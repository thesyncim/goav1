// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 self-guided restoration box sums. The Go amd64 assembler exposes every
// AVX2 mnemonic used here by name, so no WORD/BYTE encoding is required.
// Working vectors are Y0/Y1 only; the amd64 ABI has no callee-saved YMM
// registers, and the GP registers used (SI/DI, R8-R13, AX/BX/CX/DX) avoid R14
// (g) and BP (frame pointer), so nothing needs preserving across the call.
//
// Lane semantics (bit-exact with the pure-Go reference):
//   VMOVDQU (mem), Yn          load 8 int32 lanes (unaligned)
//   VPMULLD Ya, Yb, Yd         per-lane int32 multiply (low 32 bits, exact)
//   VPADDD  Ya, Yb, Yd         per-lane int32 add (two's-complement wrap)
//   VMOVDQU Yn, (mem)          store 8 int32 lanes

// func boxsumInteriorBandAVX2Asm(src *int32, dst *int32, avxCols, radius, rows,
//                                srcStride, squared uintptr)
//
// For each group of 8 interior output columns, accumulate (in an 8-lane int32
// register) the source values over `rows` vertical rows and the 2r+1 horizontal
// taps, then store the eight sums. `src` points at the top-left of the first
// lane's window (row y0, column col-r). squared != 0 squares each value first.
TEXT ·boxsumInteriorBandAVX2Asm(SB), NOSPLIT, $0-56
	MOVQ src+0(FP), SI
	MOVQ dst+8(FP), DI
	MOVQ avxCols+16(FP), R8
	MOVQ radius+24(FP), R10
	MOVQ rows+32(FP), R11
	MOVQ srcStride+40(FP), R12
	MOVQ squared+48(FP), R13
	SHLQ $2, R12               // srcStride -> bytes
	LEAQ 1(R10)(R10*1), R10    // R10 = 2*radius + 1 (horizontal taps)
	XORQ R9, R9                // R9 = group byte offset (advances 32/group)

cgLoop:
	CMPQ R8, $0
	JEQ  cgDone
	VPXOR Y0, Y0, Y0          // accumulator

	LEAQ (SI)(R9*1), AX       // AX = src + group offset (row 0 leftmost tap)
	MOVQ R11, DX              // remaining rows

cgRow:
	MOVQ AX, BX              // BX = start of this row's window
	MOVQ R10, CX             // remaining horizontal taps

cgTap:
	VMOVDQU (BX), Y1
	CMPQ    R13, $0
	JEQ     cgAdd
	VPMULLD Y1, Y1, Y1       // square

cgAdd:
	VPADDD Y1, Y0, Y0
	ADDQ   $4, BX            // next horizontal tap (1 element)
	SUBQ   $1, CX
	JNE    cgTap

	ADDQ R12, AX            // next source row
	SUBQ $1, DX
	JNE  cgRow

	LEAQ    (DI)(R9*1), BX  // BX = dst + group offset
	VMOVDQU Y0, (BX)

	ADDQ $32, R9           // advance group by 8 columns
	SUBQ $8, R8
	JNE  cgLoop

cgDone:
	VZEROUPPER
	RET
