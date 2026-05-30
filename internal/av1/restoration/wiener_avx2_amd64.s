// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// NEON-equivalent AVX2 Wiener separable passes. The Go amd64 assembler exposes
// every AVX2 mnemonic used here by name, so no WORD/BYTE encoding is required.
// Working vectors are Y0-Y10 plus the X-register aliases X11/X12 for the
// VPSRAD count; the amd64 ABI has no callee-saved YMM registers, so nothing
// needs preserving across the call.
//
// Lane semantics (all bit-exact with the pure-Go reference):
//   VPMOVZXWD (mem), Yn        load 8 u16 and zero-extend to 8 s32 lanes
//   VPMULLD   Ya, Yb, Yd       per-lane s32 multiply (low 32 bits, exact here)
//   VPADDD    Ya, Yb, Yd       per-lane s32 add
//   VPBROADCASTD (mem), Yn     broadcast one s32 from memory into 8 lanes
//   VPSRAD    Xc, Yn, Yn       per-lane arithmetic right shift by Xc[0]
//   VPMAXSD / VPMINSD          per-lane signed clamp (lower 0, upper bound)
//   VPACKUSDW Yn, Yn, Yn       saturating s32 -> u16 narrow (already in range)
//   VPERMQ    $0xd8, Yn, Yn    fix the per-128-lane interleave from VPACKUSDW
//   VMOVDQU   Xn, (mem)        store 8 u16 outputs

// wienerAVX2HorizCtx field offsets (see wiener_avx2_amd64.go).
#define H_DST    0
#define H_SRC    8
#define H_SRCSTR 16
#define H_WIDTH  24
#define H_ROWS   32
#define H_TAPS   40
#define H_SEED   48
#define H_SHIFT  52
#define H_MAXCL  56

// func wienerHorizontalAVX2Asm(ctx *wienerAVX2HorizCtx)
//
// Horizontal 7-tap Wiener pass. For each of `rows` rows the inner loop emits 8
// temp columns per iteration: for tap i load the 8 u16 samples at window
// offset i (zero-extended to s32), multiply by the broadcast tap, accumulate,
// then apply the folded rounding shift and the [0, maxClamp] clamp.
TEXT ·wienerHorizontalAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ H_DST(AX), DI      // temp output base
	MOVQ H_SRC(AX), SI      // src tap-window base (row -3, col -3)
	MOVQ H_SRCSTR(AX), R8   // src stride (elements)
	SHLQ $1, R8             // -> bytes
	MOVQ H_WIDTH(AX), R9    // width (multiple of 8)
	MOVQ H_ROWS(AX), R10    // rows
	MOVQ H_TAPS(AX), R11    // taps ptr (7 int32)

	LEAQ H_SEED(AX), R12
	VPBROADCASTD (R12), Y8  // seed broadcast (s32 lanes)
	LEAQ H_SHIFT(AX), R13
	VMOVD (R13), X9         // shift count (round0)
	LEAQ H_MAXCL(AX), R14
	VPBROADCASTD (R14), Y10 // maxClamp broadcast
	VPXOR Y11, Y11, Y11     // lower clamp bound 0

	// Preload the 7 broadcast taps into Y0..Y6 (loop-invariant per call).
	VPBROADCASTD 0(R11), Y0
	VPBROADCASTD 4(R11), Y1
	VPBROADCASTD 8(R11), Y2
	VPBROADCASTD 12(R11), Y3
	VPBROADCASTD 16(R11), Y4
	VPBROADCASTD 20(R11), Y5
	VPBROADCASTD 24(R11), Y6

	MOVQ R9, R15
	SHLQ $1, R15           // dst row stride in bytes (= width * 2)

hRowLoop:
	CMPQ R10, $0
	JEQ  hDone
	MOVQ DI, BX            // temp column cursor
	MOVQ SI, CX            // src window cursor
	MOVQ R9, DX            // remaining columns

hColLoop:
	VMOVDQA Y8, Y7         // seed
	VPMOVZXWD 0(CX), Y12
	VPMULLD Y0, Y12, Y12
	VPADDD  Y12, Y7, Y7
	VPMOVZXWD 2(CX), Y12
	VPMULLD Y1, Y12, Y12
	VPADDD  Y12, Y7, Y7
	VPMOVZXWD 4(CX), Y12
	VPMULLD Y2, Y12, Y12
	VPADDD  Y12, Y7, Y7
	VPMOVZXWD 6(CX), Y12
	VPMULLD Y3, Y12, Y12
	VPADDD  Y12, Y7, Y7
	VPMOVZXWD 8(CX), Y12
	VPMULLD Y4, Y12, Y12
	VPADDD  Y12, Y7, Y7
	VPMOVZXWD 10(CX), Y12
	VPMULLD Y5, Y12, Y12
	VPADDD  Y12, Y7, Y7
	VPMOVZXWD 12(CX), Y12
	VPMULLD Y6, Y12, Y12
	VPADDD  Y12, Y7, Y7

	VPSRAD  X9, Y7, Y7     // arith >> round0
	VPMAXSD Y11, Y7, Y7    // clamp lo 0
	VPMINSD Y10, Y7, Y7    // clamp hi maxClamp
	VPACKUSDW Y7, Y7, Y7   // s32 -> u16
	VPERMQ  $0xd8, Y7, Y7  // gather the 8 low words contiguously
	VMOVDQU X7, (BX)       // store 8 u16

	ADDQ $16, BX           // temp += 8 u16
	ADDQ $16, CX           // src window += 8 u16
	SUBQ $8, DX
	JNE  hColLoop

	ADDQ R15, DI           // temp += rowStride bytes
	ADDQ R8, SI            // src += srcStride bytes
	SUBQ $1, R10
	JNE  hRowLoop

hDone:
	VZEROUPPER
	RET

// wienerAVX2VertCtx field offsets (see wiener_avx2_amd64.go).
#define V_DST    0
#define V_SRC    8
#define V_DSTSTR 16
#define V_SRCSTR 24
#define V_WIDTH  32
#define V_ROWS   40
#define V_TAPS   48
#define V_SEED   56
#define V_SHIFT  60
#define V_MAXCL  64

// func wienerVerticalAVX2Asm(ctx *wienerAVX2VertCtx)
//
// Vertical 7-tap Wiener pass over the u16 temp buffer. For each output row the
// inner loop emits 8 columns per iteration: MAC the 7 stacked temp rows
// [row..row+6] (center row+3) with the taps, then apply the folded rounding
// shift and the [0, max] clamp.
TEXT ·wienerVerticalAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ V_DST(AX), DI      // dst row base
	MOVQ V_SRC(AX), SI      // temp tap-window base (row 0)
	MOVQ V_DSTSTR(AX), R8   // dst stride (elements)
	SHLQ $1, R8             // -> bytes
	MOVQ V_SRCSTR(AX), R9   // temp stride (elements)
	SHLQ $1, R9             // -> bytes
	MOVQ V_WIDTH(AX), R10   // width (multiple of 8)
	MOVQ V_ROWS(AX), R11    // rows = height
	MOVQ V_TAPS(AX), R12    // taps ptr (7 int32)

	LEAQ V_SEED(AX), R13
	VPBROADCASTD (R13), Y8  // seed broadcast
	LEAQ V_SHIFT(AX), R14
	VMOVD (R14), X9         // shift count (round1)
	LEAQ V_MAXCL(AX), R15
	VPBROADCASTD (R15), Y10 // max broadcast
	VPXOR Y11, Y11, Y11     // lower clamp bound 0

	VPBROADCASTD 0(R12), Y0
	VPBROADCASTD 4(R12), Y1
	VPBROADCASTD 8(R12), Y2
	VPBROADCASTD 12(R12), Y3
	VPBROADCASTD 16(R12), Y4
	VPBROADCASTD 20(R12), Y5
	VPBROADCASTD 24(R12), Y6

vRowLoop:
	CMPQ R11, $0
	JEQ  vDone
	MOVQ DI, BX            // dst column cursor
	MOVQ SI, CX            // temp row-window base for this output row
	MOVQ R10, DX           // remaining columns

vColLoop:
	MOVQ CX, R13           // R13 walks the 7 tap rows
	VMOVDQA Y8, Y7         // seed

	VPMOVZXWD (R13), Y12
	VPMULLD Y0, Y12, Y12
	VPADDD  Y12, Y7, Y7
	ADDQ R9, R13
	VPMOVZXWD (R13), Y12
	VPMULLD Y1, Y12, Y12
	VPADDD  Y12, Y7, Y7
	ADDQ R9, R13
	VPMOVZXWD (R13), Y12
	VPMULLD Y2, Y12, Y12
	VPADDD  Y12, Y7, Y7
	ADDQ R9, R13
	VPMOVZXWD (R13), Y12
	VPMULLD Y3, Y12, Y12
	VPADDD  Y12, Y7, Y7
	ADDQ R9, R13
	VPMOVZXWD (R13), Y12
	VPMULLD Y4, Y12, Y12
	VPADDD  Y12, Y7, Y7
	ADDQ R9, R13
	VPMOVZXWD (R13), Y12
	VPMULLD Y5, Y12, Y12
	VPADDD  Y12, Y7, Y7
	ADDQ R9, R13
	VPMOVZXWD (R13), Y12
	VPMULLD Y6, Y12, Y12
	VPADDD  Y12, Y7, Y7

	VPSRAD  X9, Y7, Y7     // arith >> round1
	VPMAXSD Y11, Y7, Y7    // clamp lo 0
	VPMINSD Y10, Y7, Y7    // clamp hi max
	VPACKUSDW Y7, Y7, Y7
	VPERMQ  $0xd8, Y7, Y7
	VMOVDQU X7, (BX)

	ADDQ $16, BX           // dst += 8 u16
	ADDQ $16, CX           // temp window += 8 u16
	SUBQ $8, DX
	JNE  vColLoop

	ADDQ R8, DI            // dst += dstStride bytes
	ADDQ R9, SI            // temp += tempStride bytes
	SUBQ $1, R11
	JNE  vRowLoop

vDone:
	VZEROUPPER
	RET
