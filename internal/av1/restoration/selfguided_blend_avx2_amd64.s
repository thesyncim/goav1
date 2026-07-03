// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 self-guided restoration per-pixel blend. The Go amd64 assembler exposes
// every AVX2 mnemonic used here by name, so no WORD/BYTE encoding is required.
// The 9 stencil weights and the rounding bias are preloaded into Y0..Y9;
// working vectors are Y10..Y13 plus X14 (the VPSRAD shift count). The amd64 ABI
// has no callee-saved YMM registers, and the GP registers used (AX/SI/DI,
// R8-R13,R15) avoid R14 (g) and BP (frame pointer), so nothing needs preserving.
//
// Lane semantics (bit-exact with the pure-Go reference):
//   VPBROADCASTD (mem), Yn     broadcast one int32 into 8 lanes
//   VMOVDQU (mem), Yn          load 8 int32 lanes (unaligned)
//   VPMULLD Ya, Yb, Yd         per-lane int32 multiply (low 32 bits, exact)
//   VPADDD  Ya, Yb, Yd         per-lane int32 add (two's-complement wrap)
//   VPSRAD  Xc, Yn, Yn         per-lane arithmetic right shift by Xc[0]
//   VMOVDQU Yn, (mem)          store 8 int32 lanes
//
// res = (accA*dgd + accB + bias) >> shift, where accA/accB are the 3x3
// weighted stencils over the A/B buffers. This reproduces
// roundPowerOfTwo(a*dgd+b, shift) bit-for-bit (the bias is 1<<(shift-1)).

// sgrBlendAVX2Ctx field offsets (see selfguided_blend_avx2_amd64.go).
#define S_DST   0
#define S_DGD   8
#define S_APREV 16
#define S_ACUR  24
#define S_ANEXT 32
#define S_BPREV 40
#define S_BCUR  48
#define S_BNEXT 56
#define S_COLS  64
#define S_WPL   72
#define S_WP    76
#define S_WPR   80
#define S_WCL   84
#define S_WC    88
#define S_WCR   92
#define S_WNL   96
#define S_WN    100
#define S_WNR   104
#define S_BIAS  108
#define S_SHIFT 112

// func sgrBlendRowAVX2Asm(ctx *sgrBlendAVX2Ctx)
//
// Computes one blended output row, 8 columns per iteration. Each row pointer
// addresses index (j-1) for column 0, so the L/C/R windows are the unaligned
// loads at byte offsets 0/4/8.
TEXT ·sgrBlendRowAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX

	// Preload the 9 weights + bias broadcasts (loop-invariant per call).
	VPBROADCASTD S_WPL(AX), Y0
	VPBROADCASTD S_WP(AX), Y1
	VPBROADCASTD S_WPR(AX), Y2
	VPBROADCASTD S_WCL(AX), Y3
	VPBROADCASTD S_WC(AX), Y4
	VPBROADCASTD S_WCR(AX), Y5
	VPBROADCASTD S_WNL(AX), Y6
	VPBROADCASTD S_WN(AX), Y7
	VPBROADCASTD S_WNR(AX), Y8
	VPBROADCASTD S_BIAS(AX), Y9
	VMOVD        S_SHIFT(AX), X14  // shift count (positive)

	MOVQ S_DST(AX), DI
	MOVQ S_DGD(AX), SI
	MOVQ S_APREV(AX), R8
	MOVQ S_ACUR(AX), R9
	MOVQ S_ANEXT(AX), R10
	MOVQ S_BPREV(AX), R11
	MOVQ S_BCUR(AX), R12
	MOVQ S_BNEXT(AX), R13
	MOVQ S_COLS(AX), R15          // remaining columns (multiple of 8)

bLoop:
	CMPQ R15, $0
	JEQ  bDone
	VPXOR Y10, Y10, Y10          // accA = 0
	VPXOR Y11, Y11, Y11          // accB = 0

	// --- A previous row ---
	VMOVDQU 0(R8), Y12           // L
	VPMULLD Y0, Y12, Y12
	VPADDD  Y12, Y10, Y10
	VMOVDQU 4(R8), Y12           // C
	VPMULLD Y1, Y12, Y12
	VPADDD  Y12, Y10, Y10
	VMOVDQU 8(R8), Y12           // R
	VPMULLD Y2, Y12, Y12
	VPADDD  Y12, Y10, Y10

	// --- A current row ---
	VMOVDQU 0(R9), Y12
	VPMULLD Y3, Y12, Y12
	VPADDD  Y12, Y10, Y10
	VMOVDQU 4(R9), Y12
	VPMULLD Y4, Y12, Y12
	VPADDD  Y12, Y10, Y10
	VMOVDQU 8(R9), Y12
	VPMULLD Y5, Y12, Y12
	VPADDD  Y12, Y10, Y10

	// --- A next row ---
	VMOVDQU 0(R10), Y12
	VPMULLD Y6, Y12, Y12
	VPADDD  Y12, Y10, Y10
	VMOVDQU 4(R10), Y12
	VPMULLD Y7, Y12, Y12
	VPADDD  Y12, Y10, Y10
	VMOVDQU 8(R10), Y12
	VPMULLD Y8, Y12, Y12
	VPADDD  Y12, Y10, Y10

	// --- B previous row ---
	VMOVDQU 0(R11), Y12
	VPMULLD Y0, Y12, Y12
	VPADDD  Y12, Y11, Y11
	VMOVDQU 4(R11), Y12
	VPMULLD Y1, Y12, Y12
	VPADDD  Y12, Y11, Y11
	VMOVDQU 8(R11), Y12
	VPMULLD Y2, Y12, Y12
	VPADDD  Y12, Y11, Y11

	// --- B current row ---
	VMOVDQU 0(R12), Y12
	VPMULLD Y3, Y12, Y12
	VPADDD  Y12, Y11, Y11
	VMOVDQU 4(R12), Y12
	VPMULLD Y4, Y12, Y12
	VPADDD  Y12, Y11, Y11
	VMOVDQU 8(R12), Y12
	VPMULLD Y5, Y12, Y12
	VPADDD  Y12, Y11, Y11

	// --- B next row ---
	VMOVDQU 0(R13), Y12
	VPMULLD Y6, Y12, Y12
	VPADDD  Y12, Y11, Y11
	VMOVDQU 4(R13), Y12
	VPMULLD Y7, Y12, Y12
	VPADDD  Y12, Y11, Y11
	VMOVDQU 8(R13), Y12
	VPMULLD Y8, Y12, Y12
	VPADDD  Y12, Y11, Y11

	// --- combine: res = (accA*dgd + accB + bias) >> shift ---
	VMOVDQU (SI), Y13            // dgd
	VPMULLD Y13, Y10, Y10        // accA*dgd
	VPADDD  Y11, Y10, Y10        // + accB
	VPADDD  Y9, Y10, Y10         // + bias
	VPSRAD  X14, Y10, Y10        // arith >> shift
	VMOVDQU Y10, (DI)

	// advance all cursors by 8 elements (32 bytes)
	ADDQ $32, DI
	ADDQ $32, SI
	ADDQ $32, R8
	ADDQ $32, R9
	ADDQ $32, R10
	ADDQ $32, R11
	ADDQ $32, R12
	ADDQ $32, R13
	SUBQ $8, R15
	JNE  bLoop

bDone:
	VZEROUPPER
	RET
