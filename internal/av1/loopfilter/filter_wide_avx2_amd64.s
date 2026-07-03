// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// AVX2 8-bit wide (six-, eight- and fourteen-sample) deblocking kernels. Sixteen
// horizontal edge positions are processed per 256-bit vector. Bit-exact with the
// pure-Go references (filter6.go / filter8.go / filter14.go); the Go wrappers in
// filter_wide_avx2_amd64.go restrict them to 8-bit horizontal edges processed in
// groups of sixteen positions.
//
// All lane arithmetic runs in signed 16-bit lanes: sixteen bytes widen to one
// ymm of sixteen int16 with VPMOVZXBW, the centred ps/qs values stay in
// [-128,127] and the flat weighted sums stay under 4088, so clamps use
// VPMINSW/VPMAXSW and shifts use VPSRAW, exactly matching signedClamp and the
// integer shifts of the references. needsFilter/flat/flat2/hev are reproduced as
// lane masks; each output tap is blended through those masks with VPBLENDVB so
// the conditional writes match the reference branch structure, then the sixteen
// int16 results are narrowed to sixteen contiguous bytes (VPACKUSWB + VPERMQ).
//
// Weighted sums are accumulated by repeated VPADDW of each tap (weight-many
// times) into a zeroed accumulator, so no extra scratch register is needed to
// materialise multiplied taps. roundPowerOfTwo(v, n) = (v + (1<<(n-1))) >> n is
// implemented by subtracting a negative bias ((-1)<<(n-1)) then VPSRAW $n; every
// sum is non-negative so the arithmetic shift matches the reference exactly.

//go:build amd64 && !purego

#include "textflag.h"

// ---------------------------------------------------------------------------
// filter6: ctx field offsets
// ---------------------------------------------------------------------------
#define F6_P2 0
#define F6_P1 8
#define F6_P0 16
#define F6_Q0 24
#define F6_Q1 32
#define F6_Q2 40
#define F6_COUNT 48
#define F6_LIMIT 56
#define F6_BLIMIT 64
#define F6_HEV 72
#define F6_THR 80

// func filter6EdgeAVX2Asm(ctx *filter6AVX2Ctx)
// The four output rows are staged in a 64-byte stack scratch and copied to pix
// only after all four are computed, so every tap read within a group sees the
// original samples (matching the reference, which reads a position's whole
// window before writing any of it).
TEXT ·filter6EdgeAVX2Asm(SB), NOSPLIT, $64-8
	MOVQ ctx+0(FP), AX
	MOVQ F6_P2(AX), R8
	MOVQ F6_P1(AX), R9
	MOVQ F6_P0(AX), R10
	MOVQ F6_Q0(AX), R11
	MOVQ F6_Q1(AX), R12
	MOVQ F6_Q2(AX), R13
	MOVQ F6_COUNT(AX), CX
	XORQ DX, DX // byte offset of current 16-position group

	// Clamp constants for the 8-bit narrow update: center 128, min -128, max 127.
	MOVL         $128, BX
	MOVL         BX, X3
	VPBROADCASTW X3, Y3
	MOVL         $-128, BX
	MOVL         BX, X4
	VPBROADCASTW X4, Y4
	MOVL         $127, BX
	MOVL         BX, X5
	VPBROADCASTW X5, Y5

f6loop:
	TESTQ CX, CX
	JZ    f6done

	VPMOVZXBW (R8)(DX*1), Y6   // p2
	VPMOVZXBW (R9)(DX*1), Y7   // p1
	VPMOVZXBW (R10)(DX*1), Y8  // p0
	VPMOVZXBW (R11)(DX*1), Y9  // q0
	VPMOVZXBW (R12)(DX*1), Y10 // q1
	VPMOVZXBW (R13)(DX*1), Y11 // q2

	// ---- need mask -> Y0 (= ~fail) ----
	MOVQ         F6_LIMIT(AX), BX
	MOVL         BX, X12
	VPBROADCASTW X12, Y12
	VPSUBW   Y7, Y6, Y13   // p2-p1
	VPABSW   Y13, Y13
	VPCMPGTW Y12, Y13, Y0
	VPSUBW   Y8, Y7, Y13   // p1-p0
	VPABSW   Y13, Y13
	VPCMPGTW Y12, Y13, Y13
	VPOR     Y13, Y0, Y0
	VPSUBW   Y9, Y10, Y13  // q1-q0
	VPABSW   Y13, Y13
	VPCMPGTW Y12, Y13, Y13
	VPOR     Y13, Y0, Y0
	VPSUBW   Y10, Y11, Y13 // q2-q1
	VPABSW   Y13, Y13
	VPCMPGTW Y12, Y13, Y13
	VPOR     Y13, Y0, Y0
	VPSUBW   Y9, Y8, Y13   // p0-q0
	VPABSW   Y13, Y13
	VPSLLW   $1, Y13, Y13
	VPSUBW   Y10, Y7, Y14  // p1-q1
	VPABSW   Y14, Y14
	VPSRAW   $1, Y14, Y14
	VPADDW   Y14, Y13, Y13
	MOVQ         F6_BLIMIT(AX), BX
	MOVL         BX, X12
	VPBROADCASTW X12, Y12
	VPCMPGTW Y12, Y13, Y13
	VPOR     Y13, Y0, Y0
	VPCMPEQW Y13, Y13, Y13
	VPXOR    Y13, Y0, Y0   // need

	// ---- flat mask -> Y1 ----
	MOVQ         F6_THR(AX), BX
	MOVL         BX, X12
	VPBROADCASTW X12, Y12
	VPSUBW   Y8, Y7, Y13   // p1-p0
	VPABSW   Y13, Y13
	VPCMPGTW Y12, Y13, Y1
	VPSUBW   Y9, Y10, Y13  // q1-q0
	VPABSW   Y13, Y13
	VPCMPGTW Y12, Y13, Y13
	VPOR     Y13, Y1, Y1
	VPSUBW   Y8, Y6, Y13   // p2-p0
	VPABSW   Y13, Y13
	VPCMPGTW Y12, Y13, Y13
	VPOR     Y13, Y1, Y1
	VPSUBW   Y9, Y11, Y13  // q2-q0
	VPABSW   Y13, Y13
	VPCMPGTW Y12, Y13, Y13
	VPOR     Y13, Y1, Y1
	VPCMPEQW Y13, Y13, Y13
	VPXOR    Y13, Y1, Y1   // flat

	// ---- hev mask -> Y2 ----
	MOVQ         F6_HEV(AX), BX
	MOVL         BX, X12
	VPBROADCASTW X12, Y12
	VPSUBW   Y8, Y7, Y13   // p1-p0
	VPABSW   Y13, Y13
	VPCMPGTW Y12, Y13, Y2
	VPSUBW   Y9, Y10, Y13  // q1-q0
	VPABSW   Y13, Y13
	VPCMPGTW Y12, Y13, Y13
	VPOR     Y13, Y2, Y2   // hev

	// ---- narrow filter derivation: filter1 Y13, filter2 Y14, outer Y15 ----
	VPSUBW  Y10, Y7, Y12  // p1-q1
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12
	VPAND   Y2, Y12, Y12  // hev ? clamp : 0
	VPSUBW  Y8, Y9, Y13   // q0-p0
	VPADDW  Y13, Y13, Y14
	VPADDW  Y13, Y14, Y13 // 3*(q0-p0)
	VPADDW  Y13, Y12, Y12
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12  // filter
	VPCMPEQW Y13, Y13, Y13
	VPSLLW  $2, Y13, Y13
	VPSUBW  Y13, Y12, Y13
	VPMINSW Y5, Y13, Y13
	VPMAXSW Y4, Y13, Y13
	VPSRAW  $3, Y13, Y13  // filter1
	VPCMPEQW Y15, Y15, Y15
	VPADDW  Y15, Y15, Y14
	VPADDW  Y15, Y14, Y14 // -3
	VPSUBW  Y14, Y12, Y14
	VPMINSW Y5, Y14, Y14
	VPMAXSW Y4, Y14, Y14
	VPSRAW  $3, Y14, Y14  // filter2
	VPCMPEQW Y15, Y15, Y15
	VPSUBW  Y15, Y13, Y15
	VPSRAW  $1, Y15, Y15  // outer

	// ================= p1' (store R9) =================
	VPMOVZXBW (R9)(DX*1), Y7 // origP1
	VPXOR Y11, Y11, Y11
	VPMOVZXBW (R8)(DX*1), Y6 // p2 *3
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // p1 *2
	VPADDW Y7, Y11, Y11
	VPMOVZXBW (R10)(DX*1), Y6 // p0 *2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R11)(DX*1), Y6 // q0 *1
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y12, Y12, Y12
	VPSLLW $2, Y12, Y12
	VPSUBW Y12, Y11, Y11
	VPSRAW $3, Y11, Y11      // flatVal
	VPSUBW Y3, Y7, Y12      // narrow p1' = clamp(ps1+outer)+128
	VPADDW Y15, Y12, Y12
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12
	VPADDW Y3, Y12, Y12
	VPBLENDVB Y2, Y7, Y12, Y12  // hev ? orig : narrow
	VPBLENDVB Y1, Y11, Y12, Y11 // flat ? flatVal : narrowVal
	VPBLENDVB Y0, Y11, Y7, Y11  // need ? res : orig
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 0(SP)

	// ================= p0' (store R10) =================
	VPMOVZXBW (R10)(DX*1), Y7 // origP0
	VPXOR Y11, Y11, Y11
	VPMOVZXBW (R8)(DX*1), Y6 // p2 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R9)(DX*1), Y6 // p1 *2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // p0 *2
	VPADDW Y7, Y11, Y11
	VPMOVZXBW (R11)(DX*1), Y6 // q0 *2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R12)(DX*1), Y6 // q1 *1
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y12, Y12, Y12
	VPSLLW $2, Y12, Y12
	VPSUBW Y12, Y11, Y11
	VPSRAW $3, Y11, Y11
	VPSUBW Y3, Y7, Y12      // narrow p0' = clamp(ps0+filter2)+128
	VPADDW Y14, Y12, Y12
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12
	VPADDW Y3, Y12, Y12
	VPBLENDVB Y1, Y11, Y12, Y11 // flat ? flatVal : narrowVal
	VPBLENDVB Y0, Y11, Y7, Y11  // need ? res : orig
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 16(SP)

	// ================= q0' (store R11) =================
	VPMOVZXBW (R11)(DX*1), Y7 // origQ0
	VPXOR Y11, Y11, Y11
	VPMOVZXBW (R9)(DX*1), Y6 // p1 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R10)(DX*1), Y6 // p0 *2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // q0 *2
	VPADDW Y7, Y11, Y11
	VPMOVZXBW (R12)(DX*1), Y6 // q1 *2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R13)(DX*1), Y6 // q2 *1
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y12, Y12, Y12
	VPSLLW $2, Y12, Y12
	VPSUBW Y12, Y11, Y11
	VPSRAW $3, Y11, Y11
	VPSUBW Y3, Y7, Y12      // narrow q0' = clamp(qs0-filter1)+128
	VPSUBW Y13, Y12, Y12
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12
	VPADDW Y3, Y12, Y12
	VPBLENDVB Y1, Y11, Y12, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 32(SP)

	// ================= q1' (store R12) =================
	VPMOVZXBW (R12)(DX*1), Y7 // origQ1
	VPXOR Y11, Y11, Y11
	VPMOVZXBW (R10)(DX*1), Y6 // p0 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R11)(DX*1), Y6 // q0 *2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // q1 *2
	VPADDW Y7, Y11, Y11
	VPMOVZXBW (R13)(DX*1), Y6 // q2 *3
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y12, Y12, Y12
	VPSLLW $2, Y12, Y12
	VPSUBW Y12, Y11, Y11
	VPSRAW $3, Y11, Y11
	VPSUBW Y3, Y7, Y12      // narrow q1' = clamp(qs1-outer)+128
	VPSUBW Y15, Y12, Y12
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12
	VPADDW Y3, Y12, Y12
	VPBLENDVB Y2, Y7, Y12, Y12  // hev ? orig : narrow
	VPBLENDVB Y1, Y11, Y12, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 48(SP)

	// copy staged outputs to pix (p1,p0,q0,q1)
	VMOVDQU 0(SP), X6
	VMOVDQU X6, (R9)(DX*1)
	VMOVDQU 16(SP), X6
	VMOVDQU X6, (R10)(DX*1)
	VMOVDQU 32(SP), X6
	VMOVDQU X6, (R11)(DX*1)
	VMOVDQU 48(SP), X6
	VMOVDQU X6, (R12)(DX*1)

	ADDQ $16, DX
	DECQ CX
	JMP  f6loop

f6done:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// filter8: ctx field offsets
// ---------------------------------------------------------------------------
#define F8_P3 0
#define F8_P2 8
#define F8_P1 16
#define F8_P0 24
#define F8_Q0 32
#define F8_Q1 40
#define F8_Q2 48
#define F8_Q3 56
#define F8_COUNT 64
#define F8_LIMIT 72
#define F8_BLIMIT 80
#define F8_HEV 88
#define F8_THR 96

// func filter8EdgeAVX2Asm(ctx *filter8AVX2Ctx)
// Outputs are staged in a 96-byte stack scratch and copied to pix only after all
// six are computed, so tap reads within a group see original samples.
TEXT ·filter8EdgeAVX2Asm(SB), NOSPLIT, $96-8
	MOVQ ctx+0(FP), AX
	MOVQ F8_P3(AX), R8
	MOVQ F8_P2(AX), R9
	MOVQ F8_P1(AX), R10
	MOVQ F8_P0(AX), R11
	MOVQ F8_Q0(AX), R12
	MOVQ F8_Q1(AX), R13
	MOVQ F8_Q2(AX), SI
	MOVQ F8_Q3(AX), R15
	MOVQ F8_COUNT(AX), CX
	XORQ DX, DX

	MOVL         $128, BX
	MOVL         BX, X3
	VPBROADCASTW X3, Y3
	MOVL         $-128, BX
	MOVL         BX, X4
	VPBROADCASTW X4, Y4
	MOVL         $127, BX
	MOVL         BX, X5
	VPBROADCASTW X5, Y5

f8loop:
	TESTQ CX, CX
	JZ    f8done

	VPMOVZXBW (R8)(DX*1), Y6   // p3
	VPMOVZXBW (R9)(DX*1), Y7   // p2
	VPMOVZXBW (R10)(DX*1), Y8  // p1
	VPMOVZXBW (R11)(DX*1), Y9  // p0
	VPMOVZXBW (R12)(DX*1), Y10 // q0
	VPMOVZXBW (R13)(DX*1), Y11 // q1
	VPMOVZXBW (SI)(DX*1), Y12 // q2
	VPMOVZXBW (R15)(DX*1), Y13 // q3

	// ---- need mask -> Y0 ----
	MOVQ         F8_LIMIT(AX), BX
	MOVL         BX, X14
	VPBROADCASTW X14, Y14
	VPSUBW   Y7, Y6, Y15   // p3-p2
	VPABSW   Y15, Y15
	VPCMPGTW Y14, Y15, Y0
	VPSUBW   Y8, Y7, Y15   // p2-p1
	VPABSW   Y15, Y15
	VPCMPGTW Y14, Y15, Y15
	VPOR     Y15, Y0, Y0
	VPSUBW   Y9, Y8, Y15   // p1-p0
	VPABSW   Y15, Y15
	VPCMPGTW Y14, Y15, Y15
	VPOR     Y15, Y0, Y0
	VPSUBW   Y10, Y11, Y15 // q1-q0
	VPABSW   Y15, Y15
	VPCMPGTW Y14, Y15, Y15
	VPOR     Y15, Y0, Y0
	VPSUBW   Y11, Y12, Y15 // q2-q1
	VPABSW   Y15, Y15
	VPCMPGTW Y14, Y15, Y15
	VPOR     Y15, Y0, Y0
	VPSUBW   Y12, Y13, Y15 // q3-q2
	VPABSW   Y15, Y15
	VPCMPGTW Y14, Y15, Y15
	VPOR     Y15, Y0, Y0
	VPSUBW   Y10, Y9, Y15  // p0-q0
	VPABSW   Y15, Y15
	VPSLLW   $1, Y15, Y15
	VPSUBW   Y11, Y8, Y14  // p1-q1 (limit no longer needed)
	VPABSW   Y14, Y14
	VPSRAW   $1, Y14, Y14
	VPADDW   Y14, Y15, Y15
	MOVQ         F8_BLIMIT(AX), BX
	MOVL         BX, X14
	VPBROADCASTW X14, Y14
	VPCMPGTW Y14, Y15, Y15
	VPOR     Y15, Y0, Y0
	VPCMPEQW Y15, Y15, Y15
	VPXOR    Y15, Y0, Y0   // need

	// ---- flat mask -> Y1 ----
	MOVQ         F8_THR(AX), BX
	MOVL         BX, X14
	VPBROADCASTW X14, Y14
	VPSUBW   Y9, Y8, Y15   // p1-p0
	VPABSW   Y15, Y15
	VPCMPGTW Y14, Y15, Y1
	VPSUBW   Y10, Y11, Y15 // q1-q0
	VPABSW   Y15, Y15
	VPCMPGTW Y14, Y15, Y15
	VPOR     Y15, Y1, Y1
	VPSUBW   Y9, Y7, Y15   // p2-p0
	VPABSW   Y15, Y15
	VPCMPGTW Y14, Y15, Y15
	VPOR     Y15, Y1, Y1
	VPSUBW   Y10, Y12, Y15 // q2-q0
	VPABSW   Y15, Y15
	VPCMPGTW Y14, Y15, Y15
	VPOR     Y15, Y1, Y1
	VPSUBW   Y9, Y6, Y15   // p3-p0
	VPABSW   Y15, Y15
	VPCMPGTW Y14, Y15, Y15
	VPOR     Y15, Y1, Y1
	VPSUBW   Y10, Y13, Y15 // q3-q0
	VPABSW   Y15, Y15
	VPCMPGTW Y14, Y15, Y15
	VPOR     Y15, Y1, Y1
	VPCMPEQW Y15, Y15, Y15
	VPXOR    Y15, Y1, Y1   // flat

	// ---- hev mask -> Y2 ----
	MOVQ         F8_HEV(AX), BX
	MOVL         BX, X14
	VPBROADCASTW X14, Y14
	VPSUBW   Y9, Y8, Y15   // p1-p0
	VPABSW   Y15, Y15
	VPCMPGTW Y14, Y15, Y2
	VPSUBW   Y10, Y11, Y15 // q1-q0
	VPABSW   Y15, Y15
	VPCMPGTW Y14, Y15, Y15
	VPOR     Y15, Y2, Y2   // hev

	// ---- narrow filter derivation: filter1 Y13, filter2 Y14, outer Y15 ----
	// (taps p3/p2/q2/q3 are no longer needed; reuse those regs freely.)
	VPSUBW  Y11, Y8, Y12  // p1-q1
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12
	VPAND   Y2, Y12, Y12
	VPSUBW  Y9, Y10, Y6   // q0-p0
	VPADDW  Y6, Y6, Y7
	VPADDW  Y6, Y7, Y6    // 3*(q0-p0)
	VPADDW  Y6, Y12, Y12
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12  // filter
	VPCMPEQW Y13, Y13, Y13
	VPSLLW  $2, Y13, Y13
	VPSUBW  Y13, Y12, Y13
	VPMINSW Y5, Y13, Y13
	VPMAXSW Y4, Y13, Y13
	VPSRAW  $3, Y13, Y13  // filter1
	VPCMPEQW Y15, Y15, Y15
	VPADDW  Y15, Y15, Y14
	VPADDW  Y15, Y14, Y14 // -3
	VPSUBW  Y14, Y12, Y14
	VPMINSW Y5, Y14, Y14
	VPMAXSW Y4, Y14, Y14
	VPSRAW  $3, Y14, Y14  // filter2
	VPCMPEQW Y15, Y15, Y15
	VPSUBW  Y15, Y13, Y15
	VPSRAW  $1, Y15, Y15  // outer

	// ================= p2' (store R9) : flat ? f8flat : orig =================
	VPMOVZXBW (R9)(DX*1), Y7 // origP2
	VPXOR Y11, Y11, Y11
	VPMOVZXBW (R8)(DX*1), Y6 // p3 *3
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // p2 *2
	VPADDW Y7, Y11, Y11
	VPMOVZXBW (R10)(DX*1), Y6 // p1 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R11)(DX*1), Y6 // p0 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R12)(DX*1), Y6 // q0 *1
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y6, Y6, Y6
	VPSLLW $2, Y6, Y6
	VPSUBW Y6, Y11, Y11
	VPSRAW $3, Y11, Y11
	VPBLENDVB Y1, Y11, Y7, Y11  // flat ? f8flat : orig
	VPBLENDVB Y0, Y11, Y7, Y11  // need ? res : orig
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 0(SP)

	// ================= p1' (store R10) : flat ? f8flat : (hev?orig:narrow) ===
	VPMOVZXBW (R10)(DX*1), Y7 // origP1
	VPXOR Y11, Y11, Y11
	VPMOVZXBW (R8)(DX*1), Y6 // p3 *2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R9)(DX*1), Y6 // p2 *1
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // p1 *2
	VPADDW Y7, Y11, Y11
	VPMOVZXBW (R11)(DX*1), Y6 // p0 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R12)(DX*1), Y6 // q0 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R13)(DX*1), Y6 // q1 *1
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y6, Y6, Y6
	VPSLLW $2, Y6, Y6
	VPSUBW Y6, Y11, Y11
	VPSRAW $3, Y11, Y11      // f8flat
	VPSUBW Y3, Y7, Y12      // narrow p1'
	VPADDW Y15, Y12, Y12
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12
	VPADDW Y3, Y12, Y12
	VPBLENDVB Y2, Y7, Y12, Y12  // hev ? orig : narrow
	VPBLENDVB Y1, Y11, Y12, Y11 // flat ? f8flat : notflat
	VPBLENDVB Y0, Y11, Y7, Y11  // need ? res : orig
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 16(SP)

	// ================= p0' (store R11) : flat ? f8flat : narrow =================
	VPMOVZXBW (R11)(DX*1), Y7 // origP0
	VPXOR Y11, Y11, Y11
	VPMOVZXBW (R8)(DX*1), Y6 // p3 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R9)(DX*1), Y6 // p2 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R10)(DX*1), Y6 // p1 *1
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // p0 *2
	VPADDW Y7, Y11, Y11
	VPMOVZXBW (R12)(DX*1), Y6 // q0 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R13)(DX*1), Y6 // q1 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (SI)(DX*1), Y6 // q2 *1
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y6, Y6, Y6
	VPSLLW $2, Y6, Y6
	VPSUBW Y6, Y11, Y11
	VPSRAW $3, Y11, Y11
	VPSUBW Y3, Y7, Y12      // narrow p0'
	VPADDW Y14, Y12, Y12
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12
	VPADDW Y3, Y12, Y12
	VPBLENDVB Y1, Y11, Y12, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 32(SP)

	// ================= q0' (store R12) : flat ? f8flat : narrow =================
	VPMOVZXBW (R12)(DX*1), Y7 // origQ0
	VPXOR Y11, Y11, Y11
	VPMOVZXBW (R9)(DX*1), Y6 // p2 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R10)(DX*1), Y6 // p1 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R11)(DX*1), Y6 // p0 *1
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // q0 *2
	VPADDW Y7, Y11, Y11
	VPMOVZXBW (R13)(DX*1), Y6 // q1 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (SI)(DX*1), Y6 // q2 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R15)(DX*1), Y6 // q3 *1
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y6, Y6, Y6
	VPSLLW $2, Y6, Y6
	VPSUBW Y6, Y11, Y11
	VPSRAW $3, Y11, Y11
	VPSUBW Y3, Y7, Y12      // narrow q0'
	VPSUBW Y13, Y12, Y12
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12
	VPADDW Y3, Y12, Y12
	VPBLENDVB Y1, Y11, Y12, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 48(SP)

	// ================= q1' (store R13) : flat ? f8flat : (hev?orig:narrow) ===
	VPMOVZXBW (R13)(DX*1), Y7 // origQ1
	VPXOR Y11, Y11, Y11
	VPMOVZXBW (R10)(DX*1), Y6 // p1 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R11)(DX*1), Y6 // p0 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R12)(DX*1), Y6 // q0 *1
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // q1 *2
	VPADDW Y7, Y11, Y11
	VPMOVZXBW (SI)(DX*1), Y6 // q2 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R15)(DX*1), Y6 // q3 *2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y6, Y6, Y6
	VPSLLW $2, Y6, Y6
	VPSUBW Y6, Y11, Y11
	VPSRAW $3, Y11, Y11
	VPSUBW Y3, Y7, Y12      // narrow q1'
	VPSUBW Y15, Y12, Y12
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12
	VPADDW Y3, Y12, Y12
	VPBLENDVB Y2, Y7, Y12, Y12
	VPBLENDVB Y1, Y11, Y12, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 64(SP)

	// ================= q2' (store SI) : flat ? f8flat : orig =================
	VPMOVZXBW (SI)(DX*1), Y7 // origQ2
	VPXOR Y11, Y11, Y11
	VPMOVZXBW (R11)(DX*1), Y6 // p0 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R12)(DX*1), Y6 // q0 *1
	VPADDW Y6, Y11, Y11
	VPMOVZXBW (R13)(DX*1), Y6 // q1 *1
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // q2 *2
	VPADDW Y7, Y11, Y11
	VPMOVZXBW (R15)(DX*1), Y6 // q3 *3
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y6, Y6, Y6
	VPSLLW $2, Y6, Y6
	VPSUBW Y6, Y11, Y11
	VPSRAW $3, Y11, Y11
	VPBLENDVB Y1, Y11, Y7, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 80(SP)

	// copy staged outputs to pix (p2,p1,p0,q0,q1,q2)
	VMOVDQU 0(SP), X6
	VMOVDQU X6, (R9)(DX*1)
	VMOVDQU 16(SP), X6
	VMOVDQU X6, (R10)(DX*1)
	VMOVDQU 32(SP), X6
	VMOVDQU X6, (R11)(DX*1)
	VMOVDQU 48(SP), X6
	VMOVDQU X6, (R12)(DX*1)
	VMOVDQU 64(SP), X6
	VMOVDQU X6, (R13)(DX*1)
	VMOVDQU 80(SP), X6
	VMOVDQU X6, (SI)(DX*1)

	ADDQ $16, DX
	DECQ CX
	JMP  f8loop

f8done:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// filter14: ctx field offsets. Fourteen taps do not fit in registers, so the
// asm loads each tap pointer from the context (AX) on demand into BX and indexes
// by the running group offset DX; the destination pointer for the current output
// is kept in SI across its tap loads.
// ---------------------------------------------------------------------------
#define FE_P6 0
#define FE_P5 8
#define FE_P4 16
#define FE_P3 24
#define FE_P2 32
#define FE_P1 40
#define FE_P0 48
#define FE_Q0 56
#define FE_Q1 64
#define FE_Q2 72
#define FE_Q3 80
#define FE_Q4 88
#define FE_Q5 96
#define FE_Q6 104
#define FE_COUNT 112
#define FE_LIMIT 120
#define FE_BLIMIT 128
#define FE_HEV 136
#define FE_THR 144

// func filter14EdgeAVX2Asm(ctx *filter14AVX2Ctx)
// The twelve outputs are staged in a 192-byte stack scratch and copied to pix
// only after all are computed, so every tap read within a group sees original
// samples (the wide path reads rows that other outputs overwrite).
TEXT ·filter14EdgeAVX2Asm(SB), NOSPLIT, $192-8
	MOVQ ctx+0(FP), AX
	MOVQ FE_COUNT(AX), CX
	XORQ DX, DX

	MOVL         $128, BX
	MOVL         BX, X3
	VPBROADCASTW X3, Y3
	MOVL         $-128, BX
	MOVL         BX, X4
	VPBROADCASTW X4, Y4
	MOVL         $127, BX
	MOVL         BX, X5
	VPBROADCASTW X5, Y5

feloop:
	TESTQ CX, CX
	JZ    fedone

	// ---- need mask -> Y0 : needsFilter8(p3,p2,p1,p0,q0,q1,q2,q3) ----
	MOVQ         FE_LIMIT(AX), BX
	MOVL         BX, X9
	VPBROADCASTW X9, Y9   // limit
	MOVQ FE_P3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	MOVQ FE_P2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y7
	VPSUBW Y7, Y6, Y8     // p3-p2
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y0
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y6, Y7, Y8     // p2-p1
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y0, Y0
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y7
	VPSUBW Y7, Y6, Y8     // p1-p0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y0, Y0
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y7
	VPSUBW Y6, Y7, Y8     // q1-q0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y0, Y0
	MOVQ FE_Q2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y7, Y6, Y8     // q2-q1
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y0, Y0
	MOVQ FE_Q3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y7
	VPSUBW Y6, Y7, Y8     // q3-q2
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y0, Y0
	// blimit term: |p0-q0|*2 + |p1-q1|/2
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y7
	VPSUBW Y7, Y6, Y8     // p0-q0
	VPABSW Y8, Y8
	VPSLLW $1, Y8, Y8
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y7
	VPSUBW Y7, Y6, Y9     // p1-q1 (limit dead)
	VPABSW Y9, Y9
	VPSRAW $1, Y9, Y9
	VPADDW Y9, Y8, Y8
	MOVQ         FE_BLIMIT(AX), BX
	MOVL         BX, X9
	VPBROADCASTW X9, Y9
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y0, Y0
	VPCMPEQW Y8, Y8, Y8
	VPXOR Y8, Y0, Y0      // need

	// ---- flat mask -> Y1 : flatMask4(thr,p3,p2,p1,p0,q0,q1,q2,q3) ----
	MOVQ         FE_THR(AX), BX
	MOVL         BX, X9
	VPBROADCASTW X9, Y9   // thr
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y7 // p0 (reused as base for many diffs)
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y7, Y6, Y8     // p1-p0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y1
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y10 // q0 (reused base)
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y10, Y6, Y8    // q1-q0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y1, Y1
	MOVQ FE_P2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y7, Y6, Y8     // p2-p0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y1, Y1
	MOVQ FE_Q2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y10, Y6, Y8    // q2-q0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y1, Y1
	MOVQ FE_P3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y7, Y6, Y8     // p3-p0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y1, Y1
	MOVQ FE_Q3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y10, Y6, Y8    // q3-q0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y1, Y1
	VPCMPEQW Y8, Y8, Y8
	VPXOR Y8, Y1, Y1      // flat

	// ---- flat2 -> Y15, then wide = flat & flat2 -> Y15 ----
	// flatMask4(thr,p6,p5,p4,p0,q0,q4,q5,q6): checks |p4-p0|,|q4-q0|,|p5-p0|,
	// |q5-q0|,|p6-p0|,|q6-q0|. Y7=p0, Y10=q0 still live.
	MOVQ FE_P4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y7, Y6, Y8     // p4-p0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y15
	MOVQ FE_Q4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y10, Y6, Y8    // q4-q0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y15, Y15
	MOVQ FE_P5(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y7, Y6, Y8     // p5-p0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y15, Y15
	MOVQ FE_Q5(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y10, Y6, Y8    // q5-q0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y15, Y15
	MOVQ FE_P6(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y7, Y6, Y8     // p6-p0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y15, Y15
	MOVQ FE_Q6(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y10, Y6, Y8    // q6-q0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y15, Y15
	VPCMPEQW Y8, Y8, Y8
	VPXOR Y8, Y15, Y15    // flat2
	VPAND Y1, Y15, Y15    // wide = flat & flat2

	// ---- hev mask -> Y2 ----
	MOVQ         FE_HEV(AX), BX
	MOVL         BX, X9
	VPBROADCASTW X9, Y9
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y7
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y7, Y6, Y8     // p1-p0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y2
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y7
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPSUBW Y7, Y6, Y8     // q1-q0
	VPABSW Y8, Y8
	VPCMPGTW Y9, Y8, Y8
	VPOR Y8, Y2, Y2       // hev

	// ---- narrow filter derivation: filter1 Y12, filter2 Y13, outer Y14 ----
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p1
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y7 // q1
	VPSUBW Y7, Y6, Y8     // p1-q1
	VPMINSW Y5, Y8, Y8
	VPMAXSW Y4, Y8, Y8
	VPAND Y2, Y8, Y8      // hev ? clamp : 0
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p0
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y7 // q0
	VPSUBW Y6, Y7, Y9     // q0-p0
	VPADDW Y9, Y9, Y10
	VPADDW Y9, Y10, Y9    // 3*(q0-p0)
	VPADDW Y9, Y8, Y8
	VPMINSW Y5, Y8, Y8
	VPMAXSW Y4, Y8, Y8    // filter
	VPCMPEQW Y10, Y10, Y10
	VPSLLW $2, Y10, Y10
	VPSUBW Y10, Y8, Y12
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12
	VPSRAW $3, Y12, Y12   // filter1 Y12
	VPCMPEQW Y10, Y10, Y10
	VPADDW Y10, Y10, Y11
	VPADDW Y10, Y11, Y11  // -3
	VPSUBW Y11, Y8, Y13
	VPMINSW Y5, Y13, Y13
	VPMAXSW Y4, Y13, Y13
	VPSRAW $3, Y13, Y13   // filter2 Y13
	VPCMPEQW Y10, Y10, Y10
	VPSUBW Y10, Y12, Y14
	VPSRAW $1, Y14, Y14   // outer Y14

	// Persistent from here: need Y0, flat Y1, hev Y2, center Y3, min Y4, max Y5,
	// filter1 Y12, filter2 Y13, outer Y14, wide Y15. Scratch Y6..Y11.
	// Each output computes the 14-tap wide value (n=4, bias +8) and, where the
	// not-wide path differs, the filter8 flat value (n=3, bias +4) and/or the
	// narrow update, then blends wide?/flat?/hev? and finally need?.

	// ================= p5' (store FE_P5) : (need&wide)? wide : orig =========
	MOVQ FE_P5(AX), SI
	VPMOVZXBW (SI)(DX*1), Y7 // origP5
	VPXOR Y11, Y11, Y11
	MOVQ FE_P6(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p6*7
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // p5*2 (orig)
	VPADDW Y7, Y11, Y11
	MOVQ FE_P4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p4*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_P3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p3*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p2*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q0*1
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $3, Y8, Y8
	VPSUBW Y8, Y11, Y11
	VPSRAW $4, Y11, Y11
	VPBLENDVB Y15, Y11, Y7, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 0(SP)

	// ================= p4' (store FE_P4) : (need&wide)? wide : orig =========
	MOVQ FE_P4(AX), SI
	VPMOVZXBW (SI)(DX*1), Y7 // origP4
	VPXOR Y11, Y11, Y11
	MOVQ FE_P6(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p6*5
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_P5(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p5*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // p4*2 (orig)
	VPADDW Y7, Y11, Y11
	MOVQ FE_P3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p3*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_P2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p2*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q1*1
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $3, Y8, Y8
	VPSUBW Y8, Y11, Y11
	VPSRAW $4, Y11, Y11
	VPBLENDVB Y15, Y11, Y7, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 16(SP)

	// ================= p3' (store FE_P3) : (need&wide)? wide : orig =========
	MOVQ FE_P3(AX), SI
	VPMOVZXBW (SI)(DX*1), Y7 // origP3
	VPXOR Y11, Y11, Y11
	MOVQ FE_P6(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p6*4
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_P5(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p5*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p4*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // p3*2 (orig)
	VPADDW Y7, Y11, Y11
	MOVQ FE_P2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p2*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q2*1
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $3, Y8, Y8
	VPSUBW Y8, Y11, Y11
	VPSRAW $4, Y11, Y11
	VPBLENDVB Y15, Y11, Y7, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 32(SP)

	// ================= p2' (store FE_P2) : wide? wide : (flat? f8 : orig) ====
	MOVQ FE_P2(AX), SI
	VPMOVZXBW (SI)(DX*1), Y7 // origP2
	VPXOR Y11, Y11, Y11
	MOVQ FE_P6(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p6*3
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_P5(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p5*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p4*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p3*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // p2*2 (orig)
	VPADDW Y7, Y11, Y11
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p1*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q2*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q3*1
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $3, Y8, Y8
	VPSUBW Y8, Y11, Y11
	VPSRAW $4, Y11, Y11      // wideVal Y11
	// f8flat p2 = 3p3+2p2(orig)+p1+p0+q0
	VPXOR Y10, Y10, Y10
	MOVQ FE_P3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	VPADDW Y6, Y10, Y10
	VPADDW Y6, Y10, Y10
	VPADDW Y7, Y10, Y10
	VPADDW Y7, Y10, Y10
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $2, Y8, Y8
	VPSUBW Y8, Y10, Y10
	VPSRAW $3, Y10, Y10      // f8flat Y10
	VPBLENDVB Y1, Y10, Y7, Y10  // flat ? f8 : orig
	VPBLENDVB Y15, Y11, Y10, Y11 // wide ? wide : inner
	VPBLENDVB Y0, Y11, Y7, Y11  // need ? res : orig
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 48(SP)

	// ================= p1' (store FE_P1) : wide? : (flat? f8 : (hev?orig:narrow))
	MOVQ FE_P1(AX), SI
	VPMOVZXBW (SI)(DX*1), Y7 // origP1
	VPXOR Y11, Y11, Y11
	MOVQ FE_P6(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p6*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_P5(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p5*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p4*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p3*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p2*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // p1*2 (orig)
	VPADDW Y7, Y11, Y11
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p0*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q2*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q3*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q4*1
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $3, Y8, Y8
	VPSUBW Y8, Y11, Y11
	VPSRAW $4, Y11, Y11      // wideVal Y11
	// f8flat p1 = 2p3+p2+2p1(orig)+p0+q0+q1
	VPXOR Y10, Y10, Y10
	MOVQ FE_P3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	VPADDW Y6, Y10, Y10
	MOVQ FE_P2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	VPADDW Y7, Y10, Y10
	VPADDW Y7, Y10, Y10
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $2, Y8, Y8
	VPSUBW Y8, Y10, Y10
	VPSRAW $3, Y10, Y10      // f8flat Y10
	// narrow p1 = clamp(p1-center+outer)+center
	VPSUBW Y3, Y7, Y9
	VPADDW Y14, Y9, Y9
	VPMINSW Y5, Y9, Y9
	VPMAXSW Y4, Y9, Y9
	VPADDW Y3, Y9, Y9
	VPBLENDVB Y2, Y7, Y9, Y9    // hev ? orig : narrow
	VPBLENDVB Y1, Y10, Y9, Y10  // flat ? f8 : notflat
	VPBLENDVB Y15, Y11, Y10, Y11 // wide ? wide : inner
	VPBLENDVB Y0, Y11, Y7, Y11  // need ? res : orig
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 64(SP)

	// ================= p0' (store FE_P0) : wide? : (flat? f8 : narrow) =======
	MOVQ FE_P0(AX), SI
	VPMOVZXBW (SI)(DX*1), Y7 // origP0
	VPXOR Y11, Y11, Y11
	MOVQ FE_P6(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p6*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P5(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p5*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p4*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p3*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p2*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p1*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // p0*2 (orig)
	VPADDW Y7, Y11, Y11
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q0*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q2*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q3*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q4*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q5(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q5*1
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $3, Y8, Y8
	VPSUBW Y8, Y11, Y11
	VPSRAW $4, Y11, Y11      // wideVal Y11
	// f8flat p0 = p3+p2+p1+2p0(orig)+q0+q1+q2
	VPXOR Y10, Y10, Y10
	MOVQ FE_P3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_P2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	VPADDW Y7, Y10, Y10
	VPADDW Y7, Y10, Y10
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_Q2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $2, Y8, Y8
	VPSUBW Y8, Y10, Y10
	VPSRAW $3, Y10, Y10      // f8flat Y10
	// narrow p0 = clamp(p0-center+filter2)+center
	VPSUBW Y3, Y7, Y9
	VPADDW Y13, Y9, Y9
	VPMINSW Y5, Y9, Y9
	VPMAXSW Y4, Y9, Y9
	VPADDW Y3, Y9, Y9
	VPBLENDVB Y1, Y10, Y9, Y10  // flat ? f8 : narrow
	VPBLENDVB Y15, Y11, Y10, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 80(SP)

	// ================= q0' (store FE_Q0) : wide? : (flat? f8 : narrow) =======
	MOVQ FE_Q0(AX), SI
	VPMOVZXBW (SI)(DX*1), Y7 // origQ0
	VPXOR Y11, Y11, Y11
	MOVQ FE_P5(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p5*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p4*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p3*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p2*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p0*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // q0*2 (orig)
	VPADDW Y7, Y11, Y11
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q1*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q2*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q3*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q4*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q5(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q5*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q6(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q6*1
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $3, Y8, Y8
	VPSUBW Y8, Y11, Y11
	VPSRAW $4, Y11, Y11      // wideVal Y11
	// f8flat q0 = p2+p1+p0+2q0(orig)+q1+q2+q3
	VPXOR Y10, Y10, Y10
	MOVQ FE_P2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	VPADDW Y7, Y10, Y10
	VPADDW Y7, Y10, Y10
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_Q2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_Q3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $2, Y8, Y8
	VPSUBW Y8, Y10, Y10
	VPSRAW $3, Y10, Y10      // f8flat Y10
	// narrow q0 = clamp(q0-center-filter1)+center
	VPSUBW Y3, Y7, Y9
	VPSUBW Y12, Y9, Y9
	VPMINSW Y5, Y9, Y9
	VPMAXSW Y4, Y9, Y9
	VPADDW Y3, Y9, Y9
	VPBLENDVB Y1, Y10, Y9, Y10
	VPBLENDVB Y15, Y11, Y10, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 96(SP)

	// ================= q1' (store FE_Q1) : wide? : (flat? f8 : (hev?orig:narrow))
	MOVQ FE_Q1(AX), SI
	VPMOVZXBW (SI)(DX*1), Y7 // origQ1
	VPXOR Y11, Y11, Y11
	MOVQ FE_P4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p4*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p3*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p2*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q0*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // q1*2 (orig)
	VPADDW Y7, Y11, Y11
	MOVQ FE_Q2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q2*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q3*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q4*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q5(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q5*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q6(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q6*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $3, Y8, Y8
	VPSUBW Y8, Y11, Y11
	VPSRAW $4, Y11, Y11      // wideVal Y11
	// f8flat q1 = p1+p0+q0+2q1(orig)+q2+2q3
	VPXOR Y10, Y10, Y10
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	VPADDW Y7, Y10, Y10
	VPADDW Y7, Y10, Y10
	MOVQ FE_Q2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_Q3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	VPADDW Y6, Y10, Y10
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $2, Y8, Y8
	VPSUBW Y8, Y10, Y10
	VPSRAW $3, Y10, Y10      // f8flat Y10
	// narrow q1 = clamp(q1-center-outer)+center
	VPSUBW Y3, Y7, Y9
	VPSUBW Y14, Y9, Y9
	VPMINSW Y5, Y9, Y9
	VPMAXSW Y4, Y9, Y9
	VPADDW Y3, Y9, Y9
	VPBLENDVB Y2, Y7, Y9, Y9    // hev ? orig : narrow
	VPBLENDVB Y1, Y10, Y9, Y10  // flat ? f8 : notflat
	VPBLENDVB Y15, Y11, Y10, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 112(SP)

	// ================= q2' (store FE_Q2) : wide? : (flat? f8 : orig) =========
	MOVQ FE_Q2(AX), SI
	VPMOVZXBW (SI)(DX*1), Y7 // origQ2
	VPXOR Y11, Y11, Y11
	MOVQ FE_P3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p3*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p2*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q1*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // q2*2 (orig)
	VPADDW Y7, Y11, Y11
	MOVQ FE_Q3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q3*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q4*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q5(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q5*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q6(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q6*3
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $3, Y8, Y8
	VPSUBW Y8, Y11, Y11
	VPSRAW $4, Y11, Y11      // wideVal Y11
	// f8flat q2 = p0+q0+q1+2q2(orig)+3q3
	VPXOR Y10, Y10, Y10
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	VPADDW Y7, Y10, Y10
	VPADDW Y7, Y10, Y10
	MOVQ FE_Q3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6
	VPADDW Y6, Y10, Y10
	VPADDW Y6, Y10, Y10
	VPADDW Y6, Y10, Y10
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $2, Y8, Y8
	VPSUBW Y8, Y10, Y10
	VPSRAW $3, Y10, Y10      // f8flat Y10
	VPBLENDVB Y1, Y10, Y7, Y10  // flat ? f8 : orig
	VPBLENDVB Y15, Y11, Y10, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 128(SP)

	// ================= q3' (store FE_Q3) : (need&wide)? wide : orig =========
	MOVQ FE_Q3(AX), SI
	VPMOVZXBW (SI)(DX*1), Y7 // origQ3
	VPXOR Y11, Y11, Y11
	MOVQ FE_P2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p2*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q2*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // q3*2 (orig)
	VPADDW Y7, Y11, Y11
	MOVQ FE_Q4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q4*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q5(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q5*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q6(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q6*4
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $3, Y8, Y8
	VPSUBW Y8, Y11, Y11
	VPSRAW $4, Y11, Y11
	VPBLENDVB Y15, Y11, Y7, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 144(SP)

	// ================= q4' (store FE_Q4) : (need&wide)? wide : orig =========
	MOVQ FE_Q4(AX), SI
	VPMOVZXBW (SI)(DX*1), Y7 // origQ4
	VPXOR Y11, Y11, Y11
	MOVQ FE_P1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q2*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q3*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // q4*2 (orig)
	VPADDW Y7, Y11, Y11
	MOVQ FE_Q5(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q5*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q6(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q6*5
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $3, Y8, Y8
	VPSUBW Y8, Y11, Y11
	VPSRAW $4, Y11, Y11
	VPBLENDVB Y15, Y11, Y7, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 160(SP)

	// ================= q5' (store FE_Q5) : (need&wide)? wide : orig =========
	MOVQ FE_Q5(AX), SI
	VPMOVZXBW (SI)(DX*1), Y7 // origQ5
	VPXOR Y11, Y11, Y11
	MOVQ FE_P0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // p0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q0(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q0*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q1(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q1*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q2(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q2*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q3(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q3*1
	VPADDW Y6, Y11, Y11
	MOVQ FE_Q4(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q4*2
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y7, Y11, Y11      // q5*2 (orig)
	VPADDW Y7, Y11, Y11
	MOVQ FE_Q6(AX), BX
	VPMOVZXBW (BX)(DX*1), Y6 // q6*7
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPADDW Y6, Y11, Y11
	VPCMPEQW Y8, Y8, Y8
	VPSLLW $3, Y8, Y8
	VPSUBW Y8, Y11, Y11
	VPSRAW $4, Y11, Y11
	VPBLENDVB Y15, Y11, Y7, Y11
	VPBLENDVB Y0, Y11, Y7, Y11
	VPACKUSWB Y11, Y11, Y11
	VPERMQ $0xD8, Y11, Y11
	VMOVDQU X11, 176(SP)

	// copy staged outputs to pix rows p5..q5
	MOVQ FE_P5(AX), BX
	VMOVDQU 0(SP), X6
	VMOVDQU X6, (BX)(DX*1)
	MOVQ FE_P4(AX), BX
	VMOVDQU 16(SP), X6
	VMOVDQU X6, (BX)(DX*1)
	MOVQ FE_P3(AX), BX
	VMOVDQU 32(SP), X6
	VMOVDQU X6, (BX)(DX*1)
	MOVQ FE_P2(AX), BX
	VMOVDQU 48(SP), X6
	VMOVDQU X6, (BX)(DX*1)
	MOVQ FE_P1(AX), BX
	VMOVDQU 64(SP), X6
	VMOVDQU X6, (BX)(DX*1)
	MOVQ FE_P0(AX), BX
	VMOVDQU 80(SP), X6
	VMOVDQU X6, (BX)(DX*1)
	MOVQ FE_Q0(AX), BX
	VMOVDQU 96(SP), X6
	VMOVDQU X6, (BX)(DX*1)
	MOVQ FE_Q1(AX), BX
	VMOVDQU 112(SP), X6
	VMOVDQU X6, (BX)(DX*1)
	MOVQ FE_Q2(AX), BX
	VMOVDQU 128(SP), X6
	VMOVDQU X6, (BX)(DX*1)
	MOVQ FE_Q3(AX), BX
	VMOVDQU 144(SP), X6
	VMOVDQU X6, (BX)(DX*1)
	MOVQ FE_Q4(AX), BX
	VMOVDQU 160(SP), X6
	VMOVDQU X6, (BX)(DX*1)
	MOVQ FE_Q5(AX), BX
	VMOVDQU 176(SP), X6
	VMOVDQU X6, (BX)(DX*1)

	ADDQ $16, DX
	DECQ CX
	JMP  feloop

fedone:
	VZEROUPPER
	RET
