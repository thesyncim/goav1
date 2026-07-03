// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// AVX2 8-bit AV1 filter-intra predictor. The kernel is bit-exact with
// predictFilterIntraBlockDirect8 in filter_intra.go; the Go wrapper in
// filter_intra_avx2_amd64.go restricts it to the 8-bit path and resolves the
// row-pair pointers / edge samples. It mirrors the arm64 NEON kernel in
// filter_intra_neon_arm64.s.
//
// AV1 filter-intra predicts a transform block in 4x2-pixel batches. Each batch
// applies a 7-tap filter (taps depend only on the mode) to the seven neighbour
// samples p0..p6 and produces eight outputs (row0 cols 0-3 in lanes 0-3, row1
// cols 0-3 in lanes 4-7). Within a row pair the batches march left to right:
// p1..p4 (top) and p0 (top-left) slide along the caller-built `above` buffer,
// while p5/p6 (the left column) chain from the previous batch's lane-3 (row0)
// and lane-7 (row1) outputs. The rounding is (sum+8)>>4 (VPADDW +8, VPSRAW #4,
// arithmetic) then VPACKUSWB's signed-16 -> unsigned-8 saturation to [0,255],
// matching roundPowerOfTwo(sum, 4) then the [0,255] clamp.
//
// dav1d reference: third_party/upstream/dav1d/src/x86/ipred_avx2.asm
// (ipred_filter_8bpc).

//go:build amd64 && !purego

#include "textflag.h"

// func filterIntraRow8AVX2Asm(dst0 *byte, dst1 *byte, above *uint8, left0 uint32, left1 uint32, taps *int16, n uint64)
//
// Fills one row pair. n = width/4 batches. taps points at the transposed
// per-mode filter (7 vectors of 8 int16, tap-major: lane k = filter for output
// k). above[0] = p0 of the first batch (top-left), above[1..] = the top row.
TEXT ·filterIntraRow8AVX2Asm(SB), NOSPLIT, $0-48
	MOVQ  dst0+0(FP), DI
	MOVQ  dst1+8(FP), R8
	MOVQ  above+16(FP), SI
	MOVLQZX left0+24(FP), R9
	MOVLQZX left1+28(FP), R10
	MOVQ  taps+32(FP), R11
	MOVQ  n+40(FP), R12

	VMOVDQU (R11), X1     // tap0
	VMOVDQU 16(R11), X2   // tap1
	VMOVDQU 32(R11), X3   // tap2
	VMOVDQU 48(R11), X4   // tap3
	VMOVDQU 64(R11), X5   // tap4
	VMOVDQU 80(R11), X6   // tap5
	VMOVDQU 96(R11), X7   // tap6

	MOVL         $8, R13
	MOVQ         R13, X8
	VPBROADCASTW X8, X8   // rounding constant 8 (8 i16 lanes)

filterRow:
	// p0*tap0
	MOVBLZX      0(SI), R13
	MOVQ         R13, X9
	VPBROADCASTW X9, X9
	VPMULLW      X1, X9, X0
	// p1*tap1
	MOVBLZX      1(SI), R13
	MOVQ         R13, X9
	VPBROADCASTW X9, X9
	VPMULLW      X2, X9, X9
	VPADDW       X9, X0, X0
	// p2*tap2
	MOVBLZX      2(SI), R13
	MOVQ         R13, X9
	VPBROADCASTW X9, X9
	VPMULLW      X3, X9, X9
	VPADDW       X9, X0, X0
	// p3*tap3
	MOVBLZX      3(SI), R13
	MOVQ         R13, X9
	VPBROADCASTW X9, X9
	VPMULLW      X4, X9, X9
	VPADDW       X9, X0, X0
	// p4*tap4
	MOVBLZX      4(SI), R13
	MOVQ         R13, X9
	VPBROADCASTW X9, X9
	VPMULLW      X5, X9, X9
	VPADDW       X9, X0, X0
	// p5*tap5 (left col row0)
	MOVQ         R9, X9
	VPBROADCASTW X9, X9
	VPMULLW      X6, X9, X9
	VPADDW       X9, X0, X0
	// p6*tap6 (left col row1)
	MOVQ         R10, X9
	VPBROADCASTW X9, X9
	VPMULLW      X7, X9, X9
	VPADDW       X9, X0, X0

	VPADDW    X8, X0, X0   // +8
	VPSRAW    $4, X0, X0   // >>4 (arithmetic)
	VPACKUSWB X0, X0, X0   // saturate i16 -> u8 [0,255]; bytes 0-7 valid

	VPEXTRD $0, X0, (DI)   // row0 cols 0-3
	VPEXTRD $1, X0, (R8)   // row1 cols 0-3

	// chain: p5 next = row0 col3 (byte 3), p6 next = row1 col3 (byte 7)
	VPEXTRB $3, X0, R9
	VPEXTRB $7, X0, R10

	ADDQ $4, SI
	ADDQ $4, DI
	ADDQ $4, R8
	SUBQ $1, R12
	JNZ  filterRow
	VZEROUPPER
	RET
