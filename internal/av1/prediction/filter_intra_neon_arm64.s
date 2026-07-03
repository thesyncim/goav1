// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Code generated with the help of an offline NEON encoder; do not edit by hand.
// Every WORD below is the little-endian encoding of the mnemonic in its
// trailing comment, assembled and verified with the system assembler (clang
// -arch arm64). The kernel is bit-exact with predictFilterIntraBlockDirect8 in
// filter_intra.go; the Go wrapper in filter_intra_neon_arm64.go restricts it to
// the 8-bit path and resolves the row-pair pointers / edge samples.
//
// AV1 filter-intra predicts a transform block in 4x2-pixel batches. Each batch
// applies a 7-tap filter (taps depend only on the mode) to the seven neighbour
// samples p0..p6 and produces eight outputs (row0 cols 0-3 in lanes 0-3, row1
// cols 0-3 in lanes 4-7). Within a row pair the batches march left to right:
// p1..p4 (top) and p0 (top-left) slide along the caller-built `above` buffer,
// while p5/p6 (the left column) chain from the previous batch's lane-3 (row0)
// and lane-7 (row1) outputs. The rounding is sqrshrun #4 = clamp((sum+8)>>4,
// 0, 255), matching roundPowerOfTwo(sum, 4) then the [0,255] clamp.

//go:build arm64 && !purego

#include "textflag.h"

// func filterIntraRow8NEONAsm(dst0 *byte, dst1 *byte, above *uint8, left0 uint32, left1 uint32, taps *int16, n uintptr)
//
// Fills one row pair. n = width/4 batches. taps points at the transposed
// per-mode filter (7 vectors of 8 int16, tap-major: lane k = filter for output
// k). above[0] = p0 of the first batch (top-left), above[1..] = the top row;
// the loader reads 8 bytes per batch so the caller pads `above` past width.
TEXT ·filterIntraRow8NEONAsm(SB), NOSPLIT, $0-48
	MOVD  dst0+0(FP), R0
	MOVD  dst1+8(FP), R1
	MOVD  above+16(FP), R2
	MOVWU left0+24(FP), R3
	MOVWU left1+28(FP), R4
	MOVD  taps+32(FP), R5
	MOVD  n+40(FP), R6

	WORD $0x4cdf24b0 // ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x5], #64
	WORD $0x4c4064b4 // ld1 {v20.8h, v21.8h, v22.8h}, [x5]
	WORD $0x4e0e1c61 // mov v1.h[3], w3   (p5 seed = left0)
	WORD $0x4e1e1c81 // mov v1.h[7], w4   (p6 seed = left1)
filterRow:
	WORD $0x0c407040 // ld1 {v0.8b}, [x2]      (p0..p4 in lanes 0-4)
	WORD $0x2f08a400 // uxtl v0.8h, v0.8b
	WORD $0x4f508222 // mul  v2.8h, v17.8h, v0.h[1]   (p1*tap1)
	WORD $0x6f600242 // mla  v2.8h, v18.8h, v0.h[2]   (p2*tap2)
	WORD $0x6f700262 // mla  v2.8h, v19.8h, v0.h[3]   (p3*tap3)
	WORD $0x6f400a82 // mla  v2.8h, v20.8h, v0.h[4]   (p4*tap4)
	WORD $0x6f400202 // mla  v2.8h, v16.8h, v0.h[0]   (p0*tap0)
	WORD $0x6f7102a2 // mla  v2.8h, v21.8h, v1.h[3]   (p5*tap5)
	WORD $0x6f710ac2 // mla  v2.8h, v22.8h, v1.h[7]   (p6*tap6)
	WORD $0x2f0c8c42 // sqrshrun v2.8b, v2.8h, #4
	WORD $0x0d9f8002 // st1 {v2.s}[0], [x0], #4   (row0 cols 0-3)
	WORD $0x0d9f9022 // st1 {v2.s}[1], [x1], #4   (row1 cols 0-3)
	WORD $0x2f08a441 // uxtl v1.8h, v2.8b   (chain: lane3=row0 col3, lane7=row1 col3)
	ADD  $4, R2, R2
	SUB  $1, R6, R6
	CBNZ R6, filterRow
	RET
