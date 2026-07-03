// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Code generated with the help of an offline NEON encoder; do not edit by hand.
// Every WORD below is the little-endian encoding of the mnemonic in its
// trailing comment, assembled and verified with the system assembler (clang
// -arch arm64). The kernel is bit-exact with predictFilterIntraBlockDirect16 in
// filter_intra.go; the Go wrapper in filter_intra16_neon_arm64.go restricts it
// to the 16-bit (high-bit-depth) path and resolves the row-pair pointers / edge
// samples.
//
// This mirrors the 8-bit kernel (filter_intra_neon_arm64.s): AV1 filter-intra
// predicts a transform block in 4x2-pixel batches, each applying a 7-tap filter
// (taps depend only on the mode) to the seven neighbour samples p0..p6 to
// produce eight outputs (row0 cols 0-3 in lanes 0-3, row1 cols 0-3 in lanes
// 4-7). Within a row pair the batches march left to right: p1..p4 (top) and p0
// (top-left) slide along the caller-built `above` buffer, while p5/p6 (the left
// column) chain from the previous batch's lane-3 (row0) and lane-7 (row1)
// outputs. Unlike the 8-bit path this accumulates each output in 32-bit lanes
// (smull/smlal, mirroring dav1d ipred16.S ipred_filter_16bpc_neon) so the tap
// sums cannot overflow at 12-bit; sqrshrun #4 = roundPowerOfTwo(sum, 4) then the
// [0,max] clamp low, and smin against the broadcast max clamps high.

//go:build arm64 && !purego

#include "textflag.h"

// func filterIntraRow16NEONAsm(dst0 *byte, dst1 *byte, above *uint16, left0 uint32, left1 uint32, taps *int16, n uintptr, max uint32)
//
// Fills one row pair. n = width/4 batches. taps points at the transposed
// per-mode filter (7 vectors of 8 int16, tap-major: lane k = filter for output
// k). above[0] = p0 of the first batch (top-left), above[1..] = the top row;
// the loader reads 8 halfwords per batch so the caller pads `above` past width.
// max = (1<<bitdepth)-1 broadcast for the high clamp.
TEXT ·filterIntraRow16NEONAsm(SB), NOSPLIT, $0-52
	MOVD  dst0+0(FP), R0
	MOVD  dst1+8(FP), R1
	MOVD  above+16(FP), R2
	MOVWU left0+24(FP), R3
	MOVWU left1+28(FP), R4
	MOVD  taps+32(FP), R5
	MOVD  n+40(FP), R6
	MOVWU max+48(FP), R7

	WORD $0x4cdf24b0 // ld1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x5], #64
	WORD $0x4c4064b4 // ld1 {v20.8h, v21.8h, v22.8h}, [x5]
	WORD $0x4e020cff // dup v31.8h, w7   (max clamp)
	WORD $0x4e0e1c61 // mov v1.h[3], w3  (p5 seed = left0)
	WORD $0x4e1e1c81 // mov v1.h[7], w4  (p6 seed = left1)
filterRow:
	WORD $0x4c407440 // ld1 {v0.8h}, [x2]      (p0..p4 in lanes 0-4)
	WORD $0x0f50a238 // smull  v24.4s, v17.4h, v0.h[1]   (p1*tap1, out0-3)
	WORD $0x4f50a239 // smull2 v25.4s, v17.8h, v0.h[1]   (p1*tap1, out4-7)
	WORD $0x0f602258 // smlal  v24.4s, v18.4h, v0.h[2]   (p2*tap2)
	WORD $0x4f602259 // smlal2 v25.4s, v18.8h, v0.h[2]
	WORD $0x0f702278 // smlal  v24.4s, v19.4h, v0.h[3]   (p3*tap3)
	WORD $0x4f702279 // smlal2 v25.4s, v19.8h, v0.h[3]
	WORD $0x0f402a98 // smlal  v24.4s, v20.4h, v0.h[4]   (p4*tap4)
	WORD $0x4f402a99 // smlal2 v25.4s, v20.8h, v0.h[4]
	WORD $0x0f402218 // smlal  v24.4s, v16.4h, v0.h[0]   (p0*tap0)
	WORD $0x4f402219 // smlal2 v25.4s, v16.8h, v0.h[0]
	WORD $0x0f7122b8 // smlal  v24.4s, v21.4h, v1.h[3]   (p5*tap5)
	WORD $0x4f7122b9 // smlal2 v25.4s, v21.8h, v1.h[3]
	WORD $0x0f712ad8 // smlal  v24.4s, v22.4h, v1.h[7]   (p6*tap6)
	WORD $0x4f712ad9 // smlal2 v25.4s, v22.8h, v1.h[7]
	WORD $0x2f1c8f02 // sqrshrun  v2.4h, v24.4s, #4   (out0-3, clamp low 0)
	WORD $0x6f1c8f22 // sqrshrun2 v2.8h, v25.4s, #4   (out4-7)
	WORD $0x4e7f6c42 // smin v2.8h, v2.8h, v31.8h     (clamp high max)
	WORD $0x0d9f8402 // st1 {v2.d}[0], [x0], #8   (row0 cols 0-3)
	WORD $0x4d9f8422 // st1 {v2.d}[1], [x1], #8   (row1 cols 0-3)
	WORD $0x4ea21c41 // mov v1.16b, v2.16b   (chain: lane3=row0 col3, lane7=row1 col3)
	ADD  $8, R2, R2
	SUB  $1, R6, R6
	CBNZ R6, filterRow
	RET
