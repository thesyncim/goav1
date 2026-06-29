// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// DOTPROD-tier resident lowbd compound-X convolve. This mirrors SVT's
// dist_wtd_convolve_x_neon_dotprod do_average == 0 branch: subtract 128 from
// input samples, halve the even AV1 filter taps, fold the sample-range
// correction and CONV_BUF round offset into the accumulator, then SDOT and
// non-rounding shift by ROUND0_BITS-1.

#define DP_DST        0
#define DP_REF        8
#define DP_KERNEL     16
#define DP_PERMUTE    24
#define DP_REFSTR     32
#define DP_WIDTH      40
#define DP_HEIGHT     48
#define DP_CORRECTION 56

// func compoundX8DotProdAsm(ctx *compoundX8DotProdCtx)
TEXT ·compoundX8DotProdAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD DP_DST(R0), R1
	MOVD DP_REF(R0), R2
	MOVD DP_KERNEL(R0), R3
	MOVD DP_PERMUTE(R0), R12
	MOVD DP_REFSTR(R0), R5
	MOVD DP_WIDTH(R0), R6
	MOVD DP_HEIGHT(R0), R7
	MOVD DP_CORRECTION(R0), R11

	WORD $0x4c407460 // ld1 {v0.8h}, [x3]     load int16 filter taps
	WORD $0x4f1f0400 // sshr v0.8h, v0.8h, #1 halve even filters
	WORD $0x0e212800 // xtn  v0.8b, v0.8h     pack to int8 filter lanes

	VLD1.P 16(R12), [V28.B16]
	VLD1.P 16(R12), [V29.B16]
	VLD1   (R12), [V30.B16]

	WORD $0x4f04e418 // movi v24.16b, #128
	WORD $0x4e040d79 // dup  v25.4s, w11      correction

dpXRowLoop:
	CBZ  R7, dpXDone
	MOVD R1, R10 // dst cursor
	MOVD R2, R9  // ref tap-window cursor
	MOVD R6, R8  // remaining columns

dpXColLoop:
	VLD1 (R9), [V1.B16]
	WORD $0x6e388421 // sub v1.16b, v1.16b, v24.16b
	WORD $0x4e1c0025 // tbl v5.16b, {v1.16b}, v28.16b
	WORD $0x4e1d0026 // tbl v6.16b, {v1.16b}, v29.16b
	WORD $0x4e1e0027 // tbl v7.16b, {v1.16b}, v30.16b

	WORD $0x4eb91f30 // mov v16.16b, v25.16b
	WORD $0x4eb91f31 // mov v17.16b, v25.16b
	WORD $0x4f80e0b0 // sdot v16.4s, v5.16b, v0.4b[0]
	WORD $0x4fa0e0d0 // sdot v16.4s, v6.16b, v0.4b[1]
	WORD $0x4f80e0d1 // sdot v17.4s, v6.16b, v0.4b[0]
	WORD $0x4fa0e0f1 // sdot v17.4s, v7.16b, v0.4b[1]
	WORD $0x0f1e8610 // shrn  v16.4h, v16.4s, #2
	WORD $0x4f1e8630 // shrn2 v16.8h, v17.4s, #2
	WORD $0x4c007550 // st1 {v16.8h}, [x10]

	ADD  $16, R10, R10
	ADD  $8, R9, R9
	SUB  $8, R8, R8
	CBNZ R8, dpXColLoop

	ADD  R6<<1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, dpXRowLoop

dpXDone:
	RET
