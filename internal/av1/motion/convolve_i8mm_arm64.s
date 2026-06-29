// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// I8MM-tier resident lowbd compound-X convolve. This mirrors SVT's
// dist_wtd_convolve_x_8tap_neon_i8mm do_average == 0 branch: halve the even
// AV1 filter taps, stagger f1..f7 for USMMLA, subtract tap 0 separately, add
// the CONV_BUF round shim, then shift by ROUND0_BITS-1.

#define I8_DST       0
#define I8_REF       8
#define I8_FILTER    16
#define I8_PERMUTE   24
#define I8_REFSTR    32
#define I8_WIDTH     40
#define I8_HEIGHT    48
#define I8_ROUNDSHIM 56
#define I8_F0        64

#define XSR_DST      0
#define XSR_REF      8
#define XSR_FILTER   16
#define XSR_PERMUTE  24
#define XSR_DSTSTR   32
#define XSR_REFSTR   40
#define XSR_WIDTH    48
#define XSR_HEIGHT   56
#define XSR_F0       64

// func compoundX8I8MMAsm(ctx *compoundX8I8MMCtx)
TEXT ·compoundX8I8MMAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD I8_DST(R0), R1
	MOVD I8_REF(R0), R2
	MOVD I8_FILTER(R0), R3
	MOVD I8_PERMUTE(R0), R12
	MOVD I8_REFSTR(R0), R5
	MOVD I8_WIDTH(R0), R6
	MOVD I8_HEIGHT(R0), R7
	MOVD I8_ROUNDSHIM(R0), R11
	MOVD I8_F0(R0), R14

	VLD1   (R3), [V0.B16]
	VLD1.P 16(R12), [V28.B16]
	VLD1   (R12), [V29.B16]
	WORD $0x4e020d79 // dup v25.8h, w11     round shim
	WORD $0x0e010dda // dup v26.8b, w14     -tap0 >> 1

i8XRowLoop:
	CBZ  R7, i8XDone
	MOVD R1, R10 // dst cursor
	MOVD R2, R9  // ref tap-window cursor
	MOVD R6, R8  // remaining columns

i8XColLoop:
	VLD1 (R9), [V1.B16]
	WORD $0x4e1c0025 // tbl v5.16b, {v1.16b}, v28.16b
	WORD $0x4e1d0026 // tbl v6.16b, {v1.16b}, v29.16b
	WORD $0x4f000410 // movi v16.4s, #0
	WORD $0x4f000411 // movi v17.4s, #0
	WORD $0x4e80acb0 // usmmla v16.4s, v5.16b, v0.16b
	WORD $0x4e80acd1 // usmmla v17.4s, v6.16b, v0.16b
	WORD $0x0e612a10 // xtn  v16.4h, v16.4s
	WORD $0x4e612a30 // xtn2 v16.8h, v17.4s
	WORD $0x2e3aa030 // umlsl v16.8h, v1.8b, v26.8b
	WORD $0x4e798610 // add  v16.8h, v16.8h, v25.8h
	WORD $0x6f1e0610 // ushr v16.8h, v16.8h, #2
	WORD $0x4c007550 // st1  {v16.8h}, [x10]

	ADD  $16, R10, R10
	ADD  $8, R9, R9
	SUB  $8, R8, R8
	CBNZ R8, i8XColLoop

	ADD  R6<<1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, i8XRowLoop

i8XDone:
	RET

// func convolveX8I8MMAsm(ctx *convolveX8I8MMCtx)
TEXT ·convolveX8I8MMAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD XSR_DST(R0), R1
	MOVD XSR_REF(R0), R2
	MOVD XSR_FILTER(R0), R3
	MOVD XSR_PERMUTE(R0), R12
	MOVD XSR_DSTSTR(R0), R4
	MOVD XSR_REFSTR(R0), R5
	MOVD XSR_WIDTH(R0), R6
	MOVD XSR_HEIGHT(R0), R7
	MOVD XSR_F0(R0), R14

	VLD1   (R3), [V0.B16]
	VLD1.P 16(R12), [V28.B16]
	VLD1   (R12), [V29.B16]
	WORD $0x4f008459 // movi v25.8h, #2      ((1 << (ROUND0_BITS-1)) / 2)
	WORD $0x0e010dda // dup  v26.8b, w14     -tap0 >> 1

xsrI8RowLoop:
	CBZ  R7, xsrI8Done
	MOVD R1, R10 // dst cursor
	MOVD R2, R9  // ref tap-window cursor
	MOVD R6, R8  // remaining columns

xsrI8ColLoop:
	VLD1 (R9), [V1.B16]
	WORD $0x4e1c0025 // tbl v5.16b, {v1.16b}, v28.16b
	WORD $0x4e1d0026 // tbl v6.16b, {v1.16b}, v29.16b
	WORD $0x4f000410 // movi v16.4s, #0
	WORD $0x4f000411 // movi v17.4s, #0
	WORD $0x4e80acb0 // usmmla v16.4s, v5.16b, v0.16b
	WORD $0x4e80acd1 // usmmla v17.4s, v6.16b, v0.16b
	WORD $0x0e612a10 // xtn  v16.4h, v16.4s
	WORD $0x4e612a30 // xtn2 v16.8h, v17.4s
	WORD $0x2e3aa030 // umlsl v16.8h, v1.8b, v26.8b
	WORD $0x4e798610 // add  v16.8h, v16.8h, v25.8h
	WORD $0x2f0a8e10 // sqrshrun v16.8b, v16.8h, #6
	WORD $0x0c007150 // st1  {v16.8b}, [x10]

	ADD  $8, R10, R10
	ADD  $8, R9, R9
	SUB  $8, R8, R8
	CBNZ R8, xsrI8ColLoop

	ADD  R4, R1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, xsrI8RowLoop

xsrI8Done:
	RET
