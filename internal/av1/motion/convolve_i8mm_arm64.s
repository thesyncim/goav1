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

#define YSR_DST      0
#define YSR_REF      8
#define YSR_FILTER   16
#define YSR_MERGE    24
#define YSR_DSTSTR   32
#define YSR_REFSTR   40
#define YSR_WIDTH    48
#define YSR_HEIGHT   56

#define CYI8_DST         0
#define CYI8_REF         8
#define CYI8_FILTER      16
#define CYI8_MERGE       24
#define CYI8_DSTSTR      32
#define CYI8_REFSTR      40
#define CYI8_WIDTH       48
#define CYI8_HEIGHT      56
#define CYI8_ROUNDOFFSET 64

#define C2DI8_DST     0
#define C2DI8_REF     8
#define C2DI8_XFILTER 16
#define C2DI8_PERMUTE 24
#define C2DI8_YKERNEL 32
#define C2DI8_REFSTR  40
#define C2DI8_WIDTH   48
#define C2DI8_HEIGHT  56
#define C2DI8_IM      64
#define C2DI8_IMSTR   72
#define C2DI8_F0      80

#define T2D_DST      0
#define T2D_REF      8
#define T2D_XFILTER  16
#define T2D_PERMUTE  24
#define T2D_YKERNEL  32
#define T2D_DSTSTR   40
#define T2D_REFSTR   48
#define T2D_WIDTH    56
#define T2D_HEIGHT   64
#define T2D_IM       72
#define T2D_IMSTR    80
#define T2D_F0       88

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

// func compoundY8I8MMAsm(ctx *compoundY8I8MMCtx)
//
// Resident lowbd compound-Y convolve, 8-tap/6-tap tier. This mirrors the
// transposed USDOT path used by convolveY8I8MMAsm, but stores CONV_BUF output:
// round((sum / 2), ROUND0_BITS-1) + roundOffset.
TEXT ·compoundY8I8MMAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD CYI8_DST(R0), R1
	MOVD CYI8_REF(R0), R2
	MOVD CYI8_FILTER(R0), R3
	MOVD CYI8_MERGE(R0), R12
	MOVD CYI8_DSTSTR(R0), R4
	MOVD CYI8_REFSTR(R0), R5
	MOVD CYI8_WIDTH(R0), R6
	MOVD CYI8_HEIGHT(R0), R7
	MOVD CYI8_ROUNDOFFSET(R0), R11

	VLD1   (R3), [V0.B16]
	VLD1.P 16(R12), [V28.B16]
	VLD1.P 16(R12), [V29.B16]
	VLD1   (R12), [V30.B16]
	WORD $0x4e020d7f     // dup v31.8h, w11  roundOffset

	CMP $4, R6
	BEQ cy8I8W4Col

cy8I8ColLoop:
	MOVD R2, R9
	MOVD R1, R10
	MOVD R7, R11

	WORD $0x0cc57121
	WORD $0x0cc57122
	WORD $0x0cc57123
	WORD $0x0cc57124
	WORD $0x0cc57125
	WORD $0x0cc57126
	WORD $0x0cc57127

	WORD $0x4e03383a
	WORD $0x4e04385b
	WORD $0x4e1b3b48
	WORD $0x4e1b7b49
	WORD $0x4e04385a
	WORD $0x4e05387b
	WORD $0x4e1b3b4a
	WORD $0x4e1b7b4b
	WORD $0x4e05387a
	WORD $0x4e06389b
	WORD $0x4e1b3b4c
	WORD $0x4e1b7b4d
	WORD $0x4e06389a
	WORD $0x4e0738bb
	WORD $0x4e1b3b4e
	WORD $0x4e1b7b52

cy8I8Row4Loop:
	WORD $0x0cc57121
	WORD $0x0cc57122
	WORD $0x0cc57123
	WORD $0x0cc57124
	WORD $0x4e03383a
	WORD $0x4e04385b
	WORD $0x4e1b3b4f
	WORD $0x4e1b7b53

	WORD $0x4e1c21d4
	WORD $0x4e1c2255
	WORD $0x4e1d21d6
	WORD $0x4e1d2257
	WORD $0x4e1e21d8
	WORD $0x4e1e2259

	WORD $0x4f000410
	WORD $0x4f000411
	WORD $0x4f80f110
	WORD $0x4fa0f290
	WORD $0x4f80f131
	WORD $0x4fa0f2b1
	WORD $0x4f3e2610     // srshr v16.4s, v16.4s, #2
	WORD $0x4f3e2631     // srshr v17.4s, v17.4s, #2
	WORD $0x0e614a10
	WORD $0x4e614a30
	WORD $0x4e7f8610     // add v16.8h, v16.8h, v31.8h
	WORD $0x4c007550
	ADD  R4, R10, R10

	WORD $0x4f000410
	WORD $0x4f000411
	WORD $0x4f80f150
	WORD $0x4fa0f2d0
	WORD $0x4f80f171
	WORD $0x4fa0f2f1
	WORD $0x4f3e2610
	WORD $0x4f3e2631
	WORD $0x0e614a10
	WORD $0x4e614a30
	WORD $0x4e7f8610
	WORD $0x4c007550
	ADD  R4, R10, R10

	WORD $0x4f000410
	WORD $0x4f000411
	WORD $0x4f80f190
	WORD $0x4fa0f310
	WORD $0x4f80f1b1
	WORD $0x4fa0f331
	WORD $0x4f3e2610
	WORD $0x4f3e2631
	WORD $0x0e614a10
	WORD $0x4e614a30
	WORD $0x4e7f8610
	WORD $0x4c007550
	ADD  R4, R10, R10

	WORD $0x4f000410
	WORD $0x4f000411
	WORD $0x4f80f1d0
	WORD $0x4fa0f1f0
	WORD $0x4f80f251
	WORD $0x4fa0f271
	WORD $0x4f3e2610
	WORD $0x4f3e2631
	WORD $0x0e614a10
	WORD $0x4e614a30
	WORD $0x4e7f8610
	WORD $0x4c007550
	ADD  R4, R10, R10

	WORD $0x4eb41e88
	WORD $0x4eb51ea9
	WORD $0x4eb61eca
	WORD $0x4eb71eeb
	WORD $0x4eb81f0c
	WORD $0x4eb91f2d
	WORD $0x4eaf1dee
	WORD $0x4eb31e72

	SUB  $4, R11, R11
	CBNZ R11, cy8I8Row4Loop

	ADD  $8, R2, R2
	ADD  $16, R1, R1
	SUB  $8, R6, R6
	CBNZ R6, cy8I8ColLoop
	RET

cy8I8W4Col:
	MOVD R2, R9
	MOVD R1, R10
	MOVD R7, R11

	WORD $0x0dc58121
	WORD $0x0dc58122
	WORD $0x0dc58123
	WORD $0x0dc58124
	WORD $0x0dc58125
	WORD $0x0dc58126
	WORD $0x0dc58127

	WORD $0x4e03383a
	WORD $0x4e04385b
	WORD $0x4e1b3b48
	WORD $0x4e04385a
	WORD $0x4e05387b
	WORD $0x4e1b3b4a
	WORD $0x4e05387a
	WORD $0x4e06389b
	WORD $0x4e1b3b4c
	WORD $0x4e06389a
	WORD $0x4e0738bb
	WORD $0x4e1b3b4e

cy8I8W4RowLoop:
	WORD $0x0dc58121
	WORD $0x0dc58122
	WORD $0x0dc58123
	WORD $0x0dc58124
	WORD $0x4e03383a
	WORD $0x4e04385b
	WORD $0x4e1b3b4f

	WORD $0x4e1c21d4
	WORD $0x4e1d21d6
	WORD $0x4e1e21d8

	WORD $0x4f000410
	WORD $0x4f80f110
	WORD $0x4fa0f290
	WORD $0x4f3e2610
	WORD $0x0e614a10
	WORD $0x4e7f8610
	WORD $0x0c007550
	ADD  R4, R10, R10

	WORD $0x4f000410
	WORD $0x4f80f150
	WORD $0x4fa0f2d0
	WORD $0x4f3e2610
	WORD $0x0e614a10
	WORD $0x4e7f8610
	WORD $0x0c007550
	ADD  R4, R10, R10

	WORD $0x4f000410
	WORD $0x4f80f190
	WORD $0x4fa0f310
	WORD $0x4f3e2610
	WORD $0x0e614a10
	WORD $0x4e7f8610
	WORD $0x0c007550
	ADD  R4, R10, R10

	WORD $0x4f000410
	WORD $0x4f80f1d0
	WORD $0x4fa0f1f0
	WORD $0x4f3e2610
	WORD $0x0e614a10
	WORD $0x4e7f8610
	WORD $0x0c007550
	ADD  R4, R10, R10

	WORD $0x4eb41e88
	WORD $0x4eb61eca
	WORD $0x4eb81f0c
	WORD $0x4eaf1dee

	SUB  $4, R11, R11
	CBNZ R11, cy8I8W4RowLoop
	RET

// func compoundY4TapI8MMAsm(ctx *compoundY8I8MMCtx)
//
// Resident lowbd compound-Y convolve, 4-tap tier. Mirrors
// convolveY4TapI8MMAsm with CONV_BUF rounding and uint16 stores.
TEXT ·compoundY4TapI8MMAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD CYI8_DST(R0), R1
	MOVD CYI8_REF(R0), R2
	MOVD CYI8_FILTER(R0), R3
	MOVD CYI8_MERGE(R0), R12
	MOVD CYI8_DSTSTR(R0), R4
	MOVD CYI8_REFSTR(R0), R5
	MOVD CYI8_WIDTH(R0), R6
	MOVD CYI8_HEIGHT(R0), R7
	MOVD CYI8_ROUNDOFFSET(R0), R11

	VLD1   (R3), [V0.B16]
	VLD1.P 16(R12), [V28.B16]
	VLD1.P 16(R12), [V29.B16]
	VLD1   (R12), [V30.B16]
	WORD $0x4e020d7f     // dup v31.8h, w11

	CMP $4, R6
	BEQ cy4I8W4Col

cy4I8ColLoop:
	MOVD R2, R9
	MOVD R1, R10
	MOVD R7, R11

	WORD $0x0cc57124
	WORD $0x0cc57125
	WORD $0x0cc57126
	WORD $0x0cc57127
	WORD $0x4e06389a
	WORD $0x4e0738bb
	WORD $0x4e1b3b4e
	WORD $0x4e1b7b52

cy4I8Row4Loop:
	WORD $0x0cc57121
	WORD $0x0cc57122
	WORD $0x0cc57123
	WORD $0x0cc57124
	WORD $0x4e03383a
	WORD $0x4e04385b
	WORD $0x4e1b3b4f
	WORD $0x4e1b7b53

	WORD $0x4e1c21d4
	WORD $0x4e1c2255
	WORD $0x4e1d21d6
	WORD $0x4e1d2257
	WORD $0x4e1e21d8
	WORD $0x4e1e2259

	WORD $0x4f000410
	WORD $0x4f000411
	WORD $0x4f80f1d0
	WORD $0x4f80f251
	WORD $0x4f3e2610
	WORD $0x4f3e2631
	WORD $0x0e614a10
	WORD $0x4e614a30
	WORD $0x4e7f8610
	WORD $0x4c007550
	ADD  R4, R10, R10

	WORD $0x4f000410
	WORD $0x4f000411
	WORD $0x4f80f290
	WORD $0x4f80f2b1
	WORD $0x4f3e2610
	WORD $0x4f3e2631
	WORD $0x0e614a10
	WORD $0x4e614a30
	WORD $0x4e7f8610
	WORD $0x4c007550
	ADD  R4, R10, R10

	WORD $0x4f000410
	WORD $0x4f000411
	WORD $0x4f80f2d0
	WORD $0x4f80f2f1
	WORD $0x4f3e2610
	WORD $0x4f3e2631
	WORD $0x0e614a10
	WORD $0x4e614a30
	WORD $0x4e7f8610
	WORD $0x4c007550
	ADD  R4, R10, R10

	WORD $0x4f000410
	WORD $0x4f000411
	WORD $0x4f80f310
	WORD $0x4f80f331
	WORD $0x4f3e2610
	WORD $0x4f3e2631
	WORD $0x0e614a10
	WORD $0x4e614a30
	WORD $0x4e7f8610
	WORD $0x4c007550
	ADD  R4, R10, R10

	WORD $0x4eaf1dee
	WORD $0x4eb31e72

	SUB  $4, R11, R11
	CBNZ R11, cy4I8Row4Loop

	ADD  $8, R2, R2
	ADD  $16, R1, R1
	SUB  $8, R6, R6
	CBNZ R6, cy4I8ColLoop
	RET

cy4I8W4Col:
	MOVD R2, R9
	MOVD R1, R10
	MOVD R7, R11

	WORD $0x0dc58124
	WORD $0x0dc58125
	WORD $0x0dc58126
	WORD $0x0dc58127
	WORD $0x4e06389a
	WORD $0x4e0738bb
	WORD $0x4e1b3b4e

cy4I8W4RowLoop:
	WORD $0x0dc58121
	WORD $0x0dc58122
	WORD $0x0dc58123
	WORD $0x0dc58124
	WORD $0x4e03383a
	WORD $0x4e04385b
	WORD $0x4e1b3b4f

	WORD $0x4e1c21d4
	WORD $0x4e1d21d6
	WORD $0x4e1e21d8

	WORD $0x4f000410
	WORD $0x4f80f1d0
	WORD $0x4f3e2610
	WORD $0x0e614a10
	WORD $0x4e7f8610
	WORD $0x0c007550
	ADD  R4, R10, R10

	WORD $0x4f000410
	WORD $0x4f80f290
	WORD $0x4f3e2610
	WORD $0x0e614a10
	WORD $0x4e7f8610
	WORD $0x0c007550
	ADD  R4, R10, R10

	WORD $0x4f000410
	WORD $0x4f80f2d0
	WORD $0x4f3e2610
	WORD $0x0e614a10
	WORD $0x4e7f8610
	WORD $0x0c007550
	ADD  R4, R10, R10

	WORD $0x4f000410
	WORD $0x4f80f310
	WORD $0x4f3e2610
	WORD $0x0e614a10
	WORD $0x4e7f8610
	WORD $0x0c007550
	ADD  R4, R10, R10

	WORD $0x4eaf1dee

	SUB  $4, R11, R11
	CBNZ R11, cy4I8W4RowLoop
	RET

// func compound2D8I8MMAsm(ctx *compound2D8I8MMCtx)
//
// Resident width>=8 lowbd compound 2D convolve. The horizontal pass mirrors
// SVT's dist_wtd_convolve_2d_horiz_8tap_neon_i8mm and the local
// convolve2D8I8MMAsm pass: halve the even AV1 taps, use USMMLA, subtract tap 0
// separately when it is nonzero, add the ROUND0 shim, then shift by
// ROUND0_BITS-1 into int16. Regular and smooth filters take a zero-tap path
// that omits that multiply. The vertical pass writes four adjacent rows from
// one 11-row sliding window, then handles any defensive tail, at CONV_BUF
// precision:
//     dst = round((1 << offsetBits) + vertical_sum, COMPOUND_ROUND1_BITS)
TEXT ·compound2D8I8MMAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C2DI8_DST(R0), R1
	MOVD C2DI8_REF(R0), R2
	MOVD C2DI8_XFILTER(R0), R3
	MOVD C2DI8_PERMUTE(R0), R12
	MOVD C2DI8_REFSTR(R0), R5
	MOVD C2DI8_WIDTH(R0), R6
	MOVD C2DI8_HEIGHT(R0), R7
	MOVD C2DI8_IM(R0), R13
	MOVD C2DI8_IMSTR(R0), R14
	MOVD C2DI8_F0(R0), R15
	LSL  $1, R14, R14

	ADD  $7, R7, R11

	VLD1   (R3), [V0.B16]
	VLD1.P 16(R12), [V28.B16]
	VLD1   (R12), [V29.B16]
	MOVD   $8194, R3
	WORD   $0x4e020c79 // dup v25.8h, w3
	CBNZ   R15, ct2dHNonZeroStart

	MOVD R2, R16
	MOVD R13, R17

ct2dHZeroRowLoop:
	MOVD R16, R9
	MOVD R17, R10
	MOVD R6, R8

ct2dHZeroColLoop:
	VLD1 (R9), [V1.B16]
	WORD $0x4e1c0025 // tbl v5.16b, {v1.16b}, v28.16b
	WORD $0x4e1d0026 // tbl v6.16b, {v1.16b}, v29.16b
	WORD $0x4f000410 // movi v16.4s, #0
	WORD $0x4f000411 // movi v17.4s, #0
	WORD $0x4e80acb0 // usmmla v16.4s, v5.16b, v0.16b
	WORD $0x4e80acd1 // usmmla v17.4s, v6.16b, v0.16b
	WORD $0x0e612a10 // xtn  v16.4h, v16.4s
	WORD $0x4e612a30 // xtn2 v16.8h, v17.4s
	WORD $0x4e798610 // add  v16.8h, v16.8h, v25.8h
	WORD $0x6f1e0610 // ushr v16.8h, v16.8h, #2
	WORD $0x4c007550 // st1  {v16.8h}, [x10]

	ADD  $8, R9, R9
	ADD  $16, R10, R10
	SUB  $8, R8, R8
	CBNZ R8, ct2dHZeroColLoop

	ADD  R5, R16, R16
	ADD  R14, R17, R17
	SUB  $1, R11, R11
	CBNZ R11, ct2dHZeroRowLoop
	B    ct2dHDone

ct2dHNonZeroStart:
	WORD $0x0e010dfa // dup v26.8b, w15
	MOVD R2, R16
	MOVD R13, R17

ct2dHNonZeroRowLoop:
	MOVD R16, R9
	MOVD R17, R10
	MOVD R6, R8

ct2dHNonZeroColLoop:
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

	ADD  $8, R9, R9
	ADD  $16, R10, R10
	SUB  $8, R8, R8
	CBNZ R8, ct2dHNonZeroColLoop

	ADD  R5, R16, R16
	ADD  R14, R17, R17
	SUB  $1, R11, R11
	CBNZ R11, ct2dHNonZeroRowLoop

ct2dHDone:
	MOVD C2DI8_YKERNEL(R0), R3
	WORD $0x4c407460 // ld1 {v0.8h}, [x3]
	MOVD $524288, R11
	WORD $0x4e040d72 // dup v18.4s, w11

	MOVD R13, R17
	CMP  $4, R7
	BLT  ct2dVTail

ct2dVRow4Loop:
	MOVD R1, R10
	MOVD R17, R11
	MOVD R6, R8

ct2dVCol4Loop:
	MOVD R11, R9
	WORD $0x4eb21e50 // mov v16.16b, v18.16b
	WORD $0x4eb21e51 // mov v17.16b, v18.16b
	WORD $0x4eb21e53 // mov v19.16b, v18.16b
	WORD $0x4eb21e54 // mov v20.16b, v18.16b
	WORD $0x4eb21e55 // mov v21.16b, v18.16b
	WORD $0x4eb21e56 // mov v22.16b, v18.16b
	WORD $0x4eb21e57 // mov v23.16b, v18.16b
	WORD $0x4eb21e58 // mov v24.16b, v18.16b

	WORD $0x4cce7521 // ld1 {v1.8h}, [x9], x14
	WORD $0x4cce7522 // ld1 {v2.8h}, [x9], x14
	WORD $0x4cce7523 // ld1 {v3.8h}, [x9], x14
	WORD $0x4cce7524 // ld1 {v4.8h}, [x9], x14
	WORD $0x4cce7525 // ld1 {v5.8h}, [x9], x14
	WORD $0x4cce7526 // ld1 {v6.8h}, [x9], x14
	WORD $0x4cce7527 // ld1 {v7.8h}, [x9], x14
	WORD $0x4cce7528 // ld1 {v8.8h}, [x9], x14
	WORD $0x4cce7529 // ld1 {v9.8h}, [x9], x14
	WORD $0x4cce752a // ld1 {v10.8h}, [x9], x14
	WORD $0x4cce752b // ld1 {v11.8h}, [x9], x14

	// Four adjacent outputs share this 11-row window. Schedule each tap
	// across eight independent accumulators before returning to an output.
	WORD $0x0f402030 // smlal  v16.4s, v1.4h, v0.h[0]
	WORD $0x0f402053 // smlal  v19.4s, v2.4h, v0.h[0]
	WORD $0x0f402075 // smlal  v21.4s, v3.4h, v0.h[0]
	WORD $0x0f402097 // smlal  v23.4s, v4.4h, v0.h[0]
	WORD $0x4f402031 // smlal2 v17.4s, v1.8h, v0.h[0]
	WORD $0x4f402054 // smlal2 v20.4s, v2.8h, v0.h[0]
	WORD $0x4f402076 // smlal2 v22.4s, v3.8h, v0.h[0]
	WORD $0x4f402098 // smlal2 v24.4s, v4.8h, v0.h[0]
	WORD $0x0f502050 // smlal  v16.4s, v2.4h, v0.h[1]
	WORD $0x0f502073 // smlal  v19.4s, v3.4h, v0.h[1]
	WORD $0x0f502095 // smlal  v21.4s, v4.4h, v0.h[1]
	WORD $0x0f5020b7 // smlal  v23.4s, v5.4h, v0.h[1]
	WORD $0x4f502051 // smlal2 v17.4s, v2.8h, v0.h[1]
	WORD $0x4f502074 // smlal2 v20.4s, v3.8h, v0.h[1]
	WORD $0x4f502096 // smlal2 v22.4s, v4.8h, v0.h[1]
	WORD $0x4f5020b8 // smlal2 v24.4s, v5.8h, v0.h[1]
	WORD $0x0f602070 // smlal  v16.4s, v3.4h, v0.h[2]
	WORD $0x0f602093 // smlal  v19.4s, v4.4h, v0.h[2]
	WORD $0x0f6020b5 // smlal  v21.4s, v5.4h, v0.h[2]
	WORD $0x0f6020d7 // smlal  v23.4s, v6.4h, v0.h[2]
	WORD $0x4f602071 // smlal2 v17.4s, v3.8h, v0.h[2]
	WORD $0x4f602094 // smlal2 v20.4s, v4.8h, v0.h[2]
	WORD $0x4f6020b6 // smlal2 v22.4s, v5.8h, v0.h[2]
	WORD $0x4f6020d8 // smlal2 v24.4s, v6.8h, v0.h[2]
	WORD $0x0f702090 // smlal  v16.4s, v4.4h, v0.h[3]
	WORD $0x0f7020b3 // smlal  v19.4s, v5.4h, v0.h[3]
	WORD $0x0f7020d5 // smlal  v21.4s, v6.4h, v0.h[3]
	WORD $0x0f7020f7 // smlal  v23.4s, v7.4h, v0.h[3]
	WORD $0x4f702091 // smlal2 v17.4s, v4.8h, v0.h[3]
	WORD $0x4f7020b4 // smlal2 v20.4s, v5.8h, v0.h[3]
	WORD $0x4f7020d6 // smlal2 v22.4s, v6.8h, v0.h[3]
	WORD $0x4f7020f8 // smlal2 v24.4s, v7.8h, v0.h[3]
	WORD $0x0f4028b0 // smlal  v16.4s, v5.4h, v0.h[4]
	WORD $0x0f4028d3 // smlal  v19.4s, v6.4h, v0.h[4]
	WORD $0x0f4028f5 // smlal  v21.4s, v7.4h, v0.h[4]
	WORD $0x0f402917 // smlal  v23.4s, v8.4h, v0.h[4]
	WORD $0x4f4028b1 // smlal2 v17.4s, v5.8h, v0.h[4]
	WORD $0x4f4028d4 // smlal2 v20.4s, v6.8h, v0.h[4]
	WORD $0x4f4028f6 // smlal2 v22.4s, v7.8h, v0.h[4]
	WORD $0x4f402918 // smlal2 v24.4s, v8.8h, v0.h[4]
	WORD $0x0f5028d0 // smlal  v16.4s, v6.4h, v0.h[5]
	WORD $0x0f5028f3 // smlal  v19.4s, v7.4h, v0.h[5]
	WORD $0x0f502915 // smlal  v21.4s, v8.4h, v0.h[5]
	WORD $0x0f502937 // smlal  v23.4s, v9.4h, v0.h[5]
	WORD $0x4f5028d1 // smlal2 v17.4s, v6.8h, v0.h[5]
	WORD $0x4f5028f4 // smlal2 v20.4s, v7.8h, v0.h[5]
	WORD $0x4f502916 // smlal2 v22.4s, v8.8h, v0.h[5]
	WORD $0x4f502938 // smlal2 v24.4s, v9.8h, v0.h[5]
	WORD $0x0f6028f0 // smlal  v16.4s, v7.4h, v0.h[6]
	WORD $0x0f602913 // smlal  v19.4s, v8.4h, v0.h[6]
	WORD $0x0f602935 // smlal  v21.4s, v9.4h, v0.h[6]
	WORD $0x0f602957 // smlal  v23.4s, v10.4h, v0.h[6]
	WORD $0x4f6028f1 // smlal2 v17.4s, v7.8h, v0.h[6]
	WORD $0x4f602914 // smlal2 v20.4s, v8.8h, v0.h[6]
	WORD $0x4f602936 // smlal2 v22.4s, v9.8h, v0.h[6]
	WORD $0x4f602958 // smlal2 v24.4s, v10.8h, v0.h[6]
	WORD $0x0f702910 // smlal  v16.4s, v8.4h, v0.h[7]
	WORD $0x0f702933 // smlal  v19.4s, v9.4h, v0.h[7]
	WORD $0x0f702955 // smlal  v21.4s, v10.4h, v0.h[7]
	WORD $0x0f702977 // smlal  v23.4s, v11.4h, v0.h[7]
	WORD $0x4f702911 // smlal2 v17.4s, v8.8h, v0.h[7]
	WORD $0x4f702934 // smlal2 v20.4s, v9.8h, v0.h[7]
	WORD $0x4f702956 // smlal2 v22.4s, v10.8h, v0.h[7]
	WORD $0x4f702978 // smlal2 v24.4s, v11.8h, v0.h[7]

	WORD $0x4f392610 // srshr v16.4s, v16.4s, #7
	WORD $0x4f392631 // srshr v17.4s, v17.4s, #7
	WORD $0x4f392673 // srshr v19.4s, v19.4s, #7
	WORD $0x4f392694 // srshr v20.4s, v20.4s, #7
	WORD $0x4f3926b5 // srshr v21.4s, v21.4s, #7
	WORD $0x4f3926d6 // srshr v22.4s, v22.4s, #7
	WORD $0x4f3926f7 // srshr v23.4s, v23.4s, #7
	WORD $0x4f392718 // srshr v24.4s, v24.4s, #7
	WORD $0x0e614a10 // sqxtn  v16.4h, v16.4s
	WORD $0x4e614a30 // sqxtn2 v16.8h, v17.4s
	WORD $0x0e614a73 // sqxtn  v19.4h, v19.4s
	WORD $0x4e614a93 // sqxtn2 v19.8h, v20.4s
	WORD $0x0e614ab5 // sqxtn  v21.4h, v21.4s
	WORD $0x4e614ad5 // sqxtn2 v21.8h, v22.4s
	WORD $0x0e614af7 // sqxtn  v23.4h, v23.4s
	WORD $0x4e614b17 // sqxtn2 v23.8h, v24.4s

	MOVD R10, R9
	WORD $0x4c007530 // st1 {v16.8h}, [x9]
	ADD  R6<<1, R9, R9
	WORD $0x4c007533 // st1 {v19.8h}, [x9]
	ADD  R6<<1, R9, R9
	WORD $0x4c007535 // st1 {v21.8h}, [x9]
	ADD  R6<<1, R9, R9
	WORD $0x4c007537 // st1 {v23.8h}, [x9]

	ADD  $16, R10, R10
	ADD  $16, R11, R11
	SUB  $8, R8, R8
	CBNZ R8, ct2dVCol4Loop

	ADD R6<<3, R1, R1
	ADD R14<<2, R17, R17
	SUB $4, R7, R7
	CMP $4, R7
	BGE ct2dVRow4Loop

ct2dVTail:
	CBZ R7, ct2dVDone

ct2dVTailRowLoop:
	MOVD R1, R10
	MOVD R17, R11
	MOVD R6, R8

ct2dVTailColLoop:
	MOVD R11, R9
	WORD $0x4eb21e50 // mov v16.16b, v18.16b
	WORD $0x4eb21e51 // mov v17.16b, v18.16b

	WORD $0x4cce7521 // ld1 {v1.8h}, [x9], x14
	WORD $0x0f402030 // smlal  v16.4s, v1.4h, v0.h[0]
	WORD $0x4f402031 // smlal2 v17.4s, v1.8h, v0.h[0]
	WORD $0x4cce7521
	WORD $0x0f502030
	WORD $0x4f502031
	WORD $0x4cce7521
	WORD $0x0f602030
	WORD $0x4f602031
	WORD $0x4cce7521
	WORD $0x0f702030
	WORD $0x4f702031
	WORD $0x4cce7521
	WORD $0x0f402830
	WORD $0x4f402831
	WORD $0x4cce7521
	WORD $0x0f502830
	WORD $0x4f502831
	WORD $0x4cce7521
	WORD $0x0f602830
	WORD $0x4f602831
	WORD $0x4cce7521
	WORD $0x0f702830
	WORD $0x4f702831

	WORD $0x4f392610 // srshr v16.4s, v16.4s, #7
	WORD $0x4f392631 // srshr v17.4s, v17.4s, #7
	WORD $0x0e614a10 // sqxtn  v16.4h, v16.4s
	WORD $0x4e614a30 // sqxtn2 v16.8h, v17.4s
	WORD $0x4c007550 // st1 {v16.8h}, [x10]

	ADD  $16, R10, R10
	ADD  $16, R11, R11
	SUB  $8, R8, R8
	CBNZ R8, ct2dVTailColLoop

	ADD  R6<<1, R1
	ADD  R14, R17, R17
	SUB  $1, R7, R7
	CBNZ R7, ct2dVTailRowLoop

ct2dVDone:
	RET

// func compound2D4TapW4I8MMAsm(ctx *compound2D8I8MMCtx)
//
// Resident width-4 lowbd compound 2D convolve. The horizontal pass mirrors
// SVT's convolve_2d_sr_horiz_4tap_neon_i8mm W4 path over the local small-block
// 4-tap kernels; the vertical pass writes CONV_BUF precision.
TEXT ·compound2D4TapW4I8MMAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C2DI8_DST(R0), R1
	MOVD C2DI8_REF(R0), R2
	MOVD C2DI8_XFILTER(R0), R3
	MOVD C2DI8_PERMUTE(R0), R12
	MOVD C2DI8_REFSTR(R0), R5
	MOVD C2DI8_HEIGHT(R0), R7
	MOVD C2DI8_IM(R0), R13
	MOVD C2DI8_IMSTR(R0), R14
	LSL  $1, R14, R14

	ADD  $7, R7, R6

	VLD1 (R3), [V0.B16]
	VLD1 (R12), [V28.B16]
	MOVD $8194, R11
	WORD $0x4e040d72 // dup v18.4s, w11

	MOVD R2, R16
	MOVD R13, R17

ct2dH4RowLoop:
	CBZ  R6, ct2dH4Done
	MOVD R16, R9
	WORD $0x4eb21e50 // mov v16.16b, v18.16b
	WORD $0x0c407121 // ld1 {v1.8b}, [x9]
	WORD $0x4e1c0025 // tbl v5.16b, {v1.16b}, v28.16b
	WORD $0x4f80f0b0 // usdot v16.4s, v5.16b, v0.4b[0]
	WORD $0x0f1e8610 // shrn v16.4h, v16.4s, #2
	WORD $0x0c007630 // st1 {v16.4h}, [x17]

	ADD  R5, R16, R16
	ADD  R14, R17, R17
	SUB  $1, R6, R6
	CBNZ R6, ct2dH4RowLoop

ct2dH4Done:
	MOVD C2DI8_YKERNEL(R0), R3
	WORD $0x4c407460 // ld1 {v0.8h}, [x3]
	MOVD $524288, R11
	WORD $0x4e040d72 // dup v18.4s, w11

	MOVD R13, R17

ct2dV4RowLoop:
	CBZ  R7, ct2dV4Done
	MOVD R17, R9
	MOVD R1, R10
	WORD $0x4eb21e50 // mov v16.16b, v18.16b

	WORD $0x0cce7521 // ld1 {v1.4h}, [x9], x14
	WORD $0x0f402030 // smlal v16.4s, v1.4h, v0.h[0]
	WORD $0x0cce7521
	WORD $0x0f502030
	WORD $0x0cce7521
	WORD $0x0f602030
	WORD $0x0cce7521
	WORD $0x0f702030
	WORD $0x0cce7521
	WORD $0x0f402830
	WORD $0x0cce7521
	WORD $0x0f502830
	WORD $0x0cce7521
	WORD $0x0f602830
	WORD $0x0cce7521
	WORD $0x0f702830

	WORD $0x4f392610 // srshr v16.4s, v16.4s, #7
	WORD $0x0e614a10 // sqxtn v16.4h, v16.4s
	WORD $0x0c007550 // st1 {v16.4h}, [x10]

	ADD  $8, R1, R1
	ADD  R14, R17, R17
	SUB  $1, R7, R7
	CBNZ R7, ct2dV4RowLoop

ct2dV4Done:
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

// func convolveY8I8MMAsm(ctx *convolveY8I8MMCtx)
//
// Resident lowbd Y convolve, 8-tap/6-tap tier. This mirrors SVT's
// convolve_y_sr_8tap_neon_i8mm: halve the even AV1 taps, transpose/concatenate
// 4-row windows, merge sliding windows with svt_kDotProdMergeBlockTbl, then use
// USDOT lane 0/1 and vqrshrun_n_s16(..., FILTER_BITS-1).
TEXT ·convolveY8I8MMAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD YSR_DST(R0), R1
	MOVD YSR_REF(R0), R2
	MOVD YSR_FILTER(R0), R3
	MOVD YSR_MERGE(R0), R12
	MOVD YSR_DSTSTR(R0), R4
	MOVD YSR_REFSTR(R0), R5
	MOVD YSR_WIDTH(R0), R6
	MOVD YSR_HEIGHT(R0), R7

	VLD1   (R3), [V0.B16]
	VLD1.P 16(R12), [V28.B16]
	VLD1.P 16(R12), [V29.B16]
	VLD1   (R12), [V30.B16]

	CMP $4, R6
	BEQ y8I8W4Col

y8I8ColLoop:
	MOVD R2, R9          // src row cursor for this 8-column stripe
	MOVD R1, R10         // dst row cursor for this 8-column stripe
	MOVD R7, R11         // remaining rows

	// load_u8_8x7(s0..s6)
	WORD $0x0cc57121     // ld1 {v1.8b}, [x9], x5
	WORD $0x0cc57122     // ld1 {v2.8b}, [x9], x5
	WORD $0x0cc57123     // ld1 {v3.8b}, [x9], x5
	WORD $0x0cc57124     // ld1 {v4.8b}, [x9], x5
	WORD $0x0cc57125     // ld1 {v5.8b}, [x9], x5
	WORD $0x0cc57126     // ld1 {v6.8b}, [x9], x5
	WORD $0x0cc57127     // ld1 {v7.8b}, [x9], x5

	// transpose_concat_elems_u8_8x4(s0,s1,s2,s3) -> v8/v9
	WORD $0x4e03383a     // zip1 v26.16b, v1.16b, v3.16b
	WORD $0x4e04385b     // zip1 v27.16b, v2.16b, v4.16b
	WORD $0x4e1b3b48     // zip1 v8.16b,  v26.16b, v27.16b
	WORD $0x4e1b7b49     // zip2 v9.16b,  v26.16b, v27.16b
	// transpose_concat_elems_u8_8x4(s1,s2,s3,s4) -> v10/v11
	WORD $0x4e04385a     // zip1 v26.16b, v2.16b, v4.16b
	WORD $0x4e05387b     // zip1 v27.16b, v3.16b, v5.16b
	WORD $0x4e1b3b4a     // zip1 v10.16b, v26.16b, v27.16b
	WORD $0x4e1b7b4b     // zip2 v11.16b, v26.16b, v27.16b
	// transpose_concat_elems_u8_8x4(s2,s3,s4,s5) -> v12/v13
	WORD $0x4e05387a     // zip1 v26.16b, v3.16b, v5.16b
	WORD $0x4e06389b     // zip1 v27.16b, v4.16b, v6.16b
	WORD $0x4e1b3b4c     // zip1 v12.16b, v26.16b, v27.16b
	WORD $0x4e1b7b4d     // zip2 v13.16b, v26.16b, v27.16b
	// transpose_concat_elems_u8_8x4(s3,s4,s5,s6) -> v14/v18
	WORD $0x4e06389a     // zip1 v26.16b, v4.16b, v6.16b
	WORD $0x4e0738bb     // zip1 v27.16b, v5.16b, v7.16b
	WORD $0x4e1b3b4e     // zip1 v14.16b, v26.16b, v27.16b
	WORD $0x4e1b7b52     // zip2 v18.16b, v26.16b, v27.16b

y8I8Row4Loop:
	// load_u8_8x4(s7..sA)
	WORD $0x0cc57121     // ld1 {v1.8b}, [x9], x5
	WORD $0x0cc57122     // ld1 {v2.8b}, [x9], x5
	WORD $0x0cc57123     // ld1 {v3.8b}, [x9], x5
	WORD $0x0cc57124     // ld1 {v4.8b}, [x9], x5
	// transpose_concat_elems_u8_8x4(s7,s8,s9,sA) -> v15/v19
	WORD $0x4e03383a     // zip1 v26.16b, v1.16b, v3.16b
	WORD $0x4e04385b     // zip1 v27.16b, v2.16b, v4.16b
	WORD $0x4e1b3b4f     // zip1 v15.16b, v26.16b, v27.16b
	WORD $0x4e1b7b53     // zip2 v19.16b, v26.16b, v27.16b

	// Merge shifted windows from {s3456,s789A}.
	WORD $0x4e1c21d4     // tbl v20.16b, {v14.16b, v15.16b}, v28.16b
	WORD $0x4e1c2255     // tbl v21.16b, {v18.16b, v19.16b}, v28.16b
	WORD $0x4e1d21d6     // tbl v22.16b, {v14.16b, v15.16b}, v29.16b
	WORD $0x4e1d2257     // tbl v23.16b, {v18.16b, v19.16b}, v29.16b
	WORD $0x4e1e21d8     // tbl v24.16b, {v14.16b, v15.16b}, v30.16b
	WORD $0x4e1e2259     // tbl v25.16b, {v18.16b, v19.16b}, v30.16b

	// d0 = convolve8_8_y(s0123, s4567)
	WORD $0x4f000410     // movi v16.4s, #0
	WORD $0x4f000411     // movi v17.4s, #0
	WORD $0x4f80f110     // usdot v16.4s, v8.16b,  v0.4b[0]
	WORD $0x4fa0f290     // usdot v16.4s, v20.16b, v0.4b[1]
	WORD $0x4f80f131     // usdot v17.4s, v9.16b,  v0.4b[0]
	WORD $0x4fa0f2b1     // usdot v17.4s, v21.16b, v0.4b[1]
	WORD $0x0e612a10     // xtn  v16.4h, v16.4s
	WORD $0x4e612a30     // xtn2 v16.8h, v17.4s
	WORD $0x2f0a8e10     // sqrshrun v16.8b, v16.8h, #6
	WORD $0x0c007150     // st1 {v16.8b}, [x10]
	ADD  R4, R10, R10

	// d1 = convolve8_8_y(s1234, s5678)
	WORD $0x4f000410     // movi v16.4s, #0
	WORD $0x4f000411     // movi v17.4s, #0
	WORD $0x4f80f150     // usdot v16.4s, v10.16b, v0.4b[0]
	WORD $0x4fa0f2d0     // usdot v16.4s, v22.16b, v0.4b[1]
	WORD $0x4f80f171     // usdot v17.4s, v11.16b, v0.4b[0]
	WORD $0x4fa0f2f1     // usdot v17.4s, v23.16b, v0.4b[1]
	WORD $0x0e612a10     // xtn  v16.4h, v16.4s
	WORD $0x4e612a30     // xtn2 v16.8h, v17.4s
	WORD $0x2f0a8e10     // sqrshrun v16.8b, v16.8h, #6
	WORD $0x0c007150     // st1 {v16.8b}, [x10]
	ADD  R4, R10, R10

	// d2 = convolve8_8_y(s2345, s6789)
	WORD $0x4f000410     // movi v16.4s, #0
	WORD $0x4f000411     // movi v17.4s, #0
	WORD $0x4f80f190     // usdot v16.4s, v12.16b, v0.4b[0]
	WORD $0x4fa0f310     // usdot v16.4s, v24.16b, v0.4b[1]
	WORD $0x4f80f1b1     // usdot v17.4s, v13.16b, v0.4b[0]
	WORD $0x4fa0f331     // usdot v17.4s, v25.16b, v0.4b[1]
	WORD $0x0e612a10     // xtn  v16.4h, v16.4s
	WORD $0x4e612a30     // xtn2 v16.8h, v17.4s
	WORD $0x2f0a8e10     // sqrshrun v16.8b, v16.8h, #6
	WORD $0x0c007150     // st1 {v16.8b}, [x10]
	ADD  R4, R10, R10

	// d3 = convolve8_8_y(s3456, s789A)
	WORD $0x4f000410     // movi v16.4s, #0
	WORD $0x4f000411     // movi v17.4s, #0
	WORD $0x4f80f1d0     // usdot v16.4s, v14.16b, v0.4b[0]
	WORD $0x4fa0f1f0     // usdot v16.4s, v15.16b, v0.4b[1]
	WORD $0x4f80f251     // usdot v17.4s, v18.16b, v0.4b[0]
	WORD $0x4fa0f271     // usdot v17.4s, v19.16b, v0.4b[1]
	WORD $0x0e612a10     // xtn  v16.4h, v16.4s
	WORD $0x4e612a30     // xtn2 v16.8h, v17.4s
	WORD $0x2f0a8e10     // sqrshrun v16.8b, v16.8h, #6
	WORD $0x0c007150     // st1 {v16.8b}, [x10]
	ADD  R4, R10, R10

	// Shuffle everything up four rows.
	WORD $0x4eb41e88     // mov v8.16b,  v20.16b
	WORD $0x4eb51ea9     // mov v9.16b,  v21.16b
	WORD $0x4eb61eca     // mov v10.16b, v22.16b
	WORD $0x4eb71eeb     // mov v11.16b, v23.16b
	WORD $0x4eb81f0c     // mov v12.16b, v24.16b
	WORD $0x4eb91f2d     // mov v13.16b, v25.16b
	WORD $0x4eaf1dee     // mov v14.16b, v15.16b
	WORD $0x4eb31e72     // mov v18.16b, v19.16b

	SUB  $4, R11, R11
	CBNZ R11, y8I8Row4Loop

	ADD  $8, R2, R2
	ADD  $8, R1, R1
	SUB  $8, R6, R6
	CBNZ R6, y8I8ColLoop
	RET

y8I8W4Col:
	MOVD R2, R9
	MOVD R1, R10
	MOVD R7, R11

	// load_u8_4x7(s0..s6)
	WORD $0x0dc58121     // ld1 {v1.s}[0], [x9], x5
	WORD $0x0dc58122     // ld1 {v2.s}[0], [x9], x5
	WORD $0x0dc58123     // ld1 {v3.s}[0], [x9], x5
	WORD $0x0dc58124     // ld1 {v4.s}[0], [x9], x5
	WORD $0x0dc58125     // ld1 {v5.s}[0], [x9], x5
	WORD $0x0dc58126     // ld1 {v6.s}[0], [x9], x5
	WORD $0x0dc58127     // ld1 {v7.s}[0], [x9], x5

	// transpose_concat_elems_u8_4x4(s0,s1,s2,s3) -> v8
	WORD $0x4e03383a
	WORD $0x4e04385b
	WORD $0x4e1b3b48
	// transpose_concat_elems_u8_4x4(s1,s2,s3,s4) -> v10
	WORD $0x4e04385a
	WORD $0x4e05387b
	WORD $0x4e1b3b4a
	// transpose_concat_elems_u8_4x4(s2,s3,s4,s5) -> v12
	WORD $0x4e05387a
	WORD $0x4e06389b
	WORD $0x4e1b3b4c
	// transpose_concat_elems_u8_4x4(s3,s4,s5,s6) -> v14
	WORD $0x4e06389a
	WORD $0x4e0738bb
	WORD $0x4e1b3b4e

y8I8W4RowLoop:
	// load_u8_4x4(s7..sA)
	WORD $0x0dc58121
	WORD $0x0dc58122
	WORD $0x0dc58123
	WORD $0x0dc58124
	// transpose_concat_elems_u8_4x4(s7,s8,s9,sA) -> v15
	WORD $0x4e03383a
	WORD $0x4e04385b
	WORD $0x4e1b3b4f

	WORD $0x4e1c21d4     // tbl v20.16b, {v14.16b, v15.16b}, v28.16b
	WORD $0x4e1d21d6     // tbl v22.16b, {v14.16b, v15.16b}, v29.16b
	WORD $0x4e1e21d8     // tbl v24.16b, {v14.16b, v15.16b}, v30.16b

	// d0
	WORD $0x4f000410
	WORD $0x4f80f110     // usdot v16.4s, v8.16b,  v0.4b[0]
	WORD $0x4fa0f290     // usdot v16.4s, v20.16b, v0.4b[1]
	WORD $0x0e612a10
	WORD $0x2f0a8e10
	WORD $0x0d008150     // st1 {v16.s}[0], [x10]
	ADD  R4, R10, R10

	// d1
	WORD $0x4f000410
	WORD $0x4f80f150     // usdot v16.4s, v10.16b, v0.4b[0]
	WORD $0x4fa0f2d0     // usdot v16.4s, v22.16b, v0.4b[1]
	WORD $0x0e612a10
	WORD $0x2f0a8e10
	WORD $0x0d008150
	ADD  R4, R10, R10

	// d2
	WORD $0x4f000410
	WORD $0x4f80f190     // usdot v16.4s, v12.16b, v0.4b[0]
	WORD $0x4fa0f310     // usdot v16.4s, v24.16b, v0.4b[1]
	WORD $0x0e612a10
	WORD $0x2f0a8e10
	WORD $0x0d008150
	ADD  R4, R10, R10

	// d3
	WORD $0x4f000410
	WORD $0x4f80f1d0     // usdot v16.4s, v14.16b, v0.4b[0]
	WORD $0x4fa0f1f0     // usdot v16.4s, v15.16b, v0.4b[1]
	WORD $0x0e612a10
	WORD $0x2f0a8e10
	WORD $0x0d008150
	ADD  R4, R10, R10

	WORD $0x4eb41e88     // mov v8.16b,  v20.16b
	WORD $0x4eb61eca     // mov v10.16b, v22.16b
	WORD $0x4eb81f0c     // mov v12.16b, v24.16b
	WORD $0x4eaf1dee     // mov v14.16b, v15.16b

	SUB  $4, R11, R11
	CBNZ R11, y8I8W4RowLoop
	RET

// func convolveY4TapI8MMAsm(ctx *convolveY8I8MMCtx)
//
// Resident lowbd Y convolve, 4-tap tier. Mirrors
// convolve_y_sr_4tap_neon_i8mm with y_filter_ptr+2 packed into USDOT lane 0.
TEXT ·convolveY4TapI8MMAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD YSR_DST(R0), R1
	MOVD YSR_REF(R0), R2
	MOVD YSR_FILTER(R0), R3
	MOVD YSR_MERGE(R0), R12
	MOVD YSR_DSTSTR(R0), R4
	MOVD YSR_REFSTR(R0), R5
	MOVD YSR_WIDTH(R0), R6
	MOVD YSR_HEIGHT(R0), R7

	VLD1   (R3), [V0.B16]
	VLD1.P 16(R12), [V28.B16]
	VLD1.P 16(R12), [V29.B16]
	VLD1   (R12), [V30.B16]

	CMP $4, R6
	BEQ y4I8W4Col

y4I8ColLoop:
	MOVD R2, R9
	MOVD R1, R10
	MOVD R7, R11

	// load_u8_8x4(s0..s3) into v4..v7, then transpose -> v14/v18.
	WORD $0x0cc57124     // ld1 {v4.8b}, [x9], x5
	WORD $0x0cc57125     // ld1 {v5.8b}, [x9], x5
	WORD $0x0cc57126     // ld1 {v6.8b}, [x9], x5
	WORD $0x0cc57127     // ld1 {v7.8b}, [x9], x5
	WORD $0x4e06389a     // zip1 v26.16b, v4.16b, v6.16b
	WORD $0x4e0738bb     // zip1 v27.16b, v5.16b, v7.16b
	WORD $0x4e1b3b4e     // zip1 v14.16b, v26.16b, v27.16b
	WORD $0x4e1b7b52     // zip2 v18.16b, v26.16b, v27.16b

y4I8Row4Loop:
	// load_u8_8x4(s4..s7), transpose -> v15/v19.
	WORD $0x0cc57121
	WORD $0x0cc57122
	WORD $0x0cc57123
	WORD $0x0cc57124
	WORD $0x4e03383a
	WORD $0x4e04385b
	WORD $0x4e1b3b4f
	WORD $0x4e1b7b53

	WORD $0x4e1c21d4     // tbl v20.16b, {v14.16b, v15.16b}, v28.16b
	WORD $0x4e1c2255     // tbl v21.16b, {v18.16b, v19.16b}, v28.16b
	WORD $0x4e1d21d6     // tbl v22.16b, {v14.16b, v15.16b}, v29.16b
	WORD $0x4e1d2257     // tbl v23.16b, {v18.16b, v19.16b}, v29.16b
	WORD $0x4e1e21d8     // tbl v24.16b, {v14.16b, v15.16b}, v30.16b
	WORD $0x4e1e2259     // tbl v25.16b, {v18.16b, v19.16b}, v30.16b

	// d0 = convolve4_8_y(s0123)
	WORD $0x4f000410
	WORD $0x4f000411
	WORD $0x4f80f1d0     // usdot v16.4s, v14.16b, v0.4b[0]
	WORD $0x4f80f251     // usdot v17.4s, v18.16b, v0.4b[0]
	WORD $0x0e612a10
	WORD $0x4e612a30
	WORD $0x2f0a8e10
	WORD $0x0c007150
	ADD  R4, R10, R10

	// d1 = convolve4_8_y(s1234)
	WORD $0x4f000410
	WORD $0x4f000411
	WORD $0x4f80f290     // usdot v16.4s, v20.16b, v0.4b[0]
	WORD $0x4f80f2b1     // usdot v17.4s, v21.16b, v0.4b[0]
	WORD $0x0e612a10
	WORD $0x4e612a30
	WORD $0x2f0a8e10
	WORD $0x0c007150
	ADD  R4, R10, R10

	// d2 = convolve4_8_y(s2345)
	WORD $0x4f000410
	WORD $0x4f000411
	WORD $0x4f80f2d0     // usdot v16.4s, v22.16b, v0.4b[0]
	WORD $0x4f80f2f1     // usdot v17.4s, v23.16b, v0.4b[0]
	WORD $0x0e612a10
	WORD $0x4e612a30
	WORD $0x2f0a8e10
	WORD $0x0c007150
	ADD  R4, R10, R10

	// d3 = convolve4_8_y(s3456)
	WORD $0x4f000410
	WORD $0x4f000411
	WORD $0x4f80f310     // usdot v16.4s, v24.16b, v0.4b[0]
	WORD $0x4f80f331     // usdot v17.4s, v25.16b, v0.4b[0]
	WORD $0x0e612a10
	WORD $0x4e612a30
	WORD $0x2f0a8e10
	WORD $0x0c007150
	ADD  R4, R10, R10

	WORD $0x4eaf1dee     // mov v14.16b, v15.16b
	WORD $0x4eb31e72     // mov v18.16b, v19.16b

	SUB  $4, R11, R11
	CBNZ R11, y4I8Row4Loop

	ADD  $8, R2, R2
	ADD  $8, R1, R1
	SUB  $8, R6, R6
	CBNZ R6, y4I8ColLoop
	RET

y4I8W4Col:
	MOVD R2, R9
	MOVD R1, R10
	MOVD R7, R11

	// load_u8_4x4(s0..s3), transpose -> v14.
	WORD $0x0dc58124
	WORD $0x0dc58125
	WORD $0x0dc58126
	WORD $0x0dc58127
	WORD $0x4e06389a
	WORD $0x4e0738bb
	WORD $0x4e1b3b4e

y4I8W4RowLoop:
	// load_u8_4x4(s4..s7), transpose -> v15.
	WORD $0x0dc58121
	WORD $0x0dc58122
	WORD $0x0dc58123
	WORD $0x0dc58124
	WORD $0x4e03383a
	WORD $0x4e04385b
	WORD $0x4e1b3b4f

	WORD $0x4e1c21d4     // tbl v20.16b, {v14.16b, v15.16b}, v28.16b
	WORD $0x4e1d21d6     // tbl v22.16b, {v14.16b, v15.16b}, v29.16b
	WORD $0x4e1e21d8     // tbl v24.16b, {v14.16b, v15.16b}, v30.16b

	// d0
	WORD $0x4f000410
	WORD $0x4f80f1d0     // usdot v16.4s, v14.16b, v0.4b[0]
	WORD $0x0e612a10
	WORD $0x2f0a8e10
	WORD $0x0d008150
	ADD  R4, R10, R10

	// d1
	WORD $0x4f000410
	WORD $0x4f80f290     // usdot v16.4s, v20.16b, v0.4b[0]
	WORD $0x0e612a10
	WORD $0x2f0a8e10
	WORD $0x0d008150
	ADD  R4, R10, R10

	// d2
	WORD $0x4f000410
	WORD $0x4f80f2d0     // usdot v16.4s, v22.16b, v0.4b[0]
	WORD $0x0e612a10
	WORD $0x2f0a8e10
	WORD $0x0d008150
	ADD  R4, R10, R10

	// d3
	WORD $0x4f000410
	WORD $0x4f80f310     // usdot v16.4s, v24.16b, v0.4b[0]
	WORD $0x0e612a10
	WORD $0x2f0a8e10
	WORD $0x0d008150
	ADD  R4, R10, R10

	WORD $0x4eaf1dee     // mov v14.16b, v15.16b

	SUB  $4, R11, R11
	CBNZ R11, y4I8W4RowLoop
	RET

// func convolve2D4TapW4I8MMAsm(ctx *convolve2D8I8MMCtx)
//
// Resident width-4 lowbd 2D convolve. The horizontal pass mirrors SVT's
// convolve_2d_sr_horiz_4tap_neon_i8mm W4 path, and the vertical pass mirrors
// convolve2D8NEONAsmW4's byte-output finalization.
TEXT ·convolve2D4TapW4I8MMAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD T2D_DST(R0), R1
	MOVD T2D_REF(R0), R2
	MOVD T2D_XFILTER(R0), R3
	MOVD T2D_PERMUTE(R0), R12
	MOVD T2D_DSTSTR(R0), R4
	MOVD T2D_REFSTR(R0), R5
	MOVD T2D_HEIGHT(R0), R7
	MOVD T2D_IM(R0), R13
	MOVD T2D_IMSTR(R0), R14
	LSL  $1, R14, R14

	ADD  $7, R7, R6

	VLD1 (R3), [V0.B16]
	VLD1 (R12), [V28.B16]
	MOVD $8194, R11
	WORD $0x4e040d72 // dup v18.4s, w11

	MOVD R2, R16
	MOVD R13, R17

t2dH4I8RowLoop:
	CBZ  R6, t2dH4I8Done
	MOVD R16, R9
	WORD $0x4eb21e50 // mov v16.16b, v18.16b
	WORD $0x0c407121 // ld1 {v1.8b}, [x9]
	WORD $0x4e1c0025 // tbl v5.16b, {v1.16b}, v28.16b
	WORD $0x4f80f0b0 // usdot v16.4s, v5.16b, v0.4b[0]
	WORD $0x0f1e8610 // shrn v16.4h, v16.4s, #2
	WORD $0x0c007630 // st1 {v16.4h}, [x17]

	ADD  R5, R16, R16
	ADD  R14, R17, R17
	SUB  $1, R6, R6
	CBNZ R6, t2dH4I8RowLoop

t2dH4I8Done:
	MOVD T2D_YKERNEL(R0), R3
	WORD $0x4c407460 // ld1 {v0.8h}, [x3]
	MOVD $524288, R11
	WORD $0x4e040d72 // dup v18.4s, w11
	MOVD $384, R11
	WORD $0x4e040d73 // dup v19.4s, w11

	MOVD R13, R17

t2dV4I8RowLoop:
	CBZ  R7, t2dV4I8Done
	MOVD R17, R9
	WORD $0x4eb21e50 // mov v16.16b, v18.16b

	WORD $0x0cce7521 // ld1 {v1.4h}, [x9], x14
	WORD $0x0f402030 // smlal v16.4s, v1.4h, v0.h[0]
	WORD $0x0cce7521
	WORD $0x0f502030
	WORD $0x0cce7521
	WORD $0x0f602030
	WORD $0x0cce7521
	WORD $0x0f702030
	WORD $0x0cce7521
	WORD $0x0f402830
	WORD $0x0cce7521
	WORD $0x0f502830
	WORD $0x0cce7521
	WORD $0x0f602830
	WORD $0x0cce7521
	WORD $0x0f702830

	WORD $0x4f352610 // srshr v16.4s, v16.4s, #11
	WORD $0x6eb38610 // sub v16.4s, v16.4s, v19.4s
	WORD $0x0e614a10 // sqxtn v16.4h, v16.4s
	WORD $0x2e212a10 // sqxtun v16.8b, v16.8h
	WORD $0x0d008030 // st1 {v16.s}[0], [x1]

	ADD  R4, R1, R1
	ADD  R14, R17, R17
	SUB  $1, R7, R7
	CBNZ R7, t2dV4I8RowLoop

t2dV4I8Done:
	RET

// func convolve2D8I8MMAsm(ctx *convolve2D8I8MMCtx)
//
// Resident width>=8 lowbd 2D convolve. The horizontal pass mirrors SVT's
// convolve_2d_sr_horiz_8tap_neon_i8mm: halve the even AV1 taps, stagger f1..f7
// for USMMLA, subtract tap 0 separately when nonzero, add the ROUND0 shim,
// then shift by ROUND0_BITS-1 into the int16 intermediate. Regular and smooth
// phases take a zero-tap specialization that omits the multiply. The vertical
// pass shares an 11-row window across four outputs and folds its integer output
// offset into the accumulator seed.
TEXT ·convolve2D8I8MMAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD T2D_DST(R0), R1      // dst row base
	MOVD T2D_REF(R0), R2      // ref tap-window base (row refY-foY, col refX-foX)
	MOVD T2D_XFILTER(R0), R3  // packed horizontal f1..f7 filter
	MOVD T2D_PERMUTE(R0), R12 // SVT kMatMul8 permute table
	MOVD T2D_DSTSTR(R0), R4   // dst stride (bytes)
	MOVD T2D_REFSTR(R0), R5   // ref stride (bytes)
	MOVD T2D_WIDTH(R0), R6    // width
	MOVD T2D_HEIGHT(R0), R7   // height
	MOVD T2D_IM(R0), R13      // im base ptr
	MOVD T2D_IMSTR(R0), R14   // im row stride (int16 elements)
	MOVD T2D_F0(R0), R15      // -tap0 >> 1
	LSL  $1, R14, R14         // im row stride in bytes

	// imH = height + filterTaps - 1 = height + 7
	ADD  $7, R7, R11

	// Horizontal pass setup.
	VLD1   (R3), [V0.B16]
	VLD1.P 16(R12), [V28.B16]
	VLD1   (R12), [V29.B16]
	MOVD   $8194, R3        // (1 << (8+FILTER_BITS-2)) + (1 << ((ROUND0_BITS-1)-1))
	WORD   $0x4e020c79     // dup v25.8h, w3
	CBNZ   R15, t2dHNonZeroStart

	MOVD R2, R16           // ref row cursor
	MOVD R13, R17          // im row cursor

t2dHZeroRowLoop:
	MOVD R16, R9           // ref column cursor
	MOVD R17, R10          // im column cursor
	MOVD R6, R8            // remaining columns

t2dHZeroColLoop:
	VLD1 (R9), [V1.B16]
	WORD $0x4e1c0025       // tbl v5.16b, {v1.16b}, v28.16b
	WORD $0x4e1d0026       // tbl v6.16b, {v1.16b}, v29.16b
	WORD $0x4f000410       // movi v16.4s, #0
	WORD $0x4f000411       // movi v17.4s, #0
	WORD $0x4e80acb0       // usmmla v16.4s, v5.16b, v0.16b
	WORD $0x4e80acd1       // usmmla v17.4s, v6.16b, v0.16b
	WORD $0x0e612a10       // xtn  v16.4h, v16.4s
	WORD $0x4e612a30       // xtn2 v16.8h, v17.4s
	WORD $0x4e798610       // add  v16.8h, v16.8h, v25.8h
	WORD $0x6f1e0610       // ushr v16.8h, v16.8h, #2
	WORD $0x4c007550       // st1  {v16.8h}, [x10]

	ADD  $8, R9, R9
	ADD  $16, R10, R10
	SUB  $8, R8, R8
	CBNZ R8, t2dHZeroColLoop

	ADD  R5, R16, R16
	ADD  R14, R17, R17
	SUB  $1, R11, R11
	CBNZ R11, t2dHZeroRowLoop
	B    t2dHDone

t2dHNonZeroStart:
	WORD $0x0e010dfa // dup v26.8b, w15
	MOVD R2, R16
	MOVD R13, R17

t2dHNonZeroRowLoop:
	MOVD R16, R9
	MOVD R17, R10
	MOVD R6, R8

t2dHNonZeroColLoop:
	VLD1 (R9), [V1.B16]
	WORD $0x4e1c0025       // tbl v5.16b, {v1.16b}, v28.16b
	WORD $0x4e1d0026       // tbl v6.16b, {v1.16b}, v29.16b
	WORD $0x4f000410       // movi v16.4s, #0
	WORD $0x4f000411       // movi v17.4s, #0
	WORD $0x4e80acb0       // usmmla v16.4s, v5.16b, v0.16b
	WORD $0x4e80acd1       // usmmla v17.4s, v6.16b, v0.16b
	WORD $0x0e612a10       // xtn  v16.4h, v16.4s
	WORD $0x4e612a30       // xtn2 v16.8h, v17.4s
	WORD $0x2e3aa030       // umlsl v16.8h, v1.8b, v26.8b
	WORD $0x4e798610       // add  v16.8h, v16.8h, v25.8h
	WORD $0x6f1e0610       // ushr v16.8h, v16.8h, #2
	WORD $0x4c007550       // st1  {v16.8h}, [x10]

	ADD  $8, R9, R9
	ADD  $16, R10, R10
	SUB  $8, R8, R8
	CBNZ R8, t2dHNonZeroColLoop

	ADD  R5, R16, R16
	ADD  R14, R17, R17
	SUB  $1, R11, R11
	CBNZ R11, t2dHNonZeroRowLoop

t2dHDone:
	// Vertical pass setup.
	MOVD T2D_YKERNEL(R0), R3
	WORD $0x4c407460       // ld1 {v0.8h}, [x3]  load 8 vertical taps
	// Fold the integer round-offset subtraction into the accumulator seed:
	// yBias - (roundOffset << ROUND1_BITS) = 524288 - (384 << 11).
	MOVD $-262144, R11
	WORD $0x4e040d72       // dup v18.4s, w11

	MOVD R13, R17          // im row-window base for output row 0
	CMP  $4, R7
	BLT  t2dVTail

t2dVRow4Loop:
	MOVD R1, R10           // dst column cursor
	MOVD R17, R11          // im row-window base for this output row
	MOVD R6, R8            // remaining columns

t2dVCol4Loop:
	MOVD R11, R9           // R9 walks the 11 shared rows; post-index by imStride
	WORD $0x4eb21e50       // mov v16.16b, v18.16b   init acc to folded seed
	WORD $0x4eb21e51       // mov v17.16b, v18.16b
	WORD $0x4eb21e53       // mov v19.16b, v18.16b
	WORD $0x4eb21e54       // mov v20.16b, v18.16b
	WORD $0x4eb21e55       // mov v21.16b, v18.16b
	WORD $0x4eb21e56       // mov v22.16b, v18.16b
	WORD $0x4eb21e57       // mov v23.16b, v18.16b
	WORD $0x4eb21e58       // mov v24.16b, v18.16b

	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x4cce7522       // ld1 {v2.8h}, [x9], x14
	WORD $0x4cce7523       // ld1 {v3.8h}, [x9], x14
	WORD $0x4cce7524       // ld1 {v4.8h}, [x9], x14
	WORD $0x4cce7525       // ld1 {v5.8h}, [x9], x14
	WORD $0x4cce7526       // ld1 {v6.8h}, [x9], x14
	WORD $0x4cce7527       // ld1 {v7.8h}, [x9], x14
	WORD $0x4cce7539       // ld1 {v25.8h}, [x9], x14
	WORD $0x4cce753a       // ld1 {v26.8h}, [x9], x14
	WORD $0x4cce753b       // ld1 {v27.8h}, [x9], x14
	WORD $0x4cce753c       // ld1 {v28.8h}, [x9], x14

	WORD $0x0f402030       // smlal  v16.4s, v1.4h, v0.h[0]
	WORD $0x0f402053       // smlal  v19.4s, v2.4h, v0.h[0]
	WORD $0x0f402075       // smlal  v21.4s, v3.4h, v0.h[0]
	WORD $0x0f402097       // smlal  v23.4s, v4.4h, v0.h[0]
	WORD $0x4f402031       // smlal2 v17.4s, v1.8h, v0.h[0]
	WORD $0x4f402054       // smlal2 v20.4s, v2.8h, v0.h[0]
	WORD $0x4f402076       // smlal2 v22.4s, v3.8h, v0.h[0]
	WORD $0x4f402098       // smlal2 v24.4s, v4.8h, v0.h[0]
	WORD $0x0f502050       // smlal  v16.4s, v2.4h, v0.h[1]
	WORD $0x0f502073       // smlal  v19.4s, v3.4h, v0.h[1]
	WORD $0x0f502095       // smlal  v21.4s, v4.4h, v0.h[1]
	WORD $0x0f5020b7       // smlal  v23.4s, v5.4h, v0.h[1]
	WORD $0x4f502051       // smlal2 v17.4s, v2.8h, v0.h[1]
	WORD $0x4f502074       // smlal2 v20.4s, v3.8h, v0.h[1]
	WORD $0x4f502096       // smlal2 v22.4s, v4.8h, v0.h[1]
	WORD $0x4f5020b8       // smlal2 v24.4s, v5.8h, v0.h[1]
	WORD $0x0f602070       // smlal  v16.4s, v3.4h, v0.h[2]
	WORD $0x0f602093       // smlal  v19.4s, v4.4h, v0.h[2]
	WORD $0x0f6020b5       // smlal  v21.4s, v5.4h, v0.h[2]
	WORD $0x0f6020d7       // smlal  v23.4s, v6.4h, v0.h[2]
	WORD $0x4f602071       // smlal2 v17.4s, v3.8h, v0.h[2]
	WORD $0x4f602094       // smlal2 v20.4s, v4.8h, v0.h[2]
	WORD $0x4f6020b6       // smlal2 v22.4s, v5.8h, v0.h[2]
	WORD $0x4f6020d8       // smlal2 v24.4s, v6.8h, v0.h[2]
	WORD $0x0f702090       // smlal  v16.4s, v4.4h, v0.h[3]
	WORD $0x0f7020b3       // smlal  v19.4s, v5.4h, v0.h[3]
	WORD $0x0f7020d5       // smlal  v21.4s, v6.4h, v0.h[3]
	WORD $0x0f7020f7       // smlal  v23.4s, v7.4h, v0.h[3]
	WORD $0x4f702091       // smlal2 v17.4s, v4.8h, v0.h[3]
	WORD $0x4f7020b4       // smlal2 v20.4s, v5.8h, v0.h[3]
	WORD $0x4f7020d6       // smlal2 v22.4s, v6.8h, v0.h[3]
	WORD $0x4f7020f8       // smlal2 v24.4s, v7.8h, v0.h[3]
	WORD $0x0f4028b0       // smlal  v16.4s, v5.4h, v0.h[4]
	WORD $0x0f4028d3       // smlal  v19.4s, v6.4h, v0.h[4]
	WORD $0x0f4028f5       // smlal  v21.4s, v7.4h, v0.h[4]
	WORD $0x0f402b37       // smlal  v23.4s, v25.4h, v0.h[4]
	WORD $0x4f4028b1       // smlal2 v17.4s, v5.8h, v0.h[4]
	WORD $0x4f4028d4       // smlal2 v20.4s, v6.8h, v0.h[4]
	WORD $0x4f4028f6       // smlal2 v22.4s, v7.8h, v0.h[4]
	WORD $0x4f402b38       // smlal2 v24.4s, v25.8h, v0.h[4]
	WORD $0x0f5028d0       // smlal  v16.4s, v6.4h, v0.h[5]
	WORD $0x0f5028f3       // smlal  v19.4s, v7.4h, v0.h[5]
	WORD $0x0f502b35       // smlal  v21.4s, v25.4h, v0.h[5]
	WORD $0x0f502b57       // smlal  v23.4s, v26.4h, v0.h[5]
	WORD $0x4f5028d1       // smlal2 v17.4s, v6.8h, v0.h[5]
	WORD $0x4f5028f4       // smlal2 v20.4s, v7.8h, v0.h[5]
	WORD $0x4f502b36       // smlal2 v22.4s, v25.8h, v0.h[5]
	WORD $0x4f502b58       // smlal2 v24.4s, v26.8h, v0.h[5]
	WORD $0x0f6028f0       // smlal  v16.4s, v7.4h, v0.h[6]
	WORD $0x0f602b33       // smlal  v19.4s, v25.4h, v0.h[6]
	WORD $0x0f602b55       // smlal  v21.4s, v26.4h, v0.h[6]
	WORD $0x0f602b77       // smlal  v23.4s, v27.4h, v0.h[6]
	WORD $0x4f6028f1       // smlal2 v17.4s, v7.8h, v0.h[6]
	WORD $0x4f602b34       // smlal2 v20.4s, v25.8h, v0.h[6]
	WORD $0x4f602b56       // smlal2 v22.4s, v26.8h, v0.h[6]
	WORD $0x4f602b78       // smlal2 v24.4s, v27.8h, v0.h[6]
	WORD $0x0f702b30       // smlal  v16.4s, v25.4h, v0.h[7]
	WORD $0x0f702b53       // smlal  v19.4s, v26.4h, v0.h[7]
	WORD $0x0f702b75       // smlal  v21.4s, v27.4h, v0.h[7]
	WORD $0x0f702b97       // smlal  v23.4s, v28.4h, v0.h[7]
	WORD $0x4f702b31       // smlal2 v17.4s, v25.8h, v0.h[7]
	WORD $0x4f702b54       // smlal2 v20.4s, v26.8h, v0.h[7]
	WORD $0x4f702b76       // smlal2 v22.4s, v27.8h, v0.h[7]
	WORD $0x4f702b98       // smlal2 v24.4s, v28.8h, v0.h[7]

	WORD $0x0f159e10       // sqrshrn  v16.4h, v16.4s, #11
	WORD $0x4f159e30       // sqrshrn2 v16.8h, v17.4s, #11
	WORD $0x2e212a10       // sqxtun v16.8b, v16.8h
	WORD $0x0f159e73       // sqrshrn  v19.4h, v19.4s, #11
	WORD $0x4f159e93       // sqrshrn2 v19.8h, v20.4s, #11
	WORD $0x2e212a73       // sqxtun v19.8b, v19.8h
	WORD $0x0f159eb5       // sqrshrn  v21.4h, v21.4s, #11
	WORD $0x4f159ed5       // sqrshrn2 v21.8h, v22.4s, #11
	WORD $0x2e212ab5       // sqxtun v21.8b, v21.8h
	WORD $0x0f159ef7       // sqrshrn  v23.4h, v23.4s, #11
	WORD $0x4f159f17       // sqrshrn2 v23.8h, v24.4s, #11
	WORD $0x2e212af7       // sqxtun v23.8b, v23.8h

	MOVD R10, R9
	WORD $0x0c007130       // st1 {v16.8b}, [x9]
	ADD  R4, R9, R9
	WORD $0x0c007133       // st1 {v19.8b}, [x9]
	ADD  R4, R9, R9
	WORD $0x0c007135       // st1 {v21.8b}, [x9]
	ADD  R4, R9, R9
	WORD $0x0c007137       // st1 {v23.8b}, [x9]

	ADD  $8, R10, R10
	ADD  $16, R11, R11
	SUB  $8, R8, R8
	CBNZ R8, t2dVCol4Loop

	ADD R4<<2, R1, R1
	ADD R14<<2, R17, R17
	SUB $4, R7, R7
	CMP $4, R7
	BGE t2dVRow4Loop

t2dVTail:
	CBZ R7, t2dVDone

t2dVTailRowLoop:
	MOVD R1, R10
	MOVD R17, R11
	MOVD R6, R8

t2dVTailColLoop:
	MOVD R11, R9
	WORD $0x4eb21e50       // mov v16.16b, v18.16b
	WORD $0x4eb21e51       // mov v17.16b, v18.16b
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f402030       // smlal  v16.4s, v1.4h, v0.h[0]
	WORD $0x4f402031       // smlal2 v17.4s, v1.8h, v0.h[0]
	WORD $0x4cce7521
	WORD $0x0f502030
	WORD $0x4f502031
	WORD $0x4cce7521
	WORD $0x0f602030
	WORD $0x4f602031
	WORD $0x4cce7521
	WORD $0x0f702030
	WORD $0x4f702031
	WORD $0x4cce7521
	WORD $0x0f402830
	WORD $0x4f402831
	WORD $0x4cce7521
	WORD $0x0f502830
	WORD $0x4f502831
	WORD $0x4cce7521
	WORD $0x0f602830
	WORD $0x4f602831
	WORD $0x4cce7521
	WORD $0x0f702830
	WORD $0x4f702831
	WORD $0x0f159e10       // sqrshrn  v16.4h, v16.4s, #11
	WORD $0x4f159e30       // sqrshrn2 v16.8h, v17.4s, #11
	WORD $0x2e212a10       // sqxtun v16.8b, v16.8h
	WORD $0x0c007150       // st1 {v16.8b}, [x10]

	ADD  $8, R10, R10
	ADD  $16, R11, R11
	SUB  $8, R8, R8
	CBNZ R8, t2dVTailColLoop

	ADD  R4, R1, R1
	ADD  R14, R17, R17
	SUB  $1, R7, R7
	CBNZ R7, t2dVTailRowLoop

t2dVDone:
	RET
