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

// func convolve2D8I8MMAsm(ctx *convolve2D8I8MMCtx)
//
// Resident width>=8 lowbd 2D convolve. The horizontal pass mirrors SVT's
// convolve_2d_sr_horiz_8tap_neon_i8mm: halve the even AV1 taps, stagger f1..f7
// for USMMLA, subtract tap 0 separately, add the ROUND0 shim, then shift by
// ROUND0_BITS-1 into the int16 intermediate. The vertical pass is the same
// proven NEON int16->uint8 path used by convolve2D8NEONAsm.
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
	WORD   $0x0e010dfa     // dup v26.8b, w15

	MOVD R2, R16           // ref row cursor
	MOVD R13, R17          // im row cursor

t2dHRowLoop:
	CBZ  R11, t2dHDone
	MOVD R16, R9           // ref column cursor
	MOVD R17, R10          // im column cursor
	MOVD R6, R8            // remaining columns

t2dHColLoop:
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
	CBNZ R8, t2dHColLoop

	ADD  R5, R16, R16
	ADD  R14, R17, R17
	SUB  $1, R11, R11
	CBNZ R11, t2dHRowLoop

t2dHDone:
	// Vertical pass setup.
	MOVD T2D_YKERNEL(R0), R3
	WORD $0x4c407460       // ld1 {v0.8h}, [x3]  load 8 vertical taps
	MOVD $524288, R11      // yBias = 1 << offsetBits (offsetBits = 19)
	WORD $0x4e040d72       // dup v18.4s, w11    yBias broadcast
	MOVD $384, R11         // roundOffset = (1<<8) + (1<<7)
	WORD $0x4e040d73       // dup v19.4s, w11    roundOffset broadcast

	MOVD R13, R17          // im row-window base for output row 0

t2dVRowLoop:
	CBZ  R7, t2dVDone
	MOVD R1, R10           // dst column cursor
	MOVD R17, R11          // im row-window base for this output row
	MOVD R6, R8            // remaining columns

t2dVColLoop:
	MOVD R11, R9           // R9 walks the 8 tap rows; post-index by imStride bytes
	WORD $0x4eb21e50       // mov v16.16b, v18.16b   init acc to yBias
	WORD $0x4eb21e51       // mov v17.16b, v18.16b

	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f402030       // smlal  v16.4s, v1.4h, v0.h[0]
	WORD $0x4f402031       // smlal2 v17.4s, v1.8h, v0.h[0]
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f502030       // smlal  v16.4s, v1.4h, v0.h[1]
	WORD $0x4f502031       // smlal2 v17.4s, v1.8h, v0.h[1]
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f602030       // smlal  v16.4s, v1.4h, v0.h[2]
	WORD $0x4f602031       // smlal2 v17.4s, v1.8h, v0.h[2]
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f702030       // smlal  v16.4s, v1.4h, v0.h[3]
	WORD $0x4f702031       // smlal2 v17.4s, v1.8h, v0.h[3]
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f402830       // smlal  v16.4s, v1.4h, v0.h[4]
	WORD $0x4f402831       // smlal2 v17.4s, v1.8h, v0.h[4]
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f502830       // smlal  v16.4s, v1.4h, v0.h[5]
	WORD $0x4f502831       // smlal2 v17.4s, v1.8h, v0.h[5]
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f602830       // smlal  v16.4s, v1.4h, v0.h[6]
	WORD $0x4f602831       // smlal2 v17.4s, v1.8h, v0.h[6]
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f702830       // smlal  v16.4s, v1.4h, v0.h[7]
	WORD $0x4f702831       // smlal2 v17.4s, v1.8h, v0.h[7]

	WORD $0x4f352610       // srshr v16.4s, v16.4s, #11
	WORD $0x4f352631       // srshr v17.4s, v17.4s, #11
	WORD $0x6eb38610       // sub v16.4s, v16.4s, v19.4s
	WORD $0x6eb38631       // sub v17.4s, v17.4s, v19.4s
	WORD $0x0e614a10       // sqxtn  v16.4h, v16.4s
	WORD $0x4e614a30       // sqxtn2 v16.8h, v17.4s
	WORD $0x2e212a10       // sqxtun v16.8b, v16.8h
	WORD $0x0c007150       // st1 {v16.8b}, [x10]

	ADD  $8, R10, R10
	ADD  $16, R11, R11
	SUB  $8, R8, R8
	CBNZ R8, t2dVColLoop

	ADD  R4, R1, R1
	ADD  R14, R17, R17
	SUB  $1, R7, R7
	CBNZ R7, t2dVRowLoop

t2dVDone:
	RET
