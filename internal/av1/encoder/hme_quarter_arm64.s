// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

#define HME_DST        0
#define HME_SRC        8
#define HME_SRC_STRIDE 16
#define HME_DST_STRIDE 24
#define HME_WIDTH      32
#define HME_HEIGHT     40

// HME 4:1 luma-pyramid box average. The loads are source-shaped after
// libaom v3.13.1's av1/common/arm/resize_neon.c 4-to-1 phase kernels:
// deinterleave 64 source bytes with ld4, then keep one 16-lane vector per
// source column phase. Unlike libaom's resize phase-0 drop/bilinear filters,
// HME intentionally computes the local box average:
//   dst[x] = (sum(src[4*y+r][4*x+c], r,c in 0..3) + 8) >> 4.
//
// Instructions missing from Go's assembler are emitted as WORD:
//   ld4   {v0.16b-v3.16b}, [x12] -> 0x4c400180
//   uaddl v16.8h, v0.8b, v1.8b   -> 0x2e210010
//   uaddw v16.8h, v16.8h, v2.8b  -> 0x2e221210
//   rshrn v18.8b, v16.8h, #4     -> 0x0f0c8e12

// func buildQuarterPlaneNEONAsm(ctx *buildQuarterPlaneNEONCtx)
TEXT ·buildQuarterPlaneNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD HME_DST(R0), R1
	MOVD HME_SRC(R0), R2
	MOVD HME_SRC_STRIDE(R0), R3
	MOVD HME_DST_STRIDE(R0), R4
	MOVD HME_WIDTH(R0), R5
	MOVD HME_HEIGHT(R0), R6
	LSL  $2, R3, R11

quarterHeightLoop:
	MOVD R2, R7
	MOVD R1, R8
	MOVD R5, R9

quarterWidthLoop:
	MOVD R7, R12
	ADD  R3, R12, R13
	ADD  R3, R13, R14
	ADD  R3, R14, R15

	WORD $0x4c400180 // ld4 {v0.16b,  v1.16b,  v2.16b,  v3.16b},  [x12]
	WORD $0x4c4001a4 // ld4 {v4.16b,  v5.16b,  v6.16b,  v7.16b},  [x13]
	WORD $0x4c4001c8 // ld4 {v8.16b,  v9.16b,  v10.16b, v11.16b}, [x14]
	WORD $0x4c4001ec // ld4 {v12.16b, v13.16b, v14.16b, v15.16b}, [x15]

	WORD $0x2e210010 // uaddl  v16.8h, v0.8b,   v1.8b
	WORD $0x6e210011 // uaddl2 v17.8h, v0.16b,  v1.16b
	WORD $0x2e221210 // uaddw  v16.8h, v16.8h,  v2.8b
	WORD $0x6e221231 // uaddw2 v17.8h, v17.8h,  v2.16b
	WORD $0x2e231210 // uaddw  v16.8h, v16.8h,  v3.8b
	WORD $0x6e231231 // uaddw2 v17.8h, v17.8h,  v3.16b
	WORD $0x2e241210 // uaddw  v16.8h, v16.8h,  v4.8b
	WORD $0x6e241231 // uaddw2 v17.8h, v17.8h,  v4.16b
	WORD $0x2e251210 // uaddw  v16.8h, v16.8h,  v5.8b
	WORD $0x6e251231 // uaddw2 v17.8h, v17.8h,  v5.16b
	WORD $0x2e261210 // uaddw  v16.8h, v16.8h,  v6.8b
	WORD $0x6e261231 // uaddw2 v17.8h, v17.8h,  v6.16b
	WORD $0x2e271210 // uaddw  v16.8h, v16.8h,  v7.8b
	WORD $0x6e271231 // uaddw2 v17.8h, v17.8h,  v7.16b
	WORD $0x2e281210 // uaddw  v16.8h, v16.8h,  v8.8b
	WORD $0x6e281231 // uaddw2 v17.8h, v17.8h,  v8.16b
	WORD $0x2e291210 // uaddw  v16.8h, v16.8h,  v9.8b
	WORD $0x6e291231 // uaddw2 v17.8h, v17.8h,  v9.16b
	WORD $0x2e2a1210 // uaddw  v16.8h, v16.8h,  v10.8b
	WORD $0x6e2a1231 // uaddw2 v17.8h, v17.8h,  v10.16b
	WORD $0x2e2b1210 // uaddw  v16.8h, v16.8h,  v11.8b
	WORD $0x6e2b1231 // uaddw2 v17.8h, v17.8h,  v11.16b
	WORD $0x2e2c1210 // uaddw  v16.8h, v16.8h,  v12.8b
	WORD $0x6e2c1231 // uaddw2 v17.8h, v17.8h,  v12.16b
	WORD $0x2e2d1210 // uaddw  v16.8h, v16.8h,  v13.8b
	WORD $0x6e2d1231 // uaddw2 v17.8h, v17.8h,  v13.16b
	WORD $0x2e2e1210 // uaddw  v16.8h, v16.8h,  v14.8b
	WORD $0x6e2e1231 // uaddw2 v17.8h, v17.8h,  v14.16b
	WORD $0x2e2f1210 // uaddw  v16.8h, v16.8h,  v15.8b
	WORD $0x6e2f1231 // uaddw2 v17.8h, v17.8h,  v15.16b

	WORD $0x0f0c8e12 // rshrn  v18.8b,  v16.8h, #4
	WORD $0x4f0c8e32 // rshrn2 v18.16b, v17.8h, #4
	WORD $0x3d800112 // str q18, [x8]

	ADD  $64, R7
	ADD  $16, R8
	SUB  $16, R9
	CBNZ R9, quarterWidthLoop

	ADD  R11, R2
	ADD  R4, R1
	SUB  $1, R6
	CBNZ R6, quarterHeightLoop
	RET
