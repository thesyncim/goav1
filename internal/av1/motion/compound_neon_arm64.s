// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

#define C_DST         0
#define C_REF         8
#define C_REFSTR      16
#define C_WIDTH       24
#define C_HEIGHT      32
#define C_ROUNDOFFSET 40

// func compoundCopy8NEONAsm(ctx *compoundCopy8NEONCtx)
//
// SVT's joint-convolve copy path writes 8-bit pixels into a 16-bit CONV_BUF:
//     dst = src << (2*FILTER_BITS - COMPOUND_ROUND1_BITS - ROUND0_BITS)
//           + roundOffset
// For 8-bit AV1 compound prediction ROUND0_BITS is 3, so the scale is << 4.
// The Go wrapper only routes resident width-multiple-of-eight blocks here.
TEXT ·compoundCopy8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_REF(R0), R2
	MOVD C_REFSTR(R0), R3
	MOVD C_WIDTH(R0), R5
	MOVD C_HEIGHT(R0), R6
	MOVD C_ROUNDOFFSET(R0), R7

	WORD $0x4e020ce4 // dup v4.8h, w7

rowLoop:
	CBZ  R6, done
	MOVD R2, R8  // reference row cursor
	MOVD R1, R9  // destination row cursor
	MOVD R5, R10 // columns remaining

col16:
	CMP  $16, R10
	BLT  col8
	WORD $0x4c407100 // ld1 {v0.16b}, [x8]
	WORD $0x2f0ca401 // ushll  v1.8h, v0.8b,  #4
	WORD $0x6f0ca402 // ushll2 v2.8h, v0.16b, #4
	WORD $0x4e648421 // add v1.8h, v1.8h, v4.8h
	WORD $0x4e648442 // add v2.8h, v2.8h, v4.8h
	WORD $0x4c00a521 // st1 {v1.8h, v2.8h}, [x9]
	ADD  $16, R8, R8
	ADD  $32, R9, R9
	SUB  $16, R10, R10
	B    col16

col8:
	CBZ  R10, nextRow
	WORD $0x0c407100 // ld1 {v0.8b}, [x8]
	WORD $0x2f0ca401 // ushll v1.8h, v0.8b, #4
	WORD $0x4e648421 // add v1.8h, v1.8h, v4.8h
	WORD $0x4c007521 // st1 {v1.8h}, [x9]

nextRow:
	ADD  R3, R2, R2
	ADD  R5<<1, R1
	SUB  $1, R6, R6
	B    rowLoop

done:
	RET

// func compoundCopyHighBDNEONAsmS4(ctx *compoundCopy8NEONCtx)
//
// Resident high-bit-depth copy to CONV_BUF. For 10-bit compound prediction
// ROUND0_BITS is 3, so the copy scale is << 4. The wrapper only routes
// resident width-multiple-of-eight blocks here.
TEXT ·compoundCopyHighBDNEONAsmS4(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_REF(R0), R2
	MOVD C_REFSTR(R0), R3
	MOVD C_WIDTH(R0), R5
	MOVD C_HEIGHT(R0), R6
	MOVD C_ROUNDOFFSET(R0), R7

	WORD $0x4e020ce4 // dup v4.8h, w7

hccS4RowLoop:
	CBZ  R6, hccS4Done
	MOVD R2, R8
	MOVD R1, R9
	MOVD R5, R10

hccS4ColLoop:
	WORD $0x4c407501 // ld1 {v1.8h}, [x8]
	WORD $0x4f145421 // shl v1.8h, v1.8h, #4
	WORD $0x4e648421 // add v1.8h, v1.8h, v4.8h
	WORD $0x4c007521 // st1 {v1.8h}, [x9]
	ADD  $16, R8, R8
	ADD  $16, R9, R9
	SUB  $8, R10, R10
	CBNZ R10, hccS4ColLoop

	ADD  R3, R2, R2
	ADD  R5<<1, R1
	SUB  $1, R6, R6
	CBNZ R6, hccS4RowLoop

hccS4Done:
	RET

// func compoundCopyHighBDNEONAsmS2(ctx *compoundCopy8NEONCtx)
//
// The 12-bit path uses ROUND0_BITS == 5, leaving a << 2 copy scale before the
// compound CONV_BUF round offset.
TEXT ·compoundCopyHighBDNEONAsmS2(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_REF(R0), R2
	MOVD C_REFSTR(R0), R3
	MOVD C_WIDTH(R0), R5
	MOVD C_HEIGHT(R0), R6
	MOVD C_ROUNDOFFSET(R0), R7

	WORD $0x4e020ce4 // dup v4.8h, w7

hccS2RowLoop:
	CBZ  R6, hccS2Done
	MOVD R2, R8
	MOVD R1, R9
	MOVD R5, R10

hccS2ColLoop:
	WORD $0x4c407501 // ld1 {v1.8h}, [x8]
	WORD $0x4f125421 // shl v1.8h, v1.8h, #2
	WORD $0x4e648421 // add v1.8h, v1.8h, v4.8h
	WORD $0x4c007521 // st1 {v1.8h}, [x9]
	ADD  $16, R8, R8
	ADD  $16, R9, R9
	SUB  $8, R10, R10
	CBNZ R10, hccS2ColLoop

	ADD  R3, R2, R2
	ADD  R5<<1, R1
	SUB  $1, R6, R6
	CBNZ R6, hccS2RowLoop

hccS2Done:
	RET

#define X_DST         0
#define X_REF         8
#define X_KERNEL      16
#define X_REFSTR      24
#define X_WIDTH       32
#define X_HEIGHT      40
#define X_ROUNDOFFSET 48

#define HX_DST         0
#define HX_REF         8
#define HX_KERNEL      16
#define HX_REFSTR      24
#define HX_WIDTH       32
#define HX_HEIGHT      40
#define HX_ROUND0      48
#define HX_ROUNDOFFSET 56

// func compoundXHighBDNEONAsm(ctx *compoundFilterHighBDNEONCtx)
//
// Resident high-bit-depth horizontal joint convolve with do_average == 0:
//     dst = round(k0*s0 + ... + k7*s7, round0) + roundOffset
// This is the local round0-aware form of SVT's highbd_jnt_convolve_x_neon.
TEXT ·compoundXHighBDNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD HX_DST(R0), R1
	MOVD HX_REF(R0), R2
	MOVD HX_KERNEL(R0), R3
	MOVD HX_REFSTR(R0), R5
	MOVD HX_WIDTH(R0), R6
	MOVD HX_HEIGHT(R0), R7
	MOVD HX_ROUND0(R0), R11
	MOVD HX_ROUNDOFFSET(R0), R12

	WORD $0x4c407460       // ld1 {v0.8h}, [x3]  load 8 taps
	NEG  R11, R11
	WORD $0x4e040d74       // dup v20.4s, w11    -round0
	MOVD R12, R11
	WORD $0x4e040d75       // dup v21.4s, w11    roundOffset

hcxRowLoop:
	CBZ  R7, hcxDone
	MOVD R1, R10
	MOVD R2, R9
	MOVD R6, R8

hcxColLoop:
	WORD $0x4c407521       // ld1 {v1.8h}, [x9]       s0
	ADD  $2, R9, R15
	WORD $0x4c4075e2       // ld1 {v2.8h}, [x15]      s1
	ADD  $4, R9, R15
	WORD $0x4c4075e3       // ld1 {v3.8h}, [x15]      s2
	ADD  $6, R9, R15
	WORD $0x4c4075e4       // ld1 {v4.8h}, [x15]      s3
	ADD  $8, R9, R15
	WORD $0x4c4075e5       // ld1 {v5.8h}, [x15]      s4
	ADD  $10, R9, R15
	WORD $0x4c4075e6       // ld1 {v6.8h}, [x15]      s5
	ADD  $12, R9, R15
	WORD $0x4c4075e7       // ld1 {v7.8h}, [x15]      s6
	ADD  $14, R9, R15
	WORD $0x4c4075e8       // ld1 {v8.8h}, [x15]      s7

	WORD $0x4f000410       // movi v16.4s, #0
	WORD $0x4f000411       // movi v17.4s, #0
	WORD $0x0f402030       // smlal  v16.4s, v1.4h, v0.h[0]
	WORD $0x4f402031       // smlal2 v17.4s, v1.8h, v0.h[0]
	WORD $0x0f502050       // smlal  v16.4s, v2.4h, v0.h[1]
	WORD $0x4f502051       // smlal2 v17.4s, v2.8h, v0.h[1]
	WORD $0x0f602070       // smlal  v16.4s, v3.4h, v0.h[2]
	WORD $0x4f602071       // smlal2 v17.4s, v3.8h, v0.h[2]
	WORD $0x0f702090       // smlal  v16.4s, v4.4h, v0.h[3]
	WORD $0x4f702091       // smlal2 v17.4s, v4.8h, v0.h[3]
	WORD $0x0f4028b0       // smlal  v16.4s, v5.4h, v0.h[4]
	WORD $0x4f4028b1       // smlal2 v17.4s, v5.8h, v0.h[4]
	WORD $0x0f5028d0       // smlal  v16.4s, v6.4h, v0.h[5]
	WORD $0x4f5028d1       // smlal2 v17.4s, v6.8h, v0.h[5]
	WORD $0x0f6028f0       // smlal  v16.4s, v7.4h, v0.h[6]
	WORD $0x4f6028f1       // smlal2 v17.4s, v7.8h, v0.h[6]
	WORD $0x0f702910       // smlal  v16.4s, v8.4h, v0.h[7]
	WORD $0x4f702911       // smlal2 v17.4s, v8.8h, v0.h[7]

	WORD $0x4eb45610       // srshl v16.4s, v16.4s, v20.4s
	WORD $0x4eb45631       // srshl v17.4s, v17.4s, v20.4s
	WORD $0x4eb58610       // add v16.4s, v16.4s, v21.4s
	WORD $0x4eb58631       // add v17.4s, v17.4s, v21.4s
	WORD $0x0e612a10       // xtn  v16.4h, v16.4s
	WORD $0x4e612a30       // xtn2 v16.8h, v17.4s
	WORD $0x4c007550       // st1 {v16.8h}, [x10]

	ADD  $16, R10, R10
	ADD  $16, R9, R9
	SUB  $8, R8, R8
	CBNZ R8, hcxColLoop

	ADD  R6<<1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, hcxRowLoop

hcxDone:
	RET

// func compoundXHighBDNEONAsmW4(ctx *compoundFilterHighBDNEONCtx)
//
// Width-4 high-bit-depth compound-X. The wrapper passes refX-1 and
// x_filter_ptr+2, matching SVT's width-4 4-tap branch.
TEXT ·compoundXHighBDNEONAsmW4(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD HX_DST(R0), R1
	MOVD HX_REF(R0), R2
	MOVD HX_KERNEL(R0), R3
	MOVD HX_REFSTR(R0), R5
	MOVD HX_HEIGHT(R0), R7
	MOVD HX_ROUND0(R0), R11
	MOVD HX_ROUNDOFFSET(R0), R12

	WORD $0x0c407460       // ld1 {v0.4h}, [x3]  load k2..k5
	NEG  R11, R11
	WORD $0x4e040d74       // dup v20.4s, w11    -round0
	MOVD R12, R11
	WORD $0x4e040d75       // dup v21.4s, w11    roundOffset

hcx4RowLoop:
	CBZ  R7, hcx4Done
	MOVD R1, R10
	MOVD R2, R9
	WORD $0x4c407521       // ld1 {v1.8h}, [x9]  s0..s7
	WORD $0x6e011022       // ext v2.16b, v1.16b, v1.16b, #2
	WORD $0x6e012023       // ext v3.16b, v1.16b, v1.16b, #4
	WORD $0x6e013024       // ext v4.16b, v1.16b, v1.16b, #6
	WORD $0x4f000410       // movi v16.4s, #0
	WORD $0x0f402030       // smlal v16.4s, v1.4h, v0.h[0]
	WORD $0x0f502050       // smlal v16.4s, v2.4h, v0.h[1]
	WORD $0x0f602070       // smlal v16.4s, v3.4h, v0.h[2]
	WORD $0x0f702090       // smlal v16.4s, v4.4h, v0.h[3]
	WORD $0x4eb45610       // srshl v16.4s, v16.4s, v20.4s
	WORD $0x4eb58610       // add v16.4s, v16.4s, v21.4s
	WORD $0x0e612a10       // xtn v16.4h, v16.4s
	WORD $0x0c007550       // st1 {v16.4h}, [x10]

	ADD  $8, R1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, hcx4RowLoop

hcx4Done:
	RET

// func compoundYHighBDNEONAsm(ctx *compoundFilterHighBDNEONCtx)
//
// Resident high-bit-depth vertical joint convolve with do_average == 0:
//     dst = round(k0*s0 + ... + k7*s7, round0) + roundOffset
TEXT ·compoundYHighBDNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD HX_DST(R0), R1
	MOVD HX_REF(R0), R2
	MOVD HX_KERNEL(R0), R3
	MOVD HX_REFSTR(R0), R5
	MOVD HX_WIDTH(R0), R6
	MOVD HX_HEIGHT(R0), R7
	MOVD HX_ROUND0(R0), R11
	MOVD HX_ROUNDOFFSET(R0), R12

	WORD $0x4c407460       // ld1 {v0.8h}, [x3]  load 8 taps
	NEG  R11, R11
	WORD $0x4e040d74       // dup v20.4s, w11    -round0
	MOVD R12, R11
	WORD $0x4e040d75       // dup v21.4s, w11    roundOffset

hcyRowLoop:
	CBZ  R7, hcyDone
	MOVD R1, R10           // dst column cursor
	MOVD R2, R13           // ref row-window base
	MOVD R6, R8            // remaining columns

hcyColLoop:
	MOVD R13, R9           // R9 walks tap rows
	WORD $0x4cc57521       // ld1 {v1.8h}, [x9], x5
	WORD $0x4cc57522       // ld1 {v2.8h}, [x9], x5
	WORD $0x4cc57523       // ld1 {v3.8h}, [x9], x5
	WORD $0x4cc57524       // ld1 {v4.8h}, [x9], x5
	WORD $0x4cc57525       // ld1 {v5.8h}, [x9], x5
	WORD $0x4cc57526       // ld1 {v6.8h}, [x9], x5
	WORD $0x4cc57527       // ld1 {v7.8h}, [x9], x5
	WORD $0x4cc57528       // ld1 {v8.8h}, [x9], x5

	WORD $0x4f000410       // movi v16.4s, #0
	WORD $0x4f000411       // movi v17.4s, #0
	WORD $0x0f402030       // smlal  v16.4s, v1.4h, v0.h[0]
	WORD $0x4f402031       // smlal2 v17.4s, v1.8h, v0.h[0]
	WORD $0x0f502050       // smlal  v16.4s, v2.4h, v0.h[1]
	WORD $0x4f502051       // smlal2 v17.4s, v2.8h, v0.h[1]
	WORD $0x0f602070       // smlal  v16.4s, v3.4h, v0.h[2]
	WORD $0x4f602071       // smlal2 v17.4s, v3.8h, v0.h[2]
	WORD $0x0f702090       // smlal  v16.4s, v4.4h, v0.h[3]
	WORD $0x4f702091       // smlal2 v17.4s, v4.8h, v0.h[3]
	WORD $0x0f4028b0       // smlal  v16.4s, v5.4h, v0.h[4]
	WORD $0x4f4028b1       // smlal2 v17.4s, v5.8h, v0.h[4]
	WORD $0x0f5028d0       // smlal  v16.4s, v6.4h, v0.h[5]
	WORD $0x4f5028d1       // smlal2 v17.4s, v6.8h, v0.h[5]
	WORD $0x0f6028f0       // smlal  v16.4s, v7.4h, v0.h[6]
	WORD $0x4f6028f1       // smlal2 v17.4s, v7.8h, v0.h[6]
	WORD $0x0f702910       // smlal  v16.4s, v8.4h, v0.h[7]
	WORD $0x4f702911       // smlal2 v17.4s, v8.8h, v0.h[7]

	WORD $0x4eb45610       // srshl v16.4s, v16.4s, v20.4s
	WORD $0x4eb45631       // srshl v17.4s, v17.4s, v20.4s
	WORD $0x4eb58610       // add v16.4s, v16.4s, v21.4s
	WORD $0x4eb58631       // add v17.4s, v17.4s, v21.4s
	WORD $0x0e612a10       // xtn  v16.4h, v16.4s
	WORD $0x4e612a30       // xtn2 v16.8h, v17.4s
	WORD $0x4c007550       // st1 {v16.8h}, [x10]

	ADD  $16, R10, R10
	ADD  $16, R13, R13
	SUB  $8, R8, R8
	CBNZ R8, hcyColLoop

	ADD  R6<<1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, hcyRowLoop

hcyDone:
	RET

// func compoundYHighBDNEONAsmW4(ctx *compoundFilterHighBDNEONCtx)
TEXT ·compoundYHighBDNEONAsmW4(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD HX_DST(R0), R1
	MOVD HX_REF(R0), R2
	MOVD HX_KERNEL(R0), R3
	MOVD HX_REFSTR(R0), R5
	MOVD HX_HEIGHT(R0), R7
	MOVD HX_ROUND0(R0), R11
	MOVD HX_ROUNDOFFSET(R0), R12

	WORD $0x4c407460       // ld1 {v0.8h}, [x3]  load 8 taps
	NEG  R11, R11
	WORD $0x4e040d74       // dup v20.4s, w11    -round0
	MOVD R12, R11
	WORD $0x4e040d75       // dup v21.4s, w11    roundOffset

hcy4RowLoop:
	CBZ  R7, hcy4Done
	MOVD R1, R10
	MOVD R2, R9
	WORD $0x0cc57521       // ld1 {v1.4h}, [x9], x5
	WORD $0x0cc57522       // ld1 {v2.4h}, [x9], x5
	WORD $0x0cc57523       // ld1 {v3.4h}, [x9], x5
	WORD $0x0cc57524       // ld1 {v4.4h}, [x9], x5
	WORD $0x0cc57525       // ld1 {v5.4h}, [x9], x5
	WORD $0x0cc57526       // ld1 {v6.4h}, [x9], x5
	WORD $0x0cc57527       // ld1 {v7.4h}, [x9], x5
	WORD $0x0cc57528       // ld1 {v8.4h}, [x9], x5

	WORD $0x4f000410       // movi v16.4s, #0
	WORD $0x0f402030       // smlal v16.4s, v1.4h, v0.h[0]
	WORD $0x0f502050       // smlal v16.4s, v2.4h, v0.h[1]
	WORD $0x0f602070       // smlal v16.4s, v3.4h, v0.h[2]
	WORD $0x0f702090       // smlal v16.4s, v4.4h, v0.h[3]
	WORD $0x0f4028b0       // smlal v16.4s, v5.4h, v0.h[4]
	WORD $0x0f5028d0       // smlal v16.4s, v6.4h, v0.h[5]
	WORD $0x0f6028f0       // smlal v16.4s, v7.4h, v0.h[6]
	WORD $0x0f702910       // smlal v16.4s, v8.4h, v0.h[7]
	WORD $0x4eb45610       // srshl v16.4s, v16.4s, v20.4s
	WORD $0x4eb58610       // add v16.4s, v16.4s, v21.4s
	WORD $0x0e612a10       // xtn v16.4h, v16.4s
	WORD $0x0c007550       // st1 {v16.4h}, [x10]

	ADD  $8, R1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, hcy4RowLoop

hcy4Done:
	RET

#define HD2D_DST    0
#define HD2D_REF    8
#define HD2D_KERNEL 16
#define HD2D_XKERN  24
#define HD2D_REFSTR 32
#define HD2D_WIDTH  40
#define HD2D_HEIGHT 48
#define HD2D_IM     56
#define HD2D_IMSTR  64
#define HD2D_ROUND0 72
#define HD2D_XBIAS  80
#define HD2D_YBIAS  88

// func compound2DHighBDNEONAsm(ctx *compound2DHighBDNEONCtx)
//
// Resident high-bit-depth 2D joint convolve with do_average == 0.
//   Pass1: im[int32] = round(xBias + horizontal_sum, round0).
//   Pass2: dst       = round(yBias + vertical_sum, COMPOUND_ROUND1_BITS).
TEXT ·compound2DHighBDNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD HD2D_DST(R0), R1
	MOVD HD2D_REF(R0), R2
	MOVD HD2D_KERNEL(R0), R3
	MOVD HD2D_XKERN(R0), R12
	MOVD HD2D_REFSTR(R0), R5
	MOVD HD2D_WIDTH(R0), R6
	MOVD HD2D_HEIGHT(R0), R7
	MOVD HD2D_IM(R0), R17
	MOVD HD2D_IMSTR(R0), R14
	LSL  $2, R14, R14

	ADD  $7, R7, R19
	WORD $0x4c407580       // ld1 {v0.8h}, [x12]  horizontal taps
	MOVD HD2D_XBIAS(R0), R11
	WORD $0x4e040d72       // dup v18.4s, w11     xBias
	MOVD HD2D_ROUND0(R0), R11
	NEG  R11, R11
	WORD $0x4e040d74       // dup v20.4s, w11     -round0

	MOVD R2, R21           // ref row cursor
	MOVD R17, R22          // im row cursor

hc2hRowLoop:
	CBZ  R19, hc2hDone
	MOVD R21, R9
	MOVD R22, R10
	MOVD R6, R8

hc2hColLoop:
	WORD $0x4c407521       // ld1 {v1.8h}, [x9]   s0..s7
	ADD  $16, R9, R15
	WORD $0x0c4075e3       // ld1 {v3.4h}, [x15]  s8..s11
	WORD $0x4eb21e50       // mov v16.16b, v18.16b
	WORD $0x0f402030       // smlal v16.4s, v1.4h, v0.h[0]
	WORD $0x6e031024       // ext v4.16b, v1.16b, v3.16b, #2
	WORD $0x0f502090       // smlal v16.4s, v4.4h, v0.h[1]
	WORD $0x6e032024       // ext v4.16b, v1.16b, v3.16b, #4
	WORD $0x0f602090       // smlal v16.4s, v4.4h, v0.h[2]
	WORD $0x6e033024       // ext v4.16b, v1.16b, v3.16b, #6
	WORD $0x0f702090       // smlal v16.4s, v4.4h, v0.h[3]
	WORD $0x6e034024       // ext v4.16b, v1.16b, v3.16b, #8
	WORD $0x0f402890       // smlal v16.4s, v4.4h, v0.h[4]
	WORD $0x6e035024       // ext v4.16b, v1.16b, v3.16b, #10
	WORD $0x0f502890       // smlal v16.4s, v4.4h, v0.h[5]
	WORD $0x6e036024       // ext v4.16b, v1.16b, v3.16b, #12
	WORD $0x0f602890       // smlal v16.4s, v4.4h, v0.h[6]
	WORD $0x6e037024       // ext v4.16b, v1.16b, v3.16b, #14
	WORD $0x0f702890       // smlal v16.4s, v4.4h, v0.h[7]

	WORD $0x4eb45610       // srshl v16.4s, v16.4s, v20.4s
	WORD $0x4c007950       // st1 {v16.4s}, [x10]

	ADD  $8, R9, R9
	ADD  $16, R10, R10
	SUB  $4, R8, R8
	CBNZ R8, hc2hColLoop

	ADD  R5, R21, R21
	ADD  R14, R22, R22
	SUB  $1, R19, R19
	CBNZ R19, hc2hRowLoop

hc2hDone:
	WORD $0x4c407460       // ld1 {v0.8h}, [x3]   vertical taps
	MOVD HD2D_YBIAS(R0), R11
	WORD $0x4e040d72       // dup v18.4s, w11     yBias

	MOVD R17, R22          // im row-window base

hc2vRowLoop:
	CBZ  R7, hc2vDone
	MOVD R1, R10
	MOVD R22, R23
	MOVD R6, R8

hc2vColLoop:
	MOVD R23, R9
	MOVD R3, R12
	WORD $0x4eb21e50       // mov v16.16b, v18.16b

	WORD $0x4cce7921       // ld1 {v1.4s}, [x9], x14
	MOVH (R12), R13
	ADD  $2, R12, R12
	MOVD R13, R11
	WORD $0x4e040d7c       // dup v28.4s, w11
	WORD $0x4ebc9430       // mla v16.4s, v1.4s, v28.4s
	WORD $0x4cce7921       // ld1 {v1.4s}, [x9], x14
	MOVH (R12), R13
	ADD  $2, R12, R12
	MOVD R13, R11
	WORD $0x4e040d7c
	WORD $0x4ebc9430
	WORD $0x4cce7921
	MOVH (R12), R13
	ADD  $2, R12, R12
	MOVD R13, R11
	WORD $0x4e040d7c
	WORD $0x4ebc9430
	WORD $0x4cce7921
	MOVH (R12), R13
	ADD  $2, R12, R12
	MOVD R13, R11
	WORD $0x4e040d7c
	WORD $0x4ebc9430
	WORD $0x4cce7921
	MOVH (R12), R13
	ADD  $2, R12, R12
	MOVD R13, R11
	WORD $0x4e040d7c
	WORD $0x4ebc9430
	WORD $0x4cce7921
	MOVH (R12), R13
	ADD  $2, R12, R12
	MOVD R13, R11
	WORD $0x4e040d7c
	WORD $0x4ebc9430
	WORD $0x4cce7921
	MOVH (R12), R13
	ADD  $2, R12, R12
	MOVD R13, R11
	WORD $0x4e040d7c
	WORD $0x4ebc9430
	WORD $0x4cce7921
	MOVH (R12), R13
	MOVD R13, R11
	WORD $0x4e040d7c
	WORD $0x4ebc9430

	WORD $0x4f392610       // srshr v16.4s, v16.4s, #7
	WORD $0x0e612a10       // xtn v16.4h, v16.4s
	WORD $0x0c007550       // st1 {v16.4h}, [x10]

	ADD  $8, R10, R10
	ADD  $16, R23, R23
	SUB  $4, R8, R8
	CBNZ R8, hc2vColLoop

	ADD  R6<<1, R1
	ADD  R14, R22, R22
	SUB  $1, R7, R7
	CBNZ R7, hc2vRowLoop

hc2vDone:
	RET

// func compoundX8NEONAsm(ctx *compoundFilter8NEONCtx)
//
// Resident horizontal 8-tap joint convolve with do_average == 0:
//     dst = round((k0*s0 + ... + k7*s7), ROUND0_BITS) + roundOffset
// The accumulator and tap order match convolveX8NEONAsm; unlike the pixel
// predictor, compound CONV_BUF output stops before the second round and clamp.
TEXT ·compoundX8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD X_DST(R0), R1
	MOVD X_REF(R0), R2
	MOVD X_KERNEL(R0), R3
	MOVD X_REFSTR(R0), R5
	MOVD X_WIDTH(R0), R6
	MOVD X_HEIGHT(R0), R7
	MOVD X_ROUNDOFFSET(R0), R11

	WORD $0x4c407460 // ld1 {v0.8h}, [x3]   load 8 taps
	WORD $0x4e020d65 // dup v5.8h, w11      roundOffset

cxRowLoop:
	CBZ  R7, cxDone
	MOVD R1, R10 // dst cursor
	MOVD R2, R9  // ref window cursor
	MOVD R6, R8  // remaining columns

cxColLoop:
	WORD $0x4f000410 // movi v16.4s, #0
	WORD $0x4f000411 // movi v17.4s, #0
	WORD $0x4c407121 // ld1 {v1.16b}, [x9]
	WORD $0x2f08a422 // ushll  v2.8h, v1.8b, #0
	WORD $0x6f08a423 // ushll2 v3.8h, v1.16b, #0
	WORD $0x0f402050 // smlal  v16.4s, v2.4h, v0.h[0]
	WORD $0x4f402051 // smlal2 v17.4s, v2.8h, v0.h[0]
	WORD $0x6e031044 // ext v4.16b, v2.16b, v3.16b, #2
	WORD $0x0f502090 // smlal  v16.4s, v4.4h, v0.h[1]
	WORD $0x4f502091 // smlal2 v17.4s, v4.8h, v0.h[1]
	WORD $0x6e032044 // ext v4.16b, v2.16b, v3.16b, #4
	WORD $0x0f602090 // smlal  v16.4s, v4.4h, v0.h[2]
	WORD $0x4f602091 // smlal2 v17.4s, v4.8h, v0.h[2]
	WORD $0x6e033044 // ext v4.16b, v2.16b, v3.16b, #6
	WORD $0x0f702090 // smlal  v16.4s, v4.4h, v0.h[3]
	WORD $0x4f702091 // smlal2 v17.4s, v4.8h, v0.h[3]
	WORD $0x6e034044 // ext v4.16b, v2.16b, v3.16b, #8
	WORD $0x0f402890 // smlal  v16.4s, v4.4h, v0.h[4]
	WORD $0x4f402891 // smlal2 v17.4s, v4.8h, v0.h[4]
	WORD $0x6e035044 // ext v4.16b, v2.16b, v3.16b, #10
	WORD $0x0f502890 // smlal  v16.4s, v4.4h, v0.h[5]
	WORD $0x4f502891 // smlal2 v17.4s, v4.8h, v0.h[5]
	WORD $0x6e036044 // ext v4.16b, v2.16b, v3.16b, #12
	WORD $0x0f602890 // smlal  v16.4s, v4.4h, v0.h[6]
	WORD $0x4f602891 // smlal2 v17.4s, v4.8h, v0.h[6]
	WORD $0x6e037044 // ext v4.16b, v2.16b, v3.16b, #14
	WORD $0x0f702890 // smlal  v16.4s, v4.4h, v0.h[7]
	WORD $0x4f702891 // smlal2 v17.4s, v4.8h, v0.h[7]

	WORD $0x4f3d2610 // srshr v16.4s, v16.4s, #3
	WORD $0x4f3d2631 // srshr v17.4s, v17.4s, #3
	WORD $0x0e614a10 // sqxtn  v16.4h, v16.4s
	WORD $0x4e614a30 // sqxtn2 v16.8h, v17.4s
	WORD $0x4e658610 // add v16.8h, v16.8h, v5.8h
	WORD $0x4c007550 // st1 {v16.8h}, [x10]

	ADD  $16, R10, R10
	ADD  $8, R9, R9
	SUB  $8, R8, R8
	CBNZ R8, cxColLoop

	ADD  R6<<1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, cxRowLoop

cxDone:
	RET

// func compoundY8NEONAsm(ctx *compoundFilter8NEONCtx)
//
// Resident vertical 8-tap joint convolve with do_average == 0. The vertical
// sum carries FILTER_BITS precision, so compound output is round(sum, 3) plus
// the CONV_BUF round offset.
TEXT ·compoundY8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD X_DST(R0), R1
	MOVD X_REF(R0), R2
	MOVD X_KERNEL(R0), R3
	MOVD X_REFSTR(R0), R5
	MOVD X_WIDTH(R0), R6
	MOVD X_HEIGHT(R0), R7
	MOVD X_ROUNDOFFSET(R0), R12

	WORD $0x4c407460 // ld1 {v0.8h}, [x3]   load 8 taps
	WORD $0x4e020d85 // dup v5.8h, w12      roundOffset

cyRowLoop:
	CBZ  R7, cyDone
	MOVD R1, R10 // dst cursor
	MOVD R2, R11 // ref row-window base for this output row
	MOVD R6, R8  // remaining columns

cyColLoop:
	MOVD R11, R9 // R9 walks the 8 tap rows; post-index by ref stride
	WORD $0x4f000410 // movi v16.4s, #0
	WORD $0x4f000411 // movi v17.4s, #0

	WORD $0x0cc57121 // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f402030 // smlal  v16.4s, v1.4h, v0.h[0]
	WORD $0x4f402031 // smlal2 v17.4s, v1.8h, v0.h[0]
	WORD $0x0cc57121 // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f502030 // smlal  v16.4s, v1.4h, v0.h[1]
	WORD $0x4f502031 // smlal2 v17.4s, v1.8h, v0.h[1]
	WORD $0x0cc57121 // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f602030 // smlal  v16.4s, v1.4h, v0.h[2]
	WORD $0x4f602031 // smlal2 v17.4s, v1.8h, v0.h[2]
	WORD $0x0cc57121 // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f702030 // smlal  v16.4s, v1.4h, v0.h[3]
	WORD $0x4f702031 // smlal2 v17.4s, v1.8h, v0.h[3]
	WORD $0x0cc57121 // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f402830 // smlal  v16.4s, v1.4h, v0.h[4]
	WORD $0x4f402831 // smlal2 v17.4s, v1.8h, v0.h[4]
	WORD $0x0cc57121 // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f502830 // smlal  v16.4s, v1.4h, v0.h[5]
	WORD $0x4f502831 // smlal2 v17.4s, v1.8h, v0.h[5]
	WORD $0x0cc57121 // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f602830 // smlal  v16.4s, v1.4h, v0.h[6]
	WORD $0x4f602831 // smlal2 v17.4s, v1.8h, v0.h[6]
	WORD $0x0cc57121 // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f702830 // smlal  v16.4s, v1.4h, v0.h[7]
	WORD $0x4f702831 // smlal2 v17.4s, v1.8h, v0.h[7]

	WORD $0x4f3d2610 // srshr v16.4s, v16.4s, #3
	WORD $0x4f3d2631 // srshr v17.4s, v17.4s, #3
	WORD $0x0e614a10 // sqxtn  v16.4h, v16.4s
	WORD $0x4e614a30 // sqxtn2 v16.8h, v17.4s
	WORD $0x4e658610 // add v16.8h, v16.8h, v5.8h
	WORD $0x4c007550 // st1 {v16.8h}, [x10]

	ADD  $16, R10, R10
	ADD  $8, R11, R11
	SUB  $8, R8, R8
	CBNZ R8, cyColLoop

	ADD  R6<<1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, cyRowLoop

cyDone:
	RET

// Width-4 compound CONV_BUF kernels. These mirror the proven width-4 8-bit
// convolve kernels, but they store four uint16 CONV_BUF samples per row instead
// of clipped pixels.

// func compoundCopy8NEONAsmW4(ctx *compoundCopy8NEONCtx)
TEXT ·compoundCopy8NEONAsmW4(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_REF(R0), R2
	MOVD C_REFSTR(R0), R3
	MOVD C_HEIGHT(R0), R6
	MOVD C_ROUNDOFFSET(R0), R7
	MOVD $4, R5

	WORD $0x4e020ce4 // dup v4.8h, w7

cc4RowLoop:
	CBZ  R6, cc4Done
	MOVD R2, R9
	MOVD R1, R17
	WORD $0x0dc58121 // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f0ca421 // ushll v1.8h, v1.8b, #4
	WORD $0x4e648421 // add v1.8h, v1.8h, v4.8h
	WORD $0x0c007621 // st1 {v1.4h}, [x17]

	ADD  R3, R2, R2
	ADD  $8, R1, R1
	SUB  $1, R6, R6
	CBNZ R6, cc4RowLoop

cc4Done:
	RET

// func compoundX8NEONAsmW4(ctx *compoundFilter8NEONCtx)
TEXT ·compoundX8NEONAsmW4(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD X_DST(R0), R1
	MOVD X_REF(R0), R2
	MOVD X_KERNEL(R0), R3
	MOVD X_REFSTR(R0), R5
	MOVD X_HEIGHT(R0), R7
	MOVD X_ROUNDOFFSET(R0), R11

	WORD $0x4c407460 // ld1 {v0.8h}, [x3]   load 8 taps

cx4RowLoop:
	CBZ  R7, cx4Done
	MOVD R2, R9
	MOVD R1, R17
	ADD  $8, R9, R15
	WORD $0x4f000410 // movi v16.4s, #0
	WORD $0x0c407121 // ld1 {v1.8b}, [x9]
	WORD $0x2f08a422 // ushll v2.8h, v1.8b, #0
	WORD $0x0d4081e5 // ld1 {v5.s}[0], [x15]
	WORD $0x2f08a4a3 // ushll v3.8h, v5.8b, #0
	WORD $0x0f402050 // smlal v16.4s, v2.4h, v0.h[0]
	WORD $0x6e031044 // ext v4.16b, v2.16b, v3.16b, #2
	WORD $0x0f502090 // smlal v16.4s, v4.4h, v0.h[1]
	WORD $0x6e032044 // ext v4.16b, v2.16b, v3.16b, #4
	WORD $0x0f602090 // smlal v16.4s, v4.4h, v0.h[2]
	WORD $0x6e033044 // ext v4.16b, v2.16b, v3.16b, #6
	WORD $0x0f702090 // smlal v16.4s, v4.4h, v0.h[3]
	WORD $0x6e034044 // ext v4.16b, v2.16b, v3.16b, #8
	WORD $0x0f402890 // smlal v16.4s, v4.4h, v0.h[4]
	WORD $0x6e035044 // ext v4.16b, v2.16b, v3.16b, #10
	WORD $0x0f502890 // smlal v16.4s, v4.4h, v0.h[5]
	WORD $0x6e036044 // ext v4.16b, v2.16b, v3.16b, #12
	WORD $0x0f602890 // smlal v16.4s, v4.4h, v0.h[6]
	WORD $0x6e037044 // ext v4.16b, v2.16b, v3.16b, #14
	WORD $0x0f702890 // smlal v16.4s, v4.4h, v0.h[7]

	WORD $0x4f3d2610 // srshr v16.4s, v16.4s, #3
	WORD $0x0e614a10 // sqxtn v16.4h, v16.4s
	WORD $0x4e020d65 // dup v5.8h, w11
	WORD $0x4e658610 // add v16.8h, v16.8h, v5.8h
	WORD $0x0c007630 // st1 {v16.4h}, [x17]

	ADD  $8, R1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, cx4RowLoop

cx4Done:
	RET

// func compoundY8NEONAsmW4(ctx *compoundFilter8NEONCtx)
TEXT ·compoundY8NEONAsmW4(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD X_DST(R0), R1
	MOVD X_REF(R0), R2
	MOVD X_KERNEL(R0), R3
	MOVD X_REFSTR(R0), R5
	MOVD X_HEIGHT(R0), R7
	MOVD X_ROUNDOFFSET(R0), R12

	WORD $0x4c407460 // ld1 {v0.8h}, [x3]   load 8 taps

cy4RowLoop:
	CBZ  R7, cy4Done
	MOVD R2, R9
	MOVD R1, R17
	WORD $0x4f000410 // movi v16.4s, #0

	WORD $0x0dc58121 // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f402030 // smlal v16.4s, v1.4h, v0.h[0]
	WORD $0x0dc58121 // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f502030 // smlal v16.4s, v1.4h, v0.h[1]
	WORD $0x0dc58121 // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f602030 // smlal v16.4s, v1.4h, v0.h[2]
	WORD $0x0dc58121 // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f702030 // smlal v16.4s, v1.4h, v0.h[3]
	WORD $0x0dc58121 // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f402830 // smlal v16.4s, v1.4h, v0.h[4]
	WORD $0x0dc58121 // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f502830 // smlal v16.4s, v1.4h, v0.h[5]
	WORD $0x0dc58121 // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f602830 // smlal v16.4s, v1.4h, v0.h[6]
	WORD $0x0dc58121 // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421 // ushll v1.8h, v1.8b, #0
	WORD $0x0f702830 // smlal v16.4s, v1.4h, v0.h[7]

	WORD $0x4f3d2610 // srshr v16.4s, v16.4s, #3
	WORD $0x0e614a10 // sqxtn v16.4h, v16.4s
	WORD $0x4e020d85 // dup v5.8h, w12
	WORD $0x4e658610 // add v16.8h, v16.8h, v5.8h
	WORD $0x0c007630 // st1 {v16.4h}, [x17]

	ADD  $8, R1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, cy4RowLoop

cy4Done:
	RET

#define D_DST    0
#define D_REF    8
#define D_KERNEL 16
#define D_XKERN  24
#define D_REFSTR 32
#define D_WIDTH  40
#define D_HEIGHT 48
#define D_IM     56
#define D_IMSTR  64

// func compound2D8NEONAsm(ctx *compound2D8NEONCtx)
//
// Resident 8-bit 2D joint convolve with do_average == 0. The horizontal pass
// matches SVT's int16 intermediate: round(xBias + horizontal_sum, 3). The
// vertical pass writes CONV_BUF precision: round(yBias + vertical_sum, 7).
TEXT ·compound2D8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD D_DST(R0), R1
	MOVD D_REF(R0), R2
	MOVD D_KERNEL(R0), R3
	MOVD D_XKERN(R0), R12
	MOVD D_REFSTR(R0), R5
	MOVD D_WIDTH(R0), R6
	MOVD D_HEIGHT(R0), R7
	MOVD D_IM(R0), R13
	MOVD D_IMSTR(R0), R14
	LSL  $1, R14, R14 // int16 elements -> bytes

	ADD  $7, R7, R15 // imH = height + filterTaps - 1

	WORD $0x4c407580 // ld1 {v0.8h}, [x12]  horizontal taps
	MOVD $16384, R11 // xBias = 1 << (8 + FILTER_BITS - 1)
	WORD $0x4e040d72 // dup v18.4s, w11

	MOVD R2, R16  // ref row cursor
	MOVD R13, R17 // im row cursor

c2hRowLoop:
	CBZ  R15, c2hDone
	MOVD R16, R9
	MOVD R17, R10
	MOVD R6, R8

c2hColLoop:
	WORD $0x4f000410 // movi v16.4s, #0
	WORD $0x4f000411 // movi v17.4s, #0
	WORD $0x4c407121 // ld1 {v1.16b}, [x9]
	WORD $0x2f08a422 // ushll  v2.8h, v1.8b, #0
	WORD $0x6f08a423 // ushll2 v3.8h, v1.16b, #0
	WORD $0x0f402050 // smlal  v16.4s, v2.4h, v0.h[0]
	WORD $0x4f402051 // smlal2 v17.4s, v2.8h, v0.h[0]
	WORD $0x6e031044 // ext v4.16b, v2.16b, v3.16b, #2
	WORD $0x0f502090 // smlal  v16.4s, v4.4h, v0.h[1]
	WORD $0x4f502091 // smlal2 v17.4s, v4.8h, v0.h[1]
	WORD $0x6e032044 // ext v4.16b, v2.16b, v3.16b, #4
	WORD $0x0f602090 // smlal  v16.4s, v4.4h, v0.h[2]
	WORD $0x4f602091 // smlal2 v17.4s, v4.8h, v0.h[2]
	WORD $0x6e033044 // ext v4.16b, v2.16b, v3.16b, #6
	WORD $0x0f702090 // smlal  v16.4s, v4.4h, v0.h[3]
	WORD $0x4f702091 // smlal2 v17.4s, v4.8h, v0.h[3]
	WORD $0x6e034044 // ext v4.16b, v2.16b, v3.16b, #8
	WORD $0x0f402890 // smlal  v16.4s, v4.4h, v0.h[4]
	WORD $0x4f402891 // smlal2 v17.4s, v4.8h, v0.h[4]
	WORD $0x6e035044 // ext v4.16b, v2.16b, v3.16b, #10
	WORD $0x0f502890 // smlal  v16.4s, v4.4h, v0.h[5]
	WORD $0x4f502891 // smlal2 v17.4s, v4.8h, v0.h[5]
	WORD $0x6e036044 // ext v4.16b, v2.16b, v3.16b, #12
	WORD $0x0f602890 // smlal  v16.4s, v4.4h, v0.h[6]
	WORD $0x4f602891 // smlal2 v17.4s, v4.8h, v0.h[6]
	WORD $0x6e037044 // ext v4.16b, v2.16b, v3.16b, #14
	WORD $0x0f702890 // smlal  v16.4s, v4.4h, v0.h[7]
	WORD $0x4f702891 // smlal2 v17.4s, v4.8h, v0.h[7]

	WORD $0x4eb28610 // add v16.4s, v16.4s, v18.4s
	WORD $0x4eb28631 // add v17.4s, v17.4s, v18.4s
	WORD $0x4f3d2610 // srshr v16.4s, v16.4s, #3
	WORD $0x4f3d2631 // srshr v17.4s, v17.4s, #3
	WORD $0x0e612a10 // xtn  v16.4h, v16.4s
	WORD $0x4e612a30 // xtn2 v16.8h, v17.4s
	WORD $0x4c007550 // st1 {v16.8h}, [x10]

	ADD  $8, R9, R9
	ADD  $16, R10, R10
	SUB  $8, R8, R8
	CBNZ R8, c2hColLoop

	ADD  R5, R16, R16
	ADD  R14, R17, R17
	SUB  $1, R15, R15
	CBNZ R15, c2hRowLoop

c2hDone:
	WORD $0x4c407460 // ld1 {v0.8h}, [x3]  vertical taps
	MOVD $524288, R11 // yBias = 1 << 19
	WORD $0x4e040d72 // dup v18.4s, w11

	MOVD R13, R17 // im row-window base

c2vRowLoop:
	CBZ  R7, c2vDone
	MOVD R1, R10
	MOVD R17, R11
	MOVD R6, R8

c2vColLoop:
	MOVD R11, R9
	WORD $0x4eb21e50 // mov v16.16b, v18.16b
	WORD $0x4eb21e51 // mov v17.16b, v18.16b

	WORD $0x4cce7521 // ld1 {v1.8h}, [x9], x14
	WORD $0x0f402030 // smlal  v16.4s, v1.4h, v0.h[0]
	WORD $0x4f402031 // smlal2 v17.4s, v1.8h, v0.h[0]
	WORD $0x4cce7521 // ld1 {v1.8h}, [x9], x14
	WORD $0x0f502030 // smlal  v16.4s, v1.4h, v0.h[1]
	WORD $0x4f502031 // smlal2 v17.4s, v1.8h, v0.h[1]
	WORD $0x4cce7521 // ld1 {v1.8h}, [x9], x14
	WORD $0x0f602030 // smlal  v16.4s, v1.4h, v0.h[2]
	WORD $0x4f602031 // smlal2 v17.4s, v1.8h, v0.h[2]
	WORD $0x4cce7521 // ld1 {v1.8h}, [x9], x14
	WORD $0x0f702030 // smlal  v16.4s, v1.4h, v0.h[3]
	WORD $0x4f702031 // smlal2 v17.4s, v1.8h, v0.h[3]
	WORD $0x4cce7521 // ld1 {v1.8h}, [x9], x14
	WORD $0x0f402830 // smlal  v16.4s, v1.4h, v0.h[4]
	WORD $0x4f402831 // smlal2 v17.4s, v1.8h, v0.h[4]
	WORD $0x4cce7521 // ld1 {v1.8h}, [x9], x14
	WORD $0x0f502830 // smlal  v16.4s, v1.4h, v0.h[5]
	WORD $0x4f502831 // smlal2 v17.4s, v1.8h, v0.h[5]
	WORD $0x4cce7521 // ld1 {v1.8h}, [x9], x14
	WORD $0x0f602830 // smlal  v16.4s, v1.4h, v0.h[6]
	WORD $0x4f602831 // smlal2 v17.4s, v1.8h, v0.h[6]
	WORD $0x4cce7521 // ld1 {v1.8h}, [x9], x14
	WORD $0x0f702830 // smlal  v16.4s, v1.4h, v0.h[7]
	WORD $0x4f702831 // smlal2 v17.4s, v1.8h, v0.h[7]

	WORD $0x4f392610 // srshr v16.4s, v16.4s, #7
	WORD $0x4f392631 // srshr v17.4s, v17.4s, #7
	WORD $0x0e614a10 // sqxtn  v16.4h, v16.4s
	WORD $0x4e614a30 // sqxtn2 v16.8h, v17.4s
	WORD $0x4c007550 // st1 {v16.8h}, [x10]

	ADD  $16, R10, R10
	ADD  $16, R11, R11
	SUB  $8, R8, R8
	CBNZ R8, c2vColLoop

	ADD  R6<<1, R1
	ADD  R14, R17, R17
	SUB  $1, R7, R7
	CBNZ R7, c2vRowLoop

c2vDone:
	RET

// func compound2D8NEONAsmW4(ctx *compound2D8NEONCtx)
TEXT ·compound2D8NEONAsmW4(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD D_DST(R0), R1
	MOVD D_REF(R0), R2
	MOVD D_KERNEL(R0), R3
	MOVD D_XKERN(R0), R12
	MOVD D_REFSTR(R0), R5
	MOVD D_HEIGHT(R0), R7
	MOVD D_IM(R0), R13
	MOVD D_IMSTR(R0), R14
	LSL  $1, R14, R14

	ADD  $7, R7, R6 // imH = height + filterTaps - 1

	WORD $0x4c407580 // ld1 {v0.8h}, [x12]  horizontal taps
	MOVD $16384, R11
	WORD $0x4e040d72 // dup v18.4s, w11

	MOVD R2, R16
	MOVD R13, R17

c2h4RowLoop:
	CBZ  R6, c2h4Done
	MOVD R16, R9
	ADD  $8, R9, R15
	WORD $0x4eb21e50 // mov v16.16b, v18.16b
	WORD $0x0c407121 // ld1 {v1.8b}, [x9]
	WORD $0x2f08a422 // ushll v2.8h, v1.8b, #0
	WORD $0x0d4081e5 // ld1 {v5.s}[0], [x15]
	WORD $0x2f08a4a3 // ushll v3.8h, v5.8b, #0
	WORD $0x0f402050 // smlal v16.4s, v2.4h, v0.h[0]
	WORD $0x6e031044 // ext v4.16b, v2.16b, v3.16b, #2
	WORD $0x0f502090 // smlal v16.4s, v4.4h, v0.h[1]
	WORD $0x6e032044 // ext v4.16b, v2.16b, v3.16b, #4
	WORD $0x0f602090 // smlal v16.4s, v4.4h, v0.h[2]
	WORD $0x6e033044 // ext v4.16b, v2.16b, v3.16b, #6
	WORD $0x0f702090 // smlal v16.4s, v4.4h, v0.h[3]
	WORD $0x6e034044 // ext v4.16b, v2.16b, v3.16b, #8
	WORD $0x0f402890 // smlal v16.4s, v4.4h, v0.h[4]
	WORD $0x6e035044 // ext v4.16b, v2.16b, v3.16b, #10
	WORD $0x0f502890 // smlal v16.4s, v4.4h, v0.h[5]
	WORD $0x6e036044 // ext v4.16b, v2.16b, v3.16b, #12
	WORD $0x0f602890 // smlal v16.4s, v4.4h, v0.h[6]
	WORD $0x6e037044 // ext v4.16b, v2.16b, v3.16b, #14
	WORD $0x0f702890 // smlal v16.4s, v4.4h, v0.h[7]

	WORD $0x4f3d2610 // srshr v16.4s, v16.4s, #3
	WORD $0x0e612a10 // xtn v16.4h, v16.4s
	WORD $0x0c007630 // st1 {v16.4h}, [x17]

	ADD  R5, R16, R16
	ADD  R14, R17, R17
	SUB  $1, R6, R6
	CBNZ R6, c2h4RowLoop

c2h4Done:
	WORD $0x4c407460 // ld1 {v0.8h}, [x3]  vertical taps
	MOVD $524288, R11
	WORD $0x4e040d72 // dup v18.4s, w11

	MOVD R13, R17

c2v4RowLoop:
	CBZ  R7, c2v4Done
	MOVD R17, R9
	MOVD R1, R10
	WORD $0x4eb21e50 // mov v16.16b, v18.16b

	WORD $0x0cce7521 // ld1 {v1.4h}, [x9], x14
	WORD $0x0f402030 // smlal v16.4s, v1.4h, v0.h[0]
	WORD $0x0cce7521 // ld1 {v1.4h}, [x9], x14
	WORD $0x0f502030 // smlal v16.4s, v1.4h, v0.h[1]
	WORD $0x0cce7521 // ld1 {v1.4h}, [x9], x14
	WORD $0x0f602030 // smlal v16.4s, v1.4h, v0.h[2]
	WORD $0x0cce7521 // ld1 {v1.4h}, [x9], x14
	WORD $0x0f702030 // smlal v16.4s, v1.4h, v0.h[3]
	WORD $0x0cce7521 // ld1 {v1.4h}, [x9], x14
	WORD $0x0f402830 // smlal v16.4s, v1.4h, v0.h[4]
	WORD $0x0cce7521 // ld1 {v1.4h}, [x9], x14
	WORD $0x0f502830 // smlal v16.4s, v1.4h, v0.h[5]
	WORD $0x0cce7521 // ld1 {v1.4h}, [x9], x14
	WORD $0x0f602830 // smlal v16.4s, v1.4h, v0.h[6]
	WORD $0x0cce7521 // ld1 {v1.4h}, [x9], x14
	WORD $0x0f702830 // smlal v16.4s, v1.4h, v0.h[7]

	WORD $0x4f392610 // srshr v16.4s, v16.4s, #7
	WORD $0x0e614a10 // sqxtn v16.4h, v16.4s
	WORD $0x0c007550 // st1 {v16.4h}, [x10]

	ADD  $8, R1, R1
	ADD  R14, R17, R17
	SUB  $1, R7, R7
	CBNZ R7, c2v4RowLoop

c2v4Done:
	RET
