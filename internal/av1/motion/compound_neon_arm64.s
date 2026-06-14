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

#define X_DST         0
#define X_REF         8
#define X_KERNEL      16
#define X_REFSTR      24
#define X_WIDTH       32
#define X_HEIGHT      40
#define X_ROUNDOFFSET 48

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
