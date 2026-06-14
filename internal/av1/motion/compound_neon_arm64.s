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

// func compoundX8NEONAsm(ctx *compoundX8NEONCtx)
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
