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
