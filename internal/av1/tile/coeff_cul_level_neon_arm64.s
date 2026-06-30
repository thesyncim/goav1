// NEON TXB cumulative-level helper.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

#include "textflag.h"

// func coeffCulLevelNEONAsm(coeffs *int16, scan *int16, eob uintptr) uint8
//
// Source-shaped slice of SVT-AV1's svt_av1_compute_cul_level_neon, adapted to
// goav1's int16 quantized coefficient storage. The scan-order gather remains
// scalar like SVT; the absolute-value accumulation is kept in NEON lanes.
TEXT ·coeffCulLevelNEONAsm(SB), NOSPLIT, $0-25
	MOVD coeffs+0(FP), R0
	MOVD scan+8(FP), R1
	MOVD eob+16(FP), R2

	MOVD $0, R13
	WORD $0x4f000400 // movi.4s v0, #0

	CMP  $4, R2
	BLT  culTail

culLoop4:
	MOVH 0(R1), R5
	MOVH 2(R1), R6
	MOVH 4(R1), R7
	MOVH 6(R1), R8

	LSL  $1, R5, R5
	ADD  R0, R5, R5
	MOVH (R5), R5
	LSL  $1, R6, R6
	ADD  R0, R6, R6
	MOVH (R6), R6
	LSL  $1, R7, R7
	ADD  R0, R7, R7
	MOVH (R7), R7
	LSL  $1, R8, R8
	ADD  R0, R8, R8
	MOVH (R8), R8

	WORD $0x4e041ca1 // mov.s v1[0], w5
	WORD $0x4e0c1cc1 // mov.s v1[1], w6
	WORD $0x4e141ce1 // mov.s v1[2], w7
	WORD $0x4e1c1d01 // mov.s v1[3], w8
	WORD $0x4ea0b821 // abs.4s v1, v1
	WORD $0x4ea18400 // add.4s v0, v0, v1

	ADD  $8, R1, R1
	SUB  $4, R2, R2
	CMP  $4, R2
	BGE  culLoop4

culTail:
	CBZ R2, culReduce

culTailLoop:
	MOVH (R1), R5
	ADD  $2, R1, R1
	LSL  $1, R5, R5
	ADD  R0, R5, R5
	MOVH (R5), R5
	CMP  $0, R5
	BGE  culTailAbsDone
	NEG  R5, R5
culTailAbsDone:
	ADD  R5, R13, R13
	SUB  $1, R2, R2
	CBNZ R2, culTailLoop

culReduce:
	WORD $0x4eb1b800 // addv.4s s0, v0
	WORD $0x1e26000e // fmov w14, s0
	ADD  R13, R14, R14
	CMP  $7, R14
	BLE  culSign
	MOVD $7, R14

culSign:
	MOVH (R0), R5
	CMP  $0, R5
	BLT  culNegativeDC
	BGT  culPositiveDC
	MOVB R14, ret+24(FP)
	RET

culNegativeDC:
	ORR  $(1<<3), R14, R14
	MOVB R14, ret+24(FP)
	RET

culPositiveDC:
	ADD  $(2<<3), R14, R14
	MOVB R14, ret+24(FP)
	RET
