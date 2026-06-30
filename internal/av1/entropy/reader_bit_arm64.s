// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego && !goav1_trace_rng

#include "textflag.h"

#define CURSOR_SRC_PTR       0
#define CURSOR_SRC_LEN       8
#define CURSOR_POS          24
#define CURSOR_DIF          28
#define CURSOR_RNG          32
#define CURSOR_CNT          34
#define CURSOR_TELL_OFFS    36

// func readBitTrustedARM64(c *Cursor) uint8
//
// Source-shaped arm64 range-decoder bit path, mapped to dav1d's
// msac_decode_bool_equi_neon scalar state update while preserving goav1's
// exact refillState semantics for tile tails.
TEXT ·readBitTrustedARM64(SB), NOSPLIT, $0-9
	MOVD c+0(FP), R0
	MOVWU CURSOR_DIF(R0), R7
	MOVHU CURSOR_RNG(R0), R8

	LSR   $8, R8, R9
	LSL   $7, R9, R10
	ADD   $4, R10, R10
	LSL   $16, R10, R11
	MOVD  $1, R15
	CMPW  R11, R7
	BCC   bitNoSub
	SUB   R11, R7, R7
	SUB   R10, R8, R8
	MOVD  $0, R15
	B     bitRenorm

bitNoSub:
	MOVD R10, R8

bitRenorm:
	MOVH CURSOR_CNT(R0), R5
	ADD  $1, R7, R7
	UBFX $0, R8, $16, R12
	CLZ  R12, R12
	SUB  $48, R12, R12
	LSL  R12, R7, R7
	SUB  $1, R7, R7
	LSL  R12, R8, R8
	SUB  R12, R5, R5
	TBZ  $63, R5, bitStore

	MOVD  CURSOR_SRC_PTR(R0), R2
	MOVD  CURSOR_SRC_LEN(R0), R3
	MOVWU CURSOR_POS(R0), R6
	MOVH  CURSOR_TELL_OFFS(R0), R10
	MOVD  $8, R11
	SUB   R5, R11, R11 // shift = 8 - cnt
	SUB   R6, R3, R12 // remaining = len(src) - pos

	CMP $8, R11
	BLT refillThree
	CMP $16, R11
	BGE refillThree
	CMP $2, R12
	BLT refillSlow
	ADD   R2, R6, R13
	MOVBU (R13), R14
	LSL   R11, R14, R14
	EOR   R14, R7, R7
	MOVBU 1(R13), R14
	SUB   $8, R11, R13
	LSL   R13, R14, R14
	EOR   R14, R7, R7
	ADD   $2, R6, R6
	ADD   $16, R5, R5
	B     refillTailCheck

refillThree:
	CMP $24, R11
	BGE refillSlow
	CMP $3, R12
	BLT refillSlow
	ADD   R2, R6, R13
	MOVBU (R13), R14
	LSL   R11, R14, R14
	EOR   R14, R7, R7
	MOVBU 1(R13), R14
	SUB   $8, R11, R4
	LSL   R4, R14, R14
	EOR   R14, R7, R7
	MOVBU 2(R13), R14
	SUB   $16, R11, R4
	LSL   R4, R14, R14
	EOR   R14, R7, R7
	ADD   $3, R6, R6
	ADD   $24, R5, R5
	B     refillTailCheck

refillSlow:
	CMP $0, R11
	BLT refillTailCheck
	CMP R3, R6
	BGE refillTailCheck
	ADD   R2, R6, R13
	MOVBU (R13), R14
	LSL   R11, R14, R14
	EOR   R14, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     refillSlow

refillTailCheck:
	CMP R3, R6
	BLT refillStore
	MOVD $0x4000, R12
	SUB  R5, R12, R13
	ADD  R13, R10, R10
	MOVD R12, R5

refillStore:
	MOVW R6, CURSOR_POS(R0)
	MOVH R10, CURSOR_TELL_OFFS(R0)

bitStore:
	MOVW R7, CURSOR_DIF(R0)
	MOVH R8, CURSOR_RNG(R0)
	MOVH R5, CURSOR_CNT(R0)
	MOVB R15, ret+8(FP)
	RET
