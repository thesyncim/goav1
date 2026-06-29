// NEON TXB coefficient-prep helpers.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

#include "textflag.h"

// func txbInitLevels8x8NEONAsm(coeffs *int16, levels *uint8)
//
// Source-shaped slice of SVT-AV1's svt_av1_txb_init_levels_neon for an 8x8
// TXB. goav1's level scratch is column-major with four horizontal pad slots:
//     levels[col*(8+TX_PAD_HOR)+row] = min(abs(coeff[col*8+row]), 127)
// The caller supplies zeroed scratch; this kernel also clears the four pad
// bytes for each written column so direct tests do not depend on stack zeroing.
TEXT ·txbInitLevels8x8NEONAsm(SB), NOSPLIT, $0-16
	MOVD coeffs+0(FP), R0
	MOVD levels+8(FP), R1
	MOVD $8, R2

txb8x8InitLoop:
	WORD $0x4cdf7400 // ld1 {v0.8h}, [x0], #16
	WORD $0x4e607800 // sqabs v0.8h, v0.8h
	WORD $0x0e214801 // sqxtn v1.8b, v0.8h
	WORD $0x0c007021 // st1 {v1.8b}, [x1]
	WORD $0xb900083f // str wzr, [x1, #8]
	ADD  $12, R1, R1
	SUB  $1, R2, R2
	CBNZ R2, txb8x8InitLoop
	RET
