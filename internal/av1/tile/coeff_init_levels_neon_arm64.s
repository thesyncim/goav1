// NEON TXB coefficient level-init helpers.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

#include "textflag.h"

// These kernels are source-shaped slices of SVT-AV1's
// svt_av1_txb_init_levels_neon. goav1 stores TXB levels column-major for the
// coefficient coder, so each loop processes one coefficient column:
//
//     levels[col*(height+TX_PAD_HOR)+row] = min(abs(coeff[col*height+row]), 127)
//
// coeffInitLevelsArch clears the full scratch window first, matching libaom's
// av1_txb_init_levels_c padding semantics. The assembly only fills live rows.

// func coeffInitLevels4NEONAsm(coeffs *int16, levels *uint8, columns uintptr)
TEXT ·coeffInitLevels4NEONAsm(SB), NOSPLIT, $0-24
	MOVD coeffs+0(FP), R0
	MOVD levels+8(FP), R1
	MOVD columns+16(FP), R2

coeffInitLevels4Loop:
	WORD $0x0cdf7400 // ld1 {v0.4h}, [x0], #8
	WORD $0x4e607800 // sqabs v0.8h, v0.8h
	WORD $0x0e214801 // sqxtn v1.8b, v0.8h
	WORD $0x0d008021 // st1 {v1.s}[0], [x1]
	ADD  $8, R1, R1
	SUB  $1, R2, R2
	CBNZ R2, coeffInitLevels4Loop
	RET

// func coeffInitLevels8NEONAsm(coeffs *int16, levels *uint8, columns uintptr)
TEXT ·coeffInitLevels8NEONAsm(SB), NOSPLIT, $0-24
	MOVD coeffs+0(FP), R0
	MOVD levels+8(FP), R1
	MOVD columns+16(FP), R2

coeffInitLevels8Loop:
	WORD $0x4cdf7400 // ld1 {v0.8h}, [x0], #16
	WORD $0x4e607800 // sqabs v0.8h, v0.8h
	WORD $0x0e214801 // sqxtn v1.8b, v0.8h
	WORD $0x0c007021 // st1 {v1.8b}, [x1]
	ADD  $12, R1, R1
	SUB  $1, R2, R2
	CBNZ R2, coeffInitLevels8Loop
	RET

// func coeffInitLevels16NEONAsm(coeffs *int16, levels *uint8, columns uintptr)
TEXT ·coeffInitLevels16NEONAsm(SB), NOSPLIT, $0-24
	MOVD coeffs+0(FP), R0
	MOVD levels+8(FP), R1
	MOVD columns+16(FP), R2

coeffInitLevels16Loop:
	WORD $0x4cdf7400 // ld1 {v0.8h}, [x0], #16
	WORD $0x4cdf7401 // ld1 {v1.8h}, [x0], #16
	WORD $0x4e607800 // sqabs v0.8h, v0.8h
	WORD $0x4e607821 // sqabs v1.8h, v1.8h
	WORD $0x0e214804 // sqxtn v4.8b, v0.8h
	WORD $0x0e214825 // sqxtn v5.8b, v1.8h
	WORD $0x0c007024 // st1 {v4.8b}, [x1]
	ADD  $8, R1, R3
	WORD $0x0c007065 // st1 {v5.8b}, [x3]
	ADD  $20, R1, R1
	SUB  $1, R2, R2
	CBNZ R2, coeffInitLevels16Loop
	RET

// func coeffInitLevels32NEONAsm(coeffs *int16, levels *uint8, columns uintptr)
TEXT ·coeffInitLevels32NEONAsm(SB), NOSPLIT, $0-24
	MOVD coeffs+0(FP), R0
	MOVD levels+8(FP), R1
	MOVD columns+16(FP), R2

coeffInitLevels32Loop:
	WORD $0x4cdf7400 // ld1 {v0.8h}, [x0], #16
	WORD $0x4cdf7401 // ld1 {v1.8h}, [x0], #16
	WORD $0x4cdf7402 // ld1 {v2.8h}, [x0], #16
	WORD $0x4cdf7403 // ld1 {v3.8h}, [x0], #16
	WORD $0x4e607800 // sqabs v0.8h, v0.8h
	WORD $0x4e607821 // sqabs v1.8h, v1.8h
	WORD $0x4e607842 // sqabs v2.8h, v2.8h
	WORD $0x4e607863 // sqabs v3.8h, v3.8h
	WORD $0x0e214804 // sqxtn v4.8b, v0.8h
	WORD $0x0e214825 // sqxtn v5.8b, v1.8h
	WORD $0x0e214846 // sqxtn v6.8b, v2.8h
	WORD $0x0e214867 // sqxtn v7.8b, v3.8h
	WORD $0x0c007024 // st1 {v4.8b}, [x1]
	ADD  $8, R1, R3
	WORD $0x0c007065 // st1 {v5.8b}, [x3]
	ADD  $16, R1, R3
	WORD $0x0c007066 // st1 {v6.8b}, [x3]
	ADD  $24, R1, R3
	WORD $0x0c007067 // st1 {v7.8b}, [x3]
	ADD  $36, R1, R1
	SUB  $1, R2, R2
	CBNZ R2, coeffInitLevels32Loop
	RET
