// NEON TXB NZ-map context helpers.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

#include "textflag.h"

// These kernels are source-shaped slices of SVT-AV1's
// svt_av1_get_nz_map_contexts_neon:
//
//   count = min((sum(min(level[neighbour], 3)) + 1) >> 1, 4) + pos_offset
//
// goav1 stores TXB levels column-major, so each vector covers 16 rows from one
// coefficient column instead of SVT's row-major 16-wide stripe.

// func coeffNZMapContexts16Rows2DNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)
TEXT ·coeffNZMapContexts16Rows2DNEONAsm(SB), NOSPLIT, $0-40
	MOVD levels+0(FP), R0
	MOVD offsets+8(FP), R1
	MOVD contexts+16(FP), R2
	MOVD columns+24(FP), R3
	MOVD stride+32(FP), R4

	WORD $0x4f00e47f // movi v31.16b, #3
	WORD $0x4f00e49e // movi v30.16b, #4

coeffNZMap16Loop:
	ADD  $1, R0, R5
	ADD  R4, R0, R6
	ADD  $1, R6, R7
	ADD  R4, R6, R8
	ADD  $2, R0, R9
	WORD $0x3dc000a0 // ldr q0, [x5]  level + 1
	WORD $0x3dc000c1 // ldr q1, [x6]  level + stride
	WORD $0x3dc000e2 // ldr q2, [x7]  level + stride + 1
	WORD $0x3dc00103 // ldr q3, [x8]  level + 2*stride
	WORD $0x3dc00124 // ldr q4, [x9]  level + 2
	WORD $0x3dc00025 // ldr q5, [x1]  positional offset
	WORD $0x6e3f6c00 // umin v0.16b, v0.16b, v31.16b
	WORD $0x6e3f6c21 // umin v1.16b, v1.16b, v31.16b
	WORD $0x6e3f6c42 // umin v2.16b, v2.16b, v31.16b
	WORD $0x6e3f6c63 // umin v3.16b, v3.16b, v31.16b
	WORD $0x6e3f6c84 // umin v4.16b, v4.16b, v31.16b
	WORD $0x4e218400 // add v0.16b, v0.16b, v1.16b
	WORD $0x4e228400 // add v0.16b, v0.16b, v2.16b
	WORD $0x4e238400 // add v0.16b, v0.16b, v3.16b
	WORD $0x4e248400 // add v0.16b, v0.16b, v4.16b
	WORD $0x6f0f2400 // urshr v0.16b, v0.16b, #1
	WORD $0x6e3e6c00 // umin v0.16b, v0.16b, v30.16b
	WORD $0x4e258400 // add v0.16b, v0.16b, v5.16b
	WORD $0x3d800040 // str q0, [x2]
	ADD  R4, R0, R0
	ADD  $16, R1, R1
	ADD  $16, R2, R2
	SUB  $1, R3, R3
	CBNZ R3, coeffNZMap16Loop
	RET

// func coeffNZMapContexts32Rows2DNEONAsm(levels *uint8, offsets *uint8, contexts *int8, columns uintptr, stride uintptr)
TEXT ·coeffNZMapContexts32Rows2DNEONAsm(SB), NOSPLIT, $0-40
	MOVD levels+0(FP), R0
	MOVD offsets+8(FP), R1
	MOVD contexts+16(FP), R2
	MOVD columns+24(FP), R3
	MOVD stride+32(FP), R4

	WORD $0x4f00e47f // movi v31.16b, #3
	WORD $0x4f00e49e // movi v30.16b, #4

coeffNZMap32Loop:
	ADD  $1, R0, R5
	ADD  R4, R0, R6
	ADD  $1, R6, R7
	ADD  R4, R6, R8
	ADD  $2, R0, R9
	WORD $0x3dc000a0 // ldr q0, [x5]
	WORD $0x3dc000c1 // ldr q1, [x6]
	WORD $0x3dc000e2 // ldr q2, [x7]
	WORD $0x3dc00103 // ldr q3, [x8]
	WORD $0x3dc00124 // ldr q4, [x9]
	WORD $0x3dc00025 // ldr q5, [x1]
	WORD $0x6e3f6c00 // umin v0.16b, v0.16b, v31.16b
	WORD $0x6e3f6c21 // umin v1.16b, v1.16b, v31.16b
	WORD $0x6e3f6c42 // umin v2.16b, v2.16b, v31.16b
	WORD $0x6e3f6c63 // umin v3.16b, v3.16b, v31.16b
	WORD $0x6e3f6c84 // umin v4.16b, v4.16b, v31.16b
	WORD $0x4e218400 // add v0.16b, v0.16b, v1.16b
	WORD $0x4e228400 // add v0.16b, v0.16b, v2.16b
	WORD $0x4e238400 // add v0.16b, v0.16b, v3.16b
	WORD $0x4e248400 // add v0.16b, v0.16b, v4.16b
	WORD $0x6f0f2400 // urshr v0.16b, v0.16b, #1
	WORD $0x6e3e6c00 // umin v0.16b, v0.16b, v30.16b
	WORD $0x4e258400 // add v0.16b, v0.16b, v5.16b
	WORD $0x3d800040 // str q0, [x2]

	ADD  $16, R0, R10
	ADD  $16, R1, R11
	ADD  $16, R2, R12
	ADD  $1, R10, R5
	ADD  R4, R10, R6
	ADD  $1, R6, R7
	ADD  R4, R6, R8
	ADD  $2, R10, R9
	WORD $0x3dc000a0 // ldr q0, [x5]
	WORD $0x3dc000c1 // ldr q1, [x6]
	WORD $0x3dc000e2 // ldr q2, [x7]
	WORD $0x3dc00103 // ldr q3, [x8]
	WORD $0x3dc00124 // ldr q4, [x9]
	WORD $0x3dc00165 // ldr q5, [x11]
	WORD $0x6e3f6c00 // umin v0.16b, v0.16b, v31.16b
	WORD $0x6e3f6c21 // umin v1.16b, v1.16b, v31.16b
	WORD $0x6e3f6c42 // umin v2.16b, v2.16b, v31.16b
	WORD $0x6e3f6c63 // umin v3.16b, v3.16b, v31.16b
	WORD $0x6e3f6c84 // umin v4.16b, v4.16b, v31.16b
	WORD $0x4e218400 // add v0.16b, v0.16b, v1.16b
	WORD $0x4e228400 // add v0.16b, v0.16b, v2.16b
	WORD $0x4e238400 // add v0.16b, v0.16b, v3.16b
	WORD $0x4e248400 // add v0.16b, v0.16b, v4.16b
	WORD $0x6f0f2400 // urshr v0.16b, v0.16b, #1
	WORD $0x6e3e6c00 // umin v0.16b, v0.16b, v30.16b
	WORD $0x4e258400 // add v0.16b, v0.16b, v5.16b
	WORD $0x3d800180 // str q0, [x12]

	ADD  R4, R0, R0
	ADD  $32, R1, R1
	ADD  $32, R2, R2
	SUB  $1, R3, R3
	CBNZ R3, coeffNZMap32Loop
	RET
