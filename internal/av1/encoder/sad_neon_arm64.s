// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// NEON 8x8 sum of absolute differences. Eight rows of 8 bytes are loaded from
// src and ref, accumulated with the widening absolute-difference instructions
// (UABDL for row 0, UABAL for rows 1-7, emitted via WORD because the Go
// assembler lacks the mnemonics), then reduced with UADDLV.
//
//   uabdl v2.8h, v0.8b, v1.8b   -> 0x2e217002
//   uabal v2.8h, v0.8b, v1.8b   -> 0x2e215002
//   uaddlv s3, v2.8h            -> 0x6e703843

#define SRC    0
#define REF    8
#define STRIDE 16
#define SUM    24

// func sad8x8NEONAsm(ctx *sad8x8NEONCtx)
TEXT ·sad8x8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD SRC(R0), R1
	MOVD REF(R0), R2
	MOVD STRIDE(R0), R3

	// Row 0: initialize the uint16 accumulator lanes.
	VLD1 (R1), [V0.B8]
	VLD1 (R2), [V1.B8]
	WORD $0x2e217002 // uabdl v2.8h, v0.8b, v1.8b
	ADD  R3, R1
	ADD  R3, R2

	// Rows 1-7: accumulate.
	MOVD $7, R4
loop:
	VLD1 (R1), [V0.B8]
	VLD1 (R2), [V1.B8]
	WORD $0x2e215002 // uabal v2.8h, v0.8b, v1.8b
	ADD  R3, R1
	ADD  R3, R2
	SUB  $1, R4
	CBNZ R4, loop

	WORD $0x6e703843 // uaddlv s3, v2.8h
	VMOV V3.S[0], R5
	MOVD R5, SUM(R0)
	RET
