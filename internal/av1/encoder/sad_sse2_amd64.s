// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// SSE2 8x8 sum of absolute differences. Each iteration loads one 8-byte row
// from src and ref into the low halves of X0/X1 and accumulates PSADBW's
// 16-bit row SAD into X2; the result is the low quadword of the accumulator.

#define SRC    0
#define REF    8
#define STRIDE 16
#define SUM    24

// func sad8x8SSE2Asm(ctx *sad8x8SSE2Ctx)
TEXT ·sad8x8SSE2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ SRC(AX), SI
	MOVQ REF(AX), DI
	MOVQ STRIDE(AX), DX

	PXOR X2, X2

	MOVQ $8, CX
loop:
	MOVQ (SI), X0
	MOVQ (DI), X1
	PSADBW X1, X0
	PADDQ X0, X2
	ADDQ DX, SI
	ADDQ DX, DI
	DECQ CX
	JNZ  loop

	MOVQ X2, BX
	MOVQ BX, SUM(AX)
	RET
