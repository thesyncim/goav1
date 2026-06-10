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

// SSE2 16x16 sum of absolute differences: each iteration loads one 16-byte
// row pair with MOVOU and accumulates PSADBW's two 64-bit partial sums; the
// final reduction folds the high quadword into the low one.

// func sad16x16SSE2Asm(ctx *sad8x8SSE2Ctx)
TEXT ·sad16x16SSE2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ SRC(AX), SI
	MOVQ REF(AX), DI
	MOVQ STRIDE(AX), DX

	PXOR X2, X2

	MOVQ $16, CX
loop16:
	MOVOU (SI), X0
	MOVOU (DI), X1
	PSADBW X1, X0
	PADDQ X0, X2
	ADDQ DX, SI
	ADDQ DX, DI
	DECQ CX
	JNZ  loop16

	PSHUFD $0xEE, X2, X3
	PADDQ X3, X2
	MOVQ X2, BX
	MOVQ BX, SUM(AX)
	RET

#define DSRC       0
#define DREF       8
#define DSRCSTRIDE 16
#define DREFSTRIDE 24
#define DSUM       32

// func sad8x8DualSSE2Asm(ctx *sad8x8DualSSE2Ctx)
TEXT ·sad8x8DualSSE2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ DSRC(AX), SI
	MOVQ DREF(AX), DI
	MOVQ DSRCSTRIDE(AX), DX
	MOVQ DREFSTRIDE(AX), R8

	PXOR X2, X2

	MOVQ $8, CX
dloop:
	MOVQ (SI), X0
	MOVQ (DI), X1
	PSADBW X1, X0
	PADDQ X0, X2
	ADDQ DX, SI
	ADDQ R8, DI
	DECQ CX
	JNZ  dloop

	MOVQ X2, BX
	MOVQ BX, DSUM(AX)
	RET
