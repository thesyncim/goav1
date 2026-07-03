// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 realtime 8x8 block average: (Σbytes + 32) >> 6. PSADBW against a zero
// register sums 8 bytes per lane exactly; the quad variant reads 16-byte rows so
// one PSADBW yields the left (cols 0-7) and right (cols 8-15) quadrant sums per
// row. Exact integer arithmetic, so bit-exact with realtimeAvg8x8PureGo.

// realtimeAvgAVX2Ctx field offsets.
#define AVSRC    0
#define AVSTRIDE 8
#define AVOUT0   16
#define AVOUT1   24
#define AVOUT2   32
#define AVOUT3   40

// func realtimeAvg8x8AVX2Asm(ctx *realtimeAvgAVX2Ctx)
TEXT ·realtimeAvg8x8AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ AVSRC(AX), SI
	MOVQ AVSTRIDE(AX), DX

	PXOR X3, X3
	PXOR X2, X2

	MOVQ $8, CX
loopAvg8:
	MOVQ   (SI), X0
	PSADBW X3, X0
	PADDQ  X0, X2
	ADDQ DX, SI
	DECQ CX
	JNZ  loopAvg8

	MOVQ X2, BX
	ADDQ $32, BX
	SHRQ $6, BX
	MOVQ BX, AVOUT0(AX)
	RET

// func realtimeAvg8x8QuadAVX2Asm(ctx *realtimeAvgAVX2Ctx)
// Four 8x8 averages over a 16x16 region: X2 accumulates the top two quadrant
// sums (rows 0-7), X4 the bottom two (rows 8-15).
TEXT ·realtimeAvg8x8QuadAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ AVSRC(AX), SI
	MOVQ AVSTRIDE(AX), DX

	PXOR X3, X3
	PXOR X2, X2
	PXOR X4, X4

	MOVQ $8, CX
loopTop:
	MOVOU  (SI), X0
	PSADBW X3, X0
	PADDQ  X0, X2
	ADDQ DX, SI
	DECQ CX
	JNZ  loopTop

	MOVQ $8, CX
loopBot:
	MOVOU  (SI), X0
	PSADBW X3, X0
	PADDQ  X0, X4
	ADDQ DX, SI
	DECQ CX
	JNZ  loopBot

	// Out0 = top-left (X2 low), Out1 = top-right (X2 high).
	MOVQ X2, BX
	ADDQ $32, BX
	SHRQ $6, BX
	MOVQ BX, AVOUT0(AX)
	PEXTRQ $1, X2, BX
	ADDQ $32, BX
	SHRQ $6, BX
	MOVQ BX, AVOUT1(AX)

	// Out2 = bottom-left (X4 low), Out3 = bottom-right (X4 high).
	MOVQ X4, BX
	ADDQ $32, BX
	SHRQ $6, BX
	MOVQ BX, AVOUT2(AX)
	PEXTRQ $1, X4, BX
	ADDQ $32, BX
	SHRQ $6, BX
	MOVQ BX, AVOUT3(AX)
	RET
