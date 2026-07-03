// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 prediction residual extraction: dst[r*w+c] = int16(src) - int16(pred).
// Bytes are zero-extended to int16 lanes and subtracted (|diff| ≤ 255 fits
// int16), then stored packed at stride w. Width is a multiple of 8; each row is
// consumed in 16-column YMM chunks with an optional trailing 8-column XMM chunk.
// Exact integer arithmetic, so bit-exact with residualBlockPureGo.

// residualAVX2Ctx field offsets.
#define RSRC        0
#define RSRCSTRIDE  8
#define RPRED       16
#define RPREDSTRIDE 24
#define RDST        32
#define RW          40
#define RH          48

// func residualBlockAVX2Asm(ctx *residualAVX2Ctx)
TEXT ·residualBlockAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ RSRC(AX), SI
	MOVQ RSRCSTRIDE(AX), DX
	MOVQ RPRED(AX), DI
	MOVQ RPREDSTRIDE(AX), R8
	MOVQ RDST(AX), BX
	MOVQ RW(AX), R10
	MOVQ RH(AX), R9

rowRes:
	MOVQ SI, R11
	MOVQ DI, R12
	MOVQ BX, R13
	MOVQ R10, R14

col16Res:
	CMPQ R14, $16
	JL   col8Res
	VPMOVZXBW (R11), Y0
	VPMOVZXBW (R12), Y1
	VPSUBW    Y1, Y0, Y0
	VMOVDQU   Y0, (R13)
	ADDQ $16, R11
	ADDQ $16, R12
	ADDQ $32, R13
	SUBQ $16, R14
	JMP  col16Res

col8Res:
	CMPQ R14, $8
	JL   rowResNext
	VPMOVZXBW (R11), X0
	VPMOVZXBW (R12), X1
	VPSUBW    X1, X0, X0
	VMOVDQU   X0, (R13)
	ADDQ $8, R11
	ADDQ $8, R12
	ADDQ $16, R13

rowResNext:
	ADDQ DX, SI
	ADDQ R8, DI
	// dst advances by W int16 lanes = 2*W bytes.
	MOVQ R10, R15
	SHLQ $1, R15
	ADDQ R15, BX
	DECQ R9
	JNZ  rowRes

	VZEROUPPER
	RET
