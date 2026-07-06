// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 final self-guided projection for 8-bit pixels (see
// selfguided_u8_avx2_amd64.go).

// sgrWeightedU8AVX2Ctx field offsets (see selfguided_u8_avx2_amd64.go).
#define W_DST  0
#define W_SRC  8
#define W_F0   16
#define W_F1   24
#define W_COLS 32
#define W_XQ0  40
#define W_XQ1  44
#define W_BIAS 48
#define W_MAXV 52

// func sgrWeightedRowU8AVX2Asm(ctx *sgrWeightedU8AVX2Ctx)
//
// Projects one output row, 8 columns per iteration.
TEXT ·sgrWeightedRowU8AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ W_DST(AX), DI        // dst cursor
	MOVQ W_SRC(AX), SI        // src cursor
	MOVQ W_F0(AX), R8         // flt0 cursor
	MOVQ W_F1(AX), R9         // flt1 cursor
	MOVQ W_COLS(AX), R10      // remaining columns (multiple of 8)

	LEAQ W_XQ0(AX), R11
	VPBROADCASTD (R11), Y10   // xq0 broadcast
	LEAQ W_XQ1(AX), R11
	VPBROADCASTD (R11), Y11   // xq1 broadcast
	LEAQ W_BIAS(AX), R11
	VPBROADCASTD (R11), Y12   // rounding bias 1<<10
	LEAQ W_MAXV(AX), R11
	VPBROADCASTD (R11), Y14   // upper clamp 255
	VPXOR Y13, Y13, Y13       // lower clamp bound 0

wColLoop:
	VPMOVZXBD (SI), Y1        // 8 source u8 -> s32 lanes (s)
	VPSLLD $4, Y1, Y2         // u = s << 4
	VPSLLD $11, Y1, Y3        // v = u << 7 = s << 11
	VMOVDQU (R8), Y4          // f0
	VPSUBD Y2, Y4, Y4         // f0 - u
	VPMULLD Y10, Y4, Y4       // * xq0
	VPADDD Y4, Y3, Y3         // v += xq0*(f0-u)
	VMOVDQU (R9), Y5          // f1
	VPSUBD Y2, Y5, Y5         // f1 - u
	VPMULLD Y11, Y5, Y5       // * xq1
	VPADDD Y5, Y3, Y3         // v += xq1*(f1-u)
	VPADDD Y12, Y3, Y3        // + 1<<10 (wraps like Go int32)
	VPSRAD $11, Y3, Y3        // arith >> 11
	VPSLLD $16, Y3, Y3        // int16 wrap:
	VPSRAD $16, Y3, Y3        //   sign-extend the low 16 bits
	VPMAXSD Y13, Y3, Y3       // clamp lo 0
	VPMINSD Y14, Y3, Y3       // clamp hi 255
	VPACKUSDW Y3, Y3, Y3      // s32 -> u16 (in range)
	VPERMQ $0xd8, Y3, Y3      // gather the 8 low words contiguously
	VPACKUSWB X3, X3, X3      // u16 -> u8 (in range)
	MOVQ X3, (DI)             // store 8 u8

	ADDQ $8, DI               // dst  += 8 u8
	ADDQ $8, SI               // src  += 8 u8
	ADDQ $32, R8              // flt0 += 8 s32
	ADDQ $32, R9              // flt1 += 8 s32
	SUBQ $8, R10
	JNE  wColLoop

	VZEROUPPER
	RET
