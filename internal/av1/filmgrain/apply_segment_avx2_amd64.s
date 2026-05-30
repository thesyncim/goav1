// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// AVX2 film-grain apply kernel. The kernel is bit-exact with
// applyGrainSegmentPureGo (see apply_segment.go); the Go wrapper in
// apply_segment_avx2_amd64.go handles the (<8)-lane tail.
//
// Per eight-sample group:
//   scale (u16) -> u32, grain (i16) -> i32, src (u16) -> u32
//   prod  = scale * grain                 (VPMULLD, 32-bit lanes)
//   noise = (prod + roundBias) >> shift    (VPADDD + arithmetic VPSRAD)
//   out   = clip(src + noise, min, max)    (VPADDD + VPMAXSD + VPMINSD)
//   store out narrowed back to u16         (VPACKSSDW + VPERMQ)
// The clamped result lies in [min,max] within [0,4095], so the signed
// VPACKSSDW narrow never saturates.

//go:build amd64 && !purego

#include "textflag.h"

#define DST 0
#define SRC 8
#define SCALE 16
#define GRAIN 24
#define GROUPS 32
#define ROUNDBIAS 40
#define SHIFT 44
#define MINVALUE 48
#define MAXVALUE 52

// func applyGrainSegmentAVX2Asm(ctx *applyGrainSegmentAVX2Ctx)
TEXT ·applyGrainSegmentAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R8
	MOVQ DST(R8), DI
	MOVQ SRC(R8), SI
	MOVQ SCALE(R8), R9
	MOVQ GRAIN(R8), R10
	MOVQ GROUPS(R8), CX

	// Broadcast the scalar lane-parameters into YMM registers.
	VPBROADCASTD ROUNDBIAS(R8), Y3 // roundBias in all 8 dwords
	VPBROADCASTD MINVALUE(R8), Y4  // minValue
	VPBROADCASTD MAXVALUE(R8), Y5  // maxValue
	// Arithmetic shift count: VPSRAD takes an XMM count operand.
	MOVL SHIFT(R8), AX
	MOVQ AX, X6

loop:
	TESTQ CX, CX
	JZ    done

	VPMOVZXWD (R9), Y0  // scale -> u32
	VPMOVSXWD (R10), Y1 // grain -> i32
	VPMOVZXWD (SI), Y2  // src   -> u32

	VPMULLD Y1, Y0, Y0  // prod = scale * grain
	VPADDD  Y3, Y0, Y0  // + roundBias
	VPSRAD  X6, Y0, Y0  // >> shift (arithmetic)
	VPADDD  Y2, Y0, Y0  // + src
	VPMAXSD Y4, Y0, Y0  // clip low
	VPMINSD Y5, Y0, Y0  // clip high

	VPACKSSDW Y0, Y0, Y0   // i32 -> i16 (lane-wise, duplicated halves)
	VPERMQ    $0x08, Y0, Y0 // gather the two useful 64-bit halves into X0
	VMOVDQU   X0, (DI)      // store 8 u16

	ADDQ $16, DI
	ADDQ $16, SI
	ADDQ $16, R9
	ADDQ $16, R10
	DECQ CX
	JMP  loop

done:
	VZEROUPPER
	RET
