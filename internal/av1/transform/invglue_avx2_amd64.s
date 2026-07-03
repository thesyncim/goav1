// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 inverse-transform glue kernels, bit-exact with clampRoundPureGo and
// narrowStorePureGo (see hybrid.go) and with the NEON clampRound/narrowStore
// kernels. clampRound performs a signed rounding shift right by `shift` (a
// pre-shift add of round=1<<(shift-1) then an arithmetic VPSRAD, matching
// srshl and roundShift) and clamps to [min, max]; narrowStore rounding-shifts
// every lane right by four and saturates to int16 (VPACKSSDW == sqxtn ==
// clipInt16). The 32-bit adds mirror the NEON kernels exactly; the stage-range
// values a real transform produces are far below the int32 overflow boundary,
// so the result equals the pure-Go int64 round for every real input.

// func clampRoundAVX2Asm(ptr *int32, n uintptr, shift uint64, round uint32, min uint32, max uint32)
//
// Rounds-and-shifts every int32 lane right by `shift` (zero passes through) and
// clamps to [min, max] in place; n is a multiple of eight. `shift` is the
// scalar VPSRAD count; `round` is 1<<(shift-1) for shift>0 and 0 otherwise.
TEXT ·clampRoundAVX2Asm(SB), NOSPLIT, $0-36
	MOVQ ptr+0(FP), DI
	MOVQ n+8(FP), CX
	MOVQ shift+16(FP), R9
	MOVL round+24(FP), R8
	MOVL min+28(FP), R10
	MOVL max+32(FP), R11

	MOVL         R8, X1
	VPBROADCASTD X1, Y1 // round add
	MOVQ         R9, X2 // shift count (scalar, low 64 bits)
	MOVL         R10, X3
	VPBROADCASTD X3, Y3 // min
	MOVL         R11, X4
	VPBROADCASTD X4, Y4 // max

clampLoop:
	TESTQ   CX, CX
	JZ      clampDone
	VMOVDQU (DI), Y0
	VPADDD  Y1, Y0, Y0
	VPSRAD  X2, Y0, Y0 // arithmetic shift right by scalar count
	VPMAXSD Y3, Y0, Y0
	VPMINSD Y4, Y0, Y0
	VMOVDQU Y0, (DI)
	ADDQ    $32, DI
	SUBQ    $8, CX
	JMP     clampLoop

clampDone:
	VZEROUPPER
	RET

// func narrowStoreAVX2Asm(dst *int16, dstStride uintptr, src *int32, width uintptr, height uintptr)
//
// Writes the final residual rows: every int32 lane is rounding-shifted right by
// four (VPADDD 8, VPSRAD $4) and saturated to int16 (VPACKSSDW). Width is four
// or a multiple of eight; src rows are contiguous, dst rows dstStride int16
// elements apart.
TEXT ·narrowStoreAVX2Asm(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ dstStride+8(FP), R8
	SHLQ $1, R8 // dst stride in bytes
	MOVQ src+16(FP), SI
	MOVQ width+24(FP), R9
	MOVQ height+32(FP), R10

	MOVL         $8, R11
	MOVL         R11, X8
	VPBROADCASTD X8, Y8 // rounding add of 8

	CMPQ R9, $4
	JEQ  narrowW4

narrowRow:
	TESTQ R10, R10
	JZ    narrowDone
	MOVQ  DI, R12 // dst row cursor
	MOVQ  R9, R13 // columns remaining

narrowCol:
	VMOVDQU      (SI), Y0
	VPADDD       Y8, Y0, Y0
	VPSRAD       $4, Y0, Y0
	VEXTRACTI128 $1, Y0, X1
	VPACKSSDW    X1, X0, X0 // 8 s32 -> 8 int16 (signed saturation)
	VMOVDQU      X0, (R12)
	ADDQ         $32, SI
	ADDQ         $16, R12
	SUBQ         $8, R13
	JNZ          narrowCol
	ADDQ         R8, DI
	DECQ         R10
	JMP          narrowRow

narrowW4:
	TESTQ     R10, R10
	JZ        narrowDone
	VMOVDQU   (SI), X0
	VPADDD    X8, X0, X0
	VPSRAD    $4, X0, X0
	VPACKSSDW X0, X0, X0 // low 4 s32 -> low 4 int16
	MOVQ      X0, (DI)
	ADDQ      $16, SI
	ADDQ      R8, DI
	DECQ      R10
	JMP       narrowW4

narrowDone:
	VZEROUPPER
	RET
