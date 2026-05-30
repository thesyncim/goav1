// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// AVX2 AddResidualPlaneBlock kernels. They are bit-exact with
// AddResidualPlaneBlock's scalar inner loop (see plane.go): widen prediction
// and residual to s32 lanes, add, clamp to [0, max] with VPMAXSD(0)/VPMINSD,
// then narrow back. The s32 intermediate matches the reference, which performs
// the add and clamp in Go int and so never overflows.
//
// Each kernel processes the leading width&^7 columns of every row in groups of
// eight lanes; the Go wrapper handles the width&7 tail and all validation.
// Eight s32 lanes occupy one ymm; the per-group narrowing extracts the high
// 128 bits and re-packs so the eight results land contiguously.

//go:build amd64 && !purego

#include "textflag.h"

// func addResidual8AVX2Asm(dst *byte, dstStride uintptr, res *int16, resStride uintptr, max uint32, groups uintptr, height uintptr)
//
// 8-bit destination: each sample is one byte. groups is the number of 8-lane
// vector groups per row (width>>3). dstStride/resStride are in bytes; res
// advances by 16 bytes (8 int16) per group, dst by 8 bytes per group.
TEXT ·addResidual8AVX2Asm(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), AX
	MOVQ dstStride+8(FP), BX
	MOVQ res+16(FP), CX
	MOVQ resStride+24(FP), DX
	MOVL max+32(FP), R8
	MOVQ groups+40(FP), R9
	MOVQ height+48(FP), R10

	VPXOR        Y2, Y2, Y2 // zero (clamp lower bound)
	MOVL         R8, X3
	VPBROADCASTD X3, Y3 // max broadcast across 8 s32 lanes

rowLoop8:
	TESTQ R10, R10
	JZ    done8
	MOVQ  AX, R11 // dst row cursor
	MOVQ  CX, R12 // res row cursor
	MOVQ  R9, R13 // groups remaining

colLoop8:
	TESTQ      R13, R13
	JZ         rowAdvance8
	VPMOVZXBD  (R11), Y0 // 8 predicted bytes -> 8 s32
	VPMOVSXWD  (R12), Y1 // 8 residual int16 -> 8 s32
	VPADDD     Y1, Y0, Y0
	VPMAXSD    Y2, Y0, Y0
	VPMINSD    Y3, Y0, Y0
	VEXTRACTI128 $1, Y0, X4
	VPACKUSDW  X4, X0, X0 // 8 s32 -> 8 u16 (in range, no saturation)
	VPACKUSWB  X0, X0, X0 // low 8 u16 -> low 8 bytes
	MOVQ       X0, (R11)
	ADDQ       $8, R11
	ADDQ       $16, R12
	DECQ       R13
	JMP        colLoop8

rowAdvance8:
	ADDQ BX, AX
	ADDQ DX, CX
	DECQ R10
	JMP  rowLoop8

done8:
	VZEROUPPER
	RET

// func addResidual16AVX2Asm(dst *byte, dstStride uintptr, res *int16, resStride uintptr, max uint32, groups uintptr, height uintptr)
//
// High-bit-depth destination: each sample is a little-endian uint16. groups is
// the number of 8-lane vector groups per row (width>>3). dstStride/resStride
// are in bytes; both dst and res advance by 16 bytes per group.
TEXT ·addResidual16AVX2Asm(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), AX
	MOVQ dstStride+8(FP), BX
	MOVQ res+16(FP), CX
	MOVQ resStride+24(FP), DX
	MOVL max+32(FP), R8
	MOVQ groups+40(FP), R9
	MOVQ height+48(FP), R10

	VPXOR        Y2, Y2, Y2
	MOVL         R8, X3
	VPBROADCASTD X3, Y3

rowLoop16:
	TESTQ R10, R10
	JZ    done16
	MOVQ  AX, R11
	MOVQ  CX, R12
	MOVQ  R9, R13

colLoop16:
	TESTQ      R13, R13
	JZ         rowAdvance16
	VPMOVZXWD  (R11), Y0 // 8 predicted u16 -> 8 s32
	VPMOVSXWD  (R12), Y1 // 8 residual int16 -> 8 s32
	VPADDD     Y1, Y0, Y0
	VPMAXSD    Y2, Y0, Y0
	VPMINSD    Y3, Y0, Y0
	VEXTRACTI128 $1, Y0, X4
	VPACKUSDW  X4, X0, X0 // 8 s32 -> 8 u16 (in range, no saturation)
	VMOVDQU    X0, (R11)
	ADDQ       $16, R11
	ADDQ       $16, R12
	DECQ       R13
	JMP        colLoop16

rowAdvance16:
	ADDQ BX, AX
	ADDQ DX, CX
	DECQ R10
	JMP  rowLoop16

done16:
	VZEROUPPER
	RET
