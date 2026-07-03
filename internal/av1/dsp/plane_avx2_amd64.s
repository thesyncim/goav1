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

// AVX2 AddRawTransformPlaneBlock kernels. They fuse the inverse-transform
// column-pass output into an already-predicted block, bit-exact with
// addRawTransformPlaneBlockPureGo (see plane.go) and with the NEON
// addRawTransform*NEONAsm kernels: each raw int32 sample is rounded by four
// with a 32-bit add of 8 (VPADDD) and an arithmetic shift (VPSRAD $4), saturated
// to the int16 range with VPMINSD(32767)/VPMAXSD(-32768) — matching NEON's
// add/sshr/sqxtn and rawTransformResidual's (v+8)>>4 then int16 clamp — then
// added to the predicted pixel and clamped to [0, max] with VPMAXSD(0)/VPMINSD.
// The 32-bit add of 8 mirrors NEON's 32-bit add exactly; the raw values a
// decode/encode reconstruction produces are far below the int32 overflow
// boundary, so this is equal to the pure-Go int64 round for every real input.
//
// Each kernel processes the leading width&^7 columns of every row in groups of
// eight lanes; the Go wrapper handles the width&7 tail and all validation.

// func addRawTransform8AVX2Asm(dst *byte, dstStride uintptr, raw *int32, rawStride uintptr, max uint32, groups uintptr, height uintptr)
//
// 8-bit destination with raw int32 inverse-transform samples. dstStride/rawStride
// are in bytes; raw advances by 32 bytes (8 int32) per group, dst by 8 bytes.
TEXT ·addRawTransform8AVX2Asm(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), AX
	MOVQ dstStride+8(FP), BX
	MOVQ raw+16(FP), CX
	MOVQ rawStride+24(FP), DX
	MOVL max+32(FP), R8
	MOVQ groups+40(FP), R9
	MOVQ height+48(FP), R10

	VPXOR        Y2, Y2, Y2 // zero (clamp lower bound)
	MOVL         R8, X3
	VPBROADCASTD X3, Y3     // max broadcast across 8 s32 lanes
	MOVL         $8, R14
	MOVL         R14, X4
	VPBROADCASTD X4, Y4     // rounding add of 8
	MOVL         $32767, R14
	MOVL         R14, X5
	VPBROADCASTD X5, Y5     // int16 max
	MOVL         $-32768, R14
	MOVL         R14, X6
	VPBROADCASTD X6, Y6     // int16 min

rawRowLoop8:
	TESTQ R10, R10
	JZ    rawDone8
	MOVQ  AX, R11 // dst row cursor
	MOVQ  CX, R12 // raw row cursor
	MOVQ  R9, R13 // groups remaining

rawColLoop8:
	TESTQ        R13, R13
	JZ           rawRowAdvance8
	VMOVDQU      (R12), Y1  // 8 raw int32
	VPADDD       Y4, Y1, Y1 // + 8
	VPSRAD       $4, Y1, Y1 // >> 4 (arithmetic)
	VPMINSD      Y5, Y1, Y1 // clamp <= 32767
	VPMAXSD      Y6, Y1, Y1 // clamp >= -32768 -> int16 residual as s32
	VPMOVZXBD    (R11), Y0  // 8 predicted bytes -> 8 s32
	VPADDD       Y1, Y0, Y0
	VPMAXSD      Y2, Y0, Y0
	VPMINSD      Y3, Y0, Y0
	VEXTRACTI128 $1, Y0, X7
	VPACKUSDW    X7, X0, X0 // 8 s32 -> 8 u16 (in range, no saturation)
	VPACKUSWB    X0, X0, X0 // low 8 u16 -> low 8 bytes
	MOVQ         X0, (R11)
	ADDQ         $8, R11
	ADDQ         $32, R12
	DECQ         R13
	JMP          rawColLoop8

rawRowAdvance8:
	ADDQ BX, AX
	ADDQ DX, CX
	DECQ R10
	JMP  rawRowLoop8

rawDone8:
	VZEROUPPER
	RET

// func addRawTransform16AVX2Asm(dst *byte, dstStride uintptr, raw *int32, rawStride uintptr, max uint32, groups uintptr, height uintptr)
//
// High-bit-depth destination: each sample is a little-endian uint16. dst
// advances by 16 bytes per group, raw by 32 bytes (8 int32).
TEXT ·addRawTransform16AVX2Asm(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), AX
	MOVQ dstStride+8(FP), BX
	MOVQ raw+16(FP), CX
	MOVQ rawStride+24(FP), DX
	MOVL max+32(FP), R8
	MOVQ groups+40(FP), R9
	MOVQ height+48(FP), R10

	VPXOR        Y2, Y2, Y2
	MOVL         R8, X3
	VPBROADCASTD X3, Y3
	MOVL         $8, R14
	MOVL         R14, X4
	VPBROADCASTD X4, Y4
	MOVL         $32767, R14
	MOVL         R14, X5
	VPBROADCASTD X5, Y5
	MOVL         $-32768, R14
	MOVL         R14, X6
	VPBROADCASTD X6, Y6

rawRowLoop16:
	TESTQ R10, R10
	JZ    rawDone16
	MOVQ  AX, R11
	MOVQ  CX, R12
	MOVQ  R9, R13

rawColLoop16:
	TESTQ        R13, R13
	JZ           rawRowAdvance16
	VMOVDQU      (R12), Y1
	VPADDD       Y4, Y1, Y1
	VPSRAD       $4, Y1, Y1
	VPMINSD      Y5, Y1, Y1
	VPMAXSD      Y6, Y1, Y1
	VPMOVZXWD    (R11), Y0  // 8 predicted u16 -> 8 s32
	VPADDD       Y1, Y0, Y0
	VPMAXSD      Y2, Y0, Y0
	VPMINSD      Y3, Y0, Y0
	VEXTRACTI128 $1, Y0, X7
	VPACKUSDW    X7, X0, X0
	VMOVDQU      X0, (R11)
	ADDQ         $16, R11
	ADDQ         $32, R12
	DECQ         R13
	JMP          rawColLoop16

rawRowAdvance16:
	ADDQ BX, AX
	ADDQ DX, CX
	DECQ R10
	JMP  rawRowLoop16

rawDone16:
	VZEROUPPER
	RET
