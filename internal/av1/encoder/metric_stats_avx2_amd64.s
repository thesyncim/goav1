// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 pixel-domain distortion statistics and the SATD coefficient reducer.
//
// pixelStats returns (sse, sum) with d = src - ref per sample: sum = Σd (signed)
// and sse = Σd². Bytes are zero-extended to int16 lanes and subtracted to form
// the exact signed diff (|d| ≤ 255 fits int16); VPMADDWD then folds pairs:
// madd(diff, ones) = d0+d1 (signed partial sum) and madd(diff, diff) = d0²+d1²
// (partial sse). Both are exact integer math, so the reductions are bit-exact
// with pixelStatsPureGo. This follows libaom aom_dsp/x86/variance_avx2.c.
//
// satdCoeffs returns Σ|coeff[i]| over count int32 coefficients (VPABSD), matching
// svt_aom_satd_c exactly.

// onesW16 holds sixteen int16 lanes of value 1 for the VPMADDWD diff-sum fold.
DATA onesW16<>+0(SB)/8, $0x0001000100010001
DATA onesW16<>+8(SB)/8, $0x0001000100010001
DATA onesW16<>+16(SB)/8, $0x0001000100010001
DATA onesW16<>+24(SB)/8, $0x0001000100010001
GLOBL onesW16<>(SB), RODATA|NOPTR, $32

// pixelStatsAVX2Ctx field offsets.
#define PSSRC       0
#define PSSRCSTRIDE 8
#define PSREF       16
#define PSREFSTRIDE 24
#define PSW         32
#define PSH         40
#define PSSSE       48
#define PSSUM       56

// func pixelStatsWideAVX2Asm(ctx *pixelStatsAVX2Ctx)
// Width is a multiple of 16 (16/32/64). Each 16-byte column chunk is expanded to
// a full YMM of int16 diffs. Y2 accumulates the signed diff-sum, Y3 the sse.
TEXT ·pixelStatsWideAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ PSSRC(AX), SI
	MOVQ PSSRCSTRIDE(AX), DX
	MOVQ PSREF(AX), DI
	MOVQ PSREFSTRIDE(AX), R8
	MOVQ PSW(AX), R10
	MOVQ PSH(AX), R9

	VPXOR      Y2, Y2, Y2
	VPXOR      Y3, Y3, Y3
	VMOVDQU    onesW16<>(SB), Y15

rowWide:
	MOVQ SI, R11
	MOVQ DI, R12
	MOVQ R10, R13
colWide:
	VPMOVZXBW (R11), Y0
	VPMOVZXBW (R12), Y1
	VPSUBW    Y1, Y0, Y0
	VPMADDWD  Y15, Y0, Y4
	VPADDD    Y4, Y2, Y2
	VPMADDWD  Y0, Y0, Y5
	VPADDD    Y5, Y3, Y3
	ADDQ $16, R11
	ADDQ $16, R12
	SUBQ $16, R13
	JNZ  colWide

	ADDQ DX, SI
	ADDQ R8, DI
	DECQ R9
	JNZ  rowWide

	// Reduce Y2 (signed diff-sum) -> Sum.
	VEXTRACTI128 $1, Y2, X0
	VPADDD       X0, X2, X2
	VPSHUFD      $0xEE, X2, X0
	VPADDD       X0, X2, X2
	VPSHUFD      $0x01, X2, X0
	VPADDD       X0, X2, X2
	VMOVD        X2, BX
	MOVLQSX      BX, BX
	MOVQ         BX, PSSUM(AX)

	// Reduce Y3 (sse, non-negative) -> SSE.
	VEXTRACTI128 $1, Y3, X0
	VPADDD       X0, X3, X3
	VPSHUFD      $0xEE, X3, X0
	VPADDD       X0, X3, X3
	VPSHUFD      $0x01, X3, X0
	VPADDD       X0, X3, X3
	VMOVD        X3, CX
	MOVQ         CX, PSSSE(AX)
	VZEROUPPER
	RET

// func pixelStats8AVX2Asm(ctx *pixelStatsAVX2Ctx)
// Width 8: one XMM (8 int16 lanes) per row.
TEXT ·pixelStats8AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ PSSRC(AX), SI
	MOVQ PSSRCSTRIDE(AX), DX
	MOVQ PSREF(AX), DI
	MOVQ PSREFSTRIDE(AX), R8
	MOVQ PSH(AX), R9

	VPXOR   X2, X2, X2
	VPXOR   X3, X3, X3
	VMOVDQU onesW16<>(SB), X7

row8:
	VPMOVZXBW (SI), X0
	VPMOVZXBW (DI), X1
	VPSUBW    X1, X0, X0
	VPMADDWD  X7, X0, X4
	VPADDD    X4, X2, X2
	VPMADDWD  X0, X0, X5
	VPADDD    X5, X3, X3
	ADDQ DX, SI
	ADDQ R8, DI
	DECQ R9
	JNZ  row8

	VPSHUFD $0xEE, X2, X0
	VPADDD  X0, X2, X2
	VPSHUFD $0x01, X2, X0
	VPADDD  X0, X2, X2
	VMOVD   X2, BX
	MOVLQSX BX, BX
	MOVQ    BX, PSSUM(AX)

	VPSHUFD $0xEE, X3, X0
	VPADDD  X0, X3, X3
	VPSHUFD $0x01, X3, X0
	VPADDD  X0, X3, X3
	VMOVD   X3, CX
	MOVQ    CX, PSSSE(AX)
	VZEROUPPER
	RET

// func pixelStats4AVX2Asm(ctx *pixelStatsAVX2Ctx)
// Width 4: load 4 bytes/row (upper XMM lanes zeroed), expand to 8 int16 lanes
// where lanes 4..7 are zero and contribute nothing to either fold.
TEXT ·pixelStats4AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ PSSRC(AX), SI
	MOVQ PSSRCSTRIDE(AX), DX
	MOVQ PSREF(AX), DI
	MOVQ PSREFSTRIDE(AX), R8
	MOVQ PSH(AX), R9

	VPXOR   X2, X2, X2
	VPXOR   X3, X3, X3
	VMOVDQU onesW16<>(SB), X7

row4:
	VMOVD     (SI), X0
	VMOVD     (DI), X1
	VPMOVZXBW X0, X0
	VPMOVZXBW X1, X1
	VPSUBW    X1, X0, X0
	VPMADDWD  X7, X0, X4
	VPADDD    X4, X2, X2
	VPMADDWD  X0, X0, X5
	VPADDD    X5, X3, X3
	ADDQ DX, SI
	ADDQ R8, DI
	DECQ R9
	JNZ  row4

	VPSHUFD $0xEE, X2, X0
	VPADDD  X0, X2, X2
	VPSHUFD $0x01, X2, X0
	VPADDD  X0, X2, X2
	VMOVD   X2, BX
	MOVLQSX BX, BX
	MOVQ    BX, PSSUM(AX)

	VPSHUFD $0xEE, X3, X0
	VPADDD  X0, X3, X3
	VPSHUFD $0x01, X3, X0
	VPADDD  X0, X3, X3
	VMOVD   X3, CX
	MOVQ    CX, PSSSE(AX)
	VZEROUPPER
	RET

// satdCoeffsAVX2Ctx field offsets.
#define SATDCOEFF 0
#define SATDCOUNT 8
#define SATDSUM   16

// func satdCoeffsAVX2Asm(ctx *satdCoeffsAVX2Ctx)
// Σ|coeff[i]| over count int32 coefficients: VPABSD folds 8 lanes/iteration; a
// scalar tail covers counts that are not a multiple of 8.
TEXT ·satdCoeffsAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ SATDCOEFF(AX), SI
	MOVQ SATDCOUNT(AX), CX

	VPXOR Y2, Y2, Y2

chunk8:
	CMPQ CX, $8
	JL   tailInit
	VMOVDQU (SI), Y0
	VPABSD  Y0, Y0
	VPADDD  Y0, Y2, Y2
	ADDQ $32, SI
	SUBQ $8, CX
	JMP  chunk8

tailInit:
	// Reduce Y2 (non-negative) into R14.
	VEXTRACTI128 $1, Y2, X0
	VPADDD       X0, X2, X2
	VPSHUFD      $0xEE, X2, X0
	VPADDD       X0, X2, X2
	VPSHUFD      $0x01, X2, X0
	VPADDD       X0, X2, X2
	VMOVD        X2, BX
	MOVQ         BX, R14

tail:
	TESTQ CX, CX
	JZ    done
	MOVLQSX (SI), BX
	TESTQ   BX, BX
	JGE     tailAdd
	NEGQ    BX
tailAdd:
	ADDQ BX, R14
	ADDQ $4, SI
	DECQ CX
	JMP  tail

done:
	MOVQ R14, SATDSUM(AX)
	VZEROUPPER
	RET
