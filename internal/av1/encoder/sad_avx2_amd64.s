// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 sum-of-absolute-differences kernels. Each row's SAD is accumulated with
// (V)PSADBW, whose byte-lane |a-b| sums land in the 64-bit lanes of the result
// (one lane per 8 source bytes). The x4 kernels keep four independent lane
// accumulators (one per reference); the step-4 variant derives its four
// references from a single base at +0/+4/+8/+12. SAD is exact integer math, so
// these are bit-exact with the portable references by construction.

// sadX4AMD64Ctx field offsets.
#define X4SRC    0
#define X4REF0   8
#define X4REF1   16
#define X4REF2   24
#define X4REF3   32
#define X4STRIDE 40
#define X4SUM0   48
#define X4SUM1   56
#define X4SUM2   64
#define X4SUM3   72

// sadStep4AMD64Ctx field offsets.
#define S4SRC    0
#define S4REF    8
#define S4STRIDE 16
#define S4SUM0   24
#define S4SUM1   32
#define S4SUM2   40
#define S4SUM3   48

// func sad8x8x4AVX2Asm(ctx *sadX4AMD64Ctx)
// Four 8x8 SADs; each row is 8 bytes so PSADBW leaves the whole block sum in
// the low quadword of each accumulator.
TEXT ·sad8x8x4AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ X4SRC(AX), SI
	MOVQ X4REF0(AX), R8
	MOVQ X4REF1(AX), R9
	MOVQ X4REF2(AX), R10
	MOVQ X4REF3(AX), R11
	MOVQ X4STRIDE(AX), DX

	PXOR X4, X4
	PXOR X5, X5
	PXOR X6, X6
	PXOR X7, X7

	MOVQ $8, CX
loop8x8x4:
	MOVQ (SI), X0
	MOVQ (R8), X1
	PSADBW X0, X1
	PADDQ X1, X4
	MOVQ (R9), X2
	PSADBW X0, X2
	PADDQ X2, X5
	MOVQ (R10), X3
	PSADBW X0, X3
	PADDQ X3, X6
	MOVQ (R11), X1
	PSADBW X0, X1
	PADDQ X1, X7
	ADDQ DX, SI
	ADDQ DX, R8
	ADDQ DX, R9
	ADDQ DX, R10
	ADDQ DX, R11
	DECQ CX
	JNZ  loop8x8x4

	MOVQ X4, BX
	MOVQ BX, X4SUM0(AX)
	MOVQ X5, BX
	MOVQ BX, X4SUM1(AX)
	MOVQ X6, BX
	MOVQ BX, X4SUM2(AX)
	MOVQ X7, BX
	MOVQ BX, X4SUM3(AX)
	RET

// func sad16x16x4AVX2Asm(ctx *sadX4AMD64Ctx)
// Four 16x16 SADs; each 16-byte row's PSADBW yields two partial sums (bytes
// 0-7 and 8-15) that are folded together in the final reduction.
TEXT ·sad16x16x4AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ X4SRC(AX), SI
	MOVQ X4REF0(AX), R8
	MOVQ X4REF1(AX), R9
	MOVQ X4REF2(AX), R10
	MOVQ X4REF3(AX), R11
	MOVQ X4STRIDE(AX), DX

	PXOR X4, X4
	PXOR X5, X5
	PXOR X6, X6
	PXOR X7, X7

	MOVQ $16, CX
loop16x16x4:
	MOVOU (SI), X0
	MOVOU (R8), X1
	PSADBW X0, X1
	PADDQ X1, X4
	MOVOU (R9), X2
	PSADBW X0, X2
	PADDQ X2, X5
	MOVOU (R10), X3
	PSADBW X0, X3
	PADDQ X3, X6
	MOVOU (R11), X1
	PSADBW X0, X1
	PADDQ X1, X7
	ADDQ DX, SI
	ADDQ DX, R8
	ADDQ DX, R9
	ADDQ DX, R10
	ADDQ DX, R11
	DECQ CX
	JNZ  loop16x16x4

	PSHUFD $0xEE, X4, X0
	PADDQ  X0, X4
	MOVQ   X4, BX
	MOVQ   BX, X4SUM0(AX)
	PSHUFD $0xEE, X5, X0
	PADDQ  X0, X5
	MOVQ   X5, BX
	MOVQ   BX, X4SUM1(AX)
	PSHUFD $0xEE, X6, X0
	PADDQ  X0, X6
	MOVQ   X6, BX
	MOVQ   BX, X4SUM2(AX)
	PSHUFD $0xEE, X7, X0
	PADDQ  X0, X7
	MOVQ   X7, BX
	MOVQ   BX, X4SUM3(AX)
	RET

// func sad32x32x4AVX2Asm(ctx *sadX4AMD64Ctx)
// Four 32x32 SADs; each 32-byte row's VPSADBW yields four partial sums per
// accumulator YMM, folded to one in the final reduction.
TEXT ·sad32x32x4AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ X4SRC(AX), SI
	MOVQ X4REF0(AX), R8
	MOVQ X4REF1(AX), R9
	MOVQ X4REF2(AX), R10
	MOVQ X4REF3(AX), R11
	MOVQ X4STRIDE(AX), DX

	VPXOR Y4, Y4, Y4
	VPXOR Y5, Y5, Y5
	VPXOR Y6, Y6, Y6
	VPXOR Y7, Y7, Y7

	MOVQ $32, CX
loop32x32x4:
	VMOVDQU (SI), Y0
	VMOVDQU (R8), Y1
	VPSADBW Y0, Y1, Y1
	VPADDQ  Y1, Y4, Y4
	VMOVDQU (R9), Y2
	VPSADBW Y0, Y2, Y2
	VPADDQ  Y2, Y5, Y5
	VMOVDQU (R10), Y3
	VPSADBW Y0, Y3, Y3
	VPADDQ  Y3, Y6, Y6
	VMOVDQU (R11), Y1
	VPSADBW Y0, Y1, Y1
	VPADDQ  Y1, Y7, Y7
	ADDQ DX, SI
	ADDQ DX, R8
	ADDQ DX, R9
	ADDQ DX, R10
	ADDQ DX, R11
	DECQ CX
	JNZ  loop32x32x4

	VEXTRACTI128 $1, Y4, X0
	VPADDQ       X0, X4, X4
	VPSHUFD      $0xEE, X4, X0
	VPADDQ       X0, X4, X4
	VMOVQ        X4, BX
	MOVQ         BX, X4SUM0(AX)
	VEXTRACTI128 $1, Y5, X0
	VPADDQ       X0, X5, X5
	VPSHUFD      $0xEE, X5, X0
	VPADDQ       X0, X5, X5
	VMOVQ        X5, BX
	MOVQ         BX, X4SUM1(AX)
	VEXTRACTI128 $1, Y6, X0
	VPADDQ       X0, X6, X6
	VPSHUFD      $0xEE, X6, X0
	VPADDQ       X0, X6, X6
	VMOVQ        X6, BX
	MOVQ         BX, X4SUM2(AX)
	VEXTRACTI128 $1, Y7, X0
	VPADDQ       X0, X7, X7
	VPSHUFD      $0xEE, X7, X0
	VPADDQ       X0, X7, X7
	VMOVQ        X7, BX
	MOVQ         BX, X4SUM3(AX)
	VZEROUPPER
	RET

// func sad8x8x4Step4AVX2Asm(ctx *sadStep4AMD64Ctx)
TEXT ·sad8x8x4Step4AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ S4SRC(AX), SI
	MOVQ S4REF(AX), R8
	LEAQ 4(R8), R9
	LEAQ 8(R8), R10
	LEAQ 12(R8), R11
	MOVQ S4STRIDE(AX), DX

	PXOR X4, X4
	PXOR X5, X5
	PXOR X6, X6
	PXOR X7, X7

	MOVQ $8, CX
loop8x8s4:
	MOVQ (SI), X0
	MOVQ (R8), X1
	PSADBW X0, X1
	PADDQ X1, X4
	MOVQ (R9), X2
	PSADBW X0, X2
	PADDQ X2, X5
	MOVQ (R10), X3
	PSADBW X0, X3
	PADDQ X3, X6
	MOVQ (R11), X1
	PSADBW X0, X1
	PADDQ X1, X7
	ADDQ DX, SI
	ADDQ DX, R8
	ADDQ DX, R9
	ADDQ DX, R10
	ADDQ DX, R11
	DECQ CX
	JNZ  loop8x8s4

	MOVQ X4, BX
	MOVQ BX, S4SUM0(AX)
	MOVQ X5, BX
	MOVQ BX, S4SUM1(AX)
	MOVQ X6, BX
	MOVQ BX, S4SUM2(AX)
	MOVQ X7, BX
	MOVQ BX, S4SUM3(AX)
	RET

// func sad16x16x4Step4AVX2Asm(ctx *sadStep4AMD64Ctx)
TEXT ·sad16x16x4Step4AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ S4SRC(AX), SI
	MOVQ S4REF(AX), R8
	LEAQ 4(R8), R9
	LEAQ 8(R8), R10
	LEAQ 12(R8), R11
	MOVQ S4STRIDE(AX), DX

	PXOR X4, X4
	PXOR X5, X5
	PXOR X6, X6
	PXOR X7, X7

	MOVQ $16, CX
loop16x16s4:
	MOVOU (SI), X0
	MOVOU (R8), X1
	PSADBW X0, X1
	PADDQ X1, X4
	MOVOU (R9), X2
	PSADBW X0, X2
	PADDQ X2, X5
	MOVOU (R10), X3
	PSADBW X0, X3
	PADDQ X3, X6
	MOVOU (R11), X1
	PSADBW X0, X1
	PADDQ X1, X7
	ADDQ DX, SI
	ADDQ DX, R8
	ADDQ DX, R9
	ADDQ DX, R10
	ADDQ DX, R11
	DECQ CX
	JNZ  loop16x16s4

	PSHUFD $0xEE, X4, X0
	PADDQ  X0, X4
	MOVQ   X4, BX
	MOVQ   BX, S4SUM0(AX)
	PSHUFD $0xEE, X5, X0
	PADDQ  X0, X5
	MOVQ   X5, BX
	MOVQ   BX, S4SUM1(AX)
	PSHUFD $0xEE, X6, X0
	PADDQ  X0, X6
	MOVQ   X6, BX
	MOVQ   BX, S4SUM2(AX)
	PSHUFD $0xEE, X7, X0
	PADDQ  X0, X7
	MOVQ   X7, BX
	MOVQ   BX, S4SUM3(AX)
	RET

// func sad32x32x4Step4AVX2Asm(ctx *sadStep4AMD64Ctx)
TEXT ·sad32x32x4Step4AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ S4SRC(AX), SI
	MOVQ S4REF(AX), R8
	LEAQ 4(R8), R9
	LEAQ 8(R8), R10
	LEAQ 12(R8), R11
	MOVQ S4STRIDE(AX), DX

	VPXOR Y4, Y4, Y4
	VPXOR Y5, Y5, Y5
	VPXOR Y6, Y6, Y6
	VPXOR Y7, Y7, Y7

	MOVQ $32, CX
loop32x32s4:
	VMOVDQU (SI), Y0
	VMOVDQU (R8), Y1
	VPSADBW Y0, Y1, Y1
	VPADDQ  Y1, Y4, Y4
	VMOVDQU (R9), Y2
	VPSADBW Y0, Y2, Y2
	VPADDQ  Y2, Y5, Y5
	VMOVDQU (R10), Y3
	VPSADBW Y0, Y3, Y3
	VPADDQ  Y3, Y6, Y6
	VMOVDQU (R11), Y1
	VPSADBW Y0, Y1, Y1
	VPADDQ  Y1, Y7, Y7
	ADDQ DX, SI
	ADDQ DX, R8
	ADDQ DX, R9
	ADDQ DX, R10
	ADDQ DX, R11
	DECQ CX
	JNZ  loop32x32s4

	VEXTRACTI128 $1, Y4, X0
	VPADDQ       X0, X4, X4
	VPSHUFD      $0xEE, X4, X0
	VPADDQ       X0, X4, X4
	VMOVQ        X4, BX
	MOVQ         BX, S4SUM0(AX)
	VEXTRACTI128 $1, Y5, X0
	VPADDQ       X0, X5, X5
	VPSHUFD      $0xEE, X5, X0
	VPADDQ       X0, X5, X5
	VMOVQ        X5, BX
	MOVQ         BX, S4SUM1(AX)
	VEXTRACTI128 $1, Y6, X0
	VPADDQ       X0, X6, X6
	VPSHUFD      $0xEE, X6, X0
	VPADDQ       X0, X6, X6
	VMOVQ        X6, BX
	MOVQ         BX, S4SUM2(AX)
	VEXTRACTI128 $1, Y7, X0
	VPADDQ       X0, X7, X7
	VPSHUFD      $0xEE, X7, X0
	VPADDQ       X0, X7, X7
	VMOVQ        X7, BX
	MOVQ         BX, S4SUM3(AX)
	VZEROUPPER
	RET

// --- single-block and dual-stride leaves + compound average ---------------

// sad8x8SSE2Ctx field offsets (Src, Ref, Stride, Sum), reused by sad32x32.
#define SGSRC    0
#define SGREF    8
#define SGSTRIDE 16
#define SGSUM    24

// sad8x8DualSSE2Ctx field offsets (Src, Ref, SrcStride, RefStride, Sum).
#define DLSRC       0
#define DLREF       8
#define DLSRCSTRIDE 16
#define DLREFSTRIDE 24
#define DLSUM       32

// sadCompoundAMD64Ctx field offsets.
#define CPSRC        0
#define CPSRCSTRIDE  8
#define CPREF0       16
#define CPREF0STRIDE 24
#define CPREF1       32
#define CPREF1STRIDE 40
#define CPSUM        48

// func sad32x32AVX2Asm(ctx *sad8x8SSE2Ctx)
// One 32x32 SAD: each 32-byte row's VPSADBW yields four partial sums folded to
// one in the reduction.
TEXT ·sad32x32AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ SGSRC(AX), SI
	MOVQ SGREF(AX), DI
	MOVQ SGSTRIDE(AX), DX

	VPXOR Y2, Y2, Y2

	MOVQ $32, CX
loop32single:
	VMOVDQU (SI), Y0
	VMOVDQU (DI), Y1
	VPSADBW Y0, Y1, Y1
	VPADDQ  Y1, Y2, Y2
	ADDQ DX, SI
	ADDQ DX, DI
	DECQ CX
	JNZ  loop32single

	VEXTRACTI128 $1, Y2, X0
	VPADDQ       X0, X2, X2
	VPSHUFD      $0xEE, X2, X0
	VPADDQ       X0, X2, X2
	VMOVQ        X2, BX
	MOVQ         BX, SGSUM(AX)
	VZEROUPPER
	RET

// func sad16x16DualAVX2Asm(ctx *sad8x8DualSSE2Ctx)
// One 16x16 SAD with independent source and reference strides.
TEXT ·sad16x16DualAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ DLSRC(AX), SI
	MOVQ DLREF(AX), DI
	MOVQ DLSRCSTRIDE(AX), DX
	MOVQ DLREFSTRIDE(AX), R8

	PXOR X2, X2

	MOVQ $16, CX
loop16dual:
	MOVOU  (SI), X0
	MOVOU  (DI), X1
	PSADBW X0, X1
	PADDQ  X1, X2
	ADDQ DX, SI
	ADDQ R8, DI
	DECQ CX
	JNZ  loop16dual

	PSHUFD $0xEE, X2, X0
	PADDQ  X0, X2
	MOVQ   X2, BX
	MOVQ   BX, DLSUM(AX)
	RET

// func sad32x32DualAVX2Asm(ctx *sad8x8DualSSE2Ctx)
// One 32x32 SAD with independent source and reference strides.
TEXT ·sad32x32DualAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ DLSRC(AX), SI
	MOVQ DLREF(AX), DI
	MOVQ DLSRCSTRIDE(AX), DX
	MOVQ DLREFSTRIDE(AX), R8

	VPXOR Y2, Y2, Y2

	MOVQ $32, CX
loop32dual:
	VMOVDQU (SI), Y0
	VMOVDQU (DI), Y1
	VPSADBW Y0, Y1, Y1
	VPADDQ  Y1, Y2, Y2
	ADDQ DX, SI
	ADDQ R8, DI
	DECQ CX
	JNZ  loop32dual

	VEXTRACTI128 $1, Y2, X0
	VPADDQ       X0, X2, X2
	VPSHUFD      $0xEE, X2, X0
	VPADDQ       X0, X2, X2
	VMOVQ        X2, BX
	MOVQ         BX, DLSUM(AX)
	VZEROUPPER
	RET

// func sad8x8CompoundAvgBlockAVX2Asm(ctx *sadCompoundAMD64Ctx)
// SAD(src, round((ref0+ref1)/2)) over an 8x8 block; PAVGB forms the byte
// average (a+b+1)>>1 exactly, matching the portable reference.
TEXT ·sad8x8CompoundAvgBlockAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ CPSRC(AX), SI
	MOVQ CPSRCSTRIDE(AX), DX
	MOVQ CPREF0(AX), DI
	MOVQ CPREF0STRIDE(AX), R8
	MOVQ CPREF1(AX), R9
	MOVQ CPREF1STRIDE(AX), R10

	PXOR X3, X3

	MOVQ $8, CX
loop8compound:
	MOVQ   (DI), X1
	MOVQ   (R9), X2
	PAVGB  X2, X1
	MOVQ   (SI), X0
	PSADBW X0, X1
	PADDQ  X1, X3
	ADDQ DX, SI
	ADDQ R8, DI
	ADDQ R10, R9
	DECQ CX
	JNZ  loop8compound

	MOVQ X3, BX
	MOVQ BX, CPSUM(AX)
	RET
