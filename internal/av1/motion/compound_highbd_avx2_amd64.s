// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 high-bit-depth (10/12-bit) compound kernels, bit-exact with the
// *HighBD*PureGo references in compound.go. Each kernel processes 4 output
// columns per group into a 128-bit X register (4 int32 lanes). uint16 samples
// are zero-extended (VPMOVZXWD); int32 intermediates are loaded directly. round0
// (and its bias 1<<(round0-1)) are passed in and applied with a register-count
// VPSRAD. The CONV_BUF store is a truncating uint16 narrow (cmpLowWord + VMOVQ).
//
// cmpLowWordHBD selects, within the low 128 bits, bytes {0,1,4,5,8,9,12,13}
// into the low 8 bytes: the truncating (mod 2^16) int32->uint16 narrow of 4
// lanes. (A file-local copy; the 8-bit file has its own cmpLowWord<>.)
GLOBL cmpLowWordHBD<>(SB), RODATA|NOPTR, $16
DATA cmpLowWordHBD<>+0(SB)/8, $0x0d0c090805040100
DATA cmpLowWordHBD<>+8(SB)/8, $0x8080808080808080

// compoundConvAVX2Ctx field offsets (must match compound_avx2_amd64.go).
#define DST     0
#define REF     8
#define KERNEL  16
#define XKERN   24
#define REFSTR  32
#define WIDTH   40
#define HEIGHT  48
#define IM      56
#define IMSTR   64
#define ROUND0  72
#define SCALE   80
#define RNDOFF  88
#define XBIAS   96
#define YBIAS   104

// func compoundCopyHighBDAVX2Asm(ctx *compoundConvAVX2Ctx)
//
// out[x] = uint16(ref[x]*scale + roundOffset).
TEXT ·compoundCopyHighBDAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ DST(R10), DI      // CONV_BUF running cursor
	MOVQ REF(R10), SI      // ref row base
	MOVQ REFSTR(R10), R9
	MOVQ HEIGHT(R10), R11
	MOVQ SCALE(R10), AX
	MOVL AX, X13
	VPBROADCASTD X13, X13
	MOVQ RNDOFF(R10), AX
	MOVL AX, X12
	VPBROADCASTD X12, X12
	MOVQ WIDTH(R10), R10

hbcRowLoop:
	TESTQ R11, R11
	JZ    hbcDone
	MOVQ  SI, R13
	MOVQ  R10, CX

hbcColLoop:
	VPMOVZXWD (R13), X0
	VPMULLD X13, X0, X0
	VPADDD X12, X0, X0
	VPSHUFB cmpLowWordHBD<>(SB), X0, X0
	VMOVQ X0, (DI)
	ADDQ $8, R13           // 4 samples * 2 bytes
	ADDQ $8, DI            // 4 uint16
	SUBQ $4, CX
	JNZ  hbcColLoop

	ADDQ R9, SI
	DECQ R11
	JMP  hbcRowLoop

hbcDone:
	VZEROUPPER
	RET

// func compoundXHighBDAVX2Asm(ctx *compoundConvAVX2Ctx)
//
// out[x] = uint16(((sum+round0Bias)>>round0)*scale + roundOffset), sum = 8-tap
// horizontal MAC over uint16 samples. ref addresses the first tap (refX-fo).
TEXT ·compoundXHighBDAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ DST(R10), DI
	MOVQ REF(R10), SI
	MOVQ KERNEL(R10), BX
	MOVQ REFSTR(R10), R9
	MOVQ HEIGHT(R10), R11
	MOVQ ROUND0(R10), R14  // round0 shift
	MOVQ SCALE(R10), AX
	MOVL AX, X13
	VPBROADCASTD X13, X13  // scale
	MOVQ RNDOFF(R10), AX
	MOVL AX, X12
	VPBROADCASTD X12, X12  // roundOffset
	MOVL $1, AX            // round0 bias = 1<<(round0-1)
	MOVQ R14, CX
	DECQ CX
	SHLL CX, AX
	MOVL AX, X14
	VPBROADCASTD X14, X14
	MOVQ WIDTH(R10), R10

hbxRowLoop:
	TESTQ R11, R11
	JZ    hbxDone
	MOVQ  SI, R13
	MOVQ  R10, CX

hbxColLoop:
	VPXOR X0, X0, X0
	VPMOVZXWD (R13), X1
	MOVWLSX (BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 2(R13), X1
	MOVWLSX 2(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 4(R13), X1
	MOVWLSX 4(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 6(R13), X1
	MOVWLSX 6(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 8(R13), X1
	MOVWLSX 8(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 10(R13), X1
	MOVWLSX 10(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 12(R13), X1
	MOVWLSX 12(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 14(R13), X1
	MOVWLSX 14(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0

	VPADDD X14, X0, X0     // + round0Bias
	VMOVD R14, X3
	VPSRAD X3, X0, X0      // >> round0
	VPMULLD X13, X0, X0    // * scale
	VPADDD X12, X0, X0     // + roundOffset
	VPSHUFB cmpLowWordHBD<>(SB), X0, X0
	VMOVQ X0, (DI)

	ADDQ $8, R13
	ADDQ $8, DI
	SUBQ $4, CX
	JNZ  hbxColLoop

	ADDQ R9, SI
	DECQ R11
	JMP  hbxRowLoop

hbxDone:
	VZEROUPPER
	RET

// func compoundYHighBDAVX2Asm(ctx *compoundConvAVX2Ctx)
//
// out[x] = uint16(roundPowerOfTwo7(sum*scale) + roundOffset), sum = 8-tap
// vertical MAC over uint16 rows. ref addresses the first tap row (refY-fo).
TEXT ·compoundYHighBDAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ DST(R10), DI
	MOVQ REF(R10), SI
	MOVQ KERNEL(R10), BX
	MOVQ REFSTR(R10), R9
	MOVQ HEIGHT(R10), R11
	MOVQ SCALE(R10), AX
	MOVL AX, X13
	VPBROADCASTD X13, X13  // scale
	MOVQ RNDOFF(R10), AX
	MOVL AX, X12
	VPBROADCASTD X12, X12  // roundOffset
	MOVL $64, AX           // 1<<(compoundRound1Bits-1) = 64
	MOVL AX, X15
	VPBROADCASTD X15, X15
	MOVQ WIDTH(R10), R10

hbyRowLoop:
	TESTQ R11, R11
	JZ    hbyDone
	MOVQ  SI, R13
	MOVQ  R10, CX

hbyColLoop:
	VPXOR X0, X0, X0
	MOVQ  R13, DX
	VPMOVZXWD (DX), X1
	MOVWLSX (BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	ADDQ R9, DX
	VPMOVZXWD (DX), X1
	MOVWLSX 2(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	ADDQ R9, DX
	VPMOVZXWD (DX), X1
	MOVWLSX 4(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	ADDQ R9, DX
	VPMOVZXWD (DX), X1
	MOVWLSX 6(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	ADDQ R9, DX
	VPMOVZXWD (DX), X1
	MOVWLSX 8(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	ADDQ R9, DX
	VPMOVZXWD (DX), X1
	MOVWLSX 10(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	ADDQ R9, DX
	VPMOVZXWD (DX), X1
	MOVWLSX 12(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	ADDQ R9, DX
	VPMOVZXWD (DX), X1
	MOVWLSX 14(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0

	VPMULLD X13, X0, X0    // * scale
	VPADDD X15, X0, X0     // + 64
	VPSRAD $7, X0, X0      // >> 7
	VPADDD X12, X0, X0     // + roundOffset
	VPSHUFB cmpLowWordHBD<>(SB), X0, X0
	VMOVQ X0, (DI)

	ADDQ $8, R13
	ADDQ $8, DI
	SUBQ $4, CX
	JNZ  hbyColLoop

	ADDQ R9, SI
	DECQ R11
	JMP  hbyRowLoop

hbyDone:
	VZEROUPPER
	RET

// func compound2DHighBDAVX2Asm(ctx *compoundConvAVX2Ctx)
//
// Pass 1 (horizontal, imH=height+7 rows): 8-tap MAC over uint16 samples, +
// xBias, + round0Bias, >> round0, store int32 into `im`.
// Pass 2 (vertical, height rows): 8-tap MAC over int32 `im` rows, + yBias,
// round by compoundRound1Bits=7, truncating uint16 narrow into the CONV_BUF.
TEXT ·compound2DHighBDAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ REF(R10), SI      // ref tap-window base
	MOVQ XKERN(R10), R14   // horizontal kernel
	MOVQ REFSTR(R10), R9   // ref stride
	MOVQ IM(R10), R15      // im base
	MOVQ IMSTR(R10), R12   // im stride (int32 elements)
	SHLQ $2, R12           // -> bytes
	MOVQ HEIGHT(R10), R11  // height
	MOVQ ROUND0(R10), R8   // round0 shift
	MOVQ XBIAS(R10), AX
	MOVL AX, X13
	VPBROADCASTD X13, X13  // xBias
	MOVL $1, AX            // round0Bias = 1<<(round0-1)
	MOVQ R8, CX
	DECQ CX
	SHLL CX, AX
	MOVL AX, X12
	VPBROADCASTD X12, X12
	MOVQ WIDTH(R10), R10   // width

	// imH = height + 7
	MOVQ R11, DI
	ADDQ $7, DI

hb2RowLoop:
	TESTQ DI, DI
	JZ    hb2Done
	MOVQ  SI, DX           // ref column cursor
	MOVQ  R15, BX          // im column cursor
	MOVQ  R10, CX

hb2ColLoop:
	VPXOR X0, X0, X0
	VPMOVZXWD (DX), X1
	MOVWLSX (R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 2(DX), X1
	MOVWLSX 2(R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 4(DX), X1
	MOVWLSX 4(R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 6(DX), X1
	MOVWLSX 6(R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 8(DX), X1
	MOVWLSX 8(R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 10(DX), X1
	MOVWLSX 10(R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 12(DX), X1
	MOVWLSX 12(R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 14(DX), X1
	MOVWLSX 14(R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0

	VPADDD X13, X0, X0     // + xBias
	VPADDD X12, X0, X0     // + round0Bias
	VMOVD R8, X3
	VPSRAD X3, X0, X0      // >> round0
	VMOVDQU X0, (BX)       // store 4 int32

	ADDQ $8, DX
	ADDQ $16, BX
	SUBQ $4, CX
	JNZ  hb2ColLoop

	ADDQ R9, SI
	ADDQ R12, R15
	DECQ DI
	JMP  hb2RowLoop

hb2Done:
	// ---- Pass 2 ----
	MOVQ ctx+0(FP), AX
	MOVQ DST(AX), DI       // CONV_BUF running cursor
	MOVQ IM(AX), R15       // im base (reload)
	MOVQ KERNEL(AX), BX    // vertical kernel
	MOVQ YBIAS(AX), AX
	MOVL AX, X13
	VPBROADCASTD X13, X13  // yBias
	MOVL $64, AX
	MOVL AX, X15
	VPBROADCASTD X15, X15  // round1 bias 64

hbv2RowLoop:
	TESTQ R11, R11
	JZ    hbv2Done
	MOVQ  R15, SI          // im tap-window base for this output row
	MOVQ  R10, CX

hbv2ColLoop:
	VPXOR X0, X0, X0
	MOVQ  SI, DX
	VMOVDQU (DX), X1
	MOVWLSX (BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	ADDQ R12, DX
	VMOVDQU (DX), X1
	MOVWLSX 2(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	ADDQ R12, DX
	VMOVDQU (DX), X1
	MOVWLSX 4(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	ADDQ R12, DX
	VMOVDQU (DX), X1
	MOVWLSX 6(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	ADDQ R12, DX
	VMOVDQU (DX), X1
	MOVWLSX 8(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	ADDQ R12, DX
	VMOVDQU (DX), X1
	MOVWLSX 10(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	ADDQ R12, DX
	VMOVDQU (DX), X1
	MOVWLSX 12(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	ADDQ R12, DX
	VMOVDQU (DX), X1
	MOVWLSX 14(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0

	VPADDD X13, X0, X0     // + yBias
	VPADDD X15, X0, X0     // + 64
	VPSRAD $7, X0, X0      // >> 7
	VPSHUFB cmpLowWordHBD<>(SB), X0, X0
	VMOVQ X0, (DI)

	ADDQ $16, SI           // im tap-window += 4 int32
	ADDQ $8, DI            // dst += 4 uint16
	SUBQ $4, CX
	JNZ  hbv2ColLoop

	ADDQ R12, R15
	DECQ R11
	JMP  hbv2RowLoop

hbv2Done:
	VZEROUPPER
	RET
