// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 8-bit compound inter-prediction kernels, bit-exact with the *PureGo
// references in compound.go. Each kernel processes 8 output columns per inner
// iteration into a Y register holding 8 int32 accumulators (lane i == column i,
// natural order from VPMOVZXBD/VPMOVZXWD).
//
// See compound_avx2_amd64.go for the arithmetic/bit-exactness rationale.

// cmpLowWord gathers, within each 128-bit lane, the low 16 bits of the four
// int32 words into the low 8 bytes (bytes {0,1,4,5,8,9,12,13}); the upper 8
// bytes are zeroed. This performs the truncating (mod 2^16) int->uint16 narrow.
GLOBL cmpLowWord<>(SB), RODATA|NOPTR, $32
DATA cmpLowWord<>+0(SB)/8, $0x0d0c090805040100
DATA cmpLowWord<>+8(SB)/8, $0x8080808080808080
DATA cmpLowWord<>+16(SB)/8, $0x0d0c090805040100
DATA cmpLowWord<>+24(SB)/8, $0x8080808080808080

// compoundConvAVX2Ctx field offsets.
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

// compoundBlendAVX2Ctx field offsets.
#define B_DST    0
#define B_SRC0   8
#define B_SRC1   16
#define B_DSTSTR 24
#define B_WIDTH  32
#define B_HEIGHT 40
#define B_FWD    48
#define B_BCK    56
#define B_RNDOFF 64

// func blendCompoundAvg8AVX2Asm(ctx *compoundBlendAVX2Ctx)
//
// tmp = s0*fwd + s1*bck; tmp >>= 4; tmp -= roundOffset;
// dst = clip8((tmp + 8) >> 4). src0/src1 are contiguous uint16 (width-strided);
// dst is a byte plane (dstStr).
TEXT ·blendCompoundAvg8AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ B_DST(R10), DI    // dst row base
	MOVQ B_SRC0(R10), SI   // src0 running cursor
	MOVQ B_SRC1(R10), DX   // src1 running cursor
	MOVQ B_DSTSTR(R10), R8 // dst stride
	MOVQ B_HEIGHT(R10), R11
	MOVQ B_FWD(R10), AX
	MOVL AX, X13
	VPBROADCASTD X13, Y13  // fwd
	MOVQ B_BCK(R10), AX
	MOVL AX, X12
	VPBROADCASTD X12, Y12  // bck
	MOVQ B_RNDOFF(R10), AX
	MOVL AX, X11
	VPBROADCASTD X11, Y11  // roundOffset
	MOVL $8, AX            // 1<<(roundBits-1) = 8
	MOVL AX, X10
	VPBROADCASTD X10, Y10
	MOVQ B_WIDTH(R10), R10 // width (R10 reused after ctx loaded)

blendRowLoop:
	TESTQ R11, R11
	JZ    blendDone
	MOVQ  DI, R12
	MOVQ  R10, CX

blendColLoop:
	VPMOVZXWD (SI), Y0
	VPMULLD Y13, Y0, Y0
	VPMOVZXWD (DX), Y1
	VPMULLD Y12, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPSRAD $4, Y0, Y0      // >> DIST_PRECISION_BITS
	VPSUBD Y11, Y0, Y0     // - roundOffset
	VPADDD Y10, Y0, Y0     // + 8
	VPSRAD $4, Y0, Y0      // >> roundBits
	VPACKSSDW Y0, Y0, Y0
	VPERMQ $0xD8, Y0, Y0
	VPACKUSWB Y0, Y0, Y0
	VMOVQ X0, (R12)
	ADDQ $16, SI
	ADDQ $16, DX
	ADDQ $8, R12
	SUBQ $8, CX
	JNZ  blendColLoop

	ADDQ R8, DI
	DECQ R11
	JMP  blendRowLoop

blendDone:
	VZEROUPPER
	RET

// func compoundCopy8AVX2Asm(ctx *compoundConvAVX2Ctx)
//
// out[x] = uint16(ref[x]*scale + roundOffset).
TEXT ·compoundCopy8AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ DST(R10), DI      // CONV_BUF running cursor (contiguous)
	MOVQ REF(R10), SI      // ref row base
	MOVQ REFSTR(R10), R9
	MOVQ HEIGHT(R10), R11
	MOVQ SCALE(R10), AX
	MOVL AX, X13
	VPBROADCASTD X13, Y13
	MOVQ RNDOFF(R10), AX
	MOVL AX, X12
	VPBROADCASTD X12, Y12
	MOVQ WIDTH(R10), R10

copyRowLoop:
	TESTQ R11, R11
	JZ    copyDone
	MOVQ  SI, R13
	MOVQ  R10, CX

copyColLoop:
	VPMOVZXBD (R13), Y0
	VPMULLD Y13, Y0, Y0
	VPADDD Y12, Y0, Y0
	VPSHUFB cmpLowWord<>(SB), Y0, Y0
	VPERMQ $0xD8, Y0, Y0
	VMOVDQU X0, (DI)
	ADDQ $8, R13
	ADDQ $16, DI
	SUBQ $8, CX
	JNZ  copyColLoop

	ADDQ R9, SI
	DECQ R11
	JMP  copyRowLoop

copyDone:
	VZEROUPPER
	RET

// func compoundX8AVX2Asm(ctx *compoundConvAVX2Ctx)
//
// out[x] = uint16(roundPowerOfTwo3(sum) + roundOffset), sum = 8-tap horizontal
// MAC of contiguous bytes. The ref pointer addresses the first tap (refX-fo).
TEXT ·compoundX8AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ DST(R10), DI
	MOVQ REF(R10), SI
	MOVQ KERNEL(R10), BX
	MOVQ REFSTR(R10), R9
	MOVQ HEIGHT(R10), R11
	MOVQ RNDOFF(R10), AX
	MOVL AX, X12
	VPBROADCASTD X12, Y12  // roundOffset
	MOVL $4, AX            // 1<<(round0Bits-1) = 4
	MOVL AX, X14
	VPBROADCASTD X14, Y14
	MOVQ WIDTH(R10), R10

cxRowLoop:
	TESTQ R11, R11
	JZ    cxDone
	MOVQ  SI, R13
	MOVQ  R10, CX

cxColLoop:
	VPXOR Y0, Y0, Y0
	VPMOVZXBD (R13), Y1
	MOVWLSX (BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPMOVZXBD 1(R13), Y1
	MOVWLSX 2(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPMOVZXBD 2(R13), Y1
	MOVWLSX 4(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPMOVZXBD 3(R13), Y1
	MOVWLSX 6(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPMOVZXBD 4(R13), Y1
	MOVWLSX 8(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPMOVZXBD 5(R13), Y1
	MOVWLSX 10(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPMOVZXBD 6(R13), Y1
	MOVWLSX 12(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPMOVZXBD 7(R13), Y1
	MOVWLSX 14(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0

	VPADDD Y14, Y0, Y0     // + 4
	VPSRAD $3, Y0, Y0      // >> 3
	VPADDD Y12, Y0, Y0     // + roundOffset
	VPSHUFB cmpLowWord<>(SB), Y0, Y0
	VPERMQ $0xD8, Y0, Y0
	VMOVDQU X0, (DI)

	ADDQ $8, R13
	ADDQ $16, DI
	SUBQ $8, CX
	JNZ  cxColLoop

	ADDQ R9, SI
	DECQ R11
	JMP  cxRowLoop

cxDone:
	VZEROUPPER
	RET

// func compoundY8AVX2Asm(ctx *compoundConvAVX2Ctx)
//
// out[x] = uint16(roundPowerOfTwo7(sum*scale) + roundOffset), sum = 8-tap
// vertical MAC of byte rows. The ref pointer addresses the first tap (refY-fo).
TEXT ·compoundY8AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ DST(R10), DI
	MOVQ REF(R10), SI
	MOVQ KERNEL(R10), BX
	MOVQ REFSTR(R10), R9
	MOVQ HEIGHT(R10), R11
	MOVQ SCALE(R10), AX
	MOVL AX, X13
	VPBROADCASTD X13, Y13  // scale
	MOVQ RNDOFF(R10), AX
	MOVL AX, X12
	VPBROADCASTD X12, Y12  // roundOffset
	MOVL $64, AX           // 1<<(compoundRound1Bits-1) = 64
	MOVL AX, X15
	VPBROADCASTD X15, Y15
	MOVQ WIDTH(R10), R10

cyRowLoop:
	TESTQ R11, R11
	JZ    cyDone
	MOVQ  SI, R13
	MOVQ  R10, CX

cyColLoop:
	VPXOR Y0, Y0, Y0
	MOVQ  R13, DX
	VPMOVZXBD (DX), Y1
	MOVWLSX (BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R9, DX
	VPMOVZXBD (DX), Y1
	MOVWLSX 2(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R9, DX
	VPMOVZXBD (DX), Y1
	MOVWLSX 4(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R9, DX
	VPMOVZXBD (DX), Y1
	MOVWLSX 6(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R9, DX
	VPMOVZXBD (DX), Y1
	MOVWLSX 8(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R9, DX
	VPMOVZXBD (DX), Y1
	MOVWLSX 10(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R9, DX
	VPMOVZXBD (DX), Y1
	MOVWLSX 12(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R9, DX
	VPMOVZXBD (DX), Y1
	MOVWLSX 14(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0

	VPMULLD Y13, Y0, Y0    // * scale
	VPADDD Y15, Y0, Y0     // + 64
	VPSRAD $7, Y0, Y0      // >> 7
	VPADDD Y12, Y0, Y0     // + roundOffset
	VPSHUFB cmpLowWord<>(SB), Y0, Y0
	VPERMQ $0xD8, Y0, Y0
	VMOVDQU X0, (DI)

	ADDQ $8, R13
	ADDQ $16, DI
	SUBQ $8, CX
	JNZ  cyColLoop

	ADDQ R9, SI
	DECQ R11
	JMP  cyRowLoop

cyDone:
	VZEROUPPER
	RET

// func compound2D8AVX2Asm(ctx *compoundConvAVX2Ctx)
//
// Pass 1 (horizontal, imH=height+7 rows): 8-tap MAC over bytes, + xBias, round
// by round0Bits=3, store the full int32 into the intermediate `im`.
// Pass 2 (vertical, height rows): 8-tap MAC over int32 `im` rows, + yBias, round
// by compoundRound1Bits=7, then truncating uint16 narrow into the CONV_BUF.
TEXT ·compound2D8AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ REF(R10), SI      // ref tap-window base
	MOVQ XKERN(R10), R14   // horizontal kernel
	MOVQ REFSTR(R10), R9   // ref stride
	MOVQ IM(R10), R15      // im base
	MOVQ IMSTR(R10), R12   // im stride (int32 elements)
	SHLQ $2, R12           // -> bytes
	MOVQ HEIGHT(R10), R11  // height
	MOVQ XBIAS(R10), AX
	MOVL AX, X13
	VPBROADCASTD X13, Y13  // xBias
	MOVL $4, AX            // round0 bias (1<<2)
	MOVL AX, X12
	VPBROADCASTD X12, Y12
	MOVQ WIDTH(R10), R10   // width

	// imH = height + 7
	MOVQ R11, DI
	ADDQ $7, DI

h2RowLoop:
	TESTQ DI, DI
	JZ    h2Done
	MOVQ  SI, DX           // ref column cursor
	MOVQ  R15, BX          // im column cursor
	MOVQ  R10, CX

h2ColLoop:
	VPXOR Y0, Y0, Y0
	VPMOVZXBD (DX), Y1
	MOVWLSX (R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPMOVZXBD 1(DX), Y1
	MOVWLSX 2(R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPMOVZXBD 2(DX), Y1
	MOVWLSX 4(R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPMOVZXBD 3(DX), Y1
	MOVWLSX 6(R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPMOVZXBD 4(DX), Y1
	MOVWLSX 8(R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPMOVZXBD 5(DX), Y1
	MOVWLSX 10(R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPMOVZXBD 6(DX), Y1
	MOVWLSX 12(R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	VPMOVZXBD 7(DX), Y1
	MOVWLSX 14(R14), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0

	VPADDD Y13, Y0, Y0     // + xBias
	VPADDD Y12, Y0, Y0     // + round0 bias (4)
	VPSRAD $3, Y0, Y0      // >> 3
	VMOVDQU Y0, (BX)       // store 8 int32

	ADDQ $8, DX
	ADDQ $32, BX
	SUBQ $8, CX
	JNZ  h2ColLoop

	ADDQ R9, SI
	ADDQ R12, R15
	DECQ DI
	JMP  h2RowLoop

h2Done:
	// ---- Pass 2 ----
	MOVQ ctx+0(FP), AX
	MOVQ DST(AX), DI       // CONV_BUF running cursor
	MOVQ IM(AX), R15       // im base (reload)
	MOVQ KERNEL(AX), BX    // vertical kernel
	MOVQ YBIAS(AX), AX
	MOVL AX, X13
	VPBROADCASTD X13, Y13  // yBias
	MOVL $64, AX           // round1 bias (1<<(compoundRound1Bits-1)) = 64
	MOVL AX, X12
	VPBROADCASTD X12, Y12

v2RowLoop:
	TESTQ R11, R11
	JZ    v2Done
	MOVQ  R15, SI          // im tap-window base for this output row
	MOVQ  R10, CX

v2ColLoop:
	VPXOR Y0, Y0, Y0
	MOVQ  SI, DX
	VMOVDQU (DX), Y1
	MOVWLSX (BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R12, DX
	VMOVDQU (DX), Y1
	MOVWLSX 2(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R12, DX
	VMOVDQU (DX), Y1
	MOVWLSX 4(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R12, DX
	VMOVDQU (DX), Y1
	MOVWLSX 6(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R12, DX
	VMOVDQU (DX), Y1
	MOVWLSX 8(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R12, DX
	VMOVDQU (DX), Y1
	MOVWLSX 10(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R12, DX
	VMOVDQU (DX), Y1
	MOVWLSX 12(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R12, DX
	VMOVDQU (DX), Y1
	MOVWLSX 14(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0

	VPADDD Y13, Y0, Y0     // + yBias
	VPADDD Y12, Y0, Y0     // + 64
	VPSRAD $7, Y0, Y0      // >> 7
	VPSHUFB cmpLowWord<>(SB), Y0, Y0
	VPERMQ $0xD8, Y0, Y0
	VMOVDQU X0, (DI)

	ADDQ $32, SI           // im tap-window += 8 int32
	ADDQ $16, DI           // dst += 8 uint16
	SUBQ $8, CX
	JNZ  v2ColLoop

	ADDQ R12, R15
	DECQ R11
	JMP  v2RowLoop

v2Done:
	VZEROUPPER
	RET
