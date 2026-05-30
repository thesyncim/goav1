// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 high-bit-depth (10/12-bit) convolve kernels, bit-exact with the
// *HighBD*PureGo references in vector.go. Samples are little-endian uint16.
//
// Each kernel processes 4 output columns per inner iteration into one X register
// holding 4 int32 accumulators (lane i == column i). uint16 samples are
// zero-extended to int32 (VPMOVZXWD); int16 intermediates are sign-extended
// (VPMOVSXWD). Each tap multiplies by an int32-broadcast tap (VPMULLD) and
// accumulates (VPADDD). int32 cannot overflow for the 8-tap HBD sums.
//
// roundPowerOfTwo(v,b) = (v + (1<<(b-1))) >> b is VPADDD bias + VPSRAD b. The
// round bias is supplied via the context (shifts are bit-depth dependent). When
// the shift count is 0 the pure-Go path is a no-op; the asm guards b==0 and
// skips the add+shift to match exactly.
//
// clipPixelHighBD(v,max) is VPMAXSD against 0 then VPMINSD against max; the
// narrow to uint16 is VPACKUSDW (already in [0,max] so saturation is a no-op),
// and VMOVQ stores the 4 contiguous uint16 outputs.

// ctx field offsets (see convolveHighBDAVX2Ctx in convolve_highbd_avx2_amd64.go).
#define DST     0
#define REF     8
#define KERNEL  16
#define XKERN   24
#define DSTSTR  32
#define REFSTR  40
#define WIDTH   48
#define HEIGHT  56
#define IM      64
#define IMSTR   72
#define ROUND0  80
#define ROUND1  88
#define BITS    96
#define RNDOFF  104
#define XBIAS   112
#define YBIAS   120
#define MAXVAL  128

// func convolveXHighBDAVX2Asm(ctx *convolveHighBDAVX2Ctx)
//
// Horizontal HBD 8-tap convolve: roundPowerOfTwo by round0, then by
// bits=FILTER_BITS-round0, then clip to [0,max]. The ref pointer addresses the
// first tap sample (col refX-fo). round0/round1(bits) come from the context.
TEXT ·convolveXHighBDAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ DST(R10), DI
	MOVQ REF(R10), SI
	MOVQ KERNEL(R10), BX
	MOVQ DSTSTR(R10), R8
	MOVQ REFSTR(R10), R9
	MOVQ HEIGHT(R10), R11
	MOVQ ROUND0(R10), R14   // round0 shift
	MOVQ ROUND1(R10), R15   // bits shift
	MOVQ MAXVAL(R10), AX
	MOVL AX, X13
	VPBROADCASTD X13, X13   // max broadcast
	MOVQ WIDTH(R10), R10

	// round0 bias = 1<<(round0-1); round1 bias = 1<<(bits-1).
	MOVL $1, AX
	MOVQ R14, CX
	DECQ CX
	SHLL CX, AX
	MOVL AX, X14
	VPBROADCASTD X14, X14   // round0 bias
	MOVL $1, AX
	MOVQ R15, CX
	DECQ CX
	SHLL CX, AX
	MOVL AX, X15
	VPBROADCASTD X15, X15   // round1 bias

	VPXOR X12, X12, X12     // zero (clip lower bound)

hbxRowLoop:
	TESTQ R11, R11
	JZ    hbxDone
	MOVQ  DI, R12
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

	// round0 (>= 1 for all HBD): + bias, >> round0
	VPADDD X14, X0, X0
	VMOVD R14, X3
	VPSRAD X3, X0, X0
	// round1 (bits): + bias, >> bits  (bits = FILTER_BITS-round0, always >=2)
	VPADDD X15, X0, X0
	VMOVD R15, X3
	VPSRAD X3, X0, X0

	// clip to [0,max], narrow to u16, store 4 samples.
	VPMAXSD X12, X0, X0
	VPMINSD X13, X0, X0
	VPACKUSDW X0, X0, X0
	VMOVQ X0, (R12)

	ADDQ $8, R12           // 4 u16 = 8 bytes
	ADDQ $8, R13           // 4 samples window slide
	SUBQ $4, CX
	JNZ  hbxColLoop

	ADDQ R8, DI
	ADDQ R9, SI
	DECQ R11
	JMP  hbxRowLoop

hbxDone:
	VZEROUPPER
	RET

// func convolveYHighBDAVX2Asm(ctx *convolveHighBDAVX2Ctx)
//
// Vertical HBD 8-tap convolve: single roundPowerOfTwo by FILTER_BITS (passed in
// round0), then clip to [0,max]. The ref pointer addresses the first tap row
// (refY-fo).
TEXT ·convolveYHighBDAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ DST(R10), DI
	MOVQ REF(R10), SI
	MOVQ KERNEL(R10), BX
	MOVQ DSTSTR(R10), R8
	MOVQ REFSTR(R10), R9
	MOVQ HEIGHT(R10), R11
	MOVQ ROUND0(R10), R14   // = FILTER_BITS (7)
	MOVQ MAXVAL(R10), AX
	MOVL AX, X13
	VPBROADCASTD X13, X13
	MOVQ WIDTH(R10), R10

	MOVL $1, AX
	MOVQ R14, CX
	DECQ CX
	SHLL CX, AX
	MOVL AX, X14
	VPBROADCASTD X14, X14   // round bias = 1<<(FILTER_BITS-1) = 64
	VMOVD R14, X5           // shift amount

	VPXOR X12, X12, X12

hbyRowLoop:
	TESTQ R11, R11
	JZ    hbyDone
	MOVQ  DI, R12
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

	VPADDD X14, X0, X0
	VPSRAD X5, X0, X0
	VPMAXSD X12, X0, X0
	VPMINSD X13, X0, X0
	VPACKUSDW X0, X0, X0
	VMOVQ X0, (R12)

	ADDQ $8, R12
	ADDQ $8, R13
	SUBQ $4, CX
	JNZ  hbyColLoop

	ADDQ R8, DI
	ADDQ R9, SI
	DECQ R11
	JMP  hbyRowLoop

hbyDone:
	VZEROUPPER
	RET

// func convolve2DHighBDAVX2Asm(ctx *convolveHighBDAVX2Ctx)
//
// HBD 2D convolve, bit-exact with convolve2DHighBDPureGo.
//
// Pass 1 (horizontal): for each of imH=height+7 rows, 8-tap MAC over 4 columns,
// add xBias, round by round0, store int32 into the int32 intermediate `im`.
// Pass 2 (vertical): for each of height rows, 8-tap MAC over 4 int32 `im` rows,
// add yBias, round by round1, subtract roundOffset, round by bits, clip [0,max].
TEXT ·convolve2DHighBDAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ REF(R10), SI      // ref tap-window base
	MOVQ XKERN(R10), BX    // horizontal kernel
	MOVQ REFSTR(R10), R9   // ref stride
	MOVQ IM(R10), R15      // im base
	MOVQ IMSTR(R10), R12   // im stride (int32 elements)
	SHLQ $2, R12           // -> bytes (4 bytes / int32)
	MOVQ HEIGHT(R10), R11
	MOVQ ROUND0(R10), R14
	MOVQ XBIAS(R10), AX
	MOVL AX, X13
	VPBROADCASTD X13, X13  // xBias
	MOVQ WIDTH(R10), R10

	// imH = height + 7
	MOVQ R11, R13
	ADDQ $7, R13

	MOVL $1, AX            // round0 bias
	MOVQ R14, CX
	DECQ CX
	SHLL CX, AX
	MOVL AX, X14
	VPBROADCASTD X14, X14
	VMOVD R14, X5          // round0 shift amount

	// SI = ref row cursor, R15 = im row cursor.
hb2RowLoop:
	TESTQ R13, R13
	JZ    hb2Done
	MOVQ  SI, DX           // ref column cursor
	MOVQ  R15, DI          // im column cursor (DI free until pass 2 loads DST)
	MOVQ  R10, CX

hb2ColLoop:
	VPXOR X0, X0, X0
	VPMOVZXWD (DX), X1
	MOVWLSX (BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 2(DX), X1
	MOVWLSX 2(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 4(DX), X1
	MOVWLSX 4(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 6(DX), X1
	MOVWLSX 6(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 8(DX), X1
	MOVWLSX 8(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 10(DX), X1
	MOVWLSX 10(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 12(DX), X1
	MOVWLSX 12(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0
	VPMOVZXWD 14(DX), X1
	MOVWLSX 14(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, X2
	VPMULLD X2, X1, X1
	VPADDD X1, X0, X0

	VPADDD X13, X0, X0     // + xBias
	VPADDD X14, X0, X0     // + round0 bias
	VPSRAD X5, X0, X0      // >> round0
	VMOVDQU X0, (DI)       // store 4 int32 (16 bytes)

	ADDQ $8, DX            // ref += 4 samples (8 bytes)
	ADDQ $16, DI           // im += 4 int32
	SUBQ $4, CX
	JNZ  hb2ColLoop

	ADDQ R9, SI
	ADDQ R12, R15
	DECQ R13
	JMP  hb2RowLoop

hb2Done:
	// ---- Pass 2: vertical over int32 intermediate ----
	MOVQ ctx+0(FP), R10
	MOVQ DST(R10), DI
	MOVQ KERNEL(R10), BX   // vertical kernel
	MOVQ DSTSTR(R10), R8
	MOVQ IM(R10), R15      // reload im base
	MOVQ ROUND1(R10), R14  // round1 shift
	MOVQ BITS(R10), AX

	// yBias / round1 bias / roundOffset / bits-bias / max constants.
	MOVQ YBIAS(R10), AX
	MOVL AX, X13
	VPBROADCASTD X13, X13  // yBias
	MOVL $1, AX            // round1 bias = 1<<(round1-1)
	MOVQ R14, CX
	DECQ CX
	SHLL CX, AX
	MOVL AX, X14
	VPBROADCASTD X14, X14
	VMOVD R14, X5          // round1 shift amount
	MOVQ RNDOFF(R10), AX
	MOVL AX, X11
	VPBROADCASTD X11, X11  // roundOffset
	MOVQ MAXVAL(R10), AX
	MOVL AX, X10
	VPBROADCASTD X10, X10  // max
	VPXOR X9, X9, X9       // zero

	// bits-stage bias + shift (bits = 2*FILTER_BITS-round0-round1; >=0).
	MOVQ BITS(R10), R13    // bits shift
	MOVL $0, AX
	TESTQ R13, R13
	JZ    hb2BitsBiasZero
	MOVL $1, AX
	MOVQ R13, CX
	DECQ CX
	SHLL CX, AX
hb2BitsBiasZero:
	MOVL AX, X8
	VPBROADCASTD X8, X8    // bits bias (0 when bits==0)
	VMOVD R13, X7          // bits shift amount

	MOVQ HEIGHT(R10), R11
	MOVQ IMSTR(R10), R12   // reload im stride (int32 elements)
	SHLQ $2, R12           // -> bytes
	MOVQ WIDTH(R10), R10

hbv2RowLoop:
	TESTQ R11, R11
	JZ    hbv2Done
	MOVQ  DI, R13          // dst column cursor
	MOVQ  R15, SI          // im tap-window base for this output row
	MOVQ  R10, CX

hbv2ColLoop:
	VPXOR X0, X0, X0
	MOVQ  SI, DX

	VMOVDQU (DX), X1       // 4 int32 already
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
	VPADDD X14, X0, X0     // + round1 bias
	VPSRAD X5, X0, X0      // >> round1
	VPSUBD X11, X0, X0     // - roundOffset
	VPADDD X8, X0, X0      // + bits bias
	VPSRAD X7, X0, X0      // >> bits
	VPMAXSD X9, X0, X0
	VPMINSD X10, X0, X0
	VPACKUSDW X0, X0, X0
	VMOVQ X0, (R13)

	ADDQ $8, R13           // 4 u16
	ADDQ $16, SI           // im tap-window += 4 int32
	SUBQ $4, CX
	JNZ  hbv2ColLoop

	ADDQ R8, DI
	ADDQ R12, R15
	DECQ R11
	JMP  hbv2RowLoop

hbv2Done:
	VZEROUPPER
	RET
