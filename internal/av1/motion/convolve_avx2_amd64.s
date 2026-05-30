// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 8-bit convolve kernels, bit-exact with the *PureGo references in
// vector.go. Each kernel processes 8 output columns per inner iteration into a
// single Y register holding 8 int32 accumulators (lane i == column i, natural
// order, because VPMOVZXBD zero-extends 8 input bytes to lanes 0..7 in order).
//
// Per-tap MAC: 8 source bytes are zero-extended to int32 (VPMOVZXBD), multiplied
// by the int32-broadcast tap (VPMULLD), and added (VPADDD). int32 cannot
// overflow for the 8-tap subpel sums, so the accumulation matches the pure-Go
// int sum exactly.
//
// roundPowerOfTwo(v,b) = (v + (1<<(b-1))) >> b is reproduced by VPADDD of the
// bias followed by an arithmetic VPSRAD by b. clipPixel is VPACKSSDW (s32->s16
// saturate) then VPACKUSWB (s16->u8 saturate), i.e. clamp to [0,255]. AVX2 pack
// instructions act per 128-bit lane, so a VPERMQ $0xD8 first brings the two
// valid 64-bit halves of the 8-lane result into the low 128 bits; the low-128
// VMOVQ then stores the 8 contiguous output bytes.
//
// The 2D horizontal pass stores a *truncating* int16 narrow (Go int16(), not
// saturating). It is reproduced by VPSHUFB-gathering the low halfword of each
// int32 lane via shufLowWord<> then a VPERMQ $0xD8 + 16-byte VMOVDQU store.

// shufLowWord gathers, within each 128-bit lane, the low 16 bits of the four
// int32 words into the low 8 bytes (bytes {0,1,4,5,8,9,12,13}); the upper 8
// bytes are zeroed (0x80 selects zero in VPSHUFB). Same pattern in both lanes.
GLOBL shufLowWord<>(SB), RODATA|NOPTR, $32
DATA shufLowWord<>+0(SB)/8, $0x0d0c090805040100
DATA shufLowWord<>+8(SB)/8, $0x8080808080808080
DATA shufLowWord<>+16(SB)/8, $0x0d0c090805040100
DATA shufLowWord<>+24(SB)/8, $0x8080808080808080

// ctx field offsets (see convolveAVX2Ctx in convolve_avx2_amd64.go).
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

// func convolveX8AVX2Asm(ctx *convolveAVX2Ctx)
//
// Horizontal 8-tap convolve. For each row and group of 8 columns, taps 0..7
// read 8 contiguous bytes at increasing offsets from the column base; the
// two-stage round (round0Bits=3 then FILTER_BITS-round0Bits=4) and the [0,255]
// clip follow. The ref pointer addresses the first tap sample (refX-fo).
TEXT ·convolveX8AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ DST(R10), DI      // dst row base
	MOVQ REF(R10), SI      // ref row base (col refX-fo)
	MOVQ KERNEL(R10), BX   // kernel ptr
	MOVQ DSTSTR(R10), R8   // dst stride
	MOVQ REFSTR(R10), R9   // ref stride
	MOVQ HEIGHT(R10), R11  // height
	MOVQ WIDTH(R10), R10   // width (R10 reused after ctx loaded)

	MOVL $4, AX            // 1<<(round0Bits-1) = 4
	MOVL AX, X14
	VPBROADCASTD X14, Y14
	MOVL $8, AX            // 1<<(bits-1) = 8
	MOVL AX, X15
	VPBROADCASTD X15, Y15

xRowLoop:
	TESTQ R11, R11
	JZ    xDone
	MOVQ  DI, R12          // dst column cursor
	MOVQ  SI, R13          // ref column cursor
	MOVQ  R10, CX          // remaining columns

xColLoop:
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

	VPADDD Y14, Y0, Y0     // round0: + 4
	VPSRAD $3, Y0, Y0      //         >> 3
	VPADDD Y15, Y0, Y0     // round1: + 8
	VPSRAD $4, Y0, Y0      //         >> 4

	VPACKSSDW Y0, Y0, Y0
	VPERMQ $0xD8, Y0, Y0
	VPACKUSWB Y0, Y0, Y0
	VMOVQ X0, (R12)

	ADDQ $8, R12
	ADDQ $8, R13
	SUBQ $8, CX
	JNZ  xColLoop

	ADDQ R8, DI
	ADDQ R9, SI
	DECQ R11
	JMP  xRowLoop

xDone:
	VZEROUPPER
	RET

// func convolveY8AVX2Asm(ctx *convolveAVX2Ctx)
//
// Vertical 8-tap convolve. For each output row and group of 8 columns, taps
// 0..7 read 8 contiguous bytes from rows (base + k*refStride); a single
// rounding shift by FILTER_BITS=7 and the [0,255] clip follow. The ref pointer
// addresses the first tap row (refY-fo).
TEXT ·convolveY8AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ DST(R10), DI
	MOVQ REF(R10), SI
	MOVQ KERNEL(R10), BX
	MOVQ DSTSTR(R10), R8
	MOVQ REFSTR(R10), R9
	MOVQ HEIGHT(R10), R11
	MOVQ WIDTH(R10), R10

	MOVL $64, AX           // 1<<(FILTER_BITS-1) = 64
	MOVL AX, X15
	VPBROADCASTD X15, Y15

yRowLoop:
	TESTQ R11, R11
	JZ    yDone
	MOVQ  DI, R12
	MOVQ  SI, R13          // ref tap-row base for this output row
	MOVQ  R10, CX

yColLoop:
	VPXOR Y0, Y0, Y0
	MOVQ  R13, DX          // DX walks the 8 tap rows

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

	VPADDD Y15, Y0, Y0     // + 64
	VPSRAD $7, Y0, Y0      // >> 7

	VPACKSSDW Y0, Y0, Y0
	VPERMQ $0xD8, Y0, Y0
	VPACKUSWB Y0, Y0, Y0
	VMOVQ X0, (R12)

	ADDQ $8, R12
	ADDQ $8, R13
	SUBQ $8, CX
	JNZ  yColLoop

	ADDQ R8, DI
	ADDQ R9, SI
	DECQ R11
	JMP  yRowLoop

yDone:
	VZEROUPPER
	RET

// func convolve2D8AVX2Asm(ctx *convolveAVX2Ctx)
//
// Both-axes-fractional 8-bit 2D convolve, bit-exact with convolve2D8PureGo.
//
// Pass 1 (horizontal): for each of imH = height+7 rows, 8-tap MAC over 8
// columns, add xBias=16384, round by round0Bits=3, store the truncating int16
// narrow into the int16 intermediate `im`.
//
// Pass 2 (vertical): for each of height rows, 8-tap MAC over 8 int16 `im` rows,
// add yBias=524288, round by round1Bits=11, subtract roundOffset=384, then
// saturate to [0,255] (final round `bits` is 0).
TEXT ·convolve2D8AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), R10
	MOVQ DST(R10), DI      // dst row base
	MOVQ REF(R10), SI      // ref tap-window base
	MOVQ KERNEL(R10), BX   // vertical kernel ptr
	MOVQ XKERN(R10), R14   // horizontal kernel ptr
	MOVQ DSTSTR(R10), R8   // dst stride
	MOVQ REFSTR(R10), R9   // ref stride
	MOVQ IM(R10), R15      // im base ptr
	MOVQ IMSTR(R10), R12   // im row stride (int16 elements)
	SHLQ $1, R12           // -> bytes
	MOVQ HEIGHT(R10), R11  // height
	MOVQ WIDTH(R10), R10   // width

	// imH = height + 7
	MOVQ R11, R13
	ADDQ $7, R13

	MOVL $16384, AX        // xBias
	MOVL AX, X13
	VPBROADCASTD X13, Y13
	MOVL $4, AX            // round0 bias (1<<2)
	MOVL AX, X12
	VPBROADCASTD X12, Y12

	// ---- Pass 1: horizontal -> int16 intermediate ----
	// SI = ref row cursor, R15 = im row cursor (both advance per row).
h2RowLoop:
	TESTQ R13, R13
	JZ    h2Done
	MOVQ  SI, DX           // ref column cursor
	MOVQ  R15, BX          // im column cursor (BX is free in pass 1)
	MOVQ  R10, CX          // remaining columns

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

	// truncating int16 narrow (Go int16()): gather low words, store 8 int16.
	VPSHUFB shufLowWord<>(SB), Y0, Y0
	VPERMQ $0xD8, Y0, Y0
	VMOVDQU X0, (BX)       // 16 bytes = 8 int16

	ADDQ $8, DX            // ref += 8 columns
	ADDQ $16, BX           // im  += 8 int16
	SUBQ $8, CX
	JNZ  h2ColLoop

	ADDQ R9, SI            // ref row += refStride
	ADDQ R12, R15          // im row += imStride bytes
	DECQ R13
	JMP  h2RowLoop

h2Done:
	// ---- Pass 2: vertical over int16 intermediate ----
	// Reload im base into R15 (pass 1 advanced it) and vertical kernel in BX.
	MOVQ ctx+0(FP), AX
	MOVQ IM(AX), R15       // im base ptr
	MOVQ KERNEL(AX), BX    // vertical kernel ptr

	MOVL $524288, AX       // yBias
	MOVL AX, X13
	VPBROADCASTD X13, Y13
	MOVL $1024, AX         // round1 bias (1<<(round1Bits-1)) = 1<<10
	MOVL AX, X12
	VPBROADCASTD X12, Y12
	MOVL $384, AX          // roundOffset
	MOVL AX, X11
	VPBROADCASTD X11, Y11

	// R15 = im row-window base for the current output row; advances by one im
	// row (R12 bytes) per output row. DI = dst row base.
v2RowLoop:
	TESTQ R11, R11
	JZ    v2Done
	MOVQ  DI, R13          // dst column cursor
	MOVQ  R15, SI          // im tap-window base for this output row
	MOVQ  R10, CX          // remaining columns

v2ColLoop:
	VPXOR Y0, Y0, Y0
	MOVQ  SI, DX           // DX walks the 8 tap rows (post-indexed by R12)

	VPMOVSXWD (DX), Y1
	MOVWLSX (BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R12, DX
	VPMOVSXWD (DX), Y1
	MOVWLSX 2(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R12, DX
	VPMOVSXWD (DX), Y1
	MOVWLSX 4(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R12, DX
	VPMOVSXWD (DX), Y1
	MOVWLSX 6(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R12, DX
	VPMOVSXWD (DX), Y1
	MOVWLSX 8(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R12, DX
	VPMOVSXWD (DX), Y1
	MOVWLSX 10(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R12, DX
	VPMOVSXWD (DX), Y1
	MOVWLSX 12(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0
	ADDQ R12, DX
	VPMOVSXWD (DX), Y1
	MOVWLSX 14(BX), AX
	MOVL AX, X2
	VPBROADCASTD X2, Y2
	VPMULLD Y2, Y1, Y1
	VPADDD Y1, Y0, Y0

	VPADDD Y13, Y0, Y0     // + yBias (524288)
	VPADDD Y12, Y0, Y0     // + round1 bias (1024)
	VPSRAD $11, Y0, Y0     // >> round1Bits (11)
	VPSUBD Y11, Y0, Y0     // - roundOffset (384)
	// final round `bits` is 0 -> no shift; clip to [0,255].
	VPACKSSDW Y0, Y0, Y0
	VPERMQ $0xD8, Y0, Y0
	VPACKUSWB Y0, Y0, Y0
	VMOVQ X0, (R13)

	ADDQ $8, R13           // dst += 8 pixels
	ADDQ $16, SI           // im tap-window += 8 int16 (16 bytes)
	SUBQ $8, CX
	JNZ  v2ColLoop

	ADDQ R8, DI            // dst row += dstStride
	ADDQ R12, R15          // im row += one im row
	DECQ R11
	JMP  v2RowLoop

v2Done:
	VZEROUPPER
	RET
