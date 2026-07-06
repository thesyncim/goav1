// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 8-bit-pixel Wiener separable passes (see wiener_u8_avx2_amd64.go).
// Structure is identical to wiener_avx2_amd64.s; only the sample loads/stores
// change domain:
//   VPMOVZXBD (mem), Yn        load 8 u8 and zero-extend to 8 s32 lanes
//   VPACKUSWB Xn, Xn, Xn       saturating s16 -> u8 narrow (already in range)
//   MOVQ      Xn, (mem)        store 8 u8 outputs

// wienerU8AVX2HorizCtx field offsets (see wiener_u8_avx2_amd64.go).
#define H_DST    0
#define H_SRC    8
#define H_SRCSTR 16
#define H_WIDTH  24
#define H_ROWS   32
#define H_TAPS   40
#define H_SEED   48
#define H_SHIFT  52
#define H_MAXCL  56

// func wienerHorizontalU8AVX2Asm(ctx *wienerU8AVX2HorizCtx)
//
// Horizontal 7-tap Wiener pass over uint8 source samples. For tap i the 8
// window samples at byte offset i are loaded with VPMOVZXBD (so the MAC sees
// the same s32 values the u16 kernel loads with VPMOVZXWD), then the folded
// rounding shift and the [0, maxClamp] clamp produce the u16 temp output.
TEXT ·wienerHorizontalU8AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ H_DST(AX), DI      // temp output base
	MOVQ H_SRC(AX), SI      // src tap-window base (row -3, col -3)
	MOVQ H_SRCSTR(AX), R8   // src stride (elements == bytes)
	MOVQ H_WIDTH(AX), R9    // width (multiple of 8)
	MOVQ H_ROWS(AX), R10    // rows
	MOVQ H_TAPS(AX), R11    // taps ptr (7 int32)

	LEAQ H_SEED(AX), R12
	VPBROADCASTD (R12), Y8  // seed broadcast (s32 lanes)
	LEAQ H_SHIFT(AX), R13
	VMOVD (R13), X9         // shift count (round0)
	LEAQ H_MAXCL(AX), R14
	VPBROADCASTD (R14), Y10 // maxClamp broadcast
	VPXOR Y11, Y11, Y11     // lower clamp bound 0

	// Preload the 7 broadcast taps into Y0..Y6 (loop-invariant per call).
	VPBROADCASTD 0(R11), Y0
	VPBROADCASTD 4(R11), Y1
	VPBROADCASTD 8(R11), Y2
	VPBROADCASTD 12(R11), Y3
	VPBROADCASTD 16(R11), Y4
	VPBROADCASTD 20(R11), Y5
	VPBROADCASTD 24(R11), Y6

	MOVQ R9, R15
	SHLQ $1, R15           // dst row stride in bytes (= width * 2)

hRowLoop:
	CMPQ R10, $0
	JEQ  hDone
	MOVQ DI, BX            // temp column cursor
	MOVQ SI, CX            // src window cursor
	MOVQ R9, DX            // remaining columns

hColLoop:
	VMOVDQA Y8, Y7         // seed
	VPMOVZXBD 0(CX), Y12
	VPMULLD Y0, Y12, Y12
	VPADDD  Y12, Y7, Y7
	VPMOVZXBD 1(CX), Y12
	VPMULLD Y1, Y12, Y12
	VPADDD  Y12, Y7, Y7
	VPMOVZXBD 2(CX), Y12
	VPMULLD Y2, Y12, Y12
	VPADDD  Y12, Y7, Y7
	VPMOVZXBD 3(CX), Y12
	VPMULLD Y3, Y12, Y12
	VPADDD  Y12, Y7, Y7
	VPMOVZXBD 4(CX), Y12
	VPMULLD Y4, Y12, Y12
	VPADDD  Y12, Y7, Y7
	VPMOVZXBD 5(CX), Y12
	VPMULLD Y5, Y12, Y12
	VPADDD  Y12, Y7, Y7
	VPMOVZXBD 6(CX), Y12
	VPMULLD Y6, Y12, Y12
	VPADDD  Y12, Y7, Y7

	VPSRAD  X9, Y7, Y7     // arith >> round0
	VPMAXSD Y11, Y7, Y7    // clamp lo 0
	VPMINSD Y10, Y7, Y7    // clamp hi maxClamp
	VPACKUSDW Y7, Y7, Y7   // s32 -> u16
	VPERMQ  $0xd8, Y7, Y7  // gather the 8 low words contiguously
	VMOVDQU X7, (BX)       // store 8 u16

	ADDQ $16, BX           // temp += 8 u16
	ADDQ $8, CX            // src window += 8 u8
	SUBQ $8, DX
	JNE  hColLoop

	ADDQ R15, DI           // temp += rowStride bytes
	ADDQ R8, SI            // src += srcStride bytes
	SUBQ $1, R10
	JNE  hRowLoop

hDone:
	VZEROUPPER
	RET

// wienerU8AVX2VertCtx field offsets (see wiener_u8_avx2_amd64.go).
#define V_DST    0
#define V_SRC    8
#define V_DSTSTR 16
#define V_SRCSTR 24
#define V_WIDTH  32
#define V_ROWS   40
#define V_TAPS   48
#define V_SEED   56
#define V_SHIFT  60
#define V_MAXCL  64

// func wienerVerticalU8AVX2Asm(ctx *wienerU8AVX2VertCtx)
//
// Vertical 7-tap Wiener pass over the u16 temp buffer, writing uint8 output.
// Identical to wienerVerticalAVX2Asm except the store: after the [0,255]
// clamp the 8 s32 lanes are packed to u16 then to u8 (both packs in-range)
// and stored as 8 bytes; the dst stride is already in bytes.
TEXT ·wienerVerticalU8AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ V_DST(AX), DI      // dst row base
	MOVQ V_SRC(AX), SI      // temp tap-window base (row 0)
	MOVQ V_DSTSTR(AX), R8   // dst stride (elements == bytes)
	MOVQ V_SRCSTR(AX), R9   // temp stride (elements)
	SHLQ $1, R9             // -> bytes
	MOVQ V_WIDTH(AX), R10   // width (multiple of 8)
	MOVQ V_ROWS(AX), R11    // rows = height
	MOVQ V_TAPS(AX), R12    // taps ptr (7 int32)

	LEAQ V_SEED(AX), R13
	VPBROADCASTD (R13), Y8  // seed broadcast
	LEAQ V_SHIFT(AX), R14
	VMOVD (R14), X9         // shift count (round1)
	LEAQ V_MAXCL(AX), R15
	VPBROADCASTD (R15), Y10 // max broadcast (255)
	VPXOR Y11, Y11, Y11     // lower clamp bound 0

	VPBROADCASTD 0(R12), Y0
	VPBROADCASTD 4(R12), Y1
	VPBROADCASTD 8(R12), Y2
	VPBROADCASTD 12(R12), Y3
	VPBROADCASTD 16(R12), Y4
	VPBROADCASTD 20(R12), Y5
	VPBROADCASTD 24(R12), Y6

vRowLoop:
	CMPQ R11, $0
	JEQ  vDone
	MOVQ DI, BX            // dst column cursor
	MOVQ SI, CX            // temp row-window base for this output row
	MOVQ R10, DX           // remaining columns

vColLoop:
	MOVQ CX, R13           // R13 walks the 7 tap rows
	VMOVDQA Y8, Y7         // seed

	VPMOVZXWD (R13), Y12
	VPMULLD Y0, Y12, Y12
	VPADDD  Y12, Y7, Y7
	ADDQ R9, R13
	VPMOVZXWD (R13), Y12
	VPMULLD Y1, Y12, Y12
	VPADDD  Y12, Y7, Y7
	ADDQ R9, R13
	VPMOVZXWD (R13), Y12
	VPMULLD Y2, Y12, Y12
	VPADDD  Y12, Y7, Y7
	ADDQ R9, R13
	VPMOVZXWD (R13), Y12
	VPMULLD Y3, Y12, Y12
	VPADDD  Y12, Y7, Y7
	ADDQ R9, R13
	VPMOVZXWD (R13), Y12
	VPMULLD Y4, Y12, Y12
	VPADDD  Y12, Y7, Y7
	ADDQ R9, R13
	VPMOVZXWD (R13), Y12
	VPMULLD Y5, Y12, Y12
	VPADDD  Y12, Y7, Y7
	ADDQ R9, R13
	VPMOVZXWD (R13), Y12
	VPMULLD Y6, Y12, Y12
	VPADDD  Y12, Y7, Y7

	VPSRAD  X9, Y7, Y7     // arith >> round1
	VPMAXSD Y11, Y7, Y7    // clamp lo 0
	VPMINSD Y10, Y7, Y7    // clamp hi 255
	VPACKUSDW Y7, Y7, Y7   // s32 -> u16 (in range)
	VPERMQ  $0xd8, Y7, Y7  // gather the 8 low words contiguously
	VPACKUSWB X7, X7, X7   // u16 -> u8 (in range)
	MOVQ    X7, (BX)       // store 8 u8

	ADDQ $8, BX            // dst += 8 u8
	ADDQ $16, CX           // temp window += 8 u16
	SUBQ $8, DX
	JNE  vColLoop

	ADDQ R8, DI            // dst += dstStride bytes
	ADDQ R9, SI            // temp += tempStride bytes
	SUBQ $1, R11
	JNE  vRowLoop

vDone:
	VZEROUPPER
	RET
