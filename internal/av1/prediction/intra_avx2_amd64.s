// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// AVX2 intra static predictors, CfL apply and DC edge sum. The kernels are
// bit-exact with the *PureGo references in intra_static.go and
// intra_kernels.go; the Go wrappers in intra_avx2_amd64.go restrict them to
// the 8-bit shapes they handle (PAETH: width%16==0; the rest: width%16==0,
// stepping 8 columns/iteration).
//
// PAETH processes 16 int16 columns per iteration with VPABSW/VPCMPGTW/blend.
// The SMOOTH family and CfL widen 8 columns to u32/s32 lanes (VPMOVZXWD /
// VPMOVSXWD) so the weighted products and rounding shift run without overflow,
// then pack back with VPACKUSDW/VPACKUSWB.

//go:build amd64 && !purego

#include "textflag.h"

// ---------------------------------------------------------------------------
// PAETH
// ---------------------------------------------------------------------------
#define P_DST 0
#define P_ABOVE 8
#define P_LEFT 16
#define P_DSTSTR 24
#define P_WIDTH 32
#define P_HEIGHT 40
#define P_ABOVELEFT 48

// func predictPaethAVX2Asm(ctx *paethAVX2Ctx)
TEXT ·predictPaethAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ P_DST(AX), DI
	MOVQ P_ABOVE(AX), SI
	MOVQ P_LEFT(AX), BX
	MOVQ P_DSTSTR(AX), R8
	MOVQ P_WIDTH(AX), R9
	MOVQ P_HEIGHT(AX), R10
	MOVQ P_ABOVELEFT(AX), R11

	MOVQ         R11, X0
	VPBROADCASTW X0, Y0 // topLeft

	XORQ R12, R12 // row
paethRowLoop:
	CMPQ R12, R10
	JGE  paethDone
	MOVWLZX      (BX)(R12*2), R13
	MOVQ         R13, X1
	VPBROADCASTW X1, Y1 // left[row]

	MOVQ DI, R14
	MOVQ SI, R15
	XORQ CX, CX
paethColLoop:
	VMOVDQU   (R15), Y2     // top
	VPADDW    Y1, Y2, Y3
	VPSUBW    Y0, Y3, Y3    // base = top+left-tl
	VPSUBW    Y1, Y3, Y4
	VPABSW    Y4, Y4        // pLeft
	VPSUBW    Y2, Y3, Y5
	VPABSW    Y5, Y5        // pTop
	VPSUBW    Y0, Y3, Y6
	VPABSW    Y6, Y6        // pTopLeft
	VPCMPGTW  Y5, Y4, Y7    // pLeft>pTop
	VPCMPGTW  Y6, Y4, Y8    // pLeft>pTL
	VPOR      Y8, Y7, Y7    // NOT(maskLeft)
	VPCMPGTW  Y6, Y5, Y9    // pTop>pTL
	VPBLENDVB Y9, Y0, Y2, Y10 // (pTop>pTL)?tl:top
	VPBLENDVB Y7, Y10, Y1, Y11 // NOT(maskLeft)?selAbove:left
	VPACKUSWB Y11, Y11, Y11
	VPERMQ    $0xD8, Y11, Y11
	VMOVDQU   X11, (R14)
	ADDQ $32, R15
	ADDQ $16, R14
	ADDQ $16, CX
	CMPQ CX, R9
	JLT  paethColLoop

	ADDQ R8, DI
	INCQ R12
	JMP  paethRowLoop
paethDone:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// SMOOTH (full)
// ---------------------------------------------------------------------------
#define S_DST 0
#define S_ABOVE 8
#define S_LEFT 16
#define S_WEIGHTSW 24
#define S_WEIGHTSH 32
#define S_DSTSTR 40
#define S_WIDTH 48
#define S_HEIGHT 56
#define S_BELOW 64
#define S_RIGHT 72

// func predictSmoothAVX2Asm(ctx *smoothAVX2Ctx)
//
// pred = wH*above + (256-wH)*below + wW*left + (256-wW)*right, (sum+256)>>9.
// 8 columns/iteration in u32 lanes.
TEXT ·predictSmoothAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ S_DST(AX), DI
	MOVQ S_ABOVE(AX), SI
	MOVQ S_LEFT(AX), BX
	MOVQ S_WEIGHTSW(AX), DX
	MOVQ S_WEIGHTSH(AX), R8
	MOVQ S_DSTSTR(AX), R9
	MOVQ S_WIDTH(AX), R10
	MOVQ S_HEIGHT(AX), R11
	MOVQ S_BELOW(AX), R12
	MOVQ S_RIGHT(AX), R13

	MOVL         $256, R14
	MOVQ         R14, X15
	VPBROADCASTD X15, Y15 // 256 (u32)
	MOVQ         R12, X14
	VPBROADCASTD X14, Y14 // below
	MOVQ         R13, X13
	VPBROADCASTD X13, Y13 // right

	XORQ R12, R12 // row
smRowLoop:
	CMPQ R12, R11
	JGE  smDone
	MOVWLZX      (R8)(R12*2), CX
	MOVQ         CX, X0
	VPBROADCASTD X0, Y0          // wH
	VPSUBD       Y0, Y15, Y1     // 256-wH
	MOVWLZX      (BX)(R12*2), CX
	MOVQ         CX, X2
	VPBROADCASTD X2, Y2          // left[row]
	VPMULLD      Y14, Y1, Y3     // below*(256-wH) const for row

	MOVQ DI, R14   // dst cursor
	MOVQ SI, R15   // above cursor
	MOVQ DX, R13   // weightsW cursor
	XORQ CX, CX    // col
smColLoop:
	VPMOVZXWD (R15), Y5          // above
	VPMULLD   Y0, Y5, Y5         // above*wH
	VPADDD    Y3, Y5, Y5         // + below*(256-wH)
	VPMOVZXWD (R13), Y6          // wW
	VPSUBD    Y6, Y15, Y7        // 256-wW
	VPMULLD   Y2, Y6, Y6         // left*wW
	VPADDD    Y6, Y5, Y5
	VPMULLD   Y13, Y7, Y7        // right*(256-wW)
	VPADDD    Y7, Y5, Y5
	VPADDD    Y15, Y5, Y5        // +256
	VPSRLD    $9, Y5, Y5
	VPACKUSDW Y5, Y5, Y5
	VPERMQ    $0xD8, Y5, Y5
	VPACKUSWB Y5, Y5, Y5
	MOVQ      X5, (R14)
	ADDQ $16, R15
	ADDQ $16, R13
	ADDQ $8,  R14
	ADDQ $8,  CX
	CMPQ CX, R10
	JLT  smColLoop

	ADDQ R9, DI
	INCQ R12
	JMP  smRowLoop
smDone:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// SMOOTH_V / SMOOTH_H
// ---------------------------------------------------------------------------
#define D_DST 0
#define D_PRIMARY 8
#define D_WEIGHTS 16
#define D_DSTSTR 24
#define D_WIDTH 32
#define D_HEIGHT 40
#define D_SECONDARY 48
#define D_ISH 56

// func predictSmooth1DAVX2Asm(ctx *smooth1DAVX2Ctx)
//
// SMOOTH_V (isH==0): pred = w[row]*above[col] + (256-w[row])*below.
// SMOOTH_H (isH==1): pred = w[col]*left[row] + (256-w[col])*right.
// Both round (sum+128)>>8. 8 columns/iteration in u32 lanes.
TEXT ·predictSmooth1DAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ D_DST(AX), DI
	MOVQ D_PRIMARY(AX), SI
	MOVQ D_WEIGHTS(AX), BX
	MOVQ D_DSTSTR(AX), R9
	MOVQ D_WIDTH(AX), R10
	MOVQ D_HEIGHT(AX), R11
	MOVQ D_SECONDARY(AX), R12
	MOVQ D_ISH(AX), R13

	MOVL         $256, R14
	MOVQ         R14, X15
	VPBROADCASTD X15, Y15 // 256
	MOVQ         R12, X14
	VPBROADCASTD X14, Y14 // secondary
	MOVL         $128, R14
	MOVQ         R14, X12
	VPBROADCASTD X12, Y12 // 128 (rounding)

	CMPQ R13, $0
	JNE  shMode

	// ---- SMOOTH_V : above varies per col, w scalar per row ----
	XORQ R12, R12 // row
svRowLoop:
	CMPQ R12, R11
	JGE  d1Done
	MOVWLZX      (BX)(R12*2), CX
	MOVQ         CX, X0
	VPBROADCASTD X0, Y0          // w[row]
	VPSUBD       Y0, Y15, Y1
	VPMULLD      Y14, Y1, Y1     // below*(256-w) const per row

	MOVQ DI, R14
	MOVQ SI, R15
	XORQ CX, CX
svColLoop:
	VPMOVZXWD (R15), Y5
	VPMULLD   Y0, Y5, Y5      // above*w
	VPADDD    Y1, Y5, Y5      // + below*(256-w)
	VPADDD    Y12, Y5, Y5     // +128
	VPSRLD    $8, Y5, Y5
	VPACKUSDW Y5, Y5, Y5
	VPERMQ    $0xD8, Y5, Y5
	VPACKUSWB Y5, Y5, Y5
	MOVQ      X5, (R14)
	ADDQ $16, R15
	ADDQ $8,  R14
	ADDQ $8,  CX
	CMPQ CX, R10
	JLT  svColLoop

	ADDQ R9, DI
	INCQ R12
	JMP  svRowLoop

shMode:
	// ---- SMOOTH_H : w varies per col, left scalar per row ----
	XORQ R12, R12 // row
shRowLoop:
	CMPQ R12, R11
	JGE  d1Done
	MOVWLZX      (SI)(R12*2), CX
	MOVQ         CX, X0
	VPBROADCASTD X0, Y0          // left[row]

	MOVQ DI, R14
	MOVQ BX, R15  // weights cursor (col-indexed)
	XORQ CX, CX
shColLoop:
	VPMOVZXWD (R15), Y6          // w[col]
	VPSUBD    Y6, Y15, Y7        // 256-w
	VPMULLD   Y0, Y6, Y6         // left*w
	VPMULLD   Y14, Y7, Y7        // right*(256-w)
	VPADDD    Y7, Y6, Y6
	VPADDD    Y12, Y6, Y6        // +128
	VPSRLD    $8, Y6, Y6
	VPACKUSDW Y6, Y6, Y6
	VPERMQ    $0xD8, Y6, Y6
	VPACKUSWB Y6, Y6, Y6
	MOVQ      X6, (R14)
	ADDQ $16, R15
	ADDQ $8,  R14
	ADDQ $8,  CX
	CMPQ CX, R10
	JLT  shColLoop

	ADDQ R9, DI
	INCQ R12
	JMP  shRowLoop

d1Done:
	VZEROUPPER
	RET

// ---------------------------------------------------------------------------
// CfL apply (8-bit)
// ---------------------------------------------------------------------------
// func applyCFLRowAVX2Asm(dst *byte, ac *int16, alpha uint64, n uint64)
//
// scaled = sign(ac*alpha) * ((|ac*alpha|+32)>>6); dst = clamp(dst+scaled,0,255).
// alpha is the Q3 alpha sign-extended into a 32-bit value. 8 columns/iter.
TEXT ·applyCFLRowAVX2Asm(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ ac+8(FP), SI
	MOVQ alpha+16(FP), R8
	MOVQ n+24(FP), R9

	MOVQ         R8, X0
	VPBROADCASTD X0, Y0   // alpha (s32)
	MOVL         $32, R10
	MOVQ         R10, X1
	VPBROADCASTD X1, Y1   // 32
	VPXOR        Y8, Y8, Y8 // zero

	XORQ CX, CX
cflLoop:
	VPMOVSXWD (SI)(CX*2), Y2 // ac (8×s32)
	VPMULLD   Y0, Y2, Y2     // v = ac*alpha (s32)
	VPABSD    Y2, Y3         // |v|
	VPADDD    Y1, Y3, Y3     // |v|+32
	VPSRLD    $6, Y3, Y3     // (|v|+32)>>6
	VPSUBD    Y3, Y8, Y4     // -magnitude
	VPCMPGTD  Y2, Y8, Y5     // 0 > v  (v negative)
	VPBLENDVB Y5, Y4, Y3, Y3 // v<0 ? -mag : mag  (scaled, s32)
	VPMOVZXBD (DI)(CX*1), Y6 // chroma DC (8×u32)
	VPADDD    Y6, Y3, Y3     // dc + scaled
	VPACKUSDW Y3, Y3, Y3     // s32->u16 sat [0,65535] (neg -> 0)
	VPERMQ    $0xD8, Y3, Y3
	VPACKUSWB Y3, Y3, Y3     // u16->u8 sat [0,255]
	MOVQ      X3, (DI)(CX*1)
	ADDQ $8, CX
	CMPQ CX, R9
	JLT  cflLoop
	VZEROUPPER
	RET

// func sumSamplesAVX2Asm(samples *uint16, n uint64) uint64
//
// Sums n (multiple of 16) uint16 samples into a 64-bit total.
TEXT ·sumSamplesAVX2Asm(SB), NOSPLIT, $0-24
	MOVQ samples+0(FP), SI
	MOVQ n+8(FP), R9
	VPXOR Y1, Y1, Y1   // u64 accumulator
	VPXOR Y2, Y2, Y2   // zero for unpack
	XORQ CX, CX
sumLoop:
	VMOVDQU   (SI)(CX*2), Y0  // 16 u16
	VPMOVZXWD X0, Y3          // low 8 -> u32
	VEXTRACTI128 $1, Y0, X0
	VPMOVZXWD X0, Y4          // high 8 -> u32
	VPADDD    Y4, Y3, Y3      // 8 u32 partial sums
	// widen 8×u32 -> u64 and accumulate
	VPMOVZXDQ X3, Y5
	VEXTRACTI128 $1, Y3, X3
	VPMOVZXDQ X3, Y6
	VPADDQ    Y6, Y5, Y5
	VPADDQ    Y5, Y1, Y1
	ADDQ $16, CX
	CMPQ CX, R9
	JLT  sumLoop
	// horizontal add of 4 u64 lanes in Y1
	VEXTRACTI128 $1, Y1, X2
	VPADDQ    X2, X1, X1
	VPSHUFD   $0x4E, X1, X3   // swap the two 64-bit halves
	VPADDQ    X3, X1, X1
	MOVQ      X1, AX
	MOVQ      AX, ret+16(FP)
	VZEROUPPER
	RET
