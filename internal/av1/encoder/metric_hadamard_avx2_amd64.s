// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 low-bitdepth Hadamard producers (4x4, 8x8). The SVT/AOM Walsh-Hadamard
// butterfly runs in int16 lanes (VPADDW/VPSUBW wrap exactly like the portable
// int16 arithmetic) with a full SSE lane transpose between the two passes, so
// the emitted coefficient order matches svt_aom_hadamard_{4x4,8x8}_neon (the
// transpose of the portable order). SATD is order-invariant and the encoder's
// parity tests accept this NEON order, so it is bit-exact with the reference.
//
// VEX operand order (Go plan9): "OP src2, src1, dst" -> dst = src1 (op) src2.

// hadamardAVX2Ctx field offsets.
#define HSRC       0
#define HSRCSTRIDE 8
#define HCOEFF     16

// func hadamard8x8AVX2Asm(ctx *hadamardAVX2Ctx)
TEXT ·hadamard8x8AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ HSRC(AX), SI
	MOVQ HSRCSTRIDE(AX), DX
	MOVQ HCOEFF(AX), DI
	SHLQ $1, DX // stride in int16 elements -> bytes

	// Load 8 rows (each 8 int16) into X0..X7.
	VMOVDQU (SI), X0
	ADDQ    DX, SI
	VMOVDQU (SI), X1
	ADDQ    DX, SI
	VMOVDQU (SI), X2
	ADDQ    DX, SI
	VMOVDQU (SI), X3
	ADDQ    DX, SI
	VMOVDQU (SI), X4
	ADDQ    DX, SI
	VMOVDQU (SI), X5
	ADDQ    DX, SI
	VMOVDQU (SI), X6
	ADDQ    DX, SI
	VMOVDQU (SI), X7

	// ---- pass 1 butterfly (inputs X0..X7 -> outputs o0..o7 in X8..X15) ----
	VPADDW X1, X0, X8   // b0 = x0+x1
	VPSUBW X1, X0, X9   // b1 = x0-x1
	VPADDW X3, X2, X10  // b2 = x2+x3
	VPSUBW X3, X2, X11  // b3 = x2-x3
	VPADDW X5, X4, X12  // b4 = x4+x5
	VPSUBW X5, X4, X13  // b5 = x4-x5
	VPADDW X7, X6, X14  // b6 = x6+x7
	VPSUBW X7, X6, X15  // b7 = x6-x7

	VPADDW X10, X8, X0  // c0 = b0+b2
	VPADDW X11, X9, X1  // c1 = b1+b3
	VPSUBW X10, X8, X2  // c2 = b0-b2
	VPSUBW X11, X9, X3  // c3 = b1-b3
	VPADDW X14, X12, X4 // c4 = b4+b6
	VPADDW X15, X13, X5 // c5 = b5+b7
	VPSUBW X14, X12, X6 // c6 = b4-b6
	VPSUBW X15, X13, X7 // c7 = b5-b7

	VPADDW X4, X0, X8   // o0 = c0+c4
	VPSUBW X6, X2, X9   // o1 = c2-c6
	VPSUBW X4, X0, X10  // o2 = c0-c4
	VPADDW X6, X2, X11  // o3 = c2+c6
	VPADDW X7, X3, X12  // o4 = c3+c7
	VPSUBW X7, X3, X13  // o5 = c3-c7
	VPSUBW X5, X1, X14  // o6 = c1-c5
	VPADDW X5, X1, X15  // o7 = c1+c5

	// ---- transpose o0..o7 (X8..X15) -> t0..t7 (X0..X7) ----
	VPUNPCKLWD X9, X8, X0    // a0
	VPUNPCKHWD X9, X8, X1    // a1
	VPUNPCKLWD X11, X10, X2  // a2
	VPUNPCKHWD X11, X10, X3  // a3
	VPUNPCKLWD X13, X12, X4  // a4
	VPUNPCKHWD X13, X12, X5  // a5
	VPUNPCKLWD X15, X14, X6  // a6
	VPUNPCKHWD X15, X14, X7  // a7

	VPUNPCKLDQ X2, X0, X8    // b0
	VPUNPCKHDQ X2, X0, X9    // b1
	VPUNPCKLDQ X3, X1, X10   // b2
	VPUNPCKHDQ X3, X1, X11   // b3
	VPUNPCKLDQ X6, X4, X12   // b4
	VPUNPCKHDQ X6, X4, X13   // b5
	VPUNPCKLDQ X7, X5, X14   // b6
	VPUNPCKHDQ X7, X5, X15   // b7

	VPUNPCKLQDQ X12, X8, X0  // t0
	VPUNPCKHQDQ X12, X8, X1  // t1
	VPUNPCKLQDQ X13, X9, X2  // t2
	VPUNPCKHQDQ X13, X9, X3  // t3
	VPUNPCKLQDQ X14, X10, X4 // t4
	VPUNPCKHQDQ X14, X10, X5 // t5
	VPUNPCKLQDQ X15, X11, X6 // t6
	VPUNPCKHQDQ X15, X11, X7 // t7

	// ---- pass 2 butterfly (inputs t0..t7 in X0..X7 -> p0..p7 in X8..X15) ----
	VPADDW X1, X0, X8
	VPSUBW X1, X0, X9
	VPADDW X3, X2, X10
	VPSUBW X3, X2, X11
	VPADDW X5, X4, X12
	VPSUBW X5, X4, X13
	VPADDW X7, X6, X14
	VPSUBW X7, X6, X15

	VPADDW X10, X8, X0
	VPADDW X11, X9, X1
	VPSUBW X10, X8, X2
	VPSUBW X11, X9, X3
	VPADDW X14, X12, X4
	VPADDW X15, X13, X5
	VPSUBW X14, X12, X6
	VPSUBW X15, X13, X7

	VPADDW X4, X0, X8   // p0 = o0
	VPSUBW X6, X2, X9   // p1
	VPSUBW X4, X0, X10  // p2
	VPADDW X6, X2, X11  // p3
	VPADDW X7, X3, X12  // p4
	VPSUBW X7, X3, X13  // p5
	VPSUBW X5, X1, X14  // p6
	VPADDW X5, X1, X15  // p7

	// ---- sign-extend each row (8 int16 -> 8 int32) and store ----
	VPMOVSXWD X8, Y0
	VMOVDQU   Y0, 0(DI)
	VPMOVSXWD X9, Y0
	VMOVDQU   Y0, 32(DI)
	VPMOVSXWD X10, Y0
	VMOVDQU   Y0, 64(DI)
	VPMOVSXWD X11, Y0
	VMOVDQU   Y0, 96(DI)
	VPMOVSXWD X12, Y0
	VMOVDQU   Y0, 128(DI)
	VPMOVSXWD X13, Y0
	VMOVDQU   Y0, 160(DI)
	VPMOVSXWD X14, Y0
	VMOVDQU   Y0, 192(DI)
	VPMOVSXWD X15, Y0
	VMOVDQU   Y0, 224(DI)
	VZEROUPPER
	RET

// func hadamard4x4AVX2Asm(ctx *hadamardAVX2Ctx)
// Two >>1-halving butterfly passes with a 4x4 transpose between; store rows.
TEXT ·hadamard4x4AVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ HSRC(AX), SI
	MOVQ HSRCSTRIDE(AX), DX
	MOVQ HCOEFF(AX), DI
	SHLQ $1, DX

	// Load 4 rows (each 4 int16 = 8 bytes) into X0..X3.
	VMOVQ (SI), X0
	ADDQ  DX, SI
	VMOVQ (SI), X1
	ADDQ  DX, SI
	VMOVQ (SI), X2
	ADDQ  DX, SI
	VMOVQ (SI), X3

	// ---- pass 1: onePass with >>1 (inputs X0..X3 -> X4..X7) ----
	VPADDW X1, X0, X4  // x0+x1
	VPSUBW X1, X0, X5  // x0-x1
	VPADDW X3, X2, X6  // x2+x3
	VPSUBW X3, X2, X7  // x2-x3
	VPSRAW $1, X4, X4  // b0
	VPSRAW $1, X5, X5  // b1
	VPSRAW $1, X6, X6  // b2
	VPSRAW $1, X7, X7  // b3
	VPADDW X6, X4, X0  // a0 = b0+b2
	VPADDW X7, X5, X1  // a1 = b1+b3
	VPSUBW X6, X4, X2  // a2 = b0-b2
	VPSUBW X7, X5, X3  // a3 = b1-b3

	// ---- transpose 4x4 (rows X0..X3, low 4 words each) ----
	VPUNPCKLWD X1, X0, X4  // [a0.0,a1.0,a0.1,a1.1,a0.2,a1.2,a0.3,a1.3]
	VPUNPCKLWD X3, X2, X5  // [a2.0,a3.0,a2.1,a3.1,...]
	VPUNPCKLDQ X5, X4, X6  // low: col0 [a0.0,a1.0,a2.0,a3.0], high: col1
	VPUNPCKHDQ X5, X4, X7  // low: col2, high: col3
	VMOVDQA     X6, X0          // T0 = col0 (low 4 words)
	VPUNPCKHQDQ X6, X6, X1      // T1 = col1
	VMOVDQA     X7, X2          // T2 = col2
	VPUNPCKHQDQ X7, X7, X3      // T3 = col3

	// ---- pass 2: onePass with >>1 (inputs X0..X3 -> X4..X7) ----
	VPADDW X1, X0, X4
	VPSUBW X1, X0, X5
	VPADDW X3, X2, X6
	VPSUBW X3, X2, X7
	VPSRAW $1, X4, X4
	VPSRAW $1, X5, X5
	VPSRAW $1, X6, X6
	VPSRAW $1, X7, X7
	VPADDW X6, X4, X0  // a0
	VPADDW X7, X5, X1  // a1
	VPSUBW X6, X4, X2  // a2
	VPSUBW X7, X5, X3  // a3

	// ---- sign-extend low 4 words of each row to int32 and store ----
	VPMOVSXWD X0, X4
	VMOVDQU   X4, 0(DI)
	VPMOVSXWD X1, X4
	VMOVDQU   X4, 16(DI)
	VPMOVSXWD X2, X4
	VMOVDQU   X4, 32(DI)
	VPMOVSXWD X3, X4
	VMOVDQU   X4, 48(DI)
	RET
