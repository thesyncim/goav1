// AVX2-accelerated inverse-transform row/column kernels (amd64).
//
// Each kernel transforms two adjacent rows (row pass) or two adjacent columns
// (column pass) at once, holding the two values for a given coefficient index
// in the low 64-bit lanes (lane0, lane1) of a YMM register. All multiplies use
// VPMULDQ (signed 32x32 -> 64), sums/differences use VPADDQ/VPSUBQ, and the
// fixed-point rounding shift and per-stage clamp are emulated with AVX2-only
// integer ops so the arithmetic is bit-for-bit identical to the pure-Go
// references for every supported bit depth (8/10/12).
//
// AVX2 has no 64-bit arithmetic shift; the rounding shift roundShift(x, N) is
// emulated as ((x + 2^(N-1) + 2^62) >>u N) - 2^(62-N), valid for every
// transform intermediate magnitude. The clamp uses VPCMPGTQ + VPBLENDVB.
//
// Bound only when cpu.Detected.AVX2 is set (see colpass/rowpass_dispatch_amd64.go).
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build amd64 && !purego

#include "textflag.h"

// Bias constants for the arithmetic-rounding-shift emulation (B=62):
//   roundShift(x,N) = ((x + 2^(N-1) + 2^62) >>u N) - 2^(62-N)
DATA add8<>+0(SB)/8, $((1<<7)+(1<<62))
GLOBL add8<>(SB), RODATA|NOPTR, $8
DATA sub8<>+0(SB)/8, $(1<<54)
GLOBL sub8<>(SB), RODATA|NOPTR, $8
DATA add11<>+0(SB)/8, $((1<<10)+(1<<62))
GLOBL add11<>(SB), RODATA|NOPTR, $8
DATA sub11<>+0(SB)/8, $(1<<51)
GLOBL sub11<>(SB), RODATA|NOPTR, $8
DATA add12<>+0(SB)/8, $((1<<11)+(1<<62))
GLOBL add12<>(SB), RODATA|NOPTR, $8
DATA sub12<>+0(SB)/8, $(1<<50)
GLOBL sub12<>(SB), RODATA|NOPTR, $8

// min in Y14, max in Y15, scratch Y13. CLAMP uses Y13.
#define CLAMP(r) \
    VPCMPGTQ r, Y14, Y13 \
    VPBLENDVB Y13, Y14, r, r \
    VPCMPGTQ Y15, r, Y13 \
    VPBLENDVB Y13, Y15, r, r
// rounding shifts via broadcast-from-memory bias
#define RS8(r)  VPBROADCASTQ add8<>(SB), Y13; VPADDQ Y13, r, r; VPSRLQ $8, r, r; VPBROADCASTQ sub8<>(SB), Y13; VPSUBQ Y13, r, r
#define RS11(r) VPBROADCASTQ add11<>(SB), Y13; VPADDQ Y13, r, r; VPSRLQ $11, r, r; VPBROADCASTQ sub11<>(SB), Y13; VPSUBQ Y13, r, r
#define RS12(r) VPBROADCASTQ add12<>(SB), Y13; VPADDQ Y13, r, r; VPSRLQ $12, r, r; VPBROADCASTQ sub12<>(SB), Y13; VPSUBQ Y13, r, r

// func dct8row(r0p, r1p *int32, minv, maxv int64)
TEXT ·inverseDCT8Row2AVX2(SB), NOSPLIT, $256-32
    MOVQ r0+0(FP), R8
    MOVQ r1+8(FP), R9
    MOVQ minv+16(FP), AX
    MOVQ AX, X0
    VPBROADCASTQ X0, Y14
    MOVQ maxv+24(FP), AX
    MOVQ AX, X0
    VPBROADCASTQ X0, Y15
    VPMOVSXDQ (R8), Y0
    VPMOVSXDQ 16(R8), Y1
    VPMOVSXDQ (R9), Y2
    VPMOVSXDQ 16(R9), Y3
    VPUNPCKLQDQ Y2, Y0, Y4
    VPUNPCKHQDQ Y2, Y0, Y5
    VPUNPCKLQDQ Y3, Y1, Y6
    VPUNPCKHQDQ Y3, Y1, Y7
    VMOVDQU X4, 0(SP)
    VMOVDQU X5, 16(SP)
    VEXTRACTI128 $1, Y4, X1
    VMOVDQU X1, 32(SP)
    VEXTRACTI128 $1, Y5, X1
    VMOVDQU X1, 48(SP)
    VMOVDQU X6, 64(SP)
    VMOVDQU X7, 80(SP)
    VEXTRACTI128 $1, Y6, X1
    VMOVDQU X1, 96(SP)
    VEXTRACTI128 $1, Y7, X1
    VMOVDQU X1, 112(SP)

    // const181 in Y10 (reused), 1567 in Y11, (3784-4096) in Y12
    MOVL $181, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    // even half
    VMOVDQU 0(SP), Y2     // e0
    VMOVDQU 64(SP), Y3    // e2
    VPADDQ Y3, Y2, Y4
    VPMULDQ Y10, Y4, Y4
    RS8(Y4)               // t0 -> Y4
    VPSUBQ Y3, Y2, Y5
    VPMULDQ Y10, Y5, Y5
    RS8(Y5)               // t1 -> Y5
    VMOVDQU 32(SP), Y2    // e1=c2
    VMOVDQU 96(SP), Y3    // e3=c6
    MOVL $1567, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    MOVL $(3784-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y12
    VPMULDQ Y11, Y2, Y6   // e1*1567
    VPMULDQ Y12, Y3, Y0   // e3*(3784-4096)
    VPSUBQ Y0, Y6, Y6
    RS12(Y6)
    VPSUBQ Y3, Y6, Y6     // t2 = ... - e3  -> Y6
    VPMULDQ Y12, Y2, Y7   // e1*(3784-4096)
    VPMULDQ Y11, Y3, Y0   // e3*1567
    VPADDQ Y0, Y7, Y7
    RS12(Y7)
    VPADDQ Y2, Y7, Y7     // t3 = ... + e1  -> Y7
    // even outputs (clamped) stored to slots 128..191
    VPADDQ Y7, Y4, Y0     // t0+t3
    CLAMP(Y0)
    VMOVDQU X0, 128(SP)   // even0
    VPADDQ Y6, Y5, Y0
    CLAMP(Y0)
    VMOVDQU X0, 144(SP)   // even1
    VPSUBQ Y6, Y5, Y0
    CLAMP(Y0)
    VMOVDQU X0, 160(SP)   // even2
    VPSUBQ Y7, Y4, Y0
    CLAMP(Y0)
    VMOVDQU X0, 176(SP)   // even3

    // odd half: in1=c1,in3=c3,in5=c5,in7=c7
    VMOVDQU 16(SP), Y2    // in1
    VMOVDQU 112(SP), Y3   // in7
    MOVL $799, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $(4017-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10, Y2, Y4   // in1*799
    VPMULDQ Y11, Y3, Y0   // in7*(4017-4096)
    VPSUBQ Y0, Y4, Y4
    RS12(Y4)
    VPSUBQ Y3, Y4, Y4     // t4a -> Y4
    VPMULDQ Y11, Y2, Y7   // in1*(4017-4096)
    VPMULDQ Y10, Y3, Y0   // in7*799
    VPADDQ Y0, Y7, Y7
    RS12(Y7)
    VPADDQ Y2, Y7, Y7     // t7a -> Y7
    VMOVDQU 80(SP), Y2    // in5
    VMOVDQU 48(SP), Y3    // in3
    MOVL $1703, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $1138, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10, Y2, Y5   // in5*1703
    VPMULDQ Y11, Y3, Y0   // in3*1138
    VPSUBQ Y0, Y5, Y5
    RS11(Y5)              // t5a -> Y5
    VPMULDQ Y11, Y2, Y6   // in5*1138
    VPMULDQ Y10, Y3, Y0   // in3*1703
    VPADDQ Y0, Y6, Y6
    RS11(Y6)              // t6a -> Y6
    // t4 = clamp(t4a+t5a); t5a=clamp(t4a-t5a); t7=clamp(t7a+t6a); t6a=clamp(t7a-t6a)
    VPADDQ Y5, Y4, Y8     // t4
    CLAMP(Y8)
    VPSUBQ Y5, Y4, Y5     // t5a
    CLAMP(Y5)
    VPADDQ Y6, Y7, Y9     // t7
    CLAMP(Y9)
    VPSUBQ Y6, Y7, Y6     // t6a
    CLAMP(Y6)
    // t5 = RS8((t6a - t5a)*181); t6 = RS8((t6a + t5a)*181)
    MOVL $181, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    VPSUBQ Y5, Y6, Y4     // t6a - t5a
    VPMULDQ Y10, Y4, Y4
    RS8(Y4)              // t5
    VPADDQ Y5, Y6, Y7    // t6a + t5a
    VPMULDQ Y10, Y7, Y7
    RS8(Y7)              // t6
    // outputs: c0=even0+t7; c1=even1+t6; c2=even2+t5; c3=even3+t4
    //          c4=even3-t4; c5=even2-t5; c6=even1-t6; c7=even0-t7
    // t7=Y9, t6=Y7, t5=Y4, t4=Y8
    VMOVDQU 128(SP), Y0   // even0
    VMOVDQU 144(SP), Y1   // even1
    VMOVDQU 160(SP), Y2   // even2
    VMOVDQU 176(SP), Y3   // even3
    VPADDQ Y9, Y0, Y10; CLAMP(Y10); VMOVDQU X10, 0(SP)
    VPADDQ Y7, Y1, Y10; CLAMP(Y10); VMOVDQU X10, 16(SP)
    VPADDQ Y4, Y2, Y10; CLAMP(Y10); VMOVDQU X10, 32(SP)
    VPADDQ Y8, Y3, Y10; CLAMP(Y10); VMOVDQU X10, 48(SP)
    VPSUBQ Y8, Y3, Y10; CLAMP(Y10); VMOVDQU X10, 64(SP)
    VPSUBQ Y4, Y2, Y10; CLAMP(Y10); VMOVDQU X10, 80(SP)
    VPSUBQ Y7, Y1, Y10; CLAMP(Y10); VMOVDQU X10, 96(SP)
    VPSUBQ Y9, Y0, Y10; CLAMP(Y10); VMOVDQU X10, 112(SP)
    // store back: each slot has 2 int64 (row0,row1). Narrow to int32 and scatter.
    // build row0 vector (8 int32) and row1 vector. Take lane0 of each slot.
    // Load pairs and pack.
    // We'll extract low int32 of each lane via VPSHUFD/cvt. Simpler: read int64, write int32.
    MOVQ $0, CX
storeloop:
    MOVQ 0(SP)(CX*1), DX     // row0 value (int64, low part = int32)
    MOVL DX, (R8)
    ADDQ $4, R8
    MOVQ 8(SP)(CX*1), DX     // row1 value
    MOVL DX, (R9)
    ADDQ $4, R9
    ADDQ $16, CX
    CMPQ CX, $128
    JL storeloop
    VZEROUPPER
    RET


// hbtf-ish helpers via macros operating on slots.
// MAC2(dstY, slotA, cA, slotB, cB, opSign): dst = a*cA (op) b*cB  where op is VPADDQ/VPSUBQ
// We'll just inline.

// func dct16row(r0p, r1p *int32, minv, maxv int64)
// slot layout: c[i] at 16*i (i=0..15). even results stored 256+16*k (k=0..7).
TEXT ·inverseDCT16Row2AVX2(SB), NOSPLIT, $512-32
    MOVQ r0+0(FP), R8
    MOVQ r1+8(FP), R9
    MOVQ minv+16(FP), AX
    MOVQ AX, X0
    VPBROADCASTQ X0, Y14
    MOVQ maxv+24(FP), AX
    MOVQ AX, X0
    VPBROADCASTQ X0, Y15
    VPMOVSXDQ (R8), Y0
    VPMOVSXDQ 16(R8), Y1
    VPMOVSXDQ (R9), Y2
    VPMOVSXDQ 16(R9), Y3
    VPUNPCKLQDQ Y2, Y0, Y4
    VPUNPCKHQDQ Y2, Y0, Y5
    VPUNPCKLQDQ Y3, Y1, Y6
    VPUNPCKHQDQ Y3, Y1, Y7
    VMOVDQU X4, 0(SP)
    VMOVDQU X5, 16(SP)
    VEXTRACTI128 $1, Y4, X1; VMOVDQU X1, 32(SP)
    VEXTRACTI128 $1, Y5, X1; VMOVDQU X1, 48(SP)
    VMOVDQU X6, 64(SP)
    VMOVDQU X7, 80(SP)
    VEXTRACTI128 $1, Y6, X1; VMOVDQU X1, 96(SP)
    VEXTRACTI128 $1, Y7, X1; VMOVDQU X1, 112(SP)
    VPMOVSXDQ 32(R8), Y0
    VPMOVSXDQ 48(R8), Y1
    VPMOVSXDQ 32(R9), Y2
    VPMOVSXDQ 48(R9), Y3
    VPUNPCKLQDQ Y2, Y0, Y4
    VPUNPCKHQDQ Y2, Y0, Y5
    VPUNPCKLQDQ Y3, Y1, Y6
    VPUNPCKHQDQ Y3, Y1, Y7
    VMOVDQU X4, 128(SP)
    VMOVDQU X5, 144(SP)
    VEXTRACTI128 $1, Y4, X1; VMOVDQU X1, 160(SP)
    VEXTRACTI128 $1, Y5, X1; VMOVDQU X1, 176(SP)
    VMOVDQU X6, 192(SP)
    VMOVDQU X7, 208(SP)
    VEXTRACTI128 $1, Y6, X1; VMOVDQU X1, 224(SP)
    VEXTRACTI128 $1, Y7, X1; VMOVDQU X1, 240(SP)

    // ===== even half: DCT8 over indices 0,2,4,6,8,10,12,14 (e0..e7) =====
    // e_k = c[2k]; slot offset 32*k.
    // ---- inner DCT4 over even-of-even: e0,e2,e4,e6 = c0,c4,c8,c12 ----
    MOVL $181, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    VMOVDQU 0(SP), Y2     // e0=c0
    VMOVDQU 128(SP), Y3   // e2=c8
    VPADDQ Y3,Y2,Y4; VPMULDQ Y10,Y4,Y4; RS8(Y4)   // q0
    VPSUBQ Y3,Y2,Y5; VPMULDQ Y10,Y5,Y5; RS8(Y5)   // q1
    VMOVDQU 64(SP), Y2    // e2grp? actually e2_inner = c4 (index2 of even)
    VMOVDQU 192(SP), Y3   // c12
    MOVL $1567, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    MOVL $(3784-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y12
    VPMULDQ Y11,Y2,Y6; VPMULDQ Y12,Y3,Y0; VPSUBQ Y0,Y6,Y6; RS12(Y6); VPSUBQ Y3,Y6,Y6 // q2
    VPMULDQ Y12,Y2,Y7; VPMULDQ Y11,Y3,Y0; VPADDQ Y0,Y7,Y7; RS12(Y7); VPADDQ Y2,Y7,Y7 // q3
    VPADDQ Y7,Y4,Y0; CLAMP(Y0); VMOVDQU X0, 256(SP) // r0
    VPADDQ Y6,Y5,Y0; CLAMP(Y0); VMOVDQU X0, 272(SP) // r1
    VPSUBQ Y6,Y5,Y0; CLAMP(Y0); VMOVDQU X0, 288(SP) // r2
    VPSUBQ Y7,Y4,Y0; CLAMP(Y0); VMOVDQU X0, 304(SP) // r3
    // ---- g4a..g7a from e1,e3,e5,e7 = c2,c6,c10,c14 ----
    VMOVDQU 32(SP), Y2    // e1=c2
    VMOVDQU 224(SP), Y3   // e7=c14
    MOVL $799, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $(4017-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10,Y2,Y4; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y4,Y4; RS12(Y4); VPSUBQ Y3,Y4,Y4 // g4a
    VPMULDQ Y11,Y2,Y7; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y7,Y7; RS12(Y7); VPADDQ Y2,Y7,Y7 // g7a
    VMOVDQU 160(SP), Y2   // e5=c10
    VMOVDQU 96(SP), Y3    // e3=c6
    MOVL $1703, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $1138, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10,Y2,Y5; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y5,Y5; RS11(Y5) // g5a
    VPMULDQ Y11,Y2,Y6; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y6,Y6; RS11(Y6) // g6a
    VPADDQ Y5,Y4,Y8; CLAMP(Y8)         // g4
    VPSUBQ Y5,Y4,Y5; CLAMP(Y5)         // g5a'
    VPADDQ Y6,Y7,Y9; CLAMP(Y9)         // g7
    VPSUBQ Y6,Y7,Y6; CLAMP(Y6)         // g6a'
    MOVL $181, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    VPSUBQ Y5,Y6,Y4; VPMULDQ Y10,Y4,Y4; RS8(Y4)   // g5
    VPADDQ Y5,Y6,Y7; VPMULDQ Y10,Y7,Y7; RS8(Y7)   // g6
    // even0..7 = r0+g7, r1+g6, r2+g5, r3+g4, r3-g4, r2-g5, r1-g6, r0-g7
    VMOVDQU 256(SP), Y0   // r0
    VMOVDQU 272(SP), Y1   // r1
    VMOVDQU 288(SP), Y2   // r2
    VMOVDQU 304(SP), Y3   // r3
    VPADDQ Y9,Y0,Y10; CLAMP(Y10); VMOVDQU X10, 256(SP) // even0
    VPADDQ Y7,Y1,Y10; CLAMP(Y10); VMOVDQU X10, 272(SP) // even1
    VPADDQ Y4,Y2,Y10; CLAMP(Y10); VMOVDQU X10, 288(SP) // even2
    VPADDQ Y8,Y3,Y10; CLAMP(Y10); VMOVDQU X10, 304(SP) // even3
    VPSUBQ Y8,Y3,Y10; CLAMP(Y10); VMOVDQU X10, 320(SP) // even4
    VPSUBQ Y4,Y2,Y10; CLAMP(Y10); VMOVDQU X10, 336(SP) // even5
    VPSUBQ Y7,Y1,Y10; CLAMP(Y10); VMOVDQU X10, 352(SP) // even6
    VPSUBQ Y9,Y0,Y10; CLAMP(Y10); VMOVDQU X10, 368(SP) // even7

    // ===== odd half: t8a..t15a from c1,c3,...,c15 =====
    // t8a = RS12(in1*401 - in15*(4076-4096)) - in15
    VMOVDQU 16(SP), Y2    // in1=c1
    VMOVDQU 240(SP), Y3   // in15=c15
    MOVL $401, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $(4076-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10,Y2,Y4; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y4,Y4; RS12(Y4); VPSUBQ Y3,Y4,Y4 // t8a -> 384
    VMOVDQU X4, 384(SP)
    VPMULDQ Y11,Y2,Y5; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y5,Y5; RS12(Y5); VPADDQ Y2,Y5,Y5 // t15a -> 496
    VMOVDQU X5, 496(SP)
    // t9a = RS11(in9*1583 - in7*1299); t14a = RS11(in9*1299 + in7*1583)
    VMOVDQU 144(SP), Y2   // in9=c9
    VMOVDQU 112(SP), Y3   // in7=c7
    MOVL $1583, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $1299, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10,Y2,Y4; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y4,Y4; RS11(Y4); VMOVDQU X4, 400(SP) // t9a
    VPMULDQ Y11,Y2,Y5; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y5,Y5; RS11(Y5); VMOVDQU X5, 480(SP) // t14a
    // t10a = RS12(in5*1931 - in11*(3612-4096)) - in11
    VMOVDQU 80(SP), Y2    // in5=c5
    VMOVDQU 176(SP), Y3   // in11=c11
    MOVL $1931, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $(3612-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10,Y2,Y4; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y4,Y4; RS12(Y4); VPSUBQ Y3,Y4,Y4; VMOVDQU X4, 416(SP) // t10a
    VPMULDQ Y11,Y2,Y5; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y5,Y5; RS12(Y5); VPADDQ Y2,Y5,Y5; VMOVDQU X5, 464(SP) // t13a
    // t11a = RS12(in13*(3920-4096) - in3*1189) + in13
    VMOVDQU 208(SP), Y2   // in13=c13
    VMOVDQU 48(SP), Y3    // in3=c3
    MOVL $(3920-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $1189, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10,Y2,Y4; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y4,Y4; RS12(Y4); VPADDQ Y2,Y4,Y4; VMOVDQU X4, 432(SP) // t11a
    // t12a = RS12(in13*1189 + in3*(3920-4096)) + in3
    VPMULDQ Y11,Y2,Y5; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y5,Y5; RS12(Y5); VPADDQ Y3,Y5,Y5; VMOVDQU X5, 448(SP) // t12a

    // combine: t8=clamp(t8a+t9a) etc. Load from slots.
    VMOVDQU 384(SP), Y0 // t8a
    VMOVDQU 400(SP), Y1 // t9a
    VMOVDQU 416(SP), Y2 // t10a
    VMOVDQU 432(SP), Y3 // t11a
    VMOVDQU 448(SP), Y4 // t12a
    VMOVDQU 464(SP), Y5 // t13a
    VMOVDQU 480(SP), Y6 // t14a
    VMOVDQU 496(SP), Y7 // t15a
    VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 384(SP)  // t8
    VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 400(SP)  // t9
    VPSUBQ Y2,Y3,Y8; CLAMP(Y8); VMOVDQU X8, 416(SP)  // t10
    VPADDQ Y2,Y3,Y8; CLAMP(Y8); VMOVDQU X8, 432(SP)  // t11
    VPADDQ Y5,Y4,Y8; CLAMP(Y8); VMOVDQU X8, 448(SP)  // t12
    VPSUBQ Y5,Y4,Y8; CLAMP(Y8); VMOVDQU X8, 464(SP)  // t13
    VPSUBQ Y6,Y7,Y8; CLAMP(Y8); VMOVDQU X8, 480(SP)  // t14
    VPADDQ Y6,Y7,Y8; CLAMP(Y8); VMOVDQU X8, 496(SP)  // t15

    // stage: t9a = RS12(t14*1567 - t9*(3784-4096)) - t9; t14a = RS12(t14*(3784-4096)+t9*1567)+t14
    VMOVDQU 480(SP), Y2 // t14
    VMOVDQU 400(SP), Y3 // t9
    MOVL $1567, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $(3784-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10,Y2,Y4; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y4,Y4; RS12(Y4); VPSUBQ Y3,Y4,Y4; VMOVDQU X4, 400(SP) // t9a
    VPMULDQ Y11,Y2,Y5; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y5,Y5; RS12(Y5); VPADDQ Y2,Y5,Y5; VMOVDQU X5, 480(SP) // t14a
    // t10a = RS12(-(t13*(3784-4096)+t10*1567)) - t13 ; t13a = RS12(t13*1567 - t10*(3784-4096)) - t10
    VMOVDQU 464(SP), Y2 // t13
    VMOVDQU 416(SP), Y3 // t10
    VPMULDQ Y11,Y2,Y4; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y4,Y4
    VPXOR Y1,Y1,Y1; VPSUBQ Y4,Y1,Y4   // negate
    RS12(Y4); VPSUBQ Y2,Y4,Y4; VMOVDQU X4, 416(SP) // t10a
    VPMULDQ Y10,Y2,Y5; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y5,Y5; RS12(Y5); VPSUBQ Y3,Y5,Y5; VMOVDQU X5, 464(SP) // t13a

    // t8a=clamp(t8+t11); t9=clamp(t9a+t10a); t10=clamp(t9a-t10a); t11a=clamp(t8-t11)
    VMOVDQU 384(SP), Y0 // t8
    VMOVDQU 432(SP), Y1 // t11
    VMOVDQU 400(SP), Y2 // t9a
    VMOVDQU 416(SP), Y3 // t10a
    VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 384(SP) // t8a
    VPADDQ Y3,Y2,Y8; CLAMP(Y8); VMOVDQU X8, 400(SP) // t9
    VPSUBQ Y3,Y2,Y8; CLAMP(Y8); VMOVDQU X8, 416(SP) // t10
    VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 432(SP) // t11a
    // t12a=clamp(t15-t12); t13=clamp(t14a-t13a); t14=clamp(t14a+t13a); t15a=clamp(t15+t12)
    VMOVDQU 496(SP), Y0 // t15
    VMOVDQU 448(SP), Y1 // t12
    VMOVDQU 480(SP), Y2 // t14a
    VMOVDQU 464(SP), Y3 // t13a
    VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 448(SP) // t12a
    VPSUBQ Y3,Y2,Y8; CLAMP(Y8); VMOVDQU X8, 464(SP) // t13
    VPADDQ Y3,Y2,Y8; CLAMP(Y8); VMOVDQU X8, 480(SP) // t14
    VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 496(SP) // t15a

    // t10a=RS8((t13-t10)*181); t13a=RS8((t13+t10)*181); t11=RS8((t12a-t11a)*181); t12=RS8((t12a+t11a)*181)
    MOVL $181, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    VMOVDQU 464(SP), Y2 // t13
    VMOVDQU 416(SP), Y3 // t10
    VPSUBQ Y3,Y2,Y4; VPMULDQ Y10,Y4,Y4; RS8(Y4); VMOVDQU X4, 416(SP) // t10a
    VPADDQ Y3,Y2,Y5; VPMULDQ Y10,Y5,Y5; RS8(Y5); VMOVDQU X5, 464(SP) // t13a
    VMOVDQU 448(SP), Y2 // t12a
    VMOVDQU 432(SP), Y3 // t11a
    VPSUBQ Y3,Y2,Y4; VPMULDQ Y10,Y4,Y4; RS8(Y4); VMOVDQU X4, 432(SP) // t11
    VPADDQ Y3,Y2,Y5; VPMULDQ Y10,Y5,Y5; RS8(Y5); VMOVDQU X5, 448(SP) // t12

    // outputs c[k]=clamp(even[k]+t[15a..8a]) and c[15-k]=clamp(even[k]-...)
    // order per pure-Go:
    // c0=even0+t15a; c1=even1+t14; c2=even2+t13a; c3=even3+t12; c4=even4+t11; c5=even5+t10a; c6=even6+t9; c7=even7+t8a
    // c8=even7-t8a; c9=even6-t9; c10=even5-t10a; c11=even4-t11; c12=even3-t12; c13=even2-t13a; c14=even1-t14; c15=even0-t15a
    // even k at 256+16k; t array: t15a@496 t14@480 t13a@464 t12@448 t11@432 t10a@416 t9@400 t8a@384
    VMOVDQU 256(SP), Y0; VMOVDQU 496(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 0(SP)
    VMOVDQU 272(SP), Y0; VMOVDQU 480(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 16(SP)
    VMOVDQU 288(SP), Y0; VMOVDQU 464(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 32(SP)
    VMOVDQU 304(SP), Y0; VMOVDQU 448(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 48(SP)
    VMOVDQU 320(SP), Y0; VMOVDQU 432(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 64(SP)
    VMOVDQU 336(SP), Y0; VMOVDQU 416(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 80(SP)
    VMOVDQU 352(SP), Y0; VMOVDQU 400(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 96(SP)
    VMOVDQU 368(SP), Y0; VMOVDQU 384(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 112(SP)
    VMOVDQU 368(SP), Y0; VMOVDQU 384(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 128(SP)
    VMOVDQU 352(SP), Y0; VMOVDQU 400(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 144(SP)
    VMOVDQU 336(SP), Y0; VMOVDQU 416(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 160(SP)
    VMOVDQU 320(SP), Y0; VMOVDQU 432(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 176(SP)
    VMOVDQU 304(SP), Y0; VMOVDQU 448(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 192(SP)
    VMOVDQU 288(SP), Y0; VMOVDQU 464(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 208(SP)
    VMOVDQU 272(SP), Y0; VMOVDQU 480(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 224(SP)
    VMOVDQU 256(SP), Y0; VMOVDQU 496(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 240(SP)
    // store back to rows
    MOVQ $0, CX
dct16store:
    MOVQ 0(SP)(CX*1), DX; MOVL DX, (R8); ADDQ $4, R8
    MOVQ 8(SP)(CX*1), DX; MOVL DX, (R9); ADDQ $4, R9
    ADDQ $16, CX
    CMPQ CX, $256
    JL dct16store
    VZEROUPPER
    RET

// func dct8col(base *int32, rowStrideBytes, minv, maxv int64)
// Two adjacent columns; arithmetic identical to DCT8Row, strided I/O. Bit-exact
// to the pure-Go column kernel run on each of the two columns.
TEXT ·inverseDCT8Col2AVX2(SB), NOSPLIT, $256-32
	MOVQ base+0(FP), R8
	MOVQ rowStrideBytes+8(FP), R10
	MOVQ minv+16(FP), AX
	MOVQ AX, X0
	VPBROADCASTQ X0, Y14
	MOVQ maxv+24(FP), AX
	MOVQ AX, X0
	VPBROADCASTQ X0, Y15
	MOVQ R8, R11
	MOVQ $0, CX
dct8col_load:
	MOVQ (R11), AX
	MOVLQSX AX, DX
	MOVQ DX, (SP)(CX*1)
	SARQ $32, AX
	MOVQ AX, 8(SP)(CX*1)
	ADDQ R10, R11
	ADDQ $16, CX
	CMPQ CX, $128
	JL dct8col_load

    // const181 in Y10 (reused), 1567 in Y11, (3784-4096) in Y12
    MOVL $181, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    // even half
    VMOVDQU 0(SP), Y2     // e0
    VMOVDQU 64(SP), Y3    // e2
    VPADDQ Y3, Y2, Y4
    VPMULDQ Y10, Y4, Y4
    RS8(Y4)               // t0 -> Y4
    VPSUBQ Y3, Y2, Y5
    VPMULDQ Y10, Y5, Y5
    RS8(Y5)               // t1 -> Y5
    VMOVDQU 32(SP), Y2    // e1=c2
    VMOVDQU 96(SP), Y3    // e3=c6
    MOVL $1567, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    MOVL $(3784-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y12
    VPMULDQ Y11, Y2, Y6   // e1*1567
    VPMULDQ Y12, Y3, Y0   // e3*(3784-4096)
    VPSUBQ Y0, Y6, Y6
    RS12(Y6)
    VPSUBQ Y3, Y6, Y6     // t2 = ... - e3  -> Y6
    VPMULDQ Y12, Y2, Y7   // e1*(3784-4096)
    VPMULDQ Y11, Y3, Y0   // e3*1567
    VPADDQ Y0, Y7, Y7
    RS12(Y7)
    VPADDQ Y2, Y7, Y7     // t3 = ... + e1  -> Y7
    // even outputs (clamped) stored to slots 128..191
    VPADDQ Y7, Y4, Y0     // t0+t3
    CLAMP(Y0)
    VMOVDQU X0, 128(SP)   // even0
    VPADDQ Y6, Y5, Y0
    CLAMP(Y0)
    VMOVDQU X0, 144(SP)   // even1
    VPSUBQ Y6, Y5, Y0
    CLAMP(Y0)
    VMOVDQU X0, 160(SP)   // even2
    VPSUBQ Y7, Y4, Y0
    CLAMP(Y0)
    VMOVDQU X0, 176(SP)   // even3

    // odd half: in1=c1,in3=c3,in5=c5,in7=c7
    VMOVDQU 16(SP), Y2    // in1
    VMOVDQU 112(SP), Y3   // in7
    MOVL $799, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $(4017-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10, Y2, Y4   // in1*799
    VPMULDQ Y11, Y3, Y0   // in7*(4017-4096)
    VPSUBQ Y0, Y4, Y4
    RS12(Y4)
    VPSUBQ Y3, Y4, Y4     // t4a -> Y4
    VPMULDQ Y11, Y2, Y7   // in1*(4017-4096)
    VPMULDQ Y10, Y3, Y0   // in7*799
    VPADDQ Y0, Y7, Y7
    RS12(Y7)
    VPADDQ Y2, Y7, Y7     // t7a -> Y7
    VMOVDQU 80(SP), Y2    // in5
    VMOVDQU 48(SP), Y3    // in3
    MOVL $1703, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $1138, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10, Y2, Y5   // in5*1703
    VPMULDQ Y11, Y3, Y0   // in3*1138
    VPSUBQ Y0, Y5, Y5
    RS11(Y5)              // t5a -> Y5
    VPMULDQ Y11, Y2, Y6   // in5*1138
    VPMULDQ Y10, Y3, Y0   // in3*1703
    VPADDQ Y0, Y6, Y6
    RS11(Y6)              // t6a -> Y6
    // t4 = clamp(t4a+t5a); t5a=clamp(t4a-t5a); t7=clamp(t7a+t6a); t6a=clamp(t7a-t6a)
    VPADDQ Y5, Y4, Y8     // t4
    CLAMP(Y8)
    VPSUBQ Y5, Y4, Y5     // t5a
    CLAMP(Y5)
    VPADDQ Y6, Y7, Y9     // t7
    CLAMP(Y9)
    VPSUBQ Y6, Y7, Y6     // t6a
    CLAMP(Y6)
    // t5 = RS8((t6a - t5a)*181); t6 = RS8((t6a + t5a)*181)
    MOVL $181, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    VPSUBQ Y5, Y6, Y4     // t6a - t5a
    VPMULDQ Y10, Y4, Y4
    RS8(Y4)              // t5
    VPADDQ Y5, Y6, Y7    // t6a + t5a
    VPMULDQ Y10, Y7, Y7
    RS8(Y7)              // t6
    // outputs: c0=even0+t7; c1=even1+t6; c2=even2+t5; c3=even3+t4
    //          c4=even3-t4; c5=even2-t5; c6=even1-t6; c7=even0-t7
    // t7=Y9, t6=Y7, t5=Y4, t4=Y8
    VMOVDQU 128(SP), Y0   // even0
    VMOVDQU 144(SP), Y1   // even1
    VMOVDQU 160(SP), Y2   // even2
    VMOVDQU 176(SP), Y3   // even3
    VPADDQ Y9, Y0, Y10; CLAMP(Y10); VMOVDQU X10, 0(SP)
    VPADDQ Y7, Y1, Y10; CLAMP(Y10); VMOVDQU X10, 16(SP)
    VPADDQ Y4, Y2, Y10; CLAMP(Y10); VMOVDQU X10, 32(SP)
    VPADDQ Y8, Y3, Y10; CLAMP(Y10); VMOVDQU X10, 48(SP)
    VPSUBQ Y8, Y3, Y10; CLAMP(Y10); VMOVDQU X10, 64(SP)
    VPSUBQ Y4, Y2, Y10; CLAMP(Y10); VMOVDQU X10, 80(SP)
    VPSUBQ Y7, Y1, Y10; CLAMP(Y10); VMOVDQU X10, 96(SP)
    VPSUBQ Y9, Y0, Y10; CLAMP(Y10); VMOVDQU X10, 112(SP)
    // store back: each slot has 2 int64 (row0,row1). Narrow to int32 and scatter.
    // build row0 vector (8 int32) and row1 vector. Take lane0 of each slot.
    // Load pairs and pack.
    // We'll extract low int32 of each lane via VPSHUFD/cvt. Simpler: read int64, write int32.
	MOVQ R8, R11
	MOVQ $0, CX
dct8col_store:
	MOVQ (SP)(CX*1), DX
	MOVL DX, (R11)
	MOVQ 8(SP)(CX*1), DX
	MOVL DX, 4(R11)
	ADDQ R10, R11
	ADDQ $16, CX
	CMPQ CX, $128
	JL dct8col_store
	VZEROUPPER
	RET

// func dct16col(base *int32, rowStrideBytes, minv, maxv int64)
// Two adjacent columns; arithmetic identical to DCT16Row, strided I/O. Bit-exact
// to the pure-Go column kernel run on each of the two columns.
TEXT ·inverseDCT16Col2AVX2(SB), NOSPLIT, $544-32
	MOVQ base+0(FP), R8
	MOVQ rowStrideBytes+8(FP), R10
	MOVQ minv+16(FP), AX
	MOVQ AX, X0
	VPBROADCASTQ X0, Y14
	MOVQ maxv+24(FP), AX
	MOVQ AX, X0
	VPBROADCASTQ X0, Y15
	MOVQ R8, R11
	MOVQ $0, CX
dct16col_load:
	MOVQ (R11), AX
	MOVLQSX AX, DX
	MOVQ DX, (SP)(CX*1)
	SARQ $32, AX
	MOVQ AX, 8(SP)(CX*1)
	ADDQ R10, R11
	ADDQ $16, CX
	CMPQ CX, $256
	JL dct16col_load

    // ===== even half: DCT8 over indices 0,2,4,6,8,10,12,14 (e0..e7) =====
    // e_k = c[2k]; slot offset 32*k.
    // ---- inner DCT4 over even-of-even: e0,e2,e4,e6 = c0,c4,c8,c12 ----
    MOVL $181, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    VMOVDQU 0(SP), Y2     // e0=c0
    VMOVDQU 128(SP), Y3   // e2=c8
    VPADDQ Y3,Y2,Y4; VPMULDQ Y10,Y4,Y4; RS8(Y4)   // q0
    VPSUBQ Y3,Y2,Y5; VPMULDQ Y10,Y5,Y5; RS8(Y5)   // q1
    VMOVDQU 64(SP), Y2    // e2grp? actually e2_inner = c4 (index2 of even)
    VMOVDQU 192(SP), Y3   // c12
    MOVL $1567, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    MOVL $(3784-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y12
    VPMULDQ Y11,Y2,Y6; VPMULDQ Y12,Y3,Y0; VPSUBQ Y0,Y6,Y6; RS12(Y6); VPSUBQ Y3,Y6,Y6 // q2
    VPMULDQ Y12,Y2,Y7; VPMULDQ Y11,Y3,Y0; VPADDQ Y0,Y7,Y7; RS12(Y7); VPADDQ Y2,Y7,Y7 // q3
    VPADDQ Y7,Y4,Y0; CLAMP(Y0); VMOVDQU X0, 256(SP) // r0
    VPADDQ Y6,Y5,Y0; CLAMP(Y0); VMOVDQU X0, 272(SP) // r1
    VPSUBQ Y6,Y5,Y0; CLAMP(Y0); VMOVDQU X0, 288(SP) // r2
    VPSUBQ Y7,Y4,Y0; CLAMP(Y0); VMOVDQU X0, 304(SP) // r3
    // ---- g4a..g7a from e1,e3,e5,e7 = c2,c6,c10,c14 ----
    VMOVDQU 32(SP), Y2    // e1=c2
    VMOVDQU 224(SP), Y3   // e7=c14
    MOVL $799, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $(4017-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10,Y2,Y4; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y4,Y4; RS12(Y4); VPSUBQ Y3,Y4,Y4 // g4a
    VPMULDQ Y11,Y2,Y7; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y7,Y7; RS12(Y7); VPADDQ Y2,Y7,Y7 // g7a
    VMOVDQU 160(SP), Y2   // e5=c10
    VMOVDQU 96(SP), Y3    // e3=c6
    MOVL $1703, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $1138, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10,Y2,Y5; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y5,Y5; RS11(Y5) // g5a
    VPMULDQ Y11,Y2,Y6; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y6,Y6; RS11(Y6) // g6a
    VPADDQ Y5,Y4,Y8; CLAMP(Y8)         // g4
    VPSUBQ Y5,Y4,Y5; CLAMP(Y5)         // g5a'
    VPADDQ Y6,Y7,Y9; CLAMP(Y9)         // g7
    VPSUBQ Y6,Y7,Y6; CLAMP(Y6)         // g6a'
    MOVL $181, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    VPSUBQ Y5,Y6,Y4; VPMULDQ Y10,Y4,Y4; RS8(Y4)   // g5
    VPADDQ Y5,Y6,Y7; VPMULDQ Y10,Y7,Y7; RS8(Y7)   // g6
    // even0..7 = r0+g7, r1+g6, r2+g5, r3+g4, r3-g4, r2-g5, r1-g6, r0-g7
    VMOVDQU 256(SP), Y0   // r0
    VMOVDQU 272(SP), Y1   // r1
    VMOVDQU 288(SP), Y2   // r2
    VMOVDQU 304(SP), Y3   // r3
    VPADDQ Y9,Y0,Y10; CLAMP(Y10); VMOVDQU X10, 256(SP) // even0
    VPADDQ Y7,Y1,Y10; CLAMP(Y10); VMOVDQU X10, 272(SP) // even1
    VPADDQ Y4,Y2,Y10; CLAMP(Y10); VMOVDQU X10, 288(SP) // even2
    VPADDQ Y8,Y3,Y10; CLAMP(Y10); VMOVDQU X10, 304(SP) // even3
    VPSUBQ Y8,Y3,Y10; CLAMP(Y10); VMOVDQU X10, 320(SP) // even4
    VPSUBQ Y4,Y2,Y10; CLAMP(Y10); VMOVDQU X10, 336(SP) // even5
    VPSUBQ Y7,Y1,Y10; CLAMP(Y10); VMOVDQU X10, 352(SP) // even6
    VPSUBQ Y9,Y0,Y10; CLAMP(Y10); VMOVDQU X10, 368(SP) // even7

    // ===== odd half: t8a..t15a from c1,c3,...,c15 =====
    // t8a = RS12(in1*401 - in15*(4076-4096)) - in15
    VMOVDQU 16(SP), Y2    // in1=c1
    VMOVDQU 240(SP), Y3   // in15=c15
    MOVL $401, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $(4076-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10,Y2,Y4; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y4,Y4; RS12(Y4); VPSUBQ Y3,Y4,Y4 // t8a -> 384
    VMOVDQU X4, 384(SP)
    VPMULDQ Y11,Y2,Y5; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y5,Y5; RS12(Y5); VPADDQ Y2,Y5,Y5 // t15a -> 496
    VMOVDQU X5, 496(SP)
    // t9a = RS11(in9*1583 - in7*1299); t14a = RS11(in9*1299 + in7*1583)
    VMOVDQU 144(SP), Y2   // in9=c9
    VMOVDQU 112(SP), Y3   // in7=c7
    MOVL $1583, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $1299, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10,Y2,Y4; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y4,Y4; RS11(Y4); VMOVDQU X4, 400(SP) // t9a
    VPMULDQ Y11,Y2,Y5; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y5,Y5; RS11(Y5); VMOVDQU X5, 480(SP) // t14a
    // t10a = RS12(in5*1931 - in11*(3612-4096)) - in11
    VMOVDQU 80(SP), Y2    // in5=c5
    VMOVDQU 176(SP), Y3   // in11=c11
    MOVL $1931, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $(3612-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10,Y2,Y4; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y4,Y4; RS12(Y4); VPSUBQ Y3,Y4,Y4; VMOVDQU X4, 416(SP) // t10a
    VPMULDQ Y11,Y2,Y5; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y5,Y5; RS12(Y5); VPADDQ Y2,Y5,Y5; VMOVDQU X5, 464(SP) // t13a
    // t11a = RS12(in13*(3920-4096) - in3*1189) + in13
    VMOVDQU 208(SP), Y2   // in13=c13
    VMOVDQU 48(SP), Y3    // in3=c3
    MOVL $(3920-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $1189, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10,Y2,Y4; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y4,Y4; RS12(Y4); VPADDQ Y2,Y4,Y4; VMOVDQU X4, 432(SP) // t11a
    // t12a = RS12(in13*1189 + in3*(3920-4096)) + in3
    VPMULDQ Y11,Y2,Y5; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y5,Y5; RS12(Y5); VPADDQ Y3,Y5,Y5; VMOVDQU X5, 448(SP) // t12a

    // combine: t8=clamp(t8a+t9a) etc. Load from slots.
    VMOVDQU 384(SP), Y0 // t8a
    VMOVDQU 400(SP), Y1 // t9a
    VMOVDQU 416(SP), Y2 // t10a
    VMOVDQU 432(SP), Y3 // t11a
    VMOVDQU 448(SP), Y4 // t12a
    VMOVDQU 464(SP), Y5 // t13a
    VMOVDQU 480(SP), Y6 // t14a
    VMOVDQU 496(SP), Y7 // t15a
    VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 384(SP)  // t8
    VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 400(SP)  // t9
    VPSUBQ Y2,Y3,Y8; CLAMP(Y8); VMOVDQU X8, 416(SP)  // t10
    VPADDQ Y2,Y3,Y8; CLAMP(Y8); VMOVDQU X8, 432(SP)  // t11
    VPADDQ Y5,Y4,Y8; CLAMP(Y8); VMOVDQU X8, 448(SP)  // t12
    VPSUBQ Y5,Y4,Y8; CLAMP(Y8); VMOVDQU X8, 464(SP)  // t13
    VPSUBQ Y6,Y7,Y8; CLAMP(Y8); VMOVDQU X8, 480(SP)  // t14
    VPADDQ Y6,Y7,Y8; CLAMP(Y8); VMOVDQU X8, 496(SP)  // t15

    // stage: t9a = RS12(t14*1567 - t9*(3784-4096)) - t9; t14a = RS12(t14*(3784-4096)+t9*1567)+t14
    VMOVDQU 480(SP), Y2 // t14
    VMOVDQU 400(SP), Y3 // t9
    MOVL $1567, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    MOVL $(3784-4096), AX; MOVQ AX, X0; VPBROADCASTQ X0, Y11
    VPMULDQ Y10,Y2,Y4; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y4,Y4; RS12(Y4); VPSUBQ Y3,Y4,Y4; VMOVDQU X4, 400(SP) // t9a
    VPMULDQ Y11,Y2,Y5; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y5,Y5; RS12(Y5); VPADDQ Y2,Y5,Y5; VMOVDQU X5, 480(SP) // t14a
    // t10a = RS12(-(t13*(3784-4096)+t10*1567)) - t13 ; t13a = RS12(t13*1567 - t10*(3784-4096)) - t10
    VMOVDQU 464(SP), Y2 // t13
    VMOVDQU 416(SP), Y3 // t10
    VPMULDQ Y11,Y2,Y4; VPMULDQ Y10,Y3,Y0; VPADDQ Y0,Y4,Y4
    VPXOR Y1,Y1,Y1; VPSUBQ Y4,Y1,Y4   // negate
    RS12(Y4); VPSUBQ Y2,Y4,Y4; VMOVDQU X4, 416(SP) // t10a
    VPMULDQ Y10,Y2,Y5; VPMULDQ Y11,Y3,Y0; VPSUBQ Y0,Y5,Y5; RS12(Y5); VPSUBQ Y3,Y5,Y5; VMOVDQU X5, 464(SP) // t13a

    // t8a=clamp(t8+t11); t9=clamp(t9a+t10a); t10=clamp(t9a-t10a); t11a=clamp(t8-t11)
    VMOVDQU 384(SP), Y0 // t8
    VMOVDQU 432(SP), Y1 // t11
    VMOVDQU 400(SP), Y2 // t9a
    VMOVDQU 416(SP), Y3 // t10a
    VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 384(SP) // t8a
    VPADDQ Y3,Y2,Y8; CLAMP(Y8); VMOVDQU X8, 400(SP) // t9
    VPSUBQ Y3,Y2,Y8; CLAMP(Y8); VMOVDQU X8, 416(SP) // t10
    VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 432(SP) // t11a
    // t12a=clamp(t15-t12); t13=clamp(t14a-t13a); t14=clamp(t14a+t13a); t15a=clamp(t15+t12)
    VMOVDQU 496(SP), Y0 // t15
    VMOVDQU 448(SP), Y1 // t12
    VMOVDQU 480(SP), Y2 // t14a
    VMOVDQU 464(SP), Y3 // t13a
    VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 448(SP) // t12a
    VPSUBQ Y3,Y2,Y8; CLAMP(Y8); VMOVDQU X8, 464(SP) // t13
    VPADDQ Y3,Y2,Y8; CLAMP(Y8); VMOVDQU X8, 480(SP) // t14
    VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 496(SP) // t15a

    // t10a=RS8((t13-t10)*181); t13a=RS8((t13+t10)*181); t11=RS8((t12a-t11a)*181); t12=RS8((t12a+t11a)*181)
    MOVL $181, AX; MOVQ AX, X0; VPBROADCASTQ X0, Y10
    VMOVDQU 464(SP), Y2 // t13
    VMOVDQU 416(SP), Y3 // t10
    VPSUBQ Y3,Y2,Y4; VPMULDQ Y10,Y4,Y4; RS8(Y4); VMOVDQU X4, 416(SP) // t10a
    VPADDQ Y3,Y2,Y5; VPMULDQ Y10,Y5,Y5; RS8(Y5); VMOVDQU X5, 464(SP) // t13a
    VMOVDQU 448(SP), Y2 // t12a
    VMOVDQU 432(SP), Y3 // t11a
    VPSUBQ Y3,Y2,Y4; VPMULDQ Y10,Y4,Y4; RS8(Y4); VMOVDQU X4, 432(SP) // t11
    VPADDQ Y3,Y2,Y5; VPMULDQ Y10,Y5,Y5; RS8(Y5); VMOVDQU X5, 448(SP) // t12

    // outputs c[k]=clamp(even[k]+t[15a..8a]) and c[15-k]=clamp(even[k]-...)
    // order per pure-Go:
    // c0=even0+t15a; c1=even1+t14; c2=even2+t13a; c3=even3+t12; c4=even4+t11; c5=even5+t10a; c6=even6+t9; c7=even7+t8a
    // c8=even7-t8a; c9=even6-t9; c10=even5-t10a; c11=even4-t11; c12=even3-t12; c13=even2-t13a; c14=even1-t14; c15=even0-t15a
    // even k at 256+16k; t array: t15a@496 t14@480 t13a@464 t12@448 t11@432 t10a@416 t9@400 t8a@384
    VMOVDQU 256(SP), Y0; VMOVDQU 496(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 0(SP)
    VMOVDQU 272(SP), Y0; VMOVDQU 480(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 16(SP)
    VMOVDQU 288(SP), Y0; VMOVDQU 464(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 32(SP)
    VMOVDQU 304(SP), Y0; VMOVDQU 448(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 48(SP)
    VMOVDQU 320(SP), Y0; VMOVDQU 432(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 64(SP)
    VMOVDQU 336(SP), Y0; VMOVDQU 416(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 80(SP)
    VMOVDQU 352(SP), Y0; VMOVDQU 400(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 96(SP)
    VMOVDQU 368(SP), Y0; VMOVDQU 384(SP), Y1; VPADDQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 112(SP)
    VMOVDQU 368(SP), Y0; VMOVDQU 384(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 128(SP)
    VMOVDQU 352(SP), Y0; VMOVDQU 400(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 144(SP)
    VMOVDQU 336(SP), Y0; VMOVDQU 416(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 160(SP)
    VMOVDQU 320(SP), Y0; VMOVDQU 432(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 176(SP)
    VMOVDQU 304(SP), Y0; VMOVDQU 448(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 192(SP)
    VMOVDQU 288(SP), Y0; VMOVDQU 464(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 208(SP)
    VMOVDQU 272(SP), Y0; VMOVDQU 480(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 224(SP)
    VMOVDQU 256(SP), Y0; VMOVDQU 496(SP), Y1; VPSUBQ Y1,Y0,Y8; CLAMP(Y8); VMOVDQU X8, 240(SP)
    // store back to rows
	MOVQ R8, R11
	MOVQ $0, CX
dct16col_store:
	MOVQ (SP)(CX*1), DX
	MOVL DX, (R11)
	MOVQ 8(SP)(CX*1), DX
	MOVL DX, 4(R11)
	ADDQ R10, R11
	ADDQ $16, CX
	CMPQ CX, $256
	JL dct16col_store
	VZEROUPPER
	RET

