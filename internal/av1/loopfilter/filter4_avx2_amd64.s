// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// AVX2 8-bit narrow (four-sample) deblocking kernel. Sixteen horizontal edge
// positions are processed per 256-bit vector. Bit-exact with filter4EdgePureGo
// (see filter4.go); the Go wrapper in filter4_avx2_amd64.go restricts it to
// 8-bit horizontal edges processed in groups of sixteen positions.
//
// All lane arithmetic runs in signed 16-bit lanes: sixteen bytes widen to one
// ymm of sixteen int16 with VPMOVZXBW, the centred ps/qs values stay in
// [-128,127], clamps use VPMINSW/VPMAXSW and shifts use VPSRAW, exactly matching
// signedClamp and the integer shifts in the reference. needsFilter4 and hev are
// reproduced as lane masks; each filtered tap is blended with the original
// through VPBLENDVB so failing lanes keep their bytes, then the sixteen int16
// results are narrowed to sixteen contiguous bytes (VPACKUSWB + VPERMQ).

//go:build amd64 && !purego

#include "textflag.h"

#define P1 0
#define P0 8
#define Q0 16
#define Q1 24
#define COUNT 32
#define LIMIT 40
#define BLIMIT 48
#define HEV 56

// func filter4EdgeAVX2Asm(ctx *filter4AVX2Ctx)
TEXT ·filter4EdgeAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ P1(AX), R8
	MOVQ P0(AX), R9
	MOVQ Q0(AX), R10
	MOVQ Q1(AX), R11
	MOVQ COUNT(AX), R12

	// Broadcast the constants that live for the whole kernel.
	MOVL         $128, BX
	MOVL         BX, X3
	VPBROADCASTW X3, Y3 // 128 (center)
	MOVL         $-128, BX
	MOVL         BX, X4
	VPBROADCASTW X4, Y4 // -128 (clamp min)
	MOVL         $127, BX
	MOVL         BX, X5
	VPBROADCASTW X5, Y5 // 127 (clamp max)

loop:
	TESTQ R12, R12
	JZ    done

	// Load and widen the four taps to int16 (sixteen lanes each).
	VPMOVZXBW (R8), Y8   // p1
	VPMOVZXBW (R9), Y9   // p0
	VPMOVZXBW (R10), Y10 // q0
	VPMOVZXBW (R11), Y11 // q1

	// abs diffs: Y6=|p1-p0|, Y7=|q1-q0| (kept for hev below).
	VPSUBW Y9, Y8, Y6
	VPABSW Y6, Y6
	VPSUBW Y10, Y11, Y7
	VPABSW Y7, Y7

	// needMask: built as NOT(fail), fail accumulated in Y12.
	// limit broadcast in Y0 (transient), blimit in Y1 (transient).
	MOVQ         LIMIT(AX), BX
	MOVL         BX, X0
	VPBROADCASTW X0, Y0
	VPCMPGTW Y0, Y6, Y12 // |p1-p0| > limit
	VPCMPGTW Y0, Y7, Y13 // |q1-q0| > limit
	VPOR     Y13, Y12, Y12
	VPSUBW   Y10, Y9, Y13 // p0-q0
	VPABSW   Y13, Y13
	VPSLLW   $1, Y13, Y13 // |p0-q0|*2
	VPSUBW   Y11, Y8, Y14 // p1-q1
	VPABSW   Y14, Y14
	VPSRAW   $1, Y14, Y14 // |p1-q1|/2
	VPADDW   Y14, Y13, Y13
	MOVQ         BLIMIT(AX), BX
	MOVL         BX, X1
	VPBROADCASTW X1, Y1
	VPCMPGTW Y1, Y13, Y13 // sum > blimit
	VPOR     Y13, Y12, Y12 // fail
	VPCMPEQW Y13, Y13, Y13 // all ones
	VPXOR    Y13, Y12, Y0  // Y0 = needMask (~fail)

	// hevMask into Y1: |p1-p0| > hev || |q1-q0| > hev.
	MOVQ         HEV(AX), BX
	MOVL         BX, X1
	VPBROADCASTW X1, Y1
	VPCMPGTW Y1, Y6, Y6
	VPCMPGTW Y1, Y7, Y7
	VPOR     Y7, Y6, Y1 // Y1 = hevMask

	// filter = clamp( (hev ? clamp(p1-q1) : 0) + 3*(q0-p0) )
	VPSUBW  Y11, Y8, Y12 // p1-q1
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12
	VPAND   Y1, Y12, Y12 // hev ? clamp : 0
	VPSUBW  Y9, Y10, Y13 // q0-p0
	VPADDW  Y13, Y13, Y14 // 2*(q0-p0)
	VPADDW  Y13, Y14, Y13 // 3*(q0-p0)
	VPADDW  Y13, Y12, Y12
	VPMINSW Y5, Y12, Y12
	VPMAXSW Y4, Y12, Y12 // Y12 = filter

	// all-ones (-1 per lane) in Y2 for the +1 / +3 / +4 reconstructions.
	VPCMPEQW Y2, Y2, Y2

	// filter1 = clamp(filter+4) >> 3. filter+4 = filter - (-4) = filter + (4).
	// build +4 by subtracting (-1) four times is wasteful; use shifts of -1:
	// (-1) is 0xFFFF; -4 = (-1)<<2 == 0xFFFC. So filter - ((-1)<<2) = filter+4.
	VPSLLW  $2, Y2, Y13   // -4
	VPSUBW  Y13, Y12, Y13 // filter + 4
	VPMINSW Y5, Y13, Y13
	VPMAXSW Y4, Y13, Y13
	VPSRAW  $3, Y13, Y13  // Y13 = filter1

	// filter2 = clamp(filter+3) >> 3. +3 = -((-1)<<1) - (-1) ... simpler:
	// filter + 3 = (filter + 4) - 1 = (Y_(filter+4 pre-shift)) - 1, but that was
	// clamped/shifted. Recompute from filter: -3 = ((-1)<<1)+(-1) = 0xFFFD.
	VPADDW  Y2, Y2, Y14   // -2
	VPADDW  Y2, Y14, Y14  // -3
	VPSUBW  Y14, Y12, Y14 // filter + 3
	VPMINSW Y5, Y14, Y14
	VPMAXSW Y4, Y14, Y14
	VPSRAW  $3, Y14, Y14  // Y14 = filter2

	// outer = (filter1 + 1) >> 1.  +1 = -(-1).
	VPSUBW  Y2, Y13, Y15  // filter1 + 1
	VPSRAW  $1, Y15, Y15  // Y15 = outer

	// needMask & ~hev -> Y2 (for the p1/q1 conditional writes).
	VPANDN  Y0, Y1, Y2    // (~hev) & need  == VPANDN(hev, need)

	// ---- p0' = clamp((p0-128)+filter2)+128 ; blend(need) ; store to [R9] ----
	VPSUBW  Y3, Y9, Y6
	VPADDW  Y14, Y6, Y6
	VPMINSW Y5, Y6, Y6
	VPMAXSW Y4, Y6, Y6
	VPADDW  Y3, Y6, Y6        // p0'
	VPBLENDVB Y0, Y6, Y9, Y6  // need ? p0' : p0
	VPACKUSWB Y6, Y6, Y6
	VPERMQ  $0xD8, Y6, Y6     // gather low+high lane bytes into low 128
	VMOVDQU X6, (R9)

	// ---- q0' = clamp((q0-128)-filter1)+128 ; blend(need) ; store to [R10] ----
	VPSUBW  Y3, Y10, Y7
	VPSUBW  Y13, Y7, Y7
	VPMINSW Y5, Y7, Y7
	VPMAXSW Y4, Y7, Y7
	VPADDW  Y3, Y7, Y7        // q0'
	VPBLENDVB Y0, Y7, Y10, Y7 // need ? q0' : q0
	VPACKUSWB Y7, Y7, Y7
	VPERMQ  $0xD8, Y7, Y7
	VMOVDQU X7, (R10)

	// ---- p1' = clamp((p1-128)+outer)+128 ; blend(need&~hev) ; store to [R8] --
	VPSUBW  Y3, Y8, Y6
	VPADDW  Y15, Y6, Y6
	VPMINSW Y5, Y6, Y6
	VPMAXSW Y4, Y6, Y6
	VPADDW  Y3, Y6, Y6        // p1'
	VPBLENDVB Y2, Y6, Y8, Y6  // (need&~hev) ? p1' : p1
	VPACKUSWB Y6, Y6, Y6
	VPERMQ  $0xD8, Y6, Y6
	VMOVDQU X6, (R8)

	// ---- q1' = clamp((q1-128)-outer)+128 ; blend(need&~hev) ; store to [R11] -
	VPSUBW  Y3, Y11, Y7
	VPSUBW  Y15, Y7, Y7
	VPMINSW Y5, Y7, Y7
	VPMAXSW Y4, Y7, Y7
	VPADDW  Y3, Y7, Y7        // q1'
	VPBLENDVB Y2, Y7, Y11, Y7 // (need&~hev) ? q1' : q1
	VPACKUSWB Y7, Y7, Y7
	VPERMQ  $0xD8, Y7, Y7
	VMOVDQU X7, (R11)

	ADDQ  $16, R8
	ADDQ  $16, R9
	ADDQ  $16, R10
	ADDQ  $16, R11
	DECQ  R12
	JMP   loop

done:
	VZEROUPPER
	RET
