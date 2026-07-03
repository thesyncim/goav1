// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// AVX2 CfL luma subsampling and directional Z1/Z2/Z3 edge interpolation. The
// kernels are bit-exact with the *PureGo references in intra_kernels.go; the Go
// wrappers in intra_kernels_avx2_amd64.go restrict them to the 8-bit,
// width-multiple-of-8 shapes they handle. They mirror the arm64 NEON kernels in
// intra_kernels_neon_arm64.s.
//
// dav1d reference: third_party/upstream/dav1d/src/x86/ipred_avx2.asm
// (ipred_cfl_ac_420/422/444, ipred_z1/z2/z3).

//go:build amd64 && !purego

#include "textflag.h"

// func subsampleLuma8AVX2Asm(out *uint16, top *uint8, bot *uint8, mode uint64)
//
// Produces 8 Q3 output samples. mode: 0 = no subsample (x<<3), 1 = horizontal
// ((a+b)<<2), 2 = 2x2 ((a+b+c+d)<<1). bot is only read for mode 2. The pairwise
// byte adds use VPMADDUBSW against a vector of ones.
TEXT ·subsampleLuma8AVX2Asm(SB), NOSPLIT, $0-32
	MOVQ out+0(FP), DI
	MOVQ top+8(FP), SI
	MOVQ bot+16(FP), BX
	MOVQ mode+24(FP), R8

	MOVL         $0x01010101, R9
	MOVQ         R9, X15
	VPBROADCASTD X15, X15 // ones (16 bytes of 0x01)

	CMPQ R8, $2
	JEQ  ssXY
	CMPQ R8, $1
	JEQ  ssX

	// mode 0: out = top<<3
	VPMOVZXBW (SI), X0 // 8 bytes -> 8 u16
	VPSLLW    $3, X0, X0
	VMOVDQU   X0, (DI)
	RET

ssX:
	// mode 1: out = (a+b)<<2  (pairwise adjacent)
	VMOVDQU    (SI), X0     // 16 bytes
	VPMADDUBSW X15, X0, X0  // pairwise a+b -> 8 i16
	VPSLLW     $2, X0, X0
	VMOVDQU    X0, (DI)
	RET

ssXY:
	// mode 2: out = (a+b+c+d)<<1
	VMOVDQU    (SI), X0
	VMOVDQU    (BX), X1
	VPMADDUBSW X15, X0, X0
	VPMADDUBSW X15, X1, X1
	VPADDW     X1, X0, X0
	VPSLLW     $1, X0, X0
	VMOVDQU    X0, (DI)
	RET

// func dirInterp8AVX2Asm(dst *byte, src *uint16, weight32 uint64, weightShift uint64, n uint64)
//
// Fills n (multiple of 8) destination bytes with
//   roundPowerOfTwo(p0*(32-shift) + p1*shift, 5)
// where p0 = src[i], p1 = src[i+1]. weight32 = 32-shift, weightShift = shift.
// Callers only invoke this over runs entirely inside the interpolation region
// (src[i] and src[i+1] always valid, no clamp). Serves both dirRowInterp8 (Z1
// row) and dirAboveRun8 (Z2 above run): identical arithmetic.
TEXT ·dirInterp8AVX2Asm(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ weight32+16(FP), R8
	MOVQ weightShift+24(FP), R9
	MOVQ n+32(FP), R10

	MOVQ         R8, X0
	VPBROADCASTD X0, Y0 // 32-shift (u32)
	MOVQ         R9, X1
	VPBROADCASTD X1, Y1 // shift (u32)
	MOVL         $16, R11
	MOVQ         R11, X2
	VPBROADCASTD X2, Y2 // 16 (rounding)

diLoop:
	VPMOVZXWD (SI), Y3    // p0 = src[i..i+7]
	VPMOVZXWD 2(SI), Y4   // p1 = src[i+1..i+8]
	VPMULLD   Y0, Y3, Y3  // p0*(32-shift)
	VPMULLD   Y1, Y4, Y4  // p1*shift
	VPADDD    Y4, Y3, Y3
	VPADDD    Y2, Y3, Y3  // +16
	VPSRLD    $5, Y3, Y3
	VPACKUSDW Y3, Y3, Y3
	VPERMQ    $0xD8, Y3, Y3
	VPACKUSWB Y3, Y3, Y3
	MOVQ      X3, (DI)    // store 8 bytes
	ADDQ $16, SI
	ADDQ $8, DI
	SUBQ $8, R10
	JNZ  diLoop
	VZEROUPPER
	RET

// func dirLeftCol8AVX2Asm(dst *byte, stride uint64, ref *uint16, weight32 uint64, weightShift uint64, n uint64)
//
// Fills n (multiple of 8) destination samples down a single zone-3 column with
//   roundPowerOfTwo(ref[i]*(32-shift) + ref[i+1]*shift, 5)
// where the source index advances by one per row (a forward, contiguous run
// over the pre-validated left edge, so no clamp). The 8 interpolated bytes per
// iteration are scattered into the column with strided single-byte stores.
TEXT ·dirLeftCol8AVX2Asm(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), DI
	MOVQ stride+8(FP), R11
	MOVQ ref+16(FP), SI
	MOVQ weight32+24(FP), R8
	MOVQ weightShift+32(FP), R9
	MOVQ n+40(FP), R10

	MOVQ         R8, X0
	VPBROADCASTD X0, Y0 // 32-shift
	MOVQ         R9, X1
	VPBROADCASTD X1, Y1 // shift
	MOVL         $16, R12
	MOVQ         R12, X2
	VPBROADCASTD X2, Y2 // 16

lcLoop:
	VPMOVZXWD (SI), Y3
	VPMOVZXWD 2(SI), Y4
	VPMULLD   Y0, Y3, Y3
	VPMULLD   Y1, Y4, Y4
	VPADDD    Y4, Y3, Y3
	VPADDD    Y2, Y3, Y3
	VPSRLD    $5, Y3, Y3
	VPACKUSDW Y3, Y3, Y3
	VPERMQ    $0xD8, Y3, Y3
	VPACKUSWB Y3, Y3, Y3
	VPEXTRB   $0, X3, (DI)
	ADDQ      R11, DI
	VPEXTRB   $1, X3, (DI)
	ADDQ      R11, DI
	VPEXTRB   $2, X3, (DI)
	ADDQ      R11, DI
	VPEXTRB   $3, X3, (DI)
	ADDQ      R11, DI
	VPEXTRB   $4, X3, (DI)
	ADDQ      R11, DI
	VPEXTRB   $5, X3, (DI)
	ADDQ      R11, DI
	VPEXTRB   $6, X3, (DI)
	ADDQ      R11, DI
	VPEXTRB   $7, X3, (DI)
	ADDQ      R11, DI
	ADDQ $16, SI
	SUBQ $8, R10
	JNZ  lcLoop
	VZEROUPPER
	RET
