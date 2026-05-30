// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// AVX2 per-column dequant kernel plus a CPUID/XGETBV feature probe. The kernel
// is bit-exact with dequantColumnPureGo (see dequant.go); the Go wrapper in
// dequant_avx2_amd64.go handles the n%8 tail with the scalar reference and
// never invokes the asm when blocks8 == 0. Every mnemonic below is accepted
// directly by the Go assembler at this GOARCH, so no manual encoding is needed.

//go:build amd64 && !purego

#include "textflag.h"

#define DST 0
#define COEFF 8
#define BLOCKS8 16
#define SCALE 24
#define TXSCALE 32
#define DQMIN 40
#define DQMAX 48

// func dequantColumnAVX2Asm(ctx *dequantColumnAVX2Ctx)
TEXT ·dequantColumnAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ DST(AX), DI
	MOVQ COEFF(AX), SI
	MOVQ BLOCKS8(AX), CX

	// Broadcast the scalar parameters into YMM lanes.
	VPBROADCASTD SCALE(AX), Y4 // scale in every int32 lane

	MOVL $0x00ffffff, DX
	MOVQ DX, X5
	VPBROADCASTD X5, Y5 // 24-bit mask

	VPBROADCASTD DQMIN(AX), Y6 // dqMin
	VPBROADCASTD DQMAX(AX), Y7 // dqMax

	// txScale count for VPSRAD must live in the low 64 bits of an XMM. The
	// upper bytes are ignored by the shift; load the full 64-bit field.
	MOVQ TXSCALE(AX), DX
	MOVQ DX, X8

	TESTQ CX, CX
	JZ    done

loop:
	// Load 8 int16 coefficients and sign-extend to 8 int32 lanes.
	VPMOVSXWD (SI), Y0

	// signmask = arithmetic (orig >> 31): all-ones where negative.
	VPSRAD $31, Y0, Y1

	// magnitude = |coeff|, INT16_MIN-safe in int32 lanes.
	VPABSD Y0, Y2

	// product = (magnitude * scale) low 32 bits.
	VPMULLD Y4, Y2, Y2

	// product &= 0x00ffffff (libaom tran_low_t mask).
	VPAND Y5, Y2, Y2

	// product >>= txScale (arithmetic; product is non-negative so it equals
	// a logical shift and matches dequantScalar).
	VPSRAD X8, Y2, Y2

	// reapply sign: (product ^ signmask) - signmask.
	VPXOR  Y1, Y2, Y2
	VPSUBD Y1, Y2, Y2

	// clamp to [dqMin, dqMax].
	VPMINSD Y7, Y2, Y2
	VPMAXSD Y6, Y2, Y2

	VMOVDQU Y2, (DI)

	ADDQ $16, SI // advance coeff by 8 int16
	ADDQ $32, DI // advance dst by 8 int32
	DECQ CX
	JNZ  loop

done:
	VZEROUPPER
	RET

// func cpuidQuantize(eax, ecx uint32) (a, b, c, d uint32)
TEXT ·cpuidQuantize(SB), NOSPLIT, $0-24
	MOVL eax+0(FP), AX
	MOVL ecx+4(FP), CX
	CPUID
	MOVL AX, a+8(FP)
	MOVL BX, b+12(FP)
	MOVL CX, c+16(FP)
	MOVL DX, d+20(FP)
	RET

// func xgetbvQuantize() (lo uint32)
TEXT ·xgetbvQuantize(SB), NOSPLIT, $0-4
	MOVL  $0, CX
	XGETBV
	MOVL  AX, lo+0(FP)
	RET
