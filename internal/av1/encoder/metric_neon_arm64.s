// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// Pixel-domain SSE/signed-sum statistics over a width-multiple-of-eight block:
//   diff = src - ref
//   SSE += diff*diff
//   Sum += diff
//
// The shape mirrors SVT's variance/SSE NEON kernels while keeping one generic
// body for goav1's active square 8/16/32 metric sizes. Instructions missing
// from Go's assembler are emitted as WORD with source mnemonics from clang
// -target arm64-apple-macos:
//
//   usubl  v2.8h,  v0.8b,   v1.8b    -> 0x2e212002
//   usubl2 v3.8h,  v0.16b,  v1.16b   -> 0x6e212003
//   saddlp v4.4s,  v2.8h             -> 0x4e602844
//   saddlp v5.4s,  v3.8h             -> 0x4e602865
//   add    v18.4s, v18.4s, v4.4s     -> 0x4ea48652
//   add    v18.4s, v18.4s, v5.4s     -> 0x4ea58652
//   smlal  v16.4s, v2.4h,  v2.4h     -> 0x0e628050
//   smlal2 v17.4s, v2.8h,  v2.8h     -> 0x4e628051
//   smlal  v16.4s, v3.4h,  v3.4h     -> 0x0e638070
//   smlal2 v17.4s, v3.8h,  v3.8h     -> 0x4e638071

#define M_SRC       0
#define M_SRCSTRIDE 8
#define M_REF       16
#define M_REFSTRIDE 24
#define M_W         32
#define M_H         40
#define M_SSE       48
#define M_SUM       56

// func pixelStatsNEONAsm(ctx *pixelStatsNEONCtx)
TEXT ·pixelStatsNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD M_SRC(R0), R1
	MOVD M_SRCSTRIDE(R0), R2
	MOVD M_REF(R0), R3
	MOVD M_REFSTRIDE(R0), R4
	MOVD M_W(R0), R5
	MOVD M_H(R0), R6

	WORD $0x6e301e10 // eor v16.16b, v16.16b, v16.16b
	WORD $0x6e311e31 // eor v17.16b, v17.16b, v17.16b
	WORD $0x6e321e52 // eor v18.16b, v18.16b, v18.16b

mrow:
	MOVD R1, R8
	MOVD R3, R9
	MOVD R5, R10
mcol16:
	CMP  $16, R10
	BLT  mcol8
	VLD1.P 16(R8), [V0.B16]
	VLD1.P 16(R9), [V1.B16]
	WORD $0x2e212002 // usubl  v2.8h,  v0.8b,  v1.8b
	WORD $0x6e212003 // usubl2 v3.8h,  v0.16b, v1.16b
	WORD $0x4e602844 // saddlp v4.4s,  v2.8h
	WORD $0x4e602865 // saddlp v5.4s,  v3.8h
	WORD $0x4ea48652 // add    v18.4s, v18.4s, v4.4s
	WORD $0x4ea58652 // add    v18.4s, v18.4s, v5.4s
	WORD $0x0e628050 // smlal  v16.4s, v2.4h,  v2.4h
	WORD $0x4e628051 // smlal2 v17.4s, v2.8h,  v2.8h
	WORD $0x0e638070 // smlal  v16.4s, v3.4h,  v3.4h
	WORD $0x4e638071 // smlal2 v17.4s, v3.8h,  v3.8h
	SUB  $16, R10
	B    mcol16
mcol8:
	CBZ  R10, mnext
	VLD1 (R8), [V0.B8]
	VLD1 (R9), [V1.B8]
	WORD $0x2e212002 // usubl  v2.8h,  v0.8b, v1.8b
	WORD $0x4e602844 // saddlp v4.4s,  v2.8h
	WORD $0x4ea48652 // add    v18.4s, v18.4s, v4.4s
	WORD $0x0e628050 // smlal  v16.4s, v2.4h, v2.4h
	WORD $0x4e628051 // smlal2 v17.4s, v2.8h, v2.8h
mnext:
	ADD  R2, R1
	ADD  R4, R3
	SUB  $1, R6
	CBNZ R6, mrow

	WORD $0x4eb18610 // add    v16.4s, v16.4s, v17.4s
	WORD $0x6eb03a00 // uaddlv d0,     v16.4s
	WORD $0x4eb03a41 // saddlv d1,     v18.4s
	VMOV V0.D[0], R7
	MOVD R7, M_SSE(R0)
	VMOV V1.D[0], R7
	MOVD R7, M_SUM(R0)
	RET

// DOTPROD-tier pixel-domain SSE/signed-sum statistics over a width-multiple-of
// eight block. Sum still uses the same widening diff as the baseline NEON
// kernel; SSE is reconstructed as:
//   sum(src*src) + sum(ref*ref) - 2*sum(src*ref)
// This mirrors the DOTPROD variance/SSE surface SVT can select at `--asm max`
// while preserving the same `(SSE, signed residual sum)` contract.
//
//   udot v16.4s, v0.16b, v0.16b -> 0x6e809410
//   udot v17.4s, v1.16b, v1.16b -> 0x6e819431
//   udot v19.4s, v0.16b, v1.16b -> 0x6e819413
//
// func pixelStatsDotProdAsm(ctx *pixelStatsNEONCtx)
TEXT ·pixelStatsDotProdAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD M_SRC(R0), R1
	MOVD M_SRCSTRIDE(R0), R2
	MOVD M_REF(R0), R3
	MOVD M_REFSTRIDE(R0), R4
	MOVD M_W(R0), R5
	MOVD M_H(R0), R6

	WORD $0x6e301e10 // eor v16.16b, v16.16b, v16.16b  src*src
	WORD $0x6e311e31 // eor v17.16b, v17.16b, v17.16b  ref*ref
	WORD $0x6e321e52 // eor v18.16b, v18.16b, v18.16b  residual sum
	WORD $0x6e331e73 // eor v19.16b, v19.16b, v19.16b  src*ref

mdprow:
	MOVD R1, R8
	MOVD R3, R9
	MOVD R5, R10
mdpcol16:
	CMP  $16, R10
	BLT  mdpcol8
	VLD1.P 16(R8), [V0.B16]
	VLD1.P 16(R9), [V1.B16]
	WORD $0x6e809410 // udot v16.4s, v0.16b, v0.16b
	WORD $0x6e819431 // udot v17.4s, v1.16b, v1.16b
	WORD $0x6e819413 // udot v19.4s, v0.16b, v1.16b
	WORD $0x2e212002 // usubl  v2.8h,  v0.8b,  v1.8b
	WORD $0x6e212003 // usubl2 v3.8h,  v0.16b, v1.16b
	WORD $0x4e602844 // saddlp v4.4s,  v2.8h
	WORD $0x4e602865 // saddlp v5.4s,  v3.8h
	WORD $0x4ea48652 // add    v18.4s, v18.4s, v4.4s
	WORD $0x4ea58652 // add    v18.4s, v18.4s, v5.4s
	SUB  $16, R10
	B    mdpcol16
mdpcol8:
	CBZ  R10, mdpnext
	WORD $0x4f000400 // movi v0.4s, #0
	WORD $0x4f000401 // movi v1.4s, #0
	VLD1 (R8), [V0.B8]
	VLD1 (R9), [V1.B8]
	WORD $0x6e809410 // udot v16.4s, v0.16b, v0.16b
	WORD $0x6e819431 // udot v17.4s, v1.16b, v1.16b
	WORD $0x6e819413 // udot v19.4s, v0.16b, v1.16b
	WORD $0x2e212002 // usubl  v2.8h,  v0.8b, v1.8b
	WORD $0x4e602844 // saddlp v4.4s,  v2.8h
	WORD $0x4ea48652 // add    v18.4s, v18.4s, v4.4s
mdpnext:
	ADD  R2, R1
	ADD  R4, R3
	SUB  $1, R6
	CBNZ R6, mdprow

	WORD $0x6eb03a00 // uaddlv d0, v16.4s
	VMOV V0.D[0], R7
	WORD $0x6eb03a20 // uaddlv d0, v17.4s
	VMOV V0.D[0], R8
	WORD $0x6eb03a60 // uaddlv d0, v19.4s
	VMOV V0.D[0], R9
	ADD  R8, R7
	LSL  $1, R9
	SUB  R9, R7
	MOVD R7, M_SSE(R0)
	WORD $0x4eb03a41 // saddlv d1, v18.4s
	VMOV V1.D[0], R7
	MOVD R7, M_SUM(R0)
	RET

// Pixel-domain SSE/signed-sum statistics over a 4-wide block. This mirrors the
// generic width>=8 metric kernel but uses exact four-byte row loads so edge
// probes do not overread the row. The upper four byte lanes are zeroed before
// each row, making the existing 8-lane widening math contribute only lanes 0..3.
//
//   movi v0.4s, #0               -> 0x4f000400
//   movi v1.4s, #0               -> 0x4f000401
//   ld1  {v0.s}[0], [x8]         -> 0x0d408100
//   ld1  {v1.s}[0], [x9]         -> 0x0d408121
//
// func pixelStats4NEONAsm(ctx *pixelStatsNEONCtx)
TEXT ·pixelStats4NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD M_SRC(R0), R1
	MOVD M_SRCSTRIDE(R0), R2
	MOVD M_REF(R0), R3
	MOVD M_REFSTRIDE(R0), R4
	MOVD M_H(R0), R6

	WORD $0x6e301e10 // eor v16.16b, v16.16b, v16.16b
	WORD $0x6e311e31 // eor v17.16b, v17.16b, v17.16b
	WORD $0x6e321e52 // eor v18.16b, v18.16b, v18.16b

m4row:
	MOVD R1, R8
	MOVD R3, R9
	WORD $0x4f000400 // movi v0.4s, #0
	WORD $0x4f000401 // movi v1.4s, #0
	WORD $0x0d408100 // ld1 {v0.s}[0], [x8]
	WORD $0x0d408121 // ld1 {v1.s}[0], [x9]
	WORD $0x2e212002 // usubl  v2.8h,  v0.8b, v1.8b
	WORD $0x4e602844 // saddlp v4.4s,  v2.8h
	WORD $0x4ea48652 // add    v18.4s, v18.4s, v4.4s
	WORD $0x0e628050 // smlal  v16.4s, v2.4h, v2.4h
	WORD $0x4e628051 // smlal2 v17.4s, v2.8h, v2.8h
	ADD  R2, R1
	ADD  R4, R3
	SUB  $1, R6
	CBNZ R6, m4row

	WORD $0x4eb18610 // add    v16.4s, v16.4s, v17.4s
	WORD $0x6eb03a00 // uaddlv d0,     v16.4s
	WORD $0x4eb03a41 // saddlv d1,     v18.4s
	VMOV V0.D[0], R7
	MOVD R7, M_SSE(R0)
	VMOV V1.D[0], R7
	MOVD R7, M_SUM(R0)
	RET
