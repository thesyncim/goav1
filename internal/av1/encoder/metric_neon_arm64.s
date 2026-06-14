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

// SVT-shaped coefficient SATD reducer:
//   for each int32 coefficient: sum += abs(coeff[i])
//
// SVT's `svt_aom_satd_neon` processes 16 TranLow coefficients per iteration.
// This mirrors that loop shape for the AV1 TX sizes {16,64,256,1024}.
//
//   abs   v0.4s,  v0.4s             -> 0x4ea0b800
//   abs   v1.4s,  v1.4s             -> 0x4ea0b821
//   abs   v2.4s,  v2.4s             -> 0x4ea0b842
//   abs   v3.4s,  v3.4s             -> 0x4ea0b863
//   add   v16.4s, v16.4s, v0.4s     -> 0x4ea08610
//   add   v16.4s, v16.4s, v1.4s     -> 0x4ea18610
//   add   v16.4s, v16.4s, v2.4s     -> 0x4ea28610
//   add   v16.4s, v16.4s, v3.4s     -> 0x4ea38610
//   saddlv d0,     v16.4s           -> 0x4eb03a00

#define S_COEFF 0
#define S_COUNT 8
#define S_SUM   16

// func satdCoeffsNEONAsm(ctx *satdCoeffsNEONCtx)
TEXT ·satdCoeffsNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD S_COEFF(R0), R1
	MOVD S_COUNT(R0), R2
	WORD $0x6e301e10 // eor v16.16b, v16.16b, v16.16b
	LSR  $4, R2
sloop:
	VLD1.P 16(R1), [V0.S4]
	VLD1.P 16(R1), [V1.S4]
	VLD1.P 16(R1), [V2.S4]
	VLD1.P 16(R1), [V3.S4]
	WORD $0x4ea0b800 // abs v0.4s, v0.4s
	WORD $0x4ea0b821 // abs v1.4s, v1.4s
	WORD $0x4ea0b842 // abs v2.4s, v2.4s
	WORD $0x4ea0b863 // abs v3.4s, v3.4s
	WORD $0x4ea08610 // add v16.4s, v16.4s, v0.4s
	WORD $0x4ea18610 // add v16.4s, v16.4s, v1.4s
	WORD $0x4ea28610 // add v16.4s, v16.4s, v2.4s
	WORD $0x4ea38610 // add v16.4s, v16.4s, v3.4s
	SUB  $1, R2
	CBNZ R2, sloop
	WORD $0x4eb03a00 // saddlv d0, v16.4s
	VMOV V0.D[0], R3
	MOVD R3, S_SUM(R0)
	RET

// SVT-shaped low-bitdepth 8x8 Hadamard producer. This mirrors
// svt_aom_hadamard_8x8_neon: load eight int16 residual rows, run the 8-point
// Hadamard butterfly, transpose, rerun the same butterfly, and sign-extend
// the int16 outputs to int32 coefficients in SVT's NEON order. That order is
// the transpose of svt_aom_hadamard_8x8_c, which SVT permits because SATD is
// order-invariant.

#define H_SRC       0
#define H_SRCSTRIDE 8
#define H_COEFF     16

// SVT-shaped low-bitdepth 4x4 Hadamard producer. This mirrors
// svt_aom_hadamard_4x4_neon: load four int16 residual rows, run the 4-point
// signed-halving Hadamard butterfly, transpose, rerun the butterfly, and
// sign-extend the int16 outputs to int32 coefficients.
//
//   shadd v16.4h, v0.4h,  v1.4h  -> 0x0e610410
//   shsub v17.4h, v0.4h,  v1.4h  -> 0x0e612411
//   shadd v18.4h, v2.4h,  v3.4h  -> 0x0e630452
//   shsub v19.4h, v2.4h,  v3.4h  -> 0x0e632453
//   add   v0.4h,  v16.4h, v18.4h -> 0x0e728600
//   add   v1.4h,  v17.4h, v19.4h -> 0x0e738621
//   sub   v2.4h,  v16.4h, v18.4h -> 0x2e728602
//   sub   v3.4h,  v17.4h, v19.4h -> 0x2e738623
//   trn1  v16.4h, v0.4h,  v1.4h  -> 0x0e412810
//   trn2  v17.4h, v0.4h,  v1.4h  -> 0x0e416811
//   trn1  v18.4h, v2.4h,  v3.4h  -> 0x0e432852
//   trn2  v19.4h, v2.4h,  v3.4h  -> 0x0e436853
//   trn1  v0.2s,  v16.2s, v18.2s -> 0x0e922a00
//   trn2  v2.2s,  v16.2s, v18.2s -> 0x0e926a02
//   trn1  v1.2s,  v17.2s, v19.2s -> 0x0e932a21
//   trn2  v3.2s,  v17.2s, v19.2s -> 0x0e936a23
//   sshll v16.4s, v0.4h,  #0     -> 0x0f10a410
//   sshll v17.4s, v1.4h,  #0     -> 0x0f10a431
//   sshll v18.4s, v2.4h,  #0     -> 0x0f10a452
//   sshll v19.4s, v3.4h,  #0     -> 0x0f10a473

// func hadamard4x4NEONAsm(ctx *hadamard8x8NEONCtx)
TEXT ·hadamard4x4NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD H_SRC(R0), R1
	MOVD H_SRCSTRIDE(R0), R2
	MOVD H_COEFF(R0), R3
	LSL  $1, R2

	VLD1 (R1), [V0.H4]
	ADD  R2, R1
	VLD1 (R1), [V1.H4]
	ADD  R2, R1
	VLD1 (R1), [V2.H4]
	ADD  R2, R1
	VLD1 (R1), [V3.H4]

	WORD $0x0e610410 // shadd v16.4h, v0.4h, v1.4h
	WORD $0x0e612411 // shsub v17.4h, v0.4h, v1.4h
	WORD $0x0e630452 // shadd v18.4h, v2.4h, v3.4h
	WORD $0x0e632453 // shsub v19.4h, v2.4h, v3.4h
	WORD $0x0e728600 // add v0.4h, v16.4h, v18.4h
	WORD $0x0e738621 // add v1.4h, v17.4h, v19.4h
	WORD $0x2e728602 // sub v2.4h, v16.4h, v18.4h
	WORD $0x2e738623 // sub v3.4h, v17.4h, v19.4h

	WORD $0x0e412810 // trn1 v16.4h, v0.4h, v1.4h
	WORD $0x0e416811 // trn2 v17.4h, v0.4h, v1.4h
	WORD $0x0e432852 // trn1 v18.4h, v2.4h, v3.4h
	WORD $0x0e436853 // trn2 v19.4h, v2.4h, v3.4h
	WORD $0x0e922a00 // trn1 v0.2s, v16.2s, v18.2s
	WORD $0x0e926a02 // trn2 v2.2s, v16.2s, v18.2s
	WORD $0x0e932a21 // trn1 v1.2s, v17.2s, v19.2s
	WORD $0x0e936a23 // trn2 v3.2s, v17.2s, v19.2s

	WORD $0x0e610410 // shadd v16.4h, v0.4h, v1.4h
	WORD $0x0e612411 // shsub v17.4h, v0.4h, v1.4h
	WORD $0x0e630452 // shadd v18.4h, v2.4h, v3.4h
	WORD $0x0e632453 // shsub v19.4h, v2.4h, v3.4h
	WORD $0x0e728600 // add v0.4h, v16.4h, v18.4h
	WORD $0x0e738621 // add v1.4h, v17.4h, v19.4h
	WORD $0x2e728602 // sub v2.4h, v16.4h, v18.4h
	WORD $0x2e738623 // sub v3.4h, v17.4h, v19.4h

	WORD $0x0f10a410 // sshll v16.4s, v0.4h, #0
	WORD $0x0f10a431 // sshll v17.4s, v1.4h, #0
	WORD $0x0f10a452 // sshll v18.4s, v2.4h, #0
	WORD $0x0f10a473 // sshll v19.4s, v3.4h, #0
	VST1 [V16.S4], (R3)
	ADD  $16, R3
	VST1 [V17.S4], (R3)
	ADD  $16, R3
	VST1 [V18.S4], (R3)
	ADD  $16, R3
	VST1 [V19.S4], (R3)
	RET

// func hadamard8x8NEONAsm(ctx *hadamard8x8NEONCtx)
TEXT ·hadamard8x8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD H_SRC(R0), R1
	MOVD H_SRCSTRIDE(R0), R2
	MOVD H_COEFF(R0), R3
	LSL  $1, R2

	VLD1 (R1), [V0.H8]
	ADD  R2, R1
	VLD1 (R1), [V1.H8]
	ADD  R2, R1
	VLD1 (R1), [V2.H8]
	ADD  R2, R1
	VLD1 (R1), [V3.H8]
	ADD  R2, R1
	VLD1 (R1), [V4.H8]
	ADD  R2, R1
	VLD1 (R1), [V5.H8]
	ADD  R2, R1
	VLD1 (R1), [V6.H8]
	ADD  R2, R1
	VLD1 (R1), [V7.H8]

	WORD $0x4e618410 // add v16.8h, v0.8h, v1.8h
	WORD $0x6e618411 // sub v17.8h, v0.8h, v1.8h
	WORD $0x4e638452 // add v18.8h, v2.8h, v3.8h
	WORD $0x6e638453 // sub v19.8h, v2.8h, v3.8h
	WORD $0x4e658494 // add v20.8h, v4.8h, v5.8h
	WORD $0x6e658495 // sub v21.8h, v4.8h, v5.8h
	WORD $0x4e6784d6 // add v22.8h, v6.8h, v7.8h
	WORD $0x6e6784d7 // sub v23.8h, v6.8h, v7.8h
	WORD $0x4e728618 // add v24.8h, v16.8h, v18.8h
	WORD $0x4e738639 // add v25.8h, v17.8h, v19.8h
	WORD $0x6e72861a // sub v26.8h, v16.8h, v18.8h
	WORD $0x6e73863b // sub v27.8h, v17.8h, v19.8h
	WORD $0x4e76869c // add v28.8h, v20.8h, v22.8h
	WORD $0x4e7786bd // add v29.8h, v21.8h, v23.8h
	WORD $0x6e76869e // sub v30.8h, v20.8h, v22.8h
	WORD $0x6e7786bf // sub v31.8h, v21.8h, v23.8h
	WORD $0x4e7c8700 // add v0.8h, v24.8h, v28.8h
	WORD $0x6e7e8741 // sub v1.8h, v26.8h, v30.8h
	WORD $0x6e7c8702 // sub v2.8h, v24.8h, v28.8h
	WORD $0x4e7e8743 // add v3.8h, v26.8h, v30.8h
	WORD $0x4e7f8764 // add v4.8h, v27.8h, v31.8h
	WORD $0x6e7f8765 // sub v5.8h, v27.8h, v31.8h
	WORD $0x6e7d8726 // sub v6.8h, v25.8h, v29.8h
	WORD $0x4e7d8727 // add v7.8h, v25.8h, v29.8h

	WORD $0x4e412810 // trn1 v16.8h, v0.8h, v1.8h
	WORD $0x4e416811 // trn2 v17.8h, v0.8h, v1.8h
	WORD $0x4e432852 // trn1 v18.8h, v2.8h, v3.8h
	WORD $0x4e436853 // trn2 v19.8h, v2.8h, v3.8h
	WORD $0x4e452894 // trn1 v20.8h, v4.8h, v5.8h
	WORD $0x4e456895 // trn2 v21.8h, v4.8h, v5.8h
	WORD $0x4e4728d6 // trn1 v22.8h, v6.8h, v7.8h
	WORD $0x4e4768d7 // trn2 v23.8h, v6.8h, v7.8h
	WORD $0x4e922a18 // trn1 v24.4s, v16.4s, v18.4s
	WORD $0x4e926a19 // trn2 v25.4s, v16.4s, v18.4s
	WORD $0x4e932a3a // trn1 v26.4s, v17.4s, v19.4s
	WORD $0x4e936a3b // trn2 v27.4s, v17.4s, v19.4s
	WORD $0x4e962a9c // trn1 v28.4s, v20.4s, v22.4s
	WORD $0x4e966a9d // trn2 v29.4s, v20.4s, v22.4s
	WORD $0x4e972abe // trn1 v30.4s, v21.4s, v23.4s
	WORD $0x4e976abf // trn2 v31.4s, v21.4s, v23.4s
	WORD $0x4edc3b00 // zip1 v0.2d, v24.2d, v28.2d
	WORD $0x4ede3b41 // zip1 v1.2d, v26.2d, v30.2d
	WORD $0x4edd3b22 // zip1 v2.2d, v25.2d, v29.2d
	WORD $0x4edf3b63 // zip1 v3.2d, v27.2d, v31.2d
	WORD $0x4edc7b04 // zip2 v4.2d, v24.2d, v28.2d
	WORD $0x4ede7b45 // zip2 v5.2d, v26.2d, v30.2d
	WORD $0x4edd7b26 // zip2 v6.2d, v25.2d, v29.2d
	WORD $0x4edf7b67 // zip2 v7.2d, v27.2d, v31.2d

	WORD $0x4e618410 // add v16.8h, v0.8h, v1.8h
	WORD $0x6e618411 // sub v17.8h, v0.8h, v1.8h
	WORD $0x4e638452 // add v18.8h, v2.8h, v3.8h
	WORD $0x6e638453 // sub v19.8h, v2.8h, v3.8h
	WORD $0x4e658494 // add v20.8h, v4.8h, v5.8h
	WORD $0x6e658495 // sub v21.8h, v4.8h, v5.8h
	WORD $0x4e6784d6 // add v22.8h, v6.8h, v7.8h
	WORD $0x6e6784d7 // sub v23.8h, v6.8h, v7.8h
	WORD $0x4e728618 // add v24.8h, v16.8h, v18.8h
	WORD $0x4e738639 // add v25.8h, v17.8h, v19.8h
	WORD $0x6e72861a // sub v26.8h, v16.8h, v18.8h
	WORD $0x6e73863b // sub v27.8h, v17.8h, v19.8h
	WORD $0x4e76869c // add v28.8h, v20.8h, v22.8h
	WORD $0x4e7786bd // add v29.8h, v21.8h, v23.8h
	WORD $0x6e76869e // sub v30.8h, v20.8h, v22.8h
	WORD $0x6e7786bf // sub v31.8h, v21.8h, v23.8h
	WORD $0x4e7c8700 // add v0.8h, v24.8h, v28.8h
	WORD $0x6e7e8741 // sub v1.8h, v26.8h, v30.8h
	WORD $0x6e7c8702 // sub v2.8h, v24.8h, v28.8h
	WORD $0x4e7e8743 // add v3.8h, v26.8h, v30.8h
	WORD $0x4e7f8764 // add v4.8h, v27.8h, v31.8h
	WORD $0x6e7f8765 // sub v5.8h, v27.8h, v31.8h
	WORD $0x6e7d8726 // sub v6.8h, v25.8h, v29.8h
	WORD $0x4e7d8727 // add v7.8h, v25.8h, v29.8h

	WORD $0x0f10a410 // sshll v16.4s, v0.4h, #0
	WORD $0x4f10a411 // sshll2 v17.4s, v0.8h, #0
	VST1 [V16.S4], (R3)
	ADD  $16, R3
	VST1 [V17.S4], (R3)
	ADD  $16, R3
	WORD $0x0f10a430 // sshll v16.4s, v1.4h, #0
	WORD $0x4f10a431 // sshll2 v17.4s, v1.8h, #0
	VST1 [V16.S4], (R3)
	ADD  $16, R3
	VST1 [V17.S4], (R3)
	ADD  $16, R3
	WORD $0x0f10a450 // sshll v16.4s, v2.4h, #0
	WORD $0x4f10a451 // sshll2 v17.4s, v2.8h, #0
	VST1 [V16.S4], (R3)
	ADD  $16, R3
	VST1 [V17.S4], (R3)
	ADD  $16, R3
	WORD $0x0f10a470 // sshll v16.4s, v3.4h, #0
	WORD $0x4f10a471 // sshll2 v17.4s, v3.8h, #0
	VST1 [V16.S4], (R3)
	ADD  $16, R3
	VST1 [V17.S4], (R3)
	ADD  $16, R3
	WORD $0x0f10a490 // sshll v16.4s, v4.4h, #0
	WORD $0x4f10a491 // sshll2 v17.4s, v4.8h, #0
	VST1 [V16.S4], (R3)
	ADD  $16, R3
	VST1 [V17.S4], (R3)
	ADD  $16, R3
	WORD $0x0f10a4b0 // sshll v16.4s, v5.4h, #0
	WORD $0x4f10a4b1 // sshll2 v17.4s, v5.8h, #0
	VST1 [V16.S4], (R3)
	ADD  $16, R3
	VST1 [V17.S4], (R3)
	ADD  $16, R3
	WORD $0x0f10a4d0 // sshll v16.4s, v6.4h, #0
	WORD $0x4f10a4d1 // sshll2 v17.4s, v6.8h, #0
	VST1 [V16.S4], (R3)
	ADD  $16, R3
	VST1 [V17.S4], (R3)
	ADD  $16, R3
	WORD $0x0f10a4f0 // sshll v16.4s, v7.4h, #0
	WORD $0x4f10a4f1 // sshll2 v17.4s, v7.8h, #0
	VST1 [V16.S4], (R3)
	ADD  $16, R3
	VST1 [V17.S4], (R3)
	RET

// SVT-shaped 16x16 Hadamard quadrant combine. The caller has already produced
// four 8x8 NEON-order coefficient quadrants at offsets 0, 64, 128, and 192.
// This mirrors svt_aom_hadamard_16x16_neon's signed halving add/sub combine
// and row-quarter store order.
//
//   shadd v4.4s, v0.4s, v1.4s  -> 0x4ea10404
//   shsub v5.4s, v0.4s, v1.4s  -> 0x4ea12405
//   shadd v6.4s, v2.4s, v3.4s  -> 0x4ea30446
//   shsub v7.4s, v2.4s, v3.4s  -> 0x4ea32447

// func hadamard16x16CombineNEONAsm(coeff unsafe.Pointer)
TEXT ·hadamard16x16CombineNEONAsm(SB), NOSPLIT, $0-8
	MOVD coeff+0(FP), R0
	MOVD $4, R1
h16loop:
	MOVD R0, R4
	MOVD R0, R8
	MOVD R0, R5
	ADD  $256, R5
	MOVD R5, R9
	MOVD R0, R6
	ADD  $512, R6
	MOVD R6, R10
	MOVD R0, R7
	ADD  $768, R7
	MOVD R7, R11

	VLD1 (R4), [V0.S4]
	VLD1 (R5), [V1.S4]
	VLD1 (R6), [V2.S4]
	VLD1 (R7), [V3.S4]
	WORD $0x4ea10404 // shadd v4.4s, v0.4s, v1.4s
	WORD $0x4ea12405 // shsub v5.4s, v0.4s, v1.4s
	WORD $0x4ea30446 // shadd v6.4s, v2.4s, v3.4s
	WORD $0x4ea32447 // shsub v7.4s, v2.4s, v3.4s
	VADD V6.S4, V4.S4, V16.S4
	VADD V7.S4, V5.S4, V17.S4
	VSUB V6.S4, V4.S4, V18.S4
	VSUB V7.S4, V5.S4, V19.S4

	ADD $16, R4
	ADD $16, R5
	ADD $16, R6
	ADD $16, R7
	VLD1 (R4), [V0.S4]
	VLD1 (R5), [V1.S4]
	VLD1 (R6), [V2.S4]
	VLD1 (R7), [V3.S4]
	WORD $0x4ea10404 // shadd v4.4s, v0.4s, v1.4s
	WORD $0x4ea12405 // shsub v5.4s, v0.4s, v1.4s
	WORD $0x4ea30446 // shadd v6.4s, v2.4s, v3.4s
	WORD $0x4ea32447 // shsub v7.4s, v2.4s, v3.4s
	VADD V6.S4, V4.S4, V20.S4
	VADD V7.S4, V5.S4, V21.S4
	VSUB V6.S4, V4.S4, V22.S4
	VSUB V7.S4, V5.S4, V23.S4

	ADD $16, R4
	ADD $16, R5
	ADD $16, R6
	ADD $16, R7
	VLD1 (R4), [V0.S4]
	VLD1 (R5), [V1.S4]
	VLD1 (R6), [V2.S4]
	VLD1 (R7), [V3.S4]
	WORD $0x4ea10404 // shadd v4.4s, v0.4s, v1.4s
	WORD $0x4ea12405 // shsub v5.4s, v0.4s, v1.4s
	WORD $0x4ea30446 // shadd v6.4s, v2.4s, v3.4s
	WORD $0x4ea32447 // shsub v7.4s, v2.4s, v3.4s
	VADD V6.S4, V4.S4, V24.S4
	VADD V7.S4, V5.S4, V25.S4
	VSUB V6.S4, V4.S4, V26.S4
	VSUB V7.S4, V5.S4, V27.S4

	ADD $16, R4
	ADD $16, R5
	ADD $16, R6
	ADD $16, R7
	VLD1 (R4), [V0.S4]
	VLD1 (R5), [V1.S4]
	VLD1 (R6), [V2.S4]
	VLD1 (R7), [V3.S4]
	WORD $0x4ea10404 // shadd v4.4s, v0.4s, v1.4s
	WORD $0x4ea12405 // shsub v5.4s, v0.4s, v1.4s
	WORD $0x4ea30446 // shadd v6.4s, v2.4s, v3.4s
	WORD $0x4ea32447 // shsub v7.4s, v2.4s, v3.4s
	VADD V6.S4, V4.S4, V28.S4
	VADD V7.S4, V5.S4, V29.S4
	VSUB V6.S4, V4.S4, V30.S4
	VSUB V7.S4, V5.S4, V31.S4

	VST1 [V16.S4], (R8)
	MOVD R8, R12
	ADD  $16, R12
	VST1 [V24.S4], (R12)
	MOVD R8, R12
	ADD  $32, R12
	VST1 [V20.S4], (R12)
	MOVD R8, R12
	ADD  $48, R12
	VST1 [V28.S4], (R12)

	VST1 [V17.S4], (R9)
	MOVD R9, R12
	ADD  $16, R12
	VST1 [V25.S4], (R12)
	MOVD R9, R12
	ADD  $32, R12
	VST1 [V21.S4], (R12)
	MOVD R9, R12
	ADD  $48, R12
	VST1 [V29.S4], (R12)

	VST1 [V18.S4], (R10)
	MOVD R10, R12
	ADD  $16, R12
	VST1 [V26.S4], (R12)
	MOVD R10, R12
	ADD  $32, R12
	VST1 [V22.S4], (R12)
	MOVD R10, R12
	ADD  $48, R12
	VST1 [V30.S4], (R12)

	VST1 [V19.S4], (R11)
	MOVD R11, R12
	ADD  $16, R12
	VST1 [V27.S4], (R12)
	MOVD R11, R12
	ADD  $32, R12
	VST1 [V23.S4], (R12)
	MOVD R11, R12
	ADD  $48, R12
	VST1 [V31.S4], (R12)

	ADD $64, R0
	SUB $1, R1
	CBNZ R1, h16loop
	RET

// SVT-shaped 32x32 Hadamard quadrant combine. The caller has already produced
// four 16x16 NEON-order coefficient quadrants at offsets 0, 256, 512, and 768.
// This mirrors svt_aom_hadamard_32x32_neon's add/sub, signed shift-right by
// two, and final quadrant stores.
//
//   sshr v4.4s, v4.4s, #2 -> 0x4f3e0484
//   sshr v5.4s, v5.4s, #2 -> 0x4f3e04a5
//   sshr v6.4s, v6.4s, #2 -> 0x4f3e04c6
//   sshr v7.4s, v7.4s, #2 -> 0x4f3e04e7

// func hadamard32x32CombineNEONAsm(coeff unsafe.Pointer)
TEXT ·hadamard32x32CombineNEONAsm(SB), NOSPLIT, $0-8
	MOVD coeff+0(FP), R0
	MOVD $64, R1
h32loop:
	MOVD R0, R2
	ADD  $1024, R2
	MOVD R0, R3
	ADD  $2048, R3
	MOVD R0, R4
	ADD  $3072, R4

	VLD1 (R0), [V0.S4]
	VLD1 (R2), [V1.S4]
	VLD1 (R3), [V2.S4]
	VLD1 (R4), [V3.S4]

	VADD V1.S4, V0.S4, V4.S4
	VSUB V1.S4, V0.S4, V5.S4
	VADD V3.S4, V2.S4, V6.S4
	VSUB V3.S4, V2.S4, V7.S4
	WORD $0x4f3e0484 // sshr v4.4s, v4.4s, #2
	WORD $0x4f3e04a5 // sshr v5.4s, v5.4s, #2
	WORD $0x4f3e04c6 // sshr v6.4s, v6.4s, #2
	WORD $0x4f3e04e7 // sshr v7.4s, v7.4s, #2

	VADD V6.S4, V4.S4, V16.S4
	VADD V7.S4, V5.S4, V17.S4
	VSUB V6.S4, V4.S4, V18.S4
	VSUB V7.S4, V5.S4, V19.S4

	VST1 [V16.S4], (R0)
	VST1 [V17.S4], (R2)
	VST1 [V18.S4], (R3)
	VST1 [V19.S4], (R4)

	ADD $16, R0
	SUB $1, R1
	CBNZ R1, h32loop
	RET
