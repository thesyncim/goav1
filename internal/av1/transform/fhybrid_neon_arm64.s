// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

#define CTXIN        0
#define CTXINSTRIDE  8
#define CTXOUT      16
#define CTXOUTSTRIDE 24

// NEON forward 8x8 IDTX. SVT's svt_av1_fwd_txfm2d_8x8_neon handles IDTX by
// running identity in both dimensions around the common 8x8 load/store path.
// For AV1's fwd_shift_8x8={2,-1,0}, two identity passes collapse to residual
// << 3. The output is stored in decoder coefficient layout:
// coeff[c*stride + r] = residual[r*residualStride + c] << 3.

// func fidtx8x8NEONAsm(ctx *fdct8x8NEONCtx)
TEXT ·fidtx8x8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD CTXIN(R0), R1
	MOVD CTXINSTRIDE(R0), R2
	MOVD CTXOUT(R0), R3
	MOVD CTXOUTSTRIDE(R0), R4
	LSL  $1, R2 // residual stride bytes (int16)
	LSL  $2, R4 // coeff stride bytes (int32)

	VLD1 (R1), [V16.H8]
	WORD $0x0f13a600 // sshll  v0.4s, v16.4h, #3
	WORD $0x4f13a608 // sshll2 v8.4s, v16.8h, #3
	ADD R2, R1
	VLD1 (R1), [V16.H8]
	WORD $0x0f13a601 // sshll  v1.4s, v16.4h, #3
	WORD $0x4f13a609 // sshll2 v9.4s, v16.8h, #3
	ADD R2, R1
	VLD1 (R1), [V16.H8]
	WORD $0x0f13a602 // sshll  v2.4s, v16.4h, #3
	WORD $0x4f13a60a // sshll2 v10.4s, v16.8h, #3
	ADD R2, R1
	VLD1 (R1), [V16.H8]
	WORD $0x0f13a603 // sshll  v3.4s, v16.4h, #3
	WORD $0x4f13a60b // sshll2 v11.4s, v16.8h, #3
	ADD R2, R1
	VLD1 (R1), [V16.H8]
	WORD $0x0f13a604 // sshll  v4.4s, v16.4h, #3
	WORD $0x4f13a60c // sshll2 v12.4s, v16.8h, #3
	ADD R2, R1
	VLD1 (R1), [V16.H8]
	WORD $0x0f13a605 // sshll  v5.4s, v16.4h, #3
	WORD $0x4f13a60d // sshll2 v13.4s, v16.8h, #3
	ADD R2, R1
	VLD1 (R1), [V16.H8]
	WORD $0x0f13a606 // sshll  v6.4s, v16.4h, #3
	WORD $0x4f13a60e // sshll2 v14.4s, v16.8h, #3
	ADD R2, R1
	VLD1 (R1), [V16.H8]
	WORD $0x0f13a607 // sshll  v7.4s, v16.4h, #3
	WORD $0x4f13a60f // sshll2 v15.4s, v16.8h, #3

	// Transpose low source columns 0..3 for rows 0..3 into V20..V23.
	WORD $0x4e812810 // trn1 v16.4s, v0.4s, v1.4s
	WORD $0x4e816811 // trn2 v17.4s, v0.4s, v1.4s
	WORD $0x4e832852 // trn1 v18.4s, v2.4s, v3.4s
	WORD $0x4e836853 // trn2 v19.4s, v2.4s, v3.4s
	WORD $0x4ed23a14 // zip1 v20.2d, v16.2d, v18.2d
	WORD $0x4ed33a35 // zip1 v21.2d, v17.2d, v19.2d
	WORD $0x4ed27a16 // zip2 v22.2d, v16.2d, v18.2d
	WORD $0x4ed37a37 // zip2 v23.2d, v17.2d, v19.2d

	// Transpose low source columns 0..3 for rows 4..7 into V28..V31.
	WORD $0x4e852898 // trn1 v24.4s, v4.4s, v5.4s
	WORD $0x4e856899 // trn2 v25.4s, v4.4s, v5.4s
	WORD $0x4e8728da // trn1 v26.4s, v6.4s, v7.4s
	WORD $0x4e8768db // trn2 v27.4s, v6.4s, v7.4s
	WORD $0x4eda3b1c // zip1 v28.2d, v24.2d, v26.2d
	WORD $0x4edb3b3d // zip1 v29.2d, v25.2d, v27.2d
	WORD $0x4eda7b1e // zip2 v30.2d, v24.2d, v26.2d
	WORD $0x4edb7b3f // zip2 v31.2d, v25.2d, v27.2d

	VST1 [V20.S4], (R3)
	ADD  $16, R3
	VST1 [V28.S4], (R3)
	SUB  $16, R3
	ADD  R4, R3
	VST1 [V21.S4], (R3)
	ADD  $16, R3
	VST1 [V29.S4], (R3)
	SUB  $16, R3
	ADD  R4, R3
	VST1 [V22.S4], (R3)
	ADD  $16, R3
	VST1 [V30.S4], (R3)
	SUB  $16, R3
	ADD  R4, R3
	VST1 [V23.S4], (R3)
	ADD  $16, R3
	VST1 [V31.S4], (R3)
	SUB  $16, R3
	ADD  R4, R3

	// Transpose high source columns 4..7 for rows 0..3 into V0..V3.
	WORD $0x4e892910 // trn1 v16.4s, v8.4s, v9.4s
	WORD $0x4e896911 // trn2 v17.4s, v8.4s, v9.4s
	WORD $0x4e8b2952 // trn1 v18.4s, v10.4s, v11.4s
	WORD $0x4e8b6953 // trn2 v19.4s, v10.4s, v11.4s
	WORD $0x4ed23a00 // zip1 v0.2d, v16.2d, v18.2d
	WORD $0x4ed33a21 // zip1 v1.2d, v17.2d, v19.2d
	WORD $0x4ed27a02 // zip2 v2.2d, v16.2d, v18.2d
	WORD $0x4ed37a23 // zip2 v3.2d, v17.2d, v19.2d

	// Transpose high source columns 4..7 for rows 4..7 into V4..V7.
	WORD $0x4e8d2990 // trn1 v16.4s, v12.4s, v13.4s
	WORD $0x4e8d6991 // trn2 v17.4s, v12.4s, v13.4s
	WORD $0x4e8f29d2 // trn1 v18.4s, v14.4s, v15.4s
	WORD $0x4e8f69d3 // trn2 v19.4s, v14.4s, v15.4s
	WORD $0x4ed23a04 // zip1 v4.2d, v16.2d, v18.2d
	WORD $0x4ed33a25 // zip1 v5.2d, v17.2d, v19.2d
	WORD $0x4ed27a06 // zip2 v6.2d, v16.2d, v18.2d
	WORD $0x4ed37a27 // zip2 v7.2d, v17.2d, v19.2d

	VST1 [V0.S4], (R3)
	ADD  $16, R3
	VST1 [V4.S4], (R3)
	SUB  $16, R3
	ADD  R4, R3
	VST1 [V1.S4], (R3)
	ADD  $16, R3
	VST1 [V5.S4], (R3)
	SUB  $16, R3
	ADD  R4, R3
	VST1 [V2.S4], (R3)
	ADD  $16, R3
	VST1 [V6.S4], (R3)
	SUB  $16, R3
	ADD  R4, R3
	VST1 [V3.S4], (R3)
	ADD  $16, R3
	VST1 [V7.S4], (R3)
	SUB  $16, R3
	RET
