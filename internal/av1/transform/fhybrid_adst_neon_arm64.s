// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

#define CTXIN        0
#define CTXINSTRIDE  8
#define CTXOUT      16
#define CTXOUTSTRIDE 24

// NEON forward 8x8 ADST_DCT. SVT's svt_av1_fwd_txfm2d_8x8_neon runs an
// ADST column pass, applies the common 8x8 -1 shift, transposes, and then runs
// a DCT row pass. This mirrors that shape while matching the local scalar
// fwdADST8Values/fwdDCT8Values equations exactly.

// func fadstDCT8x8NEONAsm(ctx *fdct8x8NEONCtx)
TEXT ·fadstDCT8x8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD CTXIN(R0), R1
	MOVD CTXINSTRIDE(R0), R2  // residual stride in elements
	MOVD CTXOUT(R0), R3
	MOVD CTXOUTSTRIDE(R0), R4 // coeff stride in elements
	MOVD $·fadst8Cospi(SB), R5
	VLD1 (R5), [V26.S4, V27.S4, V28.S4]
	LSL  $1, R2 // stride bytes (int16)
	LSL  $2, R4 // stride bytes (int32)

	// Load residual rows, widening with the shift[0]=2 fused into SSHLL.
	VLD1 (R1), [V16.H8]
	WORD $0x0f12a600 // sshll v0.4s, v16.4h, #2
	WORD $0x4f12a608 // sshll2 v8.4s, v16.8h, #2
	ADD  R2, R1
	VLD1 (R1), [V16.H8]
	WORD $0x0f12a601 // sshll v1.4s, v16.4h, #2
	WORD $0x4f12a609 // sshll2 v9.4s, v16.8h, #2
	ADD  R2, R1
	VLD1 (R1), [V16.H8]
	WORD $0x0f12a602 // sshll v2.4s, v16.4h, #2
	WORD $0x4f12a60a // sshll2 v10.4s, v16.8h, #2
	ADD  R2, R1
	VLD1 (R1), [V16.H8]
	WORD $0x0f12a603 // sshll v3.4s, v16.4h, #2
	WORD $0x4f12a60b // sshll2 v11.4s, v16.8h, #2
	ADD  R2, R1
	VLD1 (R1), [V16.H8]
	WORD $0x0f12a604 // sshll v4.4s, v16.4h, #2
	WORD $0x4f12a60c // sshll2 v12.4s, v16.8h, #2
	ADD  R2, R1
	VLD1 (R1), [V16.H8]
	WORD $0x0f12a605 // sshll v5.4s, v16.4h, #2
	WORD $0x4f12a60d // sshll2 v13.4s, v16.8h, #2
	ADD  R2, R1
	VLD1 (R1), [V16.H8]
	WORD $0x0f12a606 // sshll v6.4s, v16.4h, #2
	WORD $0x4f12a60e // sshll2 v14.4s, v16.8h, #2
	ADD  R2, R1
	VLD1 (R1), [V16.H8]
	WORD $0x0f12a607 // sshll v7.4s, v16.4h, #2
	WORD $0x4f12a60f // sshll2 v15.4s, v16.8h, #2

	// ADST8 column pass for lane group V0-V7.
	WORD $0x4ea01c10 // mov v16.16b, v0.16b ; s0 = x0
	WORD $0x6ea0b8f1 // neg v17.4s, v7.4s ; s1 = -x7
	WORD $0x4f9a8078 // mul v24.4s, v3.4s, v26.s[0] ; s2 = half(-c32*x3 + c32*x4)
	WORD $0x6ea0bb18 // neg v24.4s, v24.4s
	WORD $0x6f9a0098 // mla v24.4s, v4.4s, v26.s[0]
	WORD $0x4f332712 // srshr v18.4s, v24.4s, #13
	WORD $0x4f9a8078 // mul v24.4s, v3.4s, v26.s[0] ; s3 = half(-c32*x3 - c32*x4)
	WORD $0x6ea0bb18 // neg v24.4s, v24.4s
	WORD $0x6f9a4098 // mls v24.4s, v4.4s, v26.s[0]
	WORD $0x4f332713 // srshr v19.4s, v24.4s, #13
	WORD $0x6ea0b834 // neg v20.4s, v1.4s ; s4 = -x1
	WORD $0x4ea61cd5 // mov v21.16b, v6.16b ; s5 = x6
	WORD $0x4f9a8058 // mul v24.4s, v2.4s, v26.s[0] ; s6 = half(c32*x2 - c32*x5)
	WORD $0x6f9a40b8 // mls v24.4s, v5.4s, v26.s[0]
	WORD $0x4f332716 // srshr v22.4s, v24.4s, #13
	WORD $0x4f9a8058 // mul v24.4s, v2.4s, v26.s[0] ; s7 = half(c32*x2 + c32*x5)
	WORD $0x6f9a00b8 // mla v24.4s, v5.4s, v26.s[0]
	WORD $0x4f332717 // srshr v23.4s, v24.4s, #13
	WORD $0x4eb28600 // add v0.4s, v16.4s, v18.4s ; t0 = s0 + s2
	WORD $0x4eb38621 // add v1.4s, v17.4s, v19.4s ; t1 = s1 + s3
	WORD $0x6eb28602 // sub v2.4s, v16.4s, v18.4s ; t2 = s0 - s2
	WORD $0x6eb38623 // sub v3.4s, v17.4s, v19.4s ; t3 = s1 - s3
	WORD $0x4eb68684 // add v4.4s, v20.4s, v22.4s ; t4 = s4 + s6
	WORD $0x4eb786a5 // add v5.4s, v21.4s, v23.4s ; t5 = s5 + s7
	WORD $0x6eb68686 // sub v6.4s, v20.4s, v22.4s ; t6 = s4 - s6
	WORD $0x6eb786a7 // sub v7.4s, v21.4s, v23.4s ; t7 = s5 - s7
	WORD $0x4fba8098 // mul v24.4s, v4.4s, v26.s[1] ; s4 = half(c16*t4 + c48*t5)
	WORD $0x6f9a08b8 // mla v24.4s, v5.4s, v26.s[2]
	WORD $0x4f332714 // srshr v20.4s, v24.4s, #13
	WORD $0x4f9a8898 // mul v24.4s, v4.4s, v26.s[2] ; s5 = half(c48*t4 - c16*t5)
	WORD $0x6fba40b8 // mls v24.4s, v5.4s, v26.s[1]
	WORD $0x4f332715 // srshr v21.4s, v24.4s, #13
	WORD $0x4f9a88d8 // mul v24.4s, v6.4s, v26.s[2] ; s6 = half(-c48*t6 + c16*t7)
	WORD $0x6ea0bb18 // neg v24.4s, v24.4s
	WORD $0x6fba00f8 // mla v24.4s, v7.4s, v26.s[1]
	WORD $0x4f332716 // srshr v22.4s, v24.4s, #13
	WORD $0x4fba80d8 // mul v24.4s, v6.4s, v26.s[1] ; s7 = half(c16*t6 + c48*t7)
	WORD $0x6f9a08f8 // mla v24.4s, v7.4s, v26.s[2]
	WORD $0x4f332717 // srshr v23.4s, v24.4s, #13
	WORD $0x6eb48404 // sub v4.4s, v0.4s, v20.4s ; t4 = t0 - s4
	WORD $0x4eb48400 // add v0.4s, v0.4s, v20.4s ; t0 = t0 + s4
	WORD $0x6eb58425 // sub v5.4s, v1.4s, v21.4s ; t5 = t1 - s5
	WORD $0x4eb58421 // add v1.4s, v1.4s, v21.4s ; t1 = t1 + s5
	WORD $0x6eb68446 // sub v6.4s, v2.4s, v22.4s ; t6 = t2 - s6
	WORD $0x4eb68442 // add v2.4s, v2.4s, v22.4s ; t2 = t2 + s6
	WORD $0x6eb78467 // sub v7.4s, v3.4s, v23.4s ; t7 = t3 - s7
	WORD $0x4eb78463 // add v3.4s, v3.4s, v23.4s ; t3 = t3 + s7
	WORD $0x4fba8818 // mul v24.4s, v0.4s, v26.s[3] ; s0 = half(c4*t0 + c60*t1)
	WORD $0x6f9b0038 // mla v24.4s, v1.4s, v27.s[0]
	WORD $0x4f332710 // srshr v16.4s, v24.4s, #13
	WORD $0x4f9b8018 // mul v24.4s, v0.4s, v27.s[0] ; s1 = half(c60*t0 - c4*t1)
	WORD $0x6fba4838 // mls v24.4s, v1.4s, v26.s[3]
	WORD $0x4f332711 // srshr v17.4s, v24.4s, #13
	WORD $0x4fbb8058 // mul v24.4s, v2.4s, v27.s[1] ; s2 = half(c20*t2 + c44*t3)
	WORD $0x6f9b0878 // mla v24.4s, v3.4s, v27.s[2]
	WORD $0x4f332712 // srshr v18.4s, v24.4s, #13
	WORD $0x4f9b8858 // mul v24.4s, v2.4s, v27.s[2] ; s3 = half(c44*t2 - c20*t3)
	WORD $0x6fbb4078 // mls v24.4s, v3.4s, v27.s[1]
	WORD $0x4f332713 // srshr v19.4s, v24.4s, #13
	WORD $0x4fbb8898 // mul v24.4s, v4.4s, v27.s[3] ; s4 = half(c36*t4 + c28*t5)
	WORD $0x6f9c00b8 // mla v24.4s, v5.4s, v28.s[0]
	WORD $0x4f332714 // srshr v20.4s, v24.4s, #13
	WORD $0x4f9c8098 // mul v24.4s, v4.4s, v28.s[0] ; s5 = half(c28*t4 - c36*t5)
	WORD $0x6fbb48b8 // mls v24.4s, v5.4s, v27.s[3]
	WORD $0x4f332715 // srshr v21.4s, v24.4s, #13
	WORD $0x4fbc80d8 // mul v24.4s, v6.4s, v28.s[1] ; s6 = half(c52*t6 + c12*t7)
	WORD $0x6f9c08f8 // mla v24.4s, v7.4s, v28.s[2]
	WORD $0x4f332716 // srshr v22.4s, v24.4s, #13
	WORD $0x4f9c88d8 // mul v24.4s, v6.4s, v28.s[2] ; s7 = half(c12*t6 - c52*t7)
	WORD $0x6fbc40f8 // mls v24.4s, v7.4s, v28.s[1]
	WORD $0x4f332717 // srshr v23.4s, v24.4s, #13
	WORD $0x4eb11e20 // mov v0.16b, v17.16b ; out0 = s1
	WORD $0x4eb61ec1 // mov v1.16b, v22.16b ; out1 = s6
	WORD $0x4eb31e62 // mov v2.16b, v19.16b ; out2 = s3
	WORD $0x4eb41e83 // mov v3.16b, v20.16b ; out3 = s4
	WORD $0x4eb51ea4 // mov v4.16b, v21.16b ; out4 = s5
	WORD $0x4eb21e45 // mov v5.16b, v18.16b ; out5 = s2
	WORD $0x4eb71ee6 // mov v6.16b, v23.16b ; out6 = s7
	WORD $0x4eb01e07 // mov v7.16b, v16.16b ; out7 = s0

	// ADST8 column pass for lane group V8-V15.
	WORD $0x4ea81d10 // mov v16.16b, v8.16b ; s0 = x0
	WORD $0x6ea0b9f1 // neg v17.4s, v15.4s ; s1 = -x7
	WORD $0x4f9a8178 // mul v24.4s, v11.4s, v26.s[0] ; s2 = half(-c32*x3 + c32*x4)
	WORD $0x6ea0bb18 // neg v24.4s, v24.4s
	WORD $0x6f9a0198 // mla v24.4s, v12.4s, v26.s[0]
	WORD $0x4f332712 // srshr v18.4s, v24.4s, #13
	WORD $0x4f9a8178 // mul v24.4s, v11.4s, v26.s[0] ; s3 = half(-c32*x3 - c32*x4)
	WORD $0x6ea0bb18 // neg v24.4s, v24.4s
	WORD $0x6f9a4198 // mls v24.4s, v12.4s, v26.s[0]
	WORD $0x4f332713 // srshr v19.4s, v24.4s, #13
	WORD $0x6ea0b934 // neg v20.4s, v9.4s ; s4 = -x1
	WORD $0x4eae1dd5 // mov v21.16b, v14.16b ; s5 = x6
	WORD $0x4f9a8158 // mul v24.4s, v10.4s, v26.s[0] ; s6 = half(c32*x2 - c32*x5)
	WORD $0x6f9a41b8 // mls v24.4s, v13.4s, v26.s[0]
	WORD $0x4f332716 // srshr v22.4s, v24.4s, #13
	WORD $0x4f9a8158 // mul v24.4s, v10.4s, v26.s[0] ; s7 = half(c32*x2 + c32*x5)
	WORD $0x6f9a01b8 // mla v24.4s, v13.4s, v26.s[0]
	WORD $0x4f332717 // srshr v23.4s, v24.4s, #13
	WORD $0x4eb28608 // add v8.4s, v16.4s, v18.4s ; t0 = s0 + s2
	WORD $0x4eb38629 // add v9.4s, v17.4s, v19.4s ; t1 = s1 + s3
	WORD $0x6eb2860a // sub v10.4s, v16.4s, v18.4s ; t2 = s0 - s2
	WORD $0x6eb3862b // sub v11.4s, v17.4s, v19.4s ; t3 = s1 - s3
	WORD $0x4eb6868c // add v12.4s, v20.4s, v22.4s ; t4 = s4 + s6
	WORD $0x4eb786ad // add v13.4s, v21.4s, v23.4s ; t5 = s5 + s7
	WORD $0x6eb6868e // sub v14.4s, v20.4s, v22.4s ; t6 = s4 - s6
	WORD $0x6eb786af // sub v15.4s, v21.4s, v23.4s ; t7 = s5 - s7
	WORD $0x4fba8198 // mul v24.4s, v12.4s, v26.s[1] ; s4 = half(c16*t4 + c48*t5)
	WORD $0x6f9a09b8 // mla v24.4s, v13.4s, v26.s[2]
	WORD $0x4f332714 // srshr v20.4s, v24.4s, #13
	WORD $0x4f9a8998 // mul v24.4s, v12.4s, v26.s[2] ; s5 = half(c48*t4 - c16*t5)
	WORD $0x6fba41b8 // mls v24.4s, v13.4s, v26.s[1]
	WORD $0x4f332715 // srshr v21.4s, v24.4s, #13
	WORD $0x4f9a89d8 // mul v24.4s, v14.4s, v26.s[2] ; s6 = half(-c48*t6 + c16*t7)
	WORD $0x6ea0bb18 // neg v24.4s, v24.4s
	WORD $0x6fba01f8 // mla v24.4s, v15.4s, v26.s[1]
	WORD $0x4f332716 // srshr v22.4s, v24.4s, #13
	WORD $0x4fba81d8 // mul v24.4s, v14.4s, v26.s[1] ; s7 = half(c16*t6 + c48*t7)
	WORD $0x6f9a09f8 // mla v24.4s, v15.4s, v26.s[2]
	WORD $0x4f332717 // srshr v23.4s, v24.4s, #13
	WORD $0x6eb4850c // sub v12.4s, v8.4s, v20.4s ; t4 = t0 - s4
	WORD $0x4eb48508 // add v8.4s, v8.4s, v20.4s ; t0 = t0 + s4
	WORD $0x6eb5852d // sub v13.4s, v9.4s, v21.4s ; t5 = t1 - s5
	WORD $0x4eb58529 // add v9.4s, v9.4s, v21.4s ; t1 = t1 + s5
	WORD $0x6eb6854e // sub v14.4s, v10.4s, v22.4s ; t6 = t2 - s6
	WORD $0x4eb6854a // add v10.4s, v10.4s, v22.4s ; t2 = t2 + s6
	WORD $0x6eb7856f // sub v15.4s, v11.4s, v23.4s ; t7 = t3 - s7
	WORD $0x4eb7856b // add v11.4s, v11.4s, v23.4s ; t3 = t3 + s7
	WORD $0x4fba8918 // mul v24.4s, v8.4s, v26.s[3] ; s0 = half(c4*t0 + c60*t1)
	WORD $0x6f9b0138 // mla v24.4s, v9.4s, v27.s[0]
	WORD $0x4f332710 // srshr v16.4s, v24.4s, #13
	WORD $0x4f9b8118 // mul v24.4s, v8.4s, v27.s[0] ; s1 = half(c60*t0 - c4*t1)
	WORD $0x6fba4938 // mls v24.4s, v9.4s, v26.s[3]
	WORD $0x4f332711 // srshr v17.4s, v24.4s, #13
	WORD $0x4fbb8158 // mul v24.4s, v10.4s, v27.s[1] ; s2 = half(c20*t2 + c44*t3)
	WORD $0x6f9b0978 // mla v24.4s, v11.4s, v27.s[2]
	WORD $0x4f332712 // srshr v18.4s, v24.4s, #13
	WORD $0x4f9b8958 // mul v24.4s, v10.4s, v27.s[2] ; s3 = half(c44*t2 - c20*t3)
	WORD $0x6fbb4178 // mls v24.4s, v11.4s, v27.s[1]
	WORD $0x4f332713 // srshr v19.4s, v24.4s, #13
	WORD $0x4fbb8998 // mul v24.4s, v12.4s, v27.s[3] ; s4 = half(c36*t4 + c28*t5)
	WORD $0x6f9c01b8 // mla v24.4s, v13.4s, v28.s[0]
	WORD $0x4f332714 // srshr v20.4s, v24.4s, #13
	WORD $0x4f9c8198 // mul v24.4s, v12.4s, v28.s[0] ; s5 = half(c28*t4 - c36*t5)
	WORD $0x6fbb49b8 // mls v24.4s, v13.4s, v27.s[3]
	WORD $0x4f332715 // srshr v21.4s, v24.4s, #13
	WORD $0x4fbc81d8 // mul v24.4s, v14.4s, v28.s[1] ; s6 = half(c52*t6 + c12*t7)
	WORD $0x6f9c09f8 // mla v24.4s, v15.4s, v28.s[2]
	WORD $0x4f332716 // srshr v22.4s, v24.4s, #13
	WORD $0x4f9c89d8 // mul v24.4s, v14.4s, v28.s[2] ; s7 = half(c12*t6 - c52*t7)
	WORD $0x6fbc41f8 // mls v24.4s, v15.4s, v28.s[1]
	WORD $0x4f332717 // srshr v23.4s, v24.4s, #13
	WORD $0x4eb11e28 // mov v8.16b, v17.16b ; out0 = s1
	WORD $0x4eb61ec9 // mov v9.16b, v22.16b ; out1 = s6
	WORD $0x4eb31e6a // mov v10.16b, v19.16b ; out2 = s3
	WORD $0x4eb41e8b // mov v11.16b, v20.16b ; out3 = s4
	WORD $0x4eb51eac // mov v12.16b, v21.16b ; out4 = s5
	WORD $0x4eb21e4d // mov v13.16b, v18.16b ; out5 = s2
	WORD $0x4eb71eee // mov v14.16b, v23.16b ; out6 = s7
	WORD $0x4eb01e0f // mov v15.16b, v16.16b ; out7 = s0

	// Apply the common 8x8 shift[1] == -1 symmetric round shift.
	WORD $0x4f00043d // movi v29.4s, #1
	WORD $0x4f210410 // sshr v16.4s, v0.4s, #31
	WORD $0x4eb08400 // add v0.4s, v0.4s, v16.4s
	WORD $0x4ebd8400 // add v0.4s, v0.4s, v29.4s
	WORD $0x4f3f0400 // sshr v0.4s, v0.4s, #1
	WORD $0x4f210430 // sshr v16.4s, v1.4s, #31
	WORD $0x4eb08421 // add v1.4s, v1.4s, v16.4s
	WORD $0x4ebd8421 // add v1.4s, v1.4s, v29.4s
	WORD $0x4f3f0421 // sshr v1.4s, v1.4s, #1
	WORD $0x4f210450 // sshr v16.4s, v2.4s, #31
	WORD $0x4eb08442 // add v2.4s, v2.4s, v16.4s
	WORD $0x4ebd8442 // add v2.4s, v2.4s, v29.4s
	WORD $0x4f3f0442 // sshr v2.4s, v2.4s, #1
	WORD $0x4f210470 // sshr v16.4s, v3.4s, #31
	WORD $0x4eb08463 // add v3.4s, v3.4s, v16.4s
	WORD $0x4ebd8463 // add v3.4s, v3.4s, v29.4s
	WORD $0x4f3f0463 // sshr v3.4s, v3.4s, #1
	WORD $0x4f210490 // sshr v16.4s, v4.4s, #31
	WORD $0x4eb08484 // add v4.4s, v4.4s, v16.4s
	WORD $0x4ebd8484 // add v4.4s, v4.4s, v29.4s
	WORD $0x4f3f0484 // sshr v4.4s, v4.4s, #1
	WORD $0x4f2104b0 // sshr v16.4s, v5.4s, #31
	WORD $0x4eb084a5 // add v5.4s, v5.4s, v16.4s
	WORD $0x4ebd84a5 // add v5.4s, v5.4s, v29.4s
	WORD $0x4f3f04a5 // sshr v5.4s, v5.4s, #1
	WORD $0x4f2104d0 // sshr v16.4s, v6.4s, #31
	WORD $0x4eb084c6 // add v6.4s, v6.4s, v16.4s
	WORD $0x4ebd84c6 // add v6.4s, v6.4s, v29.4s
	WORD $0x4f3f04c6 // sshr v6.4s, v6.4s, #1
	WORD $0x4f2104f0 // sshr v16.4s, v7.4s, #31
	WORD $0x4eb084e7 // add v7.4s, v7.4s, v16.4s
	WORD $0x4ebd84e7 // add v7.4s, v7.4s, v29.4s
	WORD $0x4f3f04e7 // sshr v7.4s, v7.4s, #1
	WORD $0x4f210510 // sshr v16.4s, v8.4s, #31
	WORD $0x4eb08508 // add v8.4s, v8.4s, v16.4s
	WORD $0x4ebd8508 // add v8.4s, v8.4s, v29.4s
	WORD $0x4f3f0508 // sshr v8.4s, v8.4s, #1
	WORD $0x4f210530 // sshr v16.4s, v9.4s, #31
	WORD $0x4eb08529 // add v9.4s, v9.4s, v16.4s
	WORD $0x4ebd8529 // add v9.4s, v9.4s, v29.4s
	WORD $0x4f3f0529 // sshr v9.4s, v9.4s, #1
	WORD $0x4f210550 // sshr v16.4s, v10.4s, #31
	WORD $0x4eb0854a // add v10.4s, v10.4s, v16.4s
	WORD $0x4ebd854a // add v10.4s, v10.4s, v29.4s
	WORD $0x4f3f054a // sshr v10.4s, v10.4s, #1
	WORD $0x4f210570 // sshr v16.4s, v11.4s, #31
	WORD $0x4eb0856b // add v11.4s, v11.4s, v16.4s
	WORD $0x4ebd856b // add v11.4s, v11.4s, v29.4s
	WORD $0x4f3f056b // sshr v11.4s, v11.4s, #1
	WORD $0x4f210590 // sshr v16.4s, v12.4s, #31
	WORD $0x4eb0858c // add v12.4s, v12.4s, v16.4s
	WORD $0x4ebd858c // add v12.4s, v12.4s, v29.4s
	WORD $0x4f3f058c // sshr v12.4s, v12.4s, #1
	WORD $0x4f2105b0 // sshr v16.4s, v13.4s, #31
	WORD $0x4eb085ad // add v13.4s, v13.4s, v16.4s
	WORD $0x4ebd85ad // add v13.4s, v13.4s, v29.4s
	WORD $0x4f3f05ad // sshr v13.4s, v13.4s, #1
	WORD $0x4f2105d0 // sshr v16.4s, v14.4s, #31
	WORD $0x4eb085ce // add v14.4s, v14.4s, v16.4s
	WORD $0x4ebd85ce // add v14.4s, v14.4s, v29.4s
	WORD $0x4f3f05ce // sshr v14.4s, v14.4s, #1
	WORD $0x4f2105f0 // sshr v16.4s, v15.4s, #31
	WORD $0x4eb085ef // add v15.4s, v15.4s, v16.4s
	WORD $0x4ebd85ef // add v15.4s, v15.4s, v29.4s
	WORD $0x4f3f05ef // sshr v15.4s, v15.4s, #1

	MOVD $·fdct8Cospi(SB), R5
	VLD1 (R5), [V30.S4, V31.S4]

	// Transpose the 8x8 int32 matrix held as rows (V0-V7 | V8-V15).
	WORD $0x4e812810 // trn1 v16.4s, v0.4s, v1.4s
	WORD $0x4e816811 // trn2 v17.4s, v0.4s, v1.4s
	WORD $0x4e832852 // trn1 v18.4s, v2.4s, v3.4s
	WORD $0x4e836853 // trn2 v19.4s, v2.4s, v3.4s
	WORD $0x4ed23a00 // zip1 v0.2d, v16.2d, v18.2d
	WORD $0x4ed33a21 // zip1 v1.2d, v17.2d, v19.2d
	WORD $0x4ed27a02 // zip2 v2.2d, v16.2d, v18.2d
	WORD $0x4ed37a23 // zip2 v3.2d, v17.2d, v19.2d
	WORD $0x4e892910 // trn1 v16.4s, v8.4s, v9.4s
	WORD $0x4e896911 // trn2 v17.4s, v8.4s, v9.4s
	WORD $0x4e8b2952 // trn1 v18.4s, v10.4s, v11.4s
	WORD $0x4e8b6953 // trn2 v19.4s, v10.4s, v11.4s
	WORD $0x4ed23a08 // zip1 v8.2d, v16.2d, v18.2d
	WORD $0x4ed33a29 // zip1 v9.2d, v17.2d, v19.2d
	WORD $0x4ed27a0a // zip2 v10.2d, v16.2d, v18.2d
	WORD $0x4ed37a2b // zip2 v11.2d, v17.2d, v19.2d
	WORD $0x4e852890 // trn1 v16.4s, v4.4s, v5.4s
	WORD $0x4e856891 // trn2 v17.4s, v4.4s, v5.4s
	WORD $0x4e8728d2 // trn1 v18.4s, v6.4s, v7.4s
	WORD $0x4e8768d3 // trn2 v19.4s, v6.4s, v7.4s
	WORD $0x4ed23a04 // zip1 v4.2d, v16.2d, v18.2d
	WORD $0x4ed33a25 // zip1 v5.2d, v17.2d, v19.2d
	WORD $0x4ed27a06 // zip2 v6.2d, v16.2d, v18.2d
	WORD $0x4ed37a27 // zip2 v7.2d, v17.2d, v19.2d
	WORD $0x4e8d2990 // trn1 v16.4s, v12.4s, v13.4s
	WORD $0x4e8d6991 // trn2 v17.4s, v12.4s, v13.4s
	WORD $0x4e8f29d2 // trn1 v18.4s, v14.4s, v15.4s
	WORD $0x4e8f69d3 // trn2 v19.4s, v14.4s, v15.4s
	WORD $0x4ed23a0c // zip1 v12.2d, v16.2d, v18.2d
	WORD $0x4ed33a2d // zip1 v13.2d, v17.2d, v19.2d
	WORD $0x4ed27a0e // zip2 v14.2d, v16.2d, v18.2d
	WORD $0x4ed37a2f // zip2 v15.2d, v17.2d, v19.2d

	// Row pass: lo lane group is (V0-V3, V8-V11), hi is (V4-V7, V12-V15).
	WORD $0x4eab8410 // add v16.4s, v0.4s, v11.4s
	WORD $0x4eaa8431 // add v17.4s, v1.4s, v10.4s
	WORD $0x4ea98452 // add v18.4s, v2.4s, v9.4s
	WORD $0x4ea88473 // add v19.4s, v3.4s, v8.4s
	WORD $0x6ea88474 // sub v20.4s, v3.4s, v8.4s
	WORD $0x6ea98455 // sub v21.4s, v2.4s, v9.4s
	WORD $0x6eaa8436 // sub v22.4s, v1.4s, v10.4s
	WORD $0x6eab8417 // sub v23.4s, v0.4s, v11.4s
	WORD $0x4eb38600 // add v0.4s, v16.4s, v19.4s
	WORD $0x4eb28621 // add v1.4s, v17.4s, v18.4s
	WORD $0x6eb28622 // sub v2.4s, v17.4s, v18.4s
	WORD $0x6eb38603 // sub v3.4s, v16.4s, v19.4s
	WORD $0x4eb41e88 // mov v8.16b, v20.16b
	WORD $0x4f9e82d8 // mul v24.4s, v22.4s, v30.s[0]
	WORD $0x6f9e42b8 // mls v24.4s, v21.4s, v30.s[0]
	WORD $0x4f332709 // srshr v9.4s, v24.4s, #13
	WORD $0x4f9e82d8 // mul v24.4s, v22.4s, v30.s[0]
	WORD $0x6f9e02b8 // mla v24.4s, v21.4s, v30.s[0]
	WORD $0x4f33270a // srshr v10.4s, v24.4s, #13
	WORD $0x4eb71eeb // mov v11.16b, v23.16b
	WORD $0x4f9e8018 // mul v24.4s, v0.4s, v30.s[0]
	WORD $0x6f9e0038 // mla v24.4s, v1.4s, v30.s[0]
	WORD $0x4f332710 // srshr v16.4s, v24.4s, #13
	WORD $0x4f9e8018 // mul v24.4s, v0.4s, v30.s[0]
	WORD $0x6f9e4038 // mls v24.4s, v1.4s, v30.s[0]
	WORD $0x4f332711 // srshr v17.4s, v24.4s, #13
	WORD $0x4f9e8858 // mul v24.4s, v2.4s, v30.s[2]
	WORD $0x6fbe0078 // mla v24.4s, v3.4s, v30.s[1]
	WORD $0x4f332712 // srshr v18.4s, v24.4s, #13
	WORD $0x4f9e8878 // mul v24.4s, v3.4s, v30.s[2]
	WORD $0x6fbe4058 // mls v24.4s, v2.4s, v30.s[1]
	WORD $0x4f332713 // srshr v19.4s, v24.4s, #13
	WORD $0x4ea98514 // add v20.4s, v8.4s, v9.4s
	WORD $0x6ea98515 // sub v21.4s, v8.4s, v9.4s
	WORD $0x6eaa8576 // sub v22.4s, v11.4s, v10.4s
	WORD $0x4eaa8577 // add v23.4s, v11.4s, v10.4s
	WORD $0x4eb01e00 // mov v0.16b, v16.16b
	WORD $0x4eb11e28 // mov v8.16b, v17.16b
	WORD $0x4eb21e42 // mov v2.16b, v18.16b
	WORD $0x4eb31e6a // mov v10.16b, v19.16b
	WORD $0x4f9f8a98 // mul v24.4s, v20.4s, v31.s[2]
	WORD $0x6fbe0af8 // mla v24.4s, v23.4s, v30.s[3]
	WORD $0x4f332701 // srshr v1.4s, v24.4s, #13
	WORD $0x4f9f82b8 // mul v24.4s, v21.4s, v31.s[0]
	WORD $0x6fbf02d8 // mla v24.4s, v22.4s, v31.s[1]
	WORD $0x4f332709 // srshr v9.4s, v24.4s, #13
	WORD $0x4f9f82d8 // mul v24.4s, v22.4s, v31.s[0]
	WORD $0x6fbf42b8 // mls v24.4s, v21.4s, v31.s[1]
	WORD $0x4f332703 // srshr v3.4s, v24.4s, #13
	WORD $0x4f9f8af8 // mul v24.4s, v23.4s, v31.s[2]
	WORD $0x6fbe4a98 // mls v24.4s, v20.4s, v30.s[3]
	WORD $0x4f33270b // srshr v11.4s, v24.4s, #13
	WORD $0x4eaf8490 // add v16.4s, v4.4s, v15.4s
	WORD $0x4eae84b1 // add v17.4s, v5.4s, v14.4s
	WORD $0x4ead84d2 // add v18.4s, v6.4s, v13.4s
	WORD $0x4eac84f3 // add v19.4s, v7.4s, v12.4s
	WORD $0x6eac84f4 // sub v20.4s, v7.4s, v12.4s
	WORD $0x6ead84d5 // sub v21.4s, v6.4s, v13.4s
	WORD $0x6eae84b6 // sub v22.4s, v5.4s, v14.4s
	WORD $0x6eaf8497 // sub v23.4s, v4.4s, v15.4s
	WORD $0x4eb38604 // add v4.4s, v16.4s, v19.4s
	WORD $0x4eb28625 // add v5.4s, v17.4s, v18.4s
	WORD $0x6eb28626 // sub v6.4s, v17.4s, v18.4s
	WORD $0x6eb38607 // sub v7.4s, v16.4s, v19.4s
	WORD $0x4eb41e8c // mov v12.16b, v20.16b
	WORD $0x4f9e82d8 // mul v24.4s, v22.4s, v30.s[0]
	WORD $0x6f9e42b8 // mls v24.4s, v21.4s, v30.s[0]
	WORD $0x4f33270d // srshr v13.4s, v24.4s, #13
	WORD $0x4f9e82d8 // mul v24.4s, v22.4s, v30.s[0]
	WORD $0x6f9e02b8 // mla v24.4s, v21.4s, v30.s[0]
	WORD $0x4f33270e // srshr v14.4s, v24.4s, #13
	WORD $0x4eb71eef // mov v15.16b, v23.16b
	WORD $0x4f9e8098 // mul v24.4s, v4.4s, v30.s[0]
	WORD $0x6f9e00b8 // mla v24.4s, v5.4s, v30.s[0]
	WORD $0x4f332710 // srshr v16.4s, v24.4s, #13
	WORD $0x4f9e8098 // mul v24.4s, v4.4s, v30.s[0]
	WORD $0x6f9e40b8 // mls v24.4s, v5.4s, v30.s[0]
	WORD $0x4f332711 // srshr v17.4s, v24.4s, #13
	WORD $0x4f9e88d8 // mul v24.4s, v6.4s, v30.s[2]
	WORD $0x6fbe00f8 // mla v24.4s, v7.4s, v30.s[1]
	WORD $0x4f332712 // srshr v18.4s, v24.4s, #13
	WORD $0x4f9e88f8 // mul v24.4s, v7.4s, v30.s[2]
	WORD $0x6fbe40d8 // mls v24.4s, v6.4s, v30.s[1]
	WORD $0x4f332713 // srshr v19.4s, v24.4s, #13
	WORD $0x4ead8594 // add v20.4s, v12.4s, v13.4s
	WORD $0x6ead8595 // sub v21.4s, v12.4s, v13.4s
	WORD $0x6eae85f6 // sub v22.4s, v15.4s, v14.4s
	WORD $0x4eae85f7 // add v23.4s, v15.4s, v14.4s
	WORD $0x4eb01e04 // mov v4.16b, v16.16b
	WORD $0x4eb11e2c // mov v12.16b, v17.16b
	WORD $0x4eb21e46 // mov v6.16b, v18.16b
	WORD $0x4eb31e6e // mov v14.16b, v19.16b
	WORD $0x4f9f8a98 // mul v24.4s, v20.4s, v31.s[2]
	WORD $0x6fbe0af8 // mla v24.4s, v23.4s, v30.s[3]
	WORD $0x4f332705 // srshr v5.4s, v24.4s, #13
	WORD $0x4f9f82b8 // mul v24.4s, v21.4s, v31.s[0]
	WORD $0x6fbf02d8 // mla v24.4s, v22.4s, v31.s[1]
	WORD $0x4f33270d // srshr v13.4s, v24.4s, #13
	WORD $0x4f9f82d8 // mul v24.4s, v22.4s, v31.s[0]
	WORD $0x6fbf42b8 // mls v24.4s, v21.4s, v31.s[1]
	WORD $0x4f332707 // srshr v7.4s, v24.4s, #13
	WORD $0x4f9f8af8 // mul v24.4s, v23.4s, v31.s[2]
	WORD $0x6fbe4a98 // mls v24.4s, v20.4s, v30.s[3]
	WORD $0x4f33270f // srshr v15.4s, v24.4s, #13

	// Store: vector for output frequency c carries rows; lo half lanes
	// r0-3 and hi half lanes r4-7 of coeff[c*stride+r].
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
	ADD  R4, R3
	VST1 [V8.S4], (R3)
	ADD  $16, R3
	VST1 [V12.S4], (R3)
	SUB  $16, R3
	ADD  R4, R3
	VST1 [V9.S4], (R3)
	ADD  $16, R3
	VST1 [V13.S4], (R3)
	SUB  $16, R3
	ADD  R4, R3
	VST1 [V10.S4], (R3)
	ADD  $16, R3
	VST1 [V14.S4], (R3)
	SUB  $16, R3
	ADD  R4, R3
	VST1 [V11.S4], (R3)
	ADD  $16, R3
	VST1 [V15.S4], (R3)
	SUB  $16, R3
	RET

// fadst8Cospi holds Q13 weights by vector lane:
// v26 = [cospi32, cospi16, cospi48, cospi4],
// v27 = [cospi60, cospi20, cospi44, cospi36],
// v28 = [cospi28, cospi52, cospi12, 0].
GLOBL ·fadst8Cospi(SB), RODATA|NOPTR, $48
DATA ·fadst8Cospi+0(SB)/4, $5793
DATA ·fadst8Cospi+4(SB)/4, $7568
DATA ·fadst8Cospi+8(SB)/4, $3135
DATA ·fadst8Cospi+12(SB)/4, $8153
DATA ·fadst8Cospi+16(SB)/4, $803
DATA ·fadst8Cospi+20(SB)/4, $7225
DATA ·fadst8Cospi+24(SB)/4, $3862
DATA ·fadst8Cospi+28(SB)/4, $5197
DATA ·fadst8Cospi+32(SB)/4, $6333
DATA ·fadst8Cospi+36(SB)/4, $2378
DATA ·fadst8Cospi+40(SB)/4, $7839
DATA ·fadst8Cospi+44(SB)/4, $0
