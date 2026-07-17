// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

#define I_DST         0
#define I_DSTSTRIDE   8
#define I_COEFF       16
#define I_COEFFSTRIDE 24
#define I_ROWGROUPS   32
#define I_ROWMIN      40
#define I_ROWMAX      44
#define I_COLMIN      48
#define I_COLMAX      52

// inverseIdentity16Rows4NEONAsm transforms four rows by four columns per tile.
// Four column-major vector loads become four row-major vectors after a 4x4
// transpose. All fixed-point stages and clamps precede the transpose because
// they are lane-local; the final SQRSHRN both rounds by four and saturates to
// the int16 residual representation.
TEXT ·inverseIdentity16Rows4NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD I_DST(R0), R1
	MOVD I_DSTSTRIDE(R0), R2
	LSL  $1, R2, R2
	MOVD I_COEFF(R0), R3
	MOVD I_COEFFSTRIDE(R0), R4
	LSL  $2, R4, R4
	MOVD I_ROWGROUPS(R0), R5

	MOVW I_ROWMIN(R0), R6
	WORD $0x4e040cd4 // dup v20.4s, w6
	MOVW I_ROWMAX(R0), R6
	WORD $0x4e040cd5 // dup v21.4s, w6
	MOVW I_COLMIN(R0), R6
	WORD $0x4e040cd6 // dup v22.4s, w6
	MOVW I_COLMAX(R0), R6
	WORD $0x4e040cd7 // dup v23.4s, w6
	MOVD $1697, R6
	WORD $0x4e040cd8 // dup v24.4s, w6
	MOVD $1024, R6
	WORD $0x4e040cd9 // dup v25.4s, w6

identity16RowGroup:
	MOVD R1, R10
	MOVD R3, R11
	MOVD $4, R9

identity16Tile:
	VLD1 (R11), [V0.S4]
	ADD  R4, R11
	VLD1 (R11), [V1.S4]
	ADD  R4, R11
	VLD1 (R11), [V2.S4]
	ADD  R4, R11
	VLD1 (R11), [V3.S4]
	ADD  R4, R11

	// Row-stage clamp and horizontal identity16.
	WORD $0x4eb46400 // smax v0.4s, v0.4s, v20.4s
	WORD $0x4eb56c00 // smin v0.4s, v0.4s, v21.4s
	WORD $0x4eb89c08 // mul v8.4s, v0.4s, v24.4s
	WORD $0x4eb98508 // add v8.4s, v8.4s, v25.4s
	WORD $0x4f350508 // sshr v8.4s, v8.4s, #11
	WORD $0x4f215400 // shl v0.4s, v0.4s, #1
	WORD $0x4ea88400 // add v0.4s, v0.4s, v8.4s

	WORD $0x4eb46421 // smax v1.4s, v1.4s, v20.4s
	WORD $0x4eb56c21 // smin v1.4s, v1.4s, v21.4s
	WORD $0x4eb89c29 // mul v9.4s, v1.4s, v24.4s
	WORD $0x4eb98529 // add v9.4s, v9.4s, v25.4s
	WORD $0x4f350529 // sshr v9.4s, v9.4s, #11
	WORD $0x4f215421 // shl v1.4s, v1.4s, #1
	WORD $0x4ea98421 // add v1.4s, v1.4s, v9.4s

	WORD $0x4eb46442 // smax v2.4s, v2.4s, v20.4s
	WORD $0x4eb56c42 // smin v2.4s, v2.4s, v21.4s
	WORD $0x4eb89c4a // mul v10.4s, v2.4s, v24.4s
	WORD $0x4eb9854a // add v10.4s, v10.4s, v25.4s
	WORD $0x4f35054a // sshr v10.4s, v10.4s, #11
	WORD $0x4f215442 // shl v2.4s, v2.4s, #1
	WORD $0x4eaa8442 // add v2.4s, v2.4s, v10.4s

	WORD $0x4eb46463 // smax v3.4s, v3.4s, v20.4s
	WORD $0x4eb56c63 // smin v3.4s, v3.4s, v21.4s
	WORD $0x4eb89c6b // mul v11.4s, v3.4s, v24.4s
	WORD $0x4eb9856b // add v11.4s, v11.4s, v25.4s
	WORD $0x4f35056b // sshr v11.4s, v11.4s, #11
	WORD $0x4f215463 // shl v3.4s, v3.4s, #1
	WORD $0x4eab8463 // add v3.4s, v3.4s, v11.4s

	// 16x16 uses the AV1 mid-pass shift of two, then the column clamp.
	WORD $0x4f3e2400 // srshr v0.4s, v0.4s, #2
	WORD $0x4f3e2421 // srshr v1.4s, v1.4s, #2
	WORD $0x4f3e2442 // srshr v2.4s, v2.4s, #2
	WORD $0x4f3e2463 // srshr v3.4s, v3.4s, #2
	WORD $0x4eb66400 // smax v0.4s, v0.4s, v22.4s
	WORD $0x4eb76c00 // smin v0.4s, v0.4s, v23.4s
	WORD $0x4eb66421 // smax v1.4s, v1.4s, v22.4s
	WORD $0x4eb76c21 // smin v1.4s, v1.4s, v23.4s
	WORD $0x4eb66442 // smax v2.4s, v2.4s, v22.4s
	WORD $0x4eb76c42 // smin v2.4s, v2.4s, v23.4s
	WORD $0x4eb66463 // smax v3.4s, v3.4s, v22.4s
	WORD $0x4eb76c63 // smin v3.4s, v3.4s, v23.4s

	// Vertical identity16.
	WORD $0x4eb89c08 // mul v8.4s, v0.4s, v24.4s
	WORD $0x4eb98508 // add v8.4s, v8.4s, v25.4s
	WORD $0x4f350508 // sshr v8.4s, v8.4s, #11
	WORD $0x4f215400 // shl v0.4s, v0.4s, #1
	WORD $0x4ea88400 // add v0.4s, v0.4s, v8.4s
	WORD $0x4eb89c29 // mul v9.4s, v1.4s, v24.4s
	WORD $0x4eb98529 // add v9.4s, v9.4s, v25.4s
	WORD $0x4f350529 // sshr v9.4s, v9.4s, #11
	WORD $0x4f215421 // shl v1.4s, v1.4s, #1
	WORD $0x4ea98421 // add v1.4s, v1.4s, v9.4s
	WORD $0x4eb89c4a // mul v10.4s, v2.4s, v24.4s
	WORD $0x4eb9854a // add v10.4s, v10.4s, v25.4s
	WORD $0x4f35054a // sshr v10.4s, v10.4s, #11
	WORD $0x4f215442 // shl v2.4s, v2.4s, #1
	WORD $0x4eaa8442 // add v2.4s, v2.4s, v10.4s
	WORD $0x4eb89c6b // mul v11.4s, v3.4s, v24.4s
	WORD $0x4eb9856b // add v11.4s, v11.4s, v25.4s
	WORD $0x4f35056b // sshr v11.4s, v11.4s, #11
	WORD $0x4f215463 // shl v3.4s, v3.4s, #1
	WORD $0x4eab8463 // add v3.4s, v3.4s, v11.4s

	// Transpose the four coefficient columns into output rows.
	WORD $0x4e812810 // trn1 v16.4s, v0.4s, v1.4s
	WORD $0x4e816811 // trn2 v17.4s, v0.4s, v1.4s
	WORD $0x4e832852 // trn1 v18.4s, v2.4s, v3.4s
	WORD $0x4e836853 // trn2 v19.4s, v2.4s, v3.4s
	WORD $0x4ed23a04 // zip1 v4.2d, v16.2d, v18.2d
	WORD $0x4ed33a25 // zip1 v5.2d, v17.2d, v19.2d
	WORD $0x4ed27a06 // zip2 v6.2d, v16.2d, v18.2d
	WORD $0x4ed37a27 // zip2 v7.2d, v17.2d, v19.2d
	WORD $0x0f1c9c84 // sqrshrn v4.4h, v4.4s, #4
	WORD $0x0f1c9ca5 // sqrshrn v5.4h, v5.4s, #4
	WORD $0x0f1c9cc6 // sqrshrn v6.4h, v6.4s, #4
	WORD $0x0f1c9ce7 // sqrshrn v7.4h, v7.4s, #4

	VST1 [V4.H4], (R10)
	ADD  R2, R10, R12
	VST1 [V5.H4], (R12)
	ADD  R2, R12, R13
	VST1 [V6.H4], (R13)
	ADD  R2, R13, R14
	VST1 [V7.H4], (R14)
	ADD  $8, R10
	SUBS $1, R9, R9
	BNE  identity16Tile

	LSL  $2, R2, R12
	ADD  R12, R1, R1
	ADD  $16, R3
	SUBS $1, R5, R5
	BNE  identity16RowGroup
	RET
