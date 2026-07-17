// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// inverseIdentityCol4NEON applies the AV1 inverse-identity scale for length
// 4/8/16/32 to four adjacent int32 columns in place. The four columns are
// contiguous within each row; successive rows are rowStrideBytes apart.
//
// The caller limits inputs to the AV1 stage envelope (+/-2^19), so the length
// 4 and 16 fixed-point products are exact in int32 lanes. Length 8 and 32 use
// saturating shifts, retaining identity1DValue's full int32 saturation.
//
// func inverseIdentityCol4NEON(base *int32, rowStrideBytes, length int64)
TEXT ·inverseIdentityCol4NEON(SB), NOSPLIT, $0-24
	MOVD base+0(FP), R0
	MOVD rowStrideBytes+8(FP), R1
	MOVD length+16(FP), R2
	CMP  $4, R2
	BEQ  identity4Setup
	CMP  $8, R2
	BEQ  identity8Loop
	CMP  $16, R2
	BEQ  identity16Setup
	CMP  $32, R2
	BEQ  identity32Loop
	RET

identity4Setup:
	MOVD $1697, R3
	WORD $0x4e040c70 // dup v16.4s, w3
	MOVD $2048, R3
	WORD $0x4e040c71 // dup v17.4s, w3
identity4Loop:
	VLD1 (R0), [V0.S4]
	WORD $0x4eb09c01 // mul v1.4s, v0.4s, v16.4s
	WORD $0x4eb18421 // add v1.4s, v1.4s, v17.4s
	WORD $0x4f340421 // sshr v1.4s, v1.4s, #12
	WORD $0x4ea18400 // add v0.4s, v0.4s, v1.4s
	VST1 [V0.S4], (R0)
	ADD  R1, R0
	SUBS $1, R2, R2
	BNE  identity4Loop
	RET

identity8Loop:
	VLD1 (R0), [V0.S4]
	WORD $0x4f217400 // sqshl v0.4s, v0.4s, #1
	VST1 [V0.S4], (R0)
	ADD  R1, R0
	SUBS $1, R2, R2
	BNE  identity8Loop
	RET

identity16Setup:
	MOVD $1697, R3
	WORD $0x4e040c70 // dup v16.4s, w3
	MOVD $1024, R3
	WORD $0x4e040c71 // dup v17.4s, w3
identity16Loop:
	VLD1 (R0), [V0.S4]
	WORD $0x4eb09c01 // mul v1.4s, v0.4s, v16.4s
	WORD $0x4eb18421 // add v1.4s, v1.4s, v17.4s
	WORD $0x4f350421 // sshr v1.4s, v1.4s, #11
	WORD $0x4f215402 // shl v2.4s, v0.4s, #1
	WORD $0x4ea18440 // add v0.4s, v2.4s, v1.4s
	VST1 [V0.S4], (R0)
	ADD  R1, R0
	SUBS $1, R2, R2
	BNE  identity16Loop
	RET

identity32Loop:
	VLD1 (R0), [V0.S4]
	WORD $0x4f227400 // sqshl v0.4s, v0.4s, #2
	VST1 [V0.S4], (R0)
	ADD  R1, R0
	SUBS $1, R2, R2
	BNE  identity32Loop
	RET
