// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

#define SCALE_DST    0
#define SCALE_SRC    8
#define SCALE_GROUPS 16

// Exact nearest-neighbor 2:1 and 4:1 row downscalers for WebRTC spatial layer
// preparation. Each group emits 16 output pixels by deinterleaving the source
// row and keeping phase zero, matching:
//
//   dst[x] = src[x*2]  for 2:1
//   dst[x] = src[x*4]  for 4:1
//
// Instructions missing from Go's assembler are emitted as WORD:
//   ld2 {v0.16b, v1.16b}, [x2]             -> 0x4c408040
//   ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x2] -> 0x4c400040
//   ld2 {v0.8h,  v1.8h},  [x2]             -> 0x4c408440
//   ld4 {v0.8h,  v1.8h,  v2.8h,  v3.8h},  [x2] -> 0x4c400440

// func scaleNearestRowDown2NEON(ctx *scaleNearestRowCtx)
TEXT ·scaleNearestRowDown2NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD SCALE_DST(R0), R1
	MOVD SCALE_SRC(R0), R2
	MOVD SCALE_GROUPS(R0), R3
	CBZ  R3, down2Done

down2Loop:
	WORD $0x4c408040 // ld2 {v0.16b, v1.16b}, [x2]
	WORD $0x3d800020 // str q0, [x1]
	ADD  $32, R2
	ADD  $16, R1
	SUB  $1, R3
	CBNZ R3, down2Loop

down2Done:
	RET

// func scaleNearestRowDown4NEON(ctx *scaleNearestRowCtx)
TEXT ·scaleNearestRowDown4NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD SCALE_DST(R0), R1
	MOVD SCALE_SRC(R0), R2
	MOVD SCALE_GROUPS(R0), R3
	CBZ  R3, down4Done

down4Loop:
	WORD $0x4c400040 // ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x2]
	WORD $0x3d800020 // str q0, [x1]
	ADD  $64, R2
	ADD  $16, R1
	SUB  $1, R3
	CBNZ R3, down4Loop

down4Done:
	RET

// func scaleNearestRow16Down2NEON(ctx *scaleNearestRowCtx)
TEXT ·scaleNearestRow16Down2NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD SCALE_DST(R0), R1
	MOVD SCALE_SRC(R0), R2
	MOVD SCALE_GROUPS(R0), R3
	CBZ  R3, down2x16Done

down2x16Loop:
	WORD $0x4c408440 // ld2 {v0.8h, v1.8h}, [x2]
	WORD $0x3d800020 // str q0, [x1]
	ADD  $32, R2
	ADD  $16, R1
	SUB  $1, R3
	CBNZ R3, down2x16Loop

down2x16Done:
	RET

// func scaleNearestRow16Down4NEON(ctx *scaleNearestRowCtx)
TEXT ·scaleNearestRow16Down4NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD SCALE_DST(R0), R1
	MOVD SCALE_SRC(R0), R2
	MOVD SCALE_GROUPS(R0), R3
	CBZ  R3, down4x16Done

down4x16Loop:
	WORD $0x4c400440 // ld4 {v0.8h, v1.8h, v2.8h, v3.8h}, [x2]
	WORD $0x3d800020 // str q0, [x1]
	ADD  $64, R2
	ADD  $16, R1
	SUB  $1, R3
	CBNZ R3, down4x16Loop

down4x16Done:
	RET
