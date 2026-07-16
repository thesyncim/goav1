// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

#define WH_TMP     0
#define WH_REF     8
#define WH_FILTER  16
#define WH_PERMUTE 24
#define WH_REFSTR  32
#define WH_SXSTART 40
#define WH_ALPHA   48
#define WH_BETA    56

// func warpHorizontal8DotProdAsm(ctx *warpHorizontal8DotProdCtx)
//
// Resident 8-bit horizontal affine-warp pass. Each of 15 rows loads the common
// 15-byte source span once, gathers the eight per-column int8 filters, and uses
// four SDOTs plus two pairwise adds to produce eight exact int32 intermediates.
// Pixels are biased by -128 so SDOT can consume signed bytes. Since every warp
// filter sums to 128, the per-column seed 32772 restores that subtraction,
// adds the scalar path's 16384 horizontal bias, and supplies the +4 round term
// before the exact >>3.
TEXT ·warpHorizontal8DotProdAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD WH_TMP(R0), R1
	MOVD WH_REF(R0), R2
	MOVD WH_FILTER(R0), R3
	MOVD WH_PERMUTE(R0), R4
	MOVD WH_REFSTR(R0), R5
	MOVD WH_SXSTART(R0), R10
	MOVD WH_ALPHA(R0), R7
	MOVD WH_BETA(R0), R8
	MOVD $15, R9

	VLD1.P 16(R4), [V28.B16]
	VLD1.P 16(R4), [V29.B16]
	VLD1.P 16(R4), [V30.B16]
	VLD1   (R4), [V31.B16]
	MOVD $32772, R14
	WORD $0x4e081dd8 // mov v24.d[0], x14
	WORD $0x4e181dd8 // mov v24.d[1], x14
	WORD $0x4f04e417 // movi v23.16b, #128

warpHRowLoop:
	// Filter index = ((sx + 512) >> 10) + 64. Folding 64<<10 into
	// the input gives (sx + 66048) >> 10; the Go-side corner check proves all
	// eight indices are in [0,192].
	ADD $66048, R10, R12

	ASR $10, R12, R13
	MOVD (R3)(R13<<3), R19
	ADD R7, R12, R12
	ASR $10, R12, R13
	MOVD (R3)(R13<<3), R20
	WORD $0x4e081e60 // mov v0.d[0], x19
	WORD $0x4e181e80 // mov v0.d[1], x20
	ADD R7, R12, R12

	ASR $10, R12, R13
	MOVD (R3)(R13<<3), R19
	ADD R7, R12, R12
	ASR $10, R12, R13
	MOVD (R3)(R13<<3), R20
	WORD $0x4e081e61 // mov v1.d[0], x19
	WORD $0x4e181e81 // mov v1.d[1], x20
	ADD R7, R12, R12

	ASR $10, R12, R13
	MOVD (R3)(R13<<3), R19
	ADD R7, R12, R12
	ASR $10, R12, R13
	MOVD (R3)(R13<<3), R20
	WORD $0x4e081e62 // mov v2.d[0], x19
	WORD $0x4e181e82 // mov v2.d[1], x20
	ADD R7, R12, R12

	ASR $10, R12, R13
	MOVD (R3)(R13<<3), R19
	ADD R7, R12, R12
	ASR $10, R12, R13
	MOVD (R3)(R13<<3), R20
	WORD $0x4e081e63 // mov v3.d[0], x19
	WORD $0x4e181e83 // mov v3.d[1], x20

	VLD1 (R2), [V4.B16]
	WORD $0x6e378484 // sub v4.16b, v4.16b, v23.16b
	WORD $0x4e1c0085 // tbl v5.16b, {v4.16b}, v28.16b
	WORD $0x4e1d0086 // tbl v6.16b, {v4.16b}, v29.16b
	WORD $0x4e1e0087 // tbl v7.16b, {v4.16b}, v30.16b
	WORD $0x4e1f0088 // tbl v8.16b, {v4.16b}, v31.16b

	WORD $0x4eb81f10 // mov v16.16b, v24.16b
	WORD $0x4eb81f11 // mov v17.16b, v24.16b
	WORD $0x4eb81f12 // mov v18.16b, v24.16b
	WORD $0x4eb81f13 // mov v19.16b, v24.16b
	WORD $0x4e8094b0 // sdot v16.4s, v5.16b, v0.16b
	WORD $0x4e8194d1 // sdot v17.4s, v6.16b, v1.16b
	WORD $0x4e8294f2 // sdot v18.4s, v7.16b, v2.16b
	WORD $0x4e839513 // sdot v19.4s, v8.16b, v3.16b
	WORD $0x4eb1be10 // addp v16.4s, v16.4s, v17.4s
	WORD $0x4eb3be52 // addp v18.4s, v18.4s, v19.4s
	WORD $0x4f3d0610 // sshr v16.4s, v16.4s, #3
	WORD $0x4f3d0652 // sshr v18.4s, v18.4s, #3
	VST1 [V16.S4], (R1)
	ADD $16, R1, R1
	VST1 [V18.S4], (R1)
	ADD $16, R1, R1

	ADD R5, R2, R2
	ADD R8, R10, R10
	SUB $1, R9, R9
	CBNZ R9, warpHRowLoop
	RET
