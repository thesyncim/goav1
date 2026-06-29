// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

#define IP_DST    0
#define IP_REF    8
#define IP_STRIDE 16
#define IP_WIDTH  24
#define IP_HEIGHT 32

// Low-bitdepth integer projection row and column kernels, source-shaped from
// libaom's aom_int_pro_row_neon and aom_int_pro_col_neon. The Go dispatch
// mirrors libaom's preconditions: width is a multiple of 16, height is a
// multiple of 4, and realtime 64x64 projection uses norm_factor == 5.
//
// Instructions missing from Go's assembler are emitted as WORD:
//   uaddl  v16.8h, v0.8b,  v1.8b   -> 0x2e210010
//   uaddl2 v17.8h, v0.16b, v1.16b  -> 0x6e210011
//   uaddl  v18.8h, v2.8b,  v3.8b   -> 0x2e230052
//   uaddl2 v19.8h, v2.16b, v3.16b  -> 0x6e230053
//   uaddlp v16.8h, v0.16b          -> 0x6e202810
//   uadalp v16.8h, v0.16b          -> 0x6e206810
//   uaddlv s20,    v16.8h          -> 0x6e703a14
//   ushr   v16.8h, v16.8h, #5      -> 0x6f1b0610

// func realtimeIntProRowNEONAsm(ctx *realtimeIntProNEONCtx)
TEXT ·realtimeIntProRowNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD IP_DST(R0), R1
	MOVD IP_REF(R0), R2
	MOVD IP_STRIDE(R0), R3
	MOVD IP_WIDTH(R0), R4
	MOVD IP_HEIGHT(R0), R5

rowWidthLoop:
	MOVD R2, R6

	VLD1 (R6), [V0.B16]
	ADD  R3, R6
	VLD1 (R6), [V1.B16]
	ADD  R3, R6
	VLD1 (R6), [V2.B16]
	ADD  R3, R6
	VLD1 (R6), [V3.B16]
	ADD  R3, R6

	WORD $0x2e210010 // uaddl  v16.8h, v0.8b,  v1.8b
	WORD $0x6e210011 // uaddl2 v17.8h, v0.16b, v1.16b
	WORD $0x2e230052 // uaddl  v18.8h, v2.8b,  v3.8b
	WORD $0x6e230053 // uaddl2 v19.8h, v2.16b, v3.16b
	VADD V18.H8, V16.H8, V16.H8
	VADD V19.H8, V17.H8, V17.H8

	MOVD R5, R7
	SUB  $4, R7
rowHeightLoop:
	CBZ R7, rowStore

	VLD1 (R6), [V0.B16]
	ADD  R3, R6
	VLD1 (R6), [V1.B16]
	ADD  R3, R6
	VLD1 (R6), [V2.B16]
	ADD  R3, R6
	VLD1 (R6), [V3.B16]
	ADD  R3, R6

	WORD $0x2e210012 // uaddl  v18.8h, v0.8b,  v1.8b
	WORD $0x6e210013 // uaddl2 v19.8h, v0.16b, v1.16b
	WORD $0x2e230054 // uaddl  v20.8h, v2.8b,  v3.8b
	WORD $0x6e230055 // uaddl2 v21.8h, v2.16b, v3.16b
	VADD V18.H8, V16.H8, V16.H8
	VADD V19.H8, V17.H8, V17.H8
	VADD V20.H8, V16.H8, V16.H8
	VADD V21.H8, V17.H8, V17.H8

	SUB  $4, R7
	B    rowHeightLoop

rowStore:
	WORD $0x6f1b0610 // ushr v16.8h, v16.8h, #5
	WORD $0x6f1b0631 // ushr v17.8h, v17.8h, #5
	VST1 [V16.H8], (R1)
	ADD  $16, R1
	VST1 [V17.H8], (R1)
	ADD  $16, R1
	ADD  $16, R2
	SUB  $16, R4
	CBNZ R4, rowWidthLoop
	RET

// func realtimeIntProColNEONAsm(ctx *realtimeIntProNEONCtx)
TEXT ·realtimeIntProColNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD IP_DST(R0), R1
	MOVD IP_REF(R0), R2
	MOVD IP_STRIDE(R0), R3
	MOVD IP_WIDTH(R0), R4
	MOVD IP_HEIGHT(R0), R5
	LSL  $2, R3, R11

colHeightLoop:
	MOVD R2, R6
	VLD1 (R6), [V0.B16]
	ADD  R3, R6
	VLD1 (R6), [V1.B16]
	ADD  R3, R6
	VLD1 (R6), [V2.B16]
	ADD  R3, R6
	VLD1 (R6), [V3.B16]

	WORD $0x6e202810 // uaddlp v16.8h, v0.16b
	WORD $0x6e202831 // uaddlp v17.8h, v1.16b
	WORD $0x6e202852 // uaddlp v18.8h, v2.16b
	WORD $0x6e202873 // uaddlp v19.8h, v3.16b

	MOVD R4, R7
	SUB  $16, R7
	MOVD R2, R8
	ADD  $16, R8
colWidthLoop:
	CBZ R7, colStore
	MOVD R8, R6
	VLD1 (R6), [V0.B16]
	ADD  R3, R6
	VLD1 (R6), [V1.B16]
	ADD  R3, R6
	VLD1 (R6), [V2.B16]
	ADD  R3, R6
	VLD1 (R6), [V3.B16]

	WORD $0x6e206810 // uadalp v16.8h, v0.16b
	WORD $0x6e206831 // uadalp v17.8h, v1.16b
	WORD $0x6e206852 // uadalp v18.8h, v2.16b
	WORD $0x6e206873 // uadalp v19.8h, v3.16b

	ADD  $16, R8
	SUB  $16, R7
	B    colWidthLoop

colStore:
	WORD $0x6e703a14 // uaddlv s20, v16.8h
	WORD $0x6e703a35 // uaddlv s21, v17.8h
	WORD $0x6e703a56 // uaddlv s22, v18.8h
	WORD $0x6e703a77 // uaddlv s23, v19.8h

	VMOV V20.S[0], R6
	LSR  $5, R6
	MOVH R6, 0(R1)
	VMOV V21.S[0], R6
	LSR  $5, R6
	MOVH R6, 2(R1)
	VMOV V22.S[0], R6
	LSR  $5, R6
	MOVH R6, 4(R1)
	VMOV V23.S[0], R6
	LSR  $5, R6
	MOVH R6, 6(R1)

	ADD  $8, R1
	ADD  R11, R2
	SUB  $4, R5
	CBNZ R5, colHeightLoop
	RET
