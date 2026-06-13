// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// NEON 8x8 sum of absolute differences. Eight rows of 8 bytes are loaded from
// src and ref, accumulated with the widening absolute-difference instructions
// (UABDL for row 0, UABAL for rows 1-7, emitted via WORD because the Go
// assembler lacks the mnemonics), then reduced with UADDLV.
//
//   uabdl v2.8h, v0.8b, v1.8b   -> 0x2e217002
//   uabal v2.8h, v0.8b, v1.8b   -> 0x2e215002
//   uaddlv s3, v2.8h            -> 0x6e703843

#define SRC    0
#define REF    8
#define STRIDE 16
#define SUM    24

// func sad8x8NEONAsm(ctx *sad8x8NEONCtx)
TEXT ·sad8x8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD SRC(R0), R1
	MOVD REF(R0), R2
	MOVD STRIDE(R0), R3

	// Row 0: initialize the uint16 accumulator lanes.
	VLD1 (R1), [V0.B8]
	VLD1 (R2), [V1.B8]
	WORD $0x2e217002 // uabdl v2.8h, v0.8b, v1.8b
	ADD  R3, R1
	ADD  R3, R2

	// Rows 1-7: accumulate.
	MOVD $7, R4
loop:
	VLD1 (R1), [V0.B8]
	VLD1 (R2), [V1.B8]
	WORD $0x2e215002 // uabal v2.8h, v0.8b, v1.8b
	ADD  R3, R1
	ADD  R3, R2
	SUB  $1, R4
	CBNZ R4, loop

	WORD $0x6e703843 // uaddlv s3, v2.8h
	VMOV V3.S[0], R5
	MOVD R5, SUM(R0)
	RET

#define X4SUM0 24
#define X4SUM1 32
#define X4SUM2 40
#define X4SUM3 48

// NEON 8x8 SAD for four horizontal candidates separated by four pixels. Each
// row loads one source vector plus enough reference bytes for windows at
// ref+0, ref+4, ref+8, and ref+12.
//
//   ext   v10.16b, v1.16b, v5.16b, #4   -> 0x6e05202a
//   ext   v11.16b, v1.16b, v5.16b, #8   -> 0x6e05402b
//   ext   v12.16b, v1.16b, v5.16b, #12  -> 0x6e05602c

// func sad8x8x4Step4NEONAsm(ctx *sad8x8x4Step4NEONCtx)
TEXT ·sad8x8x4Step4NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD SRC(R0), R1
	MOVD REF(R0), R2
	MOVD STRIDE(R0), R3

	VLD1 (R1), [V0.B8]
	VLD1 (R2), [V1.B16]
	ADD  $16, R2, R7
	VLD1 (R7), [V5.B16]
	WORD $0x6e05202a // ext   v10.16b, v1.16b, v5.16b, #4
	WORD $0x6e05402b // ext   v11.16b, v1.16b, v5.16b, #8
	WORD $0x6e05602c // ext   v12.16b, v1.16b, v5.16b, #12
	WORD $0x2e217002 // uabdl v2.8h, v0.8b, v1.8b
	WORD $0x2e2a7003 // uabdl v3.8h, v0.8b, v10.8b
	WORD $0x2e2b7004 // uabdl v4.8h, v0.8b, v11.8b
	WORD $0x2e2c7006 // uabdl v6.8h, v0.8b, v12.8b
	ADD  R3, R1
	ADD  R3, R2

	MOVD $7, R4
x4loop:
	VLD1 (R1), [V0.B8]
	VLD1 (R2), [V1.B16]
	ADD  $16, R2, R7
	VLD1 (R7), [V5.B16]
	WORD $0x6e05202a // ext   v10.16b, v1.16b, v5.16b, #4
	WORD $0x6e05402b // ext   v11.16b, v1.16b, v5.16b, #8
	WORD $0x6e05602c // ext   v12.16b, v1.16b, v5.16b, #12
	WORD $0x2e215002 // uabal v2.8h, v0.8b, v1.8b
	WORD $0x2e2a5003 // uabal v3.8h, v0.8b, v10.8b
	WORD $0x2e2b5004 // uabal v4.8h, v0.8b, v11.8b
	WORD $0x2e2c5006 // uabal v6.8h, v0.8b, v12.8b
	ADD  R3, R1
	ADD  R3, R2
	SUB  $1, R4
	CBNZ R4, x4loop

	WORD $0x6e70384d // uaddlv s13, v2.8h
	WORD $0x6e70386e // uaddlv s14, v3.8h
	WORD $0x6e70388f // uaddlv s15, v4.8h
	WORD $0x6e7038d0 // uaddlv s16, v6.8h
	VMOV V13.S[0], R5
	MOVD R5, X4SUM0(R0)
	VMOV V14.S[0], R5
	MOVD R5, X4SUM1(R0)
	VMOV V15.S[0], R5
	MOVD R5, X4SUM2(R0)
	VMOV V16.S[0], R5
	MOVD R5, X4SUM3(R0)
	RET

// NEON 16x16 SAD for four horizontal candidates separated by four pixels.
// Each candidate keeps separate low/high uint16 accumulators, matching the
// single-candidate 16x16 kernel while sharing source loads across four refs.

// func sad16x16x4Step4NEONAsm(ctx *sad8x8x4Step4NEONCtx)
TEXT ·sad16x16x4Step4NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD SRC(R0), R1
	MOVD REF(R0), R2
	MOVD STRIDE(R0), R3

	VLD1 (R1), [V0.B16]
	VLD1 (R2), [V1.B16]
	ADD  $16, R2, R10
	VLD1 (R10), [V5.B16]
	WORD $0x6e05202a // ext    v10.16b, v1.16b, v5.16b, #4
	WORD $0x6e05402b // ext    v11.16b, v1.16b, v5.16b, #8
	WORD $0x6e05602c // ext    v12.16b, v1.16b, v5.16b, #12
	WORD $0x2e217002 // uabdl  v2.8h, v0.8b,  v1.8b
	WORD $0x6e217003 // uabdl2 v3.8h, v0.16b, v1.16b
	WORD $0x2e2a7004 // uabdl  v4.8h, v0.8b,  v10.8b
	WORD $0x6e2a7006 // uabdl2 v6.8h, v0.16b, v10.16b
	WORD $0x2e2b7007 // uabdl  v7.8h, v0.8b,  v11.8b
	WORD $0x6e2b7008 // uabdl2 v8.8h, v0.16b, v11.16b
	WORD $0x2e2c7009 // uabdl  v9.8h, v0.8b,  v12.8b
	WORD $0x6e2c700d // uabdl2 v13.8h, v0.16b, v12.16b
	ADD  R3, R1
	ADD  R3, R2

	MOVD $15, R4
x4loop16:
	VLD1 (R1), [V0.B16]
	VLD1 (R2), [V1.B16]
	ADD  $16, R2, R10
	VLD1 (R10), [V5.B16]
	WORD $0x6e05202a // ext    v10.16b, v1.16b, v5.16b, #4
	WORD $0x6e05402b // ext    v11.16b, v1.16b, v5.16b, #8
	WORD $0x6e05602c // ext    v12.16b, v1.16b, v5.16b, #12
	WORD $0x2e215002 // uabal  v2.8h, v0.8b,  v1.8b
	WORD $0x6e215003 // uabal2 v3.8h, v0.16b, v1.16b
	WORD $0x2e2a5004 // uabal  v4.8h, v0.8b,  v10.8b
	WORD $0x6e2a5006 // uabal2 v6.8h, v0.16b, v10.16b
	WORD $0x2e2b5007 // uabal  v7.8h, v0.8b,  v11.8b
	WORD $0x6e2b5008 // uabal2 v8.8h, v0.16b, v11.16b
	WORD $0x2e2c5009 // uabal  v9.8h, v0.8b,  v12.8b
	WORD $0x6e2c500d // uabal2 v13.8h, v0.16b, v12.16b
	ADD  R3, R1
	ADD  R3, R2
	SUB  $1, R4
	CBNZ R4, x4loop16

	VADD V3.H8, V2.H8, V2.H8
	VADD V6.H8, V4.H8, V4.H8
	VADD V8.H8, V7.H8, V7.H8
	VADD V13.H8, V9.H8, V9.H8
	WORD $0x6e70384e // uaddlv s14, v2.8h
	WORD $0x6e70388f // uaddlv s15, v4.8h
	WORD $0x6e7038f0 // uaddlv s16, v7.8h
	WORD $0x6e703931 // uaddlv s17, v9.8h
	VMOV V14.S[0], R5
	MOVD R5, X4SUM0(R0)
	VMOV V15.S[0], R5
	MOVD R5, X4SUM1(R0)
	VMOV V16.S[0], R5
	MOVD R5, X4SUM2(R0)
	VMOV V17.S[0], R5
	MOVD R5, X4SUM3(R0)
	RET

// NEON 32x32 SAD for four horizontal candidates separated by four pixels. Each
// row consumes two 16-byte source chunks; the second chunk accumulates into the
// same low/high candidate sums as the first.

// func sad32x32x4Step4NEONAsm(ctx *sad8x8x4Step4NEONCtx)
TEXT ·sad32x32x4Step4NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD SRC(R0), R1
	MOVD REF(R0), R2
	MOVD STRIDE(R0), R3

	VLD1 (R1), [V0.B16]
	ADD  $16, R1, R11
	VLD1 (R11), [V15.B16]
	VLD1 (R2), [V1.B16]
	ADD  $16, R2, R10
	VLD1 (R10), [V5.B16]
	ADD  $32, R2, R12
	VLD1 (R12), [V14.B16]
	WORD $0x6e05202a // ext    v10.16b, v1.16b,  v5.16b,  #4
	WORD $0x6e05402b // ext    v11.16b, v1.16b,  v5.16b,  #8
	WORD $0x6e05602c // ext    v12.16b, v1.16b,  v5.16b,  #12
	WORD $0x6e0e20b2 // ext    v18.16b, v5.16b,  v14.16b, #4
	WORD $0x6e0e40b3 // ext    v19.16b, v5.16b,  v14.16b, #8
	WORD $0x6e0e60b4 // ext    v20.16b, v5.16b,  v14.16b, #12
	WORD $0x2e217002 // uabdl  v2.8h,  v0.8b,   v1.8b
	WORD $0x6e217003 // uabdl2 v3.8h,  v0.16b,  v1.16b
	WORD $0x2e2a7004 // uabdl  v4.8h,  v0.8b,   v10.8b
	WORD $0x6e2a7006 // uabdl2 v6.8h,  v0.16b,  v10.16b
	WORD $0x2e2b7007 // uabdl  v7.8h,  v0.8b,   v11.8b
	WORD $0x6e2b7008 // uabdl2 v8.8h,  v0.16b,  v11.16b
	WORD $0x2e2c7009 // uabdl  v9.8h,  v0.8b,   v12.8b
	WORD $0x6e2c700d // uabdl2 v13.8h, v0.16b,  v12.16b
	WORD $0x2e2551e2 // uabal  v2.8h,  v15.8b,  v5.8b
	WORD $0x6e2551e3 // uabal2 v3.8h,  v15.16b, v5.16b
	WORD $0x2e3251e4 // uabal  v4.8h,  v15.8b,  v18.8b
	WORD $0x6e3251e6 // uabal2 v6.8h,  v15.16b, v18.16b
	WORD $0x2e3351e7 // uabal  v7.8h,  v15.8b,  v19.8b
	WORD $0x6e3351e8 // uabal2 v8.8h,  v15.16b, v19.16b
	WORD $0x2e3451e9 // uabal  v9.8h,  v15.8b,  v20.8b
	WORD $0x6e3451ed // uabal2 v13.8h, v15.16b, v20.16b
	ADD  R3, R1
	ADD  R3, R2

	MOVD $31, R4
x4loop32:
	VLD1 (R1), [V0.B16]
	ADD  $16, R1, R11
	VLD1 (R11), [V15.B16]
	VLD1 (R2), [V1.B16]
	ADD  $16, R2, R10
	VLD1 (R10), [V5.B16]
	ADD  $32, R2, R12
	VLD1 (R12), [V14.B16]
	WORD $0x6e05202a // ext    v10.16b, v1.16b,  v5.16b,  #4
	WORD $0x6e05402b // ext    v11.16b, v1.16b,  v5.16b,  #8
	WORD $0x6e05602c // ext    v12.16b, v1.16b,  v5.16b,  #12
	WORD $0x6e0e20b2 // ext    v18.16b, v5.16b,  v14.16b, #4
	WORD $0x6e0e40b3 // ext    v19.16b, v5.16b,  v14.16b, #8
	WORD $0x6e0e60b4 // ext    v20.16b, v5.16b,  v14.16b, #12
	WORD $0x2e215002 // uabal  v2.8h,  v0.8b,   v1.8b
	WORD $0x6e215003 // uabal2 v3.8h,  v0.16b,  v1.16b
	WORD $0x2e2a5004 // uabal  v4.8h,  v0.8b,   v10.8b
	WORD $0x6e2a5006 // uabal2 v6.8h,  v0.16b,  v10.16b
	WORD $0x2e2b5007 // uabal  v7.8h,  v0.8b,   v11.8b
	WORD $0x6e2b5008 // uabal2 v8.8h,  v0.16b,  v11.16b
	WORD $0x2e2c5009 // uabal  v9.8h,  v0.8b,   v12.8b
	WORD $0x6e2c500d // uabal2 v13.8h, v0.16b,  v12.16b
	WORD $0x2e2551e2 // uabal  v2.8h,  v15.8b,  v5.8b
	WORD $0x6e2551e3 // uabal2 v3.8h,  v15.16b, v5.16b
	WORD $0x2e3251e4 // uabal  v4.8h,  v15.8b,  v18.8b
	WORD $0x6e3251e6 // uabal2 v6.8h,  v15.16b, v18.16b
	WORD $0x2e3351e7 // uabal  v7.8h,  v15.8b,  v19.8b
	WORD $0x6e3351e8 // uabal2 v8.8h,  v15.16b, v19.16b
	WORD $0x2e3451e9 // uabal  v9.8h,  v15.8b,  v20.8b
	WORD $0x6e3451ed // uabal2 v13.8h, v15.16b, v20.16b
	ADD  R3, R1
	ADD  R3, R2
	SUB  $1, R4
	CBNZ R4, x4loop32

	VADD V3.H8, V2.H8, V2.H8
	VADD V6.H8, V4.H8, V4.H8
	VADD V8.H8, V7.H8, V7.H8
	VADD V13.H8, V9.H8, V9.H8
	WORD $0x6e70384e // uaddlv s14, v2.8h
	WORD $0x6e70388f // uaddlv s15, v4.8h
	WORD $0x6e7038f0 // uaddlv s16, v7.8h
	WORD $0x6e703931 // uaddlv s17, v9.8h
	VMOV V14.S[0], R5
	MOVD R5, X4SUM0(R0)
	VMOV V15.S[0], R5
	MOVD R5, X4SUM1(R0)
	VMOV V16.S[0], R5
	MOVD R5, X4SUM2(R0)
	VMOV V17.S[0], R5
	MOVD R5, X4SUM3(R0)
	RET

// NEON 16x16 sum of absolute differences: 16-byte rows accumulated with the
// paired widening instructions (UABDL/UABDL2 for row 0, UABAL/UABAL2 for the
// rest), the two uint16 accumulators added and reduced with UADDLV.
//
//   uabdl  v2.8h, v0.8b,  v1.8b    -> 0x2e217002
//   uabdl2 v3.8h, v0.16b, v1.16b   -> 0x6e217003
//   uabal  v2.8h, v0.8b,  v1.8b    -> 0x2e215002
//   uabal2 v3.8h, v0.16b, v1.16b   -> 0x6e215003
//   uaddlv s4,    v2.8h            -> 0x6e703844

// func sad16x16NEONAsm(ctx *sad8x8NEONCtx)
TEXT ·sad16x16NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD SRC(R0), R1
	MOVD REF(R0), R2
	MOVD STRIDE(R0), R3

	// Row 0: initialize both uint16 accumulators.
	VLD1 (R1), [V0.B16]
	VLD1 (R2), [V1.B16]
	WORD $0x2e217002 // uabdl  v2.8h, v0.8b, v1.8b
	WORD $0x6e217003 // uabdl2 v3.8h, v0.16b, v1.16b
	ADD  R3, R1
	ADD  R3, R2

	// Rows 1-15: accumulate.
	MOVD $15, R4
loop16:
	VLD1 (R1), [V0.B16]
	VLD1 (R2), [V1.B16]
	WORD $0x2e215002 // uabal  v2.8h, v0.8b, v1.8b
	WORD $0x6e215003 // uabal2 v3.8h, v0.16b, v1.16b
	ADD  R3, R1
	ADD  R3, R2
	SUB  $1, R4
	CBNZ R4, loop16

	VADD V3.H8, V2.H8, V2.H8
	WORD $0x6e703844 // uaddlv s4, v2.8h
	VMOV V4.S[0], R5
	MOVD R5, SUM(R0)
	RET

// NEON 32x32 sum of absolute differences: two 16-byte chunks per row, four
// uint16 accumulators over 32 rows. Each accumulator lane tops out at
// 32*255=8160 before the final vector adds, so uint16 lanes cannot overflow.
//
//   uabdl  v2.8h, v0.8b,  v1.8b    -> 0x2e217002
//   uabdl2 v3.8h, v0.16b, v1.16b   -> 0x6e217003
//   uabdl  v6.8h, v4.8b,  v5.8b    -> 0x2e257086
//   uabdl2 v7.8h, v4.16b, v5.16b   -> 0x6e257087
//   uabal  v2.8h, v0.8b,  v1.8b    -> 0x2e215002
//   uabal2 v3.8h, v0.16b, v1.16b   -> 0x6e215003
//   uabal  v6.8h, v4.8b,  v5.8b    -> 0x2e255086
//   uabal2 v7.8h, v4.16b, v5.16b   -> 0x6e255087

// func sad32x32NEONAsm(ctx *sad8x8NEONCtx)
TEXT ·sad32x32NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD SRC(R0), R1
	MOVD REF(R0), R2
	MOVD STRIDE(R0), R3

	VLD1 (R1), [V0.B16]
	ADD  $16, R1, R6
	VLD1 (R6), [V4.B16]
	VLD1 (R2), [V1.B16]
	ADD  $16, R2, R7
	VLD1 (R7), [V5.B16]
	WORD $0x2e217002 // uabdl  v2.8h, v0.8b,  v1.8b
	WORD $0x6e217003 // uabdl2 v3.8h, v0.16b, v1.16b
	WORD $0x2e257086 // uabdl  v6.8h, v4.8b,  v5.8b
	WORD $0x6e257087 // uabdl2 v7.8h, v4.16b, v5.16b
	ADD  R3, R1
	ADD  R3, R2

	MOVD $31, R4
loop32:
	VLD1 (R1), [V0.B16]
	ADD  $16, R1, R6
	VLD1 (R6), [V4.B16]
	VLD1 (R2), [V1.B16]
	ADD  $16, R2, R7
	VLD1 (R7), [V5.B16]
	WORD $0x2e215002 // uabal  v2.8h, v0.8b,  v1.8b
	WORD $0x6e215003 // uabal2 v3.8h, v0.16b, v1.16b
	WORD $0x2e255086 // uabal  v6.8h, v4.8b,  v5.8b
	WORD $0x6e255087 // uabal2 v7.8h, v4.16b, v5.16b
	ADD  R3, R1
	ADD  R3, R2
	SUB  $1, R4
	CBNZ R4, loop32

	VADD V3.H8, V2.H8, V2.H8
	VADD V7.H8, V6.H8, V6.H8
	VADD V6.H8, V2.H8, V2.H8
	WORD $0x6e703844 // uaddlv s4, v2.8h
	VMOV V4.S[0], R5
	MOVD R5, SUM(R0)
	RET

#define DSRC       0
#define DREF       8
#define DSRCSTRIDE 16
#define DREFSTRIDE 24
#define DSUM       32

// func sad8x8DualNEONAsm(ctx *sad8x8DualNEONCtx)
TEXT ·sad8x8DualNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD DSRC(R0), R1
	MOVD DREF(R0), R2
	MOVD DSRCSTRIDE(R0), R3
	MOVD DREFSTRIDE(R0), R6

	VLD1 (R1), [V0.B8]
	VLD1 (R2), [V1.B8]
	WORD $0x2e217002 // uabdl v2.8h, v0.8b, v1.8b
	ADD  R3, R1
	ADD  R6, R2

	MOVD $7, R4
dloop:
	VLD1 (R1), [V0.B8]
	VLD1 (R2), [V1.B8]
	WORD $0x2e215002 // uabal v2.8h, v0.8b, v1.8b
	ADD  R3, R1
	ADD  R6, R2
	SUB  $1, R4
	CBNZ R4, dloop

	WORD $0x6e703843 // uaddlv s3, v2.8h
	VMOV V3.S[0], R5
	MOVD R5, DSUM(R0)
	RET

// func sad16x16DualNEONAsm(ctx *sad8x8DualNEONCtx)
TEXT ·sad16x16DualNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD DSRC(R0), R1
	MOVD DREF(R0), R2
	MOVD DSRCSTRIDE(R0), R3
	MOVD DREFSTRIDE(R0), R6

	VLD1 (R1), [V0.B16]
	VLD1 (R2), [V1.B16]
	WORD $0x2e217002 // uabdl  v2.8h, v0.8b, v1.8b
	WORD $0x6e217003 // uabdl2 v3.8h, v0.16b, v1.16b
	ADD  R3, R1
	ADD  R6, R2

	MOVD $15, R4
dloop16:
	VLD1 (R1), [V0.B16]
	VLD1 (R2), [V1.B16]
	WORD $0x2e215002 // uabal  v2.8h, v0.8b, v1.8b
	WORD $0x6e215003 // uabal2 v3.8h, v0.16b, v1.16b
	ADD  R3, R1
	ADD  R6, R2
	SUB  $1, R4
	CBNZ R4, dloop16

	VADD V3.H8, V2.H8, V2.H8
	WORD $0x6e703844 // uaddlv s4, v2.8h
	VMOV V4.S[0], R5
	MOVD R5, DSUM(R0)
	RET

// func sad32x32DualNEONAsm(ctx *sad8x8DualNEONCtx)
TEXT ·sad32x32DualNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD DSRC(R0), R1
	MOVD DREF(R0), R2
	MOVD DSRCSTRIDE(R0), R3
	MOVD DREFSTRIDE(R0), R6

	VLD1 (R1), [V0.B16]
	ADD  $16, R1, R7
	VLD1 (R7), [V4.B16]
	VLD1 (R2), [V1.B16]
	ADD  $16, R2, R8
	VLD1 (R8), [V5.B16]
	WORD $0x2e217002 // uabdl  v2.8h, v0.8b,  v1.8b
	WORD $0x6e217003 // uabdl2 v3.8h, v0.16b, v1.16b
	WORD $0x2e257086 // uabdl  v6.8h, v4.8b,  v5.8b
	WORD $0x6e257087 // uabdl2 v7.8h, v4.16b, v5.16b
	ADD  R3, R1
	ADD  R6, R2

	MOVD $31, R4
dloop32:
	VLD1 (R1), [V0.B16]
	ADD  $16, R1, R7
	VLD1 (R7), [V4.B16]
	VLD1 (R2), [V1.B16]
	ADD  $16, R2, R8
	VLD1 (R8), [V5.B16]
	WORD $0x2e215002 // uabal  v2.8h, v0.8b,  v1.8b
	WORD $0x6e215003 // uabal2 v3.8h, v0.16b, v1.16b
	WORD $0x2e255086 // uabal  v6.8h, v4.8b,  v5.8b
	WORD $0x6e255087 // uabal2 v7.8h, v4.16b, v5.16b
	ADD  R3, R1
	ADD  R6, R2
	SUB  $1, R4
	CBNZ R4, dloop32

	VADD V3.H8, V2.H8, V2.H8
	VADD V7.H8, V6.H8, V6.H8
	VADD V6.H8, V2.H8, V2.H8
	WORD $0x6e703844 // uaddlv s4, v2.8h
	VMOV V4.S[0], R5
	MOVD R5, DSUM(R0)
	RET

#define CSRC        0
#define CREF0       8
#define CREF1       16
#define CSRCSTRIDE  24
#define CREF0STRIDE 32
#define CREF1STRIDE 40
#define CSUM        48

// NEON 8x8 SAD against the rounded average of two reference blocks:
// pred = urhadd(ref0, ref1), sum(abs(src - pred)).
//
//   urhadd v3.8b, v1.8b, v2.8b  -> 0x2e221423
//   uabdl  v4.8h, v0.8b, v3.8b  -> 0x2e237004
//   uabal  v4.8h, v0.8b, v3.8b  -> 0x2e235004
//   uaddlv s5,    v4.8h         -> 0x6e703885

// func sad8x8CompoundAvgNEONAsm(ctx *sad8x8CompoundAvgNEONCtx)
TEXT ·sad8x8CompoundAvgNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD CSRC(R0), R1
	MOVD CREF0(R0), R2
	MOVD CREF1(R0), R7
	MOVD CSRCSTRIDE(R0), R3
	MOVD CREF0STRIDE(R0), R6
	MOVD CREF1STRIDE(R0), R8

	VLD1 (R1), [V0.B8]
	VLD1 (R2), [V1.B8]
	VLD1 (R7), [V2.B8]
	WORD $0x2e221423 // urhadd v3.8b, v1.8b, v2.8b
	WORD $0x2e237004 // uabdl  v4.8h, v0.8b, v3.8b
	ADD  R3, R1
	ADD  R6, R2
	ADD  R8, R7

	MOVD $7, R4
cloop:
	VLD1 (R1), [V0.B8]
	VLD1 (R2), [V1.B8]
	VLD1 (R7), [V2.B8]
	WORD $0x2e221423 // urhadd v3.8b, v1.8b, v2.8b
	WORD $0x2e235004 // uabal  v4.8h, v0.8b, v3.8b
	ADD  R3, R1
	ADD  R6, R2
	ADD  R8, R7
	SUB  $1, R4
	CBNZ R4, cloop

	WORD $0x6e703885 // uaddlv s5, v4.8h
	VMOV V5.S[0], R5
	MOVD R5, CSUM(R0)
	RET
