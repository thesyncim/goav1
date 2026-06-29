// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

#define A_SRC    0
#define A_STRIDE 8
#define A_AVG    16

// Low-bitdepth 8x8 average, source-shaped from libaom's aom_avg_8x8_neon:
// load eight rows, widen-accumulate bytes to u16 lanes, horizontally add, and
// return (sum + 32) >> 6.
//
// Instructions missing from Go's assembler are emitted as WORD with source
// mnemonics from clang -target arm64-apple-macos:
//   uaddl  v16.8h, v0.8b,  v1.8b  -> 0x2e210010
//   uaddw  v16.8h, v16.8h, v2.8b  -> 0x2e221210
//   uaddw  v16.8h, v16.8h, v3.8b  -> 0x2e231210
//   uaddw  v16.8h, v16.8h, v4.8b  -> 0x2e241210
//   uaddw  v16.8h, v16.8h, v5.8b  -> 0x2e251210
//   uaddw  v16.8h, v16.8h, v6.8b  -> 0x2e261210
//   uaddw  v16.8h, v16.8h, v7.8b  -> 0x2e271210
//   uaddlv s0,     v16.8h         -> 0x6e703a00
//
// func realtimeAvg8x8NEONAsm(ctx *realtimeAvg8x8NEONCtx)
TEXT ·realtimeAvg8x8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD A_SRC(R0), R1
	MOVD A_STRIDE(R0), R2

	VLD1 (R1), [V0.B8]
	ADD  R2, R1
	VLD1 (R1), [V1.B8]
	ADD  R2, R1
	VLD1 (R1), [V2.B8]
	ADD  R2, R1
	VLD1 (R1), [V3.B8]
	ADD  R2, R1
	VLD1 (R1), [V4.B8]
	ADD  R2, R1
	VLD1 (R1), [V5.B8]
	ADD  R2, R1
	VLD1 (R1), [V6.B8]
	ADD  R2, R1
	VLD1 (R1), [V7.B8]

	WORD $0x2e210010 // uaddl  v16.8h, v0.8b,  v1.8b
	WORD $0x2e221210 // uaddw  v16.8h, v16.8h, v2.8b
	WORD $0x2e231210 // uaddw  v16.8h, v16.8h, v3.8b
	WORD $0x2e241210 // uaddw  v16.8h, v16.8h, v4.8b
	WORD $0x2e251210 // uaddw  v16.8h, v16.8h, v5.8b
	WORD $0x2e261210 // uaddw  v16.8h, v16.8h, v6.8b
	WORD $0x2e271210 // uaddw  v16.8h, v16.8h, v7.8b
	WORD $0x6e703a00 // uaddlv s0,     v16.8h
	VMOV V0.S[0], R3
	ADD  $32, R3
	LSR  $6, R3
	MOVD R3, A_AVG(R0)
	RET
