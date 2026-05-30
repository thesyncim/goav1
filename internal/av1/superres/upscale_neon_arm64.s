// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// NEON super-res horizontal upscale (per-output-pixel 8-tap polyphase filter).
//
// The Go arm64 assembler does not expose the integer vector widening
// multiply-accumulate or the across-lane reduction as named mnemonics, so the
// SIMD instructions are emitted via WORD with hand-verified encodings
// (cross-checked against the system assembler). Pointer arithmetic, the
// fixed-point cursor, the final rounding/clamp, and control flow use ordinary
// Plan 9 GP-register instructions, mirroring the pure-Go reference exactly.
//
// Lane semantics (all bit-exact with upscaleRowPureGo central case):
//   ld1   {v1.8h}, [x9]            load 8 source uint16 samples src[base..+7]
//   ld1   {v2.8h}, [x12]           load 8 int16 phase coefficients
//   smull  v3.4s, v1.4h, v2.4h     widen-multiply low  4 lanes -> int32
//   smlal2 v3.4s, v1.8h, v2.8h     widen-MAC   high 4 lanes -> int32
//   addv   s4, v3.4s               horizontal add of the 4 int32 lanes
//   mov    w11, v4.s[0]            move scalar sum to a GP register
// then sum = (sum + 64) >> 7, clamped to [0, maxValue], stored as uint16.

// ctx field offsets (see upscaleRowNEONCtx in upscale_neon_arm64.go).
#define SRC      0
#define DST      8
#define FILTER   16
#define COUNT    24
#define SRCX0    32
#define STEPX    40
#define MAXVAL   48

// func upscaleRowNEONAsm(ctx *upscaleRowNEONCtx)
TEXT ·upscaleRowNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD SRC(R0), R1       // src row base (uint16*)
	MOVD DST(R0), R2       // dst cursor (uint16*), first central pixel
	MOVD FILTER(R0), R3    // filter table base (int16*)
	MOVD COUNT(R0), R4     // remaining central pixels
	MOVD SRCX0(R0), R5     // srcX cursor
	MOVD STEPX(R0), R6     // stepX
	MOVD MAXVAL(R0), R7    // maxValue

	CBZ  R4, done

	// Two-wide main loop: run two independent per-pixel pipelines so the
	// across-lane addv reduction of one pixel overlaps the widening MACs of
	// the other. R5 holds srcX for pixel 0; R15 holds srcX for pixel 1.
	ADD  R6, R5, R15       // srcX1 = srcX0 + stepX
	LSL  $1, R6, R16       // step2 = 2*stepX (per two-pixel iteration)

	LSR  $1, R4, R17       // pairs = count / 2
	CBZ  R17, tail

pairLoop:
	// --- pixel 0 ---
	ASR  $14, R5, R8
	SUB  $3, R8, R8
	ADD  R8<<1, R1, R9     // srcAddr0
	AND  $0x3fff, R5, R10
	LSR  $8, R10, R10
	ADD  R10<<4, R3, R12   // filtAddr0
	// --- pixel 1 ---
	ASR  $14, R15, R8
	SUB  $3, R8, R8
	ADD  R8<<1, R1, R13    // srcAddr1
	AND  $0x3fff, R15, R10
	LSR  $8, R10, R10
	ADD  R10<<4, R3, R14   // filtAddr1

	WORD $0x4c407521      // ld1 {v1.8h}, [x9]
	WORD $0x4c407582      // ld1 {v2.8h}, [x12]
	WORD $0x4c4075a5      // ld1 {v5.8h}, [x13]
	WORD $0x4c4075c6      // ld1 {v6.8h}, [x14]
	WORD $0x0e62c023      // smull  v3.4s, v1.4h, v2.4h
	WORD $0x0e66c0a7      // smull  v7.4s, v5.4h, v6.4h
	WORD $0x4e628023      // smlal2 v3.4s, v1.8h, v2.8h
	WORD $0x4e6680a7      // smlal2 v7.4s, v5.8h, v6.8h
	WORD $0x4eb1b864      // addv   s4, v3.4s
	WORD $0x4eb1b8e8      // addv   s8, v7.4s
	WORD $0x0e043c8b      // mov    w11, v4.s[0]
	WORD $0x0e043d0e      // mov    w14, v8.s[0]
	SXTW R11, R11
	SXTW R14, R14

	// round + clamp pixel 0
	ADD  $64, R11, R11
	ASR  $7, R11, R11
	CMP  $0, R11
	CSEL LT, ZR, R11, R11
	CMP  R7, R11
	CSEL GT, R7, R11, R11
	MOVH R11, (R2)
	// round + clamp pixel 1
	ADD  $64, R14, R14
	ASR  $7, R14, R14
	CMP  $0, R14
	CSEL LT, ZR, R14, R14
	CMP  R7, R14
	CSEL GT, R7, R14, R14
	MOVH R14, 2(R2)

	ADD  $4, R2, R2        // dst += 2 pixels
	ADD  R16, R5, R5       // srcX0 += 2*stepX
	ADD  R16, R15, R15     // srcX1 += 2*stepX
	SUB  $1, R17, R17
	CBNZ R17, pairLoop

tail:
	// Odd trailing pixel (count is odd). R5 already points at it.
	AND  $1, R4, R4
	CBZ  R4, done

	ASR  $14, R5, R8
	SUB  $3, R8, R8
	ADD  R8<<1, R1, R9
	AND  $0x3fff, R5, R10
	LSR  $8, R10, R10
	ADD  R10<<4, R3, R12

	WORD $0x4c407521      // ld1 {v1.8h}, [x9]
	WORD $0x4c407582      // ld1 {v2.8h}, [x12]
	WORD $0x0e62c023      // smull  v3.4s, v1.4h, v2.4h
	WORD $0x4e628023      // smlal2 v3.4s, v1.8h, v2.8h
	WORD $0x4eb1b864      // addv   s4, v3.4s
	WORD $0x0e043c8b      // mov    w11, v4.s[0]
	SXTW R11, R11

	ADD  $64, R11, R11
	ASR  $7, R11, R11
	CMP  $0, R11
	CSEL LT, ZR, R11, R11
	CMP  R7, R11
	CSEL GT, R7, R11, R11
	MOVH R11, (R2)

done:
	RET
