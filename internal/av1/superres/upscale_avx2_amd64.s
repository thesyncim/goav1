// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 super-res horizontal upscale (per-output-pixel 8-tap polyphase filter).
//
// Pointer arithmetic, the fixed-point cursor, the final rounding/clamp, and
// control flow use ordinary GP-register instructions, mirroring the pure-Go
// reference exactly. The 8-tap dot product is computed with VPMADDWD on packed
// 16-bit lanes:
//
//   VMOVDQU (srcAddr), X1     load 8 source uint16 samples src[base..+7]
//   VMOVDQU (filtAddr), X2    load 8 int16 phase coefficients
//   VPMADDWD X2, X1, X0       signed16xsigned16 -> int32, add adjacent pairs
//                             -> 4 partial int32 sums
//   VPHADDD X0, X0, X0        reduce 4 -> 2
//   VPHADDD X0, X0, X0        reduce 2 -> 1 (scalar sum in low int32 lane)
//   MOVD X0, R                move scalar sum to a GP register
// then sum = (sum + 64) >> 7, clamped to [0, maxValue], stored as uint16.
//
// Source samples are non-negative (<= 4095), so the signed-16 interpretation
// used by VPMADDWD reproduces the reference int products exactly. The sum fits
// well within int32 for all <=12-bit inputs.

// ctx field offsets (see upscaleRowAVX2Ctx in upscale_avx2_amd64.go).
#define SRC      0
#define DST      8
#define FILTER   16
#define COUNT    24
#define SRCX0    32
#define STEPX    40
#define MAXVAL   48

// func upscaleRowAVX2Asm(ctx *upscaleRowAVX2Ctx)
TEXT ·upscaleRowAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ SRC(AX), SI       // src row base (uint16*)
	MOVQ DST(AX), DI       // dst cursor (uint16*), first central pixel
	MOVQ FILTER(AX), BX    // filter table base (int16*)
	MOVQ COUNT(AX), CX     // remaining central pixels
	MOVQ SRCX0(AX), R8     // srcX cursor
	MOVQ STEPX(AX), R9     // stepX
	MOVQ MAXVAL(AX), R10   // maxValue

	TESTQ CX, CX
	JZ    done

loop:
	// base = (srcX >> 14) - 3 ; srcAddr = src + base*2
	MOVQ  R8, R11
	SARQ  $14, R11         // srcXPx = srcX >> 14 (arithmetic)
	SUBQ  $3, R11          // base = srcXPx - FilterOffset
	LEAQ  (SI)(R11*2), R12 // srcAddr = src + base*2

	// subpel = (srcX & 0x3fff) >> 8 ; filtAddr = filter + subpel*16
	MOVQ  R8, R13
	ANDQ  $0x3fff, R13     // srcX & ScaleMask
	SHRQ  $8, R13          // >> ExtraBits (=8); subpel index
	SHLQ  $4, R13          // subpel * 16 bytes (8 int16 per phase)
	LEAQ  (BX)(R13*1), R14 // filtAddr

	VMOVDQU (R12), X1      // 8 source uint16
	VMOVDQU (R14), X2      // 8 int16 coefficients
	VPMADDWD X2, X1, X0    // 4 int32 partial sums
	VPHADDD  X0, X0, X0    // reduce to 2
	VPHADDD  X0, X0, X0    // reduce to 1 (low lane = scalar sum)
	MOVD     X0, R15       // 32-bit sum
	MOVLQSX  R15, R15      // sign-extend to 64-bit

	// round + clamp: (sum + 64) >> 7, clamp [0, maxValue]
	ADDQ  $64, R15
	SARQ  $7, R15
	// clamp low: if R15 < 0 -> 0
	XORQ  AX, AX
	CMPQ  R15, AX
	CMOVQLT AX, R15
	// clamp high: if R15 > maxValue -> maxValue
	CMPQ  R15, R10
	CMOVQGT R10, R15
	MOVW  R15, (DI)        // store uint16

	ADDQ  $2, DI           // dst += 1 pixel
	ADDQ  R9, R8           // srcX += stepX
	DECQ  CX
	JNZ   loop

done:
	VZEROUPPER
	RET
