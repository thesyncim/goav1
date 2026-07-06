// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego && !goav1_trace_rng

#include "textflag.h"

#define CURSOR_SRC_PTR       0
#define CURSOR_SRC_LEN       8
#define CURSOR_DIF          24
#define CURSOR_POS          32
#define CURSOR_RNG          36
#define CURSOR_CNT          38
#define CURSOR_TELL_OFFS    40

// The refill blocks below are the 64-bit-window port of refillState: the fast
// path mirrors dav1d's single 8-byte refill load (src/arm/64/msac.S L(refill))
// while keeping goav1's XOR-into-ones window convention, so the leaked partial
// byte below bit shift&7 must be masked off before the XOR (dav1d's OR of
// inverted bytes is idempotent and needs no mask; XOR is not). With at least 8
// bytes remaining the fast path consumes at most 7, so end-of-buffer tell
// accounting stays on the byte-loop path, exactly as in the Go refillState.

// func readBitTrustedARM64(c *Cursor) uint8
//
// Source-shaped arm64 range-decoder bit path, mapped to dav1d's
// msac_decode_bool_equi_neon scalar state update while preserving goav1's
// exact refillState semantics for tile tails.
TEXT ·readBitTrustedARM64(SB), NOSPLIT, $0-9
	MOVD c+0(FP), R0
	MOVD CURSOR_DIF(R0), R7
	MOVHU CURSOR_RNG(R0), R8

	LSR   $8, R8, R9
	LSL   $7, R9, R10
	ADD   $4, R10, R10
	LSL   $48, R10, R11
	MOVD  $1, R15
	CMP   R11, R7
	BCC   bitNoSub
	SUB   R11, R7, R7
	SUB   R10, R8, R8
	MOVD  $0, R15
	B     bitRenorm

bitNoSub:
	MOVD R10, R8

bitRenorm:
	MOVH CURSOR_CNT(R0), R5
	ADD  $1, R7, R7
	UBFX $0, R8, $16, R12
	CLZ  R12, R12
	SUB  $48, R12, R12
	LSL  R12, R7, R7
	SUB  $1, R7, R7
	LSL  R12, R8, R8
	SUB  R12, R5, R5
	TBZ  $63, R5, bitStore

	MOVD  CURSOR_SRC_PTR(R0), R2
	MOVD  CURSOR_SRC_LEN(R0), R3
	MOVWU CURSOR_POS(R0), R6
	MOVH  CURSOR_TELL_OFFS(R0), R10
	MOVD  $40, R11
	SUB   R5, R11, R11 // shift = ecWindow-9-(cnt+15) = 40 - cnt
	SUB   R6, R3, R12  // remaining = len(src) - pos

	CMP $55, R11
	BHI refillSlow // shift > ecWindow-9 (or negative): byte loop
	CMP $8, R12
	BLT refillSlow
	ADD  R2, R6, R13
	MOVD (R13), R14
	REV  R14, R14 // u = big-endian 8-byte load
	MOVD $56, R4
	SUB  R11, R4, R4
	LSR  R4, R14, R14 // u >> (56 - shift)
	AND  $7, R11, R4
	MOVD $1, R16
	LSL  R4, R16, R16
	SUB  $1, R16, R16
	BIC  R16, R14, R14 // mask the partial-byte leak below shift&7
	EOR  R14, R7, R7
	LSR  $3, R11, R4
	ADD  $1, R4, R4 // n = shift>>3 + 1
	ADD  R4, R6, R6 // pos += n
	LSL  $3, R4, R4
	ADD  R4, R5, R5    // cnt += 8*n
	B    refillStore   // n <= 7 < remaining: end of buffer unreachable

refillSlow:
	CMP $0, R11
	BLT refillTailCheck
	CMP R3, R6
	BGE refillTailCheck
	ADD   R2, R6, R13
	MOVBU (R13), R14
	LSL   R11, R14, R14
	EOR   R14, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     refillSlow

refillTailCheck:
	CMP R3, R6
	BLT refillStore
	MOVD $0x4000, R12
	SUB  R5, R12, R13
	ADD  R13, R10, R10
	MOVD R12, R5

refillStore:
	MOVW R6, CURSOR_POS(R0)
	MOVH R10, CURSOR_TELL_OFFS(R0)

bitStore:
	MOVD R7, CURSOR_DIF(R0)
	MOVH R8, CURSOR_RNG(R0)
	MOVH R5, CURSOR_CNT(R0)
	MOVB R15, ret+8(FP)
	RET

// func readCDF4HighTokenUpdateArch(c *Cursor, values *[MaxSymbols + 1]uint16) uint8
//
// Complete ARM64 high-token chain for coefficient BR syntax. This is the scalar
// source-shaped unit corresponding to dav1d's msac_decode_hi_tok_neon: decode up
// to four chained 4-symbol reads, adapt the same CDF row after each read, and
// refill inside the same leaf.
TEXT ·readCDF4HighTokenUpdateArch(SB), NOSPLIT, $0-17
	MOVD  c+0(FP), R0
	MOVD  values+8(FP), R1
	MOVD  CURSOR_SRC_PTR(R0), R2
	MOVD  CURSOR_SRC_LEN(R0), R3
	MOVWU CURSOR_POS(R0), R6
	MOVD  CURSOR_DIF(R0), R7
	MOVHU CURSOR_RNG(R0), R8
	MOVH  CURSOR_CNT(R0), R5
	MOVH  CURSOR_TELL_OFFS(R0), R10
	MOVHU 0(R1), R20 // cdf[0]
	MOVHU 2(R1), R21 // cdf[1]
	MOVHU 4(R1), R22 // cdf[2]
	MOVHU 8(R1), R23 // cdf[4] adaptation count
	MOVD  $0, R24    // accumulated level
	MOVD  $0, R27    // decoded symbols in this chain

hiTokLoop:
	MOVD R8, R16      // upper = rng
	LSR  $48, R7, R12 // coded = dif >> (ecWindow - 16)
	LSR  $8, R8, R9   // rngHi = rng >> 8

	LSR  $6, R20, R17
	MUL  R9, R17, R17
	LSR  $1, R17, R17
	ADD  $12, R17, R17 // lower0
	MOVD $0, R15        // symbol
	CMPW R17, R12
	BCC  hiTokSym1
	B    hiTokDecoded

hiTokSym1:
	MOVD $1, R15
	MOVD R17, R16
	LSR  $6, R21, R17
	MUL  R9, R17, R17
	LSR  $1, R17, R17
	ADD  $8, R17, R17 // lower1
	CMPW R17, R12
	BCC  hiTokSym2
	B    hiTokDecoded

hiTokSym2:
	MOVD $2, R15
	MOVD R17, R16
	LSR  $6, R22, R17
	MUL  R9, R17, R17
	LSR  $1, R17, R17
	ADD  $4, R17, R17 // lower2
	CMPW R17, R12
	BCC  hiTokSym3
	B    hiTokDecoded

hiTokSym3:
	MOVD $3, R15
	MOVD R17, R16
	MOVD $0, R17

hiTokDecoded:
	LSL  $48, R17, R11
	SUB  R11, R7, R7 // dif -= lower << (ecWindow - 16)
	SUB  R17, R16, R8
	ADD  $1, R7, R7
	UBFX $0, R8, $16, R12
	CLZ  R12, R12
	SUB  $48, R12, R12
	LSL  R12, R7, R7
	SUB  $1, R7, R7
	LSL  R12, R8, R8
	SUB  R12, R5, R5
	TBZ  $63, R5, hiTokAfterRefill

	MOVD $40, R11
	SUB  R5, R11, R11 // shift = ecWindow-9-(cnt+15) = 40 - cnt
	SUB  R6, R3, R12  // remaining = len(src) - pos

	CMP $55, R11
	BHI hiTokRefillSlow // shift > ecWindow-9 (or negative): byte loop
	CMP $8, R12
	BLT hiTokRefillSlow
	ADD  R2, R6, R13
	MOVD (R13), R14
	REV  R14, R14 // u = big-endian 8-byte load
	MOVD $56, R4
	SUB  R11, R4, R4
	LSR  R4, R14, R14 // u >> (56 - shift)
	AND  $7, R11, R4
	MOVD $1, R17
	LSL  R4, R17, R17
	SUB  $1, R17, R17
	BIC  R17, R14, R14 // mask the partial-byte leak below shift&7
	EOR  R14, R7, R7
	LSR  $3, R11, R4
	ADD  $1, R4, R4          // n = shift>>3 + 1
	ADD  R4, R6, R6          // pos += n
	LSL  $3, R4, R4
	ADD  R4, R5, R5          // cnt += 8*n
	B    hiTokAfterRefill    // n <= 7 < remaining: end of buffer unreachable

hiTokRefillSlow:
	CMP $0, R11
	BLT hiTokRefillTailCheck
	CMP R3, R6
	BGE hiTokRefillTailCheck
	ADD   R2, R6, R13
	MOVBU (R13), R14
	LSL   R11, R14, R14
	EOR   R14, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     hiTokRefillSlow

hiTokRefillTailCheck:
	CMP R3, R6
	BLT hiTokAfterRefill
	MOVD $0x4000, R12
	SUB  R5, R12, R13
	ADD  R13, R10, R10
	MOVD R12, R5

hiTokAfterRefill:
	LSR $4, R23, R19
	ADD $5, R19, R19 // rate = 5 + count >> 4
	CMP $0, R15
	BEQ hiTokUpdate0
	CMP $1, R15
	BEQ hiTokUpdate1
	CMP $2, R15
	BEQ hiTokUpdate2
	B   hiTokUpdate3

hiTokUpdate0:
	LSR R19, R20, R25
	SUB R25, R20, R20
	LSR R19, R21, R25
	SUB R25, R21, R21
	LSR R19, R22, R25
	SUB R25, R22, R22
	B   hiTokCount

hiTokUpdate1:
	MOVD $32768, R25
	SUB  R20, R25, R25
	LSR  R19, R25, R25
	ADD  R25, R20, R20
	LSR  R19, R21, R25
	SUB  R25, R21, R21
	LSR  R19, R22, R25
	SUB  R25, R22, R22
	B    hiTokCount

hiTokUpdate2:
	MOVD $32768, R25
	SUB  R20, R25, R25
	LSR  R19, R25, R25
	ADD  R25, R20, R20
	MOVD $32768, R25
	SUB  R21, R25, R25
	LSR  R19, R25, R25
	ADD  R25, R21, R21
	LSR  R19, R22, R25
	SUB  R25, R22, R22
	B    hiTokCount

hiTokUpdate3:
	MOVD $32768, R25
	SUB  R20, R25, R25
	LSR  R19, R25, R25
	ADD  R25, R20, R20
	MOVD $32768, R25
	SUB  R21, R25, R25
	LSR  R19, R25, R25
	ADD  R25, R21, R21
	MOVD $32768, R25
	SUB  R22, R25, R25
	LSR  R19, R25, R25
	ADD  R25, R22, R22

hiTokCount:
	CMP $32, R23
	BGE hiTokCountDone
	ADD $1, R23, R23

hiTokCountDone:
	ADD $1, R27, R27
	ADD R15, R24, R24
	CMP $3, R15
	BNE hiTokDone
	CMP $4, R27
	BLT hiTokLoop

hiTokDone:
	MOVW R6, CURSOR_POS(R0)
	MOVD R7, CURSOR_DIF(R0)
	MOVH R8, CURSOR_RNG(R0)
	MOVH R5, CURSOR_CNT(R0)
	MOVH R10, CURSOR_TELL_OFFS(R0)
	MOVH R20, 0(R1)
	MOVH R21, 2(R1)
	MOVH R22, 4(R1)
	MOVH R23, 8(R1)
	MOVB R24, ret+16(FP)
	RET
