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

// entropy.CDF row layout (compile-time pinned in reader_coeff_base_arm64.go):
// symbols u8 at +0, values [17]u16 at +2. For the 2-symbol DC-sign row c0
// lives at +2 and the adaptation count (values[2]) at +6.
#define CDF_C0    2
#define CDF_CNT2  6

// func coeffSignGolombARM64(c *Cursor, dirty *int16, n int, coeffs *int16,
//	dcSign *CDF, update bool) (culLevel, dcValue, maxScanLine int)
//
// M-D3 D3-c kernel (spec: internal/av1/tile/coeff_asm_arm64_spec.go §6).
// Scalar source-shaped port of the P3 dirty-list sign/golomb replay shared by
// tile.readCoefficientsTXBWithGeo and readCoefficientsTXBTracked2DWithGeo:
// walk the packed (level<<10|pos) list from index n-1 down to 0; the DC entry
// (pos==0, only ever the last-appended) reads its sign from the DC-sign
// binary CDF (readBinaryCDFKnown shape, updateCDF2 adaptation when update is
// set), every other entry reads one equiprobable bit (ReadBitTrustedInline
// shape), and a level saturated at MaxBaseBRRange (15) appends the Exp-Golomb
// tail (ReadUnsignedGolombTrusted, maxLength 20). Signed coefficients are
// stored to coeffs[pos] and the list entries unpacked back to bare positions.
// culLevel returns -1 for the Go loop's validation errors (golomb prefix
// beyond maxLength; level > 32767); the failing entry's stores are skipped
// and the cursor holds the exact post-failure reader state, matching the Go
// error path byte-for-byte. The refill blocks are verbatim copies of the
// established reader_bit_arm64.s / reader_coeff_base_arm64.s refill; src
// ptr/len and the cursor pointer are reloaded from the stack args inside them
// so the loop keeps three more persistent registers.
//
// Register plan (D3-b conventions): persistent R1 dirty, R2 coeffs,
// R4 culLevel, R5 cnt, R6 pos, R7 dif, R8 rng, R10 tellOffs, R19 i,
// R20 maxScanLine, R21 dcValue, R22 update flag, R23 level, R24 negative,
// R25 entry pos, R26 isDC flag; golomb R14 length / R15 accumulator survive
// the bit reads. Iteration scratch R0, R9, R11-R13, R16, R17 (R14 doubles as
// the DC CDF row pointer and survives refill; R16 holds the decoded
// bit/symbol and survives refill + adapt); refill clobbers only
// {R0, R9, R11, R12, R13, R17}. V8-V15, R18, R27-R30 untouched.
TEXT ·coeffSignGolombARM64(SB), NOSPLIT, $0-72
	MOVD  c+0(FP), R0
	MOVD  CURSOR_DIF(R0), R7
	MOVHU CURSOR_RNG(R0), R8
	MOVH  CURSOR_CNT(R0), R5
	MOVWU CURSOR_POS(R0), R6
	MOVH  CURSOR_TELL_OFFS(R0), R10
	MOVD  dirty+8(FP), R1
	MOVD  n+16(FP), R19
	MOVD  coeffs+24(FP), R2
	MOVBU update+40(FP), R22
	MOVD  $0, R4  // culLevel
	MOVD  $0, R20 // maxScanLine
	MOVD  $0, R21 // dcValue
	SUB   $1, R19, R19
	TBNZ  $63, R19, done // n <= 0: nothing to replay

	// First entry (i = n-1): the only one that can be the DC position.
	ADD   R19<<1, R1, R13
	MOVHU (R13), R13
	AND   $1023, R13, R25 // pos = packed & coeffDirtyPosMask
	LSR   $10, R13, R23   // level = packed >> 10
	MOVD  $0, R26
	CBNZ  R25, signRead

	// DC sign: 2-symbol CDF read on the DC-sign row (readBinaryCDFKnown):
	// symbol 1 (negative) when coded < lower.
	MOVD  $1, R26
	MOVD  dcSign+32(FP), R14
	LSR   $48, R7, R12     // coded = dif >> (ecWindow-16)
	LSR   $8, R8, R9       // rngHi = rng >> 8
	MOVHU CDF_C0(R14), R11 // c0
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $4, R17, R17 // lower (+ecMinProb)
	MOVD  $0, R16
	CMP   R17, R12
	BCC   dcSym1
	LSL   $48, R17, R11
	SUB   R11, R7, R7 // dif -= lower << (ecWindow-16)
	SUB   R17, R8, R8 // rng = upper - lower
	B     dcRenorm

dcSym1:
	MOVD $1, R16
	MOVD R17, R8 // upper = lower, lower = 0: rng = lower

dcRenorm:
	ADD  $1, R7, R7
	// R8 stays zero-extended in 1..65535, so direct 64-bit CLZ yields the
	// AV1 16-bit normalization shift after subtracting 48.
	CLZ  R8, R12
	SUB  $48, R12, R12
	LSL  R12, R7, R7
	SUB  $1, R7, R7
	LSL  R12, R8, R8
	SUB  R12, R5, R5
	TBZ  $63, R5, dcAdapt

	// Refill (64-bit-window port shared with reader_bit_arm64.s); preserves
	// R14 row and R16 symbol.
	MOVD $40, R11
	SUB  R5, R11, R11 // shift = ecWindow-9-(cnt+15) = 40 - cnt
	MOVD c+0(FP), R0
	MOVD CURSOR_SRC_PTR(R0), R12
	MOVD CURSOR_SRC_LEN(R0), R13
	SUB  R6, R13, R9  // remaining = len(src) - pos
	CMP  $55, R11
	BHI  dcRefillSlow // shift > ecWindow-9 (or negative): byte loop
	CMP  $8, R9
	BLT  dcRefillSlow
	ADD  R12, R6, R0
	MOVD (R0), R17
	REV  R17, R17 // u = big-endian 8-byte load
	MOVD $56, R0
	SUB  R11, R0, R0
	LSR  R0, R17, R17 // u >> (56 - shift)
	AND  $7, R11, R0
	MOVD $1, R9
	LSL  R0, R9, R9
	SUB  $1, R9, R9
	BIC  R9, R17, R17 // mask the partial-byte leak below shift&7
	EOR  R17, R7, R7
	LSR  $3, R11, R0
	ADD  $1, R0, R0 // n = shift>>3 + 1
	ADD  R0, R6, R6 // pos += n
	LSL  $3, R0, R0
	ADD  R0, R5, R5 // cnt += 8*n
	B    dcAdapt    // n <= 7 < remaining: end of buffer unreachable

dcRefillSlow:
	CMP $0, R11
	BLT dcRefillTail
	CMP R13, R6
	BGE dcRefillTail
	ADD   R12, R6, R0
	MOVBU (R0), R17
	LSL   R11, R17, R17
	EOR   R17, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     dcRefillSlow

dcRefillTail:
	CMP  R13, R6
	BLT  dcAdapt
	MOVD $0x4000, R9
	SUB  R5, R9, R0
	ADD  R0, R10, R10
	MOVD R9, R5

dcAdapt:
	// updateCDF2: rate = 4 + count>>4 (2 symbols: no rate bias); count is
	// values[2], saturating at 32.
	CBZ   R22, dcAdaptDone
	MOVHU CDF_CNT2(R14), R13
	LSR   $4, R13, R11
	ADD   $4, R11, R11 // rate
	MOVHU CDF_C0(R14), R12
	CBZ   R16, dcUpd0
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12 // symbol 1: c0 += (32768-c0)>>rate
	B     dcUpdStore

dcUpd0:
	LSR R11, R12, R9
	SUB R9, R12, R12 // symbol 0: c0 -= c0>>rate

dcUpdStore:
	MOVH R12, CDF_C0(R14)
	CMP  $32, R13
	BGE  dcAdaptDone
	ADD  $1, R13, R13
	MOVH R13, CDF_CNT2(R14)

dcAdaptDone:
	MOVD R16, R24 // negative = symbol
	B    tail

	// Equiprobable sign bit (ReadBitTrustedInline shape): bit = 1 when
	// dif < window; negative when bit != 0.
signRead:
	LSR  $8, R8, R11
	LSL  $7, R11, R11
	ADD  $4, R11, R11  // split
	LSL  $48, R11, R12 // window = split << (ecWindow-16)
	MOVD $1, R16
	CMP  R12, R7
	BCC  sbSplit
	SUB  R12, R7, R7 // dif -= window
	SUB  R11, R8, R8 // rng -= split
	MOVD $0, R16
	B    sbRenorm

sbSplit:
	MOVD R11, R8 // rng = split

sbRenorm:
	ADD  $1, R7, R7
	CLZ  R8, R12
	SUB  $48, R12, R12
	LSL  R12, R7, R7
	SUB  $1, R7, R7
	LSL  R12, R8, R8
	SUB  R12, R5, R5
	TBZ  $63, R5, sbDone

	MOVD $40, R11
	SUB  R5, R11, R11
	MOVD c+0(FP), R0
	MOVD CURSOR_SRC_PTR(R0), R12
	MOVD CURSOR_SRC_LEN(R0), R13
	SUB  R6, R13, R9
	CMP  $55, R11
	BHI  sbRefillSlow
	CMP  $8, R9
	BLT  sbRefillSlow
	ADD  R12, R6, R0
	MOVD (R0), R17
	REV  R17, R17
	MOVD $56, R0
	SUB  R11, R0, R0
	LSR  R0, R17, R17
	AND  $7, R11, R0
	MOVD $1, R9
	LSL  R0, R9, R9
	SUB  $1, R9, R9
	BIC  R9, R17, R17
	EOR  R17, R7, R7
	LSR  $3, R11, R0
	ADD  $1, R0, R0
	ADD  R0, R6, R6
	LSL  $3, R0, R0
	ADD  R0, R5, R5
	B    sbDone

sbRefillSlow:
	CMP $0, R11
	BLT sbRefillTail
	CMP R13, R6
	BGE sbRefillTail
	ADD   R12, R6, R0
	MOVBU (R0), R17
	LSL   R11, R17, R17
	EOR   R17, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     sbRefillSlow

sbRefillTail:
	CMP  R13, R6
	BLT  sbDone
	MOVD $0x4000, R9
	SUB  R5, R9, R0
	ADD  R0, R10, R10
	MOVD R9, R5

sbDone:
	MOVD R16, R24 // negative = bit

tail:
	// maxScanLine covers every replayed position; the DC pos 0 never raises
	// it, exactly like the split Go blocks.
	CMP  R20, R25
	CSEL HI, R25, R20, R20
	CMP  $15, R23 // MaxBaseBRRange
	BLT  tailStore

	// Exp-Golomb tail (ReadUnsignedGolombTrusted, maxLength 20): unary
	// prefix of length bits (terminating 1 included), then length-1 suffix
	// bits accumulated MSB-first. R15 starts at 1 so it finishes as
	// (1<<(length-1))|suffix; value = R15 - 1.
	MOVD $0, R14 // length

gpLoop:
	LSR  $8, R8, R11
	LSL  $7, R11, R11
	ADD  $4, R11, R11
	LSL  $48, R11, R12
	MOVD $1, R16
	CMP  R12, R7
	BCC  gpSplit
	SUB  R12, R7, R7
	SUB  R11, R8, R8
	MOVD $0, R16
	B    gpRenorm

gpSplit:
	MOVD R11, R8

gpRenorm:
	ADD  $1, R7, R7
	CLZ  R8, R12
	SUB  $48, R12, R12
	LSL  R12, R7, R7
	SUB  $1, R7, R7
	LSL  R12, R8, R8
	SUB  R12, R5, R5
	TBZ  $63, R5, gpDone

	MOVD $40, R11
	SUB  R5, R11, R11
	MOVD c+0(FP), R0
	MOVD CURSOR_SRC_PTR(R0), R12
	MOVD CURSOR_SRC_LEN(R0), R13
	SUB  R6, R13, R9
	CMP  $55, R11
	BHI  gpRefillSlow
	CMP  $8, R9
	BLT  gpRefillSlow
	ADD  R12, R6, R0
	MOVD (R0), R17
	REV  R17, R17
	MOVD $56, R0
	SUB  R11, R0, R0
	LSR  R0, R17, R17
	AND  $7, R11, R0
	MOVD $1, R9
	LSL  R0, R9, R9
	SUB  $1, R9, R9
	BIC  R9, R17, R17
	EOR  R17, R7, R7
	LSR  $3, R11, R0
	ADD  $1, R0, R0
	ADD  R0, R6, R6
	LSL  $3, R0, R0
	ADD  R0, R5, R5
	B    gpDone

gpRefillSlow:
	CMP $0, R11
	BLT gpRefillTail
	CMP R13, R6
	BGE gpRefillTail
	ADD   R12, R6, R0
	MOVBU (R0), R17
	LSL   R11, R17, R17
	EOR   R17, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     gpRefillSlow

gpRefillTail:
	CMP  R13, R6
	BLT  gpDone
	MOVD $0x4000, R9
	SUB  R5, R9, R0
	ADD  R0, R10, R10
	MOVD R9, R5

gpDone:
	ADD $1, R14, R14
	CMP $20, R14
	BGT fail          // prefix beyond maxLength: validation error
	CBZ R16, gpLoop   // bit 0: prefix continues

	MOVD $1, R15
	SUB  $1, R14, R14 // remaining suffix bits = length-1

gsLoop:
	CBZ  R14, gsEnd
	LSR  $8, R8, R11
	LSL  $7, R11, R11
	ADD  $4, R11, R11
	LSL  $48, R11, R12
	MOVD $1, R16
	CMP  R12, R7
	BCC  gsSplit
	SUB  R12, R7, R7
	SUB  R11, R8, R8
	MOVD $0, R16
	B    gsRenorm

gsSplit:
	MOVD R11, R8

gsRenorm:
	ADD  $1, R7, R7
	CLZ  R8, R12
	SUB  $48, R12, R12
	LSL  R12, R7, R7
	SUB  $1, R7, R7
	LSL  R12, R8, R8
	SUB  R12, R5, R5
	TBZ  $63, R5, gsDone

	MOVD $40, R11
	SUB  R5, R11, R11
	MOVD c+0(FP), R0
	MOVD CURSOR_SRC_PTR(R0), R12
	MOVD CURSOR_SRC_LEN(R0), R13
	SUB  R6, R13, R9
	CMP  $55, R11
	BHI  gsRefillSlow
	CMP  $8, R9
	BLT  gsRefillSlow
	ADD  R12, R6, R0
	MOVD (R0), R17
	REV  R17, R17
	MOVD $56, R0
	SUB  R11, R0, R0
	LSR  R0, R17, R17
	AND  $7, R11, R0
	MOVD $1, R9
	LSL  R0, R9, R9
	SUB  $1, R9, R9
	BIC  R9, R17, R17
	EOR  R17, R7, R7
	LSR  $3, R11, R0
	ADD  $1, R0, R0
	ADD  R0, R6, R6
	LSL  $3, R0, R0
	ADD  R0, R5, R5
	B    gsDone

gsRefillSlow:
	CMP $0, R11
	BLT gsRefillTail
	CMP R13, R6
	BGE gsRefillTail
	ADD   R12, R6, R0
	MOVBU (R0), R17
	LSL   R11, R17, R17
	EOR   R17, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     gsRefillSlow

gsRefillTail:
	CMP  R13, R6
	BLT  gsDone
	MOVD $0x4000, R9
	SUB  R5, R9, R0
	ADD  R0, R10, R10
	MOVD R9, R5

gsDone:
	ADD R15<<1, R16, R15 // acc = acc<<1 | bit
	SUB $1, R14, R14
	B   gsLoop

gsEnd:
	SUB $1, R15, R15   // value = ((1<<(length-1))|suffix) - 1
	ADD R15, R23, R23  // level += value

tailStore:
	ADD  R23, R4, R4 // culLevel += level
	MOVD $32767, R13
	CMP  R13, R23
	BGT  fail            // level > math.MaxInt16: validation error
	NEG  R23, R13
	CMP  $0, R24
	CSEL EQ, R23, R13, R12 // signed = negative ? -level : level
	ADD  R25<<1, R2, R13
	MOVH R12, (R13)        // coeffs[pos] = signed
	CBZ  R26, tailNoDC
	MOVD R12, R21          // dcValue = signed

tailNoDC:
	ADD  R19<<1, R1, R13
	MOVH R25, (R13) // dirty[i] = pos (unpacked)
	SUB  $1, R19, R19
	TBNZ $63, R19, done
	ADD  R19<<1, R1, R13
	MOVHU (R13), R13
	AND  $1023, R13, R25
	LSR  $10, R13, R23
	MOVD $0, R26 // pos 0 only ever appends last: later entries are never DC
	B    signRead

fail:
	MOVD $-1, R4

done:
	MOVD c+0(FP), R0
	MOVD R7, CURSOR_DIF(R0)
	MOVW R6, CURSOR_POS(R0)
	MOVH R8, CURSOR_RNG(R0)
	MOVH R5, CURSOR_CNT(R0)
	MOVH R10, CURSOR_TELL_OFFS(R0)
	MOVD R4, culLevel+48(FP)
	MOVD R21, dcValue+56(FP)
	MOVD R20, maxScanLine+64(FP)
	RET
