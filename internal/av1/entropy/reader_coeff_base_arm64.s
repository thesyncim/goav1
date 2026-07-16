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
// symbols u8 at +0, values [17]u16 at +2. For the 4-symbol coefficient rows
// c0/c1/c2 live at +2/+4/+6 and the adaptation count at +10; rows are 36
// bytes apart in the context arrays.
#define CDF_C0   2
#define CDF_C1   4
#define CDF_C2   6
#define CDF_CNT 10

// func coeffBaseLevelsARM64(c *Cursor, scanHot unsafe.Pointer, cHi, cLo int,
//	levels *uint8, stride int, base *CDF, br *CDF,
//	dirty *int16, levelDirty *int16, class int, update bool) int
//
// M-D3 D3-b kernel (spec: internal/av1/tile/coeff_asm_arm64_spec.go §3).
// Scalar source-shaped port of the 2D/horizontal/vertical base-levels loops
// with the BR high-token chain (dav1d msac_decode_hi_tok shape, already proven
// as readCDF4HighTokenUpdateArch) inlined, over goav1's 64-bit XOR-into-ones
// window. The refill blocks are
// verbatim copies of the established reader_bit_arm64.s refill; src ptr/len
// and the cursor pointer are reloaded from the stack args inside them so the
// loop keeps three more persistent registers.
//
// Register plan (see spec §3): persistent R1 base, R2 br, R3 scanHot,
// R4 levels, R5 cnt, R6 pos, R7 dif, R8 rng, R10 tellOffs, R19 c, R20 cLo,
// R21 stride, R22 dirty cursor, R23 levelDirty cursor, R24 update flag,
// R25 appended, R15 level, R26 scanHot entry / BR extra accumulator, R27 class.
// Iteration scratch R0, R9, R11-R14, R16, R17; refill clobbers only
// {R0, R9, R11, R12, R13, R17}. V8-V15, R18, R28-R30 untouched.
TEXT ·coeffBaseLevelsARM64(SB), NOSPLIT, $0-104
	MOVD  c+0(FP), R0
	MOVD  CURSOR_DIF(R0), R7
	MOVHU CURSOR_RNG(R0), R8
	MOVH  CURSOR_CNT(R0), R5
	MOVWU CURSOR_POS(R0), R6
	MOVH  CURSOR_TELL_OFFS(R0), R10
	MOVD  scanHot+8(FP), R3
	MOVD  cHi+16(FP), R19
	MOVD  cLo+24(FP), R20
	MOVD  levels+32(FP), R4
	MOVD  stride+40(FP), R21
	MOVD  base+48(FP), R1
	MOVD  br+56(FP), R2
	MOVD  dirty+64(FP), R22
	MOVD  levelDirty+72(FP), R23
	MOVD  class+80(FP), R27
	MOVBU update+88(FP), R24
	MOVD  $0, R25
	CMP   R20, R19
	BLT   done

loop:
	MOVD (R3)(R19<<3), R26 // scanHot[c]: pos | padded<<16 | lower2D<<32 | br2D<<40
	CBNZ R27, ctx1D

	// Base context: 0 for the DC position, else
	// min((Σ clip3(neighbour)+1)>>1, 4) + lower2DOffset. clip3 is computed as
	// 3 + ((x-3)&((x-3)>>63)); the +3 terms are folded into the final +16
	// (= 5*3 + the AV1 (mag+1) rounding).
	UBFX  $0, R26, $16, R12 // pos
	CBZ   R12, ctxZero
	UBFX  $16, R26, $16, R13 // padded
	ADD   R13, R21, R14      // s1 = padded + stride
	MOVBU (R4)(R14), R11     // levels[s1]
	SUB   $3, R11, R11
	AND   R11->63, R11, R16
	ADD   $1, R13, R15
	MOVBU (R4)(R15), R11 // levels[padded+1]
	SUB   $3, R11, R11
	AND   R11->63, R11, R11
	ADD   R11, R16, R16
	ADD   $1, R14, R15
	MOVBU (R4)(R15), R11 // levels[s1+1]
	SUB   $3, R11, R11
	AND   R11->63, R11, R11
	ADD   R11, R16, R16
	ADD   R21, R14, R15
	MOVBU (R4)(R15), R11 // levels[padded+2*stride]
	SUB   $3, R11, R11
	AND   R11->63, R11, R11
	ADD   R11, R16, R16
	ADD   $2, R13, R15
	MOVBU (R4)(R15), R11 // levels[padded+2]
	SUB   $3, R11, R11
	AND   R11->63, R11, R11
	ADD   R11, R16, R16
	ADD   $16, R16, R16 // mag+1
	LSR   $1, R16, R16  // (mag+1)>>1
	MOVD  $4, R11
	CMP   R11, R16
	CSEL  LO, R16, R11, R16 // min(.., 4)
	SBFX  $32, R26, $8, R11 // lower2DOffset
	ADD   R11, R16, R16
	B     ctxDone

ctxZero:
	MOVD $0, R16
	B    ctxDone

ctx1D:
	// Horizontal transforms use {s1,p1,s2,s3,s4}; vertical transforms use
	// {s1,p1,p2,p3,p4}. The scan table stores the corresponding 1D class
	// offset in the same byte used by lower2DOffset.
	UBFX  $16, R26, $16, R13 // padded
	ADD   R13, R21, R14      // s1
	MOVBU (R4)(R14), R11
	SUB   $3, R11, R11
	AND   R11->63, R11, R16
	ADD   $1, R13, R15       // p1
	MOVBU (R4)(R15), R11
	SUB   $3, R11, R11
	AND   R11->63, R11, R11
	ADD   R11, R16, R16
	CMP   $1, R27
	BNE   ctxVertTail

	ADD   R14, R21, R15 // s2
	MOVBU (R4)(R15), R11
	SUB   $3, R11, R11
	AND   R11->63, R11, R11
	ADD   R11, R16, R16
	ADD   R15, R21, R15 // s3
	MOVBU (R4)(R15), R11
	SUB   $3, R11, R11
	AND   R11->63, R11, R11
	ADD   R11, R16, R16
	ADD   R15, R21, R15 // s4
	MOVBU (R4)(R15), R11
	SUB   $3, R11, R11
	AND   R11->63, R11, R11
	ADD   R11, R16, R16
	B     ctx1DReduce

ctxVertTail:
	ADD   $2, R13, R15 // p2
	MOVBU (R4)(R15), R11
	SUB   $3, R11, R11
	AND   R11->63, R11, R11
	ADD   R11, R16, R16
	ADD   $3, R13, R15 // p3
	MOVBU (R4)(R15), R11
	SUB   $3, R11, R11
	AND   R11->63, R11, R11
	ADD   R11, R16, R16
	ADD   $4, R13, R15 // p4
	MOVBU (R4)(R15), R11
	SUB   $3, R11, R11
	AND   R11->63, R11, R11
	ADD   R11, R16, R16

ctx1DReduce:
	ADD   $16, R16, R16
	LSR   $1, R16, R16
	MOVD  $4, R11
	CMP   R11, R16
	CSEL  LO, R16, R11, R16
	SBFX  $32, R26, $8, R11
	ADD   R11, R16, R16

ctxDone:
	ADD R16<<3, R16, R11 // 9*ctx
	ADD R11<<2, R1, R14  // row = base + 36*ctx

	// 4-symbol inverse-CDF search on row R14 (readCDF4UpdateKnown shape):
	// symbol -> R16, upper -> R13, lower -> R17.
	MOVD  R8, R13      // upper = rng
	LSR   $48, R7, R12 // coded = dif >> (ecWindow-16)
	LSR   $8, R8, R9   // rngHi = rng >> 8
	MOVHU CDF_C0(R14), R11
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $12, R17, R17 // lower0 (+3*ecMinProb)
	MOVD  $0, R16
	CMP   R17, R12
	BCS   baseRenorm
	MOVD  $1, R16
	MOVD  R17, R13
	MOVHU CDF_C1(R14), R11
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $8, R17, R17 // lower1 (+2*ecMinProb)
	CMP   R17, R12
	BCS   baseRenorm
	MOVD  $2, R16
	MOVD  R17, R13
	MOVHU CDF_C2(R14), R11
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $4, R17, R17 // lower2 (+ecMinProb)
	CMP   R17, R12
	BCS   baseRenorm
	MOVD  $3, R16
	MOVD  R17, R13
	MOVD  $0, R17

baseRenorm:
	LSL  $48, R17, R11
	SUB  R11, R7, R7  // dif -= lower << (ecWindow-16)
	SUB  R17, R13, R8 // rng = upper - lower
	ADD  $1, R7, R7
	UBFX $0, R8, $16, R12
	CLZ  R12, R12
	SUB  $48, R12, R12
	LSL  R12, R7, R7
	SUB  $1, R7, R7
	LSL  R12, R8, R8
	SUB  R12, R5, R5
	TBZ  $63, R5, baseAdapt

	// Refill (64-bit-window port shared with reader_bit_arm64.s); preserves
	// R14 row and R16 symbol.
	MOVD $40, R11
	SUB  R5, R11, R11 // shift = ecWindow-9-(cnt+15) = 40 - cnt
	MOVD c+0(FP), R0
	MOVD CURSOR_SRC_PTR(R0), R12
	MOVD CURSOR_SRC_LEN(R0), R13
	SUB  R6, R13, R9  // remaining = len(src) - pos
	CMP  $55, R11
	BHI  baseRefillSlow // shift > ecWindow-9 (or negative): byte loop
	CMP  $8, R9
	BLT  baseRefillSlow
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
	B    baseAdapt  // n <= 7 < remaining: end of buffer unreachable

baseRefillSlow:
	CMP $0, R11
	BLT baseRefillTail
	CMP R13, R6
	BGE baseRefillTail
	ADD   R12, R6, R0
	MOVBU (R0), R17
	LSL   R11, R17, R17
	EOR   R17, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     baseRefillSlow

baseRefillTail:
	CMP  R13, R6
	BLT  baseAdapt
	MOVD $0x4000, R9
	SUB  R5, R9, R0
	ADD  R0, R10, R10
	MOVD R9, R5

baseAdapt:
	// updateCDFWindow for symbols==4: rate = 5 + count>>4 (the symbols>3
	// bias is baked into the 5); adapt in memory, count saturates at 32.
	CBZ   R24, baseAdaptDone
	MOVHU CDF_CNT(R14), R13
	LSR   $4, R13, R11
	ADD   $5, R11, R11 // rate
	CBZ   R16, baseUpd0
	CMP   $1, R16
	BEQ   baseUpd1
	CMP   $2, R16
	BEQ   baseUpd2

	// symbol 3: up c0, up c1, up c2
	MOVHU CDF_C0(R14), R12
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	MOVH  R12, CDF_C0(R14)
	MOVHU CDF_C1(R14), R12
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	MOVH  R12, CDF_C1(R14)
	MOVHU CDF_C2(R14), R12
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	MOVH  R12, CDF_C2(R14)
	B     baseUpdCount

baseUpd0:
	MOVHU CDF_C0(R14), R12
	LSR   R11, R12, R9
	SUB   R9, R12, R12
	MOVH  R12, CDF_C0(R14)
	MOVHU CDF_C1(R14), R12
	LSR   R11, R12, R9
	SUB   R9, R12, R12
	MOVH  R12, CDF_C1(R14)
	MOVHU CDF_C2(R14), R12
	LSR   R11, R12, R9
	SUB   R9, R12, R12
	MOVH  R12, CDF_C2(R14)
	B     baseUpdCount

baseUpd1:
	MOVHU CDF_C0(R14), R12
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	MOVH  R12, CDF_C0(R14)
	MOVHU CDF_C1(R14), R12
	LSR   R11, R12, R9
	SUB   R9, R12, R12
	MOVH  R12, CDF_C1(R14)
	MOVHU CDF_C2(R14), R12
	LSR   R11, R12, R9
	SUB   R9, R12, R12
	MOVH  R12, CDF_C2(R14)
	B     baseUpdCount

baseUpd2:
	MOVHU CDF_C0(R14), R12
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	MOVH  R12, CDF_C0(R14)
	MOVHU CDF_C1(R14), R12
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	MOVH  R12, CDF_C1(R14)
	MOVHU CDF_C2(R14), R12
	LSR   R11, R12, R9
	SUB   R9, R12, R12
	MOVH  R12, CDF_C2(R14)

baseUpdCount:
	CMP  $32, R13
	BGE  baseAdaptDone
	ADD  $1, R13, R13
	MOVH R13, CDF_CNT(R14)

baseAdaptDone:
	MOVD R16, R15 // level = symbol
	CMP  $3, R16
	BNE  store

	// BR high-token chain (readCDF4HighToken{UpdateLoop,Known} shape):
	// brCtx = min((l[p1]+l[s1]+l[s1+1]+1)>>1, 6) + br2DOffset; then up to
	// four chained reads of the same row, extra += symbol, stop when
	// symbol < 3. extra == 3k after k saturating reads, so extra < 12 is
	// exactly the reads < 4 bound and no chain counter is needed.
	CBNZ  R27, brCtx1D
	UBFX  $16, R26, $16, R13 // padded
	ADD   R13, R21, R14      // s1
	MOVBU (R4)(R14), R16     // levels[s1]
	ADD   $1, R13, R11
	MOVBU (R4)(R11), R12 // levels[padded+1]
	ADD   R12, R16, R16
	ADD   $1, R14, R11
	MOVBU (R4)(R11), R12 // levels[s1+1]
	ADD   R12, R16, R16
	ADD   $1, R16, R16
	LSR   $1, R16, R16
	MOVD  $6, R11
	CMP   R11, R16
	CSEL  LO, R16, R11, R16 // min(mag, 6)
	B     brCtxOffset

brCtx1D:
	UBFX  $16, R26, $16, R13 // padded
	ADD   R13, R21, R14      // s1
	MOVBU (R4)(R14), R16
	ADD   $1, R13, R11       // p1
	MOVBU (R4)(R11), R12
	ADD   R12, R16, R16
	CMP   $1, R27
	BNE   brCtxVert
	ADD   R14, R21, R11 // s2
	MOVBU (R4)(R11), R12
	B     brCtx1DLast

brCtxVert:
	ADD   $2, R13, R11 // p2
	MOVBU (R4)(R11), R12

brCtx1DLast:
	ADD   R12, R16, R16

brCtxReduce:
	ADD   $1, R16, R16
	LSR   $1, R16, R16
	MOVD  $6, R11
	CMP   R11, R16
	CSEL  LO, R16, R11, R16

brCtxOffset:
	SBFX  $40, R26, $8, R11 // class-specific BR context offset
	ADD   R11, R16, R16     // brCtx
	ADD   R16<<3, R16, R11
	ADD   R11<<2, R2, R14   // row = br + 36*brCtx
	MOVD  $0, R26           // extra accumulator (entry reloaded after chain)

brLoop:
	MOVD  R8, R13
	LSR   $48, R7, R12
	LSR   $8, R8, R9
	MOVHU CDF_C0(R14), R11
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $12, R17, R17
	MOVD  $0, R16
	CMP   R17, R12
	BCS   brRenorm
	MOVD  $1, R16
	MOVD  R17, R13
	MOVHU CDF_C1(R14), R11
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $8, R17, R17
	CMP   R17, R12
	BCS   brRenorm
	MOVD  $2, R16
	MOVD  R17, R13
	MOVHU CDF_C2(R14), R11
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $4, R17, R17
	CMP   R17, R12
	BCS   brRenorm
	MOVD  $3, R16
	MOVD  R17, R13
	MOVD  $0, R17

brRenorm:
	LSL  $48, R17, R11
	SUB  R11, R7, R7
	SUB  R17, R13, R8
	ADD  $1, R7, R7
	UBFX $0, R8, $16, R12
	CLZ  R12, R12
	SUB  $48, R12, R12
	LSL  R12, R7, R7
	SUB  $1, R7, R7
	LSL  R12, R8, R8
	SUB  R12, R5, R5
	TBZ  $63, R5, brAdapt

	MOVD $40, R11
	SUB  R5, R11, R11
	MOVD c+0(FP), R0
	MOVD CURSOR_SRC_PTR(R0), R12
	MOVD CURSOR_SRC_LEN(R0), R13
	SUB  R6, R13, R9
	CMP  $55, R11
	BHI  brRefillSlow
	CMP  $8, R9
	BLT  brRefillSlow
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
	B    brAdapt

brRefillSlow:
	CMP $0, R11
	BLT brRefillTail
	CMP R13, R6
	BGE brRefillTail
	ADD   R12, R6, R0
	MOVBU (R0), R17
	LSL   R11, R17, R17
	EOR   R17, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     brRefillSlow

brRefillTail:
	CMP  R13, R6
	BLT  brAdapt
	MOVD $0x4000, R9
	SUB  R5, R9, R0
	ADD  R0, R10, R10
	MOVD R9, R5

brAdapt:
	CBZ   R24, brAdaptDone
	MOVHU CDF_CNT(R14), R13
	LSR   $4, R13, R11
	ADD   $5, R11, R11
	CBZ   R16, brUpd0
	CMP   $1, R16
	BEQ   brUpd1
	CMP   $2, R16
	BEQ   brUpd2

	MOVHU CDF_C0(R14), R12
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	MOVH  R12, CDF_C0(R14)
	MOVHU CDF_C1(R14), R12
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	MOVH  R12, CDF_C1(R14)
	MOVHU CDF_C2(R14), R12
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	MOVH  R12, CDF_C2(R14)
	B     brUpdCount

brUpd0:
	MOVHU CDF_C0(R14), R12
	LSR   R11, R12, R9
	SUB   R9, R12, R12
	MOVH  R12, CDF_C0(R14)
	MOVHU CDF_C1(R14), R12
	LSR   R11, R12, R9
	SUB   R9, R12, R12
	MOVH  R12, CDF_C1(R14)
	MOVHU CDF_C2(R14), R12
	LSR   R11, R12, R9
	SUB   R9, R12, R12
	MOVH  R12, CDF_C2(R14)
	B     brUpdCount

brUpd1:
	MOVHU CDF_C0(R14), R12
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	MOVH  R12, CDF_C0(R14)
	MOVHU CDF_C1(R14), R12
	LSR   R11, R12, R9
	SUB   R9, R12, R12
	MOVH  R12, CDF_C1(R14)
	MOVHU CDF_C2(R14), R12
	LSR   R11, R12, R9
	SUB   R9, R12, R12
	MOVH  R12, CDF_C2(R14)
	B     brUpdCount

brUpd2:
	MOVHU CDF_C0(R14), R12
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	MOVH  R12, CDF_C0(R14)
	MOVHU CDF_C1(R14), R12
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	MOVH  R12, CDF_C1(R14)
	MOVHU CDF_C2(R14), R12
	LSR   R11, R12, R9
	SUB   R9, R12, R12
	MOVH  R12, CDF_C2(R14)

brUpdCount:
	CMP  $32, R13
	BGE  brAdaptDone
	ADD  $1, R13, R13
	MOVH R13, CDF_CNT(R14)

brAdaptDone:
	ADD R16, R26, R26 // extra += symbol
	CMP $3, R16
	BNE brDone
	CMP $12, R26
	BLT brLoop

brDone:
	ADD  R26, R15, R15     // level += extra
	MOVD (R3)(R19<<3), R26 // reload scanHot[c]

store:
	CBZ  R15, next
	UBFX $16, R26, $16, R13 // padded
	ADD  R4, R13, R11
	MOVB R15, (R11)         // levels[padded] = level
	MOVH R13, (R23)         // levelDirty append: padded
	ADD  $2, R23, R23
	UBFX $0, R26, $16, R12  // pos
	ORR  R15<<10, R12, R12  // packed = level<<10 | pos
	MOVH R12, (R22)         // dirty append
	ADD  $2, R22, R22
	ADD  $1, R25, R25

next:
	SUB $1, R19, R19
	CMP R20, R19
	BGE loop

done:
	MOVD c+0(FP), R0
	MOVD R7, CURSOR_DIF(R0)
	MOVW R6, CURSOR_POS(R0)
	MOVH R8, CURSOR_RNG(R0)
	MOVH R5, CURSOR_CNT(R0)
	MOVH R10, CURSOR_TELL_OFFS(R0)
	MOVD R25, ret+96(FP)
	RET
