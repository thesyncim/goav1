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
// symbols u8 at +0, values [17]u16 at +2. Binary rows keep c0 at +2 and the
// adaptation count (values[2]) at +6; 4-symbol rows keep c0/c1/c2 at +2/+4/+6
// and the count (values[4]) at +10; the 11-symbol classes row keeps its count
// (values[11]) at +24.
#define CDF_C0     2
#define CDF_C1     4
#define CDF_C2     6
#define CDF_CNT2   6
#define CDF_CNT4  10
#define CDF_CNT11 24

// tile.MVCDFs layout (compile-time pinned in internal/av1/tile/mv_asm.go):
// Joint at +0, Components[0] at +36, Components[1] at +684; within a
// component: Classes +0, Class0FP[0/1] +36/+72, FP +108, Sign +144,
// Class0HP +180, HP +216, Class0 +252, Bits[i] +288+36*i.
#define MV_COMP0     36
#define MV_COMP1    684
#define MV_CLASS0FP  36
#define MV_FP       108
#define MV_SIGN     144
#define MV_CLASS0HP 180
#define MV_HP       216
#define MV_CLASS0   252
#define MV_BITS     288

// func mvResidualARM64(c *Cursor, mvCDFs unsafe.Pointer, precision int64,
//	update bool) (joint, comp0, comp1 int64)
//
// M-D3 D3-e kernel (spec: internal/av1/tile/coeff_asm_arm64_spec.go §7).
// Scalar source-shaped port of tile.ReadMotionVector's symbol chain (dav1d
// read_mv_residual/read_mv_component_diff shape): one 4-symbol joint read,
// then per coded component the sign binary read, the 11-symbol class read,
// the class-0 binary integer read OR the class-long binary bit loop, and the
// fraction/high-precision reads gated on precision — every read adapting its
// row in place exactly like the Cursor readers it replaces (readCDF4Known /
// readCDF11Known / readBinaryCDFKnown + updateCDF2). The range-decoder window
// stays in registers for the whole chain; the refill blocks are verbatim
// copies of the established reader_bit_arm64.s refill with src ptr/len and
// the cursor pointer reloaded from the stack args inside them.
//
// Component results come back packed (0 when the joint skips the component):
// diff int16 in bits 0-15, integerPart in 16-25, class in 26-29, fraction in
// 30-31, highPrecision in bit 32, sign in bit 33. The chain cannot fail: the
// wrapper pre-validates every row shape and |diff| <= 16384 by construction.
//
// Register plan (D3-b/D3-c conventions): persistent R1 mvCDFs, R5 cnt,
// R6 pos, R7 dif, R8 rng, R10 tellOffs, R19 component index, R20 component
// base, R21 precision, R22 update flag, R23/R24 packed comp0/comp1, R25
// joint; component-body locals R2 sign, R3 class, R4 fraction, R15 bits-loop
// index then mag, R26 integerPart. R14 holds the current CDF row and R16 the
// decoded symbol (both survive refill + adapt); iteration scratch R0, R9,
// R11-R13, R17; refill clobbers only {R0, R9, R11, R12, R13, R17}.
// V8-V15, R18, R27-R30 untouched. NOSPLIT leaf, zero locals.
TEXT ·mvResidualARM64(SB), NOSPLIT, $0-56
	MOVD  c+0(FP), R0
	MOVD  CURSOR_DIF(R0), R7
	MOVHU CURSOR_RNG(R0), R8
	MOVH  CURSOR_CNT(R0), R5
	MOVWU CURSOR_POS(R0), R6
	MOVH  CURSOR_TELL_OFFS(R0), R10
	MOVD  mvCDFs+8(FP), R1
	MOVD  precision+16(FP), R21
	MOVBU update+24(FP), R22
	MOVD  $0, R23
	MOVD  $0, R24

	// Joint: 4-symbol read on the row at mvCDFs+0 (readCDF4Known shape).
	MOVD  R1, R14
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
	BCS   jrRenorm
	MOVD  $1, R16
	MOVD  R17, R13
	MOVHU CDF_C1(R14), R11
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $8, R17, R17 // lower1 (+2*ecMinProb)
	CMP   R17, R12
	BCS   jrRenorm
	MOVD  $2, R16
	MOVD  R17, R13
	MOVHU CDF_C2(R14), R11
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $4, R17, R17 // lower2 (+ecMinProb)
	CMP   R17, R12
	BCS   jrRenorm
	MOVD  $3, R16
	MOVD  R17, R13
	MOVD  $0, R17

jrRenorm:
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
	TBZ  $63, R5, jrAdapt

	// Refill (64-bit-window port shared with reader_bit_arm64.s); preserves
	// R14 row and R16 symbol.
	MOVD $40, R11
	SUB  R5, R11, R11 // shift = ecWindow-9-(cnt+15) = 40 - cnt
	MOVD c+0(FP), R0
	MOVD CURSOR_SRC_PTR(R0), R12
	MOVD CURSOR_SRC_LEN(R0), R13
	SUB  R6, R13, R9  // remaining = len(src) - pos
	CMP  $55, R11
	BHI  jrRefillSlow // shift > ecWindow-9 (or negative): byte loop
	CMP  $8, R9
	BLT  jrRefillSlow
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
	B    jrAdapt    // n <= 7 < remaining: end of buffer unreachable

jrRefillSlow:
	CMP $0, R11
	BLT jrRefillTail
	CMP R13, R6
	BGE jrRefillTail
	ADD   R12, R6, R0
	MOVBU (R0), R17
	LSL   R11, R17, R17
	EOR   R17, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     jrRefillSlow

jrRefillTail:
	CMP  R13, R6
	BLT  jrAdapt
	MOVD $0x4000, R9
	SUB  R5, R9, R0
	ADD  R0, R10, R10
	MOVD R9, R5

jrAdapt:
	// updateCDFWindow for symbols==4: rate = 5 + count>>4 (the symbols>3
	// bias is baked into the 5); adapt in memory, count saturates at 32.
	CBZ   R22, jrAdaptDone
	MOVHU CDF_CNT4(R14), R13
	LSR   $4, R13, R11
	ADD   $5, R11, R11 // rate
	CBZ   R16, jrUpd0
	CMP   $1, R16
	BEQ   jrUpd1
	CMP   $2, R16
	BEQ   jrUpd2

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
	B     jrUpdCount

jrUpd0:
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
	B     jrUpdCount

jrUpd1:
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
	B     jrUpdCount

jrUpd2:
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

jrUpdCount:
	CMP $32, R13
	BGE jrAdaptDone
	ADD $1, R13, R13
	MOVH R13, CDF_CNT4(R14)

jrAdaptDone:
	MOVD R16, R25 // joint

	// Component loop: index 0 decodes the vertical (row) component when
	// joint bit 1 is set, index 1 the horizontal (col) component when joint
	// bit 0 is set — MVJoint{Zero,Horizontal,Vertical,Both} = 0..3.
	MOVD $0, R19

compLoop:
	CBNZ R19, compTestH
	TBZ  $1, R25, compNext
	ADD  $MV_COMP0, R1, R20
	B    compBody

compTestH:
	TBZ $0, R25, compNext
	ADD $MV_COMP1, R1, R20

compBody:
	// Sign: binary read on the component Sign row (readBinaryCDFKnown +
	// updateCDF2 shape): symbol 1 when coded < lower.
	ADD   $MV_SIGN, R20, R14
	LSR   $48, R7, R12
	LSR   $8, R8, R9
	MOVHU CDF_C0(R14), R11
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $4, R17, R17
	MOVD  $0, R16
	CMP   R17, R12
	BCC   sgSym1
	LSL   $48, R17, R11
	SUB   R11, R7, R7
	SUB   R17, R8, R8
	B     sgRenorm

sgSym1:
	MOVD $1, R16
	MOVD R17, R8

sgRenorm:
	ADD  $1, R7, R7
	UBFX $0, R8, $16, R12
	CLZ  R12, R12
	SUB  $48, R12, R12
	LSL  R12, R7, R7
	SUB  $1, R7, R7
	LSL  R12, R8, R8
	SUB  R12, R5, R5
	TBZ  $63, R5, sgAdapt

	MOVD $40, R11
	SUB  R5, R11, R11
	MOVD c+0(FP), R0
	MOVD CURSOR_SRC_PTR(R0), R12
	MOVD CURSOR_SRC_LEN(R0), R13
	SUB  R6, R13, R9
	CMP  $55, R11
	BHI  sgRefillSlow
	CMP  $8, R9
	BLT  sgRefillSlow
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
	B    sgAdapt

sgRefillSlow:
	CMP $0, R11
	BLT sgRefillTail
	CMP R13, R6
	BGE sgRefillTail
	ADD   R12, R6, R0
	MOVBU (R0), R17
	LSL   R11, R17, R17
	EOR   R17, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     sgRefillSlow

sgRefillTail:
	CMP  R13, R6
	BLT  sgAdapt
	MOVD $0x4000, R9
	SUB  R5, R9, R0
	ADD  R0, R10, R10
	MOVD R9, R5

sgAdapt:
	// updateCDF2: rate = 4 + count>>4 (2 symbols: no rate bias).
	CBZ   R22, sgAdaptDone
	MOVHU CDF_CNT2(R14), R13
	LSR   $4, R13, R11
	ADD   $4, R11, R11
	MOVHU CDF_C0(R14), R12
	CBZ   R16, sgUpd0
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	B     sgUpdStore

sgUpd0:
	LSR R11, R12, R9
	SUB R9, R12, R12

sgUpdStore:
	MOVH R12, CDF_C0(R14)
	CMP  $32, R13
	BGE  sgAdaptDone
	ADD  $1, R13, R13
	MOVH R13, CDF_CNT2(R14)

sgAdaptDone:
	MOVD R16, R2 // sign

	// Class: 11-symbol read on the component Classes row (readCDF11Known
	// shape, loop form of the inverse-CDF search).
	MOVD  R20, R14
	MOVD  R8, R13      // upper = rng
	LSR   $48, R7, R12 // coded
	LSR   $8, R8, R9   // rngHi
	ADD   $2, R14, R11 // values base
	MOVD  $40, R0      // minTerm = 10*ecMinProb
	MOVD  $0, R16

clLoop:
	MOVHU (R11)(R16<<1), R17
	LSR   $6, R17, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   R0, R17, R17 // lower
	CMP   R17, R12
	BCS   clRenorm     // coded >= lower: symbol found
	ADD   $1, R16, R16
	MOVD  R17, R13     // upper = lower
	SUB   $4, R0, R0   // minTerm -= ecMinProb
	CMP   $10, R16
	BLT   clLoop
	MOVD  $0, R17      // symbol == last(10): lower = 0

clRenorm:
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
	TBZ  $63, R5, clAdapt

	MOVD $40, R11
	SUB  R5, R11, R11
	MOVD c+0(FP), R0
	MOVD CURSOR_SRC_PTR(R0), R12
	MOVD CURSOR_SRC_LEN(R0), R13
	SUB  R6, R13, R9
	CMP  $55, R11
	BHI  clRefillSlow
	CMP  $8, R9
	BLT  clRefillSlow
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
	B    clAdapt

clRefillSlow:
	CMP $0, R11
	BLT clRefillTail
	CMP R13, R6
	BGE clRefillTail
	ADD   R12, R6, R0
	MOVBU (R0), R17
	LSL   R11, R17, R17
	EOR   R17, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     clRefillSlow

clRefillTail:
	CMP  R13, R6
	BLT  clAdapt
	MOVD $0x4000, R9
	SUB  R5, R9, R0
	ADD  R0, R10, R10
	MOVD R9, R5

clAdapt:
	// readCDF11Known adaptation: rate = 5 + count>>4; values[i] moves up for
	// i < symbol and down for symbol <= i < 10; count (values[11]) saturates
	// at 32. R3/R15 are still free here (class/mag are assigned below).
	CBZ   R22, clAdaptDone
	MOVHU CDF_CNT11(R14), R13
	LSR   $4, R13, R11
	ADD   $5, R11, R11 // rate
	ADD   $2, R14, R12 // values base
	MOVD  $0, R0

clUp:
	CMP   R16, R0
	BGE   clDn
	ADD   R0<<1, R12, R9
	MOVHU (R9), R17
	MOVD  $32768, R15
	SUB   R17, R15, R15
	LSR   R11, R15, R15
	ADD   R15, R17, R17
	MOVH  R17, (R9)
	ADD   $1, R0, R0
	B     clUp

clDn:
	CMP   $10, R0
	BGE   clCnt
	ADD   R0<<1, R12, R9
	MOVHU (R9), R17
	LSR   R11, R17, R15
	SUB   R15, R17, R17
	MOVH  R17, (R9)
	ADD   $1, R0, R0
	B     clDn

clCnt:
	CMP  $32, R13
	BGE  clAdaptDone
	ADD  $1, R13, R13
	MOVH R13, CDF_CNT11(R14)

clAdaptDone:
	MOVD R16, R3 // class

	MOVD $0, R26 // integerPart
	MOVD $0, R15 // mag
	MOVD $3, R4  // fraction default
	CBNZ R3, bitsInt

	// Class 0: one binary integer read on the Class0 row.
	ADD   $MV_CLASS0, R20, R14
	LSR   $48, R7, R12
	LSR   $8, R8, R9
	MOVHU CDF_C0(R14), R11
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $4, R17, R17
	MOVD  $0, R16
	CMP   R17, R12
	BCC   c0Sym1
	LSL   $48, R17, R11
	SUB   R11, R7, R7
	SUB   R17, R8, R8
	B     c0Renorm

c0Sym1:
	MOVD $1, R16
	MOVD R17, R8

c0Renorm:
	ADD  $1, R7, R7
	UBFX $0, R8, $16, R12
	CLZ  R12, R12
	SUB  $48, R12, R12
	LSL  R12, R7, R7
	SUB  $1, R7, R7
	LSL  R12, R8, R8
	SUB  R12, R5, R5
	TBZ  $63, R5, c0Adapt

	MOVD $40, R11
	SUB  R5, R11, R11
	MOVD c+0(FP), R0
	MOVD CURSOR_SRC_PTR(R0), R12
	MOVD CURSOR_SRC_LEN(R0), R13
	SUB  R6, R13, R9
	CMP  $55, R11
	BHI  c0RefillSlow
	CMP  $8, R9
	BLT  c0RefillSlow
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
	B    c0Adapt

c0RefillSlow:
	CMP $0, R11
	BLT c0RefillTail
	CMP R13, R6
	BGE c0RefillTail
	ADD   R12, R6, R0
	MOVBU (R0), R17
	LSL   R11, R17, R17
	EOR   R17, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     c0RefillSlow

c0RefillTail:
	CMP  R13, R6
	BLT  c0Adapt
	MOVD $0x4000, R9
	SUB  R5, R9, R0
	ADD  R0, R10, R10
	MOVD R9, R5

c0Adapt:
	CBZ   R22, c0AdaptDone
	MOVHU CDF_CNT2(R14), R13
	LSR   $4, R13, R11
	ADD   $4, R11, R11
	MOVHU CDF_C0(R14), R12
	CBZ   R16, c0Upd0
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	B     c0UpdStore

c0Upd0:
	LSR R11, R12, R9
	SUB R9, R12, R12

c0UpdStore:
	MOVH R12, CDF_C0(R14)
	CMP  $32, R13
	BGE  c0AdaptDone
	ADD  $1, R13, R13
	MOVH R13, CDF_CNT2(R14)

c0AdaptDone:
	MOVD R16, R26 // integerPart
	B    intDone

bitsInt:
	// Class > 0: n = class binary Bits[i] reads build the integer part LSB
	// first (MVClass0Bits == 1 makes n = class). R15 doubles as the loop
	// index (mag is assigned after the loop) and R14 walks the Bits rows —
	// both survive the refill clobber set.
	ADD $MV_BITS, R20, R14

bitsLoop:
	LSR   $48, R7, R12
	LSR   $8, R8, R9
	MOVHU CDF_C0(R14), R11
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $4, R17, R17
	MOVD  $0, R16
	CMP   R17, R12
	BCC   blSym1
	LSL   $48, R17, R11
	SUB   R11, R7, R7
	SUB   R17, R8, R8
	B     blRenorm

blSym1:
	MOVD $1, R16
	MOVD R17, R8

blRenorm:
	ADD  $1, R7, R7
	UBFX $0, R8, $16, R12
	CLZ  R12, R12
	SUB  $48, R12, R12
	LSL  R12, R7, R7
	SUB  $1, R7, R7
	LSL  R12, R8, R8
	SUB  R12, R5, R5
	TBZ  $63, R5, blAdapt

	MOVD $40, R11
	SUB  R5, R11, R11
	MOVD c+0(FP), R0
	MOVD CURSOR_SRC_PTR(R0), R12
	MOVD CURSOR_SRC_LEN(R0), R13
	SUB  R6, R13, R9
	CMP  $55, R11
	BHI  blRefillSlow
	CMP  $8, R9
	BLT  blRefillSlow
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
	B    blAdapt

blRefillSlow:
	CMP $0, R11
	BLT blRefillTail
	CMP R13, R6
	BGE blRefillTail
	ADD   R12, R6, R0
	MOVBU (R0), R17
	LSL   R11, R17, R17
	EOR   R17, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     blRefillSlow

blRefillTail:
	CMP  R13, R6
	BLT  blAdapt
	MOVD $0x4000, R9
	SUB  R5, R9, R0
	ADD  R0, R10, R10
	MOVD R9, R5

blAdapt:
	CBZ   R22, blAdaptDone
	MOVHU CDF_CNT2(R14), R13
	LSR   $4, R13, R11
	ADD   $4, R11, R11
	MOVHU CDF_C0(R14), R12
	CBZ   R16, blUpd0
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	B     blUpdStore

blUpd0:
	LSR R11, R12, R9
	SUB R9, R12, R12

blUpdStore:
	MOVH R12, CDF_C0(R14)
	CMP  $32, R13
	BGE  blAdaptDone
	ADD  $1, R13, R13
	MOVH R13, CDF_CNT2(R14)

blAdaptDone:
	LSL R15, R16, R16
	ORR R16, R26, R26 // integerPart |= bit << i
	ADD $36, R14, R14
	ADD $1, R15, R15
	CMP R3, R15
	BLT bitsLoop

	// mag = MVClass0Size << (class+2) = 1 << (class+3)
	ADD  $3, R3, R11
	MOVD $1, R15
	LSL  R11, R15, R15

intDone:
	TBNZ $63, R21, hpDefault // precision < 0: no subpel, fraction/hp defaults

	// Fraction: 4-symbol read on Class0FP[integerPart] (class 0) or FP.
	CBNZ R3, frFP
	ADD  $MV_CLASS0FP, R20, R14
	ADD  R26<<5, R14, R14
	ADD  R26<<2, R14, R14 // + 36*integerPart
	B    frRead

frFP:
	ADD $MV_FP, R20, R14

frRead:
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
	BCS   frRenorm
	MOVD  $1, R16
	MOVD  R17, R13
	MOVHU CDF_C1(R14), R11
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $8, R17, R17
	CMP   R17, R12
	BCS   frRenorm
	MOVD  $2, R16
	MOVD  R17, R13
	MOVHU CDF_C2(R14), R11
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $4, R17, R17
	CMP   R17, R12
	BCS   frRenorm
	MOVD  $3, R16
	MOVD  R17, R13
	MOVD  $0, R17

frRenorm:
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
	TBZ  $63, R5, frAdapt

	MOVD $40, R11
	SUB  R5, R11, R11
	MOVD c+0(FP), R0
	MOVD CURSOR_SRC_PTR(R0), R12
	MOVD CURSOR_SRC_LEN(R0), R13
	SUB  R6, R13, R9
	CMP  $55, R11
	BHI  frRefillSlow
	CMP  $8, R9
	BLT  frRefillSlow
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
	B    frAdapt

frRefillSlow:
	CMP $0, R11
	BLT frRefillTail
	CMP R13, R6
	BGE frRefillTail
	ADD   R12, R6, R0
	MOVBU (R0), R17
	LSL   R11, R17, R17
	EOR   R17, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     frRefillSlow

frRefillTail:
	CMP  R13, R6
	BLT  frAdapt
	MOVD $0x4000, R9
	SUB  R5, R9, R0
	ADD  R0, R10, R10
	MOVD R9, R5

frAdapt:
	CBZ   R22, frAdaptDone
	MOVHU CDF_CNT4(R14), R13
	LSR   $4, R13, R11
	ADD   $5, R11, R11
	CBZ   R16, frUpd0
	CMP   $1, R16
	BEQ   frUpd1
	CMP   $2, R16
	BEQ   frUpd2

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
	B     frUpdCount

frUpd0:
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
	B     frUpdCount

frUpd1:
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
	B     frUpdCount

frUpd2:
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

frUpdCount:
	CMP $32, R13
	BGE frAdaptDone
	ADD $1, R13, R13
	MOVH R13, CDF_CNT4(R14)

frAdaptDone:
	MOVD R16, R4 // fraction

	// High precision: only when precision > 0; binary read on Class0HP/HP.
	CMP  $0, R21
	BLE  hpDefault
	CBNZ R3, hpFP
	ADD  $MV_CLASS0HP, R20, R14
	B    hpRead

hpFP:
	ADD $MV_HP, R20, R14

hpRead:
	LSR   $48, R7, R12
	LSR   $8, R8, R9
	MOVHU CDF_C0(R14), R11
	LSR   $6, R11, R17
	MUL   R9, R17, R17
	LSR   $1, R17, R17
	ADD   $4, R17, R17
	MOVD  $0, R16
	CMP   R17, R12
	BCC   hpSym1
	LSL   $48, R17, R11
	SUB   R11, R7, R7
	SUB   R17, R8, R8
	B     hpRenorm

hpSym1:
	MOVD $1, R16
	MOVD R17, R8

hpRenorm:
	ADD  $1, R7, R7
	UBFX $0, R8, $16, R12
	CLZ  R12, R12
	SUB  $48, R12, R12
	LSL  R12, R7, R7
	SUB  $1, R7, R7
	LSL  R12, R8, R8
	SUB  R12, R5, R5
	TBZ  $63, R5, hpAdapt

	MOVD $40, R11
	SUB  R5, R11, R11
	MOVD c+0(FP), R0
	MOVD CURSOR_SRC_PTR(R0), R12
	MOVD CURSOR_SRC_LEN(R0), R13
	SUB  R6, R13, R9
	CMP  $55, R11
	BHI  hpRefillSlow
	CMP  $8, R9
	BLT  hpRefillSlow
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
	B    hpAdapt

hpRefillSlow:
	CMP $0, R11
	BLT hpRefillTail
	CMP R13, R6
	BGE hpRefillTail
	ADD   R12, R6, R0
	MOVBU (R0), R17
	LSL   R11, R17, R17
	EOR   R17, R7, R7
	ADD   $8, R5, R5
	SUB   $8, R11, R11
	ADD   $1, R6, R6
	B     hpRefillSlow

hpRefillTail:
	CMP  R13, R6
	BLT  hpAdapt
	MOVD $0x4000, R9
	SUB  R5, R9, R0
	ADD  R0, R10, R10
	MOVD R9, R5

hpAdapt:
	CBZ   R22, hpAdaptDone
	MOVHU CDF_CNT2(R14), R13
	LSR   $4, R13, R11
	ADD   $4, R11, R11
	MOVHU CDF_C0(R14), R12
	CBZ   R16, hpUpd0
	MOVD  $32768, R9
	SUB   R12, R9, R9
	LSR   R11, R9, R9
	ADD   R9, R12, R12
	B     hpUpdStore

hpUpd0:
	LSR R11, R12, R9
	SUB R9, R12, R12

hpUpdStore:
	MOVH R12, CDF_C0(R14)
	CMP  $32, R13
	BGE  hpAdaptDone
	ADD  $1, R13, R13
	MOVH R13, CDF_CNT2(R14)

hpAdaptDone:
	MOVD R16, R17 // hp
	B    packComp

hpDefault:
	MOVD $1, R17 // hp default (fraction keeps its default 3 when skipped)

packComp:
	// mag += ((integerPart<<3) | (fraction<<1) | hp) + 1
	LSL  $3, R26, R11
	ORR  R4<<1, R11, R11
	ORR  R17, R11, R11
	ADD  $1, R11, R11
	ADD  R11, R15, R15
	// diff = sign ? -mag : mag (|diff| <= 16384: int16-safe by construction)
	NEG  R15, R12
	CMP  $0, R2
	CSEL NE, R12, R15, R12
	// pack: diff u16 | integerPart<<16 | class<<26 | fraction<<30 | hp<<32 |
	// sign<<33
	AND  $0xffff, R12, R13
	ORR  R26<<16, R13, R13
	ORR  R3<<26, R13, R13
	ORR  R4<<30, R13, R13
	ORR  R17<<32, R13, R13
	ORR  R2<<33, R13, R13
	CBNZ R19, packC1
	MOVD R13, R23
	B    compNext

packC1:
	MOVD R13, R24

compNext:
	ADD $1, R19, R19
	CMP $2, R19
	BLT compLoop

	MOVD c+0(FP), R0
	MOVD R7, CURSOR_DIF(R0)
	MOVW R6, CURSOR_POS(R0)
	MOVH R8, CURSOR_RNG(R0)
	MOVH R5, CURSOR_CNT(R0)
	MOVH R10, CURSOR_TELL_OFFS(R0)
	MOVD R25, joint+32(FP)
	MOVD R23, comp0+40(FP)
	MOVD R24, comp1+48(FP)
	RET
