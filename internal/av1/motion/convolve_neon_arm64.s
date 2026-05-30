// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// NEON 8-bit convolve kernels. The Go arm64 assembler does not expose integer
// vector multiply-accumulate, rounding shifts, or saturating narrows as named
// mnemonics, so the SIMD instructions are emitted via WORD with hand-verified
// encodings (cross-checked against the system assembler). Pointer arithmetic,
// loop counters, and control flow use ordinary Plan 9 GP-register instructions.
//
// Register / lane semantics used below (all bit-exact with the pure-Go ref):
//   ld1   {Vn.8B}, [Xm](, Xs)  load 8 source bytes, optional post-index by Xs
//   ushll Vn.8H, Vn.8B, #0     zero-extend 8 bytes to 8 halfwords (== uint8 load)
//   movi  Vn.4S, #0            zero a 32-bit-lane accumulator
//   smlal/smlal2 Vd.4S, Vn.{4,8}H, V0.H[i]  signed widening MAC by tap i
//   srshr Vd.4S, Vn.4S, #b     (v + (1<<(b-1))) >> b, signed == roundPowerOfTwo
//   sqxtn/sqxtn2 Vd.{4,8}H, Vn.4S  saturating narrow s32 -> s16
//   sqxtun Vd.8B, Vn.8H        unsigned saturating narrow s16 -> u8 == clipPixel
//   st1   {Vn.8B}, [Xm]        store 8 output bytes

// ctx field offsets (see convolveNEONCtx in convolve_neon_arm64.go).
#define DST     0
#define REF     8
#define KERNEL  16
#define XKERN   24
#define DSTSTR  32
#define REFSTR  40
#define WIDTH   48
#define HEIGHT  56

// func convolveY8NEONAsm(ctx *convolveNEONCtx)
//
// Vertical 8-tap convolve. For each row, processes 8 output columns per
// iteration. The reference pointer addresses the first tap row (refY-fo).
TEXT ·convolveY8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD DST(R0), R1        // dst row base
	MOVD REF(R0), R2        // ref tap-window base (row refY-fo)
	MOVD KERNEL(R0), R3     // kernel ptr
	MOVD DSTSTR(R0), R4     // dst stride
	MOVD REFSTR(R0), R5     // ref stride
	MOVD WIDTH(R0), R6      // width
	MOVD HEIGHT(R0), R7     // height

	WORD $0x4c407460        // ld1 {v0.8h}, [x3]   load 8 taps

yRowLoop:
	CBZ  R7, yDone
	MOVD R1, R10            // dst column cursor
	MOVD R2, R11            // ref row-window base for this output row
	MOVD R6, R8             // remaining columns

yColLoop:
	MOVD R11, R9           // R9 walks the 8 tap rows; post-index by R5
	WORD $0x4f000410       // movi v16.4s, #0
	WORD $0x4f000411       // movi v17.4s, #0

	WORD $0x0cc57121       // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f402030       // smlal  v16.4s, v1.4h, v0.h[0]
	WORD $0x4f402031       // smlal2 v17.4s, v1.8h, v0.h[0]
	WORD $0x0cc57121       // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f502030       // smlal  v16.4s, v1.4h, v0.h[1]
	WORD $0x4f502031       // smlal2 v17.4s, v1.8h, v0.h[1]
	WORD $0x0cc57121       // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f602030       // smlal  v16.4s, v1.4h, v0.h[2]
	WORD $0x4f602031       // smlal2 v17.4s, v1.8h, v0.h[2]
	WORD $0x0cc57121       // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f702030       // smlal  v16.4s, v1.4h, v0.h[3]
	WORD $0x4f702031       // smlal2 v17.4s, v1.8h, v0.h[3]
	WORD $0x0cc57121       // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f402830       // smlal  v16.4s, v1.4h, v0.h[4]
	WORD $0x4f402831       // smlal2 v17.4s, v1.8h, v0.h[4]
	WORD $0x0cc57121       // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f502830       // smlal  v16.4s, v1.4h, v0.h[5]
	WORD $0x4f502831       // smlal2 v17.4s, v1.8h, v0.h[5]
	WORD $0x0cc57121       // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f602830       // smlal  v16.4s, v1.4h, v0.h[6]
	WORD $0x4f602831       // smlal2 v17.4s, v1.8h, v0.h[6]
	WORD $0x0cc57121       // ld1 {v1.8b}, [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f702830       // smlal  v16.4s, v1.4h, v0.h[7]
	WORD $0x4f702831       // smlal2 v17.4s, v1.8h, v0.h[7]

	WORD $0x4f392610       // srshr v16.4s, v16.4s, #7
	WORD $0x4f392631       // srshr v17.4s, v17.4s, #7
	WORD $0x0e614a10       // sqxtn  v16.4h, v16.4s
	WORD $0x4e614a30       // sqxtn2 v16.8h, v17.4s
	WORD $0x2e212a10       // sqxtun v16.8b, v16.8h
	WORD $0x0c007150       // st1 {v16.8b}, [x10]

	ADD  $8, R10, R10      // advance dst by 8 pixels
	ADD  $8, R11, R11      // advance ref window by 8 columns
	SUB  $8, R8, R8
	CBNZ R8, yColLoop

	ADD  R4, R1, R1        // dst += dstStride
	ADD  R5, R2, R2        // ref += refStride
	SUB  $1, R7, R7
	CBNZ R7, yRowLoop

yDone:
	RET

// func convolveX8NEONAsm(ctx *convolveNEONCtx)
//
// Horizontal 8-tap convolve. Loads 16 source bytes per 8-column group, slides
// the tap window with EXT, then applies the two-stage round (round0=3 then
// FILTER_BITS-round0=4) and clips. The reference pointer addresses the first
// tap sample of the row (refX-fo).
TEXT ·convolveX8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD DST(R0), R1
	MOVD REF(R0), R2
	MOVD KERNEL(R0), R3
	MOVD DSTSTR(R0), R4
	MOVD REFSTR(R0), R5
	MOVD WIDTH(R0), R6
	MOVD HEIGHT(R0), R7

	WORD $0x4c407460        // ld1 {v0.8h}, [x3]   load 8 taps

xRowLoop:
	CBZ  R7, xDone
	MOVD R1, R10           // dst column cursor
	MOVD R2, R9            // ref column cursor
	MOVD R6, R8            // remaining columns

xColLoop:
	WORD $0x4f000410       // movi v16.4s, #0
	WORD $0x4f000411       // movi v17.4s, #0
	WORD $0x4c407121       // ld1 {v1.16b}, [x9]
	WORD $0x2f08a422       // ushll  v2.8h, v1.8b, #0
	WORD $0x6f08a423       // ushll2 v3.8h, v1.16b, #0
	WORD $0x0f402050       // smlal  v16.4s, v2.4h, v0.h[0]
	WORD $0x4f402051       // smlal2 v17.4s, v2.8h, v0.h[0]
	WORD $0x6e031044       // ext v4.16b, v2.16b, v3.16b, #2
	WORD $0x0f502090       // smlal  v16.4s, v4.4h, v0.h[1]
	WORD $0x4f502091       // smlal2 v17.4s, v4.8h, v0.h[1]
	WORD $0x6e032044       // ext v4.16b, v2.16b, v3.16b, #4
	WORD $0x0f602090       // smlal  v16.4s, v4.4h, v0.h[2]
	WORD $0x4f602091       // smlal2 v17.4s, v4.8h, v0.h[2]
	WORD $0x6e033044       // ext v4.16b, v2.16b, v3.16b, #6
	WORD $0x0f702090       // smlal  v16.4s, v4.4h, v0.h[3]
	WORD $0x4f702091       // smlal2 v17.4s, v4.8h, v0.h[3]
	WORD $0x6e034044       // ext v4.16b, v2.16b, v3.16b, #8
	WORD $0x0f402890       // smlal  v16.4s, v4.4h, v0.h[4]
	WORD $0x4f402891       // smlal2 v17.4s, v4.8h, v0.h[4]
	WORD $0x6e035044       // ext v4.16b, v2.16b, v3.16b, #10
	WORD $0x0f502890       // smlal  v16.4s, v4.4h, v0.h[5]
	WORD $0x4f502891       // smlal2 v17.4s, v4.8h, v0.h[5]
	WORD $0x6e036044       // ext v4.16b, v2.16b, v3.16b, #12
	WORD $0x0f602890       // smlal  v16.4s, v4.4h, v0.h[6]
	WORD $0x4f602891       // smlal2 v17.4s, v4.8h, v0.h[6]
	WORD $0x6e037044       // ext v4.16b, v2.16b, v3.16b, #14
	WORD $0x0f702890       // smlal  v16.4s, v4.4h, v0.h[7]
	WORD $0x4f702891       // smlal2 v17.4s, v4.8h, v0.h[7]

	WORD $0x4f3d2610       // srshr v16.4s, v16.4s, #3
	WORD $0x4f3d2631       // srshr v17.4s, v17.4s, #3
	WORD $0x4f3c2610       // srshr v16.4s, v16.4s, #4
	WORD $0x4f3c2631       // srshr v17.4s, v17.4s, #4
	WORD $0x0e614a10       // sqxtn  v16.4h, v16.4s
	WORD $0x4e614a30       // sqxtn2 v16.8h, v17.4s
	WORD $0x2e212a10       // sqxtun v16.8b, v16.8h
	WORD $0x0c007150       // st1 {v16.8b}, [x10]

	ADD  $8, R10, R10
	ADD  $8, R9, R9
	SUB  $8, R8, R8
	CBNZ R8, xColLoop

	ADD  R4, R1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, xRowLoop

xDone:
	RET
