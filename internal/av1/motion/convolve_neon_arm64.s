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
#define IM      64
#define IMSTR   72

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

// func convolve2D8NEONAsm(ctx *convolveNEONCtx)
//
// Both-axes-fractional 8-bit 2D convolve, bit-exact with convolve2D8PureGo.
//
// Pass 1 (horizontal): for each of imH = height+7 rows, load 16 ref bytes per
// 8-column group, slide the 8-tap window with EXT, widen-MAC (smlal/smlal2),
// add the 1<<(8+FILTER_BITS-1)=16384 bias, round by round0Bits=3 (srshr) and
// store the truncating int16 narrow (xtn/xtn2 == Go int16() wraparound) into
// the int16 intermediate `im`.
//
// Pass 2 (vertical): for each of height rows, MAC 8 int16 `im` rows with the
// vertical kernel (smlal/smlal2 widen int16->int32), add yBias=1<<19=524288,
// round by round1Bits=11 (srshr), subtract roundOffset=384, then saturate to
// [0,255] (sqxtn+sqxtun == clipPixel, since the final round `bits` is 0).
TEXT ·convolve2D8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD DST(R0), R1        // dst row base
	MOVD REF(R0), R2        // ref tap-window base (row refY-foY, col refX-foX)
	MOVD KERNEL(R0), R3     // vertical kernel ptr
	MOVD XKERN(R0), R12     // horizontal kernel ptr
	MOVD DSTSTR(R0), R4     // dst stride (bytes)
	MOVD REFSTR(R0), R5     // ref stride (bytes)
	MOVD WIDTH(R0), R6      // width
	MOVD HEIGHT(R0), R7     // height
	MOVD IM(R0), R13        // im base ptr
	MOVD IMSTR(R0), R14     // im row stride (int16 elements)
	LSL  $1, R14, R14       // im row stride in bytes (2 bytes / int16)

	// imH = height + filterTaps - 1 = height + 7
	ADD  $7, R7, R15        // R15 = imH (horizontal pass row count)

	// Horizontal pass setup.
	WORD $0x4c407580       // ld1 {v0.8h}, [x12]  load 8 horizontal taps
	MOVD $16384, R11       // xBias = 1 << (8 + FILTER_BITS - 1)
	WORD $0x4e040d72       // dup v18.4s, w11     bias broadcast (s32 lanes)

	MOVD R2, R16           // ref row cursor
	MOVD R13, R17          // im row cursor

h2RowLoop:
	CBZ  R15, h2Done
	MOVD R16, R9           // ref column cursor
	MOVD R17, R10          // im column cursor
	MOVD R6, R8            // remaining columns

h2ColLoop:
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

	WORD $0x4eb28610       // add v16.4s, v16.4s, v18.4s   += xBias
	WORD $0x4eb28631       // add v17.4s, v17.4s, v18.4s
	WORD $0x4f3d2610       // srshr v16.4s, v16.4s, #3
	WORD $0x4f3d2631       // srshr v17.4s, v17.4s, #3
	WORD $0x0e612a10       // xtn  v16.4h, v16.4s   truncating narrow (== int16())
	WORD $0x4e612a30       // xtn2 v16.8h, v17.4s
	WORD $0x4c007550       // st1 {v16.8h}, [x10]   store 8 int16 intermediates

	ADD  $8, R9, R9        // ref += 8 columns
	ADD  $16, R10, R10     // im  += 8 int16 (16 bytes)
	SUB  $8, R8, R8
	CBNZ R8, h2ColLoop

	ADD  R5, R16, R16      // ref row += refStride
	ADD  R14, R17, R17     // im  row += imStride bytes
	SUB  $1, R15, R15
	CBNZ R15, h2RowLoop

h2Done:
	// Vertical pass setup.
	WORD $0x4c407460       // ld1 {v0.8h}, [x3]  load 8 vertical taps
	MOVD $524288, R11      // yBias = 1 << offsetBits (offsetBits = 19)
	WORD $0x4e040d72       // dup v18.4s, w11    yBias broadcast
	MOVD $384, R11         // roundOffset = (1<<8) + (1<<7)
	WORD $0x4e040d73       // dup v19.4s, w11    roundOffset broadcast

	MOVD R13, R17          // im row-window base for output row 0

v2RowLoop:
	CBZ  R7, v2Done
	MOVD R1, R10           // dst column cursor
	MOVD R17, R11          // im row-window base for this output row
	MOVD R6, R8            // remaining columns

v2ColLoop:
	MOVD R11, R9           // R9 walks the 8 tap rows; post-index by imStride bytes
	WORD $0x4eb21e50       // mov v16.16b, v18.16b   init acc to yBias
	WORD $0x4eb21e51       // mov v17.16b, v18.16b

	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f402030       // smlal  v16.4s, v1.4h, v0.h[0]
	WORD $0x4f402031       // smlal2 v17.4s, v1.8h, v0.h[0]
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f502030       // smlal  v16.4s, v1.4h, v0.h[1]
	WORD $0x4f502031       // smlal2 v17.4s, v1.8h, v0.h[1]
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f602030       // smlal  v16.4s, v1.4h, v0.h[2]
	WORD $0x4f602031       // smlal2 v17.4s, v1.8h, v0.h[2]
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f702030       // smlal  v16.4s, v1.4h, v0.h[3]
	WORD $0x4f702031       // smlal2 v17.4s, v1.8h, v0.h[3]
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f402830       // smlal  v16.4s, v1.4h, v0.h[4]
	WORD $0x4f402831       // smlal2 v17.4s, v1.8h, v0.h[4]
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f502830       // smlal  v16.4s, v1.4h, v0.h[5]
	WORD $0x4f502831       // smlal2 v17.4s, v1.8h, v0.h[5]
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f602830       // smlal  v16.4s, v1.4h, v0.h[6]
	WORD $0x4f602831       // smlal2 v17.4s, v1.8h, v0.h[6]
	WORD $0x4cce7521       // ld1 {v1.8h}, [x9], x14
	WORD $0x0f702830       // smlal  v16.4s, v1.4h, v0.h[7]
	WORD $0x4f702831       // smlal2 v17.4s, v1.8h, v0.h[7]

	WORD $0x4f352610       // srshr v16.4s, v16.4s, #11
	WORD $0x4f352631       // srshr v17.4s, v17.4s, #11
	WORD $0x6eb38610       // sub v16.4s, v16.4s, v19.4s   -= roundOffset
	WORD $0x6eb38631       // sub v17.4s, v17.4s, v19.4s
	WORD $0x0e614a10       // sqxtn  v16.4h, v16.4s
	WORD $0x4e614a30       // sqxtn2 v16.8h, v17.4s
	WORD $0x2e212a10       // sqxtun v16.8b, v16.8h   clip to [0,255]
	WORD $0x0c007150       // st1 {v16.8b}, [x10]

	ADD  $8, R10, R10      // dst += 8 pixels
	ADD  $16, R11, R11     // im window += 8 columns (16 bytes)
	SUB  $8, R8, R8
	CBNZ R8, v2ColLoop

	ADD  R4, R1, R1        // dst row += dstStride
	ADD  R14, R17, R17     // im base row += imStride bytes
	SUB  $1, R7, R7
	CBNZ R7, v2RowLoop

v2Done:
	RET

// Width-4 NEON kernels. AV1's smallest inter blocks are 4 wide (every 4:2:0
// chroma block of an 8x8 luma block is 4x4, plus 4xN luma), and they are too
// narrow for the width-8 loops above. These produce exactly 4 output pixels per
// row using the low halves of the NEON registers (.4h/.4s lanes, no smlal2) and
// store 4 bytes with st1 {Vn.s}[0]. The arithmetic is identical to the width-8
// kernels, so they stay bit-exact with the pure-Go reference. Source over-read
// is bounded to at most 1 byte past the resident tap window (the X / 2D
// horizontal pass loads bytes 8..11 to fill the slide register but only
// consumes samples 8..10), matching the width-8 kernels' existing tolerance.
//
// Additional WORD encodings used here (cross-checked against the system
// assembler), in addition to those documented above:
//   ld1 {v1.s}[0], [x9], x5   0dc58121  load 4 bytes (one s-lane), post-index Xs
//   ld1 {v5.s}[0], [x15]      0d4081e5  load 4 bytes (one s-lane) from Xn
//   st1 {v16.s}[0], [x1]      0d008030  store 4 bytes (one s-lane)
//   st1 {v16.4h}, [x17]       0c007630  store 4 int16
//   ld1 {v1.4h}, [x9], x14    0cce7521  load 4 int16, post-index by x14

// func convolveY8NEONAsmW4(ctx *convolveNEONCtx)
//
// Vertical 8-tap convolve, width 4. Mirrors convolveY8NEONAsm but loads exactly
// 4 source bytes per tap row (no over-read) and emits 4 output bytes per row.
TEXT ·convolveY8NEONAsmW4(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD DST(R0), R1
	MOVD REF(R0), R2
	MOVD KERNEL(R0), R3
	MOVD DSTSTR(R0), R4
	MOVD REFSTR(R0), R5
	MOVD HEIGHT(R0), R7

	WORD $0x4c407460       // ld1 {v0.8h}, [x3]   load 8 taps

y4RowLoop:
	CBZ  R7, y4Done
	MOVD R2, R9            // R9 walks the 8 tap rows; post-index by R5
	WORD $0x4f000410       // movi v16.4s, #0

	WORD $0x0dc58121       // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f402030       // smlal v16.4s, v1.4h, v0.h[0]
	WORD $0x0dc58121       // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f502030       // smlal v16.4s, v1.4h, v0.h[1]
	WORD $0x0dc58121       // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f602030       // smlal v16.4s, v1.4h, v0.h[2]
	WORD $0x0dc58121       // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f702030       // smlal v16.4s, v1.4h, v0.h[3]
	WORD $0x0dc58121       // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f402830       // smlal v16.4s, v1.4h, v0.h[4]
	WORD $0x0dc58121       // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f502830       // smlal v16.4s, v1.4h, v0.h[5]
	WORD $0x0dc58121       // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f602830       // smlal v16.4s, v1.4h, v0.h[6]
	WORD $0x0dc58121       // ld1 {v1.s}[0], [x9], x5
	WORD $0x2f08a421       // ushll v1.8h, v1.8b, #0
	WORD $0x0f702830       // smlal v16.4s, v1.4h, v0.h[7]

	WORD $0x4f392610       // srshr v16.4s, v16.4s, #7
	WORD $0x0e614a10       // sqxtn  v16.4h, v16.4s
	WORD $0x2e212a10       // sqxtun v16.8b, v16.8h
	WORD $0x0d008030       // st1 {v16.s}[0], [x1]   store 4 output bytes

	ADD  R4, R1, R1        // dst += dstStride
	ADD  R5, R2, R2        // ref += refStride
	SUB  $1, R7, R7
	CBNZ R7, y4RowLoop

y4Done:
	RET

// func convolveX8NEONAsmW4(ctx *convolveNEONCtx)
//
// Horizontal 8-tap convolve, width 4. Mirrors convolveX8NEONAsm but loads the
// 8-tap window as bytes 0..7 (v2) plus bytes 8..11 (v3, only lanes 0..2 used)
// and emits 4 output bytes per row.
TEXT ·convolveX8NEONAsmW4(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD DST(R0), R1
	MOVD REF(R0), R2
	MOVD KERNEL(R0), R3
	MOVD DSTSTR(R0), R4
	MOVD REFSTR(R0), R5
	MOVD HEIGHT(R0), R7

	WORD $0x4c407460       // ld1 {v0.8h}, [x3]   load 8 taps

x4RowLoop:
	CBZ  R7, x4Done
	MOVD R2, R9            // ref column cursor
	ADD  $8, R9, R15       // R15 = ref + 8 (upper 4 source bytes)
	WORD $0x4f000410       // movi v16.4s, #0
	WORD $0x0c407121       // ld1 {v1.8b}, [x9]      bytes 0..7
	WORD $0x2f08a422       // ushll v2.8h, v1.8b, #0  samples 0..7
	WORD $0x0d4081e5       // ld1 {v5.s}[0], [x15]   bytes 8..11
	WORD $0x2f08a4a3       // ushll v3.8h, v5.8b, #0  samples 8..11 (lanes 0..3)
	WORD $0x0f402050       // smlal v16.4s, v2.4h, v0.h[0]
	WORD $0x6e031044       // ext v4.16b, v2.16b, v3.16b, #2
	WORD $0x0f502090       // smlal v16.4s, v4.4h, v0.h[1]
	WORD $0x6e032044       // ext v4.16b, v2.16b, v3.16b, #4
	WORD $0x0f602090       // smlal v16.4s, v4.4h, v0.h[2]
	WORD $0x6e033044       // ext v4.16b, v2.16b, v3.16b, #6
	WORD $0x0f702090       // smlal v16.4s, v4.4h, v0.h[3]
	WORD $0x6e034044       // ext v4.16b, v2.16b, v3.16b, #8
	WORD $0x0f402890       // smlal v16.4s, v4.4h, v0.h[4]
	WORD $0x6e035044       // ext v4.16b, v2.16b, v3.16b, #10
	WORD $0x0f502890       // smlal v16.4s, v4.4h, v0.h[5]
	WORD $0x6e036044       // ext v4.16b, v2.16b, v3.16b, #12
	WORD $0x0f602890       // smlal v16.4s, v4.4h, v0.h[6]
	WORD $0x6e037044       // ext v4.16b, v2.16b, v3.16b, #14
	WORD $0x0f702890       // smlal v16.4s, v4.4h, v0.h[7]

	WORD $0x4f3d2610       // srshr v16.4s, v16.4s, #3
	WORD $0x4f3c2610       // srshr v16.4s, v16.4s, #4
	WORD $0x0e614a10       // sqxtn  v16.4h, v16.4s
	WORD $0x2e212a10       // sqxtun v16.8b, v16.8h
	WORD $0x0d008030       // st1 {v16.s}[0], [x1]

	ADD  R4, R1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, x4RowLoop

x4Done:
	RET

// func convolve2D8NEONAsmW4(ctx *convolveNEONCtx)
//
// Both-axes-fractional 2D convolve, width 4. Same two-pass structure as
// convolve2D8NEONAsm: a horizontal pass writes 4 int16 intermediates per row,
// then a vertical pass MACs 8 int16 rows and clips to [0,255].
TEXT ·convolve2D8NEONAsmW4(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD DST(R0), R1        // dst row base
	MOVD REF(R0), R2        // ref tap-window base
	MOVD KERNEL(R0), R3     // vertical kernel ptr
	MOVD XKERN(R0), R12     // horizontal kernel ptr
	MOVD DSTSTR(R0), R4
	MOVD REFSTR(R0), R5
	MOVD HEIGHT(R0), R7
	MOVD IM(R0), R13        // im base ptr
	MOVD IMSTR(R0), R14     // im row stride (int16 elements)
	LSL  $1, R14, R14       // im row stride in bytes

	ADD  $7, R7, R6         // R6 = imH = height + 7 (horizontal row count)

	// Horizontal pass setup.
	WORD $0x4c407580       // ld1 {v0.8h}, [x12]  load 8 horizontal taps
	MOVD $16384, R11       // xBias = 1 << (8 + FILTER_BITS - 1)
	WORD $0x4e040d72       // dup v18.4s, w11

	MOVD R2, R16           // ref row cursor
	MOVD R13, R17          // im row cursor (bytes)

h4RowLoop:
	CBZ  R6, h4Done
	MOVD R16, R9           // ref column cursor
	ADD  $8, R9, R15       // R15 = ref + 8
	WORD $0x4eb21e50       // mov v16.16b, v18.16b   acc = xBias
	WORD $0x0c407121       // ld1 {v1.8b}, [x9]      bytes 0..7
	WORD $0x2f08a422       // ushll v2.8h, v1.8b, #0
	WORD $0x0d4081e5       // ld1 {v5.s}[0], [x15]   bytes 8..11
	WORD $0x2f08a4a3       // ushll v3.8h, v5.8b, #0
	WORD $0x0f402050       // smlal v16.4s, v2.4h, v0.h[0]
	WORD $0x6e031044       // ext v4.16b, v2.16b, v3.16b, #2
	WORD $0x0f502090       // smlal v16.4s, v4.4h, v0.h[1]
	WORD $0x6e032044       // ext v4.16b, v2.16b, v3.16b, #4
	WORD $0x0f602090       // smlal v16.4s, v4.4h, v0.h[2]
	WORD $0x6e033044       // ext v4.16b, v2.16b, v3.16b, #6
	WORD $0x0f702090       // smlal v16.4s, v4.4h, v0.h[3]
	WORD $0x6e034044       // ext v4.16b, v2.16b, v3.16b, #8
	WORD $0x0f402890       // smlal v16.4s, v4.4h, v0.h[4]
	WORD $0x6e035044       // ext v4.16b, v2.16b, v3.16b, #10
	WORD $0x0f502890       // smlal v16.4s, v4.4h, v0.h[5]
	WORD $0x6e036044       // ext v4.16b, v2.16b, v3.16b, #12
	WORD $0x0f602890       // smlal v16.4s, v4.4h, v0.h[6]
	WORD $0x6e037044       // ext v4.16b, v2.16b, v3.16b, #14
	WORD $0x0f702890       // smlal v16.4s, v4.4h, v0.h[7]

	WORD $0x4f3d2610       // srshr v16.4s, v16.4s, #3
	WORD $0x0e612a10       // xtn v16.4h, v16.4s    truncating narrow (== int16())
	WORD $0x0c007630       // st1 {v16.4h}, [x17]   store 4 int16 intermediates

	ADD  R5, R16, R16      // ref row += refStride
	ADD  R14, R17, R17     // im row += imStride bytes
	SUB  $1, R6, R6
	CBNZ R6, h4RowLoop

h4Done:
	// Vertical pass setup.
	WORD $0x4c407460       // ld1 {v0.8h}, [x3]  load 8 vertical taps
	MOVD $524288, R11      // yBias = 1 << offsetBits (offsetBits = 19)
	WORD $0x4e040d72       // dup v18.4s, w11
	MOVD $384, R11         // roundOffset = (1<<8) + (1<<7)
	WORD $0x4e040d73       // dup v19.4s, w11

	MOVD R13, R17          // im row-window base for output row 0 (bytes)
	MOVD HEIGHT(R0), R7

v4RowLoop:
	CBZ  R7, v4Done
	MOVD R17, R9           // R9 walks the 8 tap rows; post-index by R14 bytes
	WORD $0x4eb21e50       // mov v16.16b, v18.16b   acc = yBias

	WORD $0x0cce7521       // ld1 {v1.4h}, [x9], x14
	WORD $0x0f402030       // smlal v16.4s, v1.4h, v0.h[0]
	WORD $0x0cce7521       // ld1 {v1.4h}, [x9], x14
	WORD $0x0f502030       // smlal v16.4s, v1.4h, v0.h[1]
	WORD $0x0cce7521       // ld1 {v1.4h}, [x9], x14
	WORD $0x0f602030       // smlal v16.4s, v1.4h, v0.h[2]
	WORD $0x0cce7521       // ld1 {v1.4h}, [x9], x14
	WORD $0x0f702030       // smlal v16.4s, v1.4h, v0.h[3]
	WORD $0x0cce7521       // ld1 {v1.4h}, [x9], x14
	WORD $0x0f402830       // smlal v16.4s, v1.4h, v0.h[4]
	WORD $0x0cce7521       // ld1 {v1.4h}, [x9], x14
	WORD $0x0f502830       // smlal v16.4s, v1.4h, v0.h[5]
	WORD $0x0cce7521       // ld1 {v1.4h}, [x9], x14
	WORD $0x0f602830       // smlal v16.4s, v1.4h, v0.h[6]
	WORD $0x0cce7521       // ld1 {v1.4h}, [x9], x14
	WORD $0x0f702830       // smlal v16.4s, v1.4h, v0.h[7]

	WORD $0x4f352610       // srshr v16.4s, v16.4s, #11
	WORD $0x6eb38610       // sub v16.4s, v16.4s, v19.4s   -= roundOffset
	WORD $0x0e614a10       // sqxtn  v16.4h, v16.4s
	WORD $0x2e212a10       // sqxtun v16.8b, v16.8h   clip to [0,255]
	WORD $0x0d008030       // st1 {v16.s}[0], [x1]

	ADD  R4, R1, R1        // dst row += dstStride
	ADD  R14, R17, R17     // im base row += imStride bytes
	SUB  $1, R7, R7
	CBNZ R7, v4RowLoop

v4Done:
	RET
