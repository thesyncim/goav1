// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// NEON high-bit-depth (10/12-bit) convolve kernels. Samples are little-endian
// uint16; all values are < 4096 so they sit in the positive int16 range and a
// plain ld1 {Vn.Nh} loads them as signed halfwords ready for smull/smlal.
//
// Bit-depth-dependent rounds are applied with SRSHL by a negative lane amount
// (signed rounding shift left == roundPowerOfTwo for a negative count), so the
// asm needs no per-bit-depth WORD immediates. The final clip is a vector
// smax(0)/smin(max) clamp because the upper bound is bit-depth dependent.
//
// Encodings emitted via WORD (Go's arm64 assembler lacks these mnemonics),
// cross-checked against the system assembler:
//   ld1 {v1.8h}, [x9]            4c407521
//   ld1 {v1.4h}, [x9]            0c407521
//   ld1 {v1.4h}, [x9], x5        0cc57521  (post-index by Xs)
//   ld1 {v1.4s}, [x9]            4c407921
//   ld1 {v1.4s}, [x9], x14       4cce7921  (post-index by x14)
//   ext v4.16b, v1.16b, v3.16b, #imm  6e0i_024 (i encodes imm)
//   smull  v16.4s, v1.4h, v0.h[i]
//   smlal  v16.4s, v4.4h, v0.h[i]
//   mla    v16.4s, v1.4s, v20.4s     4eb49430
//   movi   v16.4s, #0                4f000410
//   mov    v16.16b, v18.16b          4eb21e50
//   add    v16.4s, v16.4s, v18.4s    4eb28610
//   sub    v16.4s, v16.4s, v19.4s    6eb38610
//   srshl  v16.4s, v16.4s, v20.4s    4eb45610
//   smax   v16.4s, v16.4s, v21.4s    4eb56610
//   smin   v16.4s, v16.4s, v22.4s    4eb66e10
//   xtn    v16.4h, v16.4s            0e612a10
//   st1 {v16.4h}, [x10]              0c007550
//   dup    v20.4s, w11               4e040d74

// convolveHighBDNEONCtx field offsets.
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
#define ROUND0  80
#define ROUND1  88
#define BITS    96
#define RNDOFF  104
#define XBIAS   112
#define YBIAS   120
#define MAXVAL  128

// func convolveYHighBDNEONAsm(ctx *convolveHighBDNEONCtx)
//
// Vertical 8-tap HBD convolve. out = clipHBD(roundPowerOfTwo(sum, FILTER_BITS)).
// round0 field carries FILTER_BITS. Processes 4 samples per column group.
TEXT ·convolveYHighBDNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD DST(R0), R1
	MOVD REF(R0), R2
	MOVD KERNEL(R0), R3
	MOVD DSTSTR(R0), R4
	MOVD REFSTR(R0), R5
	MOVD WIDTH(R0), R6
	MOVD HEIGHT(R0), R7
	MOVD ROUND0(R0), R11   // FILTER_BITS
	MOVD MAXVAL(R0), R12

	WORD $0x4c407460       // ld1 {v0.8h}, [x3]  load 8 taps
	NEG  R11, R11
	WORD $0x4e040d74       // dup v20.4s, w11    negative round-shift
	WORD $0x4f000415       // movi v21.4s, #0    v21 = 0 (clip lower bound)
	MOVD R12, R11
	WORD $0x4e040d76       // dup v22.4s, w11    v22 = max (clip upper bound)

yRowLoop:
	CBZ  R7, yDone
	MOVD R1, R10           // dst column cursor
	MOVD R2, R13           // ref row-window base for this output row
	MOVD R6, R8            // remaining columns

yColLoop:
	MOVD R13, R9           // R9 walks the 8 tap rows; post-index by R5
	WORD $0x4f000410       // movi v16.4s, #0
	WORD $0x0cc57521       // ld1 {v1.4h}, [x9], x5
	WORD $0x0f40a030       // smull v16.4s, v1.4h, v0.h[0]
	WORD $0x0cc57521       // ld1 {v1.4h}, [x9], x5
	WORD $0x0f502030       // smlal v16.4s, v1.4h, v0.h[1]
	WORD $0x0cc57521       // ld1 {v1.4h}, [x9], x5
	WORD $0x0f602030       // smlal v16.4s, v1.4h, v0.h[2]
	WORD $0x0cc57521       // ld1 {v1.4h}, [x9], x5
	WORD $0x0f702030       // smlal v16.4s, v1.4h, v0.h[3]
	WORD $0x0cc57521       // ld1 {v1.4h}, [x9], x5
	WORD $0x0f402830       // smlal v16.4s, v1.4h, v0.h[4]
	WORD $0x0cc57521       // ld1 {v1.4h}, [x9], x5
	WORD $0x0f502830       // smlal v16.4s, v1.4h, v0.h[5]
	WORD $0x0cc57521       // ld1 {v1.4h}, [x9], x5
	WORD $0x0f602830       // smlal v16.4s, v1.4h, v0.h[6]
	WORD $0x0cc57521       // ld1 {v1.4h}, [x9], x5
	WORD $0x0f702830       // smlal v16.4s, v1.4h, v0.h[7]

	WORD $0x4eb45610       // srshl v16.4s, v16.4s, v20.4s   round by FILTER_BITS
	WORD $0x4eb56610       // smax v16.4s, v16.4s, v21.4s    clip >= 0
	WORD $0x4eb66e10       // smin v16.4s, v16.4s, v22.4s    clip <= max
	WORD $0x0e612a10       // xtn v16.4h, v16.4s
	WORD $0x0c007550       // st1 {v16.4h}, [x10]

	ADD  $8, R10, R10      // dst += 4 samples (8 bytes)
	ADD  $8, R13, R13      // ref window += 4 columns
	SUB  $4, R8, R8
	CBNZ R8, yColLoop

	ADD  R4, R1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, yRowLoop

yDone:
	RET

// func convolveXHighBDNEONAsm(ctx *convolveHighBDNEONCtx)
//
// Horizontal 8-tap HBD convolve. res = round(sum, round0); out =
// clipHBD(round(res, bits)). round0 in field ROUND0, bits in field ROUND1.
TEXT ·convolveXHighBDNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD DST(R0), R1
	MOVD REF(R0), R2
	MOVD KERNEL(R0), R3
	MOVD DSTSTR(R0), R4
	MOVD REFSTR(R0), R5
	MOVD WIDTH(R0), R6
	MOVD HEIGHT(R0), R7
	MOVD ROUND0(R0), R11
	MOVD ROUND1(R0), R14
	MOVD MAXVAL(R0), R12

	WORD $0x4c407460       // ld1 {v0.8h}, [x3]  load 8 taps
	NEG  R11, R11
	WORD $0x4e040d74       // dup v20.4s, w11    -round0
	NEG  R14, R14
	MOVD R14, R11
	WORD $0x4e040d75       // dup v21.4s, w11    -bits
	WORD $0x4f000416       // movi v22.4s, #0    clip lower
	MOVD R12, R11
	WORD $0x4e040d77       // dup v23.4s, w11    clip upper = max

xRowLoop:
	CBZ  R7, xDone
	MOVD R1, R10
	MOVD R2, R9
	MOVD R6, R8

xColLoop:
	WORD $0x4c407521       // ld1 {v1.8h}, [x9]       s0..s7
	MOVD R9, R15
	ADD  $16, R15, R15
	WORD $0x4c4075e3       // ld1 {v3.8h}, [x15]      s8..s15
	WORD $0x0f40a030       // smull v16.4s, v1.4h, v0.h[0]
	WORD $0x6e031024       // ext v4.16b, v1.16b, v3.16b, #2
	WORD $0x0f502090       // smlal v16.4s, v4.4h, v0.h[1]
	WORD $0x6e032024       // ext v4.16b, v1.16b, v3.16b, #4
	WORD $0x0f602090       // smlal v16.4s, v4.4h, v0.h[2]
	WORD $0x6e033024       // ext v4.16b, v1.16b, v3.16b, #6
	WORD $0x0f702090       // smlal v16.4s, v4.4h, v0.h[3]
	WORD $0x6e034024       // ext v4.16b, v1.16b, v3.16b, #8
	WORD $0x0f402890       // smlal v16.4s, v4.4h, v0.h[4]
	WORD $0x6e035024       // ext v4.16b, v1.16b, v3.16b, #10
	WORD $0x0f502890       // smlal v16.4s, v4.4h, v0.h[5]
	WORD $0x6e036024       // ext v4.16b, v1.16b, v3.16b, #12
	WORD $0x0f602890       // smlal v16.4s, v4.4h, v0.h[6]
	WORD $0x6e037024       // ext v4.16b, v1.16b, v3.16b, #14
	WORD $0x0f702890       // smlal v16.4s, v4.4h, v0.h[7]

	WORD $0x4eb45610       // srshl v16.4s, v16.4s, v20.4s   round by round0
	WORD $0x4eb55610       // srshl v16.4s, v16.4s, v21.4s   round by bits
	WORD $0x4eb66610       // smax v16.4s, v16.4s, v22.4s    clip >= 0
	WORD $0x4eb76e10       // smin v16.4s, v16.4s, v23.4s    clip <= max
	WORD $0x0e612a10       // xtn v16.4h, v16.4s
	WORD $0x0c007550       // st1 {v16.4h}, [x10]

	ADD  $8, R10, R10      // dst += 4 samples
	ADD  $8, R9, R9        // ref += 4 samples
	SUB  $4, R8, R8
	CBNZ R8, xColLoop

	ADD  R4, R1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, xRowLoop

xDone:
	RET

// func convolve2DHighBDNEONAsm(ctx *convolveHighBDNEONCtx)
//
// Both-axes-fractional HBD 2D convolve.
//   Pass1 (horizontal): im[int32] = round(xBias + sum_x, round0).
//   Pass2 (vertical):   res = round(yBias + sum_y, round1) - roundOffset;
//                       out = clipHBD(round(res, bits)).
// Pass2 MACs int32 intermediates by int16 taps; products fit int32 so a 32-bit
// MLA with each tap broadcast (sign-extended) is exact.
TEXT ·convolve2DHighBDNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD DST(R0), R1
	MOVD REF(R0), R2
	MOVD KERNEL(R0), R3    // vertical kernel
	MOVD XKERN(R0), R16    // horizontal kernel
	MOVD DSTSTR(R0), R4
	MOVD REFSTR(R0), R5
	MOVD WIDTH(R0), R6
	MOVD HEIGHT(R0), R7
	MOVD IM(R0), R17
	MOVD IMSTR(R0), R24
	LSL  $2, R24, R24      // im row stride in bytes (4 bytes / int32)

	ADD  $7, R7, R19       // imH = height + 7

	// Horizontal pass setup.
	MOVD R16, R20
	WORD $0x4c407680       // ld1 {v0.8h}, [x20]  load 8 horizontal taps
	MOVD XBIAS(R0), R11
	WORD $0x4e040d72       // dup v18.4s, w11     xBias
	MOVD ROUND0(R0), R11
	NEG  R11, R11
	WORD $0x4e040d74       // dup v20.4s, w11     -round0

	MOVD R2, R21           // ref row cursor
	MOVD R17, R22          // im row cursor (bytes)

h2RowLoop:
	CBZ  R19, h2Done
	MOVD R21, R9           // ref column cursor
	MOVD R22, R10          // im column cursor
	MOVD R6, R8            // remaining columns

h2ColLoop:
	WORD $0x4c407521       // ld1 {v1.8h}, [x9]       s0..s7
	MOVD R9, R15
	ADD  $16, R15, R15
	WORD $0x4c4075e3       // ld1 {v3.8h}, [x15]      s8..s15
	WORD $0x4eb21e50       // mov v16.16b, v18.16b     acc = xBias
	WORD $0x0f402030       // smlal v16.4s, v1.4h, v0.h[0]
	WORD $0x6e031024       // ext v4.16b, v1.16b, v3.16b, #2
	WORD $0x0f502090       // smlal v16.4s, v4.4h, v0.h[1]
	WORD $0x6e032024       // ext v4.16b, v1.16b, v3.16b, #4
	WORD $0x0f602090       // smlal v16.4s, v4.4h, v0.h[2]
	WORD $0x6e033024       // ext v4.16b, v1.16b, v3.16b, #6
	WORD $0x0f702090       // smlal v16.4s, v4.4h, v0.h[3]
	WORD $0x6e034024       // ext v4.16b, v1.16b, v3.16b, #8
	WORD $0x0f402890       // smlal v16.4s, v4.4h, v0.h[4]
	WORD $0x6e035024       // ext v4.16b, v1.16b, v3.16b, #10
	WORD $0x0f502890       // smlal v16.4s, v4.4h, v0.h[5]
	WORD $0x6e036024       // ext v4.16b, v1.16b, v3.16b, #12
	WORD $0x0f602890       // smlal v16.4s, v4.4h, v0.h[6]
	WORD $0x6e037024       // ext v4.16b, v1.16b, v3.16b, #14
	WORD $0x0f702890       // smlal v16.4s, v4.4h, v0.h[7]

	WORD $0x4eb45610       // srshl v16.4s, v16.4s, v20.4s   round by round0
	WORD $0x4c007950       // st1 {v16.4s}, [x10]            store 4 int32

	ADD  $8, R9, R9        // ref += 4 samples (8 bytes)
	ADD  $16, R10, R10     // im += 4 int32 (16 bytes)
	SUB  $4, R8, R8
	CBNZ R8, h2ColLoop

	ADD  R5, R21, R21      // ref row += refStride
	ADD  R24, R22, R22     // im row += imStride bytes
	SUB  $1, R19, R19
	CBNZ R19, h2RowLoop

h2Done:
	// Vertical pass setup.
	MOVD YBIAS(R0), R11
	WORD $0x4e040d72       // dup v18.4s, w11     yBias
	MOVD RNDOFF(R0), R11
	WORD $0x4e040d73       // dup v19.4s, w11     roundOffset
	MOVD ROUND1(R0), R11
	NEG  R11, R11
	WORD $0x4e040d74       // dup v20.4s, w11     -round1
	MOVD BITS(R0), R11
	NEG  R11, R11
	WORD $0x4e040d75       // dup v21.4s, w11     -bits
	WORD $0x4f000416       // movi v22.4s, #0     clip lower
	MOVD MAXVAL(R0), R11
	WORD $0x4e040d77       // dup v23.4s, w11     clip upper = max

	MOVD R17, R22          // im row-window base for output row 0 (bytes)

v2RowLoop:
	CBZ  R7, v2Done
	MOVD R1, R10           // dst column cursor
	MOVD R22, R23          // im row-window base for this output row
	MOVD R6, R8

v2ColLoop:
	MOVD R23, R9           // R9 walks the 8 tap rows; post-index by R24 bytes
	MOVD R3, R12           // vertical kernel ptr (re-read each column)
	WORD $0x4eb21e50       // mov v16.16b, v18.16b   acc = yBias

	WORD $0x4cd87921       // ld1 {v1.4s}, [x9], x18
	MOVH (R12), R13        // sign-extended tap0
	ADD  $2, R12, R12
	MOVD R13, R11
	WORD $0x4e040d7c       // dup v28.4s, w11
	WORD $0x4ebc9430       // mla v16.4s, v1.4s, v28.4s
	WORD $0x4cd87921       // ld1 {v1.4s}, [x9], x18
	MOVH (R12), R13
	ADD  $2, R12, R12
	MOVD R13, R11
	WORD $0x4e040d7c       // dup v28.4s, w11
	WORD $0x4ebc9430       // mla v16.4s, v1.4s, v28.4s
	WORD $0x4cd87921       // ld1 {v1.4s}, [x9], x18
	MOVH (R12), R13
	ADD  $2, R12, R12
	MOVD R13, R11
	WORD $0x4e040d7c       // dup v28.4s, w11
	WORD $0x4ebc9430       // mla v16.4s, v1.4s, v28.4s
	WORD $0x4cd87921       // ld1 {v1.4s}, [x9], x18
	MOVH (R12), R13
	ADD  $2, R12, R12
	MOVD R13, R11
	WORD $0x4e040d7c       // dup v28.4s, w11
	WORD $0x4ebc9430       // mla v16.4s, v1.4s, v28.4s
	WORD $0x4cd87921       // ld1 {v1.4s}, [x9], x18
	MOVH (R12), R13
	ADD  $2, R12, R12
	MOVD R13, R11
	WORD $0x4e040d7c       // dup v28.4s, w11
	WORD $0x4ebc9430       // mla v16.4s, v1.4s, v28.4s
	WORD $0x4cd87921       // ld1 {v1.4s}, [x9], x18
	MOVH (R12), R13
	ADD  $2, R12, R12
	MOVD R13, R11
	WORD $0x4e040d7c       // dup v28.4s, w11
	WORD $0x4ebc9430       // mla v16.4s, v1.4s, v28.4s
	WORD $0x4cd87921       // ld1 {v1.4s}, [x9], x18
	MOVH (R12), R13
	ADD  $2, R12, R12
	MOVD R13, R11
	WORD $0x4e040d7c       // dup v28.4s, w11
	WORD $0x4ebc9430       // mla v16.4s, v1.4s, v28.4s
	WORD $0x4cd87921       // ld1 {v1.4s}, [x9], x18
	MOVH (R12), R13
	MOVD R13, R11
	WORD $0x4e040d7c       // dup v28.4s, w11
	WORD $0x4ebc9430       // mla v16.4s, v1.4s, v28.4s

	WORD $0x4eb45610       // srshl v16.4s, v16.4s, v20.4s   round by round1
	WORD $0x6eb38610       // sub v16.4s, v16.4s, v19.4s     -= roundOffset
	WORD $0x4eb55610       // srshl v16.4s, v16.4s, v21.4s   round by bits
	WORD $0x4eb66610       // smax v16.4s, v16.4s, v22.4s    clip >= 0
	WORD $0x4eb76e10       // smin v16.4s, v16.4s, v23.4s    clip <= max
	WORD $0x0e612a10       // xtn v16.4h, v16.4s
	WORD $0x0c007550       // st1 {v16.4h}, [x10]

	ADD  $8, R10, R10      // dst += 4 samples
	ADD  $16, R23, R23     // im window += 4 columns (16 bytes)
	SUB  $4, R8, R8
	CBNZ R8, v2ColLoop

	ADD  R4, R1, R1        // dst row += dstStride
	ADD  R24, R22, R22     // im base row += imStride bytes
	SUB  $1, R7, R7
	CBNZ R7, v2RowLoop

v2Done:
	RET
