// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// I8MM-tier resident lowbd compound-X convolve. This mirrors SVT's
// dist_wtd_convolve_x_8tap_neon_i8mm do_average == 0 branch: halve the even
// AV1 filter taps, stagger f1..f7 for USMMLA, subtract tap 0 separately, add
// the CONV_BUF round shim, then shift by ROUND0_BITS-1.

#define I8_DST       0
#define I8_REF       8
#define I8_FILTER    16
#define I8_PERMUTE   24
#define I8_REFSTR    32
#define I8_WIDTH     40
#define I8_HEIGHT    48
#define I8_ROUNDSHIM 56
#define I8_F0        64

#define XSR_DST      0
#define XSR_REF      8
#define XSR_FILTER   16
#define XSR_PERMUTE  24
#define XSR_DSTSTR   32
#define XSR_REFSTR   40
#define XSR_WIDTH    48
#define XSR_HEIGHT   56
#define XSR_F0       64

#define T2D_DST      0
#define T2D_REF      8
#define T2D_XFILTER  16
#define T2D_PERMUTE  24
#define T2D_YKERNEL  32
#define T2D_DSTSTR   40
#define T2D_REFSTR   48
#define T2D_WIDTH    56
#define T2D_HEIGHT   64
#define T2D_IM       72
#define T2D_IMSTR    80
#define T2D_F0       88

// func compoundX8I8MMAsm(ctx *compoundX8I8MMCtx)
TEXT ·compoundX8I8MMAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD I8_DST(R0), R1
	MOVD I8_REF(R0), R2
	MOVD I8_FILTER(R0), R3
	MOVD I8_PERMUTE(R0), R12
	MOVD I8_REFSTR(R0), R5
	MOVD I8_WIDTH(R0), R6
	MOVD I8_HEIGHT(R0), R7
	MOVD I8_ROUNDSHIM(R0), R11
	MOVD I8_F0(R0), R14

	VLD1   (R3), [V0.B16]
	VLD1.P 16(R12), [V28.B16]
	VLD1   (R12), [V29.B16]
	WORD $0x4e020d79 // dup v25.8h, w11     round shim
	WORD $0x0e010dda // dup v26.8b, w14     -tap0 >> 1

i8XRowLoop:
	CBZ  R7, i8XDone
	MOVD R1, R10 // dst cursor
	MOVD R2, R9  // ref tap-window cursor
	MOVD R6, R8  // remaining columns

i8XColLoop:
	VLD1 (R9), [V1.B16]
	WORD $0x4e1c0025 // tbl v5.16b, {v1.16b}, v28.16b
	WORD $0x4e1d0026 // tbl v6.16b, {v1.16b}, v29.16b
	WORD $0x4f000410 // movi v16.4s, #0
	WORD $0x4f000411 // movi v17.4s, #0
	WORD $0x4e80acb0 // usmmla v16.4s, v5.16b, v0.16b
	WORD $0x4e80acd1 // usmmla v17.4s, v6.16b, v0.16b
	WORD $0x0e612a10 // xtn  v16.4h, v16.4s
	WORD $0x4e612a30 // xtn2 v16.8h, v17.4s
	WORD $0x2e3aa030 // umlsl v16.8h, v1.8b, v26.8b
	WORD $0x4e798610 // add  v16.8h, v16.8h, v25.8h
	WORD $0x6f1e0610 // ushr v16.8h, v16.8h, #2
	WORD $0x4c007550 // st1  {v16.8h}, [x10]

	ADD  $16, R10, R10
	ADD  $8, R9, R9
	SUB  $8, R8, R8
	CBNZ R8, i8XColLoop

	ADD  R6<<1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, i8XRowLoop

i8XDone:
	RET

// func convolveX8I8MMAsm(ctx *convolveX8I8MMCtx)
TEXT ·convolveX8I8MMAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD XSR_DST(R0), R1
	MOVD XSR_REF(R0), R2
	MOVD XSR_FILTER(R0), R3
	MOVD XSR_PERMUTE(R0), R12
	MOVD XSR_DSTSTR(R0), R4
	MOVD XSR_REFSTR(R0), R5
	MOVD XSR_WIDTH(R0), R6
	MOVD XSR_HEIGHT(R0), R7
	MOVD XSR_F0(R0), R14

	VLD1   (R3), [V0.B16]
	VLD1.P 16(R12), [V28.B16]
	VLD1   (R12), [V29.B16]
	WORD $0x4f008459 // movi v25.8h, #2      ((1 << (ROUND0_BITS-1)) / 2)
	WORD $0x0e010dda // dup  v26.8b, w14     -tap0 >> 1

xsrI8RowLoop:
	CBZ  R7, xsrI8Done
	MOVD R1, R10 // dst cursor
	MOVD R2, R9  // ref tap-window cursor
	MOVD R6, R8  // remaining columns

xsrI8ColLoop:
	VLD1 (R9), [V1.B16]
	WORD $0x4e1c0025 // tbl v5.16b, {v1.16b}, v28.16b
	WORD $0x4e1d0026 // tbl v6.16b, {v1.16b}, v29.16b
	WORD $0x4f000410 // movi v16.4s, #0
	WORD $0x4f000411 // movi v17.4s, #0
	WORD $0x4e80acb0 // usmmla v16.4s, v5.16b, v0.16b
	WORD $0x4e80acd1 // usmmla v17.4s, v6.16b, v0.16b
	WORD $0x0e612a10 // xtn  v16.4h, v16.4s
	WORD $0x4e612a30 // xtn2 v16.8h, v17.4s
	WORD $0x2e3aa030 // umlsl v16.8h, v1.8b, v26.8b
	WORD $0x4e798610 // add  v16.8h, v16.8h, v25.8h
	WORD $0x2f0a8e10 // sqrshrun v16.8b, v16.8h, #6
	WORD $0x0c007150 // st1  {v16.8b}, [x10]

	ADD  $8, R10, R10
	ADD  $8, R9, R9
	SUB  $8, R8, R8
	CBNZ R8, xsrI8ColLoop

	ADD  R4, R1, R1
	ADD  R5, R2, R2
	SUB  $1, R7, R7
	CBNZ R7, xsrI8RowLoop

xsrI8Done:
	RET

// func convolve2D8I8MMAsm(ctx *convolve2D8I8MMCtx)
//
// Resident width>=8 lowbd 2D convolve. The horizontal pass mirrors SVT's
// convolve_2d_sr_horiz_8tap_neon_i8mm: halve the even AV1 taps, stagger f1..f7
// for USMMLA, subtract tap 0 separately, add the ROUND0 shim, then shift by
// ROUND0_BITS-1 into the int16 intermediate. The vertical pass is the same
// proven NEON int16->uint8 path used by convolve2D8NEONAsm.
TEXT ·convolve2D8I8MMAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD T2D_DST(R0), R1      // dst row base
	MOVD T2D_REF(R0), R2      // ref tap-window base (row refY-foY, col refX-foX)
	MOVD T2D_XFILTER(R0), R3  // packed horizontal f1..f7 filter
	MOVD T2D_PERMUTE(R0), R12 // SVT kMatMul8 permute table
	MOVD T2D_DSTSTR(R0), R4   // dst stride (bytes)
	MOVD T2D_REFSTR(R0), R5   // ref stride (bytes)
	MOVD T2D_WIDTH(R0), R6    // width
	MOVD T2D_HEIGHT(R0), R7   // height
	MOVD T2D_IM(R0), R13      // im base ptr
	MOVD T2D_IMSTR(R0), R14   // im row stride (int16 elements)
	MOVD T2D_F0(R0), R15      // -tap0 >> 1
	LSL  $1, R14, R14         // im row stride in bytes

	// imH = height + filterTaps - 1 = height + 7
	ADD  $7, R7, R11

	// Horizontal pass setup.
	VLD1   (R3), [V0.B16]
	VLD1.P 16(R12), [V28.B16]
	VLD1   (R12), [V29.B16]
	MOVD   $8194, R3        // (1 << (8+FILTER_BITS-2)) + (1 << ((ROUND0_BITS-1)-1))
	WORD   $0x4e020c79     // dup v25.8h, w3
	WORD   $0x0e010dfa     // dup v26.8b, w15

	MOVD R2, R16           // ref row cursor
	MOVD R13, R17          // im row cursor

t2dHRowLoop:
	CBZ  R11, t2dHDone
	MOVD R16, R9           // ref column cursor
	MOVD R17, R10          // im column cursor
	MOVD R6, R8            // remaining columns

t2dHColLoop:
	VLD1 (R9), [V1.B16]
	WORD $0x4e1c0025       // tbl v5.16b, {v1.16b}, v28.16b
	WORD $0x4e1d0026       // tbl v6.16b, {v1.16b}, v29.16b
	WORD $0x4f000410       // movi v16.4s, #0
	WORD $0x4f000411       // movi v17.4s, #0
	WORD $0x4e80acb0       // usmmla v16.4s, v5.16b, v0.16b
	WORD $0x4e80acd1       // usmmla v17.4s, v6.16b, v0.16b
	WORD $0x0e612a10       // xtn  v16.4h, v16.4s
	WORD $0x4e612a30       // xtn2 v16.8h, v17.4s
	WORD $0x2e3aa030       // umlsl v16.8h, v1.8b, v26.8b
	WORD $0x4e798610       // add  v16.8h, v16.8h, v25.8h
	WORD $0x6f1e0610       // ushr v16.8h, v16.8h, #2
	WORD $0x4c007550       // st1  {v16.8h}, [x10]

	ADD  $8, R9, R9
	ADD  $16, R10, R10
	SUB  $8, R8, R8
	CBNZ R8, t2dHColLoop

	ADD  R5, R16, R16
	ADD  R14, R17, R17
	SUB  $1, R11, R11
	CBNZ R11, t2dHRowLoop

t2dHDone:
	// Vertical pass setup.
	MOVD T2D_YKERNEL(R0), R3
	WORD $0x4c407460       // ld1 {v0.8h}, [x3]  load 8 vertical taps
	MOVD $524288, R11      // yBias = 1 << offsetBits (offsetBits = 19)
	WORD $0x4e040d72       // dup v18.4s, w11    yBias broadcast
	MOVD $384, R11         // roundOffset = (1<<8) + (1<<7)
	WORD $0x4e040d73       // dup v19.4s, w11    roundOffset broadcast

	MOVD R13, R17          // im row-window base for output row 0

t2dVRowLoop:
	CBZ  R7, t2dVDone
	MOVD R1, R10           // dst column cursor
	MOVD R17, R11          // im row-window base for this output row
	MOVD R6, R8            // remaining columns

t2dVColLoop:
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
	WORD $0x6eb38610       // sub v16.4s, v16.4s, v19.4s
	WORD $0x6eb38631       // sub v17.4s, v17.4s, v19.4s
	WORD $0x0e614a10       // sqxtn  v16.4h, v16.4s
	WORD $0x4e614a30       // sqxtn2 v16.8h, v17.4s
	WORD $0x2e212a10       // sqxtun v16.8b, v16.8h
	WORD $0x0c007150       // st1 {v16.8b}, [x10]

	ADD  $8, R10, R10
	ADD  $16, R11, R11
	SUB  $8, R8, R8
	CBNZ R8, t2dVColLoop

	ADD  R4, R1, R1
	ADD  R14, R17, R17
	SUB  $1, R7, R7
	CBNZ R7, t2dVRowLoop

t2dVDone:
	RET
