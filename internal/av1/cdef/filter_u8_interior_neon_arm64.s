// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// CDEF interior 8-bit-dst .16b kernels: the dav1d fully-edged 8bpc fast path
// (src/arm/64/cdef.S cdef_filter{8,4}_{pri,sec,pri_sec}_edged_8bpc_neon,
// handle_pixel_8) applied to goav1's per-block CDEF walk. dav1d branches to
// this path when edges == 0xf (every neighbour present, no CDEF_VERY_LARGE
// border), which lets it filter uint8 samples 16-at-a-time in .16b lanes,
// 2 rows/iter for w=8 and 4 rows/iter for w=4, with plain umin/umax min/max
// (no sentinel eor/xor dance). goav1's tap buffer is the uint16 CDEF input
// (libaom cdef_prepare_fb geometry) so each tap pair is loaded as two .8h rows
// and narrowed (xtn/xtn2) to a .16b of two/four rows; the walk only routes a
// block here when its whole tap footprint is sentinel-free (cdefUnitInteriorU8),
// so the narrow is lossless (all taps in 0..255) and the block is bit-identical
// to filterBlockU8PureGo (the goav1 scalar reference mirrored below).
//
// Per-tap math (dav1d handle_pixel_8, uint8 lanes):
//   clip      = uqsub(threshold, abs(p - x) >> shift)   (unsigned saturate)
//   sign      = cmhi(x, p)                               (0xff where p < x)
//   constrain = bsl(sign, -umin(abs, clip), umin(abs, clip))
//   sumA/sumB = mla(sum, constrain, tap)                (two 8-bit accumulators;
//               |sum per side| <= 4*15+2*15 + 2*4+2*4+1*4+1*4 = 114 < 128, and
//               the true sumA+sumB in [-229,227] is reconstructed with the
//               halving adds below, matching goav1's int accumulation exactly)
// Finalize (dav1d, bit-exact to goav1 x + ((8 + sum - (sum<0)) >> 4)):
//   sumA starts at -1 (0xff) and sumB at 0, so sumA+sumB = sum - 1;
//   srhadd = (sum-1+1)>>1 = sum>>1, shadd = (sum-1)>>1;
//   pick shadd where sum<0 else srhadd, then srshr #3 gives the int8 delta;
//   pri_sec clamps with the usqadd/umin/umax [min,max] clip (== goav1 clamp);
//   pri-only and sec-only add without clip (== goav1 byte() wrap; the interior
//   contract keeps the result in 0..255 so wrap and saturate agree).

#define C_DST 0
#define C_INPUT 8
#define C_DSTSTR 16
#define C_HEIGHT 24
#define C_PRI0 32
#define C_PRI1 40
#define C_SEC0 48
#define C_SEC1 56
#define C_SEC2 64
#define C_SEC3 72
#define C_PRITAP0 80
#define C_PRITAP1 88
#define C_SECTAP0 96
#define C_SECTAP1 104
#define C_PRISTR 112
#define C_SECSTR 120
#define C_PRISHIFT 128
#define C_SECSHIFT 136

// func cdefFilterBlock8InteriorU8NEON(ctx *filterBlockU8NEONCtx)
TEXT ·cdefFilterBlock8InteriorU8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_INPUT(R0), R2
	MOVD C_DSTSTR(R0), R3
	MOVD C_HEIGHT(R0), R4
	MOVD C_PRISTR(R0), R5
	WORD $0x4e010cb9 // dup v25.16b, w5  pri threshold
	MOVD C_PRISHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e010cb8 // dup v24.16b, w5  -pri shift
	MOVD C_PRITAP0(R0), R5
	WORD $0x4e010cbc // dup v28.16b, w5  pri tap0
	MOVD C_PRITAP1(R0), R5
	WORD $0x4e010cbd // dup v29.16b, w5  pri tap1
	MOVD C_SECSTR(R0), R5
	WORD $0x4e010cbb // dup v27.16b, w5  sec threshold
	MOVD C_SECSHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e010cba // dup v26.16b, w5  -sec shift
	MOVD C_SECTAP0(R0), R5
	WORD $0x4e010cbe // dup v30.16b, w5  sec tap0
	MOVD C_SECTAP1(R0), R5
	WORD $0x4e010cbf // dup v31.16b, w5  sec tap1
	MOVD C_PRI0(R0), R6
	LSL  $1, R6
	MOVD C_PRI1(R0), R7
	LSL  $1, R7
	MOVD C_SEC0(R0), R8
	LSL  $1, R8
	MOVD C_SEC1(R0), R9
	LSL  $1, R9
	MOVD C_SEC2(R0), R10
	LSL  $1, R10
	MOVD C_SEC3(R0), R11
	LSL  $1, R11

u8i_cdefFilterBlock8InteriorU8NEON_row:
	VLD1 (R2), [V7.H8]
	ADD  $288, R2, R16
	VLD1 (R16), [V8.H8]
	WORD $0x0e2128e0 // xtn v0.8b, v7.8h   px row0
	WORD $0x4e212900 // xtn2 v0.16b, v8.8h  px row1
	WORD $0x4f07e7e1 // movi v1.16b, #255  sumA = -1
	WORD $0x4f00e402 // movi v2.16b, #0   sumB = 0
	WORD $0x4ea01c03 // mov v3.16b, v0.16b  min = px
	WORD $0x4ea01c04 // mov v4.16b, v0.16b  max = px
	// pri tap pri0
	ADD  R6, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R6, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e256c63 // umin v3.16b, v3.16b, v5.16b  min
	WORD $0x6e256484 // umax v4.16b, v4.16b, v5.16b  max
	WORD $0x6e266c63 // umin v3.16b, v3.16b, v6.16b
	WORD $0x6e266484 // umax v4.16b, v4.16b, v6.16b
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e384611 // ushl v17.16b, v16.16b, v24.16b
	WORD $0x6e384695 // ushl v21.16b, v20.16b, v24.16b
	WORD $0x6e312f31 // uqsub v17.16b, v25.16b, v17.16b
	WORD $0x6e352f35 // uqsub v21.16b, v25.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3c9641 // mla v1.16b, v18.16b, v28.16b
	WORD $0x4e3c96c2 // mla v2.16b, v22.16b, v28.16b
	// pri tap pri1
	ADD  R7, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R7, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e256c63 // umin v3.16b, v3.16b, v5.16b  min
	WORD $0x6e256484 // umax v4.16b, v4.16b, v5.16b  max
	WORD $0x6e266c63 // umin v3.16b, v3.16b, v6.16b
	WORD $0x6e266484 // umax v4.16b, v4.16b, v6.16b
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e384611 // ushl v17.16b, v16.16b, v24.16b
	WORD $0x6e384695 // ushl v21.16b, v20.16b, v24.16b
	WORD $0x6e312f31 // uqsub v17.16b, v25.16b, v17.16b
	WORD $0x6e352f35 // uqsub v21.16b, v25.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3d9641 // mla v1.16b, v18.16b, v29.16b
	WORD $0x4e3d96c2 // mla v2.16b, v22.16b, v29.16b
	// sec tap sec0
	ADD  R8, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R8, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e256c63 // umin v3.16b, v3.16b, v5.16b  min
	WORD $0x6e256484 // umax v4.16b, v4.16b, v5.16b  max
	WORD $0x6e266c63 // umin v3.16b, v3.16b, v6.16b
	WORD $0x6e266484 // umax v4.16b, v4.16b, v6.16b
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3e9641 // mla v1.16b, v18.16b, v30.16b
	WORD $0x4e3e96c2 // mla v2.16b, v22.16b, v30.16b
	// sec tap sec1
	ADD  R9, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R9, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e256c63 // umin v3.16b, v3.16b, v5.16b  min
	WORD $0x6e256484 // umax v4.16b, v4.16b, v5.16b  max
	WORD $0x6e266c63 // umin v3.16b, v3.16b, v6.16b
	WORD $0x6e266484 // umax v4.16b, v4.16b, v6.16b
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3e9641 // mla v1.16b, v18.16b, v30.16b
	WORD $0x4e3e96c2 // mla v2.16b, v22.16b, v30.16b
	// sec tap sec2
	ADD  R10, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R10, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e256c63 // umin v3.16b, v3.16b, v5.16b  min
	WORD $0x6e256484 // umax v4.16b, v4.16b, v5.16b  max
	WORD $0x6e266c63 // umin v3.16b, v3.16b, v6.16b
	WORD $0x6e266484 // umax v4.16b, v4.16b, v6.16b
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3f9641 // mla v1.16b, v18.16b, v31.16b
	WORD $0x4e3f96c2 // mla v2.16b, v22.16b, v31.16b
	// sec tap sec3
	ADD  R11, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R11, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e256c63 // umin v3.16b, v3.16b, v5.16b  min
	WORD $0x6e256484 // umax v4.16b, v4.16b, v5.16b  max
	WORD $0x6e266c63 // umin v3.16b, v3.16b, v6.16b
	WORD $0x6e266484 // umax v4.16b, v4.16b, v6.16b
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3f9641 // mla v1.16b, v18.16b, v31.16b
	WORD $0x4e3f96c2 // mla v2.16b, v22.16b, v31.16b
	// finalize
	WORD $0x4e221425 // srhadd v5.16b, v1.16b, v2.16b  sum>>1
	WORD $0x4e220426 // shadd v6.16b, v1.16b, v2.16b   (sum-1)>>1
	WORD $0x4e20a8a7 // cmlt v7.16b, v5.16b, #0
	WORD $0x6e651cc7 // bsl v7.16b, v6.16b, v5.16b
	WORD $0x4f0d24e7 // srshr v7.16b, v7.16b, #3  int8 delta
	WORD $0x6e2038e0 // usqadd v0.16b, v7.16b
	WORD $0x6e246c00 // umin v0.16b, v0.16b, v4.16b
	WORD $0x6e236400 // umax v0.16b, v0.16b, v3.16b
	WORD $0x0d008420 // st1 {v0.d}[0], [x1]  row0
	ADD  R3, R1
	WORD $0x4d008420 // st1 {v0.d}[1], [x1]  row1
	ADD  R3, R1
	ADD  $576, R2  // 2*BStride bytes
	SUB  $2, R4
	CBNZ R4, u8i_cdefFilterBlock8InteriorU8NEON_row
	RET

// func cdefFilterBlock8PrimaryInteriorU8NEON(ctx *filterBlockU8NEONCtx)
TEXT ·cdefFilterBlock8PrimaryInteriorU8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_INPUT(R0), R2
	MOVD C_DSTSTR(R0), R3
	MOVD C_HEIGHT(R0), R4
	MOVD C_PRISTR(R0), R5
	WORD $0x4e010cb9 // dup v25.16b, w5  pri threshold
	MOVD C_PRISHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e010cb8 // dup v24.16b, w5  -pri shift
	MOVD C_PRITAP0(R0), R5
	WORD $0x4e010cbc // dup v28.16b, w5  pri tap0
	MOVD C_PRITAP1(R0), R5
	WORD $0x4e010cbd // dup v29.16b, w5  pri tap1
	MOVD C_PRI0(R0), R6
	LSL  $1, R6
	MOVD C_PRI1(R0), R7
	LSL  $1, R7

u8i_cdefFilterBlock8PrimaryInteriorU8NEON_row:
	VLD1 (R2), [V7.H8]
	ADD  $288, R2, R16
	VLD1 (R16), [V8.H8]
	WORD $0x0e2128e0 // xtn v0.8b, v7.8h   px row0
	WORD $0x4e212900 // xtn2 v0.16b, v8.8h  px row1
	WORD $0x4f07e7e1 // movi v1.16b, #255  sumA = -1
	WORD $0x4f00e402 // movi v2.16b, #0   sumB = 0
	// pri tap pri0
	ADD  R6, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R6, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e384611 // ushl v17.16b, v16.16b, v24.16b
	WORD $0x6e384695 // ushl v21.16b, v20.16b, v24.16b
	WORD $0x6e312f31 // uqsub v17.16b, v25.16b, v17.16b
	WORD $0x6e352f35 // uqsub v21.16b, v25.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3c9641 // mla v1.16b, v18.16b, v28.16b
	WORD $0x4e3c96c2 // mla v2.16b, v22.16b, v28.16b
	// pri tap pri1
	ADD  R7, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R7, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e384611 // ushl v17.16b, v16.16b, v24.16b
	WORD $0x6e384695 // ushl v21.16b, v20.16b, v24.16b
	WORD $0x6e312f31 // uqsub v17.16b, v25.16b, v17.16b
	WORD $0x6e352f35 // uqsub v21.16b, v25.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3d9641 // mla v1.16b, v18.16b, v29.16b
	WORD $0x4e3d96c2 // mla v2.16b, v22.16b, v29.16b
	// finalize
	WORD $0x4e221425 // srhadd v5.16b, v1.16b, v2.16b  sum>>1
	WORD $0x4e220426 // shadd v6.16b, v1.16b, v2.16b   (sum-1)>>1
	WORD $0x4e20a8a7 // cmlt v7.16b, v5.16b, #0
	WORD $0x6e651cc7 // bsl v7.16b, v6.16b, v5.16b
	WORD $0x4f0d24e7 // srshr v7.16b, v7.16b, #3  int8 delta
	WORD $0x4e278400 // add v0.16b, v0.16b, v7.16b  wrap
	WORD $0x0d008420 // st1 {v0.d}[0], [x1]  row0
	ADD  R3, R1
	WORD $0x4d008420 // st1 {v0.d}[1], [x1]  row1
	ADD  R3, R1
	ADD  $576, R2  // 2*BStride bytes
	SUB  $2, R4
	CBNZ R4, u8i_cdefFilterBlock8PrimaryInteriorU8NEON_row
	RET

// func cdefFilterBlock8SecondaryInteriorU8NEON(ctx *filterBlockU8NEONCtx)
TEXT ·cdefFilterBlock8SecondaryInteriorU8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_INPUT(R0), R2
	MOVD C_DSTSTR(R0), R3
	MOVD C_HEIGHT(R0), R4
	MOVD C_SECSTR(R0), R5
	WORD $0x4e010cbb // dup v27.16b, w5  sec threshold
	MOVD C_SECSHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e010cba // dup v26.16b, w5  -sec shift
	MOVD C_SECTAP0(R0), R5
	WORD $0x4e010cbe // dup v30.16b, w5  sec tap0
	MOVD C_SECTAP1(R0), R5
	WORD $0x4e010cbf // dup v31.16b, w5  sec tap1
	MOVD C_SEC0(R0), R8
	LSL  $1, R8
	MOVD C_SEC1(R0), R9
	LSL  $1, R9
	MOVD C_SEC2(R0), R10
	LSL  $1, R10
	MOVD C_SEC3(R0), R11
	LSL  $1, R11

u8i_cdefFilterBlock8SecondaryInteriorU8NEON_row:
	VLD1 (R2), [V7.H8]
	ADD  $288, R2, R16
	VLD1 (R16), [V8.H8]
	WORD $0x0e2128e0 // xtn v0.8b, v7.8h   px row0
	WORD $0x4e212900 // xtn2 v0.16b, v8.8h  px row1
	WORD $0x4f07e7e1 // movi v1.16b, #255  sumA = -1
	WORD $0x4f00e402 // movi v2.16b, #0   sumB = 0
	// sec tap sec0
	ADD  R8, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R8, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3e9641 // mla v1.16b, v18.16b, v30.16b
	WORD $0x4e3e96c2 // mla v2.16b, v22.16b, v30.16b
	// sec tap sec1
	ADD  R9, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R9, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3e9641 // mla v1.16b, v18.16b, v30.16b
	WORD $0x4e3e96c2 // mla v2.16b, v22.16b, v30.16b
	// sec tap sec2
	ADD  R10, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R10, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3f9641 // mla v1.16b, v18.16b, v31.16b
	WORD $0x4e3f96c2 // mla v2.16b, v22.16b, v31.16b
	// sec tap sec3
	ADD  R11, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R11, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V7.H8]
	VLD1 (R17), [V8.H8]
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3f9641 // mla v1.16b, v18.16b, v31.16b
	WORD $0x4e3f96c2 // mla v2.16b, v22.16b, v31.16b
	// finalize
	WORD $0x4e221425 // srhadd v5.16b, v1.16b, v2.16b  sum>>1
	WORD $0x4e220426 // shadd v6.16b, v1.16b, v2.16b   (sum-1)>>1
	WORD $0x4e20a8a7 // cmlt v7.16b, v5.16b, #0
	WORD $0x6e651cc7 // bsl v7.16b, v6.16b, v5.16b
	WORD $0x4f0d24e7 // srshr v7.16b, v7.16b, #3  int8 delta
	WORD $0x4e278400 // add v0.16b, v0.16b, v7.16b  wrap
	WORD $0x0d008420 // st1 {v0.d}[0], [x1]  row0
	ADD  R3, R1
	WORD $0x4d008420 // st1 {v0.d}[1], [x1]  row1
	ADD  R3, R1
	ADD  $576, R2  // 2*BStride bytes
	SUB  $2, R4
	CBNZ R4, u8i_cdefFilterBlock8SecondaryInteriorU8NEON_row
	RET

// func cdefFilterBlock4InteriorU8NEON(ctx *filterBlockU8NEONCtx)
TEXT ·cdefFilterBlock4InteriorU8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_INPUT(R0), R2
	MOVD C_DSTSTR(R0), R3
	MOVD C_HEIGHT(R0), R4
	MOVD C_PRISTR(R0), R5
	WORD $0x4e010cb9 // dup v25.16b, w5  pri threshold
	MOVD C_PRISHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e010cb8 // dup v24.16b, w5  -pri shift
	MOVD C_PRITAP0(R0), R5
	WORD $0x4e010cbc // dup v28.16b, w5  pri tap0
	MOVD C_PRITAP1(R0), R5
	WORD $0x4e010cbd // dup v29.16b, w5  pri tap1
	MOVD C_SECSTR(R0), R5
	WORD $0x4e010cbb // dup v27.16b, w5  sec threshold
	MOVD C_SECSHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e010cba // dup v26.16b, w5  -sec shift
	MOVD C_SECTAP0(R0), R5
	WORD $0x4e010cbe // dup v30.16b, w5  sec tap0
	MOVD C_SECTAP1(R0), R5
	WORD $0x4e010cbf // dup v31.16b, w5  sec tap1
	MOVD C_PRI0(R0), R6
	LSL  $1, R6
	MOVD C_PRI1(R0), R7
	LSL  $1, R7
	MOVD C_SEC0(R0), R8
	LSL  $1, R8
	MOVD C_SEC1(R0), R9
	LSL  $1, R9
	MOVD C_SEC2(R0), R10
	LSL  $1, R10
	MOVD C_SEC3(R0), R11
	LSL  $1, R11

u8i_cdefFilterBlock4InteriorU8NEON_row:
	WORD $0x0c407447 // ld1 {v7.4h}, [x2]   px row0
	ADD  $288, R2, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  px row1
	ADD  $576, R2, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  px row2
	ADD  $864, R2, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  px row3
	WORD $0x0e2128e0 // xtn v0.8b, v7.8h
	WORD $0x4e212900 // xtn2 v0.16b, v8.8h
	WORD $0x4f07e7e1 // movi v1.16b, #255  sumA = -1
	WORD $0x4f00e402 // movi v2.16b, #0   sumB = 0
	WORD $0x4ea01c03 // mov v3.16b, v0.16b  min = px
	WORD $0x4ea01c04 // mov v4.16b, v0.16b  max = px
	// pri tap pri0
	ADD  R6, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s1 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s1 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s1 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s1 row3
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R6, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s2 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s2 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s2 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s2 row3
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e256c63 // umin v3.16b, v3.16b, v5.16b  min
	WORD $0x6e256484 // umax v4.16b, v4.16b, v5.16b  max
	WORD $0x6e266c63 // umin v3.16b, v3.16b, v6.16b
	WORD $0x6e266484 // umax v4.16b, v4.16b, v6.16b
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e384611 // ushl v17.16b, v16.16b, v24.16b
	WORD $0x6e384695 // ushl v21.16b, v20.16b, v24.16b
	WORD $0x6e312f31 // uqsub v17.16b, v25.16b, v17.16b
	WORD $0x6e352f35 // uqsub v21.16b, v25.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3c9641 // mla v1.16b, v18.16b, v28.16b
	WORD $0x4e3c96c2 // mla v2.16b, v22.16b, v28.16b
	// pri tap pri1
	ADD  R7, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s1 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s1 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s1 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s1 row3
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R7, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s2 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s2 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s2 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s2 row3
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e256c63 // umin v3.16b, v3.16b, v5.16b  min
	WORD $0x6e256484 // umax v4.16b, v4.16b, v5.16b  max
	WORD $0x6e266c63 // umin v3.16b, v3.16b, v6.16b
	WORD $0x6e266484 // umax v4.16b, v4.16b, v6.16b
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e384611 // ushl v17.16b, v16.16b, v24.16b
	WORD $0x6e384695 // ushl v21.16b, v20.16b, v24.16b
	WORD $0x6e312f31 // uqsub v17.16b, v25.16b, v17.16b
	WORD $0x6e352f35 // uqsub v21.16b, v25.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3d9641 // mla v1.16b, v18.16b, v29.16b
	WORD $0x4e3d96c2 // mla v2.16b, v22.16b, v29.16b
	// sec tap sec0
	ADD  R8, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s1 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s1 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s1 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s1 row3
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R8, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s2 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s2 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s2 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s2 row3
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e256c63 // umin v3.16b, v3.16b, v5.16b  min
	WORD $0x6e256484 // umax v4.16b, v4.16b, v5.16b  max
	WORD $0x6e266c63 // umin v3.16b, v3.16b, v6.16b
	WORD $0x6e266484 // umax v4.16b, v4.16b, v6.16b
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3e9641 // mla v1.16b, v18.16b, v30.16b
	WORD $0x4e3e96c2 // mla v2.16b, v22.16b, v30.16b
	// sec tap sec1
	ADD  R9, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s1 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s1 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s1 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s1 row3
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R9, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s2 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s2 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s2 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s2 row3
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e256c63 // umin v3.16b, v3.16b, v5.16b  min
	WORD $0x6e256484 // umax v4.16b, v4.16b, v5.16b  max
	WORD $0x6e266c63 // umin v3.16b, v3.16b, v6.16b
	WORD $0x6e266484 // umax v4.16b, v4.16b, v6.16b
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3e9641 // mla v1.16b, v18.16b, v30.16b
	WORD $0x4e3e96c2 // mla v2.16b, v22.16b, v30.16b
	// sec tap sec2
	ADD  R10, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s1 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s1 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s1 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s1 row3
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R10, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s2 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s2 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s2 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s2 row3
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e256c63 // umin v3.16b, v3.16b, v5.16b  min
	WORD $0x6e256484 // umax v4.16b, v4.16b, v5.16b  max
	WORD $0x6e266c63 // umin v3.16b, v3.16b, v6.16b
	WORD $0x6e266484 // umax v4.16b, v4.16b, v6.16b
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3f9641 // mla v1.16b, v18.16b, v31.16b
	WORD $0x4e3f96c2 // mla v2.16b, v22.16b, v31.16b
	// sec tap sec3
	ADD  R11, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s1 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s1 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s1 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s1 row3
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R11, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s2 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s2 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s2 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s2 row3
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e256c63 // umin v3.16b, v3.16b, v5.16b  min
	WORD $0x6e256484 // umax v4.16b, v4.16b, v5.16b  max
	WORD $0x6e266c63 // umin v3.16b, v3.16b, v6.16b
	WORD $0x6e266484 // umax v4.16b, v4.16b, v6.16b
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3f9641 // mla v1.16b, v18.16b, v31.16b
	WORD $0x4e3f96c2 // mla v2.16b, v22.16b, v31.16b
	// finalize
	WORD $0x4e221425 // srhadd v5.16b, v1.16b, v2.16b  sum>>1
	WORD $0x4e220426 // shadd v6.16b, v1.16b, v2.16b   (sum-1)>>1
	WORD $0x4e20a8a7 // cmlt v7.16b, v5.16b, #0
	WORD $0x6e651cc7 // bsl v7.16b, v6.16b, v5.16b
	WORD $0x4f0d24e7 // srshr v7.16b, v7.16b, #3  int8 delta
	WORD $0x6e2038e0 // usqadd v0.16b, v7.16b
	WORD $0x6e246c00 // umin v0.16b, v0.16b, v4.16b
	WORD $0x6e236400 // umax v0.16b, v0.16b, v3.16b
	WORD $0x0d008020 // st1 {v0.s}[0], [x1]  row0
	ADD  R3, R1
	WORD $0x0d009020 // st1 {v0.s}[1], [x1]  row1
	ADD  R3, R1
	WORD $0x4d008020 // st1 {v0.s}[2], [x1]  row2
	ADD  R3, R1
	WORD $0x4d009020 // st1 {v0.s}[3], [x1]  row3
	ADD  R3, R1
	ADD  $1152, R2  // 4*BStride bytes
	SUB  $4, R4
	CBNZ R4, u8i_cdefFilterBlock4InteriorU8NEON_row
	RET

// func cdefFilterBlock4PrimaryInteriorU8NEON(ctx *filterBlockU8NEONCtx)
TEXT ·cdefFilterBlock4PrimaryInteriorU8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_INPUT(R0), R2
	MOVD C_DSTSTR(R0), R3
	MOVD C_HEIGHT(R0), R4
	MOVD C_PRISTR(R0), R5
	WORD $0x4e010cb9 // dup v25.16b, w5  pri threshold
	MOVD C_PRISHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e010cb8 // dup v24.16b, w5  -pri shift
	MOVD C_PRITAP0(R0), R5
	WORD $0x4e010cbc // dup v28.16b, w5  pri tap0
	MOVD C_PRITAP1(R0), R5
	WORD $0x4e010cbd // dup v29.16b, w5  pri tap1
	MOVD C_PRI0(R0), R6
	LSL  $1, R6
	MOVD C_PRI1(R0), R7
	LSL  $1, R7

u8i_cdefFilterBlock4PrimaryInteriorU8NEON_row:
	WORD $0x0c407447 // ld1 {v7.4h}, [x2]   px row0
	ADD  $288, R2, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  px row1
	ADD  $576, R2, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  px row2
	ADD  $864, R2, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  px row3
	WORD $0x0e2128e0 // xtn v0.8b, v7.8h
	WORD $0x4e212900 // xtn2 v0.16b, v8.8h
	WORD $0x4f07e7e1 // movi v1.16b, #255  sumA = -1
	WORD $0x4f00e402 // movi v2.16b, #0   sumB = 0
	// pri tap pri0
	ADD  R6, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s1 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s1 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s1 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s1 row3
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R6, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s2 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s2 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s2 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s2 row3
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e384611 // ushl v17.16b, v16.16b, v24.16b
	WORD $0x6e384695 // ushl v21.16b, v20.16b, v24.16b
	WORD $0x6e312f31 // uqsub v17.16b, v25.16b, v17.16b
	WORD $0x6e352f35 // uqsub v21.16b, v25.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3c9641 // mla v1.16b, v18.16b, v28.16b
	WORD $0x4e3c96c2 // mla v2.16b, v22.16b, v28.16b
	// pri tap pri1
	ADD  R7, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s1 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s1 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s1 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s1 row3
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R7, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s2 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s2 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s2 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s2 row3
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e384611 // ushl v17.16b, v16.16b, v24.16b
	WORD $0x6e384695 // ushl v21.16b, v20.16b, v24.16b
	WORD $0x6e312f31 // uqsub v17.16b, v25.16b, v17.16b
	WORD $0x6e352f35 // uqsub v21.16b, v25.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3d9641 // mla v1.16b, v18.16b, v29.16b
	WORD $0x4e3d96c2 // mla v2.16b, v22.16b, v29.16b
	// finalize
	WORD $0x4e221425 // srhadd v5.16b, v1.16b, v2.16b  sum>>1
	WORD $0x4e220426 // shadd v6.16b, v1.16b, v2.16b   (sum-1)>>1
	WORD $0x4e20a8a7 // cmlt v7.16b, v5.16b, #0
	WORD $0x6e651cc7 // bsl v7.16b, v6.16b, v5.16b
	WORD $0x4f0d24e7 // srshr v7.16b, v7.16b, #3  int8 delta
	WORD $0x4e278400 // add v0.16b, v0.16b, v7.16b  wrap
	WORD $0x0d008020 // st1 {v0.s}[0], [x1]  row0
	ADD  R3, R1
	WORD $0x0d009020 // st1 {v0.s}[1], [x1]  row1
	ADD  R3, R1
	WORD $0x4d008020 // st1 {v0.s}[2], [x1]  row2
	ADD  R3, R1
	WORD $0x4d009020 // st1 {v0.s}[3], [x1]  row3
	ADD  R3, R1
	ADD  $1152, R2  // 4*BStride bytes
	SUB  $4, R4
	CBNZ R4, u8i_cdefFilterBlock4PrimaryInteriorU8NEON_row
	RET

// func cdefFilterBlock4SecondaryInteriorU8NEON(ctx *filterBlockU8NEONCtx)
TEXT ·cdefFilterBlock4SecondaryInteriorU8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_INPUT(R0), R2
	MOVD C_DSTSTR(R0), R3
	MOVD C_HEIGHT(R0), R4
	MOVD C_SECSTR(R0), R5
	WORD $0x4e010cbb // dup v27.16b, w5  sec threshold
	MOVD C_SECSHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e010cba // dup v26.16b, w5  -sec shift
	MOVD C_SECTAP0(R0), R5
	WORD $0x4e010cbe // dup v30.16b, w5  sec tap0
	MOVD C_SECTAP1(R0), R5
	WORD $0x4e010cbf // dup v31.16b, w5  sec tap1
	MOVD C_SEC0(R0), R8
	LSL  $1, R8
	MOVD C_SEC1(R0), R9
	LSL  $1, R9
	MOVD C_SEC2(R0), R10
	LSL  $1, R10
	MOVD C_SEC3(R0), R11
	LSL  $1, R11

u8i_cdefFilterBlock4SecondaryInteriorU8NEON_row:
	WORD $0x0c407447 // ld1 {v7.4h}, [x2]   px row0
	ADD  $288, R2, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  px row1
	ADD  $576, R2, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  px row2
	ADD  $864, R2, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  px row3
	WORD $0x0e2128e0 // xtn v0.8b, v7.8h
	WORD $0x4e212900 // xtn2 v0.16b, v8.8h
	WORD $0x4f07e7e1 // movi v1.16b, #255  sumA = -1
	WORD $0x4f00e402 // movi v2.16b, #0   sumB = 0
	// sec tap sec0
	ADD  R8, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s1 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s1 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s1 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s1 row3
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R8, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s2 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s2 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s2 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s2 row3
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3e9641 // mla v1.16b, v18.16b, v30.16b
	WORD $0x4e3e96c2 // mla v2.16b, v22.16b, v30.16b
	// sec tap sec1
	ADD  R9, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s1 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s1 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s1 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s1 row3
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R9, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s2 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s2 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s2 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s2 row3
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3e9641 // mla v1.16b, v18.16b, v30.16b
	WORD $0x4e3e96c2 // mla v2.16b, v22.16b, v30.16b
	// sec tap sec2
	ADD  R10, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s1 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s1 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s1 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s1 row3
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R10, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s2 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s2 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s2 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s2 row3
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3f9641 // mla v1.16b, v18.16b, v31.16b
	WORD $0x4e3f96c2 // mla v2.16b, v22.16b, v31.16b
	// sec tap sec3
	ADD  R11, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s1 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s1 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s1 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s1 row3
	WORD $0x0e2128e5 // xtn v5.8b, v7.8h
	WORD $0x4e212905 // xtn2 v5.16b, v8.8h
	SUB  R11, R2, R16
	WORD $0x0c407607 // ld1 {v7.4h}, [x16]  s2 row0
	ADD  $288, R16
	WORD $0x4d408607 // ld1 {v7.d}[1], [x16]  s2 row1
	ADD  $288, R16
	WORD $0x0c407608 // ld1 {v8.4h}, [x16]  s2 row2
	ADD  $288, R16
	WORD $0x4d408608 // ld1 {v8.d}[1], [x16]  s2 row3
	WORD $0x0e2128e6 // xtn v6.8b, v7.8h
	WORD $0x4e212906 // xtn2 v6.16b, v8.8h
	WORD $0x6e257410 // uabd v16.16b, v0.16b, v5.16b
	WORD $0x6e267414 // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611 // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695 // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71 // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75 // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e253412 // cmhi v18.16b, v0.16b, v5.16b
	WORD $0x6e263416 // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31 // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5 // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30 // neg v16.16b, v17.16b
	WORD $0x6e20bab4 // neg v20.16b, v21.16b
	WORD $0x6e711e12 // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96 // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e3f9641 // mla v1.16b, v18.16b, v31.16b
	WORD $0x4e3f96c2 // mla v2.16b, v22.16b, v31.16b
	// finalize
	WORD $0x4e221425 // srhadd v5.16b, v1.16b, v2.16b  sum>>1
	WORD $0x4e220426 // shadd v6.16b, v1.16b, v2.16b   (sum-1)>>1
	WORD $0x4e20a8a7 // cmlt v7.16b, v5.16b, #0
	WORD $0x6e651cc7 // bsl v7.16b, v6.16b, v5.16b
	WORD $0x4f0d24e7 // srshr v7.16b, v7.16b, #3  int8 delta
	WORD $0x4e278400 // add v0.16b, v0.16b, v7.16b  wrap
	WORD $0x0d008020 // st1 {v0.s}[0], [x1]  row0
	ADD  R3, R1
	WORD $0x0d009020 // st1 {v0.s}[1], [x1]  row1
	ADD  R3, R1
	WORD $0x4d008020 // st1 {v0.s}[2], [x1]  row2
	ADD  R3, R1
	WORD $0x4d009020 // st1 {v0.s}[3], [x1]  row3
	ADD  R3, R1
	ADD  $1152, R2  // 4*BStride bytes
	SUB  $4, R4
	CBNZ R4, u8i_cdefFilterBlock4SecondaryInteriorU8NEON_row
	RET
