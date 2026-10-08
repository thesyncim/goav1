// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// NEON 8-bit warped-motion prediction. These reproduce the pure-Go
// warpHorizontal8Resident and warpVertical8Full(Gamma0) in warp.go bit-for-bit.
// The SIMD instructions the Go assembler does not name are emitted via WORD,
// each encoding cross-checked against the system assembler. Working vectors are
// V0-V7 and V16-V23 only, so the callee-saved V8-V15 are never touched.
//
// Horizontal (warpHorizontalU8NEONAsm) mirrors dav1d's warp_filter_horz shape
// (src/arm/64/mc.S): 16 source bytes per row are XOR-128 centred so an int8
// SMULL against the int8 warp coefficients keeps the per-column accumulator
// inside int16 for the ADDP reduction tree. That centring shifts the raw dot
// product Sum(pix*coeff) by -128*Sum(coeff). Every warpedFilter row sums to 128
// and 128*128 == 1<<offsetBitsHoriz (bd=8), so after SRSHR #reduceBitsHoriz the
// dav1d-style result is exactly goav1's roundPowerOfTwo(Sum(pix*coeff) +
// (1<<offsetBitsHoriz), reduceBitsHoriz) minus a constant
// (1<<(offsetBitsHoriz+1)) >> reduceBitsHoriz == 4096. Re-adding 4096 before the
// int32 widen restores goav1's tmp exactly. The 16-byte row load overshoots the
// scalar's 15-sample window by one byte on the right, which is inside the
// resident guard's ix4+8 <= ref.Width and the reference plane's SB-aligned
// border (same contract as the u8 wiener kernel).
//
// Vertical (warpVerticalU8NEONAsm) mirrors dav1d's per-column int8 filter
// transpose (util.S transpose_8x8b_xtl) but multiplies goav1's int32 tmp
// (narrowed to int16 -- every tmp lies in [0, 1<<14)) so the products match the
// scalar coeff*tmp exactly. The accumulator is preloaded with seed =
// (1<<offsetBitsVert) - ((1<<7)+(1<<8)) << reduceBitsVert, which folds the
// vertical offset and the -((1<<7)+(1<<8)) clip bias through the SRSHR
// #reduceBitsVert; SQXTUN+UQXTN then apply clipPixel's [0,255] clamp.

// warpH8ResidentNEONCtx field offsets (see warp_neon_arm64.go).
#define H_SRC    0
#define H_TMP    8
#define H_FILTER 16
#define H_SRCSTR 24
#define H_MX0    32
#define H_ALPHA  36
#define H_BETA   40

// func warpHorizontalU8NEONAsm(ctx *warpH8ResidentNEONCtx)
TEXT ·warpHorizontalU8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD H_SRC(R0), R2      // src tap-window base (row k=-7, col ix4-7)
	MOVD H_TMP(R0), R1      // tmp output base
	MOVD H_FILTER(R0), R11  // &warpedFilterI8[64] (index-0 anchor)
	MOVD H_SRCSTR(R0), R3   // src stride (bytes)
	MOVW H_MX0(R0), R4      // mx: row-0 filter phase
	MOVW H_ALPHA(R0), R5    // alpha: per-column phase step
	MOVW H_BETA(R0), R6     // beta: per-row phase step

	WORD $0x4f04e416       // movi v22.16b, #0x80        (XOR-128 centring)
	WORD $0x4f00a617       // movi v23.8h, #0x10, lsl #8 (4096 bias re-add)
	MOVD $15, R7           // 15 intermediate rows

hRow:
	WORD $0x0cc3a050       // ld1 {v16.8b,v17.8b},[x2],x3  16 src, advance row
	WORD $0x2e361e10       // eor v16.8b, v16.8b, v22.8b   pix-128 (low 8)
	WORD $0x2e361e31       // eor v17.8b, v17.8b, v22.8b   pix-128 (high 8)

	ADD  $512, R4, R9      // wt = mx + 512 (roundPowerOfTwo bias)
	// Load the 8 per-column filters into d0..d7. Each: offs = wt>>10 relative
	// to the index-0 anchor at &warpedFilterI8[64]; advance wt by alpha.
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400100       // ldr d0, [x8]
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400101       // ldr d1, [x8]
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400102       // ldr d2, [x8]
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400103       // ldr d3, [x8]
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400104       // ldr d4, [x8]
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400105       // ldr d5, [x8]
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400106       // ldr d6, [x8]
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400107       // ldr d7, [x8]

	// Slide the 16-byte window by the column index and int8-SMULL each column
	// against its filter (v0..v7 hold filters, overwritten by products).
	WORD $0x2e110a12       // ext v18.8b, v16, v17, #1
	WORD $0x2e111213       // ext v19.8b, v16, v17, #2
	WORD $0x0e30c000       // smull v0.8h, v0.8b, v16.8b   col0
	WORD $0x0e32c021       // smull v1.8h, v1.8b, v18.8b   col1
	WORD $0x2e111a12       // ext v18.8b, v16, v17, #3
	WORD $0x2e112214       // ext v20.8b, v16, v17, #4
	WORD $0x0e33c042       // smull v2.8h, v2.8b, v19.8b   col2
	WORD $0x0e32c063       // smull v3.8h, v3.8b, v18.8b   col3
	WORD $0x2e112a12       // ext v18.8b, v16, v17, #5
	WORD $0x2e113213       // ext v19.8b, v16, v17, #6
	WORD $0x0e34c084       // smull v4.8h, v4.8b, v20.8b   col4
	WORD $0x0e32c0a5       // smull v5.8h, v5.8b, v18.8b   col5
	WORD $0x2e113a12       // ext v18.8b, v16, v17, #7
	WORD $0x0e33c0c6       // smull v6.8h, v6.8b, v19.8b   col6
	WORD $0x0e32c0e7       // smull v7.8h, v7.8b, v18.8b   col7

	// ADDP reduction: v0.8h lane c becomes column c's dot product.
	WORD $0x4e61bc00       // addp v0.8h, v0.8h, v1.8h
	WORD $0x4e63bc42       // addp v2.8h, v2.8h, v3.8h
	WORD $0x4e65bc84       // addp v4.8h, v4.8h, v5.8h
	WORD $0x4e67bcc6       // addp v6.8h, v6.8h, v7.8h
	WORD $0x4e62bc00       // addp v0.8h, v0.8h, v2.8h
	WORD $0x4e66bc84       // addp v4.8h, v4.8h, v6.8h
	WORD $0x4e64bc00       // addp v0.8h, v0.8h, v4.8h

	WORD $0x4f1d2400       // srshr v0.8h, v0.8h, #3       reduceBitsHoriz
	WORD $0x4e778400       // add v0.8h, v0.8h, v23.8h     + 4096 bias
	WORD $0x0f10a402       // sxtl v2.4s, v0.4h            cols0-3 -> int32
	WORD $0x4f10a403       // sxtl2 v3.4s, v0.8h           cols4-7 -> int32
	WORD $0x4c9fa822       // st1 {v2.4s,v3.4s},[x1],#32   store 8 int32

	ADD  R6, R4, R4        // mx += beta (next row phase)
	SUB  $1, R7, R7
	CBNZ R7, hRow
	RET

// warpV8FullNEONCtx field offsets (see warp_neon_arm64.go).
#define V_DST    0
#define V_TMP    8
#define V_FILTER 16
#define V_DSTSTR 24
#define V_MY0    32
#define V_GAMMA  36
#define V_DELTA  40
#define V_SEED   44

// func warpVerticalU8NEONAsm(ctx *warpV8FullNEONCtx)
TEXT ·warpVerticalU8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD V_DST(R0), R1      // dst output base (r=0,c=0)
	MOVD V_TMP(R0), R2      // tmp base (row 0)
	MOVD V_FILTER(R0), R11  // &warpedFilterI8[64]
	MOVD V_DSTSTR(R0), R3   // dst stride (bytes)
	MOVW V_MY0(R0), R4      // my: output-row-0 filter phase
	MOVW V_GAMMA(R0), R5    // gamma: per-column phase step
	MOVW V_DELTA(R0), R6    // delta: per-row phase step
	MOVW V_SEED(R0), R12    // accumulator seed
	MOVD R2, R10           // tmp window base for output row 0
	MOVD $8, R7            // 8 output rows

vRow:
	// Load the 8 per-column filters into d0..d7 (advance phase by gamma).
	ADD  $512, R4, R9      // wt = my + 512
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400100       // ldr d0, [x8]
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400101       // ldr d1, [x8]
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400102       // ldr d2, [x8]
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400103       // ldr d3, [x8]
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400104       // ldr d4, [x8]
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400105       // ldr d5, [x8]
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400106       // ldr d6, [x8]
	ASR  $10, R9, R13
	ADD  R5, R9, R9
	ADD  R13<<3, R11, R8
	WORD $0xfd400107       // ldr d7, [x8]

	// Transpose the 8x8 int8 filter matrix (columns down, taps across) and
	// sign-extend, leaving v0..v7 = tap-m coefficients across the 8 columns.
	WORD $0x4e013800       // zip1 v0.16b, v0.16b, v1.16b
	WORD $0x4e033842       // zip1 v2.16b, v2.16b, v3.16b
	WORD $0x4e053884       // zip1 v4.16b, v4.16b, v5.16b
	WORD $0x4e0738c6       // zip1 v6.16b, v6.16b, v7.16b
	WORD $0x4e422801       // trn1 v1.8h, v0.8h, v2.8h
	WORD $0x4e426803       // trn2 v3.8h, v0.8h, v2.8h
	WORD $0x4e462885       // trn1 v5.8h, v4.8h, v6.8h
	WORD $0x4e466887       // trn2 v7.8h, v4.8h, v6.8h
	WORD $0x4e852820       // trn1 v0.4s, v1.4s, v5.4s
	WORD $0x4e856822       // trn2 v2.4s, v1.4s, v5.4s
	WORD $0x4e872861       // trn1 v1.4s, v3.4s, v7.4s
	WORD $0x4e876863       // trn2 v3.4s, v3.4s, v7.4s
	WORD $0x4f08a404       // sxtl2 v4.8h, v0.16b
	WORD $0x0f08a400       // sxtl v0.8h, v0.8b
	WORD $0x4f08a446       // sxtl2 v6.8h, v2.16b
	WORD $0x0f08a442       // sxtl v2.8h, v2.8b
	WORD $0x4f08a425       // sxtl2 v5.8h, v1.16b
	WORD $0x0f08a421       // sxtl v1.8h, v1.8b
	WORD $0x4f08a467       // sxtl2 v7.8h, v3.16b
	WORD $0x0f08a463       // sxtl v3.8h, v3.8b

	WORD $0x4e040d90       // dup v16.4s, w12   seed (cols 0-3)
	WORD $0x4e040d91       // dup v17.4s, w12   seed (cols 4-7)

	MOVD R10, R9           // tmp window cursor for this output row
	WORD $0x4cdfa932       // ld1 {v18.4s,v19.4s},[x9],#32  tap row 0
	WORD $0x0e612a55       // xtn v21.4h, v18.4s
	WORD $0x4e612a75       // xtn2 v21.8h, v19.4s
	WORD $0x0e6082b0       // smlal v16.4s, v21.4h, v0.4h
	WORD $0x4e6082b1       // smlal2 v17.4s, v21.8h, v0.8h
	WORD $0x4cdfa932       // tap row 1
	WORD $0x0e612a55
	WORD $0x4e612a75
	WORD $0x0e6182b0       // smlal v16.4s, v21.4h, v1.4h
	WORD $0x4e6182b1       // smlal2 v17.4s, v21.8h, v1.8h
	WORD $0x4cdfa932       // tap row 2
	WORD $0x0e612a55
	WORD $0x4e612a75
	WORD $0x0e6282b0       // smlal ... v2
	WORD $0x4e6282b1
	WORD $0x4cdfa932       // tap row 3
	WORD $0x0e612a55
	WORD $0x4e612a75
	WORD $0x0e6382b0       // smlal ... v3
	WORD $0x4e6382b1
	WORD $0x4cdfa932       // tap row 4
	WORD $0x0e612a55
	WORD $0x4e612a75
	WORD $0x0e6482b0       // smlal ... v4
	WORD $0x4e6482b1
	WORD $0x4cdfa932       // tap row 5
	WORD $0x0e612a55
	WORD $0x4e612a75
	WORD $0x0e6582b0       // smlal ... v5
	WORD $0x4e6582b1
	WORD $0x4cdfa932       // tap row 6
	WORD $0x0e612a55
	WORD $0x4e612a75
	WORD $0x0e6682b0       // smlal ... v6
	WORD $0x4e6682b1
	WORD $0x4cdfa932       // tap row 7
	WORD $0x0e612a55
	WORD $0x4e612a75
	WORD $0x0e6782b0       // smlal ... v7
	WORD $0x4e6782b1

	WORD $0x4f352610       // srshr v16.4s, v16.4s, #11  reduceBitsVert
	WORD $0x4f352631       // srshr v17.4s, v17.4s, #11
	WORD $0x2e612a14       // sqxtun v20.4h, v16.4s      s32->u16, clamp lo 0
	WORD $0x6e612a34       // sqxtun2 v20.8h, v17.4s
	WORD $0x2e214a94       // uqxtn v20.8b, v20.8h        u16->u8, clamp hi 255
	WORD $0x0c007034       // st1 {v20.8b},[x1]           store 8 output px

	ADD  R3, R1, R1        // dst += stride
	ADD  $32, R10, R10     // tmp window base += 1 row
	ADD  R6, R4, R4        // my += delta
	SUB  $1, R7, R7
	CBNZ R7, vRow
	RET
