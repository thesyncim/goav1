// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// NEON final self-guided projection for 8-bit pixels (dav1d sgr_weighted2
// 8bpc role, libaom arithmetic — see selfguided_u8_neon_arm64.go). SIMD
// instructions the Go assembler does not name are emitted via WORD with
// encodings cross-checked against the system assembler. Working vectors are
// V1-V7, V16, V20, V21 only, so the callee-saved V8-V15 are never touched.
//
// Encoding helpers (new on top of selfguided_neon_arm64.s):
//   ld1   {Vt.8b}, [Xn]            0x0c407000 | (n<<5) | t
//   uxtl  Vd.8h, Vn.8b             0x2f08a400 | (n<<5) | d
//   ushll Vd.4s, Vn.4h, #4         0x2f14a400 | (n<<5) | d
//   ushll2 Vd.4s, Vn.8h, #4        0x6f14a400 | (n<<5) | d
//   shl   Vd.4s, Vn.4s, #7         0x4f275400 | (n<<5) | d
//   sub   Vd.4s, Vn.4s, Vm.4s      0x6ea08400 | (m<<16) | (n<<5) | d
//   srshr Vd.4s, Vn.4s, #11        0x4f352400 | (n<<5) | d
//   xtn   Vd.4h, Vn.4s             0x0e612800 | (n<<5) | d
//   xtn2  Vd.8h, Vn.4s             0x4e612800 | (n<<5) | d
//   sqxtun Vd.8b, Vn.8h            0x2e212800 | (n<<5) | d
//   st1   {Vt.8b}, [Xn]            0x0c007000 | (n<<5) | t

// sgrWeightedU8Ctx field offsets (see selfguided_u8_neon_arm64.go).
#define W_DST  0
#define W_SRC  8
#define W_F0   16
#define W_F1   24
#define W_COLS 32
#define W_XQ0  40
#define W_XQ1  44

// func sgrWeightedRowU8NEONAsm(ctx *sgrWeightedU8Ctx)
//
// Projects one output row, 8 columns per iteration (two 4-lane s32 halves).
TEXT ·sgrWeightedRowU8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD W_DST(R0), R1        // dst cursor
	MOVD W_SRC(R0), R2        // src cursor
	MOVD W_F0(R0), R3         // flt0 cursor
	MOVD W_F1(R0), R4         // flt1 cursor
	MOVD W_COLS(R0), R5       // remaining columns (multiple of 8)
	MOVW W_XQ0(R0), R11
	MOVW W_XQ1(R0), R12
	WORD $0x4e040d74          // dup v20.4s, w11   xq0 broadcast
	WORD $0x4e040d95          // dup v21.4s, w12   xq1 broadcast

wCol:
	CBZ  R5, wDone
	WORD $0x0c407041          // ld1 {v1.8b}, [x2]      8 source u8
	WORD $0x2f08a421          // uxtl v1.8h, v1.8b      -> u16 lanes
	WORD $0x2f14a422          // ushll  v2.4s, v1.4h, #4   u lanes 0..3
	WORD $0x6f14a423          // ushll2 v3.4s, v1.8h, #4   u lanes 4..7
	WORD $0x4c40a864          // ld1 {v4.4s, v5.4s}, [x3]  f0
	WORD $0x4c40a886          // ld1 {v6.4s, v7.4s}, [x4]  f1
	WORD $0x6ea28484          // sub v4.4s, v4.4s, v2.4s   f0-u lo
	WORD $0x6ea384a5          // sub v5.4s, v5.4s, v3.4s   f0-u hi
	WORD $0x6ea284c6          // sub v6.4s, v6.4s, v2.4s   f1-u lo
	WORD $0x6ea384e7          // sub v7.4s, v7.4s, v3.4s   f1-u hi
	WORD $0x4f275442          // shl v2.4s, v2.4s, #7      v = u<<7 lo
	WORD $0x4f275463          // shl v3.4s, v3.4s, #7      v = u<<7 hi
	WORD $0x4eb49482          // mla v2.4s, v4.4s, v20.4s  v += xq0*(f0-u) lo
	WORD $0x4eb594c2          // mla v2.4s, v6.4s, v21.4s  v += xq1*(f1-u) lo
	WORD $0x4eb494a3          // mla v3.4s, v5.4s, v20.4s  v += xq0*(f0-u) hi
	WORD $0x4eb594e3          // mla v3.4s, v7.4s, v21.4s  v += xq1*(f1-u) hi
	WORD $0x4f352442          // srshr v2.4s, v2.4s, #11   roundPowerOfTwo lo
	WORD $0x4f352463          // srshr v3.4s, v3.4s, #11   roundPowerOfTwo hi
	WORD $0x0e612850          // xtn  v16.4h, v2.4s        int16 wrap lo
	WORD $0x4e612870          // xtn2 v16.8h, v3.4s        int16 wrap hi
	WORD $0x2e212a10          // sqxtun v16.8b, v16.8h     clamp [0,255] -> u8
	WORD $0x0c007030          // st1 {v16.8b}, [x1]

	ADD  $8, R1, R1           // dst  += 8 u8
	ADD  $8, R2, R2           // src  += 8 u8
	ADD  $32, R3, R3          // flt0 += 8 s32
	ADD  $32, R4, R4          // flt1 += 8 s32
	SUB  $8, R5, R5
	B    wCol

wDone:
	RET
