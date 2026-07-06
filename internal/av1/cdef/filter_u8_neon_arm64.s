// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// NEON 8-bit-dst CDEF pri+sec block filter: the uint16 kernels of
// filter_neon_arm64.s (dav1d cdef_filter{8,4}_pri_sec, src/arm/64/cdef_tmpl.S
// filter_func with pri=1, sec=1, min=1) with dav1d's 8bpc store epilogue
// (src/arm/64/cdef.S: xtn to bytes, st1 {v0.8b} / st1 {v0.s}[lane]). The tap
// loads still read the uint16 CDEF input buffer; dst is the frame's uint8
// plane and its stride arrives in bytes (no element shift). The narrowed
// store is lossless under the 8-bit contract in filter_u8.go. Field offsets
// mirror filterBlockU8NEONCtx.
//
// Per-tap math (dav1d handle_pixel):
//   clip      = uqsub(threshold, abs(p - x) >> shift)   (unsigned saturate)
//   constrain = smax(smin(p - x, clip), -clip)
//   sum       = mla(sum, constrain, tap)                (int16 lanes; the
//               worst case |sum| = 2*(4+2)*pri_strength + 4*(2+1)*sec_strength
//               is 3648 at 12-bit, so 16-bit accumulation never wraps)
// and the finalize is dav1d's cmlt/add/srshr #4 rounding, followed by the
// [min, max] clip.
//
// dav1d pads its tmp buffer with INT16_MIN so plain umin/smax skip padding
// lanes; goav1's input keeps libaom's 0x4000 VeryLarge sentinel, so the max
// fold runs in an xor domain instead: real samples s < 0x4000 map to
// s ^ 0x4000 = s + 0x4000 while sentinel lanes map to 0 and always lose;
// xoring the fold result by 0x4000 restores the true tap maximum, and px is
// folded in afterwards with umax (px itself may legally be the sentinel in
// border blocks, in which case it must win the max like the scalar
// reference; when every tap is a sentinel the restored fold is 0x4000, which
// is harmless because sum is then 0 and y == px >= min). The min fold uses
// plain umin exactly like dav1d (the sentinel is larger than every sample
// either way).

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

// func cdefFilterBlock8U8NEON(ctx *filterBlockU8NEONCtx)
// dav1d cdef_filter8_pri_sec_neon (src/arm/64/cdef_tmpl.S filter_func 8,
// pri=1, sec=1, min=1): one 8-wide row per iteration.
TEXT ·cdefFilterBlock8U8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_INPUT(R0), R2
	MOVD C_DSTSTR(R0), R3
	MOVD C_HEIGHT(R0), R4
	MOVD C_PRISTR(R0), R5
	WORD $0x4e020cb9 // dup v25.8h, w5
	MOVD C_SECSTR(R0), R5
	WORD $0x4e020cbb // dup v27.8h, w5
	MOVD C_PRISHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e020cb8 // dup v24.8h, w5
	MOVD C_SECSHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e020cba // dup v26.8h, w5
	MOVD C_PRITAP0(R0), R5
	WORD $0x4e020cbc // dup v28.8h, w5
	MOVD C_PRITAP1(R0), R5
	WORD $0x4e020cbd // dup v29.8h, w5
	MOVD C_SECTAP0(R0), R5
	WORD $0x4e020cbe // dup v30.8h, w5
	MOVD C_SECTAP1(R0), R5
	WORD $0x4e020cbf // dup v31.8h, w5
	MOVD $0x4000, R5 // VeryLarge sentinel
	WORD $0x4e020ca8 // dup v8.8h, w5
	// Tap byte offsets.
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

u8prisec8_row:
	VLD1 (R2), [V0.H8] // px
	WORD $0x6e211c21 // eor v1.16b, v1.16b, v1.16b
	VMOV V0.B16, V2.B16 // min = px
	WORD $0x6e231c63 // eor v3.16b, v3.16b, v3.16b

	// pri taps k=0 (dav1d handle_pixel v4, v5, pri thresh/shift, pri_taps[0], min=1)
	ADD  R6, R2, R16
	SUB  R6, R2, R17
	VLD1 (R16), [V4.H8]
	VLD1 (R17), [V5.H8]
	WORD $0x6e646c42 // umin v2.8h, v2.8h, v4.8h
	WORD $0x6e656c42 // umin v2.8h, v2.8h, v5.8h
	WORD $0x6e281c90 // eor v16.16b, v4.16b, v8.16b
	WORD $0x6e281cb4 // eor v20.16b, v5.16b, v8.16b
	WORD $0x6e706463 // umax v3.8h, v3.8h, v16.8h
	WORD $0x6e746463 // umax v3.8h, v3.8h, v20.8h
	WORD $0x6e647410 // uabd v16.8h, v0.8h, v4.8h
	WORD $0x6e657414 // uabd v20.8h, v0.8h, v5.8h
	WORD $0x6e784611 // ushl v17.8h, v16.8h, v24.8h
	WORD $0x6e784695 // ushl v21.8h, v20.8h, v24.8h
	WORD $0x6e712f31 // uqsub v17.8h, v25.8h, v17.8h
	WORD $0x6e752f35 // uqsub v21.8h, v25.8h, v21.8h
	WORD $0x6e608492 // sub v18.8h, v4.8h, v0.8h
	WORD $0x6e6084b6 // sub v22.8h, v5.8h, v0.8h
	WORD $0x6e60ba30 // neg v16.8h, v17.8h
	WORD $0x6e60bab4 // neg v20.8h, v21.8h
	WORD $0x4e716e52 // smin v18.8h, v18.8h, v17.8h
	WORD $0x4e756ed6 // smin v22.8h, v22.8h, v21.8h
	WORD $0x4e706652 // smax v18.8h, v18.8h, v16.8h
	WORD $0x4e7466d6 // smax v22.8h, v22.8h, v20.8h
	WORD $0x4e7c9641 // mla v1.8h, v18.8h, v28.8h
	WORD $0x4e7c96c1 // mla v1.8h, v22.8h, v28.8h

	// pri taps k=1
	ADD  R7, R2, R16
	SUB  R7, R2, R17
	VLD1 (R16), [V4.H8]
	VLD1 (R17), [V5.H8]
	WORD $0x6e646c42 // umin v2.8h, v2.8h, v4.8h
	WORD $0x6e656c42 // umin v2.8h, v2.8h, v5.8h
	WORD $0x6e281c90 // eor v16.16b, v4.16b, v8.16b
	WORD $0x6e281cb4 // eor v20.16b, v5.16b, v8.16b
	WORD $0x6e706463 // umax v3.8h, v3.8h, v16.8h
	WORD $0x6e746463 // umax v3.8h, v3.8h, v20.8h
	WORD $0x6e647410 // uabd v16.8h, v0.8h, v4.8h
	WORD $0x6e657414 // uabd v20.8h, v0.8h, v5.8h
	WORD $0x6e784611 // ushl v17.8h, v16.8h, v24.8h
	WORD $0x6e784695 // ushl v21.8h, v20.8h, v24.8h
	WORD $0x6e712f31 // uqsub v17.8h, v25.8h, v17.8h
	WORD $0x6e752f35 // uqsub v21.8h, v25.8h, v21.8h
	WORD $0x6e608492 // sub v18.8h, v4.8h, v0.8h
	WORD $0x6e6084b6 // sub v22.8h, v5.8h, v0.8h
	WORD $0x6e60ba30 // neg v16.8h, v17.8h
	WORD $0x6e60bab4 // neg v20.8h, v21.8h
	WORD $0x4e716e52 // smin v18.8h, v18.8h, v17.8h
	WORD $0x4e756ed6 // smin v22.8h, v22.8h, v21.8h
	WORD $0x4e706652 // smax v18.8h, v18.8h, v16.8h
	WORD $0x4e7466d6 // smax v22.8h, v22.8h, v20.8h
	WORD $0x4e7d9641 // mla v1.8h, v18.8h, v29.8h
	WORD $0x4e7d96c1 // mla v1.8h, v22.8h, v29.8h

	// sec taps k=0, off2 (dir + 2)
	ADD  R8, R2, R16
	SUB  R8, R2, R17
	VLD1 (R16), [V4.H8]
	VLD1 (R17), [V5.H8]
	WORD $0x6e646c42 // umin v2.8h, v2.8h, v4.8h
	WORD $0x6e656c42 // umin v2.8h, v2.8h, v5.8h
	WORD $0x6e281c90 // eor v16.16b, v4.16b, v8.16b
	WORD $0x6e281cb4 // eor v20.16b, v5.16b, v8.16b
	WORD $0x6e706463 // umax v3.8h, v3.8h, v16.8h
	WORD $0x6e746463 // umax v3.8h, v3.8h, v20.8h
	WORD $0x6e647410 // uabd v16.8h, v0.8h, v4.8h
	WORD $0x6e657414 // uabd v20.8h, v0.8h, v5.8h
	WORD $0x6e7a4611 // ushl v17.8h, v16.8h, v26.8h
	WORD $0x6e7a4695 // ushl v21.8h, v20.8h, v26.8h
	WORD $0x6e712f71 // uqsub v17.8h, v27.8h, v17.8h
	WORD $0x6e752f75 // uqsub v21.8h, v27.8h, v21.8h
	WORD $0x6e608492 // sub v18.8h, v4.8h, v0.8h
	WORD $0x6e6084b6 // sub v22.8h, v5.8h, v0.8h
	WORD $0x6e60ba30 // neg v16.8h, v17.8h
	WORD $0x6e60bab4 // neg v20.8h, v21.8h
	WORD $0x4e716e52 // smin v18.8h, v18.8h, v17.8h
	WORD $0x4e756ed6 // smin v22.8h, v22.8h, v21.8h
	WORD $0x4e706652 // smax v18.8h, v18.8h, v16.8h
	WORD $0x4e7466d6 // smax v22.8h, v22.8h, v20.8h
	WORD $0x4e7e9641 // mla v1.8h, v18.8h, v30.8h
	WORD $0x4e7e96c1 // mla v1.8h, v22.8h, v30.8h

	// sec taps k=0, off3 (dir - 2)
	ADD  R9, R2, R16
	SUB  R9, R2, R17
	VLD1 (R16), [V4.H8]
	VLD1 (R17), [V5.H8]
	WORD $0x6e646c42 // umin v2.8h, v2.8h, v4.8h
	WORD $0x6e656c42 // umin v2.8h, v2.8h, v5.8h
	WORD $0x6e281c90 // eor v16.16b, v4.16b, v8.16b
	WORD $0x6e281cb4 // eor v20.16b, v5.16b, v8.16b
	WORD $0x6e706463 // umax v3.8h, v3.8h, v16.8h
	WORD $0x6e746463 // umax v3.8h, v3.8h, v20.8h
	WORD $0x6e647410 // uabd v16.8h, v0.8h, v4.8h
	WORD $0x6e657414 // uabd v20.8h, v0.8h, v5.8h
	WORD $0x6e7a4611 // ushl v17.8h, v16.8h, v26.8h
	WORD $0x6e7a4695 // ushl v21.8h, v20.8h, v26.8h
	WORD $0x6e712f71 // uqsub v17.8h, v27.8h, v17.8h
	WORD $0x6e752f75 // uqsub v21.8h, v27.8h, v21.8h
	WORD $0x6e608492 // sub v18.8h, v4.8h, v0.8h
	WORD $0x6e6084b6 // sub v22.8h, v5.8h, v0.8h
	WORD $0x6e60ba30 // neg v16.8h, v17.8h
	WORD $0x6e60bab4 // neg v20.8h, v21.8h
	WORD $0x4e716e52 // smin v18.8h, v18.8h, v17.8h
	WORD $0x4e756ed6 // smin v22.8h, v22.8h, v21.8h
	WORD $0x4e706652 // smax v18.8h, v18.8h, v16.8h
	WORD $0x4e7466d6 // smax v22.8h, v22.8h, v20.8h
	WORD $0x4e7e9641 // mla v1.8h, v18.8h, v30.8h
	WORD $0x4e7e96c1 // mla v1.8h, v22.8h, v30.8h

	// sec taps k=1, off2 (dir + 2)
	ADD  R10, R2, R16
	SUB  R10, R2, R17
	VLD1 (R16), [V4.H8]
	VLD1 (R17), [V5.H8]
	WORD $0x6e646c42 // umin v2.8h, v2.8h, v4.8h
	WORD $0x6e656c42 // umin v2.8h, v2.8h, v5.8h
	WORD $0x6e281c90 // eor v16.16b, v4.16b, v8.16b
	WORD $0x6e281cb4 // eor v20.16b, v5.16b, v8.16b
	WORD $0x6e706463 // umax v3.8h, v3.8h, v16.8h
	WORD $0x6e746463 // umax v3.8h, v3.8h, v20.8h
	WORD $0x6e647410 // uabd v16.8h, v0.8h, v4.8h
	WORD $0x6e657414 // uabd v20.8h, v0.8h, v5.8h
	WORD $0x6e7a4611 // ushl v17.8h, v16.8h, v26.8h
	WORD $0x6e7a4695 // ushl v21.8h, v20.8h, v26.8h
	WORD $0x6e712f71 // uqsub v17.8h, v27.8h, v17.8h
	WORD $0x6e752f75 // uqsub v21.8h, v27.8h, v21.8h
	WORD $0x6e608492 // sub v18.8h, v4.8h, v0.8h
	WORD $0x6e6084b6 // sub v22.8h, v5.8h, v0.8h
	WORD $0x6e60ba30 // neg v16.8h, v17.8h
	WORD $0x6e60bab4 // neg v20.8h, v21.8h
	WORD $0x4e716e52 // smin v18.8h, v18.8h, v17.8h
	WORD $0x4e756ed6 // smin v22.8h, v22.8h, v21.8h
	WORD $0x4e706652 // smax v18.8h, v18.8h, v16.8h
	WORD $0x4e7466d6 // smax v22.8h, v22.8h, v20.8h
	WORD $0x4e7f9641 // mla v1.8h, v18.8h, v31.8h
	WORD $0x4e7f96c1 // mla v1.8h, v22.8h, v31.8h

	// sec taps k=1, off3 (dir - 2)
	ADD  R11, R2, R16
	SUB  R11, R2, R17
	VLD1 (R16), [V4.H8]
	VLD1 (R17), [V5.H8]
	WORD $0x6e646c42 // umin v2.8h, v2.8h, v4.8h
	WORD $0x6e656c42 // umin v2.8h, v2.8h, v5.8h
	WORD $0x6e281c90 // eor v16.16b, v4.16b, v8.16b
	WORD $0x6e281cb4 // eor v20.16b, v5.16b, v8.16b
	WORD $0x6e706463 // umax v3.8h, v3.8h, v16.8h
	WORD $0x6e746463 // umax v3.8h, v3.8h, v20.8h
	WORD $0x6e647410 // uabd v16.8h, v0.8h, v4.8h
	WORD $0x6e657414 // uabd v20.8h, v0.8h, v5.8h
	WORD $0x6e7a4611 // ushl v17.8h, v16.8h, v26.8h
	WORD $0x6e7a4695 // ushl v21.8h, v20.8h, v26.8h
	WORD $0x6e712f71 // uqsub v17.8h, v27.8h, v17.8h
	WORD $0x6e752f75 // uqsub v21.8h, v27.8h, v21.8h
	WORD $0x6e608492 // sub v18.8h, v4.8h, v0.8h
	WORD $0x6e6084b6 // sub v22.8h, v5.8h, v0.8h
	WORD $0x6e60ba30 // neg v16.8h, v17.8h
	WORD $0x6e60bab4 // neg v20.8h, v21.8h
	WORD $0x4e716e52 // smin v18.8h, v18.8h, v17.8h
	WORD $0x4e756ed6 // smin v22.8h, v22.8h, v21.8h
	WORD $0x4e706652 // smax v18.8h, v18.8h, v16.8h
	WORD $0x4e7466d6 // smax v22.8h, v22.8h, v20.8h
	WORD $0x4e7f9641 // mla v1.8h, v18.8h, v31.8h
	WORD $0x4e7f96c1 // mla v1.8h, v22.8h, v31.8h

	// finalize: px + ((8 + sum - (sum < 0)) >> 4), clipped to [min, max]
	WORD $0x4e60a830 // cmlt v16.8h, v1.8h, #0
	WORD $0x4e708421 // add v1.8h, v1.8h, v16.8h
	WORD $0x4f1c2421 // srshr v1.8h, v1.8h, #4
	WORD $0x6e281c63 // eor v3.16b, v3.16b, v8.16b
	WORD $0x6e606463 // umax v3.8h, v3.8h, v0.8h
	WORD $0x4e618400 // add v0.8h, v0.8h, v1.8h
	WORD $0x4e636c00 // smin v0.8h, v0.8h, v3.8h
	WORD $0x4e626400 // smax v0.8h, v0.8h, v2.8h
	WORD $0x0e212800 // xtn v0.8b, v0.8h
	FMOVD F0, (R1) // 8 bytes
	ADD  R3, R1
	ADD  $288, R2 // BStride bytes
	SUB  $1, R4
	CBNZ R4, u8prisec8_row
	RET

// func cdefFilterBlock4U8NEON(ctx *filterBlockU8NEONCtx)
// dav1d cdef_filter4_pri_sec_neon: two 4-wide rows packed into one vector
// per iteration (dav1d load_px w=4), so height must be even (CDEF chroma
// blocks are 4x4 or 4x8).
TEXT ·cdefFilterBlock4U8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_INPUT(R0), R2
	MOVD C_DSTSTR(R0), R3
	MOVD C_HEIGHT(R0), R4
	MOVD C_PRISTR(R0), R5
	WORD $0x4e020cb9 // dup v25.8h, w5
	MOVD C_SECSTR(R0), R5
	WORD $0x4e020cbb // dup v27.8h, w5
	MOVD C_PRISHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e020cb8 // dup v24.8h, w5
	MOVD C_SECSHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e020cba // dup v26.8h, w5
	MOVD C_PRITAP0(R0), R5
	WORD $0x4e020cbc // dup v28.8h, w5
	MOVD C_PRITAP1(R0), R5
	WORD $0x4e020cbd // dup v29.8h, w5
	MOVD C_SECTAP0(R0), R5
	WORD $0x4e020cbe // dup v30.8h, w5
	MOVD C_SECTAP1(R0), R5
	WORD $0x4e020cbf // dup v31.8h, w5
	MOVD $0x4000, R5
	WORD $0x4e020ca8 // dup v8.8h, w5
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

u8prisec4_row:
	ADD  $288, R2, R16
	VLD1 (R2), [V0.H4] // px row 0
	WORD $0x4d408600 // ld1 {v0.d}[1], [x16]
	WORD $0x6e211c21 // eor v1.16b, v1.16b, v1.16b
	VMOV V0.B16, V2.B16
	WORD $0x6e231c63 // eor v3.16b, v3.16b, v3.16b

	// pri taps k=0
	ADD  R6, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V4.H4]
	WORD $0x4d408624 // ld1 {v4.d}[1], [x17]
	SUB  R6, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V5.H4]
	WORD $0x4d408625 // ld1 {v5.d}[1], [x17]
	WORD $0x6e646c42 // umin v2.8h, v2.8h, v4.8h
	WORD $0x6e656c42 // umin v2.8h, v2.8h, v5.8h
	WORD $0x6e281c90 // eor v16.16b, v4.16b, v8.16b
	WORD $0x6e281cb4 // eor v20.16b, v5.16b, v8.16b
	WORD $0x6e706463 // umax v3.8h, v3.8h, v16.8h
	WORD $0x6e746463 // umax v3.8h, v3.8h, v20.8h
	WORD $0x6e647410 // uabd v16.8h, v0.8h, v4.8h
	WORD $0x6e657414 // uabd v20.8h, v0.8h, v5.8h
	WORD $0x6e784611 // ushl v17.8h, v16.8h, v24.8h
	WORD $0x6e784695 // ushl v21.8h, v20.8h, v24.8h
	WORD $0x6e712f31 // uqsub v17.8h, v25.8h, v17.8h
	WORD $0x6e752f35 // uqsub v21.8h, v25.8h, v21.8h
	WORD $0x6e608492 // sub v18.8h, v4.8h, v0.8h
	WORD $0x6e6084b6 // sub v22.8h, v5.8h, v0.8h
	WORD $0x6e60ba30 // neg v16.8h, v17.8h
	WORD $0x6e60bab4 // neg v20.8h, v21.8h
	WORD $0x4e716e52 // smin v18.8h, v18.8h, v17.8h
	WORD $0x4e756ed6 // smin v22.8h, v22.8h, v21.8h
	WORD $0x4e706652 // smax v18.8h, v18.8h, v16.8h
	WORD $0x4e7466d6 // smax v22.8h, v22.8h, v20.8h
	WORD $0x4e7c9641 // mla v1.8h, v18.8h, v28.8h
	WORD $0x4e7c96c1 // mla v1.8h, v22.8h, v28.8h

	// pri taps k=1
	ADD  R7, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V4.H4]
	WORD $0x4d408624 // ld1 {v4.d}[1], [x17]
	SUB  R7, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V5.H4]
	WORD $0x4d408625 // ld1 {v5.d}[1], [x17]
	WORD $0x6e646c42 // umin v2.8h, v2.8h, v4.8h
	WORD $0x6e656c42 // umin v2.8h, v2.8h, v5.8h
	WORD $0x6e281c90 // eor v16.16b, v4.16b, v8.16b
	WORD $0x6e281cb4 // eor v20.16b, v5.16b, v8.16b
	WORD $0x6e706463 // umax v3.8h, v3.8h, v16.8h
	WORD $0x6e746463 // umax v3.8h, v3.8h, v20.8h
	WORD $0x6e647410 // uabd v16.8h, v0.8h, v4.8h
	WORD $0x6e657414 // uabd v20.8h, v0.8h, v5.8h
	WORD $0x6e784611 // ushl v17.8h, v16.8h, v24.8h
	WORD $0x6e784695 // ushl v21.8h, v20.8h, v24.8h
	WORD $0x6e712f31 // uqsub v17.8h, v25.8h, v17.8h
	WORD $0x6e752f35 // uqsub v21.8h, v25.8h, v21.8h
	WORD $0x6e608492 // sub v18.8h, v4.8h, v0.8h
	WORD $0x6e6084b6 // sub v22.8h, v5.8h, v0.8h
	WORD $0x6e60ba30 // neg v16.8h, v17.8h
	WORD $0x6e60bab4 // neg v20.8h, v21.8h
	WORD $0x4e716e52 // smin v18.8h, v18.8h, v17.8h
	WORD $0x4e756ed6 // smin v22.8h, v22.8h, v21.8h
	WORD $0x4e706652 // smax v18.8h, v18.8h, v16.8h
	WORD $0x4e7466d6 // smax v22.8h, v22.8h, v20.8h
	WORD $0x4e7d9641 // mla v1.8h, v18.8h, v29.8h
	WORD $0x4e7d96c1 // mla v1.8h, v22.8h, v29.8h

	// sec taps k=0, off2 (dir + 2)
	ADD  R8, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V4.H4]
	WORD $0x4d408624 // ld1 {v4.d}[1], [x17]
	SUB  R8, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V5.H4]
	WORD $0x4d408625 // ld1 {v5.d}[1], [x17]
	WORD $0x6e646c42 // umin v2.8h, v2.8h, v4.8h
	WORD $0x6e656c42 // umin v2.8h, v2.8h, v5.8h
	WORD $0x6e281c90 // eor v16.16b, v4.16b, v8.16b
	WORD $0x6e281cb4 // eor v20.16b, v5.16b, v8.16b
	WORD $0x6e706463 // umax v3.8h, v3.8h, v16.8h
	WORD $0x6e746463 // umax v3.8h, v3.8h, v20.8h
	WORD $0x6e647410 // uabd v16.8h, v0.8h, v4.8h
	WORD $0x6e657414 // uabd v20.8h, v0.8h, v5.8h
	WORD $0x6e7a4611 // ushl v17.8h, v16.8h, v26.8h
	WORD $0x6e7a4695 // ushl v21.8h, v20.8h, v26.8h
	WORD $0x6e712f71 // uqsub v17.8h, v27.8h, v17.8h
	WORD $0x6e752f75 // uqsub v21.8h, v27.8h, v21.8h
	WORD $0x6e608492 // sub v18.8h, v4.8h, v0.8h
	WORD $0x6e6084b6 // sub v22.8h, v5.8h, v0.8h
	WORD $0x6e60ba30 // neg v16.8h, v17.8h
	WORD $0x6e60bab4 // neg v20.8h, v21.8h
	WORD $0x4e716e52 // smin v18.8h, v18.8h, v17.8h
	WORD $0x4e756ed6 // smin v22.8h, v22.8h, v21.8h
	WORD $0x4e706652 // smax v18.8h, v18.8h, v16.8h
	WORD $0x4e7466d6 // smax v22.8h, v22.8h, v20.8h
	WORD $0x4e7e9641 // mla v1.8h, v18.8h, v30.8h
	WORD $0x4e7e96c1 // mla v1.8h, v22.8h, v30.8h

	// sec taps k=0, off3 (dir - 2)
	ADD  R9, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V4.H4]
	WORD $0x4d408624 // ld1 {v4.d}[1], [x17]
	SUB  R9, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V5.H4]
	WORD $0x4d408625 // ld1 {v5.d}[1], [x17]
	WORD $0x6e646c42 // umin v2.8h, v2.8h, v4.8h
	WORD $0x6e656c42 // umin v2.8h, v2.8h, v5.8h
	WORD $0x6e281c90 // eor v16.16b, v4.16b, v8.16b
	WORD $0x6e281cb4 // eor v20.16b, v5.16b, v8.16b
	WORD $0x6e706463 // umax v3.8h, v3.8h, v16.8h
	WORD $0x6e746463 // umax v3.8h, v3.8h, v20.8h
	WORD $0x6e647410 // uabd v16.8h, v0.8h, v4.8h
	WORD $0x6e657414 // uabd v20.8h, v0.8h, v5.8h
	WORD $0x6e7a4611 // ushl v17.8h, v16.8h, v26.8h
	WORD $0x6e7a4695 // ushl v21.8h, v20.8h, v26.8h
	WORD $0x6e712f71 // uqsub v17.8h, v27.8h, v17.8h
	WORD $0x6e752f75 // uqsub v21.8h, v27.8h, v21.8h
	WORD $0x6e608492 // sub v18.8h, v4.8h, v0.8h
	WORD $0x6e6084b6 // sub v22.8h, v5.8h, v0.8h
	WORD $0x6e60ba30 // neg v16.8h, v17.8h
	WORD $0x6e60bab4 // neg v20.8h, v21.8h
	WORD $0x4e716e52 // smin v18.8h, v18.8h, v17.8h
	WORD $0x4e756ed6 // smin v22.8h, v22.8h, v21.8h
	WORD $0x4e706652 // smax v18.8h, v18.8h, v16.8h
	WORD $0x4e7466d6 // smax v22.8h, v22.8h, v20.8h
	WORD $0x4e7e9641 // mla v1.8h, v18.8h, v30.8h
	WORD $0x4e7e96c1 // mla v1.8h, v22.8h, v30.8h

	// sec taps k=1, off2 (dir + 2)
	ADD  R10, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V4.H4]
	WORD $0x4d408624 // ld1 {v4.d}[1], [x17]
	SUB  R10, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V5.H4]
	WORD $0x4d408625 // ld1 {v5.d}[1], [x17]
	WORD $0x6e646c42 // umin v2.8h, v2.8h, v4.8h
	WORD $0x6e656c42 // umin v2.8h, v2.8h, v5.8h
	WORD $0x6e281c90 // eor v16.16b, v4.16b, v8.16b
	WORD $0x6e281cb4 // eor v20.16b, v5.16b, v8.16b
	WORD $0x6e706463 // umax v3.8h, v3.8h, v16.8h
	WORD $0x6e746463 // umax v3.8h, v3.8h, v20.8h
	WORD $0x6e647410 // uabd v16.8h, v0.8h, v4.8h
	WORD $0x6e657414 // uabd v20.8h, v0.8h, v5.8h
	WORD $0x6e7a4611 // ushl v17.8h, v16.8h, v26.8h
	WORD $0x6e7a4695 // ushl v21.8h, v20.8h, v26.8h
	WORD $0x6e712f71 // uqsub v17.8h, v27.8h, v17.8h
	WORD $0x6e752f75 // uqsub v21.8h, v27.8h, v21.8h
	WORD $0x6e608492 // sub v18.8h, v4.8h, v0.8h
	WORD $0x6e6084b6 // sub v22.8h, v5.8h, v0.8h
	WORD $0x6e60ba30 // neg v16.8h, v17.8h
	WORD $0x6e60bab4 // neg v20.8h, v21.8h
	WORD $0x4e716e52 // smin v18.8h, v18.8h, v17.8h
	WORD $0x4e756ed6 // smin v22.8h, v22.8h, v21.8h
	WORD $0x4e706652 // smax v18.8h, v18.8h, v16.8h
	WORD $0x4e7466d6 // smax v22.8h, v22.8h, v20.8h
	WORD $0x4e7f9641 // mla v1.8h, v18.8h, v31.8h
	WORD $0x4e7f96c1 // mla v1.8h, v22.8h, v31.8h

	// sec taps k=1, off3 (dir - 2)
	ADD  R11, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V4.H4]
	WORD $0x4d408624 // ld1 {v4.d}[1], [x17]
	SUB  R11, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V5.H4]
	WORD $0x4d408625 // ld1 {v5.d}[1], [x17]
	WORD $0x6e646c42 // umin v2.8h, v2.8h, v4.8h
	WORD $0x6e656c42 // umin v2.8h, v2.8h, v5.8h
	WORD $0x6e281c90 // eor v16.16b, v4.16b, v8.16b
	WORD $0x6e281cb4 // eor v20.16b, v5.16b, v8.16b
	WORD $0x6e706463 // umax v3.8h, v3.8h, v16.8h
	WORD $0x6e746463 // umax v3.8h, v3.8h, v20.8h
	WORD $0x6e647410 // uabd v16.8h, v0.8h, v4.8h
	WORD $0x6e657414 // uabd v20.8h, v0.8h, v5.8h
	WORD $0x6e7a4611 // ushl v17.8h, v16.8h, v26.8h
	WORD $0x6e7a4695 // ushl v21.8h, v20.8h, v26.8h
	WORD $0x6e712f71 // uqsub v17.8h, v27.8h, v17.8h
	WORD $0x6e752f75 // uqsub v21.8h, v27.8h, v21.8h
	WORD $0x6e608492 // sub v18.8h, v4.8h, v0.8h
	WORD $0x6e6084b6 // sub v22.8h, v5.8h, v0.8h
	WORD $0x6e60ba30 // neg v16.8h, v17.8h
	WORD $0x6e60bab4 // neg v20.8h, v21.8h
	WORD $0x4e716e52 // smin v18.8h, v18.8h, v17.8h
	WORD $0x4e756ed6 // smin v22.8h, v22.8h, v21.8h
	WORD $0x4e706652 // smax v18.8h, v18.8h, v16.8h
	WORD $0x4e7466d6 // smax v22.8h, v22.8h, v20.8h
	WORD $0x4e7f9641 // mla v1.8h, v18.8h, v31.8h
	WORD $0x4e7f96c1 // mla v1.8h, v22.8h, v31.8h

	// finalize both rows, then store row halves like dav1d's st1 {v0.d}[0/1]
	WORD $0x4e60a830 // cmlt v16.8h, v1.8h, #0
	WORD $0x4e708421 // add v1.8h, v1.8h, v16.8h
	WORD $0x4f1c2421 // srshr v1.8h, v1.8h, #4
	WORD $0x6e281c63 // eor v3.16b, v3.16b, v8.16b
	WORD $0x6e606463 // umax v3.8h, v3.8h, v0.8h
	WORD $0x4e618400 // add v0.8h, v0.8h, v1.8h
	WORD $0x4e636c00 // smin v0.8h, v0.8h, v3.8h
	WORD $0x4e626400 // smax v0.8h, v0.8h, v2.8h
	WORD $0x0e212800 // xtn v0.8b, v0.8h
	FMOVS F0, (R1) // row 0: 4 bytes
	ADD  R3, R1
	WORD $0x0d009020 // st1 {v0.s}[1], [x1] (row 1: 4 bytes)
	ADD  R3, R1
	ADD  $576, R2 // 2*BStride bytes
	SUB  $2, R4
	CBNZ R4, u8prisec4_row
	RET
