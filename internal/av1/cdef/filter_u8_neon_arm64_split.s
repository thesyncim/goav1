// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// Strength-specialized 8-bit-dst CDEF kernels: the uint16 kernels of
// filter_neon_arm64_split.s (dav1d cdef_filter{8,4}_pri / cdef_filter{8,4}_sec,
// src/arm/64/cdef_tmpl.S filter_func with min=0) with dav1d's 8bpc store
// epilogue (src/arm/64/cdef.S: xtn to bytes, st1 {v0.8b} / st1 {v0.s}[lane]).
// Tap loads still read the uint16 CDEF input buffer; dst is the frame's uint8
// plane with a byte stride. See filter_u8_neon_arm64.s and filter_u8.go for
// the narrowing bit-exactness argument. Primary-only kernels add each tap
// pair before applying its runtime weight, avoiding four serial MLAs.
// Secondary-only kernels accumulate the four +/- chains independently and
// fold the fixed {2,1} weights after the constraints, shortening the critical
// dependency path without changing the integer result.

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

// func cdefFilterBlock8PrimaryU8NEON(ctx *filterBlockU8NEONCtx)
// dav1d cdef_filter8_pri_neon.
TEXT ·cdefFilterBlock8PrimaryU8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_INPUT(R0), R2
	MOVD C_DSTSTR(R0), R3
	MOVD C_HEIGHT(R0), R4
	MOVD C_PRISTR(R0), R5
	WORD $0x4e020cb9 // dup v25.8h, w5
	MOVD C_PRISHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e020cb8 // dup v24.8h, w5
	MOVD C_PRITAP0(R0), R5
	WORD $0x4e020cbc // dup v28.8h, w5
	MOVD C_PRITAP1(R0), R5
	WORD $0x4e020cbd // dup v29.8h, w5
	MOVD C_PRI0(R0), R6
	LSL  $1, R6
	MOVD C_PRI1(R0), R7
	LSL  $1, R7

u8pri8_row:
	VLD1 (R2), [V0.H8] // px

	// pri taps k=0
	ADD  R6, R2, R16
	SUB  R6, R2, R17
	VLD1 (R16), [V4.H8]
	VLD1 (R17), [V5.H8]
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
	WORD $0x4e768641 // add v1.8h, v18.8h, v22.8h
	WORD $0x4e7c9c21 // mul v1.8h, v1.8h, v28.8h

	// pri taps k=1
	ADD  R7, R2, R16
	SUB  R7, R2, R17
	VLD1 (R16), [V4.H8]
	VLD1 (R17), [V5.H8]
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
	WORD $0x4e768642 // add v2.8h, v18.8h, v22.8h
	WORD $0x4e7d9c42 // mul v2.8h, v2.8h, v29.8h
	WORD $0x4e628421 // add v1.8h, v1.8h, v2.8h

	// finalize: px + ((8 + sum - (sum < 0)) >> 4)
	WORD $0x4e60a830 // cmlt v16.8h, v1.8h, #0
	WORD $0x4e708421 // add v1.8h, v1.8h, v16.8h
	WORD $0x4f1c2421 // srshr v1.8h, v1.8h, #4
	WORD $0x4e618400 // add v0.8h, v0.8h, v1.8h
	WORD $0x0e212800 // xtn v0.8b, v0.8h
	FMOVD F0, (R1) // 8 bytes
	ADD  R3, R1
	ADD  $288, R2 // BStride bytes
	SUB  $1, R4
	CBNZ R4, u8pri8_row
	RET

// func cdefFilterBlock8SecondaryU8NEON(ctx *filterBlockU8NEONCtx)
// dav1d cdef_filter8_sec_neon.
TEXT ·cdefFilterBlock8SecondaryU8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_INPUT(R0), R2
	MOVD C_DSTSTR(R0), R3
	MOVD C_HEIGHT(R0), R4
	MOVD C_SECSTR(R0), R5
	WORD $0x4e020cbb // dup v27.8h, w5
	MOVD C_SECSHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e020cba // dup v26.8h, w5
	MOVD C_SEC0(R0), R8
	LSL  $1, R8
	MOVD C_SEC1(R0), R9
	LSL  $1, R9
	MOVD C_SEC2(R0), R10
	LSL  $1, R10
	MOVD C_SEC3(R0), R11
	LSL  $1, R11

u8sec8_row:
	VLD1 (R2), [V0.H8] // px

	// sec taps k=0, off2 (dir + 2)
	ADD  R8, R2, R16
	SUB  R8, R2, R17
	VLD1 (R16), [V4.H8]
	VLD1 (R17), [V5.H8]
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
	WORD $0x4e706641 // smax v1.8h, v18.8h, v16.8h (seed tap-0 accumulators)
	WORD $0x4e7466c2 // smax v2.8h, v22.8h, v20.8h

	// sec taps k=0, off3 (dir - 2)
	ADD  R9, R2, R16
	SUB  R9, R2, R17
	VLD1 (R16), [V4.H8]
	VLD1 (R17), [V5.H8]
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
	WORD $0x4e728421 // add v1.8h, v1.8h, v18.8h
	WORD $0x4e768442 // add v2.8h, v2.8h, v22.8h

	// sec taps k=1, off2 (dir + 2)
	ADD  R10, R2, R16
	SUB  R10, R2, R17
	VLD1 (R16), [V4.H8]
	VLD1 (R17), [V5.H8]
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
	WORD $0x4e706643 // smax v3.8h, v18.8h, v16.8h (seed tap-1 accumulators)
	WORD $0x4e7466c6 // smax v6.8h, v22.8h, v20.8h

	// sec taps k=1, off3 (dir - 2)
	ADD  R11, R2, R16
	SUB  R11, R2, R17
	VLD1 (R16), [V4.H8]
	VLD1 (R17), [V5.H8]
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
	WORD $0x4e728463 // add v3.8h, v3.8h, v18.8h
	WORD $0x4e7684c6 // add v6.8h, v6.8h, v22.8h

	// finalize
	WORD $0x4e628421 // add v1.8h, v1.8h, v2.8h
	WORD $0x4f115421 // shl v1.8h, v1.8h, #1
	WORD $0x4e668463 // add v3.8h, v3.8h, v6.8h
	WORD $0x4e638421 // add v1.8h, v1.8h, v3.8h
	WORD $0x4e60a830 // cmlt v16.8h, v1.8h, #0
	WORD $0x4e708421 // add v1.8h, v1.8h, v16.8h
	WORD $0x4f1c2421 // srshr v1.8h, v1.8h, #4
	WORD $0x4e618400 // add v0.8h, v0.8h, v1.8h
	WORD $0x0e212800 // xtn v0.8b, v0.8h
	FMOVD F0, (R1) // 8 bytes
	ADD  R3, R1
	ADD  $288, R2
	SUB  $1, R4
	CBNZ R4, u8sec8_row
	RET

// func cdefFilterBlock4PrimaryU8NEON(ctx *filterBlockU8NEONCtx)
// dav1d cdef_filter4_pri_neon: two 4-wide rows per iteration.
TEXT ·cdefFilterBlock4PrimaryU8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_INPUT(R0), R2
	MOVD C_DSTSTR(R0), R3
	MOVD C_HEIGHT(R0), R4
	MOVD C_PRISTR(R0), R5
	WORD $0x4e020cb9 // dup v25.8h, w5
	MOVD C_PRISHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e020cb8 // dup v24.8h, w5
	MOVD C_PRITAP0(R0), R5
	WORD $0x4e020cbc // dup v28.8h, w5
	MOVD C_PRITAP1(R0), R5
	WORD $0x4e020cbd // dup v29.8h, w5
	MOVD C_PRI0(R0), R6
	LSL  $1, R6
	MOVD C_PRI1(R0), R7
	LSL  $1, R7

u8pri4_row:
	ADD  $288, R2, R16
	VLD1 (R2), [V0.H4]
	WORD $0x4d408600 // ld1 {v0.d}[1], [x16]

	// pri taps k=0
	ADD  R6, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V4.H4]
	WORD $0x4d408624 // ld1 {v4.d}[1], [x17]
	SUB  R6, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V5.H4]
	WORD $0x4d408625 // ld1 {v5.d}[1], [x17]
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
	WORD $0x4e768641 // add v1.8h, v18.8h, v22.8h
	WORD $0x4e7c9c21 // mul v1.8h, v1.8h, v28.8h

	// pri taps k=1
	ADD  R7, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V4.H4]
	WORD $0x4d408624 // ld1 {v4.d}[1], [x17]
	SUB  R7, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V5.H4]
	WORD $0x4d408625 // ld1 {v5.d}[1], [x17]
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
	WORD $0x4e768642 // add v2.8h, v18.8h, v22.8h
	WORD $0x4e7d9c42 // mul v2.8h, v2.8h, v29.8h
	WORD $0x4e628421 // add v1.8h, v1.8h, v2.8h

	// finalize
	WORD $0x4e60a830 // cmlt v16.8h, v1.8h, #0
	WORD $0x4e708421 // add v1.8h, v1.8h, v16.8h
	WORD $0x4f1c2421 // srshr v1.8h, v1.8h, #4
	WORD $0x4e618400 // add v0.8h, v0.8h, v1.8h
	WORD $0x0e212800 // xtn v0.8b, v0.8h
	FMOVS F0, (R1) // row 0: 4 bytes
	ADD  R3, R1
	WORD $0x0d009020 // st1 {v0.s}[1], [x1] (row 1: 4 bytes)
	ADD  R3, R1
	ADD  $576, R2
	SUB  $2, R4
	CBNZ R4, u8pri4_row
	RET

// func cdefFilterBlock4SecondaryU8NEON(ctx *filterBlockU8NEONCtx)
// dav1d cdef_filter4_sec_neon: two 4-wide rows per iteration.
TEXT ·cdefFilterBlock4SecondaryU8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_INPUT(R0), R2
	MOVD C_DSTSTR(R0), R3
	MOVD C_HEIGHT(R0), R4
	MOVD C_SECSTR(R0), R5
	WORD $0x4e020cbb // dup v27.8h, w5
	MOVD C_SECSHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e020cba // dup v26.8h, w5
	MOVD C_SEC0(R0), R8
	LSL  $1, R8
	MOVD C_SEC1(R0), R9
	LSL  $1, R9
	MOVD C_SEC2(R0), R10
	LSL  $1, R10
	MOVD C_SEC3(R0), R11
	LSL  $1, R11

u8sec4_row:
	ADD  $288, R2, R16
	VLD1 (R2), [V0.H4]
	WORD $0x4d408600 // ld1 {v0.d}[1], [x16]

	// sec taps k=0, off2 (dir + 2)
	ADD  R8, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V4.H4]
	WORD $0x4d408624 // ld1 {v4.d}[1], [x17]
	SUB  R8, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V5.H4]
	WORD $0x4d408625 // ld1 {v5.d}[1], [x17]
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
	WORD $0x4e706641 // smax v1.8h, v18.8h, v16.8h (seed tap-0 accumulators)
	WORD $0x4e7466c2 // smax v2.8h, v22.8h, v20.8h

	// sec taps k=0, off3 (dir - 2)
	ADD  R9, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V4.H4]
	WORD $0x4d408624 // ld1 {v4.d}[1], [x17]
	SUB  R9, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V5.H4]
	WORD $0x4d408625 // ld1 {v5.d}[1], [x17]
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
	WORD $0x4e728421 // add v1.8h, v1.8h, v18.8h
	WORD $0x4e768442 // add v2.8h, v2.8h, v22.8h

	// sec taps k=1, off2 (dir + 2)
	ADD  R10, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V4.H4]
	WORD $0x4d408624 // ld1 {v4.d}[1], [x17]
	SUB  R10, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V5.H4]
	WORD $0x4d408625 // ld1 {v5.d}[1], [x17]
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
	WORD $0x4e706643 // smax v3.8h, v18.8h, v16.8h (seed tap-1 accumulators)
	WORD $0x4e7466c6 // smax v6.8h, v22.8h, v20.8h

	// sec taps k=1, off3 (dir - 2)
	ADD  R11, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V4.H4]
	WORD $0x4d408624 // ld1 {v4.d}[1], [x17]
	SUB  R11, R2, R16
	ADD  $288, R16, R17
	VLD1 (R16), [V5.H4]
	WORD $0x4d408625 // ld1 {v5.d}[1], [x17]
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
	WORD $0x4e728463 // add v3.8h, v3.8h, v18.8h
	WORD $0x4e7684c6 // add v6.8h, v6.8h, v22.8h

	// finalize
	WORD $0x4e628421 // add v1.8h, v1.8h, v2.8h
	WORD $0x4f115421 // shl v1.8h, v1.8h, #1
	WORD $0x4e668463 // add v3.8h, v3.8h, v6.8h
	WORD $0x4e638421 // add v1.8h, v1.8h, v3.8h
	WORD $0x4e60a830 // cmlt v16.8h, v1.8h, #0
	WORD $0x4e708421 // add v1.8h, v1.8h, v16.8h
	WORD $0x4f1c2421 // srshr v1.8h, v1.8h, #4
	WORD $0x4e618400 // add v0.8h, v0.8h, v1.8h
	WORD $0x0e212800 // xtn v0.8b, v0.8h
	FMOVS F0, (R1) // row 0: 4 bytes
	ADD  R3, R1
	WORD $0x0d009020 // st1 {v0.s}[1], [x1] (row 1: 4 bytes)
	ADD  R3, R1
	ADD  $576, R2
	SUB  $2, R4
	CBNZ R4, u8sec4_row
	RET
