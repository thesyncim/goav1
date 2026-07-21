// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

#define S_DST    0
#define S_INPUT  8
#define S_DSTSTR 16
#define S_SECSTR 24
#define S_SECSH  32

// func cdefFilterBlock8SecondaryByteU8NEON(ctx *filterBlockU8SecondaryByteNEONCtx)
//
// Direction zero makes the secondary taps axis-aligned: +/-1 row/column has
// weight 2 and +/-2 rows/columns has weight 1. Two output rows occupy the 16
// byte lanes together. The vertical neighbours are formed from the two-row
// centre/up2/down2 vectors, sharing the overlapping rows instead of loading
// all four vertical tap vectors independently. Secondary-only sums are
// bounded by +/-48, so the exact filter arithmetic fits in signed bytes.
TEXT ·cdefFilterBlock8SecondaryByteU8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD S_DST(R0), R1
	MOVD S_INPUT(R0), R2
	MOVD S_DSTSTR(R0), R3
	MOVD S_SECSTR(R0), R5
	WORD $0x4e010cbb       // dup v27.16b, w5  secondary strength
	MOVD S_SECSH(R0), R5
	NEG  R5, R5
	WORD $0x4e010cba       // dup v26.16b, w5  negative constrain shift
	WORD $0x4f00e45c       // movi v28.16b, #2 tap-0 weight
	MOVD $4, R4

u8byte_sec8_rows2:
	// Centre rows r,r+1.
	VLD1 (R2), [V0.B8]
	ADD  $144, R2, R12
	WORD $0x4d408580       // ld1 {v0.d}[1], [x12]

	// Rows r-2,r-1 and r+2,r+3. The +/-1 vectors below are their
	// overlapping halves joined to the centre vector.
	SUB  $288, R2, R12
	VLD1 (R12), [V6.B8]
	ADD  $144, R12, R13
	WORD $0x4d4085a6       // ld1 {v6.d}[1], [x13]
	ADD  $288, R2, R12
	VLD1 (R12), [V7.B8]
	ADD  $144, R12, R13
	WORD $0x4d4085a7       // ld1 {v7.d}[1], [x13]

	// Horizontal +/-1, tap weight 2.
	ADD  $1, R2, R12
	VLD1 (R12), [V4.B8]
	ADD  $144, R12, R13
	WORD $0x4d4085a4       // ld1 {v4.d}[1], [x13]
	SUB  $1, R2, R12
	VLD1 (R12), [V5.B8]
	ADD  $144, R12, R13
	WORD $0x4d4085a5       // ld1 {v5.d}[1], [x13]
	WORD $0x6e247410       // uabd v16.16b, v0.16b, v4.16b
	WORD $0x6e257414       // uabd v20.16b, v0.16b, v5.16b
	WORD $0x6e3a4611       // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695       // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71       // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75       // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e243412       // cmhi v18.16b, v0.16b, v4.16b
	WORD $0x6e253416       // cmhi v22.16b, v0.16b, v5.16b
	WORD $0x6e306e31       // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5       // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30       // neg v16.16b, v17.16b
	WORD $0x6e20bab4       // neg v20.16b, v21.16b
	WORD $0x6e711e12       // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96       // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e368652       // add v18.16b, v18.16b, v22.16b
	WORD $0x4f095641       // shl v1.16b, v18.16b, #1  seed sum

	// Vertical +/-1, tap weight 2.
	WORD $0x6e0040c4       // ext v4.16b, v6.16b, v0.16b, #8
	WORD $0x6e074005       // ext v5.16b, v0.16b, v7.16b, #8
	WORD $0x6e247410       // uabd v16.16b, v0.16b, v4.16b
	WORD $0x6e257414       // uabd v20.16b, v0.16b, v5.16b
	WORD $0x6e3a4611       // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695       // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71       // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75       // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e243412       // cmhi v18.16b, v0.16b, v4.16b
	WORD $0x6e253416       // cmhi v22.16b, v0.16b, v5.16b
	WORD $0x6e306e31       // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5       // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30       // neg v16.16b, v17.16b
	WORD $0x6e20bab4       // neg v20.16b, v21.16b
	WORD $0x6e711e12       // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96       // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e368652       // add v18.16b, v18.16b, v22.16b
	WORD $0x4e3c9641       // mla v1.16b, v18.16b, v28.16b

	// Horizontal +/-2, tap weight 1.
	ADD  $2, R2, R12
	VLD1 (R12), [V4.B8]
	ADD  $144, R12, R13
	WORD $0x4d4085a4       // ld1 {v4.d}[1], [x13]
	SUB  $2, R2, R12
	VLD1 (R12), [V5.B8]
	ADD  $144, R12, R13
	WORD $0x4d4085a5       // ld1 {v5.d}[1], [x13]
	WORD $0x6e247410       // uabd v16.16b, v0.16b, v4.16b
	WORD $0x6e257414       // uabd v20.16b, v0.16b, v5.16b
	WORD $0x6e3a4611       // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695       // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71       // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75       // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e243412       // cmhi v18.16b, v0.16b, v4.16b
	WORD $0x6e253416       // cmhi v22.16b, v0.16b, v5.16b
	WORD $0x6e306e31       // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5       // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30       // neg v16.16b, v17.16b
	WORD $0x6e20bab4       // neg v20.16b, v21.16b
	WORD $0x6e711e12       // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96       // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e368652       // add v18.16b, v18.16b, v22.16b
	WORD $0x4e328421       // add v1.16b, v1.16b, v18.16b

	// Vertical +/-2, tap weight 1; v7/v6 are the down/up vectors.
	WORD $0x6e277410       // uabd v16.16b, v0.16b, v7.16b
	WORD $0x6e267414       // uabd v20.16b, v0.16b, v6.16b
	WORD $0x6e3a4611       // ushl v17.16b, v16.16b, v26.16b
	WORD $0x6e3a4695       // ushl v21.16b, v20.16b, v26.16b
	WORD $0x6e312f71       // uqsub v17.16b, v27.16b, v17.16b
	WORD $0x6e352f75       // uqsub v21.16b, v27.16b, v21.16b
	WORD $0x6e273412       // cmhi v18.16b, v0.16b, v7.16b
	WORD $0x6e263416       // cmhi v22.16b, v0.16b, v6.16b
	WORD $0x6e306e31       // umin v17.16b, v17.16b, v16.16b
	WORD $0x6e346eb5       // umin v21.16b, v21.16b, v20.16b
	WORD $0x6e20ba30       // neg v16.16b, v17.16b
	WORD $0x6e20bab4       // neg v20.16b, v21.16b
	WORD $0x6e711e12       // bsl v18.16b, v16.16b, v17.16b
	WORD $0x6e751e96       // bsl v22.16b, v20.16b, v21.16b
	WORD $0x4e368652       // add v18.16b, v18.16b, v22.16b
	WORD $0x4e328421       // add v1.16b, v1.16b, v18.16b

	// Exact CDEF rounding and unsigned-saturating pixel add.
	WORD $0x4e20a830       // cmlt v16.16b, v1.16b, #0
	WORD $0x4e308421       // add v1.16b, v1.16b, v16.16b
	WORD $0x4f0c2421       // srshr v1.16b, v1.16b, #4
	WORD $0x6e203820       // usqadd v0.16b, v1.16b
	FMOVD F0, (R1)
	ADD  R3, R1, R1
	WORD $0x4d008420       // st1 {v0.d}[1], [x1]
	ADD  R3, R1, R1
	ADD  $288, R2, R2
	SUB  $1, R4, R4
	CBNZ R4, u8byte_sec8_rows2
	RET

// General two-row byte-input kernels. These use dav1d's two signed-byte
// accumulator representation: each +/- tap chain is bounded to one byte,
// then SRHADD/SHADD reconstruct the exact nine-bit sum and CDEF rounding.
#define G_DST     0
#define G_INPUT   8
#define G_DSTSTR  16
#define G_HEIGHT  24
#define G_PRI0    32
#define G_PRI1    40
#define G_SEC0    48
#define G_SEC1    56
#define G_SEC2    64
#define G_SEC3    72
#define G_PRITAP0 80
#define G_PRITAP1 88
#define G_PRISTR  112
#define G_SECSTR  120
#define G_PRISH   128
#define G_SECSH   136

#define G_LOAD_CENTER VLD1 (R2), [V0.B8]; ADD $144, R2, R12; WORD $0x4d408580
#define G_LOAD_PAIR(off) ADD off, R2, R12; SUB off, R2, R13; VLD1 (R12), [V5.B8]; VLD1 (R13), [V6.B8]; ADD $144, R12, R14; ADD $144, R13, R15; WORD $0x4d4085c5; WORD $0x4d4085e6
#define G_INIT_SUM WORD $0x4f07e7e1; WORD $0x4f00e402
#define G_INIT_CLIP WORD $0x4ea01c03; WORD $0x4ea01c04
#define G_CLIP_PAIR WORD $0x6e256c63; WORD $0x6e256484; WORD $0x6e266c63; WORD $0x6e266484
#define G_CONSTRAIN WORD $0x6e257410; WORD $0x6e267414; WORD $0x6e3a4611; WORD $0x6e3a4695; WORD $0x6e312f71; WORD $0x6e352f75; WORD $0x6e253412; WORD $0x6e263416; WORD $0x6e306e31; WORD $0x6e346eb5; WORD $0x6e20ba30; WORD $0x6e20bab4; WORD $0x6e711e12; WORD $0x6e751e96
#define G_CONSTRAIN_PRI WORD $0x6e257410; WORD $0x6e267414; WORD $0x6e384611; WORD $0x6e384695; WORD $0x6e312f31; WORD $0x6e352f35; WORD $0x6e253412; WORD $0x6e263416; WORD $0x6e306e31; WORD $0x6e346eb5; WORD $0x6e20ba30; WORD $0x6e20bab4; WORD $0x6e711e12; WORD $0x6e751e96
#define G_ACC28 WORD $0x4e3c9641; WORD $0x4e3c96c2
#define G_ACC29 WORD $0x4e3d9641; WORD $0x4e3d96c2
#define G_ACC30 WORD $0x4e3e9641; WORD $0x4e3e96c2
#define G_ACC31 WORD $0x4e3f9641; WORD $0x4e3f96c2
#define G_FINAL WORD $0x4e221425; WORD $0x4e220426; WORD $0x4e20a8a1; WORD $0x6e651cc1; WORD $0x4f0d2421; WORD $0x6e203820
#define G_FINAL_CLIP WORD $0x6e246c00; WORD $0x6e236400
#define G_STORE FMOVD F0, (R1); ADD R3, R1, R1; WORD $0x4d008420; ADD R3, R1, R1; ADD $288, R2, R2; SUB $2, R4, R4

// func cdefFilterBlock8PrimaryByteU8NEON(ctx *filterBlockU8ByteNEONCtx)
TEXT ·cdefFilterBlock8PrimaryByteU8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD G_DST(R0), R1
	MOVD G_INPUT(R0), R2
	MOVD G_DSTSTR(R0), R3
	MOVD G_HEIGHT(R0), R4
	MOVD G_PRISTR(R0), R5
	WORD $0x4e010cbb       // dup v27.16b, w5
	MOVD G_PRISH(R0), R5
	NEG  R5, R5
	WORD $0x4e010cba       // dup v26.16b, w5
	MOVD G_PRITAP0(R0), R5
	WORD $0x4e010cbc       // dup v28.16b, w5
	MOVD G_PRITAP1(R0), R5
	WORD $0x4e010cbd       // dup v29.16b, w5
	MOVD G_PRI0(R0), R6
	MOVD G_PRI1(R0), R7

u8byte_pri8_rows2:
	G_LOAD_CENTER
	G_INIT_SUM
	G_LOAD_PAIR(R6)
	G_CONSTRAIN
	G_ACC28
	G_LOAD_PAIR(R7)
	G_CONSTRAIN
	G_ACC29
	G_FINAL
	G_STORE
	CBNZ R4, u8byte_pri8_rows2
	RET

// func cdefFilterBlock8SecondaryGeneralByteU8NEON(ctx *filterBlockU8ByteNEONCtx)
TEXT ·cdefFilterBlock8SecondaryGeneralByteU8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD G_DST(R0), R1
	MOVD G_INPUT(R0), R2
	MOVD G_DSTSTR(R0), R3
	MOVD G_HEIGHT(R0), R4
	MOVD G_SECSTR(R0), R5
	WORD $0x4e010cbb       // dup v27.16b, w5
	MOVD G_SECSH(R0), R5
	NEG  R5, R5
	WORD $0x4e010cba       // dup v26.16b, w5
	WORD $0x4f00e45e       // movi v30.16b, #2
	WORD $0x4f00e43f       // movi v31.16b, #1
	MOVD G_SEC0(R0), R6
	MOVD G_SEC1(R0), R7
	MOVD G_SEC2(R0), R8
	MOVD G_SEC3(R0), R9

u8byte_secgen8_rows2:
	G_LOAD_CENTER
	G_INIT_SUM
	G_LOAD_PAIR(R6)
	G_CONSTRAIN
	G_ACC30
	G_LOAD_PAIR(R7)
	G_CONSTRAIN
	G_ACC30
	G_LOAD_PAIR(R8)
	G_CONSTRAIN
	G_ACC31
	G_LOAD_PAIR(R9)
	G_CONSTRAIN
	G_ACC31
	G_FINAL
	G_STORE
	CBNZ R4, u8byte_secgen8_rows2
	RET

// func cdefFilterBlock8FusedByteU8NEON(ctx *filterBlockU8ByteNEONCtx)
TEXT ·cdefFilterBlock8FusedByteU8NEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD G_DST(R0), R1
	MOVD G_INPUT(R0), R2
	MOVD G_DSTSTR(R0), R3
	MOVD G_HEIGHT(R0), R4
	MOVD G_PRISTR(R0), R5
	WORD $0x4e010cb9       // dup v25.16b, w5
	MOVD G_PRISH(R0), R5
	NEG  R5, R5
	WORD $0x4e010cb8       // dup v24.16b, w5
	MOVD G_SECSTR(R0), R5
	WORD $0x4e010cbb       // dup v27.16b, w5
	MOVD G_SECSH(R0), R5
	NEG  R5, R5
	WORD $0x4e010cba       // dup v26.16b, w5
	MOVD G_PRITAP0(R0), R5
	WORD $0x4e010cbc       // dup v28.16b, w5
	MOVD G_PRITAP1(R0), R5
	WORD $0x4e010cbd       // dup v29.16b, w5
	WORD $0x4f00e45e       // movi v30.16b, #2
	WORD $0x4f00e43f       // movi v31.16b, #1

u8byte_fused8_rows2:
	G_LOAD_CENTER
	G_INIT_SUM
	G_INIT_CLIP
	MOVD G_PRI0(R0), R6
	G_LOAD_PAIR(R6)
	G_CLIP_PAIR
	G_CONSTRAIN_PRI
	G_ACC28
	MOVD G_PRI1(R0), R6
	G_LOAD_PAIR(R6)
	G_CLIP_PAIR
	G_CONSTRAIN_PRI
	G_ACC29
	MOVD G_SEC0(R0), R6
	G_LOAD_PAIR(R6)
	G_CLIP_PAIR
	G_CONSTRAIN
	G_ACC30
	MOVD G_SEC1(R0), R6
	G_LOAD_PAIR(R6)
	G_CLIP_PAIR
	G_CONSTRAIN
	G_ACC30
	MOVD G_SEC2(R0), R6
	G_LOAD_PAIR(R6)
	G_CLIP_PAIR
	G_CONSTRAIN
	G_ACC31
	MOVD G_SEC3(R0), R6
	G_LOAD_PAIR(R6)
	G_CLIP_PAIR
	G_CONSTRAIN
	G_ACC31
	G_FINAL
	G_FINAL_CLIP
	G_STORE
	CBNZ R4, u8byte_fused8_rows2
	RET
