// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// Source-shaped CDEF strength split following dav1d's cdef_filter*_pri entry.
// This 8-wide variant keeps the generic cdefFilterBlock8NEON path for
// secondary-only and both-strength blocks, and removes the per-row
// strength/clipping branches from primary-only blocks.

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

// func cdefFilterBlock8PrimaryNEON(ctx *filterBlockNEONCtx)
TEXT ·cdefFilterBlock8PrimaryNEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD C_DST(R0), R1
	MOVD C_INPUT(R0), R2
	MOVD C_DSTSTR(R0), R3
	LSL  $1, R3
	MOVD C_HEIGHT(R0), R4
	MOVD C_PRISTR(R0), R5
	WORD $0x4e020cb0 // dup v16.8h, w5
	MOVD C_PRISHIFT(R0), R5
	NEG  R5, R5
	WORD $0x4e020cb2 // dup v18.8h, w5
	MOVD C_PRITAP0(R0), R5
	WORD $0x4e020cb4 // dup v20.8h, w5
	MOVD C_PRITAP1(R0), R5
	WORD $0x4e020cb5 // dup v21.8h, w5
	MOVD C_PRI0(R0), R6
	LSL  $1, R6
	MOVD C_PRI1(R0), R7
	LSL  $1, R7
	MOVD $(144*2), R15

primary8_row:
	VLD1 (R2), [V0.H8]
	WORD $0x6e211c21 // eor v1.16b, v1.16b, v1.16b
	WORD $0x6e221c42 // eor v2.16b, v2.16b, v2.16b

	ADD R6, R2, R16
	VLD1 (R16), [V5.H8]
	WORD $0x6e6084a6 // sub v6.8h, v5.8h, v0.8h
	WORD $0x4e60b8c7 // abs v7.8h, v6.8h
	WORD $0x6e7244e8 // ushl v8.8h, v7.8h, v18.8h
	WORD $0x6e688608 // sub v8.8h, v16.8h, v8.8h
	WORD $0x6e291d29 // eor v9.16b, v9.16b, v9.16b
	WORD $0x4e696508 // smax v8.8h, v8.8h, v9.8h
	WORD $0x4e676d08 // smin v8.8h, v8.8h, v7.8h
	WORD $0x4f1104c9 // sshr v9.8h, v6.8h, #15
	WORD $0x6e291d08 // eor v8.16b, v8.16b, v9.16b
	WORD $0x6e698508 // sub v8.8h, v8.8h, v9.8h
	WORD $0x0e748101 // smlal v1.4s, v8.4h, v20.4h
	WORD $0x4e748102 // smlal2 v2.4s, v8.8h, v20.8h

	SUB R6, R2, R16
	VLD1 (R16), [V5.H8]
	WORD $0x6e6084a6 // sub v6.8h, v5.8h, v0.8h
	WORD $0x4e60b8c7 // abs v7.8h, v6.8h
	WORD $0x6e7244e8 // ushl v8.8h, v7.8h, v18.8h
	WORD $0x6e688608 // sub v8.8h, v16.8h, v8.8h
	WORD $0x6e291d29 // eor v9.16b, v9.16b, v9.16b
	WORD $0x4e696508 // smax v8.8h, v8.8h, v9.8h
	WORD $0x4e676d08 // smin v8.8h, v8.8h, v7.8h
	WORD $0x4f1104c9 // sshr v9.8h, v6.8h, #15
	WORD $0x6e291d08 // eor v8.16b, v8.16b, v9.16b
	WORD $0x6e698508 // sub v8.8h, v8.8h, v9.8h
	WORD $0x0e748101 // smlal v1.4s, v8.4h, v20.4h
	WORD $0x4e748102 // smlal2 v2.4s, v8.8h, v20.8h

	ADD R7, R2, R16
	VLD1 (R16), [V5.H8]
	WORD $0x6e6084a6 // sub v6.8h, v5.8h, v0.8h
	WORD $0x4e60b8c7 // abs v7.8h, v6.8h
	WORD $0x6e7244e8 // ushl v8.8h, v7.8h, v18.8h
	WORD $0x6e688608 // sub v8.8h, v16.8h, v8.8h
	WORD $0x6e291d29 // eor v9.16b, v9.16b, v9.16b
	WORD $0x4e696508 // smax v8.8h, v8.8h, v9.8h
	WORD $0x4e676d08 // smin v8.8h, v8.8h, v7.8h
	WORD $0x4f1104c9 // sshr v9.8h, v6.8h, #15
	WORD $0x6e291d08 // eor v8.16b, v8.16b, v9.16b
	WORD $0x6e698508 // sub v8.8h, v8.8h, v9.8h
	WORD $0x0e758101 // smlal v1.4s, v8.4h, v21.4h
	WORD $0x4e758102 // smlal2 v2.4s, v8.8h, v21.8h

	SUB R7, R2, R16
	VLD1 (R16), [V5.H8]
	WORD $0x6e6084a6 // sub v6.8h, v5.8h, v0.8h
	WORD $0x4e60b8c7 // abs v7.8h, v6.8h
	WORD $0x6e7244e8 // ushl v8.8h, v7.8h, v18.8h
	WORD $0x6e688608 // sub v8.8h, v16.8h, v8.8h
	WORD $0x6e291d29 // eor v9.16b, v9.16b, v9.16b
	WORD $0x4e696508 // smax v8.8h, v8.8h, v9.8h
	WORD $0x4e676d08 // smin v8.8h, v8.8h, v7.8h
	WORD $0x4f1104c9 // sshr v9.8h, v6.8h, #15
	WORD $0x6e291d08 // eor v8.16b, v8.16b, v9.16b
	WORD $0x6e698508 // sub v8.8h, v8.8h, v9.8h
	WORD $0x0e758101 // smlal v1.4s, v8.4h, v21.4h
	WORD $0x4e758102 // smlal2 v2.4s, v8.8h, v21.8h

primary8_finalize:
	WORD $0x4f210425 // sshr v5.4s, v1.4s, #31
	WORD $0x4ea58421 // add v1.4s, v1.4s, v5.4s
	WORD $0x4f210445 // sshr v5.4s, v2.4s, #31
	WORD $0x4ea58442 // add v2.4s, v2.4s, v5.4s
	MOVD $8, R16
	VDUP R16, V5.S4
	WORD $0x4ea58421 // add v1.4s, v1.4s, v5.4s
	WORD $0x4ea58442 // add v2.4s, v2.4s, v5.4s
	WORD $0x4f3c0421 // sshr v1.4s, v1.4s, #4
	WORD $0x4f3c0442 // sshr v2.4s, v2.4s, #4
	WORD $0x0e612825 // xtn v5.4h, v1.4s
	WORD $0x4e612845 // xtn2 v5.8h, v2.4s
	WORD $0x4e6084a5 // add v5.8h, v5.8h, v0.8h
	VST1 [V5.H8], (R1)
	ADD R3, R1
	ADD R15, R2
	SUB $1, R4
	CBNZ R4, primary8_row
	RET
