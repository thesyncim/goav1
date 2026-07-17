// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

#define L_DST    0
#define L_SRC    8
#define L_DSTSTR 16
#define L_SRCSTR 24
#define L_WIDTH  32
#define L_HEIGHT 40

#define WIDEN16 \
	WORD $0x4cdf7040 \
	WORD $0x2f08a401 \
	WORD $0x6f08a402 \
	WORD $0x4c9fa421

#define WIDEN8 \
	WORD $0x0cdf7040 \
	WORD $0x2f08a401 \
	WORD $0x4c9f7421

// func loadSampleRows8NEONAsm(ctx *loadSampleRows8NEONCtx)
//
// Widening row copy u8 -> u16 for postfilter sample staging:
//     dst[y][x] = uint16(src[y][x])
// Sixteen columns per iteration (ld1 16b, ushll/ushll2, st1 {2x8h}), an
// 8-wide group for the mid tail, and one overlapping 8-wide group aligned to
// the row end for the final 1..7 columns (rewriting identical values). The
// wrapper only routes width >= 8 here.
TEXT ·loadSampleRows8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD L_DST(R0), R1
	MOVD L_SRC(R0), R2
	MOVD L_DSTSTR(R0), R3
	MOVD L_SRCSTR(R0), R5
	MOVD L_WIDTH(R0), R6
	MOVD L_HEIGHT(R0), R7
	LSL  $1, R3, R3 // dst stride in bytes
	CMP  $8, R6
	BEQ  lsrRows8
	CMP  $40, R6
	BEQ  lsrRows40
	CMP  $48, R6
	BEQ  lsrRows48
	CMP  $72, R6
	BEQ  lsrRows72
	CMP  $80, R6
	BEQ  lsrRows80

lsrRowLoop:
	MOVD R2, R8  // src row cursor
	MOVD R1, R9  // dst row cursor
	MOVD R6, R10 // columns remaining

lsrCol16:
	CMP  $16, R10
	BLT  lsrCol8
	WORD $0x4cdf7100 // ld1 {v0.16b}, [x8], #16
	WORD $0x2f08a401 // ushll  v1.8h, v0.8b,  #0
	WORD $0x6f08a402 // ushll2 v2.8h, v0.16b, #0
	WORD $0x4c9fa521 // st1 {v1.8h, v2.8h}, [x9], #32
	SUB  $16, R10, R10
	B    lsrCol16

lsrCol8:
	CMP  $8, R10
	BLT  lsrTail
	WORD $0x0cdf7100 // ld1 {v0.8b}, [x8], #8
	WORD $0x2f08a401 // ushll v1.8h, v0.8b, #0
	WORD $0x4c9f7521 // st1 {v1.8h}, [x9], #16
	SUB  $8, R10, R10

lsrTail:
	CBZ  R10, lsrNextRow
	// Overlapping final 8-wide group ending at the row's last column.
	ADD  R6, R2, R8
	SUB  $8, R8, R8  // src row + width - 8
	ADD  R6<<1, R1, R9
	SUB  $16, R9, R9 // dst row + 2*(width - 8)
	WORD $0x0c407100 // ld1 {v0.8b}, [x8]
	WORD $0x2f08a401 // ushll v1.8h, v0.8b, #0
	WORD $0x4c007521 // st1 {v1.8h}, [x9]

lsrNextRow:
	ADD  R5, R2, R2
	ADD  R3, R1, R1
	SUB  $1, R7, R7
	CBNZ R7, lsrRowLoop

	RET

// The CDEF unit assembler overwhelmingly copies 8-wide left halos, 40/72-wide
// chroma/luma bodies, and 48/80-wide top/bottom halos. Keep those row widths
// on fixed straight-line paths: the generic column loop otherwise spends as
// many scalar compare/branch instructions as it does vector work.
lsrRows8:
	SUB $8, R5, R8
	SUB $16, R3, R9
lsrRows8Loop:
	WIDEN8
	ADD R8, R2
	ADD R9, R1
	SUB $1, R7
	CBNZ R7, lsrRows8Loop
	RET

lsrRows40:
	SUB $40, R5, R8
	SUB $80, R3, R9
lsrRows40Loop:
	WIDEN16
	WIDEN16
	WIDEN8
	ADD R8, R2
	ADD R9, R1
	SUB $1, R7
	CBNZ R7, lsrRows40Loop
	RET

lsrRows48:
	SUB $48, R5, R8
	SUB $96, R3, R9
lsrRows48Loop:
	WIDEN16
	WIDEN16
	WIDEN16
	ADD R8, R2
	ADD R9, R1
	SUB $1, R7
	CBNZ R7, lsrRows48Loop
	RET

lsrRows72:
	SUB $72, R5, R8
	SUB $144, R3, R9
lsrRows72Loop:
	WIDEN16
	WIDEN16
	WIDEN16
	WIDEN16
	WIDEN8
	ADD R8, R2
	ADD R9, R1
	SUB $1, R7
	CBNZ R7, lsrRows72Loop
	RET

lsrRows80:
	SUB $80, R5, R8
	SUB $160, R3, R9
lsrRows80Loop:
	WIDEN16
	WIDEN16
	WIDEN16
	WIDEN16
	WIDEN16
	ADD R8, R2
	ADD R9, R1
	SUB $1, R7
	CBNZ R7, lsrRows80Loop
	RET
