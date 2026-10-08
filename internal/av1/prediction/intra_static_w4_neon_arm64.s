// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Code generated with the help of an offline NEON encoder; do not edit by hand.
// Every WORD below is the little-endian encoding of the mnemonic in its
// trailing comment, assembled and verified with the system assembler (as
// -arch arm64). These are the width==4 (8-bit) counterparts of the 8-wide
// PAETH/SMOOTH kernels in intra_neon_arm64.s. TX_4X4 is the most frequent block
// shape on all-intra content, so the scalar fallback for width 4 is hot. Like
// dav1d's ipred.S w4 paths, two output rows (4 columns each) are packed into a
// single 8-lane vector -- the low half is row r, the high half is row r+1 -- so
// the whole block is produced in height/2 iterations of one asm call, amortising
// the call overhead across the block. The above row (and, for SMOOTH, the width
// weights) is broadcast into both halves once; the per-row scalars (left, the
// height weight) are loaded as a {r,r+1} pair each iteration. The arithmetic is
// identical to the 8-wide kernels and bit-exact with the *PureGo references in
// intra_static.go; the Go wrappers gate these on width==4 and an even height.
// Working vectors stay in V0-V7 and V16-V31.

//go:build arm64 && !purego

#include "textflag.h"

// paethNEONCtx field offsets (see intra_neon_arm64.go).
#define P_DST 0
#define P_ABOVE 8
#define P_LEFT 16
#define P_DSTSTR 24
#define P_WIDTH 32
#define P_HEIGHT 40
#define P_ABOVELEFT 48

// func predictPaethW4NEONAsm(ctx *paethNEONCtx)
//
// Width-4 PAETH. above[0..3] is broadcast into both halves of v1; each iteration
// loads left[row] and left[row+1] and builds the left pair v2. base, the three
// SABD diffs and the CMGE/BSL tie-break are identical to the 8-wide kernel; the
// 8 result bytes split into row r (lane word 0) and row r+1 (lane word 1).
TEXT ·predictPaethW4NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD P_DST(R0), R1
	MOVD P_ABOVE(R0), R2
	MOVD P_LEFT(R0), R3
	MOVD P_DSTSTR(R0), R4
	MOVD P_HEIGHT(R0), R6
	MOVD P_ABOVELEFT(R0), R7
	WORD $0x0c407441 // ld1 {v1.4h}, [x2]     (above[0..3])
	WORD $0x4e080421 // dup v1.2d, v1.d[0]    (above pair -> both halves)
	WORD $0x4e020ce3 // dup v3.8h, w7         (topLeft broadcast)

	MOVD ZR, R8 // row index
paethW4Row:
	CMP   R6, R8
	BGE   paethW4Done
	MOVHU (R3), R9    // left[row]
	MOVHU 2(R3), R12  // left[row+1]
	WORD  $0x0e020d22 // dup v2.4h, w9
	WORD  $0x0e020d96 // dup v22.4h, w12
	WORD  $0x6e1806c2 // mov v2.d[1], v22.d[0]  (left pair {L_r x4, L_r1 x4})
	MOVD  R1, R10     // dst_r
	ADD   R4, R1, R11 // dst_r1 = dst_r + stride
	WORD  $0x4e628424 // add v4.8h, v1.8h, v2.8h
	WORD  $0x6e638484 // sub v4.8h, v4.8h, v3.8h    (base = top + left - tl)
	WORD  $0x4e627485 // sabd v5.8h, v4.8h, v2.8h   (pLeft)
	WORD  $0x4e617486 // sabd v6.8h, v4.8h, v1.8h   (pTop)
	WORD  $0x4e637487 // sabd v7.8h, v4.8h, v3.8h   (pTopLeft)
	WORD  $0x4e653cd0 // cmge v16.8h, v6.8h, v5.8h  (pTop >= pLeft)
	WORD  $0x4e653cf1 // cmge v17.8h, v7.8h, v5.8h  (pTopLeft >= pLeft)
	WORD  $0x4e311e10 // and v16.16b, v16.16b, v17.16b (maskLeft)
	WORD  $0x4e663cf2 // cmge v18.8h, v7.8h, v6.8h  (maskTop)
	WORD  $0x6e631c32 // bsl v18.16b, v1.16b, v3.16b (maskTop ? top : tl)
	WORD  $0x6e721c50 // bsl v16.16b, v2.16b, v18.16b (maskLeft ? left : ...)
	WORD  $0x0e212a10 // xtn v16.8b, v16.8h
	WORD  $0x0d008150 // st1 {v16.s}[0], [x10]   (row r cols 0..3)
	WORD  $0x0d009170 // st1 {v16.s}[1], [x11]   (row r+1 cols 0..3)
	ADD   R4, R11, R1 // dst += 2*stride
	ADD   $4, R3, R3  // left += 2 rows (uint16)
	ADD   $2, R8, R8
	B     paethW4Row
paethW4Done:
	RET

// smoothNEONCtx field offsets (see intra_neon_arm64.go).
#define S_DST 0
#define S_ABOVE 8
#define S_LEFT 16
#define S_WEIGHTSW 24
#define S_WEIGHTSH 32
#define S_DSTSTR 40
#define S_WIDTH 48
#define S_HEIGHT 56
#define S_BELOW 64
#define S_RIGHT 72

// func predictSmoothW4NEONAsm(ctx *smoothNEONCtx)
//
// Width-4 full SMOOTH. above (v1) and weightsW (v8) are broadcast into both
// halves once; per iteration the wH (v5) and left (v6) pairs carry row r in the
// low half and row r+1 in the high half. Accumulate in u32 lanes and round with
// (sum+256)>>9, identical to divideRound(pred, 9).
TEXT ·predictSmoothW4NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD S_DST(R0), R1
	MOVD S_ABOVE(R0), R2
	MOVD S_LEFT(R0), R3
	MOVD S_WEIGHTSW(R0), R13
	MOVD S_WEIGHTSH(R0), R14
	MOVD S_DSTSTR(R0), R4
	MOVD S_HEIGHT(R0), R6
	MOVD S_BELOW(R0), R7
	MOVD S_RIGHT(R0), R15
	WORD $0x0c407441 // ld1 {v1.4h}, [x2]     (above[0..3])
	WORD $0x4e080421 // dup v1.2d, v1.d[0]    (above pair)
	WORD $0x0c4075a8 // ld1 {v8.4h}, [x13]    (weightsW[0..3])
	WORD $0x4e080508 // dup v8.2d, v8.d[0]    (weightsW pair)
	WORD $0x4f00a43e // movi v30.8h, #1, lsl #8  (256)
	WORD $0x6e6887dc // sub v28.8h, v30.8h, v8.8h  (256-wW)
	WORD $0x4e020ce3 // dup v3.8h, w7   (belowPred)
	WORD $0x4e020de4 // dup v4.8h, w15  (rightPred)

	MOVD ZR, R8 // row index
smW4Row:
	CMP   R6, R8
	BGE   smW4Done
	MOVHU (R14), R9   // weightsH[row]
	MOVHU 2(R14), R12 // weightsH[row+1]
	WORD  $0x0e020d25 // dup v5.4h, w9
	WORD  $0x0e020d96 // dup v22.4h, w12
	WORD  $0x6e1806c5 // mov v5.d[1], v22.d[0]   (wH pair)
	WORD  $0x6e6587dd // sub v29.8h, v30.8h, v5.8h  (256-wH)
	MOVHU (R3), R16   // left[row]
	MOVHU 2(R3), R17  // left[row+1]
	WORD  $0x0e020e06 // dup v6.4h, w16
	WORD  $0x0e020e37 // dup v23.4h, w17
	WORD  $0x6e1806e6 // mov v6.d[1], v23.d[0]   (left pair)
	MOVD  R1, R10
	ADD   R4, R1, R11
	WORD  $0x2e65c030 // umull  v16.4s, v1.4h, v5.4h    (above*wH)
	WORD  $0x6e65c031 // umull2 v17.4s, v1.8h, v5.8h
	WORD  $0x2e7d8070 // umlal  v16.4s, v3.4h, v29.4h   (below*(256-wH))
	WORD  $0x6e7d8071 // umlal2 v17.4s, v3.8h, v29.8h
	WORD  $0x2e6880d0 // umlal  v16.4s, v6.4h, v8.4h    (left*wW)
	WORD  $0x6e6880d1 // umlal2 v17.4s, v6.8h, v8.8h
	WORD  $0x2e7c8090 // umlal  v16.4s, v4.4h, v28.4h   (right*(256-wW))
	WORD  $0x6e7c8091 // umlal2 v17.4s, v4.8h, v28.8h
	WORD  $0x6f372610 // urshr v16.4s, v16.4s, #9
	WORD  $0x6f372631 // urshr v17.4s, v17.4s, #9
	WORD  $0x0e612a10 // xtn  v16.4h, v16.4s
	WORD  $0x4e612a30 // xtn2 v16.8h, v17.4s
	WORD  $0x0e212a10 // xtn  v16.8b, v16.8h
	WORD  $0x0d008150 // st1 {v16.s}[0], [x10]
	WORD  $0x0d009170 // st1 {v16.s}[1], [x11]
	ADD   R4, R11, R1 // dst += 2*stride
	ADD   $4, R3, R3  // left += 2 rows
	ADD   $4, R14, R14 // weightsH += 2 rows
	ADD   $2, R8, R8
	B     smW4Row
smW4Done:
	RET

// smooth1DNEONCtx field offsets (see intra_neon_arm64.go).
#define D_DST 0
#define D_PRIMARY 8
#define D_WEIGHTS 16
#define D_DSTSTR 24
#define D_WIDTH 32
#define D_HEIGHT 40
#define D_SECONDARY 48

// func predictSmoothVerticalW4NEONAsm(ctx *smooth1DNEONCtx)
//
// Width-4 SMOOTH_V. above (v1) broadcast into both halves; per iteration the wH
// pair (v5) carries the two rows. pred = w*above + (256-w)*below, (sum+128)>>8.
TEXT ·predictSmoothVerticalW4NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD D_DST(R0), R1
	MOVD D_PRIMARY(R0), R2  // above[]
	MOVD D_WEIGHTS(R0), R14 // weights (row-indexed)
	MOVD D_DSTSTR(R0), R4
	MOVD D_HEIGHT(R0), R6
	MOVD D_SECONDARY(R0), R7 // belowPred
	WORD $0x0c407441         // ld1 {v1.4h}, [x2]
	WORD $0x4e080421         // dup v1.2d, v1.d[0]  (above pair)
	WORD $0x4f00a43e         // movi v30.8h, #256
	WORD $0x4e020ce3         // dup v3.8h, w7   (belowPred)

	MOVD ZR, R8 // row index
svW4Row:
	CMP   R6, R8
	BGE   svW4Done
	MOVHU (R14), R9   // weights[row]
	MOVHU 2(R14), R12 // weights[row+1]
	WORD  $0x0e020d25 // dup v5.4h, w9
	WORD  $0x0e020d96 // dup v22.4h, w12
	WORD  $0x6e1806c5 // mov v5.d[1], v22.d[0]   (wH pair)
	WORD  $0x6e6587dd // sub v29.8h, v30.8h, v5.8h  (256-w)
	MOVD  R1, R10
	ADD   R4, R1, R11
	WORD  $0x2e65c030 // umull  v16.4s, v1.4h, v5.4h
	WORD  $0x6e65c031 // umull2 v17.4s, v1.8h, v5.8h
	WORD  $0x2e7d8070 // umlal  v16.4s, v3.4h, v29.4h
	WORD  $0x6e7d8071 // umlal2 v17.4s, v3.8h, v29.8h
	WORD  $0x6f382610 // urshr v16.4s, v16.4s, #8
	WORD  $0x6f382631 // urshr v17.4s, v17.4s, #8
	WORD  $0x0e612a10 // xtn  v16.4h, v16.4s
	WORD  $0x4e612a30 // xtn2 v16.8h, v17.4s
	WORD  $0x0e212a10 // xtn  v16.8b, v16.8h
	WORD  $0x0d008150 // st1 {v16.s}[0], [x10]
	WORD  $0x0d009170 // st1 {v16.s}[1], [x11]
	ADD   R4, R11, R1  // dst += 2*stride
	ADD   $4, R14, R14 // weights += 2 rows
	ADD   $2, R8, R8
	B     svW4Row
svW4Done:
	RET

// func predictSmoothHorizontalW4NEONAsm(ctx *smooth1DNEONCtx)
//
// Width-4 SMOOTH_H. weightsW (v8) broadcast into both halves; per iteration the
// left pair (v6) carries the two rows. pred = w*left + (256-w)*right,
// (sum+128)>>8.
TEXT ·predictSmoothHorizontalW4NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD D_DST(R0), R1
	MOVD D_PRIMARY(R0), R3  // left[]
	MOVD D_WEIGHTS(R0), R13 // weights (col-indexed)
	MOVD D_DSTSTR(R0), R4
	MOVD D_HEIGHT(R0), R6
	MOVD D_SECONDARY(R0), R7 // rightPred
	WORD $0x0c4075a8         // ld1 {v8.4h}, [x13]   (weightsW[0..3])
	WORD $0x4e080508         // dup v8.2d, v8.d[0]   (weightsW pair)
	WORD $0x4f00a43e         // movi v30.8h, #256
	WORD $0x6e6887dc         // sub v28.8h, v30.8h, v8.8h  (256-w)
	WORD $0x4e020ce4         // dup v4.8h, w7   (rightPred)

	MOVD ZR, R8 // row index
shW4Row:
	CMP   R6, R8
	BGE   shW4Done
	MOVHU (R3), R16   // left[row]
	MOVHU 2(R3), R17  // left[row+1]
	WORD  $0x0e020e06 // dup v6.4h, w16
	WORD  $0x0e020e37 // dup v23.4h, w17
	WORD  $0x6e1806e6 // mov v6.d[1], v23.d[0]   (left pair)
	MOVD  R1, R10
	ADD   R4, R1, R11
	WORD  $0x2e68c0d0 // umull  v16.4s, v6.4h, v8.4h    (left*w)
	WORD  $0x6e68c0d1 // umull2 v17.4s, v6.8h, v8.8h
	WORD  $0x2e7c8090 // umlal  v16.4s, v4.4h, v28.4h   (right*(256-w))
	WORD  $0x6e7c8091 // umlal2 v17.4s, v4.8h, v28.8h
	WORD  $0x6f382610 // urshr v16.4s, v16.4s, #8
	WORD  $0x6f382631 // urshr v17.4s, v17.4s, #8
	WORD  $0x0e612a10 // xtn  v16.4h, v16.4s
	WORD  $0x4e612a30 // xtn2 v16.8h, v17.4s
	WORD  $0x0e212a10 // xtn  v16.8b, v16.8h
	WORD  $0x0d008150 // st1 {v16.s}[0], [x10]
	WORD  $0x0d009170 // st1 {v16.s}[1], [x11]
	ADD   R4, R11, R1 // dst += 2*stride
	ADD   $4, R3, R3  // left += 2 rows
	ADD   $2, R8, R8
	B     shW4Row
shW4Done:
	RET
