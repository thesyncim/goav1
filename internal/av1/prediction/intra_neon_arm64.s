// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Code generated with the help of an offline NEON encoder; do not edit by hand.
// Every WORD below is the little-endian encoding of the mnemonic in its
// trailing comment, assembled and verified with the system assembler. The
// kernels are bit-exact with the *PureGo references in intra_static.go; the Go
// wrappers in intra_neon_arm64.go restrict them to 8-bit, width-multiple-of-8
// blocks. Working vectors stay in V0-V7 and V16-V31 to avoid clobbering the
// callee-saved V8-V15.

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

// func predictPaethNEONAsm(ctx *paethNEONCtx)
//
// PAETH predictor. Processes 8 output columns per inner iteration as int16
// lanes. base = top + left - topLeft; the three abs diffs use SABD; the libaom
// tie-break order (left, then top, then topLeft) is reproduced with CMGE masks
// and BSL selects.
TEXT ·predictPaethNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD P_DST(R0), R1       // dst row base
	MOVD P_ABOVE(R0), R2     // above[] base (uint16)
	MOVD P_LEFT(R0), R3      // left[] base (uint16)
	MOVD P_DSTSTR(R0), R4    // dst stride (bytes)
	MOVD P_WIDTH(R0), R5     // width
	MOVD P_HEIGHT(R0), R6    // height
	MOVD P_ABOVELEFT(R0), R7 // aboveLeft
	WORD $0x4e020ce3         // dup v3.8h, w7   (topLeft broadcast)

	MOVD ZR, R8 // row index
paethRow:
	CMP   R6, R8
	BGE   paethDone
	MOVHU (R3), R9     // left[row] zero-extended
	WORD  $0x4e020d22  // dup v2.8h, w9   (left[row] broadcast)
	MOVD  R1, R10      // dst column cursor
	MOVD  R2, R11      // above column cursor
	MOVD  R5, R12      // remaining columns
paethCol:
	WORD $0x4cdf7561 // ld1 {v1.8h}, [x11], #16   (8 above samples)
	WORD $0x4e628424 // add v4.8h, v1.8h, v2.8h
	WORD $0x6e638484 // sub v4.8h, v4.8h, v3.8h    (base = top + left - tl)
	WORD $0x4e627485 // sabd v5.8h, v4.8h, v2.8h   (pLeft = |base-left|)
	WORD $0x4e617486 // sabd v6.8h, v4.8h, v1.8h   (pTop = |base-top|)
	WORD $0x4e637487 // sabd v7.8h, v4.8h, v3.8h   (pTopLeft = |base-tl|)
	WORD $0x4e653cd0 // cmge v16.8h, v6.8h, v5.8h  (pTop >= pLeft)
	WORD $0x4e653cf1 // cmge v17.8h, v7.8h, v5.8h  (pTopLeft >= pLeft)
	WORD $0x4e311e10 // and v16.16b, v16.16b, v17.16b (maskLeft)
	WORD $0x4e663cf2 // cmge v18.8h, v7.8h, v6.8h  (pTopLeft >= pTop -> maskTop)
	WORD $0x6e631c32 // bsl v18.16b, v1.16b, v3.16b (maskTop ? top : tl)
	WORD $0x6e721c50 // bsl v16.16b, v2.16b, v18.16b (maskLeft ? left : selectedAbove)
	WORD $0x0e212a10 // xtn v16.8b, v16.8h
	WORD $0x0c9f7150 // st1 {v16.8b}, [x10], #8
	SUB  $8, R12, R12
	CBNZ R12, paethCol

	ADD R4, R1, R1 // dst += stride
	ADD $2, R3, R3 // left++ (uint16)
	ADD $1, R8, R8
	B   paethRow
paethDone:
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

// func predictSmoothNEONAsm(ctx *smoothNEONCtx)
//
// Full SMOOTH predictor. For each row, weightsH[row], left[row], belowPred and
// rightPred are scalars; above[col] and weightsW[col] vary across the row. The
// per-pixel value
//   wH*above + (256-wH)*below + wW*left + (256-wW)*right
// is accumulated in u32 lanes and rounded by (sum + 256) >> 9, identical to
// divideRound(pred, 9).
TEXT ·predictSmoothNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD S_DST(R0), R1
	MOVD S_ABOVE(R0), R2
	MOVD S_LEFT(R0), R3
	MOVD S_WEIGHTSW(R0), R13
	MOVD S_WEIGHTSH(R0), R14
	MOVD S_DSTSTR(R0), R4
	MOVD S_WIDTH(R0), R5
	MOVD S_HEIGHT(R0), R6
	MOVD S_BELOW(R0), R7
	MOVD S_RIGHT(R0), R15
	WORD $0x4f00a43e // movi v30.8h, #1, lsl #8  (256)
	WORD $0x4e020ce3 // dup v3.8h, w7   (belowPred broadcast)
	WORD $0x4e020de4 // dup v4.8h, w15  (rightPred broadcast)

	MOVD ZR, R8 // row index
smoothRow:
	CMP   R6, R8
	BGE   smoothDone
	MOVHU (R14), R9  // weightsH[row]
	MOVHU (R3), R16  // left[row]
	WORD  $0x4e020d25 // dup v5.8h, w9   (wH)
	WORD  $0x6e6587dd // sub v29.8h, v30.8h, v5.8h  (256-wH)
	WORD  $0x4e020e06 // dup v6.8h, w16  (left[row])
	MOVD  R1, R10     // dst cursor
	MOVD  R2, R11     // above cursor
	MOVD  R13, R17    // weightsW cursor
	MOVD  R5, R12     // remaining cols
smoothCol:
	WORD $0x4cdf7561 // ld1 {v1.8h}, [x11], #16   (8 above)
	WORD $0x4cdf7628 // ld1 {v8.8h}, [x17], #16   (8 weightsW)
	WORD $0x6e6887dc // sub v28.8h, v30.8h, v8.8h  (256-wW)
	WORD $0x2e65c030 // umull  v16.4s, v1.4h, v5.4h    (above*wH)
	WORD $0x6e65c031 // umull2 v17.4s, v1.8h, v5.8h
	WORD $0x2e7d8070 // umlal  v16.4s, v3.4h, v29.4h   (below*(256-wH))
	WORD $0x6e7d8071 // umlal2 v17.4s, v3.8h, v29.8h
	WORD $0x2e6880d0 // umlal  v16.4s, v6.4h, v8.4h    (left*wW)
	WORD $0x6e6880d1 // umlal2 v17.4s, v6.8h, v8.8h
	WORD $0x2e7c8090 // umlal  v16.4s, v4.4h, v28.4h   (right*(256-wW))
	WORD $0x6e7c8091 // umlal2 v17.4s, v4.8h, v28.8h
	WORD $0x6f372610 // urshr v16.4s, v16.4s, #9
	WORD $0x6f372631 // urshr v17.4s, v17.4s, #9
	WORD $0x0e612a10 // xtn  v16.4h, v16.4s
	WORD $0x4e612a30 // xtn2 v16.8h, v17.4s
	WORD $0x0e212a10 // xtn  v16.8b, v16.8h
	WORD $0x0c9f7150 // st1 {v16.8b}, [x10], #8
	SUB  $8, R12, R12
	CBNZ R12, smoothCol

	ADD R4, R1, R1 // dst += stride
	ADD $2, R3, R3 // left++
	ADD $2, R14, R14 // weightsH++
	ADD $1, R8, R8
	B   smoothRow
smoothDone:
	RET

// smooth1DNEONCtx field offsets (see intra_neon_arm64.go).
#define D_DST 0
#define D_PRIMARY 8
#define D_WEIGHTS 16
#define D_DSTSTR 24
#define D_WIDTH 32
#define D_HEIGHT 40
#define D_SECONDARY 48

// func predictSmoothVerticalNEONAsm(ctx *smooth1DNEONCtx)
//
// SMOOTH_V predictor. primary=above[] (varies per col), weights is row-indexed,
// secondary=belowPred. pred = w*above + (256-w)*below, rounded by (sum+128)>>8.
TEXT ·predictSmoothVerticalNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD D_DST(R0), R1
	MOVD D_PRIMARY(R0), R2   // above[]
	MOVD D_WEIGHTS(R0), R14  // weights (row-indexed)
	MOVD D_DSTSTR(R0), R4
	MOVD D_WIDTH(R0), R5
	MOVD D_HEIGHT(R0), R6
	MOVD D_SECONDARY(R0), R7 // belowPred
	WORD $0x4f00a43e         // movi v30.8h, #256
	WORD $0x4e020ce3         // dup v3.8h, w7   (belowPred)

	MOVD ZR, R8 // row index
svRow:
	CMP   R6, R8
	BGE   svDone
	MOVHU (R14), R9   // weights[row]
	WORD  $0x4e020d25 // dup v5.8h, w9   (w)
	WORD  $0x6e6587dd // sub v29.8h, v30.8h, v5.8h  (256-w)
	MOVD  R1, R10
	MOVD  R2, R11
	MOVD  R5, R12
svCol:
	WORD $0x4cdf7561 // ld1 {v1.8h}, [x11], #16   (8 above)
	WORD $0x2e65c030 // umull  v16.4s, v1.4h, v5.4h
	WORD $0x6e65c031 // umull2 v17.4s, v1.8h, v5.8h
	WORD $0x2e7d8070 // umlal  v16.4s, v3.4h, v29.4h
	WORD $0x6e7d8071 // umlal2 v17.4s, v3.8h, v29.8h
	WORD $0x6f382610 // urshr v16.4s, v16.4s, #8
	WORD $0x6f382631 // urshr v17.4s, v17.4s, #8
	WORD $0x0e612a10 // xtn  v16.4h, v16.4s
	WORD $0x4e612a30 // xtn2 v16.8h, v17.4s
	WORD $0x0e212a10 // xtn  v16.8b, v16.8h
	WORD $0x0c9f7150 // st1 {v16.8b}, [x10], #8
	SUB  $8, R12, R12
	CBNZ R12, svCol

	ADD R4, R1, R1
	ADD $2, R14, R14 // weights++ (row-indexed)
	ADD $1, R8, R8
	B   svRow
svDone:
	RET

// func predictSmoothHorizontalNEONAsm(ctx *smooth1DNEONCtx)
//
// SMOOTH_H predictor. primary=left[] (row-indexed scalar), weights is
// col-indexed (varies per col), secondary=rightPred. pred = w*left + (256-w)*
// right, rounded by (sum+128)>>8.
TEXT ·predictSmoothHorizontalNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD D_DST(R0), R1
	MOVD D_PRIMARY(R0), R3   // left[]
	MOVD D_WEIGHTS(R0), R13  // weights (col-indexed)
	MOVD D_DSTSTR(R0), R4
	MOVD D_WIDTH(R0), R5
	MOVD D_HEIGHT(R0), R6
	MOVD D_SECONDARY(R0), R7 // rightPred
	WORD $0x4f00a43e         // movi v30.8h, #256
	WORD $0x4e020ce4         // dup v4.8h, w7   (rightPred)

	MOVD ZR, R8 // row index
shRow:
	CMP   R6, R8
	BGE   shDone
	MOVHU (R3), R16   // left[row]
	WORD  $0x4e020e06 // dup v6.8h, w16  (left[row])
	MOVD  R1, R10
	MOVD  R13, R17    // weights cursor
	MOVD  R5, R12
shCol:
	WORD $0x4cdf7628 // ld1 {v8.8h}, [x17], #16   (8 weightsW)
	WORD $0x6e6887dc // sub v28.8h, v30.8h, v8.8h  (256-w)
	WORD $0x2e68c0d0 // umull  v16.4s, v6.4h, v8.4h    (left*w)
	WORD $0x6e68c0d1 // umull2 v17.4s, v6.8h, v8.8h
	WORD $0x2e7c8090 // umlal  v16.4s, v4.4h, v28.4h   (right*(256-w))
	WORD $0x6e7c8091 // umlal2 v17.4s, v4.8h, v28.8h
	WORD $0x6f382610 // urshr v16.4s, v16.4s, #8
	WORD $0x6f382631 // urshr v17.4s, v17.4s, #8
	WORD $0x0e612a10 // xtn  v16.4h, v16.4s
	WORD $0x4e612a30 // xtn2 v16.8h, v17.4s
	WORD $0x0e212a10 // xtn  v16.8b, v16.8h
	WORD $0x0c9f7150 // st1 {v16.8b}, [x10], #8
	SUB  $8, R12, R12
	CBNZ R12, shCol

	ADD R4, R1, R1
	ADD $2, R3, R3 // left++ (row-indexed)
	ADD $1, R8, R8
	B   shRow
shDone:
	RET
