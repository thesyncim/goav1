// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Code generated with the help of an offline NEON encoder; do not edit by hand.
// Every WORD below is the little-endian encoding of the mnemonic in its
// trailing comment, assembled and verified with the system assembler (as
// -arch arm64). These are the high-bit-depth (10/12-bit, bytesPerSample==2)
// counterparts of the 8-bit PAETH/SMOOTH kernels in intra_neon_arm64.s: the
// samples are already int16-native, so the arithmetic is identical to the
// 8-bit lanes (PAETH base = top+left-topLeft still fits int16 for 12-bit; the
// SMOOTH u32 accumulators still fit) and only the final narrowing/store changes
// -- the result stays in uint16 lanes and is written with a 16-byte st1.8h
// instead of narrowing to bytes. They are bit-exact with the *PureGo references
// in intra_static.go. The mnemonics mirror dav1d's ipred16.S paeth/smooth
// hbd kernels. Working vectors stay in V0-V7 and V16-V31.

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

// func predictPaeth16NEONAsm(ctx *paethNEONCtx)
//
// High-bit-depth PAETH. Processes 8 output columns per inner iteration as int16
// lanes exactly like the 8-bit kernel; base = top + left - topLeft, the three
// abs diffs use SABD and the tie-break order (left, then top, then topLeft) is
// reproduced with CMGE masks and BSL selects. The 8 uint16 results are stored
// directly (st1 {v16.8h}) rather than narrowed to bytes.
TEXT ·predictPaeth16NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD P_DST(R0), R1
	MOVD P_ABOVE(R0), R2
	MOVD P_LEFT(R0), R3
	MOVD P_DSTSTR(R0), R4
	MOVD P_WIDTH(R0), R5
	MOVD P_HEIGHT(R0), R6
	MOVD P_ABOVELEFT(R0), R7
	WORD $0x4e020ce3 // dup v3.8h, w7   (topLeft broadcast)

	MOVD ZR, R8 // row index
paeth16Row:
	CMP   R6, R8
	BGE   paeth16Done
	MOVHU (R3), R9    // left[row] zero-extended
	WORD  $0x4e020d22 // dup v2.8h, w9   (left[row] broadcast)
	MOVD  R1, R10     // dst column cursor
	MOVD  R2, R11     // above column cursor
	MOVD  R5, R12     // remaining columns
paeth16Col:
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
	WORD $0x4c9f7550 // st1 {v16.8h}, [x10], #16
	SUB  $8, R12, R12
	CBNZ R12, paeth16Col

	ADD R4, R1, R1 // dst += stride (bytes)
	ADD $2, R3, R3 // left++ (uint16)
	ADD $1, R8, R8
	B   paeth16Row
paeth16Done:
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

// func predictSmooth16NEONAsm(ctx *smoothNEONCtx)
//
// High-bit-depth full SMOOTH. Accumulates wH*above + (256-wH)*below + wW*left +
// (256-wW)*right in u32 lanes and rounds with (sum + 256) >> 9, identical to
// divideRound(pred, 9); the u16 results are stored directly.
TEXT ·predictSmooth16NEONAsm(SB), NOSPLIT, $0-8
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
smooth16Row:
	CMP   R6, R8
	BGE   smooth16Done
	MOVHU (R14), R9   // weightsH[row]
	MOVHU (R3), R16   // left[row]
	WORD  $0x4e020d25 // dup v5.8h, w9   (wH)
	WORD  $0x6e6587dd // sub v29.8h, v30.8h, v5.8h  (256-wH)
	WORD  $0x4e020e06 // dup v6.8h, w16  (left[row])
	MOVD  R1, R10     // dst cursor
	MOVD  R2, R11     // above cursor
	MOVD  R13, R17    // weightsW cursor
	MOVD  R5, R12     // remaining cols
smooth16Col:
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
	WORD $0x4c9f7550 // st1 {v16.8h}, [x10], #16
	SUB  $8, R12, R12
	CBNZ R12, smooth16Col

	ADD R4, R1, R1   // dst += stride
	ADD $2, R3, R3   // left++
	ADD $2, R14, R14 // weightsH++
	ADD $1, R8, R8
	B   smooth16Row
smooth16Done:
	RET

// smooth1DNEONCtx field offsets (see intra_neon_arm64.go).
#define D_DST 0
#define D_PRIMARY 8
#define D_WEIGHTS 16
#define D_DSTSTR 24
#define D_WIDTH 32
#define D_HEIGHT 40
#define D_SECONDARY 48

// func predictSmoothVertical16NEONAsm(ctx *smooth1DNEONCtx)
//
// High-bit-depth SMOOTH_V. pred = w*above + (256-w)*below rounded by
// (sum+128)>>8; u16 results stored directly.
TEXT ·predictSmoothVertical16NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD D_DST(R0), R1
	MOVD D_PRIMARY(R0), R2  // above[]
	MOVD D_WEIGHTS(R0), R14 // weights (row-indexed)
	MOVD D_DSTSTR(R0), R4
	MOVD D_WIDTH(R0), R5
	MOVD D_HEIGHT(R0), R6
	MOVD D_SECONDARY(R0), R7 // belowPred
	WORD $0x4f00a43e         // movi v30.8h, #256
	WORD $0x4e020ce3         // dup v3.8h, w7   (belowPred)

	MOVD ZR, R8 // row index
sv16Row:
	CMP   R6, R8
	BGE   sv16Done
	MOVHU (R14), R9   // weights[row]
	WORD  $0x4e020d25 // dup v5.8h, w9   (w)
	WORD  $0x6e6587dd // sub v29.8h, v30.8h, v5.8h  (256-w)
	MOVD  R1, R10
	MOVD  R2, R11
	MOVD  R5, R12
sv16Col:
	WORD $0x4cdf7561 // ld1 {v1.8h}, [x11], #16   (8 above)
	WORD $0x2e65c030 // umull  v16.4s, v1.4h, v5.4h
	WORD $0x6e65c031 // umull2 v17.4s, v1.8h, v5.8h
	WORD $0x2e7d8070 // umlal  v16.4s, v3.4h, v29.4h
	WORD $0x6e7d8071 // umlal2 v17.4s, v3.8h, v29.8h
	WORD $0x6f382610 // urshr v16.4s, v16.4s, #8
	WORD $0x6f382631 // urshr v17.4s, v17.4s, #8
	WORD $0x0e612a10 // xtn  v16.4h, v16.4s
	WORD $0x4e612a30 // xtn2 v16.8h, v17.4s
	WORD $0x4c9f7550 // st1 {v16.8h}, [x10], #16
	SUB  $8, R12, R12
	CBNZ R12, sv16Col

	ADD R4, R1, R1
	ADD $2, R14, R14 // weights++ (row-indexed)
	ADD $1, R8, R8
	B   sv16Row
sv16Done:
	RET

// func predictSmoothHorizontal16NEONAsm(ctx *smooth1DNEONCtx)
//
// High-bit-depth SMOOTH_H. pred = w*left + (256-w)*right rounded by
// (sum+128)>>8; u16 results stored directly.
TEXT ·predictSmoothHorizontal16NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD D_DST(R0), R1
	MOVD D_PRIMARY(R0), R3  // left[]
	MOVD D_WEIGHTS(R0), R13 // weights (col-indexed)
	MOVD D_DSTSTR(R0), R4
	MOVD D_WIDTH(R0), R5
	MOVD D_HEIGHT(R0), R6
	MOVD D_SECONDARY(R0), R7 // rightPred
	WORD $0x4f00a43e         // movi v30.8h, #256
	WORD $0x4e020ce4         // dup v4.8h, w7   (rightPred)

	MOVD ZR, R8 // row index
sh16Row:
	CMP   R6, R8
	BGE   sh16Done
	MOVHU (R3), R16   // left[row]
	WORD  $0x4e020e06 // dup v6.8h, w16  (left[row])
	MOVD  R1, R10
	MOVD  R13, R17    // weights cursor
	MOVD  R5, R12
sh16Col:
	WORD $0x4cdf7628 // ld1 {v8.8h}, [x17], #16   (8 weights)
	WORD $0x6e6887dc // sub v28.8h, v30.8h, v8.8h  (256-w)
	WORD $0x2e68c0d0 // umull  v16.4s, v6.4h, v8.4h    (left*w)
	WORD $0x6e68c0d1 // umull2 v17.4s, v6.8h, v8.8h
	WORD $0x2e7c8090 // umlal  v16.4s, v4.4h, v28.4h   (right*(256-w))
	WORD $0x6e7c8091 // umlal2 v17.4s, v4.8h, v28.8h
	WORD $0x6f382610 // urshr v16.4s, v16.4s, #8
	WORD $0x6f382631 // urshr v17.4s, v17.4s, #8
	WORD $0x0e612a10 // xtn  v16.4h, v16.4s
	WORD $0x4e612a30 // xtn2 v16.8h, v17.4s
	WORD $0x4c9f7550 // st1 {v16.8h}, [x10], #16
	SUB  $8, R12, R12
	CBNZ R12, sh16Col

	ADD R4, R1, R1
	ADD $2, R3, R3 // left++ (row-indexed)
	ADD $1, R8, R8
	B   sh16Row
sh16Done:
	RET
