// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// NEON 8-bit-pixel Wiener separable passes (dav1d 8bpc shape: uint8 loads
// widened in-kernel, uint8 saturating stores), sharing the u16 kernels'
// EXT/SMLAL structure from wiener_neon_arm64.s. SIMD instructions the Go
// assembler does not name are emitted via WORD with encodings cross-checked
// against the system assembler. Working vectors are V0-V7 and V16-V31 only,
// so the callee-saved V8-V15 are never touched.
//
// New lane semantics on top of wiener_neon_arm64.s:
//   ld1  {Vn.16b}, [Xm]   load 16 u8 samples (horizontal sliding window)
//   uxtl/uxtl2            zero-extend u8 -> u16 lanes (dav1d wiener h shape)
//   sqxtun/sqxtun2        s32 -> u16 saturating narrow (lower clamp 0)
//   uqxtn                 u16 -> u8 saturating narrow (upper clamp 255)
//   st1  {Vn.8b}, [Xm]    store 8 u8 outputs

// wienerU8NEONHorizCtx field offsets (see wiener_u8_neon_arm64.go).
#define H_DST    0
#define H_SRC    8
#define H_SRCSTR 16
#define H_WIDTH  24
#define H_ROWS   32
#define H_TAPS   40
#define H_SEED   48
#define H_MAXCL  52

// func wienerHorizontalU8NEONAsm(ctx *wienerU8NEONHorizCtx)
//
// Horizontal 7-tap Wiener pass over uint8 source samples. Symmetric taps are
// factored into f0*(s0+s6) + f1*(s1+s5) + f2*(s2+s4) + f3*s3, shortening each
// accumulator from seven widening MACs to four. The wrapper guarantees the
// symmetry and fixed 8-bit round0, allowing SQSHRUN to fuse the final shift and
// lower clamp. The temp output remains the 16-bit intermediate.
TEXT ·wienerHorizontalU8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD H_DST(R0), R1      // temp output base
	MOVD H_SRC(R0), R2      // src tap-window base (row -3, col -3)
	MOVD H_SRCSTR(R0), R5   // src stride (elements == bytes)
	MOVD H_WIDTH(R0), R6    // width (multiple of 8)
	MOVD H_ROWS(R0), R7     // rows = height + 2*WienerHalfwin
	MOVD H_TAPS(R0), R3     // taps ptr (8 int16)
	MOVW H_SEED(R0), R11    // accumulator seed (s32)
	MOVHU H_MAXCL(R0), R13  // maxClamp (u16)

	WORD $0x4c407460       // ld1 {v0.8h}, [x3]    load 8 taps
	WORD $0x4e040d72       // dup v18.4s, w11      seed broadcast (s32 lanes)
	WORD $0x4e020db5       // dup v21.8h, w13      maxClamp broadcast (u16 lanes)

	LSL  $1, R6, R14       // dst row stride in bytes (= width * 2)

hRowLoop:
	CBZ  R7, hDone
	MOVD R1, R10           // temp column cursor
	MOVD R2, R9            // src window cursor
	MOVD R6, R8            // remaining columns

hColLoop:
	WORD $0x4eb21e50       // mov v16.16b, v18.16b  seed lanes 0..3
	WORD $0x4eb21e51       // mov v17.16b, v18.16b  seed lanes 4..7
	WORD $0x4c407121       // ld1 {v1.16b}, [x9]    16 source u8

	WORD $0x6e013022       // ext v2.16b, v1.16b, v1.16b, #6
	WORD $0x2e220024       // uaddl v4.8h, v1.8b, v2.8b          s0+s6
	WORD $0x0f402090       // smlal  v16.4s, v4.4h, v0.h[0]
	WORD $0x4f402091       // smlal2 v17.4s, v4.8h, v0.h[0]
	WORD $0x6e010822       // ext v2.16b, v1.16b, v1.16b, #1
	WORD $0x6e012823       // ext v3.16b, v1.16b, v1.16b, #5
	WORD $0x2e230044       // uaddl v4.8h, v2.8b, v3.8b          s1+s5
	WORD $0x0f502090       // smlal  v16.4s, v4.4h, v0.h[1]
	WORD $0x4f502091       // smlal2 v17.4s, v4.8h, v0.h[1]
	WORD $0x6e011022       // ext v2.16b, v1.16b, v1.16b, #2
	WORD $0x6e012023       // ext v3.16b, v1.16b, v1.16b, #4
	WORD $0x2e230044       // uaddl v4.8h, v2.8b, v3.8b          s2+s4
	WORD $0x0f602090       // smlal  v16.4s, v4.4h, v0.h[2]
	WORD $0x4f602091       // smlal2 v17.4s, v4.8h, v0.h[2]
	WORD $0x6e011822       // ext v2.16b, v1.16b, v1.16b, #3
	WORD $0x2f08a444       // uxtl v4.8h, v2.8b                  s3
	WORD $0x0f702090       // smlal  v16.4s, v4.4h, v0.h[3]
	WORD $0x4f702091       // smlal2 v17.4s, v4.8h, v0.h[3]

	WORD $0x2f1d8614       // sqshrun  v20.4h, v16.4s, #3  round0 + clamp lo 0
	WORD $0x6f1d8634       // sqshrun2 v20.8h, v17.4s, #3
	WORD $0x6e756e94       // umin v20.8h, v20.8h, v21.8h  clamp hi maxClamp
	WORD $0x4c007554       // st1 {v20.8h}, [x10]

	ADD  $16, R10, R10     // temp += 8 u16 (16 bytes)
	ADD  $8, R9, R9        // src window += 8 u8 (8 bytes)
	SUB  $8, R8, R8
	CBNZ R8, hColLoop

	ADD  R14, R1, R1       // temp += rowStride bytes
	ADD  R5, R2, R2        // src += srcStride bytes
	SUB  $1, R7, R7
	CBNZ R7, hRowLoop

hDone:
	RET

// wienerU8NEONVertCtx field offsets (see wiener_u8_neon_arm64.go).
#define V_DST    0
#define V_SRC    8
#define V_DSTSTR 16
#define V_SRCSTR 24
#define V_WIDTH  32
#define V_ROWS   40
#define V_TAPS   48
#define V_SEED   56

// func wienerVerticalU8NEONAsm(ctx *wienerU8NEONVertCtx)
//
// Vertical 7-tap Wiener pass over the u16 temp buffer, writing uint8 output.
// Four adjacent output rows share one ten-row input window instead of loading
// seven rows for each output. Symmetric tap factoring also shortens each lane
// accumulator from seven widening MACs to four; the wrapper guarantees this
// symmetry and fixed 8-bit round1, allowing SQSHRUN+UQXTN to fuse rounding and
// the [0,255] clamp.
TEXT ·wienerVerticalU8NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD V_DST(R0), R1      // dst row base
	MOVD V_SRC(R0), R2      // temp tap-window base (row 0)
	MOVD V_DSTSTR(R0), R4   // dst stride (elements == bytes)
	MOVD V_SRCSTR(R0), R5   // temp stride (elements)
	LSL  $1, R5, R5         // -> bytes
	MOVD V_WIDTH(R0), R6    // width (multiple of 8)
	MOVD V_ROWS(R0), R7     // rows = height
	MOVD V_TAPS(R0), R3     // taps ptr (8 int16)
	MOVW V_SEED(R0), R11    // accumulator seed (s32)

	WORD $0x4c407460       // ld1 {v0.8h}, [x3]    load 8 taps
	WORD $0x4e040d72       // dup v18.4s, w11      seed broadcast
	CMP  $4, R7
	BLT  vTail

vRow4Loop:
	MOVD R1, R10           // dst column cursor
	MOVD R2, R11           // temp row-window base for this output row
	MOVD R6, R8            // remaining columns

vCol4Loop:
	MOVD R11, R9           // R9 walks the 10 shared rows; post-index by R5
	WORD $0x4eb21e50       // mov v16.16b, v18.16b  seed lanes 0..3
	WORD $0x4eb21e51       // mov v17.16b, v18.16b  seed lanes 4..7
	WORD $0x4eb21e53       // mov v19.16b, v18.16b
	WORD $0x4eb21e54       // mov v20.16b, v18.16b
	WORD $0x4eb21e55       // mov v21.16b, v18.16b
	WORD $0x4eb21e56       // mov v22.16b, v18.16b
	WORD $0x4eb21e57       // mov v23.16b, v18.16b
	WORD $0x4eb21e58       // mov v24.16b, v18.16b

	WORD $0x4cc57521       // ld1 {v1.8h}, [x9], x5
	WORD $0x4cc57522       // ld1 {v2.8h}, [x9], x5
	WORD $0x4cc57523       // ld1 {v3.8h}, [x9], x5
	WORD $0x4cc57524       // ld1 {v4.8h}, [x9], x5
	WORD $0x4cc57525       // ld1 {v5.8h}, [x9], x5
	WORD $0x4cc57526       // ld1 {v6.8h}, [x9], x5
	WORD $0x4cc57527       // ld1 {v7.8h}, [x9], x5
	WORD $0x4cc57539       // ld1 {v25.8h}, [x9], x5
	WORD $0x4cc5753a       // ld1 {v26.8h}, [x9], x5
	WORD $0x4cc5753b       // ld1 {v27.8h}, [x9], x5

	// Tap 0: (c0+c6), (c1+c7), (c2+c8), (c3+c9).
	WORD $0x4e67843c // add v28.8h, v1.8h, v7.8h
	WORD $0x4e79845d // add v29.8h, v2.8h, v25.8h
	WORD $0x4e7a847e // add v30.8h, v3.8h, v26.8h
	WORD $0x4e7b849f // add v31.8h, v4.8h, v27.8h
	WORD $0x0f402390 // smlal  v16.4s, v28.4h, v0.h[0]
	WORD $0x0f4023b3 // smlal  v19.4s, v29.4h, v0.h[0]
	WORD $0x0f4023d5 // smlal  v21.4s, v30.4h, v0.h[0]
	WORD $0x0f4023f7 // smlal  v23.4s, v31.4h, v0.h[0]
	WORD $0x4f402391 // smlal2 v17.4s, v28.8h, v0.h[0]
	WORD $0x4f4023b4 // smlal2 v20.4s, v29.8h, v0.h[0]
	WORD $0x4f4023d6 // smlal2 v22.4s, v30.8h, v0.h[0]
	WORD $0x4f4023f8 // smlal2 v24.4s, v31.8h, v0.h[0]

	// Tap 1: (c1+c5), (c2+c6), (c3+c7), (c4+c8).
	WORD $0x4e66845c // add v28.8h, v2.8h, v6.8h
	WORD $0x4e67847d // add v29.8h, v3.8h, v7.8h
	WORD $0x4e79849e // add v30.8h, v4.8h, v25.8h
	WORD $0x4e7a84bf // add v31.8h, v5.8h, v26.8h
	WORD $0x0f502390 // smlal  v16.4s, v28.4h, v0.h[1]
	WORD $0x0f5023b3 // smlal  v19.4s, v29.4h, v0.h[1]
	WORD $0x0f5023d5 // smlal  v21.4s, v30.4h, v0.h[1]
	WORD $0x0f5023f7 // smlal  v23.4s, v31.4h, v0.h[1]
	WORD $0x4f502391 // smlal2 v17.4s, v28.8h, v0.h[1]
	WORD $0x4f5023b4 // smlal2 v20.4s, v29.8h, v0.h[1]
	WORD $0x4f5023d6 // smlal2 v22.4s, v30.8h, v0.h[1]
	WORD $0x4f5023f8 // smlal2 v24.4s, v31.8h, v0.h[1]

	// Tap 2: (c2+c4), (c3+c5), (c4+c6), (c5+c7).
	WORD $0x4e65847c // add v28.8h, v3.8h, v5.8h
	WORD $0x4e66849d // add v29.8h, v4.8h, v6.8h
	WORD $0x4e6784be // add v30.8h, v5.8h, v7.8h
	WORD $0x4e7984df // add v31.8h, v6.8h, v25.8h
	WORD $0x0f602390 // smlal  v16.4s, v28.4h, v0.h[2]
	WORD $0x0f6023b3 // smlal  v19.4s, v29.4h, v0.h[2]
	WORD $0x0f6023d5 // smlal  v21.4s, v30.4h, v0.h[2]
	WORD $0x0f6023f7 // smlal  v23.4s, v31.4h, v0.h[2]
	WORD $0x4f602391 // smlal2 v17.4s, v28.8h, v0.h[2]
	WORD $0x4f6023b4 // smlal2 v20.4s, v29.8h, v0.h[2]
	WORD $0x4f6023d6 // smlal2 v22.4s, v30.8h, v0.h[2]
	WORD $0x4f6023f8 // smlal2 v24.4s, v31.8h, v0.h[2]

	// Adjusted center tap over c3, c4, c5, c6.
	WORD $0x0f702090 // smlal  v16.4s, v4.4h, v0.h[3]
	WORD $0x0f7020b3 // smlal  v19.4s, v5.4h, v0.h[3]
	WORD $0x0f7020d5 // smlal  v21.4s, v6.4h, v0.h[3]
	WORD $0x0f7020f7 // smlal  v23.4s, v7.4h, v0.h[3]
	WORD $0x4f702091 // smlal2 v17.4s, v4.8h, v0.h[3]
	WORD $0x4f7020b4 // smlal2 v20.4s, v5.8h, v0.h[3]
	WORD $0x4f7020d6 // smlal2 v22.4s, v6.8h, v0.h[3]
	WORD $0x4f7020f8 // smlal2 v24.4s, v7.8h, v0.h[3]

	WORD $0x2f158610 // sqshrun  v16.4h, v16.4s, #11
	WORD $0x6f158630 // sqshrun2 v16.8h, v17.4s, #11
	WORD $0x2e214a10 // uqxtn v16.8b, v16.8h
	WORD $0x2f158673 // sqshrun  v19.4h, v19.4s, #11
	WORD $0x6f158693 // sqshrun2 v19.8h, v20.4s, #11
	WORD $0x2e214a73 // uqxtn v19.8b, v19.8h
	WORD $0x2f1586b5 // sqshrun  v21.4h, v21.4s, #11
	WORD $0x6f1586d5 // sqshrun2 v21.8h, v22.4s, #11
	WORD $0x2e214ab5 // uqxtn v21.8b, v21.8h
	WORD $0x2f1586f7 // sqshrun  v23.4h, v23.4s, #11
	WORD $0x6f158717 // sqshrun2 v23.8h, v24.4s, #11
	WORD $0x2e214af7 // uqxtn v23.8b, v23.8h

	MOVD R10, R9
	WORD $0x0c007130 // st1 {v16.8b}, [x9]
	ADD  R4, R9, R9
	WORD $0x0c007133 // st1 {v19.8b}, [x9]
	ADD  R4, R9, R9
	WORD $0x0c007135 // st1 {v21.8b}, [x9]
	ADD  R4, R9, R9
	WORD $0x0c007137 // st1 {v23.8b}, [x9]

	ADD  $8, R10, R10      // dst += 8 u8 (8 bytes)
	ADD  $16, R11, R11     // temp window += 8 u16 (16 bytes)
	SUB  $8, R8, R8
	CBNZ R8, vCol4Loop

	ADD R4<<2, R1, R1
	ADD R5<<2, R2, R2
	SUB $4, R7, R7
	CMP $4, R7
	BGE vRow4Loop

vTail:
	CBZ R7, vDone

vTailRowLoop:
	MOVD R1, R10
	MOVD R2, R11
	MOVD R6, R8

vTailColLoop:
	MOVD R11, R9
	WORD $0x4eb21e50 // mov v16.16b, v18.16b
	WORD $0x4eb21e51 // mov v17.16b, v18.16b
	WORD $0x4cc57521 // ld1 {v1.8h}, [x9], x5
	WORD $0x4cc57522 // ld1 {v2.8h}, [x9], x5
	WORD $0x4cc57523 // ld1 {v3.8h}, [x9], x5
	WORD $0x4cc57524 // ld1 {v4.8h}, [x9], x5
	WORD $0x4cc57525 // ld1 {v5.8h}, [x9], x5
	WORD $0x4cc57526 // ld1 {v6.8h}, [x9], x5
	WORD $0x4cc57527 // ld1 {v7.8h}, [x9], x5
	WORD $0x4e678439 // add v25.8h, v1.8h, v7.8h
	WORD $0x4e66845a // add v26.8h, v2.8h, v6.8h
	WORD $0x4e65847b // add v27.8h, v3.8h, v5.8h
	WORD $0x0f402330 // smlal  v16.4s, v25.4h, v0.h[0]
	WORD $0x4f402331 // smlal2 v17.4s, v25.8h, v0.h[0]
	WORD $0x0f502350 // smlal  v16.4s, v26.4h, v0.h[1]
	WORD $0x4f502351 // smlal2 v17.4s, v26.8h, v0.h[1]
	WORD $0x0f602370 // smlal  v16.4s, v27.4h, v0.h[2]
	WORD $0x4f602371 // smlal2 v17.4s, v27.8h, v0.h[2]
	WORD $0x0f702090 // smlal  v16.4s, v4.4h, v0.h[3]
	WORD $0x4f702091 // smlal2 v17.4s, v4.8h, v0.h[3]
	WORD $0x2f158610 // sqshrun  v16.4h, v16.4s, #11
	WORD $0x6f158630 // sqshrun2 v16.8h, v17.4s, #11
	WORD $0x2e214a10 // uqxtn v16.8b, v16.8h
	WORD $0x0c007150 // st1 {v16.8b}, [x10]

	ADD  $8, R10, R10
	ADD  $16, R11, R11
	SUB  $8, R8, R8
	CBNZ R8, vTailColLoop

	ADD  R4, R1, R1        // dst += dstStride bytes
	ADD  R5, R2, R2        // temp += tempStride bytes
	SUB  $1, R7, R7
	CBNZ R7, vTailRowLoop

vDone:
	RET
