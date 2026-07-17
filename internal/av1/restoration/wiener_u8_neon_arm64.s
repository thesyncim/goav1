// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// NEON 8-bit-pixel Wiener separable passes (dav1d 8bpc shape: uint8 loads
// widened in-kernel, uint8 saturating stores), sharing the u16 kernels'
// EXT/SMLAL structure from wiener_neon_arm64.s. SIMD instructions the Go
// assembler does not name are emitted via WORD with encodings cross-checked
// against the system assembler. Working vectors are V0-V7 and V16-V21 only,
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
	WORD $0x2f08a422       // uxtl  v2.8h, v1.8b    lanes 0..7  -> u16
	WORD $0x6f08a423       // uxtl2 v3.8h, v1.16b   lanes 8..15 -> u16

	WORD $0x6e036044       // ext v4.16b, v2.16b, v3.16b, #12  s6..s13
	WORD $0x4e628484       // add v4.8h, v4.8h, v2.8h            s0+s6
	WORD $0x0f402090       // smlal  v16.4s, v4.4h, v0.h[0]
	WORD $0x4f402091       // smlal2 v17.4s, v4.8h, v0.h[0]
	WORD $0x6e031044       // ext v4.16b, v2.16b, v3.16b, #2   s1..s8
	WORD $0x6e035045       // ext v5.16b, v2.16b, v3.16b, #10  s5..s12
	WORD $0x4e658484       // add v4.8h, v4.8h, v5.8h            s1+s5
	WORD $0x0f502090       // smlal  v16.4s, v4.4h, v0.h[1]
	WORD $0x4f502091       // smlal2 v17.4s, v4.8h, v0.h[1]
	WORD $0x6e032044       // ext v4.16b, v2.16b, v3.16b, #4   s2..s9
	WORD $0x6e034045       // ext v5.16b, v2.16b, v3.16b, #8   s4..s11
	WORD $0x4e658484       // add v4.8h, v4.8h, v5.8h            s2+s4
	WORD $0x0f602090       // smlal  v16.4s, v4.4h, v0.h[2]
	WORD $0x4f602091       // smlal2 v17.4s, v4.8h, v0.h[2]
	WORD $0x6e033044       // ext v4.16b, v2.16b, v3.16b, #6
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
#define V_SHIFT  60

// func wienerVerticalU8NEONAsm(ctx *wienerU8NEONVertCtx)
//
// Vertical 7-tap Wiener pass over the u16 temp buffer, writing uint8 output.
// Identical to wienerVerticalNEONAsm except the store: the [0,255] clamp and
// u8 narrow are fused into SQXTUN (s32 -> u16, lower clamp 0) + UQXTN
// (u16 -> u8, upper clamp 255), which equals clampInt32(x, 0, 255)
// bit-for-bit, and the dst stride is already in bytes.
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
	MOVW V_SHIFT(R0), R12   // -round1 (s32)

	WORD $0x4c407460       // ld1 {v0.8h}, [x3]    load 8 taps
	WORD $0x4e040d72       // dup v18.4s, w11      seed broadcast
	WORD $0x4e040d93       // dup v19.4s, w12      shift broadcast

vRowLoop:
	CBZ  R7, vDone
	MOVD R1, R10           // dst column cursor
	MOVD R2, R11           // temp row-window base for this output row
	MOVD R6, R8            // remaining columns

vColLoop:
	MOVD R11, R9           // R9 walks the 7 tap rows; post-index by R5
	WORD $0x4eb21e50       // mov v16.16b, v18.16b  seed lanes 0..3
	WORD $0x4eb21e51       // mov v17.16b, v18.16b  seed lanes 4..7

	WORD $0x4cc57521       // ld1 {v1.8h}, [x9], x5
	WORD $0x0f402030       // smlal  v16.4s, v1.4h, v0.h[0]
	WORD $0x4f402031       // smlal2 v17.4s, v1.8h, v0.h[0]
	WORD $0x4cc57521       // ld1 {v1.8h}, [x9], x5
	WORD $0x0f502030       // smlal  v16.4s, v1.4h, v0.h[1]
	WORD $0x4f502031       // smlal2 v17.4s, v1.8h, v0.h[1]
	WORD $0x4cc57521       // ld1 {v1.8h}, [x9], x5
	WORD $0x0f602030       // smlal  v16.4s, v1.4h, v0.h[2]
	WORD $0x4f602031       // smlal2 v17.4s, v1.8h, v0.h[2]
	WORD $0x4cc57521       // ld1 {v1.8h}, [x9], x5
	WORD $0x0f702030       // smlal  v16.4s, v1.4h, v0.h[3]
	WORD $0x4f702031       // smlal2 v17.4s, v1.8h, v0.h[3]
	WORD $0x4cc57521       // ld1 {v1.8h}, [x9], x5
	WORD $0x0f402830       // smlal  v16.4s, v1.4h, v0.h[4]
	WORD $0x4f402831       // smlal2 v17.4s, v1.8h, v0.h[4]
	WORD $0x4cc57521       // ld1 {v1.8h}, [x9], x5
	WORD $0x0f502830       // smlal  v16.4s, v1.4h, v0.h[5]
	WORD $0x4f502831       // smlal2 v17.4s, v1.8h, v0.h[5]
	WORD $0x4cc57521       // ld1 {v1.8h}, [x9], x5
	WORD $0x0f602830       // smlal  v16.4s, v1.4h, v0.h[6]
	WORD $0x4f602831       // smlal2 v17.4s, v1.8h, v0.h[6]

	WORD $0x4eb34610       // sshl v16.4s, v16.4s, v19.4s  arith >> round1
	WORD $0x4eb34631       // sshl v17.4s, v17.4s, v19.4s
	WORD $0x2e612a14       // sqxtun  v20.4h, v16.4s       s32 -> u16, clamp lo 0
	WORD $0x6e612a34       // sqxtun2 v20.8h, v17.4s
	WORD $0x2e214a94       // uqxtn v20.8b, v20.8h         u16 -> u8, clamp hi 255
	WORD $0x0c007154       // st1 {v20.8b}, [x10]

	ADD  $8, R10, R10      // dst += 8 u8 (8 bytes)
	ADD  $16, R11, R11     // temp window += 8 u16 (16 bytes)
	SUB  $8, R8, R8
	CBNZ R8, vColLoop

	ADD  R4, R1, R1        // dst += dstStride bytes
	ADD  R5, R2, R2        // temp += tempStride bytes
	SUB  $1, R7, R7
	CBNZ R7, vRowLoop

vDone:
	RET
