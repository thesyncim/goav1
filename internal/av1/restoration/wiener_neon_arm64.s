// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// NEON Wiener separable passes. The Go arm64 assembler does not expose integer
// vector multiply-accumulate, the register-variable signed shift, or the
// saturating narrows as named mnemonics, so the SIMD instructions are emitted
// via WORD with hand-verified encodings (cross-checked against the system
// assembler). Pointer arithmetic, loop counters, and control flow use ordinary
// Plan 9 GP-register instructions. Working vectors are V0-V7 and V16-V21 only,
// so the callee-saved V8-V15 are never touched.
//
// Register / lane semantics (all bit-exact with the pure-Go reference):
//   ld1   {Vn.8h}, [Xm](, Xs)   load 8 u16 samples, optional post-index by Xs
//   ld1   {Vn.8h, Vm.8h}, [Xp]  load 16 u16 samples (sliding window source)
//   movi  Vn.4s, #0             zero a 32-bit-lane accumulator
//   dup   Vn.4s, Wm             broadcast a 32-bit GP value into 4 s32 lanes
//   dup   Vn.8h, Wm             broadcast a 16-bit GP value into 8 u16 lanes
//   mov   Vn.16b, Vm.16b        copy the accumulator seed into an accumulator
//   ext   Vd.16b, Vn, Vm, #b    slide the 16-u16 window by b bytes (= b/2 lanes)
//   smlal/smlal2 Vd.4s, Vn.{4,8}h, V0.h[i]  signed widening MAC by tap i
//   sshl  Vd.4s, Vn.4s, Vs.4s   per-lane signed shift; negative lane => arith. >>
//   sqxtun/sqxtun2 Vd.{4,8}h, Vn.4s  s32 -> u16 saturating narrow (lower clamp 0)
//   umin  Vd.8h, Vn.8h, Vm.8h   per-lane unsigned min (upper clamp)
//   st1   {Vn.8h}, [Xm]         store 8 u16 outputs

// wienerNEONHorizCtx field offsets (see wiener_neon_arm64.go).
#define H_DST    0
#define H_SRC    8
#define H_SRCSTR 16
#define H_WIDTH  24
#define H_ROWS   32
#define H_TAPS   40
#define H_SEED   48
#define H_SHIFT  52
#define H_MAXCL  56

// func wienerHorizontalNEONAsm(ctx *wienerNEONHorizCtx)
//
// Horizontal 7-tap Wiener pass. For each of `rows` rows the inner loop emits 8
// temp columns per iteration: load 16 source u16 (the [col-3..col+10] window),
// slide the 7-tap window with EXT, widen-MAC by each tap, then apply the folded
// rounding shift and the [0, maxClamp] clamp. The temp output stride equals the
// width (rows are stored contiguously, exactly as wienerHorizontal writes them).
TEXT ·wienerHorizontalNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD H_DST(R0), R1      // temp output base
	MOVD H_SRC(R0), R2      // src tap-window base (row -3, col -3)
	MOVD H_SRCSTR(R0), R5   // src stride (elements)
	LSL  $1, R5, R5         // -> bytes
	MOVD H_WIDTH(R0), R6    // width (multiple of 8)
	MOVD H_ROWS(R0), R7     // rows = height + 2*WienerHalfwin
	MOVD H_TAPS(R0), R3     // taps ptr (8 int16)
	MOVW H_SEED(R0), R11    // accumulator seed (s32)
	MOVW H_SHIFT(R0), R12   // -round0 (s32)
	MOVHU H_MAXCL(R0), R13  // maxClamp (u16)

	WORD $0x4c407460       // ld1 {v0.8h}, [x3]    load 8 taps
	WORD $0x4e040d72       // dup v18.4s, w11      seed broadcast (s32 lanes)
	WORD $0x4e040d93       // dup v19.4s, w12      shift broadcast (s32 lanes)
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
	WORD $0x4c40a522       // ld1 {v2.8h, v3.8h}, [x9]  16 source u16

	// AV1 Wiener symmetry (dav1d looprestoration.S wiener_filter7_h): the seven
	// taps are mirrored (f0==f6, f1==f5, f2==f4), so add the three symmetric
	// sample pairs in 16-bit lanes first, then MAC each pair-sum once by its
	// shared tap. This is 4 widening MACs (center + 3 pair-sums) in place of 7,
	// bit-identical because (a+b)*f == a*f + b*f and every source sample (<=4095)
	// keeps the pair-sum (<=8190) inside the positive int16 lane SMLAL widens.
	WORD $0x6e033041       // ext v1.16b, v2.16b, v3.16b, #6   center col+0
	WORD $0x6e036044       // ext v4.16b, v2.16b, v3.16b, #12  col+3
	WORD $0x4e648444       // add v4.8h, v2.8h, v4.8h          pair f0: (col-3)+(col+3)
	WORD $0x6e031045       // ext v5.16b, v2.16b, v3.16b, #2   col-2
	WORD $0x6e035046       // ext v6.16b, v2.16b, v3.16b, #10  col+2
	WORD $0x4e6684a5       // add v5.8h, v5.8h, v6.8h          pair f1: (col-2)+(col+2)
	WORD $0x6e032046       // ext v6.16b, v2.16b, v3.16b, #4   col-1
	WORD $0x6e034047       // ext v7.16b, v2.16b, v3.16b, #8   col+1
	WORD $0x4e6784c6       // add v6.8h, v6.8h, v7.8h          pair f2: (col-1)+(col+1)

	WORD $0x0f402090       // smlal  v16.4s, v4.4h, v0.h[0]   pair f0
	WORD $0x4f402091       // smlal2 v17.4s, v4.8h, v0.h[0]
	WORD $0x0f5020b0       // smlal  v16.4s, v5.4h, v0.h[1]   pair f1
	WORD $0x4f5020b1       // smlal2 v17.4s, v5.8h, v0.h[1]
	WORD $0x0f6020d0       // smlal  v16.4s, v6.4h, v0.h[2]   pair f2
	WORD $0x4f6020d1       // smlal2 v17.4s, v6.8h, v0.h[2]
	WORD $0x0f702030       // smlal  v16.4s, v1.4h, v0.h[3]   center f3 (1<<7 folded)
	WORD $0x4f702031       // smlal2 v17.4s, v1.8h, v0.h[3]

	WORD $0x4eb34610       // sshl v16.4s, v16.4s, v19.4s  arith >> round0
	WORD $0x4eb34631       // sshl v17.4s, v17.4s, v19.4s
	WORD $0x2e612a14       // sqxtun  v20.4h, v16.4s       s32 -> u16, clamp lo 0
	WORD $0x6e612a34       // sqxtun2 v20.8h, v17.4s
	WORD $0x6e756e94       // umin v20.8h, v20.8h, v21.8h  clamp hi maxClamp
	WORD $0x4c007554       // st1 {v20.8h}, [x10]

	ADD  $16, R10, R10     // temp += 8 u16 (16 bytes)
	ADD  $16, R9, R9       // src window += 8 u16 (16 bytes)
	SUB  $8, R8, R8
	CBNZ R8, hColLoop

	ADD  R14, R1, R1       // temp += rowStride bytes
	ADD  R5, R2, R2        // src += srcStride bytes
	SUB  $1, R7, R7
	CBNZ R7, hRowLoop

hDone:
	RET

// wienerNEONVertCtx field offsets (see wiener_neon_arm64.go).
#define V_DST    0
#define V_SRC    8
#define V_DSTSTR 16
#define V_SRCSTR 24
#define V_WIDTH  32
#define V_ROWS   40
#define V_TAPS   48
#define V_SEED   56
#define V_SHIFT  60
#define V_MAXCL  64

// func wienerVerticalNEONAsm(ctx *wienerNEONVertCtx)
//
// Vertical 7-tap Wiener pass over the int16 temp buffer. For each output row the
// inner loop emits 8 columns per iteration: MAC the 7 stacked temp rows
// [row..row+6] (the center is row+3) with the taps, then apply the folded
// rounding shift and the [0, max] clamp.
TEXT ·wienerVerticalNEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD V_DST(R0), R1      // dst row base
	MOVD V_SRC(R0), R2      // temp tap-window base (row 0)
	MOVD V_DSTSTR(R0), R4   // dst stride (elements)
	LSL  $1, R4, R4         // -> bytes
	MOVD V_SRCSTR(R0), R5   // temp stride (elements)
	LSL  $1, R5, R5         // -> bytes
	MOVD V_WIDTH(R0), R6    // width (multiple of 8)
	MOVD V_ROWS(R0), R7     // rows = height
	MOVD V_TAPS(R0), R3     // taps ptr (8 int16)
	MOVW V_SEED(R0), R11    // accumulator seed (s32)
	MOVW V_SHIFT(R0), R12   // -round1 (s32)
	MOVHU V_MAXCL(R0), R13  // max (u16)

	WORD $0x4c407460       // ld1 {v0.8h}, [x3]    load 8 taps
	WORD $0x4e040d72       // dup v18.4s, w11      seed broadcast
	WORD $0x4e040d93       // dup v19.4s, w12      shift broadcast
	WORD $0x4e020db5       // dup v21.8h, w13      max broadcast

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
	WORD $0x2e612a14       // sqxtun  v20.4h, v16.4s
	WORD $0x6e612a34       // sqxtun2 v20.8h, v17.4s
	WORD $0x6e756e94       // umin v20.8h, v20.8h, v21.8h
	WORD $0x4c007554       // st1 {v20.8h}, [x10]

	ADD  $16, R10, R10     // dst += 8 u16 (16 bytes)
	ADD  $16, R11, R11     // temp window += 8 u16 (16 bytes)
	SUB  $8, R8, R8
	CBNZ R8, vColLoop

	ADD  R4, R1, R1        // dst += dstStride bytes
	ADD  R5, R2, R2        // temp += tempStride bytes
	SUB  $1, R7, R7
	CBNZ R7, vRowLoop

vDone:
	RET
