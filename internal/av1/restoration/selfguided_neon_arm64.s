// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

#include "textflag.h"

// NEON self-guided restoration kernels. The Go arm64 assembler does not expose
// integer vector multiply / multiply-accumulate or the register-variable signed
// shift as named mnemonics, so those SIMD instructions are emitted via WORD with
// hand-verified encodings (cross-checked against the system assembler). Pointer
// arithmetic and control flow use ordinary Plan 9 GP instructions. Working
// vectors are V0-V7 and V16-V26 only, so callee-saved V8-V15 are never touched.
//
// Encoding helpers (all .4s = four 32-bit lanes):
//   movi Vd.4s, #0                  0x4f000400 | d
//   ld1  {Vt.4s}, [Xn]              0x4c407800 | (n<<5) | t
//   st1  {Vt.4s}, [Xn]              0x4c007800 | (n<<5) | t
//   dup  Vd.4s, Wn                  0x4e040c00 | (n<<5) | d
//   mla  Vd.4s, Vn.4s, Vm.4s        0x4e209400 | (m<<16) | (n<<5) | d
//   mul  Vd.4s, Vn.4s, Vm.4s        0x4ea09c00 | (m<<16) ... (see below)
//   add  Vd.4s, Vn.4s, Vm.4s        0x4e208400 | (m<<16) | (n<<5) | d
//   sshl Vd.4s, Vn.4s, Vm.4s        0x4ee04400 | (m<<16) | (n<<5) | d

// sgrBlendCtx field offsets (see selfguided_neon_arm64.go).
#define B_DST   0
#define B_DGD   8
#define B_APREV 16
#define B_ACUR  24
#define B_ANEXT 32
#define B_BPREV 40
#define B_BCUR  48
#define B_BNEXT 56
#define B_COLS  64
#define B_WPL   72
#define B_WP    76
#define B_WPR   80
#define B_WCL   84
#define B_WC    88
#define B_WCR   92
#define B_WNL   96
#define B_WN    100
#define B_WNR   104
#define B_BIAS  108
#define B_SHIFT 112

// func sgrBlendRowNEON(ctx *sgrBlendCtx)
//
// Computes one blended output row, 4 columns per iteration:
//   accA = sum over the 3x3 A stencil of A[..]*weight
//   accB = sum over the 3x3 B stencil of B[..]*weight
//   res  = roundPowerOfTwo(accA*dgd + accB, shift)  (bias folded, SSHL shift)
// The weights pick the stencil shape (full 4/3, fast-even 6/5, fast-odd 6/5).
TEXT ·sgrBlendRowNEON(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0

	// Broadcast the 9 weights + bias + shift into V16..V26.
	MOVW B_WPL(R0), R1
	WORD $0x4e040c30           // dup v16.4s, w1   (wPL)
	MOVW B_WP(R0), R1
	WORD $0x4e040c31           // dup v17.4s, w1   (wP)
	MOVW B_WPR(R0), R1
	WORD $0x4e040c32           // dup v18.4s, w1   (wPR)
	MOVW B_WCL(R0), R1
	WORD $0x4e040c33           // dup v19.4s, w1   (wCL)
	MOVW B_WC(R0), R1
	WORD $0x4e040c34           // dup v20.4s, w1   (wC)
	MOVW B_WCR(R0), R1
	WORD $0x4e040c35           // dup v21.4s, w1   (wCR)
	MOVW B_WNL(R0), R1
	WORD $0x4e040c36           // dup v22.4s, w1   (wNL)
	MOVW B_WN(R0), R1
	WORD $0x4e040c37           // dup v23.4s, w1   (wN)
	MOVW B_WNR(R0), R1
	WORD $0x4e040c38           // dup v24.4s, w1   (wNR)
	MOVW B_BIAS(R0), R1
	WORD $0x4e040c39           // dup v25.4s, w1   (bias)
	MOVW B_SHIFT(R0), R1
	WORD $0x4e040c3a           // dup v26.4s, w1   (shift, negative)

	MOVD B_DST(R0), R1         // dst base
	MOVD B_DGD(R0), R2         // dgd base
	MOVD B_APREV(R0), R3
	MOVD B_ACUR(R0), R4
	MOVD B_ANEXT(R0), R5
	MOVD B_BPREV(R0), R6
	MOVD B_BCUR(R0), R7
	MOVD B_BNEXT(R0), R9
	MOVD B_COLS(R0), R10       // remaining columns (multiple of 4)

bCol:
	CBZ  R10, bDone
	WORD $0x4f000401           // movi v1.4s, #0   (accA)
	WORD $0x4f000402           // movi v2.4s, #0   (accB)

	// --- A previous row ---
	WORD $0x4c407863           // ld1 {v3.4s}, [x3]   L
	WORD $0x4eb09461           // mla v1.4s, v3.4s, v16.4s   *wPL
	ADD  $4, R3, R15
	WORD $0x4c4079e4           // ld1 {v4.4s}, [x15]  C
	WORD $0x4eb19481           // mla v1.4s, v4.4s, v17.4s   *wP
	ADD  $8, R3, R15
	WORD $0x4c4079e5           // ld1 {v5.4s}, [x15]  R
	WORD $0x4eb294a1           // mla v1.4s, v5.4s, v18.4s   *wPR

	// --- A current row ---
	WORD $0x4c407883           // ld1 {v3.4s}, [x4]
	WORD $0x4eb39461           // mla v1.4s, v3.4s, v19.4s   *wCL
	ADD  $4, R4, R15
	WORD $0x4c4079e4           // ld1 {v4.4s}, [x15]
	WORD $0x4eb49481           // mla v1.4s, v4.4s, v20.4s   *wC
	ADD  $8, R4, R15
	WORD $0x4c4079e5           // ld1 {v5.4s}, [x15]
	WORD $0x4eb594a1           // mla v1.4s, v5.4s, v21.4s   *wCR

	// --- A next row ---
	WORD $0x4c4078a3           // ld1 {v3.4s}, [x5]
	WORD $0x4eb69461           // mla v1.4s, v3.4s, v22.4s   *wNL
	ADD  $4, R5, R15
	WORD $0x4c4079e4           // ld1 {v4.4s}, [x15]
	WORD $0x4eb79481           // mla v1.4s, v4.4s, v23.4s   *wN
	ADD  $8, R5, R15
	WORD $0x4c4079e5           // ld1 {v5.4s}, [x15]
	WORD $0x4eb894a1           // mla v1.4s, v5.4s, v24.4s   *wNR

	// --- B previous row ---
	WORD $0x4c4078c3           // ld1 {v3.4s}, [x6]
	WORD $0x4eb09462           // mla v2.4s, v3.4s, v16.4s
	ADD  $4, R6, R15
	WORD $0x4c4079e4           // ld1 {v4.4s}, [x15]
	WORD $0x4eb19482           // mla v2.4s, v4.4s, v17.4s
	ADD  $8, R6, R15
	WORD $0x4c4079e5           // ld1 {v5.4s}, [x15]
	WORD $0x4eb294a2           // mla v2.4s, v5.4s, v18.4s

	// --- B current row ---
	WORD $0x4c4078e3           // ld1 {v3.4s}, [x7]
	WORD $0x4eb39462           // mla v2.4s, v3.4s, v19.4s
	ADD  $4, R7, R15
	WORD $0x4c4079e4           // ld1 {v4.4s}, [x15]
	WORD $0x4eb49482           // mla v2.4s, v4.4s, v20.4s
	ADD  $8, R7, R15
	WORD $0x4c4079e5           // ld1 {v5.4s}, [x15]
	WORD $0x4eb594a2           // mla v2.4s, v5.4s, v21.4s

	// --- B next row ---
	WORD $0x4c407923           // ld1 {v3.4s}, [x9]
	WORD $0x4eb69462           // mla v2.4s, v3.4s, v22.4s
	ADD  $4, R9, R15
	WORD $0x4c4079e4           // ld1 {v4.4s}, [x15]
	WORD $0x4eb79482           // mla v2.4s, v4.4s, v23.4s
	ADD  $8, R9, R15
	WORD $0x4c4079e5           // ld1 {v5.4s}, [x15]
	WORD $0x4eb894a2           // mla v2.4s, v5.4s, v24.4s

	// --- combine: res = (accA*dgd + accB + bias) >> shift ---
	WORD $0x4c407840           // ld1 {v0.4s}, [x2]   dgd
	WORD $0x4ea09c26           // mul v6.4s, v1.4s, v0.4s
	WORD $0x4ea284c6           // add v6.4s, v6.4s, v2.4s
	WORD $0x4eb984c6           // add v6.4s, v6.4s, v25.4s   (+ bias)
	WORD $0x4eba44c6           // sshl v6.4s, v6.4s, v26.4s  (>> shift)
	WORD $0x4c007826           // st1 {v6.4s}, [x1]

	// advance all cursors by 4 elements (16 bytes)
	ADD  $16, R1, R1
	ADD  $16, R2, R2
	ADD  $16, R3, R3
	ADD  $16, R4, R4
	ADD  $16, R5, R5
	ADD  $16, R6, R6
	ADD  $16, R7, R7
	ADD  $16, R9, R9
	SUB  $4, R10, R10
	B    bCol

bDone:
	RET

// func boxsumInteriorBandNEON(src *int32, dst *int32, neonCols, radius, rows,
//                             srcStride, squared uintptr)
//
// For each group of 4 interior output columns, accumulate (in a 4-lane int32
// register) the source values over `rows` vertical rows and the 2r+1 horizontal
// taps, then store the four sums. `src` points at the top-left of the first
// lane's window (row y0, column col-r). squared != 0 squares each value first.
TEXT ·boxsumInteriorBandNEON(SB), NOSPLIT, $0-56
	MOVD src+0(FP), R0
	MOVD dst+8(FP), R1
	MOVD neonCols+16(FP), R2
	MOVD radius+24(FP), R3
	MOVD rows+32(FP), R4
	MOVD srcStride+40(FP), R5
	MOVD squared+48(FP), R6
	LSL  $2, R5, R5            // srcStride -> bytes
	ADD  $1, R3, R7
	ADD  R3, R7, R7           // R7 = 2*radius + 1 (horizontal taps)

	// column-group cursor: byte offset into src for the current group's
	// top-left window element. Starts at 0; advances by 16 bytes per group.
	MOVD $0, R8

cgLoop:
	CBZ  R2, cgDone
	WORD $0x4f000400          // movi v0.4s, #0   accumulator

	ADD  R0, R8, R9          // R9 = src + group offset (row 0, leftmost tap)
	MOVD R4, R10             // remaining rows

cgRow:
	MOVD R9, R11            // R11 = start of this row's window
	MOVD R7, R12            // remaining horizontal taps

cgTap:
	WORD $0x4c407961        // ld1 {v1.4s}, [x11]
	CBZ  R6, cgAdd
	WORD $0x4ea19c21        // mul v1.4s, v1.4s, v1.4s   (square)

cgAdd:
	WORD $0x4ea18400        // add v0.4s, v0.4s, v1.4s
	ADD  $4, R11, R11       // next horizontal tap (1 element)
	SUB  $1, R12, R12
	CBNZ R12, cgTap

	ADD  R5, R9, R9         // next source row
	SUB  $1, R10, R10
	CBNZ R10, cgRow

	ADD  R1, R8, R13        // R13 = dst + group offset
	WORD $0x4c0079a0        // st1 {v0.4s}, [x13]

	ADD  $16, R8, R8        // advance group by 4 columns
	SUB  $4, R2, R2
	B    cgLoop

cgDone:
	RET
