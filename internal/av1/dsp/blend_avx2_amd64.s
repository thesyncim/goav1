// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// AVX2 BlendA64Mask kernels. The Go amd64 assembler exposes every AVX2 mnemonic
// used here by name, so no WORD/BYTE encoding is required. The amd64 ABI has no
// callee-saved YMM registers, so nothing needs preserving across the call; the
// frame pointer (BP) is left untouched so the kernels pass `go vet` asm checks.
//
// Both kernels are bit-exact with blendA64MaskPureGo (see blend.go) for the two
// supported mask layouts: no subsampling, and 2x2 (subX && subY). Per group of
// eight output columns:
//   - the per-pixel weight m is gathered into 8 u16 lanes (the mask byte for
//     no-sub; the round2 of the four 2x2 neighbours for subX&&subY),
//   - dst = (m*s0 + (64-m)*s1 + 32) >> 6 is computed in u32 lanes, matching
//     blendA64 / roundPowerOfTwoPositive(...,6),
//   - any s0>max, s1>max or m>64 lane is folded into a violation accumulator so
//     the Go wrapper returns ErrInvalidBlock, matching the reference verdict.
//     The range test is VPSUBUSW(x, limit): unsigned-saturating subtract is
//     non-zero exactly when x>limit, correct for the entire uint16 range
//     (a signed compare would miss out-of-range values >= 0x8000).
//
// The four strides stay on the stack frame (re-read from FP at each row
// advance) so the loop bodies have enough general registers without touching
// BP. Fixed vector registers held across the row/col loops:
//   X10  violation accumulator (8 u16 lanes; any non-zero lane => invalid)
//   X11  broadcast max (u16) for the sample range test
//   X12  broadcast 64  (u16) for the mask-weight range test
//   Y13  broadcast 64  (s32) for computing 64-m
//   Y14  broadcast 32  (s32) rounding bias for the >>6 blend
//   X9   broadcast 2   (u16) rounding bias for the 2x2 mask round2 (subXY only)
//   X8   broadcast 0x01 (16 bytes) VPMADDUBSW multiplier        (subXY only)

//go:build amd64 && !purego

#include "textflag.h"

// func blendA64NoSubAVX2Asm(dst *uint16, dstStride uintptr, src0 *uint16, src0Stride uintptr, src1 *uint16, src1Stride uintptr, mask *uint8, maskStride uintptr, maxv uint32, groups uintptr, height uintptr) uint32
TEXT ·blendA64NoSubAVX2Asm(SB), NOSPLIT, $0-92
	MOVQ dst+0(FP), AX
	MOVQ src0+16(FP), CX
	MOVQ src1+32(FP), SI
	MOVQ mask+48(FP), R8
	MOVL maxv+64(FP), R12
	MOVQ groups+72(FP), R11
	MOVQ height+80(FP), R13

	VPXOR        X10, X10, X10 // violation accumulator
	MOVL         R12, X11
	VPBROADCASTW X11, X11      // max (u16)
	MOVL         $64, R12
	MOVL         R12, X12
	VPBROADCASTW X12, X12      // 64 (u16)
	MOVL         R12, X13
	VPBROADCASTD X13, Y13      // 64 (s32)
	MOVL         $32, R12
	MOVL         R12, X14
	VPBROADCASTD X14, Y14      // 32 (s32)

rowLoopN:
	TESTQ R13, R13
	JZ    doneN
	MOVQ  AX, R12  // dst cursor
	MOVQ  CX, R14  // src0 cursor
	MOVQ  SI, R15  // src1 cursor
	MOVQ  R8, R9   // mask cursor
	MOVQ  R11, R10 // per-row groups counter

colLoopN:
	TESTQ R10, R10
	JZ    rowAdvN
	VMOVDQU   (R14), X0      // 8 u16 src0
	VMOVDQU   (R15), X1      // 8 u16 src1
	VPMOVZXBW (R9), X2       // 8 mask bytes -> 8 u16 weights

	// Range verdict (unsigned >): VPSUBUSW is non-zero iff x > limit.
	VPSUBUSW  X11, X0, X3
	VPOR      X3, X10, X10
	VPSUBUSW  X11, X1, X3
	VPOR      X3, X10, X10
	VPSUBUSW  X12, X2, X3
	VPOR      X3, X10, X10

	// dst = (m*s0 + (64-m)*s1 + 32) >> 6 in u32 lanes.
	VPMOVZXWD X0, Y4         // s0 -> s32
	VPMOVZXWD X1, Y5         // s1 -> s32
	VPMOVZXWD X2, Y6         // m  -> s32
	VPSUBD    Y6, Y13, Y7    // 64 - m
	VPMULLD   Y6, Y4, Y4     // m*s0
	VPMULLD   Y7, Y5, Y5     // (64-m)*s1
	VPADDD    Y5, Y4, Y4
	VPADDD    Y14, Y4, Y4    // + 32
	VPSRLD    $6, Y4, Y4     // >> 6
	VPACKUSDW Y4, Y4, Y4     // s32 -> u16 (in range, no saturation effect)
	VPERMQ    $0xd8, Y4, Y4  // gather the 8 low words contiguously
	VMOVDQU   X4, (R12)      // store 8 u16

	ADDQ $16, R12
	ADDQ $16, R14
	ADDQ $16, R15
	ADDQ $8, R9
	DECQ R10
	JMP  colLoopN

rowAdvN:
	ADDQ dstStride+8(FP), AX
	ADDQ src0Stride+24(FP), CX
	ADDQ src1Stride+40(FP), SI
	ADDQ maskStride+56(FP), R8
	DECQ R13
	JMP  rowLoopN

doneN:
	// Reduce the 8 u16 violation lanes to a single 0/non-zero scalar by ORing
	// the two 64-bit halves. A lane is non-zero iff some s0>max, s1>max or m>64
	// occurred; ORing preserves any set bit regardless of the lane's value, so
	// (unlike a saturating narrow) a difference >= 0x8000 is never lost.
	VPEXTRQ   $1, X10, R14
	VMOVQ     X10, R12
	ORQ       R14, R12
	XORQ      R13, R13
	TESTQ     R12, R12
	JZ        okN
	MOVL      $1, R13

okN:
	MOVL R13, ret+88(FP)
	VZEROUPPER
	RET

// func blendA64SubXYAVX2Asm(dst *uint16, dstStride uintptr, src0 *uint16, src0Stride uintptr, src1 *uint16, src1Stride uintptr, mask *uint8, maskStride uintptr, maxv uint32, groups uintptr, height uintptr) uint32
//
// 2x2 subsampled mask: each of the eight output weights is the round2 of the
// four neighbours mask[2r,2c], mask[2r+1,2c], mask[2r,2c+1], mask[2r+1,2c+1].
TEXT ·blendA64SubXYAVX2Asm(SB), NOSPLIT, $0-92
	MOVQ dst+0(FP), AX
	MOVQ src0+16(FP), CX
	MOVQ src1+32(FP), SI
	MOVQ mask+48(FP), R8
	MOVL maxv+64(FP), R12
	MOVQ groups+72(FP), R11
	MOVQ height+80(FP), R13

	VPXOR        X10, X10, X10
	MOVL         R12, X11
	VPBROADCASTW X11, X11      // max (u16)
	MOVL         $64, R12
	MOVL         R12, X12
	VPBROADCASTW X12, X12      // 64 (u16)
	MOVL         R12, X13
	VPBROADCASTD X13, Y13      // 64 (s32)
	MOVL         $32, R12
	MOVL         R12, X14
	VPBROADCASTD X14, Y14      // 32 (s32)
	MOVL         $2, R12
	MOVL         R12, X9
	VPBROADCASTW X9, X9        // 2 (u16) round2 bias
	MOVQ         $0x0101010101010101, R12
	MOVQ         R12, X8
	VPUNPCKLQDQ  X8, X8, X8    // 16 bytes of 0x01 (VPMADDUBSW multiplier)

rowLoopS:
	TESTQ R13, R13
	JZ    doneS
	MOVQ  AX, R12  // dst cursor
	MOVQ  CX, R14  // src0 cursor
	MOVQ  SI, R15  // src1 cursor
	MOVQ  R8, R9   // mask row 0 cursor
	MOVQ  R11, R10 // per-row groups counter

colLoopS:
	TESTQ R10, R10
	JZ    rowAdvS
	VMOVDQU (R14), X0        // 8 u16 src0
	VMOVDQU (R15), X1        // 8 u16 src1

	// Gather 16 mask bytes from each of the two rows, pairwise-sum adjacent
	// bytes (VPMADDUBSW with multiplier 1), add the two rows, then round2.
	VMOVDQU    (R9), X2      // mask row 0, 16 bytes
	VPMADDUBSW X8, X2, X2    // 8 u16: mask0[2j]+mask0[2j+1]
	ADDQ       maskStride+56(FP), R9 // advance to mask row 1
	VMOVDQU    (R9), X3      // mask row 1, 16 bytes
	VPMADDUBSW X8, X3, X3    // 8 u16: mask1[2j]+mask1[2j+1]
	SUBQ       maskStride+56(FP), R9 // back to row 0 for the column step
	VPADDW     X3, X2, X2    // sum of all four neighbours
	VPADDW     X9, X2, X2    // + 2  (round2 bias)
	VPSRLW     $2, X2, X2    // >> 2  -> m

	// Range verdict.
	VPSUBUSW  X11, X0, X4
	VPOR      X4, X10, X10
	VPSUBUSW  X11, X1, X4
	VPOR      X4, X10, X10
	VPSUBUSW  X12, X2, X4
	VPOR      X4, X10, X10

	VPMOVZXWD X0, Y4         // s0 -> s32
	VPMOVZXWD X1, Y5         // s1 -> s32
	VPMOVZXWD X2, Y6         // m  -> s32
	VPSUBD    Y6, Y13, Y7    // 64 - m
	VPMULLD   Y6, Y4, Y4     // m*s0
	VPMULLD   Y7, Y5, Y5     // (64-m)*s1
	VPADDD    Y5, Y4, Y4
	VPADDD    Y14, Y4, Y4    // + 32
	VPSRLD    $6, Y4, Y4     // >> 6
	VPACKUSDW Y4, Y4, Y4
	VPERMQ    $0xd8, Y4, Y4
	VMOVDQU   X4, (R12)

	ADDQ $16, R12
	ADDQ $16, R14
	ADDQ $16, R15
	ADDQ $16, R9            // mask row 0 += 16 bytes (8 input pairs)
	DECQ R10
	JMP  colLoopS

rowAdvS:
	ADDQ dstStride+8(FP), AX
	ADDQ src0Stride+24(FP), CX
	ADDQ src1Stride+40(FP), SI
	ADDQ maskStride+56(FP), R8
	ADDQ maskStride+56(FP), R8 // mask advances two rows per output row
	DECQ R13
	JMP  rowLoopS

doneS:
	// See doneN: OR the two 64-bit halves so a violation lane >= 0x8000 is not
	// lost to a saturating narrow.
	VPEXTRQ   $1, X10, R14
	VMOVQ     X10, R12
	ORQ       R14, R12
	XORQ      R13, R13
	TESTQ     R12, R12
	JZ        okS
	MOVL      $1, R13

okS:
	MOVL R13, ret+88(FP)
	VZEROUPPER
	RET
