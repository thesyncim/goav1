// AVX2 TXB coefficient level-init helper.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build amd64 && !purego

#include "textflag.h"

// coeffInitLevelsAVX2Asm mirrors the arm64 svt_av1_txb_init_levels_neon slice
// into AVX2, adapted to goav1's column-major level scratch:
//
//     levels[col*stride+row] = min(abs(coeff[col*height+row]), 127)
//
// The Go wrapper clears the padded scratch first, so this kernel only fills the
// live rows of each column. Each column is processed in blocks of eight int16
// coefficients (16 bytes in / 8 bytes out); a trailing four-lane block covers
// the height-4 case. Bit-exactness with coeffInitLevelsPureGo:
//   - VPABSW takes |coeff| per int16 lane. abs(-32768) wraps to 0x8000, but the
//     following VPMINUW treats it as unsigned 32768 and clamps to 127, matching
//     coeffAbsClamp127's `if n > 127 { return 127 }` (n=32768 for INT16_MIN).
//   - all valid magnitudes are in [0,127] after the clamp, so VPACKUSWB's
//     unsigned-saturating narrow to bytes is exact.
//
// func coeffInitLevelsAVX2Asm(coeffs *int16, levels *uint8, columns uintptr, height uintptr, stride uintptr)
TEXT ·coeffInitLevelsAVX2Asm(SB), NOSPLIT, $0-40
	MOVQ coeffs+0(FP), SI
	MOVQ levels+8(FP), DI
	MOVQ columns+16(FP), CX
	MOVQ height+24(FP), R8
	MOVQ stride+32(FP), R9

	// Broadcast 127 into all eight word lanes of X15.
	MOVL         $127, AX
	MOVQ         AX, X15
	VPBROADCASTW X15, X15

colLoop:
	MOVQ DI, R10 // destination column base
	MOVQ R8, R11 // rows remaining in this column

rowLoop:
	CMPQ R11, $8
	JL   row4
	VMOVDQU   (SI), X0 // 8 int16
	VPABSW    X0, X0
	VPMINUW   X15, X0, X0
	VPACKUSWB X0, X0, X0 // low 8 bytes = clamped levels
	MOVQ      X0, (R10)
	ADDQ      $16, SI
	ADDQ      $8, R10
	SUBQ      $8, R11
	JMP       rowLoop

row4:
	TESTQ R11, R11
	JZ    colNext
	MOVQ      (SI), X0 // 4 int16; upper XMM lanes zero-extended
	VPABSW    X0, X0
	VPMINUW   X15, X0, X0
	VPACKUSWB X0, X0, X0 // low 4 bytes valid, bytes 4-7 are zero padding
	MOVQ      X0, (R10)
	ADDQ      $8, SI

colNext:
	ADDQ R9, DI // advance to next column base by stride
	DECQ CX
	JNZ  colLoop

	VZEROUPPER
	RET
