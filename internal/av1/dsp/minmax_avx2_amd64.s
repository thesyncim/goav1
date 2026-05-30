// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// AVX2 (VEX-encoded SSE-width) MinMaxAbsDiff8x8 kernels. They are bit-exact
// with minMaxAbsDiff8x8PureGo (see minmax_pure_go.go).
//
// Each kernel walks the 8 rows of an 8x8 block with byte strides, computes the
// per-lane absolute difference as max(a,b)-min(a,b), and folds a running
// per-lane min/max. After the row loop a log-step shuffle reduction collapses
// the lanes to scalars matching the reference, which scans every sample with
// absDiff8/absDiff16.

//go:build amd64 && !purego

#include "textflag.h"

// func minMaxAbsDiff8x8AVX2_8(a *byte, aStride uintptr, b *byte, bStride uintptr, out *uint32)
//
// 8-bit samples: each row is 8 bytes loaded into the low half of an xmm.
// out[0] = min absolute byte difference, out[1] = max absolute byte difference.
TEXT ·minMaxAbsDiff8x8AVX2_8(SB), NOSPLIT, $0-40
	MOVQ a+0(FP), AX
	MOVQ aStride+8(FP), BX
	MOVQ b+16(FP), CX
	MOVQ bStride+24(FP), DX
	MOVQ out+32(FP), DI
	MOVQ $8, SI

	// X2 = running min (0xff per byte lane), X3 = running max (0).
	VPCMPEQB X2, X2, X2 // all-ones (0xff lanes)
	VPXOR    X3, X3, X3

rowLoop8:
	MOVQ      (AX), X0 // low 8 bytes of a row
	MOVQ      (CX), X1 // low 8 bytes of b row
	VPMAXUB   X1, X0, X4
	VPMINUB   X1, X0, X5
	VPSUBB    X5, X4, X4 // |a-b| per byte lane
	VPMINUB   X4, X2, X2
	VPMAXUB   X4, X3, X3
	ADDQ      BX, AX
	ADDQ      DX, CX
	DECQ      SI
	JNZ       rowLoop8

	// Horizontal reduce the low 8 byte lanes of X2 (min) and X3 (max).
	VPSRLDQ $4, X2, X6
	VPMINUB X6, X2, X2
	VPSRLDQ $2, X2, X6
	VPMINUB X6, X2, X2
	VPSRLDQ $1, X2, X6
	VPMINUB X6, X2, X2

	VPSRLDQ $4, X3, X6
	VPMAXUB X6, X3, X3
	VPSRLDQ $2, X3, X6
	VPMAXUB X6, X3, X3
	VPSRLDQ $1, X3, X6
	VPMAXUB X6, X3, X3

	VPEXTRB $0, X2, R8
	VPEXTRB $0, X3, R9
	MOVL    R8, 0(DI)
	MOVL    R9, 4(DI)
	RET

// func minMaxAbsDiff8x8AVX2_16(a *byte, aStride uintptr, b *byte, bStride uintptr, out *uint32)
//
// 16-bit little-endian samples: each row is 8 words filling a full xmm.
// out[0] = min, out[1] = max.
TEXT ·minMaxAbsDiff8x8AVX2_16(SB), NOSPLIT, $0-40
	MOVQ a+0(FP), AX
	MOVQ aStride+8(FP), BX
	MOVQ b+16(FP), CX
	MOVQ bStride+24(FP), DX
	MOVQ out+32(FP), DI
	MOVQ $8, SI

	VPCMPEQW X2, X2, X2 // all-ones (0xffff per word lane) running min
	VPXOR    X3, X3, X3 // running max

rowLoop16:
	VMOVDQU   (AX), X0
	VMOVDQU   (CX), X1
	VPMAXUW   X1, X0, X4
	VPMINUW   X1, X0, X5
	VPSUBW    X5, X4, X4 // |a-b| per word lane
	VPMINUW   X4, X2, X2
	VPMAXUW   X4, X3, X3
	ADDQ      BX, AX
	ADDQ      DX, CX
	DECQ      SI
	JNZ       rowLoop16

	VPSRLDQ $8, X2, X6
	VPMINUW X6, X2, X2
	VPSRLDQ $4, X2, X6
	VPMINUW X6, X2, X2
	VPSRLDQ $2, X2, X6
	VPMINUW X6, X2, X2

	VPSRLDQ $8, X3, X6
	VPMAXUW X6, X3, X3
	VPSRLDQ $4, X3, X6
	VPMAXUW X6, X3, X3
	VPSRLDQ $2, X3, X6
	VPMAXUW X6, X3, X3

	VPEXTRW $0, X2, R8
	VPEXTRW $0, X3, R9
	MOVL    R8, 0(DI)
	MOVL    R9, 4(DI)
	RET
