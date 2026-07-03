// AVX2 TXB NZ-map context helper.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build amd64 && !purego

#include "textflag.h"

// coeffNZMapContextsAVX2Asm mirrors the arm64 svt_av1_get_nz_map_contexts_neon
// slice into AVX2:
//
//   count = min((sum(min(level[neighbour], 3)) + 1) >> 1, 4) + pos_offset
//
// goav1 stores TXB levels column-major, so each vector covers rows from one
// coefficient column. The five neighbour byte offsets (n0..n4) are precomputed
// by the Go caller from the transform class and stride, which lets one kernel
// serve the 2D, horizontal and vertical classes. Bit-exactness with the pure-Go
// reference (CoeffLowerLevelsContext / coeffNZMapContextsScalar):
//   - VPMINUB against 3 is clipMax3 per lane.
//   - VPADDB accumulates the five clipped neighbours; the sum is at most 15 so
//     no byte lane overflows.
//   - VPAVGB against a zero register computes (sum + 0 + 1) >> 1, exactly the
//     unsigned rounding half-sum (mag+1)>>1.
//   - VPMINUB against 4 caps the count, then VPADDB adds the positional offset
//     table entry, matching `minInt((mag+1)>>1, 4) + offset`.
//
// The height-32 columns are processed as two 16-byte blocks (rows 0..15 and
// 16..31). Height 4 and 8 use 32/64-bit loads/stores so the read/write footprint
// never exceeds the packed offset/context buffers or the padded level scratch.

#define CTX_LEVELS   0
#define CTX_OFFSETS  8
#define CTX_CONTEXTS 16
#define CTX_COLUMNS  24
#define CTX_STRIDE   32
#define CTX_N0       40
#define CTX_N1       48
#define CTX_N2       56
#define CTX_N3       64
#define CTX_N4       72
#define CTX_ROWBYTES 80

// NZBLOCK combines the five neighbour vectors staged in X0..X4 with the
// positional offset vector in X5, leaving the context bytes in X0.
// X12 = 4, X13 = 3, X14 = 0 (all broadcast).
#define NZBLOCK          \
	VPMINUB X13, X0, X0 \
	VPMINUB X13, X1, X1 \
	VPMINUB X13, X2, X2 \
	VPMINUB X13, X3, X3 \
	VPMINUB X13, X4, X4 \
	VPADDB  X1, X0, X0  \
	VPADDB  X2, X0, X0  \
	VPADDB  X3, X0, X0  \
	VPADDB  X4, X0, X0  \
	VPAVGB  X14, X0, X0 \
	VPMINUB X12, X0, X0 \
	VPADDB  X5, X0, X0

// func coeffNZMapContextsAVX2Asm(ctx *coeffNZMapAVX2Ctx)
TEXT ·coeffNZMapContextsAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ CTX_LEVELS(AX), SI
	MOVQ CTX_OFFSETS(AX), DI
	MOVQ CTX_CONTEXTS(AX), BX
	MOVQ CTX_COLUMNS(AX), CX
	MOVQ CTX_STRIDE(AX), R15
	MOVQ CTX_N0(AX), R8
	MOVQ CTX_N1(AX), R9
	MOVQ CTX_N2(AX), R10
	MOVQ CTX_N3(AX), R11
	MOVQ CTX_N4(AX), R12

	VPXOR        X14, X14, X14
	MOVL         $3, DX
	MOVQ         DX, X13
	VPBROADCASTB X13, X13
	MOVL         $4, DX
	MOVQ         DX, X12
	VPBROADCASTB X12, X12

	MOVQ CTX_ROWBYTES(AX), DX
	CMPQ DX, $16
	JEQ  path16
	CMPQ DX, $32
	JEQ  path32
	CMPQ DX, $8
	JEQ  path8

path4:
	TESTQ CX, CX
	JZ    done

loop4:
	MOVL (SI)(R8*1), X0
	MOVL (SI)(R9*1), X1
	MOVL (SI)(R10*1), X2
	MOVL (SI)(R11*1), X3
	MOVL (SI)(R12*1), X4
	MOVL (DI), X5
	NZBLOCK
	MOVL X0, DX
	MOVL DX, (BX)
	ADDQ R15, SI
	ADDQ $4, DI
	ADDQ $4, BX
	DECQ CX
	JNZ  loop4
	JMP  done

path8:
	TESTQ CX, CX
	JZ    done

loop8:
	MOVQ (SI)(R8*1), X0
	MOVQ (SI)(R9*1), X1
	MOVQ (SI)(R10*1), X2
	MOVQ (SI)(R11*1), X3
	MOVQ (SI)(R12*1), X4
	MOVQ (DI), X5
	NZBLOCK
	MOVQ X0, (BX)
	ADDQ R15, SI
	ADDQ $8, DI
	ADDQ $8, BX
	DECQ CX
	JNZ  loop8
	JMP  done

path16:
	TESTQ CX, CX
	JZ    done

loop16:
	VMOVDQU (SI)(R8*1), X0
	VMOVDQU (SI)(R9*1), X1
	VMOVDQU (SI)(R10*1), X2
	VMOVDQU (SI)(R11*1), X3
	VMOVDQU (SI)(R12*1), X4
	VMOVDQU (DI), X5
	NZBLOCK
	VMOVDQU X0, (BX)
	ADDQ    R15, SI
	ADDQ    $16, DI
	ADDQ    $16, BX
	DECQ    CX
	JNZ     loop16
	JMP     done

path32:
	TESTQ CX, CX
	JZ    done

loop32:
	// rows 0..15 at the column base.
	VMOVDQU (SI)(R8*1), X0
	VMOVDQU (SI)(R9*1), X1
	VMOVDQU (SI)(R10*1), X2
	VMOVDQU (SI)(R11*1), X3
	VMOVDQU (SI)(R12*1), X4
	VMOVDQU (DI), X5
	NZBLOCK
	VMOVDQU X0, (BX)

	// rows 16..31 at base+16.
	LEAQ    16(SI), DX
	VMOVDQU (DX)(R8*1), X0
	VMOVDQU (DX)(R9*1), X1
	VMOVDQU (DX)(R10*1), X2
	VMOVDQU (DX)(R11*1), X3
	VMOVDQU (DX)(R12*1), X4
	VMOVDQU 16(DI), X5
	NZBLOCK
	VMOVDQU X0, 16(BX)

	ADDQ R15, SI
	ADDQ $32, DI
	ADDQ $32, BX
	DECQ CX
	JNZ  loop32

done:
	VZEROUPPER
	RET
