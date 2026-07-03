// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// func loadSampleRows8AVX2Asm(dst *uint16, src *byte, dstStride uintptr, srcStride uintptr, width uintptr, height uintptr)
//
// Widening row copy u8 -> u16 for postfilter sample staging:
//     dst[y][x] = uint16(src[y][x])
// dstStride is in uint16 elements; srcStride is in bytes. Sixteen columns per
// iteration (VPMOVZXBW 16 u8 -> 16 u16), an 8-wide group for the mid tail, and
// one overlapping 8-wide group aligned to the row end for the final 1..7
// columns (rewriting identical values). The wrapper only routes width >= 8
// here, so the overlapping tail never reads before the row start. Output is
// bit-identical to loadSampleRows8PureGo.
TEXT ·loadSampleRows8AVX2Asm(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ dstStride+16(FP), R8 // dst stride in uint16 elements
	SHLQ $1, R8               // dst stride in bytes
	MOVQ srcStride+24(FP), R9 // src stride in bytes
	MOVQ width+32(FP), R10
	MOVQ height+40(FP), R11

rowLoop:
	TESTQ R11, R11
	JZ    done
	MOVQ  SI, AX  // src row cursor
	MOVQ  DI, BX  // dst row cursor
	MOVQ  R10, CX // columns remaining

col16:
	CMPQ      CX, $16
	JL        col8
	VPMOVZXBW (AX), Y0 // 16 u8 -> 16 u16
	VMOVDQU   Y0, (BX)
	ADDQ      $16, AX
	ADDQ      $32, BX
	SUBQ      $16, CX
	JMP       col16

col8:
	CMPQ      CX, $8
	JL        tail
	VPMOVZXBW (AX), X0 // 8 u8 -> 8 u16
	VMOVDQU   X0, (BX)
	ADDQ      $8, AX
	ADDQ      $16, BX
	SUBQ      $8, CX

tail:
	TESTQ CX, CX
	JZ    nextRow
	// Overlapping final 8-wide group ending at the row's last column.
	LEAQ      -8(SI)(R10*1), AX // src row + width - 8
	LEAQ      (DI)(R10*2), BX   // dst row + 2*width
	SUBQ      $16, BX           // dst row + 2*(width - 8)
	VPMOVZXBW (AX), X0
	VMOVDQU   X0, (BX)

nextRow:
	ADDQ R9, SI
	ADDQ R8, DI
	DECQ R11
	JMP  rowLoop

done:
	VZEROUPPER
	RET
