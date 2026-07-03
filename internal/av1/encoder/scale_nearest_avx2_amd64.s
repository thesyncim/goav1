// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 nearest-neighbour downscale row kernels for the exact 2x and 4x ratios.
// Each kernel writes one 16-byte destination group per Groups iteration, picking
// every 2nd (down2) or 4th (down4) source sample. Wanted lanes are isolated with
// an in-range PAND mask then compacted with PACKUSDW/PACKUSWB; because the masked
// values stay within [0,255] / [0,65535] the saturating packs are lossless, so
// the result is bit-exact with scalePlaneNearestPureGo.

// maskW00FF: keep the low byte of each 16-bit lane (8-bit down2 even-byte pick).
DATA maskW00FF<>+0(SB)/8, $0x00ff00ff00ff00ff
DATA maskW00FF<>+8(SB)/8, $0x00ff00ff00ff00ff
GLOBL maskW00FF<>(SB), RODATA|NOPTR, $16

// maskD000000FF: keep the low byte of each 32-bit lane (8-bit down4 pick).
DATA maskD000000FF<>+0(SB)/8, $0x000000ff000000ff
DATA maskD000000FF<>+8(SB)/8, $0x000000ff000000ff
GLOBL maskD000000FF<>(SB), RODATA|NOPTR, $16

// maskD0000FFFF: keep the low word of each 32-bit lane (16-bit even-word pick).
DATA maskD0000FFFF<>+0(SB)/8, $0x0000ffff0000ffff
DATA maskD0000FFFF<>+8(SB)/8, $0x0000ffff0000ffff
GLOBL maskD0000FFFF<>(SB), RODATA|NOPTR, $16

// scaleNearestRowAVX2Ctx field offsets.
#define SCDST    0
#define SCSRC    8
#define SCGROUPS 16

// func scaleNearestRowDown2AVX2(ctx *scaleNearestRowAVX2Ctx)
// 8-bit down2: 16 dst bytes from 32 src bytes, even bytes.
TEXT ·scaleNearestRowDown2AVX2(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ SCDST(AX), DI
	MOVQ SCSRC(AX), SI
	MOVQ SCGROUPS(AX), CX
	MOVOU maskW00FF<>(SB), X7
loopD2b:
	MOVOU    (SI), X0
	MOVOU    16(SI), X1
	PAND     X7, X0
	PAND     X7, X1
	PACKUSWB X1, X0
	MOVOU    X0, (DI)
	ADDQ $32, SI
	ADDQ $16, DI
	DECQ CX
	JNZ  loopD2b
	RET

// func scaleNearestRowDown4AVX2(ctx *scaleNearestRowAVX2Ctx)
// 8-bit down4: 16 dst bytes from 64 src bytes, every 4th byte.
TEXT ·scaleNearestRowDown4AVX2(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ SCDST(AX), DI
	MOVQ SCSRC(AX), SI
	MOVQ SCGROUPS(AX), CX
	MOVOU maskD000000FF<>(SB), X7
loopD4b:
	MOVOU    (SI), X0
	MOVOU    16(SI), X1
	MOVOU    32(SI), X2
	MOVOU    48(SI), X3
	PAND     X7, X0
	PAND     X7, X1
	PAND     X7, X2
	PAND     X7, X3
	PACKUSDW X1, X0
	PACKUSDW X3, X2
	PACKUSWB X2, X0
	MOVOU    X0, (DI)
	ADDQ $64, SI
	ADDQ $16, DI
	DECQ CX
	JNZ  loopD4b
	RET

// func scaleNearestRow16Down2AVX2(ctx *scaleNearestRowAVX2Ctx)
// 16-bit down2: 8 dst words from 16 src words, even words.
TEXT ·scaleNearestRow16Down2AVX2(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ SCDST(AX), DI
	MOVQ SCSRC(AX), SI
	MOVQ SCGROUPS(AX), CX
	MOVOU maskD0000FFFF<>(SB), X7
loopD2w:
	MOVOU    (SI), X0
	MOVOU    16(SI), X1
	PAND     X7, X0
	PAND     X7, X1
	PACKUSDW X1, X0
	MOVOU    X0, (DI)
	ADDQ $32, SI
	ADDQ $16, DI
	DECQ CX
	JNZ  loopD2w
	RET

// func scaleNearestRow16Down4AVX2(ctx *scaleNearestRowAVX2Ctx)
// 16-bit down4: 8 dst words from 32 src words, every 4th word. Even-word pick
// applied twice: 32 -> 16 -> 8.
TEXT ·scaleNearestRow16Down4AVX2(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ SCDST(AX), DI
	MOVQ SCSRC(AX), SI
	MOVQ SCGROUPS(AX), CX
	MOVOU maskD0000FFFF<>(SB), X7
loopD4w:
	MOVOU    (SI), X0
	MOVOU    16(SI), X1
	MOVOU    32(SI), X2
	MOVOU    48(SI), X3
	PAND     X7, X0
	PAND     X7, X1
	PAND     X7, X2
	PAND     X7, X3
	PACKUSDW X1, X0
	PACKUSDW X3, X2
	// X0 = src even words 0..14, X2 = even words 16..30; pick every other again.
	PAND     X7, X0
	PAND     X7, X2
	PACKUSDW X2, X0
	MOVOU    X0, (DI)
	ADDQ $64, SI
	ADDQ $16, DI
	DECQ CX
	JNZ  loopD4w
	RET
