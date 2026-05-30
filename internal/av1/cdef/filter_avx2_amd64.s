// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// AVX2 CDEF per-block filter kernel, bit-exact with filterBlockPureGo
// (see filter.go) and structurally mirroring the proven NEON kernel in
// filter_neon_arm64.s. One CDEF row is eight 16-bit samples, processed in the
// low 128 bits of the AVX2 registers; the int32 tap accumulator uses a full
// 256-bit register. The Go wrapper in filter_avx2_amd64.go restricts this to
// 8-wide blocks.
//
//go:build amd64 && !purego

#include "textflag.h"

#define DST 0
#define INPUT 8
#define DSTSTR 16
#define HEIGHT 24
#define PRI0 32
#define PRI1 40
#define SEC0 48
#define SEC1 56
#define SEC2 64
#define SEC3 72
#define PRITAP0 80
#define PRITAP1 88
#define SECTAP0 96
#define SECTAP1 104
#define PRISTRENGTH 112
#define SECSTRENGTH 120
#define PRISHIFT 128
#define SECSHIFT 136
#define ENABLEPRIMARY 144
#define ENABLESECONDARY 152
#define CLIPPING 160

// Register conventions:
//   SI            = pointer to current input row (input[inputOrigin] + row*BStride*2)
//   DI            = pointer to current dst row
//   CX            = remaining row count (height)
//   R8..R13       = signed byte offsets for the six direction pairs
//                   (pri0,pri1,sec0,sec1,sec2,sec3)
//   R14           = BStride*2 (input row stride in bytes)
//   R15           = dst stride in bytes
//   AX,BX,DX      = scratch
//
//   X0            = current pixel row x (int16x8)
//   X2            = active constrain strength (int16x8, set per phase)
//   X4            = active constrain shift count (low 64 bits, set per phase)
//   X5            = zero (int16x8)
//   X6            = VeryLarge sentinel 0x4000 (int16x8)
//   X7            = running max (int16x8, clipping path only)
//   X15           = running min (int16x8, clipping path only)
//   Y10..Y13      = tap broadcasts (int32x8): priTap0,priTap1,secTap0,secTap1
//   Y14           = int32x8 accumulator (sum)
//   X1,X3,X8,X9   = constrain scratch
//   Y8,Y9         = widen/multiply scratch (aliases X8/X9; reloaded each tap)

// CONSTRAIN computes one tap contribution into the accumulator Y14.
// addrReg holds the byte address of the tap sample. tapY is the int32x8 tap.
// Reads X0 (x), X2 (strength), X4 (shift), X5 (zero). Clobbers X1,X3,X8,X9,Y8.
// v -> diff -> abs -> abs>>shift -> strength-shifted -> max(.,0) -> min(.,abs)
// -> -limit / mask(diff<0) -> blend -> widen -> tap*value -> accumulate.
#define CONSTRAIN(addrReg, tapY) \
	VMOVDQU (addrReg), X8 \
	VPSUBW  X0, X8, X9 \
	VPABSW  X9, X1 \
	VPSRLW  X4, X1, X3 \
	VPSUBW  X3, X2, X3 \
	VPMAXSW X5, X3, X3 \
	VPMINSW X1, X3, X3 \
	VPSUBW  X3, X5, X1 \
	VPSRAW  $15, X9, X9 \
	VPBLENDVB X9, X1, X3, X8 \
	VPMOVSXWD X8, Y8 \
	VPMULLD tapY, Y8, Y8 \
	VPADDD  Y8, Y14, Y14

// CLIPUPDATE folds the raw sample at addrReg into max(X7)/min(X15).
// The VeryLarge sentinel is replaced by 0 for the max (matching maxClip);
// min uses the raw value (matching min4). Loads v into X8 and leaves it there
// so a following CONSTRAIN reuses it without reloading. Clobbers X1,X3,X8.
#define CLIPLOAD(addrReg) \
	VMOVDQU (addrReg), X8 \
	VPCMPEQW X6, X8, X3 \
	VPBLENDVB X3, X5, X8, X1 \
	VPMAXSW X1, X7, X7 \
	VPMINSW X8, X15, X15

// CONSTRAINV reuses the sample already loaded into X8 by CLIPLOAD.
// Reads X0,X2,X4,X5,X8. Clobbers X1,X3,X8,X9,Y8.
#define CONSTRAINV(tapY) \
	VPSUBW  X0, X8, X9 \
	VPABSW  X9, X1 \
	VPSRLW  X4, X1, X3 \
	VPSUBW  X3, X2, X3 \
	VPMAXSW X5, X3, X3 \
	VPMINSW X1, X3, X3 \
	VPSUBW  X3, X5, X1 \
	VPSRAW  $15, X9, X9 \
	VPBLENDVB X9, X1, X3, X8 \
	VPMOVSXWD X8, Y8 \
	VPMULLD tapY, Y8, Y8 \
	VPADDD  Y8, Y14, Y14

// func filterBlockAVX2Asm(ctx *filterBlockAVX2Ctx)
TEXT ·filterBlockAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX

	MOVQ DST(AX), DI
	MOVQ INPUT(AX), SI
	MOVQ DSTSTR(AX), R15
	SHLQ $1, R15            // dst stride in bytes
	MOVQ HEIGHT(AX), CX

	// BStride*2 input row stride in bytes. BStride is a compile-time const in
	// Go (144); encode 2*144 = 288 here.
	MOVQ $288, R14

	// Direction byte offsets.
	MOVQ PRI0(AX), R8
	SHLQ $1, R8
	MOVQ PRI1(AX), R9
	SHLQ $1, R9
	MOVQ SEC0(AX), R10
	SHLQ $1, R10
	MOVQ SEC1(AX), R11
	SHLQ $1, R11
	MOVQ SEC2(AX), R12
	SHLQ $1, R12
	MOVQ SEC3(AX), R13
	SHLQ $1, R13

	// zero and VeryLarge sentinel.
	VPXOR X5, X5, X5
	MOVQ $0x4000, BX
	VMOVQ BX, X6
	VPBROADCASTW X6, X6

	// Tap int32 broadcasts.
	MOVQ PRITAP0(AX), BX
	VMOVQ BX, X10
	VPBROADCASTD X10, Y10
	MOVQ PRITAP1(AX), BX
	VMOVQ BX, X11
	VPBROADCASTD X11, Y11
	MOVQ SECTAP0(AX), BX
	VMOVQ BX, X12
	VPBROADCASTD X12, Y12
	MOVQ SECTAP1(AX), BX
	VMOVQ BX, X13
	VPBROADCASTD X13, Y13

	// Stash strengths/shifts and flags in stack-free GP regs by reloading from
	// ctx as needed. Keep ctx pointer in AX for the duration.

rowLoop:
	TESTQ CX, CX
	JEQ   done

	VMOVDQU (SI), X0          // x = current row
	VPXOR   Y14, Y14, Y14     // sum = 0

	// Decide clipping vs plain.
	MOVQ CLIPPING(AX), BX
	TESTQ BX, BX
	JEQ  plainPath

	// ---- clipping path: both strengths non-zero ----
	VMOVDQA X0, X7            // max = x
	VMOVDQA X0, X15           // min = x

	// primary phase strength/shift
	MOVQ PRISTRENGTH(AX), BX
	VMOVQ BX, X2
	VPBROADCASTW X2, X2
	MOVQ PRISHIFT(AX), BX
	VMOVQ BX, X4

	LEAQ (SI)(R8*1), DX
	CLIPLOAD(DX)
	CONSTRAINV(Y10)
	MOVQ SI, DX
	SUBQ R8, DX
	CLIPLOAD(DX)
	CONSTRAINV(Y10)
	LEAQ (SI)(R9*1), DX
	CLIPLOAD(DX)
	CONSTRAINV(Y11)
	MOVQ SI, DX
	SUBQ R9, DX
	CLIPLOAD(DX)
	CONSTRAINV(Y11)

	// secondary phase strength/shift
	MOVQ SECSTRENGTH(AX), BX
	VMOVQ BX, X2
	VPBROADCASTW X2, X2
	MOVQ SECSHIFT(AX), BX
	VMOVQ BX, X4

	LEAQ (SI)(R10*1), DX
	CLIPLOAD(DX)
	CONSTRAINV(Y12)
	MOVQ SI, DX
	SUBQ R10, DX
	CLIPLOAD(DX)
	CONSTRAINV(Y12)
	LEAQ (SI)(R12*1), DX
	CLIPLOAD(DX)
	CONSTRAINV(Y12)
	MOVQ SI, DX
	SUBQ R12, DX
	CLIPLOAD(DX)
	CONSTRAINV(Y12)
	LEAQ (SI)(R11*1), DX
	CLIPLOAD(DX)
	CONSTRAINV(Y13)
	MOVQ SI, DX
	SUBQ R11, DX
	CLIPLOAD(DX)
	CONSTRAINV(Y13)
	LEAQ (SI)(R13*1), DX
	CLIPLOAD(DX)
	CONSTRAINV(Y13)
	MOVQ SI, DX
	SUBQ R13, DX
	CLIPLOAD(DX)
	CONSTRAINV(Y13)

	// fold with clamp to [min,max]
	VPSRAD $31, Y14, Y8       // sign = sum>>31 (-1 if neg)
	MOVQ $8, BX
	VMOVQ BX, X9
	VPBROADCASTD X9, Y9
	VPADDD Y9, Y14, Y14       // +8
	VPADDD Y8, Y14, Y14       // + sign
	VPSRAD $4, Y14, Y14       // >>4 (arithmetic)
	VPMOVZXWD X0, Y8          // x widened (zero-extend)
	VPADDD Y8, Y14, Y14       // + x
	VPMOVSXWD X15, Y8         // min (signed)
	VPMAXSD Y8, Y14, Y14
	VPMOVSXWD X7, Y8          // max (signed)
	VPMINSD Y8, Y14, Y14
	JMP store

plainPath:
	MOVQ ENABLEPRIMARY(AX), BX
	TESTQ BX, BX
	JEQ  plainNoPrimary

	MOVQ PRISTRENGTH(AX), BX
	VMOVQ BX, X2
	VPBROADCASTW X2, X2
	MOVQ PRISHIFT(AX), BX
	VMOVQ BX, X4

	LEAQ (SI)(R8*1), DX
	CONSTRAIN(DX, Y10)
	MOVQ SI, DX
	SUBQ R8, DX
	CONSTRAIN(DX, Y10)
	LEAQ (SI)(R9*1), DX
	CONSTRAIN(DX, Y11)
	MOVQ SI, DX
	SUBQ R9, DX
	CONSTRAIN(DX, Y11)

plainNoPrimary:
	MOVQ ENABLESECONDARY(AX), BX
	TESTQ BX, BX
	JEQ  plainNoSecondary

	MOVQ SECSTRENGTH(AX), BX
	VMOVQ BX, X2
	VPBROADCASTW X2, X2
	MOVQ SECSHIFT(AX), BX
	VMOVQ BX, X4

	LEAQ (SI)(R10*1), DX
	CONSTRAIN(DX, Y12)
	MOVQ SI, DX
	SUBQ R10, DX
	CONSTRAIN(DX, Y12)
	LEAQ (SI)(R12*1), DX
	CONSTRAIN(DX, Y12)
	MOVQ SI, DX
	SUBQ R12, DX
	CONSTRAIN(DX, Y12)
	LEAQ (SI)(R11*1), DX
	CONSTRAIN(DX, Y13)
	MOVQ SI, DX
	SUBQ R11, DX
	CONSTRAIN(DX, Y13)
	LEAQ (SI)(R13*1), DX
	CONSTRAIN(DX, Y13)
	MOVQ SI, DX
	SUBQ R13, DX
	CONSTRAIN(DX, Y13)

plainNoSecondary:
	VPSRAD $31, Y14, Y8
	MOVQ $8, BX
	VMOVQ BX, X9
	VPBROADCASTD X9, Y9
	VPADDD Y9, Y14, Y14
	VPADDD Y8, Y14, Y14
	VPSRAD $4, Y14, Y14
	VPMOVZXWD X0, Y8
	VPADDD Y8, Y14, Y14

store:
	// Truncate int32 lanes to uint16 (matching uint16(y)) and store 8 samples.
	MOVQ $0xffff, BX
	VMOVQ BX, X9
	VPBROADCASTD X9, Y9
	VPAND Y9, Y14, Y14        // mask low 16 bits per lane
	VEXTRACTI128 $1, Y14, X1  // high 4 lanes
	VPACKUSDW X1, X14, X8     // pack to 8 uint16 (no clamp after masking)
	VMOVDQU X8, (DI)

	ADDQ R14, SI
	ADDQ R15, DI
	SUBQ $1, CX
	JMP  rowLoop

done:
	VZEROUPPER
	RET
