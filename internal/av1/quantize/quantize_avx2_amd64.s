// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

#include "textflag.h"

// AVX2 aom_quantize_b lane math, eight int32 coefficients per iteration:
//
//   keep  = |c| >= zbin           (as NOT(zbin > |c|))
//   tmp   = min(|c| + round, 32767)
//   t     = (tmp * quant) >> 16   (the invert_quant factor lies in
//                                  (-32768, 1], so the 32-bit product is
//                                  exact: |tmp*quant| <= 32767<<15)
//   level = (t + tmp) >> (l - ts) (invert_quant's power-of-two multiply
//                                  folded into the variable shift)
//
// signed by the coefficient, masked by keep, packed to int16.

#define CCOEFF 0
#define COUT   8
#define CCOUNT 16
#define CQUANT 24
#define CROUND 32
#define CZBIN  40
#define CSHIFT 48

// func quantizeBAVX2Asm(ctx *quantizeBAVX2Ctx)
TEXT ·quantizeBAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ CCOEFF(AX), SI
	MOVQ COUT(AX), DX
	MOVQ CCOUNT(AX), CX
	VPBROADCASTD CQUANT(AX), Y11
	VPBROADCASTD CROUND(AX), Y12
	VPBROADCASTD CZBIN(AX), Y13
	VMOVQ CSHIFT(AX), X10 // right-shift count (l - txScale)
	MOVL $32767, BX
	VMOVD BX, X0
	VPBROADCASTD X0, Y14
	SHRQ $3, CX // eight lanes per iteration

bloop:
	VMOVDQU (SI), Y0
	VPABSD  Y0, Y1
	VPCMPGTD Y1, Y13, Y3  // zbin > |c| -> drop mask
	VPADDD  Y12, Y1, Y4
	VPMINSD Y14, Y4, Y4   // tmp
	VPMULLD Y11, Y4, Y5
	VPSRAD  $16, Y5, Y5   // t
	VPADDD  Y5, Y4, Y4
	VPSRAD  X10, Y4, Y4   // level
	VPMINSD Y14, Y4, Y4
	VPSIGND Y0, Y4, Y4    // sign by the coefficient
	VPANDN  Y4, Y3, Y4    // keep = ~drop
	VEXTRACTI128 $1, Y4, X5
	VPACKSSDW X5, X4, X4
	VMOVDQU X4, (DX)
	ADDQ $32, SI
	ADDQ $16, DX
	SUBQ $1, CX
	JNZ  bloop
	VZEROUPPER
	RET

// AVX2 av1_quantize_fp lane math, eight int32 coefficients per iteration:
//
//   keep  = (|c| << (1+ts)) >= dequant
//   a     = min(|c| + round, 32767)
//   level = (a * quant) >> (16 - ts)  (quant <= 1<<14 keeps the product
//                                      inside int32)
//
// signed by the coefficient, masked by keep, packed to int16.

#define FCOEFF  0
#define FOUT    8
#define FCOUNT  16
#define FQUANT  24
#define FROUND  32
#define FDEQ    40
#define FSHIFTL 48
#define FSHIFTR 56

// func quantizeFPAVX2Asm(ctx *quantizeFPAVX2Ctx)
TEXT ·quantizeFPAVX2Asm(SB), NOSPLIT, $0-8
	MOVQ ctx+0(FP), AX
	MOVQ FCOEFF(AX), SI
	MOVQ FOUT(AX), DX
	MOVQ FCOUNT(AX), CX
	VPBROADCASTD FQUANT(AX), Y11
	VPBROADCASTD FROUND(AX), Y12
	VPBROADCASTD FDEQ(AX), Y13
	VMOVQ FSHIFTL(AX), X9
	VMOVQ FSHIFTR(AX), X10
	MOVL $32767, BX
	VMOVD BX, X0
	VPBROADCASTD X0, Y14
	SHRQ $3, CX

floop:
	VMOVDQU (SI), Y0
	VPABSD  Y0, Y1
	VPSLLD  X9, Y1, Y2
	VPCMPGTD Y2, Y13, Y3  // dequant > (|c| << (1+ts)) -> drop mask
	VPADDD  Y12, Y1, Y4
	VPMINSD Y14, Y4, Y4   // a
	VPMULLD Y11, Y4, Y4
	VPSRAD  X10, Y4, Y4   // level
	VPMINSD Y14, Y4, Y4
	VPSIGND Y0, Y4, Y4
	VPANDN  Y4, Y3, Y4
	VEXTRACTI128 $1, Y4, X5
	VPACKSSDW X5, X4, X4
	VMOVDQU X4, (DX)
	ADDQ $32, SI
	ADDQ $16, DX
	SUBQ $1, CX
	JNZ  floop
	VZEROUPPER
	RET
