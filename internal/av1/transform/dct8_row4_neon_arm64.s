// NEON-accelerated inverse DCT8 four-row kernel.
//
// Four stride-1 rows are transposed so coefficient i occupies the four int32
// lanes of one vector. The generated butterfly uses the same coefficient
// rewriting and stage clamps as the checked DCT32 four-lane kernel; generator
// range proofs and differential tests cover every AV1 8/10/12-bit clamp.
//
// Generated from the pure-Go inverseDCT8 butterfly by tools/itxgen.
//
// SPDX-License-Identifier: BSD-2-Clause
//
//go:build arm64 && !purego

#include "textflag.h"

// func inverseDCT8Row4NEON(r0, r1, r2, r3 *int32, min, max int64)
TEXT ·inverseDCT8Row4NEON(SB), NOSPLIT, $0-48
	MOVD r0+0(FP), R0
	MOVD r1+8(FP), R1
	MOVD r2+16(FP), R2
	MOVD r3+24(FP), R3
	MOVD min+32(FP), R4
	MOVD max+40(FP), R5
	WORD $0xad400400 // ldp	q0, q1, [x0]
	WORD $0xad400c22 // ldp	q2, q3, [x1]
	WORD $0x4e822813 // trn1.4s	v19, v0, v2
	WORD $0x4e826802 // trn2.4s	v2, v0, v2
	WORD $0xad401840 // ldp	q0, q6, [x2]
	WORD $0xad401c65 // ldp	q5, q7, [x3]
	WORD $0x4e852810 // trn1.4s	v16, v0, v5
	WORD $0x4e856800 // trn2.4s	v0, v0, v5
	WORD $0x4ed07a72 // zip2.2d	v18, v19, v16
	WORD $0x6e180613 // mov.d	v19[1], v16[0]
	WORD $0x4ec07845 // zip2.2d	v5, v2, v0
	WORD $0x6e180402 // mov.d	v2[1], v0[0]
	WORD $0x4e832820 // trn1.4s	v0, v1, v3
	WORD $0x4e836823 // trn2.4s	v3, v1, v3
	WORD $0x4e8728c1 // trn1.4s	v1, v6, v7
	WORD $0x4e8768c4 // trn2.4s	v4, v6, v7
	WORD $0x4ec17806 // zip2.2d	v6, v0, v1
	WORD $0x4ea01c07 // mov.16b	v7, v0
	WORD $0x6e180427 // mov.d	v7[1], v1[0]
	WORD $0x4ea31c71 // mov.16b	v17, v3
	WORD $0x6e180491 // mov.d	v17[1], v4[0]
	WORD $0x4e040c80 // dup.4s	v0, w4
	WORD $0x4e040ca1 // dup.4s	v1, w5
	WORD $0x4ec47874 // zip2.2d	v20, v3, v4
	WORD $0x4eb384e3 // add.4s	v3, v7, v19
	WORD $0x4f0506a4 // movi.4s	v4, #0xb5
	WORD $0x4ea49c63 // mul.4s	v3, v3, v4
	WORD $0x4f382470 // srshr.4s	v16, v3, #0x8
	WORD $0x6ea78667 // sub.4s	v7, v19, v7
	WORD $0x4ea49ce7 // mul.4s	v7, v7, v4
	WORD $0x4f3824f3 // srshr.4s	v19, v7, #0x8
	WORD $0x5280c3e8 // mov	w8, #0x61f              ; =1567
	WORD $0x4e040d15 // dup.4s	v21, w8
	WORD $0x4eb59e56 // mul.4s	v22, v18, v21
	WORD $0x52802708 // mov	w8, #0x138              ; =312
	WORD $0x4e040d17 // dup.4s	v23, w8
	WORD $0x4eb794d6 // mla.4s	v22, v6, v23
	WORD $0x4f3426d6 // srshr.4s	v22, v22, #0xc
	WORD $0x6ea686d6 // sub.4s	v22, v22, v6
	WORD $0x128026e8 // mov	w8, #-0x138             ; =-312
	WORD $0x4e040d17 // dup.4s	v23, w8
	WORD $0x4eb79e57 // mul.4s	v23, v18, v23
	WORD $0x4eb594d7 // mla.4s	v23, v6, v21
	WORD $0x4f3436f2 // srsra.4s	v18, v23, #0xc
	WORD $0x6eb28606 // sub.4s	v6, v16, v18
	WORD $0x4f383472 // srsra.4s	v18, v3, #0x8
	WORD $0x4ea06643 // smax.4s	v3, v18, v0
	WORD $0x4ea16c63 // smin.4s	v3, v3, v1
	WORD $0x6eb68672 // sub.4s	v18, v19, v22
	WORD $0x4f3834f6 // srsra.4s	v22, v7, #0x8
	WORD $0x4ea066c7 // smax.4s	v7, v22, v0
	WORD $0x4ea16cf0 // smin.4s	v16, v7, v1
	WORD $0x4ea06647 // smax.4s	v7, v18, v0
	WORD $0x4ea16ce7 // smin.4s	v7, v7, v1
	WORD $0x4ea064c6 // smax.4s	v6, v6, v0
	WORD $0x4ea16cc6 // smin.4s	v6, v6, v1
	WORD $0x528063e8 // mov	w8, #0x31f              ; =799
	WORD $0x4e040d12 // dup.4s	v18, w8
	WORD $0x4eb29c53 // mul.4s	v19, v2, v18
	WORD $0x4f0205f5 // movi.4s	v21, #0x4f
	WORD $0x4eb59693 // mla.4s	v19, v20, v21
	WORD $0x4f342673 // srshr.4s	v19, v19, #0xc
	WORD $0x6eb48673 // sub.4s	v19, v19, v20
	WORD $0x5280d4e8 // mov	w8, #0x6a7              ; =1703
	WORD $0x4e040d15 // dup.4s	v21, w8
	WORD $0x12808e28 // mov	w8, #-0x472             ; =-1138
	WORD $0x4e040d16 // dup.4s	v22, w8
	WORD $0x4eb69cb6 // mul.4s	v22, v5, v22
	WORD $0x4eb59636 // mla.4s	v22, v17, v21
	WORD $0x4f3526d7 // srshr.4s	v23, v22, #0xb
	WORD $0x52808e48 // mov	w8, #0x472              ; =1138
	WORD $0x4e040d18 // dup.4s	v24, w8
	WORD $0x4eb59ca5 // mul.4s	v5, v5, v21
	WORD $0x4eb89625 // mla.4s	v5, v17, v24
	WORD $0x4f3524b1 // srshr.4s	v17, v5, #0xb
	WORD $0x6f0205d5 // mvni.4s	v21, #0x4e
	WORD $0x4eb59c55 // mul.4s	v21, v2, v21
	WORD $0x4eb29695 // mla.4s	v21, v20, v18
	WORD $0x4f3436a2 // srsra.4s	v2, v21, #0xc
	WORD $0x6eb78672 // sub.4s	v18, v19, v23
	WORD $0x4f3536d3 // srsra.4s	v19, v22, #0xb
	WORD $0x4ea06673 // smax.4s	v19, v19, v0
	WORD $0x4ea16e73 // smin.4s	v19, v19, v1
	WORD $0x4ea06652 // smax.4s	v18, v18, v0
	WORD $0x4ea16e52 // smin.4s	v18, v18, v1
	WORD $0x6eb18451 // sub.4s	v17, v2, v17
	WORD $0x4f3534a2 // srsra.4s	v2, v5, #0xb
	WORD $0x4ea06442 // smax.4s	v2, v2, v0
	WORD $0x4ea16c42 // smin.4s	v2, v2, v1
	WORD $0x4ea06625 // smax.4s	v5, v17, v0
	WORD $0x4ea16ca5 // smin.4s	v5, v5, v1
	WORD $0x6eb284b1 // sub.4s	v17, v5, v18
	WORD $0x4ea49e31 // mul.4s	v17, v17, v4
	WORD $0x4f382634 // srshr.4s	v20, v17, #0x8
	WORD $0x4eb284a5 // add.4s	v5, v5, v18
	WORD $0x4ea49ca4 // mul.4s	v4, v5, v4
	WORD $0x4f382485 // srshr.4s	v5, v4, #0x8
	WORD $0x4ea38452 // add.4s	v18, v2, v3
	WORD $0x4ea06652 // smax.4s	v18, v18, v0
	WORD $0x4ea16e52 // smin.4s	v18, v18, v1
	WORD $0x6ea58605 // sub.4s	v5, v16, v5
	WORD $0x4f383490 // srsra.4s	v16, v4, #0x8
	WORD $0x4ea06604 // smax.4s	v4, v16, v0
	WORD $0x4ea16c84 // smin.4s	v4, v4, v1
	WORD $0x6eb484f0 // sub.4s	v16, v7, v20
	WORD $0x4f383627 // srsra.4s	v7, v17, #0x8
	WORD $0x4ea064e7 // smax.4s	v7, v7, v0
	WORD $0x4ea16ce7 // smin.4s	v7, v7, v1
	WORD $0x4ea68671 // add.4s	v17, v19, v6
	WORD $0x4ea06631 // smax.4s	v17, v17, v0
	WORD $0x4ea16e31 // smin.4s	v17, v17, v1
	WORD $0x6eb384c6 // sub.4s	v6, v6, v19
	WORD $0x4ea064c6 // smax.4s	v6, v6, v0
	WORD $0x4ea16cc6 // smin.4s	v6, v6, v1
	WORD $0x4ea06610 // smax.4s	v16, v16, v0
	WORD $0x4ea16e10 // smin.4s	v16, v16, v1
	WORD $0x4ea064a5 // smax.4s	v5, v5, v0
	WORD $0x4ea16ca5 // smin.4s	v5, v5, v1
	WORD $0x6ea28462 // sub.4s	v2, v3, v2
	WORD $0x4ea06440 // smax.4s	v0, v2, v0
	WORD $0x4ea16c00 // smin.4s	v0, v0, v1
	WORD $0x4e842a41 // trn1.4s	v1, v18, v4
	WORD $0x4e846a42 // trn2.4s	v2, v18, v4
	WORD $0x4e9128e3 // trn1.4s	v3, v7, v17
	WORD $0x4e9168e4 // trn2.4s	v4, v7, v17
	WORD $0x4ec37827 // zip2.2d	v7, v1, v3
	WORD $0x6e180461 // mov.d	v1[1], v3[0]
	WORD $0x4ec47843 // zip2.2d	v3, v2, v4
	WORD $0x6e180482 // mov.d	v2[1], v4[0]
	WORD $0x3d800001 // str	q1, [x0]
	WORD $0x3d800022 // str	q2, [x1]
	WORD $0x3d800047 // str	q7, [x2]
	WORD $0x3d800063 // str	q3, [x3]
	WORD $0x4e9028c1 // trn1.4s	v1, v6, v16
	WORD $0x4e9068c2 // trn2.4s	v2, v6, v16
	WORD $0x4e8028a3 // trn1.4s	v3, v5, v0
	WORD $0x4e8068a0 // trn2.4s	v0, v5, v0
	WORD $0x4ec37824 // zip2.2d	v4, v1, v3
	WORD $0x6e180461 // mov.d	v1[1], v3[0]
	WORD $0x4ec07843 // zip2.2d	v3, v2, v0
	WORD $0x6e180402 // mov.d	v2[1], v0[0]
	WORD $0x3d800401 // str	q1, [x0, #0x10]
	WORD $0x3d800422 // str	q2, [x1, #0x10]
	WORD $0x3d800444 // str	q4, [x2, #0x10]
	WORD $0x3d800463 // str	q3, [x3, #0x10]
	WORD $0xd65f03c0 // ret

