// NEON-accelerated inverse ADST8 four-row kernel.
//
// Four stride-1 rows are transposed so coefficient i occupies the four int32
// lanes of one vector. Coefficient rewrites keep every multiply accumulator in
// range under the checked +/-2^19 stage envelope; the generator harness and Go
// differential tests cover all AV1 8/10/12-bit clamps.
//
// Generated from inverseADST8Core by tools/itxgen.
//
// SPDX-License-Identifier: BSD-2-Clause
//
//go:build arm64 && !purego

#include "textflag.h"

// func inverseADST8Row4NEON(r0, r1, r2, r3 *int32, min, max int64)
TEXT ·inverseADST8Row4NEON(SB), NOSPLIT, $0-48
	MOVD r0+0(FP), R0
	MOVD r1+8(FP), R1
	MOVD r2+16(FP), R2
	MOVD r3+24(FP), R3
	MOVD min+32(FP), R4
	MOVD max+40(FP), R5
	WORD $0xad400400 // ldp	q0, q1, [x0]
	WORD $0xad400c22 // ldp	q2, q3, [x1]
	WORD $0x4e822804 // trn1.4s	v4, v0, v2
	WORD $0x4e826802 // trn2.4s	v2, v0, v2
	WORD $0xad401440 // ldp	q0, q5, [x2]
	WORD $0xad401c66 // ldp	q6, q7, [x3]
	WORD $0x4e862810 // trn1.4s	v16, v0, v6
	WORD $0x4e866800 // trn2.4s	v0, v0, v6
	WORD $0x4ed07891 // zip2.2d	v17, v4, v16
	WORD $0x4ea41c86 // mov.16b	v6, v4
	WORD $0x6e180606 // mov.d	v6[1], v16[0]
	WORD $0x4ec07850 // zip2.2d	v16, v2, v0
	WORD $0x6e180402 // mov.d	v2[1], v0[0]
	WORD $0x4e832832 // trn1.4s	v18, v1, v3
	WORD $0x4e836824 // trn2.4s	v4, v1, v3
	WORD $0x4e8728a1 // trn1.4s	v1, v5, v7
	WORD $0x4e8768a7 // trn2.4s	v7, v5, v7
	WORD $0x4ec17a43 // zip2.2d	v3, v18, v1
	WORD $0x6e180432 // mov.d	v18[1], v1[0]
	WORD $0x4ec77885 // zip2.2d	v5, v4, v7
	WORD $0x6e1804e4 // mov.d	v4[1], v7[0]
	WORD $0x4e040c80 // dup.4s	v0, w4
	WORD $0x4e040ca1 // dup.4s	v1, w5
	WORD $0x6f000667 // mvni.4s	v7, #0x13
	WORD $0x52803228 // mov	w8, #0x191              ; =401
	WORD $0x4e040d13 // dup.4s	v19, w8
	WORD $0x4eb39cd4 // mul.4s	v20, v6, v19
	WORD $0x4ea794b4 // mla.4s	v20, v5, v7
	WORD $0x4f000687 // movi.4s	v7, #0x14
	WORD $0x4ea79cc7 // mul.4s	v7, v6, v7
	WORD $0x4eb394a7 // mla.4s	v7, v5, v19
	WORD $0x4f343685 // srsra.4s	v5, v20, #0xc
	WORD $0x4f3424e7 // srshr.4s	v7, v7, #0xc
	WORD $0x6ea684e6 // sub.4s	v6, v7, v6
	WORD $0x12803c68 // mov	w8, #-0x1e4             ; =-484
	WORD $0x4e040d07 // dup.4s	v7, w8
	WORD $0x5280f168 // mov	w8, #0x78b              ; =1931
	WORD $0x4e040d13 // dup.4s	v19, w8
	WORD $0x4eb39e34 // mul.4s	v20, v17, v19
	WORD $0x4ea79494 // mla.4s	v20, v4, v7
	WORD $0x52803c88 // mov	w8, #0x1e4              ; =484
	WORD $0x4e040d07 // dup.4s	v7, w8
	WORD $0x4ea79e27 // mul.4s	v7, v17, v7
	WORD $0x4eb39487 // mla.4s	v7, v4, v19
	WORD $0x4f343684 // srsra.4s	v4, v20, #0xc
	WORD $0x4f3424e7 // srshr.4s	v7, v7, #0xc
	WORD $0x5280a268 // mov	w8, #0x513              ; =1299
	WORD $0x4e040d13 // dup.4s	v19, w8
	WORD $0x6eb184e7 // sub.4s	v7, v7, v17
	WORD $0x4eb39e11 // mul.4s	v17, v16, v19
	WORD $0x5280c5e8 // mov	w8, #0x62f              ; =1583
	WORD $0x4e040d13 // dup.4s	v19, w8
	WORD $0x4eb39651 // mla.4s	v17, v18, v19
	WORD $0x4f352634 // srshr.4s	v20, v17, #0xb
	WORD $0x4eb39e10 // mul.4s	v16, v16, v19
	WORD $0x1280a248 // mov	w8, #-0x513             ; =-1299
	WORD $0x4e040d13 // dup.4s	v19, w8
	WORD $0x4eb39650 // mla.4s	v16, v18, v19
	WORD $0x4f352612 // srshr.4s	v18, v16, #0xb
	WORD $0x528094a8 // mov	w8, #0x4a5              ; =1189
	WORD $0x4e040d13 // dup.4s	v19, w8
	WORD $0x4eb39c53 // mul.4s	v19, v2, v19
	WORD $0x6f0505f5 // mvni.4s	v21, #0xaf
	WORD $0x4eb59473 // mla.4s	v19, v3, v21
	WORD $0x4eb59c55 // mul.4s	v21, v2, v21
	WORD $0x12809488 // mov	w8, #-0x4a5             ; =-1189
	WORD $0x4e040d16 // dup.4s	v22, w8
	WORD $0x4eb69475 // mla.4s	v21, v3, v22
	WORD $0x4f343663 // srsra.4s	v3, v19, #0xc
	WORD $0x4f3436a2 // srsra.4s	v2, v21, #0xc
	WORD $0x6eb484b3 // sub.4s	v19, v5, v20
	WORD $0x4f353625 // srsra.4s	v5, v17, #0xb
	WORD $0x4ea064a5 // smax.4s	v5, v5, v0
	WORD $0x4ea16ca5 // smin.4s	v5, v5, v1
	WORD $0x6eb284d1 // sub.4s	v17, v6, v18
	WORD $0x4f353606 // srsra.4s	v6, v16, #0xb
	WORD $0x4ea064c6 // smax.4s	v6, v6, v0
	WORD $0x4ea16cc6 // smin.4s	v6, v6, v1
	WORD $0x4ea48470 // add.4s	v16, v3, v4
	WORD $0x4ea06610 // smax.4s	v16, v16, v0
	WORD $0x4ea16e10 // smin.4s	v16, v16, v1
	WORD $0x4ea78452 // add.4s	v18, v2, v7
	WORD $0x4ea06652 // smax.4s	v18, v18, v0
	WORD $0x4ea16e52 // smin.4s	v18, v18, v1
	WORD $0x4ea06673 // smax.4s	v19, v19, v0
	WORD $0x4ea16e73 // smin.4s	v19, v19, v1
	WORD $0x4ea06631 // smax.4s	v17, v17, v0
	WORD $0x4ea16e31 // smin.4s	v17, v17, v1
	WORD $0x6ea38483 // sub.4s	v3, v4, v3
	WORD $0x4ea06463 // smax.4s	v3, v3, v0
	WORD $0x4ea16c63 // smin.4s	v3, v3, v1
	WORD $0x6ea284e2 // sub.4s	v2, v7, v2
	WORD $0x4ea06442 // smax.4s	v2, v2, v0
	WORD $0x128026e8 // mov	w8, #-0x138             ; =-312
	WORD $0x4e040d04 // dup.4s	v4, w8
	WORD $0x4ea16c42 // smin.4s	v2, v2, v1
	WORD $0x4ea49e67 // mul.4s	v7, v19, v4
	WORD $0x5280c3e8 // mov	w8, #0x61f              ; =1567
	WORD $0x4e040d14 // dup.4s	v20, w8
	WORD $0x4eb49627 // mla.4s	v7, v17, v20
	WORD $0x4eb49e75 // mul.4s	v21, v19, v20
	WORD $0x4f3434f3 // srsra.4s	v19, v7, #0xc
	WORD $0x52802708 // mov	w8, #0x138              ; =312
	WORD $0x4e040d07 // dup.4s	v7, w8
	WORD $0x4ea79635 // mla.4s	v21, v17, v7
	WORD $0x4f3426a7 // srshr.4s	v7, v21, #0xc
	WORD $0x6eb184e7 // sub.4s	v7, v7, v17
	WORD $0x1280c3c8 // mov	w8, #-0x61f             ; =-1567
	WORD $0x4e040d11 // dup.4s	v17, w8
	WORD $0x4eb19c71 // mul.4s	v17, v3, v17
	WORD $0x4ea49451 // mla.4s	v17, v2, v4
	WORD $0x4ea49c64 // mul.4s	v4, v3, v4
	WORD $0x4eb49444 // mla.4s	v4, v2, v20
	WORD $0x4f343622 // srsra.4s	v2, v17, #0xc
	WORD $0x4f343483 // srsra.4s	v3, v4, #0xc
	WORD $0x4ea58604 // add.4s	v4, v16, v5
	WORD $0x4ea06484 // smax.4s	v4, v4, v0
	WORD $0x4ea16c84 // smin.4s	v4, v4, v1
	WORD $0x4ea68651 // add.4s	v17, v18, v6
	WORD $0x4ea06631 // smax.4s	v17, v17, v0
	WORD $0x4ea16e31 // smin.4s	v17, v17, v1
	WORD $0x6ea0ba31 // neg.4s	v17, v17
	WORD $0x6eb084a5 // sub.4s	v5, v5, v16
	WORD $0x4ea064a5 // smax.4s	v5, v5, v0
	WORD $0x4ea16ca5 // smin.4s	v5, v5, v1
	WORD $0x6eb284c6 // sub.4s	v6, v6, v18
	WORD $0x4ea064c6 // smax.4s	v6, v6, v0
	WORD $0x4ea16cc6 // smin.4s	v6, v6, v1
	WORD $0x4eb38450 // add.4s	v16, v2, v19
	WORD $0x4ea06610 // smax.4s	v16, v16, v0
	WORD $0x4ea16e10 // smin.4s	v16, v16, v1
	WORD $0x6ea0ba10 // neg.4s	v16, v16
	WORD $0x4ea78472 // add.4s	v18, v3, v7
	WORD $0x4ea06652 // smax.4s	v18, v18, v0
	WORD $0x4ea16e52 // smin.4s	v18, v18, v1
	WORD $0x6ea28662 // sub.4s	v2, v19, v2
	WORD $0x4ea06442 // smax.4s	v2, v2, v0
	WORD $0x4ea16c42 // smin.4s	v2, v2, v1
	WORD $0x6ea384e3 // sub.4s	v3, v7, v3
	WORD $0x4ea06460 // smax.4s	v0, v3, v0
	WORD $0x4ea16c00 // smin.4s	v0, v0, v1
	WORD $0x4ea584c1 // add.4s	v1, v6, v5
	WORD $0x4f0506a3 // movi.4s	v3, #0xb5
	WORD $0x4ea39c21 // mul.4s	v1, v1, v3
	WORD $0x4f382421 // srshr.4s	v1, v1, #0x8
	WORD $0x6ea0b821 // neg.4s	v1, v1
	WORD $0x6ea684a5 // sub.4s	v5, v5, v6
	WORD $0x4ea39ca5 // mul.4s	v5, v5, v3
	WORD $0x4f3824a5 // srshr.4s	v5, v5, #0x8
	WORD $0x4ea28406 // add.4s	v6, v0, v2
	WORD $0x4ea39cc6 // mul.4s	v6, v6, v3
	WORD $0x4f3824c6 // srshr.4s	v6, v6, #0x8
	WORD $0x6ea08440 // sub.4s	v0, v2, v0
	WORD $0x4ea39c00 // mul.4s	v0, v0, v3
	WORD $0x4f382400 // srshr.4s	v0, v0, #0x8
	WORD $0x6ea0b800 // neg.4s	v0, v0
	WORD $0x4e902882 // trn1.4s	v2, v4, v16
	WORD $0x4e906883 // trn2.4s	v3, v4, v16
	WORD $0x4e8128c4 // trn1.4s	v4, v6, v1
	WORD $0x4e8168c1 // trn2.4s	v1, v6, v1
	WORD $0x4ec47846 // zip2.2d	v6, v2, v4
	WORD $0x6e180482 // mov.d	v2[1], v4[0]
	WORD $0x4ec17864 // zip2.2d	v4, v3, v1
	WORD $0x6e180423 // mov.d	v3[1], v1[0]
	WORD $0x3d800002 // str	q2, [x0]
	WORD $0x3d800023 // str	q3, [x1]
	WORD $0x3d800046 // str	q6, [x2]
	WORD $0x3d800064 // str	q4, [x3]
	WORD $0x4e8028a1 // trn1.4s	v1, v5, v0
	WORD $0x4e8068a0 // trn2.4s	v0, v5, v0
	WORD $0x4e912a42 // trn1.4s	v2, v18, v17
	WORD $0x4e916a43 // trn2.4s	v3, v18, v17
	WORD $0x4ec27824 // zip2.2d	v4, v1, v2
	WORD $0x6e180441 // mov.d	v1[1], v2[0]
	WORD $0x4ec37802 // zip2.2d	v2, v0, v3
	WORD $0x6e180460 // mov.d	v0[1], v3[0]
	WORD $0x3d800401 // str	q1, [x0, #0x10]
	WORD $0x3d800420 // str	q0, [x1, #0x10]
	WORD $0x3d800444 // str	q4, [x2, #0x10]
	WORD $0x3d800462 // str	q2, [x3, #0x10]
	WORD $0xd65f03c0 // ret

