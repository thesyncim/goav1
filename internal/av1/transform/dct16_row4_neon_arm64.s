// NEON-accelerated inverse DCT16 four-row kernel.
//
// Four stride-1 rows are transposed so coefficient i occupies the four int32
// lanes of one vector. The generated butterfly uses the same coefficient
// rewriting and stage clamps as the checked DCT32 four-lane kernel; generator
// range proofs and differential tests cover every AV1 8/10/12-bit clamp.
// The generated leaf saves its callee-saved vector registers in a 64-byte
// manual frame, comfortably inside the NOSPLIT stack budget.
//
// Generated from the pure-Go inverseDCT16 butterfly by tools/itxgen.
//
// SPDX-License-Identifier: BSD-2-Clause
//
//go:build arm64 && !purego

#include "textflag.h"

// func inverseDCT16Row4NEON(r0, r1, r2, r3 *int32, min, max int64)
TEXT ·inverseDCT16Row4NEON(SB), NOSPLIT, $0-48
	MOVD r0+0(FP), R0
	MOVD r1+8(FP), R1
	MOVD r2+16(FP), R2
	MOVD r3+24(FP), R3
	MOVD min+32(FP), R4
	MOVD max+40(FP), R5
	WORD $0x6dbc3bef // stp	d15, d14, [sp, #-0x40]!
	WORD $0x6d0133ed // stp	d13, d12, [sp, #0x10]
	WORD $0x6d022beb // stp	d11, d10, [sp, #0x20]
	WORD $0x6d0323e9 // stp	d9, d8, [sp, #0x30]
	WORD $0xad400400 // ldp	q0, q1, [x0]
	WORD $0xad400c22 // ldp	q2, q3, [x1]
	WORD $0x4e822811 // trn1.4s	v17, v0, v2
	WORD $0x4e826804 // trn2.4s	v4, v0, v2
	WORD $0xad401840 // ldp	q0, q6, [x2]
	WORD $0xad401c65 // ldp	q5, q7, [x3]
	WORD $0x4e852810 // trn1.4s	v16, v0, v5
	WORD $0x4ed07a22 // zip2.2d	v2, v17, v16
	WORD $0x6e180611 // mov.d	v17[1], v16[0]
	WORD $0x4e856800 // trn2.4s	v0, v0, v5
	WORD $0x4ec07885 // zip2.2d	v5, v4, v0
	WORD $0x6e180404 // mov.d	v4[1], v0[0]
	WORD $0x4e832833 // trn1.4s	v19, v1, v3
	WORD $0x4e836820 // trn2.4s	v0, v1, v3
	WORD $0x4e8728c1 // trn1.4s	v1, v6, v7
	WORD $0x4ec17a63 // zip2.2d	v3, v19, v1
	WORD $0x6e180433 // mov.d	v19[1], v1[0]
	WORD $0x4e8768c1 // trn2.4s	v1, v6, v7
	WORD $0x4ec17810 // zip2.2d	v16, v0, v1
	WORD $0x4ea01c07 // mov.16b	v7, v0
	WORD $0x6e180427 // mov.d	v7[1], v1[0]
	WORD $0xad410400 // ldp	q0, q1, [x0, #0x20]
	WORD $0xad414826 // ldp	q6, q18, [x1, #0x20]
	WORD $0x4e862814 // trn1.4s	v20, v0, v6
	WORD $0x4e866800 // trn2.4s	v0, v0, v6
	WORD $0xad415c46 // ldp	q6, q23, [x2, #0x20]
	WORD $0xad416075 // ldp	q21, q24, [x3, #0x20]
	WORD $0x4e9528d6 // trn1.4s	v22, v6, v21
	WORD $0x4e9568c6 // trn2.4s	v6, v6, v21
	WORD $0x4ed67a95 // zip2.2d	v21, v20, v22
	WORD $0x6e1806d4 // mov.d	v20[1], v22[0]
	WORD $0x4ea01c16 // mov.16b	v22, v0
	WORD $0x6e1804d6 // mov.d	v22[1], v6[0]
	WORD $0x4ec6781c // zip2.2d	v28, v0, v6
	WORD $0x4e922820 // trn1.4s	v0, v1, v18
	WORD $0x4e926821 // trn2.4s	v1, v1, v18
	WORD $0x4e982ae6 // trn1.4s	v6, v23, v24
	WORD $0x4e986af2 // trn2.4s	v18, v23, v24
	WORD $0x4ec6781a // zip2.2d	v26, v0, v6
	WORD $0x4ea01c18 // mov.16b	v24, v0
	WORD $0x6e1804d8 // mov.d	v24[1], v6[0]
	WORD $0x4ea11c37 // mov.16b	v23, v1
	WORD $0x6e180657 // mov.d	v23[1], v18[0]
	WORD $0x4ed2783d // zip2.2d	v29, v1, v18
	WORD $0x4e040c80 // dup.4s	v0, w4
	WORD $0x4e040ca1 // dup.4s	v1, w5
	WORD $0x4eb18692 // add.4s	v18, v20, v17
	WORD $0x4f0506a6 // movi.4s	v6, #0xb5
	WORD $0x4ea69e5b // mul.4s	v27, v18, v6
	WORD $0x4f38277e // srshr.4s	v30, v27, #0x8
	WORD $0x6eb48631 // sub.4s	v17, v17, v20
	WORD $0x4ea69e3f // mul.4s	v31, v17, v6
	WORD $0x4f3827e8 // srshr.4s	v8, v31, #0x8
	WORD $0x5280c3e8 // mov	w8, #0x61f              ; =1567
	WORD $0x4e040d11 // dup.4s	v17, w8
	WORD $0x52802708 // mov	w8, #0x138              ; =312
	WORD $0x4e040d12 // dup.4s	v18, w8
	WORD $0x4eb19e74 // mul.4s	v20, v19, v17
	WORD $0x4eb29714 // mla.4s	v20, v24, v18
	WORD $0x4f342694 // srshr.4s	v20, v20, #0xc
	WORD $0x128026e8 // mov	w8, #-0x138             ; =-312
	WORD $0x4e040d19 // dup.4s	v25, w8
	WORD $0x6eb88689 // sub.4s	v9, v20, v24
	WORD $0x4eb99e74 // mul.4s	v20, v19, v25
	WORD $0x4eb19714 // mla.4s	v20, v24, v17
	WORD $0x4f343693 // srsra.4s	v19, v20, #0xc
	WORD $0x6eb387de // sub.4s	v30, v30, v19
	WORD $0x4f383773 // srsra.4s	v19, v27, #0x8
	WORD $0x4ea06673 // smax.4s	v19, v19, v0
	WORD $0x4ea16e74 // smin.4s	v20, v19, v1
	WORD $0x6ea98513 // sub.4s	v19, v8, v9
	WORD $0x4f3837e9 // srsra.4s	v9, v31, #0x8
	WORD $0x4ea06538 // smax.4s	v24, v9, v0
	WORD $0x4ea16f1b // smin.4s	v27, v24, v1
	WORD $0x4ea06673 // smax.4s	v19, v19, v0
	WORD $0x4ea16e78 // smin.4s	v24, v19, v1
	WORD $0x4ea067d3 // smax.4s	v19, v30, v0
	WORD $0x528063e8 // mov	w8, #0x31f              ; =799
	WORD $0x4e040d1e // dup.4s	v30, w8
	WORD $0x4ea16e73 // smin.4s	v19, v19, v1
	WORD $0x4ebe9c5f // mul.4s	v31, v2, v30
	WORD $0x4f0205e8 // movi.4s	v8, #0x4f
	WORD $0x4ea8975f // mla.4s	v31, v26, v8
	WORD $0x4f3427ff // srshr.4s	v31, v31, #0xc
	WORD $0x6eba87ff // sub.4s	v31, v31, v26
	WORD $0x5280d4e8 // mov	w8, #0x6a7              ; =1703
	WORD $0x4e040d08 // dup.4s	v8, w8
	WORD $0x12808e28 // mov	w8, #-0x472             ; =-1138
	WORD $0x4e040d09 // dup.4s	v9, w8
	WORD $0x4ea99c69 // mul.4s	v9, v3, v9
	WORD $0x4ea896a9 // mla.4s	v9, v21, v8
	WORD $0x4f35252a // srshr.4s	v10, v9, #0xb
	WORD $0x52808e48 // mov	w8, #0x472              ; =1138
	WORD $0x4e040d0b // dup.4s	v11, w8
	WORD $0x4ea89c63 // mul.4s	v3, v3, v8
	WORD $0x4eab96a3 // mla.4s	v3, v21, v11
	WORD $0x4f352475 // srshr.4s	v21, v3, #0xb
	WORD $0x6f0205c8 // mvni.4s	v8, #0x4e
	WORD $0x4ea89c48 // mul.4s	v8, v2, v8
	WORD $0x4ebe9748 // mla.4s	v8, v26, v30
	WORD $0x4f343502 // srsra.4s	v2, v8, #0xc
	WORD $0x6eaa87fa // sub.4s	v26, v31, v10
	WORD $0x4f35353f // srsra.4s	v31, v9, #0xb
	WORD $0x4ea067fe // smax.4s	v30, v31, v0
	WORD $0x4ea16fde // smin.4s	v30, v30, v1
	WORD $0x4ea0675a // smax.4s	v26, v26, v0
	WORD $0x4ea16f5a // smin.4s	v26, v26, v1
	WORD $0x6eb58455 // sub.4s	v21, v2, v21
	WORD $0x4f353462 // srsra.4s	v2, v3, #0xb
	WORD $0x4ea06442 // smax.4s	v2, v2, v0
	WORD $0x4ea16c5f // smin.4s	v31, v2, v1
	WORD $0x4ea066a2 // smax.4s	v2, v21, v0
	WORD $0x4ea16c42 // smin.4s	v2, v2, v1
	WORD $0x6eba8443 // sub.4s	v3, v2, v26
	WORD $0x4ea69c75 // mul.4s	v21, v3, v6
	WORD $0x4f3826a8 // srshr.4s	v8, v21, #0x8
	WORD $0x4eba8442 // add.4s	v2, v2, v26
	WORD $0x4ea69c43 // mul.4s	v3, v2, v6
	WORD $0x4f38247a // srshr.4s	v26, v3, #0x8
	WORD $0x4eb487e2 // add.4s	v2, v31, v20
	WORD $0x4ea06442 // smax.4s	v2, v2, v0
	WORD $0x4ea16c42 // smin.4s	v2, v2, v1
	WORD $0x6eba8769 // sub.4s	v9, v27, v26
	WORD $0x4f38347b // srsra.4s	v27, v3, #0x8
	WORD $0x4ea06763 // smax.4s	v3, v27, v0
	WORD $0x4ea16c63 // smin.4s	v3, v3, v1
	WORD $0x6ea88708 // sub.4s	v8, v24, v8
	WORD $0x4f3836b8 // srsra.4s	v24, v21, #0x8
	WORD $0x4ea06715 // smax.4s	v21, v24, v0
	WORD $0x4ea16ebb // smin.4s	v27, v21, v1
	WORD $0x4eb387d5 // add.4s	v21, v30, v19
	WORD $0x4ea066b5 // smax.4s	v21, v21, v0
	WORD $0x4ea16eba // smin.4s	v26, v21, v1
	WORD $0x6ebe8673 // sub.4s	v19, v19, v30
	WORD $0x4ea06673 // smax.4s	v19, v19, v0
	WORD $0x4ea16e78 // smin.4s	v24, v19, v1
	WORD $0x4ea06513 // smax.4s	v19, v8, v0
	WORD $0x4ea16e75 // smin.4s	v21, v19, v1
	WORD $0x4ea06533 // smax.4s	v19, v9, v0
	WORD $0x4ea16e73 // smin.4s	v19, v19, v1
	WORD $0x6ebf8694 // sub.4s	v20, v20, v31
	WORD $0x4ea06694 // smax.4s	v20, v20, v0
	WORD $0x4ea16e94 // smin.4s	v20, v20, v1
	WORD $0x52803228 // mov	w8, #0x191              ; =401
	WORD $0x4e040d1e // dup.4s	v30, w8
	WORD $0x4ebe9c9f // mul.4s	v31, v4, v30
	WORD $0x4f000688 // movi.4s	v8, #0x14
	WORD $0x4ea897bf // mla.4s	v31, v29, v8
	WORD $0x4f3427ff // srshr.4s	v31, v31, #0xc
	WORD $0x5280c5e8 // mov	w8, #0x62f              ; =1583
	WORD $0x4e040d08 // dup.4s	v8, w8
	WORD $0x6ebd87ff // sub.4s	v31, v31, v29
	WORD $0x1280a248 // mov	w8, #-0x513             ; =-1299
	WORD $0x4e040d09 // dup.4s	v9, w8
	WORD $0x4ea99e09 // mul.4s	v9, v16, v9
	WORD $0x4ea896c9 // mla.4s	v9, v22, v8
	WORD $0x4f35252a // srshr.4s	v10, v9, #0xb
	WORD $0x5280f168 // mov	w8, #0x78b              ; =1931
	WORD $0x4e040d0b // dup.4s	v11, w8
	WORD $0x4eab9cec // mul.4s	v12, v7, v11
	WORD $0x52803c88 // mov	w8, #0x1e4              ; =484
	WORD $0x4e040d0d // dup.4s	v13, w8
	WORD $0x4ead978c // mla.4s	v12, v28, v13
	WORD $0x4f34258c // srshr.4s	v12, v12, #0xc
	WORD $0x6ebc858c // sub.4s	v12, v12, v28
	WORD $0x6f0505ed // mvni.4s	v13, #0xaf
	WORD $0x12809488 // mov	w8, #-0x4a5             ; =-1189
	WORD $0x4e040d0e // dup.4s	v14, w8
	WORD $0x4eae9cae // mul.4s	v14, v5, v14
	WORD $0x4ead96ee // mla.4s	v14, v23, v13
	WORD $0x528094a8 // mov	w8, #0x4a5              ; =1189
	WORD $0x4e040d0f // dup.4s	v15, w8
	WORD $0x4ead9cad // mul.4s	v13, v5, v13
	WORD $0x4eaf96ed // mla.4s	v13, v23, v15
	WORD $0x4f3435d7 // srsra.4s	v23, v14, #0xc
	WORD $0x4f3435a5 // srsra.4s	v5, v13, #0xc
	WORD $0x12803c68 // mov	w8, #-0x1e4             ; =-484
	WORD $0x4e040d0d // dup.4s	v13, w8
	WORD $0x4ead9ced // mul.4s	v13, v7, v13
	WORD $0x4eab978d // mla.4s	v13, v28, v11
	WORD $0x4f3435a7 // srsra.4s	v7, v13, #0xc
	WORD $0x5280a268 // mov	w8, #0x513              ; =1299
	WORD $0x4e040d1c // dup.4s	v28, w8
	WORD $0x4ea89e10 // mul.4s	v16, v16, v8
	WORD $0x4ebc96d0 // mla.4s	v16, v22, v28
	WORD $0x4f352616 // srshr.4s	v22, v16, #0xb
	WORD $0x6f00067c // mvni.4s	v28, #0x13
	WORD $0x4ebc9c9c // mul.4s	v28, v4, v28
	WORD $0x4ebe97bc // mla.4s	v28, v29, v30
	WORD $0x4f343784 // srsra.4s	v4, v28, #0xc
	WORD $0x6eaa87fc // sub.4s	v28, v31, v10
	WORD $0x4f35353f // srsra.4s	v31, v9, #0xb
	WORD $0x4ea067fd // smax.4s	v29, v31, v0
	WORD $0x4ea16fbd // smin.4s	v29, v29, v1
	WORD $0x4ea0679c // smax.4s	v28, v28, v0
	WORD $0x4ea16f9c // smin.4s	v28, v28, v1
	WORD $0x6eac86fe // sub.4s	v30, v23, v12
	WORD $0x4ea067de // smax.4s	v30, v30, v0
	WORD $0x4ea16fde // smin.4s	v30, v30, v1
	WORD $0x4eac86f7 // add.4s	v23, v23, v12
	WORD $0x4ea066f7 // smax.4s	v23, v23, v0
	WORD $0x4ea16ef7 // smin.4s	v23, v23, v1
	WORD $0x4ea584ff // add.4s	v31, v7, v5
	WORD $0x4ea067ff // smax.4s	v31, v31, v0
	WORD $0x4ea16fff // smin.4s	v31, v31, v1
	WORD $0x6ea784a5 // sub.4s	v5, v5, v7
	WORD $0x4ea064a5 // smax.4s	v5, v5, v0
	WORD $0x4ea16ca5 // smin.4s	v5, v5, v1
	WORD $0x6eb68487 // sub.4s	v7, v4, v22
	WORD $0x4ea064e7 // smax.4s	v7, v7, v0
	WORD $0x4ea16cf6 // smin.4s	v22, v7, v1
	WORD $0x4f353604 // srsra.4s	v4, v16, #0xb
	WORD $0x4ea06484 // smax.4s	v4, v4, v0
	WORD $0x4ea16c84 // smin.4s	v4, v4, v1
	WORD $0x4eb29f87 // mul.4s	v7, v28, v18
	WORD $0x4eb196c7 // mla.4s	v7, v22, v17
	WORD $0x4f3424e7 // srshr.4s	v7, v7, #0xc
	WORD $0x6ebc84e8 // sub.4s	v8, v7, v28
	WORD $0x4eb19f87 // mul.4s	v7, v28, v17
	WORD $0x4eb996c7 // mla.4s	v7, v22, v25
	WORD $0x4f3434f6 // srsra.4s	v22, v7, #0xc
	WORD $0x1280c3c8 // mov	w8, #-0x61f             ; =-1567
	WORD $0x4e040d07 // dup.4s	v7, w8
	WORD $0x4ea79fc7 // mul.4s	v7, v30, v7
	WORD $0x4eb294a7 // mla.4s	v7, v5, v18
	WORD $0x4f3424e7 // srshr.4s	v7, v7, #0xc
	WORD $0x6ea584f9 // sub.4s	v25, v7, v5
	WORD $0x4eb29fc7 // mul.4s	v7, v30, v18
	WORD $0x4eb194a7 // mla.4s	v7, v5, v17
	WORD $0x4f3424e5 // srshr.4s	v5, v7, #0xc
	WORD $0x6ebe84a5 // sub.4s	v5, v5, v30
	WORD $0x4ebd86e7 // add.4s	v7, v23, v29
	WORD $0x4ea064e7 // smax.4s	v7, v7, v0
	WORD $0x4ea16ce7 // smin.4s	v7, v7, v1
	WORD $0x4ea88730 // add.4s	v16, v25, v8
	WORD $0x4ea06610 // smax.4s	v16, v16, v0
	WORD $0x4ea16e10 // smin.4s	v16, v16, v1
	WORD $0x6eb98511 // sub.4s	v17, v8, v25
	WORD $0x4ea06631 // smax.4s	v17, v17, v0
	WORD $0x4ea16e31 // smin.4s	v17, v17, v1
	WORD $0x6eb787b2 // sub.4s	v18, v29, v23
	WORD $0x4ea06652 // smax.4s	v18, v18, v0
	WORD $0x4ea16e52 // smin.4s	v18, v18, v1
	WORD $0x6ebf8497 // sub.4s	v23, v4, v31
	WORD $0x4ea066f7 // smax.4s	v23, v23, v0
	WORD $0x4ea16efc // smin.4s	v28, v23, v1
	WORD $0x6ea586d7 // sub.4s	v23, v22, v5
	WORD $0x4ea066f7 // smax.4s	v23, v23, v0
	WORD $0x4ea16efd // smin.4s	v29, v23, v1
	WORD $0x4eb684a5 // add.4s	v5, v5, v22
	WORD $0x4ea064a5 // smax.4s	v5, v5, v0
	WORD $0x4ea16ca5 // smin.4s	v5, v5, v1
	WORD $0x4ebf8484 // add.4s	v4, v4, v31
	WORD $0x4ea06484 // smax.4s	v4, v4, v0
	WORD $0x4ea16c84 // smin.4s	v4, v4, v1
	WORD $0x6eb187b6 // sub.4s	v22, v29, v17
	WORD $0x4ea69ed7 // mul.4s	v23, v22, v6
	WORD $0x4f3826f9 // srshr.4s	v25, v23, #0x8
	WORD $0x4eb187b1 // add.4s	v17, v29, v17
	WORD $0x4ea69e36 // mul.4s	v22, v17, v6
	WORD $0x4f3826dd // srshr.4s	v29, v22, #0x8
	WORD $0x6eb28791 // sub.4s	v17, v28, v18
	WORD $0x4ea69e3e // mul.4s	v30, v17, v6
	WORD $0x4f3827df // srshr.4s	v31, v30, #0x8
	WORD $0x4eb28791 // add.4s	v17, v28, v18
	WORD $0x4ea69e3c // mul.4s	v28, v17, v6
	WORD $0x4f382788 // srshr.4s	v8, v28, #0x8
	WORD $0x4ea28486 // add.4s	v6, v4, v2
	WORD $0x4ea064c6 // smax.4s	v6, v6, v0
	WORD $0x4ea16cd1 // smin.4s	v17, v6, v1
	WORD $0x4ea384a6 // add.4s	v6, v5, v3
	WORD $0x4ea064c6 // smax.4s	v6, v6, v0
	WORD $0x4ea16cd2 // smin.4s	v18, v6, v1
	WORD $0x6ebd8766 // sub.4s	v6, v27, v29
	WORD $0x4f3836db // srsra.4s	v27, v22, #0x8
	WORD $0x4ea06776 // smax.4s	v22, v27, v0
	WORD $0x4ea16edb // smin.4s	v27, v22, v1
	WORD $0x6ea88756 // sub.4s	v22, v26, v8
	WORD $0x4f38379a // srsra.4s	v26, v28, #0x8
	WORD $0x4ea0675a // smax.4s	v26, v26, v0
	WORD $0x4ea16f5a // smin.4s	v26, v26, v1
	WORD $0x6ebf871c // sub.4s	v28, v24, v31
	WORD $0x4f3837d8 // srsra.4s	v24, v30, #0x8
	WORD $0x4ea06718 // smax.4s	v24, v24, v0
	WORD $0x4ea16f18 // smin.4s	v24, v24, v1
	WORD $0x6eb986b9 // sub.4s	v25, v21, v25
	WORD $0x4f3836f5 // srsra.4s	v21, v23, #0x8
	WORD $0x4ea066b5 // smax.4s	v21, v21, v0
	WORD $0x4ea16eb5 // smin.4s	v21, v21, v1
	WORD $0x4eb38617 // add.4s	v23, v16, v19
	WORD $0x4ea066f7 // smax.4s	v23, v23, v0
	WORD $0x4ea16ef7 // smin.4s	v23, v23, v1
	WORD $0x4eb484fd // add.4s	v29, v7, v20
	WORD $0x4ea067bd // smax.4s	v29, v29, v0
	WORD $0x4ea16fbd // smin.4s	v29, v29, v1
	WORD $0x6ea78687 // sub.4s	v7, v20, v7
	WORD $0x4ea064e7 // smax.4s	v7, v7, v0
	WORD $0x4ea16ce7 // smin.4s	v7, v7, v1
	WORD $0x6eb08670 // sub.4s	v16, v19, v16
	WORD $0x4ea06610 // smax.4s	v16, v16, v0
	WORD $0x4ea16e10 // smin.4s	v16, v16, v1
	WORD $0x4ea06733 // smax.4s	v19, v25, v0
	WORD $0x4ea16e73 // smin.4s	v19, v19, v1
	WORD $0x4ea06794 // smax.4s	v20, v28, v0
	WORD $0x4ea16e94 // smin.4s	v20, v20, v1
	WORD $0x4e922a39 // trn1.4s	v25, v17, v18
	WORD $0x4e926a31 // trn2.4s	v17, v17, v18
	WORD $0x4e9a2b72 // trn1.4s	v18, v27, v26
	WORD $0x4e9a6b7a // trn2.4s	v26, v27, v26
	WORD $0x4ed27b3b // zip2.2d	v27, v25, v18
	WORD $0x6e180659 // mov.d	v25[1], v18[0]
	WORD $0x4eda7a32 // zip2.2d	v18, v17, v26
	WORD $0x6e180751 // mov.d	v17[1], v26[0]
	WORD $0x3d800019 // str	q25, [x0]
	WORD $0x3d800031 // str	q17, [x1]
	WORD $0x3d80005b // str	q27, [x2]
	WORD $0x3d800072 // str	q18, [x3]
	WORD $0x4e952b11 // trn1.4s	v17, v24, v21
	WORD $0x4e956b12 // trn2.4s	v18, v24, v21
	WORD $0x4e9d2af5 // trn1.4s	v21, v23, v29
	WORD $0x4e9d6af7 // trn2.4s	v23, v23, v29
	WORD $0x4ed57a38 // zip2.2d	v24, v17, v21
	WORD $0x4ed77a59 // zip2.2d	v25, v18, v23
	WORD $0x4ea066d6 // smax.4s	v22, v22, v0
	WORD $0x4ea16ed6 // smin.4s	v22, v22, v1
	WORD $0x6e1806b1 // mov.d	v17[1], v21[0]
	WORD $0x6e1806f2 // mov.d	v18[1], v23[0]
	WORD $0x3d800411 // str	q17, [x0, #0x10]
	WORD $0x3d800432 // str	q18, [x1, #0x10]
	WORD $0x3d800458 // str	q24, [x2, #0x10]
	WORD $0x3d800479 // str	q25, [x3, #0x10]
	WORD $0x4e9028f1 // trn1.4s	v17, v7, v16
	WORD $0x4e9068e7 // trn2.4s	v7, v7, v16
	WORD $0x4e942a70 // trn1.4s	v16, v19, v20
	WORD $0x4e946a72 // trn2.4s	v18, v19, v20
	WORD $0x4ed07a33 // zip2.2d	v19, v17, v16
	WORD $0x6e180611 // mov.d	v17[1], v16[0]
	WORD $0x4ed278f0 // zip2.2d	v16, v7, v18
	WORD $0x4ea064c6 // smax.4s	v6, v6, v0
	WORD $0x3d800811 // str	q17, [x0, #0x20]
	WORD $0x4ea16cc6 // smin.4s	v6, v6, v1
	WORD $0x6ea58463 // sub.4s	v3, v3, v5
	WORD $0x6e180647 // mov.d	v7[1], v18[0]
	WORD $0x3d800827 // str	q7, [x1, #0x20]
	WORD $0x4ea06463 // smax.4s	v3, v3, v0
	WORD $0x4ea16c63 // smin.4s	v3, v3, v1
	WORD $0x3d800853 // str	q19, [x2, #0x20]
	WORD $0x6ea48442 // sub.4s	v2, v2, v4
	WORD $0x3d800870 // str	q16, [x3, #0x20]
	WORD $0x4ea06440 // smax.4s	v0, v2, v0
	WORD $0x4ea16c00 // smin.4s	v0, v0, v1
	WORD $0x4e862ac1 // trn1.4s	v1, v22, v6
	WORD $0x4e802862 // trn1.4s	v2, v3, v0
	WORD $0x4ec27824 // zip2.2d	v4, v1, v2
	WORD $0x4e866ac5 // trn2.4s	v5, v22, v6
	WORD $0x4e806860 // trn2.4s	v0, v3, v0
	WORD $0x4ec078a3 // zip2.2d	v3, v5, v0
	WORD $0x6e180441 // mov.d	v1[1], v2[0]
	WORD $0x3d800c01 // str	q1, [x0, #0x30]
	WORD $0x6e180405 // mov.d	v5[1], v0[0]
	WORD $0x3d800c25 // str	q5, [x1, #0x30]
	WORD $0x3d800c44 // str	q4, [x2, #0x30]
	WORD $0x3d800c63 // str	q3, [x3, #0x30]
	WORD $0x6d4323e9 // ldp	d9, d8, [sp, #0x30]
	WORD $0x6d422beb // ldp	d11, d10, [sp, #0x20]
	WORD $0x6d4133ed // ldp	d13, d12, [sp, #0x10]
	WORD $0x6cc43bef // ldp	d15, d14, [sp], #0x40
	WORD $0xd65f03c0 // ret

