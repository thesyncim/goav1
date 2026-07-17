// Generated NEON four-column inverse kernels for DCT8, DCT16, and ADST8.
//
// Four adjacent columns ride the four int32 lanes of each vector. Inputs are
// pre-clamped to the transform stage range; the generated butterflies use the
// same fixed-point products, rounded shifts, and stage clamps as the scalar
// references. tools/itxgen validates the compiled kernels differentially and
// transcribes their straight-line bodies as WORD opcodes.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

#include "textflag.h"

// func inverseDCT8Col4NEON(base *int32, rowStrideBytes, min, max int64)
TEXT ·inverseDCT8Col4NEON(SB), NOSPLIT, $0-32
	MOVD base+0(FP), R0
	MOVD rowStrideBytes+8(FP), R1
	MOVD min+16(FP), R2
	MOVD max+24(FP), R3
	WORD $0x3dc00002 // ldr	q2, [x0]
	WORD $0x3ce16803 // ldr	q3, [x0, x1]
	WORD $0xd37ff828 // lsl	x8, x1, #1
	WORD $0x3ce86805 // ldr	q5, [x0, x8]
	WORD $0x8b010109 // add	x9, x8, x1
	WORD $0x3ce96810 // ldr	q16, [x0, x9]
	WORD $0xd37ef42a // lsl	x10, x1, #2
	WORD $0x3cea6806 // ldr	q6, [x0, x10]
	WORD $0x8b01014b // add	x11, x10, x1
	WORD $0x3ceb6811 // ldr	q17, [x0, x11]
	WORD $0x528000cc // mov	w12, #0x6               ; =6
	WORD $0x9b0c7c2c // mul	x12, x1, x12
	WORD $0x3cec6807 // ldr	q7, [x0, x12]
	WORD $0xd37df02d // lsl	x13, x1, #3
	WORD $0xcb01000e // sub	x14, x0, x1
	WORD $0x3ced69d2 // ldr	q18, [x14, x13]
	WORD $0x4e040c40 // dup.4s	v0, w2
	WORD $0x4e040c61 // dup.4s	v1, w3
	WORD $0x4ea284d3 // add.4s	v19, v6, v2
	WORD $0x4f0506a4 // movi.4s	v4, #0xb5
	WORD $0x4ea49e73 // mul.4s	v19, v19, v4
	WORD $0x4f382674 // srshr.4s	v20, v19, #0x8
	WORD $0x6ea68442 // sub.4s	v2, v2, v6
	WORD $0x4ea49c46 // mul.4s	v6, v2, v4
	WORD $0x4f3824d5 // srshr.4s	v21, v6, #0x8
	WORD $0x5280c3ef // mov	w15, #0x61f             ; =1567
	WORD $0x4e040de2 // dup.4s	v2, w15
	WORD $0x4ea29cb6 // mul.4s	v22, v5, v2
	WORD $0x5280270f // mov	w15, #0x138             ; =312
	WORD $0x4e040df7 // dup.4s	v23, w15
	WORD $0x4eb794f6 // mla.4s	v22, v7, v23
	WORD $0x4f3426d6 // srshr.4s	v22, v22, #0xc
	WORD $0x6ea786d6 // sub.4s	v22, v22, v7
	WORD $0x128026ef // mov	w15, #-0x138            ; =-312
	WORD $0x4e040df7 // dup.4s	v23, w15
	WORD $0x4eb79cb7 // mul.4s	v23, v5, v23
	WORD $0x4ea294f7 // mla.4s	v23, v7, v2
	WORD $0x4f3436e5 // srsra.4s	v5, v23, #0xc
	WORD $0x6ea58694 // sub.4s	v20, v20, v5
	WORD $0x4f383665 // srsra.4s	v5, v19, #0x8
	WORD $0x4ea064a2 // smax.4s	v2, v5, v0
	WORD $0x4ea16c42 // smin.4s	v2, v2, v1
	WORD $0x6eb686a5 // sub.4s	v5, v21, v22
	WORD $0x4f3834d6 // srsra.4s	v22, v6, #0x8
	WORD $0x4ea066c6 // smax.4s	v6, v22, v0
	WORD $0x4ea16cc7 // smin.4s	v7, v6, v1
	WORD $0x4ea064a5 // smax.4s	v5, v5, v0
	WORD $0x4ea16ca6 // smin.4s	v6, v5, v1
	WORD $0x4ea06685 // smax.4s	v5, v20, v0
	WORD $0x4ea16ca5 // smin.4s	v5, v5, v1
	WORD $0x528063ef // mov	w15, #0x31f             ; =799
	WORD $0x4e040df3 // dup.4s	v19, w15
	WORD $0x4eb39c74 // mul.4s	v20, v3, v19
	WORD $0x4f0205f5 // movi.4s	v21, #0x4f
	WORD $0x4eb59654 // mla.4s	v20, v18, v21
	WORD $0x4f342694 // srshr.4s	v20, v20, #0xc
	WORD $0x5280d4ef // mov	w15, #0x6a7             ; =1703
	WORD $0x4e040df5 // dup.4s	v21, w15
	WORD $0x12808e2f // mov	w15, #-0x472            ; =-1138
	WORD $0x4e040df6 // dup.4s	v22, w15
	WORD $0x6eb28694 // sub.4s	v20, v20, v18
	WORD $0x4eb69e16 // mul.4s	v22, v16, v22
	WORD $0x4eb59636 // mla.4s	v22, v17, v21
	WORD $0x52808e4f // mov	w15, #0x472             ; =1138
	WORD $0x4e040df7 // dup.4s	v23, w15
	WORD $0x4f3526d8 // srshr.4s	v24, v22, #0xb
	WORD $0x4eb59e10 // mul.4s	v16, v16, v21
	WORD $0x4eb79630 // mla.4s	v16, v17, v23
	WORD $0x4f352611 // srshr.4s	v17, v16, #0xb
	WORD $0x6f0205d5 // mvni.4s	v21, #0x4e
	WORD $0x4eb59c75 // mul.4s	v21, v3, v21
	WORD $0x4eb39655 // mla.4s	v21, v18, v19
	WORD $0x4f3436a3 // srsra.4s	v3, v21, #0xc
	WORD $0x6eb88692 // sub.4s	v18, v20, v24
	WORD $0x4f3536d4 // srsra.4s	v20, v22, #0xb
	WORD $0x4ea06693 // smax.4s	v19, v20, v0
	WORD $0x4ea16e73 // smin.4s	v19, v19, v1
	WORD $0x4ea06652 // smax.4s	v18, v18, v0
	WORD $0x4ea16e52 // smin.4s	v18, v18, v1
	WORD $0x6eb18471 // sub.4s	v17, v3, v17
	WORD $0x4f353603 // srsra.4s	v3, v16, #0xb
	WORD $0x4ea06463 // smax.4s	v3, v3, v0
	WORD $0x4ea16c63 // smin.4s	v3, v3, v1
	WORD $0x4ea06630 // smax.4s	v16, v17, v0
	WORD $0x4ea16e10 // smin.4s	v16, v16, v1
	WORD $0x6eb28611 // sub.4s	v17, v16, v18
	WORD $0x4ea49e31 // mul.4s	v17, v17, v4
	WORD $0x4f382634 // srshr.4s	v20, v17, #0x8
	WORD $0x4eb28610 // add.4s	v16, v16, v18
	WORD $0x4ea49e04 // mul.4s	v4, v16, v4
	WORD $0x4f382490 // srshr.4s	v16, v4, #0x8
	WORD $0x4ea28472 // add.4s	v18, v3, v2
	WORD $0x4ea06652 // smax.4s	v18, v18, v0
	WORD $0x4ea16e52 // smin.4s	v18, v18, v1
	WORD $0x6eb084f0 // sub.4s	v16, v7, v16
	WORD $0x4f383487 // srsra.4s	v7, v4, #0x8
	WORD $0x4ea064e4 // smax.4s	v4, v7, v0
	WORD $0x4ea16c84 // smin.4s	v4, v4, v1
	WORD $0x6eb484c7 // sub.4s	v7, v6, v20
	WORD $0x4f383626 // srsra.4s	v6, v17, #0x8
	WORD $0x4ea064c6 // smax.4s	v6, v6, v0
	WORD $0x4ea16cc6 // smin.4s	v6, v6, v1
	WORD $0x4ea58671 // add.4s	v17, v19, v5
	WORD $0x4ea06631 // smax.4s	v17, v17, v0
	WORD $0x4ea16e31 // smin.4s	v17, v17, v1
	WORD $0x6eb384a5 // sub.4s	v5, v5, v19
	WORD $0x4ea064a5 // smax.4s	v5, v5, v0
	WORD $0x4ea16ca5 // smin.4s	v5, v5, v1
	WORD $0x4ea064e7 // smax.4s	v7, v7, v0
	WORD $0x4ea16ce7 // smin.4s	v7, v7, v1
	WORD $0x4ea06610 // smax.4s	v16, v16, v0
	WORD $0x4ea16e10 // smin.4s	v16, v16, v1
	WORD $0x6ea38442 // sub.4s	v2, v2, v3
	WORD $0x3d800012 // str	q18, [x0]
	WORD $0x3ca16804 // str	q4, [x0, x1]
	WORD $0x3ca86806 // str	q6, [x0, x8]
	WORD $0x3ca96811 // str	q17, [x0, x9]
	WORD $0x3caa6805 // str	q5, [x0, x10]
	WORD $0x3cab6807 // str	q7, [x0, x11]
	WORD $0x4ea06440 // smax.4s	v0, v2, v0
	WORD $0x4ea16c00 // smin.4s	v0, v0, v1
	WORD $0x3cac6810 // str	q16, [x0, x12]
	WORD $0x3cad69c0 // str	q0, [x14, x13]
	WORD $0xd65f03c0 // ret

// func inverseDCT16Col4NEON(base *int32, rowStrideBytes, min, max int64)
TEXT ·inverseDCT16Col4NEON(SB), NOSPLIT, $0-32
	MOVD base+0(FP), R0
	MOVD rowStrideBytes+8(FP), R1
	MOVD min+16(FP), R2
	MOVD max+24(FP), R3
	WORD $0x6dbc3bef // stp	d15, d14, [sp, #-0x40]!
	WORD $0x6d0133ed // stp	d13, d12, [sp, #0x10]
	WORD $0x6d022beb // stp	d11, d10, [sp, #0x20]
	WORD $0x6d0323e9 // stp	d9, d8, [sp, #0x30]
	WORD $0x4e040c40 // dup.4s	v0, w2
	WORD $0x4e040c61 // dup.4s	v1, w3
	WORD $0x3dc00003 // ldr	q3, [x0]
	WORD $0xd37df028 // lsl	x8, x1, #3
	WORD $0x3ce86804 // ldr	q4, [x0, x8]
	WORD $0x4ea38485 // add.4s	v5, v4, v3
	WORD $0x4f0506a2 // movi.4s	v2, #0xb5
	WORD $0x4ea29cb0 // mul.4s	v16, v5, v2
	WORD $0x4f382611 // srshr.4s	v17, v16, #0x8
	WORD $0x6ea48463 // sub.4s	v3, v3, v4
	WORD $0x4ea29c63 // mul.4s	v3, v3, v2
	WORD $0x4f382464 // srshr.4s	v4, v3, #0x8
	WORD $0x5280c3e9 // mov	w9, #0x61f              ; =1567
	WORD $0x4e040d25 // dup.4s	v5, w9
	WORD $0x52802709 // mov	w9, #0x138              ; =312
	WORD $0x4e040d26 // dup.4s	v6, w9
	WORD $0xd37ef42a // lsl	x10, x1, #2
	WORD $0x3cea6812 // ldr	q18, [x0, x10]
	WORD $0x4ea59e47 // mul.4s	v7, v18, v5
	WORD $0x52800189 // mov	w9, #0xc                ; =12
	WORD $0x9b097c29 // mul	x9, x1, x9
	WORD $0x3ce96813 // ldr	q19, [x0, x9]
	WORD $0x4ea69667 // mla.4s	v7, v19, v6
	WORD $0x4f3424e7 // srshr.4s	v7, v7, #0xc
	WORD $0x6eb384f4 // sub.4s	v20, v7, v19
	WORD $0x128026eb // mov	w11, #-0x138            ; =-312
	WORD $0x4e040d67 // dup.4s	v7, w11
	WORD $0x4ea79e55 // mul.4s	v21, v18, v7
	WORD $0x4ea59675 // mla.4s	v21, v19, v5
	WORD $0x4f3436b2 // srsra.4s	v18, v21, #0xc
	WORD $0x6eb28633 // sub.4s	v19, v17, v18
	WORD $0x4f383612 // srsra.4s	v18, v16, #0x8
	WORD $0x4ea06650 // smax.4s	v16, v18, v0
	WORD $0x4ea16e11 // smin.4s	v17, v16, v1
	WORD $0x6eb48490 // sub.4s	v16, v4, v20
	WORD $0xd37ff82c // lsl	x12, x1, #1
	WORD $0x4f383474 // srsra.4s	v20, v3, #0x8
	WORD $0x4ea06684 // smax.4s	v4, v20, v0
	WORD $0x3cec6803 // ldr	q3, [x0, x12]
	WORD $0x4ea16c84 // smin.4s	v4, v4, v1
	WORD $0x528000cd // mov	w13, #0x6               ; =6
	WORD $0x4ea06610 // smax.4s	v16, v16, v0
	WORD $0x528001cb // mov	w11, #0xe               ; =14
	WORD $0x4ea16e12 // smin.4s	v18, v16, v1
	WORD $0x9b0b7c2b // mul	x11, x1, x11
	WORD $0x4ea06670 // smax.4s	v16, v19, v0
	WORD $0x4ea16e10 // smin.4s	v16, v16, v1
	WORD $0x528063ee // mov	w14, #0x31f             ; =799
	WORD $0x4e040dd3 // dup.4s	v19, w14
	WORD $0x4eb39c74 // mul.4s	v20, v3, v19
	WORD $0x3ceb6815 // ldr	q21, [x0, x11]
	WORD $0x4f0205f6 // movi.4s	v22, #0x4f
	WORD $0x4eb696b4 // mla.4s	v20, v21, v22
	WORD $0x4f342694 // srshr.4s	v20, v20, #0xc
	WORD $0x5280d4ee // mov	w14, #0x6a7             ; =1703
	WORD $0x4e040dd6 // dup.4s	v22, w14
	WORD $0x12808e2e // mov	w14, #-0x472            ; =-1138
	WORD $0x4e040dd7 // dup.4s	v23, w14
	WORD $0x9b0d7c2e // mul	x14, x1, x13
	WORD $0x6eb58694 // sub.4s	v20, v20, v21
	WORD $0x3cee6818 // ldr	q24, [x0, x14]
	WORD $0x4eb79f1e // mul.4s	v30, v24, v23
	WORD $0x5280014d // mov	w13, #0xa               ; =10
	WORD $0x9b0d7c2d // mul	x13, x1, x13
	WORD $0x3ced6817 // ldr	q23, [x0, x13]
	WORD $0x4eb696fe // mla.4s	v30, v23, v22
	WORD $0x52808e4f // mov	w15, #0x472             ; =1138
	WORD $0x4e040df9 // dup.4s	v25, w15
	WORD $0x4f3527df // srshr.4s	v31, v30, #0xb
	WORD $0x4eb69f08 // mul.4s	v8, v24, v22
	WORD $0x4eb996e8 // mla.4s	v8, v23, v25
	WORD $0x4f352509 // srshr.4s	v9, v8, #0xb
	WORD $0x6f0205d6 // mvni.4s	v22, #0x4e
	WORD $0x4eb69c79 // mul.4s	v25, v3, v22
	WORD $0x3ce16816 // ldr	q22, [x0, x1]
	WORD $0x4eb396b9 // mla.4s	v25, v21, v19
	WORD $0x8b010185 // add	x5, x12, x1
	WORD $0x3ce56817 // ldr	q23, [x0, x5]
	WORD $0x8b010146 // add	x6, x10, x1
	WORD $0x3ce66818 // ldr	q24, [x0, x6]
	WORD $0xcb010104 // sub	x4, x8, x1
	WORD $0x3ce4681a // ldr	q26, [x0, x4]
	WORD $0x4f343723 // srsra.4s	v3, v25, #0xc
	WORD $0x8b010103 // add	x3, x8, x1
	WORD $0x3ce3681b // ldr	q27, [x0, x3]
	WORD $0x5280016f // mov	w15, #0xb               ; =11
	WORD $0x9b0f7c22 // mul	x2, x1, x15
	WORD $0x3ce2681d // ldr	q29, [x0, x2]
	WORD $0x528001af // mov	w15, #0xd               ; =13
	WORD $0x9b0f7c31 // mul	x17, x1, x15
	WORD $0x3cf16819 // ldr	q25, [x0, x17]
	WORD $0xd37cec2f // lsl	x15, x1, #4
	WORD $0xcb010010 // sub	x16, x0, x1
	WORD $0x3cef6a1c // ldr	q28, [x16, x15]
	WORD $0x6ebf8693 // sub.4s	v19, v20, v31
	WORD $0x4f3537d4 // srsra.4s	v20, v30, #0xb
	WORD $0x4ea06694 // smax.4s	v20, v20, v0
	WORD $0x4ea16e9e // smin.4s	v30, v20, v1
	WORD $0x4ea06673 // smax.4s	v19, v19, v0
	WORD $0x4ea16e73 // smin.4s	v19, v19, v1
	WORD $0x6ea98474 // sub.4s	v20, v3, v9
	WORD $0x4f353503 // srsra.4s	v3, v8, #0xb
	WORD $0x4ea06463 // smax.4s	v3, v3, v0
	WORD $0x4ea16c7f // smin.4s	v31, v3, v1
	WORD $0x4ea06683 // smax.4s	v3, v20, v0
	WORD $0x4ea16c63 // smin.4s	v3, v3, v1
	WORD $0x6eb38474 // sub.4s	v20, v3, v19
	WORD $0x4ea29e94 // mul.4s	v20, v20, v2
	WORD $0x4f382695 // srshr.4s	v21, v20, #0x8
	WORD $0x4eb38463 // add.4s	v3, v3, v19
	WORD $0x4ea29c73 // mul.4s	v19, v3, v2
	WORD $0x4f382668 // srshr.4s	v8, v19, #0x8
	WORD $0x4eb187e3 // add.4s	v3, v31, v17
	WORD $0x4ea06463 // smax.4s	v3, v3, v0
	WORD $0x4ea16c63 // smin.4s	v3, v3, v1
	WORD $0x6ea88488 // sub.4s	v8, v4, v8
	WORD $0x4f383664 // srsra.4s	v4, v19, #0x8
	WORD $0x4ea06484 // smax.4s	v4, v4, v0
	WORD $0x4ea16c84 // smin.4s	v4, v4, v1
	WORD $0x6eb58649 // sub.4s	v9, v18, v21
	WORD $0x4f383692 // srsra.4s	v18, v20, #0x8
	WORD $0x4ea06652 // smax.4s	v18, v18, v0
	WORD $0x4ea16e55 // smin.4s	v21, v18, v1
	WORD $0x4eb087d2 // add.4s	v18, v30, v16
	WORD $0x4ea06652 // smax.4s	v18, v18, v0
	WORD $0x4ea16e54 // smin.4s	v20, v18, v1
	WORD $0x6ebe8610 // sub.4s	v16, v16, v30
	WORD $0x4ea06610 // smax.4s	v16, v16, v0
	WORD $0x4ea16e13 // smin.4s	v19, v16, v1
	WORD $0x4ea06530 // smax.4s	v16, v9, v0
	WORD $0x4ea16e12 // smin.4s	v18, v16, v1
	WORD $0x4ea06510 // smax.4s	v16, v8, v0
	WORD $0x4ea16e10 // smin.4s	v16, v16, v1
	WORD $0x6ebf8631 // sub.4s	v17, v17, v31
	WORD $0x4ea06631 // smax.4s	v17, v17, v0
	WORD $0x4ea16e31 // smin.4s	v17, v17, v1
	WORD $0x52803227 // mov	w7, #0x191              ; =401
	WORD $0x4e040cfe // dup.4s	v30, w7
	WORD $0x4ebe9edf // mul.4s	v31, v22, v30
	WORD $0x4f000688 // movi.4s	v8, #0x14
	WORD $0x4ea8979f // mla.4s	v31, v28, v8
	WORD $0x4f3427ff // srshr.4s	v31, v31, #0xc
	WORD $0x5280c5e7 // mov	w7, #0x62f              ; =1583
	WORD $0x4e040ce8 // dup.4s	v8, w7
	WORD $0x1280a247 // mov	w7, #-0x513             ; =-1299
	WORD $0x4e040ce9 // dup.4s	v9, w7
	WORD $0x6ebc87ff // sub.4s	v31, v31, v28
	WORD $0x4ea99f49 // mul.4s	v9, v26, v9
	WORD $0x4ea89769 // mla.4s	v9, v27, v8
	WORD $0x5280f167 // mov	w7, #0x78b              ; =1931
	WORD $0x4e040cea // dup.4s	v10, w7
	WORD $0x4f35252b // srshr.4s	v11, v9, #0xb
	WORD $0x4eaa9f0c // mul.4s	v12, v24, v10
	WORD $0x52803c87 // mov	w7, #0x1e4              ; =484
	WORD $0x4e040ced // dup.4s	v13, w7
	WORD $0x4ead97ac // mla.4s	v12, v29, v13
	WORD $0x4f34258c // srshr.4s	v12, v12, #0xc
	WORD $0x6ebd858c // sub.4s	v12, v12, v29
	WORD $0x6f0505ed // mvni.4s	v13, #0xaf
	WORD $0x12809487 // mov	w7, #-0x4a5             ; =-1189
	WORD $0x4e040cee // dup.4s	v14, w7
	WORD $0x4eae9eee // mul.4s	v14, v23, v14
	WORD $0x4ead972e // mla.4s	v14, v25, v13
	WORD $0x528094a7 // mov	w7, #0x4a5              ; =1189
	WORD $0x4e040cef // dup.4s	v15, w7
	WORD $0x4ead9eed // mul.4s	v13, v23, v13
	WORD $0x4eaf972d // mla.4s	v13, v25, v15
	WORD $0x4f3435d9 // srsra.4s	v25, v14, #0xc
	WORD $0x4f3435b7 // srsra.4s	v23, v13, #0xc
	WORD $0x12803c67 // mov	w7, #-0x1e4             ; =-484
	WORD $0x4e040ced // dup.4s	v13, w7
	WORD $0x4ead9f0d // mul.4s	v13, v24, v13
	WORD $0x4eaa97ad // mla.4s	v13, v29, v10
	WORD $0x4f3435b8 // srsra.4s	v24, v13, #0xc
	WORD $0x5280a267 // mov	w7, #0x513              ; =1299
	WORD $0x4e040cfd // dup.4s	v29, w7
	WORD $0x4ea89f5a // mul.4s	v26, v26, v8
	WORD $0x4ebd977a // mla.4s	v26, v27, v29
	WORD $0x4f35275b // srshr.4s	v27, v26, #0xb
	WORD $0x6f00067d // mvni.4s	v29, #0x13
	WORD $0x4ebd9edd // mul.4s	v29, v22, v29
	WORD $0x4ebe979d // mla.4s	v29, v28, v30
	WORD $0x4f3437b6 // srsra.4s	v22, v29, #0xc
	WORD $0x6eab87fc // sub.4s	v28, v31, v11
	WORD $0x4f35353f // srsra.4s	v31, v9, #0xb
	WORD $0x4ea067fd // smax.4s	v29, v31, v0
	WORD $0x4ea16fbd // smin.4s	v29, v29, v1
	WORD $0x4ea0679c // smax.4s	v28, v28, v0
	WORD $0x4ea16f9c // smin.4s	v28, v28, v1
	WORD $0x6eac873e // sub.4s	v30, v25, v12
	WORD $0x4ea067de // smax.4s	v30, v30, v0
	WORD $0x4ea16fde // smin.4s	v30, v30, v1
	WORD $0x4eac8739 // add.4s	v25, v25, v12
	WORD $0x4ea06739 // smax.4s	v25, v25, v0
	WORD $0x4ea16f39 // smin.4s	v25, v25, v1
	WORD $0x4eb7871f // add.4s	v31, v24, v23
	WORD $0x4ea067ff // smax.4s	v31, v31, v0
	WORD $0x4ea16fff // smin.4s	v31, v31, v1
	WORD $0x6eb886f7 // sub.4s	v23, v23, v24
	WORD $0x4ea066f7 // smax.4s	v23, v23, v0
	WORD $0x4ea16ef7 // smin.4s	v23, v23, v1
	WORD $0x6ebb86d8 // sub.4s	v24, v22, v27
	WORD $0x4ea06718 // smax.4s	v24, v24, v0
	WORD $0x4ea16f18 // smin.4s	v24, v24, v1
	WORD $0x4f353756 // srsra.4s	v22, v26, #0xb
	WORD $0x4ea066d6 // smax.4s	v22, v22, v0
	WORD $0x4ea16eda // smin.4s	v26, v22, v1
	WORD $0x4ea69f96 // mul.4s	v22, v28, v6
	WORD $0x4ea59716 // mla.4s	v22, v24, v5
	WORD $0x4f3426d6 // srshr.4s	v22, v22, #0xc
	WORD $0x6ebc86db // sub.4s	v27, v22, v28
	WORD $0x4ea59f96 // mul.4s	v22, v28, v5
	WORD $0x4ea79716 // mla.4s	v22, v24, v7
	WORD $0x4f3436d8 // srsra.4s	v24, v22, #0xc
	WORD $0x1280c3c7 // mov	w7, #-0x61f             ; =-1567
	WORD $0x4e040ce7 // dup.4s	v7, w7
	WORD $0x4ea79fc7 // mul.4s	v7, v30, v7
	WORD $0x4ea696e7 // mla.4s	v7, v23, v6
	WORD $0x4f3424e7 // srshr.4s	v7, v7, #0xc
	WORD $0x6eb784fc // sub.4s	v28, v7, v23
	WORD $0x4ea69fc6 // mul.4s	v6, v30, v6
	WORD $0x4ea596e6 // mla.4s	v6, v23, v5
	WORD $0x4f3424c5 // srshr.4s	v5, v6, #0xc
	WORD $0x6ebe84a5 // sub.4s	v5, v5, v30
	WORD $0x4ebd8726 // add.4s	v6, v25, v29
	WORD $0x4ea064c6 // smax.4s	v6, v6, v0
	WORD $0x4ea16cc7 // smin.4s	v7, v6, v1
	WORD $0x4ebb8786 // add.4s	v6, v28, v27
	WORD $0x4ea064c6 // smax.4s	v6, v6, v0
	WORD $0x4ea16cd6 // smin.4s	v22, v6, v1
	WORD $0x6ebc8766 // sub.4s	v6, v27, v28
	WORD $0x4ea064c6 // smax.4s	v6, v6, v0
	WORD $0x4ea16cdb // smin.4s	v27, v6, v1
	WORD $0x6eb987a6 // sub.4s	v6, v29, v25
	WORD $0x4ea064c6 // smax.4s	v6, v6, v0
	WORD $0x4ea16cd9 // smin.4s	v25, v6, v1
	WORD $0x6ebf8746 // sub.4s	v6, v26, v31
	WORD $0x4ea064c6 // smax.4s	v6, v6, v0
	WORD $0x4ea16cdc // smin.4s	v28, v6, v1
	WORD $0x6ea58706 // sub.4s	v6, v24, v5
	WORD $0x4ea064c6 // smax.4s	v6, v6, v0
	WORD $0x4ea16cdd // smin.4s	v29, v6, v1
	WORD $0x4eb884a5 // add.4s	v5, v5, v24
	WORD $0x4ea064a5 // smax.4s	v5, v5, v0
	WORD $0x4ea16ca6 // smin.4s	v6, v5, v1
	WORD $0x4ebf8745 // add.4s	v5, v26, v31
	WORD $0x4ea064a5 // smax.4s	v5, v5, v0
	WORD $0x4ea16ca5 // smin.4s	v5, v5, v1
	WORD $0x6ebb87b7 // sub.4s	v23, v29, v27
	WORD $0x4ea29ef7 // mul.4s	v23, v23, v2
	WORD $0x4f3826f8 // srshr.4s	v24, v23, #0x8
	WORD $0x4ebb87ba // add.4s	v26, v29, v27
	WORD $0x4ea29f5a // mul.4s	v26, v26, v2
	WORD $0x4f38275b // srshr.4s	v27, v26, #0x8
	WORD $0x6eb9879d // sub.4s	v29, v28, v25
	WORD $0x4ea29fbd // mul.4s	v29, v29, v2
	WORD $0x4f3827be // srshr.4s	v30, v29, #0x8
	WORD $0x4eb98799 // add.4s	v25, v28, v25
	WORD $0x4ea29f39 // mul.4s	v25, v25, v2
	WORD $0x4f38273c // srshr.4s	v28, v25, #0x8
	WORD $0x4ea384a2 // add.4s	v2, v5, v3
	WORD $0x4ea06442 // smax.4s	v2, v2, v0
	WORD $0x4ea16c42 // smin.4s	v2, v2, v1
	WORD $0x4ea484df // add.4s	v31, v6, v4
	WORD $0x4ea067ff // smax.4s	v31, v31, v0
	WORD $0x4ea16fff // smin.4s	v31, v31, v1
	WORD $0x6ebb86bb // sub.4s	v27, v21, v27
	WORD $0x4f383755 // srsra.4s	v21, v26, #0x8
	WORD $0x4ea066b5 // smax.4s	v21, v21, v0
	WORD $0x4ea16eb5 // smin.4s	v21, v21, v1
	WORD $0x6ebc869a // sub.4s	v26, v20, v28
	WORD $0x4f383734 // srsra.4s	v20, v25, #0x8
	WORD $0x4ea06694 // smax.4s	v20, v20, v0
	WORD $0x4ea16e94 // smin.4s	v20, v20, v1
	WORD $0x6ebe8679 // sub.4s	v25, v19, v30
	WORD $0x4f3837b3 // srsra.4s	v19, v29, #0x8
	WORD $0x4ea06673 // smax.4s	v19, v19, v0
	WORD $0x4ea16e73 // smin.4s	v19, v19, v1
	WORD $0x6eb88658 // sub.4s	v24, v18, v24
	WORD $0x3d800002 // str	q2, [x0]
	WORD $0x3ca1681f // str	q31, [x0, x1]
	WORD $0x4f3836f2 // srsra.4s	v18, v23, #0x8
	WORD $0x4ea06642 // smax.4s	v2, v18, v0
	WORD $0x4ea16c42 // smin.4s	v2, v2, v1
	WORD $0x4eb086d2 // add.4s	v18, v22, v16
	WORD $0x4ea06652 // smax.4s	v18, v18, v0
	WORD $0x4ea16e52 // smin.4s	v18, v18, v1
	WORD $0x4eb184f7 // add.4s	v23, v7, v17
	WORD $0x3cac6815 // str	q21, [x0, x12]
	WORD $0x3ca56814 // str	q20, [x0, x5]
	WORD $0x3caa6813 // str	q19, [x0, x10]
	WORD $0x3ca66802 // str	q2, [x0, x6]
	WORD $0x4ea066e2 // smax.4s	v2, v23, v0
	WORD $0x4ea16c42 // smin.4s	v2, v2, v1
	WORD $0x6ea78627 // sub.4s	v7, v17, v7
	WORD $0x4ea064e7 // smax.4s	v7, v7, v0
	WORD $0x4ea16ce7 // smin.4s	v7, v7, v1
	WORD $0x6eb68610 // sub.4s	v16, v16, v22
	WORD $0x4ea06610 // smax.4s	v16, v16, v0
	WORD $0x4ea16e10 // smin.4s	v16, v16, v1
	WORD $0x3cae6812 // str	q18, [x0, x14]
	WORD $0x3ca46802 // str	q2, [x0, x4]
	WORD $0x4ea06702 // smax.4s	v2, v24, v0
	WORD $0x4ea16c42 // smin.4s	v2, v2, v1
	WORD $0x4ea06731 // smax.4s	v17, v25, v0
	WORD $0x3ca86807 // str	q7, [x0, x8]
	WORD $0x4ea16e27 // smin.4s	v7, v17, v1
	WORD $0x3ca36810 // str	q16, [x0, x3]
	WORD $0x4ea06750 // smax.4s	v16, v26, v0
	WORD $0x4ea16e10 // smin.4s	v16, v16, v1
	WORD $0x3cad6802 // str	q2, [x0, x13]
	WORD $0x4ea06762 // smax.4s	v2, v27, v0
	WORD $0x4ea16c42 // smin.4s	v2, v2, v1
	WORD $0x3ca26807 // str	q7, [x0, x2]
	WORD $0x3ca96810 // str	q16, [x0, x9]
	WORD $0x6ea68484 // sub.4s	v4, v4, v6
	WORD $0x4ea06484 // smax.4s	v4, v4, v0
	WORD $0x3cb16802 // str	q2, [x0, x17]
	WORD $0x4ea16c82 // smin.4s	v2, v4, v1
	WORD $0x6ea58463 // sub.4s	v3, v3, v5
	WORD $0x3cab6802 // str	q2, [x0, x11]
	WORD $0x4ea06460 // smax.4s	v0, v3, v0
	WORD $0x4ea16c00 // smin.4s	v0, v0, v1
	WORD $0x3caf6a00 // str	q0, [x16, x15]
	WORD $0x6d4323e9 // ldp	d9, d8, [sp, #0x30]
	WORD $0x6d422beb // ldp	d11, d10, [sp, #0x20]
	WORD $0x6d4133ed // ldp	d13, d12, [sp, #0x10]
	WORD $0x6cc43bef // ldp	d15, d14, [sp], #0x40
	WORD $0xd65f03c0 // ret

// func inverseADST8Col4NEON(base *int32, rowStrideBytes, min, max int64)
TEXT ·inverseADST8Col4NEON(SB), NOSPLIT, $0-32
	MOVD base+0(FP), R0
	MOVD rowStrideBytes+8(FP), R1
	MOVD min+16(FP), R2
	MOVD max+24(FP), R3
	WORD $0x3ce16802 // ldr	q2, [x0, x1]
	WORD $0xd37ff828 // lsl	x8, x1, #1
	WORD $0x3ce86806 // ldr	q6, [x0, x8]
	WORD $0x8b010109 // add	x9, x8, x1
	WORD $0x3ce96810 // ldr	q16, [x0, x9]
	WORD $0xd37ef42a // lsl	x10, x1, #2
	WORD $0x3cea6811 // ldr	q17, [x0, x10]
	WORD $0x8b01014b // add	x11, x10, x1
	WORD $0x3ceb6803 // ldr	q3, [x0, x11]
	WORD $0x528000cc // mov	w12, #0x6               ; =6
	WORD $0x9b0c7c2c // mul	x12, x1, x12
	WORD $0x3cec6804 // ldr	q4, [x0, x12]
	WORD $0xd37df02d // lsl	x13, x1, #3
	WORD $0xcb01000e // sub	x14, x0, x1
	WORD $0x3ced69c5 // ldr	q5, [x14, x13]
	WORD $0x3dc00007 // ldr	q7, [x0]
	WORD $0x4e040c40 // dup.4s	v0, w2
	WORD $0x4e040c61 // dup.4s	v1, w3
	WORD $0x6f000672 // mvni.4s	v18, #0x13
	WORD $0x5280322f // mov	w15, #0x191             ; =401
	WORD $0x4e040df3 // dup.4s	v19, w15
	WORD $0x4eb39cf4 // mul.4s	v20, v7, v19
	WORD $0x4eb294b4 // mla.4s	v20, v5, v18
	WORD $0x4f000692 // movi.4s	v18, #0x14
	WORD $0x4eb29cf2 // mul.4s	v18, v7, v18
	WORD $0x4eb394b2 // mla.4s	v18, v5, v19
	WORD $0x4f343685 // srsra.4s	v5, v20, #0xc
	WORD $0x4f342652 // srshr.4s	v18, v18, #0xc
	WORD $0x6ea78647 // sub.4s	v7, v18, v7
	WORD $0x12803c6f // mov	w15, #-0x1e4            ; =-484
	WORD $0x4e040df2 // dup.4s	v18, w15
	WORD $0x5280f16f // mov	w15, #0x78b             ; =1931
	WORD $0x4e040df3 // dup.4s	v19, w15
	WORD $0x4eb39cd4 // mul.4s	v20, v6, v19
	WORD $0x4eb29474 // mla.4s	v20, v3, v18
	WORD $0x52803c8f // mov	w15, #0x1e4             ; =484
	WORD $0x4e040df2 // dup.4s	v18, w15
	WORD $0x4eb29cd2 // mul.4s	v18, v6, v18
	WORD $0x4eb39472 // mla.4s	v18, v3, v19
	WORD $0x4f343683 // srsra.4s	v3, v20, #0xc
	WORD $0x4f342652 // srshr.4s	v18, v18, #0xc
	WORD $0x6ea68646 // sub.4s	v6, v18, v6
	WORD $0x5280a26f // mov	w15, #0x513             ; =1299
	WORD $0x4e040df2 // dup.4s	v18, w15
	WORD $0x4eb29e12 // mul.4s	v18, v16, v18
	WORD $0x5280c5ef // mov	w15, #0x62f             ; =1583
	WORD $0x4e040df3 // dup.4s	v19, w15
	WORD $0x4eb39632 // mla.4s	v18, v17, v19
	WORD $0x4f352654 // srshr.4s	v20, v18, #0xb
	WORD $0x4eb39e10 // mul.4s	v16, v16, v19
	WORD $0x1280a24f // mov	w15, #-0x513            ; =-1299
	WORD $0x4e040df3 // dup.4s	v19, w15
	WORD $0x4eb39630 // mla.4s	v16, v17, v19
	WORD $0x4f352611 // srshr.4s	v17, v16, #0xb
	WORD $0x528094af // mov	w15, #0x4a5             ; =1189
	WORD $0x4e040df3 // dup.4s	v19, w15
	WORD $0x4eb39c53 // mul.4s	v19, v2, v19
	WORD $0x6f0505f5 // mvni.4s	v21, #0xaf
	WORD $0x4eb59493 // mla.4s	v19, v4, v21
	WORD $0x4eb59c55 // mul.4s	v21, v2, v21
	WORD $0x1280948f // mov	w15, #-0x4a5            ; =-1189
	WORD $0x4e040df6 // dup.4s	v22, w15
	WORD $0x4eb69495 // mla.4s	v21, v4, v22
	WORD $0x4f343664 // srsra.4s	v4, v19, #0xc
	WORD $0x4f3436a2 // srsra.4s	v2, v21, #0xc
	WORD $0x6eb484b3 // sub.4s	v19, v5, v20
	WORD $0x4f353645 // srsra.4s	v5, v18, #0xb
	WORD $0x4ea064a5 // smax.4s	v5, v5, v0
	WORD $0x4ea16ca5 // smin.4s	v5, v5, v1
	WORD $0x6eb184f1 // sub.4s	v17, v7, v17
	WORD $0x4f353607 // srsra.4s	v7, v16, #0xb
	WORD $0x4ea064e7 // smax.4s	v7, v7, v0
	WORD $0x4ea16ce7 // smin.4s	v7, v7, v1
	WORD $0x4ea38490 // add.4s	v16, v4, v3
	WORD $0x4ea06610 // smax.4s	v16, v16, v0
	WORD $0x4ea16e10 // smin.4s	v16, v16, v1
	WORD $0x4ea68452 // add.4s	v18, v2, v6
	WORD $0x4ea06652 // smax.4s	v18, v18, v0
	WORD $0x4ea16e52 // smin.4s	v18, v18, v1
	WORD $0x4ea06673 // smax.4s	v19, v19, v0
	WORD $0x4ea16e73 // smin.4s	v19, v19, v1
	WORD $0x4ea06631 // smax.4s	v17, v17, v0
	WORD $0x4ea16e31 // smin.4s	v17, v17, v1
	WORD $0x6ea48463 // sub.4s	v3, v3, v4
	WORD $0x4ea06463 // smax.4s	v3, v3, v0
	WORD $0x4ea16c63 // smin.4s	v3, v3, v1
	WORD $0x6ea284c2 // sub.4s	v2, v6, v2
	WORD $0x4ea06442 // smax.4s	v2, v2, v0
	WORD $0x4ea16c42 // smin.4s	v2, v2, v1
	WORD $0x128026ef // mov	w15, #-0x138            ; =-312
	WORD $0x4e040de4 // dup.4s	v4, w15
	WORD $0x4ea49e66 // mul.4s	v6, v19, v4
	WORD $0x5280c3ef // mov	w15, #0x61f             ; =1567
	WORD $0x4e040df4 // dup.4s	v20, w15
	WORD $0x4eb49626 // mla.4s	v6, v17, v20
	WORD $0x4eb49e75 // mul.4s	v21, v19, v20
	WORD $0x4f3434d3 // srsra.4s	v19, v6, #0xc
	WORD $0x5280270f // mov	w15, #0x138             ; =312
	WORD $0x4e040de6 // dup.4s	v6, w15
	WORD $0x4ea69635 // mla.4s	v21, v17, v6
	WORD $0x4f3426a6 // srshr.4s	v6, v21, #0xc
	WORD $0x6eb184c6 // sub.4s	v6, v6, v17
	WORD $0x1280c3cf // mov	w15, #-0x61f            ; =-1567
	WORD $0x4e040df1 // dup.4s	v17, w15
	WORD $0x4eb19c71 // mul.4s	v17, v3, v17
	WORD $0x4ea49451 // mla.4s	v17, v2, v4
	WORD $0x4ea49c64 // mul.4s	v4, v3, v4
	WORD $0x4eb49444 // mla.4s	v4, v2, v20
	WORD $0x4f343622 // srsra.4s	v2, v17, #0xc
	WORD $0x4f343483 // srsra.4s	v3, v4, #0xc
	WORD $0x4ea58604 // add.4s	v4, v16, v5
	WORD $0x4ea06484 // smax.4s	v4, v4, v0
	WORD $0x4ea16c84 // smin.4s	v4, v4, v1
	WORD $0x4ea78651 // add.4s	v17, v18, v7
	WORD $0x4ea06631 // smax.4s	v17, v17, v0
	WORD $0x4ea16e31 // smin.4s	v17, v17, v1
	WORD $0x6ea0ba31 // neg.4s	v17, v17
	WORD $0x6eb084a5 // sub.4s	v5, v5, v16
	WORD $0x4ea064a5 // smax.4s	v5, v5, v0
	WORD $0x4ea16ca5 // smin.4s	v5, v5, v1
	WORD $0x6eb284e7 // sub.4s	v7, v7, v18
	WORD $0x4ea064e7 // smax.4s	v7, v7, v0
	WORD $0x4ea16ce7 // smin.4s	v7, v7, v1
	WORD $0x4eb38450 // add.4s	v16, v2, v19
	WORD $0x4ea06610 // smax.4s	v16, v16, v0
	WORD $0x4ea16e10 // smin.4s	v16, v16, v1
	WORD $0x6ea0ba10 // neg.4s	v16, v16
	WORD $0x4ea68472 // add.4s	v18, v3, v6
	WORD $0x6ea28662 // sub.4s	v2, v19, v2
	WORD $0x4ea06442 // smax.4s	v2, v2, v0
	WORD $0x4ea16c42 // smin.4s	v2, v2, v1
	WORD $0x6ea384c3 // sub.4s	v3, v6, v3
	WORD $0x4ea06463 // smax.4s	v3, v3, v0
	WORD $0x4ea16c63 // smin.4s	v3, v3, v1
	WORD $0x4ea584e6 // add.4s	v6, v7, v5
	WORD $0x4f0506b3 // movi.4s	v19, #0xb5
	WORD $0x4eb39cc6 // mul.4s	v6, v6, v19
	WORD $0x4f3824c6 // srshr.4s	v6, v6, #0x8
	WORD $0x6ea0b8c6 // neg.4s	v6, v6
	WORD $0x6ea784a5 // sub.4s	v5, v5, v7
	WORD $0x4eb39ca5 // mul.4s	v5, v5, v19
	WORD $0x4f3824a5 // srshr.4s	v5, v5, #0x8
	WORD $0x4ea28467 // add.4s	v7, v3, v2
	WORD $0x4eb39ce7 // mul.4s	v7, v7, v19
	WORD $0x4f3824e7 // srshr.4s	v7, v7, #0x8
	WORD $0x6ea38442 // sub.4s	v2, v2, v3
	WORD $0x4eb39c42 // mul.4s	v2, v2, v19
	WORD $0x4f382442 // srshr.4s	v2, v2, #0x8
	WORD $0x6ea0b842 // neg.4s	v2, v2
	WORD $0x3d800004 // str	q4, [x0]
	WORD $0x3ca16810 // str	q16, [x0, x1]
	WORD $0x3ca86807 // str	q7, [x0, x8]
	WORD $0x3ca96806 // str	q6, [x0, x9]
	WORD $0x3caa6805 // str	q5, [x0, x10]
	WORD $0x3cab6802 // str	q2, [x0, x11]
	WORD $0x4ea06640 // smax.4s	v0, v18, v0
	WORD $0x4ea16c00 // smin.4s	v0, v0, v1
	WORD $0x3cac6800 // str	q0, [x0, x12]
	WORD $0x3cad69d1 // str	q17, [x14, x13]
	WORD $0xd65f03c0 // ret


