// NEON-accelerated inverse-transform row-pass kernels.
//
// Each kernel transforms two adjacent stride-1 rows at once, holding the same
// coefficient index from each row in the two lanes of a 64-bit-element NEON
// register. All multiplies widen int32->int64 (SMULL/SMLAL), the rounding
// shifts use SRSHR and the per-stage clamps use CMGT+BSL, so the fixed-point
// arithmetic, rounding and clamp ranges are bit-for-bit identical to the
// pure-Go reference kernels for every supported bit depth (8/10/12).
//
// Go's arm64 assembler does not expose the signed widening-multiply and
// signed-rounding-shift NEON mnemonics, so the instructions are emitted as
// WORD-encoded opcodes (the disassembly is in the trailing comment of each
// line; generated from the AArch64 reference assembler).
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

#include "textflag.h"

// func inverseDCT4Row2NEON(r0, r1 *int32, min, max int64)
TEXT ·inverseDCT4Row2NEON(SB), NOSPLIT, $0-32
	MOVD r0+0(FP), R0
	MOVD r1+8(FP), R1
	MOVD min+16(FP), R2
	MOVD max+24(FP), R3
	WORD $0x4c407800 // ld1.4s	{ v0 }, [x0]
	WORD $0x4c407821 // ld1.4s	{ v1 }, [x1]
	WORD $0x4e813802 // zip1.4s	v2, v0, v1
	WORD $0x4e817803 // zip2.4s	v3, v0, v1
	WORD $0x6e024045 // ext.16b	v5, v2, v2, #0x8
	WORD $0x6e034067 // ext.16b	v7, v3, v3, #0x8
	WORD $0x528016a4 // mov	w4, #0xb5               ; =181
	WORD $0x0e040c90 // dup.2s	v16, w4
	WORD $0x5280c3e4 // mov	w4, #0x61f              ; =1567
	WORD $0x0e040c91 // dup.2s	v17, w4
	WORD $0x128026e4 // mov	w4, #-0x138             ; =-312
	WORD $0x0e040c92 // dup.2s	v18, w4
	WORD $0x0ea38454 // add.2s	v20, v2, v3
	WORD $0x0eb0c294 // smull.2d	v20, v20, v16
	WORD $0x4f782694 // srshr.2d	v20, v20, #0x8
	WORD $0x2ea38455 // sub.2s	v21, v2, v3
	WORD $0x0eb0c2b5 // smull.2d	v21, v21, v16
	WORD $0x4f7826b5 // srshr.2d	v21, v21, #0x8
	WORD $0x0eb1c0b6 // smull.2d	v22, v5, v17
	WORD $0x0eb2c0f7 // smull.2d	v23, v7, v18
	WORD $0x6ef786d6 // sub.2d	v22, v22, v23
	WORD $0x4f7426d6 // srshr.2d	v22, v22, #0xc
	WORD $0x0f20a4f8 // sshll.2d	v24, v7, #0x0
	WORD $0x6ef886d6 // sub.2d	v22, v22, v24
	WORD $0x0eb2c0b9 // smull.2d	v25, v5, v18
	WORD $0x0eb180f9 // smlal.2d	v25, v7, v17
	WORD $0x4f742739 // srshr.2d	v25, v25, #0xc
	WORD $0x0f20a4ba // sshll.2d	v26, v5, #0x0
	WORD $0x4efa8739 // add.2d	v25, v25, v26
	WORD $0x4e080c5c // dup.2d	v28, x2
	WORD $0x4e080c7d // dup.2d	v29, x3
	WORD $0x4ef98688 // add.2d	v8, v20, v25
	WORD $0x4ee8379e // cmgt.2d	v30, v28, v8
	WORD $0x6e681f9e // bsl.16b	v30, v28, v8
	WORD $0x4efd37df // cmgt.2d	v31, v30, v29
	WORD $0x6e7e1fbf // bsl.16b	v31, v29, v30
	WORD $0x4ef686a9 // add.2d	v9, v21, v22
	WORD $0x4ee9378a // cmgt.2d	v10, v28, v9
	WORD $0x6e691f8a // bsl.16b	v10, v28, v9
	WORD $0x4efd354b // cmgt.2d	v11, v10, v29
	WORD $0x6e6a1fab // bsl.16b	v11, v29, v10
	WORD $0x6ef686ac // sub.2d	v12, v21, v22
	WORD $0x4eec378d // cmgt.2d	v13, v28, v12
	WORD $0x6e6c1f8d // bsl.16b	v13, v28, v12
	WORD $0x4efd35ae // cmgt.2d	v14, v13, v29
	WORD $0x6e6d1fae // bsl.16b	v14, v29, v13
	WORD $0x6ef9868f // sub.2d	v15, v20, v25
	WORD $0x4eef3786 // cmgt.2d	v6, v28, v15
	WORD $0x6e6f1f86 // bsl.16b	v6, v28, v15
	WORD $0x4efd34c4 // cmgt.2d	v4, v6, v29
	WORD $0x6e661fa4 // bsl.16b	v4, v29, v6
	WORD $0x0ea12bff // xtn.2s	v31, v31
	WORD $0x0ea1296b // xtn.2s	v11, v11
	WORD $0x0ea129ce // xtn.2s	v14, v14
	WORD $0x0ea12884 // xtn.2s	v4, v4
	WORD $0x0d9f801f // st1.s	{ v31 }[0], [x0], #4
	WORD $0x0d9f800b // st1.s	{ v11 }[0], [x0], #4
	WORD $0x0d9f800e // st1.s	{ v14 }[0], [x0], #4
	WORD $0x0d008004 // st1.s	{ v4 }[0], [x0]
	WORD $0x0d9f903f // st1.s	{ v31 }[1], [x1], #4
	WORD $0x0d9f902b // st1.s	{ v11 }[1], [x1], #4
	WORD $0x0d9f902e // st1.s	{ v14 }[1], [x1], #4
	WORD $0x0d009024 // st1.s	{ v4 }[1], [x1]
	RET

// func inverseDCT8Row2NEON(r0, r1 *int32, min, max int64)
TEXT ·inverseDCT8Row2NEON(SB), NOSPLIT, $0-32
	MOVD r0+0(FP), R0
	MOVD r1+8(FP), R1
	MOVD min+16(FP), R2
	MOVD max+24(FP), R3
	WORD $0x4c40a800 // ld1.4s	{ v0, v1 }, [x0]
	WORD $0x4c40a822 // ld1.4s	{ v2, v3 }, [x1]
	WORD $0x4e823804 // zip1.4s	v4, v0, v2
	WORD $0x4e827805 // zip2.4s	v5, v0, v2
	WORD $0x4e833826 // zip1.4s	v6, v1, v3
	WORD $0x4e837827 // zip2.4s	v7, v1, v3
	WORD $0x6e044088 // ext.16b	v8, v4, v4, #0x8
	WORD $0x6e0540a9 // ext.16b	v9, v5, v5, #0x8
	WORD $0x6e0640ca // ext.16b	v10, v6, v6, #0x8
	WORD $0x6e0740eb // ext.16b	v11, v7, v7, #0x8
	WORD $0x528016a4 // mov	w4, #0xb5               ; =181
	WORD $0x0e040c98 // dup.2s	v24, w4
	WORD $0x5280c3e4 // mov	w4, #0x61f              ; =1567
	WORD $0x0e040c99 // dup.2s	v25, w4
	WORD $0x128026e4 // mov	w4, #-0x138             ; =-312
	WORD $0x0e040c9a // dup.2s	v26, w4
	WORD $0x528063e4 // mov	w4, #0x31f              ; =799
	WORD $0x0e040c9b // dup.2s	v27, w4
	WORD $0x128009c4 // mov	w4, #-0x4f              ; =-79
	WORD $0x0e040c9c // dup.2s	v28, w4
	WORD $0x5280d4e4 // mov	w4, #0x6a7              ; =1703
	WORD $0x0e040c9d // dup.2s	v29, w4
	WORD $0x52808e44 // mov	w4, #0x472              ; =1138
	WORD $0x0e040c9e // dup.2s	v30, w4
	WORD $0x4e080c56 // dup.2d	v22, x2
	WORD $0x4e080c77 // dup.2d	v23, x3
	WORD $0x0ea6848c // add.2s	v12, v4, v6
	WORD $0x0eb8c18c // smull.2d	v12, v12, v24
	WORD $0x4f78258c // srshr.2d	v12, v12, #0x8
	WORD $0x2ea6848d // sub.2s	v13, v4, v6
	WORD $0x0eb8c1ad // smull.2d	v13, v13, v24
	WORD $0x4f7825ad // srshr.2d	v13, v13, #0x8
	WORD $0x0eb9c0ae // smull.2d	v14, v5, v25
	WORD $0x0ebac0ef // smull.2d	v15, v7, v26
	WORD $0x6eef85ce // sub.2d	v14, v14, v15
	WORD $0x4f7425ce // srshr.2d	v14, v14, #0xc
	WORD $0x0f20a4f0 // sshll.2d	v16, v7, #0x0
	WORD $0x6ef085ce // sub.2d	v14, v14, v16
	WORD $0x0ebac0af // smull.2d	v15, v5, v26
	WORD $0x0eb980ef // smlal.2d	v15, v7, v25
	WORD $0x4f7425ef // srshr.2d	v15, v15, #0xc
	WORD $0x0f20a4b1 // sshll.2d	v17, v5, #0x0
	WORD $0x4ef185ef // add.2d	v15, v15, v17
	WORD $0x4eef8580 // add.2d	v0, v12, v15
	WORD $0x4ee036d2 // cmgt.2d	v18, v22, v0
	WORD $0x6e601ed2 // bsl.16b	v18, v22, v0
	WORD $0x4ef73640 // cmgt.2d	v0, v18, v23
	WORD $0x6e721ee0 // bsl.16b	v0, v23, v18
	WORD $0x4eee85a1 // add.2d	v1, v13, v14
	WORD $0x4ee136d2 // cmgt.2d	v18, v22, v1
	WORD $0x6e611ed2 // bsl.16b	v18, v22, v1
	WORD $0x4ef73641 // cmgt.2d	v1, v18, v23
	WORD $0x6e721ee1 // bsl.16b	v1, v23, v18
	WORD $0x6eee85a2 // sub.2d	v2, v13, v14
	WORD $0x4ee236d2 // cmgt.2d	v18, v22, v2
	WORD $0x6e621ed2 // bsl.16b	v18, v22, v2
	WORD $0x4ef73642 // cmgt.2d	v2, v18, v23
	WORD $0x6e721ee2 // bsl.16b	v2, v23, v18
	WORD $0x6eef8583 // sub.2d	v3, v12, v15
	WORD $0x4ee336d2 // cmgt.2d	v18, v22, v3
	WORD $0x6e631ed2 // bsl.16b	v18, v22, v3
	WORD $0x4ef73643 // cmgt.2d	v3, v18, v23
	WORD $0x6e721ee3 // bsl.16b	v3, v23, v18
	WORD $0x0ebbc10c // smull.2d	v12, v8, v27
	WORD $0x0ebcc16d // smull.2d	v13, v11, v28
	WORD $0x6eed858c // sub.2d	v12, v12, v13
	WORD $0x4f74258c // srshr.2d	v12, v12, #0xc
	WORD $0x0f20a570 // sshll.2d	v16, v11, #0x0
	WORD $0x6ef0858c // sub.2d	v12, v12, v16
	WORD $0x0ebdc14d // smull.2d	v13, v10, v29
	WORD $0x0ebec12e // smull.2d	v14, v9, v30
	WORD $0x6eee85ad // sub.2d	v13, v13, v14
	WORD $0x4f7525ad // srshr.2d	v13, v13, #0xb
	WORD $0x0ebec14e // smull.2d	v14, v10, v30
	WORD $0x0ebd812e // smlal.2d	v14, v9, v29
	WORD $0x4f7525ce // srshr.2d	v14, v14, #0xb
	WORD $0x0ebcc10f // smull.2d	v15, v8, v28
	WORD $0x0ebb816f // smlal.2d	v15, v11, v27
	WORD $0x4f7425ef // srshr.2d	v15, v15, #0xc
	WORD $0x0f20a511 // sshll.2d	v17, v8, #0x0
	WORD $0x4ef185ef // add.2d	v15, v15, v17
	WORD $0x4eed8584 // add.2d	v4, v12, v13
	WORD $0x4ee436d2 // cmgt.2d	v18, v22, v4
	WORD $0x6e641ed2 // bsl.16b	v18, v22, v4
	WORD $0x4ef73644 // cmgt.2d	v4, v18, v23
	WORD $0x6e721ee4 // bsl.16b	v4, v23, v18
	WORD $0x6eed8585 // sub.2d	v5, v12, v13
	WORD $0x4ee536d2 // cmgt.2d	v18, v22, v5
	WORD $0x6e651ed2 // bsl.16b	v18, v22, v5
	WORD $0x4ef73645 // cmgt.2d	v5, v18, v23
	WORD $0x6e721ee5 // bsl.16b	v5, v23, v18
	WORD $0x4eee85e6 // add.2d	v6, v15, v14
	WORD $0x4ee636d2 // cmgt.2d	v18, v22, v6
	WORD $0x6e661ed2 // bsl.16b	v18, v22, v6
	WORD $0x4ef73646 // cmgt.2d	v6, v18, v23
	WORD $0x6e721ee6 // bsl.16b	v6, v23, v18
	WORD $0x6eee85e7 // sub.2d	v7, v15, v14
	WORD $0x4ee736d2 // cmgt.2d	v18, v22, v7
	WORD $0x6e671ed2 // bsl.16b	v18, v22, v7
	WORD $0x4ef73647 // cmgt.2d	v7, v18, v23
	WORD $0x6e721ee7 // bsl.16b	v7, v23, v18
	WORD $0x6ee584ec // sub.2d	v12, v7, v5
	WORD $0x0ea1298c // xtn.2s	v12, v12
	WORD $0x0eb8c18c // smull.2d	v12, v12, v24
	WORD $0x4f78258c // srshr.2d	v12, v12, #0x8
	WORD $0x4ee584ed // add.2d	v13, v7, v5
	WORD $0x0ea129ad // xtn.2s	v13, v13
	WORD $0x0eb8c1ad // smull.2d	v13, v13, v24
	WORD $0x4f7825ad // srshr.2d	v13, v13, #0x8
	WORD $0x4ee6840e // add.2d	v14, v0, v6
	WORD $0x4eee36d2 // cmgt.2d	v18, v22, v14
	WORD $0x6e6e1ed2 // bsl.16b	v18, v22, v14
	WORD $0x4ef7364e // cmgt.2d	v14, v18, v23
	WORD $0x6e721eee // bsl.16b	v14, v23, v18
	WORD $0x6ee6840f // sub.2d	v15, v0, v6
	WORD $0x4eef36d2 // cmgt.2d	v18, v22, v15
	WORD $0x6e6f1ed2 // bsl.16b	v18, v22, v15
	WORD $0x4ef7364f // cmgt.2d	v15, v18, v23
	WORD $0x6e721eef // bsl.16b	v15, v23, v18
	WORD $0x4eed8430 // add.2d	v16, v1, v13
	WORD $0x4ef036d2 // cmgt.2d	v18, v22, v16
	WORD $0x6e701ed2 // bsl.16b	v18, v22, v16
	WORD $0x4ef73650 // cmgt.2d	v16, v18, v23
	WORD $0x6e721ef0 // bsl.16b	v16, v23, v18
	WORD $0x6eed8431 // sub.2d	v17, v1, v13
	WORD $0x4ef136d2 // cmgt.2d	v18, v22, v17
	WORD $0x6e711ed2 // bsl.16b	v18, v22, v17
	WORD $0x4ef73651 // cmgt.2d	v17, v18, v23
	WORD $0x6e721ef1 // bsl.16b	v17, v23, v18
	WORD $0x4eec8453 // add.2d	v19, v2, v12
	WORD $0x4ef336d2 // cmgt.2d	v18, v22, v19
	WORD $0x6e731ed2 // bsl.16b	v18, v22, v19
	WORD $0x4ef73653 // cmgt.2d	v19, v18, v23
	WORD $0x6e721ef3 // bsl.16b	v19, v23, v18
	WORD $0x6eec8454 // sub.2d	v20, v2, v12
	WORD $0x4ef436d2 // cmgt.2d	v18, v22, v20
	WORD $0x6e741ed2 // bsl.16b	v18, v22, v20
	WORD $0x4ef73654 // cmgt.2d	v20, v18, v23
	WORD $0x6e721ef4 // bsl.16b	v20, v23, v18
	WORD $0x4ee48475 // add.2d	v21, v3, v4
	WORD $0x4ef536d2 // cmgt.2d	v18, v22, v21
	WORD $0x6e751ed2 // bsl.16b	v18, v22, v21
	WORD $0x4ef73655 // cmgt.2d	v21, v18, v23
	WORD $0x6e721ef5 // bsl.16b	v21, v23, v18
	WORD $0x6ee4847f // sub.2d	v31, v3, v4
	WORD $0x4eff36d2 // cmgt.2d	v18, v22, v31
	WORD $0x6e7f1ed2 // bsl.16b	v18, v22, v31
	WORD $0x4ef7365f // cmgt.2d	v31, v18, v23
	WORD $0x6e721eff // bsl.16b	v31, v23, v18
	WORD $0x0ea129ce // xtn.2s	v14, v14
	WORD $0x0ea12a10 // xtn.2s	v16, v16
	WORD $0x0ea12a73 // xtn.2s	v19, v19
	WORD $0x0ea12ab5 // xtn.2s	v21, v21
	WORD $0x0ea12bff // xtn.2s	v31, v31
	WORD $0x0ea12a94 // xtn.2s	v20, v20
	WORD $0x0ea12a31 // xtn.2s	v17, v17
	WORD $0x0ea129ef // xtn.2s	v15, v15
	WORD $0x0d9f800e // st1.s	{ v14 }[0], [x0], #4
	WORD $0x0d9f8010 // st1.s	{ v16 }[0], [x0], #4
	WORD $0x0d9f8013 // st1.s	{ v19 }[0], [x0], #4
	WORD $0x0d9f8015 // st1.s	{ v21 }[0], [x0], #4
	WORD $0x0d9f801f // st1.s	{ v31 }[0], [x0], #4
	WORD $0x0d9f8014 // st1.s	{ v20 }[0], [x0], #4
	WORD $0x0d9f8011 // st1.s	{ v17 }[0], [x0], #4
	WORD $0x0d00800f // st1.s	{ v15 }[0], [x0]
	WORD $0x0d9f902e // st1.s	{ v14 }[1], [x1], #4
	WORD $0x0d9f9030 // st1.s	{ v16 }[1], [x1], #4
	WORD $0x0d9f9033 // st1.s	{ v19 }[1], [x1], #4
	WORD $0x0d9f9035 // st1.s	{ v21 }[1], [x1], #4
	WORD $0x0d9f903f // st1.s	{ v31 }[1], [x1], #4
	WORD $0x0d9f9034 // st1.s	{ v20 }[1], [x1], #4
	WORD $0x0d9f9031 // st1.s	{ v17 }[1], [x1], #4
	WORD $0x0d00902f // st1.s	{ v15 }[1], [x1]
	RET

// func inverseADST4Row2NEON(r0, r1 *int32)
TEXT ·inverseADST4Row2NEON(SB), NOSPLIT, $0-16
	MOVD r0+0(FP), R0
	MOVD r1+8(FP), R1
	WORD $0x4c407800 // ld1.4s	{ v0 }, [x0]
	WORD $0x4c407821 // ld1.4s	{ v1 }, [x1]
	WORD $0x4e813802 // zip1.4s	v2, v0, v1
	WORD $0x4e817803 // zip2.4s	v3, v0, v1
	WORD $0x6e024044 // ext.16b	v4, v2, v2, #0x8
	WORD $0x6e034065 // ext.16b	v5, v3, v3, #0x8
	WORD $0x5280a524 // mov	w4, #0x529              ; =1321
	WORD $0x0e040c90 // dup.2s	v16, w4
	WORD $0x12802484 // mov	w4, #-0x125             ; =-293
	WORD $0x0e040c91 // dup.2s	v17, w4
	WORD $0x1280c9a4 // mov	w4, #-0x64e             ; =-1614
	WORD $0x0e040c92 // dup.2s	v18, w4
	WORD $0x12805de4 // mov	w4, #-0x2f0             ; =-752
	WORD $0x0e040c93 // dup.2s	v19, w4
	WORD $0x52801a24 // mov	w4, #0xd1               ; =209
	WORD $0x0e040c94 // dup.2s	v20, w4
	WORD $0x0eb0c046 // smull.2d	v6, v2, v16
	WORD $0x0eb18066 // smlal.2d	v6, v3, v17
	WORD $0x0eb280a6 // smlal.2d	v6, v5, v18
	WORD $0x0eb38086 // smlal.2d	v6, v4, v19
	WORD $0x4f7424c6 // srshr.2d	v6, v6, #0xc
	WORD $0x0ea58467 // add.2s	v7, v3, v5
	WORD $0x0ea484e7 // add.2s	v7, v7, v4
	WORD $0x0f20a4e7 // sshll.2d	v7, v7, #0x0
	WORD $0x4ee784c6 // add.2d	v6, v6, v7
	WORD $0x0eb2c048 // smull.2d	v8, v2, v18
	WORD $0x0eb0c069 // smull.2d	v9, v3, v16
	WORD $0x6ee98508 // sub.2d	v8, v8, v9
	WORD $0x0eb1c0a9 // smull.2d	v9, v5, v17
	WORD $0x6ee98508 // sub.2d	v8, v8, v9
	WORD $0x0eb38088 // smlal.2d	v8, v4, v19
	WORD $0x4f742508 // srshr.2d	v8, v8, #0xc
	WORD $0x2ea5844a // sub.2s	v10, v2, v5
	WORD $0x0ea4854a // add.2s	v10, v10, v4
	WORD $0x0f20a54a // sshll.2d	v10, v10, #0x0
	WORD $0x4eea8508 // add.2d	v8, v8, v10
	WORD $0x2ea3844b // sub.2s	v11, v2, v3
	WORD $0x0ea5856b // add.2s	v11, v11, v5
	WORD $0x0eb4c16b // smull.2d	v11, v11, v20
	WORD $0x4f78256b // srshr.2d	v11, v11, #0x8
	WORD $0x0eb1c04c // smull.2d	v12, v2, v17
	WORD $0x0eb2806c // smlal.2d	v12, v3, v18
	WORD $0x0eb0c0ad // smull.2d	v13, v5, v16
	WORD $0x6eed858c // sub.2d	v12, v12, v13
	WORD $0x0eb3c08d // smull.2d	v13, v4, v19
	WORD $0x6eed858c // sub.2d	v12, v12, v13
	WORD $0x4f74258c // srshr.2d	v12, v12, #0xc
	WORD $0x0ea3844e // add.2s	v14, v2, v3
	WORD $0x2ea485ce // sub.2s	v14, v14, v4
	WORD $0x0f20a5ce // sshll.2d	v14, v14, #0x0
	WORD $0x4eee858c // add.2d	v12, v12, v14
	WORD $0x0ea128c6 // xtn.2s	v6, v6
	WORD $0x0ea12908 // xtn.2s	v8, v8
	WORD $0x0ea1296b // xtn.2s	v11, v11
	WORD $0x0ea1298c // xtn.2s	v12, v12
	WORD $0x0d9f8006 // st1.s	{ v6 }[0], [x0], #4
	WORD $0x0d9f8008 // st1.s	{ v8 }[0], [x0], #4
	WORD $0x0d9f800b // st1.s	{ v11 }[0], [x0], #4
	WORD $0x0d00800c // st1.s	{ v12 }[0], [x0]
	WORD $0x0d9f9026 // st1.s	{ v6 }[1], [x1], #4
	WORD $0x0d9f9028 // st1.s	{ v8 }[1], [x1], #4
	WORD $0x0d9f902b // st1.s	{ v11 }[1], [x1], #4
	WORD $0x0d00902c // st1.s	{ v12 }[1], [x1]
	RET

// func inverseADST8Row2NEON(r0, r1 *int32, min, max int64)
TEXT ·inverseADST8Row2NEON(SB), NOSPLIT, $0-32
	MOVD r0+0(FP), R0
	MOVD r1+8(FP), R1
	MOVD min+16(FP), R2
	MOVD max+24(FP), R3
	WORD $0x4c40a800 // ld1.4s	{ v0, v1 }, [x0]
	WORD $0x4c40a822 // ld1.4s	{ v2, v3 }, [x1]
	WORD $0x4e823804 // zip1.4s	v4, v0, v2
	WORD $0x4e827805 // zip2.4s	v5, v0, v2
	WORD $0x4e833826 // zip1.4s	v6, v1, v3
	WORD $0x4e837827 // zip2.4s	v7, v1, v3
	WORD $0x6e044088 // ext.16b	v8, v4, v4, #0x8
	WORD $0x6e0540a9 // ext.16b	v9, v5, v5, #0x8
	WORD $0x6e0640ca // ext.16b	v10, v6, v6, #0x8
	WORD $0x6e0740eb // ext.16b	v11, v7, v7, #0x8
	WORD $0x4e080c5e // dup.2d	v30, x2
	WORD $0x4e080c7f // dup.2d	v31, x3
	WORD $0x12800264 // mov	w4, #-0x14              ; =-20
	WORD $0x0e040c94 // dup.2s	v20, w4
	WORD $0x52803224 // mov	w4, #0x191              ; =401
	WORD $0x0e040c95 // dup.2s	v21, w4
	WORD $0x0eb4c16c // smull.2d	v12, v11, v20
	WORD $0x0eb5808c // smlal.2d	v12, v4, v21
	WORD $0x4f74258c // srshr.2d	v12, v12, #0xc
	WORD $0x0f20a56d // sshll.2d	v13, v11, #0x0
	WORD $0x4eed858c // add.2d	v12, v12, v13
	WORD $0x0eb5c16e // smull.2d	v14, v11, v21
	WORD $0x0eb4c08f // smull.2d	v15, v4, v20
	WORD $0x6eef85ce // sub.2d	v14, v14, v15
	WORD $0x4f7425ce // srshr.2d	v14, v14, #0xc
	WORD $0x0f20a48f // sshll.2d	v15, v4, #0x0
	WORD $0x6eef85ce // sub.2d	v14, v14, v15
	WORD $0x12803c64 // mov	w4, #-0x1e4             ; =-484
	WORD $0x0e040c96 // dup.2s	v22, w4
	WORD $0x5280f164 // mov	w4, #0x78b              ; =1931
	WORD $0x0e040c97 // dup.2s	v23, w4
	WORD $0x0eb6c150 // smull.2d	v16, v10, v22
	WORD $0x0eb780b0 // smlal.2d	v16, v5, v23
	WORD $0x4f742610 // srshr.2d	v16, v16, #0xc
	WORD $0x0f20a551 // sshll.2d	v17, v10, #0x0
	WORD $0x4ef18610 // add.2d	v16, v16, v17
	WORD $0x0eb7c152 // smull.2d	v18, v10, v23
	WORD $0x0eb6c0b3 // smull.2d	v19, v5, v22
	WORD $0x6ef38652 // sub.2d	v18, v18, v19
	WORD $0x4f742652 // srshr.2d	v18, v18, #0xc
	WORD $0x0f20a4b3 // sshll.2d	v19, v5, #0x0
	WORD $0x6ef38652 // sub.2d	v18, v18, v19
	WORD $0x5280a264 // mov	w4, #0x513              ; =1299
	WORD $0x0e040c98 // dup.2s	v24, w4
	WORD $0x5280c5e4 // mov	w4, #0x62f              ; =1583
	WORD $0x0e040c99 // dup.2s	v25, w4
	WORD $0x0eb8c121 // smull.2d	v1, v9, v24
	WORD $0x0eb980c1 // smlal.2d	v1, v6, v25
	WORD $0x4f752421 // srshr.2d	v1, v1, #0xb
	WORD $0x0eb9c122 // smull.2d	v2, v9, v25
	WORD $0x0eb8c0c3 // smull.2d	v3, v6, v24
	WORD $0x6ee38442 // sub.2d	v2, v2, v3
	WORD $0x4f752442 // srshr.2d	v2, v2, #0xb
	WORD $0x528094a4 // mov	w4, #0x4a5              ; =1189
	WORD $0x0e040c9a // dup.2s	v26, w4
	WORD $0x128015e4 // mov	w4, #-0xb0              ; =-176
	WORD $0x0e040c9b // dup.2s	v27, w4
	WORD $0x0ebac103 // smull.2d	v3, v8, v26
	WORD $0x0ebb80e3 // smlal.2d	v3, v7, v27
	WORD $0x4f742463 // srshr.2d	v3, v3, #0xc
	WORD $0x0f20a4e0 // sshll.2d	v0, v7, #0x0
	WORD $0x4ee08463 // add.2d	v3, v3, v0
	WORD $0x0ebbc100 // smull.2d	v0, v8, v27
	WORD $0x0ebac0fc // smull.2d	v28, v7, v26
	WORD $0x6efc8400 // sub.2d	v0, v0, v28
	WORD $0x4f742400 // srshr.2d	v0, v0, #0xc
	WORD $0x0f20a51c // sshll.2d	v28, v8, #0x0
	WORD $0x4efc8400 // add.2d	v0, v0, v28
	WORD $0x4ee18584 // add.2d	v4, v12, v1
	WORD $0x4ee437dd // cmgt.2d	v29, v30, v4
	WORD $0x6e641fdd // bsl.16b	v29, v30, v4
	WORD $0x4eff37a4 // cmgt.2d	v4, v29, v31
	WORD $0x6e7d1fe4 // bsl.16b	v4, v31, v29
	WORD $0x4ee285c5 // add.2d	v5, v14, v2
	WORD $0x4ee537dd // cmgt.2d	v29, v30, v5
	WORD $0x6e651fdd // bsl.16b	v29, v30, v5
	WORD $0x4eff37a5 // cmgt.2d	v5, v29, v31
	WORD $0x6e7d1fe5 // bsl.16b	v5, v31, v29
	WORD $0x4ee38606 // add.2d	v6, v16, v3
	WORD $0x4ee637dd // cmgt.2d	v29, v30, v6
	WORD $0x6e661fdd // bsl.16b	v29, v30, v6
	WORD $0x4eff37a6 // cmgt.2d	v6, v29, v31
	WORD $0x6e7d1fe6 // bsl.16b	v6, v31, v29
	WORD $0x4ee08647 // add.2d	v7, v18, v0
	WORD $0x4ee737dd // cmgt.2d	v29, v30, v7
	WORD $0x6e671fdd // bsl.16b	v29, v30, v7
	WORD $0x4eff37a7 // cmgt.2d	v7, v29, v31
	WORD $0x6e7d1fe7 // bsl.16b	v7, v31, v29
	WORD $0x6ee18588 // sub.2d	v8, v12, v1
	WORD $0x4ee837dd // cmgt.2d	v29, v30, v8
	WORD $0x6e681fdd // bsl.16b	v29, v30, v8
	WORD $0x4eff37a8 // cmgt.2d	v8, v29, v31
	WORD $0x6e7d1fe8 // bsl.16b	v8, v31, v29
	WORD $0x6ee285c9 // sub.2d	v9, v14, v2
	WORD $0x4ee937dd // cmgt.2d	v29, v30, v9
	WORD $0x6e691fdd // bsl.16b	v29, v30, v9
	WORD $0x4eff37a9 // cmgt.2d	v9, v29, v31
	WORD $0x6e7d1fe9 // bsl.16b	v9, v31, v29
	WORD $0x6ee3860a // sub.2d	v10, v16, v3
	WORD $0x4eea37dd // cmgt.2d	v29, v30, v10
	WORD $0x6e6a1fdd // bsl.16b	v29, v30, v10
	WORD $0x4eff37aa // cmgt.2d	v10, v29, v31
	WORD $0x6e7d1fea // bsl.16b	v10, v31, v29
	WORD $0x6ee0864b // sub.2d	v11, v18, v0
	WORD $0x4eeb37dd // cmgt.2d	v29, v30, v11
	WORD $0x6e6b1fdd // bsl.16b	v29, v30, v11
	WORD $0x4eff37ab // cmgt.2d	v11, v29, v31
	WORD $0x6e7d1feb // bsl.16b	v11, v31, v29
	WORD $0x128026e4 // mov	w4, #-0x138             ; =-312
	WORD $0x0e040c94 // dup.2s	v20, w4
	WORD $0x5280c3e4 // mov	w4, #0x61f              ; =1567
	WORD $0x0e040c95 // dup.2s	v21, w4
	WORD $0x0ea1290c // xtn.2s	v12, v8
	WORD $0x0ea1292d // xtn.2s	v13, v9
	WORD $0x0ea1294e // xtn.2s	v14, v10
	WORD $0x0ea1296f // xtn.2s	v15, v11
	WORD $0x0eb4c190 // smull.2d	v16, v12, v20
	WORD $0x0eb581b0 // smlal.2d	v16, v13, v21
	WORD $0x4f742610 // srshr.2d	v16, v16, #0xc
	WORD $0x4ee88610 // add.2d	v16, v16, v8
	WORD $0x0eb5c191 // smull.2d	v17, v12, v21
	WORD $0x0eb4c1b2 // smull.2d	v18, v13, v20
	WORD $0x6ef28631 // sub.2d	v17, v17, v18
	WORD $0x4f742631 // srshr.2d	v17, v17, #0xc
	WORD $0x6ee98631 // sub.2d	v17, v17, v9
	WORD $0x0eb4c1f2 // smull.2d	v18, v15, v20
	WORD $0x0eb5c1d3 // smull.2d	v19, v14, v21
	WORD $0x6ef38652 // sub.2d	v18, v18, v19
	WORD $0x4f742652 // srshr.2d	v18, v18, #0xc
	WORD $0x4eeb8652 // add.2d	v18, v18, v11
	WORD $0x0eb5c1f3 // smull.2d	v19, v15, v21
	WORD $0x0eb481d3 // smlal.2d	v19, v14, v20
	WORD $0x4f742673 // srshr.2d	v19, v19, #0xc
	WORD $0x4eea8673 // add.2d	v19, v19, v10
	WORD $0x4ee68480 // add.2d	v0, v4, v6
	WORD $0x4ee037dd // cmgt.2d	v29, v30, v0
	WORD $0x6e601fdd // bsl.16b	v29, v30, v0
	WORD $0x4eff37a0 // cmgt.2d	v0, v29, v31
	WORD $0x6e7d1fe0 // bsl.16b	v0, v31, v29
	WORD $0x4ee784a1 // add.2d	v1, v5, v7
	WORD $0x4ee137dd // cmgt.2d	v29, v30, v1
	WORD $0x6e611fdd // bsl.16b	v29, v30, v1
	WORD $0x4eff37a1 // cmgt.2d	v1, v29, v31
	WORD $0x6e7d1fe1 // bsl.16b	v1, v31, v29
	WORD $0x6ee0b821 // neg.2d	v1, v1
	WORD $0x6ee68482 // sub.2d	v2, v4, v6
	WORD $0x4ee237dd // cmgt.2d	v29, v30, v2
	WORD $0x6e621fdd // bsl.16b	v29, v30, v2
	WORD $0x4eff37a2 // cmgt.2d	v2, v29, v31
	WORD $0x6e7d1fe2 // bsl.16b	v2, v31, v29
	WORD $0x6ee784a3 // sub.2d	v3, v5, v7
	WORD $0x4ee337dd // cmgt.2d	v29, v30, v3
	WORD $0x6e631fdd // bsl.16b	v29, v30, v3
	WORD $0x4eff37a3 // cmgt.2d	v3, v29, v31
	WORD $0x6e7d1fe3 // bsl.16b	v3, v31, v29
	WORD $0x4ef28604 // add.2d	v4, v16, v18
	WORD $0x4ee437dd // cmgt.2d	v29, v30, v4
	WORD $0x6e641fdd // bsl.16b	v29, v30, v4
	WORD $0x4eff37a4 // cmgt.2d	v4, v29, v31
	WORD $0x6e7d1fe4 // bsl.16b	v4, v31, v29
	WORD $0x6ee0b884 // neg.2d	v4, v4
	WORD $0x4ef38625 // add.2d	v5, v17, v19
	WORD $0x4ee537dd // cmgt.2d	v29, v30, v5
	WORD $0x6e651fdd // bsl.16b	v29, v30, v5
	WORD $0x4eff37a5 // cmgt.2d	v5, v29, v31
	WORD $0x6e7d1fe5 // bsl.16b	v5, v31, v29
	WORD $0x6ef28606 // sub.2d	v6, v16, v18
	WORD $0x4ee637dd // cmgt.2d	v29, v30, v6
	WORD $0x6e661fdd // bsl.16b	v29, v30, v6
	WORD $0x4eff37a6 // cmgt.2d	v6, v29, v31
	WORD $0x6e7d1fe6 // bsl.16b	v6, v31, v29
	WORD $0x6ef38627 // sub.2d	v7, v17, v19
	WORD $0x4ee737dd // cmgt.2d	v29, v30, v7
	WORD $0x6e671fdd // bsl.16b	v29, v30, v7
	WORD $0x4eff37a7 // cmgt.2d	v7, v29, v31
	WORD $0x6e7d1fe7 // bsl.16b	v7, v31, v29
	WORD $0x528016a4 // mov	w4, #0xb5               ; =181
	WORD $0x0e040c94 // dup.2s	v20, w4
	WORD $0x4ee38448 // add.2d	v8, v2, v3
	WORD $0x0ea12908 // xtn.2s	v8, v8
	WORD $0x0eb4c108 // smull.2d	v8, v8, v20
	WORD $0x4f782508 // srshr.2d	v8, v8, #0x8
	WORD $0x6ee0b908 // neg.2d	v8, v8
	WORD $0x6ee38449 // sub.2d	v9, v2, v3
	WORD $0x0ea12929 // xtn.2s	v9, v9
	WORD $0x0eb4c129 // smull.2d	v9, v9, v20
	WORD $0x4f782529 // srshr.2d	v9, v9, #0x8
	WORD $0x4ee784ca // add.2d	v10, v6, v7
	WORD $0x0ea1294a // xtn.2s	v10, v10
	WORD $0x0eb4c14a // smull.2d	v10, v10, v20
	WORD $0x4f78254a // srshr.2d	v10, v10, #0x8
	WORD $0x6ee784cb // sub.2d	v11, v6, v7
	WORD $0x0ea1296b // xtn.2s	v11, v11
	WORD $0x0eb4c16b // smull.2d	v11, v11, v20
	WORD $0x4f78256b // srshr.2d	v11, v11, #0x8
	WORD $0x6ee0b96b // neg.2d	v11, v11
	WORD $0x0ea12800 // xtn.2s	v0, v0
	WORD $0x0ea12884 // xtn.2s	v4, v4
	WORD $0x0ea1294a // xtn.2s	v10, v10
	WORD $0x0ea12908 // xtn.2s	v8, v8
	WORD $0x0ea12929 // xtn.2s	v9, v9
	WORD $0x0ea1296b // xtn.2s	v11, v11
	WORD $0x0ea128a5 // xtn.2s	v5, v5
	WORD $0x0ea12821 // xtn.2s	v1, v1
	WORD $0x0d9f8000 // st1.s	{ v0 }[0], [x0], #4
	WORD $0x0d9f8004 // st1.s	{ v4 }[0], [x0], #4
	WORD $0x0d9f800a // st1.s	{ v10 }[0], [x0], #4
	WORD $0x0d9f8008 // st1.s	{ v8 }[0], [x0], #4
	WORD $0x0d9f8009 // st1.s	{ v9 }[0], [x0], #4
	WORD $0x0d9f800b // st1.s	{ v11 }[0], [x0], #4
	WORD $0x0d9f8005 // st1.s	{ v5 }[0], [x0], #4
	WORD $0x0d008001 // st1.s	{ v1 }[0], [x0]
	WORD $0x0d9f9020 // st1.s	{ v0 }[1], [x1], #4
	WORD $0x0d9f9024 // st1.s	{ v4 }[1], [x1], #4
	WORD $0x0d9f902a // st1.s	{ v10 }[1], [x1], #4
	WORD $0x0d9f9028 // st1.s	{ v8 }[1], [x1], #4
	WORD $0x0d9f9029 // st1.s	{ v9 }[1], [x1], #4
	WORD $0x0d9f902b // st1.s	{ v11 }[1], [x1], #4
	WORD $0x0d9f9025 // st1.s	{ v5 }[1], [x1], #4
	WORD $0x0d009021 // st1.s	{ v1 }[1], [x1]
	RET

// func inverseDCT16Row2NEON(r0, r1 *int32, min, max int64)
// Uses a 128-byte local frame to spill the eight even-half results while the
// odd half is computed.
TEXT ·inverseDCT16Row2NEON(SB), NOSPLIT, $128-32
	MOVD r0+0(FP), R0
	MOVD r1+8(FP), R1
	MOVD min+16(FP), R2
	MOVD max+24(FP), R3
	WORD $0x4c402800 // ld1.4s	{ v0, v1, v2, v3 }, [x0]
	WORD $0x4c402824 // ld1.4s	{ v4, v5, v6, v7 }, [x1]
	WORD $0x4e843808 // zip1.4s	v8, v0, v4
	WORD $0x4e847809 // zip2.4s	v9, v0, v4
	WORD $0x4e85382a // zip1.4s	v10, v1, v5
	WORD $0x4e85782b // zip2.4s	v11, v1, v5
	WORD $0x4e86384c // zip1.4s	v12, v2, v6
	WORD $0x4e86784d // zip2.4s	v13, v2, v6
	WORD $0x4e87386e // zip1.4s	v14, v3, v7
	WORD $0x4e87786f // zip2.4s	v15, v3, v7
	WORD $0x4e080c5e // dup.2d	v30, x2
	WORD $0x4e080c7f // dup.2d	v31, x3
	WORD $0x528016a4 // mov	w4, #0xb5               ; =181
	WORD $0x0e040c90 // dup.2s	v16, w4
	WORD $0x0eac8511 // add.2s	v17, v8, v12
	WORD $0x0eb0c231 // smull.2d	v17, v17, v16
	WORD $0x4f782631 // srshr.2d	v17, v17, #0x8
	WORD $0x2eac8512 // sub.2s	v18, v8, v12
	WORD $0x0eb0c252 // smull.2d	v18, v18, v16
	WORD $0x4f782652 // srshr.2d	v18, v18, #0x8
	WORD $0x5280c3e4 // mov	w4, #0x61f              ; =1567
	WORD $0x0e040c93 // dup.2s	v19, w4
	WORD $0x128026e4 // mov	w4, #-0x138             ; =-312
	WORD $0x0e040c94 // dup.2s	v20, w4
	WORD $0x0eb3c155 // smull.2d	v21, v10, v19
	WORD $0x0eb4c1d6 // smull.2d	v22, v14, v20
	WORD $0x6ef686b5 // sub.2d	v21, v21, v22
	WORD $0x4f7426b5 // srshr.2d	v21, v21, #0xc
	WORD $0x0f20a5d6 // sshll.2d	v22, v14, #0x0
	WORD $0x6ef686b5 // sub.2d	v21, v21, v22
	WORD $0x0eb4c156 // smull.2d	v22, v10, v20
	WORD $0x0eb381d6 // smlal.2d	v22, v14, v19
	WORD $0x4f7426d6 // srshr.2d	v22, v22, #0xc
	WORD $0x0f20a557 // sshll.2d	v23, v10, #0x0
	WORD $0x4ef786d6 // add.2d	v22, v22, v23
	WORD $0x4ef68637 // add.2d	v23, v17, v22
	WORD $0x4ef737dd // cmgt.2d	v29, v30, v23
	WORD $0x6e771fdd // bsl.16b	v29, v30, v23
	WORD $0x4eff37b7 // cmgt.2d	v23, v29, v31
	WORD $0x6e7d1ff7 // bsl.16b	v23, v31, v29
	WORD $0x4ef58658 // add.2d	v24, v18, v21
	WORD $0x4ef837dd // cmgt.2d	v29, v30, v24
	WORD $0x6e781fdd // bsl.16b	v29, v30, v24
	WORD $0x4eff37b8 // cmgt.2d	v24, v29, v31
	WORD $0x6e7d1ff8 // bsl.16b	v24, v31, v29
	WORD $0x6ef58659 // sub.2d	v25, v18, v21
	WORD $0x4ef937dd // cmgt.2d	v29, v30, v25
	WORD $0x6e791fdd // bsl.16b	v29, v30, v25
	WORD $0x4eff37b9 // cmgt.2d	v25, v29, v31
	WORD $0x6e7d1ff9 // bsl.16b	v25, v31, v29
	WORD $0x6ef6863a // sub.2d	v26, v17, v22
	WORD $0x4efa37dd // cmgt.2d	v29, v30, v26
	WORD $0x6e7a1fdd // bsl.16b	v29, v30, v26
	WORD $0x4eff37ba // cmgt.2d	v26, v29, v31
	WORD $0x6e7d1ffa // bsl.16b	v26, v31, v29
	WORD $0x528063e4 // mov	w4, #0x31f              ; =799
	WORD $0x0e040c80 // dup.2s	v0, w4
	WORD $0x128009c4 // mov	w4, #-0x4f              ; =-79
	WORD $0x0e040c81 // dup.2s	v1, w4
	WORD $0x0ea0c122 // smull.2d	v2, v9, v0
	WORD $0x0ea1c1e3 // smull.2d	v3, v15, v1
	WORD $0x6ee38442 // sub.2d	v2, v2, v3
	WORD $0x4f742442 // srshr.2d	v2, v2, #0xc
	WORD $0x0f20a5e3 // sshll.2d	v3, v15, #0x0
	WORD $0x6ee38442 // sub.2d	v2, v2, v3
	WORD $0x5280d4e4 // mov	w4, #0x6a7              ; =1703
	WORD $0x0e040c84 // dup.2s	v4, w4
	WORD $0x52808e44 // mov	w4, #0x472              ; =1138
	WORD $0x0e040c85 // dup.2s	v5, w4
	WORD $0x0ea4c1a3 // smull.2d	v3, v13, v4
	WORD $0x0ea5c166 // smull.2d	v6, v11, v5
	WORD $0x6ee68463 // sub.2d	v3, v3, v6
	WORD $0x4f752463 // srshr.2d	v3, v3, #0xb
	WORD $0x0ea5c1a6 // smull.2d	v6, v13, v5
	WORD $0x0ea48166 // smlal.2d	v6, v11, v4
	WORD $0x4f7524c6 // srshr.2d	v6, v6, #0xb
	WORD $0x0ea1c127 // smull.2d	v7, v9, v1
	WORD $0x0ea081e7 // smlal.2d	v7, v15, v0
	WORD $0x4f7424e7 // srshr.2d	v7, v7, #0xc
	WORD $0x0f20a520 // sshll.2d	v0, v9, #0x0
	WORD $0x4ee084e7 // add.2d	v7, v7, v0
	WORD $0x4ee38440 // add.2d	v0, v2, v3
	WORD $0x4ee037dd // cmgt.2d	v29, v30, v0
	WORD $0x6e601fdd // bsl.16b	v29, v30, v0
	WORD $0x4eff37a0 // cmgt.2d	v0, v29, v31
	WORD $0x6e7d1fe0 // bsl.16b	v0, v31, v29
	WORD $0x6ee38441 // sub.2d	v1, v2, v3
	WORD $0x4ee137dd // cmgt.2d	v29, v30, v1
	WORD $0x6e611fdd // bsl.16b	v29, v30, v1
	WORD $0x4eff37a1 // cmgt.2d	v1, v29, v31
	WORD $0x6e7d1fe1 // bsl.16b	v1, v31, v29
	WORD $0x4ee684e2 // add.2d	v2, v7, v6
	WORD $0x4ee237dd // cmgt.2d	v29, v30, v2
	WORD $0x6e621fdd // bsl.16b	v29, v30, v2
	WORD $0x4eff37a2 // cmgt.2d	v2, v29, v31
	WORD $0x6e7d1fe2 // bsl.16b	v2, v31, v29
	WORD $0x6ee684e3 // sub.2d	v3, v7, v6
	WORD $0x4ee337dd // cmgt.2d	v29, v30, v3
	WORD $0x6e631fdd // bsl.16b	v29, v30, v3
	WORD $0x4eff37a3 // cmgt.2d	v3, v29, v31
	WORD $0x6e7d1fe3 // bsl.16b	v3, v31, v29
	WORD $0x528016a4 // mov	w4, #0xb5               ; =181
	WORD $0x0e040c90 // dup.2s	v16, w4
	WORD $0x6ee18466 // sub.2d	v6, v3, v1
	WORD $0x0ea128c6 // xtn.2s	v6, v6
	WORD $0x0eb0c0c6 // smull.2d	v6, v6, v16
	WORD $0x4f7824c6 // srshr.2d	v6, v6, #0x8
	WORD $0x4ee18467 // add.2d	v7, v3, v1
	WORD $0x0ea128e7 // xtn.2s	v7, v7
	WORD $0x0eb0c0e7 // smull.2d	v7, v7, v16
	WORD $0x4f7824e7 // srshr.2d	v7, v7, #0x8
	WORD $0x4ee286f0 // add.2d	v16, v23, v2
	WORD $0x4ef037dd // cmgt.2d	v29, v30, v16
	WORD $0x6e701fdd // bsl.16b	v29, v30, v16
	WORD $0x4eff37b0 // cmgt.2d	v16, v29, v31
	WORD $0x6e7d1ff0 // bsl.16b	v16, v31, v29
	WORD $0x3d8003f0 // str	q16, [sp]
	WORD $0x4ee78710 // add.2d	v16, v24, v7
	WORD $0x4ef037dd // cmgt.2d	v29, v30, v16
	WORD $0x6e701fdd // bsl.16b	v29, v30, v16
	WORD $0x4eff37b0 // cmgt.2d	v16, v29, v31
	WORD $0x6e7d1ff0 // bsl.16b	v16, v31, v29
	WORD $0x3d8007f0 // str	q16, [sp, #0x10]
	WORD $0x4ee68730 // add.2d	v16, v25, v6
	WORD $0x4ef037dd // cmgt.2d	v29, v30, v16
	WORD $0x6e701fdd // bsl.16b	v29, v30, v16
	WORD $0x4eff37b0 // cmgt.2d	v16, v29, v31
	WORD $0x6e7d1ff0 // bsl.16b	v16, v31, v29
	WORD $0x3d800bf0 // str	q16, [sp, #0x20]
	WORD $0x4ee08750 // add.2d	v16, v26, v0
	WORD $0x4ef037dd // cmgt.2d	v29, v30, v16
	WORD $0x6e701fdd // bsl.16b	v29, v30, v16
	WORD $0x4eff37b0 // cmgt.2d	v16, v29, v31
	WORD $0x6e7d1ff0 // bsl.16b	v16, v31, v29
	WORD $0x3d800ff0 // str	q16, [sp, #0x30]
	WORD $0x6ee08750 // sub.2d	v16, v26, v0
	WORD $0x4ef037dd // cmgt.2d	v29, v30, v16
	WORD $0x6e701fdd // bsl.16b	v29, v30, v16
	WORD $0x4eff37b0 // cmgt.2d	v16, v29, v31
	WORD $0x6e7d1ff0 // bsl.16b	v16, v31, v29
	WORD $0x3d8013f0 // str	q16, [sp, #0x40]
	WORD $0x6ee68730 // sub.2d	v16, v25, v6
	WORD $0x4ef037dd // cmgt.2d	v29, v30, v16
	WORD $0x6e701fdd // bsl.16b	v29, v30, v16
	WORD $0x4eff37b0 // cmgt.2d	v16, v29, v31
	WORD $0x6e7d1ff0 // bsl.16b	v16, v31, v29
	WORD $0x3d8017f0 // str	q16, [sp, #0x50]
	WORD $0x6ee78710 // sub.2d	v16, v24, v7
	WORD $0x4ef037dd // cmgt.2d	v29, v30, v16
	WORD $0x6e701fdd // bsl.16b	v29, v30, v16
	WORD $0x4eff37b0 // cmgt.2d	v16, v29, v31
	WORD $0x6e7d1ff0 // bsl.16b	v16, v31, v29
	WORD $0x3d801bf0 // str	q16, [sp, #0x60]
	WORD $0x6ee286f0 // sub.2d	v16, v23, v2
	WORD $0x4ef037dd // cmgt.2d	v29, v30, v16
	WORD $0x6e701fdd // bsl.16b	v29, v30, v16
	WORD $0x4eff37b0 // cmgt.2d	v16, v29, v31
	WORD $0x6e7d1ff0 // bsl.16b	v16, v31, v29
	WORD $0x3d801ff0 // str	q16, [sp, #0x70]
	WORD $0x6e084108 // ext.16b	v8, v8, v8, #0x8
	WORD $0x6e094129 // ext.16b	v9, v9, v9, #0x8
	WORD $0x6e0a414a // ext.16b	v10, v10, v10, #0x8
	WORD $0x6e0b416b // ext.16b	v11, v11, v11, #0x8
	WORD $0x6e0c418c // ext.16b	v12, v12, v12, #0x8
	WORD $0x6e0d41ad // ext.16b	v13, v13, v13, #0x8
	WORD $0x6e0e41ce // ext.16b	v14, v14, v14, #0x8
	WORD $0x6e0f41ef // ext.16b	v15, v15, v15, #0x8
	WORD $0x52803224 // mov	w4, #0x191              ; =401
	WORD $0x0e040c90 // dup.2s	v16, w4
	WORD $0x12800264 // mov	w4, #-0x14              ; =-20
	WORD $0x0e040c91 // dup.2s	v17, w4
	WORD $0x0eb0c112 // smull.2d	v18, v8, v16
	WORD $0x0eb1c1f3 // smull.2d	v19, v15, v17
	WORD $0x6ef38652 // sub.2d	v18, v18, v19
	WORD $0x4f742652 // srshr.2d	v18, v18, #0xc
	WORD $0x0f20a5f3 // sshll.2d	v19, v15, #0x0
	WORD $0x6ef38652 // sub.2d	v18, v18, v19
	WORD $0x5280c5e4 // mov	w4, #0x62f              ; =1583
	WORD $0x0e040c80 // dup.2s	v0, w4
	WORD $0x5280a264 // mov	w4, #0x513              ; =1299
	WORD $0x0e040c81 // dup.2s	v1, w4
	WORD $0x0ea0c193 // smull.2d	v19, v12, v0
	WORD $0x0ea1c174 // smull.2d	v20, v11, v1
	WORD $0x6ef48673 // sub.2d	v19, v19, v20
	WORD $0x4f752673 // srshr.2d	v19, v19, #0xb
	WORD $0x5280f164 // mov	w4, #0x78b              ; =1931
	WORD $0x0e040c82 // dup.2s	v2, w4
	WORD $0x12803c64 // mov	w4, #-0x1e4             ; =-484
	WORD $0x0e040c83 // dup.2s	v3, w4
	WORD $0x0ea2c154 // smull.2d	v20, v10, v2
	WORD $0x0ea3c1b5 // smull.2d	v21, v13, v3
	WORD $0x6ef58694 // sub.2d	v20, v20, v21
	WORD $0x4f742694 // srshr.2d	v20, v20, #0xc
	WORD $0x0f20a5b5 // sshll.2d	v21, v13, #0x0
	WORD $0x6ef58694 // sub.2d	v20, v20, v21
	WORD $0x128015e4 // mov	w4, #-0xb0              ; =-176
	WORD $0x0e040c84 // dup.2s	v4, w4
	WORD $0x528094a4 // mov	w4, #0x4a5              ; =1189
	WORD $0x0e040c85 // dup.2s	v5, w4
	WORD $0x0ea4c1d5 // smull.2d	v21, v14, v4
	WORD $0x0ea5c136 // smull.2d	v22, v9, v5
	WORD $0x6ef686b5 // sub.2d	v21, v21, v22
	WORD $0x4f7426b5 // srshr.2d	v21, v21, #0xc
	WORD $0x0f20a5d6 // sshll.2d	v22, v14, #0x0
	WORD $0x4ef686b5 // add.2d	v21, v21, v22
	WORD $0x0ea5c1d6 // smull.2d	v22, v14, v5
	WORD $0x0ea48136 // smlal.2d	v22, v9, v4
	WORD $0x4f7426d6 // srshr.2d	v22, v22, #0xc
	WORD $0x0f20a537 // sshll.2d	v23, v9, #0x0
	WORD $0x4ef786d6 // add.2d	v22, v22, v23
	WORD $0x0ea3c157 // smull.2d	v23, v10, v3
	WORD $0x0ea281b7 // smlal.2d	v23, v13, v2
	WORD $0x4f7426f7 // srshr.2d	v23, v23, #0xc
	WORD $0x0f20a558 // sshll.2d	v24, v10, #0x0
	WORD $0x4ef886f7 // add.2d	v23, v23, v24
	WORD $0x0ea1c198 // smull.2d	v24, v12, v1
	WORD $0x0ea08178 // smlal.2d	v24, v11, v0
	WORD $0x4f752718 // srshr.2d	v24, v24, #0xb
	WORD $0x0eb1c119 // smull.2d	v25, v8, v17
	WORD $0x0eb081f9 // smlal.2d	v25, v15, v16
	WORD $0x4f742739 // srshr.2d	v25, v25, #0xc
	WORD $0x0f20a51a // sshll.2d	v26, v8, #0x0
	WORD $0x4efa8739 // add.2d	v25, v25, v26
	WORD $0x4ef38640 // add.2d	v0, v18, v19
	WORD $0x4ee037dd // cmgt.2d	v29, v30, v0
	WORD $0x6e601fdd // bsl.16b	v29, v30, v0
	WORD $0x4eff37a0 // cmgt.2d	v0, v29, v31
	WORD $0x6e7d1fe0 // bsl.16b	v0, v31, v29
	WORD $0x6ef38641 // sub.2d	v1, v18, v19
	WORD $0x4ee137dd // cmgt.2d	v29, v30, v1
	WORD $0x6e611fdd // bsl.16b	v29, v30, v1
	WORD $0x4eff37a1 // cmgt.2d	v1, v29, v31
	WORD $0x6e7d1fe1 // bsl.16b	v1, v31, v29
	WORD $0x6ef486a2 // sub.2d	v2, v21, v20
	WORD $0x4ee237dd // cmgt.2d	v29, v30, v2
	WORD $0x6e621fdd // bsl.16b	v29, v30, v2
	WORD $0x4eff37a2 // cmgt.2d	v2, v29, v31
	WORD $0x6e7d1fe2 // bsl.16b	v2, v31, v29
	WORD $0x4ef486a3 // add.2d	v3, v21, v20
	WORD $0x4ee337dd // cmgt.2d	v29, v30, v3
	WORD $0x6e631fdd // bsl.16b	v29, v30, v3
	WORD $0x4eff37a3 // cmgt.2d	v3, v29, v31
	WORD $0x6e7d1fe3 // bsl.16b	v3, v31, v29
	WORD $0x4ef786c4 // add.2d	v4, v22, v23
	WORD $0x4ee437dd // cmgt.2d	v29, v30, v4
	WORD $0x6e641fdd // bsl.16b	v29, v30, v4
	WORD $0x4eff37a4 // cmgt.2d	v4, v29, v31
	WORD $0x6e7d1fe4 // bsl.16b	v4, v31, v29
	WORD $0x6ef786c5 // sub.2d	v5, v22, v23
	WORD $0x4ee537dd // cmgt.2d	v29, v30, v5
	WORD $0x6e651fdd // bsl.16b	v29, v30, v5
	WORD $0x4eff37a5 // cmgt.2d	v5, v29, v31
	WORD $0x6e7d1fe5 // bsl.16b	v5, v31, v29
	WORD $0x6ef88726 // sub.2d	v6, v25, v24
	WORD $0x4ee637dd // cmgt.2d	v29, v30, v6
	WORD $0x6e661fdd // bsl.16b	v29, v30, v6
	WORD $0x4eff37a6 // cmgt.2d	v6, v29, v31
	WORD $0x6e7d1fe6 // bsl.16b	v6, v31, v29
	WORD $0x4ef88727 // add.2d	v7, v25, v24
	WORD $0x4ee737dd // cmgt.2d	v29, v30, v7
	WORD $0x6e671fdd // bsl.16b	v29, v30, v7
	WORD $0x4eff37a7 // cmgt.2d	v7, v29, v31
	WORD $0x6e7d1fe7 // bsl.16b	v7, v31, v29
	WORD $0x5280c3e4 // mov	w4, #0x61f              ; =1567
	WORD $0x0e040c90 // dup.2s	v16, w4
	WORD $0x128026e4 // mov	w4, #-0x138             ; =-312
	WORD $0x0e040c91 // dup.2s	v17, w4
	WORD $0x0ea12832 // xtn.2s	v18, v1
	WORD $0x0ea12853 // xtn.2s	v19, v2
	WORD $0x0ea128b4 // xtn.2s	v20, v5
	WORD $0x0ea128d5 // xtn.2s	v21, v6
	WORD $0x0eb0c2b6 // smull.2d	v22, v21, v16
	WORD $0x0eb1c257 // smull.2d	v23, v18, v17
	WORD $0x6ef786d6 // sub.2d	v22, v22, v23
	WORD $0x4f7426d6 // srshr.2d	v22, v22, #0xc
	WORD $0x6ee186d6 // sub.2d	v22, v22, v1
	WORD $0x0eb1c2b7 // smull.2d	v23, v21, v17
	WORD $0x0eb08257 // smlal.2d	v23, v18, v16
	WORD $0x4f7426f7 // srshr.2d	v23, v23, #0xc
	WORD $0x4ee686f7 // add.2d	v23, v23, v6
	WORD $0x0eb1c298 // smull.2d	v24, v20, v17
	WORD $0x0eb08278 // smlal.2d	v24, v19, v16
	WORD $0x6ee0bb18 // neg.2d	v24, v24
	WORD $0x4f742718 // srshr.2d	v24, v24, #0xc
	WORD $0x6ee58718 // sub.2d	v24, v24, v5
	WORD $0x0eb0c299 // smull.2d	v25, v20, v16
	WORD $0x0eb1c27a // smull.2d	v26, v19, v17
	WORD $0x6efa8739 // sub.2d	v25, v25, v26
	WORD $0x4f742739 // srshr.2d	v25, v25, #0xc
	WORD $0x6ee28739 // sub.2d	v25, v25, v2
	WORD $0x4ee38410 // add.2d	v16, v0, v3
	WORD $0x4ef037dd // cmgt.2d	v29, v30, v16
	WORD $0x6e701fdd // bsl.16b	v29, v30, v16
	WORD $0x4eff37b0 // cmgt.2d	v16, v29, v31
	WORD $0x6e7d1ff0 // bsl.16b	v16, v31, v29
	WORD $0x4ef886d1 // add.2d	v17, v22, v24
	WORD $0x4ef137dd // cmgt.2d	v29, v30, v17
	WORD $0x6e711fdd // bsl.16b	v29, v30, v17
	WORD $0x4eff37b1 // cmgt.2d	v17, v29, v31
	WORD $0x6e7d1ff1 // bsl.16b	v17, v31, v29
	WORD $0x6ef886d2 // sub.2d	v18, v22, v24
	WORD $0x4ef237dd // cmgt.2d	v29, v30, v18
	WORD $0x6e721fdd // bsl.16b	v29, v30, v18
	WORD $0x4eff37b2 // cmgt.2d	v18, v29, v31
	WORD $0x6e7d1ff2 // bsl.16b	v18, v31, v29
	WORD $0x6ee38413 // sub.2d	v19, v0, v3
	WORD $0x4ef337dd // cmgt.2d	v29, v30, v19
	WORD $0x6e731fdd // bsl.16b	v29, v30, v19
	WORD $0x4eff37b3 // cmgt.2d	v19, v29, v31
	WORD $0x6e7d1ff3 // bsl.16b	v19, v31, v29
	WORD $0x6ee484f4 // sub.2d	v20, v7, v4
	WORD $0x4ef437dd // cmgt.2d	v29, v30, v20
	WORD $0x6e741fdd // bsl.16b	v29, v30, v20
	WORD $0x4eff37b4 // cmgt.2d	v20, v29, v31
	WORD $0x6e7d1ff4 // bsl.16b	v20, v31, v29
	WORD $0x6ef986f5 // sub.2d	v21, v23, v25
	WORD $0x4ef537dd // cmgt.2d	v29, v30, v21
	WORD $0x6e751fdd // bsl.16b	v29, v30, v21
	WORD $0x4eff37b5 // cmgt.2d	v21, v29, v31
	WORD $0x6e7d1ff5 // bsl.16b	v21, v31, v29
	WORD $0x4ef986fa // add.2d	v26, v23, v25
	WORD $0x4efa37dd // cmgt.2d	v29, v30, v26
	WORD $0x6e7a1fdd // bsl.16b	v29, v30, v26
	WORD $0x4eff37ba // cmgt.2d	v26, v29, v31
	WORD $0x6e7d1ffa // bsl.16b	v26, v31, v29
	WORD $0x4ee484fb // add.2d	v27, v7, v4
	WORD $0x4efb37dd // cmgt.2d	v29, v30, v27
	WORD $0x6e7b1fdd // bsl.16b	v29, v30, v27
	WORD $0x4eff37bb // cmgt.2d	v27, v29, v31
	WORD $0x6e7d1ffb // bsl.16b	v27, v31, v29
	WORD $0x528016a4 // mov	w4, #0xb5               ; =181
	WORD $0x0e040c9c // dup.2s	v28, w4
	WORD $0x6ef286b6 // sub.2d	v22, v21, v18
	WORD $0x0ea12ad6 // xtn.2s	v22, v22
	WORD $0x0ebcc2d6 // smull.2d	v22, v22, v28
	WORD $0x4f7826d6 // srshr.2d	v22, v22, #0x8
	WORD $0x4ef286b9 // add.2d	v25, v21, v18
	WORD $0x0ea12b39 // xtn.2s	v25, v25
	WORD $0x0ebcc339 // smull.2d	v25, v25, v28
	WORD $0x4f782739 // srshr.2d	v25, v25, #0x8
	WORD $0x6ef38697 // sub.2d	v23, v20, v19
	WORD $0x0ea12af7 // xtn.2s	v23, v23
	WORD $0x0ebcc2f7 // smull.2d	v23, v23, v28
	WORD $0x4f7826f7 // srshr.2d	v23, v23, #0x8
	WORD $0x4ef38698 // add.2d	v24, v20, v19
	WORD $0x0ea12b18 // xtn.2s	v24, v24
	WORD $0x0ebcc318 // smull.2d	v24, v24, v28
	WORD $0x4f782718 // srshr.2d	v24, v24, #0x8
	WORD $0x3dc003e0 // ldr	q0, [sp]
	WORD $0x4efb8401 // add.2d	v1, v0, v27
	WORD $0x4ee137dd // cmgt.2d	v29, v30, v1
	WORD $0x6e611fdd // bsl.16b	v29, v30, v1
	WORD $0x4eff37a1 // cmgt.2d	v1, v29, v31
	WORD $0x6e7d1fe1 // bsl.16b	v1, v31, v29
	WORD $0x6efb8402 // sub.2d	v2, v0, v27
	WORD $0x4ee237dd // cmgt.2d	v29, v30, v2
	WORD $0x6e621fdd // bsl.16b	v29, v30, v2
	WORD $0x4eff37a2 // cmgt.2d	v2, v29, v31
	WORD $0x6e7d1fe2 // bsl.16b	v2, v31, v29
	WORD $0x0ea12821 // xtn.2s	v1, v1
	WORD $0x0ea12842 // xtn.2s	v2, v2
	WORD $0x0d008001 // st1.s	{ v1 }[0], [x0]
	WORD $0x9100f005 // add	x5, x0, #0x3c
	WORD $0x0d0080a2 // st1.s	{ v2 }[0], [x5]
	WORD $0x0d009021 // st1.s	{ v1 }[1], [x1]
	WORD $0x9100f026 // add	x6, x1, #0x3c
	WORD $0x0d0090c2 // st1.s	{ v2 }[1], [x6]
	WORD $0x3dc007e0 // ldr	q0, [sp, #0x10]
	WORD $0x4efa8401 // add.2d	v1, v0, v26
	WORD $0x4ee137dd // cmgt.2d	v29, v30, v1
	WORD $0x6e611fdd // bsl.16b	v29, v30, v1
	WORD $0x4eff37a1 // cmgt.2d	v1, v29, v31
	WORD $0x6e7d1fe1 // bsl.16b	v1, v31, v29
	WORD $0x6efa8402 // sub.2d	v2, v0, v26
	WORD $0x4ee237dd // cmgt.2d	v29, v30, v2
	WORD $0x6e621fdd // bsl.16b	v29, v30, v2
	WORD $0x4eff37a2 // cmgt.2d	v2, v29, v31
	WORD $0x6e7d1fe2 // bsl.16b	v2, v31, v29
	WORD $0x0ea12821 // xtn.2s	v1, v1
	WORD $0x0ea12842 // xtn.2s	v2, v2
	WORD $0x91001005 // add	x5, x0, #0x4
	WORD $0x0d0080a1 // st1.s	{ v1 }[0], [x5]
	WORD $0x9100e005 // add	x5, x0, #0x38
	WORD $0x0d0080a2 // st1.s	{ v2 }[0], [x5]
	WORD $0x91001026 // add	x6, x1, #0x4
	WORD $0x0d0090c1 // st1.s	{ v1 }[1], [x6]
	WORD $0x9100e026 // add	x6, x1, #0x38
	WORD $0x0d0090c2 // st1.s	{ v2 }[1], [x6]
	WORD $0x3dc00be0 // ldr	q0, [sp, #0x20]
	WORD $0x4ef98401 // add.2d	v1, v0, v25
	WORD $0x4ee137dd // cmgt.2d	v29, v30, v1
	WORD $0x6e611fdd // bsl.16b	v29, v30, v1
	WORD $0x4eff37a1 // cmgt.2d	v1, v29, v31
	WORD $0x6e7d1fe1 // bsl.16b	v1, v31, v29
	WORD $0x6ef98402 // sub.2d	v2, v0, v25
	WORD $0x4ee237dd // cmgt.2d	v29, v30, v2
	WORD $0x6e621fdd // bsl.16b	v29, v30, v2
	WORD $0x4eff37a2 // cmgt.2d	v2, v29, v31
	WORD $0x6e7d1fe2 // bsl.16b	v2, v31, v29
	WORD $0x0ea12821 // xtn.2s	v1, v1
	WORD $0x0ea12842 // xtn.2s	v2, v2
	WORD $0x91002005 // add	x5, x0, #0x8
	WORD $0x0d0080a1 // st1.s	{ v1 }[0], [x5]
	WORD $0x9100d005 // add	x5, x0, #0x34
	WORD $0x0d0080a2 // st1.s	{ v2 }[0], [x5]
	WORD $0x91002026 // add	x6, x1, #0x8
	WORD $0x0d0090c1 // st1.s	{ v1 }[1], [x6]
	WORD $0x9100d026 // add	x6, x1, #0x34
	WORD $0x0d0090c2 // st1.s	{ v2 }[1], [x6]
	WORD $0x3dc00fe0 // ldr	q0, [sp, #0x30]
	WORD $0x4ef88401 // add.2d	v1, v0, v24
	WORD $0x4ee137dd // cmgt.2d	v29, v30, v1
	WORD $0x6e611fdd // bsl.16b	v29, v30, v1
	WORD $0x4eff37a1 // cmgt.2d	v1, v29, v31
	WORD $0x6e7d1fe1 // bsl.16b	v1, v31, v29
	WORD $0x6ef88402 // sub.2d	v2, v0, v24
	WORD $0x4ee237dd // cmgt.2d	v29, v30, v2
	WORD $0x6e621fdd // bsl.16b	v29, v30, v2
	WORD $0x4eff37a2 // cmgt.2d	v2, v29, v31
	WORD $0x6e7d1fe2 // bsl.16b	v2, v31, v29
	WORD $0x0ea12821 // xtn.2s	v1, v1
	WORD $0x0ea12842 // xtn.2s	v2, v2
	WORD $0x91003005 // add	x5, x0, #0xc
	WORD $0x0d0080a1 // st1.s	{ v1 }[0], [x5]
	WORD $0x9100c005 // add	x5, x0, #0x30
	WORD $0x0d0080a2 // st1.s	{ v2 }[0], [x5]
	WORD $0x91003026 // add	x6, x1, #0xc
	WORD $0x0d0090c1 // st1.s	{ v1 }[1], [x6]
	WORD $0x9100c026 // add	x6, x1, #0x30
	WORD $0x0d0090c2 // st1.s	{ v2 }[1], [x6]
	WORD $0x3dc013e0 // ldr	q0, [sp, #0x40]
	WORD $0x4ef78401 // add.2d	v1, v0, v23
	WORD $0x4ee137dd // cmgt.2d	v29, v30, v1
	WORD $0x6e611fdd // bsl.16b	v29, v30, v1
	WORD $0x4eff37a1 // cmgt.2d	v1, v29, v31
	WORD $0x6e7d1fe1 // bsl.16b	v1, v31, v29
	WORD $0x6ef78402 // sub.2d	v2, v0, v23
	WORD $0x4ee237dd // cmgt.2d	v29, v30, v2
	WORD $0x6e621fdd // bsl.16b	v29, v30, v2
	WORD $0x4eff37a2 // cmgt.2d	v2, v29, v31
	WORD $0x6e7d1fe2 // bsl.16b	v2, v31, v29
	WORD $0x0ea12821 // xtn.2s	v1, v1
	WORD $0x0ea12842 // xtn.2s	v2, v2
	WORD $0x91004005 // add	x5, x0, #0x10
	WORD $0x0d0080a1 // st1.s	{ v1 }[0], [x5]
	WORD $0x9100b005 // add	x5, x0, #0x2c
	WORD $0x0d0080a2 // st1.s	{ v2 }[0], [x5]
	WORD $0x91004026 // add	x6, x1, #0x10
	WORD $0x0d0090c1 // st1.s	{ v1 }[1], [x6]
	WORD $0x9100b026 // add	x6, x1, #0x2c
	WORD $0x0d0090c2 // st1.s	{ v2 }[1], [x6]
	WORD $0x3dc017e0 // ldr	q0, [sp, #0x50]
	WORD $0x4ef68401 // add.2d	v1, v0, v22
	WORD $0x4ee137dd // cmgt.2d	v29, v30, v1
	WORD $0x6e611fdd // bsl.16b	v29, v30, v1
	WORD $0x4eff37a1 // cmgt.2d	v1, v29, v31
	WORD $0x6e7d1fe1 // bsl.16b	v1, v31, v29
	WORD $0x6ef68402 // sub.2d	v2, v0, v22
	WORD $0x4ee237dd // cmgt.2d	v29, v30, v2
	WORD $0x6e621fdd // bsl.16b	v29, v30, v2
	WORD $0x4eff37a2 // cmgt.2d	v2, v29, v31
	WORD $0x6e7d1fe2 // bsl.16b	v2, v31, v29
	WORD $0x0ea12821 // xtn.2s	v1, v1
	WORD $0x0ea12842 // xtn.2s	v2, v2
	WORD $0x91005005 // add	x5, x0, #0x14
	WORD $0x0d0080a1 // st1.s	{ v1 }[0], [x5]
	WORD $0x9100a005 // add	x5, x0, #0x28
	WORD $0x0d0080a2 // st1.s	{ v2 }[0], [x5]
	WORD $0x91005026 // add	x6, x1, #0x14
	WORD $0x0d0090c1 // st1.s	{ v1 }[1], [x6]
	WORD $0x9100a026 // add	x6, x1, #0x28
	WORD $0x0d0090c2 // st1.s	{ v2 }[1], [x6]
	WORD $0x3dc01be0 // ldr	q0, [sp, #0x60]
	WORD $0x4ef18401 // add.2d	v1, v0, v17
	WORD $0x4ee137dd // cmgt.2d	v29, v30, v1
	WORD $0x6e611fdd // bsl.16b	v29, v30, v1
	WORD $0x4eff37a1 // cmgt.2d	v1, v29, v31
	WORD $0x6e7d1fe1 // bsl.16b	v1, v31, v29
	WORD $0x6ef18402 // sub.2d	v2, v0, v17
	WORD $0x4ee237dd // cmgt.2d	v29, v30, v2
	WORD $0x6e621fdd // bsl.16b	v29, v30, v2
	WORD $0x4eff37a2 // cmgt.2d	v2, v29, v31
	WORD $0x6e7d1fe2 // bsl.16b	v2, v31, v29
	WORD $0x0ea12821 // xtn.2s	v1, v1
	WORD $0x0ea12842 // xtn.2s	v2, v2
	WORD $0x91006005 // add	x5, x0, #0x18
	WORD $0x0d0080a1 // st1.s	{ v1 }[0], [x5]
	WORD $0x91009005 // add	x5, x0, #0x24
	WORD $0x0d0080a2 // st1.s	{ v2 }[0], [x5]
	WORD $0x91006026 // add	x6, x1, #0x18
	WORD $0x0d0090c1 // st1.s	{ v1 }[1], [x6]
	WORD $0x91009026 // add	x6, x1, #0x24
	WORD $0x0d0090c2 // st1.s	{ v2 }[1], [x6]
	WORD $0x3dc01fe0 // ldr	q0, [sp, #0x70]
	WORD $0x4ef08401 // add.2d	v1, v0, v16
	WORD $0x4ee137dd // cmgt.2d	v29, v30, v1
	WORD $0x6e611fdd // bsl.16b	v29, v30, v1
	WORD $0x4eff37a1 // cmgt.2d	v1, v29, v31
	WORD $0x6e7d1fe1 // bsl.16b	v1, v31, v29
	WORD $0x6ef08402 // sub.2d	v2, v0, v16
	WORD $0x4ee237dd // cmgt.2d	v29, v30, v2
	WORD $0x6e621fdd // bsl.16b	v29, v30, v2
	WORD $0x4eff37a2 // cmgt.2d	v2, v29, v31
	WORD $0x6e7d1fe2 // bsl.16b	v2, v31, v29
	WORD $0x0ea12821 // xtn.2s	v1, v1
	WORD $0x0ea12842 // xtn.2s	v2, v2
	WORD $0x91007005 // add	x5, x0, #0x1c
	WORD $0x0d0080a1 // st1.s	{ v1 }[0], [x5]
	WORD $0x91008005 // add	x5, x0, #0x20
	WORD $0x0d0080a2 // st1.s	{ v2 }[0], [x5]
	WORD $0x91007026 // add	x6, x1, #0x1c
	WORD $0x0d0090c1 // st1.s	{ v1 }[1], [x6]
	WORD $0x91008026 // add	x6, x1, #0x20
	WORD $0x0d0090c2 // st1.s	{ v2 }[1], [x6]
	RET
