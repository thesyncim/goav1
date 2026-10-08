// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// The 10-bit filter8 horizontal NEON kernel remains here. HBD6 and HBD14
// horizontal filtering use Go SIMD or scalar Go. Vertical gather/scatter
// assembly shared by the SIMD kernels lives in the separate vtrn file.

//go:build arm64 && !purego

#include "textflag.h"

// ---- filter8 16-bit context offsets ----
#define F8_P3 0
#define F8_P2 8
#define F8_P1 16
#define F8_P0 24
#define F8_Q0 32
#define F8_Q1 40
#define F8_Q2 48
#define F8_Q3 56
#define F8_COUNT 64
#define F8_LIMIT 72
#define F8_BLIMIT 80
#define F8_HEV 88
#define F8_THR 96
#define F8_CENTER 104
#define F8_MIN 112
#define F8_MAX 120

TEXT ·filter8Edge16NEONAsm(SB), NOSPLIT, $0-8
	MOVD ctx+0(FP), R0
	MOVD F8_P3(R0), R9
	MOVD F8_P2(R0), R10
	MOVD F8_P1(R0), R11
	MOVD F8_P0(R0), R12
	MOVD F8_Q0(R0), R13
	MOVD F8_Q1(R0), R14
	MOVD F8_Q2(R0), R15
	MOVD F8_Q3(R0), R16
	MOVD F8_COUNT(R0), R17

	MOVD F8_LIMIT(R0), R6
	WORD $0x4e020cc0 // dup v0.8h, w6
	MOVD F8_BLIMIT(R0), R7
	WORD $0x4e020ce1 // dup v1.8h, w7
	MOVD F8_HEV(R0), R8
	WORD $0x4e020d02 // dup v2.8h, w8
	MOVD F8_THR(R0), R5
	WORD $0x4e020ca3 // dup v3.8h, w5
	MOVD F8_CENTER(R0), R6
	WORD $0x4e020cc4 // dup v4.8h, w6
	MOVD F8_MIN(R0), R6
	WORD $0x4e020cc5 // dup v5.8h, w6
	MOVD F8_MAX(R0), R6
	WORD $0x4e020cc6 // dup v6.8h, w6
	WORD $0x4f008467 // movi v7.8h, #3

loop8:
	CBZ R17, done8
	WORD $0x4c407530 // ld1 {v16.8h}, [x9]
	WORD $0x4c407551 // ld1 {v17.8h}, [x10]
	WORD $0x4c407572 // ld1 {v18.8h}, [x11]
	WORD $0x4c407593 // ld1 {v19.8h}, [x12]
	WORD $0x4c4075b4 // ld1 {v20.8h}, [x13]
	WORD $0x4c4075d5 // ld1 {v21.8h}, [x14]
	WORD $0x4c4075f6 // ld1 {v22.8h}, [x15]
	WORD $0x4c407617 // ld1 {v23.8h}, [x16]
	WORD $0x6e71861d // sub v29.8h, v16.8h, v17.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4e7d3c18 // cmge v24.8h, v0.8h, v29.8h
	WORD $0x6e72863d // sub v29.8h, v17.8h, v18.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4e7d3c1e // cmge v30.8h, v0.8h, v29.8h
	WORD $0x4e3e1f18 // and v24.16b, v24.16b, v30.16b
	WORD $0x6e73865d // sub v29.8h, v18.8h, v19.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4e7d3c1e // cmge v30.8h, v0.8h, v29.8h
	WORD $0x4e3e1f18 // and v24.16b, v24.16b, v30.16b
	WORD $0x6e7486bd // sub v29.8h, v21.8h, v20.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4e7d3c1e // cmge v30.8h, v0.8h, v29.8h
	WORD $0x4e3e1f18 // and v24.16b, v24.16b, v30.16b
	WORD $0x6e7586dd // sub v29.8h, v22.8h, v21.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4e7d3c1e // cmge v30.8h, v0.8h, v29.8h
	WORD $0x4e3e1f18 // and v24.16b, v24.16b, v30.16b
	WORD $0x6e7686fd // sub v29.8h, v23.8h, v22.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4e7d3c1e // cmge v30.8h, v0.8h, v29.8h
	WORD $0x4e3e1f18 // and v24.16b, v24.16b, v30.16b
	WORD $0x6e74867d // sub v29.8h, v19.8h, v20.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4f1157bd // shl v29.8h, v29.8h, #1
	WORD $0x6e75865e // sub v30.8h, v18.8h, v21.8h
	WORD $0x4e60bbde // abs v30.8h, v30.8h
	WORD $0x4f1f07de // sshr v30.8h, v30.8h, #1
	WORD $0x4e7e87bd // add v29.8h, v29.8h, v30.8h
	WORD $0x4e7d3c3e // cmge v30.8h, v1.8h, v29.8h
	WORD $0x4e3e1f18 // and v24.16b, v24.16b, v30.16b
	WORD $0x6e73865d // sub v29.8h, v18.8h, v19.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4e7d3c79 // cmge v25.8h, v3.8h, v29.8h
	WORD $0x6e7486bd // sub v29.8h, v21.8h, v20.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4e7d3c7e // cmge v30.8h, v3.8h, v29.8h
	WORD $0x4e3e1f39 // and v25.16b, v25.16b, v30.16b
	WORD $0x6e73863d // sub v29.8h, v17.8h, v19.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4e7d3c7e // cmge v30.8h, v3.8h, v29.8h
	WORD $0x4e3e1f39 // and v25.16b, v25.16b, v30.16b
	WORD $0x6e7486dd // sub v29.8h, v22.8h, v20.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4e7d3c7e // cmge v30.8h, v3.8h, v29.8h
	WORD $0x4e3e1f39 // and v25.16b, v25.16b, v30.16b
	WORD $0x6e73861d // sub v29.8h, v16.8h, v19.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4e7d3c7e // cmge v30.8h, v3.8h, v29.8h
	WORD $0x4e3e1f39 // and v25.16b, v25.16b, v30.16b
	WORD $0x6e7486fd // sub v29.8h, v23.8h, v20.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4e7d3c7e // cmge v30.8h, v3.8h, v29.8h
	WORD $0x4e3e1f39 // and v25.16b, v25.16b, v30.16b
	WORD $0x6e73865d // sub v29.8h, v18.8h, v19.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4e6237ba // cmgt v26.8h, v29.8h, v2.8h
	WORD $0x6e7486bd // sub v29.8h, v21.8h, v20.8h
	WORD $0x4e60bbbd // abs v29.8h, v29.8h
	WORD $0x4e6237bd // cmgt v29.8h, v29.8h, v2.8h
	WORD $0x4ebd1f5a // orr v26.16b, v26.16b, v29.16b
	WORD $0x6e64865d // sub v29.8h, v18.8h, v4.8h
	WORD $0x6e6486be // sub v30.8h, v21.8h, v4.8h
	WORD $0x6e7e87bf // sub v31.8h, v29.8h, v30.8h
	WORD $0x4e666fff // smin v31.8h, v31.8h, v6.8h
	WORD $0x4e6567ff // smax v31.8h, v31.8h, v5.8h
	WORD $0x4e3a1fff // and v31.16b, v31.16b, v26.16b
	WORD $0x6e73869d // sub v29.8h, v20.8h, v19.8h
	WORD $0x4e679fbd // mul v29.8h, v29.8h, v7.8h
	WORD $0x4e7d87ff // add v31.8h, v31.8h, v29.8h
	WORD $0x4e666fff // smin v31.8h, v31.8h, v6.8h
	WORD $0x4e6567ff // smax v31.8h, v31.8h, v5.8h
	WORD $0x4f00849d // movi v29.8h, #4
	WORD $0x4e7d87fb // add v27.8h, v31.8h, v29.8h
	WORD $0x4e666f7b // smin v27.8h, v27.8h, v6.8h
	WORD $0x4e65677b // smax v27.8h, v27.8h, v5.8h
	WORD $0x4f1d077b // sshr v27.8h, v27.8h, #3
	WORD $0x4e6787fc // add v28.8h, v31.8h, v7.8h
	WORD $0x4e666f9c // smin v28.8h, v28.8h, v6.8h
	WORD $0x4e65679c // smax v28.8h, v28.8h, v5.8h
	WORD $0x4f1d079c // sshr v28.8h, v28.8h, #3
	WORD $0x4e73865f // add v31.8h, v18.8h, v19.8h
	WORD $0x4e7487ff // add v31.8h, v31.8h, v20.8h
	WORD $0x4f11563e // shl v30.8h, v17.8h, #1
	WORD $0x4e7e87ff // add v31.8h, v31.8h, v30.8h
	WORD $0x4e679e1e // mul v30.8h, v16.8h, v7.8h
	WORD $0x4e7e87ff // add v31.8h, v31.8h, v30.8h
	WORD $0x4f1d27ff // srshr v31.8h, v31.8h, #3
	WORD $0x4eb91f3e // mov v30.16b, v25.16b
	WORD $0x6e711ffe // bsl v30.16b, v31.16b, v17.16b
	WORD $0x4eb81f1d // mov v29.16b, v24.16b
	WORD $0x6e711fdd // bsl v29.16b, v30.16b, v17.16b
	WORD $0x4c00755d // st1 {v29.8h}, [x10]
	WORD $0x4f1f277e // srshr v30.8h, v27.8h, #1
	WORD $0x6e64865d // sub v29.8h, v18.8h, v4.8h
	WORD $0x4e7e87bd // add v29.8h, v29.8h, v30.8h
	WORD $0x4e666fbd // smin v29.8h, v29.8h, v6.8h
	WORD $0x4e6567bd // smax v29.8h, v29.8h, v5.8h
	WORD $0x4e6487bd // add v29.8h, v29.8h, v4.8h
	WORD $0x4eba1f5e // mov v30.16b, v26.16b
	WORD $0x6e7d1e5e // bsl v30.16b, v18.16b, v29.16b
	WORD $0x4f11561f // shl v31.8h, v16.8h, #1
	WORD $0x4e7187ff // add v31.8h, v31.8h, v17.8h
	WORD $0x4f11565d // shl v29.8h, v18.8h, #1
	WORD $0x4e7d87ff // add v31.8h, v31.8h, v29.8h
	WORD $0x4e7387ff // add v31.8h, v31.8h, v19.8h
	WORD $0x4e7487ff // add v31.8h, v31.8h, v20.8h
	WORD $0x4e7587ff // add v31.8h, v31.8h, v21.8h
	WORD $0x4f1d27ff // srshr v31.8h, v31.8h, #3
	WORD $0x4eb91f3d // mov v29.16b, v25.16b
	WORD $0x6e7e1ffd // bsl v29.16b, v31.16b, v30.16b
	WORD $0x4eb81f1e // mov v30.16b, v24.16b
	WORD $0x6e721fbe // bsl v30.16b, v29.16b, v18.16b
	WORD $0x4c00757e // st1 {v30.8h}, [x11]
	WORD $0x6e64867d // sub v29.8h, v19.8h, v4.8h
	WORD $0x4e7c87bd // add v29.8h, v29.8h, v28.8h
	WORD $0x4e666fbd // smin v29.8h, v29.8h, v6.8h
	WORD $0x4e6567bd // smax v29.8h, v29.8h, v5.8h
	WORD $0x4e6487bd // add v29.8h, v29.8h, v4.8h
	WORD $0x4ebd1fbe // mov v30.16b, v29.16b
	WORD $0x4e71861f // add v31.8h, v16.8h, v17.8h
	WORD $0x4e7287ff // add v31.8h, v31.8h, v18.8h
	WORD $0x4f11567d // shl v29.8h, v19.8h, #1
	WORD $0x4e7d87ff // add v31.8h, v31.8h, v29.8h
	WORD $0x4e7487ff // add v31.8h, v31.8h, v20.8h
	WORD $0x4e7587ff // add v31.8h, v31.8h, v21.8h
	WORD $0x4e7687ff // add v31.8h, v31.8h, v22.8h
	WORD $0x4f1d27ff // srshr v31.8h, v31.8h, #3
	WORD $0x4eb91f3d // mov v29.16b, v25.16b
	WORD $0x6e7e1ffd // bsl v29.16b, v31.16b, v30.16b
	WORD $0x4eb81f1e // mov v30.16b, v24.16b
	WORD $0x6e731fbe // bsl v30.16b, v29.16b, v19.16b
	WORD $0x4c00759e // st1 {v30.8h}, [x12]
	WORD $0x6e64869d // sub v29.8h, v20.8h, v4.8h
	WORD $0x6e7b87bd // sub v29.8h, v29.8h, v27.8h
	WORD $0x4e666fbd // smin v29.8h, v29.8h, v6.8h
	WORD $0x4e6567bd // smax v29.8h, v29.8h, v5.8h
	WORD $0x4e6487bd // add v29.8h, v29.8h, v4.8h
	WORD $0x4ebd1fbe // mov v30.16b, v29.16b
	WORD $0x4e72863f // add v31.8h, v17.8h, v18.8h
	WORD $0x4e7387ff // add v31.8h, v31.8h, v19.8h
	WORD $0x4f11569d // shl v29.8h, v20.8h, #1
	WORD $0x4e7d87ff // add v31.8h, v31.8h, v29.8h
	WORD $0x4e7587ff // add v31.8h, v31.8h, v21.8h
	WORD $0x4e7687ff // add v31.8h, v31.8h, v22.8h
	WORD $0x4e7787ff // add v31.8h, v31.8h, v23.8h
	WORD $0x4f1d27ff // srshr v31.8h, v31.8h, #3
	WORD $0x4eb91f3d // mov v29.16b, v25.16b
	WORD $0x6e7e1ffd // bsl v29.16b, v31.16b, v30.16b
	WORD $0x4eb81f1e // mov v30.16b, v24.16b
	WORD $0x6e741fbe // bsl v30.16b, v29.16b, v20.16b
	WORD $0x4c0075be // st1 {v30.8h}, [x13]
	WORD $0x4f1f277e // srshr v30.8h, v27.8h, #1
	WORD $0x6e6486bd // sub v29.8h, v21.8h, v4.8h
	WORD $0x6e7e87bd // sub v29.8h, v29.8h, v30.8h
	WORD $0x4e666fbd // smin v29.8h, v29.8h, v6.8h
	WORD $0x4e6567bd // smax v29.8h, v29.8h, v5.8h
	WORD $0x4e6487bd // add v29.8h, v29.8h, v4.8h
	WORD $0x4eba1f5e // mov v30.16b, v26.16b
	WORD $0x6e7d1ebe // bsl v30.16b, v21.16b, v29.16b
	WORD $0x4e73865f // add v31.8h, v18.8h, v19.8h
	WORD $0x4e7487ff // add v31.8h, v31.8h, v20.8h
	WORD $0x4f1156bd // shl v29.8h, v21.8h, #1
	WORD $0x4e7d87ff // add v31.8h, v31.8h, v29.8h
	WORD $0x4e7687ff // add v31.8h, v31.8h, v22.8h
	WORD $0x4f1156fd // shl v29.8h, v23.8h, #1
	WORD $0x4e7d87ff // add v31.8h, v31.8h, v29.8h
	WORD $0x4f1d27ff // srshr v31.8h, v31.8h, #3
	WORD $0x4eb91f3d // mov v29.16b, v25.16b
	WORD $0x6e7e1ffd // bsl v29.16b, v31.16b, v30.16b
	WORD $0x4eb81f1e // mov v30.16b, v24.16b
	WORD $0x6e751fbe // bsl v30.16b, v29.16b, v21.16b
	WORD $0x4c0075de // st1 {v30.8h}, [x14]
	WORD $0x4e74867f // add v31.8h, v19.8h, v20.8h
	WORD $0x4e7587ff // add v31.8h, v31.8h, v21.8h
	WORD $0x4f1156dd // shl v29.8h, v22.8h, #1
	WORD $0x4e7d87ff // add v31.8h, v31.8h, v29.8h
	WORD $0x4e679efd // mul v29.8h, v23.8h, v7.8h
	WORD $0x4e7d87ff // add v31.8h, v31.8h, v29.8h
	WORD $0x4f1d27ff // srshr v31.8h, v31.8h, #3
	WORD $0x4eb91f3e // mov v30.16b, v25.16b
	WORD $0x6e761ffe // bsl v30.16b, v31.16b, v22.16b
	WORD $0x4eb81f1d // mov v29.16b, v24.16b
	WORD $0x6e761fdd // bsl v29.16b, v30.16b, v22.16b
	WORD $0x4c0075fd // st1 {v29.8h}, [x15]
	ADD $16, R9, R9
	ADD $16, R10, R10
	ADD $16, R11, R11
	ADD $16, R12, R12
	ADD $16, R13, R13
	ADD $16, R14, R14
	ADD $16, R15, R15
	ADD $16, R16, R16
	SUB $1, R17, R17
	B   loop8

done8:
	RET
