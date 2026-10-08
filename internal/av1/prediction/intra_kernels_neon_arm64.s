// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Code generated with the help of an offline NEON encoder; do not edit by hand.
// Every WORD below is the little-endian encoding of the mnemonic in its
// trailing comment, assembled and verified with the system assembler (as
// -arch arm64). The kernels are bit-exact with the *PureGo references in
// intra_kernels.go; the Go wrappers in intra_kernels_neon_arm64.go restrict
// them to the 8-bit, width-multiple-of-8 shapes they handle. Working vectors
// stay in V0-V7 and V16-V20 to avoid clobbering the callee-saved V8-V15.

//go:build arm64 && !purego

#include "textflag.h"

// func sumSamplesNEONAsm(samples *uint16, n uintptr) uint64
//
// Returns the 64-bit sum of n (multiple of 8) uint16 samples.
TEXT ·sumSamplesNEONAsm(SB), NOSPLIT, $0-24
	MOVD samples+0(FP), R1
	MOVD n+8(FP), R2
	WORD $0x6f00e414 // movi v20.2d, #0   (u64 accumulator)
sumLoop:
	WORD $0x4cdf7420 // ld1 {v0.8h}, [x1], #16
	WORD $0x6e602806 // uaddlp v6.4s, v0.8h
	WORD $0x6ea028c7 // uaddlp v7.2d, v6.4s
	WORD $0x4ee78694 // add v20.2d, v20.2d, v7.2d
	SUB  $8, R2, R2
	CBNZ R2, sumLoop
	WORD $0x5ef1ba88 // addp d8, v20.2d
	WORD $0x4e083d0c // mov x12, v8.d[0]
	MOVD R12, ret+16(FP)
	RET

// func subsampleLuma8NEONAsm(out *uint16, top *uint8, bot *uint8, mode uintptr, n uintptr)
//
// Produces 8 Q3 output samples. mode: 0 = no subsample (x<<3), 1 = horizontal
// ((a+b)<<2), 2 = 2x2 ((a+b+c+d)<<1). bot is only read for mode 2.
TEXT ·subsampleLuma8NEONAsm(SB), NOSPLIT, $0-40
	MOVD out+0(FP), R1
	MOVD top+8(FP), R2
	MOVD bot+16(FP), R3
	MOVD mode+24(FP), R4

	CMP  $2, R4
	BEQ  ssXY
	CMP  $1, R4
	BEQ  ssX

	// mode 0: out = top<<3
	WORD $0x0c407040 // ld1 {v0.8b}, [x2]
	WORD $0x2f08a405 // uxtl v5.8h, v0.8b
	WORD $0x4f1354a5 // shl v5.8h, v5.8h, #3
	WORD $0x4c007425 // st1 {v5.8h}, [x1]
	RET

ssX:
	// mode 1: out = (a+b)<<2  (pairwise adjacent)
	WORD $0x4c407040 // ld1 {v0.16b}, [x2]
	WORD $0x6e202802 // uaddlp v2.8h, v0.16b
	WORD $0x4f125442 // shl v2.8h, v2.8h, #2
	WORD $0x4c007422 // st1 {v2.8h}, [x1]
	RET

ssXY:
	// mode 2: out = (a+b+c+d)<<1
	WORD $0x4c407040 // ld1 {v0.16b}, [x2]
	WORD $0x4c407061 // ld1 {v1.16b}, [x3]
	WORD $0x6e202802 // uaddlp v2.8h, v0.16b
	WORD $0x6e202823 // uaddlp v3.8h, v1.16b
	WORD $0x4e638444 // add v4.8h, v2.8h, v3.8h
	WORD $0x4f115484 // shl v4.8h, v4.8h, #1
	WORD $0x4c007424 // st1 {v4.8h}, [x1]
	RET

// func applyCFLRowNEONAsm(dst *byte, ac *int16, alpha uintptr, n uintptr)
//
// Applies 8 chroma-from-luma samples in place:
//   scaled = sign(ac*alpha) * ((|ac*alpha| + 32) >> 6)
//   dst    = clamp(dst + scaled, 0, 255)
// alpha is the signed Q3 alpha sign-extended into the low 16 bits.
TEXT ·applyCFLRowNEONAsm(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R1
	MOVD ac+8(FP), R2
	MOVD alpha+16(FP), R7

	WORD $0x4e020ce0 // dup v0.8h, w7   (alpha broadcast, s16)
	WORD $0x4c407441 // ld1 {v1.8h}, [x2]   (8 ac, s16)
	WORD $0x0e60c022 // smull  v2.4s, v1.4h, v0.4h
	WORD $0x4e60c023 // smull2 v3.4s, v1.8h, v0.8h
	WORD $0x4ea0a852 // cmlt v18.4s, v2.4s, #0  (lo sign mask)
	WORD $0x4ea0a873 // cmlt v19.4s, v3.4s, #0  (hi sign mask)
	WORD $0x4ea0b844 // abs v4.4s, v2.4s
	WORD $0x4ea0b865 // abs v5.4s, v3.4s
	WORD $0x6f3a2484 // urshr v4.4s, v4.4s, #6
	WORD $0x6f3a24a5 // urshr v5.4s, v5.4s, #6
	WORD $0x6ea0b886 // neg v6.4s, v4.4s
	WORD $0x6ea0b8b0 // neg v16.4s, v5.4s
	WORD $0x6e641cd2 // bsl v18.16b, v6.16b, v4.16b   (sign ? -val : val)
	WORD $0x6e651e13 // bsl v19.16b, v16.16b, v5.16b
	WORD $0x0e612a47 // xtn  v7.4h, v18.4s
	WORD $0x4e612a67 // xtn2 v7.8h, v19.4s   (scaled, s16)
	WORD $0x0c407028 // ld1 {v8.8b}, [x1]    (chroma DC bytes)
	WORD $0x2f08a509 // uxtl v9.8h, v8.8b
	WORD $0x4e67852a // add v10.8h, v9.8h, v7.8h
	WORD $0x2e21294b // sqxtun v11.8b, v10.8h   (clamp to [0,255])
	WORD $0x0c00702b // st1 {v11.8b}, [x1]
	RET

// func dirRowInterp8NEONAsm(dst *byte, above *uint16, weight32 uintptr, weightShift uintptr, n uintptr)
//
// Fills n (multiple of 8) destination bytes with
//   roundPowerOfTwo(p0*(32-shift) + p1*shift, 5)
// where p0 = above[i], p1 = above[i+1]. weight32 = 32-shift, weightShift =
// shift. Callers only invoke this for rows entirely inside the interpolation
// region (above[base+i] and above[base+i+1] always valid, no clamp).
TEXT ·dirRowInterp8NEONAsm(SB), NOSPLIT, $0-40
	MOVD dst+0(FP), R11
	MOVD above+8(FP), R10
	MOVD weight32+16(FP), R8
	MOVD weightShift+24(FP), R9
	MOVD n+32(FP), R12

	WORD $0x4e020d00 // dup v0.8h, w8   (32-shift)
	WORD $0x4e020d21 // dup v1.8h, w9   (shift)
diRow:
	WORD $0x4c407542 // ld1 {v2.8h}, [x10]      (p0 = above[i..i+7])
	WORD $0x3cc02143 // ldur q3, [x10, #2]       (p1 = above[i+1..i+8])
	WORD $0x2e60c044 // umull  v4.4s, v2.4h, v0.4h
	WORD $0x6e60c045 // umull2 v5.4s, v2.8h, v0.8h
	WORD $0x2e618064 // umlal  v4.4s, v3.4h, v1.4h
	WORD $0x6e618065 // umlal2 v5.4s, v3.8h, v1.8h
	WORD $0x6f3b2484 // urshr v4.4s, v4.4s, #5
	WORD $0x6f3b24a5 // urshr v5.4s, v5.4s, #5
	WORD $0x0e612886 // xtn  v6.4h, v4.4s
	WORD $0x4e6128a6 // xtn2 v6.8h, v5.4s
	WORD $0x0e2128c7 // xtn v7.8b, v6.8h
	WORD $0x0c9f7167 // st1 {v7.8b}, [x11], #8
	ADD  $16, R10, R10
	SUB  $8, R12, R12
	CBNZ R12, diRow
	RET

// func dirAboveRun8NEONAsm(dst *byte, ref *uint16, weight32 uintptr, weightShift uintptr, n uintptr)
//
// Fills n (multiple of 8) destination bytes for the zone-2 "above" branch with
//   roundPowerOfTwo(ref[i]*(32-shift) + ref[i+1]*shift, 5)
// where the source index advances by one per output (a forward, contiguous run
// over the pre-validated edge span, so ref[i] and ref[i+1] are always valid and
// there is no clamp). weight32 = 32-shift, weightShift = shift. Same arithmetic
// as dirRowInterp8NEONAsm.
TEXT ·dirAboveRun8NEONAsm(SB), NOSPLIT, $0-40
	MOVD dst+0(FP), R11
	MOVD ref+8(FP), R10
	MOVD weight32+16(FP), R8
	MOVD weightShift+24(FP), R9
	MOVD n+32(FP), R12

	WORD $0x4e020d00 // dup v0.8h, w8   (32-shift)
	WORD $0x4e020d21 // dup v1.8h, w9   (shift)
arRow:
	WORD $0x4c407542 // ld1 {v2.8h}, [x10]      (p0 = ref[i..i+7])
	WORD $0x3cc02143 // ldur q3, [x10, #2]       (p1 = ref[i+1..i+8])
	WORD $0x2e60c044 // umull  v4.4s, v2.4h, v0.4h
	WORD $0x6e60c045 // umull2 v5.4s, v2.8h, v0.8h
	WORD $0x2e618064 // umlal  v4.4s, v3.4h, v1.4h
	WORD $0x6e618065 // umlal2 v5.4s, v3.8h, v1.8h
	WORD $0x6f3b2484 // urshr v4.4s, v4.4s, #5
	WORD $0x6f3b24a5 // urshr v5.4s, v5.4s, #5
	WORD $0x0e612886 // xtn  v6.4h, v4.4s
	WORD $0x4e6128a6 // xtn2 v6.8h, v5.4s
	WORD $0x0e2128c7 // xtn v7.8b, v6.8h
	WORD $0x0c9f7167 // st1 {v7.8b}, [x11], #8
	ADD  $16, R10, R10
	SUB  $8, R12, R12
	CBNZ R12, arRow
	RET

// func dirLeftCol8NEONAsm(dst *byte, stride uintptr, ref *uint16, weight32 uintptr, weightShift uintptr, n uintptr)
//
// Fills n (multiple of 8) destination samples down a single zone-3 column with
//   roundPowerOfTwo(ref[i]*(32-shift) + ref[i+1]*shift, 5)
// where the source index advances by one per row (a forward, contiguous run
// over the pre-validated left edge, so no clamp). The 8 interpolated bytes per
// iteration are scattered into the column with strided single-lane stores; the
// store base stays in R11 and the per-row byte stride stays in R1 (the lane
// store encodings post-index by Rm = x1). weight32 = 32-shift, weightShift =
// shift. Same arithmetic as dirAboveRun8NEONAsm.
TEXT ·dirLeftCol8NEONAsm(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R11
	MOVD stride+8(FP), R1
	MOVD ref+16(FP), R10
	MOVD weight32+24(FP), R8
	MOVD weightShift+32(FP), R9
	MOVD n+40(FP), R12

	WORD $0x4e020d00 // dup v0.8h, w8   (32-shift)
	WORD $0x4e020d21 // dup v1.8h, w9   (shift)
lcRow:
	WORD $0x4c407542 // ld1 {v2.8h}, [x10]      (p0 = ref[i..i+7])
	WORD $0x3cc02143 // ldur q3, [x10, #2]       (p1 = ref[i+1..i+8])
	WORD $0x2e60c044 // umull  v4.4s, v2.4h, v0.4h
	WORD $0x6e60c045 // umull2 v5.4s, v2.8h, v0.8h
	WORD $0x2e618064 // umlal  v4.4s, v3.4h, v1.4h
	WORD $0x6e618065 // umlal2 v5.4s, v3.8h, v1.8h
	WORD $0x6f3b2484 // urshr v4.4s, v4.4s, #5
	WORD $0x6f3b24a5 // urshr v5.4s, v5.4s, #5
	WORD $0x0e612886 // xtn  v6.4h, v4.4s
	WORD $0x4e6128a6 // xtn2 v6.8h, v5.4s
	WORD $0x0e2128c7 // xtn v7.8b, v6.8h
	WORD $0x0d810167 // st1 {v7.b}[0], [x11], x1
	WORD $0x0d810567 // st1 {v7.b}[1], [x11], x1
	WORD $0x0d810967 // st1 {v7.b}[2], [x11], x1
	WORD $0x0d810d67 // st1 {v7.b}[3], [x11], x1
	WORD $0x0d811167 // st1 {v7.b}[4], [x11], x1
	WORD $0x0d811567 // st1 {v7.b}[5], [x11], x1
	WORD $0x0d811967 // st1 {v7.b}[6], [x11], x1
	WORD $0x0d811d67 // st1 {v7.b}[7], [x11], x1
	ADD  $16, R10, R10
	SUB  $8, R12, R12
	CBNZ R12, lcRow
	RET

// func applyCFLRow16NEONAsm(dst *byte, ac *int16, alpha uintptr, max uintptr)
//
// High-bit-depth counterpart of applyCFLRowNEONAsm. Applies 8 chroma-from-luma
// samples in place over a uint16 destination:
//   scaled = sign(ac*alpha) * ((|ac*alpha| + 32) >> 6)
//   dst    = clamp(dst + scaled, 0, max)
// The scaled residual is computed exactly like the 8-bit kernel (and matches
// roundPowerOfTwoSigned(alpha*ac, 6)) and kept in s32 lanes; the DC is widened
// u16->u32 so the add and the [0,max] clamp (SMAX 0 / SMIN max) run in s32 and
// stay bit-exact for any DC value, then narrowed back to u16. alpha is the
// signed Q3 alpha in the low 16 bits; max = (1<<bitDepth)-1.
TEXT ·applyCFLRow16NEONAsm(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R1
	MOVD ac+8(FP), R2
	MOVD alpha+16(FP), R7
	MOVD max+24(FP), R8

	WORD $0x4e020ce0 // dup v0.8h, w7   (alpha broadcast, s16)
	WORD $0x4e040d14 // dup v20.4s, w8  (max broadcast, s32)
	WORD $0x4f00041f // movi v31.4s, #0
	WORD $0x4c407441 // ld1 {v1.8h}, [x2]   (8 ac, s16)
	WORD $0x0e60c022 // smull  v2.4s, v1.4h, v0.4h
	WORD $0x4e60c023 // smull2 v3.4s, v1.8h, v0.8h
	WORD $0x4ea0a852 // cmlt v18.4s, v2.4s, #0
	WORD $0x4ea0a873 // cmlt v19.4s, v3.4s, #0
	WORD $0x4ea0b844 // abs v4.4s, v2.4s
	WORD $0x4ea0b865 // abs v5.4s, v3.4s
	WORD $0x6f3a2484 // urshr v4.4s, v4.4s, #6
	WORD $0x6f3a24a5 // urshr v5.4s, v5.4s, #6
	WORD $0x6ea0b886 // neg v6.4s, v4.4s
	WORD $0x6ea0b8b0 // neg v16.4s, v5.4s
	WORD $0x6e641cd2 // bsl v18.16b, v6.16b, v4.16b   (scaled lo, s32)
	WORD $0x6e651e13 // bsl v19.16b, v16.16b, v5.16b  (scaled hi, s32)
	WORD $0x4c407428 // ld1 {v8.8h}, [x1]    (8 chroma DC, u16)
	WORD $0x2f10a509 // uxtl  v9.4s, v8.4h    (dc lo -> u32)
	WORD $0x6f10a50c // uxtl2 v12.4s, v8.8h   (dc hi -> u32)
	WORD $0x4eb2852a // add v10.4s, v9.4s, v18.4s
	WORD $0x4eb3858b // add v11.4s, v12.4s, v19.4s
	WORD $0x4ebf654a // smax v10.4s, v10.4s, v31.4s   (>= 0)
	WORD $0x4ebf656b // smax v11.4s, v11.4s, v31.4s
	WORD $0x4eb46d4a // smin v10.4s, v10.4s, v20.4s   (<= max)
	WORD $0x4eb46d6b // smin v11.4s, v11.4s, v20.4s
	WORD $0x0e61294a // xtn  v10.4h, v10.4s
	WORD $0x4e61296a // xtn2 v10.8h, v11.4s
	WORD $0x4c00742a // st1 {v10.8h}, [x1]
	RET

// func applyCFL4NEONAsm(dst *byte, dstStride uintptr, ac *int16, alpha uintptr, height uintptr)
//
// Width-4 (8-bit) whole-block chroma-from-luma. Like the width-4 static kernels,
// two rows (4 columns each) are packed per 8-lane vector: ac[row][0..3] fill the
// low half and ac[row+1][0..3] the high half (ac uses the CFLBufLine==32 stride,
// i.e. 64 bytes/row). scaled is the same as applyCFLRowNEONAsm and the [0,255]
// clamp reuses SQXTUN. height is even (the wrapper keeps the odd frame-edge
// remainder on the pure-Go reference).
TEXT ·applyCFL4NEONAsm(SB), NOSPLIT, $0-40
	MOVD dst+0(FP), R1
	MOVD dstStride+8(FP), R4
	MOVD ac+16(FP), R2
	MOVD alpha+24(FP), R7
	MOVD height+32(FP), R6

	WORD $0x4e020ce0 // dup v0.8h, w7   (alpha broadcast, s16)
	MOVD ZR, R8 // row index
cfl4Row:
	CMP  R6, R8
	BGE  cfl4Done
	MOVD R1, R10     // dst_r
	ADD  R4, R1, R11 // dst_r1 = dst_r + stride
	ADD  $64, R2, R3 // ac_r1 = ac_r + CFLBufLine*2 bytes
	WORD $0x0c407441 // ld1 {v1.4h}, [x2]    (ac row r: 4 samples)
	WORD $0x0c407472 // ld1 {v18.4h}, [x3]   (ac row r+1)
	WORD $0x6e180641 // mov v1.d[1], v18.d[0]  (ac pair {r,r1})
	WORD $0x0e60c022 // smull  v2.4s, v1.4h, v0.4h
	WORD $0x4e60c023 // smull2 v3.4s, v1.8h, v0.8h
	WORD $0x4ea0a852 // cmlt v18.4s, v2.4s, #0
	WORD $0x4ea0a873 // cmlt v19.4s, v3.4s, #0
	WORD $0x4ea0b844 // abs v4.4s, v2.4s
	WORD $0x4ea0b865 // abs v5.4s, v3.4s
	WORD $0x6f3a2484 // urshr v4.4s, v4.4s, #6
	WORD $0x6f3a24a5 // urshr v5.4s, v5.4s, #6
	WORD $0x6ea0b886 // neg v6.4s, v4.4s
	WORD $0x6ea0b8b0 // neg v16.4s, v5.4s
	WORD $0x6e641cd2 // bsl v18.16b, v6.16b, v4.16b
	WORD $0x6e651e13 // bsl v19.16b, v16.16b, v5.16b
	WORD $0x0e612a47 // xtn  v7.4h, v18.4s
	WORD $0x4e612a67 // xtn2 v7.8h, v19.4s   (scaled pair, s16)
	WORD $0x0d408148 // ld1 {v8.s}[0], [x10]  (dc row r: 4 bytes)
	WORD $0x0d409168 // ld1 {v8.s}[1], [x11]  (dc row r+1)
	WORD $0x2f08a509 // uxtl v9.8h, v8.8b
	WORD $0x4e67852a // add v10.8h, v9.8h, v7.8h
	WORD $0x2e21294b // sqxtun v11.8b, v10.8h   (clamp to [0,255])
	WORD $0x0d00814b // st1 {v11.s}[0], [x10]   (row r)
	WORD $0x0d00916b // st1 {v11.s}[1], [x11]   (row r+1)
	ADD  R4, R11, R1  // dst += 2*stride
	ADD  $128, R2, R2 // ac += 2 rows (2*CFLBufLine*2 bytes)
	ADD  $2, R8, R8
	B    cfl4Row
cfl4Done:
	RET
