// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// This file is a SPIKE: it reimplements the 8-bit residual-add inner loop using
// Go's native arm64 SIMD intrinsics (the simd/archsimd package, Go 1.27+ with
// GOEXPERIMENT=simd). It is compiled ONLY under GOEXPERIMENT=simd, so the normal
// Go 1.26 production build never sees it — the standard scalar/NEON-asm dispatch
// is unaffected. Its purpose is to measure Go-native SIMD against the scalar
// baseline and the existing hand-written Plan 9 NEON asm on identical real code,
// and to prove byte-identity via the differential test in plane_simd_arm64_test.go.
//
// All vector loads/stores go through the array-pointer (value-type) variants
// LoadUint8x16Array(*[16]uint8) / LoadInt16x8Array(*[8]int16) / StoreArray — no
// slice-header indirection in the hot path, matching goav1's trusted-path
// convention. Slice regions are converted to fixed-size array pointers once per
// 16-pixel chunk; the loop bounds (x+16 <= width, validated residual/dst extent)
// guarantee those conversions never see a short slice.
//
// Byte-exactness argument for the 8-bit path: the scalar loop computes
// clamp(int(dst)+int(res), 0, max) with a 32-bit add. Here we use a signed int16
// AddSaturated followed by Max(0).Min(max). Saturation only diverges from the
// exact sum when dst+res leaves [-32768, 32767]; in every such case the exact
// sum is already outside [0, max], so the subsequent clamp maps it to the same
// endpoint the scalar produces. Hence identical output for all inputs.

package dsp

import "simd/archsimd"

// addResidualPlaneBlockSIMD is the tuned Go-native-SIMD analogue of
// addResidualPlaneBlockPureGo for the 8-bit path: 16 pixels per iteration, one
// full Uint8x16 array-pointer load (both halves widened), a two-half narrowing
// pack, and one 16-byte StoreArray. Widths that are not a multiple of 16 (the
// AV1 4- and 8-wide transforms) and the 16-bit path fall back to the scalar
// reference, so the function is defined for the same inputs as the scalar one.
func addResidualPlaneBlockSIMD(block planeBlock, bytesPerSample int, max uint16, width int, residual []int16, residualStride int) {
	if bytesPerSample != 1 || width < 16 || width%16 != 0 {
		addResidualPlaneBlockPureGo(block, bytesPerSample, max, width, residual, residualStride)
		return
	}
	zero := archsimd.BroadcastInt16x8(0)
	hi := archsimd.BroadcastInt16x8(int16(max))
	for row := 0; row < block.height; row++ {
		dstRow := block.pix[row*block.stride:]
		resRow := residual[row*residualStride:]
		for x := 0; x < width; x += 16 {
			// Value-type loads: one 16-byte destination vector, two 8-lane
			// residual vectors, all from *[N]T (no slice indirection).
			dv := archsimd.LoadUint8x16Array((*[16]uint8)(dstRow[x:]))
			dLo := dv.ExtendLo8ToUint16().ConvertToInt16()
			dHi := dv.ConcatShiftBytesRight(dv, 8).ExtendLo8ToUint16().ConvertToInt16()
			rLo := archsimd.LoadInt16x8Array((*[8]int16)(resRow[x:]))
			rHi := archsimd.LoadInt16x8Array((*[8]int16)(resRow[x+8:]))
			// Byte-exact add + clamp (see file header), narrow each half to
			// uint8 (low 8 bytes valid).
			sLo := dLo.AddSaturated(rLo).Max(zero).Min(hi).SaturateToUint8()
			sHi := dHi.AddSaturated(rHi).Max(zero).Min(hi).SaturateToUint8()
			// Pack the two low-8 halves into one 16-byte vector by interleaving
			// their low 64-bit lanes, then a single value-type store.
			packed := sLo.ReshapeToUint64s().InterleaveLo(sHi.ReshapeToUint64s()).ReshapeToUint8s()
			packed.StoreArray((*[16]uint8)(dstRow[x:]))
		}
	}
}
