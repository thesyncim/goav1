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
// Byte-exactness argument for the 8-bit path: the scalar loop computes
// clamp(int(dst)+int(res), 0, max) with a 32-bit add. Here we use a signed int16
// AddSaturated followed by Max(0).Min(max). Saturation only diverges from the
// exact sum when dst+res leaves [-32768, 32767]; in every such case the exact
// sum is already outside [0, max], so the subsequent clamp maps it to the same
// endpoint the scalar produces. Hence identical output for all inputs.

package dsp

import "simd/archsimd"

// addResidualPlaneBlockSIMD is the Go-native-SIMD analogue of
// addResidualPlaneBlockPureGo. It handles the 8-bit case for widths that are a
// multiple of 8 (AV1 transform widths 8/16/32/64); width 4 and the 16-bit path
// fall back to the scalar reference so the function is always defined for the
// same inputs as the scalar one.
func addResidualPlaneBlockSIMD(block planeBlock, bytesPerSample int, max uint16, width int, residual []int16, residualStride int) {
	if bytesPerSample != 1 || width < 8 || width%8 != 0 {
		addResidualPlaneBlockPureGo(block, bytesPerSample, max, width, residual, residualStride)
		return
	}
	zero := archsimd.BroadcastInt16x8(0)
	hi := archsimd.BroadcastInt16x8(int16(max))
	for row := 0; row < block.height; row++ {
		line := block.pix[row*block.stride : row*block.stride+width : row*block.stride+width]
		resLine := residual[row*residualStride : row*residualStride+width : row*residualStride+width]
		for x := 0; x < width; x += 8 {
			// Widen 8 destination bytes (0..255) to signed int16.
			dv, _ := archsimd.LoadUint8x16Part(line[x:])
			d16 := dv.ExtendLo8ToUint16().ConvertToInt16()
			// 8 int16 residuals.
			r16, _ := archsimd.LoadInt16x8Part(resLine[x:])
			// Saturating add then clamp to [0, max] — byte-identical to the
			// scalar 32-bit add + clamp (see file header).
			sum := d16.AddSaturated(r16).Max(zero).Min(hi)
			// Narrow back to uint8 and store the 8 low bytes.
			sum.SaturateToUint8().StorePart(line[x : x+8])
		}
	}
}
