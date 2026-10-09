// Ported from libaom: av1/decoder/grain_synthesis.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package filmgrain

// buildScaleRow10 is the bit-exact per-run 10-bit scaling-LUT gather+interpolate
// that apply.go performs before the grain add+clip kernel. For every sample it
// computes exactly scaleLUT10(lut, int(src[i])) and stores it widened to uint16:
//
//	idx    = min(src[i], 1023)
//	x      = idx>>2, rem = idx&3
//	scale  = uint8(lut[x] + roundPowerOfTwo((lut[min(x+1,255)]-lut[x])*rem, 2))
//
// src holds a contiguous run of image samples (luma samples, or the
// chroma-scaling-from-luma samples with no horizontal subsampling), lut is the
// 256-entry AV1 scaling LUT, and scale receives the gathered scaling values in
// [0,255] widened to uint16. len(src) must be >= len(scale).
//
// The loop is plain Go and branch-free. Every sample is one data-dependent LUT
// lookup pair, so there is no vector form that avoids a per-lane gather; the
// clamp of x+1 to 255 makes the x==255 special case of scaleLUT10 a zero
// difference, which reproduces it without a branch. 8-bit stays scalar in
// apply.go (a bare lut[index] gather with no interpolation).
func buildScaleRow10(scale []uint16, src []uint16, lut []uint8) {
	lut = lut[:ScalingLUTSize]
	src = src[:len(scale)]
	for i := range scale {
		idx := int(src[i])
		if idx > (1<<10)-1 {
			idx = (1 << 10) - 1
		}
		x := idx >> 2
		rem := idx & 3
		start := int(lut[x])
		end := int(lut[min(x+1, ScalingLUTSize-1)])
		scale[i] = uint16(uint8(start + roundPowerOfTwo((end-start)*rem, 2)))
	}
}
