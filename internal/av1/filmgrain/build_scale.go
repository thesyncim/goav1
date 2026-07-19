// Ported from libaom: av1/decoder/grain_synthesis.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package filmgrain

// buildScaleRow10PureGo is the canonical bit-exact reference for the per-run
// 10-bit scaling-LUT gather+interpolate that apply.go performs before the grain
// add+clip kernel. It mirrors the scalar per-pixel loop in
// applyLumaBlock/applyChromaBlock:
//
//	scale[i] = uint16(scaleLUT10(lut, int(src[i])))  // interp lut[x],lut[x+1]
//
// src holds a contiguous run of image samples (luma samples, or the
// chroma-scaling-from-luma samples with no horizontal subsampling), lut is the
// 256-entry AV1 scaling LUT, and scale receives the gathered scaling values in
// [0,255] widened to uint16. len(src) must be >= len(scale). This is the
// vectorization target; the architecture entry point buildScaleRow10 (see
// build_scale_dispatch_*.go) routes to the NEON kernel when available and falls
// back here otherwise. 8-bit stays scalar in apply.go (a bare lut[index] gather
// has no interpolation to amortize a NEON per-lane gather; scalar is faster).
func buildScaleRow10PureGo(scale []uint16, src []uint16, lut []uint8) {
	src = src[:len(scale)]
	for i := range scale {
		scale[i] = uint16(scaleLUT10(lut, int(src[i])))
	}
}
