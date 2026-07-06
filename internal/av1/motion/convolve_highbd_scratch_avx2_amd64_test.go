// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package motion

import (
	"math/rand"
	"testing"
)

// TestConvolve2DHighBDAVX2WithScratchMatchesPureGo asserts the scratch-carrying
// HBD 2D AVX2 convolve stays bit-identical to the pure-Go reference with a
// deliberately poisoned scratch, proving every intermediate sample read is
// written first. Kernels are called directly (never gated on cpu.Detected) so
// the sweep runs under Rosetta too.
func TestConvolve2DHighBDAVX2WithScratchMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xa7c2d))
	const pad = filterTaps
	scratch := &ConvolveScratch{}
	poison := func() {
		for i := range scratch.imHBD {
			scratch.imHBD[i] = int32(0x7bad0000) | int32(i&0xffff)
		}
	}
	xk := subpelFilters8[6]
	yk := subpelFilters8[9]
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << bd) - 1)
		for _, w := range []int{4, 8, 16, 32, 64, 128} {
			for _, h := range []int{4, 8, 16, 32} {
				side := w
				if h > side {
					side = h
				}
				ref := randPlaneHBD(rng, side+2*pad, max)
				g, _ := testPlane(w, h, 2, w*2)
				wn, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDAVX2WithScratch(g, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk, scratch)
				convolve2DHighBDPureGo(wn, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
				diffPlanesHBD(t, g, wn, w, h, "2Dhbd-scratch", bd)

				gc, _ := testPlane(w, h, 2, w*2)
				wc, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDClampedAVX2WithScratch(gc, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk, scratch)
				convolve2DHighBDClampedPureGo(wc, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk)
				diffPlanesHBD(t, gc, wc, w, h, "2Dhbd-scratch-clamped", bd)
			}
		}
	}
}
