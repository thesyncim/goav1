// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package motion

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// TestConvolve2DHighBDScratchMatchesNoScratch asserts the pure-Go
// scratch-carrying HBD 2D convolves and the bound dispatch slots are
// bit-identical to their no-scratch references with a deliberately poisoned
// scratch, proving every intermediate sample read is written first. Runs on
// every build (including purego), exercising whichever variant the dispatch
// slots bound at init.
func TestConvolve2DHighBDScratchMatchesNoScratch(t *testing.T) {
	rng := rand.New(rand.NewSource(0x2d5c))
	const pad = filterTaps
	scratch := &ConvolveScratch{}
	poison := func() {
		for i := range scratch.imHBD {
			scratch.imHBD[i] = int32(0x7bad0000) | int32(i&0xffff)
		}
		for i := range scratch.edge16 {
			scratch.edge16[i] = byte(0xa5)
		}
	}
	xk := subpelFilters8[6]
	yk := subpelFilters8[9]
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		for _, w := range []int{4, 8, 16, 32, 64, 128} {
			for _, h := range []int{4, 8, 16, 32, 64} {
				side := w
				if h > side {
					side = h
				}
				refSide := side + 2*pad
				ref, _ := testPlane(refSide, refSide, 2, refSide*2)
				for y := 0; y < refSide; y++ {
					for x := 0; x < refSide; x++ {
						setSample(ref, 2, x, y, uint16(rng.Intn(int(max)+1)))
					}
				}
				// Pure-Go WithScratch, resident window.
				g, _ := testPlane(w, h, 2, w*2)
				wn, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDPureGoWithScratch(g, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk, scratch)
				convolve2DHighBDPureGo(wn, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
				diffHighBDScratchPlanes(t, g, wn, w, h, "puregoscratch", bd)
				// Pure-Go WithScratch, clamped overhang.
				gc, _ := testPlane(w, h, 2, w*2)
				wc, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDClampedPureGoWithScratch(gc, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk, scratch)
				convolve2DHighBDClampedPureGo(wc, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk)
				diffHighBDScratchPlanes(t, gc, wc, w, h, "puregoscratch-clamped", bd)
				// Dispatch slots (whatever the arch bound), scratch vs no-scratch.
				gd, _ := testPlane(w, h, 2, w*2)
				wd, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDWithScratchImpl(gd, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk, scratch)
				convolve2DHighBDImpl(wd, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
				diffHighBDScratchPlanes(t, gd, wd, w, h, "dispatchscratch", bd)
				gdc, _ := testPlane(w, h, 2, w*2)
				wdc, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDClampedWithScratchImpl(gdc, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk, scratch)
				convolve2DHighBDClampedImpl(wdc, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk)
				diffHighBDScratchPlanes(t, gdc, wdc, w, h, "dispatchscratch-clamped", bd)
			}
		}
	}
}

// TestBlendCompoundAvgHighBDImplMatchesReference asserts the bound HBD blend
// dispatch slot (NEON on arm64, pure-Go elsewhere) matches the canonical
// pure-Go reference on every build.
func TestBlendCompoundAvgHighBDImplMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0xb1e42))
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		round0 := compoundRound0(bd)
		offsetBits := int(bd) + 2*filterBits - round0
		roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
		roundBits := 2*filterBits - round0 - compoundRound1Bits
		for _, w := range []int{4, 8, 16, 32, 64, 128} {
			for _, h := range []int{1, 3, 4, 8, 32} {
				src0 := make([]uint16, w*h)
				src1 := make([]uint16, w*h)
				for i := range src0 {
					src0[i] = uint16(rng.Intn(1 << 16))
					src1[i] = uint16(rng.Intn(1 << 16))
				}
				for _, wt := range [][2]int{{8, 8}, {9, 7}, {13, 3}} {
					got, _ := testPlane(w, h, 2, w*2)
					want, _ := testPlane(w, h, 2, w*2)
					blendCompoundAvgHighBDImpl(got, src0, src1, max, 0, 0, w, h, wt[0], wt[1], roundOffset, roundBits)
					blendCompoundAvgHighBD(want, src0, src1, max, 0, 0, w, h, wt[0], wt[1], roundOffset, roundBits)
					diffHighBDScratchPlanes(t, got, want, w, h, "blendimpl", bd, w, h, wt)
				}
			}
		}
	}
}

func diffHighBDScratchPlanes(t *testing.T, got, want frame.Plane, w, h int, tag string, ctx ...any) {
	t.Helper()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g := getSample(got, 2, x, y)
			e := getSample(want, 2, x, y)
			if g != e {
				t.Fatalf("%s (%d,%d): scratch=%d ref=%d ctx=%v", tag, x, y, g, e, ctx)
			}
		}
	}
}
