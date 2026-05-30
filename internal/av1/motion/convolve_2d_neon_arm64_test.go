// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// convolve2DKernelPhases returns every 8-tap subpel kernel phase across the four
// filter types so the differential test exercises all 16 x/y phases per type.
func convolve2DKernelTables() [][16][filterTaps]int16 {
	return [][16][filterTaps]int16{
		subpelFilters8,
		subpelFilters8Smooth,
		subpelFilters8Sharp,
		bilinearFilters,
	}
}

// TestConvolve2D8NEONMatchesPureGo asserts the NEON 2D convolve is bit-identical
// to convolve2D8PureGo for every width/height in 4..64, all 16 x and 16 y subpel
// phases of each filter type, over both deterministic and random reference
// pixels. It also asserts the NEON path allocates nothing.
func TestConvolve2D8NEONMatchesPureGo(t *testing.T) {
	tables := convolve2DKernelTables()
	rng := rand.New(rand.NewSource(0x2d20feed))

	const pad = filterTaps
	// Pick a representative subpel phase per filter for the full width/height
	// sweep, then sweep all 16x16 phases on a couple of fixed shapes.
	sweepSizes := []int{4, 8, 12, 16, 24, 32, 48, 64}

	makeRef := func(side int, randomize bool) frame.Plane {
		refSide := side + 2*pad
		r, _ := testPlane(refSide, refSide, 1, refSide)
		if randomize {
			for i := range r.Pix {
				r.Pix[i] = byte(rng.Intn(256))
			}
		} else {
			fillMotionTestPlane(r)
		}
		return r
	}

	run := func(t *testing.T, w, h int, xk, yk [filterTaps]int16, randomize bool) {
		t.Helper()
		side := w
		if h > side {
			side = h
		}
		ref := makeRef(side, randomize)
		got, _ := testPlane(w, h, 1, w)
		want, _ := testPlane(w, h, 1, w)
		convolve2D8NEON(got, ref, 0, 0, pad, pad, w, h, xk, yk)
		convolve2D8PureGo(want, ref, 0, 0, pad, pad, w, h, xk, yk)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				g := got.Pix[y*got.Stride+x]
				e := want.Pix[y*want.Stride+x]
				if g != e {
					t.Fatalf("w=%d h=%d (%d,%d): NEON=%d PureGo=%d xk=%v yk=%v", w, h, x, y, g, e, xk, yk)
				}
			}
		}
	}

	// Full width/height sweep with a non-trivial phase per filter type.
	for ti, tbl := range tables {
		xk := tbl[3]
		yk := tbl[5]
		for _, w := range sweepSizes {
			for _, h := range sweepSizes {
				for _, randomize := range []bool{false, true} {
					t.Run("", func(t *testing.T) { _ = ti; run(t, w, h, xk, yk, randomize) })
				}
			}
		}
	}

	// All 16x16 phase combinations per filter type on fixed shapes.
	for _, tbl := range tables {
		for sx := 0; sx < 16; sx++ {
			for sy := 0; sy < 16; sy++ {
				run(t, 16, 16, tbl[sx], tbl[sy], false)
				run(t, 8, 24, tbl[sx], tbl[sy], true)
			}
		}
	}

	// Zero-alloc assertion on the NEON path (width%8==0, 8-tap -> asm).
	ref := makeRef(32, true)
	dst, _ := testPlane(32, 32, 1, 32)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	allocs := testing.AllocsPerRun(50, func() {
		convolve2D8NEON(dst, ref, 0, 0, pad, pad, 32, 32, xk, yk)
	})
	if allocs != 0 {
		t.Fatalf("convolve2D8NEON allocated %v times, want 0", allocs)
	}
}
