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

// These tests assert the width-4 NEON convolve kernels (X, Y, 2D, plus their
// clamped wrappers) are bit-identical to the pure-Go reference. Width 4 is the
// most common inter shape: every 4:2:0 chroma block of an 8x8 luma block is 4x4
// and 4xN luma blocks are frequent, so a byte-exact 4-lane kernel is what closes
// the biggest inter-clip gap.

// w4FilterTables returns one 8-tap and one 4-tap (zero end taps) table so the
// sweep exercises both the full MAC and the 4-tap-kernel-contributes-nothing
// path through the same asm.
func w4FilterTables() (eight [][16][filterTaps]int16, four [][16][filterTaps]int16) {
	eight = [][16][filterTaps]int16{subpelFilters8, subpelFilters8Smooth, subpelFilters8Sharp}
	four = [][16][filterTaps]int16{subpelFilters4, subpelFilters4Smooth, bilinearFilters}
	return eight, four
}

// TestConvolveW4NEONMatchesPureGo sweeps the non-clamped width-4 X / Y / 2D NEON
// kernels across heights 4/8/16, all 16 subpel phases, 8-tap and 4-tap kernels,
// over deterministic and random reference pixels.
func TestConvolveW4NEONMatchesPureGo(t *testing.T) {
	const pad = filterTaps
	const w = 4
	heights := []int{4, 8, 16}
	eightTables, fourTables := w4FilterTables()
	rng := rand.New(rand.NewSource(0x04f4beef))

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

	checkX := func(t *testing.T, h int, k [filterTaps]int16, randomize bool) {
		ref := makeRef(maxInt2(w, h), randomize)
		got, _ := testPlane(w, h, 1, w)
		want, _ := testPlane(w, h, 1, w)
		convolveX8NEON(got, ref, 0, 0, pad, pad, w, h, k)
		convolveX8PureGo(want, ref, 0, 0, pad, pad, w, h, k)
		assertClampedEqual(t, got, want, w, h)
	}
	checkY := func(t *testing.T, h int, k [filterTaps]int16, randomize bool) {
		ref := makeRef(maxInt2(w, h), randomize)
		got, _ := testPlane(w, h, 1, w)
		want, _ := testPlane(w, h, 1, w)
		convolveY8NEON(got, ref, 0, 0, pad, pad, w, h, k)
		convolveY8PureGo(want, ref, 0, 0, pad, pad, w, h, k)
		assertClampedEqual(t, got, want, w, h)
	}
	check2D := func(t *testing.T, h int, xk, yk [filterTaps]int16, randomize bool) {
		ref := makeRef(maxInt2(w, h), randomize)
		got, _ := testPlane(w, h, 1, w)
		want, _ := testPlane(w, h, 1, w)
		convolve2D8NEON(got, ref, 0, 0, pad, pad, w, h, xk, yk)
		convolve2D8PureGo(want, ref, 0, 0, pad, pad, w, h, xk, yk)
		assertClampedEqual(t, got, want, w, h)
	}

	for _, h := range heights {
		for _, randomize := range []bool{false, true} {
			// 8-tap: sweep all 16 phases for the first table; representative for rest.
			for ti, tbl := range eightTables {
				phases := []int{3, 5}
				if ti == 0 {
					phases = nil
					for p := 0; p < 16; p++ {
						phases = append(phases, p)
					}
				}
				for _, p := range phases {
					checkX(t, h, tbl[p], randomize)
					checkY(t, h, tbl[p], randomize)
				}
				for _, sx := range []int{1, 3, 7, 12} {
					for _, sy := range []int{2, 5, 9, 15} {
						check2D(t, h, tbl[sx], tbl[sy], randomize)
					}
				}
			}
			// 4-tap kernels (zero end taps) through the same asm.
			for _, tbl := range fourTables {
				for _, p := range []int{1, 3, 6, 11, 14} {
					checkX(t, h, tbl[p], randomize)
					checkY(t, h, tbl[p], randomize)
				}
				check2D(t, h, tbl[3], tbl[5], randomize)
				check2D(t, h, tbl[7], tbl[12], randomize)
			}
			// Mixed 4-tap X / 8-tap Y (and vice versa) for 2D.
			check2D(t, h, subpelFilters4[3], subpelFilters8[5], randomize)
			check2D(t, h, subpelFilters8[3], subpelFilters4[5], randomize)
		}
	}
}

// TestConvolveW4ClampedNEONMatchesPureGo asserts the width-4 *ClampedNEON
// wrappers match the clamped pure-Go reference for interior (resident halo) and
// genuine edge placements (where the tap window falls off the frame).
func TestConvolveW4ClampedNEONMatchesPureGo(t *testing.T) {
	const planeW, planeH = 48, 48
	ref, _ := testPlane(planeW, planeH, 1, planeW)
	fillMotionTestPlane(ref)

	xk8 := subpelFilters8[3]
	yk8 := subpelFilters8[5]
	xk4 := subpelFilters4[3]
	yk4 := subpelFilters4[5]

	type wcase struct {
		name          string
		refX, refY    int
		height        int
		four          bool
	}
	cases := []wcase{
		// Interior: full tap window resident, clamp is a no-op -> hits the NEON
		// resident fast path.
		{"interior_4x4_8tap", 20, 20, 4, false},
		{"interior_4x8_8tap", 20, 16, 8, false},
		{"interior_4x16_8tap", 20, 12, 16, false},
		{"interior_4x4_4tap", 20, 20, 4, true},
		{"interior_4x16_4tap", 20, 12, 16, true},
		// Genuine edges -> off-frame taps -> pure-Go clamped fallback.
		{"corner_4x4_8tap", 0, 0, 4, false},
		{"corner_4x16_4tap", 0, 0, 16, true},
		{"left_4x8_8tap", 0, 20, 8, false},
		{"top_4x8_8tap", 20, 0, 8, false},
		{"br_4x4_8tap", planeW - 4, planeH - 4, 4, false},
		{"right_straddle_4x8_8tap", planeW - 2, 16, 8, false},
		{"bottom_straddle_4x8_8tap", 16, planeH - 2, 8, false},
	}

	pick := func(four bool, eight, fourK [filterTaps]int16) [filterTaps]int16 {
		if four {
			return fourK
		}
		return eight
	}

	for _, c := range cases {
		xk := pick(c.four, xk8, xk4)
		yk := pick(c.four, yk8, yk4)
		const w = 4
		t.Run("X/"+c.name, func(t *testing.T) {
			got, _ := testPlane(w, c.height, 1, w)
			want, _ := testPlane(w, c.height, 1, w)
			convolveX8ClampedNEON(got, ref, 0, 0, c.refX, c.refY, w, c.height, xk)
			convolveX8ClampedPureGo(want, ref, 0, 0, c.refX, c.refY, w, c.height, xk)
			assertClampedEqual(t, got, want, w, c.height)
		})
		t.Run("Y/"+c.name, func(t *testing.T) {
			got, _ := testPlane(w, c.height, 1, w)
			want, _ := testPlane(w, c.height, 1, w)
			convolveY8ClampedNEON(got, ref, 0, 0, c.refX, c.refY, w, c.height, yk)
			convolveY8ClampedPureGo(want, ref, 0, 0, c.refX, c.refY, w, c.height, yk)
			assertClampedEqual(t, got, want, w, c.height)
		})
		t.Run("2D/"+c.name, func(t *testing.T) {
			got, _ := testPlane(w, c.height, 1, w)
			want, _ := testPlane(w, c.height, 1, w)
			convolve2D8ClampedNEON(got, ref, 0, 0, c.refX, c.refY, w, c.height, xk, yk)
			convolve2D8ClampedPureGo(want, ref, 0, 0, c.refX, c.refY, w, c.height, xk, yk)
			assertClampedEqual(t, got, want, w, c.height)
		})
	}
}

func maxInt2(a, b int) int {
	if a > b {
		return a
	}
	return b
}
