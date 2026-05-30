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

// makeHighBDRef builds a (side+2*pad)-square high-bit-depth reference plane
// whose samples are valid for the given bit depth.
func makeHighBDRef(side, pad int, max uint16, randomize bool, rng *rand.Rand) frame.Plane {
	refSide := side + 2*pad
	r, _ := testPlane(refSide, refSide, 2, refSide*2)
	if randomize {
		for y := 0; y < refSide; y++ {
			for x := 0; x < refSide; x++ {
				setSample(r, 2, x, y, uint16(rng.Intn(int(max)+1)))
			}
		}
	} else {
		fillHighBDMotionTestPlane(r, max)
	}
	return r
}

func eqHighBDBlock(t *testing.T, got, want frame.Plane, w, h int, tag string, ctx ...any) {
	t.Helper()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g := getSample(got, 2, x, y)
			e := getSample(want, 2, x, y)
			if g != e {
				t.Fatalf("%s (%d,%d): NEON=%d PureGo=%d ctx=%v", tag, x, y, g, e, ctx)
			}
		}
	}
}

// TestConvolveHighBDNEONMatchesPureGo asserts the high-bit-depth NEON X/Y/2D
// convolves are bit-identical to their pure-Go references for every width/height
// in 4..64, every subpel phase of each filter type, at bit depths 10 and 12,
// over deterministic and random reference pixels.
func TestConvolveHighBDNEONMatchesPureGo(t *testing.T) {
	tables := convolve2DKernelTables()
	rng := rand.New(rand.NewSource(0x4b1d))
	const pad = filterTaps
	sweep := []int{4, 8, 12, 16, 24, 32, 48, 64}
	bds := []uint8{10, 12}

	for _, bd := range bds {
		max, _ := highBDMax(bd)
		// Full size sweep with a representative phase per filter type.
		for _, tbl := range tables {
			xk := tbl[3]
			yk := tbl[5]
			for _, w := range sweep {
				for _, h := range sweep {
					for _, randomize := range []bool{false, true} {
						side := w
						if h > side {
							side = h
						}
						ref := makeHighBDRef(side, pad, max, randomize, rng)
						// X
						gx, _ := testPlane(w, h, 2, w*2)
						ex, _ := testPlane(w, h, 2, w*2)
						convolveXHighBDNEON(gx, ref, bd, max, 0, 0, pad, pad, w, h, xk)
						convolveXHighBDPureGo(ex, ref, bd, max, 0, 0, pad, pad, w, h, xk)
						eqHighBDBlock(t, gx, ex, w, h, "X", bd, w, h)
						// Y
						gy, _ := testPlane(w, h, 2, w*2)
						ey, _ := testPlane(w, h, 2, w*2)
						convolveYHighBDNEON(gy, ref, bd, max, 0, 0, pad, pad, w, h, yk)
						convolveYHighBDPureGo(ey, ref, bd, max, 0, 0, pad, pad, w, h, yk)
						eqHighBDBlock(t, gy, ey, w, h, "Y", bd, w, h)
						// 2D
						g2, _ := testPlane(w, h, 2, w*2)
						e2, _ := testPlane(w, h, 2, w*2)
						convolve2DHighBDNEON(g2, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
						convolve2DHighBDPureGo(e2, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
						eqHighBDBlock(t, g2, e2, w, h, "2D", bd, w, h)
					}
				}
			}
		}

		// All subpel phases per filter type on a fixed 16x16 shape.
		ref := makeHighBDRef(16, pad, max, false, rng)
		for _, tbl := range tables {
			for sp := 0; sp < 16; sp++ {
				k := tbl[sp]
				gx, _ := testPlane(16, 16, 2, 32)
				ex, _ := testPlane(16, 16, 2, 32)
				convolveXHighBDNEON(gx, ref, bd, max, 0, 0, pad, pad, 16, 16, k)
				convolveXHighBDPureGo(ex, ref, bd, max, 0, 0, pad, pad, 16, 16, k)
				eqHighBDBlock(t, gx, ex, 16, 16, "Xphase", bd, sp)

				gy, _ := testPlane(16, 16, 2, 32)
				ey, _ := testPlane(16, 16, 2, 32)
				convolveYHighBDNEON(gy, ref, bd, max, 0, 0, pad, pad, 16, 16, k)
				convolveYHighBDPureGo(ey, ref, bd, max, 0, 0, pad, pad, 16, 16, k)
				eqHighBDBlock(t, gy, ey, 16, 16, "Yphase", bd, sp)

				g2, _ := testPlane(16, 16, 2, 32)
				e2, _ := testPlane(16, 16, 2, 32)
				convolve2DHighBDNEON(g2, ref, bd, max, 0, 0, pad, pad, 16, 16, k, k)
				convolve2DHighBDPureGo(e2, ref, bd, max, 0, 0, pad, pad, 16, 16, k, k)
				eqHighBDBlock(t, g2, e2, 16, 16, "2Dphase", bd, sp)
			}
		}
	}
}

// TestConvolveHighBDNEONZeroAlloc asserts the HBD NEON wrappers allocate nothing
// on the fast (asm) path.
func TestConvolveHighBDNEONZeroAlloc(t *testing.T) {
	const pad = filterTaps
	max, _ := highBDMax(10)
	rng := rand.New(rand.NewSource(7))
	ref := makeHighBDRef(32, pad, max, true, rng)
	dst, _ := testPlane(32, 32, 2, 64)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	cases := []struct {
		name string
		fn   func()
	}{
		{"X", func() { convolveXHighBDNEON(dst, ref, 10, max, 0, 0, pad, pad, 32, 32, xk) }},
		{"Y", func() { convolveYHighBDNEON(dst, ref, 10, max, 0, 0, pad, pad, 32, 32, yk) }},
		{"2D", func() { convolve2DHighBDNEON(dst, ref, 10, max, 0, 0, pad, pad, 32, 32, xk, yk) }},
	}
	for _, c := range cases {
		if allocs := testing.AllocsPerRun(20, c.fn); allocs != 0 {
			t.Errorf("%s HBD NEON allocated %v times, want 0", c.name, allocs)
		}
	}
}

// TestConvolveClampedNEONMatchesPureGo asserts the edge-clamped NEON wrappers
// (8-bit and high-bit-depth) stay bit-identical to the pure-Go clamped
// references at genuine frame edges, where the tap window falls off the plane
// and the wrapper must use the per-tap-clamping pure-Go path. It also covers the
// in-bounds case where the wrapper routes to the fast NEON kernel.
func TestConvolveClampedNEONMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xc1a))
	sizes := []int{4, 8, 12, 16, 32}

	// 8-bit clamped: place the block at the plane origin so the negative-offset
	// halo clamps, plus an interior offset that stays resident.
	for _, w := range sizes {
		for _, h := range sizes {
			side := w
			if h > side {
				side = h
			}
			ref, _ := testPlane(side, side, 1, side)
			for i := range ref.Pix {
				ref.Pix[i] = byte(rng.Intn(256))
			}
			xk := subpelFilters8[6]
			yk := subpelFilters8[9]
			for _, org := range []int{0, 1} {
				gx, _ := testPlane(w, h, 1, w)
				ex, _ := testPlane(w, h, 1, w)
				convolveX8ClampedNEON(gx, ref, 0, 0, org, org, w, h, xk)
				convolveX8ClampedPureGo(ex, ref, 0, 0, org, org, w, h, xk)
				assertBytesEqual(t, gx, ex, w, h, "X8clamped", w, h, org)

				gy, _ := testPlane(w, h, 1, w)
				ey, _ := testPlane(w, h, 1, w)
				convolveY8ClampedNEON(gy, ref, 0, 0, org, org, w, h, yk)
				convolveY8ClampedPureGo(ey, ref, 0, 0, org, org, w, h, yk)
				assertBytesEqual(t, gy, ey, w, h, "Y8clamped", w, h, org)

				g2, _ := testPlane(w, h, 1, w)
				e2, _ := testPlane(w, h, 1, w)
				convolve2D8ClampedNEON(g2, ref, 0, 0, org, org, w, h, xk, yk)
				convolve2D8ClampedPureGo(e2, ref, 0, 0, org, org, w, h, xk, yk)
				assertBytesEqual(t, g2, e2, w, h, "2D8clamped", w, h, org)
			}
		}
	}

	// High-bit-depth clamped at bd 10 and 12.
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		for _, w := range sizes {
			for _, h := range sizes {
				side := w
				if h > side {
					side = h
				}
				ref, _ := testPlane(side, side, 2, side*2)
				for y := 0; y < side; y++ {
					for x := 0; x < side; x++ {
						setSample(ref, 2, x, y, uint16(rng.Intn(int(max)+1)))
					}
				}
				xk := subpelFilters8[6]
				yk := subpelFilters8[9]
				for _, org := range []int{0, 1} {
					gx, _ := testPlane(w, h, 2, w*2)
					ex, _ := testPlane(w, h, 2, w*2)
					convolveXHighBDClampedNEON(gx, ref, bd, max, 0, 0, org, org, w, h, xk)
					convolveXHighBDClampedPureGo(ex, ref, bd, max, 0, 0, org, org, w, h, xk)
					eqHighBDBlock(t, gx, ex, w, h, "XHBDclamped", bd, w, h, org)

					gy, _ := testPlane(w, h, 2, w*2)
					ey, _ := testPlane(w, h, 2, w*2)
					convolveYHighBDClampedNEON(gy, ref, bd, max, 0, 0, org, org, w, h, yk)
					convolveYHighBDClampedPureGo(ey, ref, bd, max, 0, 0, org, org, w, h, yk)
					eqHighBDBlock(t, gy, ey, w, h, "YHBDclamped", bd, w, h, org)

					g2, _ := testPlane(w, h, 2, w*2)
					e2, _ := testPlane(w, h, 2, w*2)
					convolve2DHighBDClampedNEON(g2, ref, bd, max, 0, 0, org, org, w, h, xk, yk)
					convolve2DHighBDClampedPureGo(e2, ref, bd, max, 0, 0, org, org, w, h, xk, yk)
					eqHighBDBlock(t, g2, e2, w, h, "2DHBDclamped", bd, w, h, org)
				}
			}
		}
	}
}

func assertBytesEqual(t *testing.T, got, want frame.Plane, w, h int, tag string, ctx ...any) {
	t.Helper()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g := got.Pix[y*got.Stride+x]
			e := want.Pix[y*want.Stride+x]
			if g != e {
				t.Fatalf("%s (%d,%d): NEON=%d PureGo=%d ctx=%v", tag, x, y, g, e, ctx)
			}
		}
	}
}
