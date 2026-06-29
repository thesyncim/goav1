// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package motion

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// These differential tests assert the AVX2 convolve kernels are bit-identical to
// the pure-Go references for every width/height and every subpel phase of each
// filter type. They call the AVX2 wrappers directly rather than through the
// dispatch slots, so they validate the asm even on hosts whose CPUID does not
// advertise AVX2 (e.g. amd64 under Rosetta 2, which translates AVX2 anyway). On
// a true non-AVX2 amd64 host these would fault; the harness that runs them is
// expected to be AVX2-capable (real CI) or AVX2-translating (Rosetta).

func avx2FilterTables() [][16][filterTaps]int16 {
	return [][16][filterTaps]int16{
		subpelFilters8,
		subpelFilters8Smooth,
		subpelFilters8Sharp,
		bilinearFilters,
	}
}

func TestConvolveX8AVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA1A2B3C4))
	const pad = filterTaps
	sizes := []int{8, 16, 24, 32, 48, 64}
	for _, tbl := range avx2FilterTables() {
		for ph := 0; ph < 16; ph++ {
			k := tbl[ph]
			for _, w := range sizes {
				for _, h := range []int{1, 4, 8, 17, 32} {
					side := w
					if h > side {
						side = h
					}
					ref := randPlane(rng, side+2*pad, 1)
					got, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					convolveX8AVX2(got, ref, 0, 0, pad, pad, w, h, k)
					convolveX8PureGo(want, ref, 0, 0, pad, pad, w, h, k)
					diffPlanes8(t, got, want, w, h, "X", k, k)
				}
			}
		}
	}
}

func TestConvolveY8AVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xB1B2C3D4))
	const pad = filterTaps
	sizes := []int{8, 16, 24, 32, 48, 64}
	for _, tbl := range avx2FilterTables() {
		for ph := 0; ph < 16; ph++ {
			k := tbl[ph]
			for _, w := range sizes {
				for _, h := range []int{1, 4, 8, 17, 32} {
					side := w
					if h > side {
						side = h
					}
					ref := randPlane(rng, side+2*pad, 1)
					got, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					convolveY8AVX2(got, ref, 0, 0, pad, pad, w, h, k)
					convolveY8PureGo(want, ref, 0, 0, pad, pad, w, h, k)
					diffPlanes8(t, got, want, w, h, "Y", k, k)
				}
			}
		}
	}
}

func TestConvolve2D8AVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x2d20feed))
	const pad = filterTaps
	sizes := []int{8, 16, 24, 32, 48, 64}
	tables := avx2FilterTables()

	// Full size sweep with one representative phase per filter type.
	for _, tbl := range tables {
		xk := tbl[3]
		yk := tbl[5]
		for _, w := range sizes {
			for _, h := range []int{4, 8, 16, 32} {
				side := w
				if h > side {
					side = h
				}
				ref := randPlane(rng, side+2*pad, 1)
				got, _ := testPlane(w, h, 1, w)
				want, _ := testPlane(w, h, 1, w)
				convolve2D8AVX2(got, ref, 0, 0, pad, pad, w, h, xk, yk)
				convolve2D8PureGo(want, ref, 0, 0, pad, pad, w, h, xk, yk)
				diffPlanes8(t, got, want, w, h, "2D", xk, yk)
			}
		}
	}

	// All 16x16 phase combinations on fixed shapes.
	for _, tbl := range tables {
		for sx := 0; sx < 16; sx++ {
			for sy := 0; sy < 16; sy++ {
				ref := randPlane(rng, 32+2*pad, 1)
				got, _ := testPlane(16, 16, 1, 16)
				want, _ := testPlane(16, 16, 1, 16)
				convolve2D8AVX2(got, ref, 0, 0, pad, pad, 16, 16, tbl[sx], tbl[sy])
				convolve2D8PureGo(want, ref, 0, 0, pad, pad, 16, 16, tbl[sx], tbl[sy])
				diffPlanes8(t, got, want, 16, 16, "2Dphase", tbl[sx], tbl[sy])
			}
		}
	}
}

func TestConvolve2D8ClampedEdgeSplitAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x2d8a11e))
	widths := []int{16, 24, 32}
	heights := []int{4, 8, 16, 32}
	phasePairs := [][2][filterTaps]int16{
		{subpelFilters8[3], subpelFilters8[5]},
		{subpelFilters8Smooth[6], subpelFilters8Smooth[11]},
		{subpelFilters8Sharp[9], subpelFilters8Sharp[13]},
		{bilinearFilters[7], bilinearFilters[2]},
	}

	for _, w := range widths {
		for _, h := range heights {
			for _, kernels := range phasePairs {
				for _, edge := range []string{"left", "right"} {
					const refW = 96
					refH := h + 2*filterTaps
					ref, _ := testPlane(refW, refH, 1, refW)
					for i := range ref.Pix {
						ref.Pix[i] = byte(rng.Intn(256))
					}
					refX := 1
					if edge == "right" {
						refX = refW - w - 3
					}
					refY := filterTaps
					got, _ := testPlane(w, h, 1, w)
					gotScratch, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					var scratch ConvolveScratch
					if !convolve2D8ClampedEdgeSplitAVX2WithScratch(got, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1], nil) {
						t.Fatalf("2D8horizontal-edge AVX2 split path was not used w=%d h=%d edge=%s", w, h, edge)
					}
					if !convolve2D8ClampedEdgeSplitAVX2WithScratch(gotScratch, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1], &scratch) {
						t.Fatalf("2D8horizontal-edge AVX2 scratch split path was not used w=%d h=%d edge=%s", w, h, edge)
					}
					convolve2D8ClampedPureGo(want, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1])
					diffPlanes8(t, got, want, w, h, "2Dclamped-edge", kernels[0], kernels[1])
					diffPlanes8(t, gotScratch, want, w, h, "2Dclamped-edge-scratch", kernels[0], kernels[1])
				}
			}
		}
	}
}

func TestConvolveHighBDAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6BD0FACE))
	const pad = filterTaps
	sizes := []int{4, 8, 12, 16, 24, 32}
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << bd) - 1)
		for _, tbl := range avx2FilterTables() {
			for ph := 0; ph < 16; ph++ {
				k := tbl[ph]
				for _, w := range sizes {
					for _, h := range []int{1, 4, 8, 16} {
						side := w
						if h > side {
							side = h
						}
						ref := randPlaneHBD(rng, side+2*pad, max)
						// X
						gx, _ := testPlane(w, h, 2, w*2)
						wx, _ := testPlane(w, h, 2, w*2)
						convolveXHighBDAVX2(gx, ref, bd, max, 0, 0, pad, pad, w, h, k)
						convolveXHighBDPureGo(wx, ref, bd, max, 0, 0, pad, pad, w, h, k)
						diffPlanesHBD(t, gx, wx, w, h, "Xhbd", bd)
						// Y
						gy, _ := testPlane(w, h, 2, w*2)
						wy, _ := testPlane(w, h, 2, w*2)
						convolveYHighBDAVX2(gy, ref, bd, max, 0, 0, pad, pad, w, h, k)
						convolveYHighBDPureGo(wy, ref, bd, max, 0, 0, pad, pad, w, h, k)
						diffPlanesHBD(t, gy, wy, w, h, "Yhbd", bd)
					}
				}
			}
		}
	}
}

func TestConvolve2DHighBDAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x123abc))
	const pad = filterTaps
	sizes := []int{4, 8, 16, 32}
	tables := avx2FilterTables()
	for _, bd := range []uint8{10, 12} {
		max := uint16((1 << bd) - 1)
		// size sweep
		for _, tbl := range tables {
			xk := tbl[3]
			yk := tbl[5]
			for _, w := range sizes {
				for _, h := range []int{4, 8, 16} {
					side := w
					if h > side {
						side = h
					}
					ref := randPlaneHBD(rng, side+2*pad, max)
					g, _ := testPlane(w, h, 2, w*2)
					wn, _ := testPlane(w, h, 2, w*2)
					convolve2DHighBDAVX2(g, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
					convolve2DHighBDPureGo(wn, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
					diffPlanesHBD(t, g, wn, w, h, "2Dhbd", bd)
				}
			}
		}
		// all phases on a fixed shape
		for _, tbl := range tables {
			for sx := 0; sx < 16; sx++ {
				for sy := 0; sy < 16; sy++ {
					ref := randPlaneHBD(rng, 16+2*pad, max)
					g, _ := testPlane(8, 8, 2, 16)
					wn, _ := testPlane(8, 8, 2, 16)
					convolve2DHighBDAVX2(g, ref, bd, max, 0, 0, pad, pad, 8, 8, tbl[sx], tbl[sy])
					convolve2DHighBDPureGo(wn, ref, bd, max, 0, 0, pad, pad, 8, 8, tbl[sx], tbl[sy])
					diffPlanesHBD(t, g, wn, 8, 8, "2Dhbdphase", bd)
				}
			}
		}
	}
}

// TestConvolveAVX2ZeroAlloc asserts the AVX2 fast paths allocate nothing.
func TestConvolveAVX2ZeroAlloc(t *testing.T) {
	const pad = filterTaps
	rng := rand.New(rand.NewSource(1))
	ref := randPlane(rng, 32+2*pad, 1)
	dst, _ := testPlane(32, 32, 1, 32)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	if a := testing.AllocsPerRun(20, func() {
		convolve2D8AVX2(dst, ref, 0, 0, pad, pad, 32, 32, xk, yk)
	}); a != 0 {
		t.Fatalf("convolve2D8AVX2 allocated %v times, want 0", a)
	}
	if a := testing.AllocsPerRun(20, func() {
		convolveX8AVX2(dst, ref, 0, 0, pad, pad, 32, 32, xk)
	}); a != 0 {
		t.Fatalf("convolveX8AVX2 allocated %v times, want 0", a)
	}
}

func randPlane(rng *rand.Rand, side, bps int) frame.Plane {
	p, _ := testPlane(side, side, bps, side*bps)
	for i := range p.Pix {
		p.Pix[i] = byte(rng.Intn(256))
	}
	return p
}

func randPlaneHBD(rng *rand.Rand, side int, max uint16) frame.Plane {
	p, _ := testPlane(side, side, 2, side*2)
	for i := 0; i+1 < len(p.Pix); i += 2 {
		v := uint16(rng.Intn(int(max) + 1))
		p.Pix[i] = byte(v)
		p.Pix[i+1] = byte(v >> 8)
	}
	return p
}

func diffPlanes8(t *testing.T, got, want frame.Plane, w, h int, tag string, xk, yk [filterTaps]int16) {
	t.Helper()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g := got.Pix[y*got.Stride+x]
			e := want.Pix[y*want.Stride+x]
			if g != e {
				t.Fatalf("%s w=%d h=%d (%d,%d): AVX2=%d PureGo=%d xk=%v yk=%v", tag, w, h, x, y, g, e, xk, yk)
			}
		}
	}
}

func diffPlanesHBD(t *testing.T, got, want frame.Plane, w, h int, tag string, bd uint8) {
	t.Helper()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			o := y*got.Stride + x*2
			g := uint16(got.Pix[o]) | uint16(got.Pix[o+1])<<8
			ow := y*want.Stride + x*2
			e := uint16(want.Pix[ow]) | uint16(want.Pix[ow+1])<<8
			if g != e {
				t.Fatalf("%s bd=%d w=%d h=%d (%d,%d): AVX2=%d PureGo=%d", tag, bd, w, h, x, y, g, e)
			}
		}
	}
}
