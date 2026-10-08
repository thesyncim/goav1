// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package motion

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

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

// TestConvolveHighBDGoSIMDMatchesPureGo asserts the high-bit-depth NEON X/Y/2D
// convolves are bit-identical to their pure-Go references for every width/height
// in 4..64, every subpel phase of each filter type, at bit depths 10 and 12,
// over deterministic and random reference pixels.
func TestConvolveHighBDGoSIMDMatchesPureGo(t *testing.T) {
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
						convolveXHighBDGoSIMD(gx, ref, bd, max, 0, 0, pad, pad, w, h, xk)
						convolveXHighBDPureGo(ex, ref, bd, max, 0, 0, pad, pad, w, h, xk)
						eqHighBDBlock(t, gx, ex, w, h, "X", bd, w, h)
						// Y
						gy, _ := testPlane(w, h, 2, w*2)
						ey, _ := testPlane(w, h, 2, w*2)
						convolveYHighBDGoSIMD(gy, ref, bd, max, 0, 0, pad, pad, w, h, yk)
						convolveYHighBDPureGo(ey, ref, bd, max, 0, 0, pad, pad, w, h, yk)
						eqHighBDBlock(t, gy, ey, w, h, "Y", bd, w, h)
						// 2D
						g2, _ := testPlane(w, h, 2, w*2)
						e2, _ := testPlane(w, h, 2, w*2)
						convolve2DHighBDGoSIMD(g2, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
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
				convolveXHighBDGoSIMD(gx, ref, bd, max, 0, 0, pad, pad, 16, 16, k)
				convolveXHighBDPureGo(ex, ref, bd, max, 0, 0, pad, pad, 16, 16, k)
				eqHighBDBlock(t, gx, ex, 16, 16, "Xphase", bd, sp)

				gy, _ := testPlane(16, 16, 2, 32)
				ey, _ := testPlane(16, 16, 2, 32)
				convolveYHighBDGoSIMD(gy, ref, bd, max, 0, 0, pad, pad, 16, 16, k)
				convolveYHighBDPureGo(ey, ref, bd, max, 0, 0, pad, pad, 16, 16, k)
				eqHighBDBlock(t, gy, ey, 16, 16, "Yphase", bd, sp)

				g2, _ := testPlane(16, 16, 2, 32)
				e2, _ := testPlane(16, 16, 2, 32)
				convolve2DHighBDGoSIMD(g2, ref, bd, max, 0, 0, pad, pad, 16, 16, k, k)
				convolve2DHighBDPureGo(e2, ref, bd, max, 0, 0, pad, pad, 16, 16, k, k)
				eqHighBDBlock(t, g2, e2, 16, 16, "2Dphase", bd, sp)
			}
		}
	}
}

// TestConvolveHighBDGoSIMDZeroAlloc asserts the HBD NEON wrappers allocate nothing
// on the fast (asm) path.
func TestConvolveHighBDGoSIMDZeroAlloc(t *testing.T) {
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
		{"X", func() { convolveXHighBDGoSIMD(dst, ref, 10, max, 0, 0, pad, pad, 32, 32, xk) }},
		{"Y", func() { convolveYHighBDGoSIMD(dst, ref, 10, max, 0, 0, pad, pad, 32, 32, yk) }},
		{"2D", func() { convolve2DHighBDGoSIMD(dst, ref, 10, max, 0, 0, pad, pad, 32, 32, xk, yk) }},
	}
	for _, c := range cases {
		if allocs := testing.AllocsPerRun(20, c.fn); allocs != 0 {
			t.Errorf("%s HBD NEON allocated %v times, want 0", c.name, allocs)
		}
	}
}

// TestConvolveClampedNEONMatchesPureGo asserts the edge-clamped NEON wrappers
// (8-bit and high-bit-depth) stay bit-identical to the pure-Go clamped
// references at genuine frame edges, where the tap window falls off the plane.
// It also covers the in-bounds case where the wrapper routes to the fast NEON
// kernel.
func TestConvolveClampedHighBDGoSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xc1a))
	sizes := []int{4, 8, 12, 16, 32}
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
					convolveXHighBDClampedGoSIMD(gx, ref, bd, max, 0, 0, org, org, w, h, xk)
					convolveXHighBDClampedPureGo(ex, ref, bd, max, 0, 0, org, org, w, h, xk)
					eqHighBDBlock(t, gx, ex, w, h, "XHBDclamped", bd, w, h, org)

					gy, _ := testPlane(w, h, 2, w*2)
					ey, _ := testPlane(w, h, 2, w*2)
					convolveYHighBDClampedGoSIMD(gy, ref, bd, max, 0, 0, org, org, w, h, yk)
					convolveYHighBDClampedPureGo(ey, ref, bd, max, 0, 0, org, org, w, h, yk)
					eqHighBDBlock(t, gy, ey, w, h, "YHBDclamped", bd, w, h, org)

					g2, _ := testPlane(w, h, 2, w*2)
					e2, _ := testPlane(w, h, 2, w*2)
					convolve2DHighBDClampedGoSIMD(g2, ref, bd, max, 0, 0, org, org, w, h, xk, yk)
					convolve2DHighBDClampedPureGo(e2, ref, bd, max, 0, 0, org, org, w, h, xk, yk)
					eqHighBDBlock(t, g2, e2, w, h, "2DHBDclamped", bd, w, h, org)
				}
			}
		}
	}
}

// TestConvolveHighBDClampedEmuEdgeMatchesPureGo drives the HBD edge-clamped
// NEON wrappers over tap windows that overhang the reference plane on every
// side (and past both corners), forcing the emu_edge halo materialization
// (emuEdgeWindow16). Every shape must stay bit-identical to the pure-Go
// per-tap-clamping reference at bit depths 10 and 12, including the max/zero
// saturation edge values.
func TestConvolveHighBDClampedEmuEdgeMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x16bced9e))
	sizes := []int{4, 8, 12, 16, 32, 64}
	const refW, refH = 24, 20
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		ref, _ := testPlane(refW, refH, 2, refW*2)
		for y := 0; y < refH; y++ {
			for x := 0; x < refW; x++ {
				v := uint16(rng.Intn(int(max) + 1))
				// Seed the corners with the saturation extremes so the
				// clamped-replication halo carries 0 and max samples.
				if (x == 0 || x == refW-1) && (y == 0 || y == refH-1) {
					if (x+y)&1 == 0 {
						v = 0
					} else {
						v = max
					}
				}
				setSample(ref, 2, x, y, v)
			}
		}
		for _, w := range sizes {
			for _, h := range sizes {
				// Offsets that push the tap window past each plane boundary and
				// diagonally past the corners, plus an interior case.
				offs := [][2]int{
					{-w, -h}, {-3, -3}, {2, -5}, {refW - 2, -1},
					{refW - 1, 5}, {refW, refH}, {5, refH - 2},
					{-4, refH - 3}, {-6, 4}, {4, 4},
				}
				xk := subpelFilters8[6]
				yk := subpelFilters8[9]
				for _, o := range offs {
					rx, ry := o[0], o[1]
					gx, _ := testPlane(w, h, 2, w*2)
					ex, _ := testPlane(w, h, 2, w*2)
					convolveXHighBDClampedGoSIMD(gx, ref, bd, max, 0, 0, rx, ry, w, h, xk)
					convolveXHighBDClampedPureGo(ex, ref, bd, max, 0, 0, rx, ry, w, h, xk)
					eqHighBDBlock(t, gx, ex, w, h, "XHBDemu", bd, w, h, o)

					gy, _ := testPlane(w, h, 2, w*2)
					ey, _ := testPlane(w, h, 2, w*2)
					convolveYHighBDClampedGoSIMD(gy, ref, bd, max, 0, 0, rx, ry, w, h, yk)
					convolveYHighBDClampedPureGo(ey, ref, bd, max, 0, 0, rx, ry, w, h, yk)
					eqHighBDBlock(t, gy, ey, w, h, "YHBDemu", bd, w, h, o)

					g2, _ := testPlane(w, h, 2, w*2)
					e2, _ := testPlane(w, h, 2, w*2)
					convolve2DHighBDClampedGoSIMD(g2, ref, bd, max, 0, 0, rx, ry, w, h, xk, yk)
					convolve2DHighBDClampedPureGo(e2, ref, bd, max, 0, 0, rx, ry, w, h, xk, yk)
					eqHighBDBlock(t, g2, e2, w, h, "2DHBDemu", bd, w, h, o)
				}
			}
		}
	}
}

// TestConvolveHighBDClampedEmuEdgeZeroAlloc proves the emu_edge halo scratch
// stays on the stack (the func-ptr dispatch slot must not force it to the heap).
func TestConvolveHighBDClampedEmuEdgeZeroAlloc(t *testing.T) {
	max, _ := highBDMax(10)
	ref, _ := testPlane(24, 20, 2, 24*2)
	for y := 0; y < 20; y++ {
		for x := 0; x < 24; x++ {
			setSample(ref, 2, x, y, uint16((x*7+y*3)&int(max)))
		}
	}
	dst, _ := testPlane(64, 64, 2, 64*2)
	xk := subpelFilters8[6]
	yk := subpelFilters8[9]
	check := func(name string, fn func()) {
		if a := testing.AllocsPerRun(20, fn); a != 0 {
			t.Fatalf("%s allocated %v times, want 0", name, a)
		}
	}
	check("2DHBDemu", func() {
		convolve2DHighBDClampedGoSIMD(dst, ref, 10, max, 0, 0, -3, -3, 64, 64, xk, yk)
	})
	check("XHBDemu", func() {
		convolveXHighBDClampedGoSIMD(dst, ref, 10, max, 0, 0, -3, -3, 64, 64, xk)
	})
	check("YHBDemu", func() {
		convolveYHighBDClampedGoSIMD(dst, ref, 10, max, 0, 0, -3, -3, 64, 64, yk)
	})
}

// BenchmarkConvolveHighBDClampedEmuEdge measures the edge-overhanging HBD 2D
// clamped convolve: the NEON emu_edge path against the pure-Go per-tap clamp.
func BenchmarkConvolveHighBDClampedEmuEdge(b *testing.B) {
	max, _ := highBDMax(10)
	ref, _ := testPlane(24, 20, 2, 24*2)
	for y := 0; y < 20; y++ {
		for x := 0; x < 24; x++ {
			setSample(ref, 2, x, y, uint16((x*7+y*3)&int(max)))
		}
	}
	xk := subpelFilters8[6]
	yk := subpelFilters8[9]
	for _, w := range []int{8, 16, 32, 64} {
		dst, _ := testPlane(w, w, 2, w*2)
		b.Run("neon_"+itoaW(w), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				convolve2DHighBDClampedGoSIMD(dst, ref, 10, max, 0, 0, -3, -3, w, w, xk, yk)
			}
		})
		b.Run("purego_"+itoaW(w), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				convolve2DHighBDClampedPureGo(dst, ref, 10, max, 0, 0, -3, -3, w, w, xk, yk)
			}
		})
	}
}

func TestConvolveHighBDGoSIMDXYMatchesPureGo(t *testing.T) {
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
						convolveXHighBDGoSIMD(gx, ref, bd, max, 0, 0, pad, pad, w, h, k)
						convolveXHighBDPureGo(wx, ref, bd, max, 0, 0, pad, pad, w, h, k)
						diffPlanesHBD(t, gx, wx, w, h, "Xhbd", bd)
						// Y
						gy, _ := testPlane(w, h, 2, w*2)
						wy, _ := testPlane(w, h, 2, w*2)
						convolveYHighBDGoSIMD(gy, ref, bd, max, 0, 0, pad, pad, w, h, k)
						convolveYHighBDPureGo(wy, ref, bd, max, 0, 0, pad, pad, w, h, k)
						diffPlanesHBD(t, gy, wy, w, h, "Yhbd", bd)
					}
				}
			}
		}
	}
}

func TestConvolve2DHighBDGoSIMDSweepMatchesPureGo(t *testing.T) {
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
					convolve2DHighBDGoSIMD(g, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
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
					convolve2DHighBDGoSIMD(g, ref, bd, max, 0, 0, pad, pad, 8, 8, tbl[sx], tbl[sy])
					convolve2DHighBDPureGo(wn, ref, bd, max, 0, 0, pad, pad, 8, 8, tbl[sx], tbl[sy])
					diffPlanesHBD(t, g, wn, 8, 8, "2Dhbdphase", bd)
				}
			}
		}
	}
}

// TestConvolveAVX2ZeroAlloc asserts the AVX2 fast paths allocate nothing.
