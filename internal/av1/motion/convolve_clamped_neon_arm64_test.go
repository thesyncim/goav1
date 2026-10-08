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

}

func TestConvolve1D8ClampedEdgeNEONMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x1d8c1a))
	xWidths := []int{4, 8, 12, 15, 16, 24, 32}
	yWidths := []int{4, 8, 16, 24, 32}
	heights := []int{4, 8, 12, 15, 16, 24, 32}
	kernels := [][filterTaps]int16{
		subpelFilters8[3],
		subpelFilters8Smooth[6],
		subpelFilters8Sharp[9],
		bilinearFilters[7],
	}

	for _, w := range xWidths {
		for _, h := range heights {
			for _, k := range kernels {
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
					want, _ := testPlane(w, h, 1, w)
					convolveX8ClampedNEON(got, ref, 0, 0, refX, refY, w, h, k)
					convolveX8ClampedPureGo(want, ref, 0, 0, refX, refY, w, h, k)
					assertBytesEqual(t, got, want, w, h, "X8horizontal-edge", w, h, edge)
				}
			}
		}
	}

	for _, w := range yWidths {
		for _, h := range heights {
			for _, k := range kernels {
				for _, edge := range []string{"top", "bottom"} {
					refW := w + 2*filterTaps
					const refH = 96
					ref, _ := testPlane(refW, refH, 1, refW)
					for i := range ref.Pix {
						ref.Pix[i] = byte(rng.Intn(256))
					}
					refX := filterTaps
					refY := 1
					if edge == "bottom" {
						refY = refH - h - 3
					}
					got, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					convolveY8ClampedNEON(got, ref, 0, 0, refX, refY, w, h, k)
					convolveY8ClampedPureGo(want, ref, 0, 0, refX, refY, w, h, k)
					assertBytesEqual(t, got, want, w, h, "Y8vertical-edge", w, h, edge)
				}
			}
		}
	}
}

func TestConvolve2D8ClampedHorizontalEdgeNEONMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x2d8c1a))
	widths := []int{8, 12, 15, 16, 24, 32}
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
					gotSplit, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					var scratch ConvolveScratch
					var splitScratch ConvolveScratch
					convolve2D8ClampedNEON(got, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1])
					convolve2D8ClampedNEONWithScratch(gotScratch, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1], &scratch)
					if !convolve2D8ClampedEdgeSplitNEONWithScratch(gotSplit, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1], &splitScratch) {
						t.Fatalf("2D8horizontal-edge split path was not used w=%d h=%d edge=%s", w, h, edge)
					}
					convolve2D8ClampedPureGo(want, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1])
					assertBytesEqual(t, got, want, w, h, "2D8horizontal-edge", w, h, edge)
					assertBytesEqual(t, gotScratch, want, w, h, "2D8horizontal-edge-scratch", w, h, edge)
					assertBytesEqual(t, gotSplit, want, w, h, "2D8horizontal-edge-split", w, h, edge)
				}
			}
		}
	}

	const refW = 64
	const w = 16
	const h = 16
	ref, _ := testPlane(refW, h+2*filterTaps, 1, refW)
	for i := range ref.Pix {
		ref.Pix[i] = byte(rng.Intn(256))
	}
	dst, _ := testPlane(w, h, 1, w)
	var scratch ConvolveScratch
	allocs := testing.AllocsPerRun(50, func() {
		convolve2D8ClampedNEONWithScratch(dst, ref, 0, 0, refW-w-3, filterTaps, w, h, subpelFilters8[3], subpelFilters8[5], &scratch)
	})
	if allocs != 0 {
		t.Fatalf("convolve2D8ClampedNEONWithScratch horizontal edge allocated %v times, want 0", allocs)
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
