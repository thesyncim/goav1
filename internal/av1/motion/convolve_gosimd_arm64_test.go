// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func TestConvolveX8GoSIMDMatchesPureGo(t *testing.T) {
	tables := [][16][filterTaps]int16{subpelFilters8, subpelFilters8Sharp, subpelFilters8Smooth, subpelFilters4, subpelFilters4Smooth, bilinearFilters}
	sizes := []struct{ w, h int }{
		{8, 8}, {8, 4}, {16, 16}, {16, 8}, {24, 32}, {32, 32}, {32, 16}, {64, 64}, {40, 8}, {8, 1}, {8, 2},
	}
	for ti, table := range tables {
		for _, size := range sizes {
			for subX := 0; subX < 16; subX++ {
				kernel := table[subX]
				pad := filterTaps
				refSide := size.w + 2*pad
				refH := size.h + 2*pad
				ref, _ := testPlane(refSide, refH, 1, refSide)
				fillMotionTestPlane(ref)
				want, _ := testPlane(size.w, size.h, 1, size.w)
				got, _ := testPlane(size.w, size.h, 1, size.w)
				convolveX8PureGo(want, ref, 0, 0, pad, pad, size.w, size.h, kernel)
				convolveX8GoSIMD(got, ref, 0, 0, pad, pad, size.w, size.h, kernel)
				for i := range want.Pix {
					if want.Pix[i] != got.Pix[i] {
						t.Fatalf("table %d size %dx%d subX %d: mismatch at %d: got %d want %d", ti, size.w, size.h, subX, i, got.Pix[i], want.Pix[i])
					}
				}
			}
		}
	}
}

func TestConvolve2D8GoSIMDMatchesPureGo(t *testing.T) {
	tables := [][16][filterTaps]int16{subpelFilters8, subpelFilters8Sharp, subpelFilters8Smooth, subpelFilters4, subpelFilters4Smooth, bilinearFilters}
	sizes := []struct{ w, h int }{
		{8, 8}, {8, 4}, {16, 16}, {16, 8}, {24, 32}, {32, 32}, {32, 16}, {64, 64}, {40, 8}, {8, 1}, {8, 2},
	}
	for ti, table := range tables {
		for _, size := range sizes {
			// Sweep a representative set of (subX, subY) phase pairs.
			for _, subX := range []int{0, 1, 3, 7, 8, 11, 15} {
				for _, subY := range []int{0, 2, 5, 8, 13, 15} {
					xk := table[subX]
					yk := table[subY]
					pad := filterTaps
					refSide := size.w + 2*pad
					refH := size.h + 2*pad
					ref, _ := testPlane(refSide, refH, 1, refSide)
					fillMotionTestPlane(ref)
					want, _ := testPlane(size.w, size.h, 1, size.w)
					got, _ := testPlane(size.w, size.h, 1, size.w)
					var scratch ConvolveScratch
					convolve2D8PureGo(want, ref, 0, 0, pad, pad, size.w, size.h, xk, yk)
					convolve2D8GoSIMDScratch(got, ref, 0, 0, pad, pad, size.w, size.h, xk, yk, &scratch)
					for i := range want.Pix {
						if want.Pix[i] != got.Pix[i] {
							t.Fatalf("table %d size %dx%d subX %d subY %d: mismatch at %d: got %d want %d", ti, size.w, size.h, subX, subY, i, got.Pix[i], want.Pix[i])
						}
					}
				}
			}
		}
	}
}

// TestConvolve2D8GoSIMDExtremeInputs hardens the byte-exact gate with saturating
// inputs: all-zero, all-max (255), and an alternating min/max checkerboard drive
// the staged rounding and the [0,255] clip to both rails, where any off-by-one in
// the SQRSHRN saturation or the folded roundOffset would surface.
func TestConvolve2D8GoSIMDExtremeInputs(t *testing.T) {
	fillers := map[string]func(p frame.Plane){
		"zero": func(p frame.Plane) {
			for i := range p.Pix {
				p.Pix[i] = 0
			}
		},
		"max": func(p frame.Plane) {
			for i := range p.Pix {
				p.Pix[i] = 255
			}
		},
		"checker": func(p frame.Plane) {
			for y := 0; y < p.Height; y++ {
				for x := 0; x < p.Width; x++ {
					if (x+y)&1 == 0 {
						p.Pix[y*p.Stride+x] = 0
					} else {
						p.Pix[y*p.Stride+x] = 255
					}
				}
			}
		},
	}
	tables := [][16][filterTaps]int16{subpelFilters8, subpelFilters8Sharp, subpelFilters8Smooth}
	for name, fill := range fillers {
		for ti, table := range tables {
			for _, subX := range []int{1, 8, 15} {
				for _, subY := range []int{2, 8, 13} {
					const w, h = 32, 16
					pad := filterTaps
					refSide := w + 2*pad
					refH := h + 2*pad
					ref, _ := testPlane(refSide, refH, 1, refSide)
					fill(ref)
					want, _ := testPlane(w, h, 1, w)
					got, _ := testPlane(w, h, 1, w)
					var scratch ConvolveScratch
					convolve2D8PureGo(want, ref, 0, 0, pad, pad, w, h, table[subX], table[subY])
					convolve2D8GoSIMDScratch(got, ref, 0, 0, pad, pad, w, h, table[subX], table[subY], &scratch)
					for i := range want.Pix {
						if want.Pix[i] != got.Pix[i] {
							t.Fatalf("%s table %d subX %d subY %d: mismatch at %d: got %d want %d", name, ti, subX, subY, i, got.Pix[i], want.Pix[i])
						}
					}
				}
			}
		}
	}
}

func BenchmarkConvolve2D8GoSIMDWithScratch_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	var scratch ConvolveScratch
	runConvolveBench(b, 32, 32, func() {
		convolve2D8GoSIMDScratch(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk, yk, &scratch)
	})
}

func BenchmarkConvolveX8GoSIMD_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters8[3]
	runConvolveBench(b, 32, 32, func() {
		convolveX8GoSIMD(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk)
	})
}

func BenchmarkConvolveY8GoSIMD_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	yk := subpelFilters8[5]
	runConvolveBench(b, 32, 32, func() {
		convolveY8GoSIMD(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, yk)
	})
}

// TestConvolveY8GoSIMDMatchesPureGo checks the Go-native SIMD vertical convolve
// is byte-identical to the pure-Go reference over every subpel phase and a range
// of block shapes, including hardened extreme-input planes.
func TestConvolveY8GoSIMDMatchesPureGo(t *testing.T) {
	tables := [][16][filterTaps]int16{subpelFilters8, subpelFilters8Sharp, subpelFilters8Smooth, subpelFilters4, subpelFilters4Smooth, bilinearFilters}
	sizes := []struct{ w, h int }{
		{8, 8}, {8, 4}, {16, 16}, {16, 8}, {24, 32}, {32, 32}, {32, 16}, {64, 64}, {40, 8}, {8, 1}, {8, 2},
	}
	for ti, table := range tables {
		for _, size := range sizes {
			for subY := 0; subY < 16; subY++ {
				kernel := table[subY]
				pad := filterTaps
				refSide := size.w + 2*pad
				refH := size.h + 2*pad
				ref, _ := testPlane(refSide, refH, 1, refSide)
				fillMotionTestPlane(ref)
				want, _ := testPlane(size.w, size.h, 1, size.w)
				got, _ := testPlane(size.w, size.h, 1, size.w)
				convolveY8PureGo(want, ref, 0, 0, pad, pad, size.w, size.h, kernel)
				convolveY8GoSIMD(got, ref, 0, 0, pad, pad, size.w, size.h, kernel)
				for i := range want.Pix {
					if want.Pix[i] != got.Pix[i] {
						t.Fatalf("table %d size %dx%d subY %d: mismatch at %d: got %d want %d", ti, size.w, size.h, subY, i, got.Pix[i], want.Pix[i])
					}
				}
			}
		}
	}
}
