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

func TestSIMDNarrowHorizontalBoundsAndRounding(t *testing.T) {
	tables := []struct {
		name    string
		filters [16][filterTaps]int16
	}{
		{"regular", subpelFilters8},
		{"sharp", subpelFilters8Sharp},
		{"smooth", subpelFilters8Smooth},
		{"4-tap", subpelFilters4},
		{"4-tap smooth", subpelFilters4Smooth},
		{"bilinear", bilinearFilters},
	}
	for _, table := range tables {
		t.Run(table.name, func(t *testing.T) {
			for phase, kernel := range table.filters {
				if _, _, ok := convolveX8I8MMFilter(kernel); !ok {
					t.Fatalf("phase %d is not a packed filter", phase)
				}
				if _, ok := simdNarrowHorizontalKernel(kernel); !ok {
					t.Fatalf("phase %d does not meet the narrow-MAC bound", phase)
				}
			}
		})
	}

	// The packing guard permits larger coefficients than the narrow accumulator
	// can safely hold. Keep that shape on the existing int32 widening path.
	wider := [filterTaps]int16{1: 254, 2: 254}
	if _, _, ok := convolveX8I8MMFilter(wider); !ok {
		t.Fatal("wide regression filter should remain in the packed-filter domain")
	}
	if _, ok := simdNarrowHorizontalKernel(wider); ok {
		t.Fatal("wide regression filter must not use the narrow-MAC path")
	}

	const maxHalfSum = 255 * 128
	for halfSum := -maxHalfSum; halfSum <= maxHalfSum; halfSum++ {
		// Collapsing the two 8-bit convolution rounds to one shift is exact
		// throughout the proven narrow-MAC range.
		staged := roundPowerOfTwo(roundPowerOfTwo(2*halfSum, round0Bits), filterBits-round0Bits)
		collapsed := (halfSum + 34) >> 6
		if collapsed != staged {
			t.Fatalf("half sum %d: collapsed X round %d, staged reference %d", halfSum, collapsed, staged)
		}
		firstStage := roundPowerOfTwo(2*halfSum, round0Bits)
		if firstStage < -1<<15 || firstStage > 1<<15-1 {
			t.Fatalf("half sum %d: X first stage %d overflows int16", halfSum, firstStage)
		}
		if got, want := roundPowerOfTwo(halfSum, compoundRound0Bits-1), roundPowerOfTwo(2*halfSum, compoundRound0Bits); got != want {
			t.Fatalf("half sum %d: collapsed compound round %d, reference %d", halfSum, got, want)
		}

		// The 2D horizontal bias can move after the half-sum round because the
		// full filter coefficients are exactly twice the half coefficients.
		xBias := 1 << (8 + filterBits - 1)
		staged2D := roundPowerOfTwo(2*halfSum+xBias, round0Bits)
		collapsed2D := roundPowerOfTwo(halfSum, round0Bits-1) + (xBias >> round0Bits)
		if collapsed2D != staged2D {
			t.Fatalf("half sum %d: collapsed 2D round %d, staged reference %d", halfSum, collapsed2D, staged2D)
		}
		if collapsed2D < -1<<15 || collapsed2D > 1<<15-1 {
			t.Fatalf("half sum %d: collapsed 2D intermediate %d overflows int16", halfSum, collapsed2D)
		}
	}
}

func TestHorizontalGoSIMDZeroAlloc(t *testing.T) {
	const w, h = 32, 32
	const pad = filterTaps
	refSide := w + 2*pad
	ref, _ := testPlane(refSide, h+2*pad, 1, refSide)
	fillMotionTestPlane(ref)
	dst, _ := testPlane(w, h, 1, w)
	kernel := subpelFilters8[3]
	yKernel := subpelFilters8[7]
	var convolveScratch ConvolveScratch
	var compoundScratch CompoundConvolveScratch
	var buf CompoundConvBuf
	out, ok := compoundConvBufView(&buf, w, h)
	if !ok {
		t.Fatal("invalid compound convolution buffer")
	}
	roundOffset := compoundRoundOffset8()

	cases := []struct {
		name string
		call func()
	}{
		{"convolveX dispatch", func() { convolveX8Impl(dst, ref, 0, 0, pad, pad, w, h, kernel) }},
		{"compoundX dispatch", func() { predictInterCompoundRef8ToConvBufXImpl(out, ref, pad, pad, w, h, kernel, roundOffset) }},
		{"convolve2D scratch dispatch", func() { convolve2D8WithScratchImpl(dst, ref, 0, 0, pad, pad, w, h, kernel, yKernel, &convolveScratch) }},
		{"compound2D scratch dispatch", func() {
			predictInterCompoundRef8ToConvBuf2DImpl(out, ref, pad, pad, w, h, kernel, yKernel, 19, &compoundScratch)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if allocs := testing.AllocsPerRun(30, tc.call); allocs != 0 {
				t.Fatalf("dispatched GoSIMD path allocated %.1f objects/run, want 0", allocs)
			}
		})
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

func TestConvolveY8GoSIMDUSDOTMatchesPureGo(t *testing.T) {
	tables := [][16][filterTaps]int16{subpelFilters8, subpelFilters8Sharp, subpelFilters8Smooth, subpelFilters4, subpelFilters4Smooth, bilinearFilters}
	sizes := []struct{ w, h int }{
		{8, 8}, {8, 4}, {16, 16}, {16, 8}, {24, 32}, {32, 32}, {32, 16}, {64, 64}, {40, 8}, {8, 12}, {8, 2}, {8, 6},
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
				convolveY8GoSIMDUSDOT(got, ref, 0, 0, pad, pad, size.w, size.h, kernel)
				for i := range want.Pix {
					if want.Pix[i] != got.Pix[i] {
						t.Fatalf("table %d size %dx%d subY %d: mismatch at %d: got %d want %d", ti, size.w, size.h, subY, i, got.Pix[i], want.Pix[i])
					}
				}
			}
		}
	}
}

func BenchmarkConvolveY8GoSIMDUSDOT_32(b *testing.B) {
	kernel := subpelFilters8[8]
	pad := filterTaps
	refSide := 32 + 2*pad
	ref, _ := testPlane(refSide, refSide, 1, refSide)
	fillMotionTestPlane(ref)
	dst, _ := testPlane(32, 32, 1, 32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		convolveY8GoSIMDUSDOT(dst, ref, 0, 0, pad, pad, 32, 32, kernel)
	}
}
