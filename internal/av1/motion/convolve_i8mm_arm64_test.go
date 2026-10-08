// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"math/rand"
	"strconv"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

func TestConvolveX8I8MMMatchesPureGo(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	filters := []InterpFilter{
		InterpEightTapRegular,
		InterpEightTapSmooth,
		InterpMultiTapSharp,
		InterpBilinear,
	}
	sizes := []struct {
		width  int
		height int
	}{
		{8, 1},
		{8, 4},
		{16, 7},
		{32, 13},
		{64, 16},
	}
	for _, filter := range filters {
		for _, size := range sizes {
			ref, _ := testPlane(size.width+2*filterTaps, size.height+2*filterTaps, 1, size.width+2*filterTaps)
			fillMotionTestPlane(ref)
			got, _ := testPlane(size.width, size.height, 1, size.width)
			want, _ := testPlane(size.width, size.height, 1, size.width)
			for subX := 1; subX <= subpelQ4Mask; subX++ {
				kernel, err := interpKernel(filter, size.width, subX)
				if err != nil {
					t.Fatal(err)
				}
				clear(got.Pix)
				clear(want.Pix)
				convolveX8I8MM(got, ref, 0, 0, filterTaps, filterTaps, size.width, size.height, kernel)
				convolveX8PureGo(want, ref, 0, 0, filterTaps, filterTaps, size.width, size.height, kernel)
				for y := 0; y < size.height; y++ {
					row := y * size.width
					for x := 0; x < size.width; x++ {
						i := row + x
						if got.Pix[i] != want.Pix[i] {
							t.Fatalf("filter=%d size=%dx%d subX=%d sample=(%d,%d) I8MM=%d PureGo=%d",
								filter, size.width, size.height, subX, x, y, got.Pix[i], want.Pix[i])
						}
					}
				}
			}
		}
	}
}

func TestConvolveX8I8MMFallbackMatchesNEON(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(ref)
	kernel, err := interpKernel(InterpEightTapRegular, 4, 5)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := testPlane(4, 16, 1, 4)
	want, _ := testPlane(4, 16, 1, 4)
	convolveX8I8MM(got, ref, 0, 0, filterTaps, filterTaps, 4, 16, kernel)
	convolveX8NEON(want, ref, 0, 0, filterTaps, filterTaps, 4, 16, kernel)
	for i := range got.Pix {
		if got.Pix[i] != want.Pix[i] {
			t.Fatalf("width-4 fallback sample=%d I8MM wrapper=%d NEON=%d", i, got.Pix[i], want.Pix[i])
		}
	}
}

func TestConvolveX8I8MMZeroAlloc(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	dst, ref := benchPlanes(32, 8)
	kernel, err := interpKernel(InterpEightTapRegular, 32, 5)
	if err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(50, func() {
		convolveX8I8MM(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, kernel)
	})
	if allocs != 0 {
		t.Fatalf("convolve X I8MM allocated %v times, want 0", allocs)
	}
}

func BenchmarkConvolveX8I8MM_32(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
	}
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters8[3]
	runConvolveBench(b, 32, 32, func() {
		convolveX8I8MM(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk)
	})
}

func BenchmarkConvolveX8NEONDirect_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters8[3]
	runConvolveBench(b, 32, 32, func() {
		convolveX8NEON(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk)
	})
}

func TestConvolveY8I8MMMatchesPureGo(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	rng := rand.New(rand.NewSource(0x91e7a1))
	eightTables := [][16][filterTaps]int16{subpelFilters8, subpelFilters8Smooth, subpelFilters8Sharp}
	fourTables := [][16][filterTaps]int16{subpelFilters4, subpelFilters4Smooth}
	sizes := []struct {
		width  int
		height int
	}{
		{4, 4},
		{4, 16},
		{8, 4},
		{16, 8},
		{32, 16},
		{64, 16},
	}
	for _, size := range sizes {
		ref, _ := testPlane(size.width+2*filterTaps, size.height+2*filterTaps, 1, size.width+2*filterTaps)
		for i := range ref.Pix {
			ref.Pix[i] = byte(rng.Intn(256))
		}
		got, _ := testPlane(size.width, size.height, 1, size.width)
		want, _ := testPlane(size.width, size.height, 1, size.width)
		for _, table := range eightTables {
			for subY := 1; subY <= subpelQ4Mask; subY++ {
				clear(got.Pix)
				clear(want.Pix)
				convolveY8I8MM(got, ref, 0, 0, filterTaps, filterTaps, size.width, size.height, table[subY])
				convolveY8PureGo(want, ref, 0, 0, filterTaps, filterTaps, size.width, size.height, table[subY])
				for y := 0; y < size.height; y++ {
					row := y * size.width
					for x := 0; x < size.width; x++ {
						i := row + x
						if got.Pix[i] != want.Pix[i] {
							t.Fatalf("8tap size=%dx%d subY=%d sample=(%d,%d) I8MM=%d PureGo=%d",
								size.width, size.height, subY, x, y, got.Pix[i], want.Pix[i])
						}
					}
				}
			}
		}
		for _, table := range fourTables {
			for subY := 1; subY <= subpelQ4Mask; subY++ {
				clear(got.Pix)
				clear(want.Pix)
				convolveY8I8MM(got, ref, 0, 0, filterTaps, filterTaps, size.width, size.height, table[subY])
				convolveY8PureGo(want, ref, 0, 0, filterTaps, filterTaps, size.width, size.height, table[subY])
				for y := 0; y < size.height; y++ {
					row := y * size.width
					for x := 0; x < size.width; x++ {
						i := row + x
						if got.Pix[i] != want.Pix[i] {
							t.Fatalf("4tap size=%dx%d subY=%d sample=(%d,%d) I8MM=%d PureGo=%d",
								size.width, size.height, subY, x, y, got.Pix[i], want.Pix[i])
						}
					}
				}
			}
		}
	}
}

func TestConvolveY8I8MMFallbackMatchesNEON(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := testPlane(48, 48, 1, 48)
	fillMotionTestPlane(ref)
	cases := []struct {
		name          string
		width, height int
		kernel        [filterTaps]int16
	}{
		{name: "odd_width", width: 12, height: 8, kernel: subpelFilters8[5]},
		{name: "non_multiple_height", width: 16, height: 6, kernel: subpelFilters8[7]},
		{name: "bilinear_two_tap", width: 16, height: 8, kernel: bilinearFilters[8]},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, _ := testPlane(tc.width, tc.height, 1, tc.width)
			want, _ := testPlane(tc.width, tc.height, 1, tc.width)
			convolveY8I8MM(got, ref, 0, 0, filterTaps, filterTaps, tc.width, tc.height, tc.kernel)
			convolveY8NEON(want, ref, 0, 0, filterTaps, filterTaps, tc.width, tc.height, tc.kernel)
			for i := range got.Pix {
				if got.Pix[i] != want.Pix[i] {
					t.Fatalf("%s sample=%d I8MM wrapper=%d NEON=%d", tc.name, i, got.Pix[i], want.Pix[i])
				}
			}
		})
	}
}

func TestConvolveY8I8MMZeroAlloc(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	dst, ref := benchPlanes(32, 8)
	yk := subpelFilters8[5]
	allocs := testing.AllocsPerRun(50, func() {
		convolveY8I8MM(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, yk)
	})
	if allocs != 0 {
		t.Fatalf("convolve Y I8MM allocated %v times, want 0", allocs)
	}
	yk4 := subpelFilters4[5]
	allocs = testing.AllocsPerRun(50, func() {
		convolveY8I8MM(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, yk4)
	})
	if allocs != 0 {
		t.Fatalf("convolve Y 4-tap I8MM allocated %v times, want 0", allocs)
	}
}

func BenchmarkConvolveY8I8MM_32(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
	}
	dst, ref := benchPlanes(32, 8)
	yk := subpelFilters8[5]
	runConvolveBench(b, 32, 32, func() {
		convolveY8I8MM(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, yk)
	})
}

func BenchmarkConvolveY8I8MM_4tap_32(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
	}
	dst, ref := benchPlanes(32, 8)
	yk := subpelFilters4[5]
	runConvolveBench(b, 32, 32, func() {
		convolveY8I8MM(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, yk)
	})
}

func BenchmarkConvolveY8NEONDirect_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	yk := subpelFilters8[5]
	runConvolveBench(b, 32, 32, func() {
		convolveY8NEON(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, yk)
	})
}

func BenchmarkConvolveY8NEONDirect_4tap_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	yk := subpelFilters4[5]
	runConvolveBench(b, 32, 32, func() {
		convolveY8NEON(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, yk)
	})
}

func TestConvolve2D8I8MMMatchesPureGo(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	rng := rand.New(rand.NewSource(0x2181e8dd))
	filters := []InterpFilter{
		InterpEightTapRegular,
		InterpEightTapSmooth,
		InterpMultiTapSharp,
		InterpBilinear,
	}
	sizes := []struct {
		width  int
		height int
	}{
		{8, 8},
		{16, 7},
		{32, 13},
		{64, 16},
	}
	for _, filter := range filters {
		for _, size := range sizes {
			ref, _ := testPlane(size.width+2*filterTaps, size.height+2*filterTaps, 1, size.width+2*filterTaps)
			for i := range ref.Pix {
				ref.Pix[i] = byte(rng.Intn(256))
			}
			got, _ := testPlane(size.width, size.height, 1, size.width)
			gotScratch, _ := testPlane(size.width, size.height, 1, size.width)
			want, _ := testPlane(size.width, size.height, 1, size.width)
			var scratch ConvolveScratch
			for subX := 0; subX <= subpelQ4Mask; subX++ {
				xKernel, err := interpKernel(filter, size.width, subX)
				if err != nil {
					t.Fatal(err)
				}
				for subY := 0; subY <= subpelQ4Mask; subY++ {
					yKernel, err := interpKernel(filter, size.width, subY)
					if err != nil {
						t.Fatal(err)
					}
					clear(got.Pix)
					clear(gotScratch.Pix)
					clear(want.Pix)
					convolve2D8I8MM(got, ref, 0, 0, filterTaps, filterTaps, size.width, size.height, xKernel, yKernel)
					convolve2D8I8MMWithScratch(gotScratch, ref, 0, 0, filterTaps, filterTaps, size.width, size.height, xKernel, yKernel, &scratch)
					convolve2D8PureGo(want, ref, 0, 0, filterTaps, filterTaps, size.width, size.height, xKernel, yKernel)
					for y := 0; y < size.height; y++ {
						row := y * size.width
						for x := 0; x < size.width; x++ {
							i := row + x
							if got.Pix[i] != want.Pix[i] || gotScratch.Pix[i] != want.Pix[i] {
								t.Fatalf("filter=%d size=%dx%d sub=(%d,%d) sample=(%d,%d) I8MM=%d I8MM-scratch=%d PureGo=%d",
									filter, size.width, size.height, subX, subY, x, y, got.Pix[i], gotScratch.Pix[i], want.Pix[i])
							}
						}
					}
				}
			}
		}
	}
	fourTapTables := [][16][filterTaps]int16{subpelFilters4, subpelFilters4Smooth}
	for _, table := range fourTapTables {
		for _, size := range []struct {
			width  int
			height int
		}{{8, 8}, {32, 13}} {
			ref, _ := testPlane(size.width+2*filterTaps, size.height+2*filterTaps, 1, size.width+2*filterTaps)
			for i := range ref.Pix {
				ref.Pix[i] = byte(rng.Intn(256))
			}
			got, _ := testPlane(size.width, size.height, 1, size.width)
			want, _ := testPlane(size.width, size.height, 1, size.width)
			for subX := 0; subX <= subpelQ4Mask; subX++ {
				for subY := 0; subY <= subpelQ4Mask; subY++ {
					clear(got.Pix)
					clear(want.Pix)
					convolve2D8I8MM(got, ref, 0, 0, filterTaps, filterTaps, size.width, size.height, table[subX], table[subY])
					convolve2D8PureGo(want, ref, 0, 0, filterTaps, filterTaps, size.width, size.height, table[subX], table[subY])
					for y := 0; y < size.height; y++ {
						row := y * size.width
						for x := 0; x < size.width; x++ {
							i := row + x
							if got.Pix[i] != want.Pix[i] {
								t.Fatalf("4tap size=%dx%d sub=(%d,%d) sample=(%d,%d) I8MM=%d PureGo=%d",
									size.width, size.height, subX, subY, x, y, got.Pix[i], want.Pix[i])
							}
						}
					}
				}
			}
		}
	}
}

func TestConvolve2D4TapW4I8MMMatchesPureGo(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	rng := rand.New(rand.NewSource(0x2d44))
	xTables := [][16][filterTaps]int16{subpelFilters4, subpelFilters4Smooth}
	yTables := [][16][filterTaps]int16{subpelFilters8, subpelFilters4, subpelFilters4Smooth}
	sizes := []struct {
		width  int
		height int
	}{
		{4, 4},
		{4, 16},
	}
	for _, size := range sizes {
		ref, _ := testPlane(size.width+2*filterTaps, size.height+2*filterTaps, 1, size.width+2*filterTaps)
		for i := range ref.Pix {
			ref.Pix[i] = byte(rng.Intn(256))
		}
		for _, xTable := range xTables {
			for _, yTable := range yTables {
				for subX := 0; subX <= subpelQ4Mask; subX++ {
					xKernel := xTable[subX]
					xFilter, ok := convolveX4I8MMFilter(xKernel)
					if !ok {
						continue
					}
					for subY := 0; subY <= subpelQ4Mask; subY++ {
						yKernel := yTable[subY]
						got, _ := testPlane(size.width, size.height, 1, size.width)
						gotScratch, _ := testPlane(size.width, size.height, 1, size.width)
						want, _ := testPlane(size.width, size.height, 1, size.width)
						var im [(maxBlockSize + filterTaps - 1) * 4]int16
						var scratch ConvolveScratch
						refX, refY := filterTaps, filterTaps
						convolve2D4TapW4I8MMWithIMStride(got, ref, 0, 0, refX, refY, size.height, xFilter, yKernel, &im[0], 4)
						convolve2D8I8MMWithScratch(gotScratch, ref, 0, 0, refX, refY, size.width, size.height, xKernel, yKernel, &scratch)
						convolve2D8PureGo(want, ref, 0, 0, refX, refY, size.width, size.height, xKernel, yKernel)
						for i := range want.Pix {
							if got.Pix[i] != want.Pix[i] || gotScratch.Pix[i] != want.Pix[i] {
								t.Fatalf("size=%dx%d sub=(%d,%d) sample=%d I8MM=%d I8MM-scratch=%d PureGo=%d",
									size.width, size.height, subX, subY, i, got.Pix[i], gotScratch.Pix[i], want.Pix[i])
							}
						}
					}
				}
			}
		}
	}
}

func TestConvolve2D8I8MMFallbackMatchesNEON(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(ref)
	xKernel, err := interpKernel(InterpEightTapRegular, 4, 5)
	if err != nil {
		t.Fatal(err)
	}
	yKernel, err := interpKernel(InterpEightTapSmooth, 4, 7)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := testPlane(4, 16, 1, 4)
	want, _ := testPlane(4, 16, 1, 4)
	convolve2D8I8MM(got, ref, 0, 0, filterTaps, filterTaps, 4, 16, xKernel, yKernel)
	convolve2D8NEON(want, ref, 0, 0, filterTaps, filterTaps, 4, 16, xKernel, yKernel)
	for i := range got.Pix {
		if got.Pix[i] != want.Pix[i] {
			t.Fatalf("width-4 fallback sample=%d I8MM wrapper=%d NEON=%d", i, got.Pix[i], want.Pix[i])
		}
	}
}

func TestConvolve2D8ClampedHorizontalEdgeI8MMMatchesPureGo(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	rng := rand.New(rand.NewSource(0x2d8c18))
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
					gotDispatch, _ := testPlane(w, h, 1, w)
					gotSplit, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					var scratch ConvolveScratch
					var dispatchScratch ConvolveScratch
					var splitScratch ConvolveScratch
					convolve2D8ClampedI8MMWithScratch(got, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1], &scratch)
					convolve2D8ClampedWithScratchImpl(gotDispatch, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1], &dispatchScratch)
					if !convolve2D8ClampedEdgeSplitI8MMWithScratch(gotSplit, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1], &splitScratch) {
						t.Fatalf("2D8horizontal-edge I8MM split path was not used w=%d h=%d edge=%s", w, h, edge)
					}
					convolve2D8ClampedPureGo(want, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1])
					assertBytesEqual(t, got, want, w, h, "2D8horizontal-edge-i8mm", w, h, edge)
					assertBytesEqual(t, gotDispatch, want, w, h, "2D8horizontal-edge-i8mm-dispatch", w, h, edge)
					assertBytesEqual(t, gotSplit, want, w, h, "2D8horizontal-edge-i8mm-split", w, h, edge)
				}
			}
		}
	}
}

func TestConvolve2D8I8MMZeroAlloc(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	dst, ref := benchPlanes(32, 8)
	xKernel := subpelFilters8[3]
	yKernel := subpelFilters8[5]
	var scratch ConvolveScratch
	allocs := testing.AllocsPerRun(50, func() {
		convolve2D8I8MMWithScratch(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xKernel, yKernel, &scratch)
	})
	if allocs != 0 {
		t.Fatalf("convolve 2D I8MM allocated %v times, want 0", allocs)
	}

	const plane = 64
	edgeRef, _ := testPlane(plane, plane, 1, plane)
	fillMotionTestPlane(edgeRef)
	edgeDst, _ := testPlane(16, 16, 1, 16)
	edgeX := plane - 13
	edgeY := 20
	allocs = testing.AllocsPerRun(50, func() {
		convolve2D8ClampedI8MMWithScratch(edgeDst, edgeRef, 0, 0, edgeX, edgeY, 16, 16, xKernel, yKernel, &scratch)
	})
	if allocs != 0 {
		t.Fatalf("convolve 2D clamped I8MM allocated %v times, want 0", allocs)
	}
}

func TestConvolve2D4TapW4I8MMZeroAlloc(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	dst, ref := benchPlanes(16, 8)
	xKernel := subpelFilters4[3]
	yKernel := subpelFilters8[5]
	var scratch ConvolveScratch
	allocs := testing.AllocsPerRun(50, func() {
		convolve2D8I8MMWithScratch(dst, ref, 0, 0, filterTaps, filterTaps, 4, 16, xKernel, yKernel, &scratch)
	})
	if allocs != 0 {
		t.Fatalf("convolve 2D width-4 I8MM allocated %v times, want 0", allocs)
	}
}

func BenchmarkConvolve2D8I8MM_32(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
	}
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	runConvolveBench(b, 32, 32, func() {
		convolve2D8I8MM(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk, yk)
	})
}

func BenchmarkConvolve2D8I8MMWithScratch_32(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
	}
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	var scratch ConvolveScratch
	runConvolveBench(b, 32, 32, func() {
		convolve2D8I8MMWithScratch(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk, yk, &scratch)
	})
}

func BenchmarkConvolve2D8ClampedEdgeI8MMWithScratch_16(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
	}
	const plane = 64
	ref, _ := testPlane(plane, plane, 1, plane)
	fillMotionTestPlane(ref)
	dst, _ := testPlane(16, 16, 1, 16)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	refX := plane - 13
	refY := 20
	var scratch ConvolveScratch
	runConvolveBench(b, 16, 16, func() {
		convolve2D8ClampedI8MMWithScratch(dst, ref, 0, 0, refX, refY, 16, 16, xk, yk, &scratch)
	})
}

func BenchmarkConvolve2D8ClampedEdgeNEONWithScratch_16(b *testing.B) {
	const plane = 64
	ref, _ := testPlane(plane, plane, 1, plane)
	fillMotionTestPlane(ref)
	dst, _ := testPlane(16, 16, 1, 16)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	refX := plane - 13
	refY := 20
	var scratch ConvolveScratch
	runConvolveBench(b, 16, 16, func() {
		convolve2D8ClampedNEONWithScratch(dst, ref, 0, 0, refX, refY, 16, 16, xk, yk, &scratch)
	})
}

func BenchmarkConvolve2D8I8MM_4tap_32(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
	}
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters4[3]
	yk := subpelFilters4[5]
	runConvolveBench(b, 32, 32, func() {
		convolve2D8I8MM(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk, yk)
	})
}

func BenchmarkConvolve2D4TapW4I8MMWithScratch_4x16(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
	}
	dst, ref := benchPlanes(16, 8)
	xk := subpelFilters4[3]
	yk := subpelFilters8[5]
	var scratch ConvolveScratch
	runConvolveBench(b, 4, 16, func() {
		convolve2D8I8MMWithScratch(dst, ref, 0, 0, filterTaps, filterTaps, 4, 16, xk, yk, &scratch)
	})
}

func BenchmarkConvolve2D4TapW4I8MMDirect_4x16(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
	}
	dst, ref := benchPlanes(16, 8)
	xk := subpelFilters4[3]
	yk := subpelFilters8[5]
	xFilter, ok := convolveX4I8MMFilter(xk)
	if !ok {
		b.Fatal("4-tap filter rejected")
	}
	var im [(maxBlockSize + filterTaps - 1) * 4]int16
	runConvolveBench(b, 4, 16, func() {
		convolve2D4TapW4I8MMWithIMStride(dst, ref, 0, 0, filterTaps, filterTaps, 16, xFilter, yk, &im[0], 4)
	})
}

func BenchmarkConvolve2D8NEONWithScratchDirect_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	var scratch ConvolveScratch
	runConvolveBench(b, 32, 32, func() {
		convolve2D8NEONWithScratch(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk, yk, &scratch)
	})
}

func BenchmarkConvolve2D8NEONDirect_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	runConvolveBench(b, 32, 32, func() {
		convolve2D8NEON(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk, yk)
	})
}

type i8mmVerticalTapCase struct {
	name       string
	ker        [filterTaps]int16
	taps       uintptr
	startDelta int
}

func i8mmVerticalTapCases() []i8mmVerticalTapCase {
	mixedFirstZero := subpelFilters8Sharp[9]
	mixedFirstZero[0] = 0
	mixedLastZero := subpelFilters8Sharp[9]
	mixedLastZero[filterTaps-1] = 0
	return []i8mmVerticalTapCase{
		{name: "regular", ker: subpelFilters8[6], taps: 6, startDelta: -2},
		{name: "smooth", ker: subpelFilters8Smooth[9], taps: 6, startDelta: -2},
		{name: "four-interior-coefficients", ker: subpelFilters4[6], taps: 6, startDelta: -2},
		{name: "sharp", ker: subpelFilters8Sharp[9], taps: 8, startDelta: -3},
		{name: "first-outer-zero", ker: mixedFirstZero, taps: 8, startDelta: -3},
		{name: "last-outer-zero", ker: mixedLastZero, taps: 8, startDelta: -3},
	}
}

func TestI8MM2DVerticalWindow(t *testing.T) {
	for _, tc := range i8mmVerticalTapCases() {
		t.Run(tc.name, func(t *testing.T) {
			start, taps := i8mm2DVerticalWindow(filterTaps+5, tc.ker)
			if want := filterTaps + 5 + tc.startDelta; start != want || taps != tc.taps {
				t.Fatalf("window=(%d,%d), want=(%d,%d)", start, taps, want, tc.taps)
			}
		})
	}
}

func TestConvolve2D8I8MMVerticalWindowTinyAndOddParity(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := testPlane(160, 128, 1, 160)
	fillMotionTestPlane(ref)
	xKernel := subpelFilters8[6]
	refX, refY := 16, 16
	sizes := []struct {
		width  int
		height int
		stride int
	}{
		{width: 8, height: 1, stride: 13},
		{width: 8, height: 3, stride: 13},
		{width: 8, height: 9, stride: 13},
		{width: 16, height: 1, stride: 21},
		{width: 16, height: 5, stride: 21},
	}
	for _, tc := range i8mmVerticalTapCases() {
		for _, size := range sizes {
			t.Run(tc.name+"/"+strconv.Itoa(size.width)+"x"+strconv.Itoa(size.height), func(t *testing.T) {
				got, _ := testPlane(size.width, size.height, 1, size.stride)
				gotScratch, _ := testPlane(size.width, size.height, 1, size.stride)
				want, _ := testPlane(size.width, size.height, 1, size.stride)
				for i := range got.Pix {
					got.Pix[i], gotScratch.Pix[i], want.Pix[i] = 0xa5, 0xa5, 0xa5
				}
				var scratch ConvolveScratch
				convolve2D8I8MM(got, ref, 0, 0, refX, refY, size.width, size.height, xKernel, tc.ker)
				convolve2D8I8MMWithScratch(gotScratch, ref, 0, 0, refX, refY, size.width, size.height, xKernel, tc.ker, &scratch)
				convolve2D8PureGo(want, ref, 0, 0, refX, refY, size.width, size.height, xKernel, tc.ker)
				for i := range want.Pix {
					if got.Pix[i] != want.Pix[i] || gotScratch.Pix[i] != want.Pix[i] {
						t.Fatalf("convolve byte=%d no-scratch=%d scratch=%d PureGo=%d", i, got.Pix[i], gotScratch.Pix[i], want.Pix[i])
					}
				}

			})
		}
	}
}
