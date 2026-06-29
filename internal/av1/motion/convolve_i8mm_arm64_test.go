// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

func TestCompoundX8I8MMMatchesPureGo(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := testPlane(128, 80, 1, 128)
	fillMotionTestPlane(ref)
	roundOffset := compoundRoundOffset8()
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
			for subX := 1; subX <= subpelQ4Mask; subX++ {
				kernel, err := interpKernel(filter, size.width, subX)
				if err != nil {
					t.Fatal(err)
				}
				got := make([]uint16, size.width*size.height)
				want := make([]uint16, size.width*size.height)
				refX, refY := filterTaps, filterTaps
				predictInterCompoundRef8ToConvBufXI8MM(got, ref, refX, refY, size.width, size.height, kernel, roundOffset)
				predictInterCompoundRef8ToConvBufXPureGo(want, ref, refX, refY, size.width, size.height, kernel, roundOffset)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("filter=%d size=%dx%d subX=%d sample=%d I8MM=%d PureGo=%d",
							filter, size.width, size.height, subX, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestCompoundX8I8MMFallbackMatchesPureGo(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(ref)
	roundOffset := compoundRoundOffset8()
	kernel, err := interpKernel(InterpEightTapRegular, 8, 5)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name          string
		refX, refY    int
		width, height int
	}{
		{name: "left_edge", refX: 1, refY: filterTaps, width: 8, height: 8},
		{name: "right_edge", refX: 27, refY: filterTaps, width: 8, height: 8},
		{name: "odd_width", refX: filterTaps, refY: filterTaps, width: 12, height: 8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := make([]uint16, tc.width*tc.height)
			want := make([]uint16, tc.width*tc.height)
			predictInterCompoundRef8ToConvBufXI8MM(got, ref, tc.refX, tc.refY, tc.width, tc.height, kernel, roundOffset)
			predictInterCompoundRef8ToConvBufXPureGo(want, ref, tc.refX, tc.refY, tc.width, tc.height, kernel, roundOffset)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s sample=%d I8MM wrapper=%d PureGo=%d", tc.name, i, got[i], want[i])
				}
			}
		})
	}
}

func TestCompoundX8I8MMZeroAlloc(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := testPlane(64, 64, 1, 64)
	fillMotionTestPlane(ref)
	out := make([]uint16, 32*32)
	kernel, err := interpKernel(InterpEightTapRegular, 32, 5)
	if err != nil {
		t.Fatal(err)
	}
	roundOffset := compoundRoundOffset8()
	allocs := testing.AllocsPerRun(50, func() {
		predictInterCompoundRef8ToConvBufXI8MM(out, ref, filterTaps, filterTaps, 32, 32, kernel, roundOffset)
	})
	if allocs != 0 {
		t.Fatalf("compound X I8MM allocated %v times, want 0", allocs)
	}
}

func TestCompoundY8I8MMMatchesPureGo(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := testPlane(128, 96, 1, 128)
	fillMotionTestPlane(ref)
	roundOffset := compoundRoundOffset8()
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
		{4, 4},
		{4, 16},
		{8, 4},
		{16, 8},
		{32, 12},
		{64, 16},
	}
	for _, filter := range filters {
		for _, size := range sizes {
			for subY := 1; subY <= subpelQ4Mask; subY++ {
				kernel, err := interpKernel(filter, size.height, subY)
				if err != nil {
					t.Fatal(err)
				}
				got := make([]uint16, size.width*size.height)
				want := make([]uint16, size.width*size.height)
				refX, refY := filterTaps, filterTaps
				predictInterCompoundRef8ToConvBufYI8MM(got, ref, refX, refY, size.width, size.height, kernel, compoundRound0Bits, roundOffset)
				predictInterCompoundRef8ToConvBufYPureGo(want, ref, refX, refY, size.width, size.height, kernel, compoundRound0Bits, roundOffset)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("filter=%d size=%dx%d subY=%d sample=%d I8MM=%d PureGo=%d",
							filter, size.width, size.height, subY, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestCompoundY8I8MMFallbackMatchesNEON(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(ref)
	roundOffset := compoundRoundOffset8()
	kernel, err := interpKernel(InterpEightTapRegular, 8, 5)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name          string
		refX, refY    int
		width, height int
	}{
		{name: "top_edge", refX: filterTaps, refY: 1, width: 8, height: 8},
		{name: "bottom_edge", refX: filterTaps, refY: 27, width: 8, height: 8},
		{name: "odd_width", refX: filterTaps, refY: filterTaps, width: 12, height: 8},
		{name: "odd_height", refX: filterTaps, refY: filterTaps, width: 8, height: 6},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := make([]uint16, tc.width*tc.height)
			want := make([]uint16, tc.width*tc.height)
			predictInterCompoundRef8ToConvBufYI8MM(got, ref, tc.refX, tc.refY, tc.width, tc.height, kernel, compoundRound0Bits, roundOffset)
			predictInterCompoundRef8ToConvBufYNEON(want, ref, tc.refX, tc.refY, tc.width, tc.height, kernel, compoundRound0Bits, roundOffset)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s sample=%d I8MM wrapper=%d NEON=%d", tc.name, i, got[i], want[i])
				}
			}
		})
	}
}

func TestCompoundY8I8MMZeroAlloc(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := testPlane(64, 64, 1, 64)
	fillMotionTestPlane(ref)
	out := make([]uint16, 32*32)
	kernel, err := interpKernel(InterpEightTapRegular, 32, 5)
	if err != nil {
		t.Fatal(err)
	}
	roundOffset := compoundRoundOffset8()
	allocs := testing.AllocsPerRun(50, func() {
		predictInterCompoundRef8ToConvBufYI8MM(out, ref, filterTaps, filterTaps, 32, 32, kernel, compoundRound0Bits, roundOffset)
	})
	if allocs != 0 {
		t.Fatalf("compound Y I8MM allocated %v times, want 0", allocs)
	}
}

func BenchmarkCompoundConvBufX8I8MM_32(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
	}
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	kernel, err := interpKernel(InterpEightTapRegular, 32, 3)
	if err != nil {
		b.Fatal(err)
	}
	out, ok := compoundConvBufView(&buf, 32, 32)
	if !ok {
		b.Fatal("invalid convbuf")
	}
	roundOffset := compoundRoundOffset8()
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRef8ToConvBufXI8MM(out, ref, filterTaps, filterTaps, 32, 32, kernel, roundOffset)
	})
}

func BenchmarkCompoundConvBufY8I8MM_32(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
	}
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	kernel, err := interpKernel(InterpEightTapRegular, 32, 5)
	if err != nil {
		b.Fatal(err)
	}
	out, ok := compoundConvBufView(&buf, 32, 32)
	if !ok {
		b.Fatal("invalid convbuf")
	}
	roundOffset := compoundRoundOffset8()
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRef8ToConvBufYI8MM(out, ref, filterTaps, filterTaps, 32, 32, kernel, compoundRound0Bits, roundOffset)
	})
}

func BenchmarkCompoundConvBufY8NEONDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	kernel, err := interpKernel(InterpEightTapRegular, 32, 5)
	if err != nil {
		b.Fatal(err)
	}
	out, ok := compoundConvBufView(&buf, 32, 32)
	if !ok {
		b.Fatal("invalid convbuf")
	}
	roundOffset := compoundRoundOffset8()
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRef8ToConvBufYNEON(out, ref, filterTaps, filterTaps, 32, 32, kernel, compoundRound0Bits, roundOffset)
	})
}

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
