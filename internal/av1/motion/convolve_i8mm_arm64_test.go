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

func TestCompound2D8I8MMMatchesPureGo(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	rng := rand.New(rand.NewSource(0x2d18c0ff))
	ref, _ := testPlane(160, 128, 1, 160)
	for i := range ref.Pix {
		ref.Pix[i] = byte(rng.Intn(256))
	}
	filterPairs := []struct {
		x InterpFilter
		y InterpFilter
	}{
		{InterpEightTapRegular, InterpEightTapRegular},
		{InterpEightTapSmooth, InterpMultiTapSharp},
		{InterpMultiTapSharp, InterpEightTapSmooth},
		{InterpBilinear, InterpEightTapRegular},
	}
	sizes := []struct {
		width  int
		height int
	}{
		{8, 8},
		{16, 8},
		{32, 12},
		{64, 16},
	}
	fastCases := 0
	for _, filters := range filterPairs {
		for _, size := range sizes {
			for subX := 1; subX <= subpelQ4Mask; subX++ {
				xKernel, err := interpKernel(filters.x, size.width, subX)
				if err != nil {
					t.Fatal(err)
				}
				xFilter, f0, ok := convolveX8I8MMFilter(xKernel)
				if !ok {
					continue
				}
				for subY := 1; subY <= subpelQ4Mask; subY++ {
					yKernel, err := interpKernel(filters.y, size.height, subY)
					if err != nil {
						t.Fatal(err)
					}
					got := make([]uint16, size.width*size.height)
					gotScratch := make([]uint16, size.width*size.height)
					want := make([]uint16, size.width*size.height)
					var im compoundIM16
					var scratch CompoundConvolveScratch
					refX, refY := filterTaps, filterTaps
					predictInterCompoundRef8ToConvBuf2DI8MMWithIM(got, ref, refX, refY, size.width, size.height, xFilter, f0, yKernel, &im)
					predictInterCompoundRef8ToConvBuf2DI8MM(gotScratch, ref, refX, refY, size.width, size.height, xKernel, yKernel, 19, &scratch)
					predictInterCompoundRef8ToConvBuf2DPureGo(want, ref, refX, refY, size.width, size.height, xKernel, yKernel, 19, nil)
					for i := range want {
						if got[i] != want[i] || gotScratch[i] != want[i] {
							t.Fatalf("filters=%d/%d size=%dx%d sub=(%d,%d) sample=%d I8MM=%d I8MM-scratch=%d PureGo=%d",
								filters.x, filters.y, size.width, size.height, subX, subY, i, got[i], gotScratch[i], want[i])
						}
					}
					fastCases++
				}
			}
		}
	}
	if fastCases == 0 {
		t.Fatal("no compound 2D I8MM fast-path cases covered")
	}
}

func TestCompound2D4TapW4I8MMMatchesPureGo(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	rng := rand.New(rand.NewSource(0x24d4))
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
						got := make([]uint16, size.width*size.height)
						gotScratch := make([]uint16, size.width*size.height)
						want := make([]uint16, size.width*size.height)
						var im [(maxBlockSize + filterTaps - 1) * 4]int16
						var scratch CompoundConvolveScratch
						refX, refY := filterTaps, filterTaps
						predictInterCompoundRef8ToConvBuf2DI8MMW4WithIMStride(got, ref, refX, refY, size.height, xFilter, yKernel, &im[0], 4)
						predictInterCompoundRef8ToConvBuf2DI8MM(gotScratch, ref, refX, refY, size.width, size.height, xKernel, yKernel, 19, &scratch)
						predictInterCompoundRef8ToConvBuf2DPureGo(want, ref, refX, refY, size.width, size.height, xKernel, yKernel, 19, nil)
						for i := range want {
							if got[i] != want[i] || gotScratch[i] != want[i] {
								t.Fatalf("size=%dx%d sub=(%d,%d) sample=%d I8MM=%d I8MM-scratch=%d PureGo=%d",
									size.width, size.height, subX, subY, i, got[i], gotScratch[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}

func TestCompound2D8I8MMFallbackMatchesNEON(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := testPlane(40, 40, 1, 40)
	fillMotionTestPlane(ref)
	xKernel, err := interpKernel(InterpEightTapRegular, 8, 5)
	if err != nil {
		t.Fatal(err)
	}
	yKernel, err := interpKernel(InterpEightTapSmooth, 8, 7)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name          string
		refX, refY    int
		width, height int
		offsetBits    int
	}{
		{name: "width4", refX: filterTaps, refY: filterTaps, width: 4, height: 16, offsetBits: 19},
		{name: "odd_width", refX: filterTaps, refY: filterTaps, width: 12, height: 8, offsetBits: 19},
		{name: "right_edge", refX: 33, refY: filterTaps, width: 8, height: 8, offsetBits: 19},
		{name: "bad_offset", refX: filterTaps, refY: filterTaps, width: 8, height: 8, offsetBits: 18},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := make([]uint16, tc.width*tc.height)
			want := make([]uint16, tc.width*tc.height)
			predictInterCompoundRef8ToConvBuf2DI8MM(got, ref, tc.refX, tc.refY, tc.width, tc.height, xKernel, yKernel, tc.offsetBits, nil)
			predictInterCompoundRef8ToConvBuf2DNEON(want, ref, tc.refX, tc.refY, tc.width, tc.height, xKernel, yKernel, tc.offsetBits, nil)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s sample=%d I8MM wrapper=%d NEON=%d", tc.name, i, got[i], want[i])
				}
			}
		})
	}
}

func TestCompound2D8I8MMZeroAlloc(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := benchPlanes(32, 8)
	out := make([]uint16, 32*32)
	xKernel := subpelFilters8[3]
	yKernel := subpelFilters8[5]
	var scratch CompoundConvolveScratch
	allocs := testing.AllocsPerRun(50, func() {
		predictInterCompoundRef8ToConvBuf2DI8MM(out, ref, filterTaps, filterTaps, 32, 32, xKernel, yKernel, 19, &scratch)
	})
	if allocs != 0 {
		t.Fatalf("compound 2D I8MM allocated %v times, want 0", allocs)
	}
}

func TestCompound2D4TapW4I8MMZeroAlloc(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	_, ref := benchPlanes(16, 8)
	out := make([]uint16, 4*16)
	xKernel := subpelFilters4[3]
	yKernel := subpelFilters8[5]
	var scratch CompoundConvolveScratch
	allocs := testing.AllocsPerRun(50, func() {
		predictInterCompoundRef8ToConvBuf2DI8MM(out, ref, filterTaps, filterTaps, 4, 16, xKernel, yKernel, 19, &scratch)
	})
	if allocs != 0 {
		t.Fatalf("compound 2D width-4 I8MM allocated %v times, want 0", allocs)
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

func BenchmarkCompoundConvBuf2D8I8MM_32(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
	}
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	out, ok := compoundConvBufView(&buf, 32, 32)
	if !ok {
		b.Fatal("invalid convbuf")
	}
	xKernel := subpelFilters8[3]
	yKernel := subpelFilters8[5]
	var scratch CompoundConvolveScratch
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRef8ToConvBuf2DI8MM(out, ref, filterTaps, filterTaps, 32, 32, xKernel, yKernel, 19, &scratch)
	})
}

func BenchmarkCompoundConvBuf2D4TapW4I8MM_4x16(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
	}
	_, ref := benchPlanes(16, 8)
	out := make([]uint16, 4*16)
	xKernel := subpelFilters4[3]
	yKernel := subpelFilters8[5]
	var scratch CompoundConvolveScratch
	runConvolveBench(b, 4, 16, func() {
		predictInterCompoundRef8ToConvBuf2DI8MM(out, ref, filterTaps, filterTaps, 4, 16, xKernel, yKernel, 19, &scratch)
	})
}

func BenchmarkCompoundConvBuf2D8NEONDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	out, ok := compoundConvBufView(&buf, 32, 32)
	if !ok {
		b.Fatal("invalid convbuf")
	}
	xKernel := subpelFilters8[3]
	yKernel := subpelFilters8[5]
	var scratch CompoundConvolveScratch
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRef8ToConvBuf2DNEON(out, ref, filterTaps, filterTaps, 32, 32, xKernel, yKernel, 19, &scratch)
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
