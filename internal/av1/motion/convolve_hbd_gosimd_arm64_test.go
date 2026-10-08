// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

func testHBDZeroEndpointKernel(kernel [filterTaps]int16) bool {
	return kernel[0] == 0 && kernel[filterTaps-1] == 0
}

func TestHBD6TapGoSIMDDispatchBound(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("arm64 NEON is unavailable")
	}
	cases := []struct {
		name string
		got  any
		want any
	}{
		{"convolve-X-keeps-NEON", convolveXHighBDImpl, convolveXHighBDNEON},
		{"convolve-Y-keeps-NEON", convolveYHighBDImpl, convolveYHighBDNEON},
		{"convolve-2D", convolve2DHighBDImpl, convolve2DHighBDGoSIMD},
		{"convolve-X-clamped-keeps-NEON", convolveXHighBDClampedImpl, convolveXHighBDClampedNEON},
		{"convolve-Y-clamped-keeps-NEON", convolveYHighBDClampedImpl, convolveYHighBDClampedNEON},
		{"convolve-2D-clamped", convolve2DHighBDClampedImpl, convolve2DHighBDClampedGoSIMD},
		{"compound-X-keeps-NEON", predictInterCompoundRefHighBDToConvBufXResidentImpl, predictInterCompoundRefHighBDToConvBufXResidentNEON},
		{"compound-Y-keeps-NEON", predictInterCompoundRefHighBDToConvBufYResidentImpl, predictInterCompoundRefHighBDToConvBufYResidentNEON},
		{"compound-2D", predictInterCompoundRefHighBDToConvBuf2DResidentImpl, predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD},
		{"compound-2D-clamped", predictInterCompoundRefHighBDToConvBuf2DClampedImpl, predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD},
	}
	for _, tc := range cases {
		if reflect.ValueOf(tc.got).Pointer() != reflect.ValueOf(tc.want).Pointer() {
			t.Errorf("%s dispatch does not select the six-tap Go SIMD candidate", tc.name)
		}
	}
}

func TestConvolveHighBDGoSIMD2DSixTapMatchesPureGo(t *testing.T) {
	tables := []struct {
		name string
		k    [16][filterTaps]int16
	}{
		{"regular8", subpelFilters8},
		{"smooth8", subpelFilters8Smooth},
		{"regular4", subpelFilters4},
		{"smooth4", subpelFilters4Smooth},
		{"bilinear", bilinearFilters},
		{"sharp", subpelFilters8Sharp},
	}
	sizes := []struct{ w, h int }{{8, 1}, {16, 7}, {32, 16}, {64, 9}}
	rng := rand.New(rand.NewSource(0x6bd6))

	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		for _, table := range tables {
			for phase := range 16 {
				xk := table.k[phase]
				yk := table.k[(phase+5)&15]
				xOK := testHBDZeroEndpointKernel(xk)
				yOK := testHBDZeroEndpointKernel(yk)
				if !xOK || !yOK {
					continue
				}
				for _, size := range sizes {
					side := size.w
					if size.h > side {
						side = size.h
					}
					ref := makeHighBDRef(side, filterTaps, max, true, rng)
					got2D, _ := testPlane(size.w, size.h, 2, (size.w+3)*2)
					want2D, _ := testPlane(size.w, size.h, 2, (size.w+3)*2)
					scratch := &ConvolveScratch{}
					for i := range scratch.imHBD {
						scratch.imHBD[i] = int32(0x5a5a0000 | i&0xffff)
					}
					convolve2DHighBDGoSIMDWithScratch(got2D, ref, bd, max, 0, 0, filterTaps, filterTaps, size.w, size.h, xk, yk, scratch)
					convolve2DHighBDPureGo(want2D, ref, bd, max, 0, 0, filterTaps, filterTaps, size.w, size.h, xk, yk)
					eqHighBDBlock(t, got2D, want2D, size.w, size.h, "GoSIMD-2D", bd, table.name, phase, size)
				}
			}
		}
	}
}

func TestConvolveHighBDGoSIMD2DSixTapClampedMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6bd7))
	const refW, refH = 37, 29
	const w, h = 16, 16
	positions := [][2]int{{0, 0}, {-2, -1}, {refW - w + 1, refH - h + 2}}
	xk, yk := subpelFilters8[6], subpelFilters8Smooth[9]
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		ref := makeGoSIMDHighBDPlane(refW, refH, max, rng, 3, 1)
		for _, pos := range positions {
			got2D, _ := testPlane(w, h, 2, (w+2)*2)
			want2D, _ := testPlane(w, h, 2, (w+2)*2)
			convolve2DHighBDClampedGoSIMD(got2D, ref, bd, max, 0, 0, pos[0], pos[1], w, h, xk, yk)
			convolve2DHighBDClampedPureGo(want2D, ref, bd, max, 0, 0, pos[0], pos[1], w, h, xk, yk)
			eqHighBDBlock(t, got2D, want2D, w, h, "GoSIMD-2D-clamped", bd, pos)
		}
	}
}

func TestConvolveHighBDGoSIMD2DZeroEndpointFourTapAndBilinearPhases(t *testing.T) {
	tables := []struct {
		name string
		k    [16][filterTaps]int16
	}{
		{"regular4", subpelFilters4},
		{"smooth4", subpelFilters4Smooth},
		{"bilinear", bilinearFilters},
	}
	const w, h = 32, 16
	rng := rand.New(rand.NewSource(0x6bde))

	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		ref := makeHighBDRef(64, filterTaps, max, true, rng)
		for _, table := range tables {
			for phase := range 16 {
				xk, yk := table.k[phase], table.k[(phase+7)&15]
				if !testHBDZeroEndpointKernel(xk) || !testHBDZeroEndpointKernel(yk) {
					t.Fatalf("%s phase pair %d/%d unexpectedly has a nonzero endpoint", table.name, phase, (phase+7)&15)
				}
				got, _ := testPlane(w, h, 2, (w+3)*2)
				want, _ := testPlane(w, h, 2, (w+3)*2)
				scratch := &ConvolveScratch{}
				convolve2DHighBDGoSIMDWithScratch(got, ref, bd, max, 0, 0, filterTaps, filterTaps, w, h, xk, yk, scratch)
				convolve2DHighBDPureGo(want, ref, bd, max, 0, 0, filterTaps, filterTaps, w, h, xk, yk)
				eqHighBDBlock(t, got, want, w, h, "GoSIMD-zero-endpoint", bd, table.name, phase)
			}
		}
	}

	// Mixed table pairs exercise zero-endpoint coefficients independently on
	// the horizontal and vertical pass, including the identity phase.
	for _, pair := range []struct {
		name string
		x, y [filterTaps]int16
	}{
		{"regular4-smooth4", subpelFilters4[1], subpelFilters4Smooth[14]},
		{"bilinear-regular4", bilinearFilters[1], subpelFilters4[15]},
		{"smooth4-bilinear", subpelFilters4Smooth[0], bilinearFilters[8]},
	} {
		max, _ := highBDMax(10)
		ref := makeHighBDRef(64, filterTaps, max, true, rng)
		got, _ := testPlane(w, h, 2, (w+1)*2+1)
		want, _ := testPlane(w, h, 2, (w+1)*2+1)
		convolve2DHighBDGoSIMD(got, ref, 10, max, 0, 0, filterTaps, filterTaps, w, h, pair.x, pair.y)
		convolve2DHighBDPureGo(want, ref, 10, max, 0, 0, filterTaps, filterTaps, w, h, pair.x, pair.y)
		eqHighBDBlock(t, got, want, w, h, "GoSIMD-zero-endpoint-mixed", pair.name)
	}
}

func TestHBD6TapGoSIMDShapeThresholdAndFilterEligibility(t *testing.T) {
	for _, tc := range []struct {
		w, h int
		want bool
	}{
		{8, 1, true}, {8, 2, true}, {8, 4, true}, {8, 8, true},
		{8, 16, true}, {8, 32, true},
		{16, 1, true}, {16, 4, true}, {16, 8, true}, {16, 16, true},
		{32, 4, true}, {32, 8, true},
		{128, 128, true}, {4, 64, false}, {32, 0, false}, {maxBlockSize + 1, 32, false},
	} {
		if got := hbdSIMDShape(tc.w, tc.h); got != tc.want {
			t.Errorf("hbdSIMDShape(%d,%d)=%t, want %t", tc.w, tc.h, got, tc.want)
		}
	}

	for _, table := range []struct {
		name string
		k    [16][filterTaps]int16
	}{
		{"regular8", subpelFilters8}, {"smooth8", subpelFilters8Smooth},
		{"regular4", subpelFilters4}, {"smooth4", subpelFilters4Smooth}, {"bilinear", bilinearFilters},
	} {
		for phase, kernel := range table.k {
			if !testHBDZeroEndpointKernel(kernel) {
				t.Errorf("%s phase %d should have zero endpoint taps", table.name, phase)
			}
		}
	}
	for phase, kernel := range subpelFilters8Sharp {
		if got, want := testHBDZeroEndpointKernel(kernel), phase == 0; got != want {
			t.Errorf("sharp phase %d eligibility=%t, want %t", phase, got, want)
		}
	}
}

func TestConvolveHighBDGoSIMD2DSixTapExactWindowAndMaxScratch(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6bdb))
	const bd = 12
	max, _ := highBDMax(bd)
	xk, yk := subpelFilters8[6], subpelFilters8Smooth[9]
	for _, size := range []struct{ w, h int }{{16, 16}, {maxBlockSize, maxBlockSize}} {
		// The source has only the six-tap footprint: width+5 by height+5.
		// Odd byte stride and a one-byte base offset exercise unaligned vector
		// loads/stores while the scalar oracle gets a one-pixel zero-coefficient
		// border around the exact samples it consumes.
		exact := makeGoSIMDHighBDPlane(size.w+5, size.h+5, max, rng, 1, 1)
		padded := padGoSIMDHighBDPlane(exact)
		got2D := makeGoSIMDHighBDPlane(size.w, size.h, max, rng, 1, 1)
		want2D := makeGoSIMDHighBDPlane(size.w, size.h, max, rng, 1, 1)
		scratch := &ConvolveScratch{}
		convolve2DHighBDGoSIMDWithScratch(got2D, exact, bd, max, 0, 0, 2, 2, size.w, size.h, xk, yk, scratch)
		convolve2DHighBDPureGo(want2D, padded, bd, max, 0, 0, 3, 3, size.w, size.h, xk, yk)
		eqHighBDBlock(t, got2D, want2D, size.w, size.h, "GoSIMD-2D-exact-window", size)
	}
}

func TestCompoundHighBDGoSIMD2DSixTapMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6bd8))
	sizes := []struct{ w, h int }{{16, 16}, {32, 16}}
	pairs := []struct {
		name string
		x, y [16][filterTaps]int16
	}{
		{"regular8-smooth8", subpelFilters8, subpelFilters8Smooth},
		{"regular4-smooth4", subpelFilters4, subpelFilters4Smooth},
		{"bilinear-regular4", bilinearFilters, subpelFilters4},
		{"smooth4-bilinear", subpelFilters4Smooth, bilinearFilters},
		{"sharp-identity-smooth", subpelFilters8Sharp, subpelFilters8Smooth},
	}
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		round0 := compoundRound0(bd)
		offsetBits := int(bd) + 2*filterBits - round0
		ref := makeHighBDRef(40, filterTaps, max, true, rng)
		for _, pair := range pairs {
			for phase := range 16 {
				xk := pair.x[phase]
				yk := pair.y[(phase+3)&15]
				if !testHBDZeroEndpointKernel(xk) || !testHBDZeroEndpointKernel(yk) {
					continue
				}
				for _, size := range sizes {
					got2D, want2D := make([]uint16, size.w*size.h), make([]uint16, size.w*size.h)
					var gotIM, wantIM compoundIM
					predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(got2D, ref, filterTaps, filterTaps, size.w, size.h, xk, yk, round0, offsetBits, int(bd), &gotIM)
					predictInterCompoundRefHighBDToConvBuf2DResident(want2D, ref, filterTaps, filterTaps, size.w, size.h, xk, yk, round0, offsetBits, int(bd), &wantIM)
					assertCompoundHighBDBlockEqual(t, got2D, want2D, "GoSIMD-compound-2D", bd, struct {
						filter string
						phase  int
					}{pair.name, phase}, size)
				}
			}
		}
	}
}

func TestCompoundHighBDGoSIMD2DSixTapClampedMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6bda))
	const refW, refH, w, h = 37, 29, 16, 16
	positions := [][2]int{{0, 0}, {-2, -1}, {refW - w + 1, refH - h + 2}}
	xk, yk := subpelFilters8[6], subpelFilters8Smooth[9]
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		round0 := compoundRound0(bd)
		offsetBits := int(bd) + 2*filterBits - round0
		ref := makeGoSIMDHighBDPlane(refW, refH, max, rng, 3, 1)
		for _, pos := range positions {
			got2D, want2D := make([]uint16, w*h), make([]uint16, w*h)
			var gotIM, wantIM compoundIM
			predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(got2D, ref, pos[0], pos[1], w, h, xk, yk, round0, offsetBits, int(bd), &gotIM, nil)
			predictInterCompoundRefHighBDToConvBuf2DClamped(want2D, ref, pos[0], pos[1], w, h, xk, yk, round0, offsetBits, int(bd), &wantIM)
			assertCompoundHighBDBlockEqual(t, got2D, want2D, "GoSIMD-compound-2D-clamped", bd, pos[0], struct{ w, h int }{w, h})
		}
	}
}

func TestConvolveHighBDGoSIMD2DSixTapZeroAlloc(t *testing.T) {
	const pad = filterTaps
	max, _ := highBDMax(10)
	ref := makeHighBDRef(32, pad, max, true, rand.New(rand.NewSource(0x6bd9)))
	dst, _ := testPlane(32, 16, 2, 70)
	xk, yk := subpelFilters8[6], subpelFilters8Smooth[9]
	scratch := &ConvolveScratch{}
	var im compoundIM
	out := make([]uint16, 32*16)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	cases := []struct {
		name string
		fn   func()
	}{
		{"2D-no-scratch", func() { convolve2DHighBDGoSIMD(dst, ref, 10, max, 0, 0, pad, pad, 32, 16, xk, yk) }},
		{"2D-scratch", func() { convolve2DHighBDGoSIMDWithScratch(dst, ref, 10, max, 0, 0, pad, pad, 32, 16, xk, yk, scratch) }},
		{"compound-2D", func() {
			predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(out, ref, pad, pad, 32, 16, xk, yk, round0, offsetBits, 10, &im)
		}},
	}
	for _, tc := range cases {
		if allocs := testing.AllocsPerRun(20, tc.fn); allocs != 0 {
			t.Errorf("%s allocated %v times, want zero", tc.name, allocs)
		}
	}
}

var hbd6TapBenchmarkSink uint16

type hbdSIMDBenchFilter struct {
	name string
	x, y [filterTaps]int16
}

func hbdSIMDBenchFilters() []hbdSIMDBenchFilter {
	return []hbdSIMDBenchFilter{
		{"regular8", subpelFilters8[6], subpelFilters8Smooth[9]},
		{"smooth8", subpelFilters8Smooth[6], subpelFilters8Smooth[9]},
		{"regular4", subpelFilters4[6], subpelFilters4[9]},
		{"smooth4", subpelFilters4Smooth[6], subpelFilters4Smooth[9]},
		{"bilinear", bilinearFilters[6], bilinearFilters[9]},
	}
}

// BenchmarkMotionHBD6TapConvolve compares the resident six-slot Go SIMD kernel
// and its production dispatch wrapper with NEON across common filter families
// and block shapes. The direct variant isolates kernel cost; the dispatched
// variant includes shape and endpoint checks.
func BenchmarkMotionHBD6TapConvolve(b *testing.B) {
	const bd = 10
	max, _ := highBDMax(bd)
	const pad = filterTaps
	ref := makeHighBDRef(maxBlockSize, pad, max, true, rand.New(rand.NewSource(0x6bdc)))
	sizes := []struct{ w, h int }{
		{8, 1}, {8, 2}, {8, 4}, {8, 8}, {8, 16}, {8, 32},
		{16, 1}, {16, 4}, {16, 8}, {16, 16},
		{32, 4}, {32, 8}, {32, 16}, {32, 32},
		{64, 64}, {128, 128},
	}
	for _, filter := range hbdSIMDBenchFilters() {
		filter := filter
		b.Run(filter.name, func(b *testing.B) {
			for _, size := range sizes {
				size := size
				for _, variant := range []string{"NEON", "GoSIMD-kernel", "GoSIMD-dispatch"} {
					variant := variant
					name := fmt.Sprintf("2D/%s/W%dH%d", variant, size.w, size.h)
					b.Run(name, func(b *testing.B) {
						dst, _ := testPlane(size.w, size.h, 2, size.w*2)
						scratch := &ConvolveScratch{}
						b.ReportAllocs()
						b.SetBytes(int64(size.w * size.h * 2))
						for b.Loop() {
							if variant == "NEON" {
								convolve2DHighBDNEONWithScratch(dst, ref, bd, max, 0, 0, pad, pad, size.w, size.h, filter.x, filter.y, scratch)
							} else if variant == "GoSIMD-kernel" {
								convolve2DHighBD6SIMDWithIM(dst, ref, bd, max, 0, 0, pad, pad, size.w, size.h, filter.x, filter.y, &scratch.imHBD[0])
							} else {
								convolve2DHighBDGoSIMDWithScratch(dst, ref, bd, max, 0, 0, pad, pad, size.w, size.h, filter.x, filter.y, scratch)
							}
						}
						hbd6TapBenchmarkSink = getSample(dst, 2, size.w-1, size.h-1)
					})
				}
			}
		})
	}
}

func BenchmarkMotionHBD6TapCompound(b *testing.B) {
	const bd = 10
	max, _ := highBDMax(bd)
	const pad = filterTaps
	ref := makeHighBDRef(maxBlockSize, pad, max, true, rand.New(rand.NewSource(0x6bdd)))
	round0 := compoundRound0(bd)
	offsetBits := bd + 2*filterBits - round0
	sizes := []struct{ w, h int }{
		{8, 1}, {8, 2}, {8, 4}, {8, 8}, {8, 16}, {8, 32},
		{16, 1}, {16, 4}, {16, 8}, {16, 16},
		{32, 4}, {32, 8}, {32, 16}, {32, 32},
		{64, 64}, {128, 128},
	}
	for _, filter := range hbdSIMDBenchFilters() {
		filter := filter
		b.Run(filter.name, func(b *testing.B) {
			for _, size := range sizes {
				size := size
				for _, variant := range []string{"NEON", "GoSIMD-kernel", "GoSIMD-dispatch"} {
					variant := variant
					name := fmt.Sprintf("2D/%s/W%dH%d", variant, size.w, size.h)
					b.Run(name, func(b *testing.B) {
						out := make([]uint16, size.w*size.h)
						var im compoundIM
						b.ReportAllocs()
						b.SetBytes(int64(size.w * size.h * 2))
						for b.Loop() {
							if variant == "NEON" {
								predictInterCompoundRefHighBDToConvBuf2DResidentNEON(out, ref, pad, pad, size.w, size.h, filter.x, filter.y, round0, offsetBits, bd, &im)
							} else if variant == "GoSIMD-kernel" {
								predictInterCompoundRefHighBDToConvBuf2D6SIMD(out, ref, pad, pad, size.w, size.h, filter.x, filter.y, round0, offsetBits, bd, &im)
							} else {
								predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(out, ref, pad, pad, size.w, size.h, filter.x, filter.y, round0, offsetBits, bd, &im)
							}
						}
						hbd6TapBenchmarkSink = out[len(out)-1]
					})
				}
			}
		})
	}
}

func assertCompoundHighBDBlockEqual(t *testing.T, got, want []uint16, tag string, bd uint8, phase any, size struct{ w, h int }) {
	t.Helper()
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: bd=%d phase=%v size=%dx%d sample=%d got=%d want=%d", tag, bd, phase, size.w, size.h, i, got[i], want[i])
		}
	}
}

func makeGoSIMDHighBDPlane(width, height int, max uint16, rng *rand.Rand, strideExtra, baseOffset int) frame.Plane {
	stride := width*2 + strideExtra
	pix := make([]byte, baseOffset+stride*height)
	plane := frame.Plane{Pix: pix[baseOffset:], Stride: stride, Width: width, Height: height}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			var value uint16
			switch (x + 3*y) % 3 {
			case 0:
				value = 0
			case 1:
				value = max
			default:
				value = uint16(rng.Intn(int(max) + 1))
			}
			storeHighBDSample(plane, x, y, value)
		}
	}
	return plane
}

func padGoSIMDHighBDPlane(src frame.Plane) frame.Plane {
	width, height := src.Width+2, src.Height+2
	stride := width * 2
	out := frame.Plane{Pix: make([]byte, stride*height), Stride: stride, Width: width, Height: height}
	for y := 0; y < src.Height; y++ {
		for x := 0; x < src.Width; x++ {
			storeHighBDSample(out, x+1, y+1, loadHighBDSample(src, x, y))
		}
	}
	return out
}
