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

func TestCompoundHighBDCopyGoSIMDMatchesPureGo(t *testing.T) {
	const (
		refW   = 48
		refH   = 19
		stride = 112
		refX   = 5
		refY   = 4
	)
	for _, bitDepth := range []uint8{10, 12} {
		ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
		mask := (1 << int(bitDepth)) - 1
		for y := range refH {
			for x := range refW {
				storeHighBDSample(ref, x, y, uint16((x*97+y*53+x*y*11+int(bitDepth)*17)&mask))
			}
		}
		round0 := compoundRound0(bitDepth)
		offsetBits := int(bitDepth) + 2*filterBits - round0
		roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
		for _, tc := range [...]struct {
			width  int
			height int
		}{
			{width: 8, height: 1},
			{width: 16, height: 7},
			{width: 32, height: 11},
		} {
			got := make([]uint16, tc.width*tc.height)
			want := make([]uint16, tc.width*tc.height)
			predictInterCompoundRefHighBDToConvBufCopyResidentGoSIMD(got, ref, refX, refY, tc.width, tc.height, round0, roundOffset)
			predictInterCompoundRefHighBDToConvBufCopyResidentPureGo(want, ref, refX, refY, tc.width, tc.height, round0, roundOffset)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("bd=%d %dx%d sample %d: NEON=%d PureGo=%d", bitDepth, tc.width, tc.height, i, got[i], want[i])
				}
			}
		}
	}
}

func TestCompoundHighBDXGoSIMDMatchesPureGo(t *testing.T) {
	const (
		pad = filterTaps
	)
	rng := rand.New(rand.NewSource(0x4b1d0a))
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
		{4, 13},
		{8, 1},
		{16, 7},
		{32, 11},
		{64, 16},
	}
	for _, bitDepth := range []uint8{10, 12} {
		max := uint16((1 << int(bitDepth)) - 1)
		round0 := compoundRound0(bitDepth)
		offsetBits := int(bitDepth) + 2*filterBits - round0
		roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
		for _, size := range sizes {
			refW := size.width + 2*pad
			refH := size.height + 2*pad
			stride := refW * 2
			ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
			for y := range refH {
				for x := range refW {
					storeHighBDSample(ref, x, y, uint16(rng.Intn(int(max)+1)))
				}
			}
			for _, filter := range filters {
				for subX := 1; subX <= subpelQ4Mask; subX++ {
					kernel, err := interpKernel(filter, size.width, subX)
					if err != nil {
						t.Fatal(err)
					}
					got := make([]uint16, size.width*size.height)
					want := make([]uint16, size.width*size.height)
					predictInterCompoundRefHighBDToConvBufXResidentGoSIMD(got, ref, pad, pad, size.width, size.height, kernel, round0, roundOffset)
					predictInterCompoundRefHighBDToConvBufXResident(want, ref, pad, pad, size.width, size.height, kernel, round0, roundOffset)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("bd=%d filter=%d size=%dx%d subX=%d sample=%d NEON=%d PureGo=%d",
								bitDepth, filter, size.width, size.height, subX, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestCompoundHighBDXGoSIMDFallbackMatchesPureGo(t *testing.T) {
	const (
		refW   = 48
		refH   = 24
		stride = refW * 2
		refX   = filterTaps
		refY   = filterTaps
	)
	ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
	fillHighBDMotionTestPlane(ref, 0x3ff)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	kernel := subpelFilters8[5]
	for _, tc := range []struct {
		name          string
		width, height int
		kernel        [filterTaps]int16
	}{
		{name: "odd_width", width: 12, height: 8, kernel: kernel},
		{name: "width4_non_4tap", width: 4, height: 8, kernel: kernel},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := make([]uint16, tc.width*tc.height)
			want := make([]uint16, tc.width*tc.height)
			predictInterCompoundRefHighBDToConvBufXResidentGoSIMD(got, ref, refX, refY, tc.width, tc.height, tc.kernel, round0, roundOffset)
			predictInterCompoundRefHighBDToConvBufXResident(want, ref, refX, refY, tc.width, tc.height, tc.kernel, round0, roundOffset)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s sample=%d NEON wrapper=%d PureGo=%d", tc.name, i, got[i], want[i])
				}
			}
		})
	}
}

func TestCompoundHighBDYGoSIMDMatchesPureGo(t *testing.T) {
	const pad = filterTaps
	rng := rand.New(rand.NewSource(0x7c0f21))
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
		{4, 13},
		{8, 1},
		{16, 7},
		{32, 11},
		{64, 16},
	}
	for _, bitDepth := range []uint8{10, 12} {
		max := uint16((1 << int(bitDepth)) - 1)
		round0 := compoundRound0(bitDepth)
		offsetBits := int(bitDepth) + 2*filterBits - round0
		roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
		for _, size := range sizes {
			refW := size.width + 2*pad
			refH := size.height + 2*pad
			stride := refW * 2
			ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
			for y := range refH {
				for x := range refW {
					storeHighBDSample(ref, x, y, uint16(rng.Intn(int(max)+1)))
				}
			}
			for _, filter := range filters {
				for subY := 1; subY <= subpelQ4Mask; subY++ {
					kernel, err := interpKernel(filter, size.height, subY)
					if err != nil {
						t.Fatal(err)
					}
					got := make([]uint16, size.width*size.height)
					want := make([]uint16, size.width*size.height)
					predictInterCompoundRefHighBDToConvBufYResidentGoSIMD(got, ref, pad, pad, size.width, size.height, kernel, round0, roundOffset)
					predictInterCompoundRefHighBDToConvBufYResident(want, ref, pad, pad, size.width, size.height, kernel, round0, roundOffset)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("bd=%d filter=%d size=%dx%d subY=%d sample=%d NEON=%d PureGo=%d",
								bitDepth, filter, size.width, size.height, subY, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestCompoundHighBDYGoSIMDFallbackMatchesPureGo(t *testing.T) {
	const (
		refW   = 48
		refH   = 24
		stride = refW * 2
		refX   = filterTaps
		refY   = filterTaps
	)
	ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
	fillHighBDMotionTestPlane(ref, 0x3ff)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	kernel := subpelFilters8[5]
	got := make([]uint16, 12*8)
	want := make([]uint16, 12*8)
	predictInterCompoundRefHighBDToConvBufYResidentGoSIMD(got, ref, refX, refY, 12, 8, kernel, round0, roundOffset)
	predictInterCompoundRefHighBDToConvBufYResident(want, ref, refX, refY, 12, 8, kernel, round0, roundOffset)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("odd_width sample=%d NEON wrapper=%d PureGo=%d", i, got[i], want[i])
		}
	}
}

func TestCompoundHighBD2DGoSIMDMatchesPureGo(t *testing.T) {
	const pad = filterTaps
	rng := rand.New(rand.NewSource(0x2d4c91))
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
		{4, 13},
		{8, 1},
		{16, 7},
		{32, 11},
		{64, 16},
	}
	for _, bitDepth := range []uint8{10, 12} {
		max := uint16((1 << int(bitDepth)) - 1)
		round0 := compoundRound0(bitDepth)
		offsetBits := int(bitDepth) + 2*filterBits - round0
		for _, size := range sizes {
			refW := size.width + 2*pad
			refH := size.height + 2*pad
			stride := refW * 2
			ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
			for y := range refH {
				for x := range refW {
					storeHighBDSample(ref, x, y, uint16(rng.Intn(int(max)+1)))
				}
			}
			for _, xFilter := range filters {
				for _, yFilter := range filters {
					for subX := 1; subX <= subpelQ4Mask; subX++ {
						xKernel, err := interpKernel(xFilter, size.width, subX)
						if err != nil {
							t.Fatal(err)
						}
						for subY := 1; subY <= subpelQ4Mask; subY++ {
							yKernel, err := interpKernel(yFilter, size.height, subY)
							if err != nil {
								t.Fatal(err)
							}
							got := make([]uint16, size.width*size.height)
							want := make([]uint16, size.width*size.height)
							var gotIM compoundIM
							var wantIM compoundIM
							predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(got, ref, pad, pad, size.width, size.height, xKernel, yKernel, round0, offsetBits, int(bitDepth), &gotIM)
							predictInterCompoundRefHighBDToConvBuf2DResident(want, ref, pad, pad, size.width, size.height, xKernel, yKernel, round0, offsetBits, int(bitDepth), &wantIM)
							for i := range want {
								if got[i] != want[i] {
									t.Fatalf("bd=%d filters=%d/%d size=%dx%d sub=%d/%d sample=%d NEON=%d PureGo=%d",
										bitDepth, xFilter, yFilter, size.width, size.height, subX, subY, i, got[i], want[i])
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestCompoundHighBD2DGoSIMDFallbackMatchesPureGo(t *testing.T) {
	const (
		width  = 12
		height = 8
		refX   = filterTaps
		refY   = filterTaps
	)
	fo := filterTaps/2 - 1
	refW := refX - fo + width + filterTaps - 1
	refH := refY - fo + height + filterTaps - 1
	stride := refW * 2
	ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
	fillHighBDMotionTestPlane(ref, 0x3ff)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	xKernel := subpelFilters8[3]
	yKernel := subpelFilters8[5]
	got := make([]uint16, width*height)
	want := make([]uint16, width*height)
	var gotIM compoundIM
	var wantIM compoundIM
	predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(got, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, 10, &gotIM)
	predictInterCompoundRefHighBDToConvBuf2DResident(want, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, 10, &wantIM)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("right_edge_fallback sample=%d NEON wrapper=%d PureGo=%d", i, got[i], want[i])
		}
	}
}

// TestCompoundHighBD2DClampedEmuEdgeMatchesPureGo drives the HBD compound 2D
// clamped NEON path over tap windows that overhang the reference plane on every
// side, forcing the emu_edge halo materialization, and asserts bit-identity
// with the pure-Go per-tap-clamping reference at bit depths 10 and 12.
func TestCompoundHighBD2DClampedEmuEdgeMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x2d16bced))
	const refW, refH = 24, 20
	sizes := []int{4, 8, 12, 16, 32, 64}
	for _, bitDepth := range []uint8{10, 12} {
		max := uint16((1 << int(bitDepth)) - 1)
		round0 := compoundRound0(bitDepth)
		offsetBits := int(bitDepth) + 2*filterBits - round0
		ref := frame.Plane{Pix: make([]byte, refW*2*refH), Stride: refW * 2, Width: refW, Height: refH}
		for y := 0; y < refH; y++ {
			for x := 0; x < refW; x++ {
				v := uint16(rng.Intn(int(max) + 1))
				if (x == 0 || x == refW-1) && (y == 0 || y == refH-1) {
					if (x+y)&1 == 0 {
						v = 0
					} else {
						v = max
					}
				}
				storeHighBDSample(ref, x, y, v)
			}
		}
		xKernel := subpelFilters8[6]
		yKernel := subpelFilters8[9]
		for _, w := range sizes {
			for _, h := range sizes {
				offs := [][2]int{
					{-w, -h}, {-3, -3}, {2, -5}, {refW - 2, -1},
					{refW - 1, 5}, {refW, refH}, {5, refH - 2},
					{-4, refH - 3}, {-6, 4}, {4, 4},
				}
				for _, o := range offs {
					got := make([]uint16, w*h)
					gotEdge := make([]uint16, w*h)
					want := make([]uint16, w*h)
					var gotIM, wantIM compoundIM
					var edge emuEdge16Buf
					// Poison the caller-owned edge window to prove every sample
					// read by the resident kernel was materialized first.
					for i := range edge {
						edge[i] = 0xa5
					}
					predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(got, ref, o[0], o[1], w, h, xKernel, yKernel, round0, offsetBits, int(bitDepth), &gotIM, nil)
					predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(gotEdge, ref, o[0], o[1], w, h, xKernel, yKernel, round0, offsetBits, int(bitDepth), &gotIM, &edge)
					predictInterCompoundRefHighBDToConvBuf2DClamped(want, ref, o[0], o[1], w, h, xKernel, yKernel, round0, offsetBits, int(bitDepth), &wantIM)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("bd=%d %dx%d off=%v sample=%d NEON=%d PureGo=%d",
								bitDepth, w, h, o, i, got[i], want[i])
						}
						if gotEdge[i] != want[i] {
							t.Fatalf("bd=%d %dx%d off=%v sample=%d NEON(edge)=%d PureGo=%d",
								bitDepth, w, h, o, i, gotEdge[i], want[i])
						}
					}
				}
			}
		}
	}
}

// TestCompoundHighBD2DClampedEmuEdgeZeroAlloc proves the emu_edge halo scratch
// stays on the stack through the func-ptr dispatch slot.
func TestCompoundHighBD2DClampedEmuEdgeZeroAlloc(t *testing.T) {
	const refW, refH = 24, 20
	ref := frame.Plane{Pix: make([]byte, refW*2*refH), Stride: refW * 2, Width: refW, Height: refH}
	fillHighBDMotionTestPlane(ref, 0x3ff)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	xKernel := subpelFilters8[6]
	yKernel := subpelFilters8[9]
	out := make([]uint16, 64*64)
	var im compoundIM
	if a := testing.AllocsPerRun(20, func() {
		predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(out, ref, -3, -3, 64, 64, xKernel, yKernel, round0, offsetBits, 10, &im, nil)
	}); a != 0 {
		t.Fatalf("emu_edge compound 2D clamped allocated %v times, want 0", a)
	}
	var edge emuEdge16Buf
	if a := testing.AllocsPerRun(20, func() {
		predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(out, ref, -3, -3, 64, 64, xKernel, yKernel, round0, offsetBits, 10, &im, &edge)
	}); a != 0 {
		t.Fatalf("emu_edge compound 2D clamped (caller edge) allocated %v times, want 0", a)
	}
}

func BenchmarkCompoundHighBD2DClampedEmuEdge(b *testing.B) {
	const refW, refH = 24, 20
	ref := frame.Plane{Pix: make([]byte, refW*2*refH), Stride: refW * 2, Width: refW, Height: refH}
	fillHighBDMotionTestPlane(ref, 0x3ff)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	xKernel := subpelFilters8[6]
	yKernel := subpelFilters8[9]
	for _, w := range []int{8, 16, 32, 64} {
		out := make([]uint16, w*w)
		var im compoundIM
		b.Run("neon_"+itoaW(w), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(out, ref, -3, -3, w, w, xKernel, yKernel, round0, offsetBits, 10, &im, nil)
			}
		})
		b.Run("purego_"+itoaW(w), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				predictInterCompoundRefHighBDToConvBuf2DClamped(out, ref, -3, -3, w, w, xKernel, yKernel, round0, offsetBits, 10, &im)
			}
		})
	}
}

func BenchmarkCompoundConvBufXHighBDGoSIMDDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	kernel := subpelFilters8[3]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBufXResidentGoSIMD(out, ref, filterTaps, filterTaps, 32, 32, kernel, round0, roundOffset)
	})
}

func BenchmarkCompoundConvBufXHighBDPureGoDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	kernel := subpelFilters8[3]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBufXResident(out, ref, filterTaps, filterTaps, 32, 32, kernel, round0, roundOffset)
	})
}

func BenchmarkCompoundConvBufYHighBDGoSIMDDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	kernel := subpelFilters8[5]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBufYResidentGoSIMD(out, ref, filterTaps, filterTaps, 32, 32, kernel, round0, roundOffset)
	})
}

func BenchmarkCompoundConvBufYHighBDPureGoDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	kernel := subpelFilters8[5]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBufYResident(out, ref, filterTaps, filterTaps, 32, 32, kernel, round0, roundOffset)
	})
}

func BenchmarkCompoundConvBuf2DHighBDGoSIMDDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	xKernel := subpelFilters8[3]
	yKernel := subpelFilters8[5]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	var im compoundIM
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(out, ref, filterTaps, filterTaps, 32, 32, xKernel, yKernel, round0, offsetBits, 10, &im)
	})
}

func BenchmarkCompoundConvBuf2DHighBDPureGoDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	xKernel := subpelFilters8[3]
	yKernel := subpelFilters8[5]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	var im compoundIM
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBuf2DResident(out, ref, filterTaps, filterTaps, 32, 32, xKernel, yKernel, round0, offsetBits, 10, &im)
	})
}
