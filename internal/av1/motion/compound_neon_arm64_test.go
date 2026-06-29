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

func TestCompoundHighBDCopyNEONMatchesPureGo(t *testing.T) {
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
			predictInterCompoundRefHighBDToConvBufCopyResidentNEON(got, ref, refX, refY, tc.width, tc.height, round0, roundOffset)
			predictInterCompoundRefHighBDToConvBufCopyResidentPureGo(want, ref, refX, refY, tc.width, tc.height, round0, roundOffset)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("bd=%d %dx%d sample %d: NEON=%d PureGo=%d", bitDepth, tc.width, tc.height, i, got[i], want[i])
				}
			}
		}
	}
}

func TestCompoundHighBDXNEONMatchesPureGo(t *testing.T) {
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
					predictInterCompoundRefHighBDToConvBufXResidentNEON(got, ref, pad, pad, size.width, size.height, kernel, round0, roundOffset)
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

func TestCompoundHighBDXNEONFallbackMatchesPureGo(t *testing.T) {
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
			predictInterCompoundRefHighBDToConvBufXResidentNEON(got, ref, refX, refY, tc.width, tc.height, tc.kernel, round0, roundOffset)
			predictInterCompoundRefHighBDToConvBufXResident(want, ref, refX, refY, tc.width, tc.height, tc.kernel, round0, roundOffset)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s sample=%d NEON wrapper=%d PureGo=%d", tc.name, i, got[i], want[i])
				}
			}
		})
	}
}

func TestCompoundHighBDYNEONMatchesPureGo(t *testing.T) {
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
					predictInterCompoundRefHighBDToConvBufYResidentNEON(got, ref, pad, pad, size.width, size.height, kernel, round0, roundOffset)
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

func TestCompoundHighBDYNEONFallbackMatchesPureGo(t *testing.T) {
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
	predictInterCompoundRefHighBDToConvBufYResidentNEON(got, ref, refX, refY, 12, 8, kernel, round0, roundOffset)
	predictInterCompoundRefHighBDToConvBufYResident(want, ref, refX, refY, 12, 8, kernel, round0, roundOffset)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("odd_width sample=%d NEON wrapper=%d PureGo=%d", i, got[i], want[i])
		}
	}
}

func TestCompoundHighBD2DNEONMatchesPureGo(t *testing.T) {
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
							predictInterCompoundRefHighBDToConvBuf2DResidentNEON(got, ref, pad, pad, size.width, size.height, xKernel, yKernel, round0, offsetBits, int(bitDepth), &gotIM)
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

func TestCompoundHighBD2DNEONFallbackMatchesPureGo(t *testing.T) {
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
	predictInterCompoundRefHighBDToConvBuf2DResidentNEON(got, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, 10, &gotIM)
	predictInterCompoundRefHighBDToConvBuf2DResident(want, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, 10, &wantIM)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("right_edge_fallback sample=%d NEON wrapper=%d PureGo=%d", i, got[i], want[i])
		}
	}
}

func BenchmarkCompoundConvBufXHighBDNEONDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	kernel := subpelFilters8[3]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBufXResidentNEON(out, ref, filterTaps, filterTaps, 32, 32, kernel, round0, roundOffset)
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

func BenchmarkCompoundConvBufYHighBDNEONDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	kernel := subpelFilters8[5]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBufYResidentNEON(out, ref, filterTaps, filterTaps, 32, 32, kernel, round0, roundOffset)
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

func BenchmarkCompoundConvBuf2DHighBDNEONDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	out := make([]uint16, 32*32)
	xKernel := subpelFilters8[3]
	yKernel := subpelFilters8[5]
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	var im compoundIM
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRefHighBDToConvBuf2DResidentNEON(out, ref, filterTaps, filterTaps, 32, 32, xKernel, yKernel, round0, offsetBits, 10, &im)
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
