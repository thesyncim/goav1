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

func TestBlendCompoundAvg8NEONMatchesPureGo(t *testing.T) {
	const (
		stride = 160
		planeH = 140
	)
	rng := rand.New(rand.NewSource(0x8a17ee))
	roundOffset := (1 << (19 - compoundRound1Bits)) + (1 << (19 - compoundRound1Bits - 1))
	const roundBits = 2*filterBits - compoundRound0Bits - compoundRound1Bits
	weights := [...][2]int{{8, 8}, {9, 7}, {7, 9}, {11, 5}, {5, 11}, {12, 4}, {4, 12}, {13, 3}, {3, 13}}
	sizes := [...]struct {
		width  int
		height int
	}{
		{2, 2}, {2, 8}, {4, 4}, {4, 7}, {4, 16}, {8, 2}, {8, 8}, {8, 32},
		{16, 5}, {16, 16}, {32, 8}, {32, 32}, {64, 16}, {64, 64}, {128, 128},
	}
	for _, w := range weights {
		for _, sz := range sizes {
			for _, dstOff := range [...][2]int{{0, 0}, {3, 5}} {
				src0 := make([]uint16, sz.width*sz.height)
				src1 := make([]uint16, sz.width*sz.height)
				for i := range src0 {
					switch i % 7 {
					case 0:
						src0[i] = 0
						src1[i] = 65535
					case 1:
						src0[i] = 65535
						src1[i] = 0
					default:
						src0[i] = uint16(rng.Intn(1 << 16))
						src1[i] = uint16(rng.Intn(1 << 16))
					}
				}
				got := frame.Plane{Pix: make([]byte, stride*planeH), Stride: stride, Width: stride, Height: planeH}
				want := frame.Plane{Pix: make([]byte, stride*planeH), Stride: stride, Width: stride, Height: planeH}
				for i := range got.Pix {
					got.Pix[i] = byte(i * 31)
					want.Pix[i] = byte(i * 31)
				}
				blendCompoundAvg8NEON(got, src0, src1, dstOff[0], dstOff[1], sz.width, sz.height, w[0], w[1], roundOffset, roundBits)
				blendCompoundAvg8PureGo(want, src0, src1, dstOff[0], dstOff[1], sz.width, sz.height, w[0], w[1], roundOffset, roundBits)
				for i := range got.Pix {
					if got.Pix[i] != want.Pix[i] {
						t.Fatalf("w=%v %dx%d dst=(%d,%d) byte %d: NEON=%d PureGo=%d",
							w, sz.width, sz.height, dstOff[0], dstOff[1], i, got.Pix[i], want.Pix[i])
					}
				}
			}
		}
	}
}

func TestBlendCompoundAvg8NEONZeroAlloc(t *testing.T) {
	const width, height = 32, 32
	src0 := make([]uint16, width*height)
	src1 := make([]uint16, width*height)
	for i := range src0 {
		src0[i] = uint16(i * 37)
		src1[i] = uint16(i * 91)
	}
	dst := frame.Plane{Pix: make([]byte, 64*64), Stride: 64, Width: 64, Height: 64}
	roundOffset := (1 << (19 - compoundRound1Bits)) + (1 << (19 - compoundRound1Bits - 1))
	const roundBits = 2*filterBits - compoundRound0Bits - compoundRound1Bits
	allocs := testing.AllocsPerRun(100, func() {
		blendCompoundAvg8NEON(dst, src0, src1, 0, 0, width, height, 9, 7, roundOffset, roundBits)
	})
	if allocs != 0 {
		t.Fatalf("blendCompoundAvg8NEON allocates: %v allocs/run", allocs)
	}
}

func benchBlendCompoundAvg8(b *testing.B, width, height int, fn func(dst frame.Plane, src0, src1 []uint16, dstX, dstY, w, h, fwd, bck, roundOffset, roundBits int)) {
	src0 := make([]uint16, width*height)
	src1 := make([]uint16, width*height)
	for i := range src0 {
		src0[i] = uint16(i * 37)
		src1[i] = uint16(i * 91)
	}
	dst := frame.Plane{Pix: make([]byte, 192*192), Stride: 192, Width: 192, Height: 192}
	roundOffset := (1 << (19 - compoundRound1Bits)) + (1 << (19 - compoundRound1Bits - 1))
	const roundBits = 2*filterBits - compoundRound0Bits - compoundRound1Bits
	runConvolveBench(b, width, height, func() {
		fn(dst, src0, src1, 0, 0, width, height, 9, 7, roundOffset, roundBits)
	})
}

func BenchmarkBlendCompoundAvg8NEONDirect_32(b *testing.B) {
	benchBlendCompoundAvg8(b, 32, 32, blendCompoundAvg8NEON)
}

func BenchmarkBlendCompoundAvg8PureGoDirect_32(b *testing.B) {
	benchBlendCompoundAvg8(b, 32, 32, blendCompoundAvg8PureGo)
}

func BenchmarkBlendCompoundAvg8NEONDirect_8(b *testing.B) {
	benchBlendCompoundAvg8(b, 8, 8, blendCompoundAvg8NEON)
}

func BenchmarkBlendCompoundAvg8PureGoDirect_8(b *testing.B) {
	benchBlendCompoundAvg8(b, 8, 8, blendCompoundAvg8PureGo)
}

func BenchmarkBlendCompoundAvg8NEONDirect_4x8(b *testing.B) {
	benchBlendCompoundAvg8(b, 4, 8, blendCompoundAvg8NEON)
}

func BenchmarkBlendCompoundAvg8PureGoDirect_4x8(b *testing.B) {
	benchBlendCompoundAvg8(b, 4, 8, blendCompoundAvg8PureGo)
}
