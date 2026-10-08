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

func benchBlendCompoundAvg8Fair(b *testing.B, fn func(dst frame.Plane, src0, src1 []uint16, dstX, dstY, w, h, fwd, bck, roundOffset, roundBits int)) {
	const roundOffset, roundBits = 6144, 4
	const width, height = 32, 32
	src0 := make([]uint16, width*height)
	src1 := make([]uint16, width*height)
	for i := range src0 {
		src0[i] = uint16(3000 + i%5000)
		src1[i] = uint16(4000 + (i*7)%5000)
	}
	dst, _ := testPlane(width, height, 1, width)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(dst, src0, src1, 0, 0, width, height, 8, 8, roundOffset, roundBits)
	}
}

func BenchmarkBlendCompoundAvg8NEON_32(b *testing.B) {
	benchBlendCompoundAvg8Fair(b, blendCompoundAvg8NEON)
}

func BenchmarkBlendCompoundAvg8Scalar_32(b *testing.B) {
	benchBlendCompoundAvg8Fair(b, blendCompoundAvg8PureGo)
}
