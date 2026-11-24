// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import "testing"

// TestWienerHorizontalU8Dav1d5AllFilters runs every five-tap filter AV1 can
// code (f1 in [-23, 8], f2 in [-17, 46], f3 = -2*(f1+f2)) over sample patterns
// that drive the int16 sums and the [0, 8191] clamp to both ends, against the
// scalar reference.
func TestWienerHorizontalU8Dav1d5AllFilters(t *testing.T) {
	const width, height = 32, 2
	stride := width + 2*WienerHalfwin + 16
	origin := WienerHalfwin*stride + WienerHalfwin
	rows := height + 2*WienerHalfwin + 1
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xd5)
	patterns := make([][]uint8, 0, 8)
	for p := range 8 {
		src := make([]uint8, stride*rows)
		for i := range src {
			x := i % stride
			switch p {
			case 0:
				src[i] = 0
			case 1:
				src[i] = 255
			case 2:
				src[i] = uint8(255 * (x & 1))
			case 3:
				src[i] = uint8(255 * ((x >> 1) & 1))
			case 4:
				if x%5 == 3 {
					src[i] = 255
				}
			case 5:
				if x%5 != 3 {
					src[i] = 255
				}
			default:
				src[i] = uint8(rnd.pseudoUniform(256))
			}
		}
		patterns = append(patterns, src)
	}
	round0, _ := wienerRounds(8)
	tempLen := width * (height + 2*WienerHalfwin)
	want := make([]uint16, tempLen)
	got := make([]uint16, tempLen)
	for f1 := -23; f1 <= 8; f1++ {
		for f2 := -17; f2 <= 46; f2++ {
			f3 := -2 * (f1 + f2)
			filter := WienerFilter{0, int16(f1), int16(f2), int16(f3), int16(f2), int16(f1), 0, 0}
			if !wienerU8Dav1dTapsFit(filter) {
				t.Fatalf("coded filter f1=%d f2=%d does not fit", f1, f2)
			}
			for pi, src := range patterns {
				wienerHorizontalU8(src, stride, origin, width, height, filter, round0, want)
				wienerHorizontalU8Dav1d5(src, stride, origin, width, height, filter, got)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("f1=%d f2=%d pattern %d: temp[%d] = %d, want %d", f1, f2, pi, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func BenchmarkWienerHorizontalU8Five(b *testing.B) {
	const width, height = 64, 64
	stride := width + 2*WienerHalfwin + 16
	origin := WienerHalfwin*stride + WienerHalfwin
	src := make([]uint8, stride*(height+2*WienerHalfwin+1))
	rnd := newRestorationRandom(restorationDeterministicSeed)
	for i := range src {
		src[i] = uint8(rnd.pseudoUniform(256))
	}
	filter := WienerFilter{0, -7, 21, -28, 21, -7, 0, 0}
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	b.Run("dav1d", func(b *testing.B) {
		for b.Loop() {
			wienerHorizontalU8Dav1d5(src, stride, origin, width, height, filter, temp)
		}
	})
	b.Run("packed", func(b *testing.B) {
		for b.Loop() {
			wienerHorizontalU8SIMD5(src, stride, origin, width, height, filter, WienerRound0Bits, temp)
		}
	})
}
