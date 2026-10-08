// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package restoration

import "testing"

func TestWienerVerticalU8SIMDMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5c3)
	_, round1 := wienerRounds(8)
	sizes := append([]struct{ width, height int }{{8, 8}, {16, 8}, {62, 12}, {17, 4}, {31, 7}, {7, 5}}, u8DiffSizes...)
	for _, sz := range sizes {
		for fi, info := range wienerDispatchFilters() {
			tempLen := sz.width * (sz.height + 2*WienerHalfwin)
			temp := make([]uint16, tempLen)
			for i := range temp {
				temp[i] = uint16(rnd.pseudoUniform(8192))
			}
			dstStride := sz.width + 2
			want := make([]uint8, dstStride*sz.height)
			got := make([]uint8, dstStride*sz.height)
			wienerVerticalU8(temp, sz.width, want, dstStride, sz.width, sz.height, info.VFilter, round1)
			wienerVerticalU8SIMD(temp, sz.width, got, dstStride, sz.width, sz.height, info.VFilter, round1)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("sz=%dx%d f=%d dst[%d]=%d want %d", sz.width, sz.height, fi, i, got[i], want[i])
				}
			}
		}
	}

	// The maximum horizontal-pass sample makes each symmetric pair sum to
	// 16382, still a positive int16 before the widening multiply.
	const width, height = 16, 8
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	for i := range temp {
		temp[i] = 8191
	}
	filter := wienerDispatchFilters()[0].VFilter
	want := make([]uint8, width*height)
	got := make([]uint8, width*height)
	wienerVerticalU8(temp, width, want, width, width, height, filter, round1)
	wienerVerticalU8SIMD(temp, width, got, width, width, height, filter, round1)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("max pair sum dst[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

func TestWienerVerticalU8SIMDAsymmetricFilterFallback(t *testing.T) {
	const width, height = 8, 5
	_, round1 := wienerRounds(8)
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5c4)
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	for i := range temp {
		temp[i] = uint16(rnd.pseudoUniform(8192))
	}
	filter := WienerFilter{17, -21, 8, 112, 4, -21, 19}
	want := make([]uint8, width*height)
	got := make([]uint8, width*height)
	wienerVerticalU8(temp, width, want, width, width, height, filter, round1)
	wienerVerticalU8SIMD(temp, width, got, width, width, height, filter, round1)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("asymmetric filter dst[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

var wienerU8VerticalBenchmarkSink uint8

func benchWienerVerticalU8(b *testing.B, fn func([]uint16, int, []uint8, int, int, int, WienerFilter, int), width int) {
	const height = 64
	_, round1 := wienerRounds(8)
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	for i := range temp {
		temp[i] = uint16((i*97 + 31) % 8192)
	}
	dst := make([]uint8, width*height)
	info := wienerDispatchFilters()[0]
	b.ReportAllocs()
	b.SetBytes(int64(width * height))
	b.ResetTimer()
	for b.Loop() {
		fn(temp, width, dst, width, width, height, info.VFilter, round1)
	}
	b.StopTimer()
	wienerU8VerticalBenchmarkSink = dst[len(dst)-1]
}

func BenchmarkWienerVerticalU8_8_SIMD(b *testing.B) {
	benchWienerVerticalU8(b, wienerVerticalU8SIMD, 8)
}
func BenchmarkWienerVerticalU8_8_PureGo(b *testing.B) {
	benchWienerVerticalU8(b, wienerVerticalU8, 8)
}

func BenchmarkWienerVerticalU8_32_SIMD(b *testing.B) {
	benchWienerVerticalU8(b, wienerVerticalU8SIMD, 32)
}
func BenchmarkWienerVerticalU8_32_PureGo(b *testing.B) {
	benchWienerVerticalU8(b, wienerVerticalU8, 32)
}

func BenchmarkWienerVerticalU8_64_SIMD(b *testing.B) {
	benchWienerVerticalU8(b, wienerVerticalU8SIMD, 64)
}
func BenchmarkWienerVerticalU8_64_PureGo(b *testing.B) {
	benchWienerVerticalU8(b, wienerVerticalU8, 64)
}

func BenchmarkWienerVerticalU8_256_SIMD(b *testing.B) {
	benchWienerVerticalU8(b, wienerVerticalU8SIMD, 256)
}
func BenchmarkWienerVerticalU8_256_PureGo(b *testing.B) {
	benchWienerVerticalU8(b, wienerVerticalU8, 256)
}
