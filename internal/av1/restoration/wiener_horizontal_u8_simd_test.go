// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package restoration

import (
	"fmt"
	"slices"
	"testing"
)

func TestWienerHorizontalU8SIMDLoadBounds(t *testing.T) {
	const width, height = 16, 5
	round0, _ := wienerRounds(8)
	filter := DefaultWienerInfo().HFilter
	for _, fixture := range []struct {
		name         string
		rowPad       int
		canVectorize bool
	}{
		{"minimal-border", 2 * WienerHalfwin, false},
		{"vector-border", 2*WienerHalfwin + 2, true},
	} {
		stride := width + fixture.rowPad
		origin := WienerHalfwin*stride + WienerHalfwin
		src := make([]uint8, stride*(height+2*WienerHalfwin))
		for i := range src {
			src[i] = uint8((i*37 + i/stride*19) & 0xff)
		}
		if got := wienerHorizontalU8SIMDCanRun(len(src), stride, origin, width, height); got != fixture.canVectorize {
			t.Fatalf("%s canVectorize=%v want %v", fixture.name, got, fixture.canVectorize)
		}
		want := make([]uint16, width*(height+2*WienerHalfwin))
		got := make([]uint16, len(want))
		wienerHorizontalU8(src, stride, origin, width, height, filter, round0, want)
		wienerHorizontalU8SIMDChecked(src, stride, origin, width, height, filter, round0, got)
		if !slices.Equal(got, want) {
			t.Fatalf("%s: NEON wrapper differs from scalar fallback", fixture.name)
		}
	}
}

func benchWienerHorizontalU8SIMD(b *testing.B, fn func([]uint8, int, int, int, int, WienerFilter, int, []uint16), width int) {
	const height = 64
	round0, _ := wienerRounds(8)
	stride := width + 2*WienerHalfwin + 2
	origin := WienerHalfwin*stride + WienerHalfwin
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x7c3)
	src := make([]uint8, stride*(height+2*WienerHalfwin))
	for i := range src {
		src[i] = uint8(rnd.pseudoUniform(256))
	}
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	filter := DefaultWienerInfo().HFilter
	if !wienerHorizontalU8SIMDCanRun(len(src), stride, origin, width, height) {
		b.Fatal("benchmark fixture does not enable the NEON vector kernel")
	}
	b.ReportAllocs()
	for b.Loop() {
		fn(src, stride, origin, width, height, filter, round0, temp)
	}
}

func BenchmarkWienerHorizontalU8_SIMD(b *testing.B) {
	for _, width := range []int{8, 32, 64, 256} {
		b.Run(fmt.Sprint(width), func(b *testing.B) {
			benchWienerHorizontalU8SIMD(b, wienerHorizontalU8SIMDChecked, width)
		})
	}
}

func BenchmarkWienerHorizontalU8_PureGo(b *testing.B) {
	for _, width := range []int{8, 32, 64, 256} {
		b.Run(fmt.Sprint(width), func(b *testing.B) {
			benchWienerHorizontalU8SIMD(b, wienerHorizontalU8, width)
		})
	}
}
