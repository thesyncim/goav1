// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"reflect"
	"runtime"
	"testing"
)

// TestWienerHorizontalU8SIMDIsBound guards the wiring: under the
// goexperiment.simd build the u8 horizontal dispatch slot must resolve to the
// Go-native SIMD kernel (the NEON binding in u8_dispatch_arm64.go is excluded
// there).
func TestWienerHorizontalU8SIMDIsBound(t *testing.T) {
	nameOf := func(v any) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	if got := nameOf(wienerHorizontalU8Impl); got != nameOf(wienerHorizontalU8SIMD) {
		t.Fatalf("wienerHorizontalU8Impl = %s, want wienerHorizontalU8SIMD", got)
	}
}

// TestWienerHorizontalU8SIMDMatchesReference is the byte-exactness
// differential: the Go-SIMD horizontal kernel must equal the scalar
// wienerHorizontalU8 reference AND the NEON asm wrapper sample-for-sample
// across the size corpus (vector widths, %16==8 tails, non-multiple-of-8
// fallback widths) and the full dispatch filter set, over random planes plus
// extreme pixel patterns (all-0, all-255, alternating 0/255) that maximize the
// symmetric tap-pair sums and the clamp edges.
func TestWienerHorizontalU8SIMDMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x7c1)
	round0, _ := wienerRounds(8)
	sizes := append([]struct{ width, height int }{{8, 8}, {16, 8}, {62, 12}, {17, 4}, {31, 7}, {7, 5}, {48, 3}}, u8DiffSizes...)
	fill := func(src []uint8, mode int) {
		switch mode {
		case 0:
			for i := range src {
				src[i] = uint8(rnd.pseudoUniform(256))
			}
		case 1:
			for i := range src {
				src[i] = 255
			}
		case 2:
			for i := range src {
				src[i] = 0
			}
		default:
			for i := range src {
				src[i] = uint8(255 * (i & 1))
			}
		}
	}
	for mode := 0; mode < 4; mode++ {
		for _, sz := range sizes {
			for fi, info := range wienerDispatchFilters() {
				stride := sz.width + 2*WienerHalfwin + 3
				origin := WienerHalfwin*stride + WienerHalfwin + 1
				src := make([]uint8, stride*(sz.height+2*WienerHalfwin))
				fill(src, mode)
				tempLen := sz.width * (sz.height + 2*WienerHalfwin)
				want := make([]uint16, tempLen)
				gotSIMD := make([]uint16, tempLen)
				gotASM := make([]uint16, tempLen)
				wienerHorizontalU8(src, stride, origin, sz.width, sz.height, info.HFilter, round0, want)
				wienerHorizontalU8SIMD(src, stride, origin, sz.width, sz.height, info.HFilter, round0, gotSIMD)
				wienerHorizontalU8NEON(src, stride, origin, sz.width, sz.height, info.HFilter, round0, gotASM)
				for i := range want {
					if gotSIMD[i] != want[i] {
						t.Fatalf("SIMD: mode=%d sz=%dx%d f=%d temp[%d]=%d want %d",
							mode, sz.width, sz.height, fi, i, gotSIMD[i], want[i])
					}
					if gotASM[i] != want[i] {
						t.Fatalf("NEON: mode=%d sz=%dx%d f=%d temp[%d]=%d want %d",
							mode, sz.width, sz.height, fi, i, gotASM[i], want[i])
					}
				}
			}
		}
	}
}

// TestWienerHorizontalU8SIMDIsZeroAlloc protects the hot-path zero-allocation
// contract for the SIMD horizontal kernel.
func TestWienerHorizontalU8SIMDIsZeroAlloc(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x7c2)
	round0, _ := wienerRounds(8)
	stride := 64 + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	src := randomU8Plane(rnd, stride, 64+2*WienerHalfwin)
	temp := make([]uint16, 64*(64+2*WienerHalfwin))
	info := DefaultWienerInfo()
	if allocs := testing.AllocsPerRun(200, func() {
		wienerHorizontalU8SIMD(src, stride, origin, 64, 64, info.HFilter, round0, temp)
	}); allocs != 0 {
		t.Fatalf("wienerHorizontalU8SIMD allocated %f times per call", allocs)
	}
}

func benchWienerHorizontalU8(b *testing.B, fn func([]uint8, int, int, int, int, WienerFilter, int, []uint16), width int) {
	const height = 64
	round0, _ := wienerRounds(8)
	stride := width + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x7c3)
	src := randomU8Plane(rnd, stride, height+2*WienerHalfwin)
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	filter := DefaultWienerInfo().HFilter
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		fn(src, stride, origin, width, height, filter, round0, temp)
	}
}

func BenchmarkWienerHorizontalU8_8_SIMD(b *testing.B) {
	benchWienerHorizontalU8(b, wienerHorizontalU8SIMD, 8)
}
func BenchmarkWienerHorizontalU8_8_NEON(b *testing.B) {
	benchWienerHorizontalU8(b, wienerHorizontalU8NEON, 8)
}
func BenchmarkWienerHorizontalU8_8_PureGo(b *testing.B) {
	benchWienerHorizontalU8(b, wienerHorizontalU8, 8)
}

func BenchmarkWienerHorizontalU8_32_SIMD(b *testing.B) {
	benchWienerHorizontalU8(b, wienerHorizontalU8SIMD, 32)
}
func BenchmarkWienerHorizontalU8_32_NEON(b *testing.B) {
	benchWienerHorizontalU8(b, wienerHorizontalU8NEON, 32)
}
func BenchmarkWienerHorizontalU8_32_PureGo(b *testing.B) {
	benchWienerHorizontalU8(b, wienerHorizontalU8, 32)
}

func BenchmarkWienerHorizontalU8_64_SIMD(b *testing.B) {
	benchWienerHorizontalU8(b, wienerHorizontalU8SIMD, 64)
}
func BenchmarkWienerHorizontalU8_64_NEON(b *testing.B) {
	benchWienerHorizontalU8(b, wienerHorizontalU8NEON, 64)
}
func BenchmarkWienerHorizontalU8_64_PureGo(b *testing.B) {
	benchWienerHorizontalU8(b, wienerHorizontalU8, 64)
}

func BenchmarkWienerHorizontalU8_256_SIMD(b *testing.B) {
	benchWienerHorizontalU8(b, wienerHorizontalU8SIMD, 256)
}
func BenchmarkWienerHorizontalU8_256_NEON(b *testing.B) {
	benchWienerHorizontalU8(b, wienerHorizontalU8NEON, 256)
}
