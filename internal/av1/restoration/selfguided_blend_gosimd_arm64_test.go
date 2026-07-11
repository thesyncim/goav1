// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"reflect"
	"testing"
)

// blendDiffWidths spans the 8-lane vector, its <8 tails, the odd widths that
// exercise the tail fallback, and the 64-wide processing unit.
var blendDiffWidths = []int{1, 4, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 40, 56, 63, 64}

// blendDiffXQ mixes the decode-xq extremes so the projection lands both inside
// [0,max] and well past the int16 wrap boundary (|rounded| > 32767) and the
// clamp edges (0 and max).
var blendDiffXQ = [][2]int32{
	{0, 128}, {31, 0}, {-96, 256}, {56, 72}, {128, -64}, {-128, 255}, {96, -96}, {0, 0},
}

// TestSGRWeightedRowU8SIMDMatchesReference is the byte-exactness gate for the
// 8-bit blend: the archsimd kernel must equal sgrWeightedRowU8 sample for
// sample, including the int16 wrap and [0,255] clamp edges.
func TestSGRWeightedRowU8SIMDMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5117)
	for iter := 0; iter < 40; iter++ {
		for _, width := range blendDiffWidths {
			for _, xq := range blendDiffXQ {
				src := randomU8Plane(rnd, width, 1)
				f0 := make([]int32, width)
				f1 := make([]int32, width)
				for i := range f0 {
					// Wide magnitudes so xq*(f-u) drives rounded past int16.
					f0[i] = int32(rnd.pseudoUniform(1<<22)) - (1 << 21)
					f1[i] = int32(rnd.pseudoUniform(1<<22)) - (1 << 21)
				}
				want := make([]uint8, width)
				got := make([]uint8, width)
				sgrWeightedRowU8(want, src, f0, f1, xq[0], xq[1])
				sgrWeightedRowU8SIMD(got, src, f0, f1, xq[0], xq[1])
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("u8 width=%d xq=%v dst[%d]=%d want %d", width, xq, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestSGRWeightedRowSIMDMatchesReference is the byte-exactness gate for the
// high-bit-depth blend across 8/10/12-bit maxima.
func TestSGRWeightedRowSIMDMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5217)
	for _, maxI := range []int32{255, 1023, 4095} {
		for iter := 0; iter < 20; iter++ {
			for _, width := range blendDiffWidths {
				for _, xq := range blendDiffXQ {
					src := make([]uint16, width)
					for i := range src {
						src[i] = uint16(rnd.pseudoUniform(int(maxI) + 1))
					}
					f0 := make([]int32, width)
					f1 := make([]int32, width)
					for i := range f0 {
						f0[i] = int32(rnd.pseudoUniform(1<<23)) - (1 << 22)
						f1[i] = int32(rnd.pseudoUniform(1<<23)) - (1 << 22)
					}
					want := make([]uint16, width)
					got := make([]uint16, width)
					sgrWeightedRow(want, src, f0, f1, xq[0], xq[1], maxI)
					sgrWeightedRowSIMD(got, src, f0, f1, xq[0], xq[1], maxI)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("u16 max=%d width=%d xq=%v dst[%d]=%d want %d",
								maxI, width, xq, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

// TestSGRBlendDispatchBindsSIMD confirms the goexperiment.simd build routes the
// dispatch slots through the archsimd kernels (not the NEON asm / pure-Go).
func TestSGRBlendDispatchBindsSIMD(t *testing.T) {
	if reflect.ValueOf(sgrWeightedRowU8Impl).Pointer() != reflect.ValueOf(sgrWeightedRowU8SIMD).Pointer() {
		t.Fatal("sgrWeightedRowU8Impl not bound to the SIMD kernel under goexperiment.simd")
	}
	if reflect.ValueOf(sgrWeightedRowImpl).Pointer() != reflect.ValueOf(sgrWeightedRowSIMD).Pointer() {
		t.Fatal("sgrWeightedRowImpl not bound to the SIMD kernel under goexperiment.simd")
	}
}

// TestSGRBlendSIMDIsZeroAlloc protects the hot-path contract: the staging
// arrays and broadcast constants stay on the stack, no per-call allocation.
func TestSGRBlendSIMDIsZeroAlloc(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5317)
	const width = 64
	src8 := randomU8Plane(rnd, width, 1)
	src16 := make([]uint16, width)
	dst8 := make([]uint8, width)
	dst16 := make([]uint16, width)
	f0 := make([]int32, width)
	f1 := make([]int32, width)
	for i := range f0 {
		src16[i] = uint16(rnd.pseudoUniform(1024))
		f0[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
		f1[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
	}
	if allocs := testing.AllocsPerRun(200, func() {
		sgrWeightedRowU8SIMD(dst8, src8, f0, f1, 12, 116)
		sgrWeightedRowSIMD(dst16, src16, f0, f1, 12, 116, 1023)
	}); allocs != 0 {
		t.Fatalf("SGR blend SIMD kernels allocated %f times per call", allocs)
	}
}

func benchU8Blend(b *testing.B, fn func([]uint8, []uint8, []int32, []int32, int32, int32)) {
	const width = 64
	rnd := newRestorationRandom(0x9931)
	src := randomU8Plane(rnd, width, 1)
	f0 := make([]int32, width)
	f1 := make([]int32, width)
	for i := range f0 {
		f0[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
		f1[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
	}
	dst := make([]uint8, width)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(dst, src, f0, f1, 12, 116)
	}
}

func BenchmarkSGRWeightedRowU8_SIMD(b *testing.B)   { benchU8Blend(b, sgrWeightedRowU8SIMD) }
func BenchmarkSGRWeightedRowU8_NEON(b *testing.B)   { benchU8Blend(b, sgrWeightedRowU8NEON) }
func BenchmarkSGRWeightedRowU8_PureGo(b *testing.B) { benchU8Blend(b, sgrWeightedRowU8) }

func benchU16Blend(b *testing.B, fn func([]uint16, []uint16, []int32, []int32, int32, int32, int32)) {
	const width = 64
	rnd := newRestorationRandom(0x9932)
	src := make([]uint16, width)
	f0 := make([]int32, width)
	f1 := make([]int32, width)
	for i := range f0 {
		src[i] = uint16(rnd.pseudoUniform(1024))
		f0[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
		f1[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
	}
	dst := make([]uint16, width)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(dst, src, f0, f1, 12, 116, 1023)
	}
}

func BenchmarkSGRWeightedRowU16_SIMD(b *testing.B)   { benchU16Blend(b, sgrWeightedRowSIMD) }
func BenchmarkSGRWeightedRowU16_PureGo(b *testing.B) { benchU16Blend(b, sgrWeightedRow) }
