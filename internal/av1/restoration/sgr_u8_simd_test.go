// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package restoration

import (
	"reflect"
	"runtime"
	"testing"
)

// Direct-call differential for the 8-bit Go-native SIMD self-guided projection,
// independent of the dispatch binding so the SIMD path runs on every host. Widths
// off the vector span must hit the scalar tail and still match.

func TestSGRWeightedRowU8SIMDMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x8b3)
	widths := []int{1, 4, 7, 8, 9, 15, 16, 17, 23, 24, 31, 33, 64}
	xqs := [][2]int32{{0, 128}, {31, 0}, {-96, 256}, {56, 72}, {-128, 0}, {127, 127}}
	for _, width := range widths {
		for _, xq := range xqs {
			src := randomU8Plane(rnd, width, 1)
			f0 := make([]int32, width)
			f1 := make([]int32, width)
			for i := range f0 {
				f0[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
				f1[i] = int32(rnd.pseudoUniform(1<<21)) - (1 << 20)
			}
			want := make([]uint8, width)
			got := make([]uint8, width)
			sgrWeightedRowU8(want, src, f0, f1, xq[0], xq[1])
			sgrWeightedRowU8SIMD(got, src, f0, f1, xq[0], xq[1])
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("width=%d xq=%v dst[%d]=%d want %d", width, xq, i, got[i], want[i])
				}
			}
		}
	}
}

// TestSGRWeightedRowU8SIMDExtremes hardens the projection at the int16 wrap and
// the [0,255] clamp boundaries: every sample value against extreme flt rows.
func TestSGRWeightedRowU8SIMDExtremes(t *testing.T) {
	const width = 48
	flts := []int32{-(1 << 20), -1, 0, 1, 1 << 20, 1<<21 - 1}
	xqs := [][2]int32{{0, 128}, {-96, 256}, {127, 127}, {-128, 0}}
	for s := 0; s < 256; s++ {
		src := make([]uint8, width)
		for i := range src {
			src[i] = uint8(s)
		}
		for _, f := range flts {
			f0 := make([]int32, width)
			f1 := make([]int32, width)
			for i := range f0 {
				f0[i] = f
				f1[i] = -f
			}
			for _, xq := range xqs {
				want := make([]uint8, width)
				got := make([]uint8, width)
				sgrWeightedRowU8(want, src, f0, f1, xq[0], xq[1])
				sgrWeightedRowU8SIMD(got, src, f0, f1, xq[0], xq[1])
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("s=%d f=%d xq=%v dst[%d]=%d want %d", s, f, xq, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestSGRWeightedRowU8SIMDIsZeroAlloc protects the hot-path contract that the
// 8-bit SIMD projection does not allocate per call.
func TestSGRWeightedRowU8SIMDIsZeroAlloc(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x8a4)
	src := randomU8Plane(rnd, 64, 1)
	dst := make([]uint8, 64)
	f0 := make([]int32, 64)
	f1 := make([]int32, 64)
	if allocs := testing.AllocsPerRun(200, func() {
		sgrWeightedRowU8SIMD(dst, src, f0, f1, 12, 116)
	}); allocs != 0 {
		t.Fatalf("u8 SIMD projection allocated %f times per call", allocs)
	}
}

func TestSGRWeightedRowU8SIMDIsBound(t *testing.T) {
	if !sgrSIMDDetected() {
		t.Skip("Go SIMD detection failed on this host")
	}
	nameOf := func(v any) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	if got, want := nameOf(sgrWeightedRowU8Impl), nameOf(sgrWeightedRowU8SIMD); got != want {
		t.Fatalf("sgrWeightedRowU8Impl = %s, want %s", got, want)
	}
}
