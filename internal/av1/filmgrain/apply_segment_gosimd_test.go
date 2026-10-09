// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package filmgrain

import (
	"math/rand"
	"runtime"
	"simd/archsimd"
	"testing"
)

// TestApplyGrainSegmentSIMDMatchesPureGo drives the Go SIMD kernel directly
// rather than through the dispatch bool, so the vector path is covered on every
// host. On amd64 this runs AVX2 encodings even where CPUID hides AVX2 (Rosetta 2).
func TestApplyGrainSegmentSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xF11A))
	bitDepths := []uint8{8, 10, 12}
	shifts := []int{8, 9, 10, 11}
	// Cover full groups, partial tails, and sub-group lengths.
	lengths := []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64}
	for _, bd := range bitDepths {
		maxSample := (1 << bd) - 1
		for _, shift := range shifts {
			for _, n := range lengths {
				_, src, scale, grain := fuzzSegmentInputs(rng, n, bd)
				ref := make([]uint16, n)
				got := make([]uint16, n)
				for _, restricted := range []bool{false, true} {
					minValue, maxValue := 0, maxSample
					if restricted {
						minValue = LumaLegalMin << (bd - 8)
						maxValue = LumaLegalMax << (bd - 8)
					}
					applyGrainSegmentPureGo(ref, src, scale, grain, shift, minValue, maxValue)
					applyGrainSegmentSIMD(got, src, scale, grain, shift, minValue, maxValue)
					for i := 0; i < n; i++ {
						if got[i] != ref[i] {
							t.Fatalf("bd=%d shift=%d n=%d restricted=%v i=%d: SIMD=%d pureGo=%d (src=%d scale=%d grain=%d)",
								bd, shift, n, restricted, i, got[i], ref[i], src[i], scale[i], grain[i])
						}
					}
				}
			}
		}
	}
}

// TestApplyGrainSegmentDispatchBindsSIMD asserts the dispatch selects the Go SIMD
// kernel when the platform supports it: always on arm64 (NEON is mandatory), and
// on amd64 whenever AVX2 is present.
func TestApplyGrainSegmentDispatchBindsSIMD(t *testing.T) {
	want := runtime.GOARCH == "arm64" || archsimd.X86.AVX2()
	if applyGrainSegmentUseSIMD != want {
		t.Fatalf("applyGrainSegmentUseSIMD=%v want %v", applyGrainSegmentUseSIMD, want)
	}
}

func TestApplyGrainSegmentSIMDZeroAlloc(t *testing.T) {
	const n = 64
	dst, src, scale, grain := fuzzSegmentInputs(rand.New(rand.NewSource(1)), n, 8)
	allocs := testing.AllocsPerRun(1000, func() {
		applyGrainSegmentSIMD(dst, src, scale, grain, 8, 0, 255)
	})
	if allocs != 0 {
		t.Fatalf("applyGrainSegmentSIMD allocated: %f", allocs)
	}
}

func BenchmarkApplyGrainSegmentSIMD(b *testing.B) {
	const n = 64
	dst, src, scale, grain := fuzzSegmentInputs(rand.New(rand.NewSource(7)), n, 8)
	b.ReportAllocs()
	for b.Loop() {
		applyGrainSegmentSIMD(dst, src, scale, grain, 8, 0, 255)
	}
}
