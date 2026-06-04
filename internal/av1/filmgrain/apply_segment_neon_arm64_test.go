// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package filmgrain

import (
	"math/rand"
	"testing"
	"unsafe"
)

func TestApplyGrainSegmentNEONCtxSize(t *testing.T) {
	if size := unsafe.Sizeof(applyGrainSegmentNEONCtx{}); size != 56 {
		t.Fatalf("applyGrainSegmentNEONCtx size=%d want 56", size)
	}
}

func TestApplyGrainSegmentNEONMatchesPureGo(t *testing.T) {
	if !applyGrainSegmentUseNEON {
		t.Skip("NEON apply kernel not active on this CPU")
	}
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
					applyGrainSegmentNEON(got, src, scale, grain, shift, minValue, maxValue)
					for i := 0; i < n; i++ {
						if got[i] != ref[i] {
							t.Fatalf("bd=%d shift=%d n=%d restricted=%v i=%d: NEON=%d pureGo=%d (src=%d scale=%d grain=%d)",
								bd, shift, n, restricted, i, got[i], ref[i], src[i], scale[i], grain[i])
						}
					}
				}
			}
		}
	}
}

func TestApplyGrainSegmentNEONZeroAlloc(t *testing.T) {
	if !applyGrainSegmentUseNEON {
		t.Skip("NEON apply kernel not active on this CPU")
	}
	const n = 64
	dst, src, scale, grain := fuzzSegmentInputs(rand.New(rand.NewSource(1)), n, 8)
	allocs := testing.AllocsPerRun(1000, func() {
		applyGrainSegmentNEON(dst, src, scale, grain, 8, 0, 255)
	})
	if allocs != 0 {
		t.Fatalf("applyGrainSegmentNEON allocated: %f", allocs)
	}
}

func BenchmarkApplyGrainSegmentNEON(b *testing.B) {
	if !applyGrainSegmentUseNEON {
		b.Skip("NEON apply kernel not active on this CPU")
	}
	const n = 64
	dst, src, scale, grain := fuzzSegmentInputs(rand.New(rand.NewSource(7)), n, 8)
	b.ReportAllocs()
	for b.Loop() {
		applyGrainSegmentNEON(dst, src, scale, grain, 8, 0, 255)
	}
}
