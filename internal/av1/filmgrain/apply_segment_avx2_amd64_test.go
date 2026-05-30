// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package filmgrain

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// TestApplyGrainSegmentAVX2Status prints the runtime AVX2 detection result so
// the test log records whether the AVX2 kernel executed or the build fell back
// to pure-Go (e.g. under Rosetta 2, which does not expose AVX2).
func TestApplyGrainSegmentAVX2Status(t *testing.T) {
	t.Logf("cpu.Detected.AVX2=%v applyGrainSegmentUseAVX2=%v", cpu.Detected.AVX2, applyGrainSegmentUseAVX2)
	if applyGrainSegmentUseAVX2 {
		t.Log("film-grain apply segment: AVX2 kernel ACTIVE")
	} else {
		t.Log("film-grain apply segment: AVX2 unavailable, using pure-Go fallback")
	}
}

// TestApplyGrainSegmentAVX2MatchesPureGo drives the AVX2 kernel directly rather
// than gating on applyGrainSegmentUseAVX2: the kernel is always built on
// amd64 && !purego, and on hosts where the AVX2 instructions execute (including
// Rosetta 2, which runs them even though it hides AVX2 in CPUID and so disables
// the dispatch) this gives full differential coverage. If the host genuinely
// lacks AVX2 the VEX-encoded ops fault, which surfaces as a hard test failure —
// the desired signal that something is wrong with the build environment.
func TestApplyGrainSegmentAVX2MatchesPureGo(t *testing.T) {
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
					applyGrainSegmentAVX2(got, src, scale, grain, shift, minValue, maxValue)
					for i := 0; i < n; i++ {
						if got[i] != ref[i] {
							t.Fatalf("bd=%d shift=%d n=%d restricted=%v i=%d: AVX2=%d pureGo=%d (src=%d scale=%d grain=%d)",
								bd, shift, n, restricted, i, got[i], ref[i], src[i], scale[i], grain[i])
						}
					}
				}
			}
		}
	}
}

func TestApplyGrainSegmentAVX2ZeroAlloc(t *testing.T) {
	const n = 64
	dst, src, scale, grain := fuzzSegmentInputs(rand.New(rand.NewSource(1)), n, 8)
	allocs := testing.AllocsPerRun(1000, func() {
		applyGrainSegmentAVX2(dst, src, scale, grain, 8, 0, 255)
	})
	if allocs != 0 {
		t.Fatalf("applyGrainSegmentAVX2 allocated: %f", allocs)
	}
}

func BenchmarkApplyGrainSegmentAVX2(b *testing.B) {
	const n = 64
	dst, src, scale, grain := fuzzSegmentInputs(rand.New(rand.NewSource(7)), n, 8)
	b.ReportAllocs()
	for b.Loop() {
		applyGrainSegmentAVX2(dst, src, scale, grain, 8, 0, 255)
	}
}
