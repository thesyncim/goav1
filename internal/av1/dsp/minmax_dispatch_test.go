// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package dsp

import "testing"

// TestMinMaxAbsDiff8x8DispatchMatchesPureGo confirms the dispatch indirection
// is bit-exact with the canonical pure-Go reference across every (stride,
// bytesPerSample) shape the kernel accepts. This is the round-trip guard
// for future SIMD variants: once one is wired in, this test (plus the
// existing libaom reference comparisons) will catch any divergence.
func TestMinMaxAbsDiff8x8DispatchMatchesPureGo(t *testing.T) {
	for _, bytesPerSample := range []int{1, 2} {
		var a [8 * 64]byte
		var b [8 * 64]byte
		fillMinMaxBytes(a[:], 0xdeadbeef)
		fillMinMaxBytes(b[:], 0xfeedface)
		minStride := 8 * bytesPerSample
		for aStride := minStride; aStride <= 64; aStride += minStride {
			for bStride := minStride; bStride <= 64; bStride += minStride {
				wantMin, wantMax, wantErr := minMaxAbsDiff8x8PureGo(a[:], aStride, b[:], bStride, bytesPerSample)
				gotMin, gotMax, gotErr := MinMaxAbsDiff8x8(a[:], aStride, b[:], bStride, bytesPerSample)
				if wantErr != gotErr {
					t.Fatalf("bps=%d aStride=%d bStride=%d err mismatch: dispatch=%v reference=%v", bytesPerSample, aStride, bStride, gotErr, wantErr)
				}
				if gotMin != wantMin || gotMax != wantMax {
					t.Fatalf("bps=%d aStride=%d bStride=%d dispatch=(%d,%d) reference=(%d,%d)",
						bytesPerSample, aStride, bStride, gotMin, gotMax, wantMin, wantMax)
				}
			}
		}
	}
}

// TestMinMaxAbsDiff8x8DispatchIsZeroAlloc protects the steady-state contract
// that hot DSP entries do not allocate. The dispatch indirection must remain
// allocation-free, otherwise the project's hot-path invariants regress.
func TestMinMaxAbsDiff8x8DispatchIsZeroAlloc(t *testing.T) {
	var a [8 * 64]byte
	var b [8 * 64]byte
	fillMinMaxBytes(a[:], 0x11111111)
	fillMinMaxBytes(b[:], 0x22222222)
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, err := MinMaxAbsDiff8x8(a[:], 64, b[:], 64, 1)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("dispatch allocated %f times per call", allocs)
	}
}
