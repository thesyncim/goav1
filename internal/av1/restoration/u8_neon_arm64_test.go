// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

import "testing"

// Direct-call differentials for the 8-bit NEON self-guided projection,
// independent of the dispatch bindings. Widths off the 8-lane vector must hit
// the pure-Go fallback inside the wrapper and still match.

func TestSGRWeightedRowU8NEONMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x8a3)
	widths := []int{1, 4, 7, 8, 9, 16, 23, 33, 64}
	xqs := [][2]int32{{0, 128}, {31, 0}, {-96, 256}, {56, 72}}
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
			sgrWeightedRowU8NEON(got, src, f0, f1, xq[0], xq[1])
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("width=%d xq=%v dst[%d]=%d want %d", width, xq, i, got[i], want[i])
				}
			}
		}
	}
}

// TestSGRWeightedRowU8NEONIsZeroAlloc protects the hot-path contract that the
// NEON wrapper does not allocate per call (the ctx struct must stay on the stack).
func TestSGRWeightedRowU8NEONIsZeroAlloc(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x8a4)
	src := randomU8Plane(rnd, 64, 1)
	dst := make([]uint8, 64)
	f0 := make([]int32, 64)
	f1 := make([]int32, 64)
	if allocs := testing.AllocsPerRun(200, func() {
		sgrWeightedRowU8NEON(dst, src, f0, f1, 12, 116)
	}); allocs != 0 {
		t.Fatalf("u8 NEON projection allocated %f times per call", allocs)
	}
}
