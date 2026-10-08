// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package restoration

import "testing"

// Direct-call differential for the 8-bit AVX2 self-guided projection, independent
// of cpu.Detected so the SIMD path runs even when auto-dispatch falls back.
// Widths off the 8-lane vector must hit the pure-Go tail and still match.

func TestSGRWeightedRowU8AVX2MatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x8b3)
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
			sgrWeightedRowU8AVX2(got, src, f0, f1, xq[0], xq[1])
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("width=%d xq=%v dst[%d]=%d want %d", width, xq, i, got[i], want[i])
				}
			}
		}
	}
}
