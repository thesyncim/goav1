// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

import "testing"

// Direct-call differentials for the 8-bit NEON kernels, independent of the
// dispatch bindings (which TestWiener*U8MatchesU16Widened already exercises on
// arm64). Widths below/off the 8-lane vector must hit the pure-Go fallback
// inside the wrappers and still match.

func TestWienerHorizontalU8NEONMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x8a1)
	round0, _ := wienerRounds(8)
	for _, sz := range u8DiffSizes {
		for fi, info := range wienerDispatchFilters() {
			stride := sz.width + 2*WienerHalfwin + 3
			origin := WienerHalfwin*stride + WienerHalfwin + 1
			src := randomU8Plane(rnd, stride, sz.height+2*WienerHalfwin)
			tempLen := sz.width * (sz.height + 2*WienerHalfwin)
			want := make([]uint16, tempLen)
			got := make([]uint16, tempLen)
			wienerHorizontalU8(src, stride, origin, sz.width, sz.height, info.HFilter, round0, want)
			wienerHorizontalU8NEON(src, stride, origin, sz.width, sz.height, info.HFilter, round0, got)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("sz=%dx%d f=%d temp[%d]=%d want %d", sz.width, sz.height, fi, i, got[i], want[i])
				}
			}
		}
	}
}

func TestWienerVerticalU8NEONMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x8a2)
	_, round1 := wienerRounds(8)
	for _, sz := range u8DiffSizes {
		for fi, info := range wienerDispatchFilters() {
			tempLen := sz.width * (sz.height + 2*WienerHalfwin)
			temp := make([]uint16, tempLen)
			// The horizontal pass bounds temp values by maxClamp = 8191.
			for i := range temp {
				temp[i] = uint16(rnd.pseudoUniform(8192))
			}
			dstStride := sz.width + 2
			want := make([]uint8, dstStride*sz.height)
			got := make([]uint8, dstStride*sz.height)
			wienerVerticalU8(temp, sz.width, want, dstStride, sz.width, sz.height, info.VFilter, round1)
			wienerVerticalU8NEON(temp, sz.width, got, dstStride, sz.width, sz.height, info.VFilter, round1)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("sz=%dx%d f=%d dst[%d]=%d want %d", sz.width, sz.height, fi, i, got[i], want[i])
				}
			}
		}
	}
}

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

// TestWienerU8NEONIsZeroAlloc protects the hot-path contract that the NEON
// wrappers do not allocate per call (the ctx structs must stay on the stack).
func TestWienerU8NEONIsZeroAlloc(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x8a4)
	round0, round1 := wienerRounds(8)
	stride := 64 + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	src := randomU8Plane(rnd, stride, 64+2*WienerHalfwin)
	temp := make([]uint16, 64*(64+2*WienerHalfwin))
	dst := make([]uint8, 64*64)
	info := DefaultWienerInfo()
	f0 := make([]int32, 64)
	f1 := make([]int32, 64)
	if allocs := testing.AllocsPerRun(200, func() {
		wienerHorizontalU8NEON(src, stride, origin, 64, 64, info.HFilter, round0, temp)
		wienerVerticalU8NEON(temp, 64, dst, 64, 64, 64, info.VFilter, round1)
		sgrWeightedRowU8NEON(dst[:64], src[origin:origin+64], f0, f1, 12, 116)
	}); allocs != 0 {
		t.Fatalf("u8 NEON wrappers allocated %f times per call", allocs)
	}
}
