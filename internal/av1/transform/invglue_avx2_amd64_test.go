// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package transform

import "testing"

// invGlueLCG is a tiny deterministic generator so the differential tests are
// reproducible without pulling in math/rand.
type invGlueLCG struct{ s uint64 }

func (g *invGlueLCG) next() uint64 {
	g.s = g.s*6364136223846793005 + 1442695040888963407
	return g.s >> 11
}

// stage-range-representative signed int32 in [-1<<21, 1<<21).
func (g *invGlueLCG) stageVal() int32 {
	return int32(g.next()%(1<<22)) - (1 << 21)
}

// TestClampRoundAVX2MatchesPureGo runs the AVX2 mid-pass round-and-clamp and the
// pure-Go reference over identical inputs and asserts lane-for-lane equality. It
// calls clampRoundAVX2 directly so the vector path executes even on hosts that
// do not advertise AVX2 in CPUID (notably Rosetta 2, which runs AVX2 but reports
// it absent). It sweeps shift 0 (clamp-only) and several positive shifts, with
// [min, max] windows that force both the low and high clamp, over
// multiple-of-eight lengths plus a non-multiple length to exercise the pure-Go
// tail fallback.
func TestClampRoundAVX2MatchesPureGo(t *testing.T) {
	shifts := []int{0, 1, 2, 3, 4, 6, 8, 12}
	bounds := []struct{ min, max int32 }{
		{-(1 << 18), (1 << 18) - 1},
		{-(1 << 10), (1 << 10) - 1},
		{-4096, 4095},
		{0, 255},
	}
	lengths := []int{8, 16, 24, 64, 256, 13}

	for _, shift := range shifts {
		for _, b := range bounds {
			for _, n := range lengths {
				g := invGlueLCG{s: uint64(shift*7919 + int(b.max)*31 + n*17 + 1)}
				got := make([]int32, n)
				want := make([]int32, n)
				for i := range got {
					v := g.stageVal()
					got[i] = v
					want[i] = v
				}
				clampRoundAVX2(got, shift, b.min, b.max)
				clampRoundPureGo(want, shift, b.min, b.max)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("shift=%d min=%d max=%d n=%d lane %d: avx2=%d ref=%d",
							shift, b.min, b.max, n, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestNarrowStoreAVX2MatchesPureGo runs the AVX2 final narrowing store and the
// pure-Go reference over identical inputs and asserts element-for-element
// equality (including the past-width dst region, seeded with a sentinel to catch
// stray writes). It calls narrowStoreAVX2 directly to execute the vector path
// under Rosetta 2. It sweeps width four and multiples of eight (plus a non-4,
// non-multiple-of-8 width for the tail fallback), with source values large
// enough to force int16 saturation on both sides.
func TestNarrowStoreAVX2MatchesPureGo(t *testing.T) {
	widths := []int{4, 8, 16, 24, 32, 64, 12}
	heights := []int{1, 2, 4, 7}

	for _, width := range widths {
		for _, height := range heights {
			g := invGlueLCG{s: uint64(width*131 + height*29 + 3)}
			dstStride := width + 5
			scratch := make([]int32, width*height)
			for i := range scratch {
				switch g.next() & 3 {
				case 0:
					// Large magnitude to force int16 saturation after >>4.
					scratch[i] = int32(g.next()%(1<<25)) - (1 << 24)
				default:
					scratch[i] = int32(g.next()%(1<<19)) - (1 << 18)
				}
			}

			gotDst := make([]int16, dstStride*height)
			wantDst := make([]int16, dstStride*height)
			for i := range gotDst {
				gotDst[i] = -12345
				wantDst[i] = -12345
			}

			narrowStoreAVX2(gotDst, dstStride, scratch, width, height)
			narrowStorePureGo(wantDst, dstStride, scratch, width, height)

			for i := range gotDst {
				if gotDst[i] != wantDst[i] {
					t.Fatalf("w=%d h=%d elem %d: avx2=%d ref=%d", width, height, i, gotDst[i], wantDst[i])
				}
			}
		}
	}
}

func TestInvGlueAVX2IsZeroAlloc(t *testing.T) {
	scratch := make([]int32, 1024)
	for i := range scratch {
		scratch[i] = int32(i%2048) - 1024
	}
	dst := make([]int16, 32*32)
	if a := testing.AllocsPerRun(1000, func() {
		clampRoundAVX2(scratch, 4, -4096, 4095)
	}); a != 0 {
		t.Fatalf("clampRoundAVX2 allocated: %f", a)
	}
	if a := testing.AllocsPerRun(1000, func() {
		narrowStoreAVX2(dst, 32, scratch, 32, 32)
	}); a != 0 {
		t.Fatalf("narrowStoreAVX2 allocated: %f", a)
	}
}
