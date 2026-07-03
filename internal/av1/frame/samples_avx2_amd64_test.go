// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package frame

import "testing"

// widenLCG is a tiny deterministic generator so the differential test is
// reproducible without pulling in math/rand.
type widenLCG struct{ s uint64 }

func (g *widenLCG) next() uint64 {
	g.s = g.s*6364136223846793005 + 1442695040888963407
	return g.s >> 11
}

// TestLoadSampleRows8AVX2MatchesPureGo runs the AVX2 widen and the pure-Go
// reference over identical inputs and asserts element-for-element equality. It
// calls the AVX2 kernel directly (not through the wrapper) so it exercises the
// vector path even on hosts that do not advertise AVX2 in CPUID — notably
// Rosetta 2, which executes AVX2 but reports it absent. It sweeps power-of-two
// and edge (non-multiple-of-8, non-multiple-of-16) widths >= 8 to cover the
// 16-wide body, the 8-wide mid group, and the overlapping row-end tail, with a
// dst stride wider than the visible width to catch any past-width write.
func TestLoadSampleRows8AVX2MatchesPureGo(t *testing.T) {
	widths := []int{8, 9, 13, 15, 16, 17, 23, 24, 31, 32, 33, 48, 63, 64, 65, 127}
	heights := []int{1, 2, 3, 5, 8}

	for _, width := range widths {
		for _, height := range heights {
			g := widenLCG{s: uint64(width*131 + height*17 + 7)}
			srcStride := width + 5
			dstStride := width + 11

			src := make([]byte, srcStride*height)
			for i := range src {
				src[i] = byte(g.next())
			}

			// Seed both destinations with a sentinel so any stray write past
			// the visible [0,width) region shows up as a mismatch.
			gotDst := make([]uint16, dstStride*height)
			wantDst := make([]uint16, dstStride*height)
			for i := range gotDst {
				gotDst[i] = 0xBEEF
				wantDst[i] = 0xBEEF
			}

			loadSampleRows8PureGo(wantDst, dstStride, src, srcStride, width, height)
			loadSampleRows8AVX2Asm(
				&gotDst[0], &src[0],
				uintptr(dstStride), uintptr(srcStride),
				uintptr(width), uintptr(height),
			)

			for i := range gotDst {
				if gotDst[i] != wantDst[i] {
					t.Fatalf("w=%d h=%d elem %d: avx2=%#04x ref=%#04x", width, height, i, gotDst[i], wantDst[i])
				}
			}
		}
	}
}

func TestLoadSampleRows8AVX2IsZeroAlloc(t *testing.T) {
	const w, h = 64, 64
	src := make([]byte, w*h)
	for i := range src {
		src[i] = byte(i)
	}
	dst := make([]uint16, w*h)
	allocs := testing.AllocsPerRun(1000, func() {
		loadSampleRows8AVX2Asm(&dst[0], &src[0], uintptr(w), uintptr(w), uintptr(w), uintptr(h))
	})
	if allocs != 0 {
		t.Fatalf("loadSampleRows8AVX2Asm allocated: %f", allocs)
	}
}
