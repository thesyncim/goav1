// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package prediction

import "testing"

// The AVX2 kernels are differentially validated against the pure-Go reference
// directly (not through the dispatch slot), because the cpu package stub does
// not yet report AVX2 so the runtime dispatch keeps the reference. This makes
// the AVX2 code byte-exact-tested on every amd64 build (native or Rosetta)
// regardless of the dispatch wiring.

func TestSubsampleLuma8AVX2MatchesPureGo(t *testing.T) {
	type cfg struct {
		w, h       int
		subX, subY bool
	}
	cfgs := []cfg{
		{8, 8, false, false}, {16, 16, false, false}, {32, 32, false, false},
		{24, 16, false, false},
		{16, 16, true, false}, {32, 16, true, false}, {16, 8, true, false},
		{48, 16, true, false},
		{16, 16, true, true}, {32, 32, true, true}, {16, 32, true, true},
		{8, 8, true, true}, {4, 4, true, true}, // width 4 -> pure-Go fallback
		{12, 8, false, false}, // outW=12 not %8 -> fallback
	}
	var seed uint32 = 0x1234
	for _, c := range cfgs {
		stride := c.w + 5
		input := make([]uint8, stride*c.h)
		for i := range input {
			seed = seed*1664525 + 1013904223
			input[i] = uint8(seed >> 13)
		}
		outW, outH := c.w, c.h
		if c.subX {
			outW >>= 1
		}
		if c.subY {
			outH >>= 1
		}
		got := make([]uint16, CFLBufSquare)
		want := make([]uint16, CFLBufSquare)
		subsampleLuma8AVX2(got, input, stride, c.w, c.h, outW, outH, c.subX, c.subY)
		subsampleLuma8PureGo(want, input, stride, c.w, c.h, outW, outH, c.subX, c.subY)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("subsample %+v idx=%d got=%d want=%d", c, i, got[i], want[i])
			}
		}
	}
}

// dirEdge builds a uint16 edge slice of length n holding 8-bit content.
func dirEdge(n int, seed uint32) []uint16 {
	out := make([]uint16, n)
	for i := range out {
		seed = seed*1664525 + 1013904223
		out[i] = uint16(seed >> 24)
	}
	return out
}

func TestDirRowInterp8AVX2MatchesPureGo(t *testing.T) {
	for _, width := range []int{8, 16, 24, 32, 64, 4, 12, 20} {
		above := dirEdge(width+8, 0x77+uint32(width))
		for _, shift := range []int{0, 1, 5, 12, 16, 31} {
			for _, base := range []int{0, 1, 3} {
				maxBase := width + base // fully interpolated (asm path when width%8==0)
				got := make([]byte, width)
				want := make([]byte, width)
				dirRowInterp8AVX2(got, above, base, shift, maxBase, width)
				dirRowInterp8PureGo(want, above, base, shift, maxBase, width)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("dirRowInterp w=%d shift=%d base=%d i=%d got=%d want=%d", width, shift, base, i, got[i], want[i])
					}
				}
				// Also exercise the clamp path (base+width > maxBase) -> pure-Go fallback.
				clampMax := base + width/2
				got2 := make([]byte, width)
				want2 := make([]byte, width)
				dirRowInterp8AVX2(got2, above, base, shift, clampMax, width)
				dirRowInterp8PureGo(want2, above, base, shift, clampMax, width)
				for i := range got2 {
					if got2[i] != want2[i] {
						t.Fatalf("dirRowInterp clamp w=%d shift=%d i=%d got=%d want=%d", width, shift, i, got2[i], want2[i])
					}
				}
			}
		}
	}
}

func TestDirAboveRun8AVX2MatchesPureGo(t *testing.T) {
	for _, count := range []int{8, 16, 24, 32, 64, 1, 7, 9, 15, 33} {
		ref := dirEdge(count+8, 0x314+uint32(count))
		for _, shift := range []int{0, 1, 7, 16, 30, 31} {
			got := make([]byte, count)
			want := make([]byte, count)
			dirAboveRun8AVX2(got, ref, shift, count)
			dirAboveRun8PureGo(want, ref, shift, count)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("dirAboveRun count=%d shift=%d i=%d got=%d want=%d", count, shift, i, got[i], want[i])
				}
			}
		}
	}
}

func TestDirLeftCol8AVX2MatchesPureGo(t *testing.T) {
	for _, count := range []int{8, 16, 24, 32, 64, 1, 7, 9, 15} {
		ref := dirEdge(count+8, 0x9a1+uint32(count))
		for _, stride := range []int{1, 5, 33} {
			for _, shift := range []int{0, 3, 16, 29, 31} {
				got := make([]byte, count*stride+8)
				want := make([]byte, count*stride+8)
				dirLeftCol8AVX2(got, stride, ref, shift, count)
				dirLeftCol8PureGo(want, stride, ref, shift, count)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("dirLeftCol count=%d stride=%d shift=%d i=%d got=%d want=%d", count, stride, shift, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestKernelsAVX2ZeroAlloc(t *testing.T) {
	// subsample
	input := make([]uint8, 40*32)
	for i := range input {
		input[i] = byte(i)
	}
	out := make([]uint16, CFLBufSquare)
	ss := func() { subsampleLuma8AVX2(out, input, 40, 32, 32, 16, 16, true, true) }
	if a := testing.AllocsPerRun(1000, ss); a != 0 {
		t.Fatalf("subsampleLuma8AVX2 allocated %f/call", a)
	}
	// dirRowInterp / dirAboveRun
	above := dirEdge(72, 0x5)
	dst := make([]byte, 64)
	ri := func() { dirRowInterp8AVX2(dst, above, 0, 11, 64, 64) }
	if a := testing.AllocsPerRun(1000, ri); a != 0 {
		t.Fatalf("dirRowInterp8AVX2 allocated %f/call", a)
	}
	ar := func() { dirAboveRun8AVX2(dst, above, 11, 64) }
	if a := testing.AllocsPerRun(1000, ar); a != 0 {
		t.Fatalf("dirAboveRun8AVX2 allocated %f/call", a)
	}
	// dirLeftCol
	col := make([]byte, 64*5+8)
	lc := func() { dirLeftCol8AVX2(col, 5, above, 11, 64) }
	if a := testing.AllocsPerRun(1000, lc); a != 0 {
		t.Fatalf("dirLeftCol8AVX2 allocated %f/call", a)
	}
}
