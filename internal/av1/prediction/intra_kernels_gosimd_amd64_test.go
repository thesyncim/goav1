// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

import "testing"

// The amd64 SIMD kernels are differentially validated against the pure-Go
// references directly, not through the dispatch slot, so the kernels are tested on
// every amd64 build that has SIMD (native or Rosetta). Dispatch binding is covered
// by TestFilterIntra8SIMDBinding and TestStaticSIMDBinding.

func TestPaethSIMDMatchesPureGo(t *testing.T) {
	for _, w := range []int{16, 32, 48, 64} {
		for _, h := range []int{4, 8, 16, 32, 64} {
			above, s1 := samplesFor(w, 0xff, 0x111+uint32(w))
			left, _ := samplesFor(h, 0xff, s1)
			for _, al := range []uint16{0, 1, 128, 255} {
				base := makeDispatchBlock(w, h, 1)
				got := cloneBlock(base)
				want := cloneBlock(base)
				predictPaethSIMD(got, 1, above, left, al)
				predictPaethPureGo(want, 1, above, left, al)
				diffBlocks(t, "paeth-avx2", 0xff, w, h, got, want)
			}
		}
	}
}

func TestPaethSIMDEdgeValues(t *testing.T) {
	const w, h = 32, 16
	patterns := []func(i int) uint16{
		func(i int) uint16 { return 0 },
		func(i int) uint16 { return 255 },
		func(i int) uint16 {
			if i%2 == 0 {
				return 0
			}
			return 255
		},
		func(i int) uint16 { return uint16(i * 17 % 256) },
	}
	for _, ap := range patterns {
		for _, lp := range patterns {
			above := make([]uint16, w)
			left := make([]uint16, h)
			for i := range above {
				above[i] = ap(i)
			}
			for i := range left {
				left[i] = lp(i)
			}
			for _, al := range []uint16{0, 128, 255} {
				base := makeDispatchBlock(w, h, 1)
				got := cloneBlock(base)
				want := cloneBlock(base)
				predictPaethSIMD(got, 1, above, left, al)
				predictPaethPureGo(want, 1, above, left, al)
				diffBlocks(t, "paeth-avx2-edge", 0xff, w, h, got, want)
			}
		}
	}
}

func TestSmoothSIMDMatchesPureGo(t *testing.T) {
	for _, w := range []int{16, 32, 64} {
		for _, h := range []int{4, 8, 16, 32, 64} {
			above, s1 := samplesFor(w, 0xff, 0x222+uint32(w))
			left, _ := samplesFor(h, 0xff, s1)
			weightsW, _ := smoothWeightsForSize(w)
			weightsH, _ := smoothWeightsForSize(h)
			belowPred := left[h-1]
			rightPred := above[w-1]
			base := makeDispatchBlock(w, h, 1)

			got := cloneBlock(base)
			want := cloneBlock(base)
			predictSmoothSIMD(got, 1, weightsW, weightsH, above, left, belowPred, rightPred)
			predictSmoothPureGo(want, 1, weightsW, weightsH, above, left, belowPred, rightPred)
			diffBlocks(t, "smooth-avx2", 0xff, w, h, got, want)

			got = cloneBlock(base)
			want = cloneBlock(base)
			predictSmoothVerticalSIMD(got, 1, weightsH, above, belowPred)
			predictSmoothVerticalPureGo(want, 1, weightsH, above, belowPred)
			diffBlocks(t, "smooth_v-avx2", 0xff, w, h, got, want)

			got = cloneBlock(base)
			want = cloneBlock(base)
			predictSmoothHorizontalSIMD(got, 1, weightsW, left, rightPred)
			predictSmoothHorizontalPureGo(want, 1, weightsW, left, rightPred)
			diffBlocks(t, "smooth_h-avx2", 0xff, w, h, got, want)
		}
	}
}

func TestApplyCFLSIMDMatchesPureGo(t *testing.T) {
	dims := [][2]int{{16, 16}, {32, 32}, {16, 8}, {32, 16}, {16, 32}}
	var seed uint32 = 0x7c3
	for _, dim := range dims {
		w, h := dim[0], dim[1]
		ac := make([]int16, CFLBufSquare)
		for i := range ac {
			seed = seed*1664525 + 1013904223
			ac[i] = int16(int32(seed>>8)%4096 - 2048)
		}
		for _, alpha := range []int{-16, -7, -1, 0, 1, 5, 16} {
			base := makeDispatchBlock(w, h, 1)
			for i := range base.pix {
				base.pix[i] = byte((i*13 + 7) & 0xff)
			}
			got := cloneBlock(base)
			want := cloneBlock(base)
			applyCFLSIMD(got, 1, w, h, ac, alpha, 0xff)
			applyCFLPureGo(want, 1, w, h, ac, alpha, 0xff)
			diffBlocks(t, "applycfl-avx2", 0xff, w, h, got, want)
		}
	}
}

func TestSumSamplesSIMDMatchesPureGo(t *testing.T) {
	for _, n := range []int{16, 32, 48, 64, 20, 4, 100} {
		for _, max := range []uint16{0xff, 0x3ff, 0xfff} {
			s, _ := samplesFor(n, max, 0x5a5+uint32(n)+uint32(max))
			if got, want := sumSamplesSIMD(s), sumSamplesPureGo(s); got != want {
				t.Fatalf("sumSamplesSIMD n=%d max=%d got=%d want=%d", n, max, got, want)
			}
		}
	}
}

func TestSubsampleLuma8SIMDMatchesPureGo(t *testing.T) {
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
		subsampleLuma8SIMD(got, input, stride, c.w, c.h, outW, outH, c.subX, c.subY)
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

func TestDirRowInterp8SIMDMatchesPureGo(t *testing.T) {
	for _, width := range []int{8, 16, 24, 32, 64, 4, 12, 20} {
		above := dirEdge(width+8, 0x77+uint32(width))
		for _, shift := range []int{0, 1, 5, 12, 16, 31} {
			for _, base := range []int{0, 1, 3} {
				maxBase := width + base // fully interpolated (asm path when width%8==0)
				got := make([]byte, width)
				want := make([]byte, width)
				dirRowInterp8SIMD(got, above, base, shift, maxBase, width)
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
				dirRowInterp8SIMD(got2, above, base, shift, clampMax, width)
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

func TestDirAboveRun8SIMDMatchesPureGo(t *testing.T) {
	for _, count := range []int{8, 16, 24, 32, 64, 1, 7, 9, 15, 33} {
		ref := dirEdge(count+8, 0x314+uint32(count))
		for _, shift := range []int{0, 1, 7, 16, 30, 31} {
			got := make([]byte, count)
			want := make([]byte, count)
			dirAboveRun8SIMD(got, ref, shift, count)
			dirAboveRun8PureGo(want, ref, shift, count)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("dirAboveRun count=%d shift=%d i=%d got=%d want=%d", count, shift, i, got[i], want[i])
				}
			}
		}
	}
}

func TestDirLeftCol8SIMDMatchesPureGo(t *testing.T) {
	for _, count := range []int{8, 16, 24, 32, 64, 1, 7, 9, 15} {
		ref := dirEdge(count+8, 0x9a1+uint32(count))
		for _, stride := range []int{1, 5, 33} {
			for _, shift := range []int{0, 3, 16, 29, 31} {
				got := make([]byte, count*stride+8)
				want := make([]byte, count*stride+8)
				dirLeftCol8SIMD(got, stride, ref, shift, count)
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

func TestKernelsSIMDZeroAlloc(t *testing.T) {
	// subsample
	input := make([]uint8, 40*32)
	for i := range input {
		input[i] = byte(i)
	}
	out := make([]uint16, CFLBufSquare)
	ss := func() { subsampleLuma8SIMD(out, input, 40, 32, 32, 16, 16, true, true) }
	if a := testing.AllocsPerRun(1000, ss); a != 0 {
		t.Fatalf("subsampleLuma8SIMD allocated %f/call", a)
	}
	// dirRowInterp / dirAboveRun
	above := dirEdge(72, 0x5)
	dst := make([]byte, 64)
	ri := func() { dirRowInterp8SIMD(dst, above, 0, 11, 64, 64) }
	if a := testing.AllocsPerRun(1000, ri); a != 0 {
		t.Fatalf("dirRowInterp8SIMD allocated %f/call", a)
	}
	ar := func() { dirAboveRun8SIMD(dst, above, 11, 64) }
	if a := testing.AllocsPerRun(1000, ar); a != 0 {
		t.Fatalf("dirAboveRun8SIMD allocated %f/call", a)
	}
	// dirLeftCol
	col := make([]byte, 64*5+8)
	lc := func() { dirLeftCol8SIMD(col, 5, above, 11, 64) }
	if a := testing.AllocsPerRun(1000, lc); a != 0 {
		t.Fatalf("dirLeftCol8SIMD allocated %f/call", a)
	}
}
