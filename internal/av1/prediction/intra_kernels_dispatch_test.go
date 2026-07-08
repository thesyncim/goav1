// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package prediction

import "testing"

// These tests guard the dispatched DC-sum, CfL and directional-interpolation
// kernels: the resolved dispatch slots (NEON on arm64, AVX2 on amd64 when
// active) must match the canonical pure-Go references sample for sample. They
// run on every build; on architectures with no tuned variant the slot is the
// reference itself and the comparison is trivially satisfied.

func TestSumSamplesDispatchMatchesPureGo(t *testing.T) {
	for _, n := range []int{4, 8, 12, 16, 20, 32, 64, 100} {
		for _, max := range []uint16{0xff, 0x3ff, 0xfff} {
			s, _ := samplesFor(n, max, 0x9e3d+uint32(n)+uint32(max))
			got := sumSamplesImpl(s)
			want := sumSamplesPureGo(s)
			if got != want {
				t.Fatalf("sumSamples n=%d max=%d got=%d want=%d", n, max, got, want)
			}
		}
	}
}

func TestSubsampleLuma8DispatchMatchesPureGo(t *testing.T) {
	type cfg struct {
		w, h       int
		subX, subY bool
	}
	cfgs := []cfg{
		{8, 8, false, false}, {16, 16, false, false}, {32, 32, false, false},
		{16, 16, true, false}, {32, 16, true, false}, {16, 8, true, false},
		{16, 16, true, true}, {32, 32, true, true}, {16, 32, true, true},
		{8, 8, true, true}, {4, 4, true, true}, // width 4 -> pure-Go fallback
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
		subsampleLuma8Impl(got, input, stride, c.w, c.h, outW, outH, c.subX, c.subY)
		subsampleLuma8PureGo(want, input, stride, c.w, c.h, outW, outH, c.subX, c.subY)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("subsample %+v idx=%d got=%d want=%d", c, i, got[i], want[i])
			}
		}
	}
}

func TestSubtractCFLAverageDispatchMatchesPureGo(t *testing.T) {
	var seed uint32 = 0x871
	for _, dim := range [][2]int{{4, 4}, {8, 8}, {16, 16}, {32, 32}, {8, 32}, {32, 8}} {
		w, h := dim[0], dim[1]
		src := make([]uint16, CFLBufSquare)
		for row := 0; row < h; row++ {
			for col := 0; col < w; col++ {
				seed = seed*1664525 + 1013904223
				src[row*CFLBufLine+col] = uint16(seed >> 17)
			}
		}
		log2, _ := log2PowerOfTwoInt(w * h)
		got := make([]int16, CFLBufSquare)
		want := make([]int16, CFLBufSquare)
		subtractCFLAverageImpl(src, got, w, h, log2)
		subtractCFLAveragePureGo(src, want, w, h, log2)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("subtract-average %dx%d idx=%d got=%d want=%d", w, h, i, got[i], want[i])
			}
		}
	}
}

func TestApplyCFLDispatchMatchesPureGo(t *testing.T) {
	dims := [][2]int{{8, 8}, {16, 16}, {32, 32}, {8, 16}, {16, 8}, {4, 4}, {32, 8}}
	depths := []struct {
		bps int
		max uint16
	}{{1, 0xff}, {2, 0x3ff}, {2, 0xfff}}
	var seed uint32 = 0xabc
	for _, d := range depths {
		for _, dim := range dims {
			w, h := dim[0], dim[1]
			ac := make([]int16, CFLBufSquare)
			for i := range ac {
				seed = seed*1664525 + 1013904223
				ac[i] = int16(int32(seed>>8)%4096 - 2048)
			}
			for _, alpha := range []int{-16, -7, -1, 0, 1, 5, 16} {
				base := makeDispatchBlock(w, h, d.bps)
				// seed the chroma DC with mid-range values
				for i := range base.pix {
					base.pix[i] = byte((i*7 + 31) & 0xff)
				}
				got := cloneBlock(base)
				want := cloneBlock(base)
				applyCFLImpl(got, d.bps, w, h, ac, alpha, d.max)
				applyCFLPureGo(want, d.bps, w, h, ac, alpha, d.max)
				diffBlocks(t, "applycfl", int(d.max), w*d.bps, h, got, want)
			}
		}
	}
}

func TestDirRowInterp8DispatchMatchesPureGo(t *testing.T) {
	var seed uint32 = 0x55aa
	for _, width := range []int{4, 8, 16, 32, 64} {
		for _, maxBase := range []int{width + 1, width + 8, 200, 8} {
			above := make([]uint16, 300)
			for i := range above {
				seed = seed*1664525 + 1013904223
				above[i] = uint16(seed >> 24)
			}
			for _, base := range []int{0, 1, 3, 7, maxBase - 1, maxBase, maxBase + 4} {
				if base < 0 {
					continue
				}
				for _, shift := range []int{0, 1, 5, 16, 31} {
					got := make([]byte, width)
					want := make([]byte, width)
					dirRowInterp8Impl(got, above, base, shift, maxBase, width)
					dirRowInterp8PureGo(want, above, base, shift, maxBase, width)
					for i := range got {
						if got[i] != want[i] {
							t.Fatalf("dirrow w=%d base=%d shift=%d maxBase=%d col=%d got=%d want=%d",
								width, base, shift, maxBase, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}
