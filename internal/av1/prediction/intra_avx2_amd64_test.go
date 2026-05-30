// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package prediction

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// The AVX2 kernels are differentially validated against the pure-Go reference
// directly (not through the dispatch slot), because the cpu package stub does
// not yet report AVX2 so the runtime dispatch keeps the reference. This makes
// the AVX2 code byte-exact-tested on every amd64 build (native or Rosetta)
// regardless of the dispatch wiring.

func TestAVX2DispatchProbe(t *testing.T) {
	t.Logf("cpu.Detected.AVX2=%v SSE2=%v", cpu.Detected.AVX2, cpu.Detected.SSE2)
}

func TestPaethAVX2MatchesPureGo(t *testing.T) {
	for _, w := range []int{16, 32, 48, 64} {
		for _, h := range []int{4, 8, 16, 32, 64} {
			above, s1 := samplesFor(w, 0xff, 0x111+uint32(w))
			left, _ := samplesFor(h, 0xff, s1)
			for _, al := range []uint16{0, 1, 128, 255} {
				base := makeDispatchBlock(w, h, 1)
				got := cloneBlock(base)
				want := cloneBlock(base)
				predictPaethAVX2(got, 1, above, left, al)
				predictPaethPureGo(want, 1, above, left, al)
				diffBlocks(t, "paeth-avx2", 0xff, w, h, got, want)
			}
		}
	}
}

func TestPaethAVX2EdgeValues(t *testing.T) {
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
				predictPaethAVX2(got, 1, above, left, al)
				predictPaethPureGo(want, 1, above, left, al)
				diffBlocks(t, "paeth-avx2-edge", 0xff, w, h, got, want)
			}
		}
	}
}

func TestSmoothAVX2MatchesPureGo(t *testing.T) {
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
			predictSmoothAVX2(got, 1, weightsW, weightsH, above, left, belowPred, rightPred)
			predictSmoothPureGo(want, 1, weightsW, weightsH, above, left, belowPred, rightPred)
			diffBlocks(t, "smooth-avx2", 0xff, w, h, got, want)

			got = cloneBlock(base)
			want = cloneBlock(base)
			predictSmoothVerticalAVX2(got, 1, weightsH, above, belowPred)
			predictSmoothVerticalPureGo(want, 1, weightsH, above, belowPred)
			diffBlocks(t, "smooth_v-avx2", 0xff, w, h, got, want)

			got = cloneBlock(base)
			want = cloneBlock(base)
			predictSmoothHorizontalAVX2(got, 1, weightsW, left, rightPred)
			predictSmoothHorizontalPureGo(want, 1, weightsW, left, rightPred)
			diffBlocks(t, "smooth_h-avx2", 0xff, w, h, got, want)
		}
	}
}

func TestApplyCFLAVX2MatchesPureGo(t *testing.T) {
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
			applyCFLAVX2(got, 1, w, h, ac, alpha, 0xff)
			applyCFLPureGo(want, 1, w, h, ac, alpha, 0xff)
			diffBlocks(t, "applycfl-avx2", 0xff, w, h, got, want)
		}
	}
}

func TestSumSamplesAVX2MatchesPureGo(t *testing.T) {
	for _, n := range []int{16, 32, 48, 64, 20, 4, 100} {
		for _, max := range []uint16{0xff, 0x3ff, 0xfff} {
			s, _ := samplesFor(n, max, 0x5a5+uint32(n)+uint32(max))
			if got, want := sumSamplesAVX2(s), sumSamplesPureGo(s); got != want {
				t.Fatalf("sumSamplesAVX2 n=%d max=%d got=%d want=%d", n, max, got, want)
			}
		}
	}
}
