// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package prediction

import (
	"testing"
)

// blockSizes are the AV1 intra block dimensions the predictors run on.
var dispatchBlockDims = []int{4, 8, 16, 32, 64}

func makeDispatchBlock(width, height, bytesPerSample int) planeBlock {
	rowBytes := width * bytesPerSample
	stride := rowBytes + 7*bytesPerSample // non-tight stride to catch stride bugs
	return planeBlock{
		pix:      make([]byte, stride*height),
		stride:   stride,
		width:    width,
		height:   height,
		rowBytes: rowBytes,
	}
}

func cloneBlock(b planeBlock) planeBlock {
	c := b
	c.pix = make([]byte, len(b.pix))
	copy(c.pix, b.pix)
	return c
}

func samplesFor(n int, max uint16, seed uint32) ([]uint16, uint32) {
	s := make([]uint16, n)
	for i := range s {
		seed = seed*1664525 + 1013904223
		s[i] = uint16((seed >> 8) % (uint32(max) + 1))
	}
	return s, seed
}

func diffBlocks(t *testing.T, name string, depth, w, h int, got, want planeBlock) {
	t.Helper()
	for row := 0; row < h; row++ {
		for col := 0; col < w*1; col++ {
			gi := row*got.stride + col
			wi := row*want.stride + col
			if got.pix[gi] != want.pix[wi] {
				t.Fatalf("%s depth=%d %dx%d row=%d col=%d got=%d want=%d",
					name, depth, w, h, row, col, got.pix[gi], want.pix[wi])
			}
		}
	}
}

// TestIntraStaticDispatchMatchesPureGo is the bit-exactness guard for the
// dispatched intra static predictors. It exercises the resolved dispatch slots
// (which route through NEON asm on arm64) against the canonical pure-Go
// reference across PAETH and the three SMOOTH variants, every AV1 block size,
// and 8/10/12-bit depths. Every output sample must match the reference exactly.
func TestIntraStaticDispatchMatchesPureGo(t *testing.T) {
	for _, depth := range []struct {
		bps int
		max uint16
	}{{1, 0xff}, {2, 0x3ff}, {2, 0xfff}} {
		var seed uint32 = 0x12345 + uint32(depth.max)
		for _, w := range dispatchBlockDims {
			for _, h := range dispatchBlockDims {
				above, s1 := samplesFor(w, depth.max, seed)
				left, s2 := samplesFor(h, depth.max, s1)
				seed = s2
				aboveLeft := above[0] ^ left[0]
				if aboveLeft > depth.max {
					aboveLeft = depth.max
				}
				weightsW, _ := smoothWeightsForSize(w)
				weightsH, _ := smoothWeightsForSize(h)
				belowPred := left[h-1]
				rightPred := above[w-1]

				base := makeDispatchBlock(w, h, depth.bps)

				// PAETH
				{
					got := cloneBlock(base)
					want := cloneBlock(base)
					predictPaethImpl(got, depth.bps, above, left, aboveLeft)
					predictPaethPureGo(want, depth.bps, above, left, aboveLeft)
					diffBlocks(t, "paeth", int(depth.max), w*depth.bps, h, got, want)
				}
				// SMOOTH
				{
					got := cloneBlock(base)
					want := cloneBlock(base)
					predictSmoothImpl(got, depth.bps, weightsW, weightsH, above, left, belowPred, rightPred)
					predictSmoothPureGo(want, depth.bps, weightsW, weightsH, above, left, belowPred, rightPred)
					diffBlocks(t, "smooth", int(depth.max), w*depth.bps, h, got, want)
				}
				// SMOOTH_V
				{
					got := cloneBlock(base)
					want := cloneBlock(base)
					predictSmoothVerticalImpl(got, depth.bps, weightsH, above, belowPred)
					predictSmoothVerticalPureGo(want, depth.bps, weightsH, above, belowPred)
					diffBlocks(t, "smooth_v", int(depth.max), w*depth.bps, h, got, want)
				}
				// SMOOTH_H
				{
					got := cloneBlock(base)
					want := cloneBlock(base)
					predictSmoothHorizontalImpl(got, depth.bps, weightsW, left, rightPred)
					predictSmoothHorizontalPureGo(want, depth.bps, weightsW, left, rightPred)
					diffBlocks(t, "smooth_h", int(depth.max), w*depth.bps, h, got, want)
				}
			}
		}
	}
}

// TestIntraStaticDispatchEdgeValues stresses the PAETH tie-break paths and the
// SMOOTH rounding boundary with degenerate edge samples (all zero, all max,
// alternating) which exercise the saturating / select corners of the asm.
func TestIntraStaticDispatchEdgeValues(t *testing.T) {
	const w, h = 16, 16
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
	for ai, ap := range patterns {
		for li, lp := range patterns {
			above := make([]uint16, w)
			left := make([]uint16, h)
			for i := range above {
				above[i] = ap(i)
			}
			for i := range left {
				left[i] = lp(i)
			}
			for _, al := range []uint16{0, 128, 255} {
				weightsW, _ := smoothWeightsForSize(w)
				weightsH, _ := smoothWeightsForSize(h)
				base := makeDispatchBlock(w, h, 1)

				got := cloneBlock(base)
				want := cloneBlock(base)
				predictPaethImpl(got, 1, above, left, al)
				predictPaethPureGo(want, 1, above, left, al)
				diffBlocks(t, "paeth-edge", 0xff, w, h, got, want)

				got = cloneBlock(base)
				want = cloneBlock(base)
				predictSmoothImpl(got, 1, weightsW, weightsH, above, left, left[h-1], above[w-1])
				predictSmoothPureGo(want, 1, weightsW, weightsH, above, left, left[h-1], above[w-1])
				diffBlocks(t, "smooth-edge", 0xff, w, h, got, want)
				_ = ai
				_ = li
			}
		}
	}
}

// TestIntraStaticDispatchZeroAlloc protects the hot-path contract that the
// dispatched predictors do not allocate per call.
func TestIntraStaticDispatchZeroAlloc(t *testing.T) {
	const w, h = 32, 32
	above, _ := samplesFor(w, 0xff, 7)
	left, _ := samplesFor(h, 0xff, 99)
	weightsW, _ := smoothWeightsForSize(w)
	weightsH, _ := smoothWeightsForSize(h)
	block := makeDispatchBlock(w, h, 1)

	checks := []struct {
		name string
		fn   func()
	}{
		{"paeth", func() { predictPaethImpl(block, 1, above, left, 123) }},
		{"smooth", func() { predictSmoothImpl(block, 1, weightsW, weightsH, above, left, left[h-1], above[w-1]) }},
		{"smooth_v", func() { predictSmoothVerticalImpl(block, 1, weightsH, above, left[h-1]) }},
		{"smooth_h", func() { predictSmoothHorizontalImpl(block, 1, weightsW, left, above[w-1]) }},
	}
	for _, c := range checks {
		if allocs := testing.AllocsPerRun(1000, c.fn); allocs != 0 {
			t.Fatalf("%s allocated %f times per call", c.name, allocs)
		}
	}
}
