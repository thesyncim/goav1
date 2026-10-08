// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import (
	"testing"
)

// staticSIMDShapes covers every AV1 intra block shape the SMOOTH weight tables
// define (4, 8, 16, 32, 64 on both axes), so every kernel shape is exercised.
var staticSIMDShapes = [...]directionalTestSize{
	{4, 4}, {4, 8}, {4, 16}, {4, 32}, {4, 64},
	{8, 4}, {8, 8}, {8, 16}, {8, 32}, {8, 64},
	{16, 4}, {16, 8}, {16, 16}, {16, 32}, {16, 64},
	{32, 4}, {32, 8}, {32, 16}, {32, 32}, {32, 64},
	{64, 4}, {64, 8}, {64, 16}, {64, 32}, {64, 64},
}

// staticSIMDPaethExtraShapes adds the odd and sub-4 heights that PAETH has no
// weight table for. They exercise the width-4 kernel's odd-height fallback.
var staticSIMDPaethExtraShapes = [...]directionalTestSize{
	{4, 1}, {4, 2}, {4, 3}, {4, 5}, {8, 1}, {8, 2}, {8, 3}, {16, 2}, {32, 2},
}

// staticSIMDDepths pairs each sample storage size with the bit depths it serves:
// 8-bit samples (one byte each) and 10/12-bit samples (two bytes each).
var staticSIMDDepths = [...]struct {
	bytesPerSample int
	max            uint16
}{
	{1, 0xff},
	{2, 1023},
	{2, 4095},
}

// staticSIMDEdges fills above/left/aboveLeft with pseudo-random samples in
// [0,max] from an LCG seeded by seed, and sets the weight tables for the block.
func staticSIMDEdges(width, height int, max uint16, seed uint32) (above, left []uint16, aboveLeft uint16) {
	next := func() uint32 {
		seed = seed*1664525 + 1013904223
		return seed >> 8
	}
	above = make([]uint16, width+1)
	left = make([]uint16, height+1)
	for i := range above {
		above[i] = uint16(next() % (uint32(max) + 1))
	}
	for i := range left {
		left[i] = uint16(next() % (uint32(max) + 1))
	}
	aboveLeft = uint16(next() % (uint32(max) + 1))
	return above, left, aboveLeft
}

// staticSIMDPattern builds edges from the clamp-corner patterns (all zero, all
// max, alternating, ramp) so the PAETH tie-break and the rounding extremes run.
func staticSIMDPattern(width, height int, max uint16, kind int) (above, left []uint16, aboveLeft uint16) {
	pick := func(i int) uint16 {
		switch kind {
		case 0:
			return 0
		case 1:
			return max
		case 2:
			if i%2 == 0 {
				return 0
			}
			return max
		default:
			return uint16((i * 37) % (int(max) + 1))
		}
	}
	above = make([]uint16, width+1)
	left = make([]uint16, height+1)
	for i := range above {
		above[i] = pick(i)
	}
	for i := range left {
		left[i] = pick(i + 1)
	}
	return above, left, pick(3)
}

// staticSIMDModes names the four static predictors through their routers and
// their PureGo references; each entry runs one block with the given edges.
var staticSIMDModes = [...]struct {
	name string
	simd func(block planeBlock, bps int, above, left []uint16, aboveLeft uint16)
	ref  func(block planeBlock, bps int, above, left []uint16, aboveLeft uint16)
}{
	{
		"paeth",
		func(b planeBlock, bps int, above, left []uint16, al uint16) {
			predictPaethSIMD(b, bps, above, left, al)
		},
		func(b planeBlock, bps int, above, left []uint16, al uint16) {
			predictPaethPureGo(b, bps, above, left, al)
		},
	},
	{
		"smooth",
		func(b planeBlock, bps int, above, left []uint16, al uint16) {
			wW, _ := smoothWeightsForSize(b.width)
			wH, _ := smoothWeightsForSize(b.height)
			predictSmoothSIMD(b, bps, wW, wH, above, left, left[b.height-1], above[b.width-1])
		},
		func(b planeBlock, bps int, above, left []uint16, al uint16) {
			wW, _ := smoothWeightsForSize(b.width)
			wH, _ := smoothWeightsForSize(b.height)
			predictSmoothPureGo(b, bps, wW, wH, above, left, left[b.height-1], above[b.width-1])
		},
	},
	{
		"smooth_v",
		func(b planeBlock, bps int, above, left []uint16, al uint16) {
			wH, _ := smoothWeightsForSize(b.height)
			predictSmoothVerticalSIMD(b, bps, wH, above, left[b.height-1])
		},
		func(b planeBlock, bps int, above, left []uint16, al uint16) {
			wH, _ := smoothWeightsForSize(b.height)
			predictSmoothVerticalPureGo(b, bps, wH, above, left[b.height-1])
		},
	},
	{
		"smooth_h",
		func(b planeBlock, bps int, above, left []uint16, al uint16) {
			wW, _ := smoothWeightsForSize(b.width)
			predictSmoothHorizontalSIMD(b, bps, wW, left, above[b.width-1])
		},
		func(b planeBlock, bps int, above, left []uint16, al uint16) {
			wW, _ := smoothWeightsForSize(b.width)
			predictSmoothHorizontalPureGo(b, bps, wW, left, above[b.width-1])
		},
	},
}

// TestStaticSIMDMatchesPureGo runs every static predictor through its router
// against the PureGo reference for every routed shape, both sample sizes and
// several seeded edge sets, comparing every output byte.
func TestStaticSIMDMatchesPureGo(t *testing.T) {
	for _, mode := range staticSIMDModes {
		for _, d := range staticSIMDDepths {
			for _, sz := range staticSIMDShapesFor(mode.name) {
				for _, seed := range []uint32{1, 0xabcd, 0x5f3759df} {
					above, left, al := staticSIMDEdges(sz.width, sz.height, d.max, seed)
					base := makeDispatchBlock(sz.width, sz.height, d.bytesPerSample)
					got := cloneBlock(base)
					want := cloneBlock(base)
					mode.simd(got, d.bytesPerSample, above, left, al)
					mode.ref(want, d.bytesPerSample, above, left, al)
					diffBlocks(t, mode.name, int(d.max), sz.width*d.bytesPerSample, sz.height, got, want)
				}
			}
		}
	}
}

// TestStaticSIMDEdgePatterns stresses the clamp corners and the PAETH tie-break
// with constant and alternating edges at every shape and sample size.
func TestStaticSIMDEdgePatterns(t *testing.T) {
	for _, mode := range staticSIMDModes {
		for _, d := range staticSIMDDepths {
			for _, sz := range staticSIMDShapesFor(mode.name) {
				for kind := 0; kind < 4; kind++ {
					above, left, al := staticSIMDPattern(sz.width, sz.height, d.max, kind)
					base := makeDispatchBlock(sz.width, sz.height, d.bytesPerSample)
					got := cloneBlock(base)
					want := cloneBlock(base)
					mode.simd(got, d.bytesPerSample, above, left, al)
					mode.ref(want, d.bytesPerSample, above, left, al)
					diffBlocks(t, mode.name+"-edge", int(d.max), sz.width*d.bytesPerSample, sz.height, got, want)
				}
			}
		}
	}
}

// TestStaticSIMDBinding asserts the arm64 SIMD build binds the Go-native static
// predictors to the dispatch slots, so the tests above exercise the production
// targets.
func TestStaticSIMDBinding(t *testing.T) {
	assertDispatchTarget(t, "predictPaethImpl", predictPaethImpl, "predictPaethSIMD")
	assertDispatchTarget(t, "predictSmoothImpl", predictSmoothImpl, "predictSmoothSIMD")
	assertDispatchTarget(t, "predictSmoothVerticalImpl", predictSmoothVerticalImpl, "predictSmoothVerticalSIMD")
	assertDispatchTarget(t, "predictSmoothHorizontalImpl", predictSmoothHorizontalImpl, "predictSmoothHorizontalSIMD")
}

// TestStaticSIMDZeroAlloc protects the hot-path contract: the static SIMD
// routers must not allocate per call.
func TestStaticSIMDZeroAlloc(t *testing.T) {
	const w, h = 32, 32
	above, left, al := staticSIMDEdges(w, h, 0xff, 0x24)
	block := makeDispatchBlock(w, h, 1)
	wW, _ := smoothWeightsForSize(w)
	wH, _ := smoothWeightsForSize(h)
	cases := map[string]func(){
		"paeth":    func() { predictPaethSIMD(block, 1, above, left, al) },
		"smooth":   func() { predictSmoothSIMD(block, 1, wW, wH, above, left, left[h-1], above[w-1]) },
		"smooth_v": func() { predictSmoothVerticalSIMD(block, 1, wH, above, left[h-1]) },
		"smooth_h": func() { predictSmoothHorizontalSIMD(block, 1, wW, left, above[w-1]) },
	}
	for name, fn := range cases {
		if allocs := testing.AllocsPerRun(1000, fn); allocs != 0 {
			t.Fatalf("%s SIMD allocated %f times per call", name, allocs)
		}
	}
}

// staticSIMDShapesFor returns the shapes a predictor is tested over; PAETH also
// covers the shapes without SMOOTH weights.
func staticSIMDShapesFor(name string) []directionalTestSize {
	shapes := append([]directionalTestSize(nil), staticSIMDShapes[:]...)
	if name == "paeth" {
		shapes = append(shapes, staticSIMDPaethExtraShapes[:]...)
	}
	return shapes
}
