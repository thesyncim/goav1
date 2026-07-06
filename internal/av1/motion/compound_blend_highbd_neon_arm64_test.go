// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"math/rand"
	"testing"
)

// blendHighBDRoundParams returns the CONV_BUF blend rounding constants for a
// bit depth, mirroring BlendCompoundAvg.
func blendHighBDRoundParams(bitDepth uint8) (roundOffset int, roundBits int) {
	bd := int(bitDepth)
	round0 := compoundRound0(bitDepth)
	offsetBits := bd + 2*filterBits - round0
	roundOffset = (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	roundBits = 2*filterBits - round0 - compoundRound1Bits
	return roundOffset, roundBits
}

// TestBlendCompoundAvgHighBDNEONMatchesPureGo asserts the HBD NEON compound
// average / dist-wtd blend is bit-identical to the pure-Go reference for every
// AV1 block width (including the width-4 two-rows-per-iteration variant and
// the odd-height fallback), every dist-wtd weight pair, at bit depths 10 and
// 12, over full-range uint16 CONV_BUF inputs that exercise both clip bounds.
func TestBlendCompoundAvgHighBDNEONMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xb1e4d))
	widths := []int{4, 8, 16, 24, 32, 64, 128}
	heights := []int{1, 2, 3, 4, 5, 8, 16, 32, 64}
	weights := [][2]int{{8, 8}, {9, 7}, {7, 9}, {11, 5}, {5, 11}, {13, 3}, {3, 13}}
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		roundOffset, roundBits := blendHighBDRoundParams(bd)
		for _, w := range widths {
			for _, h := range heights {
				src0 := make([]uint16, w*h)
				src1 := make([]uint16, w*h)
				for i := range src0 {
					// Full uint16 range stresses both the >= 0 and <= max clamps.
					src0[i] = uint16(rng.Intn(1 << 16))
					src1[i] = uint16(rng.Intn(1 << 16))
				}
				for _, wt := range weights {
					for _, org := range [][2]int{{0, 0}, {2, 3}} {
						dstX, dstY := org[0], org[1]
						planeW := dstX + w
						planeH := dstY + h
						got, _ := testPlane(planeW, planeH, 2, planeW*2)
						want, _ := testPlane(planeW, planeH, 2, planeW*2)
						blendCompoundAvgHighBDNEON(got, src0, src1, max, dstX, dstY, w, h, wt[0], wt[1], roundOffset, roundBits)
						blendCompoundAvgHighBD(want, src0, src1, max, dstX, dstY, w, h, wt[0], wt[1], roundOffset, roundBits)
						for y := 0; y < h; y++ {
							for x := 0; x < w; x++ {
								g := getSample(got, 2, dstX+x, dstY+y)
								e := getSample(want, 2, dstX+x, dstY+y)
								if g != e {
									t.Fatalf("bd=%d %dx%d wt=%v org=%v (%d,%d): NEON=%d PureGo=%d",
										bd, w, h, wt, org, x, y, g, e)
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestBlendCompoundAvgHighBDNEONZeroAlloc asserts the HBD blend NEON wrapper
// allocates nothing on either asm path.
func TestBlendCompoundAvgHighBDNEONZeroAlloc(t *testing.T) {
	max, _ := highBDMax(10)
	roundOffset, roundBits := blendHighBDRoundParams(10)
	src0 := make([]uint16, 32*32)
	src1 := make([]uint16, 32*32)
	for i := range src0 {
		src0[i] = uint16(i * 37)
		src1[i] = uint16(i * 91)
	}
	dst, _ := testPlane(32, 32, 2, 64)
	cases := []struct {
		name string
		fn   func()
	}{
		{"W8", func() {
			blendCompoundAvgHighBDNEON(dst, src0[:32*8], src1[:32*8], max, 0, 0, 32, 8, 9, 7, roundOffset, roundBits)
		}},
		{"W4", func() {
			blendCompoundAvgHighBDNEON(dst, src0[:4*8], src1[:4*8], max, 0, 0, 4, 8, 8, 8, roundOffset, roundBits)
		}},
	}
	for _, c := range cases {
		if a := testing.AllocsPerRun(20, c.fn); a != 0 {
			t.Errorf("%s allocated %v times, want 0", c.name, a)
		}
	}
}

// TestConvolve2DHighBDNEONWithScratchMatchesPureGo asserts the scratch-carrying
// HBD 2D NEON convolve (resident and edge-clamped emu_edge shapes) stays
// bit-identical to the pure-Go references with a deliberately poisoned scratch,
// proving every intermediate sample read is written first.
func TestConvolve2DHighBDNEONWithScratchMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5c4a7c))
	scratch := &ConvolveScratch{}
	poison := func() {
		for i := range scratch.imHBD {
			scratch.imHBD[i] = int32(0x7bad0000) | int32(i&0xffff)
		}
		for i := range scratch.edge16 {
			scratch.edge16[i] = byte(0xa5)
		}
	}
	sizes := []int{4, 8, 16, 32, 64, 128}
	const pad = filterTaps
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		xk := subpelFilters8[6]
		yk := subpelFilters8[9]
		for _, w := range sizes {
			for _, h := range sizes {
				side := w
				if h > side {
					side = h
				}
				ref := makeHighBDRef(side, pad, max, true, rng)
				// Resident window.
				got, _ := testPlane(w, h, 2, w*2)
				want, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDNEONWithScratch(got, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk, scratch)
				convolve2DHighBDPureGo(want, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
				eqHighBDBlock(t, got, want, w, h, "2DHBDscratch", bd, w, h)
				// Overhanging window (emu_edge path through scratch.edge16).
				gotC, _ := testPlane(w, h, 2, w*2)
				wantC, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDClampedNEONWithScratch(gotC, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk, scratch)
				convolve2DHighBDClampedPureGo(wantC, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk)
				eqHighBDBlock(t, gotC, wantC, w, h, "2DHBDscratch-clamped", bd, w, h)
			}
		}
	}
}

// TestConvolve2DHighBDNEONWithScratchZeroAlloc asserts the scratch path (both
// resident and emu_edge clamped) allocates nothing.
func TestConvolve2DHighBDNEONWithScratchZeroAlloc(t *testing.T) {
	max, _ := highBDMax(10)
	rng := rand.New(rand.NewSource(11))
	const pad = filterTaps
	ref := makeHighBDRef(64, pad, max, true, rng)
	dst, _ := testPlane(64, 64, 2, 128)
	xk := subpelFilters8[6]
	yk := subpelFilters8[9]
	scratch := &ConvolveScratch{}
	cases := []struct {
		name string
		fn   func()
	}{
		{"resident", func() {
			convolve2DHighBDNEONWithScratch(dst, ref, 10, max, 0, 0, pad, pad, 64, 64, xk, yk, scratch)
		}},
		{"clamped", func() {
			convolve2DHighBDClampedNEONWithScratch(dst, ref, 10, max, 0, 0, -3, -3, 64, 64, xk, yk, scratch)
		}},
	}
	for _, c := range cases {
		if a := testing.AllocsPerRun(20, c.fn); a != 0 {
			t.Errorf("%s allocated %v times, want 0", c.name, a)
		}
	}
}
