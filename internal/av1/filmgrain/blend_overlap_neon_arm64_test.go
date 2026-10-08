// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package filmgrain

import (
	"math/rand"
	"testing"
	"unsafe"
)

func TestBlendGrainRowNEONCtxSize(t *testing.T) {
	if size := unsafe.Sizeof(blendGrainRowNEONCtx{}); size != 48 {
		t.Fatalf("blendGrainRowNEONCtx size=%d want 48", size)
	}
}

// blendOverlapWeightPairs enumerates every (prevWeight, curWeight) pair the
// apply path feeds blendGrainRow: luma / vertically non-subsampled chroma use
// 27/17 and 17/27 (blendLumaOverlap); vertically subsampled chroma uses 23/22
// (blendChromaOverlap).
func blendOverlapWeightPairs() [][2]int {
	return [][2]int{{27, 17}, {17, 27}, {23, 22}}
}

func fuzzGrainRun(rng *rand.Rand, n int, bd uint8) []int16 {
	grainMin := -(1 << (bd - 1))
	grainMax := (1 << (bd - 1)) - 1
	span := grainMax - grainMin + 1
	out := make([]int16, n)
	for i := range out {
		out[i] = int16(grainMin + rng.Intn(span))
	}
	return out
}

func TestBlendGrainRowNEONMatchesPureGo(t *testing.T) {
	if !blendGrainRowUseNEON {
		t.Skip("NEON overlap-blend kernel not active on this CPU")
	}
	rng := rand.New(rand.NewSource(0xB1E4D))
	bitDepths := []uint8{8, 10, 12}
	// Cover full groups, partial tails, and sub-group lengths.
	lengths := []int{1, 2, 3, 7, 8, 9, 14, 15, 16, 17, 30, 31, 32}
	for _, bd := range bitDepths {
		grainMin := -(1 << (bd - 1))
		grainMax := (1 << (bd - 1)) - 1
		for _, wp := range blendOverlapWeightPairs() {
			for _, n := range lengths {
				prev := fuzzGrainRun(rng, n, bd)
				cur := fuzzGrainRun(rng, n, bd)
				ref := make([]int16, n)
				got := make([]int16, n)
				blendGrainRowPureGo(ref, prev, cur, wp[0], wp[1], grainMin, grainMax)
				blendGrainRowNEON(got, prev, cur, wp[0], wp[1], grainMin, grainMax)
				for i := 0; i < n; i++ {
					if got[i] != ref[i] {
						t.Fatalf("bd=%d w=%v n=%d i=%d: NEON=%d pureGo=%d (prev=%d cur=%d)",
							bd, wp, n, i, got[i], ref[i], prev[i], cur[i])
					}
				}
			}
		}
	}
}

// TestBlendGrainRowNEONExtremes drives the saturating corners: all-min and
// all-max grain runs at every weight pair so the SMAX/SMIN clamp and the XTN
// narrow are exercised against the pure-Go clipInt at the range boundaries.
func TestBlendGrainRowNEONExtremes(t *testing.T) {
	if !blendGrainRowUseNEON {
		t.Skip("NEON overlap-blend kernel not active on this CPU")
	}
	bitDepths := []uint8{8, 10, 12}
	const n = 32
	for _, bd := range bitDepths {
		grainMin := int16(-(1 << (bd - 1)))
		grainMax := int16((1 << (bd - 1)) - 1)
		for _, wp := range blendOverlapWeightPairs() {
			for _, fill := range []int16{grainMin, grainMax, 0} {
				prev := make([]int16, n)
				cur := make([]int16, n)
				for i := range prev {
					prev[i] = fill
					cur[i] = fill
				}
				ref := make([]int16, n)
				got := make([]int16, n)
				blendGrainRowPureGo(ref, prev, cur, wp[0], wp[1], int(grainMin), int(grainMax))
				blendGrainRowNEON(got, prev, cur, wp[0], wp[1], int(grainMin), int(grainMax))
				for i := 0; i < n; i++ {
					if got[i] != ref[i] {
						t.Fatalf("bd=%d w=%v fill=%d i=%d: NEON=%d pureGo=%d", bd, wp, fill, i, got[i], ref[i])
					}
				}
			}
		}
	}
}

func TestBlendGrainRowNEONZeroAlloc(t *testing.T) {
	if !blendGrainRowUseNEON {
		t.Skip("NEON overlap-blend kernel not active on this CPU")
	}
	const n = 32
	rng := rand.New(rand.NewSource(1))
	prev := fuzzGrainRun(rng, n, 10)
	cur := fuzzGrainRun(rng, n, 10)
	dst := make([]int16, n)
	allocs := testing.AllocsPerRun(1000, func() {
		blendGrainRowNEON(dst, prev, cur, 27, 17, -512, 511)
	})
	if allocs != 0 {
		t.Fatalf("blendGrainRowNEON allocated: %f", allocs)
	}
}

func BenchmarkBlendGrainRowNEON(b *testing.B) {
	if !blendGrainRowUseNEON {
		b.Skip("NEON overlap-blend kernel not active on this CPU")
	}
	const n = 30 // representative top-overlap run width (blockWidth-xStart)
	rng := rand.New(rand.NewSource(7))
	prev := fuzzGrainRun(rng, n, 10)
	cur := fuzzGrainRun(rng, n, 10)
	dst := make([]int16, n)
	b.ReportAllocs()
	for b.Loop() {
		blendGrainRowNEON(dst, prev, cur, 27, 17, -512, 511)
	}
}

func BenchmarkBlendGrainRowPureGo(b *testing.B) {
	const n = 30
	rng := rand.New(rand.NewSource(7))
	prev := fuzzGrainRun(rng, n, 10)
	cur := fuzzGrainRun(rng, n, 10)
	dst := make([]int16, n)
	b.ReportAllocs()
	for b.Loop() {
		blendGrainRowPureGo(dst, prev, cur, 27, 17, -512, 511)
	}
}
