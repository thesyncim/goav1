// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func refRowsTestVisit(miRow uint16, ref tile.ReferenceFrame, mvRow int16) tile.BlockLoopVisit {
	var v tile.BlockLoopVisit
	v.Block.MIRow = miRow
	v.Block.Size = tile.BlockSize16x16
	v.Prediction.Valid = true
	v.Prediction.InterMotion.References.Ref0, v.Prediction.InterMotion.References.Ref1 = ref, tile.ReferenceFrameNone
	v.Prediction.InterMotion.MV[0] = motion.Vector{Row: mvRow}
	return v
}

// TestFrameThreadRefNeed pins the per-reference row bound of each prediction
// kind: bottom edge + whole-row vertical motion + the filter / chroma margin.
func TestFrameThreadRefNeed(t *testing.T) {
	const bottom = (8 + 4) * 4 // MIRow 8, 16x16

	var n frameThreadRefNeed
	v := refRowsTestVisit(8, tile.ReferenceFrameLast, 8*10+3) // 10 rows + a fraction
	n.addVisit(&v, &testPayloads)
	if want := int32(bottom + 10 + refRowMargin); n[tile.ReferenceFrameLast] != want {
		t.Fatalf("translation: need %d, want %d", n[tile.ReferenceFrameLast], want)
	}

	// Upward motion never needs fewer than one row.
	n = frameThreadRefNeed{}
	v = refRowsTestVisit(0, tile.ReferenceFrameLast, -8*500)
	n.addVisit(&v, &testPayloads)
	if n[tile.ReferenceFrameLast] != 1 {
		t.Fatalf("upward: need %d, want 1", n[tile.ReferenceFrameLast])
	}

	// Compound: both references, each with its own vector.
	n = frameThreadRefNeed{}
	v = refRowsTestVisit(8, tile.ReferenceFrameLast, 8*4)
	v.Prediction.InterMotion.References.Ref1 = tile.ReferenceFrameAltref
	v.Prediction.InterMotion.References.Compound = true
	v.Prediction.InterMotion.MV[1] = motion.Vector{Row: 8 * 40}
	n.addVisit(&v, &testPayloads)
	if n[tile.ReferenceFrameLast] != bottom+4+refRowMargin || n[tile.ReferenceFrameAltref] != bottom+40+refRowMargin {
		t.Fatalf("compound: need %v", n)
	}

	// OBMC: a neighbour's vector applies over the block's own rows.
	n = frameThreadRefNeed{}
	v = refRowsTestVisit(8, tile.ReferenceFrameLast, 0)
	v.Prediction.MotionModeValid = true
	v.Prediction.MotionMode = tile.MotionModeOBMC
	v.Prediction.OverlappableNeighborsValid = true
	testMotionExtras(&v.Prediction).OverlappableNeighbors.AboveCount = 1
	testMotionExtras(&v.Prediction).OverlappableNeighbors.Above[0].MotionValid = true
	testMotionExtras(&v.Prediction).OverlappableNeighbors.Above[0].Motion.References = tile.InterReferencesResult{Ref0: tile.ReferenceFrameGolden, Ref1: tile.ReferenceFrameNone}
	testMotionExtras(&v.Prediction).OverlappableNeighbors.Above[0].Motion.MV[0] = motion.Vector{Row: 8 * 64}
	n.addVisit(&v, &testPayloads)
	if n[tile.ReferenceFrameGolden] != bottom+64+refRowMargin {
		t.Fatalf("obmc: golden need %d", n[tile.ReferenceFrameGolden])
	}

	// Sub-8x8 chroma cells use their own (neighbouring) vectors.
	n = frameThreadRefNeed{}
	v = refRowsTestVisit(8, tile.ReferenceFrameLast, 0)
	sub := testSubChroma(&v.Prediction)
	sub.Count = 1
	sub.Cells[0].Reference = tile.ReferenceFrameLast2
	sub.Cells[0].MV = motion.Vector{Row: 8 * 30}
	n.addVisit(&v, &testPayloads)
	if n[tile.ReferenceFrameLast2] != bottom+30+refRowMargin {
		t.Fatalf("sub-chroma: last2 need %d", n[tile.ReferenceFrameLast2])
	}

	// Warped and global-warp predictions wait for the whole reference.
	for _, mod := range []func(*tile.BlockLoopVisit){
		func(v *tile.BlockLoopVisit) {
			v.Prediction.MotionModeValid = true
			v.Prediction.MotionMode = tile.MotionModeWarp
		},
		func(v *tile.BlockLoopVisit) { v.Prediction.GlobalWarpedMotionValid = true },
	} {
		n = frameThreadRefNeed{}
		v = refRowsTestVisit(8, tile.ReferenceFrameLast, 0)
		mod(&v)
		n.addVisit(&v, &testPayloads)
		if n[tile.ReferenceFrameLast] != FrameProgressDone {
			t.Fatalf("warp: need %d, want the whole reference", n[tile.ReferenceFrameLast])
		}
	}

	// Intra and intra block copy read no reference.
	n = frameThreadRefNeed{}
	v = refRowsTestVisit(8, tile.ReferenceFrameLast, 0)
	v.Prediction.Intra = true
	n.addVisit(&v, &testPayloads)
	if n != (frameThreadRefNeed{}) {
		t.Fatalf("intra: need %v", n)
	}
}

// testPayloads backs the payload slots of test visits; the test prediction
// batches resolve through it. Tests in this package do not run in parallel.
var testPayloads = tile.BlockPayloads{
	Extras:    make([]tile.BlockPredictionMotionExtras, 1),
	SubChroma: make([]tile.SubChromaInterResult, 1),
	Palette:   make([]tile.PaletteModeScratch, 1),
}

// testMotionExtras returns p's out-of-line warp/OBMC payloads, naming a
// cleared test slot the first time.
func testMotionExtras(p *tile.BlockPredictionModeResult) *tile.BlockPredictionMotionExtras {
	if p.ExtrasSlot == 0 {
		testPayloads.Extras[0] = tile.BlockPredictionMotionExtras{}
		p.ExtrasSlot = 1
	}
	return &testPayloads.Extras[0]
}

// testSubChroma returns p's sub8x8 chroma cells, naming a cleared test slot
// the first time.
func testSubChroma(p *tile.BlockPredictionModeResult) *tile.SubChromaInterResult {
	if p.SubChromaSlot == 0 {
		testPayloads.SubChroma[0] = tile.SubChromaInterResult{}
		p.SubChromaSlot = 1
	}
	return &testPayloads.SubChroma[0]
}
