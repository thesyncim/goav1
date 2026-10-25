// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// Reference-row waits for frame-threaded reconstruction (dav1d's lowest_px,
// src/decode.c mc_lowest_px / obmc_lowest_px, checked in thread_task.c
// check_tile): before a superblock is reconstructed, every reference it
// predicts from must have published the luma rows its motion-compensated
// reads reach, rather than the whole frame.
//
// A block's need for a reference is its bottom edge plus the vertical motion
// (in whole luma rows, rounded down) plus refRowMargin: four rows for the
// eight-tap filter below the integer position and four more for the 4:2:0
// chroma rounding (chroma rows are needed at twice their index in luma units).
// OBMC predictions from the above and left neighbours' motion and the sub-8x8
// chroma cells (which use neighbouring blocks' motion) only cover rows of the
// block itself, so the same bound with the neighbour's vector holds. Warped
// and global-warp predictions, and references at a different size (scaled
// prediction), wait for the whole reference.
const refRowMargin = 8

// frameThreadRefNeed is the per-reference luma row count a superblock needs;
// zero means the reference is not read.
type frameThreadRefNeed [parser.InterRefsPerFrame]int32

func (n *frameThreadRefNeed) add(ref tile.ReferenceFrame, bottom int32, mv motion.Vector, whole bool) {
	if !ref.Valid() {
		return
	}
	v := FrameProgressDone
	if !whole {
		v = max(1, bottom+int32(mv.Row>>3)+refRowMargin)
	}
	if v > n[ref] {
		n[ref] = v
	}
}

// addVisit adds the reference rows one buffered block's inter prediction reads.
func (n *frameThreadRefNeed) addVisit(v *tile.BlockLoopVisit) {
	p := &v.Prediction
	if !p.Valid || p.Intra || p.Intrabc {
		return
	}
	bottom := (int32(v.Block.MIRow) + int32(tile.BlockSizeHeight4(v.Block.Size))) * 4
	refs := &p.InterMotion.References
	warp := p.MotionModeValid && p.MotionMode == tile.MotionModeWarp && !p.WarpedMotionInvalid
	if refs.Ref[1] == tile.ReferenceFrameNone || !refs.Compound {
		n.add(refs.Ref[0], bottom, p.InterMotion.MV[0], warp || p.GlobalWarpedMotionValid)
	} else {
		for i := range 2 {
			n.add(refs.Ref[i], bottom, p.InterMotion.MV[i], warp || p.GlobalWarpedMotionValid || p.GlobalWarpedMotionCompoundValid[i])
		}
	}
	if p.MotionModeValid && p.MotionMode == tile.MotionModeOBMC && p.OverlappableNeighborsValid {
		s := &p.OverlappableNeighbors
		for i := range min(int(s.AboveCount), len(s.Above)) {
			n.addNeighbor(&s.Above[i], bottom)
		}
		for i := range min(int(s.LeftCount), len(s.Left)) {
			n.addNeighbor(&s.Left[i], bottom)
		}
	}
	if p.SubChromaInterValid {
		c := &p.SubChromaInter
		for i := range min(int(c.Count), len(c.Cells)) {
			n.add(c.Cells[i].Reference, bottom, c.Cells[i].MV, false)
		}
	}
}

func (n *frameThreadRefNeed) addNeighbor(o *tile.OverlappableNeighbor, bottom int32) {
	if !o.MotionValid {
		return
	}
	n.add(o.Motion.References.Ref[0], bottom, o.Motion.MV[0], false)
}

// waitSuperblockReferences blocks until every reference the superblock's
// buffered blocks predict from has published the rows they read. It reports
// false when a reference failed to decode.
func (r *FrameThreadRecon) waitSuperblockReferences(b *FrameWorkBatch, events []frameWorkReconEvent, visits []tile.BlockLoopVisit) bool {
	var need frameThreadRefNeed
	any := false
	for i := range events {
		ev := &events[i]
		if ev.kind != frameWorkReconEventBlockBegin || int(ev.index) >= len(visits) {
			continue
		}
		need.addVisit(&visits[ev.index])
		any = true
	}
	if !any {
		return true
	}
	for i, rows := range need {
		p := r.RefProgress[i]
		if rows == 0 || p == nil {
			continue
		}
		if rows != FrameProgressDone && b != nil {
			if i >= len(b.References) || b.References[i] == nil || b.frameThreadScaledReference(i) ||
				rows >= int32(b.References[i].Y.Height) {
				rows = FrameProgressDone
			}
		}
		if !p.Wait(rows) {
			return false
		}
	}
	return true
}

// frameThreadScaledReference reports whether reference i differs in size
// from the frame being decoded (scaled prediction reads the whole reference).
func (b *FrameWorkBatch) frameThreadScaledReference(i int) bool {
	ref := b.References[i]
	return ref.Format.Width != int(b.FrameSize.CodedWidth) || ref.Format.Height != int(b.FrameSize.Height)
}
