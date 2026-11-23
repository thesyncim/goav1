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
	if refs.Ref1 == tile.ReferenceFrameNone || !refs.Compound {
		n.add(refs.Ref0, bottom, p.InterMotion.MV[0], warp || p.GlobalWarpedMotionValid)
	} else {
		for i := range 2 {
			n.add(refs.At(i), bottom, p.InterMotion.MV[i], warp || p.GlobalWarpedMotionValid || p.GlobalWarpedMotionCompoundValid[i])
		}
	}
	if p.MotionModeValid && p.MotionMode == tile.MotionModeOBMC && p.OverlappableNeighborsValid && p.Extras != nil {
		s := &p.Extras.OverlappableNeighbors
		for i := range min(int(s.AboveCount), len(s.Above)) {
			n.addNeighbor(&s.Above[i], bottom)
		}
		for i := range min(int(s.LeftCount), len(s.Left)) {
			n.addNeighbor(&s.Left[i], bottom)
		}
	}
	if c := p.SubChromaInter; c != nil {
		for i := range min(int(c.Count), len(c.Cells)) {
			n.add(c.Cells[i].Reference, bottom, c.Cells[i].MV, false)
		}
	}
}

func (n *frameThreadRefNeed) addNeighbor(o *tile.OverlappableNeighbor, bottom int32) {
	if !o.MotionValid {
		return
	}
	n.add(o.Motion.References.Ref0, bottom, o.Motion.MV[0], false)
}

// waitSuperblockReferences blocks until every reference the superblock's
// buffered blocks predict from has published the rows they read. It reports
// false when a reference failed to decode.
func (r *FrameThreadRecon) waitSuperblockReferences(b *FrameWorkBatch, events []frameWorkReconEvent, visits []tile.BlockLoopVisit) bool {
	var need frameThreadRefNeed
	if !need.addEvents(events, visits) {
		return true
	}
	return r.waitReferences(b, &need)
}

// rowReferenceNeed is the reference rows every superblock of SB row row reads
// (dav1d's per tile-sbrow lowest_pixel); it reports false when the row reads
// no reference.
func rowReferenceNeed(view *frameWorkReconView, row int) (frameThreadRefNeed, bool) {
	var need frameThreadRefNeed
	if row+1 >= len(view.rowStart) {
		return need, false
	}
	lo, hi := int(view.rowStart[row]), int(view.rowStart[row+1])
	if lo < 0 || hi < lo || hi > len(view.sbs) {
		return need, false
	}
	any := false
	for _, sb := range view.sbs[lo:hi] {
		if sb.start < 0 || sb.end < sb.start || int(sb.end) > len(view.events) {
			return need, false
		}
		if need.addEvents(view.events[sb.start:sb.end], view.visits) {
			any = true
		}
	}
	return need, any
}

// addEvents adds the reference rows the buffered blocks of events read and
// reports whether any block was found.
func (n *frameThreadRefNeed) addEvents(events []frameWorkReconEvent, visits []tile.BlockLoopVisit) bool {
	any := false
	for i := range events {
		ev := &events[i]
		if ev.kind != frameWorkReconEventBlockBegin || int(ev.index) >= len(visits) {
			continue
		}
		n.addVisit(&visits[ev.index])
		any = true
	}
	return any
}

func (r *FrameThreadRecon) waitReferences(b *FrameWorkBatch, need *frameThreadRefNeed) bool {
	for i := range need {
		p, rows := r.referenceRows(b, need, i)
		if p != nil && !p.Wait(rows) {
			return false
		}
	}
	return true
}

// referencesReady reports whether every reference has published the rows
// need names (or failed), so the row's task can run without blocking (dav1d
// check_tile).
func (r *FrameThreadRecon) referencesReady(b *FrameWorkBatch, need *frameThreadRefNeed) bool {
	for i := range need {
		p, rows := r.referenceRows(b, need, i)
		if p != nil && !p.Ready(rows) {
			return false
		}
	}
	return true
}

// referenceRows is reference i's progress and the luma rows of it need
// waits for, or nil when the reference is not read.
func (r *FrameThreadRecon) referenceRows(b *FrameWorkBatch, need *frameThreadRefNeed, i int) (*FrameProgress, int32) {
	rows := need[i]
	p := r.RefProgress[i]
	if rows == 0 || p == nil {
		return nil, 0
	}
	if rows != FrameProgressDone && b != nil {
		if i >= len(b.References) || b.References[i] == nil || b.frameThreadScaledReference(i) ||
			rows >= int32(b.References[i].Y.Height) {
			rows = FrameProgressDone
		}
	}
	return p, rows
}

// frameThreadScaledReference reports whether reference i differs in size
// from the frame being decoded (scaled prediction reads the whole reference).
func (b *FrameWorkBatch) frameThreadScaledReference(i int) bool {
	ref := b.References[i]
	return ref.Format.Width != int(b.FrameSize.CodedWidth) || ref.Format.Height != int(b.FrameSize.Height)
}
