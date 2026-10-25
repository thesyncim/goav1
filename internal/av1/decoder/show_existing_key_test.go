// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package decoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// showExistingKeyTestState binds slot r to a key frame (CDF entry 0, MV grid
// 1, intra metadata) and every other slot to an older inter frame (CDF entry
// 1, MV grid 2).
func showExistingKeyTestState(r int) *FrameWorkState {
	s := &FrameWorkState{ownedTemporalMotion: true}
	for i := range parser.RefFrames {
		cdf, grid := int8(1), uint8(2)
		meta := mvFrameMeta{valid: true, orderHint: 3}
		if i == r {
			cdf, grid = 0, 1
			meta = mvFrameMeta{valid: true, orderHint: 9, intraOnly: true}
		}
		s.tileResidualSlotCDF[i] = cdf
		s.tileResidualFrameContextValid[i] = true
		s.tileResidualCDFPoolRefs[cdf]++
		s.mvSlotGrid[i] = grid
		s.mvGridRefs[grid-1]++
		s.mvFrameStore[i] = tile.ReferenceMVFrame{Rows: 1, Cols: 1, Stride: 1}
		s.mvFrameStoreMeta[i] = meta
	}
	return s
}

func checkShowExistingKeySlots(t *testing.T, s *FrameWorkState, r int) {
	t.Helper()
	for i := range parser.RefFrames {
		if s.tileResidualSlotCDF[i] != 0 || !s.tileResidualFrameContextValid[i] {
			t.Fatalf("slot %d CDF entry %d valid=%v, want the shown key frame's entry 0", i, s.tileResidualSlotCDF[i], s.tileResidualFrameContextValid[i])
		}
		if s.mvFrameStoreMeta[i] != s.mvFrameStoreMeta[r] {
			t.Fatalf("slot %d MV metadata %+v, want %+v", i, s.mvFrameStoreMeta[i], s.mvFrameStoreMeta[r])
		}
		if i != r && (s.mvSlotGrid[i] != 0 || s.mvFrameStore[i].Rows != 0) {
			t.Fatalf("slot %d kept MV grid %d", i, s.mvSlotGrid[i])
		}
	}
	if got := s.tileResidualCDFPoolRefs[0]; got != parser.RefFrames {
		t.Fatalf("key CDF entry refs=%d want %d", got, parser.RefFrames)
	}
	if got := s.tileResidualCDFPoolRefs[1]; got != 0 {
		t.Fatalf("stale CDF entry still held: refs=%d", got)
	}
	if s.mvGridRefs[0] != 1 || s.mvGridRefs[1] != 0 {
		t.Fatalf("MV grid refs=%v want [1 0 ...]", s.mvGridRefs[:2])
	}
}

// TestShowExistingKeyResetsSlotState: a key frame shown again saves its
// state into every reference slot (AV1 7.21 with refresh_frame_flags =
// allFrames; dav1d obu.c copies cdf[r] and segmap[r] to every slot and drops
// the other slots' refmvs). A later frame whose primary_ref_frame names
// another slot must load the key frame's CDFs, not that slot's old ones.
func TestShowExistingKeyResetsSlotState(t *testing.T) {
	const r = 2
	pool := testFramePool(t, parser.RefFrames)
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	for i := range parser.RefFrames {
		surface, _, err := pool.Acquire()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := refs.Refresh(1<<uint(i), surface, releases[:]); err != nil {
			t.Fatal(err)
		}
	}
	s := showExistingKeyTestState(r)
	plan, err := s.ShowExisting(&refs, &pool, showExistingWorkEventPtr(r, parser.FrameTypeKey), releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan.ReleaseCount != parser.RefFrames-1 || pool.Available() != parser.RefFrames-1 {
		t.Fatalf("plan=%+v available=%d", plan, pool.Available())
	}
	checkShowExistingKeySlots(t, s, r)
}

// TestShowExistingFrameThreadedDefersReleases: queued in the frame-thread
// ring, a key frame shown again applies the same slot reset at once but hands
// the surfaces it drops to the ring slot instead of the pool, since frames in
// flight may still read them.
func TestShowExistingFrameThreadedDefersReleases(t *testing.T) {
	const r = 5
	pool := testFramePool(t, parser.RefFrames)
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	var surfaces [parser.RefFrames]int
	for i := range parser.RefFrames {
		surface, _, err := pool.Acquire()
		if err != nil {
			t.Fatal(err)
		}
		surfaces[i] = surface
		if _, err := refs.Refresh(1<<uint(i), surface, releases[:]); err != nil {
			t.Fatal(err)
		}
	}
	s := showExistingKeyTestState(r)
	var ft FrameThreadFrame
	plan, err := s.ShowExistingFrameThreaded(&refs, showExistingWorkEventPtr(r, parser.FrameTypeKey), &ft)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Surface != surfaces[r] || ft.ReleaseCount != parser.RefFrames-1 || pool.Available() != 0 {
		t.Fatalf("plan=%+v releases=%d available=%d", plan, ft.ReleaseCount, pool.Available())
	}
	for i := range parser.RefFrames {
		if got, _ := refs.ReferenceSlot(i); got != surfaces[r] {
			t.Fatalf("slot %d surface %d want %d", i, got, surfaces[r])
		}
	}
	checkShowExistingKeySlots(t, s, r)

	// A non-key frame shown again changes nothing.
	ft = FrameThreadFrame{}
	if _, err := s.ShowExistingFrameThreaded(&refs, showExistingWorkEventPtr(0, parser.FrameTypeInter), &ft); err != nil || ft.ReleaseCount != 0 {
		t.Fatalf("inter show-existing: releases=%d err=%v", ft.ReleaseCount, err)
	}
}
