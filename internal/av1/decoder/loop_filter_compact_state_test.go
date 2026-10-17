package decoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestFrameWorkStateCompactLoopFilterLifecycle(t *testing.T) {
	t.Run("reset deactivates owned state", func(t *testing.T) {
		state, compact, masks, cdef, buffers := compactFrameWorkStateFixture(t)
		if err := SetOwnedFrameWorkLoopFilterSideData(state, cdef, masks, compact, buffers); err != nil {
			t.Fatal(err)
		}
		state.Reset()
		if threading.FrameWorkLoopFilterCompactStateActive(compact) {
			t.Fatal("Reset left the owned compact state active")
		}
		if state.loopFilterCompactState != nil {
			t.Fatal("Reset retained the compact-state link")
		}
	})

	t.Run("rejected public rebind preserves owned state", func(t *testing.T) {
		state, compact, masks, cdef, buffers := compactFrameWorkStateFixture(t)
		if err := SetOwnedFrameWorkLoopFilterSideData(state, cdef, masks, compact, buffers); err != nil {
			t.Fatal(err)
		}
		if err := state.SetSideData(threading.FrameWorkCDEFIndexMap{}, threading.FrameWorkLoopFilterMap{}, buffers); err == nil {
			t.Fatal("SetSideData accepted invalid public views")
		}
		if !threading.FrameWorkLoopFilterCompactStateActive(compact) || state.loopFilterCompactState != compact {
			t.Fatal("rejected SetSideData invalidated the previous owned binding")
		}
	})

	t.Run("successful generic rebind deactivates owned state", func(t *testing.T) {
		state, compact, masks, cdef, buffers := compactFrameWorkStateFixture(t)
		if err := SetOwnedFrameWorkLoopFilterSideData(state, cdef, masks, compact, buffers); err != nil {
			t.Fatal(err)
		}
		lfMap := threading.FrameWorkLoopFilterMap{Records: make([]threading.FrameWorkLoopFilterBlockRecord, 16), Stride: 4, Rows: 4}
		if err := state.SetLoopFilterMap(lfMap); err != nil {
			t.Fatal(err)
		}
		if threading.FrameWorkLoopFilterCompactStateActive(compact) || state.loopFilterCompactState != nil {
			t.Fatal("SetLoopFilterMap retained the owned compact binding")
		}
		if !state.loopFilterMapValid || state.loopFilterMap.Records == nil {
			t.Fatal("SetLoopFilterMap did not install the public map")
		}
	})
}

func compactFrameWorkStateFixture(t *testing.T) (*FrameWorkState, *threading.FrameWorkLoopFilterCompactState, threading.FrameWorkLoopFilterMasks, threading.FrameWorkCDEFIndexMap, threading.FrameWorkRestorationFrameBuffers) {
	t.Helper()
	state := &FrameWorkState{active: true}
	masks := threading.FrameWorkLoopFilterMasks{
		Masks:      make([]lfmask.FilterMask, 1),
		LevelCache: make([][4]uint8, 16),
		Cols:       4,
		Rows:       4,
		SB128W:     1,
		SB128H:     1,
		Layout:     lfmask.Layout{Mono: true},
	}
	compact := &threading.FrameWorkLoopFilterCompactState{}
	if err := threading.BindFrameWorkLoopFilterCompactState(&masks, compact, make([]byte, 8), parser.LoopFilterParams{}, parser.SegmentationParams{}, parser.DeltaParams{}, true); err != nil {
		t.Fatal(err)
	}
	cdef := threading.FrameWorkCDEFIndexMap{Index: make([]uint8, 1), Read: make([]bool, 1), Stride: 1, Rows: 1}
	buffers := threading.FrameWorkRestorationFrameBuffers{Plan: tile.RestorationFramePlan{Planes: 1}}
	return state, compact, masks, cdef, buffers
}
