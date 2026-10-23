// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package decoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func segMapTestEvent(w, h uint32, primary uint8, refSlot uint8, refresh uint8, seg parser.SegmentationParams) *Event {
	var ev Event
	ev.FrameSize.CodedWidth = w
	ev.FrameSize.Height = h
	ev.FrameSize.RefreshFrameFlags = refresh
	ev.FrameHeader.PrimaryRefFrame = primary
	for i := range ev.FrameSize.RefFrameIdx {
		ev.FrameSize.RefFrameIdx[i] = refSlot
	}
	ev.Segmentation = seg
	return &ev
}

// TestSegmentMapLifecycle checks the dav1d segmap rules: a fresh map under
// update_map, the primary reference's map shared without it, the previous map
// dropped on a dimension change, show-existing key frame propagation, and the
// sequence reset.
func TestSegmentMapLifecycle(t *testing.T) {
	var store SegmentMapStore
	var f segMapFrame

	// Key frame (primary none): update_map, writes a fresh map to every slot.
	key := segMapTestEvent(64, 64, parser.PrimaryRefNone, 0, 0xff, parser.SegmentationParams{Enabled: true, UpdateMap: true})
	if err := f.prepare(&store, key); err != nil {
		t.Fatal(err)
	}
	cur, prev, stride := f.fields(&store)
	if len(cur) != 16*16 || prev != nil || stride != 16 {
		t.Fatalf("key frame maps cur=%d prev=%v stride=%d", len(cur), prev != nil, stride)
	}
	for i := range cur {
		cur[i] = 3
	}
	keyMap := f.cur
	store.publish(key.FrameSize.RefreshFrameFlags, f.cur)
	f.release(&store)
	if store.refs[keyMap-1] != parser.RefFrames {
		t.Fatalf("key map refs=%d want %d", store.refs[keyMap-1], parser.RefFrames)
	}

	// update_map=0 inherits the primary reference's map: read-only, shared.
	inherit := segMapTestEvent(64, 64, 0, 2, 0x01, parser.SegmentationParams{Enabled: true})
	if err := f.prepare(&store, inherit); err != nil {
		t.Fatal(err)
	}
	cur, prev, _ = f.fields(&store)
	if cur != nil || len(prev) != 16*16 || prev[0] != 3 || f.cur != keyMap {
		t.Fatalf("inherit cur=%v prev=%d cur handle=%d", cur != nil, len(prev), f.cur)
	}
	store.publish(inherit.FrameSize.RefreshFrameFlags, f.cur)
	f.release(&store)

	// temporal_update reads the previous map and writes a fresh one.
	temporal := segMapTestEvent(64, 64, 0, 0, 0x02, parser.SegmentationParams{Enabled: true, UpdateMap: true, TemporalUpdate: true})
	if err := f.prepare(&store, temporal); err != nil {
		t.Fatal(err)
	}
	cur, prev, _ = f.fields(&store)
	if len(cur) != 16*16 || len(prev) != 16*16 || f.cur == keyMap || f.prev != keyMap {
		t.Fatalf("temporal cur=%d prev=%d", len(cur), len(prev))
	}
	f.release(&store)

	// A frame-size change drops the previous map (dav1d ref_w/ref_h check).
	resized := segMapTestEvent(32, 64, 0, 0, 0x04, parser.SegmentationParams{Enabled: true, UpdateMap: true, TemporalUpdate: true})
	if err := f.prepare(&store, resized); err != nil {
		t.Fatal(err)
	}
	if _, prev, stride = f.fields(&store); prev != nil || stride != 8 {
		t.Fatalf("resized prev=%v stride=%d", prev != nil, stride)
	}
	store.publish(resized.FrameSize.RefreshFrameFlags, f.cur)
	f.release(&store)
	resizedMap := store.slot[2]

	// Segmentation disabled: the refreshed slots lose their map.
	off := segMapTestEvent(64, 64, 0, 0, 0x08, parser.SegmentationParams{})
	if err := f.prepare(&store, off); err != nil {
		t.Fatal(err)
	}
	store.publish(off.FrameSize.RefreshFrameFlags, f.cur)
	f.release(&store)
	if store.slot[3] != 0 {
		t.Fatalf("disabled segmentation left slot map %d", store.slot[3])
	}

	// Show-existing key frame: every slot takes the shown slot's map.
	store.showExistingKey(2)
	for i, h := range store.slot {
		if h != resizedMap {
			t.Fatalf("slot %d map %d want %d", i, h, resizedMap)
		}
	}
	if store.refs[resizedMap-1] != parser.RefFrames {
		t.Fatalf("shown map refs=%d", store.refs[resizedMap-1])
	}
	if store.refs[keyMap-1] != 0 {
		t.Fatalf("key map still held: refs=%d", store.refs[keyMap-1])
	}

	store.resetSlots()
	for i := range store.refs {
		if store.refs[i] != 0 {
			t.Fatalf("map %d still held after reset", i+1)
		}
	}
}

// TestSegmentMapSteadyStateNoAllocs checks that reusing the store across
// frames does not allocate once its maps are backed.
func TestSegmentMapSteadyStateNoAllocs(t *testing.T) {
	var store SegmentMapStore
	var f segMapFrame
	ev := segMapTestEvent(64, 64, 0, 0, 0x01, parser.SegmentationParams{Enabled: true, UpdateMap: true, TemporalUpdate: true})
	step := func() {
		if err := f.prepare(&store, ev); err != nil {
			t.Fatal(err)
		}
		store.publish(ev.FrameSize.RefreshFrameFlags, f.cur)
		f.release(&store)
	}
	for range 4 {
		step()
	}
	if n := testing.AllocsPerRun(50, step); n != 0 {
		t.Fatalf("allocs per frame = %v", n)
	}
}
