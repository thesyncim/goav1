// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package decoder

import "github.com/thesyncim/goav1/internal/av1/parser"

// frameWorkSegMaps bounds the live segmentation maps: one per reference slot,
// the planning frame's current map, and per in-flight frame its current map
// plus the previous map its entropy pass reads (which the slots may already
// have dropped).
const frameWorkSegMaps = parser.RefFrames + 1 + 2*FrameThreadMaxFrames

// FrameWorkSegMaps is the segment-map handle count the frame-threading owner
// sizes its progress table by.
const FrameWorkSegMaps = frameWorkSegMaps

// SegmentMapStore is dav1d's segmap lifecycle (src/decode.c
// dav1d_submit_frame "segmap" block and the refresh loop, src/obu.c reset and
// show-existing key frame): each map is a refcounted handle, and the
// reference slots and the frames being decoded hold handles. Handle zero means
// "no map", which dav1d reads exactly like a zero-filled map
// (get_prev_frame_segid is skipped and seg_id is 0).
//
// A FrameWorkState owns one store; decoders whose layer states share the
// stream-global reference slots share one store (ShareSegmentMapStore).
type SegmentMapStore struct {
	backing [frameWorkSegMaps][]uint8
	refs    [frameWorkSegMaps]uint8
	cols    [frameWorkSegMaps]uint32
	rows    [frameWorkSegMaps]uint32
	slot    [parser.RefFrames]uint8 // one-based handle; zero has no map
}

// Reset drops every map reference, keeping the backing for reuse.
func (p *SegmentMapStore) Reset() {
	if p == nil {
		return
	}
	p.refs = [frameWorkSegMaps]uint8{}
	p.slot = [parser.RefFrames]uint8{}
}

func (p *SegmentMapStore) release(h uint8) {
	if h != 0 && p.refs[h-1] > 0 {
		p.refs[h-1]--
	}
}

func (p *SegmentMapStore) retain(h uint8) {
	if h != 0 {
		p.refs[h-1]++
	}
}

// acquire claims a free map sized cols x rows. The backing is reused once
// grown, so a steady-state decode does not allocate.
func (p *SegmentMapStore) acquire(cols, rows uint32) (uint8, error) {
	need := int(cols) * int(rows)
	for i := range p.refs {
		if p.refs[i] != 0 {
			continue
		}
		if cap(p.backing[i]) < need {
			p.backing[i] = make([]uint8, need)
		}
		p.backing[i] = p.backing[i][:need]
		p.refs[i] = 1
		p.cols[i] = cols
		p.rows[i] = rows
		return uint8(i + 1), nil
	}
	return 0, ErrInvalidFrameWorkState
}

func (p *SegmentMapStore) data(h uint8) []uint8 {
	if h == 0 {
		return nil
	}
	return p.backing[h-1]
}

// resetSlots drops every slot map (dav1d obu.c sequence reset).
func (p *SegmentMapStore) resetSlots() {
	for i := range p.slot {
		p.release(p.slot[i])
		p.slot[i] = 0
	}
}

// publish points every refreshed slot at map h.
func (p *SegmentMapStore) publish(refresh uint8, h uint8) {
	for i := range parser.RefFrames {
		if refresh&(1<<uint(i)) == 0 {
			continue
		}
		p.release(p.slot[i])
		p.slot[i] = h
		p.retain(h)
	}
}

// showExistingKey applies the show-existing key frame refresh: every slot
// takes the shown slot's map (dav1d obu.c).
func (p *SegmentMapStore) showExistingKey(shown uint8) {
	if shown >= parser.RefFrames {
		return
	}
	h := p.slot[shown]
	for i := range p.slot {
		if uint8(i) == shown {
			continue
		}
		p.release(p.slot[i])
		p.slot[i] = h
		p.retain(h)
	}
}

// segMapFrame is one frame's segmentation-map bindings (one-based handles
// into a SegmentMapStore, each held by the frame).
type segMapFrame struct {
	cur    uint8 // map published to the refreshed slots
	prev   uint8 // map the frame's segment-id prediction reads
	write  bool  // update_map: the entropy pass writes cur
	stride uint16
}

func (f *segMapFrame) release(p *SegmentMapStore) {
	p.release(f.cur)
	p.release(f.prev)
	*f = segMapFrame{}
}

// prepare binds the frame's maps (dav1d_submit_frame "segmap").
func (f *segMapFrame) prepare(p *SegmentMapStore, event *Event) error {
	f.release(p)
	seg := &event.Segmentation
	if !seg.Enabled || event.FrameHeader.ShowExistingFrame {
		return nil
	}
	cols := mvFrameMIExtent(event.FrameSize.CodedWidth)
	rows := mvFrameMIExtent(event.FrameSize.Height)
	if cols == 0 || rows == 0 || cols > uint32(^uint16(0)) {
		return ErrInvalidFrameWorkState
	}
	f.stride = uint16(cols)

	// The previous map is needed for temporal prediction or when the map is
	// not updated; it is used only when the primary reference's map has the
	// current frame's dimensions.
	if seg.TemporalUpdate || !seg.UpdateMap {
		pri := event.FrameHeader.PrimaryRefFrame
		if pri >= parser.InterRefsPerFrame {
			return ErrInvalidFrameWorkState
		}
		slot := event.FrameSize.RefFrameIdx[pri]
		if slot >= parser.RefFrames {
			return ErrInvalidFrameWorkState
		}
		if h := p.slot[slot]; h != 0 && p.cols[h-1] == cols && p.rows[h-1] == rows {
			f.prev = h
			p.retain(h)
		}
	}

	if seg.UpdateMap {
		h, err := p.acquire(cols, rows)
		if err != nil {
			f.release(p)
			return err
		}
		f.cur = h
		f.write = true
		return nil
	}
	// No update: the frame keeps the previous map (or, without one, the
	// all-zero map that handle zero stands for).
	f.cur = f.prev
	p.retain(f.cur)
	return nil
}

// fields returns the block-loop segment maps. The current map is bound only
// when the frame writes it (update_map): dav1d writes cur_segmap only under
// update_map, and otherwise the current map is the shared previous map.
func (f *segMapFrame) fields(p *SegmentMapStore) (cur, prev []uint8, stride uint16) {
	if f.write {
		cur = p.data(f.cur)
	}
	return cur, p.data(f.prev), f.stride
}

// segMapStore returns the store this state's frames publish into.
func (s *FrameWorkState) segMapStore() *SegmentMapStore {
	if s.segShared != nil {
		return s.segShared
	}
	return &s.segOwn
}

// ShareSegmentMapStore makes s publish and read segmentation maps through
// store, for decoders whose per-layer states share the stream-global
// reference slots. It is a free function so the public FrameWorkState alias
// gains no method.
func ShareSegmentMapStore(s *FrameWorkState, store *SegmentMapStore) {
	if s != nil {
		s.segShared = store
	}
}
