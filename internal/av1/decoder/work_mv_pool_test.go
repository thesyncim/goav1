package decoder

import (
	"errors"
	"reflect"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// copyMVOracle freezes the main-10f4a6ae prepare/publish implementation. It
// retains independent backing per slot and never calls the new pool helpers.
type copyMVOracle struct {
	Sequence            threading.FrameWorkSequenceContext
	sharedFrameContexts *SharedFrameContextStore
	mvFrameStore        [parser.RefFrames]tile.ReferenceMVFrame
	mvFrameStoreBacking [parser.RefFrames][]tile.ReferenceMVEntry
	mvFrameStoreMeta    [parser.RefFrames]mvFrameMeta
	currentMVFrame      tile.ReferenceMVFrame
	currentMVFrameBack  []tile.ReferenceMVEntry
	currentMVFrameValid bool
	temporalMVs         tile.TemporalMotionField
	temporalMVsBack     []tile.TemporalMotionEntry
	temporalMVsValid    bool
	mvReferences        [parser.InterRefsPerFrame]tile.TemporalMotionReferenceFrame
}

func (s *copyMVOracle) prepareTemporalMotionField(event Event) error {
	s.currentMVFrameValid = false
	s.temporalMVsValid = false
	if s.sharedFrameContexts != nil {
		// The shared/SVC path wires per-surface MV frames through the caller's
		// store and batch callback; leave the batch pointers nil here.
		return nil
	}
	if event.FrameHeader.ShowExistingFrame {
		return nil
	}

	miCols := mvFrameMIExtent(event.FrameSize.CodedWidth)
	miRows := mvFrameMIExtent(event.FrameSize.Height)
	need, err := tile.ReferenceMVFrameEntries(miRows, miCols)
	if err != nil {
		// A zero-dimension frame carries no MV grid; nothing to project.
		return nil
	}
	if cap(s.currentMVFrameBack) < need {
		s.currentMVFrameBack = make([]tile.ReferenceMVEntry, need)
	}
	if err := s.currentMVFrame.Init(miRows, miCols, s.currentMVFrameBack[:need]); err != nil {
		return err
	}
	s.currentMVFrameValid = true

	if !event.TileInfo.UseRefFrameMVS {
		return nil
	}

	// Resolve each inter reference's saved MV_REF side data + order-hint
	// metadata from the per-slot store, keyed by RefFrameIdx exactly as libaom
	// indexes ref_frame_map. A reference whose slot was never published (no
	// prior decoded frame) leaves the temporal projection without that source.
	var refs [parser.InterRefsPerFrame]tile.TemporalMotionReferenceFrame
	for ref := 0; ref < parser.InterRefsPerFrame; ref++ {
		slot := int(event.FrameSize.RefFrameIdx[ref])
		if slot < 0 || slot >= parser.RefFrames {
			return ErrInvalidSurfaceReference
		}
		meta := s.mvFrameStoreMeta[slot]
		if !meta.valid {
			// A reference slot never written by a prior frame has no MV grid to
			// project. Leave its entry's Frame nil; Setup/projectSetupReference
			// skip nil references per-reference, matching libaom's
			// av1_setup_motion_field guarding on ref_buf[ref] != NULL.
			continue
		}
		refs[ref] = tile.TemporalMotionReferenceFrame{
			Frame:         &s.mvFrameStore[slot],
			OrderHint:     meta.orderHint,
			RefOrderHints: meta.refOrderHints,
			IntraOnly:     meta.intraOnly,
		}
	}
	s.mvReferences = refs

	if cap(s.temporalMVsBack) < need {
		s.temporalMVsBack = make([]tile.TemporalMotionEntry, need)
	}
	if err := s.temporalMVs.Init(miRows, miCols, s.temporalMVsBack[:need]); err != nil {
		return err
	}
	if _, err := s.temporalMVs.Setup(tile.TemporalMotionSetupRequest{
		EnableOrderHint:  s.Sequence.EnableOrderHint,
		OrderHintBits:    s.Sequence.OrderHintBits,
		CurrentOrderHint: event.FrameHeader.OrderHint,
		References:       s.mvReferences,
	}); err != nil {
		return err
	}
	s.temporalMVsValid = true
	return nil
}

func (s *copyMVOracle) publishTemporalMotionFrame(event Event) {
	if s == nil || s.sharedFrameContexts != nil || !s.currentMVFrameValid {
		return
	}
	if err := s.currentMVFrame.Validate(); err != nil {
		return
	}
	meta := mvFrameMeta{
		valid:         true,
		orderHint:     event.FrameHeader.OrderHint,
		refOrderHints: event.ReferenceOrderHints,
		intraOnly: event.FrameHeader.FrameType == parser.FrameTypeKey ||
			event.FrameHeader.FrameType == parser.FrameTypeIntraOnly,
	}
	for i := range parser.RefFrames {
		if (event.FrameSize.RefreshFrameFlags & (1 << uint(i))) == 0 {
			continue
		}
		need := len(s.currentMVFrame.Entries)
		if cap(s.mvFrameStoreBacking[i]) < need {
			s.mvFrameStoreBacking[i] = make([]tile.ReferenceMVEntry, need)
		}
		dst := s.mvFrameStoreBacking[i][:need]
		copy(dst, s.currentMVFrame.Entries)
		s.mvFrameStore[i] = tile.ReferenceMVFrame{
			Rows:    s.currentMVFrame.Rows,
			Cols:    s.currentMVFrame.Cols,
			Stride:  s.currentMVFrame.Stride,
			Entries: dst,
		}
		s.mvFrameStoreMeta[i] = meta
	}
}

func mvPoolEvent(n int) Event {
	e := Event{
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeInter, OrderHint: uint8(8 + n*3)},
		FrameSize:   parser.FrameSize{CodedWidth: 32, Height: 24, RefreshFrameFlags: uint8(1 << uint(n%8))},
	}
	if n%11 == 0 {
		e.FrameHeader.FrameType = parser.FrameTypeKey
		e.FrameSize.RefreshFrameFlags = 255
	}
	if n%11 == 5 {
		e.FrameHeader.FrameType = parser.FrameTypeIntraOnly
		e.FrameSize.AllowIntrabc = n%2 == 0
	}
	if n%7 == 6 {
		e.FrameSize.RefreshFrameFlags = 0
	}
	if n%7 == 3 {
		e.FrameSize.RefreshFrameFlags = 0x95
	}
	if n%13 == 12 {
		e.FrameSize.CodedWidth = 16
		e.FrameSize.Height = 40
	}
	e.TileInfo.UseRefFrameMVS = e.FrameHeader.FrameType == parser.FrameTypeInter
	for i := range e.FrameSize.RefFrameIdx {
		e.FrameSize.RefFrameIdx[i] = uint8((n + i) % 8)
		e.ReferenceOrderHints[i] = e.FrameHeader.OrderHint - uint8(i+1)
	}
	return e
}

func fillMVPoolFrame(f *tile.ReferenceMVFrame, n int) {
	for i := range f.Entries {
		f.Entries[i] = tile.ReferenceMVEntry{Ref: tile.ReferenceFrameLast, MV: motion.Vector{Row: int16((i+n)%17 - 8), Col: int16((i*3+n)%19 - 9)}, Valid: i%3 != 0}
	}
}

func checkMVPoolRefs(t *testing.T, s *FrameWorkState) {
	t.Helper()
	var want [parser.RefFrames + 1]uint8
	for _, handle := range s.mvSlotGrid {
		if handle != 0 {
			want[handle-1]++
		}
	}
	if s.currentMVGrid != 0 {
		want[s.currentMVGrid-1]++
	}
	if s.mvGridRefs != want {
		t.Fatalf("pool refs=%v want %v", s.mvGridRefs, want)
	}
}

func TestOwnedMVPoolMatchesFrozenCopy(t *testing.T) {
	var s FrameWorkState
	EnableOwnedTemporalMotionStorage(&s)
	sequence := threading.FrameWorkSequenceContext{EnableOrderHint: true, OrderHintBits: 8}
	var old copyMVOracle
	for n := 0; n < 48; n++ {
		s.Sequence = sequence
		old.Sequence = sequence
		event := mvPoolEvent(n)
		// Preparing a new writable grid must not clear any completed slot.
		var saved [parser.RefFrames][]tile.ReferenceMVEntry
		for i := range saved {
			saved[i] = append([]tile.ReferenceMVEntry(nil), s.mvFrameStore[i].Entries...)
		}
		err := s.prepareTemporalMotionField(&event)
		wantErr := old.prepareTemporalMotionField(event)
		if err != wantErr {
			t.Fatalf("frame %d prepare=%v old=%v", n, err, wantErr)
		}
		for i := range saved {
			if !reflect.DeepEqual(saved[i], s.mvFrameStore[i].Entries) {
				t.Fatalf("frame %d mutated saved slot %d during prepare", n, i)
			}
		}
		if s.temporalMVsValid != old.temporalMVsValid || !reflect.DeepEqual(s.temporalMVs, old.temporalMVs) {
			t.Fatalf("frame %d temporal projection mismatch", n)
		}
		if event.TileInfo.UseRefFrameMVS {
			for i := range s.mvReferences {
				got, want := s.mvReferences[i], old.mvReferences[i]
				if (got.Frame == nil) != (want.Frame == nil) || got.OrderHint != want.OrderHint || got.RefOrderHints != want.RefOrderHints || got.IntraOnly != want.IntraOnly {
					t.Fatalf("frame %d reference %d metadata mismatch", n, i)
				}
			}
			req := tile.TemporalMotionSetupRequest{EnableOrderHint: true, OrderHintBits: 8, CurrentOrderHint: event.FrameHeader.OrderHint, References: s.mvReferences}
			gotStats, gotErr := s.temporalMVs.Setup(req)
			req.References = old.mvReferences
			wantStats, wantErr := old.temporalMVs.Setup(req)
			if gotStats != wantStats || gotErr != wantErr {
				t.Fatalf("frame %d setup counters/errors differ: %v/%v versus %v/%v", n, gotStats, gotErr, wantStats, wantErr)
			}
		}
		if event.FrameHeader.FrameType == parser.FrameTypeInter {
			fillMVPoolFrame(&s.currentMVFrame, n)
			fillMVPoolFrame(&old.currentMVFrame, n)
		}
		s.publishTemporalMotionFrame(&event)
		old.publishTemporalMotionFrame(event)
		if s.mvFrameStoreMeta != old.mvFrameStoreMeta {
			t.Fatalf("frame %d saved metadata differs", n)
		}
		for i, meta := range s.mvFrameStoreMeta {
			if !meta.valid || meta.intraOnly {
				continue
			}
			if !reflect.DeepEqual(s.mvFrameStore[i], old.mvFrameStore[i]) {
				t.Fatalf("frame %d slot %d differs from frozen copy", n, i)
			}
		}
		checkMVPoolRefs(t, &s)
		s.resetActive()
		// Match resetActive's legacy transient state without invoking candidate code.
		old.currentMVFrame = tile.ReferenceMVFrame{}
		old.currentMVFrameValid = false
		old.temporalMVs = tile.TemporalMotionField{}
		old.temporalMVsValid = false
		old.mvReferences = [parser.InterRefsPerFrame]tile.TemporalMotionReferenceFrame{}
		checkMVPoolRefs(t, &s)
	}
}

func TestOwnedMVPoolSharingAndGenericIsolation(t *testing.T) {
	for _, owned := range []bool{false, true} {
		var s FrameWorkState
		if owned {
			EnableOwnedTemporalMotionStorage(&s)
		}
		e := mvPoolEvent(1)
		e.TileInfo.UseRefFrameMVS = false
		e.FrameSize.RefreshFrameFlags = 255
		if err := s.prepareTemporalMotionField(&e); err != nil {
			t.Fatal(err)
		}
		fillMVPoolFrame(&s.currentMVFrame, 1)
		s.publishTemporalMotionFrame(&e)
		current := &s.currentMVFrame.Entries[0]
		for i := range s.mvFrameStore {
			shared := &s.mvFrameStore[i].Entries[0] == current
			if shared != owned {
				t.Fatalf("owned=%v slot %d shared=%v", owned, i, shared)
			}
		}
		if !owned {
			saved := s.mvFrameStore[1].Entries[0]
			s.currentMVFrame.Entries[0].MV.Row++
			s.mvFrameStore[0].Entries[0].MV.Col++
			if s.mvFrameStore[1].Entries[0] != saved {
				t.Fatal("generic mutable slots alias")
			}
		}
		s.resetActive()
		if owned {
			for _, refs := range s.mvGridRefs {
				if refs != 0 && refs != 8 {
					t.Fatalf("shared completed refs=%d", refs)
				}
			}
		}
		e.FrameSize.RefreshFrameFlags = 0
		if err := s.prepareTemporalMotionField(&e); err != nil {
			t.Fatal(err)
		}
		if owned && &s.currentMVFrame.Entries[0] == current {
			t.Fatal("reused a referenced grid")
		}
		s.resetActive()
		s.resetReferenceState()
		checkMVPoolRefs(t, &s)
		for _, refs := range s.mvGridRefs {
			if refs != 0 {
				t.Fatal("sequence reset retained a slot")
			}
		}
	}
}

func TestOwnedMVPoolIntraMetadataAndFallback(t *testing.T) {
	for _, owned := range []bool{false, true} {
		for _, intrabc := range []bool{false, true} {
			var s FrameWorkState
			if owned {
				EnableOwnedTemporalMotionStorage(&s)
			}
			e := mvPoolEvent(0)
			e.FrameSize.AllowIntrabc = intrabc
			if err := s.prepareTemporalMotionField(&e); err != nil {
				t.Fatal(err)
			}
			omit := owned && !intrabc
			if s.currentMVFrameValid == omit || s.currentMVIntra != omit {
				t.Fatalf("owned=%v intrabc=%v grid=%v intra=%v", owned, intrabc, s.currentMVFrameValid, s.currentMVIntra)
			}
			if (s.motionFields().current == nil) != omit {
				t.Fatal("block-loop current-grid binding differs")
			}
			s.publishTemporalMotionFrame(&e)
			s.resetActive()
			next := mvPoolEvent(1)
			s.Sequence = threading.FrameWorkSequenceContext{EnableOrderHint: true, OrderHintBits: 8}
			if err := s.prepareTemporalMotionField(&next); err != nil {
				t.Fatal(err)
			}
			for _, ref := range s.mvReferences {
				if ref.Frame == nil || !ref.IntraOnly {
					t.Fatal("intra reference presence lost")
				}
			}
			s.resetActive()
			s.sharedFrameContexts = &SharedFrameContextStore{}
			if err := s.prepareTemporalMotionField(&next); err != nil {
				t.Fatal(err)
			}
			if s.currentMVFrameValid || s.currentMVGrid != 0 || s.currentMVIntra {
				t.Fatal("shared-store path acquired a private grid")
			}
			s.sharedFrameContexts = nil
			next.FrameHeader.ShowExistingFrame = true
			if err := s.prepareTemporalMotionField(&next); err != nil {
				t.Fatal(err)
			}
			if s.currentMVFrameValid || s.currentMVGrid != 0 || s.currentMVIntra {
				t.Fatal("show-existing acquired a grid")
			}
		}
	}
}

func TestOwnedMVPoolFailureAndReuse(t *testing.T) {
	var s FrameWorkState
	EnableOwnedTemporalMotionStorage(&s)
	e := mvPoolEvent(1)
	e.FrameSize.RefFrameIdx[3] = parser.RefFrames
	if err := s.prepareTemporalMotionField(&e); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("invalid ref error=%v", err)
	}
	if s.currentMVGrid != 0 || s.currentMVFrameValid {
		t.Fatal("failed prepare retained current grid")
	}
	checkMVPoolRefs(t, &s)
	e = mvPoolEvent(1)
	s.Sequence = threading.FrameWorkSequenceContext{EnableOrderHint: true, OrderHintBits: 0}
	if err := s.prepareTemporalMotionField(&e); err == nil {
		t.Fatal("invalid order bits accepted")
	}
	if s.currentMVGrid != 0 || s.currentMVFrameValid {
		t.Fatal("failed projection retained current grid")
	}
	s.Sequence = threading.FrameWorkSequenceContext{EnableOrderHint: true, OrderHintBits: 8}
	if err := s.prepareTemporalMotionField(&e); err != nil {
		t.Fatal(err)
	}
	checkMVPoolRefs(t, &s)
	s.resetActive()
	// All eight different slot grids can remain live while the ninth is written.
	e.TileInfo.UseRefFrameMVS = false
	for n := 0; n < 24; n++ {
		e.FrameSize.RefreshFrameFlags = 1 << uint(n%8)
		if err := s.prepareTemporalMotionField(&e); err != nil {
			t.Fatal(err)
		}
		s.publishTemporalMotionFrame(&e)
		s.resetActive()
		checkMVPoolRefs(t, &s)
	}
	s.Reset()
	if !s.ownedTemporalMotion {
		t.Fatal("Reset lost private ownership mode")
	}
	checkMVPoolRefs(t, &s)
	for _, meta := range s.mvFrameStoreMeta {
		if meta.valid {
			t.Fatal("Reset kept saved metadata")
		}
	}
}

func TestOwnedMVPoolFinishAbortFailureLifetimes(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var s FrameWorkState
	EnableOwnedTemporalMotionStorage(&s)
	event := Event{Kind: EventFrameHeader, FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey}, FrameSize: testFrameSize(16, 16)}
	event.FrameSize.AllowIntrabc = true // retain a writable grid for the lifecycle checks
	plan, _, err := s.Begin(&refs, &pool, testSequence(), &event, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	handle := s.currentMVGrid
	if handle == 0 {
		t.Fatal("Begin did not acquire current grid")
	}
	if _, err = s.Finish(&refs, &pool, &Event{}, nil); err == nil {
		t.Fatal("invalid finish accepted")
	}
	if s.currentMVGrid != handle || !s.Active() {
		t.Fatal("failed finish released current grid")
	}
	if err = s.Abort(&pool); err != nil {
		t.Fatal(err)
	}
	if s.currentMVGrid != 0 || s.Active() {
		t.Fatal("Abort kept current grid")
	}
	plan, _, err = s.Begin(&refs, &pool, testSequence(), &event, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	handle = s.currentMVGrid
	if err = pool.Release(plan.Surface); err != nil {
		t.Fatal(err)
	}
	if err = s.Abort(&pool); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("abort failure=%v", err)
	}
	if s.currentMVGrid != handle || !s.Active() {
		t.Fatal("failed abort released current grid")
	}
	s.Reset()
	checkMVPoolRefs(t, &s)
}

func TestOwnedMVPoolWarmAllocs(t *testing.T) {
	var s FrameWorkState
	EnableOwnedTemporalMotionStorage(&s)
	e := mvPoolEvent(1)
	e.TileInfo.UseRefFrameMVS = false
	e.FrameSize.RefreshFrameFlags = 255
	if err := s.PreallocTemporalMotionScratch(e.FrameSize); err != nil {
		t.Fatal(err)
	}
	var err error
	allocs := testing.AllocsPerRun(20, func() {
		for i := 0; i < 12; i++ {
			err = s.prepareTemporalMotionField(&e)
			if err != nil {
				return
			}
			s.publishTemporalMotionFrame(&e)
			s.resetActive()
		}
		s.Reset()
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("warm allocations=%v", allocs)
	}
}
