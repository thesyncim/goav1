package decoder

import (
	"errors"
	"fmt"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestPlanTileWorkTileGroupEvent(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	plan, err := PlanTileWork(events[2], 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan != (TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) {
		t.Fatalf("plan=%+v", plan)
	}
	if spans[0] != (parser.TileSpan{Tile: 0, Row: 0, Col: 0, Offset: 0, Size: 1}) {
		t.Fatalf("span=%+v", spans[0])
	}
	if jobs[0].Tile != 0 || jobs[0].SBCols != 1 || jobs[0].SBRows != 1 || jobs[0].Offset != 0 || jobs[0].Size != 1 {
		t.Fatalf("job=%+v", jobs[0])
	}
	if batches[0] != (threading.Batch{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0, Units: 1}) {
		t.Fatalf("batch=%+v", batches[0])
	}
}

func TestBeginFrameWorkFrameHeader(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	plan, output, err := BeginFrameWork(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || plan != (FrameWorkPlan{Surface: 0}) {
		t.Fatalf("plan=%+v output=%p", plan, output)
	}
	if pool.Available() != 0 {
		t.Fatalf("available=%d want 0", pool.Available())
	}
}

func TestBeginFrameWorkFrameEventPlansTiles(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	plan, output, err := BeginFrameWork(&refs, &pool, testSequence(), events[1], 32, nil, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || plan.Surface != 0 || plan.ReferenceCount != 0 || plan.Tile != (TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) {
		t.Fatalf("plan=%+v output=%p", plan, output)
	}
	if spans[0].Offset != len(reducedStillFrameHeaderPayload()) || spans[0].Size != 1 {
		t.Fatalf("span=%+v", spans[0])
	}
	if jobs[0].Tile != 0 || batches[0].Count != 1 {
		t.Fatalf("job=%+v batch=%+v", jobs[0], batches[0])
	}
}

func TestBeginFrameWorkRejectsTilePlanBeforeAcquire(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	_, _, err := BeginFrameWork(&refs, &pool, testSequence(), Event{
		Kind:        EventFrame,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, spans[:], jobs[:], batches[:])
	if !errors.Is(err, parser.ErrInvalidTileGroup) {
		t.Fatalf("BeginFrameWork err=%v want %v", err, parser.ErrInvalidTileGroup)
	}
	if pool.Available() != 1 {
		t.Fatalf("tile failure consumed a slot, available=%d", pool.Available())
	}
}

func TestBeginFrameWorkRejectsFormatAfterTilePlanBeforeAcquire(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}
	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	events[1].FrameSize.CodedWidth = 32

	var refs SurfaceReferences
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	_, _, err = BeginFrameWork(&refs, &pool, testSequence(), events[1], 32, nil, 1, spans[:], jobs[:], batches[:])
	if !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("BeginFrameWork err=%v want %v", err, frame.ErrInvalidFormat)
	}
	if pool.Available() != 1 {
		t.Fatalf("format failure consumed a slot, available=%d", pool.Available())
	}
	if spans[0].Size != 1 || jobs[0].Size != 1 || batches[0].Count != 1 {
		t.Fatalf("tile plan was not written before acquire failure: span=%+v job=%+v batch=%+v", spans[0], jobs[0], batches[0])
	}
}

func TestBeginFrameWorkRejectsNonFrameEvent(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	_, _, err := BeginFrameWork(&refs, &pool, testSequence(), Event{Kind: EventTileGroup}, 32, nil, 1, nil, nil, nil)
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("BeginFrameWork err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
	if pool.Available() != 1 {
		t.Fatalf("invalid event consumed a slot, available=%d", pool.Available())
	}
}

func TestPlanFrameTileWorkTileGroup(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	plan, err := PlanFrameTileWork(events[2], 5, 0, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan.Surface != 5 || plan.ReferenceCount != 0 || plan.Tile != (TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) {
		t.Fatalf("plan=%+v", plan)
	}
	if jobs[0].Tile != 0 || batches[0].Count != 1 {
		t.Fatalf("job=%+v batch=%+v", jobs[0], batches[0])
	}
}

func TestPlanFrameTileWorkRejectsInvalidState(t *testing.T) {
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	_, err := PlanFrameTileWork(Event{Kind: EventFrameHeader}, 0, 0, 1, spans[:], jobs[:], batches[:])
	if !errors.Is(err, ErrInvalidTileWork) {
		t.Fatalf("wrong event err=%v want %v", err, ErrInvalidTileWork)
	}
	_, err = PlanFrameTileWork(Event{Kind: EventTileGroup}, -1, 0, 1, spans[:], jobs[:], batches[:])
	if !errors.Is(err, ErrInvalidTileWork) {
		t.Fatalf("bad surface err=%v want %v", err, ErrInvalidTileWork)
	}
	_, err = PlanFrameTileWork(Event{Kind: EventTileGroup}, 0, parser.InterRefsPerFrame+1, 1, spans[:], jobs[:], batches[:])
	if !errors.Is(err, ErrInvalidTileWork) {
		t.Fatalf("bad references err=%v want %v", err, ErrInvalidTileWork)
	}
}

func TestEventDropsFrameWork(t *testing.T) {
	tests := []struct {
		name  string
		event Event
		want  bool
	}{
		{
			name:  "new sequence",
			event: Event{Kind: EventSequenceHeader, NewCodedVideoSequence: true},
			want:  true,
		},
		{
			name:  "temporal delimiter",
			event: Event{Kind: EventTemporalDelimiter, NewTemporalUnit: true},
			want:  true,
		},
		{
			name:  "show existing",
			event: Event{Kind: EventExistingFrame},
			want:  true,
		},
		{
			name:  "operating parameters",
			event: Event{Kind: EventSequenceHeader, OperatingParametersChanged: true},
			want:  false,
		},
		{
			name:  "tile group",
			event: Event{Kind: EventTileGroup},
			want:  false,
		},
	}
	for _, tt := range tests {
		if got := EventDropsFrameWork(tt.event); got != tt.want {
			t.Fatalf("%s: EventDropsFrameWork=%v want %v", tt.name, got, tt.want)
		}
	}
}

func TestEventCompletesFrameWork(t *testing.T) {
	tests := []struct {
		name  string
		event Event
		want  bool
	}{
		{
			name:  "final frame",
			event: Event{Kind: EventFrame, TileGroup: parser.TileGroup{Final: true}},
			want:  true,
		},
		{
			name:  "final tile group",
			event: Event{Kind: EventTileGroup, TileGroup: parser.TileGroup{Final: true}},
			want:  true,
		},
		{
			name:  "non-final tile group",
			event: Event{Kind: EventTileGroup},
			want:  false,
		},
		{
			name:  "frame header",
			event: Event{Kind: EventFrameHeader, TileGroup: parser.TileGroup{Final: true}},
			want:  false,
		},
	}
	for _, tt := range tests {
		if got := EventCompletesFrameWork(tt.event); got != tt.want {
			t.Fatalf("%s: EventCompletesFrameWork=%v want %v", tt.name, got, tt.want)
		}
	}
}

func TestEventOutputsFrame(t *testing.T) {
	tests := []struct {
		name  string
		event Event
		want  bool
	}{
		{
			name:  "final frame shown",
			event: Event{Kind: EventFrame, FrameHeader: parser.FrameHeaderPrefix{ShowFrame: true}, TileGroup: parser.TileGroup{Final: true}},
			want:  true,
		},
		{
			name:  "final frame not shown",
			event: Event{Kind: EventFrame, TileGroup: parser.TileGroup{Final: true}},
			want:  false,
		},
		{
			name:  "show existing",
			event: Event{Kind: EventExistingFrame},
			want:  true,
		},
		{
			name:  "frame header",
			event: Event{Kind: EventFrameHeader},
			want:  false,
		},
		{
			name:  "non-final tile group",
			event: Event{Kind: EventTileGroup},
			want:  false,
		},
	}
	for _, tt := range tests {
		if got := EventOutputsFrame(tt.event); got != tt.want {
			t.Fatalf("%s: EventOutputsFrame=%v want %v", tt.name, got, tt.want)
		}
	}
}

func TestFrameWorkStateLifecycle(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	plan, output, err := state.Begin(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || !state.Active() || state.Surface != plan.Surface || state.ReferenceCount != plan.ReferenceCount {
		t.Fatalf("state=%+v active=%v plan=%+v output=%p", state, state.Active(), plan, output)
	}
	if state.Sequence != threading.FrameWorkSequenceContextFromHeader(events[0].SequenceHeader) {
		t.Fatalf("state sequence=%+v want %+v", state.Sequence, threading.FrameWorkSequenceContextFromHeader(events[0].SequenceHeader))
	}
	if pool.Available() != 0 {
		t.Fatalf("available after begin=%d want 0", pool.Available())
	}

	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	tilePlan, err := state.PlanTile(events[2], 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if tilePlan.Surface != plan.Surface || tilePlan.ReferenceCount != plan.ReferenceCount ||
		tilePlan.Tile != (TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) {
		t.Fatalf("tile plan=%+v begin plan=%+v", tilePlan, plan)
	}
	if pool.Available() != 0 {
		t.Fatalf("continuation acquired a surface, available=%d", pool.Available())
	}

	var releases [parser.RefFrames]int
	releaseCount, err := state.Finish(&refs, &pool, events[2], releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if releaseCount != 0 || state.Active() {
		t.Fatalf("releaseCount=%d active=%v", releaseCount, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != plan.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, plan.Surface)
	}
}

func TestFrameWorkStatePlanEventLifecycle(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	step, output, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || !state.Active() || step.Kind != FrameWorkStepBegin ||
		step.Begin.Surface != state.Surface || step.Begin.ReferenceCount != state.ReferenceCount {
		t.Fatalf("step=%+v output=%p state=%+v active=%v", step, output, state, state.Active())
	}

	step, output, err = state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[2], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if output != nil || step.Kind != FrameWorkStepTile ||
		step.Tile.Surface != state.Surface ||
		step.Tile.Tile != (TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) {
		t.Fatalf("step=%+v output=%p state=%+v", step, output, state)
	}

	var releases [parser.RefFrames]int
	completed, releaseCount, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, events[2], releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if !completed || releaseCount != 0 || state.Active() {
		t.Fatalf("completed=%v releaseCount=%d active=%v", completed, releaseCount, state.Active())
	}
}

func TestFrameWorkStatePlanEventFrameOBU(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	step, output, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || step.Kind != FrameWorkStepBegin ||
		step.Begin.Tile != (TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) ||
		!state.Active() {
		t.Fatalf("step=%+v output=%p active=%v", step, output, state.Active())
	}
	if spans[0].Offset != len(reducedStillFrameHeaderPayload()) || jobs[0].Size != 1 || batches[0].Count != 1 {
		t.Fatalf("span=%+v job=%+v batch=%+v", spans[0], jobs[0], batches[0])
	}
}

func TestFrameWorkStatePlanEventDropsBoundary(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	step, output, err := state.PlanEvent(&refs, &pool, testSequence(), Event{Kind: EventTemporalDelimiter, NewTemporalUnit: true}, 32, nil, 1, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output != nil || step.Kind != FrameWorkStepDropped || !step.DroppedFrameWork || state.Active() || pool.Available() != 1 {
		t.Fatalf("step=%+v output=%p active=%v available=%d", step, output, state.Active(), pool.Available())
	}
	if _, err := pool.Frame(begin.Surface); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("dropped frame err=%v want %v", err, frame.ErrInvalidSlot)
	}
}

func TestFrameWorkStatePlanEventDropsBeforeBoundaryBegin(t *testing.T) {
	pool := testFramePool(t, 2)
	var refs SurfaceReferences
	var state FrameWorkState
	if _, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}

	step, output, err := state.PlanEvent(&refs, &pool, testSequence(), Event{
		Kind:                  EventFrameHeader,
		NewCodedVideoSequence: true,
		FrameHeader:           parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:             testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || step.Kind != FrameWorkStepBegin || !step.DroppedFrameWork || !state.Active() || pool.Available() != 1 {
		t.Fatalf("step=%+v output=%p active=%v available=%d", step, output, state.Active(), pool.Available())
	}
}

func TestFrameWorkStatePlanEventNewCodedVideoSequenceReleasesReferencesBeforeBegin(t *testing.T) {
	pool := testFramePool(t, 1)
	reference, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0xff, reference, releases[:]); err != nil {
		t.Fatal(err)
	}
	var state FrameWorkState

	step, output, err := state.PlanEvent(&refs, &pool, testSequence(), Event{
		Kind:                  EventFrameHeader,
		NewCodedVideoSequence: true,
		FrameHeader:           parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:             testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if output == nil ||
		step.Kind != FrameWorkStepBegin ||
		step.ReleaseCount != 1 ||
		!state.Active() ||
		pool.Available() != 0 ||
		refs.Holds(reference) {
		t.Fatalf("step=%+v output=%p active=%v available=%d holdsReference=%v", step, output, state.Active(), pool.Available(), refs.Holds(reference))
	}
}

func TestFrameWorkStatePlanEventNewCodedVideoSequenceReleasesReferencesOnIgnoredEvent(t *testing.T) {
	pool := testFramePool(t, 1)
	reference, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0xff, reference, releases[:]); err != nil {
		t.Fatal(err)
	}
	var state FrameWorkState
	event := Event{Kind: EventSequenceHeader, NewCodedVideoSequence: true}

	step, output, err := state.PlanEvent(&refs, &pool, testSequence(), event, 32, nil, 1, nil, nil, nil, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if output != nil ||
		step.Kind != FrameWorkStepIgnored ||
		step.ReleaseCount != 1 ||
		state.Active() ||
		pool.Available() != 1 ||
		refs.Holds(reference) {
		t.Fatalf("step=%+v output=%p active=%v available=%d holdsReference=%v", step, output, state.Active(), pool.Available(), refs.Holds(reference))
	}
	run, err := state.RunStep(&refs, &pool, event, step, nil, nil, nil, releases[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if run != (FrameWorkStepResult{ReleaseCount: 1}) {
		t.Fatalf("run=%+v", run)
	}
}

func TestFrameWorkStatePlanEventShowExisting(t *testing.T) {
	pool := testFramePool(t, 2)
	reference, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(1<<0, reference, releases[:]); err != nil {
		t.Fatal(err)
	}
	var state FrameWorkState
	if _, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}

	step, output, err := state.PlanEvent(&refs, &pool, testSequence(), showExistingWorkEvent(0, parser.FrameTypeInter), 32, nil, 1, nil, nil, nil, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if output != nil || step.Kind != FrameWorkStepShowExisting || !step.DroppedFrameWork ||
		step.ShowExisting != (ShowExistingFrameWorkPlan{Surface: reference, DroppedFrameWork: true}) ||
		state.Active() {
		t.Fatalf("step=%+v output=%p active=%v", step, output, state.Active())
	}
}

func TestFrameWorkStatePlanEventIgnoresNonWorkEvent(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	step, output, err := state.PlanEvent(&refs, &pool, testSequence(), Event{Kind: EventMetadata}, 32, nil, 1, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if step.Kind != FrameWorkStepIgnored || output != nil || state.Active() || pool.Available() != 1 {
		t.Fatalf("step=%+v output=%p active=%v available=%d", step, output, state.Active(), pool.Available())
	}
}

func TestFrameWorkStatePlanEventRejectsActiveBegin(t *testing.T) {
	pool := testFramePool(t, 2)
	var refs SurfaceReferences
	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = state.PlanEvent(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil, nil)
	if !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("PlanEvent err=%v want %v", err, ErrInvalidFrameWorkState)
	}
	if !state.Active() || state.Surface != begin.Surface || pool.Available() != 1 {
		t.Fatalf("state=%+v active=%v available=%d", state, state.Active(), pool.Available())
	}
}

func TestExecuteTileWorkUsesPlanRanges(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs := [2]tile.Job{
		{Tile: 3, SBCols: 1, SBRows: 1},
		{Tile: 99, SBCols: 1, SBRows: 1},
	}
	batches := [2]threading.Batch{
		{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 3, LastTile: 3, Units: 1},
		{Worker: 0, FirstJob: 1, Count: 1, FirstTile: 99, LastTile: 99, Units: 1},
	}
	plan := TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}
	var seen [2]uint16

	err = ExecuteTileWork(plan, workerPool, jobs[:], batches[:], func(batch threading.Batch, batchJobs []tile.Job) error {
		for i := range batchJobs {
			seen[batch.FirstJob+i] = batchJobs[i].Tile
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if seen != ([2]uint16{3, 0}) {
		t.Fatalf("seen=%v", seen)
	}
}

func TestExecuteFrameWorkStepExecutesBeginAndTile(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	step := FrameWorkStep{
		Kind:  FrameWorkStepBegin,
		Begin: FrameWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	var seen uint16
	executed, err := ExecuteFrameWorkStep(step, workerPool, jobs[:], batches[:], func(_ threading.Batch, batchJobs []tile.Job) error {
		for i := range batchJobs {
			seen += batchJobs[i].Tile + 1
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !executed || seen != 3 {
		t.Fatalf("begin executed=%v seen=%d", executed, seen)
	}

	step = FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	seen = 0
	executed, err = ExecuteFrameWorkStep(step, workerPool, jobs[:], batches[:], func(_ threading.Batch, batchJobs []tile.Job) error {
		for i := range batchJobs {
			seen += batchJobs[i].Tile + 1
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !executed || seen != 3 {
		t.Fatalf("tile executed=%v seen=%d", executed, seen)
	}
}

func TestExecuteFrameWorkStepNoopSteps(t *testing.T) {
	for _, step := range []FrameWorkStep{
		{Kind: FrameWorkStepIgnored},
		{Kind: FrameWorkStepDropped},
		{Kind: FrameWorkStepShowExisting},
		{Kind: FrameWorkStepBegin},
	} {
		executed, err := ExecuteFrameWorkStep(step, nil, nil, nil, nil)
		if err != nil || executed {
			t.Fatalf("step=%+v executed=%v err=%v", step, executed, err)
		}
	}
}

func TestExecuteFrameWorkStepWithContext(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	output := &frame.Frame{}
	reference := &frame.Frame{}
	references := [parser.InterRefsPerFrame]*frame.Frame{reference, &frame.Frame{}}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{
			Surface:        3,
			ReferenceCount: 1,
			Tile:           TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount},
		},
	}

	calls := 0
	executed, err := ExecuteFrameWorkStepWithContext(step, workerPool, output, references[:], jobs[:], batches[:], func(ctx FrameWorkBatch) error {
		calls++
		if ctx.Step != step {
			t.Fatalf("step=%+v want %+v", ctx.Step, step)
		}
		if ctx.Output != output {
			t.Fatalf("output=%p want %p", ctx.Output, output)
		}
		if ctx.Payload != nil {
			t.Fatalf("payload=%v want nil", ctx.Payload)
		}
		if len(ctx.References) != 1 || ctx.References[0] != reference {
			t.Fatalf("references=%v want %p", ctx.References, reference)
		}
		if ctx.Batch != batches[0] {
			t.Fatalf("batch=%+v want %+v", ctx.Batch, batches[0])
		}
		if len(ctx.Jobs) != 2 || ctx.Jobs[0].Tile != 0 || ctx.Jobs[1].Tile != 1 {
			t.Fatalf("jobs=%+v", ctx.Jobs)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !executed || calls != 1 {
		t.Fatalf("executed=%v calls=%d", executed, calls)
	}
}

func TestExecuteFrameWorkStepWithPayload(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	output := &frame.Frame{}
	payload := []byte{0xaa, 0xbb, 0xcc}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}

	executed, err := ExecuteFrameWorkStepWithPayload(step, workerPool, output, nil, payload, jobs[:], batches[:], func(ctx FrameWorkBatch) error {
		if ctx.Output != output {
			t.Fatalf("output=%p want %p", ctx.Output, output)
		}
		if len(ctx.Payload) != len(payload) || ctx.Payload[0] != payload[0] || ctx.Payload[2] != payload[2] {
			t.Fatalf("payload=%v want %v", ctx.Payload, payload)
		}
		if len(ctx.Jobs) != 2 {
			t.Fatalf("jobs=%+v", ctx.Jobs)
		}
		data, err := ctx.JobPayload(1)
		if err != nil {
			t.Fatal(err)
		}
		if len(data) != 2 || data[0] != 0xbb || data[1] != 0xcc {
			t.Fatalf("job payload=%v", data)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !executed {
		t.Fatal("not executed")
	}
}

func TestFrameWorkStateRunStepWithPayloadContextCarriesCDFUpdateMode(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	globalMotion := testFrameWorkGlobalMotion()
	filmGrain := testFrameWorkFilmGrain()
	event := Event{
		Kind:           EventTileGroup,
		SequenceHeader: testFrameWorkSequenceHeader(),
		FrameHeader: parser.FrameHeaderPrefix{
			DisableCDFUpdate: true,
			FrameType:        parser.FrameTypeKey,
		},
		FrameSize:    testFrameSize(128, 64),
		TileInfo:     testFrameWorkTileInfo(),
		Quantization: parser.QuantizationParams{BaseQIdx: 73},
		Segmentation: parser.SegmentationParams{
			QIndex:      [parser.MaxSegments]uint8{73},
			AllLossless: false,
		},
		Delta: parser.DeltaParams{
			DeltaQPresent:  true,
			DeltaQResLog2:  1,
			DeltaLFPresent: true,
			DeltaLFResLog2: 1,
		},
		LoopFilter:  parser.LoopFilterParams{LevelY: [2]uint8{4, 5}, LevelU: 6, LevelV: 7},
		CDEF:        parser.CDEFParams{Damping: 3, StrengthCount: 1, YStrength: [parser.MaxCDEFStrengths]uint8{8}},
		Restoration: parser.RestorationParams{Type: [3]parser.RestorationType{parser.RestorationWiener}, UnitSizeY: 64},
		TransformRef: parser.TransformReferenceParams{
			TransformMode: parser.TransformModeSwitchable,
			ReferenceMode: parser.ReferenceModeSelect,
		},
		SkipMode:            parser.SkipModeParams{Allowed: true, Enabled: true, RefFrameIdx: [2]uint8{1, 2}},
		FrameMode:           parser.FrameModeParams{AllowWarpedMotion: true, ReducedTxSet: true},
		GlobalMotion:        globalMotion,
		FilmGrain:           filmGrain,
		ReferenceOrderHints: [parser.InterRefsPerFrame]uint32{4, 5, 6, 7, 8, 9, 10},
	}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	payload := []byte{0x00, 0x00, 0x00}
	state := FrameWorkState{active: true}
	cdefContext := FrameWorkBatch{FrameWorkFrameContext: frameWorkFrameContext(event, threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader))}
	_, _, cdefMapLen, err := cdefContext.CDEFIndexMapShape()
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := cdefContext.BindCDEFIndexMap(make([]uint8, cdefMapLen), make([]bool, cdefMapLen))
	if err != nil {
		t.Fatal(err)
	}
	cdefMap.Read[0] = true
	if err := state.SetCDEFIndexMap(cdefMap); err != nil {
		t.Fatal(err)
	}
	_, _, lfMapLen, err := cdefContext.LoopFilterMapShape()
	if err != nil {
		t.Fatal(err)
	}
	lfMap, err := cdefContext.BindLoopFilterMap(make([]threading.FrameWorkLoopFilterBlockRecord, lfMapLen))
	if err != nil {
		t.Fatal(err)
	}
	lfMap.Records[0].Valid = true
	if err := state.SetLoopFilterMap(lfMap); err != nil {
		t.Fatal(err)
	}

	result, err := state.RunStepWithPayloadContext(nil, nil, event, step, workerPool, nil, nil, payload, jobs[:], batches[:], nil, func(ctx FrameWorkBatch) error {
		if !ctx.DisableCDFUpdate {
			t.Fatal("DisableCDFUpdate not propagated")
		}
		if ctx.CDEFIndexMap == nil {
			t.Fatal("CDEFIndexMap not propagated")
		}
		if ctx.CDEFIndexMap.Stride != cdefMap.Stride || ctx.CDEFIndexMap.Rows != cdefMap.Rows ||
			len(ctx.CDEFIndexMap.Index) != cdefMapLen || len(ctx.CDEFIndexMap.Read) != cdefMapLen {
			t.Fatalf("CDEFIndexMap=%+v len=%d/%d want stride=%d rows=%d len=%d", ctx.CDEFIndexMap, len(ctx.CDEFIndexMap.Index), len(ctx.CDEFIndexMap.Read), cdefMap.Stride, cdefMap.Rows, cdefMapLen)
		}
		if ctx.CDEFIndexMap.Read[0] || ctx.CDEFIndexMap.Index[0] != 0 {
			t.Fatalf("CDEFIndexMap was not reset before propagation: read=%v index=%d", ctx.CDEFIndexMap.Read[0], ctx.CDEFIndexMap.Index[0])
		}
		ctx.CDEFIndexMap.Read[0] = true
		if ctx.LoopFilterMap == nil {
			t.Fatal("LoopFilterMap not propagated")
		}
		if ctx.LoopFilterMap.Stride != lfMap.Stride || ctx.LoopFilterMap.Rows != lfMap.Rows ||
			len(ctx.LoopFilterMap.Records) != lfMapLen {
			t.Fatalf("LoopFilterMap=%+v len=%d want stride=%d rows=%d len=%d", ctx.LoopFilterMap, len(ctx.LoopFilterMap.Records), lfMap.Stride, lfMap.Rows, lfMapLen)
		}
		if ctx.LoopFilterMap.Records[0].Valid {
			t.Fatal("LoopFilterMap was not reset before propagation")
		}
		ctx.LoopFilterMap.Records[0].Valid = true
		if ctx.Quantization.BaseQIdx != 73 {
			t.Fatalf("BaseQIdx=%d want 73", ctx.Quantization.BaseQIdx)
		}
		if ctx.FrameHeader != event.FrameHeader {
			t.Fatalf("FrameHeader=%+v want %+v", ctx.FrameHeader, event.FrameHeader)
		}
		if ctx.Sequence != testFrameWorkSequenceContext() {
			t.Fatalf("Sequence=%+v want %+v", ctx.Sequence, testFrameWorkSequenceContext())
		}
		if ctx.FrameSize != event.FrameSize {
			t.Fatalf("FrameSize=%+v want %+v", ctx.FrameSize, event.FrameSize)
		}
		if ctx.TileInfo != event.TileInfo {
			t.Fatalf("TileInfo=%+v want %+v", ctx.TileInfo, event.TileInfo)
		}
		if ctx.Delta != event.Delta {
			t.Fatalf("Delta=%+v want %+v", ctx.Delta, event.Delta)
		}
		if ctx.Segmentation != event.Segmentation {
			t.Fatalf("Segmentation=%+v want %+v", ctx.Segmentation, event.Segmentation)
		}
		if ctx.LoopFilter != event.LoopFilter {
			t.Fatalf("LoopFilter=%+v want %+v", ctx.LoopFilter, event.LoopFilter)
		}
		if ctx.CDEF != event.CDEF {
			t.Fatalf("CDEF=%+v want %+v", ctx.CDEF, event.CDEF)
		}
		if ctx.Restoration != event.Restoration {
			t.Fatalf("Restoration=%+v want %+v", ctx.Restoration, event.Restoration)
		}
		if ctx.TransformRef != event.TransformRef {
			t.Fatalf("TransformRef=%+v want %+v", ctx.TransformRef, event.TransformRef)
		}
		if ctx.SkipMode != event.SkipMode {
			t.Fatalf("SkipMode=%+v want %+v", ctx.SkipMode, event.SkipMode)
		}
		if ctx.FrameMode != event.FrameMode {
			t.Fatalf("FrameMode=%+v want %+v", ctx.FrameMode, event.FrameMode)
		}
		if ctx.GlobalMotion != event.GlobalMotion {
			t.Fatalf("GlobalMotion=%+v want %+v", ctx.GlobalMotion, event.GlobalMotion)
		}
		if ctx.FilmGrain != event.FilmGrain {
			t.Fatalf("FilmGrain=%+v want %+v", ctx.FilmGrain, event.FilmGrain)
		}
		if ctx.ReferenceOrderHints != event.ReferenceOrderHints {
			t.Fatalf("ReferenceOrderHints=%v want %v", ctx.ReferenceOrderHints, event.ReferenceOrderHints)
		}
		r, err := ctx.JobEntropyReader(1)
		if err != nil {
			t.Fatal(err)
		}
		if r.AllowCDFUpdate() {
			t.Fatal("CDF update enabled")
		}
		bit, err := r.ReadBit()
		if err != nil {
			t.Fatal(err)
		}
		if bit != 0 {
			t.Fatalf("bit=%d want 0", bit)
		}
		var state tile.DecodeState
		if err := ctx.JobDecodeState(1, &state); err != nil {
			t.Fatal(err)
		}
		if state.CurrentBaseQIdx != 73 {
			t.Fatalf("CurrentBaseQIdx=%d want 73", state.CurrentBaseQIdx)
		}
		if state.Reader.AllowCDFUpdate() {
			t.Fatal("decode state CDF update enabled")
		}
		var qCDF entropy.CDF
		if err := qCDF.InitDefaultDelta(); err != nil {
			t.Fatal(err)
		}
		var lfCDF entropy.CDF
		if err := lfCDF.InitDefaultDelta(); err != nil {
			t.Fatal(err)
		}
		err = state.ReadBlockDeltas(ctx.Delta, tile.BlockDeltaContext{SBSizeMIB: 16}, tile.DeltaCDFs{Q: &qCDF, LF: &lfCDF})
		if err != nil {
			t.Fatal(err)
		}
		if state.CurrentBaseQIdx != 73 || state.DeltaLFFromBase != 0 {
			t.Fatalf("delta state q=%d lf=%d", state.CurrentBaseQIdx, state.DeltaLFFromBase)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.ExecutedTileWork {
		t.Fatal("not executed")
	}
	if !state.cdefIndexMap.Read[0] {
		t.Fatal("CDEFIndexMap update did not alias frame state")
	}
	if !state.loopFilterMap.Records[0].Valid {
		t.Fatal("LoopFilterMap update did not alias frame state")
	}
}

func TestFrameWorkStateCarriesTileResidualCDFsThroughReferenceSlots(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePool(t, 2)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int
	jobs := []tile.Job{{Tile: 0, Offset: 0, Size: 1, UpdatesFrameContext: true}}
	batches := []threading.Batch{{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0, Units: 1}}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}},
	}

	key := Event{
		Kind: EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{
			FrameType:       parser.FrameTypeKey,
			PrimaryRefFrame: parser.PrimaryRefNone,
		},
		FrameSize:    testFrameSize(16, 16),
		TileInfo:     parser.TileInfo{RefreshContext: true, ContextUpdateTileID: 0},
		Quantization: parser.QuantizationParams{BaseQIdx: 64},
	}
	plan, _, err := state.Begin(&refs, &pool, testSequence(), key, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	step.Tile.Surface = plan.Surface

	var retainedDeltaQ []uint16
	finalKey := key
	finalKey.Kind = EventTileGroup
	finalKey.Unit.Payload = []byte{0x00}
	finalKey.FrameSize.RefreshFrameFlags = 0xff
	finalKey.TileGroup = parser.TileGroup{Final: true}
	_, err = state.RunStepWithPayloadContext(&refs, &pool, finalKey, step, workerPool, nil, nil, finalKey.Unit.Payload, jobs, batches, releases[:], func(ctx FrameWorkBatch) error {
		var storage threading.FrameWorkTileResidualCDFStorage
		if err := ctx.InitTileResidualCDFStorage(&storage); err != nil {
			return err
		}
		if err := storage.DeltaQ.Update(2); err != nil {
			return err
		}
		retainedDeltaQ = append(retainedDeltaQ[:0], storage.DeltaQ.Values()...)
		retainedDeltaQ[len(retainedDeltaQ)-1] = 0
		var decodeState tile.DecodeState
		if err := ctx.JobDecodeState(0, &decodeState); err != nil {
			return err
		}
		return ctx.RetainTileResidualCDFStorage(0, &decodeState, &storage)
	})
	if err != nil {
		t.Fatal(err)
	}

	inter := Event{
		Kind: EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{
			FrameType:       parser.FrameTypeInter,
			PrimaryRefFrame: 0,
		},
		FrameSize:    testFrameSize(16, 16),
		TileInfo:     parser.TileInfo{RefreshContext: true, ContextUpdateTileID: 0},
		Quantization: parser.QuantizationParams{BaseQIdx: 64},
	}
	for i := range parser.InterRefsPerFrame {
		inter.FrameSize.RefFrameIdx[i] = 0
	}
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	plan, _, err = state.Begin(&refs, &pool, testSequence(), inter, 32, referenceSurfaces[:], 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if plan.ReferenceCount != 0 {
		count, err := ResolveFrameReferences(&pool, referenceSurfaces[:plan.ReferenceCount], referenceFrames[:])
		if err != nil {
			t.Fatal(err)
		}
		if count != plan.ReferenceCount {
			t.Fatalf("reference count=%d want %d", count, plan.ReferenceCount)
		}
	}
	step.Tile.Surface = plan.Surface
	step.Tile.ReferenceCount = plan.ReferenceCount
	finalInter := inter
	finalInter.Kind = EventTileGroup
	finalInter.Unit.Payload = []byte{0x00}
	finalInter.FrameSize.RefreshFrameFlags = 0x01
	finalInter.TileGroup = parser.TileGroup{Final: true}
	_, err = state.RunStepWithPayloadContext(&refs, &pool, finalInter, step, workerPool, nil, referenceFrames[:plan.ReferenceCount], finalInter.Unit.Payload, jobs, batches, releases[:], func(ctx FrameWorkBatch) error {
		var storage threading.FrameWorkTileResidualCDFStorage
		if err := ctx.InitTileResidualCDFStorage(&storage); err != nil {
			return err
		}
		if !testUint16sEqual(storage.DeltaQ.Values(), retainedDeltaQ) {
			return fmt.Errorf("delta q=%v want retained %v", storage.DeltaQ.Values(), retainedDeltaQ)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestExecuteFrameWorkStepWithPayloadRejectsInvalidPayloadRange(t *testing.T) {
	jobs, batches, batchCount := testExecutionWork(t)
	jobs[1].Offset = 2
	jobs[1].Size = 2
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}

	_, err := ExecuteFrameWorkStepWithPayload(step, nil, nil, nil, []byte{0xaa, 0xbb, 0xcc}, jobs[:], batches[:], func(FrameWorkBatch) error {
		t.Fatal("callback should not run")
		return nil
	})
	if !errors.Is(err, tile.ErrInvalidPlan) {
		t.Fatalf("ExecuteFrameWorkStepWithPayload err=%v want %v", err, tile.ErrInvalidPlan)
	}
}

func TestExecuteFrameWorkStepWithContextNoopSteps(t *testing.T) {
	for _, step := range []FrameWorkStep{
		{Kind: FrameWorkStepIgnored},
		{Kind: FrameWorkStepDropped},
		{Kind: FrameWorkStepShowExisting},
		{Kind: FrameWorkStepBegin},
	} {
		executed, err := ExecuteFrameWorkStepWithContext(step, nil, nil, nil, nil, nil, nil)
		if err != nil || executed {
			t.Fatalf("step=%+v executed=%v err=%v", step, executed, err)
		}
	}
}

func TestExecuteFrameWorkStepWithContextRejectsNilCallback(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	_, err = ExecuteFrameWorkStepWithContext(step, workerPool, nil, nil, jobs[:], batches[:], nil)
	if !errors.Is(err, threading.ErrInvalidCallback) {
		t.Fatalf("ExecuteFrameWorkStepWithContext err=%v want %v", err, threading.ErrInvalidCallback)
	}
}

func TestExecuteFrameWorkStepWithContextRejectsShortReferences(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{
			ReferenceCount: 1,
			Tile:           TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount},
		},
	}
	_, err = ExecuteFrameWorkStepWithContext(step, workerPool, nil, nil, jobs[:], batches[:], func(FrameWorkBatch) error {
		return nil
	})
	if !errors.Is(err, ErrSurfaceReferenceBufferTooSmall) {
		t.Fatalf("ExecuteFrameWorkStepWithContext err=%v want %v", err, ErrSurfaceReferenceBufferTooSmall)
	}
}

func TestExecuteTileWorkRejectsInvalidPlan(t *testing.T) {
	jobs, batches, batchCount := testExecutionWork(t)
	tests := []TileWorkPlan{
		{SpanCount: -1},
		{SpanCount: 1, JobCount: 0},
		{SpanCount: 1, JobCount: 1, BatchCount: 0},
		{SpanCount: 1, JobCount: 1, BatchCount: 2},
		{SpanCount: 3, JobCount: 3, BatchCount: batchCount},
		{SpanCount: 2, JobCount: 2, BatchCount: batchCount + 1},
	}
	for _, plan := range tests {
		err := ExecuteTileWork(plan, nil, jobs[:], batches[:batchCount], nil)
		if !errors.Is(err, ErrInvalidTileWork) {
			t.Fatalf("plan=%+v err=%v want %v", plan, err, ErrInvalidTileWork)
		}
	}
}

func TestExecuteFrameWorkStepRejectsInvalidKind(t *testing.T) {
	_, err := ExecuteFrameWorkStep(FrameWorkStep{Kind: FrameWorkStepKind(99)}, nil, nil, nil, nil)
	if !errors.Is(err, ErrInvalidTileWork) {
		t.Fatalf("ExecuteFrameWorkStep err=%v want %v", err, ErrInvalidTileWork)
	}
}

func TestExecuteTileWorkPropagatesCallbackError(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	want := errors.New("tile callback")
	err = ExecuteTileWork(TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}, workerPool, jobs[:], batches[:], func(threading.Batch, []tile.Job) error {
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("ExecuteTileWork err=%v want %v", err, want)
	}
}

func TestFrameWorkStateRunStepFrameOBUExecutesThenFinishes(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	step, output, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || step.Kind != FrameWorkStepBegin {
		t.Fatalf("step=%+v output=%p", step, output)
	}

	var ranWhileActive bool
	var releases [parser.RefFrames]int
	result, err := state.RunStep(&refs, &pool, events[1], step, workerPool, jobs[:], batches[:], releases[:], func(batch threading.Batch, batchJobs []tile.Job) error {
		ranWhileActive = state.Active() && batch.Count == 1 && len(batchJobs) == 1
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) || !ranWhileActive || state.Active() {
		t.Fatalf("result=%+v ranWhileActive=%v active=%v", result, ranWhileActive, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, step.Begin.Surface)
	}
}

func TestFrameWorkStateRunStepWithContextFrameOBUExecutesThenFinishes(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	step, output, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || step.Kind != FrameWorkStepBegin {
		t.Fatalf("step=%+v output=%p", step, output)
	}

	var ranWhileActive bool
	var releases [parser.RefFrames]int
	result, err := state.RunStepWithContext(&refs, &pool, events[1], step, workerPool, output, nil, jobs[:], batches[:], releases[:], func(ctx FrameWorkBatch) error {
		ranWhileActive = state.Active() &&
			ctx.Step == step &&
			ctx.Output == output &&
			len(ctx.References) == 0 &&
			ctx.Batch.Count == 1 &&
			len(ctx.Jobs) == 1
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) || !ranWhileActive || state.Active() {
		t.Fatalf("result=%+v ranWhileActive=%v active=%v", result, ranWhileActive, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, step.Begin.Surface)
	}
}

func TestFrameWorkStateRunStepTileGroupExecutesThenFinishes(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	begin, _, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	result, err := state.RunStep(&refs, &pool, events[1], begin, workerPool, jobs[:], batches[:], nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkStepResult{}) || !state.Active() {
		t.Fatalf("begin result=%+v active=%v", result, state.Active())
	}

	step, _, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[2], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	var releases [parser.RefFrames]int
	result, err = state.RunStep(&refs, &pool, events[2], step, workerPool, jobs[:], batches[:], releases[:], func(threading.Batch, []tile.Job) error {
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) || state.Active() {
		t.Fatalf("tile result=%+v active=%v", result, state.Active())
	}
}

func TestFrameWorkStateRunStepKeepsActiveOnCallbackError(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	begin, _, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	step, _, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[2], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}

	want := errors.New("tile decode")
	var releases [parser.RefFrames]int
	_, err = state.RunStep(&refs, &pool, events[2], step, workerPool, jobs[:], batches[:], releases[:], func(threading.Batch, []tile.Job) error {
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("RunStep err=%v want %v", err, want)
	}
	if !state.Active() || state.Surface != begin.Begin.Surface {
		t.Fatalf("state=%+v active=%v begin=%+v", state, state.Active(), begin)
	}
	if slot, ok := refs.ReferenceSlot(0); ok || slot != -1 {
		t.Fatalf("slot=%d ok=%v want no publication", slot, ok)
	}
}

func TestFrameWorkStateRunStepWithContextKeepsActiveOnCallbackError(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	begin, output, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	step, _, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[2], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}

	want := errors.New("tile decode")
	var releases [parser.RefFrames]int
	_, err = state.RunStepWithContext(&refs, &pool, events[2], step, workerPool, output, nil, jobs[:], batches[:], releases[:], func(ctx FrameWorkBatch) error {
		if ctx.Output != output {
			t.Fatalf("output=%p want %p", ctx.Output, output)
		}
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("RunStepWithContext err=%v want %v", err, want)
	}
	if !state.Active() || state.Surface != begin.Begin.Surface {
		t.Fatalf("state=%+v active=%v begin=%+v", state, state.Active(), begin)
	}
	if slot, ok := refs.ReferenceSlot(0); ok || slot != -1 {
		t.Fatalf("slot=%d ok=%v want no publication", slot, ok)
	}
}

func TestFrameWorkStateRunEventWithContextFrameOBU(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	wantSequence := threading.FrameWorkSequenceContextFromHeader(events[0].SequenceHeader)
	var seenOutput *frame.Frame
	result, err := state.RunEventWithContext(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx FrameWorkBatch) error {
		seenOutput = ctx.Output
		if ctx.Step.Kind != FrameWorkStepBegin ||
			len(ctx.References) != 0 ||
			len(ctx.Jobs) != 1 ||
			len(ctx.Payload) != len(events[1].Unit.Payload) ||
			ctx.Payload[ctx.Jobs[0].Offset] != 0xaa ||
			ctx.Sequence != wantSequence ||
			ctx.FrameHeader != events[1].FrameHeader ||
			ctx.FrameSize != events[1].FrameSize ||
			ctx.TileInfo != events[1].TileInfo ||
			ctx.Segmentation != events[1].Segmentation ||
			ctx.LoopFilter != events[1].LoopFilter ||
			ctx.CDEF != events[1].CDEF ||
			ctx.Restoration != events[1].Restoration ||
			ctx.TransformRef != events[1].TransformRef ||
			ctx.SkipMode != events[1].SkipMode ||
			ctx.FrameMode != events[1].FrameMode ||
			ctx.GlobalMotion != events[1].GlobalMotion ||
			ctx.FilmGrain != events[1].FilmGrain {
			t.Fatalf("ctx=%+v", ctx)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Step.Kind != FrameWorkStepBegin ||
		result.Output == nil ||
		result.Output != seenOutput ||
		result.ReferenceCount != 0 ||
		result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		state.Active() {
		t.Fatalf("result=%+v seen=%p active=%v", result, seenOutput, state.Active())
	}
	if slot, ok := refs.ReferenceSlot(0); !ok || slot != result.Step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, result.Step.Begin.Surface)
	}
}

func TestFrameWorkStateRunEventWithContextRunnerFrameOBU(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	runner := frameWorkContextRunner{
		wantKind:     FrameWorkStepBegin,
		wantSequence: threading.FrameWorkSequenceContextFromHeader(events[0].SequenceHeader),
		wantEvent:    events[1],
		wantPayload:  0xaa,
	}
	result, err := state.RunEventWithContextRunner(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &runner)
	if err != nil {
		t.Fatal(err)
	}
	if runner.err != nil {
		t.Fatal(runner.err)
	}
	if result.Step.Kind != FrameWorkStepBegin ||
		result.Output == nil ||
		result.Output != runner.output ||
		result.ReferenceCount != 0 ||
		result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		state.Active() {
		t.Fatalf("result=%+v seen=%p active=%v", result, runner.output, state.Active())
	}
	if slot, ok := refs.ReferenceSlot(0); !ok || slot != result.Step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, result.Step.Begin.Surface)
	}
}

func TestFrameWorkStateRunEventWithResidualRunnerFrameOBU(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayloadQ(64)...)
	framePayload = append(framePayload, make([]byte, 256)...)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testStillSequenceHeaderPayload(64, 64))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int
	var runner threading.FrameWorkTileResidualRunner
	runner.Workers = []threading.FrameWorkTileResidualRunnerWorker{
		{
			Int32Scratch:    make([]int32, 32768),
			ResidualScratch: make([]int16, 4096),
		},
	}

	result, err := state.RunEventWithContextRunner(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &runner)
	if err != nil {
		t.Fatal(err)
	}
	if result.Step.Kind != FrameWorkStepBegin ||
		result.Output == nil ||
		result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
	if stats := runner.Workers[0].Stats; stats.Loop.Blocks == 0 || stats.CoefficientBlocks == 0 || stats.TXBs == 0 {
		t.Fatalf("residual runner stats=%+v", stats)
	}
	if slot, ok := refs.ReferenceSlot(0); !ok || slot != result.Step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, result.Step.Begin.Surface)
	}
}

func TestFrameWorkBoundSideDataRunnerBindsActiveMaps(t *testing.T) {
	seq := testSequence()
	event := Event{
		Kind:           EventFrameHeader,
		SequenceHeader: seq,
		FrameHeader: parser.FrameHeaderPrefix{
			FrameType:       parser.FrameTypeKey,
			PrimaryRefFrame: parser.PrimaryRefNone,
		},
		FrameSize: parser.FrameSize{
			CodedWidth:          64,
			UpscaledWidth:       64,
			Height:              64,
			SuperResDenominator: 8,
		},
		LoopFilter: parser.LoopFilterParams{
			LevelY: [2]uint8{4},
		},
		CDEF: parser.CDEFParams{
			Bits:          1,
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{8},
		},
		Restoration: parser.RestorationParams{
			Type:      [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
			UnitSizeY: 64,
		},
	}
	pool := testFramePoolForSize(t, event.FrameSize.CodedWidth, event.FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	plan, output, err := state.Begin(&refs, &pool, seq, event, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx := FrameWorkBatch{
		Step:                  FrameWorkStep{Kind: FrameWorkStepBegin, Begin: plan},
		Output:                output,
		FrameWorkFrameContext: frameWorkFrameContext(event, threading.FrameWorkSequenceContextFromHeader(seq)),
	}
	_, _, cdefLen, err := ctx.CDEFIndexMapShape()
	if err != nil {
		t.Fatal(err)
	}
	_, _, lfLen, err := ctx.LoopFilterMapShape()
	if err != nil {
		t.Fatal(err)
	}
	restorationPlan, err := ctx.RestorationFramePlan()
	if err != nil {
		t.Fatal(err)
	}
	size, err := FrameWorkSideDataScratchLen(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if size.CDEF != cdefLen ||
		size.LoopFilterRecords != lfLen ||
		size.RestorationRecords != restorationPlan.UnitRecordLen() ||
		size.RestorationBoundary != restorationPlan.BoundaryBufferLen() {
		t.Fatalf("size=%+v cdef=%d lf=%d restoration=%+v", size, cdefLen, lfLen, restorationPlan)
	}
	runner, err := size.BindRunner(FrameWorkSideDataScratch{
		CDEFIndex:          make([]uint8, size.CDEF),
		CDEFRead:           make([]bool, size.CDEF),
		LoopFilterRecords:  make([]threading.FrameWorkLoopFilterBlockRecord, size.LoopFilterRecords),
		RestorationRecords: make([]tile.RestorationUnitRecord, size.RestorationRecords),
		RestorationAbove:   make([]uint16, size.RestorationBoundary),
		RestorationBelow:   make([]uint16, size.RestorationBoundary),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.BindFrameWorkSideData(&state, ctx); err != nil {
		t.Fatal(err)
	}
	cdefMap, lfMap, restorationBuffers := state.postFilterSideData()
	if cdefMap == nil || cdefMap.Stride != runner.CDEFIndexMap.Stride || len(cdefMap.Index) != cdefLen {
		t.Fatalf("cdefMap=%+v runner=%+v len=%d", cdefMap, runner.CDEFIndexMap, cdefLen)
	}
	if lfMap == nil || lfMap.Stride != runner.LoopFilterMap.Stride || len(lfMap.Records) != lfLen {
		t.Fatalf("lfMap=%+v runner=%+v len=%d", lfMap, runner.LoopFilterMap, lfLen)
	}
	if restorationBuffers == nil || restorationBuffers.Plan != runner.RestorationFrameBuffers.Plan ||
		len(restorationBuffers.Records[0]) != restorationPlan.UnitRecords[0] {
		t.Fatalf("restorationBuffers=%+v runner=%+v plan=%+v", restorationBuffers, runner.RestorationFrameBuffers, restorationPlan)
	}
}

func TestFrameWorkBoundSideDataRunnerIgnoresInactiveReusableScratch(t *testing.T) {
	seq := testSequence()
	event := Event{
		Kind:           EventFrameHeader,
		SequenceHeader: seq,
		FrameHeader: parser.FrameHeaderPrefix{
			FrameType:       parser.FrameTypeKey,
			PrimaryRefFrame: parser.PrimaryRefNone,
		},
		FrameSize: parser.FrameSize{
			CodedWidth:          64,
			UpscaledWidth:       64,
			Height:              64,
			SuperResDenominator: 8,
		},
	}
	pool := testFramePoolForSize(t, event.FrameSize.CodedWidth, event.FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	plan, output, err := state.Begin(&refs, &pool, seq, event, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx := FrameWorkBatch{
		Step:                  FrameWorkStep{Kind: FrameWorkStepBegin, Begin: plan},
		Output:                output,
		FrameWorkFrameContext: frameWorkFrameContext(event, threading.FrameWorkSequenceContextFromHeader(seq)),
	}
	size, err := FrameWorkSideDataScratchLen(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if size != (FrameWorkSideDataScratchSize{}) {
		t.Fatalf("inactive size=%+v want zero", size)
	}

	runner := FrameWorkBoundSideDataRunner{
		CDEFIndex:          make([]uint8, 4),
		CDEFRead:           make([]bool, 4),
		LoopFilterRecords:  make([]threading.FrameWorkLoopFilterBlockRecord, 256),
		RestorationRecords: make([]tile.RestorationUnitRecord, 16),
		RestorationAbove:   make([]uint16, 256),
		RestorationBelow:   make([]uint16, 256),
	}
	if err := runner.BindFrameWorkSideData(&state, ctx); err != nil {
		t.Fatal(err)
	}
	cdefMap, lfMap, restorationBuffers := state.postFilterSideData()
	if cdefMap != nil || lfMap != nil || restorationBuffers != nil {
		t.Fatalf("inactive side data cdef=%+v lf=%+v restoration=%+v", cdefMap, lfMap, restorationBuffers)
	}
	if runner.CDEFIndexMap.Stride != 0 || len(runner.CDEFIndexMap.Index) != 0 ||
		runner.LoopFilterMap.Stride != 0 || len(runner.LoopFilterMap.Records) != 0 ||
		runner.RestorationFrameBuffers.Plan.Planes != 0 {
		t.Fatalf("runner retained inactive side data: %+v", runner)
	}
}

func TestFrameWorkSideDataScratchSizeBindRunnerAllocs(t *testing.T) {
	seq := testSequence()
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: frameWorkFrameContext(Event{
			SequenceHeader: seq,
			FrameSize: parser.FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              64,
				SuperResDenominator: 8,
			},
			LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{4}},
			CDEF: parser.CDEFParams{
				Bits:          1,
				StrengthCount: 1,
				YStrength:     [parser.MaxCDEFStrengths]uint8{8},
			},
			Restoration: parser.RestorationParams{
				Type:      [3]parser.RestorationType{parser.RestorationWiener},
				UnitSizeY: 64,
			},
		}, threading.FrameWorkSequenceContextFromHeader(seq)),
	}
	size, err := FrameWorkSideDataScratchLen(ctx)
	if err != nil {
		t.Fatal(err)
	}
	scratch := FrameWorkSideDataScratch{
		CDEFIndex:          make([]uint8, size.CDEF),
		CDEFRead:           make([]bool, size.CDEF),
		LoopFilterRecords:  make([]threading.FrameWorkLoopFilterBlockRecord, size.LoopFilterRecords),
		RestorationRecords: make([]tile.RestorationUnitRecord, size.RestorationRecords),
		RestorationAbove:   make([]uint16, size.RestorationBoundary),
		RestorationBelow:   make([]uint16, size.RestorationBoundary),
	}
	allocs := testing.AllocsPerRun(1000, func() {
		runner, err := size.BindRunner(scratch)
		if err != nil {
			t.Fatal(err)
		}
		if len(runner.CDEFIndex) != size.CDEF ||
			len(runner.LoopFilterRecords) != size.LoopFilterRecords ||
			len(runner.RestorationRecords) != size.RestorationRecords {
			t.Fatalf("runner=%+v size=%+v", runner, size)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkSideDataScratchSize.BindRunner allocated: %f", allocs)
	}
}

func TestFrameWorkSideDataScratchSizeMaxAndBindRunnerErrors(t *testing.T) {
	size := FrameWorkSideDataScratchSize{
		CDEF:                2,
		LoopFilterRecords:   4,
		RestorationRecords:  6,
		RestorationBoundary: 8,
	}
	other := FrameWorkSideDataScratchSize{
		CDEF:                3,
		LoopFilterRecords:   1,
		RestorationRecords:  7,
		RestorationBoundary: 5,
	}
	if got := size.Max(other); got != (FrameWorkSideDataScratchSize{
		CDEF:                3,
		LoopFilterRecords:   4,
		RestorationRecords:  7,
		RestorationBoundary: 8,
	}) {
		t.Fatalf("Max=%+v", got)
	}

	scratch := FrameWorkSideDataScratch{
		CDEFIndex:          make([]uint8, size.CDEF),
		CDEFRead:           make([]bool, size.CDEF),
		LoopFilterRecords:  make([]threading.FrameWorkLoopFilterBlockRecord, size.LoopFilterRecords),
		RestorationRecords: make([]tile.RestorationUnitRecord, size.RestorationRecords),
		RestorationAbove:   make([]uint16, size.RestorationBoundary),
		RestorationBelow:   make([]uint16, size.RestorationBoundary),
	}
	tests := []struct {
		name   string
		size   FrameWorkSideDataScratchSize
		mutate func(*FrameWorkSideDataScratch)
	}{
		{name: "negative cdef", size: FrameWorkSideDataScratchSize{CDEF: -1}},
		{name: "short cdef index", size: size, mutate: func(s *FrameWorkSideDataScratch) { s.CDEFIndex = s.CDEFIndex[:len(s.CDEFIndex)-1] }},
		{name: "short cdef read", size: size, mutate: func(s *FrameWorkSideDataScratch) { s.CDEFRead = s.CDEFRead[:len(s.CDEFRead)-1] }},
		{name: "short loop map", size: size, mutate: func(s *FrameWorkSideDataScratch) {
			s.LoopFilterRecords = s.LoopFilterRecords[:len(s.LoopFilterRecords)-1]
		}},
		{name: "short restoration records", size: size, mutate: func(s *FrameWorkSideDataScratch) {
			s.RestorationRecords = s.RestorationRecords[:len(s.RestorationRecords)-1]
		}},
		{name: "short restoration above", size: size, mutate: func(s *FrameWorkSideDataScratch) { s.RestorationAbove = s.RestorationAbove[:len(s.RestorationAbove)-1] }},
		{name: "short restoration below", size: size, mutate: func(s *FrameWorkSideDataScratch) { s.RestorationBelow = s.RestorationBelow[:len(s.RestorationBelow)-1] }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			short := scratch
			if tt.mutate != nil {
				tt.mutate(&short)
			}
			if _, err := tt.size.BindRunner(short); !errors.Is(err, frame.ErrShortBuffer) {
				t.Fatalf("BindRunner err=%v want %v", err, frame.ErrShortBuffer)
			}
		})
	}
}

func TestFrameWorkStateRunEventWithResidualRunnerSideDataPostFilter(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayloadQ(64)...)
	framePayload = append(framePayload, make([]byte, 256)...)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testStillSequenceHeaderPayload(64, 64))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}
	events[1].LoopFilter = parser.LoopFilterParams{LevelY: [2]uint8{4}}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int
	var runner threading.FrameWorkTileResidualRunner
	runner.Workers = []threading.FrameWorkTileResidualRunnerWorker{
		{
			Int32Scratch:    make([]int32, 32768),
			ResidualScratch: make([]int16, 4096),
		},
	}
	side := FrameWorkBoundSideDataRunner{
		LoopFilterRecords: make([]threading.FrameWorkLoopFilterBlockRecord, 256),
	}
	post := FrameWorkBoundSupportedPostFilterRunner{
		Scratch: FrameWorkPostFilterScratch{
			LoopFilterEdges: make([]FrameWorkLoopFilterPostFilterEdge, 256),
		},
	}

	result, err := state.RunEventWithContextAndSideDataAndPostFilterRunners(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &side, &runner, &post)
	if err != nil {
		t.Fatal(err)
	}
	if side.LoopFilterMap.Stride == 0 {
		t.Fatalf("side data runner did not bind loop-filter map: %+v", side)
	}
	if result.Step.Kind != FrameWorkStepBegin ||
		result.Output == nil ||
		result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
	if stats := runner.Workers[0].Stats; stats.Loop.Blocks == 0 || stats.CoefficientBlocks == 0 || stats.TXBs == 0 {
		t.Fatalf("residual runner stats=%+v", stats)
	}
	coverage, err := side.LoopFilterMap.CoverageStats(side.LoopFilterMap.Stride, side.LoopFilterMap.Rows)
	if err != nil {
		t.Fatal(err)
	}
	if coverage.Blocks == 0 || coverage.Missing != 0 {
		t.Fatalf("loop-filter coverage=%+v", coverage)
	}
	if post.Result.Completed != FrameWorkPostFilterLoopFilter ||
		!post.Result.LoopFilter.Active ||
		post.Result.LoopFilter.Plan.Blocks == 0 ||
		post.Context.RemainingPostFilters() != 0 {
		t.Fatalf("postfilter result=%+v size=%+v remaining=%v", post.Result, post.Size, post.Context.RemainingPostFilters())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != result.Step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, result.Step.Begin.Surface)
	}
}

func TestFrameWorkStateRunEventWithContextTileGroup(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	begin, err := state.RunEventWithContext(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, nil)
	if err != nil {
		t.Fatal(err)
	}
	if begin.Step.Kind != FrameWorkStepBegin || begin.Output == nil || begin.Run != (FrameWorkStepResult{}) || !state.Active() {
		t.Fatalf("begin=%+v active=%v", begin, state.Active())
	}

	wantSequence := threading.FrameWorkSequenceContextFromHeader(events[0].SequenceHeader)
	events[2].SequenceHeader = parser.SequenceHeader{}
	var seenOutput *frame.Frame
	tileResult, err := state.RunEventWithContext(&refs, &pool, events[0].SequenceHeader, events[2], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx FrameWorkBatch) error {
		seenOutput = ctx.Output
		if ctx.Step.Kind != FrameWorkStepTile ||
			len(ctx.References) != 0 ||
			len(ctx.Jobs) != 1 ||
			len(ctx.Payload) != len(events[2].Unit.Payload) ||
			ctx.Sequence != wantSequence ||
			ctx.Payload[ctx.Jobs[0].Offset] != 0x80 {
			t.Fatalf("ctx=%+v", ctx)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if tileResult.Step.Kind != FrameWorkStepTile ||
		tileResult.Output != begin.Output ||
		seenOutput != begin.Output ||
		tileResult.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		state.Active() {
		t.Fatalf("tile=%+v seen=%p begin=%p active=%v", tileResult, seenOutput, begin.Output, state.Active())
	}
}

func TestFrameWorkStateRunEventWithContextRunnerTileGroup(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	begin, err := state.RunEventWithContextRunner(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, nil)
	if err != nil {
		t.Fatal(err)
	}
	if begin.Step.Kind != FrameWorkStepBegin || begin.Output == nil || begin.Run != (FrameWorkStepResult{}) || !state.Active() {
		t.Fatalf("begin=%+v active=%v", begin, state.Active())
	}

	events[2].SequenceHeader = parser.SequenceHeader{}
	runner := frameWorkContextRunner{
		wantKind:     FrameWorkStepTile,
		wantSequence: threading.FrameWorkSequenceContextFromHeader(events[0].SequenceHeader),
		wantEvent:    events[2],
		wantPayload:  0x80,
	}
	tileResult, err := state.RunEventWithContextRunner(&refs, &pool, events[0].SequenceHeader, events[2], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &runner)
	if err != nil {
		t.Fatal(err)
	}
	if runner.err != nil {
		t.Fatal(runner.err)
	}
	if tileResult.Step.Kind != FrameWorkStepTile ||
		tileResult.Output != begin.Output ||
		runner.output != begin.Output ||
		tileResult.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		state.Active() {
		t.Fatalf("tile=%+v seen=%p begin=%p active=%v", tileResult, runner.output, begin.Output, state.Active())
	}
}

func TestFrameWorkStateRunEventWithContextPostFilterBeforeFinish(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	var order [2]string
	orderIndex := 0
	result, err := state.RunEventWithContextAndPostFilter(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx FrameWorkBatch) error {
		order[orderIndex] = "tile"
		orderIndex++
		if !state.Active() {
			t.Fatal("state inactive during tile callback")
		}
		ctx.Output.Y.Pix[0] = 0x11
		return nil
	}, func(ctx FrameWorkPostFilterContext) error {
		order[orderIndex] = "post"
		orderIndex++
		if !state.Active() {
			t.Fatal("state inactive during postfilter")
		}
		if slot, ok := refs.ReferenceSlot(0); ok || slot != -1 {
			t.Fatalf("published before postfilter slot=%d ok=%v", slot, ok)
		}
		if ctx.Output.Y.Pix[0] != 0x11 {
			t.Fatalf("postfilter saw sample=%d want 0x11", ctx.Output.Y.Pix[0])
		}
		if ctx.Event.Kind != EventFrame || ctx.Step.Kind != FrameWorkStepBegin || !ctx.ExecutedTileWork || ctx.ReferenceCount != 0 {
			t.Fatalf("postfilter context=%+v", ctx)
		}
		ctx.Output.Y.Pix[0] = 0x33
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if orderIndex != 2 || order != ([2]string{"tile", "post"}) {
		t.Fatalf("order=%v", order)
	}
	if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) || state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != result.Step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, result.Step.Begin.Surface)
	}
	published, err := pool.Frame(slot)
	if err != nil {
		t.Fatal(err)
	}
	if published.Y.Pix[0] != 0x33 {
		t.Fatalf("published sample=%d want 0x33", published.Y.Pix[0])
	}
}

func TestFrameWorkStateRunEventWithContextRunnersPostFilterBeforeFinish(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	var order [2]string
	runner := frameWorkWritingRunner{
		order: &order,
		state: &state,
		value: 0x11,
	}
	post := frameWorkCheckingPostRunner{
		order: &order,
		refs:  &refs,
		state: &state,
		value: 0x33,
	}
	result, err := state.RunEventWithContextAndPostFilterRunners(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &runner, &post)
	if err != nil {
		t.Fatal(err)
	}
	if runner.err != nil {
		t.Fatal(runner.err)
	}
	if post.err != nil {
		t.Fatal(post.err)
	}
	if order != ([2]string{"tile", "post"}) {
		t.Fatalf("order=%v", order)
	}
	if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) || state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != result.Step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, result.Step.Begin.Surface)
	}
	published, err := pool.Frame(slot)
	if err != nil {
		t.Fatal(err)
	}
	if published.Y.Pix[0] != 0x33 {
		t.Fatalf("published sample=%d want 0x33", published.Y.Pix[0])
	}
}

func TestFrameWorkStateRunStepWithPostFilterCarriesSideMaps(t *testing.T) {
	pool := testFramePoolForSize(t, 64, 64, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int

	seq := testSequence()
	begin := Event{
		Kind:           EventFrameHeader,
		SequenceHeader: seq,
		FrameHeader:    parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:      testFrameSize(64, 64),
	}
	plan, _, err := state.Begin(&refs, &pool, seq, begin, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	final := begin
	final.Kind = EventTileGroup
	final.TileGroup.Final = true
	final.FrameSize.RefreshFrameFlags = 1
	final.FrameSize.UpscaledWidth = 64
	final.FrameSize.SuperResDenominator = 8
	final.CDEF = parser.CDEFParams{Damping: 5, StrengthCount: 1, YStrength: [parser.MaxCDEFStrengths]uint8{8}}
	final.LoopFilter = parser.LoopFilterParams{LevelY: [2]uint8{1}}
	final.Restoration = parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener},
		UnitSizeY: 64,
	}
	ctx := FrameWorkBatch{FrameWorkFrameContext: frameWorkFrameContext(final, threading.FrameWorkSequenceContextFromHeader(final.SequenceHeader))}
	_, _, cdefLen, err := ctx.CDEFIndexMapShape()
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := ctx.BindCDEFIndexMap(make([]uint8, cdefLen), make([]bool, cdefLen))
	if err != nil {
		t.Fatal(err)
	}
	if err := state.SetCDEFIndexMap(cdefMap); err != nil {
		t.Fatal(err)
	}
	_, _, lfLen, err := ctx.LoopFilterMapShape()
	if err != nil {
		t.Fatal(err)
	}
	lfMap, err := ctx.BindLoopFilterMap(make([]threading.FrameWorkLoopFilterBlockRecord, lfLen))
	if err != nil {
		t.Fatal(err)
	}
	if err := state.SetLoopFilterMap(lfMap); err != nil {
		t.Fatal(err)
	}
	restorationPlan, err := ctx.RestorationFramePlan()
	if err != nil {
		t.Fatal(err)
	}
	restorationBuffers, err := ctx.BindRestorationFrameBuffers(
		make([]tile.RestorationUnitRecord, restorationPlan.UnitRecordLen()),
		make([]uint16, restorationPlan.BoundaryBufferLen()),
		make([]uint16, restorationPlan.BoundaryBufferLen()),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := state.SetRestorationFrameBuffers(restorationBuffers); err != nil {
		t.Fatal(err)
	}

	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Surface: plan.Surface},
	}
	var postRan bool
	result, err := state.RunStepWithPostFilter(&refs, &pool, final, step, nil, nil, nil, releases[:], nil, func(post FrameWorkPostFilterContext) error {
		postRan = true
		if post.CDEFIndexMap == nil || post.CDEFIndexMap.Stride != cdefMap.Stride || post.CDEFIndexMap.Rows != cdefMap.Rows {
			t.Fatalf("CDEFIndexMap=%+v want stride=%d rows=%d", post.CDEFIndexMap, cdefMap.Stride, cdefMap.Rows)
		}
		if post.LoopFilterMap == nil || post.LoopFilterMap.Stride != lfMap.Stride || post.LoopFilterMap.Rows != lfMap.Rows {
			t.Fatalf("LoopFilterMap=%+v want stride=%d rows=%d", post.LoopFilterMap, lfMap.Stride, lfMap.Rows)
		}
		if post.RestorationFrameBuffers == nil || len(post.RestorationFrameBuffers.Records[0]) != restorationPlan.UnitRecords[0] {
			t.Fatalf("RestorationFrameBuffers=%+v want records=%d", post.RestorationFrameBuffers, restorationPlan.UnitRecords[0])
		}
		post.CDEFIndexMap.Read[0] = true
		post.LoopFilterMap.Records[0].Valid = true
		post.RestorationFrameBuffers.Records[0][0].Unit.Type = parser.RestorationWiener
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !postRan || !result.CompletedFrame || state.Active() {
		t.Fatalf("postRan=%v result=%+v active=%v", postRan, result, state.Active())
	}
	if !cdefMap.Read[0] {
		t.Fatal("postfilter CDEFIndexMap did not alias caller storage")
	}
	if !lfMap.Records[0].Valid {
		t.Fatal("postfilter LoopFilterMap did not alias caller storage")
	}
	if restorationBuffers.Records[0][0].Unit.Type != parser.RestorationWiener {
		t.Fatal("postfilter RestorationFrameBuffers did not alias caller storage")
	}
}

func TestFrameWorkStatePostFilterContextPreview(t *testing.T) {
	pool := testFramePool(t, 1)
	surface, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	cdefMap := threading.FrameWorkCDEFIndexMap{Index: make([]uint8, 1), Read: make([]bool, 1), Stride: 1, Rows: 1}
	lfMap := threading.FrameWorkLoopFilterMap{Records: make([]threading.FrameWorkLoopFilterBlockRecord, 1), Stride: 1, Rows: 1}
	restorationBuffers := threading.FrameWorkRestorationFrameBuffers{Plan: tile.RestorationFramePlan{Planes: 1}}
	state := FrameWorkState{
		Surface:                      surface,
		ReferenceCount:               2,
		cdefIndexMap:                 cdefMap,
		cdefIndexMapValid:            true,
		loopFilterMap:                lfMap,
		loopFilterMapValid:           true,
		restorationFrameBuffers:      restorationBuffers,
		restorationFrameBuffersValid: true,
		active:                       true,
	}
	event := finalFrameEvent(0)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{
			Surface:        surface,
			ReferenceCount: 2,
		},
	}

	ctx, err := state.PostFilterContext(&pool, event, step, true)
	if err != nil {
		t.Fatal(err)
	}
	if ctx.Event.Kind != event.Kind ||
		!ctx.Event.TileGroup.Final ||
		ctx.Step != step ||
		ctx.Output != output ||
		ctx.ReferenceCount != 2 ||
		!ctx.ExecutedTileWork ||
		ctx.CDEFIndexMap != &state.cdefIndexMap ||
		ctx.LoopFilterMap != &state.loopFilterMap ||
		ctx.RestorationFrameBuffers != &state.restorationFrameBuffers {
		t.Fatalf("ctx=%+v output=%p state=%+v", ctx, output, state)
	}
	ctx.CDEFIndexMap.Read[0] = true
	ctx.LoopFilterMap.Records[0].Valid = true
	if !cdefMap.Read[0] || !lfMap.Records[0].Valid {
		t.Fatal("preview context did not alias caller-owned side maps")
	}
	if !state.Active() {
		t.Fatal("preview context mutated active state")
	}

	nonFinal := event
	nonFinal.TileGroup.Final = false
	noop, err := state.PostFilterContext(&pool, nonFinal, step, true)
	if err != nil {
		t.Fatal(err)
	}
	if noop.Output != nil || noop.Event.Kind != EventIgnored || noop.Step.Kind != FrameWorkStepIgnored {
		t.Fatalf("non-final context=%+v want zero", noop)
	}
	if _, err := state.PostFilterContext(nil, event, step, true); !errors.Is(err, frame.ErrInvalidPool) {
		t.Fatalf("nil pool err=%v want %v", err, frame.ErrInvalidPool)
	}
	if _, err := state.PostFilterContext(&pool, event, FrameWorkStep{}, true); !errors.Is(err, ErrInvalidFrameWorkStep) {
		t.Fatalf("bad step err=%v want %v", err, ErrInvalidFrameWorkStep)
	}
	state.active = false
	if _, err := state.PostFilterContext(&pool, event, step, true); !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("inactive err=%v want %v", err, ErrInvalidFrameWorkState)
	}
}

func TestFrameWorkStateRunEventWithContextPostFilterErrorKeepsActive(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	want := errors.New("postfilter")
	var tileRan bool
	_, err = state.RunEventWithContextAndPostFilter(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx FrameWorkBatch) error {
		tileRan = true
		ctx.Output.Y.Pix[0] = 0x22
		return nil
	}, func(ctx FrameWorkPostFilterContext) error {
		if ctx.Output.Y.Pix[0] != 0x22 {
			t.Fatalf("postfilter sample=%d want 0x22", ctx.Output.Y.Pix[0])
		}
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("RunEventWithContextAndPostFilter err=%v want %v", err, want)
	}
	if !tileRan || !state.Active() {
		t.Fatalf("tileRan=%v active=%v", tileRan, state.Active())
	}
	if slot, ok := refs.ReferenceSlot(0); ok || slot != -1 {
		t.Fatalf("slot=%d ok=%v want no publication", slot, ok)
	}
	output, err := pool.Frame(state.Surface)
	if err != nil {
		t.Fatal(err)
	}
	if output.Y.Pix[0] != 0x22 {
		t.Fatalf("active output sample=%d want 0x22", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextRequireNoActivePostFilters(t *testing.T) {
	if err := (FrameWorkPostFilterContext{}).RequireNoActivePostFilters(); err != nil {
		t.Fatalf("inactive postfilter err=%v", err)
	}

	tests := []struct {
		name  string
		event Event
	}{
		{
			name: "loopfilter",
			event: Event{
				LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{1, 0}},
			},
		},
		{
			name: "cdef-indexes",
			event: Event{
				CDEF: parser.CDEFParams{Bits: 1, StrengthCount: 2},
			},
		},
		{
			name: "cdef-strength",
			event: Event{
				CDEF: parser.CDEFParams{StrengthCount: 1, YStrength: [parser.MaxCDEFStrengths]uint8{4}},
			},
		},
		{
			name: "superres",
			event: Event{
				FrameSize: parser.FrameSize{SuperResEnabled: true},
			},
		},
		{
			name: "restoration",
			event: Event{
				Restoration: parser.RestorationParams{Type: [3]parser.RestorationType{parser.RestorationWiener}},
			},
		},
		{
			name: "film-grain",
			event: Event{
				FilmGrain: parser.FilmGrainParams{Apply: true},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := (FrameWorkPostFilterContext{Event: tt.event}).RequireNoActivePostFilters()
			if !errors.Is(err, ErrUnsupportedPostFilter) {
				t.Fatalf("err=%v want %v", err, ErrUnsupportedPostFilter)
			}
		})
	}
}

func TestFrameWorkStateRunStepWithPostFilterAllocs(t *testing.T) {
	pool := testFramePool(t, 1)
	surface, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	event := finalFrameEvent(0)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Surface: surface},
	}
	post := func(ctx FrameWorkPostFilterContext) error {
		if ctx.Output != output || ctx.ExecutedTileWork {
			t.Fatalf("ctx=%+v output=%p", ctx, output)
		}
		ctx.Output.Y.Pix[0] ^= 1
		return nil
	}
	var state FrameWorkState
	allocs := testing.AllocsPerRun(1000, func() {
		var refs SurfaceReferences
		state = FrameWorkState{Surface: surface, active: true}
		if _, err := state.RunStepWithPostFilter(&refs, &pool, event, step, nil, nil, nil, nil, nil, post); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("RunStepWithPostFilter allocated: %f", allocs)
	}
}

func TestFrameWorkStatePostFilterContextAllocs(t *testing.T) {
	pool := testFramePool(t, 1)
	surface, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	event := finalFrameEvent(0)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Surface: surface},
	}
	state := FrameWorkState{Surface: surface, active: true}

	allocs := testing.AllocsPerRun(1000, func() {
		ctx, err := state.PostFilterContext(&pool, event, step, false)
		if err != nil {
			t.Fatal(err)
		}
		if ctx.Output != output || ctx.ExecutedTileWork {
			t.Fatalf("ctx=%+v output=%p", ctx, output)
		}
	})
	if allocs != 0 {
		t.Fatalf("PostFilterContext allocated: %f", allocs)
	}
}

func TestFrameWorkStateRunEventWithContextInterFrameReferences(t *testing.T) {
	keyFrame := append([]byte{}, shownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	interFrame := append([]byte{}, interFrameHeaderPayload()...)
	interFrame = append(interFrame, 0xbb)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, keyFrame)
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, interFrame)

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[2].FrameSize.CodedWidth, events[2].FrameSize.Height, 2)
	referenceSurface, referenceFrame, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(1<<0, referenceSurface, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch

	result, err := state.RunEventWithContext(&refs, &pool, events[0].SequenceHeader, events[2], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx FrameWorkBatch) error {
		if ctx.Output == nil || ctx.Output == referenceFrame {
			t.Fatalf("output=%p reference=%p", ctx.Output, referenceFrame)
		}
		if len(ctx.Payload) != len(events[2].Unit.Payload) || ctx.Payload[ctx.Jobs[0].Offset] != 0xbb {
			t.Fatalf("payload=%v", ctx.Payload)
		}
		if len(ctx.References) != parser.InterRefsPerFrame {
			t.Fatalf("references=%d", len(ctx.References))
		}
		for i := 0; i < len(ctx.References); i++ {
			if ctx.References[i] != referenceFrame {
				t.Fatalf("reference[%d]=%p want %p", i, ctx.References[i], referenceFrame)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.ReferenceCount != parser.InterRefsPerFrame ||
		result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true, ReleaseCount: 1}) ||
		state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != result.Step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, result.Step.Begin.Surface)
	}
	if _, err := pool.Frame(referenceSurface); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("reference surface err=%v want %v", err, frame.ErrInvalidSlot)
	}
}

func TestFrameWorkStateRunEventWithContextShowExisting(t *testing.T) {
	pool := testFramePool(t, 1)
	referenceSurface, referenceFrame, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(1<<0, referenceSurface, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	result, err := state.RunEventWithContext(&refs, &pool, testSequence(), showExistingWorkEvent(0, parser.FrameTypeInter), 32, nil, nil, 1, nil, nil, nil, releases[:], nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Step.Kind != FrameWorkStepShowExisting ||
		result.Output != referenceFrame ||
		result.Run != (FrameWorkStepResult{}) ||
		state.Active() {
		t.Fatalf("result=%+v output=%p active=%v", result, referenceFrame, state.Active())
	}
}

func TestFrameWorkStateRunStepRejectsMismatchedFinalStep(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	var releases [parser.RefFrames]int
	_, err = state.RunStep(&refs, &pool, finalFrameEvent(0xff), FrameWorkStep{Kind: FrameWorkStepIgnored}, nil, nil, nil, releases[:], nil)
	if !errors.Is(err, ErrInvalidFrameWorkStep) {
		t.Fatalf("RunStep err=%v want %v", err, ErrInvalidFrameWorkStep)
	}
	if !state.Active() || state.Surface != begin.Surface {
		t.Fatalf("state=%+v active=%v begin=%+v", state, state.Active(), begin)
	}
	if slot, ok := refs.ReferenceSlot(0); ok || slot != -1 {
		t.Fatalf("slot=%d ok=%v want no publication", slot, ok)
	}
}

func TestFrameWorkStateRunStepNoop(t *testing.T) {
	var state FrameWorkState
	result, err := state.RunStep(nil, nil, Event{Kind: EventMetadata}, FrameWorkStep{Kind: FrameWorkStepIgnored}, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkStepResult{}) {
		t.Fatalf("result=%+v", result)
	}
}

func TestFrameWorkStateFinishIfEventCompletesFrameWork(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	plan, _, err := state.Begin(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}

	var releases [parser.RefFrames]int
	completed, releaseCount, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, events[2], releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if !completed || releaseCount != 0 || state.Active() {
		t.Fatalf("completed=%v releaseCount=%d active=%v", completed, releaseCount, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != plan.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, plan.Surface)
	}
}

func TestFrameWorkStateFinishIfEventCompletesFrameWorkFrameOBU(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	plan, _, err := state.Begin(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}

	var releases [parser.RefFrames]int
	completed, releaseCount, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, events[1], releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if !completed || releaseCount != 0 || state.Active() {
		t.Fatalf("completed=%v releaseCount=%d active=%v", completed, releaseCount, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != plan.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, plan.Surface)
	}
}

func TestFrameWorkStateFinishIfEventCompletesFrameWorkNoop(t *testing.T) {
	var nilState *FrameWorkState
	completed, count, err := nilState.FinishIfEventCompletesFrameWork(nil, nil, Event{Kind: EventTileGroup}, nil)
	if err != nil || completed || count != 0 {
		t.Fatalf("nil state completed=%v count=%d err=%v", completed, count, err)
	}

	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	if _, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	completed, count, err = state.FinishIfEventCompletesFrameWork(&refs, &pool, Event{Kind: EventTileGroup}, nil)
	if err != nil || completed || count != 0 || !state.Active() {
		t.Fatalf("completed=%v count=%d err=%v active=%v", completed, count, err, state.Active())
	}
}

func TestFrameWorkStateFinishIfEventCompletesFrameWorkRejectsInactive(t *testing.T) {
	var state FrameWorkState
	var releases [parser.RefFrames]int
	completed, count, err := state.FinishIfEventCompletesFrameWork(nil, nil, finalFrameEvent(0xff), releases[:])
	if !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("FinishIfEventCompletesFrameWork err=%v want %v", err, ErrInvalidFrameWorkState)
	}
	if completed || count != 0 {
		t.Fatalf("completed=%v count=%d", completed, count)
	}
}

func TestFrameWorkStateFinishIfEventCompletesFrameWorkKeepsActiveOnError(t *testing.T) {
	pool := testFramePool(t, 2)
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0xff, index0, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	plan, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(index0); err != nil {
		t.Fatal(err)
	}

	completed, count, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, finalFrameEvent(0xff), releases[:])
	if !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("FinishIfEventCompletesFrameWork err=%v want %v", err, frame.ErrInvalidSlot)
	}
	if completed || count != 0 || !state.Active() || state.Surface != plan.Surface {
		t.Fatalf("completed=%v count=%d state=%+v active=%v", completed, count, state, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != index0 {
		t.Fatalf("slot=%d ok=%v want unchanged %d", slot, ok, index0)
	}
}

func TestFrameWorkStateShowExistingDropsActiveWork(t *testing.T) {
	pool := testFramePool(t, 2)
	reference, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(1<<0, reference, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	plan, err := state.ShowExisting(&refs, &pool, showExistingWorkEvent(0, parser.FrameTypeInter), releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan != (ShowExistingFrameWorkPlan{Surface: reference, DroppedFrameWork: true}) {
		t.Fatalf("plan=%+v", plan)
	}
	if state.Active() || pool.Available() != 1 {
		t.Fatalf("active=%v available=%d", state.Active(), pool.Available())
	}
	if _, err := pool.Frame(begin.Surface); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("dropped frame err=%v want %v", err, frame.ErrInvalidSlot)
	}
	if _, err := pool.Frame(reference); err != nil {
		t.Fatalf("reference frame err=%v", err)
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != reference {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, reference)
	}
}

func TestFrameWorkStateShowExistingKeyResetsReferences(t *testing.T) {
	pool := testFramePool(t, 3)
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index1, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(1<<0, index0, releases[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(1<<1, index1, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	if _, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}

	plan, err := state.ShowExisting(&refs, &pool, showExistingWorkEvent(1, parser.FrameTypeKey), releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan != (ShowExistingFrameWorkPlan{Surface: index1, ReleaseCount: 1, DroppedFrameWork: true}) || releases[0] != index0 {
		t.Fatalf("plan=%+v release=%d want surface=%d release=%d", plan, releases[0], index1, index0)
	}
	if pool.Available() != 2 {
		t.Fatalf("available=%d want 2", pool.Available())
	}
	for i := range parser.RefFrames {
		slot, ok := refs.ReferenceSlot(i)
		if !ok || slot != index1 {
			t.Fatalf("slot[%d]=%d ok=%v want %d", i, slot, ok, index1)
		}
	}
}

func TestFrameWorkStateShowExistingInactive(t *testing.T) {
	pool := testFramePool(t, 1)
	reference, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(1<<0, reference, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	plan, err := state.ShowExisting(&refs, &pool, showExistingWorkEvent(0, parser.FrameTypeInter), releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan != (ShowExistingFrameWorkPlan{Surface: reference}) {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestFrameWorkStateShowExistingRejectsInvalidEventBeforeAbort(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = state.ShowExisting(&refs, &pool, Event{Kind: EventFrameHeader}, nil)
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("ShowExisting err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
	if !state.Active() || state.Surface != begin.Surface || pool.Available() != 0 {
		t.Fatalf("state=%+v active=%v available=%d", state, state.Active(), pool.Available())
	}
}

func TestFrameWorkStateShowExistingRejectsMissingReferenceBeforeAbort(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = state.ShowExisting(&refs, &pool, showExistingWorkEvent(0, parser.FrameTypeInter), releases[:])
	if !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("ShowExisting err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	if !state.Active() || state.Surface != begin.Surface || pool.Available() != 0 {
		t.Fatalf("state=%+v active=%v available=%d", state, state.Active(), pool.Available())
	}
}

func TestFrameWorkStateShowExistingKeepsActiveOnAbortError(t *testing.T) {
	pool := testFramePool(t, 2)
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	reference, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(1<<0, reference, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = state.ShowExisting(&refs, nil, showExistingWorkEvent(0, parser.FrameTypeInter), nil)
	if !errors.Is(err, frame.ErrInvalidPool) {
		t.Fatalf("ShowExisting err=%v want %v", err, frame.ErrInvalidPool)
	}
	if !state.Active() || state.Surface != begin.Surface || pool.Available() != 0 {
		t.Fatalf("state=%+v active=%v available=%d", state, state.Active(), pool.Available())
	}
}

func TestFrameWorkStateAbortIfEventDropsFrameWork(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	plan, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	aborted, err := state.AbortIfEventDropsFrameWork(&pool, Event{Kind: EventTemporalDelimiter, NewTemporalUnit: true})
	if err != nil {
		t.Fatal(err)
	}
	if !aborted || state.Active() || pool.Available() != 1 {
		t.Fatalf("aborted=%v state=%+v active=%v available=%d", aborted, state, state.Active(), pool.Available())
	}
	if _, err := pool.Frame(plan.Surface); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("aborted frame err=%v want %v", err, frame.ErrInvalidSlot)
	}
}

func TestFrameWorkStateAbortIfShowExistingFrameDropsFrameWork(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	plan, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	aborted, err := state.AbortIfEventDropsFrameWork(&pool, Event{Kind: EventExistingFrame})
	if err != nil {
		t.Fatal(err)
	}
	if !aborted || state.Active() || pool.Available() != 1 {
		t.Fatalf("aborted=%v active=%v available=%d", aborted, state.Active(), pool.Available())
	}
	if _, err := pool.Frame(plan.Surface); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("aborted frame err=%v want %v", err, frame.ErrInvalidSlot)
	}
}

func TestFrameWorkStateAbortIfEventDropsFrameWorkNoop(t *testing.T) {
	var nilState *FrameWorkState
	aborted, err := nilState.AbortIfEventDropsFrameWork(nil, Event{Kind: EventTemporalDelimiter, NewTemporalUnit: true})
	if err != nil || aborted {
		t.Fatalf("nil state aborted=%v err=%v", aborted, err)
	}

	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	if aborted, err = state.AbortIfEventDropsFrameWork(&pool, Event{Kind: EventTemporalDelimiter, NewTemporalUnit: true}); err != nil || aborted {
		t.Fatalf("inactive state aborted=%v err=%v", aborted, err)
	}

	if _, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	aborted, err = state.AbortIfEventDropsFrameWork(&pool, Event{Kind: EventSequenceHeader, OperatingParametersChanged: true})
	if err != nil || aborted || !state.Active() || pool.Available() != 0 {
		t.Fatalf("non-drop event aborted=%v err=%v active=%v available=%d", aborted, err, state.Active(), pool.Available())
	}
}

func TestFrameWorkStateAbortIfEventDropsFrameWorkKeepsActiveOnError(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	plan, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	aborted, err := state.AbortIfEventDropsFrameWork(nil, Event{Kind: EventSequenceHeader, NewCodedVideoSequence: true})
	if !errors.Is(err, frame.ErrInvalidPool) {
		t.Fatalf("AbortIfEventDropsFrameWork err=%v want %v", err, frame.ErrInvalidPool)
	}
	if aborted || !state.Active() || state.Surface != plan.Surface || pool.Available() != 0 {
		t.Fatalf("aborted=%v state=%+v active=%v available=%d", aborted, state, state.Active(), pool.Available())
	}
}

func TestFrameWorkStateRejectsInvalidState(t *testing.T) {
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	_, err := state.PlanTile(Event{Kind: EventTileGroup}, 1, spans[:], jobs[:], batches[:])
	if !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("inactive PlanTile err=%v want %v", err, ErrInvalidFrameWorkState)
	}

	var releases [parser.RefFrames]int
	_, err = state.Finish(nil, nil, finalFrameEvent(0xff), releases[:])
	if !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("inactive Finish err=%v want %v", err, ErrInvalidFrameWorkState)
	}
	err = state.Abort(nil)
	if !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("inactive Abort err=%v want %v", err, ErrInvalidFrameWorkState)
	}

	var nilState *FrameWorkState
	err = nilState.Abort(nil)
	if !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("nil Abort err=%v want %v", err, ErrInvalidFrameWorkState)
	}

	pool := testFramePool(t, 2)
	var refs SurfaceReferences
	event := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	if _, _, err := state.Begin(&refs, &pool, testSequence(), event, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	_, _, err = state.Begin(&refs, &pool, testSequence(), event, 32, nil, 1, nil, nil, nil)
	if !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("active Begin err=%v want %v", err, ErrInvalidFrameWorkState)
	}
	if pool.Available() != 1 {
		t.Fatalf("active begin consumed a surface, available=%d", pool.Available())
	}
}

func TestFrameWorkStateAbortReleasesSurface(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	plan, output, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || pool.Available() != 0 {
		t.Fatalf("output=%p available=%d", output, pool.Available())
	}

	if err := state.Abort(&pool); err != nil {
		t.Fatal(err)
	}
	if state.Active() {
		t.Fatalf("state still active after abort: %+v", state)
	}
	if pool.Available() != 1 {
		t.Fatalf("available=%d want 1", pool.Available())
	}
	if _, err := pool.Frame(plan.Surface); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("aborted frame err=%v want %v", err, frame.ErrInvalidSlot)
	}
	if slot, ok := refs.ReferenceSlot(0); ok || slot != -1 {
		t.Fatalf("abort published reference slot=%d ok=%v", slot, ok)
	}
}

func TestFrameWorkStateKeepsActiveOnAbortError(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	plan, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	err = state.Abort(nil)
	if !errors.Is(err, frame.ErrInvalidPool) {
		t.Fatalf("Abort err=%v want %v", err, frame.ErrInvalidPool)
	}
	if !state.Active() || state.Surface != plan.Surface {
		t.Fatalf("state after abort error=%+v active=%v plan=%+v", state, state.Active(), plan)
	}
	if pool.Available() != 0 {
		t.Fatalf("failed abort released surface, available=%d", pool.Available())
	}
}

func TestFrameWorkStateKeepsActiveOnFinishError(t *testing.T) {
	pool := testFramePool(t, 2)
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0xff, index0, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	plan, _, err := state.Begin(&refs, &pool, testSequence(), Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(index0); err != nil {
		t.Fatal(err)
	}

	_, err = state.Finish(&refs, &pool, finalFrameEvent(0xff), releases[:])
	if !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("Finish err=%v want %v", err, frame.ErrInvalidSlot)
	}
	if !state.Active() || state.Surface != plan.Surface {
		t.Fatalf("state after error=%+v active=%v plan=%+v", state, state.Active(), plan)
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != index0 {
		t.Fatalf("slot=%d ok=%v want unchanged %d", slot, ok, index0)
	}
}

func TestPlanTileWorkFrameEvent(t *testing.T) {
	frame := append([]byte{}, reducedStillFrameHeaderPayload()...)
	frame = append(frame, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, frame)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	plan, err := PlanTileWork(events[1], 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan.SpanCount != 1 || plan.JobCount != 1 || plan.BatchCount != 1 {
		t.Fatalf("plan=%+v", plan)
	}
	if spans[0].Offset != len(reducedStillFrameHeaderPayload()) || spans[0].Size != 1 {
		t.Fatalf("span=%+v", spans[0])
	}
}

func TestPlanTileWorkPropagatesContextUpdateTile(t *testing.T) {
	event := Event{
		Kind: EventTileGroup,
		Unit: obu.Unit{Payload: []byte{
			0x00, // tile 0 size minus one.
			0xaa,
			0xbb,
		}},
		TileInfo: parser.TileInfo{
			RefreshContext:      true,
			ContextUpdateTileID: 1,
			TileSizeBytes:       1,
			SBCols:              2,
			SBRows:              1,
			Cols:                2,
			Rows:                1,
			ColStartSB:          [parser.MaxTileCols + 1]uint16{0, 1, 2},
			RowStartSB:          [parser.MaxTileRows + 1]uint16{0, 1},
		},
		TileGroup: parser.TileGroup{
			StartTile: 0,
			EndTile:   1,
			TileCount: 2,
			Final:     true,
		},
	}
	var spans [2]parser.TileSpan
	var jobs [2]tile.Job
	var batches [2]threading.Batch

	plan, err := PlanTileWork(event, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan != (TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: 1}) {
		t.Fatalf("plan=%+v", plan)
	}
	if jobs[0].UpdatesFrameContext {
		t.Fatalf("job[0]=%+v should not update frame context", jobs[0])
	}
	if !jobs[1].UpdatesFrameContext {
		t.Fatalf("job[1]=%+v should update frame context", jobs[1])
	}
}

func TestPlanTileWorkRejectsNonTileEvent(t *testing.T) {
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	_, err := PlanTileWork(Event{Kind: EventSequenceHeader}, 1, spans[:], jobs[:], batches[:])
	if !errors.Is(err, ErrInvalidTileWork) {
		t.Fatalf("PlanTileWork err=%v want %v", err, ErrInvalidTileWork)
	}
}

func TestPlanTileWorkAllocs(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := PlanTileWork(events[2], 1, spans[:], jobs[:], batches[:])
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PlanTileWork allocated: %f", allocs)
	}
}

func TestBeginFrameWorkAllocs(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	event := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		plan, _, err := BeginFrameWork(&refs, &pool, testSequence(), event, 32, nil, 1, nil, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if plan.Surface != 0 {
			t.Fatalf("surface=%d", plan.Surface)
		}
	})
	if allocs != 0 {
		t.Fatalf("BeginFrameWork allocated: %f", allocs)
	}
}

func TestPlanFrameTileWorkAllocs(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := PlanFrameTileWork(events[2], 0, 0, 1, spans[:], jobs[:], batches[:])
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PlanFrameTileWork allocated: %f", allocs)
	}
}

func TestFrameWorkStateAllocs(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _, err := state.Begin(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, nil, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, err = state.PlanTile(events[2], 1, spans[:], jobs[:], batches[:])
		if err != nil {
			t.Fatal(err)
		}
		if _, err = state.Finish(&refs, &pool, events[2], releases[:]); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState allocated: %f", allocs)
	}
}

func TestFrameWorkStateAbortAllocs(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	event := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		if _, _, err := state.Begin(&refs, &pool, testSequence(), event, 32, nil, 1, nil, nil, nil); err != nil {
			t.Fatal(err)
		}
		if err := state.Abort(&pool); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState Abort allocated: %f", allocs)
	}
}

func TestFrameWorkStateAbortIfEventDropsFrameWorkAllocs(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	drop := Event{Kind: EventExistingFrame}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		if _, _, err := state.Begin(&refs, &pool, testSequence(), begin, 32, nil, 1, nil, nil, nil); err != nil {
			t.Fatal(err)
		}
		aborted, err := state.AbortIfEventDropsFrameWork(&pool, drop)
		if err != nil {
			t.Fatal(err)
		}
		if !aborted {
			t.Fatal("not aborted")
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState AbortIfEventDropsFrameWork allocated: %f", allocs)
	}
}

func TestFrameWorkStateFinishIfEventCompletesFrameWorkAllocs(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		if _, _, err := state.Begin(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, nil, nil, nil); err != nil {
			t.Fatal(err)
		}
		completed, _, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, events[2], releases[:])
		if err != nil {
			t.Fatal(err)
		}
		if !completed {
			t.Fatal("not completed")
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState FinishIfEventCompletesFrameWork allocated: %f", allocs)
	}
}

func TestFrameWorkStateShowExistingAllocs(t *testing.T) {
	pool := testFramePool(t, 2)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	show := showExistingWorkEvent(0, parser.FrameTypeInter)

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		reference, _, err := pool.Acquire()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := refs.Refresh(1<<0, reference, releases[:]); err != nil {
			t.Fatal(err)
		}
		if _, _, err := state.Begin(&refs, &pool, testSequence(), begin, 32, nil, 1, nil, nil, nil); err != nil {
			t.Fatal(err)
		}
		plan, err := state.ShowExisting(&refs, &pool, show, releases[:])
		if err != nil {
			t.Fatal(err)
		}
		if plan.Surface != reference || !plan.DroppedFrameWork {
			t.Fatalf("plan=%+v reference=%d", plan, reference)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState ShowExisting allocated: %f", allocs)
	}
}

func TestFrameWorkStatePlanEventAllocs(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		step, output, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, spans[:], jobs[:], batches[:], releases[:])
		if err != nil {
			t.Fatal(err)
		}
		if step.Kind != FrameWorkStepBegin || output == nil {
			t.Fatalf("begin step=%+v output=%p", step, output)
		}
		step, output, err = state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[2], 32, nil, 1, spans[:], jobs[:], batches[:], releases[:])
		if err != nil {
			t.Fatal(err)
		}
		if step.Kind != FrameWorkStepTile || output != nil {
			t.Fatalf("tile step=%+v output=%p", step, output)
		}
		completed, _, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, events[2], releases[:])
		if err != nil {
			t.Fatal(err)
		}
		if !completed {
			t.Fatal("not completed")
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState PlanEvent allocated: %f", allocs)
	}
}

func TestExecuteFrameWorkStepAllocs(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		executed, err := ExecuteFrameWorkStep(step, workerPool, jobs[:], batches[:], func(threading.Batch, []tile.Job) error {
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if !executed {
			t.Fatal("not executed")
		}
	})
	if allocs != 0 {
		t.Fatalf("ExecuteFrameWorkStep allocated: %f", allocs)
	}
}

func TestExecuteFrameWorkStepWithContextAllocs(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	var output frame.Frame
	var reference frame.Frame
	references := [parser.InterRefsPerFrame]*frame.Frame{&reference}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{
			ReferenceCount: 1,
			Tile:           TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount},
		},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		executed, err := ExecuteFrameWorkStepWithContext(step, workerPool, &output, references[:], jobs[:], batches[:], noopFrameWorkBatch)
		if err != nil {
			t.Fatal(err)
		}
		if !executed {
			t.Fatal("not executed")
		}
	})
	if allocs != 0 {
		t.Fatalf("ExecuteFrameWorkStepWithContext allocated: %f", allocs)
	}
}

func TestExecuteFrameWorkStepWithPayloadAllocs(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	payload := []byte{0xaa, 0xbb, 0xcc}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		executed, err := ExecuteFrameWorkStepWithPayload(step, workerPool, nil, nil, payload, jobs[:], batches[:], noopFrameWorkBatch)
		if err != nil {
			t.Fatal(err)
		}
		if !executed {
			t.Fatal("not executed")
		}
	})
	if allocs != 0 {
		t.Fatalf("ExecuteFrameWorkStepWithPayload allocated: %f", allocs)
	}
}

func TestFrameWorkStateRunStepAllocs(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int
	jobs, batches, batchCount := testExecutionWork(t)
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	final := finalFrameEvent(0xff)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		if _, _, err := state.Begin(&refs, &pool, testSequence(), begin, 32, nil, 1, nil, nil, nil); err != nil {
			t.Fatal(err)
		}
		result, err := state.RunStep(&refs, &pool, final, step, workerPool, jobs[:], batches[:], releases[:], func(threading.Batch, []tile.Job) error {
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if result != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState RunStep allocated: %f", allocs)
	}
}

func TestFrameWorkStateRunStepWithContextAllocs(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int
	jobs, batches, batchCount := testExecutionWork(t)
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	final := finalFrameEvent(0xff)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, output, err := state.Begin(&refs, &pool, testSequence(), begin, 32, nil, 1, nil, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		result, err := state.RunStepWithContext(&refs, &pool, final, step, workerPool, output, nil, jobs[:], batches[:], releases[:], func(FrameWorkBatch) error {
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if result != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState RunStepWithContext allocated: %f", allocs)
	}
}

func TestFrameWorkStateRunEventWithContextAllocs(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, err := state.RunEventWithContext(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, noopFrameWorkBatch)
		if err != nil {
			t.Fatal(err)
		}
		if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState RunEventWithContext allocated: %f", allocs)
	}
}

func TestFrameWorkStateRunEventWithContextRunnerAllocs(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int
	runner := noopFrameWorkRunner{}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, err := state.RunEventWithContextRunner(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, runner)
		if err != nil {
			t.Fatal(err)
		}
		if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState RunEventWithContextRunner allocated: %f", allocs)
	}
}

func TestFrameWorkStateRunEventWithResidualRunnerAllocs(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayloadQ(64)...)
	framePayload = append(framePayload, make([]byte, 256)...)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testStillSequenceHeaderPayload(64, 64))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int
	var runner threading.FrameWorkTileResidualRunner
	runner.Workers = []threading.FrameWorkTileResidualRunnerWorker{
		{
			Int32Scratch:    make([]int32, 32768),
			ResidualScratch: make([]int16, 4096),
		},
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, err := state.RunEventWithContextRunner(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &runner)
		if err != nil {
			t.Fatal(err)
		}
		if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState RunEventWith residual runner allocated: %f", allocs)
	}
}

func TestFrameWorkStateRunEventWithResidualRunnerSideDataPostFilterAllocs(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayloadQ(64)...)
	framePayload = append(framePayload, make([]byte, 256)...)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testStillSequenceHeaderPayload(64, 64))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}
	events[1].LoopFilter = parser.LoopFilterParams{LevelY: [2]uint8{4}}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int
	var runner threading.FrameWorkTileResidualRunner
	runner.Workers = []threading.FrameWorkTileResidualRunnerWorker{
		{
			Int32Scratch:    make([]int32, 32768),
			ResidualScratch: make([]int16, 4096),
		},
	}
	side := FrameWorkBoundSideDataRunner{
		LoopFilterRecords: make([]threading.FrameWorkLoopFilterBlockRecord, 256),
	}
	post := FrameWorkBoundSupportedPostFilterRunner{
		Scratch: FrameWorkPostFilterScratch{
			LoopFilterEdges: make([]FrameWorkLoopFilterPostFilterEdge, 256),
		},
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, err := state.RunEventWithContextAndSideDataAndPostFilterRunners(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &side, &runner, &post)
		if err != nil {
			t.Fatal(err)
		}
		if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
			post.Result.Completed != FrameWorkPostFilterLoopFilter ||
			post.Context.RemainingPostFilters() != 0 {
			t.Fatalf("result=%+v post=%+v remaining=%v", result, post.Result, post.Context.RemainingPostFilters())
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState RunEventWith residual side-data postfilter allocated: %f", allocs)
	}
}

func BenchmarkPlanTileWork(b *testing.B) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 3 {
		b.Fatalf("count=%d", count)
	}
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = PlanTileWork(events[2], 1, spans[:], jobs[:], batches[:])
	}
}

func BenchmarkPlanFrameTileWork(b *testing.B) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 3 {
		b.Fatalf("count=%d", count)
	}
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = PlanFrameTileWork(events[2], 0, 0, 1, spans[:], jobs[:], batches[:])
	}
}

func BenchmarkFrameWorkState(b *testing.B) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 3 {
		b.Fatalf("count=%d", count)
	}

	pool := benchmarkFramePoolForSize(b, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _, _ = state.Begin(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, nil, nil, nil)
		_, _ = state.PlanTile(events[2], 1, spans[:], jobs[:], batches[:])
		_, _ = state.Finish(&refs, &pool, events[2], releases[:])
	}
}

func BenchmarkFrameWorkStateAbort(b *testing.B) {
	pool := benchmarkFramePool(b, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	event := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _, _ = state.Begin(&refs, &pool, testSequence(), event, 32, nil, 1, nil, nil, nil)
		_ = state.Abort(&pool)
	}
}

func BenchmarkFrameWorkStateAbortIfEventDropsFrameWork(b *testing.B) {
	pool := benchmarkFramePool(b, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	drop := Event{Kind: EventTemporalDelimiter, NewTemporalUnit: true}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _, _ = state.Begin(&refs, &pool, testSequence(), begin, 32, nil, 1, nil, nil, nil)
		_, _ = state.AbortIfEventDropsFrameWork(&pool, drop)
	}
}

func BenchmarkFrameWorkStateFinishIfEventCompletesFrameWork(b *testing.B) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 3 {
		b.Fatalf("count=%d", count)
	}

	pool := benchmarkFramePoolForSize(b, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _, _ = state.Begin(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, nil, nil, nil)
		_, _, _ = state.FinishIfEventCompletesFrameWork(&refs, &pool, events[2], releases[:])
	}
}

func BenchmarkFrameWorkStateShowExisting(b *testing.B) {
	pool := benchmarkFramePool(b, 2)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	show := showExistingWorkEvent(0, parser.FrameTypeInter)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		reference, _, _ := pool.Acquire()
		_, _ = refs.Refresh(1<<0, reference, releases[:])
		_, _, _ = state.Begin(&refs, &pool, testSequence(), begin, 32, nil, 1, nil, nil, nil)
		_, _ = state.ShowExisting(&refs, &pool, show, releases[:])
	}
}

func BenchmarkFrameWorkStatePlanEvent(b *testing.B) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 3 {
		b.Fatalf("count=%d", count)
	}

	pool := benchmarkFramePoolForSize(b, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _, _ = state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[1], 32, nil, 1, spans[:], jobs[:], batches[:], releases[:])
		_, _, _ = state.PlanEvent(&refs, &pool, events[0].SequenceHeader, events[2], 32, nil, 1, spans[:], jobs[:], batches[:], releases[:])
		_, _, _ = state.FinishIfEventCompletesFrameWork(&refs, &pool, events[2], releases[:])
	}
}

func BenchmarkExecuteFrameWorkStep(b *testing.B) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := benchmarkExecutionWork(b)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ExecuteFrameWorkStep(step, workerPool, jobs[:], batches[:], func(threading.Batch, []tile.Job) error {
			return nil
		})
	}
}

func BenchmarkExecuteFrameWorkStepWithContext(b *testing.B) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := benchmarkExecutionWork(b)
	var output frame.Frame
	var reference frame.Frame
	references := [parser.InterRefsPerFrame]*frame.Frame{&reference}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{
			ReferenceCount: 1,
			Tile:           TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount},
		},
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ExecuteFrameWorkStepWithContext(step, workerPool, &output, references[:], jobs[:], batches[:], func(FrameWorkBatch) error {
			return nil
		})
	}
}

func BenchmarkExecuteFrameWorkStepWithPayload(b *testing.B) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := benchmarkExecutionWork(b)
	payload := []byte{0xaa, 0xbb}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ExecuteFrameWorkStepWithPayload(step, workerPool, nil, nil, payload, jobs[:], batches[:], noopFrameWorkBatch)
	}
}

func BenchmarkFrameWorkStateRunStep(b *testing.B) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	pool := benchmarkFramePool(b, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int
	jobs, batches, batchCount := benchmarkExecutionWork(b)
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	final := finalFrameEvent(0xff)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _, _ = state.Begin(&refs, &pool, testSequence(), begin, 32, nil, 1, nil, nil, nil)
		_, _ = state.RunStep(&refs, &pool, final, step, workerPool, jobs[:], batches[:], releases[:], func(threading.Batch, []tile.Job) error {
			return nil
		})
	}
}

func BenchmarkFrameWorkStateRunStepWithContext(b *testing.B) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	pool := benchmarkFramePool(b, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int
	jobs, batches, batchCount := benchmarkExecutionWork(b)
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	final := finalFrameEvent(0xff)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, output, _ := state.Begin(&refs, &pool, testSequence(), begin, 32, nil, 1, nil, nil, nil)
		_, _ = state.RunStepWithContext(&refs, &pool, final, step, workerPool, output, nil, jobs[:], batches[:], releases[:], func(FrameWorkBatch) error {
			return nil
		})
	}
}

func BenchmarkFrameWorkStateRunEventWithContext(b *testing.B) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 2 {
		b.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	pool := benchmarkFramePoolForSize(b, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _ = state.RunEventWithContext(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, noopFrameWorkBatch)
	}
}

func BenchmarkFrameWorkStateRunEventWithContextRunner(b *testing.B) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 2 {
		b.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	pool := benchmarkFramePoolForSize(b, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int
	runner := noopFrameWorkRunner{}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _ = state.RunEventWithContextRunner(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, runner)
	}
}

func BenchmarkFrameWorkSideDataScratchSizeBindRunner(b *testing.B) {
	size := FrameWorkSideDataScratchSize{
		CDEF:                4,
		LoopFilterRecords:   256,
		RestorationRecords:  16,
		RestorationBoundary: 512,
	}
	scratch := FrameWorkSideDataScratch{
		CDEFIndex:          make([]uint8, size.CDEF),
		CDEFRead:           make([]bool, size.CDEF),
		LoopFilterRecords:  make([]threading.FrameWorkLoopFilterBlockRecord, size.LoopFilterRecords),
		RestorationRecords: make([]tile.RestorationUnitRecord, size.RestorationRecords),
		RestorationAbove:   make([]uint16, size.RestorationBoundary),
		RestorationBelow:   make([]uint16, size.RestorationBoundary),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runner, err := size.BindRunner(scratch)
		if err != nil {
			b.Fatal(err)
		}
		if len(runner.LoopFilterRecords) != size.LoopFilterRecords {
			b.Fatalf("runner=%+v size=%+v", runner, size)
		}
	}
}

func BenchmarkFrameWorkStateRunEventWithResidualSideDataPostFilter(b *testing.B) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayloadQ(64)...)
	framePayload = append(framePayload, make([]byte, 256)...)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testStillSequenceHeaderPayload(64, 64))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 2 {
		b.Fatalf("count=%d", count)
	}
	events[1].LoopFilter = parser.LoopFilterParams{LevelY: [2]uint8{4}}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	pool := benchmarkFramePoolForSize(b, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int
	var runner threading.FrameWorkTileResidualRunner
	runner.Workers = []threading.FrameWorkTileResidualRunnerWorker{
		{
			Int32Scratch:    make([]int32, 32768),
			ResidualScratch: make([]int16, 4096),
		},
	}
	side := FrameWorkBoundSideDataRunner{
		LoopFilterRecords: make([]threading.FrameWorkLoopFilterBlockRecord, 256),
	}
	post := FrameWorkBoundSupportedPostFilterRunner{
		Scratch: FrameWorkPostFilterScratch{
			LoopFilterEdges: make([]FrameWorkLoopFilterPostFilterEdge, 256),
		},
	}
	pool.Reset()
	refs.Reset()
	state.Reset()
	result, err := state.RunEventWithContextAndSideDataAndPostFilterRunners(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &side, &runner, &post)
	if err != nil {
		b.Fatal(err)
	}
	if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		post.Result.Completed != FrameWorkPostFilterLoopFilter ||
		post.Context.RemainingPostFilters() != 0 {
		b.Fatalf("result=%+v post=%+v remaining=%v", result, post.Result, post.Context.RemainingPostFilters())
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, err := state.RunEventWithContextAndSideDataAndPostFilterRunners(&refs, &pool, events[0].SequenceHeader, events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &side, &runner, &post)
		if err != nil {
			b.Fatal(err)
		}
		if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
			post.Result.Completed != FrameWorkPostFilterLoopFilter ||
			post.Context.RemainingPostFilters() != 0 {
			b.Fatalf("result=%+v post=%+v remaining=%v", result, post.Result, post.Context.RemainingPostFilters())
		}
	}
}

func BenchmarkBeginFrameWork(b *testing.B) {
	pool := benchmarkFramePool(b, 1)
	var refs SurfaceReferences
	event := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		_, _, _ = BeginFrameWork(&refs, &pool, testSequence(), event, 32, nil, 1, nil, nil, nil)
	}
}

func testFramePoolForSize(t *testing.T, width uint32, height uint32, count int) frame.Pool {
	t.Helper()
	pool, err := makeFramePoolForSize(width, height, count)
	if err != nil {
		t.Fatal(err)
	}
	return pool
}

func benchmarkFramePoolForSize(b *testing.B, width uint32, height uint32, count int) frame.Pool {
	b.Helper()
	pool, err := makeFramePoolForSize(width, height, count)
	if err != nil {
		b.Fatal(err)
	}
	return pool
}

func testExecutionWork(t *testing.T) ([2]tile.Job, [2]threading.Batch, int) {
	t.Helper()
	jobs := [2]tile.Job{
		{Tile: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: 1},
		{Tile: 1, SBCols: 2, SBRows: 1, Offset: 1, Size: 2},
	}
	var batches [2]threading.Batch
	n, err := threading.BuildBatches(batches[:], jobs[:], 1)
	if err != nil {
		t.Fatal(err)
	}
	return jobs, batches, n
}

func benchmarkExecutionWork(b *testing.B) ([2]tile.Job, [2]threading.Batch, int) {
	b.Helper()
	jobs := [2]tile.Job{
		{Tile: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: 1},
		{Tile: 1, SBCols: 2, SBRows: 1, Offset: 1, Size: 1},
	}
	var batches [2]threading.Batch
	n, err := threading.BuildBatches(batches[:], jobs[:], 1)
	if err != nil {
		b.Fatal(err)
	}
	return jobs, batches, n
}

func testFrameWorkTileInfo() parser.TileInfo {
	tiles := parser.TileInfo{
		SBCols: 2,
		SBRows: 1,
		Cols:   2,
		Rows:   1,
	}
	tiles.ColStartSB[1] = 1
	tiles.ColStartSB[2] = 2
	tiles.RowStartSB[1] = 1
	return tiles
}

func testFrameWorkSequenceHeader() parser.SequenceHeader {
	return parser.SequenceHeader{
		SeqProfile:           1,
		Use128x128Superblock: true,
		EnableFilterIntra:    true,
		EnableOrderHint:      true,
		OrderHintBits:        5,
		EnableCDEF:           true,
		EnableRestoration:    true,
		ColorConfig: parser.ColorConfig{
			BitDepth:     10,
			SubsamplingX: true,
			SubsamplingY: true,
		},
		FilmGrainParamsPresent: true,
	}
}

func testFrameWorkSequenceContext() threading.FrameWorkSequenceContext {
	return threading.FrameWorkSequenceContextFromHeader(testFrameWorkSequenceHeader())
}

func testFrameWorkGlobalMotion() parser.GlobalMotionParams {
	motion := parser.DefaultGlobalMotionParams()
	motion.Ref[0].Type = parser.GlobalMotionTranslation
	motion.Ref[0].Matrix[0] = 17
	motion.Ref[0].Matrix[1] = -9
	return motion
}

func testFrameWorkFilmGrain() parser.FilmGrainParams {
	return parser.FilmGrainParams{
		ParamsPresent: true,
		Apply:         true,
		Seed:          99,
		BitDepth:      8,
		NumYPoints:    1,
		YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{16, 32}},
	}
}

func noopFrameWorkBatch(FrameWorkBatch) error {
	return nil
}

func testStillSequenceHeaderPayload(width uint64, height uint64) []byte {
	var w testBitWriter
	w.writeBits(0, 3)        // seq_profile
	w.writeBool(true)        // still_picture
	w.writeBool(true)        // reduced_still_picture_header
	w.writeBits(5, 5)        // seq_level_idx[0]
	w.writeBits(7, 4)        // frame_width_bits_minus_1
	w.writeBits(7, 4)        // frame_height_bits_minus_1
	w.writeBits(width-1, 8)  // max_frame_width_minus_1
	w.writeBits(height-1, 8) // max_frame_height_minus_1
	w.writeBool(false)       // use_128x128_superblock
	w.writeBool(true)        // enable_filter_intra
	w.writeBool(true)        // enable_intra_edge_filter
	w.writeBool(false)       // enable_superres
	w.writeBool(true)        // enable_cdef
	w.writeBool(false)       // enable_restoration
	w.writeBool(false)       // high_bitdepth
	w.writeBool(false)       // mono_chrome
	w.writeBool(false)       // color_description_present_flag
	w.writeBool(false)       // color_range
	w.writeBits(0, 2)        // chroma_sample_position
	w.writeBool(true)        // separate_uv_delta_q
	w.writeBool(false)       // film_grain_params_present
	return w.trailingBits()
}

func reducedStillFrameHeaderPayloadQ(baseQIdx uint8) []byte {
	var w testBitWriter
	w.writeBool(true)  // disable_cdf_update
	w.writeBool(false) // render_and_frame_size_different
	w.writeBool(false) // uniform_tile_spacing_flag
	writeQuantParams(&w, baseQIdx)
	writeZeroSegmentationParams(&w)
	w.writeBool(false) // reduced_tx_set
	return w.bytes()
}

type noopFrameWorkRunner struct{}

func (noopFrameWorkRunner) Run(FrameWorkBatch) error {
	return nil
}

type frameWorkContextRunner struct {
	wantKind     FrameWorkStepKind
	wantSequence threading.FrameWorkSequenceContext
	wantEvent    Event
	wantPayload  byte
	output       *frame.Frame
	err          error
}

func (r *frameWorkContextRunner) Run(ctx FrameWorkBatch) error {
	r.output = ctx.Output
	if ctx.Step.Kind != r.wantKind ||
		len(ctx.References) != 0 ||
		len(ctx.Jobs) != 1 ||
		len(ctx.Payload) != len(r.wantEvent.Unit.Payload) ||
		ctx.Jobs[0].Offset < 0 ||
		ctx.Jobs[0].Offset >= len(ctx.Payload) ||
		ctx.Payload[ctx.Jobs[0].Offset] != r.wantPayload ||
		ctx.Sequence != r.wantSequence ||
		ctx.FrameHeader != r.wantEvent.FrameHeader ||
		ctx.FrameSize != r.wantEvent.FrameSize ||
		ctx.TileInfo != r.wantEvent.TileInfo ||
		ctx.Segmentation != r.wantEvent.Segmentation ||
		ctx.LoopFilter != r.wantEvent.LoopFilter ||
		ctx.CDEF != r.wantEvent.CDEF ||
		ctx.Restoration != r.wantEvent.Restoration ||
		ctx.TransformRef != r.wantEvent.TransformRef ||
		ctx.SkipMode != r.wantEvent.SkipMode ||
		ctx.FrameMode != r.wantEvent.FrameMode ||
		ctx.GlobalMotion != r.wantEvent.GlobalMotion ||
		ctx.FilmGrain != r.wantEvent.FilmGrain {
		r.err = errors.New("unexpected frame-work runner context")
		return r.err
	}
	return nil
}

type frameWorkWritingRunner struct {
	order *[2]string
	state *FrameWorkState
	value byte
	err   error
}

func (r *frameWorkWritingRunner) Run(ctx FrameWorkBatch) error {
	if !r.state.Active() || ctx.Output == nil {
		r.err = errors.New("invalid tile runner state")
		return r.err
	}
	r.order[0] = "tile"
	ctx.Output.Y.Pix[0] = r.value
	return nil
}

type frameWorkCheckingPostRunner struct {
	order *[2]string
	refs  *SurfaceReferences
	state *FrameWorkState
	value byte
	err   error
}

func (r *frameWorkCheckingPostRunner) Apply(ctx FrameWorkPostFilterContext) error {
	if !r.state.Active() {
		r.err = errors.New("state inactive during postfilter runner")
		return r.err
	}
	if slot, ok := r.refs.ReferenceSlot(0); ok || slot != -1 {
		r.err = errors.New("published before postfilter runner")
		return r.err
	}
	if r.order[0] != "tile" || ctx.Output == nil || ctx.Output.Y.Pix[0] != 0x11 {
		r.err = errors.New("postfilter runner observed invalid tile output")
		return r.err
	}
	if ctx.Event.Kind != EventFrame || ctx.Step.Kind != FrameWorkStepBegin || !ctx.ExecutedTileWork || ctx.ReferenceCount != 0 {
		r.err = errors.New("invalid postfilter runner context")
		return r.err
	}
	r.order[1] = "post"
	ctx.Output.Y.Pix[0] = r.value
	return nil
}

func testUint16sEqual(a []uint16, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func showExistingWorkEvent(index uint8, frameType parser.FrameType) Event {
	return Event{
		Kind: EventExistingFrame,
		FrameHeader: parser.FrameHeaderPrefix{
			ShowExistingFrame: true,
			ExistingFrameIdx:  index,
		},
		ExistingFrame: parser.ReferenceFrame{FrameType: frameType},
	}
}

func makeFramePoolForSize(width uint32, height uint32, count int) (frame.Pool, error) {
	format := frame.Format{
		Width:        int(width),
		Height:       int(height),
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	}
	layout, err := frame.RequiredSize(format)
	if err != nil {
		return frame.Pool{}, err
	}
	backing := make([]byte, layout.Size*count)
	frames := make([]frame.Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := frame.BindPool(backing, format, frames, free, used)
	if err != nil {
		return frame.Pool{}, err
	}
	return pool, nil
}
