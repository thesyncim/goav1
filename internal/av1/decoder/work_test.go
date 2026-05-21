package decoder

import (
	"errors"
	"testing"

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
		for i := 0; i < len(batchJobs); i++ {
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
		for i := 0; i < len(batchJobs); i++ {
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
		for i := 0; i < len(batchJobs); i++ {
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
	for i := 0; i < parser.RefFrames; i++ {
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
		{Tile: 0, SBCols: 1, SBRows: 1},
		{Tile: 1, SBCols: 2, SBRows: 1},
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
		{Tile: 0, SBCols: 1, SBRows: 1},
		{Tile: 1, SBCols: 2, SBRows: 1},
	}
	var batches [2]threading.Batch
	n, err := threading.BuildBatches(batches[:], jobs[:], 1)
	if err != nil {
		b.Fatal(err)
	}
	return jobs, batches, n
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
