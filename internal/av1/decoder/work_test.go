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
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*count)
	frames := make([]frame.Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := frame.BindPool(backing, format, frames, free, used)
	if err != nil {
		t.Fatal(err)
	}
	return pool
}
