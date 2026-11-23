package goav1

import (
	"errors"
	"os"
	"testing"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

func TestDecoderDecodeNextReusesVisibleSlice(t *testing.T) {
	ivf, err := os.ReadFile("internal/av1/testvector/testdata/profiles/profile1-444-8bit-cdef-restoration-160x128.ivf")
	if err != nil {
		t.Fatalf("read clip: %v", err)
	}
	dec, err := NewDecoderFromIVF(ivf)
	if err != nil {
		t.Fatalf("NewDecoderFromIVF: %v", err)
	}
	defer dec.Close()

	first, ok, err := dec.DecodeNext()
	if err != nil {
		t.Fatalf("DecodeNext first: %v", err)
	}
	if !ok || len(first) == 0 {
		t.Fatalf("first DecodeNext ok=%v len=%d", ok, len(first))
	}
	firstSlot := &first[0]

	second, ok, err := dec.DecodeNext()
	if err != nil {
		t.Fatalf("DecodeNext second: %v", err)
	}
	if !ok || len(second) == 0 {
		t.Fatalf("second DecodeNext ok=%v len=%d", ok, len(second))
	}
	if secondSlot := &second[0]; secondSlot != firstSlot {
		t.Fatalf("DecodeNext visible slice backing changed between frames: %p != %p", secondSlot, firstSlot)
	}

	if err := dec.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	afterReset, ok, err := dec.DecodeNext()
	if err != nil {
		t.Fatalf("DecodeNext after reset: %v", err)
	}
	if !ok || len(afterReset) == 0 {
		t.Fatalf("after-reset DecodeNext ok=%v len=%d", ok, len(afterReset))
	}
	if resetSlot := &afterReset[0]; resetSlot != firstSlot {
		t.Fatalf("DecodeNext visible slice backing changed after reset: %p != %p", resetSlot, firstSlot)
	}
}

func TestDecoderFinishFrameSurface(t *testing.T) {
	format := FrameFormat{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := internalframe.RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*2)
	var frames [2]Frame
	var free [2]int
	var used [2]bool
	pool, err := internalframe.BindPool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	size := FrameSize{CodedWidth: 16, Height: 16}

	var refs internaldecoder.SurfaceReferences
	plan0, _, err := internaldecoder.BeginFrameWork(&refs, &pool, sequence, &internaldecoder.Event{
		Kind:        internaldecoder.EventFrameHeader,
		FrameHeader: internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeKey},
		FrameSize:   size,
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if plan0.Surface != 0 || plan0.ReferenceCount != 0 || plan0.Tile != (internaldecoder.TileWorkPlan{}) {
		t.Fatalf("plan0=%+v", plan0)
	}
	plan1, _, err := internaldecoder.BeginFrameWork(&refs, &pool, sequence, &internaldecoder.Event{
		Kind:        internaldecoder.EventFrameHeader,
		FrameHeader: internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeKey},
		FrameSize:   size,
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if plan1.Surface != 1 || plan1.ReferenceCount != 0 || plan1.Tile != (internaldecoder.TileWorkPlan{}) {
		t.Fatalf("plan1=%+v", plan1)
	}

	var releases [internalparser.RefFrames]int
	if _, err := refs.Refresh(0xff, plan0.Surface, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := internaldecoder.FinishFrameSurface(&refs, &pool, &internaldecoder.Event{
		Kind:      internaldecoder.EventTileGroup,
		FrameSize: FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: internalparser.TileGroup{Final: true},
	}, plan1.Surface, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || releases[0] != plan0.Surface {
		t.Fatalf("count=%d release=%d want 1,%d", count, releases[0], plan0.Surface)
	}
	if _, err := pool.Frame(plan0.Surface); !errors.Is(err, internalframe.ErrInvalidSlot) {
		t.Fatalf("released frame err=%v want %v", err, internalframe.ErrInvalidSlot)
	}
}

func TestPlanDecoderFrameTileWork(t *testing.T) {
	event := internaldecoder.Event{
		Kind: internaldecoder.EventTileGroup,
		Unit: internalobu.Unit{Payload: []byte{0x80}},
		TileInfo: internalparser.TileInfo{
			SBCols:     1,
			SBRows:     1,
			Cols:       1,
			Rows:       1,
			ColStartSB: [internalparser.MaxTileCols + 1]uint16{0, 1},
			RowStartSB: [internalparser.MaxTileRows + 1]uint16{0, 1},
		},
		TileGroup: internalparser.TileGroup{
			TileCount:  1,
			DataOffset: 0,
			DataSize:   1,
			Final:      true,
		},
	}
	var spans [1]internalparser.TileSpan
	var jobs [1]internaltile.Job
	var batches [1]internalthreading.Batch
	plan, err := internaldecoder.PlanFrameTileWork(&event, 3, 0, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan.Surface != 3 || plan.ReferenceCount != 0 || plan.Tile != (internaldecoder.TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestTileJobPayload(t *testing.T) {
	payload := []byte{0, 1, 2, 3}
	job := internaltile.Job{Offset: 1, Size: 2}

	data, err := job.Payload(payload)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 2 || data[0] != 1 || data[1] != 2 {
		t.Fatalf("payload=%v", data)
	}
}

func TestDecoderFrameWorkBatchJobPayload(t *testing.T) {
	ctx := internaldecoder.FrameWorkBatch{
		Payload: []byte{0, 1, 2, 3},
		Jobs:    []internaltile.Job{{Offset: 2, Size: 2}},
	}

	data, err := ctx.JobPayload(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 2 || data[0] != 2 || data[1] != 3 {
		t.Fatalf("payload=%v", data)
	}
}

func TestDecoderFrameWorkBatchPlaneHelpers(t *testing.T) {
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	sequenceContext := internalthreading.FrameWorkSequenceContextFromHeader(sequence)
	if !sequenceContext.Valid() || sequenceContext.ColorConfig != sequence.ColorConfig {
		t.Fatalf("sequence context=%+v", sequenceContext)
	}

	output := testDecoderBoundFrame(t, FrameFormat{
		Width:        16,
		Height:       16,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	})
	reference := testDecoderBoundFrame(t, FrameFormat{
		Width:        16,
		Height:       16,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	})
	references := [internalparser.InterRefsPerFrame]*Frame{}
	references[int(internaldecoder.FrameWorkReferenceLast)] = &reference
	ctx := internaldecoder.FrameWorkBatch{
		Output:     &output,
		References: references[:1],
		FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
			Sequence:  sequenceContext,
			FrameSize: FrameSize{CodedWidth: 16, Height: 16},
		},
		Jobs: []internaltile.Job{{SBCols: 1, SBRows: 1}},
	}

	outputPlane, err := ctx.JobOutputPlane(0, internaldecoder.FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	var typedOutput internaldecoder.FrameWorkPlaneRegion = outputPlane
	if typedOutput.Width != 16 || typedOutput.Height != 16 || typedOutput.Plane != internaldecoder.FrameWorkPlaneY {
		t.Fatalf("output plane=%+v", typedOutput)
	}

	referencePlane, err := ctx.ReferencePlane(internaldecoder.FrameWorkReferenceLast, internaldecoder.FrameWorkPlaneU)
	if err != nil {
		t.Fatal(err)
	}
	if referencePlane.Width != 8 || referencePlane.Height != 8 || referencePlane.Plane != internaldecoder.FrameWorkPlaneU {
		t.Fatalf("reference plane=%+v", referencePlane)
	}
	jobReferencePlane, err := ctx.JobReferencePlaneWindow(0, internaldecoder.FrameWorkReferenceLast, internaldecoder.FrameWorkPlaneU, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if jobReferencePlane.Width != 8 || jobReferencePlane.Height != 8 || jobReferencePlane.Plane != internaldecoder.FrameWorkPlaneU {
		t.Fatalf("job reference plane=%+v", jobReferencePlane)
	}
}

func TestDecoderEventDropsFrameWork(t *testing.T) {
	if !internaldecoder.EventDropsFrameWork(&internaldecoder.Event{Kind: internaldecoder.EventSequenceHeader, NewCodedVideoSequence: true}) {
		t.Fatal("new sequence did not drop frame work")
	}
	if !internaldecoder.EventDropsFrameWork(&internaldecoder.Event{Kind: internaldecoder.EventTemporalDelimiter, NewTemporalUnit: true}) {
		t.Fatal("temporal delimiter did not drop frame work")
	}
	if !internaldecoder.EventDropsFrameWork(&internaldecoder.Event{Kind: internaldecoder.EventExistingFrame}) {
		t.Fatal("show existing frame did not drop frame work")
	}
	if internaldecoder.EventDropsFrameWork(&internaldecoder.Event{Kind: internaldecoder.EventSequenceHeader, OperatingParametersChanged: true}) {
		t.Fatal("operating parameter change dropped frame work")
	}
}

func TestDecoderEventCompletesFrameWork(t *testing.T) {
	if !internaldecoder.EventCompletesFrameWork(&internaldecoder.Event{Kind: internaldecoder.EventFrame, TileGroup: internalparser.TileGroup{Final: true}}) {
		t.Fatal("final frame did not complete frame work")
	}
	if !internaldecoder.EventCompletesFrameWork(&internaldecoder.Event{Kind: internaldecoder.EventTileGroup, TileGroup: internalparser.TileGroup{Final: true}}) {
		t.Fatal("final tile group did not complete frame work")
	}
	if internaldecoder.EventCompletesFrameWork(&internaldecoder.Event{Kind: internaldecoder.EventTileGroup}) {
		t.Fatal("non-final tile group completed frame work")
	}
}

func TestDecoderEventOutputsFrame(t *testing.T) {
	if !internaldecoder.EventOutputsFrame(&internaldecoder.Event{Kind: internaldecoder.EventFrame, FrameHeader: internalparser.FrameHeaderPrefix{ShowFrame: true}, TileGroup: internalparser.TileGroup{Final: true}}) {
		t.Fatal("shown final frame did not output frame")
	}
	if internaldecoder.EventOutputsFrame(&internaldecoder.Event{Kind: internaldecoder.EventFrame, TileGroup: internalparser.TileGroup{Final: true}}) {
		t.Fatal("not-shown final frame output frame")
	}
	if !internaldecoder.EventOutputsFrame(&internaldecoder.Event{Kind: internaldecoder.EventExistingFrame}) {
		t.Fatal("show existing frame did not output frame")
	}
	if internaldecoder.EventOutputsFrame(&internaldecoder.Event{Kind: internaldecoder.EventTileGroup}) {
		t.Fatal("non-final tile group output frame")
	}
}

func TestDecoderFrameWorkState(t *testing.T) {
	pool := testDecoderFramePool(t, 1)
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	header := internaldecoder.Event{
		Kind:        internaldecoder.EventFrameHeader,
		FrameHeader: internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeKey},
		FrameSize:   FrameSize{CodedWidth: 16, Height: 16},
	}
	tileGroup := internaldecoder.Event{
		Kind:      internaldecoder.EventTileGroup,
		FrameSize: FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: internalparser.TileGroup{Final: true},
	}

	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	plan, output, err := state.Begin(&refs, &pool, sequence, &header, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || !state.Active() || plan.Surface != state.Surface {
		t.Fatalf("plan=%+v output=%p state=%+v active=%v", plan, output, state, state.Active())
	}

	var releases [internalparser.RefFrames]int
	count, err := state.Finish(&refs, &pool, &tileGroup, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 || state.Active() {
		t.Fatalf("count=%d active=%v", count, state.Active())
	}
}

func TestDecoderFrameWorkStatePostFilterContext(t *testing.T) {
	pool := testDecoderFramePool(t, 1)
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	header := internaldecoder.Event{
		Kind:        internaldecoder.EventFrameHeader,
		FrameHeader: internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeKey},
		FrameSize:   FrameSize{CodedWidth: 16, Height: 16},
	}
	tileGroup := internaldecoder.Event{
		Kind:      internaldecoder.EventTileGroup,
		FrameSize: FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: internalparser.TileGroup{Final: true},
	}

	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	plan, output, err := state.Begin(&refs, &pool, sequence, &header, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	step := internaldecoder.FrameWorkStep{
		Kind: internaldecoder.FrameWorkStepTile,
		Tile: internaldecoder.FrameTileWorkPlan{Surface: plan.Surface},
	}

	ctx, err := state.PostFilterContext(&pool, &tileGroup, step, false)
	if err != nil {
		t.Fatal(err)
	}
	if ctx.Output != output ||
		ctx.Event.Kind != internaldecoder.EventTileGroup ||
		!ctx.Event.TileGroup.Final ||
		ctx.Step != step ||
		ctx.ExecutedTileWork ||
		ctx.ReferenceCount != 0 ||
		ctx.CDEFIndexMap != nil ||
		ctx.LoopFilterMap != nil ||
		ctx.RestorationFrameBuffers != nil ||
		!state.Active() {
		t.Fatalf("ctx=%+v output=%p state active=%v", ctx, output, state.Active())
	}

	nonFinal := tileGroup
	nonFinal.TileGroup.Final = false
	noop, err := state.PostFilterContext(&pool, &nonFinal, step, false)
	if err != nil {
		t.Fatal(err)
	}
	if noop.Output != nil || noop.Event.Kind != internaldecoder.EventIgnored || noop.Step.Kind != internaldecoder.FrameWorkStepIgnored {
		t.Fatalf("non-final context=%+v want zero", noop)
	}
}

func TestDecoderFrameWorkStateFinishIfEventCompletesFrameWork(t *testing.T) {
	pool := testDecoderFramePool(t, 1)
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	header := internaldecoder.Event{
		Kind:        internaldecoder.EventFrameHeader,
		FrameHeader: internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeKey},
		FrameSize:   FrameSize{CodedWidth: 16, Height: 16},
	}
	tileGroup := internaldecoder.Event{
		Kind:      internaldecoder.EventTileGroup,
		FrameSize: FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: internalparser.TileGroup{Final: true},
	}

	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	if _, _, err := state.Begin(&refs, &pool, sequence, &header, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}

	var releases [internalparser.RefFrames]int
	completed, count, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, &tileGroup, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if !completed || count != 0 || state.Active() {
		t.Fatalf("completed=%v count=%d active=%v", completed, count, state.Active())
	}
}

func TestDecoderFrameWorkStateShowExisting(t *testing.T) {
	pool := testDecoderFramePool(t, 2)
	reference, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs internaldecoder.SurfaceReferences
	var releases [internalparser.RefFrames]int
	if _, err := refs.Refresh(1<<0, reference, releases[:]); err != nil {
		t.Fatal(err)
	}

	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	header := internaldecoder.Event{
		Kind:        internaldecoder.EventFrameHeader,
		FrameHeader: internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeKey},
		FrameSize:   FrameSize{CodedWidth: 16, Height: 16},
	}

	var state internaldecoder.FrameWorkState
	if _, _, err := state.Begin(&refs, &pool, sequence, &header, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}

	plan, err := state.ShowExisting(&refs, &pool, &internaldecoder.Event{
		Kind: internaldecoder.EventExistingFrame,
		FrameHeader: internalparser.FrameHeaderPrefix{
			ShowExistingFrame: true,
			ExistingFrameIdx:  0,
		},
		ExistingFrame: internalparser.ReferenceFrame{FrameType: internalparser.FrameTypeInter},
	}, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan != (internaldecoder.ShowExistingFrameWorkPlan{Surface: reference, DroppedFrameWork: true}) || state.Active() {
		t.Fatalf("plan=%+v active=%v", plan, state.Active())
	}
}

func TestDecoderFrameWorkStatePlanEvent(t *testing.T) {
	pool := testDecoderFramePool(t, 1)
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	header := internaldecoder.Event{
		Kind:        internaldecoder.EventFrameHeader,
		FrameHeader: internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeKey},
		FrameSize:   FrameSize{CodedWidth: 16, Height: 16},
	}
	tileGroup := internaldecoder.Event{
		Kind:      internaldecoder.EventTileGroup,
		FrameSize: FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: internalparser.TileGroup{Final: true},
	}

	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	step, output, err := state.PlanEvent(&refs, &pool, sequence, &header, 32, nil, 1, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || step.Kind != internaldecoder.FrameWorkStepBegin || step.Begin.Surface != state.Surface {
		t.Fatalf("step=%+v output=%p state=%+v", step, output, state)
	}

	var releases [internalparser.RefFrames]int
	completed, count, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, &tileGroup, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if !completed || count != 0 || state.Active() {
		t.Fatalf("completed=%v count=%d active=%v", completed, count, state.Active())
	}
}

func TestDecoderFrameWorkStateRunStep(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testDecoderFramePool(t, 1)
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	header := internaldecoder.Event{
		Kind:        internaldecoder.EventFrameHeader,
		FrameHeader: internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeKey},
		FrameSize:   FrameSize{CodedWidth: 16, Height: 16},
	}
	tileGroup := internaldecoder.Event{
		Kind:      internaldecoder.EventTileGroup,
		FrameSize: FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: internalparser.TileGroup{Final: true},
	}
	jobs := [1]internaltile.Job{{Tile: 0, SBCols: 1, SBRows: 1}}
	batches := [1]internalthreading.Batch{{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0, Units: 1}}
	step := internaldecoder.FrameWorkStep{
		Kind: internaldecoder.FrameWorkStepTile,
		Tile: internaldecoder.FrameTileWorkPlan{Tile: internaldecoder.TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}},
	}

	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	if _, _, err := state.Begin(&refs, &pool, sequence, &header, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	var releases [internalparser.RefFrames]int
	result, err := state.RunStep(&refs, &pool, &tileGroup, step, workerPool, jobs[:], batches[:], releases[:], func(internalthreading.Batch, []internaltile.Job) error {
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result != (internaldecoder.FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) || state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
}

func TestResolveDecoderFrameReferences(t *testing.T) {
	pool := testDecoderFramePool(t, 1)
	surface, frame, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs [internalparser.InterRefsPerFrame]*Frame
	count, err := internaldecoder.ResolveFrameReferences(&pool, []int{surface}, refs[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || refs[0] != frame {
		t.Fatalf("count=%d ref=%p want %p", count, refs[0], frame)
	}
}

func TestDecoderFrameWorkStateAbort(t *testing.T) {
	pool := testDecoderFramePool(t, 1)
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	header := internaldecoder.Event{
		Kind:        internaldecoder.EventFrameHeader,
		FrameHeader: internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeKey},
		FrameSize:   FrameSize{CodedWidth: 16, Height: 16},
	}

	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	plan, _, err := state.Begin(&refs, &pool, sequence, &header, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := state.Abort(&pool); err != nil {
		t.Fatal(err)
	}
	if state.Active() || pool.Available() != 1 {
		t.Fatalf("state=%+v active=%v available=%d", state, state.Active(), pool.Available())
	}
	if _, err := pool.Frame(plan.Surface); !errors.Is(err, internalframe.ErrInvalidSlot) {
		t.Fatalf("aborted frame err=%v want %v", err, internalframe.ErrInvalidSlot)
	}
}

func testDecoderFramePool(t *testing.T, count int) internalframe.Pool {
	t.Helper()
	format := FrameFormat{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := internalframe.RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*count)
	frames := make([]Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := internalframe.BindPool(backing, format, frames, free, used)
	if err != nil {
		t.Fatal(err)
	}
	return pool
}

func testDecoderBoundFrame(t *testing.T, format FrameFormat) Frame {
	t.Helper()
	layout, err := internalframe.RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	buffer := make([]byte, layout.Size)
	output, err := internalframe.Bind(buffer, format)
	if err != nil {
		t.Fatal(err)
	}
	return output
}
