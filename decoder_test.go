package goav1

import (
	"errors"
	"testing"
)

func TestDecoderFinishFrameSurface(t *testing.T) {
	format := FrameFormat{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := FrameRequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*2)
	var frames [2]Frame
	var free [2]int
	var used [2]bool
	pool, err := BindFramePool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence := SequenceHeader{ColorConfig: ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	size := FrameSize{CodedWidth: 16, Height: 16}

	var refs DecoderSurfaceReferences
	plan0, _, err := BeginDecoderFrameWork(&refs, &pool, sequence, DecoderEvent{
		Kind:        DecoderEventFrameHeader,
		FrameHeader: FrameHeaderPrefix{FrameType: FrameTypeKey},
		FrameSize:   size,
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if plan0.Surface != 0 || plan0.ReferenceCount != 0 || plan0.Tile != (DecoderTileWorkPlan{}) {
		t.Fatalf("plan0=%+v", plan0)
	}
	plan1, _, err := BeginDecoderFrameWork(&refs, &pool, sequence, DecoderEvent{
		Kind:        DecoderEventFrameHeader,
		FrameHeader: FrameHeaderPrefix{FrameType: FrameTypeKey},
		FrameSize:   size,
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if plan1.Surface != 1 || plan1.ReferenceCount != 0 || plan1.Tile != (DecoderTileWorkPlan{}) {
		t.Fatalf("plan1=%+v", plan1)
	}

	var releases [RefFrames]int
	if _, err := refs.Refresh(0xff, plan0.Surface, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := DecoderFinishFrameSurface(&refs, &pool, DecoderEvent{
		Kind:      DecoderEventTileGroup,
		FrameSize: FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: TileGroup{Final: true},
	}, plan1.Surface, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || releases[0] != plan0.Surface {
		t.Fatalf("count=%d release=%d want 1,%d", count, releases[0], plan0.Surface)
	}
	if _, err := pool.Frame(plan0.Surface); !errors.Is(err, ErrFrameInvalidSlot) {
		t.Fatalf("released frame err=%v want %v", err, ErrFrameInvalidSlot)
	}
}

func TestPlanDecoderFrameTileWork(t *testing.T) {
	event := DecoderEvent{
		Kind: DecoderEventTileGroup,
		Unit: OBUUnit{Payload: []byte{0x80}},
		TileInfo: TileInfo{
			SBCols:     1,
			SBRows:     1,
			Cols:       1,
			Rows:       1,
			ColStartSB: [MaxTileCols + 1]uint16{0, 1},
			RowStartSB: [MaxTileRows + 1]uint16{0, 1},
		},
		TileGroup: TileGroup{
			TileCount:  1,
			DataOffset: 0,
			DataSize:   1,
			Final:      true,
		},
	}
	var spans [1]TileSpan
	var jobs [1]TileJob
	var batches [1]TileBatch
	plan, err := PlanDecoderFrameTileWork(event, 3, 0, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan.Surface != 3 || plan.ReferenceCount != 0 || plan.Tile != (DecoderTileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestTileJobPayload(t *testing.T) {
	payload := []byte{0, 1, 2, 3}
	job := TileJob{Offset: 1, Size: 2}

	data, err := job.Payload(payload)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 2 || data[0] != 1 || data[1] != 2 {
		t.Fatalf("payload=%v", data)
	}
}

func TestDecoderFrameWorkBatchJobPayload(t *testing.T) {
	ctx := DecoderFrameWorkBatch{
		Payload: []byte{0, 1, 2, 3},
		Jobs:    []TileJob{{Offset: 2, Size: 2}},
	}

	data, err := ctx.JobPayload(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 2 || data[0] != 2 || data[1] != 3 {
		t.Fatalf("payload=%v", data)
	}
}

func TestDecoderEventDropsFrameWork(t *testing.T) {
	if !DecoderEventDropsFrameWork(DecoderEvent{Kind: DecoderEventSequenceHeader, NewCodedVideoSequence: true}) {
		t.Fatal("new sequence did not drop frame work")
	}
	if !DecoderEventDropsFrameWork(DecoderEvent{Kind: DecoderEventTemporalDelimiter, NewTemporalUnit: true}) {
		t.Fatal("temporal delimiter did not drop frame work")
	}
	if !DecoderEventDropsFrameWork(DecoderEvent{Kind: DecoderEventExistingFrame}) {
		t.Fatal("show existing frame did not drop frame work")
	}
	if DecoderEventDropsFrameWork(DecoderEvent{Kind: DecoderEventSequenceHeader, OperatingParametersChanged: true}) {
		t.Fatal("operating parameter change dropped frame work")
	}
}

func TestDecoderEventCompletesFrameWork(t *testing.T) {
	if !DecoderEventCompletesFrameWork(DecoderEvent{Kind: DecoderEventFrame, TileGroup: TileGroup{Final: true}}) {
		t.Fatal("final frame did not complete frame work")
	}
	if !DecoderEventCompletesFrameWork(DecoderEvent{Kind: DecoderEventTileGroup, TileGroup: TileGroup{Final: true}}) {
		t.Fatal("final tile group did not complete frame work")
	}
	if DecoderEventCompletesFrameWork(DecoderEvent{Kind: DecoderEventTileGroup}) {
		t.Fatal("non-final tile group completed frame work")
	}
}

func TestDecoderFrameWorkState(t *testing.T) {
	pool := testDecoderFramePool(t, 1)
	sequence := SequenceHeader{ColorConfig: ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	header := DecoderEvent{
		Kind:        DecoderEventFrameHeader,
		FrameHeader: FrameHeaderPrefix{FrameType: FrameTypeKey},
		FrameSize:   FrameSize{CodedWidth: 16, Height: 16},
	}
	tileGroup := DecoderEvent{
		Kind:      DecoderEventTileGroup,
		FrameSize: FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: TileGroup{Final: true},
	}

	var refs DecoderSurfaceReferences
	var state DecoderFrameWorkState
	plan, output, err := state.Begin(&refs, &pool, sequence, header, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || !state.Active() || plan.Surface != state.Surface {
		t.Fatalf("plan=%+v output=%p state=%+v active=%v", plan, output, state, state.Active())
	}

	var releases [RefFrames]int
	count, err := state.Finish(&refs, &pool, tileGroup, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 || state.Active() {
		t.Fatalf("count=%d active=%v", count, state.Active())
	}
}

func TestDecoderFrameWorkStateFinishIfEventCompletesFrameWork(t *testing.T) {
	pool := testDecoderFramePool(t, 1)
	sequence := SequenceHeader{ColorConfig: ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	header := DecoderEvent{
		Kind:        DecoderEventFrameHeader,
		FrameHeader: FrameHeaderPrefix{FrameType: FrameTypeKey},
		FrameSize:   FrameSize{CodedWidth: 16, Height: 16},
	}
	tileGroup := DecoderEvent{
		Kind:      DecoderEventTileGroup,
		FrameSize: FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: TileGroup{Final: true},
	}

	var refs DecoderSurfaceReferences
	var state DecoderFrameWorkState
	if _, _, err := state.Begin(&refs, &pool, sequence, header, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}

	var releases [RefFrames]int
	completed, count, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, tileGroup, releases[:])
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

	var refs DecoderSurfaceReferences
	var releases [RefFrames]int
	if _, err := refs.Refresh(1<<0, reference, releases[:]); err != nil {
		t.Fatal(err)
	}

	sequence := SequenceHeader{ColorConfig: ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	header := DecoderEvent{
		Kind:        DecoderEventFrameHeader,
		FrameHeader: FrameHeaderPrefix{FrameType: FrameTypeKey},
		FrameSize:   FrameSize{CodedWidth: 16, Height: 16},
	}

	var state DecoderFrameWorkState
	if _, _, err := state.Begin(&refs, &pool, sequence, header, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}

	plan, err := state.ShowExisting(&refs, &pool, DecoderEvent{
		Kind: DecoderEventExistingFrame,
		FrameHeader: FrameHeaderPrefix{
			ShowExistingFrame: true,
			ExistingFrameIdx:  0,
		},
		ExistingFrame: ReferenceFrame{FrameType: FrameTypeInter},
	}, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan != (DecoderShowExistingFrameWorkPlan{Surface: reference, DroppedFrameWork: true}) || state.Active() {
		t.Fatalf("plan=%+v active=%v", plan, state.Active())
	}
}

func TestDecoderFrameWorkStatePlanEvent(t *testing.T) {
	pool := testDecoderFramePool(t, 1)
	sequence := SequenceHeader{ColorConfig: ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	header := DecoderEvent{
		Kind:        DecoderEventFrameHeader,
		FrameHeader: FrameHeaderPrefix{FrameType: FrameTypeKey},
		FrameSize:   FrameSize{CodedWidth: 16, Height: 16},
	}
	tileGroup := DecoderEvent{
		Kind:      DecoderEventTileGroup,
		FrameSize: FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: TileGroup{Final: true},
	}

	var refs DecoderSurfaceReferences
	var state DecoderFrameWorkState
	step, output, err := state.PlanEvent(&refs, &pool, sequence, header, 32, nil, 1, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || step.Kind != DecoderFrameWorkStepBegin || step.Begin.Surface != state.Surface {
		t.Fatalf("step=%+v output=%p state=%+v", step, output, state)
	}

	var releases [RefFrames]int
	completed, count, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, tileGroup, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if !completed || count != 0 || state.Active() {
		t.Fatalf("completed=%v count=%d active=%v", completed, count, state.Active())
	}
}

func TestExecuteDecoderFrameWorkStep(t *testing.T) {
	workerPool, err := NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs := [1]TileJob{{Tile: 0, SBCols: 1, SBRows: 1}}
	batches := [1]TileBatch{{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0, Units: 1}}
	step := DecoderFrameWorkStep{
		Kind: DecoderFrameWorkStepTile,
		Tile: DecoderFrameTileWorkPlan{Tile: DecoderTileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}},
	}
	seen := false
	executed, err := ExecuteDecoderFrameWorkStep(step, workerPool, jobs[:], batches[:], func(batch TileBatch, batchJobs []TileJob) error {
		seen = batch.Count == 1 && len(batchJobs) == 1 && batchJobs[0].Tile == 0
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !executed || !seen {
		t.Fatalf("executed=%v seen=%v", executed, seen)
	}
}

func TestExecuteDecoderFrameWorkStepWithContext(t *testing.T) {
	workerPool, err := NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs := [1]TileJob{{Tile: 0, SBCols: 1, SBRows: 1}}
	batches := [1]TileBatch{{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0, Units: 1}}
	step := DecoderFrameWorkStep{
		Kind: DecoderFrameWorkStepTile,
		Tile: DecoderFrameTileWorkPlan{
			ReferenceCount: 1,
			Tile:           DecoderTileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1},
		},
	}
	var output Frame
	var reference Frame
	references := [InterRefsPerFrame]*Frame{&reference}
	seen := false
	executed, err := ExecuteDecoderFrameWorkStepWithContext(step, workerPool, &output, references[:], jobs[:], batches[:], func(ctx DecoderFrameWorkBatch) error {
		seen = ctx.Step == step &&
			ctx.Output == &output &&
			len(ctx.References) == 1 &&
			ctx.References[0] == &reference &&
			ctx.Batch.Count == 1 &&
			len(ctx.Jobs) == 1 &&
			ctx.Jobs[0].Tile == 0
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !executed || !seen {
		t.Fatalf("executed=%v seen=%v", executed, seen)
	}
}

func TestExecuteDecoderFrameWorkStepWithPayload(t *testing.T) {
	workerPool, err := NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs := [1]TileJob{{Tile: 0, SBCols: 1, SBRows: 1}}
	batches := [1]TileBatch{{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0, Units: 1}}
	step := DecoderFrameWorkStep{
		Kind: DecoderFrameWorkStepTile,
		Tile: DecoderFrameTileWorkPlan{Tile: DecoderTileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}},
	}
	payload := []byte{0xab}
	seen := false
	executed, err := ExecuteDecoderFrameWorkStepWithPayload(step, workerPool, nil, nil, payload, jobs[:], batches[:], func(ctx DecoderFrameWorkBatch) error {
		seen = len(ctx.Payload) == 1 && ctx.Payload[0] == 0xab && len(ctx.Jobs) == 1
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !executed || !seen {
		t.Fatalf("executed=%v seen=%v", executed, seen)
	}
}

func TestRunDecoderFrameWorkEventWithContext(t *testing.T) {
	workerPool, err := NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testDecoderFramePool(t, 1)
	sequence := SequenceHeader{ColorConfig: ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	event := DecoderEvent{
		Kind:        DecoderEventFrame,
		Unit:        OBUUnit{Payload: []byte{0x80}},
		FrameHeader: FrameHeaderPrefix{FrameType: FrameTypeKey},
		FrameSize:   FrameSize{CodedWidth: 16, Height: 16, RefreshFrameFlags: 0xff},
		TileInfo: TileInfo{
			SBCols:     1,
			SBRows:     1,
			Cols:       1,
			Rows:       1,
			ColStartSB: [MaxTileCols + 1]uint16{0, 1},
			RowStartSB: [MaxTileRows + 1]uint16{0, 1},
		},
		TileGroup: TileGroup{
			TileCount:  1,
			DataOffset: 0,
			DataSize:   1,
			Final:      true,
		},
	}
	var refs DecoderSurfaceReferences
	var state DecoderFrameWorkState
	var referenceSurfaces [InterRefsPerFrame]int
	var referenceFrames [InterRefsPerFrame]*Frame
	var spans [1]TileSpan
	var jobs [1]TileJob
	var batches [1]TileBatch
	var releases [RefFrames]int

	var seenOutput *Frame
	result, err := RunDecoderFrameWorkEventWithContext(&state, &refs, &pool, sequence, event, 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx DecoderFrameWorkBatch) error {
		seenOutput = ctx.Output
		if len(ctx.Payload) != 1 || ctx.Payload[0] != 0x80 {
			t.Fatalf("payload=%v", ctx.Payload)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Step.Kind != DecoderFrameWorkStepBegin ||
		result.Output == nil ||
		result.Output != seenOutput ||
		result.Run != (DecoderFrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		state.Active() {
		t.Fatalf("result=%+v seen=%p active=%v", result, seenOutput, state.Active())
	}
}

func TestDecoderFrameWorkStateRunStep(t *testing.T) {
	workerPool, err := NewTileWorkerPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testDecoderFramePool(t, 1)
	sequence := SequenceHeader{ColorConfig: ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	header := DecoderEvent{
		Kind:        DecoderEventFrameHeader,
		FrameHeader: FrameHeaderPrefix{FrameType: FrameTypeKey},
		FrameSize:   FrameSize{CodedWidth: 16, Height: 16},
	}
	tileGroup := DecoderEvent{
		Kind:      DecoderEventTileGroup,
		FrameSize: FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: TileGroup{Final: true},
	}
	jobs := [1]TileJob{{Tile: 0, SBCols: 1, SBRows: 1}}
	batches := [1]TileBatch{{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0, Units: 1}}
	step := DecoderFrameWorkStep{
		Kind: DecoderFrameWorkStepTile,
		Tile: DecoderFrameTileWorkPlan{Tile: DecoderTileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}},
	}

	var refs DecoderSurfaceReferences
	var state DecoderFrameWorkState
	if _, _, err := state.Begin(&refs, &pool, sequence, header, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	var releases [RefFrames]int
	result, err := state.RunStep(&refs, &pool, tileGroup, step, workerPool, jobs[:], batches[:], releases[:], func(TileBatch, []TileJob) error {
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result != (DecoderFrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) || state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
}

func TestResolveDecoderFrameReferences(t *testing.T) {
	pool := testDecoderFramePool(t, 1)
	surface, frame, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs [InterRefsPerFrame]*Frame
	count, err := ResolveDecoderFrameReferences(&pool, []int{surface}, refs[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || refs[0] != frame {
		t.Fatalf("count=%d ref=%p want %p", count, refs[0], frame)
	}
}

func TestDecoderFrameWorkStateAbort(t *testing.T) {
	pool := testDecoderFramePool(t, 1)
	sequence := SequenceHeader{ColorConfig: ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	header := DecoderEvent{
		Kind:        DecoderEventFrameHeader,
		FrameHeader: FrameHeaderPrefix{FrameType: FrameTypeKey},
		FrameSize:   FrameSize{CodedWidth: 16, Height: 16},
	}

	var refs DecoderSurfaceReferences
	var state DecoderFrameWorkState
	plan, _, err := state.Begin(&refs, &pool, sequence, header, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := state.Abort(&pool); err != nil {
		t.Fatal(err)
	}
	if state.Active() || pool.Available() != 1 {
		t.Fatalf("state=%+v active=%v available=%d", state, state.Active(), pool.Available())
	}
	if _, err := pool.Frame(plan.Surface); !errors.Is(err, ErrFrameInvalidSlot) {
		t.Fatalf("aborted frame err=%v want %v", err, ErrFrameInvalidSlot)
	}
}

func testDecoderFramePool(t *testing.T, count int) FramePool {
	t.Helper()
	format := FrameFormat{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := FrameRequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*count)
	frames := make([]Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := BindFramePool(backing, format, frames, free, used)
	if err != nil {
		t.Fatal(err)
	}
	return pool
}
