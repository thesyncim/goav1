package goav1

import (
	"bytes"
	"errors"
	"testing"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

func TestPublicDecoderFrameWorkResidualScratchSizeMax(t *testing.T) {
	batchA := decoderFrameWorkBatchResidualScratchSize{
		Int32Scratch:     1,
		ResidualScratch:  7,
		LoopContextAbove: 3,
	}
	batchB := decoderFrameWorkBatchResidualScratchSize{
		Int32Scratch:     5,
		ResidualScratch:  2,
		LoopContextAbove: 4,
	}
	if got, want := batchA.Max(batchB), (decoderFrameWorkBatchResidualScratchSize{
		Int32Scratch:     5,
		ResidualScratch:  7,
		LoopContextAbove: 4,
	}); got != want {
		t.Fatalf("batch residual scratch max=%+v want %+v", got, want)
	}

	runnerA := decoderFrameWorkBatchResidualRunnerScratchSize{
		Workers:             1,
		Batch:               batchA,
		RestorationRequests: 1,
		Int32Scratch:        1,
		ResidualScratch:     7,
		LoopContextAbove:    3,
	}
	runnerB := decoderFrameWorkBatchResidualRunnerScratchSize{
		Workers:             3,
		Batch:               batchB,
		RestorationRequests: 2,
		Int32Scratch:        15,
		ResidualScratch:     6,
		LoopContextAbove:    12,
	}
	if got, want := runnerA.Max(runnerB), (decoderFrameWorkBatchResidualRunnerScratchSize{
		Workers:             3,
		Batch:               batchA.Max(batchB),
		RestorationRequests: 2,
		Int32Scratch:        15,
		ResidualScratch:     7,
		LoopContextAbove:    12,
	}); got != want {
		t.Fatalf("runner residual scratch max=%+v want %+v", got, want)
	}

	eventA := decoderFrameWorkResidualEventScratchSize{
		Runner:   runnerA,
		SideData: decoderFrameWorkSideDataScratchSize{CDEFIndexMap: 2, LoopFilterMap: 8},
		Plan:     internaldecoder.TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: 1},
		Outputs:  1,
	}
	eventB := decoderFrameWorkResidualEventScratchSize{
		Runner:   runnerB,
		SideData: decoderFrameWorkSideDataScratchSize{CDEFReadMap: 4, LoopFilterMap: 3},
		Plan:     internaldecoder.TileWorkPlan{SpanCount: 1, JobCount: 5, BatchCount: 2},
		Outputs:  3,
	}
	got := eventA.Max(eventB)
	if got.Runner != runnerA.Max(runnerB) ||
		got.SideData != (decoderFrameWorkSideDataScratchSize{CDEFIndexMap: 2, CDEFReadMap: 4, LoopFilterMap: 8}) ||
		got.Plan != (internaldecoder.TileWorkPlan{SpanCount: 2, JobCount: 5, BatchCount: 2}) ||
		got.Outputs != 3 {
		t.Fatalf("event residual scratch max=%+v", got)
	}

	streamA := decoderFrameWorkResidualStreamScratchSize{
		Events:    2,
		RTPBuffer: 16,
		RTPSpans:  1,
		Event:     eventA,
	}
	streamB := decoderFrameWorkResidualStreamScratchSize{
		Events:    4,
		RTPBuffer: 8,
		RTPSpans:  3,
		Event:     eventB,
	}
	streamGot := streamA.Max(streamB)
	if streamGot.Events != 4 ||
		streamGot.RTPBuffer != 16 ||
		streamGot.RTPSpans != 3 ||
		streamGot.Event != eventA.Max(eventB) {
		t.Fatalf("stream residual scratch max=%+v", streamGot)
	}

	sequenceA := SequenceHeader{ColorConfig: internalparser.ColorConfig{BitDepth: 8}}
	sequenceB := SequenceHeader{ColorConfig: internalparser.ColorConfig{BitDepth: 10}}
	planA := decoderFrameWorkResidualStreamPlan{
		Size: streamA,
		Bind: decoderFrameWorkResidualEventBindPlan{
			Sequence:   sequenceA,
			Event:      internaldecoder.Event{Kind: internaldecoder.EventExistingFrame},
			EventIndex: 0,
		},
	}
	planB := decoderFrameWorkResidualStreamPlan{
		Size: streamB,
		Bind: decoderFrameWorkResidualEventBindPlan{
			Sequence:   sequenceB,
			Event:      internaldecoder.Event{Kind: internaldecoder.EventTileGroup},
			EventIndex: 1,
		},
	}
	planGot := planA.Max(planB)
	if !planGot.HasEvent() ||
		planGot.Size != streamGot ||
		planGot.Bind.Event.Kind != internaldecoder.EventTileGroup ||
		planGot.Bind.Sequence.ColorConfig.BitDepth != 10 ||
		planGot.Bind.EventIndex != 1 {
		t.Fatalf("stream residual plan max=%+v", planGot)
	}
}

func TestPublicDecoderFrameWorkResidualStreamScratchCheckAndState(t *testing.T) {
	size := decoderFrameWorkResidualStreamScratchSize{
		Events:    2,
		RTPBuffer: 16,
		RTPSpans:  2,
		Event: decoderFrameWorkResidualEventScratchSize{
			Runner: decoderFrameWorkBatchResidualRunnerScratchSize{
				Workers:         1,
				Int32Scratch:    4,
				ResidualScratch: 4,
			},
			SideData: decoderFrameWorkSideDataScratchSize{
				CDEFIndexMap:  4,
				LoopFilterMap: 4,
			},
			Plan:    internaldecoder.TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1},
			Outputs: 2,
		},
	}
	scratch := publicDecoderResidualStreamScratch(size)
	if err := scratch.Check(size); err != nil {
		t.Fatal(err)
	}
	shortEvents := scratch
	shortEvents.Events = shortEvents.Events[:size.Events-1]
	if err := shortEvents.Check(size); !errors.Is(err, ErrDecoderEventBufferTooSmall) {
		t.Fatalf("short events err=%v want %v", err, ErrDecoderEventBufferTooSmall)
	}
	shortRTP := scratch
	shortRTP.RTPBuffer = shortRTP.RTPBuffer[:size.RTPBuffer-1]
	if err := shortRTP.Check(size); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short RTP err=%v want %v", err, ErrRTPShortBuffer)
	}
	shortOutputs := scratch
	shortOutputs.Outputs = shortOutputs.Outputs[:size.Event.Outputs-1]
	if err := shortOutputs.Check(size); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short outputs err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	shortEventScratch := scratch
	shortEventScratch.Event.Spans = shortEventScratch.Event.Spans[:0]
	if err := shortEventScratch.Check(size); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short event scratch err=%v want %v", err, internalframe.ErrShortBuffer)
	}

	var nilRunner *decoderFrameWorkResidualStreamRunner
	if state := nilRunner.State(); state != (decoderFrameWorkResidualStreamRunnerState{}) {
		t.Fatalf("nil runner state=%+v", state)
	}
	if _, ok := nilRunner.SequenceHeader(); ok {
		t.Fatal("nil runner reported sequence header")
	}

	var stream internaldecoder.Stream
	var events [4]internaldecoder.Event
	if count, err := stream.PushLowOverhead(publicDecoderResidualLowOverheadStream(), events[:]); err != nil {
		t.Fatal(err)
	} else if count != 3 || !stream.HasSequenceHeader() {
		t.Fatalf("low-overhead count=%d hasSequence=%v", count, stream.HasSequenceHeader())
	}
	runner := decoderFrameWorkResidualStreamRunner{
		Stream: &stream,
		EventRunner: decoderFrameWorkResidualEventRunner{
			Outputs: scratch.Outputs[:size.Event.Outputs],
		},
		Events:    scratch.Events[:size.Events],
		RTPBuffer: scratch.RTPBuffer[:size.RTPBuffer],
		RTPSpans:  scratch.RTPSpans[:size.RTPSpans],
	}
	state := runner.State()
	if !state.Bound ||
		!state.HasSequenceHeader ||
		state.InRTPFragment ||
		state.RTPUsed != 0 ||
		state.EventCapacity != size.Events ||
		state.RTPBufferCapacity != size.RTPBuffer ||
		state.RTPSpanCapacity != size.RTPSpans ||
		state.OutputCapacity != size.Event.Outputs {
		t.Fatalf("runner state=%+v", state)
	}
	sequence, ok := runner.SequenceHeader()
	if !ok || sequence.ColorConfig.BitDepth != 8 {
		t.Fatalf("sequence ok=%v header=%+v", ok, sequence)
	}

	frameHeader := publicDecoderResidualRTPElement(internalobu.TypeFrameHeader, publicDecoderResidualFrameHeaderPayload())
	var packet [8]byte
	n, _, more, err := PutRTPFragment(packet[:], frameHeader, 0, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	if !more {
		t.Fatal("expected retained RTP fragment")
	}
	used, count, err := stream.PushRTPPayload(runner.RTPBuffer, 0, runner.RTPSpans, runner.Events, packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	runner.RTPUsed = used
	state = runner.State()
	if count != 0 || !state.InRTPFragment || state.RTPUsed == 0 {
		t.Fatalf("fragment count=%d state=%+v", count, state)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerRequiresRunnerForTileList(t *testing.T) {
	var eventRunner decoderFrameWorkResidualEventRunner
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{BitDepth: 8, MonoChrome: true}}
	event := internaldecoder.Event{
		Kind: internaldecoder.EventTileList,
		FrameSize: FrameSize{
			CodedWidth:    64,
			UpscaledWidth: 64,
			Height:        64,
		},
		TileInfo: internalparser.TileInfo{
			Cols:       1,
			Rows:       1,
			ColStartSB: [internalparser.MaxTileCols + 1]uint16{0, 1},
			RowStartSB: [internalparser.MaxTileRows + 1]uint16{0, 1},
		},
		TileList: internalparser.TileList{
			OutputFrameWidthInTilesMinus1:  0,
			OutputFrameHeightInTilesMinus1: 0,
			TileCountMinus1:                0,
			Entries: []internalparser.TileListEntry{{
				AnchorFrameIdx:     0,
				AnchorTileRow:      0,
				AnchorTileCol:      0,
				TileDataSizeMinus1: 0,
				TileData:           []byte{0x80},
			}},
		},
	}

	result, err := eventRunner.RunEvents(sequence, []internaldecoder.Event{event}, decoderFrameWorkSideDataScratch{}, nil)
	if !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("RunEvents err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	if result.Count != 0 || result.OutputCount != 0 || result.CompletedFrames != 0 {
		t.Fatalf("result after rejected tile list=%+v", result)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerPropagatesTileListParseError(t *testing.T) {
	var eventRunner decoderFrameWorkResidualEventRunner

	_, err := eventRunner.RunEvents(SequenceHeader{}, []internaldecoder.Event{
		{Kind: internaldecoder.EventTileList, TileListErr: internalparser.ErrTileListShortEntry},
	}, decoderFrameWorkSideDataScratch{}, nil)
	if !errors.Is(err, internalparser.ErrTileListShortEntry) {
		t.Fatalf("RunEvents err=%v want %v", err, internalparser.ErrTileListShortEntry)
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerValidatesTileListLayout(t *testing.T) {
	var eventRunner decoderFrameWorkResidualEventRunner
	tileList := internalparser.TileList{
		OutputFrameWidthInTilesMinus1:  0,
		OutputFrameHeightInTilesMinus1: 0,
		TileCountMinus1:                0,
		Entries: []internalparser.TileListEntry{{
			AnchorFrameIdx:     0,
			AnchorTileRow:      1,
			AnchorTileCol:      1,
			TileDataSizeMinus1: 0,
			TileData:           []byte{0xaa},
		}},
	}
	validTiles := internalparser.TileInfo{
		Cols:       2,
		Rows:       2,
		ColStartSB: [internalparser.MaxTileCols + 1]uint16{0, 2, 4},
		RowStartSB: [internalparser.MaxTileRows + 1]uint16{0, 1, 2},
	}

	_, err := eventRunner.RunEvents(SequenceHeader{}, []internaldecoder.Event{
		{Kind: internaldecoder.EventTileList, TileList: tileList, TileInfo: validTiles},
	}, decoderFrameWorkSideDataScratch{}, nil)
	if !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("valid tile-list layout err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}

	badAnchor := validTiles
	badAnchor.Cols = 1
	_, err = eventRunner.RunEvents(SequenceHeader{}, []internaldecoder.Event{
		{Kind: internaldecoder.EventTileList, TileList: tileList, TileInfo: badAnchor},
	}, decoderFrameWorkSideDataScratch{}, nil)
	if !errors.Is(err, internalparser.ErrTileListInvalidAnchorTile) {
		t.Fatalf("bad anchor layout err=%v want %v", err, internalparser.ErrTileListInvalidAnchorTile)
	}

	nonUniform := validTiles
	nonUniform.ColStartSB[2] = 5
	_, err = eventRunner.RunEvents(SequenceHeader{}, []internaldecoder.Event{
		{Kind: internaldecoder.EventTileList, TileList: tileList, TileInfo: nonUniform},
	}, decoderFrameWorkSideDataScratch{}, nil)
	if !errors.Is(err, internalparser.ErrTileListNonUniformTileSize) {
		t.Fatalf("non-uniform layout err=%v want %v", err, internalparser.ErrTileListNonUniformTileSize)
	}
}

func TestPublicPlanDecoderTileListEntryWork(t *testing.T) {
	list := internalparser.TileList{
		OutputFrameWidthInTilesMinus1:  1,
		OutputFrameHeightInTilesMinus1: 0,
		TileCountMinus1:                1,
		Entries: []internalparser.TileListEntry{
			{
				AnchorFrameIdx:     0,
				AnchorTileRow:      0,
				AnchorTileCol:      1,
				TileDataSizeMinus1: 2,
				TileData:           []byte{0xaa, 0xbb, 0xcc},
			},
			{
				AnchorFrameIdx:     0,
				AnchorTileRow:      1,
				AnchorTileCol:      0,
				TileDataSizeMinus1: 1,
				TileData:           []byte{0xdd, 0xee},
			},
		},
	}
	tiles := internalparser.TileInfo{
		Cols:                2,
		Rows:                2,
		SBCols:              4,
		SBRows:              4,
		RefreshContext:      true,
		ContextUpdateTileID: 2,
		ColStartSB:          [internalparser.MaxTileCols + 1]uint16{0, 2, 4},
		RowStartSB:          [internalparser.MaxTileRows + 1]uint16{0, 2, 4},
	}
	var spans [1]internalparser.TileSpan
	var jobs [1]internaltile.Job
	var batches [1]internalthreading.Batch

	plan, err := internaldecoder.PlanTileListEntryWork(list, tiles, 1, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatalf("PlanDecoderTileListEntryWork: %v", err)
	}
	if plan.SpanCount != 1 || plan.JobCount != 1 || plan.BatchCount != 1 {
		t.Fatalf("plan=%+v want 1/1/1", plan)
	}
	if spans[0] != (internalparser.TileSpan{Tile: 2, Row: 1, Col: 0, Offset: 0, Size: 2}) {
		t.Fatalf("span=%+v", spans[0])
	}
	if jobs[0] != (internaltile.Job{
		Tile:                2,
		Row:                 1,
		Col:                 0,
		SBX:                 0,
		SBY:                 2,
		SBCols:              2,
		SBRows:              2,
		Offset:              0,
		Size:                2,
		LastRow:             true,
		UpdatesFrameContext: true,
	}) {
		t.Fatalf("job=%+v", jobs[0])
	}
	if batches[0] != (internalthreading.Batch{FirstJob: 0, Count: 1, Units: 4, Worker: 0, FirstTile: 2, LastTile: 2}) {
		t.Fatalf("batch=%+v", batches[0])
	}
	payload, err := jobs[0].Payload(list.Entries[1].TileData)
	if err != nil {
		t.Fatalf("job payload: %v", err)
	}
	if len(payload) != 2 || payload[0] != 0xdd || payload[1] != 0xee {
		t.Fatalf("payload=% x", payload)
	}
}

func TestPublicPlanDecoderTileListEntryWorkRejectsInvalid(t *testing.T) {
	list := internalparser.TileList{
		OutputFrameWidthInTilesMinus1:  0,
		OutputFrameHeightInTilesMinus1: 0,
		TileCountMinus1:                0,
		Entries: []internalparser.TileListEntry{{
			AnchorFrameIdx:     0,
			AnchorTileRow:      0,
			AnchorTileCol:      0,
			TileDataSizeMinus1: 0,
			TileData:           []byte{0xaa},
		}},
	}
	tiles := internalparser.TileInfo{
		Cols:       1,
		Rows:       1,
		SBCols:     2,
		SBRows:     2,
		ColStartSB: [internalparser.MaxTileCols + 1]uint16{0, 2},
		RowStartSB: [internalparser.MaxTileRows + 1]uint16{0, 2},
	}
	var spans [1]internalparser.TileSpan
	var jobs [1]internaltile.Job
	var batches [1]internalthreading.Batch

	tests := []struct {
		name  string
		list  internalparser.TileList
		tiles internalparser.TileInfo
		index int
		spans []internalparser.TileSpan
		want  error
	}{
		{name: "entry-index", list: list, tiles: tiles, index: 1, spans: spans[:], want: internalparser.ErrTileListInvalidTileCount},
		{name: "short-spans", list: list, tiles: tiles, index: 0, spans: spans[:0], want: internaldecoder.ErrInvalidTileWork},
		{name: "short-tile-data", list: func() internalparser.TileList {
			short := list
			short.Entries = append([]internalparser.TileListEntry(nil), list.Entries...)
			short.Entries[0].TileDataSizeMinus1 = 2
			short.Entries[0].TileData = []byte{0xaa, 0xbb}
			return short
		}(), tiles: tiles, index: 0, spans: spans[:], want: internalparser.ErrTileListShortTileData},
		{name: "long-tile-data", list: func() internalparser.TileList {
			long := list
			long.Entries = append([]internalparser.TileListEntry(nil), list.Entries...)
			long.Entries[0].TileDataSizeMinus1 = 0
			long.Entries[0].TileData = []byte{0xaa, 0xbb}
			return long
		}(), tiles: tiles, index: 0, spans: spans[:], want: internalparser.ErrTileListTrailingBytes},
		{name: "non-uniform-layout", list: list, tiles: func() internalparser.TileInfo {
			nonUniform := tiles
			nonUniform.Cols = 2
			nonUniform.SBCols = 5
			nonUniform.ColStartSB = [internalparser.MaxTileCols + 1]uint16{0, 2, 5}
			return nonUniform
		}(), index: 0, spans: spans[:], want: internalparser.ErrTileListNonUniformTileSize},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := internaldecoder.PlanTileListEntryWork(tc.list, tc.tiles, tc.index, 1, tc.spans, jobs[:], batches[:])
			if !errors.Is(err, tc.want) {
				t.Fatalf("PlanDecoderTileListEntryWork err=%v want %v", err, tc.want)
			}
		})
	}
}

func TestPublicPlanDecoderTileListEntryDecode(t *testing.T) {
	event := publicTileListEntryDecodeEvent()
	anchors := make([]*Frame, internalparser.TileListMaxExternalReferences)
	anchor := &Frame{Format: FrameFormat{Width: 256, Height: 256, BitDepth: 8, SubsamplingX: true, SubsamplingY: true}}
	anchors[7] = anchor
	references := []*Frame{nil, &Frame{Format: FrameFormat{Width: 1, Height: 1, BitDepth: 8}}}
	var spans [1]internalparser.TileSpan
	var jobs [1]internaltile.Job
	var batches [1]internalthreading.Batch

	plan, err := internaldecoder.PlanTileListEntryDecode(&event, 1, 3, anchors, references, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatalf("PlanDecoderTileListEntryDecode: %v", err)
	}
	if plan.Event.Kind != internaldecoder.EventTileGroup ||
		plan.Step.Kind != internaldecoder.FrameWorkStepTile ||
		plan.Step.Tile.Surface != 3 ||
		plan.ReferenceCount != 1 ||
		plan.Step.Tile.ReferenceCount != 1 ||
		plan.AnchorFrameIdx != 7 ||
		!bytes.Equal(plan.Payload, []byte{0xdd, 0xee}) ||
		!bytes.Equal(plan.Event.Unit.Payload, []byte{0xdd, 0xee}) {
		t.Fatalf("decode plan=%+v payload=% x", plan, plan.Event.Unit.Payload)
	}
	if references[0] != anchor {
		t.Fatalf("LAST reference was not bound to anchor frame: %p want %p", references[0], anchor)
	}
	if plan.Geometry.OutputFrameWidth != 256 || plan.Geometry.OutputFrameHeight != 128 {
		t.Fatalf("geometry=%+v", plan.Geometry)
	}
	if plan.Region != (internalparser.TileListTileRegion{SourceX: 0, SourceY: 128, DestX: 128, DestY: 0, Width: 128, Height: 128}) {
		t.Fatalf("region=%+v", plan.Region)
	}
	if spans[0] != (internalparser.TileSpan{Tile: 2, Row: 1, Col: 0, Offset: 0, Size: 2}) {
		t.Fatalf("span=%+v", spans[0])
	}
	if jobs[0] != (internaltile.Job{
		Tile:                2,
		Row:                 1,
		Col:                 0,
		SBX:                 0,
		SBY:                 2,
		SBCols:              2,
		SBRows:              2,
		Offset:              0,
		Size:                2,
		LastRow:             true,
		UpdatesFrameContext: true,
	}) {
		t.Fatalf("job=%+v", jobs[0])
	}

	size, scratchPlan, err := decoderTileListEntryDecodeRunnerScratchLen(&event, 1, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatalf("DecoderTileListEntryDecodeRunnerScratchLen: %v", err)
	}
	if scratchPlan != plan.Step.Tile.Tile {
		t.Fatalf("scratch plan=%+v want %+v", scratchPlan, plan.Step.Tile.Tile)
	}
	if size.Batch.Int32Scratch == 0 || size.Batch.ResidualScratch == 0 || size.Workers != 1 {
		t.Fatalf("scratch size=%+v", size)
	}
}

func TestPublicPlanDecoderTileListEntryDecodeRejectsInvalid(t *testing.T) {
	event := publicTileListEntryDecodeEvent()
	anchors := make([]*Frame, internalparser.TileListMaxExternalReferences)
	anchor := &Frame{Format: FrameFormat{Width: 256, Height: 256, BitDepth: 8, SubsamplingX: true, SubsamplingY: true}}
	anchors[7] = anchor
	stale := &Frame{}
	references := []*Frame{stale}
	var spans [1]internalparser.TileSpan
	var jobs [1]internaltile.Job
	var batches [1]internalthreading.Batch

	tests := []struct {
		name    string
		event   internaldecoder.Event
		index   int
		surface int
		anchors []*Frame
		refs    []*Frame
		want    error
	}{
		{name: "not-tile-list", event: internaldecoder.Event{Kind: internaldecoder.EventFrame}, index: 0, surface: 0, anchors: anchors, refs: references, want: internaldecoder.ErrInvalidSurfaceEvent},
		{name: "parse-error", event: func() internaldecoder.Event {
			bad := event
			bad.TileListErr = internalparser.ErrTileListShortEntry
			return bad
		}(), index: 0, surface: 0, anchors: anchors, refs: references, want: internalparser.ErrTileListShortEntry},
		{name: "negative-surface", event: event, index: 0, surface: -1, anchors: anchors, refs: references, want: internaldecoder.ErrInvalidSurfaceReference},
		{name: "short-reference-table", event: event, index: 0, surface: 0, anchors: anchors, refs: nil, want: internaldecoder.ErrSurfaceReferenceBufferTooSmall},
		{name: "missing-anchor", event: event, index: 1, surface: 0, anchors: anchors[:7], refs: references, want: internalparser.ErrTileListInvalidAnchorIndex},
		{name: "nil-anchor", event: event, index: 1, surface: 0, anchors: append([]*Frame(nil), anchors...), refs: references, want: internalparser.ErrTileListInvalidAnchorIndex},
		{name: "bad-index", event: event, index: 2, surface: 0, anchors: anchors, refs: references, want: internalparser.ErrTileListInvalidTileCount},
	}
	tests[5].anchors[7] = nil

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			references[0] = stale
			_, err := internaldecoder.PlanTileListEntryDecode(&tc.event, tc.index, tc.surface, tc.anchors, tc.refs, 1, spans[:], jobs[:], batches[:])
			if !errors.Is(err, tc.want) {
				t.Fatalf("PlanDecoderTileListEntryDecode err=%v want %v", err, tc.want)
			}
			if references[0] != stale {
				t.Fatalf("references mutated on error: %p want %p", references[0], stale)
			}
		})
	}

	if _, _, err := decoderTileListEntryDecodeRunnerScratchLen(&internaldecoder.Event{Kind: internaldecoder.EventFrame}, 0, 1, spans[:], jobs[:], batches[:]); !errors.Is(err, internaldecoder.ErrInvalidSurfaceEvent) {
		t.Fatalf("scratch non-tile-list err=%v want %v", err, internaldecoder.ErrInvalidSurfaceEvent)
	}
	parseErrEvent := event
	parseErrEvent.TileListErr = internalparser.ErrTileListShortEntry
	if _, _, err := decoderTileListEntryDecodeRunnerScratchLen(&parseErrEvent, 0, 1, spans[:], jobs[:], batches[:]); !errors.Is(err, internalparser.ErrTileListShortEntry) {
		t.Fatalf("scratch parse err=%v want %v", err, internalparser.ErrTileListShortEntry)
	}
}

func publicTileListEntryDecodeEvent() internaldecoder.Event {
	return internaldecoder.Event{
		Kind: internaldecoder.EventTileList,
		SequenceHeader: SequenceHeader{
			ColorConfig: internalparser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
		},
		FrameHeader: internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeInter},
		FrameSize: FrameSize{
			CodedWidth:        256,
			UpscaledWidth:     256,
			Height:            256,
			RenderWidth:       256,
			RenderHeight:      256,
			RefreshFrameFlags: 0,
		},
		TileInfo: internalparser.TileInfo{
			Cols:                2,
			Rows:                2,
			SBCols:              4,
			SBRows:              4,
			RefreshContext:      true,
			ContextUpdateTileID: 2,
			ColStartSB:          [internalparser.MaxTileCols + 1]uint16{0, 2, 4},
			RowStartSB:          [internalparser.MaxTileRows + 1]uint16{0, 2, 4},
		},
		TileList: internalparser.TileList{
			OutputFrameWidthInTilesMinus1:  1,
			OutputFrameHeightInTilesMinus1: 0,
			TileCountMinus1:                1,
			Entries: []internalparser.TileListEntry{
				{
					AnchorFrameIdx:     0,
					AnchorTileRow:      0,
					AnchorTileCol:      1,
					TileDataSizeMinus1: 2,
					TileData:           []byte{0xaa, 0xbb, 0xcc},
				},
				{
					AnchorFrameIdx:     7,
					AnchorTileRow:      1,
					AnchorTileCol:      0,
					TileDataSizeMinus1: 1,
					TileData:           []byte{0xdd, 0xee},
				},
			},
		},
	}
}

func TestPublicDecoderFrameWorkResidualEventRunnerRunEventsAllocs(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	sequence := SequenceHeader{
		EnableCDEF: true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	frame := publicDecoderResidualRunnerFrameEvent()
	events := [...]internaldecoder.Event{
		{
			Kind:                  internaldecoder.EventSequenceHeader,
			SequenceHeader:        sequence,
			NewCodedVideoSequence: true,
		},
		frame,
	}
	var scratchSpans [2]internalparser.TileSpan
	var scratchJobs [2]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventsScratchLen(sequence, events[:], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:      int(frame.FrameSize.CodedWidth),
		Height:     int(frame.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 1)
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var eventSideData decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	var outputs [1]*Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &frame, decoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &eventSideData,
		Stats:             &stats,
		Outputs:           outputs[:],
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, runErr := eventRunner.RunEvents(sequence, events[:], scratch.SideData, nil)
		if runErr != nil {
			err = runErr
			return
		}
		if result.Count != len(events) ||
			result.ExecutedTileWork != 1 ||
			result.CompletedFrames != 1 ||
			result.OutputCount != 1 ||
			outputs[0] != result.Last.Output ||
			stats.TXBs == 0 ||
			stats.Residuals == 0 {
			err = internalthreading.ErrInvalidBatch
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualEventRunner.RunEvents allocated: %f", allocs)
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerLowOverhead(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	payload := publicDecoderResidualLowOverheadStream()
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	count, err := probeStream.PushLowOverhead(payload, probeEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence, ok := probeStream.SequenceHeader()
	if !ok {
		t.Fatal("probe stream missing sequence")
	}
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventsScratchLen(sequence, probeEvents[:count], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(probeEvents[count-1].FrameSize.CodedWidth),
		Height:       int(probeEvents[count-1].FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	var outputs [2]*Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &probeEvents[count-1], decoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
		Outputs:           outputs[:],
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}
	runner := decoderFrameWorkResidualStreamRunner{
		Stream:          &stream,
		EventRunner:     eventRunner,
		Events:          make([]internaldecoder.Event, len(probeEvents)),
		SideDataScratch: scratch.SideData,
	}

	result, err := runner.RunLowOverhead(payload, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.EventCount != count ||
		result.Run.Count != count ||
		result.Run.ExecutedTileWork != 1 ||
		result.Run.CompletedFrames != 1 ||
		result.Run.OutputCount != 1 ||
		len(result.Run.Outputs) != 1 ||
		result.Run.Last.Output == nil ||
		result.Run.Outputs[0] != result.Run.Last.Output ||
		state.Active() ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("stream result=%+v active=%v stats=%+v", result, state.Active(), stats)
	}
	if _, ok := refs.ReferenceSlot(0); !ok {
		t.Fatal("low-overhead stream runner did not publish decoded frame")
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerReset(t *testing.T) {
	var stream internaldecoder.Stream
	var events [4]internaldecoder.Event
	if count, err := stream.PushLowOverhead(publicDecoderResidualLowOverheadStream(), events[:]); err != nil {
		t.Fatal(err)
	} else if count != 3 {
		t.Fatalf("count=%d events=%+v", count, events[:count])
	}
	runner := decoderFrameWorkResidualStreamRunner{
		Stream:  &stream,
		RTPUsed: 7,
	}
	if !stream.HasSequenceHeader() {
		t.Fatal("stream missing sequence before reset")
	}
	frameHeader := publicDecoderResidualRTPElement(internalobu.TypeFrameHeader, publicDecoderResidualFrameHeaderPayload())
	var packet [8]byte
	n, _, more, err := PutRTPFragment(packet[:], frameHeader, 0, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	if !more {
		t.Fatal("expected frame header to need a second fragment")
	}
	var rtpBuffer [128]byte
	var rtpSpans [1]RTPObuSpan
	used, count, err := stream.PushRTPPayload(rtpBuffer[:], 0, rtpSpans[:], events[:1], packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 || !stream.InRTPFragment() || used == 0 {
		t.Fatalf("fragment push used=%d count=%d inFragment=%v", used, count, stream.InRTPFragment())
	}
	runner.RTPUsed = used
	if err := runner.ResetRTP(); err != nil {
		t.Fatal(err)
	}
	if runner.RTPUsed != 0 || !stream.HasSequenceHeader() || stream.InRTPFragment() {
		t.Fatalf("ResetRTP runner=%+v hasSequence=%v inFragment=%v", runner, stream.HasSequenceHeader(), stream.InRTPFragment())
	}
	runner.RTPUsed = 9
	if err := runner.Reset(); err != nil {
		t.Fatal(err)
	}
	if runner.RTPUsed != 0 || stream.HasSequenceHeader() || stream.InRTPFragment() {
		t.Fatalf("Reset runner=%+v hasSequence=%v inFragment=%v", runner, stream.HasSequenceHeader(), stream.InRTPFragment())
	}

	var nilRunner *decoderFrameWorkResidualStreamRunner
	if err := nilRunner.Reset(); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil reset err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	if err := nilRunner.ResetRTP(); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil rtp reset err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
}

func TestPublicDecoderFrameWorkResidualEventsBindPlan(t *testing.T) {
	payload := publicDecoderResidualLowOverheadStream()
	var stream internaldecoder.Stream
	var events [4]internaldecoder.Event
	count, err := stream.PushLowOverhead(payload, events[:])
	if err != nil {
		t.Fatal(err)
	}
	plan := decoderFrameWorkResidualEventsBindPlan(SequenceHeader{}, events[:count])
	if !plan.HasEvent() ||
		plan.EventIndex != count-1 ||
		plan.Event.Kind != internaldecoder.EventTileGroup ||
		plan.Sequence.ColorConfig.BitDepth != 8 ||
		!plan.Sequence.ColorConfig.MonoChrome {
		t.Fatalf("low-overhead bind plan=%+v count=%d", plan, count)
	}

	var realtime internaldecoder.Stream
	var prime []byte
	keyFrame := append([]byte{}, publicDecoderResidualShownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	interFrame := append([]byte{}, publicDecoderResidualInterFrameHeaderPayload()...)
	interFrame = append(interFrame, 0xbb)
	prime = appendOBUForTest(prime, internalobu.TypeSequenceHeader, publicDecoderResidualRealtimeSequenceHeaderPayload())
	prime = appendOBUForTest(prime, internalobu.TypeFrame, keyFrame)
	prime = appendOBUForTest(prime, internalobu.TypeFrame, interFrame)
	var primeEvents [3]internaldecoder.Event
	if count, err := realtime.PushLowOverhead(prime, primeEvents[:]); err != nil {
		t.Fatal(err)
	} else if count != len(primeEvents) {
		t.Fatalf("prime count=%d events=%+v", count, primeEvents)
	}
	sequence, ok := realtime.SequenceHeader()
	if !ok {
		t.Fatal("realtime stream missing sequence")
	}
	var showExisting []byte
	showExisting = appendOBUForTest(showExisting, internalobu.TypeFrameHeader, publicDecoderResidualShowExistingFrameHeaderPayload(0))
	var outputEvents [1]internaldecoder.Event
	outputCount, err := realtime.PushLowOverhead(showExisting, outputEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	outputPlan := decoderFrameWorkResidualEventsBindPlan(sequence, outputEvents[:outputCount])
	if !outputPlan.HasEvent() ||
		outputPlan.EventIndex != 0 ||
		outputPlan.Event.Kind != internaldecoder.EventExistingFrame ||
		outputPlan.Event.ExistingFrame.FrameType != internalparser.FrameTypeInter ||
		outputPlan.Sequence.ColorConfig.BitDepth != 8 {
		t.Fatalf("show-existing bind plan=%+v count=%d", outputPlan, outputCount)
	}

	sequenceOnly := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:   10,
		MonoChrome: true,
	}}
	noEventPlan := decoderFrameWorkResidualEventsBindPlan(SequenceHeader{}, []internaldecoder.Event{
		{Kind: internaldecoder.EventSequenceHeader, SequenceHeader: sequenceOnly},
	})
	if noEventPlan.HasEvent() ||
		noEventPlan.EventIndex != -1 ||
		noEventPlan.Sequence.ColorConfig.BitDepth != 10 ||
		!noEventPlan.Sequence.ColorConfig.MonoChrome {
		t.Fatalf("sequence-only bind plan=%+v", noEventPlan)
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerRTPPayload(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	lowOverhead := publicDecoderResidualLowOverheadStream()
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	count, err := probeStream.PushLowOverhead(lowOverhead, probeEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence, ok := probeStream.SequenceHeader()
	if !ok {
		t.Fatal("probe stream missing sequence")
	}
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventsScratchLen(sequence, probeEvents[:count], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	rtpPayload := publicDecoderResidualRTPPayload()
	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(probeEvents[count-1].FrameSize.CodedWidth),
		Height:       int(probeEvents[count-1].FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	var outputs [1]*Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &probeEvents[count-1], decoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
		Outputs:           outputs[:],
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}
	runner := decoderFrameWorkResidualStreamRunner{
		Stream:          &stream,
		EventRunner:     eventRunner,
		Events:          make([]internaldecoder.Event, len(probeEvents)),
		SideDataScratch: scratch.SideData,
		RTPBuffer:       make([]byte, len(lowOverhead)),
		RTPSpans:        make([]RTPObuSpan, len(probeEvents)),
	}

	result, err := runner.RunRTPPayload(rtpPayload, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.EventCount != count ||
		result.RTPUsed != 0 ||
		runner.RTPUsed != 0 ||
		result.Run.Count != count ||
		result.Run.ExecutedTileWork != 1 ||
		result.Run.CompletedFrames != 1 ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("rtp result=%+v retained=%d stats=%+v", result, runner.RTPUsed, stats)
	}
	if _, ok := refs.ReferenceSlot(0); !ok {
		t.Fatal("RTP stream runner did not publish decoded frame")
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerRTPPayloads(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	lowOverhead := publicDecoderResidualLowOverheadStream()
	rtpPayloads := publicDecoderResidualFragmentedRTPPayloads()
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	count, err := probeStream.PushLowOverhead(lowOverhead, probeEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence, ok := probeStream.SequenceHeader()
	if !ok {
		t.Fatal("probe stream missing sequence")
	}
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventsScratchLen(sequence, probeEvents[:count], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(probeEvents[count-1].FrameSize.CodedWidth),
		Height:       int(probeEvents[count-1].FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 1)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	var outputs [1]*Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &probeEvents[count-1], decoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
		Outputs:           outputs[:],
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}
	runner := decoderFrameWorkResidualStreamRunner{
		Stream:          &stream,
		EventRunner:     eventRunner,
		Events:          make([]internaldecoder.Event, len(probeEvents)),
		SideDataScratch: scratch.SideData,
		RTPBuffer:       make([]byte, len(lowOverhead)),
		RTPSpans:        make([]RTPObuSpan, len(probeEvents)),
	}

	var nilRunner *decoderFrameWorkResidualStreamRunner
	if _, err := nilRunner.RunRTPPayloads(nil, nil); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil RTP payload batch runner err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	empty, err := runner.RunRTPPayloads(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if empty.EventCount != 0 || empty.RTPUsed != 0 || empty.Run.Count != 0 || empty.Run.OutputCount != 0 {
		t.Fatalf("empty RTP payload batch result=%+v", empty)
	}

	postCalls := 0
	result, err := runner.RunRTPPayloads(rtpPayloads, func(ctx internaldecoder.FrameWorkPostFilterContext) error {
		postCalls++
		if !ctx.ExecutedTileWork || ctx.Step.Kind != internaldecoder.FrameWorkStepTile {
			t.Fatalf("postfilter ctx=%+v", ctx)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.EventCount != count ||
		result.RTPUsed != 0 ||
		runner.RTPUsed != 0 ||
		result.Run.Count != count ||
		result.Run.ExecutedTileWork != 1 ||
		result.Run.CompletedFrames != 1 ||
		result.Run.OutputCount != 1 ||
		len(result.Run.Outputs) != 1 ||
		result.Run.Last.Output == nil ||
		outputs[0] != result.Run.Last.Output ||
		result.Run.Outputs[0] != result.Run.Last.Output ||
		postCalls != 1 ||
		stats.TXBs == 0 ||
		stats.Residuals == 0 {
		t.Fatalf("rtp batch result=%+v retained=%d postCalls=%d stats=%+v", result, runner.RTPUsed, postCalls, stats)
	}
	if _, ok := refs.ReferenceSlot(0); !ok {
		t.Fatal("RTP payload batch runner did not publish decoded frame")
	}
}

func TestPublicDecoderFrameWorkResidualStreamResultAccumulate(t *testing.T) {
	var firstFrame Frame
	var secondFrame Frame
	var result decoderFrameWorkResidualStreamResult
	if err := result.Accumulate(decoderFrameWorkResidualStreamResult{
		EventCount: 2,
		RTPUsed:    3,
		Run: decoderFrameWorkResidualEventsResult{
			Count:            2,
			ExecutedTileWork: 1,
			CompletedFrames:  1,
			OutputCount:      1,
			Last: internaldecoder.FrameWorkEventResult{
				Output: &firstFrame,
			},
			Stats: internalthreading.FrameWorkTileResidualStats{
				TXBs:      1,
				Residuals: 2,
			},
		},
	}); err != nil {
		t.Fatal(err)
	}
	if err := result.Accumulate(decoderFrameWorkResidualStreamResult{
		EventCount: 1,
		RTPUsed:    0,
		Run: decoderFrameWorkResidualEventsResult{
			Count:            1,
			ExecutedTileWork: 1,
			CompletedFrames:  1,
			OutputCount:      1,
			Last: internaldecoder.FrameWorkEventResult{
				Output: &secondFrame,
			},
			Stats: internalthreading.FrameWorkTileResidualStats{
				TXBs:      3,
				Residuals: 5,
			},
		},
	}); err != nil {
		t.Fatal(err)
	}
	outputs := [...]*Frame{&firstFrame, &secondFrame}
	if err := result.BindOutputs(outputs[:]); err != nil {
		t.Fatal(err)
	}
	if result.EventCount != 3 ||
		result.RTPUsed != 0 ||
		result.Run.Count != 3 ||
		result.Run.ExecutedTileWork != 2 ||
		result.Run.CompletedFrames != 2 ||
		result.Run.OutputCount != 2 ||
		result.Run.Last.Output != &secondFrame ||
		result.Run.Outputs[0] != &firstFrame ||
		result.Run.Outputs[1] != &secondFrame ||
		result.Run.Stats.TXBs != 4 ||
		result.Run.Stats.Residuals != 7 {
		t.Fatalf("accumulated result=%+v", result)
	}
	if err := result.BindOutputs(outputs[:1]); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short outputs err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	var nilStreamResult *decoderFrameWorkResidualStreamResult
	if err := nilStreamResult.Accumulate(result); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil stream accumulate err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	var nilEventsResult *decoderFrameWorkResidualEventsResult
	if err := nilEventsResult.Accumulate(result.Run); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil event accumulate err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	if err := nilEventsResult.BindOutputs(outputs[:]); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil event outputs err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
}

func TestPublicDecoderFrameWorkResidualStreamRunnerAllocs(t *testing.T) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	lowOverhead := publicDecoderResidualLowOverheadStream()
	lowOverheads := [...][]byte{
		publicDecoderResidualLowOverheadStream(),
		publicDecoderResidualLowOverheadFrameStream(),
	}
	rtpPayload := publicDecoderResidualRTPPayload()
	rtpPayloads := publicDecoderResidualFragmentedRTPPayloads()
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	count, err := probeStream.PushLowOverhead(lowOverhead, probeEvents[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence, ok := probeStream.SequenceHeader()
	if !ok {
		t.Fatal("probe stream missing sequence")
	}
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	size, err := decoderFrameWorkResidualEventsScratchLen(sequence, probeEvents[:count], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		t.Fatal(err)
	}
	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(probeEvents[count-1].FrameSize.CodedWidth),
		Height:       int(probeEvents[count-1].FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	var outputs [2]*Frame
	scratch := publicDecoderResidualEventScratch(size)
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(size, sequence, &probeEvents[count-1], decoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
		Outputs:           outputs[:],
	}, scratch, &batchRunner)
	if err != nil {
		t.Fatal(err)
	}
	var events [4]internaldecoder.Event
	var rtpBuffer [128]byte
	var rtpSpans [4]RTPObuSpan
	runner := decoderFrameWorkResidualStreamRunner{
		Stream:          &stream,
		EventRunner:     eventRunner,
		Events:          events[:],
		SideDataScratch: scratch.SideData,
		RTPBuffer:       rtpBuffer[:],
		RTPSpans:        rtpSpans[:],
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result, runErr := runner.RunLowOverhead(lowOverhead, nil)
		if runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 1 || result.Run.OutputCount != 1 || outputs[0] != result.Run.Last.Output || stats.TXBs == 0 {
			err = internalthreading.ErrInvalidBatch
			return
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result, runErr = runner.RunLowOverheads(lowOverheads[:], nil)
		if runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 2 || result.Run.OutputCount != 2 || outputs[1] != result.Run.Last.Output || stats.TXBs == 0 {
			err = internalthreading.ErrInvalidBatch
			return
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result = decoderFrameWorkResidualStreamResult{}
		if runErr = runner.RunLowOverheadsInto(&result, lowOverheads[:], nil); runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 2 || result.Run.OutputCount != 2 || outputs[1] != result.Run.Last.Output || stats.TXBs == 0 {
			err = internalthreading.ErrInvalidBatch
			return
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result, runErr = runner.RunRTPPayload(rtpPayload, nil)
		if runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 1 || result.Run.OutputCount != 1 || outputs[0] != result.Run.Last.Output || stats.TXBs == 0 || runner.RTPUsed != 0 {
			err = internalthreading.ErrInvalidBatch
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result, runErr = runner.RunRTPPayloads(rtpPayloads, nil)
		if runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 1 || result.Run.OutputCount != 1 || outputs[0] != result.Run.Last.Output || stats.TXBs == 0 || runner.RTPUsed != 0 {
			err = internalthreading.ErrInvalidBatch
			return
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result = decoderFrameWorkResidualStreamResult{}
		if runErr = runner.RunRTPPayloadsInto(&result, rtpPayloads, nil); runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 1 || result.Run.OutputCount != 1 || outputs[0] != result.Run.Last.Output || stats.TXBs == 0 || runner.RTPUsed != 0 {
			err = internalthreading.ErrInvalidBatch
			return
		}

		pool.Reset()
		refs.Reset()
		state.Reset()
		if resetErr := runner.Reset(); resetErr != nil {
			err = resetErr
			return
		}
		result = decoderFrameWorkResidualStreamResult{}
		if runErr = runner.RunRTPPayloadsInto(&result, rtpPayloads[:2], nil); runErr != nil {
			err = runErr
			return
		}
		if runner.RTPUsed == 0 {
			err = internalthreading.ErrInvalidBatch
			return
		}
		if runErr = runner.RunRTPPayloadAfterLossInto(&result, rtpPayload, nil); runErr != nil {
			err = runErr
			return
		}
		if result.Run.CompletedFrames != 1 || result.Run.OutputCount != 1 || outputs[0] != result.Run.Last.Output || stats.TXBs == 0 || runner.RTPUsed != 0 {
			err = internalthreading.ErrInvalidBatch
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualStreamRunner allocated: %f", allocs)
	}
}

func TestPublicDecoderFrameWorkResidualEventsBindPlanAllocs(t *testing.T) {
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{
		BitDepth:   8,
		MonoChrome: true,
	}}
	events := [...]internaldecoder.Event{
		{Kind: internaldecoder.EventSequenceHeader, SequenceHeader: sequence},
		{Kind: internaldecoder.EventFrameHeader, SequenceHeader: sequence},
		{Kind: internaldecoder.EventTileGroup, SequenceHeader: sequence},
	}
	var plan decoderFrameWorkResidualEventBindPlan
	allocs := testing.AllocsPerRun(1000, func() {
		plan = decoderFrameWorkResidualEventsBindPlan(SequenceHeader{}, events[:])
	})
	if !plan.HasEvent() || plan.EventIndex != 2 || plan.Event.Kind != internaldecoder.EventTileGroup {
		t.Fatalf("bind plan=%+v", plan)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualEventsBindPlan allocated: %f", allocs)
	}
}

func TestPublicDecoderFrameWorkResidualStreamCheckAndStateAllocs(t *testing.T) {
	size := decoderFrameWorkResidualStreamScratchSize{
		Events:    3,
		RTPBuffer: 32,
		RTPSpans:  3,
		Event: decoderFrameWorkResidualEventScratchSize{
			Runner: decoderFrameWorkBatchResidualRunnerScratchSize{
				Workers:         1,
				Int32Scratch:    8,
				ResidualScratch: 8,
			},
			SideData: decoderFrameWorkSideDataScratchSize{
				CDEFIndexMap:  4,
				LoopFilterMap: 4,
			},
			Plan:    internaldecoder.TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1},
			Outputs: 2,
		},
	}
	scratch := publicDecoderResidualStreamScratch(size)
	var stream internaldecoder.Stream
	runner := decoderFrameWorkResidualStreamRunner{
		Stream: &stream,
		EventRunner: decoderFrameWorkResidualEventRunner{
			Outputs: scratch.Outputs[:size.Event.Outputs],
		},
		Events:    scratch.Events[:size.Events],
		RTPBuffer: scratch.RTPBuffer[:size.RTPBuffer],
		RTPSpans:  scratch.RTPSpans[:size.RTPSpans],
		RTPUsed:   7,
	}
	var state decoderFrameWorkResidualStreamRunnerState
	var err error
	var ok bool
	var sequence SequenceHeader
	allocs := testing.AllocsPerRun(1000, func() {
		err = scratch.Check(size)
		state = runner.State()
		sequence, ok = runner.SequenceHeader()
	})
	if err != nil {
		t.Fatal(err)
	}
	if !state.Bound || state.RTPUsed != 7 || state.OutputCapacity != 2 || ok || sequence != (SequenceHeader{}) {
		t.Fatalf("state=%+v ok=%v sequence=%+v", state, ok, sequence)
	}
	if allocs != 0 {
		t.Fatalf("DecoderFrameWorkResidualStream Check/State allocated: %f", allocs)
	}
}

func publicDecoderBatchResidualRunnerScratch(size decoderFrameWorkBatchResidualRunnerScratchSize) decoderFrameWorkBatchResidualRunnerScratch {
	return decoderFrameWorkBatchResidualRunnerScratch{
		States:                  make([]internaltile.DecodeState, size.Workers+1),
		Storages:                make([]internalthreading.FrameWorkTileResidualCDFStorage, size.Workers+1),
		TileScratch:             make([]internalthreading.FrameWorkTileResidualScratch, size.Workers+1),
		RestorationRequests:     make([]internalthreading.FrameWorkTileRestorationRequest, size.RestorationRequests+1),
		PredictionScratch:       make([]internalthreading.FrameWorkPredictionScratch, size.Workers+1),
		InterPredictionScratch:  make([]internalthreading.FrameWorkInterPredictionScratch, size.Workers+1),
		Stats:                   make([]internalthreading.FrameWorkTileResidualStats, size.Workers+1),
		Int32Scratch:            make([]int32, size.Int32Scratch+1),
		ResidualScratch:         make([]int16, size.ResidualScratch+1),
		LoopContextAboveScratch: make([]internaltile.BlockLoopRootAboveContext, size.LoopContextAbove+1),
	}
}

func publicDecoderFrameWorkSideDataScratch(size decoderFrameWorkSideDataScratchSize) decoderFrameWorkSideDataScratch {
	return decoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, size.CDEFIndexMap+1),
		CDEFReadMap:              make([]bool, size.CDEFReadMap+1),
		LoopFilterMap:            make([]internalthreading.FrameWorkLoopFilterBlockRecord, size.LoopFilterMap+1),
		RestorationRecords:       make([]internaltile.RestorationUnitRecord, size.RestorationRecords+1),
		RestorationBoundaryAbove: make([]uint16, size.RestorationBoundaryAbove+1),
		RestorationBoundaryBelow: make([]uint16, size.RestorationBoundaryBelow+1),
	}
}

func publicDecoderResidualEventScratch(size decoderFrameWorkResidualEventScratchSize) decoderFrameWorkResidualEventScratch {
	return decoderFrameWorkResidualEventScratch{
		Runner:   publicDecoderBatchResidualRunnerScratch(size.Runner),
		SideData: publicDecoderFrameWorkSideDataScratch(size.SideData),
		Spans:    make([]internalparser.TileSpan, size.Plan.SpanCount+1),
		Jobs:     make([]internaltile.Job, size.Plan.JobCount+1),
		Batches:  make([]internalthreading.Batch, size.Plan.BatchCount+1),
	}
}

func publicDecoderResidualStreamScratch(size decoderFrameWorkResidualStreamScratchSize) decoderFrameWorkResidualStreamScratch {
	return decoderFrameWorkResidualStreamScratch{
		Events:    make([]internaldecoder.Event, size.Events+1),
		Event:     publicDecoderResidualEventScratch(size.Event),
		SideData:  publicDecoderFrameWorkSideDataScratch(size.Event.SideData),
		Outputs:   make([]*Frame, size.Event.Outputs+1),
		RTPBuffer: make([]byte, size.RTPBuffer+1),
		RTPSpans:  make([]RTPObuSpan, size.RTPSpans+1),
	}
}

type publicDecoderResidualBitWriter struct {
	buf [128]byte
	bit int
}

func (w *publicDecoderResidualBitWriter) writeBits(value uint64, n uint8) {
	for i := int(n) - 1; i >= 0; i-- {
		if (value>>uint(i))&1 != 0 {
			w.buf[w.bit>>3] |= 1 << uint(7-(w.bit&7))
		}
		w.bit++
	}
}

func (w *publicDecoderResidualBitWriter) writeBool(value bool) {
	if value {
		w.writeBits(1, 1)
		return
	}
	w.writeBits(0, 1)
}

func (w *publicDecoderResidualBitWriter) bytes() []byte {
	return w.buf[:(w.bit+7)>>3]
}

func (w *publicDecoderResidualBitWriter) trailingBits() []byte {
	w.writeBits(1, 1)
	for w.bit&7 != 0 {
		w.writeBits(0, 1)
	}
	return w.buf[:w.bit>>3]
}

func publicDecoderResidualLowOverheadStream() []byte {
	var stream []byte
	stream = appendOBUForTest(stream, internalobu.TypeSequenceHeader, publicDecoderResidualSequenceHeaderPayload())
	stream = appendOBUForTest(stream, internalobu.TypeFrameHeader, publicDecoderResidualFrameHeaderPayload())
	stream = appendOBUForTest(stream, internalobu.TypeTileGroup, publicDecoderResidualTileData())
	return stream
}

func publicDecoderResidualLowOverheadFrameStream() []byte {
	var stream []byte
	stream = appendOBUForTest(stream, internalobu.TypeFrameHeader, publicDecoderResidualFrameHeaderPayload())
	stream = appendOBUForTest(stream, internalobu.TypeTileGroup, publicDecoderResidualTileData())
	return stream
}

func publicDecoderResidualRTPPayload() []byte {
	elements := [...]RTPElement{
		{Data: publicDecoderResidualRTPElement(internalobu.TypeSequenceHeader, publicDecoderResidualSequenceHeaderPayload())},
		{Data: publicDecoderResidualRTPElement(internalobu.TypeFrameHeader, publicDecoderResidualFrameHeaderPayload())},
		{Data: publicDecoderResidualRTPElement(internalobu.TypeTileGroup, publicDecoderResidualTileData())},
	}
	payload := make([]byte, 128)
	n, err := PutRTPPayload(payload, RTPAggregationHeader{
		ElementCount:                uint8(len(elements)),
		StartsNewCodedVideoSequence: true,
	}, elements[:])
	if err != nil {
		panic(err)
	}
	return payload[:n]
}

func publicDecoderResidualFragmentedRTPPayloads() [][]byte {
	payloads := make([][]byte, 0, 4)

	sequence := [...]RTPElement{
		{Data: publicDecoderResidualRTPElement(internalobu.TypeSequenceHeader, publicDecoderResidualSequenceHeaderPayload())},
	}
	var packet [128]byte
	n, err := PutRTPPayload(packet[:], RTPAggregationHeader{
		ElementCount:                1,
		StartsNewCodedVideoSequence: true,
	}, sequence[:])
	if err != nil {
		panic(err)
	}
	payloads = append(payloads, append([]byte(nil), packet[:n]...))

	frameHeader := publicDecoderResidualRTPElement(internalobu.TypeFrameHeader, publicDecoderResidualFrameHeaderPayload())
	n, next, more, err := PutRTPFragment(packet[:], frameHeader, 0, 2, false)
	if err != nil {
		panic(err)
	}
	if !more {
		panic("expected frame header RTP fragment")
	}
	payloads = append(payloads, append([]byte(nil), packet[:n]...))
	n, _, more, err = PutRTPFragment(packet[:], frameHeader, next, len(packet), false)
	if err != nil {
		panic(err)
	}
	if more {
		panic("unexpected third frame header RTP fragment")
	}
	payloads = append(payloads, append([]byte(nil), packet[:n]...))

	tile := [...]RTPElement{
		{Data: publicDecoderResidualRTPElement(internalobu.TypeTileGroup, publicDecoderResidualTileData())},
	}
	n, err = PutRTPPayload(packet[:], RTPAggregationHeader{ElementCount: 1}, tile[:])
	if err != nil {
		panic(err)
	}
	payloads = append(payloads, append([]byte(nil), packet[:n]...))

	return payloads
}

func publicDecoderResidualRTPElement(typ internalobu.Type, payload []byte) []byte {
	var header [2]byte
	n, err := internalobu.PutHeader(header[:], internalobu.Header{Type: typ})
	if err != nil {
		panic(err)
	}
	element := make([]byte, 0, n+len(payload))
	element = append(element, header[:n]...)
	element = append(element, payload...)
	return element
}

func publicDecoderResidualSequenceHeaderPayload() []byte {
	var w publicDecoderResidualBitWriter
	w.writeBits(0, 3)  // seq_profile
	w.writeBool(true)  // still_picture
	w.writeBool(true)  // reduced_still_picture_header
	w.writeBits(5, 5)  // seq_level_idx[0]
	w.writeBits(7, 4)  // frame_width_bits_minus_1
	w.writeBits(7, 4)  // frame_height_bits_minus_1
	w.writeBits(15, 8) // max_frame_width_minus_1
	w.writeBits(15, 8) // max_frame_height_minus_1
	w.writeBool(false) // use_128x128_superblock
	w.writeBool(false) // enable_filter_intra
	w.writeBool(true)  // enable_intra_edge_filter
	w.writeBool(false) // enable_superres
	w.writeBool(true)  // enable_cdef
	w.writeBool(false) // enable_restoration
	w.writeBool(false) // high_bitdepth
	w.writeBool(true)  // mono_chrome
	w.writeBool(false) // color_description_present_flag
	w.writeBool(false) // color_range
	w.writeBool(false) // film_grain_params_present
	return w.trailingBits()
}

func publicDecoderResidualRealtimeSequenceHeaderPayload() []byte {
	var w publicDecoderResidualBitWriter
	w.writeBits(0, 3)  // seq_profile
	w.writeBool(false) // still_picture
	w.writeBool(false) // reduced_still_picture_header
	w.writeBool(false) // timing_info_present_flag
	w.writeBool(false) // initial_display_delay_present_flag
	w.writeBits(0, 5)  // operating_points_cnt_minus_1
	w.writeBits(0, 12) // operating_point_idc[0]
	w.writeBits(5, 5)  // seq_level_idx[0]
	w.writeBits(3, 4)  // frame_width_bits_minus_1
	w.writeBits(3, 4)  // frame_height_bits_minus_1
	w.writeBits(15, 4) // max_frame_width_minus_1
	w.writeBits(8, 4)  // max_frame_height_minus_1
	w.writeBool(false) // frame_id_numbers_present_flag
	w.writeBool(false) // use_128x128_superblock
	w.writeBool(true)  // enable_filter_intra
	w.writeBool(true)  // enable_intra_edge_filter
	w.writeBool(true)  // enable_interintra_compound
	w.writeBool(true)  // enable_masked_compound
	w.writeBool(false) // enable_warped_motion
	w.writeBool(true)  // enable_dual_filter
	w.writeBool(false) // enable_order_hint
	w.writeBool(false) // seq_choose_screen_content_tools
	w.writeBits(0, 1)  // seq_force_screen_content_tools
	w.writeBool(false) // enable_superres
	w.writeBool(true)  // enable_cdef
	w.writeBool(false) // enable_restoration
	w.writeBool(false) // high_bitdepth
	w.writeBool(false) // mono_chrome
	w.writeBool(false) // color_description_present_flag
	w.writeBool(false) // color_range
	w.writeBits(0, 2)  // chroma_sample_position
	w.writeBool(true)  // separate_uv_delta_q
	w.writeBool(false) // film_grain_params_present
	return w.trailingBits()
}

func publicDecoderResidualShownKeyFrameHeaderPayload() []byte {
	var w publicDecoderResidualBitWriter
	w.writeBool(false)                                  // show_existing_frame
	w.writeBits(uint64(internalparser.FrameTypeKey), 2) // frame_type
	w.writeBool(true)                                   // show_frame
	w.writeBool(false)                                  // disable_cdf_update
	w.writeBool(false)                                  // frame_size_override_flag
	w.writeBool(false)                                  // render_and_frame_size_different
	w.writeBool(false)                                  // disable_frame_end_update_cdf
	w.writeBool(false)                                  // uniform_tile_spacing_flag
	publicDecoderResidualWriteColorQuantParams(&w)
	publicDecoderResidualWriteZeroSegmentationParams(&w)
	w.writeBool(false) // reduced_tx_set
	return w.bytes()
}

func publicDecoderResidualInterFrameHeaderPayload() []byte {
	var w publicDecoderResidualBitWriter
	w.writeBool(false)                                    // show_existing_frame
	w.writeBits(uint64(internalparser.FrameTypeInter), 2) // frame_type
	w.writeBool(true)                                     // show_frame
	w.writeBool(false)                                    // error_resilient_mode
	w.writeBool(false)                                    // disable_cdf_update
	w.writeBool(false)                                    // frame_size_override_flag
	w.writeBits(0, 3)                                     // primary_ref_frame
	w.writeBits(0x01, 8)                                  // refresh_frame_flags
	for range internalparser.InterRefsPerFrame {
		w.writeBits(0, 3) // ref_frame_idx[i]
	}
	w.writeBool(false) // render_and_frame_size_different
	w.writeBool(false) // allow_high_precision_mv
	w.writeBool(false) // interpolation_filter is fixed
	w.writeBits(0, 2)  // interpolation_filter = EIGHTTAP
	w.writeBool(false) // is_motion_mode_switchable
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(false) // uniform_tile_spacing_flag
	publicDecoderResidualWriteColorQuantParams(&w)
	publicDecoderResidualWriteZeroSegmentationParams(&w)
	w.writeBool(false) // reference_select
	w.writeBool(false) // reduced_tx_set
	for range internalparser.InterRefsPerFrame {
		w.writeBool(false) // global_motion_is_global
	}
	return w.bytes()
}

func publicDecoderResidualShowExistingFrameHeaderPayload(index uint8) []byte {
	var w publicDecoderResidualBitWriter
	w.writeBool(true) // show_existing_frame
	w.writeBits(uint64(index&7), 3)
	return w.bytes()
}

func publicDecoderResidualWriteColorQuantParams(w *publicDecoderResidualBitWriter) {
	w.writeBits(0, 8)  // base_q_idx
	w.writeBool(false) // y_dc_delta_q
	w.writeBool(false) // diff_uv_delta
	w.writeBool(false) // u_dc_delta_q
	w.writeBool(false) // u_ac_delta_q
	w.writeBool(false) // using_qmatrix
}

func publicDecoderResidualWriteZeroSegmentationParams(w *publicDecoderResidualBitWriter) {
	w.writeBool(false) // segmentation_enabled
}

func publicDecoderResidualFrameHeaderPayload() []byte {
	var w publicDecoderResidualBitWriter
	w.writeBool(false) // disable_cdf_update
	w.writeBool(false) // allow_screen_content_tools
	w.writeBool(false) // render_and_frame_size_different
	w.writeBool(false) // uniform_tile_spacing_flag
	w.writeBits(0, 8)  // base_q_idx
	w.writeBool(false) // y_dc_delta_q
	w.writeBool(false) // using_qmatrix
	w.writeBool(false) // segmentation_enabled
	w.writeBool(false) // reduced_tx_set
	return w.trailingBits()
}

// publicDecoderResidualTileData is a valid single-tile payload for the
// 16x16 lossless monochrome still picture described by
// publicDecoderResidualSequenceHeaderPayload and
// publicDecoderResidualFrameHeaderPayload: one 16x16 DC-predicted block whose
// residual carries two non-zero samples (flat 128 with Y[0][0]=140 and
// Y[5][9]=120). It was produced by the encoder's lossless monochrome keyframe
// tile writer, and dav1d and aomdec decode the resulting stream.
func publicDecoderResidualTileData() []byte {
	return []byte{0x0a, 0x1b, 0x59, 0x03, 0xa2, 0x4c, 0x8f, 0x24, 0x87, 0xdc, 0xb4, 0x19, 0x46, 0x81, 0xa4, 0x82, 0x4f, 0x21}
}

func publicDecoderResidualRunnerFrameEvent() internaldecoder.Event {
	payload := make([]byte, 513)
	payload[0] = 0xff
	return internaldecoder.Event{
		Kind: internaldecoder.EventFrame,
		Unit: internalobu.Unit{Payload: payload},
		FrameHeader: internalparser.FrameHeaderPrefix{
			FrameType: internalparser.FrameTypeKey,
			ShowFrame: true,
		},
		FrameSize: FrameSize{
			CodedWidth:          128,
			UpscaledWidth:       128,
			Height:              64,
			SuperResDenominator: 8,
			RefreshFrameFlags:   0xff,
		},
		TileInfo: internalparser.TileInfo{
			TileSizeBytes: 1,
			SBCols:        2,
			SBRows:        1,
			Cols:          2,
			Rows:          1,
			ColStartSB:    [internalparser.MaxTileCols + 1]uint16{0, 1, 2},
			RowStartSB:    [internalparser.MaxTileRows + 1]uint16{0, 1},
		},
		TileGroup: internalparser.TileGroup{
			StartTile: 0,
			EndTile:   1,
			TileCount: 2,
			Final:     true,
		},
		CDEF:         internalparser.CDEFParams{Bits: 0, StrengthCount: 1},
		Quantization: internalparser.QuantizationParams{BaseQIdx: 64},
		TransformRef: internalparser.TransformReferenceParams{TransformMode: internalparser.TransformModeLargest},
	}
}
