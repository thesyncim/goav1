package goav1

import (
	"bytes"
	"errors"
	"testing"

	internalcdef "github.com/thesyncim/goav1/internal/av1/cdef"
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalloopfilter "github.com/thesyncim/goav1/internal/av1/loopfilter"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

func TestPublicDecoderLoopFilterMapAndPostFilterRequestBinding(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	size := FrameSize{CodedWidth: 16, UpscaledWidth: 16, Height: 16, SuperResDenominator: 8}
	cols, rows, length, err := decoderFrameWorkLoopFilterMapShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	if cols != 4 || rows != 4 || length != 16 {
		t.Fatalf("shape=%d,%d,%d want 4,4,16", cols, rows, length)
	}

	records := make([]internalthreading.FrameWorkLoopFilterBlockRecord, length+1)
	records[0].Valid = true
	filterMap, err := bindDecoderFrameWorkLoopFilterMap(sequence, size, records)
	if err != nil {
		t.Fatal(err)
	}
	if filterMap.Stride != uint16(cols) || filterMap.Rows != uint16(rows) || len(filterMap.Records) != length {
		t.Fatalf("map=%+v", filterMap)
	}
	if filterMap.Records[0].Valid {
		t.Fatal("loop-filter map bind did not reset caller-owned records")
	}
	if _, err := bindDecoderFrameWorkLoopFilterMap(sequence, size, records[:length-1]); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("short loop-filter map err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}

	record := publicDecoderLoopFilterRecordAt(0, 0, cols, rows)
	record.DeltaLF = [internaltile.FrameLoopFilterCount]int8{-2, 3, 4, -1}
	publicFillDecoderLoopFilterMap(filterMap, record)
	ctx := internaldecoder.FrameWorkPostFilterContext{
		Event: internaldecoder.Event{
			SequenceHeader: sequence,
			FrameSize:      size,
			Delta: internalparser.DeltaParams{
				DeltaLFPresent: true,
				DeltaLFMulti:   true,
			},
			LoopFilter: internalparser.LoopFilterParams{
				LevelY: [2]uint8{16, 20},
				LevelU: 8,
				LevelV: 4,
			},
		},
		LoopFilterMap: &filterMap,
	}

	scratch, err := ctx.LoopFilterPostFilterScratchLen(internaldecoder.FrameWorkLoopFilterPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req, err := bindDecoderFrameWorkLoopFilterPostFilterRequest(scratch, filterMap, make([]internaldecoder.FrameWorkLoopFilterPostFilterEdge, scratch.Edges), make([]uint32, scratch.Schedule))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := ctx.LoopFilterPostFilterPlan(req)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || int(plan.MICols) != cols || int(plan.MIRows) != rows || int(plan.Cells) != length || plan.Blocks != 1 || plan.LumaTXBs != 1 {
		t.Fatalf("plan=%+v", plan)
	}
	if got := plan.Levels[internalloopfilter.PlaneY][internalloopfilter.EdgeVertical]; got != (internaldecoder.FrameWorkLoopFilterPostFilterLevelStats{Blocks: 1, NonZero: 1, MaxLevel: 14}) {
		t.Fatalf("Y vertical=%+v", got)
	}
	if got := plan.Levels[internalloopfilter.PlaneU][internalloopfilter.EdgeHorizontal]; got.MaxLevel != 12 {
		t.Fatalf("U horizontal=%+v", got)
	}

	manualSize := internaldecoder.FrameWorkLoopFilterPostFilterScratchSize{Edges: 3, Schedule: 3}
	manualReq, err := bindDecoderFrameWorkLoopFilterPostFilterRequest(manualSize, filterMap, make([]internaldecoder.FrameWorkLoopFilterPostFilterEdge, 4), make([]uint32, 4))
	if err != nil {
		t.Fatal(err)
	}
	if len(manualReq.Edges) != 3 || len(manualReq.Schedule) != 3 {
		t.Fatalf("manual req=%+v", manualReq)
	}
	if _, err := bindDecoderFrameWorkLoopFilterPostFilterRequest(manualSize, filterMap, make([]internaldecoder.FrameWorkLoopFilterPostFilterEdge, 2), make([]uint32, 4)); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short loop-filter request err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	if _, err := bindDecoderFrameWorkLoopFilterPostFilterRequest(manualSize, filterMap, make([]internaldecoder.FrameWorkLoopFilterPostFilterEdge, 4), make([]uint32, 2)); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short loop-filter schedule err=%v want %v", err, internalframe.ErrShortBuffer)
	}
}

func TestPublicDecoderRestorationFrameBuffersBinding(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	size := FrameSize{CodedWidth: 280, UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}
	restoration := internalparser.RestorationParams{
		Type:       [3]internalparser.RestorationType{internalparser.RestorationWiener, internalparser.RestorationSGRProj, internalparser.RestorationNone},
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}
	plan, err := decoderFrameWorkRestorationFramePlan(sequence, size, restoration)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.Planes != 3 || plan.UnitRecordLen() == 0 || plan.BoundaryBufferLen() == 0 {
		t.Fatalf("plan=%+v", plan)
	}

	recordBacking := make([]internaltile.RestorationUnitRecord, plan.UnitRecordLen())
	above := make([]uint16, plan.BoundaryBufferLen())
	below := make([]uint16, plan.BoundaryBufferLen())
	buffers, err := bindDecoderFrameWorkRestorationFrameBuffers(sequence, size, restoration, recordBacking, above, below)
	if err != nil {
		t.Fatal(err)
	}
	if buffers.Plan != plan {
		t.Fatalf("buffer plan=%+v want %+v", buffers.Plan, plan)
	}
	for plane := 0; plane < int(plan.Planes); plane++ {
		if len(buffers.Records[plane]) != int(plan.UnitRecords[plane]) {
			t.Fatalf("plane %d records=%d want %d", plane, len(buffers.Records[plane]), plan.UnitRecords[plane])
		}
		if len(buffers.Boundaries[plane].Above) != int(plan.Boundaries[plane].Len) ||
			len(buffers.Boundaries[plane].Below) != int(plan.Boundaries[plane].Len) ||
			buffers.Boundaries[plane].Stride != int(plan.Boundaries[plane].Stride) {
			t.Fatalf("plane %d boundaries=%+v plan=%+v", plane, buffers.Boundaries[plane], plan.Boundaries[plane])
		}
	}
	buffers.Records[0][0].Index = 99
	buffers.Boundaries[0].Above[0] = 77
	if recordBacking[0].Index != 99 || above[0] != 77 {
		t.Fatal("restoration frame buffers do not alias caller-owned backing")
	}
	if err := buffers.ResetRecords(); err != nil {
		t.Fatal(err)
	}
	first := buffers.Records[0][0]
	if first.Index != 0 || first.Unit.Type != internalparser.RestorationNone || first.StripeCount == 0 {
		t.Fatalf("first reset record=%+v", first)
	}
	if _, err := bindDecoderFrameWorkRestorationFrameBuffers(sequence, size, restoration, recordBacking[:len(recordBacking)-1], above, below); !errors.Is(err, internaltile.ErrJobBufferTooSmall) {
		t.Fatalf("short restoration records err=%v want %v", err, internaltile.ErrJobBufferTooSmall)
	}

	allNone := internalparser.RestorationParams{}
	noneBuffers, err := bindDecoderFrameWorkRestorationFrameBuffers(sequence, size, allNone, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if noneBuffers.Plan.Active || noneBuffers.Plan.UnitRecordLen() != 0 || noneBuffers.Plan.BoundaryBufferLen() != 0 {
		t.Fatalf("none buffers=%+v", noneBuffers.Plan)
	}
}

func TestPublicDecoderFrameWorkSupportedPostFilterScratchRunner(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	sequence.ColorConfig.MonoChrome = true
	sequence.ColorConfig.SubsamplingX = false
	sequence.ColorConfig.SubsamplingY = false
	size := FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	cdef := internalparser.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [internalparser.MaxCDEFStrengths]uint8{63},
	}
	sideScratchSize, err := decoderFrameWorkSideDataScratchLen(sequence, size, cdef, internalparser.RestorationParams{})
	if err != nil {
		t.Fatal(err)
	}
	sideScratch := decoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, sideScratchSize.CDEFIndexMap),
		CDEFReadMap:              make([]bool, sideScratchSize.CDEFReadMap),
		LoopFilterMap:            make([]internalthreading.FrameWorkLoopFilterBlockRecord, sideScratchSize.LoopFilterMap),
		RestorationRecords:       make([]internaltile.RestorationUnitRecord, sideScratchSize.RestorationRecords),
		RestorationBoundaryAbove: make([]uint16, sideScratchSize.RestorationBoundaryAbove),
		RestorationBoundaryBelow: make([]uint16, sideScratchSize.RestorationBoundaryBelow),
	}
	side, err := bindDecoderFrameWorkSideData(sequence, size, cdef, internalparser.RestorationParams{}, sideScratch)
	if err != nil {
		t.Fatal(err)
	}
	side.CDEFIndexMap.Read[0] = true

	output := publicDecoderPostFilterFrame(t, FrameFormat{
		Width:      int(size.CodedWidth),
		Height:     int(size.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      32,
	})
	publicFillDecoderPostFilterPlane(output.Y)
	before := append([]byte(nil), output.Y.Pix[:output.Y.Width]...)
	ctx := internaldecoder.FrameWorkPostFilterContext{
		Event: internaldecoder.Event{
			Kind:           internaldecoder.EventTileGroup,
			SequenceHeader: sequence,
			FrameSize:      size,
			CDEF:           cdef,
			TileGroup:      internalparser.TileGroup{Final: true},
		},
		Output:                  output,
		CDEFIndexMap:            &side.CDEFIndexMap,
		LoopFilterMap:           &side.LoopFilterMap,
		RestorationFrameBuffers: &side.RestorationFrameBuffers,
	}
	var probe decoderFrameWorkSupportedPostFilterScratchRunner
	exact, err := probe.ScratchLen(ctx)
	if err != nil {
		t.Fatal(err)
	}
	arenaSize := decoderFrameWorkPostFilterRequestScratchLen(exact)
	if exact.CDEF.Input == 0 || arenaSize.Uint16Scratch == 0 {
		t.Fatalf("exact scratch=%+v arena=%+v", exact, arenaSize)
	}
	runner := decoderFrameWorkSupportedPostFilterScratchRunner{
		Scratch: publicDecoderPostFilterRequestScratch(arenaSize),
	}
	if err := runner.Apply(ctx); err != nil {
		t.Fatal(err)
	}
	if !runner.Result.Completed.Has(internaldecoder.FrameWorkPostFilterCDEF) ||
		runner.Context.RemainingPostFilters() != 0 ||
		runner.Size.CDEF.Input != exact.CDEF.Input ||
		runner.Request.CDEF.IndexMap.Rows != side.CDEFIndexMap.Rows {
		t.Fatalf("runner size=%+v request=%+v result=%+v remaining=%b", runner.Size, runner.Request, runner.Result, runner.Context.RemainingPostFilters())
	}
	if string(output.Y.Pix[:output.Y.Width]) == string(before) {
		t.Fatal("runner did not apply CDEF")
	}

	fromCtx := decoderFrameWorkPostFilterRequestSideDataFromContext(ctx)
	if fromCtx.CDEFIndexMap.Rows != side.CDEFIndexMap.Rows ||
		fromCtx.LoopFilterMap.Rows != side.LoopFilterMap.Rows ||
		len(fromCtx.RestorationRecords[0]) != len(side.RestorationFrameBuffers.Records[0]) {
		t.Fatalf("side from context=%+v side=%+v", fromCtx, side)
	}
	var nilRunner *decoderFrameWorkSupportedPostFilterScratchRunner
	if err := nilRunner.Apply(ctx); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil runner err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	shortRunner := runner
	shortRunner.Scratch.Uint16Scratch = shortRunner.Scratch.Uint16Scratch[:arenaSize.Uint16Scratch-1]
	if err := shortRunner.Apply(ctx); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short scratch err=%v want %v", err, internalframe.ErrShortBuffer)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		err = runner.Apply(ctx)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("runner allocated: %f", allocs)
	}
}

func TestPublicPostFilterScratchRunnerParallelCDEFAllocsAndParity(t *testing.T) {
	const width, height = 256, 512
	sequence := publicDecoderPostFilterSequence()
	sequence.EnableCDEF = true
	cdef := internalparser.CDEFParams{
		Damping: 5, StrengthCount: 2,
		YStrength:  [internalparser.MaxCDEFStrengths]uint8{9, 31},
		UVStrength: [internalparser.MaxCDEFStrengths]uint8{5, 17},
	}
	size := FrameSize{CodedWidth: width, UpscaledWidth: width, Height: height, SuperResDenominator: 8}
	format := FrameFormat{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	build := func(parallel *internaldecoder.FrameWorkPostFilterParallel) (*Frame, internaldecoder.FrameWorkPostFilterContext, *decoderFrameWorkSupportedPostFilterScratchRunner) {
		sideSize, err := decoderFrameWorkSideDataScratchLen(sequence, size, cdef, internalparser.RestorationParams{})
		if err != nil {
			t.Fatal(err)
		}
		sideScratch := decoderFrameWorkSideDataScratch{
			CDEFIndexMap:             make([]uint8, sideSize.CDEFIndexMap),
			CDEFReadMap:              make([]bool, sideSize.CDEFReadMap),
			LoopFilterMap:            make([]internalthreading.FrameWorkLoopFilterBlockRecord, sideSize.LoopFilterMap),
			RestorationRecords:       make([]internaltile.RestorationUnitRecord, sideSize.RestorationRecords),
			RestorationBoundaryAbove: make([]uint16, sideSize.RestorationBoundaryAbove),
			RestorationBoundaryBelow: make([]uint16, sideSize.RestorationBoundaryBelow),
		}
		side, err := bindDecoderFrameWorkSideData(sequence, size, cdef, internalparser.RestorationParams{}, sideScratch)
		if err != nil {
			t.Fatal(err)
		}
		for i := range side.CDEFIndexMap.Index {
			side.CDEFIndexMap.Index[i] = uint8(i % 2)
			side.CDEFIndexMap.Read[i] = true
		}
		out := publicDecoderPostFilterFrame(t, format)
		publicFillDecoderPostFilterPlane(out.Y)
		publicFillDecoderPostFilterPlane(out.U)
		publicFillDecoderPostFilterPlane(out.V)
		ctx := internaldecoder.FrameWorkPostFilterContext{
			Event: internaldecoder.Event{
				Kind:           internaldecoder.EventTileGroup,
				SequenceHeader: sequence,
				FrameSize:      size,
				CDEF:           cdef,
				TileGroup:      internalparser.TileGroup{Final: true},
			},
			Output:                  out,
			CDEFIndexMap:            &side.CDEFIndexMap,
			LoopFilterMap:           &side.LoopFilterMap,
			RestorationFrameBuffers: &side.RestorationFrameBuffers,
		}.WithCompletedPostFilters(internaldecoder.FrameWorkPostFilterLoopFilter)
		var probe decoderFrameWorkSupportedPostFilterScratchRunner
		exact, err := probe.ScratchLen(ctx)
		if err != nil {
			t.Fatal(err)
		}
		runner := &decoderFrameWorkSupportedPostFilterScratchRunner{
			Scratch:  publicDecoderPostFilterRequestScratch(decoderFrameWorkPostFilterRequestScratchLen(exact)),
			Parallel: parallel,
		}
		return out, ctx, runner
	}

	serialOut, serialCtx, serialRunner := build(nil)
	if err := serialRunner.Apply(serialCtx); err != nil {
		t.Fatalf("serial public post-filter: %v", err)
	}
	parallel := &internaldecoder.FrameWorkPostFilterParallel{Workers: 2}
	defer parallel.Close()
	parallelOut, parallelCtx, runner := build(parallel)
	initialY := append([]byte(nil), parallelOut.Y.Pix...)
	initialU := append([]byte(nil), parallelOut.U.Pix...)
	initialV := append([]byte(nil), parallelOut.V.Pix...)
	apply := func() {
		copy(parallelOut.Y.Pix, initialY)
		copy(parallelOut.U.Pix, initialU)
		copy(parallelOut.V.Pix, initialV)
		if err := runner.Apply(parallelCtx); err != nil {
			t.Fatalf("parallel public post-filter: %v", err)
		}
		if runner.Result.CDEF.Blocks == 0 || !runner.Result.Completed.Has(internaldecoder.FrameWorkPostFilterCDEF) {
			t.Fatalf("public runner did not complete active CDEF: %+v", runner.Result)
		}
	}
	apply()
	allocs := testing.AllocsPerRun(5, apply)
	if allocs != 0 {
		t.Fatalf("warmed public two-worker post-filter runner allocated %g times per apply", allocs)
	}
	for _, planes := range [][2]FramePlane{{serialOut.Y, parallelOut.Y}, {serialOut.U, parallelOut.U}, {serialOut.V, parallelOut.V}} {
		if !bytes.Equal(planes[0].Pix, planes[1].Pix) {
			t.Fatal("public two-worker post-filter output differs from serial output")
		}
	}
}

func TestPublicDecoderFrameWorkPostFilterScratchSizeMax(t *testing.T) {
	arenaA := decoderFrameWorkPostFilterRequestScratchSize{
		LoopFilterEdges:    1,
		LoopFilterSchedule: 6,
		CDEFDirectionGrid:  7,
		CDEFVarianceGrid:   3,
		ByteScratch:        11,
		Uint16Scratch:      5,
		Int16Scratch:       13,
		Int32Scratch:       2,
	}
	arenaB := decoderFrameWorkPostFilterRequestScratchSize{
		LoopFilterEdges:    4,
		LoopFilterSchedule: 3,
		CDEFDirectionGrid:  2,
		CDEFVarianceGrid:   8,
		ByteScratch:        6,
		Uint16Scratch:      20,
		Int16Scratch:       1,
		Int32Scratch:       9,
	}
	if got, want := arenaA.Max(arenaB), (decoderFrameWorkPostFilterRequestScratchSize{
		LoopFilterEdges:    4,
		LoopFilterSchedule: 6,
		CDEFDirectionGrid:  7,
		CDEFVarianceGrid:   8,
		ByteScratch:        11,
		Uint16Scratch:      20,
		Int16Scratch:       13,
		Int32Scratch:       9,
	}); got != want {
		t.Fatalf("postfilter scratch max=%+v want %+v", got, want)
	}

	sideA := decoderFrameWorkSideDataScratchSize{
		CDEFIndexMap:             1,
		CDEFReadMap:              5,
		LoopFilterMap:            2,
		RestorationRecords:       9,
		RestorationBoundaryAbove: 3,
		RestorationBoundaryBelow: 4,
	}
	sideB := decoderFrameWorkSideDataScratchSize{
		CDEFIndexMap:             8,
		CDEFReadMap:              2,
		LoopFilterMap:            6,
		RestorationRecords:       1,
		RestorationBoundaryAbove: 7,
		RestorationBoundaryBelow: 3,
	}
	if got, want := sideA.Max(sideB), (decoderFrameWorkSideDataScratchSize{
		CDEFIndexMap:             8,
		CDEFReadMap:              5,
		LoopFilterMap:            6,
		RestorationRecords:       9,
		RestorationBoundaryAbove: 7,
		RestorationBoundaryBelow: 4,
	}); got != want {
		t.Fatalf("side-data scratch max=%+v want %+v", got, want)
	}
}

func TestPublicDecoderFrameWorkSideDataSet(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	size := FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	cdef := internalparser.CDEFParams{Bits: 1, StrengthCount: 2}
	restoration := internalparser.RestorationParams{
		Type:       [3]internalparser.RestorationType{internalparser.RestorationWiener, internalparser.RestorationSGRProj, internalparser.RestorationNone},
		UnitSizeY:  64,
		UnitSizeUV: 64,
	}
	scratchSize, err := decoderFrameWorkSideDataScratchLen(sequence, size, cdef, restoration)
	if err != nil {
		t.Fatal(err)
	}
	sideScratch := decoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, scratchSize.CDEFIndexMap),
		CDEFReadMap:              make([]bool, scratchSize.CDEFReadMap),
		LoopFilterMap:            make([]internalthreading.FrameWorkLoopFilterBlockRecord, scratchSize.LoopFilterMap),
		RestorationRecords:       make([]internaltile.RestorationUnitRecord, scratchSize.RestorationRecords),
		RestorationBoundaryAbove: make([]uint16, scratchSize.RestorationBoundaryAbove),
		RestorationBoundaryBelow: make([]uint16, scratchSize.RestorationBoundaryBelow),
	}
	side, err := bindDecoderFrameWorkSideData(sequence, size, cdef, restoration, sideScratch)
	if err != nil {
		t.Fatal(err)
	}

	var nilState *internaldecoder.FrameWorkState
	if err := setDecoderFrameWorkSideData(nilState, side); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("nil state err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}
	var inactive internaldecoder.FrameWorkState
	if err := setDecoderFrameWorkSideData(&inactive, side); !errors.Is(err, internaldecoder.ErrInvalidFrameWorkState) {
		t.Fatalf("inactive state err=%v want %v", err, internaldecoder.ErrInvalidFrameWorkState)
	}

	pool := publicDecoderPostFilterFramePool(t, FrameFormat{
		Width:        int(size.CodedWidth),
		Height:       int(size.Height),
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	}, 1)
	begin := internaldecoder.Event{
		Kind:        internaldecoder.EventFrameHeader,
		FrameHeader: internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeKey},
		FrameSize:   size,
	}
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	plan, output, err := state.Begin(&refs, &pool, sequence, &begin, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	final := begin
	final.Kind = internaldecoder.EventTileGroup
	final.TileGroup.Final = true
	final.FrameSize.RefreshFrameFlags = 0xff
	step := internaldecoder.FrameWorkStep{
		Kind: internaldecoder.FrameWorkStepTile,
		Tile: internaldecoder.FrameTileWorkPlan{Surface: plan.Surface},
	}

	badSide := side
	badSide.LoopFilterMap = internaldecoder.FrameWorkLoopFilterMap{Stride: 1, Rows: 1}
	if err := setDecoderFrameWorkSideData(&state, badSide); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("invalid loop-filter map err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	ctx, err := state.PostFilterContext(&pool, &final, step, false)
	if err != nil {
		t.Fatal(err)
	}
	if ctx.CDEFIndexMap != nil || ctx.LoopFilterMap != nil || ctx.RestorationFrameBuffers != nil {
		t.Fatalf("failed side-data attach partially mutated state: ctx=%+v", ctx)
	}

	side.CDEFIndexMap.Index[0] = 1
	side.CDEFIndexMap.Read[0] = true
	side.LoopFilterMap.Records[0].Valid = true
	side.RestorationFrameBuffers.Records[0][0].Index = 99
	var attachErr error
	allocs := testing.AllocsPerRun(1000, func() {
		attachErr = setDecoderFrameWorkSideData(&state, side)
	})
	if attachErr != nil {
		t.Fatal(attachErr)
	}
	if allocs != 0 {
		t.Fatalf("SetDecoderFrameWorkSideData allocated: %f", allocs)
	}
	ctx, err = state.PostFilterContext(&pool, &final, step, true)
	if err != nil {
		t.Fatal(err)
	}
	if ctx.Output != output || ctx.CDEFIndexMap == nil || ctx.LoopFilterMap == nil || ctx.RestorationFrameBuffers == nil ||
		ctx.CDEFIndexMap.Stride != side.CDEFIndexMap.Stride ||
		ctx.LoopFilterMap.Stride != side.LoopFilterMap.Stride ||
		len(ctx.RestorationFrameBuffers.Records[0]) != len(side.RestorationFrameBuffers.Records[0]) ||
		!ctx.ExecutedTileWork {
		t.Fatalf("ctx=%+v output=%p side=%+v", ctx, output, side)
	}
	if ctx.CDEFIndexMap.Index[0] != 0 || ctx.CDEFIndexMap.Read[0] ||
		ctx.LoopFilterMap.Records[0].Valid ||
		ctx.RestorationFrameBuffers.Records[0][0].Index != 0 ||
		ctx.RestorationFrameBuffers.Records[0][0].Unit.Type != internalparser.RestorationNone {
		t.Fatalf("attached side data was not reset: cdef=%d/%v loop=%+v restoration=%+v",
			ctx.CDEFIndexMap.Index[0], ctx.CDEFIndexMap.Read[0], ctx.LoopFilterMap.Records[0], ctx.RestorationFrameBuffers.Records[0][0])
	}
	ctx.CDEFIndexMap.Read[0] = true
	ctx.LoopFilterMap.Records[0].Valid = true
	ctx.RestorationFrameBuffers.Records[0][0].Unit.Type = internalparser.RestorationWiener
	if !side.CDEFIndexMap.Read[0] ||
		!side.LoopFilterMap.Records[0].Valid ||
		side.RestorationFrameBuffers.Records[0][0].Unit.Type != internalparser.RestorationWiener {
		t.Fatal("attached side data did not preserve caller-owned backing")
	}
}

func TestPublicDecoderCDEFIndexMapAndPostFilterRequestBinding(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	size := FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	cols, rows, length, err := decoderFrameWorkCDEFIndexMapShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	if cols != 1 || rows != 1 || length != 1 {
		t.Fatalf("shape=%d,%d,%d want 1,1,1", cols, rows, length)
	}
	index := make([]uint8, length)
	read := make([]bool, length)
	index[0], read[0] = 3, true
	cdefMap, err := bindDecoderFrameWorkCDEFIndexMap(sequence, size, internalparser.CDEFParams{Bits: 2, StrengthCount: 4}, index, read)
	if err != nil {
		t.Fatal(err)
	}
	if cdefMap.Stride != uint16(cols) || cdefMap.Rows != uint16(rows) || len(cdefMap.Index) != length || len(cdefMap.Read) != length {
		t.Fatalf("map=%+v", cdefMap)
	}
	badIndex := []uint8{4}
	if _, err := bindDecoderFrameWorkCDEFIndexMap(sequence, size, internalparser.CDEFParams{Bits: 2, StrengthCount: 4}, badIndex, read); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("bad index err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}

	output := publicDecoderPostFilterFrame(t, FrameFormat{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	publicFillDecoderPostFilterPlane(output.Y)
	before := append([]byte(nil), output.Y.Pix[:output.Y.Width]...)
	ctx := internaldecoder.FrameWorkPostFilterContext{
		Event: internaldecoder.Event{
			SequenceHeader: sequence,
			FrameSize:      size,
			CDEF: internalparser.CDEFParams{
				Bits:          2,
				Damping:       5,
				StrengthCount: 4,
				YStrength:     [internalparser.MaxCDEFStrengths]uint8{0, 0, 0, 63},
			},
		},
		Output: output,
	}
	scratch, err := ctx.CDEFPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	var sampleScratch [3][]uint16
	var dstScratch [3][]uint16
	for plane := range 3 {
		sampleScratch[plane] = make([]uint16, scratch.Samples[plane])
		dstScratch[plane] = make([]uint16, scratch.Dst[plane])
	}
	req, err := bindDecoderFrameWorkCDEFPostFilterRequest(
		scratch,
		cdefMap,
		sampleScratch,
		dstScratch,
		make([]internalcdef.DirectionGrid, scratch.DirectionGrid),
		make([]internalcdef.VarianceGrid, scratch.VarianceGrid),
		make([]uint16, scratch.Input),
		make([]uint16, scratch.UnitDst),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ctx.ApplyCDEFPostFilter(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 1 || result.Units != 1 || result.Blocks == 0 {
		t.Fatalf("result=%+v", result)
	}
	if string(output.Y.Pix[:output.Y.Width]) == string(before) {
		t.Fatal("CDEF did not modify luma output")
	}
	if _, err := bindDecoderFrameWorkCDEFPostFilterRequest(scratch, cdefMap, sampleScratch, dstScratch, nil, nil, nil, nil); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short CDEF bind err=%v want %v", err, internalframe.ErrShortBuffer)
	}
}

func TestPublicDecoderSuperResPostFilterRequestBinding(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	sequence.EnableSuperRes = true
	size := FrameSize{CodedWidth: 64, UpscaledWidth: 96, Height: 32, SuperResEnabled: true, SuperResDenominator: 12}
	format, err := frameCodedFormatFromHeaders(sequence, size, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := publicDecoderPostFilterFrame(t, format)
	publicFillDecoderPostFilterPlane(output.Y)
	ctx := internaldecoder.FrameWorkPostFilterContext{
		Event: internaldecoder.Event{
			SequenceHeader: sequence,
			FrameSize:      size,
		},
		Output: output,
	}
	scratch, err := ctx.SuperResPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	var codedScratch [3][]uint16
	var outputScratch [3][]uint16
	for plane := range 3 {
		codedScratch[plane] = make([]uint16, scratch.CodedSamples[plane])
		outputScratch[plane] = make([]uint16, scratch.OutputSamples[plane])
	}
	req, err := bindDecoderFrameWorkSuperResPostFilterRequest(scratch, make([]byte, scratch.OutputFrame), codedScratch, outputScratch)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ctx.ApplySuperResPostFilter(req)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Plan.Active || result.Output.Format.Width != int(size.UpscaledWidth) || result.Planes != 3 {
		t.Fatalf("result=%+v output format=%+v", result, result.Output.Format)
	}
	if _, err := bindDecoderFrameWorkSuperResPostFilterRequest(scratch, nil, codedScratch, outputScratch); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short superres bind err=%v want %v", err, internalframe.ErrShortBuffer)
	}
}

func TestPublicDecoderRestorationAndFilmGrainRequestBinding(t *testing.T) {
	restorationSize := internaldecoder.FrameWorkRestorationPostFilterScratchSize{}
	restorationSize.Samples.DataLen = 11
	restorationSize.Samples.DstLen = 13
	restorationSize.Apply.Unit.Wiener = 17
	restorationSize.Apply.Unit.SGRProj = 19
	restorationSize.Apply.Boundary.Above = 23
	restorationSize.Apply.Boundary.Below = 29

	restorationReq, err := bindDecoderFrameWorkRestorationPostFilterRequest(
		restorationSize,
		[3][]internaltile.RestorationUnitRecord{{}},
		[3]internaltile.RestorationStripeBoundaries{},
		make([]uint16, 12),
		make([]uint16, 14),
		make([]uint16, 18),
		make([]int32, 20),
		make([]uint16, 24),
		make([]uint16, 30),
		true,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(restorationReq.DataScratch) != 11 ||
		len(restorationReq.DstScratch) != 13 ||
		len(restorationReq.Scratch.Unit.Wiener) != 17 ||
		len(restorationReq.Scratch.Unit.SGRProj) != 19 ||
		len(restorationReq.Scratch.Boundary.Above) != 23 ||
		len(restorationReq.Scratch.Boundary.Below) != 29 ||
		!restorationReq.Optimized {
		t.Fatalf("restoration request=%+v scratch=%+v", restorationReq, restorationReq.Scratch)
	}
	if _, err := bindDecoderFrameWorkRestorationPostFilterRequest(restorationSize, [3][]internaltile.RestorationUnitRecord{}, [3]internaltile.RestorationStripeBoundaries{}, nil, nil, nil, nil, nil, nil, false); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short restoration bind err=%v want %v", err, internalframe.ErrShortBuffer)
	}

	filmSize := internaldecoder.FrameWorkFilmGrainPostFilterScratchSize{
		OutputFrame:   29,
		LumaGrain:     31,
		ChromaGrain:   [2]int{37, 41},
		LumaSamples:   43,
		ChromaSamples: [2]int{47, 53},
	}
	filmReq, err := bindDecoderFrameWorkFilmGrainPostFilterRequest(
		filmSize,
		make([]byte, 30),
		make([]int16, 32),
		[2][]int16{make([]int16, 38), make([]int16, 42)},
		make([]uint16, 44),
		[2][]uint16{make([]uint16, 48), make([]uint16, 54)},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(filmReq.LumaGrain) != 31 ||
		len(filmReq.OutputFrame) != 29 ||
		len(filmReq.ChromaGrain[0]) != 37 ||
		len(filmReq.ChromaGrain[1]) != 41 ||
		len(filmReq.LumaSamples) != 43 ||
		len(filmReq.ChromaSamples[0]) != 47 ||
		len(filmReq.ChromaSamples[1]) != 53 {
		t.Fatalf("film request=%+v", filmReq)
	}
	if _, err := bindDecoderFrameWorkFilmGrainPostFilterRequest(filmSize, nil, nil, [2][]int16{}, nil, [2][]uint16{}); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short filmgrain bind err=%v want %v", err, internalframe.ErrShortBuffer)
	}
}

func TestPublicDecoderPostFilterRequestBinding(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	size := FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	_, _, loopFilterLength, err := decoderFrameWorkLoopFilterMapShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	loopFilterMap, err := bindDecoderFrameWorkLoopFilterMap(sequence, size, make([]internalthreading.FrameWorkLoopFilterBlockRecord, loopFilterLength))
	if err != nil {
		t.Fatal(err)
	}
	_, _, cdefLength, err := decoderFrameWorkCDEFIndexMapShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := bindDecoderFrameWorkCDEFIndexMap(sequence, size, internalparser.CDEFParams{Bits: 1, StrengthCount: 2}, make([]uint8, cdefLength), make([]bool, cdefLength))
	if err != nil {
		t.Fatal(err)
	}

	scratch := internaldecoder.FrameWorkPostFilterScratchSize{
		LoopFilter: internaldecoder.FrameWorkLoopFilterPostFilterScratchSize{Edges: 3, Schedule: 5},
		CDEF: internaldecoder.FrameWorkCDEFPostFilterScratchSize{
			Samples:       [3]int{64, 16, 16},
			Dst:           [3]int{64, 16, 16},
			DirectionGrid: 1,
			VarianceGrid:  1,
			Input:         internalcdef.InputBufferSize,
			UnitDst:       internalcdef.InputBufferSize,
		},
		SuperRes: internaldecoder.FrameWorkSuperResPostFilterScratchSize{
			OutputFrame:   128,
			CodedSamples:  [3]int{64, 16, 16},
			OutputSamples: [3]int{96, 24, 24},
		},
		Restoration: internaldecoder.FrameWorkRestorationPostFilterScratchSize{
			Samples: internaltile.RestorationFrameSampleScratchSize{DataLen: 5, DstLen: 7},
			Apply: internaltile.RestorationUnitRecordBoundaryScratchSize{
				Unit:     internaltile.RestorationUnitScratchSize{Wiener: 11, SGRProj: 13},
				Boundary: internaltile.RestorationStripeBoundaryScratchSize{Above: 17, Below: 19},
			},
		},
		FilmGrain: internaldecoder.FrameWorkFilmGrainPostFilterScratchSize{
			OutputFrame:   127,
			LumaGrain:     23,
			ChromaGrain:   [2]int{29, 31},
			LumaSamples:   37,
			ChromaSamples: [2]int{41, 43},
		},
	}
	var sampleScratch [3][]uint16
	var dstScratch [3][]uint16
	var codedScratch [3][]uint16
	var outputScratch [3][]uint16
	for plane := range 3 {
		sampleScratch[plane] = make([]uint16, scratch.CDEF.Samples[plane]+1)
		dstScratch[plane] = make([]uint16, scratch.CDEF.Dst[plane]+1)
		codedScratch[plane] = make([]uint16, scratch.SuperRes.CodedSamples[plane]+1)
		outputScratch[plane] = make([]uint16, scratch.SuperRes.OutputSamples[plane]+1)
	}
	buffers := decoderFrameWorkPostFilterRequestBuffers{
		LoopFilterMap:             loopFilterMap,
		LoopFilterTrustedCoverage: true,
		LoopFilterEdges:           make([]internaldecoder.FrameWorkLoopFilterPostFilterEdge, scratch.LoopFilter.Edges+1),
		LoopFilterSchedule:        make([]uint32, scratch.LoopFilter.Schedule+1),

		CDEFIndexMap:       cdefMap,
		CDEFSampleScratch:  sampleScratch,
		CDEFDstScratch:     dstScratch,
		CDEFDirectionGrid:  make([]internalcdef.DirectionGrid, scratch.CDEF.DirectionGrid+1),
		CDEFVarianceGrid:   make([]internalcdef.VarianceGrid, scratch.CDEF.VarianceGrid+1),
		CDEFInputScratch:   make([]uint16, scratch.CDEF.Input+1),
		CDEFUnitDstScratch: make([]uint16, scratch.CDEF.UnitDst+1),

		SuperResOutputFrame:   make([]byte, scratch.SuperRes.OutputFrame+1),
		SuperResCodedScratch:  codedScratch,
		SuperResOutputScratch: outputScratch,
		FilmGrainOutputFrame:  make([]byte, scratch.FilmGrain.OutputFrame+1),

		RestorationRecords:              [3][]internaltile.RestorationUnitRecord{{{Index: 7}}},
		RestorationDataScratch:          make([]uint16, scratch.Restoration.Samples.DataLen+1),
		RestorationDstScratch:           make([]uint16, scratch.Restoration.Samples.DstLen+1),
		RestorationWienerScratch:        make([]uint16, scratch.Restoration.Apply.Unit.Wiener+1),
		RestorationSGRProjScratch:       make([]int32, scratch.Restoration.Apply.Unit.SGRProj+1),
		RestorationBoundaryAboveScratch: make([]uint16, scratch.Restoration.Apply.Boundary.Above+1),
		RestorationBoundaryBelowScratch: make([]uint16, scratch.Restoration.Apply.Boundary.Below+1),
		RestorationOptimized:            true,

		FilmGrainLumaGrain:     make([]int16, scratch.FilmGrain.LumaGrain+1),
		FilmGrainChromaGrain:   [2][]int16{make([]int16, scratch.FilmGrain.ChromaGrain[0]+1), make([]int16, scratch.FilmGrain.ChromaGrain[1]+1)},
		FilmGrainLumaSamples:   make([]uint16, scratch.FilmGrain.LumaSamples+1),
		FilmGrainChromaSamples: [2][]uint16{make([]uint16, scratch.FilmGrain.ChromaSamples[0]+1), make([]uint16, scratch.FilmGrain.ChromaSamples[1]+1)},
	}

	req, err := bindDecoderFrameWorkPostFilterRequest(scratch, buffers)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.LoopFilter.Edges) != scratch.LoopFilter.Edges ||
		len(req.LoopFilter.Schedule) != scratch.LoopFilter.Schedule ||
		!req.LoopFilter.TrustedCoverage ||
		len(req.CDEF.SampleScratch[0]) != scratch.CDEF.Samples[0] ||
		len(req.CDEF.InputScratch) != scratch.CDEF.Input ||
		len(req.SuperRes.OutputFrame) != scratch.SuperRes.OutputFrame ||
		len(req.SuperRes.OutputScratch[1]) != scratch.SuperRes.OutputSamples[1] ||
		len(req.FilmGrain.OutputFrame) != scratch.FilmGrain.OutputFrame ||
		len(req.Restoration.DataScratch) != scratch.Restoration.Samples.DataLen ||
		len(req.Restoration.Scratch.Boundary.Below) != scratch.Restoration.Apply.Boundary.Below ||
		!req.Restoration.Optimized ||
		len(req.FilmGrain.ChromaSamples[1]) != scratch.FilmGrain.ChromaSamples[1] {
		t.Fatalf("request=%+v", req)
	}
	if req.LoopFilter.Map.Stride != loopFilterMap.Stride || req.CDEF.IndexMap.Rows != cdefMap.Rows {
		t.Fatalf("side maps loop=%+v cdef=%+v", req.LoopFilter.Map, req.CDEF.IndexMap)
	}

	if zeroReq, err := bindDecoderFrameWorkPostFilterRequest(internaldecoder.FrameWorkPostFilterScratchSize{}, decoderFrameWorkPostFilterRequestBuffers{}); err != nil || len(zeroReq.SuperRes.OutputFrame) != 0 {
		t.Fatalf("zero request=%+v err=%v", zeroReq, err)
	}

	arenaSize := decoderFrameWorkPostFilterRequestScratchLen(scratch)
	wantUint16Scratch := scratch.CDEF.Input + scratch.CDEF.UnitDst +
		scratch.Restoration.Samples.DataLen + scratch.Restoration.Samples.DstLen +
		scratch.Restoration.Apply.Unit.Wiener +
		scratch.Restoration.Apply.Boundary.Above + scratch.Restoration.Apply.Boundary.Below +
		scratch.FilmGrain.LumaSamples
	for plane := range 3 {
		wantUint16Scratch += scratch.CDEF.Samples[plane] + scratch.CDEF.Dst[plane]
		wantUint16Scratch += scratch.SuperRes.CodedSamples[plane] + scratch.SuperRes.OutputSamples[plane]
		if plane < 2 {
			wantUint16Scratch += scratch.FilmGrain.ChromaSamples[plane]
		}
	}
	wantInt16Scratch := scratch.FilmGrain.LumaGrain + scratch.FilmGrain.ChromaGrain[0] + scratch.FilmGrain.ChromaGrain[1]
	if arenaSize.LoopFilterEdges != scratch.LoopFilter.Edges ||
		arenaSize.LoopFilterSchedule != scratch.LoopFilter.Schedule ||
		arenaSize.CDEFDirectionGrid != scratch.CDEF.DirectionGrid ||
		arenaSize.CDEFVarianceGrid != scratch.CDEF.VarianceGrid ||
		arenaSize.ByteScratch != scratch.SuperRes.OutputFrame+scratch.FilmGrain.OutputFrame ||
		arenaSize.Uint16Scratch != wantUint16Scratch ||
		arenaSize.Int16Scratch != wantInt16Scratch ||
		arenaSize.Int32Scratch != scratch.Restoration.Apply.Unit.SGRProj {
		t.Fatalf("arena size=%+v want uint16=%d int16=%d", arenaSize, wantUint16Scratch, wantInt16Scratch)
	}
	side := decoderFrameWorkPostFilterRequestSideData{
		LoopFilterMap:             loopFilterMap,
		LoopFilterTrustedCoverage: true,
		CDEFIndexMap:              cdefMap,
		RestorationRecords:        buffers.RestorationRecords,
		RestorationBoundaries:     buffers.RestorationBoundaries,
		RestorationOptimized:      buffers.RestorationOptimized,
	}
	flatScratch := decoderFrameWorkPostFilterRequestScratch{
		LoopFilterEdges:    make([]internaldecoder.FrameWorkLoopFilterPostFilterEdge, arenaSize.LoopFilterEdges+1),
		LoopFilterSchedule: make([]uint32, arenaSize.LoopFilterSchedule+1),
		CDEFDirectionGrid:  make([]internalcdef.DirectionGrid, arenaSize.CDEFDirectionGrid+1),
		CDEFVarianceGrid:   make([]internalcdef.VarianceGrid, arenaSize.CDEFVarianceGrid+1),
		ByteScratch:        make([]byte, arenaSize.ByteScratch+1),
		Uint16Scratch:      make([]uint16, arenaSize.Uint16Scratch+1),
		Int16Scratch:       make([]int16, arenaSize.Int16Scratch+1),
		Int32Scratch:       make([]int32, arenaSize.Int32Scratch+1),
	}
	arenaReq, err := bindDecoderFrameWorkPostFilterRequestFromScratch(scratch, side, flatScratch)
	if err != nil {
		t.Fatal(err)
	}
	if len(arenaReq.LoopFilter.Edges) != scratch.LoopFilter.Edges ||
		len(arenaReq.CDEF.DstScratch[2]) != scratch.CDEF.Dst[2] ||
		len(arenaReq.SuperRes.CodedScratch[0]) != scratch.SuperRes.CodedSamples[0] ||
		len(arenaReq.Restoration.Scratch.Unit.SGRProj) != scratch.Restoration.Apply.Unit.SGRProj ||
		len(arenaReq.FilmGrain.LumaGrain) != scratch.FilmGrain.LumaGrain {
		t.Fatalf("arena request=%+v", arenaReq)
	}
	shortFlatScratch := flatScratch
	shortFlatScratch.Uint16Scratch = shortFlatScratch.Uint16Scratch[:arenaSize.Uint16Scratch-1]
	if _, err := bindDecoderFrameWorkPostFilterRequestFromScratch(scratch, side, shortFlatScratch); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short arena bind err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	negativeScratch := scratch
	negativeScratch.LoopFilter.Edges = -1
	if _, err := bindDecoderFrameWorkPostFilterRequestFromScratch(negativeScratch, side, flatScratch); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("negative arena bind err=%v want %v", err, internalframe.ErrShortBuffer)
	}

	shortBuffers := buffers
	shortBuffers.SuperResOutputFrame = shortBuffers.SuperResOutputFrame[:scratch.SuperRes.OutputFrame-1]
	if _, err := bindDecoderFrameWorkPostFilterRequest(scratch, shortBuffers); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short aggregate bind err=%v want %v", err, internalframe.ErrShortBuffer)
	}
}

func publicDecoderPostFilterSequence() SequenceHeader {
	return SequenceHeader{
		EnableSuperRes: true,
		EnableCDEF:     true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	}
}

func publicDecoderPostFilterFramePool(t testing.TB, format FrameFormat, count int) internalframe.Pool {
	t.Helper()
	_, backingSize, err := framePoolRequiredSize(format, count)
	if err != nil {
		t.Fatal(err)
	}
	frames := make([]Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := internalframe.BindPool(make([]byte, backingSize), format, frames, free, used)
	if err != nil {
		t.Fatal(err)
	}
	return pool
}

func publicDecoderPostFilterFrame(t testing.TB, format FrameFormat) *Frame {
	t.Helper()
	layout, err := internalframe.RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	frame, err := internalframe.Bind(make([]byte, layout.Size), format)
	if err != nil {
		t.Fatal(err)
	}
	return &frame
}

func publicDecoderPostFilterRequestScratch(size decoderFrameWorkPostFilterRequestScratchSize) decoderFrameWorkPostFilterRequestScratch {
	return decoderFrameWorkPostFilterRequestScratch{
		LoopFilterEdges:    make([]internaldecoder.FrameWorkLoopFilterPostFilterEdge, size.LoopFilterEdges),
		LoopFilterSchedule: make([]uint32, size.LoopFilterSchedule),
		CDEFDirectionGrid:  make([]internalcdef.DirectionGrid, size.CDEFDirectionGrid),
		CDEFVarianceGrid:   make([]internalcdef.VarianceGrid, size.CDEFVarianceGrid),
		ByteScratch:        make([]byte, size.ByteScratch),
		Uint16Scratch:      make([]uint16, size.Uint16Scratch),
		Int16Scratch:       make([]int16, size.Int16Scratch),
		Int32Scratch:       make([]int32, size.Int32Scratch),
	}
}

func publicDecoderLoopFilterRecordAt(col0 int, row0 int, col1 int, row1 int) internalthreading.FrameWorkLoopFilterBlockRecord {
	return internalthreading.FrameWorkLoopFilterBlockRecord{
		Valid: true,
		Block: internalthreading.FrameWorkLoopFilterBlock{
			MICol:     uint16(col0),
			MIRow:     uint16(row0),
			MIColEnd:  uint16(col1),
			MIRowEnd:  uint16(row1),
			X4:        uint8(col0),
			Y4:        uint8(row0),
			Size:      internaltile.BlockSize16x16,
			VisibleW4: uint8(col1 - col0),
			VisibleH4: uint8(row1 - row0),
		},
		TransformTree: internaltile.TransformTreeResult{
			Y:     internaltile.TransformSize16x16,
			UV:    internaltile.TransformSize8x8,
			HasUV: true,
		},
	}
}

func publicFillDecoderLoopFilterMap(filterMap internaldecoder.FrameWorkLoopFilterMap, records ...internalthreading.FrameWorkLoopFilterBlockRecord) {
	for _, record := range records {
		for row := record.Block.MIRow; row < record.Block.MIRowEnd; row++ {
			base := int(row) * int(filterMap.Stride)
			for col := record.Block.MICol; col < record.Block.MIColEnd; col++ {
				filterMap.Records[base+int(col)] = record
			}
		}
	}
}

func publicFillDecoderPostFilterPlane(plane FramePlane) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			plane.Pix[y*plane.Stride+x] = byte((x*31 + y*47 + (x^y)*13) & 0xff)
		}
	}
}
