package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicDecoderLoopFilterMapAndPostFilterRequestBinding(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	size := av1.FrameSize{CodedWidth: 16, UpscaledWidth: 16, Height: 16, SuperResDenominator: 8}
	cols, rows, length, err := av1.DecoderFrameWorkLoopFilterMapShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	if cols != 4 || rows != 4 || length != 16 {
		t.Fatalf("shape=%d,%d,%d want 4,4,16", cols, rows, length)
	}

	records := make([]av1.DecoderFrameWorkLoopFilterBlockRecord, length+1)
	records[0].Valid = true
	filterMap, err := av1.BindDecoderFrameWorkLoopFilterMap(sequence, size, records)
	if err != nil {
		t.Fatal(err)
	}
	if filterMap.Stride != cols || filterMap.Rows != rows || len(filterMap.Records) != length {
		t.Fatalf("map=%+v", filterMap)
	}
	if filterMap.Records[0].Valid {
		t.Fatal("loop-filter map bind did not reset caller-owned records")
	}
	if _, err := av1.BindDecoderFrameWorkLoopFilterMap(sequence, size, records[:length-1]); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("short loop-filter map err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}

	record := publicDecoderLoopFilterRecordAt(0, 0, cols, rows)
	record.DeltaLF = [av1.TileFrameLoopFilterCount]int8{-2, 3, 4, -1}
	publicFillDecoderLoopFilterMap(filterMap, record)
	ctx := av1.DecoderFrameWorkPostFilterContext{
		Event: av1.DecoderEvent{
			SequenceHeader: sequence,
			FrameSize:      size,
			Delta: av1.DeltaParams{
				DeltaLFPresent: true,
				DeltaLFMulti:   true,
			},
			LoopFilter: av1.LoopFilterParams{
				LevelY: [2]uint8{16, 20},
				LevelU: 8,
				LevelV: 4,
			},
		},
		LoopFilterMap: &filterMap,
	}

	scratch, err := ctx.LoopFilterPostFilterScratchLen(av1.DecoderFrameWorkLoopFilterPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req, err := av1.BindDecoderFrameWorkLoopFilterPostFilterRequest(scratch, filterMap, make([]av1.DecoderFrameWorkLoopFilterPostFilterEdge, scratch.Edges))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := ctx.LoopFilterPostFilterPlan(req)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.MICols != cols || plan.MIRows != rows || plan.Cells != length || plan.Blocks != 1 || plan.LumaTXBs != 1 {
		t.Fatalf("plan=%+v", plan)
	}
	if got := plan.Levels[av1.LoopFilterPlaneY][av1.LoopFilterEdgeVertical]; got != (av1.DecoderFrameWorkLoopFilterPostFilterLevelStats{Blocks: 1, NonZero: 1, MaxLevel: 14}) {
		t.Fatalf("Y vertical=%+v", got)
	}
	if got := plan.Levels[av1.LoopFilterPlaneU][av1.LoopFilterEdgeHorizontal]; got.MaxLevel != 12 {
		t.Fatalf("U horizontal=%+v", got)
	}

	manualSize := av1.DecoderFrameWorkLoopFilterPostFilterScratchSize{Edges: 3}
	manualReq, err := av1.BindDecoderFrameWorkLoopFilterPostFilterRequest(manualSize, filterMap, make([]av1.DecoderFrameWorkLoopFilterPostFilterEdge, 4))
	if err != nil {
		t.Fatal(err)
	}
	if len(manualReq.Edges) != 3 {
		t.Fatalf("manual edges=%d want 3", len(manualReq.Edges))
	}
	if _, err := av1.BindDecoderFrameWorkLoopFilterPostFilterRequest(manualSize, filterMap, make([]av1.DecoderFrameWorkLoopFilterPostFilterEdge, 2)); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short loop-filter request err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
}

func TestPublicDecoderSideMapsMarkAndReset(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	size := av1.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	cdef := av1.CDEFParams{Bits: 2, StrengthCount: 4}
	_, _, cdefLength, err := av1.DecoderFrameWorkCDEFIndexMapShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := av1.BindDecoderFrameWorkCDEFIndexMap(sequence, size, cdef, make([]uint8, cdefLength), make([]bool, cdefLength))
	if err != nil {
		t.Fatal(err)
	}
	cdefMap.Index[0], cdefMap.Read[0] = 3, true
	if err := av1.ResetDecoderFrameWorkCDEFIndexMap(cdefMap); err != nil {
		t.Fatal(err)
	}
	if cdefMap.Index[0] != 0 || cdefMap.Read[0] {
		t.Fatalf("cdef map reset index=%d read=%v", cdefMap.Index[0], cdefMap.Read[0])
	}
	visit := publicDecoderPredictionIntraVisit(av1.TileIntraModeDC)
	visit.Prefix.CDEFIndex = 2
	if err := av1.MarkDecoderFrameWorkCDEFIndexMapBlock(cdefMap, cdef, visit); err != nil {
		t.Fatal(err)
	}
	if cdefMap.Index[0] != 2 || !cdefMap.Read[0] {
		t.Fatalf("cdef map mark index=%d read=%v want 2,true", cdefMap.Index[0], cdefMap.Read[0])
	}

	_, _, loopLength, err := av1.DecoderFrameWorkLoopFilterMapShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	loopMap, err := av1.BindDecoderFrameWorkLoopFilterMap(sequence, size, make([]av1.DecoderFrameWorkLoopFilterBlockRecord, loopLength))
	if err != nil {
		t.Fatal(err)
	}
	loopMap.Records[0].Valid = true
	if err := av1.ResetDecoderFrameWorkLoopFilterMap(loopMap); err != nil {
		t.Fatal(err)
	}
	if loopMap.Records[0].Valid {
		t.Fatal("loop-filter map reset left record valid")
	}
	state := &av1.TileDecodeState{
		DeltaLFFromBase: -2,
		DeltaLF:         [av1.TileFrameLoopFilterCount]int8{1, 2, 3, 4},
	}
	if err := av1.MarkDecoderFrameWorkLoopFilterMapBlock(loopMap, visit, state); err != nil {
		t.Fatal(err)
	}
	record := loopMap.Records[4*loopMap.Stride+4]
	if !record.Valid || record.Block.MICol != 4 || record.Block.MIRow != 4 ||
		record.DeltaLFFromBase != -2 || record.DeltaLF != state.DeltaLF {
		t.Fatalf("loop-filter record=%+v state=%+v", record, state)
	}
}

func TestPublicDecoderRestorationFrameBuffersBinding(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	size := av1.FrameSize{CodedWidth: 280, UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}
	restoration := av1.RestorationParams{
		Type:       [3]av1.RestorationType{av1.RestorationWiener, av1.RestorationSGRProj, av1.RestorationNone},
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}
	plan, err := av1.DecoderFrameWorkRestorationFramePlan(sequence, size, restoration)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.Planes != 3 || plan.UnitRecordLen() == 0 || plan.BoundaryBufferLen() == 0 {
		t.Fatalf("plan=%+v", plan)
	}

	recordBacking := make([]av1.TileRestorationUnitRecord, plan.UnitRecordLen())
	above := make([]uint16, plan.BoundaryBufferLen())
	below := make([]uint16, plan.BoundaryBufferLen())
	buffers, err := av1.BindDecoderFrameWorkRestorationFrameBuffers(sequence, size, restoration, recordBacking, above, below)
	if err != nil {
		t.Fatal(err)
	}
	if buffers.Plan != plan {
		t.Fatalf("buffer plan=%+v want %+v", buffers.Plan, plan)
	}
	for plane := 0; plane < int(plan.Planes); plane++ {
		if len(buffers.Records[plane]) != plan.UnitRecords[plane] {
			t.Fatalf("plane %d records=%d want %d", plane, len(buffers.Records[plane]), plan.UnitRecords[plane])
		}
		if len(buffers.Boundaries[plane].Above) != plan.Boundaries[plane].Len ||
			len(buffers.Boundaries[plane].Below) != plan.Boundaries[plane].Len ||
			buffers.Boundaries[plane].Stride != plan.Boundaries[plane].Stride {
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
	if first.Index != 0 || first.Unit.Type != av1.RestorationNone || first.StripeCount == 0 {
		t.Fatalf("first reset record=%+v", first)
	}
	if _, err := av1.BindDecoderFrameWorkRestorationFrameBuffers(sequence, size, restoration, recordBacking[:len(recordBacking)-1], above, below); !errors.Is(err, av1.ErrTileJobBufferTooSmall) {
		t.Fatalf("short restoration records err=%v want %v", err, av1.ErrTileJobBufferTooSmall)
	}

	allNone := av1.RestorationParams{}
	noneBuffers, err := av1.BindDecoderFrameWorkRestorationFrameBuffers(sequence, size, allNone, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if noneBuffers.Plan.Active || noneBuffers.Plan.UnitRecordLen() != 0 || noneBuffers.Plan.BoundaryBufferLen() != 0 {
		t.Fatalf("none buffers=%+v", noneBuffers.Plan)
	}
}

func TestPublicDecoderFrameWorkSideDataBinding(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	size := av1.FrameSize{CodedWidth: 128, UpscaledWidth: 128, Height: 96, SuperResDenominator: 8}
	cdef := av1.CDEFParams{Bits: 1, StrengthCount: 2}
	restoration := av1.RestorationParams{
		Type:       [3]av1.RestorationType{av1.RestorationWiener, av1.RestorationSGRProj, av1.RestorationNone},
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}
	scratchSize, err := av1.DecoderFrameWorkSideDataScratchLen(sequence, size, cdef, restoration)
	if err != nil {
		t.Fatal(err)
	}
	_, _, cdefLen, err := av1.DecoderFrameWorkCDEFIndexMapShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	_, _, loopLen, err := av1.DecoderFrameWorkLoopFilterMapShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	restorationPlan, err := av1.DecoderFrameWorkRestorationFramePlan(sequence, size, restoration)
	if err != nil {
		t.Fatal(err)
	}
	if scratchSize.CDEFIndexMap != cdefLen || scratchSize.CDEFReadMap != cdefLen ||
		scratchSize.LoopFilterMap != loopLen ||
		scratchSize.RestorationRecords != restorationPlan.UnitRecordLen() ||
		scratchSize.RestorationBoundaryAbove != restorationPlan.BoundaryBufferLen() ||
		scratchSize.RestorationBoundaryBelow != restorationPlan.BoundaryBufferLen() {
		t.Fatalf("side scratch=%+v cdef=%d loop=%d restoration=%+v", scratchSize, cdefLen, loopLen, restorationPlan)
	}

	scratch := av1.DecoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, scratchSize.CDEFIndexMap+1),
		CDEFReadMap:              make([]bool, scratchSize.CDEFReadMap+1),
		LoopFilterMap:            make([]av1.DecoderFrameWorkLoopFilterBlockRecord, scratchSize.LoopFilterMap+1),
		RestorationRecords:       make([]av1.TileRestorationUnitRecord, scratchSize.RestorationRecords+1),
		RestorationBoundaryAbove: make([]uint16, scratchSize.RestorationBoundaryAbove+1),
		RestorationBoundaryBelow: make([]uint16, scratchSize.RestorationBoundaryBelow+1),
	}
	scratch.CDEFIndexMap[0] = 1
	scratch.CDEFReadMap[0] = true
	scratch.CDEFIndexMap[scratchSize.CDEFIndexMap] = 7
	scratch.CDEFReadMap[scratchSize.CDEFReadMap] = true
	scratch.LoopFilterMap[0].Valid = true
	scratch.LoopFilterMap[scratchSize.LoopFilterMap].Valid = true
	scratch.RestorationRecords[0].Index = 99
	scratch.RestorationRecords[scratchSize.RestorationRecords].Index = 77
	scratch.RestorationBoundaryAbove[scratchSize.RestorationBoundaryAbove] = 55
	scratch.RestorationBoundaryBelow[scratchSize.RestorationBoundaryBelow] = 66

	side, err := av1.BindDecoderFrameWorkSideData(sequence, size, cdef, restoration, scratch)
	if err != nil {
		t.Fatal(err)
	}
	if len(side.CDEFIndexMap.Index) != scratchSize.CDEFIndexMap ||
		len(side.CDEFIndexMap.Read) != scratchSize.CDEFReadMap ||
		len(side.LoopFilterMap.Records) != scratchSize.LoopFilterMap ||
		side.RestorationFrameBuffers.Plan != restorationPlan {
		t.Fatalf("side data=%+v", side)
	}
	if side.CDEFIndexMap.Index[0] != 0 || side.CDEFIndexMap.Read[0] ||
		side.LoopFilterMap.Records[0].Valid ||
		side.RestorationFrameBuffers.Records[0][0].Index != 0 ||
		side.RestorationFrameBuffers.Records[0][0].Unit.Type != av1.RestorationNone {
		t.Fatalf("side data was not reset: cdef=%d/%v loop=%+v restoration=%+v",
			side.CDEFIndexMap.Index[0], side.CDEFIndexMap.Read[0], side.LoopFilterMap.Records[0], side.RestorationFrameBuffers.Records[0][0])
	}
	if scratch.CDEFIndexMap[scratchSize.CDEFIndexMap] != 7 ||
		!scratch.CDEFReadMap[scratchSize.CDEFReadMap] ||
		!scratch.LoopFilterMap[scratchSize.LoopFilterMap].Valid ||
		scratch.RestorationRecords[scratchSize.RestorationRecords].Index != 77 ||
		scratch.RestorationBoundaryAbove[scratchSize.RestorationBoundaryAbove] != 55 ||
		scratch.RestorationBoundaryBelow[scratchSize.RestorationBoundaryBelow] != 66 {
		t.Fatal("side-data bind touched storage past requested lengths")
	}
	postSide := av1.DecoderFrameWorkPostFilterSideData(side)
	if postSide.CDEFIndexMap.Index == nil ||
		len(postSide.RestorationRecords[0]) != len(side.RestorationFrameBuffers.Records[0]) ||
		len(postSide.RestorationBoundaries[0].Above) != len(side.RestorationFrameBuffers.Boundaries[0].Above) {
		t.Fatalf("postfilter side data=%+v", postSide)
	}
	postBuffers, err := av1.BindDecoderFrameWorkPostFilterRequestBuffersFromSideData(av1.DecoderFrameWorkPostFilterScratchSize{}, side, av1.DecoderFrameWorkPostFilterRequestScratch{})
	if err != nil {
		t.Fatal(err)
	}
	if postBuffers.CDEFIndexMap.Stride != side.CDEFIndexMap.Stride ||
		postBuffers.LoopFilterMap.Stride != side.LoopFilterMap.Stride ||
		len(postBuffers.RestorationRecords[0]) != len(side.RestorationFrameBuffers.Records[0]) {
		t.Fatalf("side-data postfilter buffers=%+v side=%+v", postBuffers, side)
	}
	postReq, err := av1.BindDecoderFrameWorkPostFilterRequestFromSideData(av1.DecoderFrameWorkPostFilterScratchSize{}, side, av1.DecoderFrameWorkPostFilterRequestScratch{})
	if err != nil {
		t.Fatal(err)
	}
	if postReq.CDEF.IndexMap.Rows != side.CDEFIndexMap.Rows ||
		postReq.LoopFilter.Map.Rows != side.LoopFilterMap.Rows ||
		len(postReq.Restoration.Records[0]) != len(side.RestorationFrameBuffers.Records[0]) {
		t.Fatalf("side-data postfilter request=%+v side=%+v", postReq, side)
	}

	shortScratch := scratch
	shortScratch.RestorationBoundaryBelow = shortScratch.RestorationBoundaryBelow[:scratchSize.RestorationBoundaryBelow-1]
	if _, err := av1.BindDecoderFrameWorkSideData(sequence, size, cdef, restoration, shortScratch); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short side-data scratch err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
}

func TestPublicDecoderFrameWorkSupportedPostFilterScratchRunner(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	sequence.ColorConfig.MonoChrome = true
	sequence.ColorConfig.SubsamplingX = false
	sequence.ColorConfig.SubsamplingY = false
	size := av1.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	cdef := av1.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [av1.MaxCDEFStrengths]uint8{63},
	}
	sideScratchSize, err := av1.DecoderFrameWorkSideDataScratchLen(sequence, size, cdef, av1.RestorationParams{})
	if err != nil {
		t.Fatal(err)
	}
	sideScratch := av1.DecoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, sideScratchSize.CDEFIndexMap),
		CDEFReadMap:              make([]bool, sideScratchSize.CDEFReadMap),
		LoopFilterMap:            make([]av1.DecoderFrameWorkLoopFilterBlockRecord, sideScratchSize.LoopFilterMap),
		RestorationRecords:       make([]av1.TileRestorationUnitRecord, sideScratchSize.RestorationRecords),
		RestorationBoundaryAbove: make([]uint16, sideScratchSize.RestorationBoundaryAbove),
		RestorationBoundaryBelow: make([]uint16, sideScratchSize.RestorationBoundaryBelow),
	}
	side, err := av1.BindDecoderFrameWorkSideData(sequence, size, cdef, av1.RestorationParams{}, sideScratch)
	if err != nil {
		t.Fatal(err)
	}
	side.CDEFIndexMap.Read[0] = true

	output := publicDecoderPostFilterFrame(t, av1.FrameFormat{
		Width:      int(size.CodedWidth),
		Height:     int(size.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      32,
	})
	publicFillDecoderPostFilterPlane(output.Y)
	before := append([]byte(nil), output.Y.Pix[:output.Y.Width]...)
	ctx := av1.DecoderFrameWorkPostFilterContext{
		Event: av1.DecoderEvent{
			Kind:           av1.DecoderEventTileGroup,
			SequenceHeader: sequence,
			FrameSize:      size,
			CDEF:           cdef,
			TileGroup:      av1.TileGroup{Final: true},
		},
		Output:                  output,
		CDEFIndexMap:            &side.CDEFIndexMap,
		LoopFilterMap:           &side.LoopFilterMap,
		RestorationFrameBuffers: &side.RestorationFrameBuffers,
	}
	var probe av1.DecoderFrameWorkSupportedPostFilterScratchRunner
	exact, err := probe.ScratchLen(ctx)
	if err != nil {
		t.Fatal(err)
	}
	arenaSize := av1.DecoderFrameWorkPostFilterRequestScratchLen(exact)
	if exact.CDEF.Input == 0 || arenaSize.Uint16Scratch == 0 {
		t.Fatalf("exact scratch=%+v arena=%+v", exact, arenaSize)
	}
	runner := av1.DecoderFrameWorkSupportedPostFilterScratchRunner{
		Scratch: publicDecoderPostFilterRequestScratch(arenaSize),
	}
	if err := runner.Apply(ctx); err != nil {
		t.Fatal(err)
	}
	if !runner.Result.Completed.Has(av1.DecoderFrameWorkPostFilterCDEF) ||
		runner.Context.RemainingPostFilters() != 0 ||
		runner.Size.CDEF.Input != exact.CDEF.Input ||
		runner.Request.CDEF.IndexMap.Rows != side.CDEFIndexMap.Rows {
		t.Fatalf("runner size=%+v request=%+v result=%+v remaining=%b", runner.Size, runner.Request, runner.Result, runner.Context.RemainingPostFilters())
	}
	if string(output.Y.Pix[:output.Y.Width]) == string(before) {
		t.Fatal("runner did not apply CDEF")
	}

	fromCtx := av1.DecoderFrameWorkPostFilterRequestSideDataFromContext(ctx)
	if fromCtx.CDEFIndexMap.Rows != side.CDEFIndexMap.Rows ||
		fromCtx.LoopFilterMap.Rows != side.LoopFilterMap.Rows ||
		len(fromCtx.RestorationRecords[0]) != len(side.RestorationFrameBuffers.Records[0]) {
		t.Fatalf("side from context=%+v side=%+v", fromCtx, side)
	}
	var nilRunner *av1.DecoderFrameWorkSupportedPostFilterScratchRunner
	if err := nilRunner.Apply(ctx); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil runner err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
	shortRunner := runner
	shortRunner.Scratch.Uint16Scratch = shortRunner.Scratch.Uint16Scratch[:arenaSize.Uint16Scratch-1]
	if err := shortRunner.Apply(ctx); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short scratch err=%v want %v", err, av1.ErrFrameShortBuffer)
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

func TestPublicDecoderFrameWorkPostFilterScratchSizeMax(t *testing.T) {
	arenaA := av1.DecoderFrameWorkPostFilterRequestScratchSize{
		LoopFilterEdges:   1,
		CDEFDirectionGrid: 7,
		CDEFVarianceGrid:  3,
		ByteScratch:       11,
		Uint16Scratch:     5,
		Int16Scratch:      13,
		Int32Scratch:      2,
	}
	arenaB := av1.DecoderFrameWorkPostFilterRequestScratchSize{
		LoopFilterEdges:   4,
		CDEFDirectionGrid: 2,
		CDEFVarianceGrid:  8,
		ByteScratch:       6,
		Uint16Scratch:     20,
		Int16Scratch:      1,
		Int32Scratch:      9,
	}
	if got, want := arenaA.Max(arenaB), (av1.DecoderFrameWorkPostFilterRequestScratchSize{
		LoopFilterEdges:   4,
		CDEFDirectionGrid: 7,
		CDEFVarianceGrid:  8,
		ByteScratch:       11,
		Uint16Scratch:     20,
		Int16Scratch:      13,
		Int32Scratch:      9,
	}); got != want {
		t.Fatalf("postfilter scratch max=%+v want %+v", got, want)
	}

	sideA := av1.DecoderFrameWorkSideDataScratchSize{
		CDEFIndexMap:             1,
		CDEFReadMap:              5,
		LoopFilterMap:            2,
		RestorationRecords:       9,
		RestorationBoundaryAbove: 3,
		RestorationBoundaryBelow: 4,
	}
	sideB := av1.DecoderFrameWorkSideDataScratchSize{
		CDEFIndexMap:             8,
		CDEFReadMap:              2,
		LoopFilterMap:            6,
		RestorationRecords:       1,
		RestorationBoundaryAbove: 7,
		RestorationBoundaryBelow: 3,
	}
	if got, want := sideA.Max(sideB), (av1.DecoderFrameWorkSideDataScratchSize{
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

func TestPublicDecoderFrameWorkCallerPostFilterScratchRunner(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	sequence.ColorConfig.MonoChrome = true
	sequence.ColorConfig.SubsamplingX = false
	sequence.ColorConfig.SubsamplingY = false
	size := av1.FrameSize{
		CodedWidth:          8,
		UpscaledWidth:       13,
		Height:              32,
		SuperResEnabled:     true,
		SuperResDenominator: 13,
	}
	event := av1.DecoderEvent{
		Kind:           av1.DecoderEventTileGroup,
		SequenceHeader: sequence,
		FrameSize:      size,
		TileGroup:      av1.TileGroup{Final: true},
		FilmGrain: av1.FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			Seed:          0x1234,
			BitDepth:      8,
			NumYPoints:    1,
			YPoints:       [av1.MaxFilmGrainYPoints][2]uint8{{0, 64}},
			ScalingShift:  8,
			Overlap:       true,
		},
	}
	var scratchOutput av1.Frame
	scratchCtx, err := av1.DecoderFrameWorkPostFilterScratchContext(sequence, event, 32, nil, &scratchOutput)
	if err != nil {
		t.Fatal(err)
	}
	if scratchCtx.Output != &scratchOutput ||
		scratchOutput.Format.Width != int(size.CodedWidth) ||
		scratchOutput.Layout.Size == 0 ||
		len(scratchOutput.Y.Pix) != 0 {
		t.Fatalf("scratch ctx=%+v output=%+v", scratchCtx, scratchOutput)
	}
	if _, err := av1.DecoderFrameWorkPostFilterScratchContext(sequence, event, 32, nil, nil); !errors.Is(err, av1.ErrFrameInvalidSlot) {
		t.Fatalf("nil scratch output err=%v want %v", err, av1.ErrFrameInvalidSlot)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, err = av1.DecoderFrameWorkPostFilterScratchContext(sequence, event, 32, nil, &scratchOutput)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("scratch context allocated: %f", allocs)
	}

	format := scratchOutput.Format
	output := publicDecoderPostFilterFrame(t, format)
	for i := range output.Y.Pix {
		output.Y.Pix[i] = 100
	}
	ctx := av1.DecoderFrameWorkPostFilterContext{Event: event, Output: output}

	var runner av1.DecoderFrameWorkCallerPostFilterScratchRunner
	first, err := runner.ScratchLen(scratchCtx)
	if err != nil {
		t.Fatal(err)
	}
	if first.SuperRes.OutputFrame == 0 || first.FilmGrain != (av1.DecoderFrameWorkFilmGrainPostFilterScratchSize{}) {
		t.Fatalf("first scratch=%+v", first)
	}
	runner.Scratch = publicDecoderPostFilterRequestScratch(av1.DecoderFrameWorkPostFilterRequestScratchLen(first))
	full, err := runner.ScratchLen(scratchCtx)
	if err != nil {
		t.Fatal(err)
	}
	if full.SuperRes.OutputFrame == 0 || full.FilmGrain.LumaGrain == 0 || full.FilmGrain.LumaSamples == 0 {
		t.Fatalf("full scratch=%+v", full)
	}
	runner.Scratch = publicDecoderPostFilterRequestScratch(av1.DecoderFrameWorkPostFilterRequestScratchLen(full))
	if err := runner.Apply(ctx); err != nil {
		t.Fatal(err)
	}
	wantCompleted := av1.DecoderFrameWorkPostFilterSuperRes | av1.DecoderFrameWorkPostFilterFilmGrain
	if runner.Result.Completed != wantCompleted ||
		runner.Result.SuperRes.Output.Format.Width != int(size.UpscaledWidth) ||
		runner.Context.RemainingPostFilters() != 0 ||
		!runner.Context.DetachedPostFilterOutput() ||
		runner.Size.FilmGrain.LumaGrain != full.FilmGrain.LumaGrain {
		t.Fatalf("runner size=%+v result=%+v remaining=%b detached=%v", runner.Size, runner.Result, runner.Context.RemainingPostFilters(), runner.Context.DetachedPostFilterOutput())
	}
	var nilRunner *av1.DecoderFrameWorkCallerPostFilterScratchRunner
	if err := nilRunner.Apply(ctx); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil caller runner err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
	shortRunner := runner
	shortRunner.Scratch.ByteScratch = shortRunner.Scratch.ByteScratch[:full.SuperRes.OutputFrame-1]
	if err := shortRunner.Apply(ctx); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short caller scratch err=%v want %v", err, av1.ErrFrameShortBuffer)
	}

	allocs = testing.AllocsPerRun(1000, func() {
		err = runner.Apply(ctx)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("caller scratch runner allocated: %f", allocs)
	}
}

func TestPublicDecoderFrameWorkSideDataSet(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	size := av1.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	cdef := av1.CDEFParams{Bits: 1, StrengthCount: 2}
	restoration := av1.RestorationParams{
		Type:       [3]av1.RestorationType{av1.RestorationWiener, av1.RestorationSGRProj, av1.RestorationNone},
		UnitSizeY:  64,
		UnitSizeUV: 64,
	}
	scratchSize, err := av1.DecoderFrameWorkSideDataScratchLen(sequence, size, cdef, restoration)
	if err != nil {
		t.Fatal(err)
	}
	sideScratch := av1.DecoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, scratchSize.CDEFIndexMap),
		CDEFReadMap:              make([]bool, scratchSize.CDEFReadMap),
		LoopFilterMap:            make([]av1.DecoderFrameWorkLoopFilterBlockRecord, scratchSize.LoopFilterMap),
		RestorationRecords:       make([]av1.TileRestorationUnitRecord, scratchSize.RestorationRecords),
		RestorationBoundaryAbove: make([]uint16, scratchSize.RestorationBoundaryAbove),
		RestorationBoundaryBelow: make([]uint16, scratchSize.RestorationBoundaryBelow),
	}
	side, err := av1.BindDecoderFrameWorkSideData(sequence, size, cdef, restoration, sideScratch)
	if err != nil {
		t.Fatal(err)
	}

	var nilState *av1.DecoderFrameWorkState
	if err := av1.SetDecoderFrameWorkSideData(nilState, side); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("nil state err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}
	var inactive av1.DecoderFrameWorkState
	if err := av1.SetDecoderFrameWorkSideData(&inactive, side); !errors.Is(err, av1.ErrDecoderInvalidFrameWorkState) {
		t.Fatalf("inactive state err=%v want %v", err, av1.ErrDecoderInvalidFrameWorkState)
	}

	pool := publicDecoderPostFilterFramePool(t, av1.FrameFormat{
		Width:        int(size.CodedWidth),
		Height:       int(size.Height),
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	}, 1)
	begin := av1.DecoderEvent{
		Kind:        av1.DecoderEventFrameHeader,
		FrameHeader: av1.FrameHeaderPrefix{FrameType: av1.FrameTypeKey},
		FrameSize:   size,
	}
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	plan, output, err := state.Begin(&refs, &pool, sequence, begin, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	final := begin
	final.Kind = av1.DecoderEventTileGroup
	final.TileGroup.Final = true
	final.FrameSize.RefreshFrameFlags = 0xff
	step := av1.DecoderFrameWorkStep{
		Kind: av1.DecoderFrameWorkStepTile,
		Tile: av1.DecoderFrameTileWorkPlan{Surface: plan.Surface},
	}

	badSide := side
	badSide.LoopFilterMap = av1.DecoderFrameWorkLoopFilterMap{Stride: 1, Rows: 1}
	if err := av1.SetDecoderFrameWorkSideData(&state, badSide); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("invalid loop-filter map err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	ctx, err := state.PostFilterContext(&pool, final, step, false)
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
		attachErr = av1.SetDecoderFrameWorkSideData(&state, side)
	})
	if attachErr != nil {
		t.Fatal(attachErr)
	}
	if allocs != 0 {
		t.Fatalf("SetDecoderFrameWorkSideData allocated: %f", allocs)
	}
	ctx, err = state.PostFilterContext(&pool, final, step, true)
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
		ctx.RestorationFrameBuffers.Records[0][0].Unit.Type != av1.RestorationNone {
		t.Fatalf("attached side data was not reset: cdef=%d/%v loop=%+v restoration=%+v",
			ctx.CDEFIndexMap.Index[0], ctx.CDEFIndexMap.Read[0], ctx.LoopFilterMap.Records[0], ctx.RestorationFrameBuffers.Records[0][0])
	}
	ctx.CDEFIndexMap.Read[0] = true
	ctx.LoopFilterMap.Records[0].Valid = true
	ctx.RestorationFrameBuffers.Records[0][0].Unit.Type = av1.RestorationWiener
	if !side.CDEFIndexMap.Read[0] ||
		!side.LoopFilterMap.Records[0].Valid ||
		side.RestorationFrameBuffers.Records[0][0].Unit.Type != av1.RestorationWiener {
		t.Fatal("attached side data did not preserve caller-owned backing")
	}
}

func TestPublicDecoderCDEFIndexMapAndPostFilterRequestBinding(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	size := av1.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	cols, rows, length, err := av1.DecoderFrameWorkCDEFIndexMapShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	if cols != 1 || rows != 1 || length != 1 {
		t.Fatalf("shape=%d,%d,%d want 1,1,1", cols, rows, length)
	}
	index := make([]uint8, length)
	read := make([]bool, length)
	index[0], read[0] = 3, true
	cdefMap, err := av1.BindDecoderFrameWorkCDEFIndexMap(sequence, size, av1.CDEFParams{Bits: 2, StrengthCount: 4}, index, read)
	if err != nil {
		t.Fatal(err)
	}
	if cdefMap.Stride != cols || cdefMap.Rows != rows || len(cdefMap.Index) != length || len(cdefMap.Read) != length {
		t.Fatalf("map=%+v", cdefMap)
	}
	badIndex := []uint8{4}
	if _, err := av1.BindDecoderFrameWorkCDEFIndexMap(sequence, size, av1.CDEFParams{Bits: 2, StrengthCount: 4}, badIndex, read); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("bad index err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}

	output := publicDecoderPostFilterFrame(t, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	publicFillDecoderPostFilterPlane(output.Y)
	before := append([]byte(nil), output.Y.Pix[:output.Y.Width]...)
	ctx := av1.DecoderFrameWorkPostFilterContext{
		Event: av1.DecoderEvent{
			SequenceHeader: sequence,
			FrameSize:      size,
			CDEF: av1.CDEFParams{
				Bits:          2,
				Damping:       5,
				StrengthCount: 4,
				YStrength:     [av1.MaxCDEFStrengths]uint8{0, 0, 0, 63},
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
	for plane := 0; plane < 3; plane++ {
		sampleScratch[plane] = make([]uint16, scratch.Samples[plane])
		dstScratch[plane] = make([]uint16, scratch.Dst[plane])
	}
	req, err := av1.BindDecoderFrameWorkCDEFPostFilterRequest(
		scratch,
		cdefMap,
		sampleScratch,
		dstScratch,
		make([]av1.CDEFDirectionGrid, scratch.DirectionGrid),
		make([]av1.CDEFVarianceGrid, scratch.VarianceGrid),
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
	if _, err := av1.BindDecoderFrameWorkCDEFPostFilterRequest(scratch, cdefMap, sampleScratch, dstScratch, nil, nil, nil, nil); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short CDEF bind err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
}

func TestPublicDecoderSuperResPostFilterRequestBinding(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	sequence.EnableSuperRes = true
	size := av1.FrameSize{CodedWidth: 64, UpscaledWidth: 96, Height: 32, SuperResEnabled: true, SuperResDenominator: 12}
	format, err := av1.FrameCodedFormatFromHeaders(sequence, size, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := publicDecoderPostFilterFrame(t, format)
	publicFillDecoderPostFilterPlane(output.Y)
	ctx := av1.DecoderFrameWorkPostFilterContext{
		Event: av1.DecoderEvent{
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
	for plane := 0; plane < 3; plane++ {
		codedScratch[plane] = make([]uint16, scratch.CodedSamples[plane])
		outputScratch[plane] = make([]uint16, scratch.OutputSamples[plane])
	}
	req, err := av1.BindDecoderFrameWorkSuperResPostFilterRequest(scratch, make([]byte, scratch.OutputFrame), codedScratch, outputScratch)
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
	if _, err := av1.BindDecoderFrameWorkSuperResPostFilterRequest(scratch, nil, codedScratch, outputScratch); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short superres bind err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
}

func TestPublicDecoderRestorationAndFilmGrainRequestBinding(t *testing.T) {
	restorationSize := av1.DecoderFrameWorkRestorationPostFilterScratchSize{}
	restorationSize.Samples.DataLen = 11
	restorationSize.Samples.DstLen = 13
	restorationSize.Apply.Unit.Wiener = 17
	restorationSize.Apply.Unit.SGRProj = 19
	restorationSize.Apply.Boundary.Above = 23
	restorationSize.Apply.Boundary.Below = 29

	restorationReq, err := av1.BindDecoderFrameWorkRestorationPostFilterRequest(
		restorationSize,
		[3][]av1.TileRestorationUnitRecord{{}},
		[3]av1.TileRestorationStripeBoundaries{},
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
	if _, err := av1.BindDecoderFrameWorkRestorationPostFilterRequest(restorationSize, [3][]av1.TileRestorationUnitRecord{}, [3]av1.TileRestorationStripeBoundaries{}, nil, nil, nil, nil, nil, nil, false); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short restoration bind err=%v want %v", err, av1.ErrFrameShortBuffer)
	}

	filmSize := av1.DecoderFrameWorkFilmGrainPostFilterScratchSize{
		LumaGrain:     31,
		ChromaGrain:   [2]int{37, 41},
		LumaSamples:   43,
		ChromaSamples: [2]int{47, 53},
	}
	filmReq, err := av1.BindDecoderFrameWorkFilmGrainPostFilterRequest(
		filmSize,
		make([]int16, 32),
		[2][]int16{make([]int16, 38), make([]int16, 42)},
		make([]uint16, 44),
		[2][]uint16{make([]uint16, 48), make([]uint16, 54)},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(filmReq.LumaGrain) != 31 ||
		len(filmReq.ChromaGrain[0]) != 37 ||
		len(filmReq.ChromaGrain[1]) != 41 ||
		len(filmReq.LumaSamples) != 43 ||
		len(filmReq.ChromaSamples[0]) != 47 ||
		len(filmReq.ChromaSamples[1]) != 53 {
		t.Fatalf("film request=%+v", filmReq)
	}
	if _, err := av1.BindDecoderFrameWorkFilmGrainPostFilterRequest(filmSize, nil, [2][]int16{}, nil, [2][]uint16{}); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short filmgrain bind err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
}

func TestPublicDecoderPostFilterRequestBinding(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	size := av1.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	_, _, loopFilterLength, err := av1.DecoderFrameWorkLoopFilterMapShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	loopFilterMap, err := av1.BindDecoderFrameWorkLoopFilterMap(sequence, size, make([]av1.DecoderFrameWorkLoopFilterBlockRecord, loopFilterLength))
	if err != nil {
		t.Fatal(err)
	}
	_, _, cdefLength, err := av1.DecoderFrameWorkCDEFIndexMapShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := av1.BindDecoderFrameWorkCDEFIndexMap(sequence, size, av1.CDEFParams{Bits: 1, StrengthCount: 2}, make([]uint8, cdefLength), make([]bool, cdefLength))
	if err != nil {
		t.Fatal(err)
	}

	scratch := av1.DecoderFrameWorkPostFilterScratchSize{
		LoopFilter: av1.DecoderFrameWorkLoopFilterPostFilterScratchSize{Edges: 3},
		CDEF: av1.DecoderFrameWorkCDEFPostFilterScratchSize{
			Samples:       [3]int{64, 16, 16},
			Dst:           [3]int{64, 16, 16},
			DirectionGrid: 1,
			VarianceGrid:  1,
			Input:         av1.CDEFInputBufferSize,
			UnitDst:       av1.CDEFInputBufferSize,
		},
		SuperRes: av1.DecoderFrameWorkSuperResPostFilterScratchSize{
			OutputFrame:   128,
			CodedSamples:  [3]int{64, 16, 16},
			OutputSamples: [3]int{96, 24, 24},
		},
		Restoration: av1.DecoderFrameWorkRestorationPostFilterScratchSize{
			Samples: av1.TileRestorationFrameSampleScratchSize{DataLen: 5, DstLen: 7},
			Apply: av1.TileRestorationUnitRecordBoundaryScratchSize{
				Unit:     av1.TileRestorationUnitScratchSize{Wiener: 11, SGRProj: 13},
				Boundary: av1.TileRestorationStripeBoundaryScratchSize{Above: 17, Below: 19},
			},
		},
		FilmGrain: av1.DecoderFrameWorkFilmGrainPostFilterScratchSize{
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
	for plane := 0; plane < 3; plane++ {
		sampleScratch[plane] = make([]uint16, scratch.CDEF.Samples[plane]+1)
		dstScratch[plane] = make([]uint16, scratch.CDEF.Dst[plane]+1)
		codedScratch[plane] = make([]uint16, scratch.SuperRes.CodedSamples[plane]+1)
		outputScratch[plane] = make([]uint16, scratch.SuperRes.OutputSamples[plane]+1)
	}
	buffers := av1.DecoderFrameWorkPostFilterRequestBuffers{
		LoopFilterMap:   loopFilterMap,
		LoopFilterEdges: make([]av1.DecoderFrameWorkLoopFilterPostFilterEdge, scratch.LoopFilter.Edges+1),

		CDEFIndexMap:       cdefMap,
		CDEFSampleScratch:  sampleScratch,
		CDEFDstScratch:     dstScratch,
		CDEFDirectionGrid:  make([]av1.CDEFDirectionGrid, scratch.CDEF.DirectionGrid+1),
		CDEFVarianceGrid:   make([]av1.CDEFVarianceGrid, scratch.CDEF.VarianceGrid+1),
		CDEFInputScratch:   make([]uint16, scratch.CDEF.Input+1),
		CDEFUnitDstScratch: make([]uint16, scratch.CDEF.UnitDst+1),

		SuperResOutputFrame:   make([]byte, scratch.SuperRes.OutputFrame+1),
		SuperResCodedScratch:  codedScratch,
		SuperResOutputScratch: outputScratch,

		RestorationRecords:              [3][]av1.TileRestorationUnitRecord{{{Index: 7}}},
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

	req, err := av1.BindDecoderFrameWorkPostFilterRequest(scratch, buffers)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.LoopFilter.Edges) != scratch.LoopFilter.Edges ||
		len(req.CDEF.SampleScratch[0]) != scratch.CDEF.Samples[0] ||
		len(req.CDEF.InputScratch) != scratch.CDEF.Input ||
		len(req.SuperRes.OutputFrame) != scratch.SuperRes.OutputFrame ||
		len(req.SuperRes.OutputScratch[1]) != scratch.SuperRes.OutputSamples[1] ||
		len(req.Restoration.DataScratch) != scratch.Restoration.Samples.DataLen ||
		len(req.Restoration.Scratch.Boundary.Below) != scratch.Restoration.Apply.Boundary.Below ||
		!req.Restoration.Optimized ||
		len(req.FilmGrain.ChromaSamples[1]) != scratch.FilmGrain.ChromaSamples[1] {
		t.Fatalf("request=%+v", req)
	}
	if req.LoopFilter.Map.Stride != loopFilterMap.Stride || req.CDEF.IndexMap.Rows != cdefMap.Rows {
		t.Fatalf("side maps loop=%+v cdef=%+v", req.LoopFilter.Map, req.CDEF.IndexMap)
	}

	if zeroReq, err := av1.BindDecoderFrameWorkPostFilterRequest(av1.DecoderFrameWorkPostFilterScratchSize{}, av1.DecoderFrameWorkPostFilterRequestBuffers{}); err != nil || len(zeroReq.SuperRes.OutputFrame) != 0 {
		t.Fatalf("zero request=%+v err=%v", zeroReq, err)
	}

	arenaSize := av1.DecoderFrameWorkPostFilterRequestScratchLen(scratch)
	wantUint16Scratch := scratch.CDEF.Input + scratch.CDEF.UnitDst +
		scratch.Restoration.Samples.DataLen + scratch.Restoration.Samples.DstLen +
		scratch.Restoration.Apply.Unit.Wiener +
		scratch.Restoration.Apply.Boundary.Above + scratch.Restoration.Apply.Boundary.Below +
		scratch.FilmGrain.LumaSamples
	for plane := 0; plane < 3; plane++ {
		wantUint16Scratch += scratch.CDEF.Samples[plane] + scratch.CDEF.Dst[plane]
		wantUint16Scratch += scratch.SuperRes.CodedSamples[plane] + scratch.SuperRes.OutputSamples[plane]
		if plane < 2 {
			wantUint16Scratch += scratch.FilmGrain.ChromaSamples[plane]
		}
	}
	wantInt16Scratch := scratch.FilmGrain.LumaGrain + scratch.FilmGrain.ChromaGrain[0] + scratch.FilmGrain.ChromaGrain[1]
	if arenaSize.LoopFilterEdges != scratch.LoopFilter.Edges ||
		arenaSize.CDEFDirectionGrid != scratch.CDEF.DirectionGrid ||
		arenaSize.CDEFVarianceGrid != scratch.CDEF.VarianceGrid ||
		arenaSize.ByteScratch != scratch.SuperRes.OutputFrame ||
		arenaSize.Uint16Scratch != wantUint16Scratch ||
		arenaSize.Int16Scratch != wantInt16Scratch ||
		arenaSize.Int32Scratch != scratch.Restoration.Apply.Unit.SGRProj {
		t.Fatalf("arena size=%+v want uint16=%d int16=%d", arenaSize, wantUint16Scratch, wantInt16Scratch)
	}
	side := av1.DecoderFrameWorkPostFilterRequestSideData{
		LoopFilterMap:         loopFilterMap,
		CDEFIndexMap:          cdefMap,
		RestorationRecords:    buffers.RestorationRecords,
		RestorationBoundaries: buffers.RestorationBoundaries,
		RestorationOptimized:  buffers.RestorationOptimized,
	}
	flatScratch := av1.DecoderFrameWorkPostFilterRequestScratch{
		LoopFilterEdges:   make([]av1.DecoderFrameWorkLoopFilterPostFilterEdge, arenaSize.LoopFilterEdges+1),
		CDEFDirectionGrid: make([]av1.CDEFDirectionGrid, arenaSize.CDEFDirectionGrid+1),
		CDEFVarianceGrid:  make([]av1.CDEFVarianceGrid, arenaSize.CDEFVarianceGrid+1),
		ByteScratch:       make([]byte, arenaSize.ByteScratch+1),
		Uint16Scratch:     make([]uint16, arenaSize.Uint16Scratch+1),
		Int16Scratch:      make([]int16, arenaSize.Int16Scratch+1),
		Int32Scratch:      make([]int32, arenaSize.Int32Scratch+1),
	}
	arenaReq, err := av1.BindDecoderFrameWorkPostFilterRequestFromScratch(scratch, side, flatScratch)
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
	if _, err := av1.BindDecoderFrameWorkPostFilterRequestFromScratch(scratch, side, shortFlatScratch); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short arena bind err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
	negativeScratch := scratch
	negativeScratch.LoopFilter.Edges = -1
	if _, err := av1.BindDecoderFrameWorkPostFilterRequestFromScratch(negativeScratch, side, flatScratch); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("negative arena bind err=%v want %v", err, av1.ErrFrameShortBuffer)
	}

	shortBuffers := buffers
	shortBuffers.SuperResOutputFrame = shortBuffers.SuperResOutputFrame[:scratch.SuperRes.OutputFrame-1]
	if _, err := av1.BindDecoderFrameWorkPostFilterRequest(scratch, shortBuffers); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short aggregate bind err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
}

func TestPublicDecoderPostFilterBindingAllocs(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	size := av1.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	loopFilterSize := av1.DecoderFrameWorkLoopFilterPostFilterScratchSize{Edges: 4}
	loopFilterEdges := make([]av1.DecoderFrameWorkLoopFilterPostFilterEdge, loopFilterSize.Edges)
	cdefParams := av1.CDEFParams{Bits: 1, StrengthCount: 2}
	restorationBuffersParams := av1.RestorationParams{
		Type:       [3]av1.RestorationType{av1.RestorationWiener, av1.RestorationSGRProj, av1.RestorationNone},
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}
	sideScratchSize, err := av1.DecoderFrameWorkSideDataScratchLen(sequence, size, cdefParams, restorationBuffersParams)
	if err != nil {
		t.Fatal(err)
	}
	sideScratch := av1.DecoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, sideScratchSize.CDEFIndexMap),
		CDEFReadMap:              make([]bool, sideScratchSize.CDEFReadMap),
		LoopFilterMap:            make([]av1.DecoderFrameWorkLoopFilterBlockRecord, sideScratchSize.LoopFilterMap),
		RestorationRecords:       make([]av1.TileRestorationUnitRecord, sideScratchSize.RestorationRecords),
		RestorationBoundaryAbove: make([]uint16, sideScratchSize.RestorationBoundaryAbove),
		RestorationBoundaryBelow: make([]uint16, sideScratchSize.RestorationBoundaryBelow),
	}
	cdefSize := av1.DecoderFrameWorkCDEFPostFilterScratchSize{Samples: [3]int{64, 16, 16}, Dst: [3]int{64, 16, 16}, DirectionGrid: 1, VarianceGrid: 1, Input: av1.CDEFInputBufferSize, UnitDst: av1.CDEFInputBufferSize}
	var sampleScratch [3][]uint16
	var dstScratch [3][]uint16
	for plane := 0; plane < 3; plane++ {
		sampleScratch[plane] = make([]uint16, cdefSize.Samples[plane])
		dstScratch[plane] = make([]uint16, cdefSize.Dst[plane])
	}
	directionGrid := make([]av1.CDEFDirectionGrid, cdefSize.DirectionGrid)
	varianceGrid := make([]av1.CDEFVarianceGrid, cdefSize.VarianceGrid)
	inputScratch := make([]uint16, cdefSize.Input)
	unitDstScratch := make([]uint16, cdefSize.UnitDst)
	superSize := av1.DecoderFrameWorkSuperResPostFilterScratchSize{OutputFrame: 64, CodedSamples: [3]int{16, 4, 4}, OutputSamples: [3]int{32, 8, 8}}
	outputFrame := make([]byte, superSize.OutputFrame)
	var codedScratch [3][]uint16
	var outputScratch [3][]uint16
	for plane := 0; plane < 3; plane++ {
		codedScratch[plane] = make([]uint16, superSize.CodedSamples[plane])
		outputScratch[plane] = make([]uint16, superSize.OutputSamples[plane])
	}
	restorationSize := av1.DecoderFrameWorkRestorationPostFilterScratchSize{}
	restorationSize.Samples.DataLen = 3
	restorationSize.Samples.DstLen = 5
	restorationSize.Apply.Unit.Wiener = 7
	restorationSize.Apply.Unit.SGRProj = 11
	restorationSize.Apply.Boundary.Above = 13
	restorationSize.Apply.Boundary.Below = 17
	dataScratch := make([]uint16, 3)
	restorationDst := make([]uint16, 5)
	wienerScratch := make([]uint16, 7)
	sgrScratch := make([]int32, 11)
	aboveScratch := make([]uint16, 13)
	belowScratch := make([]uint16, 17)
	filmSize := av1.DecoderFrameWorkFilmGrainPostFilterScratchSize{LumaGrain: 3, ChromaGrain: [2]int{5, 7}, LumaSamples: 11, ChromaSamples: [2]int{13, 17}}
	lumaGrain := make([]int16, 3)
	chromaGrain := [2][]int16{make([]int16, 5), make([]int16, 7)}
	lumaSamples := make([]uint16, 11)
	chromaSamples := [2][]uint16{make([]uint16, 13), make([]uint16, 17)}
	postFilterSize := av1.DecoderFrameWorkPostFilterScratchSize{
		LoopFilter:  loopFilterSize,
		CDEF:        cdefSize,
		SuperRes:    superSize,
		Restoration: restorationSize,
		FilmGrain:   filmSize,
	}
	postFilterBuffers := av1.DecoderFrameWorkPostFilterRequestBuffers{
		LoopFilterEdges: loopFilterEdges,

		CDEFSampleScratch:  sampleScratch,
		CDEFDstScratch:     dstScratch,
		CDEFDirectionGrid:  directionGrid,
		CDEFVarianceGrid:   varianceGrid,
		CDEFInputScratch:   inputScratch,
		CDEFUnitDstScratch: unitDstScratch,

		SuperResOutputFrame:   outputFrame,
		SuperResCodedScratch:  codedScratch,
		SuperResOutputScratch: outputScratch,

		RestorationDataScratch:          dataScratch,
		RestorationDstScratch:           restorationDst,
		RestorationWienerScratch:        wienerScratch,
		RestorationSGRProjScratch:       sgrScratch,
		RestorationBoundaryAboveScratch: aboveScratch,
		RestorationBoundaryBelowScratch: belowScratch,

		FilmGrainLumaGrain:     lumaGrain,
		FilmGrainChromaGrain:   chromaGrain,
		FilmGrainLumaSamples:   lumaSamples,
		FilmGrainChromaSamples: chromaSamples,
	}
	postFilterArenaSize := av1.DecoderFrameWorkPostFilterRequestScratchLen(postFilterSize)
	postFilterSide := av1.DecoderFrameWorkPostFilterRequestSideData{}
	postFilterArena := av1.DecoderFrameWorkPostFilterRequestScratch{
		LoopFilterEdges:   make([]av1.DecoderFrameWorkLoopFilterPostFilterEdge, postFilterArenaSize.LoopFilterEdges),
		CDEFDirectionGrid: make([]av1.CDEFDirectionGrid, postFilterArenaSize.CDEFDirectionGrid),
		CDEFVarianceGrid:  make([]av1.CDEFVarianceGrid, postFilterArenaSize.CDEFVarianceGrid),
		ByteScratch:       make([]byte, postFilterArenaSize.ByteScratch),
		Uint16Scratch:     make([]uint16, postFilterArenaSize.Uint16Scratch),
		Int16Scratch:      make([]int16, postFilterArenaSize.Int16Scratch),
		Int32Scratch:      make([]int32, postFilterArenaSize.Int32Scratch),
	}

	allocs := testing.AllocsPerRun(1000, func() {
		sideData, bindErr := av1.BindDecoderFrameWorkSideData(sequence, size, cdefParams, restorationBuffersParams, sideScratch)
		if bindErr != nil {
			err = bindErr
			return
		}
		postFilterBuffers.LoopFilterMap = sideData.LoopFilterMap
		postFilterBuffers.CDEFIndexMap = sideData.CDEFIndexMap
		postFilterBuffers.RestorationRecords = sideData.RestorationFrameBuffers.Records
		postFilterBuffers.RestorationBoundaries = sideData.RestorationFrameBuffers.Boundaries
		postFilterSide = av1.DecoderFrameWorkPostFilterSideData(sideData)
		_, err = av1.BindDecoderFrameWorkLoopFilterPostFilterRequest(loopFilterSize, sideData.LoopFilterMap, loopFilterEdges)
		if err != nil {
			return
		}
		_, err = av1.BindDecoderFrameWorkCDEFPostFilterRequest(cdefSize, sideData.CDEFIndexMap, sampleScratch, dstScratch, directionGrid, varianceGrid, inputScratch, unitDstScratch)
		if err != nil {
			return
		}
		_, err = av1.BindDecoderFrameWorkSuperResPostFilterRequest(superSize, outputFrame, codedScratch, outputScratch)
		if err != nil {
			return
		}
		_, err = av1.BindDecoderFrameWorkRestorationPostFilterRequest(restorationSize, [3][]av1.TileRestorationUnitRecord{}, [3]av1.TileRestorationStripeBoundaries{}, dataScratch, restorationDst, wienerScratch, sgrScratch, aboveScratch, belowScratch, false)
		if err != nil {
			return
		}
		_, err = av1.BindDecoderFrameWorkFilmGrainPostFilterRequest(filmSize, lumaGrain, chromaGrain, lumaSamples, chromaSamples)
		if err != nil {
			return
		}
		_, err = av1.BindDecoderFrameWorkPostFilterRequest(postFilterSize, postFilterBuffers)
		if err != nil {
			return
		}
		_, err = av1.BindDecoderFrameWorkPostFilterRequestFromScratch(postFilterSize, postFilterSide, postFilterArena)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicDecoderPostFilterSequence() av1.SequenceHeader {
	return av1.SequenceHeader{
		EnableSuperRes: true,
		EnableCDEF:     true,
		ColorConfig: av1.ColorConfig{
			BitDepth:     8,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	}
}

func publicDecoderPostFilterFramePool(t testing.TB, format av1.FrameFormat, count int) av1.FramePool {
	t.Helper()
	_, backingSize, err := av1.FramePoolRequiredSize(format, count)
	if err != nil {
		t.Fatal(err)
	}
	frames := make([]av1.Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := av1.BindFramePool(make([]byte, backingSize), format, frames, free, used)
	if err != nil {
		t.Fatal(err)
	}
	return pool
}

func publicDecoderPostFilterFrame(t testing.TB, format av1.FrameFormat) *av1.Frame {
	t.Helper()
	layout, err := av1.FrameRequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	frame, err := av1.BindFrame(make([]byte, layout.Size), format)
	if err != nil {
		t.Fatal(err)
	}
	return &frame
}

func publicDecoderPostFilterRequestScratch(size av1.DecoderFrameWorkPostFilterRequestScratchSize) av1.DecoderFrameWorkPostFilterRequestScratch {
	return av1.DecoderFrameWorkPostFilterRequestScratch{
		LoopFilterEdges:   make([]av1.DecoderFrameWorkLoopFilterPostFilterEdge, size.LoopFilterEdges),
		CDEFDirectionGrid: make([]av1.CDEFDirectionGrid, size.CDEFDirectionGrid),
		CDEFVarianceGrid:  make([]av1.CDEFVarianceGrid, size.CDEFVarianceGrid),
		ByteScratch:       make([]byte, size.ByteScratch),
		Uint16Scratch:     make([]uint16, size.Uint16Scratch),
		Int16Scratch:      make([]int16, size.Int16Scratch),
		Int32Scratch:      make([]int32, size.Int32Scratch),
	}
}

func publicDecoderLoopFilterRecordAt(col0 int, row0 int, col1 int, row1 int) av1.DecoderFrameWorkLoopFilterBlockRecord {
	return av1.DecoderFrameWorkLoopFilterBlockRecord{
		Valid: true,
		Block: av1.TileBlockVisit{
			MICol:     uint32(col0),
			MIRow:     uint32(row0),
			MIColEnd:  uint32(col1),
			MIRowEnd:  uint32(row1),
			X4:        col0,
			Y4:        row0,
			Size:      av1.TileBlockSize16x16,
			VisibleW4: uint8(col1 - col0),
			VisibleH4: uint8(row1 - row0),
		},
		TransformTree: av1.TileTransformTreeResult{
			Y:     av1.TileTransformSize16x16,
			UV:    av1.TileTransformSize8x8,
			HasUV: true,
		},
	}
}

func publicFillDecoderLoopFilterMap(filterMap av1.DecoderFrameWorkLoopFilterMap, records ...av1.DecoderFrameWorkLoopFilterBlockRecord) {
	for _, record := range records {
		for row := record.Block.MIRow; row < record.Block.MIRowEnd; row++ {
			base := int(row) * filterMap.Stride
			for col := record.Block.MICol; col < record.Block.MIColEnd; col++ {
				filterMap.Records[base+int(col)] = record
			}
		}
	}
}

func publicFillDecoderPostFilterPlane(plane av1.FramePlane) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			plane.Pix[y*plane.Stride+x] = byte((x*31 + y*47 + (x^y)*13) & 0xff)
		}
	}
}
