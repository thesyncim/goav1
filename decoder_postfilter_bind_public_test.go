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

func TestPublicDecoderPostFilterBindingAllocs(t *testing.T) {
	sequence := publicDecoderPostFilterSequence()
	size := av1.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	_, _, loopFilterLength, err := av1.DecoderFrameWorkLoopFilterMapShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	loopFilterRecords := make([]av1.DecoderFrameWorkLoopFilterBlockRecord, loopFilterLength)
	loopFilterSize := av1.DecoderFrameWorkLoopFilterPostFilterScratchSize{Edges: 4}
	loopFilterEdges := make([]av1.DecoderFrameWorkLoopFilterPostFilterEdge, loopFilterSize.Edges)
	restorationBuffersParams := av1.RestorationParams{
		Type:       [3]av1.RestorationType{av1.RestorationWiener, av1.RestorationSGRProj, av1.RestorationNone},
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}
	restorationBuffersPlan, err := av1.DecoderFrameWorkRestorationFramePlan(sequence, size, restorationBuffersParams)
	if err != nil {
		t.Fatal(err)
	}
	restorationRecordBacking := make([]av1.TileRestorationUnitRecord, restorationBuffersPlan.UnitRecordLen())
	restorationAbove := make([]uint16, restorationBuffersPlan.BoundaryBufferLen())
	restorationBelow := make([]uint16, restorationBuffersPlan.BoundaryBufferLen())
	index := make([]uint8, 1)
	read := make([]bool, 1)
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

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _, err = av1.DecoderFrameWorkLoopFilterMapShape(sequence, size)
		if err != nil {
			return
		}
		loopFilterMap, bindErr := av1.BindDecoderFrameWorkLoopFilterMap(sequence, size, loopFilterRecords)
		if bindErr != nil {
			err = bindErr
			return
		}
		_, err = av1.BindDecoderFrameWorkLoopFilterPostFilterRequest(loopFilterSize, loopFilterMap, loopFilterEdges)
		if err != nil {
			return
		}
		restorationBuffers, bindErr := av1.BindDecoderFrameWorkRestorationFrameBuffers(sequence, size, restorationBuffersParams, restorationRecordBacking, restorationAbove, restorationBelow)
		if bindErr != nil {
			err = bindErr
			return
		}
		err = restorationBuffers.ResetRecords()
		if err != nil {
			return
		}
		_, _, _, err = av1.DecoderFrameWorkCDEFIndexMapShape(sequence, size)
		if err != nil {
			return
		}
		cdefMap, bindErr := av1.BindDecoderFrameWorkCDEFIndexMap(sequence, size, av1.CDEFParams{Bits: 1, StrengthCount: 2}, index, read)
		if bindErr != nil {
			err = bindErr
			return
		}
		_, err = av1.BindDecoderFrameWorkCDEFPostFilterRequest(cdefSize, cdefMap, sampleScratch, dstScratch, directionGrid, varianceGrid, inputScratch, unitDstScratch)
		_, err = av1.BindDecoderFrameWorkSuperResPostFilterRequest(superSize, outputFrame, codedScratch, outputScratch)
		_, err = av1.BindDecoderFrameWorkRestorationPostFilterRequest(restorationSize, [3][]av1.TileRestorationUnitRecord{}, [3]av1.TileRestorationStripeBoundaries{}, dataScratch, restorationDst, wienerScratch, sgrScratch, aboveScratch, belowScratch, false)
		_, err = av1.BindDecoderFrameWorkFilmGrainPostFilterRequest(filmSize, lumaGrain, chromaGrain, lumaSamples, chromaSamples)
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

func publicDecoderPostFilterFrame(t *testing.T, format av1.FrameFormat) *av1.Frame {
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
