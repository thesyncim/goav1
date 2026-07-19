package decoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

func TestFrameWorkPostFilterContextApplyCDEFPostFilterRejectsEarlierLoopFilter(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FrameSize: parser.FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              64,
				SuperResDenominator: 8,
			},
			LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{1}},
			CDEF: parser.CDEFParams{
				Damping:       5,
				StrengthCount: 1,
				YStrength:     [parser.MaxCDEFStrengths]uint8{8},
			},
		},
		Output: output,
	}
	_, err := ctx.ApplyCDEFPostFilter(FrameWorkCDEFPostFilterRequest{})
	if !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("ApplyCDEFPostFilter err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func TestFrameWorkPostFilterContextApplyCDEFPostFilterSkipsZeroStrengths(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 32})
	output.Y.Pix[0] = 0x5a
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FrameSize: parser.FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              64,
				SuperResDenominator: 8,
			},
			CDEF: parser.CDEFParams{Bits: 1, StrengthCount: 2},
		},
		Output: output,
	}
	result, err := ctx.ApplyCDEFPostFilter(FrameWorkCDEFPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkCDEFPostFilterResult{}) {
		t.Fatalf("result=%+v want zero", result)
	}
	if output.Y.Pix[0] != 0x5a {
		t.Fatalf("output sample=%d want 0x5a", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplyCDEFPostFilterFiltersLuma(t *testing.T) {
	const width = 64
	const height = 64

	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkCDEFPlane(output.Y)
	before := testCopyFrameWorkCDEFPlane(output.Y)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	result, err := ctx.ApplyCDEFPostFilter(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 1 || result.Units != 1 || result.Blocks == 0 {
		t.Fatalf("result=%+v", result)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("CDEF did not change luma samples")
	}
}

func TestFrameWorkPostFilterContextApplyCDEFPostFilterDefaultsIndexMapFromContext(t *testing.T) {
	const width = 64
	const height = 64

	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkCDEFPlane(output.Y)
	before := testCopyFrameWorkCDEFPlane(output.Y)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	cdefMap := req.IndexMap
	ctx.CDEFIndexMap = &cdefMap
	req.IndexMap = FrameWorkCDEFIndexMap{}
	result, err := ctx.ApplyCDEFPostFilter(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 1 || result.Units != 1 || result.Blocks == 0 {
		t.Fatalf("result=%+v", result)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("CDEF did not change luma samples")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRunsCDEF(t *testing.T) {
	const width = 64
	const height = 64

	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkCDEFPlane(output.Y)
	before := testCopyFrameWorkCDEFPlane(output.Y)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	cdefMap := req.IndexMap
	ctx.CDEFIndexMap = &cdefMap
	req.IndexMap = FrameWorkCDEFIndexMap{}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{CDEF: req})
	if err != nil {
		t.Fatal(err)
	}
	if size.CDEF.Input != cdef.InputBufferSize || size.CDEF.UnitDst != cdef.InputBufferSize {
		t.Fatalf("CDEF scratch=%+v", size.CDEF)
	}
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{CDEF: req})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterCDEF || result.CDEF.Planes != 1 || result.CDEF.Units != 1 {
		t.Fatalf("result=%+v", result)
	}
	if err := next.RequireNoRemainingPostFilters(); err != nil {
		t.Fatalf("RequireNoRemainingPostFilters err=%v", err)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("supported postfilter pipeline did not change luma samples")
	}
}

func TestFrameWorkSupportedPostFilterRunnerApplyRunsCDEF(t *testing.T) {
	const width = 64
	const height = 64

	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkCDEFPlane(output.Y)
	before := testCopyFrameWorkCDEFPlane(output.Y)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	cdefMap := req.IndexMap
	ctx.CDEFIndexMap = &cdefMap
	req.IndexMap = FrameWorkCDEFIndexMap{}
	runner := FrameWorkSupportedPostFilterRunner{Request: FrameWorkPostFilterRequest{CDEF: req}}
	size, err := runner.ScratchLen(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if size.CDEF.Input != cdef.InputBufferSize || size.LoopFilter != (FrameWorkLoopFilterPostFilterScratchSize{}) ||
		size.Restoration != (FrameWorkRestorationPostFilterScratchSize{}) ||
		size.FilmGrain != (FrameWorkFilmGrainPostFilterScratchSize{}) {
		t.Fatalf("runner scratch=%+v", size)
	}
	if err := runner.Apply(ctx); err != nil {
		t.Fatal(err)
	}
	if runner.Result.Completed != FrameWorkPostFilterCDEF || runner.Context.RemainingPostFilters() != 0 {
		t.Fatalf("runner result=%+v remaining=%b", runner.Result, runner.Context.RemainingPostFilters())
	}
	if runner.Context.DetachedPostFilterOutput() {
		t.Fatal("supported runner returned detached output")
	}
	if err := runner.Context.RequirePublishablePostFilterOutput(); err != nil {
		t.Fatalf("RequirePublishablePostFilterOutput err=%v", err)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("runner did not change luma samples")
	}
}

func TestFrameWorkPostFilterRunnerScratchLenRejectsNilRunner(t *testing.T) {
	var supported *FrameWorkSupportedPostFilterRunner
	if _, err := supported.ScratchLen(FrameWorkPostFilterContext{}); !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("supported runner ScratchLen err=%v want %v", err, ErrInvalidFrameWorkState)
	}
	var caller *FrameWorkCallerPostFilterRunner
	if _, err := caller.ScratchLen(FrameWorkPostFilterContext{}); !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("caller runner ScratchLen err=%v want %v", err, ErrInvalidFrameWorkState)
	}
}

func TestFrameWorkPostFilterContextSupportedPostFilterCapabilities(t *testing.T) {
	cdefEvent := Event{
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{8},
		},
	}
	ctx := FrameWorkPostFilterContext{Event: cdefEvent}
	supported, err := ctx.SupportedPostFilters()
	if err != nil {
		t.Fatal(err)
	}
	unsupported, err := ctx.UnsupportedPostFilters()
	if err != nil {
		t.Fatal(err)
	}
	if supported != FrameWorkPostFilterCDEF || !unsupported.Empty() {
		t.Fatalf("supported=%b unsupported=%b", supported, unsupported)
	}
	if err := ctx.RequireSupportedPostFilters(); err != nil {
		t.Fatalf("RequireSupportedPostFilters err=%v", err)
	}

	superResCtx := FrameWorkPostFilterContext{
		Event: Event{
			CDEF:      cdefEvent.CDEF,
			FrameSize: parser.FrameSize{SuperResEnabled: true},
		},
	}
	supported, err = superResCtx.SupportedPostFilters()
	if err != nil {
		t.Fatal(err)
	}
	unsupported, err = superResCtx.UnsupportedPostFilters()
	if err != nil {
		t.Fatal(err)
	}
	if supported != FrameWorkPostFilterCDEF || unsupported != FrameWorkPostFilterSuperRes {
		t.Fatalf("superres supported=%b unsupported=%b", supported, unsupported)
	}
	if err := superResCtx.RequireSupportedPostFilters(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequireSupportedPostFilters err=%v want %v", err, ErrUnsupportedPostFilter)
	}

	tailCtx := superResCtx.WithCompletedPostFilters(FrameWorkPostFilterCDEF)
	supported, err = tailCtx.SupportedPostFilters()
	if err != nil {
		t.Fatal(err)
	}
	unsupported, err = tailCtx.UnsupportedPostFilters()
	if err != nil {
		t.Fatal(err)
	}
	if !supported.Empty() || unsupported != FrameWorkPostFilterSuperRes {
		t.Fatalf("tail supported=%b unsupported=%b", supported, unsupported)
	}
}

func TestFrameWorkCallerPostFilterRunnerAllocs(t *testing.T) {
	runner := FrameWorkCallerPostFilterRunner{}
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := runner.ScratchLen(FrameWorkPostFilterContext{}); err != nil {
			t.Fatal(err)
		}
		if err := runner.Apply(FrameWorkPostFilterContext{}); err != nil {
			t.Fatal(err)
		}
		if runner.Context.RemainingPostFilters() != 0 || runner.Result.Completed != 0 {
			t.Fatalf("runner context=%+v result=%+v", runner.Context, runner.Result)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkCallerPostFilterRunner allocated: %f", allocs)
	}
}

func TestFrameWorkCDEFPostFilterScratchSizeBindRequest(t *testing.T) {
	size := FrameWorkCDEFPostFilterScratchSize{
		Samples:       [3]int{8, 4, 2},
		Dst:           [3]int{7, 3, 1},
		DirectionGrid: 5,
		VarianceGrid:  5,
		Input:         cdef.InputBufferSize,
		UnitDst:       cdef.InputBufferSize,
	}
	indexMap := FrameWorkCDEFIndexMap{
		Index:  make([]uint8, 6),
		Read:   make([]bool, 6),
		Stride: 3,
		Rows:   2,
	}
	samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(size)
	req, err := size.BindRequest(indexMap, samples, dst, directionGrid, varianceGrid, input, unitDst)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.SampleScratch[0]) != size.Samples[0] || len(req.DstScratch[1]) != size.Dst[1] ||
		len(req.DirectionGrid) != size.DirectionGrid || len(req.VarianceGrid) != size.VarianceGrid ||
		len(req.InputScratch) != size.Input || len(req.UnitDstScratch) != size.UnitDst {
		t.Fatalf("request lengths samples=%v dst=%v direction=%d variance=%d input=%d unitDst=%d",
			[3]int{len(req.SampleScratch[0]), len(req.SampleScratch[1]), len(req.SampleScratch[2])},
			[3]int{len(req.DstScratch[0]), len(req.DstScratch[1]), len(req.DstScratch[2])},
			len(req.DirectionGrid), len(req.VarianceGrid), len(req.InputScratch), len(req.UnitDstScratch))
	}
	if req.IndexMap.Stride != indexMap.Stride || req.IndexMap.Rows != indexMap.Rows ||
		len(req.IndexMap.Index) != len(indexMap.Index) || len(req.IndexMap.Read) != len(indexMap.Read) {
		t.Fatalf("index map=%+v want %+v", req.IndexMap, indexMap)
	}
}

func TestFrameWorkCDEFPostFilterScratchSizeBindRequestRejectsShortBuffers(t *testing.T) {
	size := FrameWorkCDEFPostFilterScratchSize{
		Samples:       [3]int{4, 3, 2},
		Dst:           [3]int{4, 3, 2},
		DirectionGrid: 2,
		VarianceGrid:  2,
		Input:         cdef.InputBufferSize,
		UnitDst:       cdef.InputBufferSize,
	}
	samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(size)
	tests := []struct {
		name     string
		size     FrameWorkCDEFPostFilterScratchSize
		samples  [3][]uint16
		dst      [3][]uint16
		dir      []cdef.DirectionGrid
		variance []cdef.VarianceGrid
		input    []uint16
		unitDst  []uint16
	}{
		{name: "sample", size: size, samples: [3][]uint16{samples[0][:3], samples[1], samples[2]}, dst: dst, dir: directionGrid, variance: varianceGrid, input: input, unitDst: unitDst},
		{name: "dst", size: size, samples: samples, dst: [3][]uint16{dst[0], dst[1][:2], dst[2]}, dir: directionGrid, variance: varianceGrid, input: input, unitDst: unitDst},
		{name: "direction", size: size, samples: samples, dst: dst, dir: directionGrid[:1], variance: varianceGrid, input: input, unitDst: unitDst},
		{name: "variance", size: size, samples: samples, dst: dst, dir: directionGrid, variance: varianceGrid[:1], input: input, unitDst: unitDst},
		{name: "input", size: size, samples: samples, dst: dst, dir: directionGrid, variance: varianceGrid, input: input[:cdef.InputBufferSize-1], unitDst: unitDst},
		{name: "unit-dst", size: size, samples: samples, dst: dst, dir: directionGrid, variance: varianceGrid, input: input, unitDst: unitDst[:cdef.InputBufferSize-1]},
		{name: "negative", size: FrameWorkCDEFPostFilterScratchSize{Samples: [3]int{-1}}, samples: samples, dst: dst, dir: directionGrid, variance: varianceGrid, input: input, unitDst: unitDst},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.size.BindRequest(FrameWorkCDEFIndexMap{}, tt.samples, tt.dst, tt.dir, tt.variance, tt.input, tt.unitDst); !errors.Is(err, frame.ErrShortBuffer) {
				t.Fatalf("BindRequest err=%v want %v", err, frame.ErrShortBuffer)
			}
		})
	}
}

func TestFrameWorkCDEFPostFilterScratchSizeBindRequestAllocs(t *testing.T) {
	size := FrameWorkCDEFPostFilterScratchSize{
		Samples:       [3]int{4096, 1024, 1024},
		Dst:           [3]int{4096, 1024, 1024},
		DirectionGrid: 16,
		VarianceGrid:  16,
		Input:         cdef.InputBufferSize,
		UnitDst:       cdef.InputBufferSize,
	}
	samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(size)
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := size.BindRequest(FrameWorkCDEFIndexMap{}, samples, dst, directionGrid, varianceGrid, input, unitDst); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("BindRequest allocated: %f", allocs)
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRejectsUnsupportedBeforeMutation(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 32})
	output.Y.Pix[0] = 0x44
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FrameSize: parser.FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              64,
				SuperResEnabled:     true,
				SuperResDenominator: 16,
			},
			CDEF: parser.CDEFParams{
				Damping:       5,
				StrengthCount: 1,
				YStrength:     [parser.MaxCDEFStrengths]uint8{63},
			},
		},
		Output: output,
	}
	if _, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{}); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("SupportedPostFilterScratchLen err=%v want %v", err, ErrUnsupportedPostFilter)
	}
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{})
	if !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("ApplySupportedPostFilters err=%v want %v", err, ErrUnsupportedPostFilter)
	}
	if next.RemainingPostFilters() != ctx.RemainingPostFilters() || result != (FrameWorkPostFilterResult{}) {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if output.Y.Pix[0] != 0x44 {
		t.Fatalf("output sample=%d want 0x44", output.Y.Pix[0])
	}
}

// TestFrameWorkPostFilterContextApplyCDEFPostFilterFiltersSVCEnhancementLayer
// regression-tests CDEF on a 1280x720 frame (an SVC enhancement-layer-sized
// resolution where the unit grid is 20x12 and the chroma grid is 640x360).
// CDEF must derive every dimension from the per-frame Event/Output, never from
// a hardcoded global resolution. This guards against future regressions where
// CDEF assumes a single global frame size and trips when the enhancement layer
// decodes at a larger size than the base layer.
func TestFrameWorkPostFilterContextApplyCDEFPostFilterFiltersSVCEnhancementLayer(t *testing.T) {
	const width = 1280
	const height = 720
	const expectedLumaCols = (width + cdef.BlockSize - 1) / cdef.BlockSize  // 20
	const expectedLumaRows = (height + cdef.BlockSize - 1) / cdef.BlockSize // 12
	const expectedUnits = expectedLumaCols * expectedLumaRows               // 240

	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
			UVStrength:    [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{
		Width: width, Height: height, BitDepth: 8,
		SubsamplingX: true, SubsamplingY: true, Align: 32,
	})
	testFillFrameWorkCDEFPlane(output.Y)
	testFillFrameWorkCDEFPlane(output.U)
	testFillFrameWorkCDEFPlane(output.V)
	beforeY := testCopyFrameWorkCDEFPlane(output.Y)
	beforeU := testCopyFrameWorkCDEFPlane(output.U)
	beforeV := testCopyFrameWorkCDEFPlane(output.V)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}

	// Verify the unit grid derives from the per-frame size, not a hardcoded
	// global resolution. An SVC base layer of 640x360 would yield 10x6=60
	// units; the enhancement layer must report 20x12=240.
	cols, rows, err := frameWorkCDEFUnitGrid(event.FrameSize)
	if err != nil {
		t.Fatalf("frameWorkCDEFUnitGrid err=%v", err)
	}
	if cols != expectedLumaCols || rows != expectedLumaRows {
		t.Fatalf("unit grid=%dx%d want %dx%d", cols, rows, expectedLumaCols, expectedLumaRows)
	}

	// Sanity-check scratch sizing scales with the frame's plane dimensions.
	size, err := ctx.CDEFPostFilterScratchLen()
	if err != nil {
		t.Fatalf("CDEFPostFilterScratchLen err=%v", err)
	}
	if size.Samples[0] < width*height || size.Dst != [3]int{} {
		t.Fatalf("luma scratch=%d dst=%v want samples >= %d and no dst scratch", size.Samples[0], size.Dst, width*height)
	}
	chromaSize := (width / 2) * (height / 2)
	if size.Samples[1] < chromaSize || size.Samples[2] < chromaSize {
		t.Fatalf("chroma sample scratch=%d/%d want >= %d", size.Samples[1], size.Samples[2], chromaSize)
	}
	if size.DirectionGrid < expectedUnits || size.VarianceGrid < expectedUnits {
		t.Fatalf("direction/variance grid=%d/%d want >= %d", size.DirectionGrid, size.VarianceGrid, expectedUnits)
	}
	// Per-unit scratch buffers stay bounded by the CDEF unit (64x64) plus
	// borders; they must not scale with frame resolution.
	if size.Input != cdef.InputBufferSize || size.UnitDst != cdef.InputBufferSize {
		t.Fatalf("input/unitDst scratch=%d/%d want %d", size.Input, size.UnitDst, cdef.InputBufferSize)
	}

	req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	if req.IndexMap.Stride != expectedLumaCols || req.IndexMap.Rows != expectedLumaRows {
		t.Fatalf("indexMap stride/rows=%d/%d want %d/%d",
			req.IndexMap.Stride, req.IndexMap.Rows, expectedLumaCols, expectedLumaRows)
	}

	result, err := ctx.ApplyCDEFPostFilter(req)
	if err != nil {
		t.Fatalf("ApplyCDEFPostFilter err=%v", err)
	}
	if result.Planes != 3 {
		t.Fatalf("planes=%d want 3", result.Planes)
	}
	// All 240 units have non-zero strengths and are marked read, so every
	// unit should run on every plane.
	if result.Units != expectedUnits*3 {
		t.Fatalf("units=%d want %d", result.Units, expectedUnits*3)
	}
	if result.Blocks == 0 {
		t.Fatalf("blocks=0; want > 0")
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, beforeY) {
		t.Fatal("CDEF did not change luma samples")
	}
	if !testFrameWorkCDEFPlaneChanged(output.U, beforeU) {
		t.Fatal("CDEF did not change U samples")
	}
	if !testFrameWorkCDEFPlaneChanged(output.V, beforeV) {
		t.Fatal("CDEF did not change V samples")
	}
}

func TestFrameWorkPostFilterContextApplyCDEFPostFilterFiltersChromaWithLumaDirectionPass(t *testing.T) {
	const width = 128
	const height = 64

	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			UVStrength:    [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkCDEFPlane(output.Y)
	testFillFrameWorkCDEFPlane(output.U)
	testFillFrameWorkCDEFPlane(output.V)
	beforeY := testCopyFrameWorkCDEFPlane(output.Y)
	beforeU := testCopyFrameWorkCDEFPlane(output.U)
	beforeV := testCopyFrameWorkCDEFPlane(output.V)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	result, err := ctx.ApplyCDEFPostFilter(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 3 || result.Units != 4 || result.Blocks == 0 {
		t.Fatalf("result=%+v", result)
	}
	if testFrameWorkCDEFPlaneChanged(output.Y, beforeY) {
		t.Fatal("direction-only luma pass changed luma samples")
	}
	if !testFrameWorkCDEFPlaneChanged(output.U, beforeU) {
		t.Fatal("CDEF did not change U samples")
	}
	if !testFrameWorkCDEFPlaneChanged(output.V, beforeV) {
		t.Fatal("CDEF did not change V samples")
	}
}

func BenchmarkFrameWorkCDEFPostFilterScratchSizeBindRequest(b *testing.B) {
	size := FrameWorkCDEFPostFilterScratchSize{
		Samples:       [3]int{1920 * 1080, 960 * 540, 960 * 540},
		Dst:           [3]int{1920 * 1080, 960 * 540, 960 * 540},
		DirectionGrid: 120 * 68,
		VarianceGrid:  120 * 68,
		Input:         cdef.InputBufferSize,
		UnitDst:       cdef.InputBufferSize,
	}
	indexMap := FrameWorkCDEFIndexMap{
		Index:  make([]uint8, size.DirectionGrid),
		Read:   make([]bool, size.DirectionGrid),
		Stride: 120,
		Rows:   68,
	}
	samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(size)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := size.BindRequest(indexMap, samples, dst, directionGrid, varianceGrid, input, unitDst); err != nil {
			b.Fatal(err)
		}
	}
}

func testFrameWorkCDEFFrame(t testing.TB, format frame.Format) *frame.Frame {
	t.Helper()
	layout, err := frame.RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	output, err := frame.Bind(make([]byte, layout.Size), format)
	if err != nil {
		t.Fatal(err)
	}
	return &output
}

func testFrameWorkCDEFPostFilterRequest(t testing.TB, ctx FrameWorkPostFilterContext, event Event) FrameWorkCDEFPostFilterRequest {
	t.Helper()
	batch := threading.FrameWorkBatch{
		FrameWorkFrameContext: threading.FrameWorkFrameContext{
			Sequence:  threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader),
			FrameSize: event.FrameSize,
			CDEF:      event.CDEF,
		},
	}
	_, _, length, err := batch.CDEFIndexMapShape()
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := batch.BindCDEFIndexMap(make([]uint8, length), make([]bool, length))
	if err != nil {
		t.Fatal(err)
	}
	for i := range cdefMap.Read {
		cdefMap.Read[i] = true
	}
	size, err := ctx.CDEFPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(size)
	req, err := size.BindRequest(cdefMap, samples, dst, directionGrid, varianceGrid, input, unitDst)
	if err != nil {
		t.Fatal(err)
	}
	return req
}

func testFrameWorkCDEFScratchStorage(size FrameWorkCDEFPostFilterScratchSize) ([3][]uint16, [3][]uint16, []cdef.DirectionGrid, []cdef.VarianceGrid, []uint16, []uint16) {
	var samples [3][]uint16
	var dst [3][]uint16
	for plane := range 3 {
		if size.Samples[plane] > 0 {
			samples[plane] = make([]uint16, size.Samples[plane])
		}
		if size.Dst[plane] > 0 {
			dst[plane] = make([]uint16, size.Dst[plane])
		}
	}
	return samples, dst,
		make([]cdef.DirectionGrid, maxInt(size.DirectionGrid, 0)),
		make([]cdef.VarianceGrid, maxInt(size.VarianceGrid, 0)),
		make([]uint16, maxInt(size.Input, 0)),
		make([]uint16, maxInt(size.UnitDst, 0))
}

func testFillFrameWorkCDEFPlane(plane frame.Plane) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			plane.Pix[y*plane.Stride+x] = byte((x*37 + y*53 + (x^y)*17) & 0xff)
		}
	}
}

func testCopyFrameWorkCDEFPlane(plane frame.Plane) []byte {
	out := make([]byte, plane.Width*plane.Height)
	for y := 0; y < plane.Height; y++ {
		copy(out[y*plane.Width:(y+1)*plane.Width], plane.Pix[y*plane.Stride:y*plane.Stride+plane.Width])
	}
	return out
}

func testFrameWorkCDEFPlaneChanged(plane frame.Plane, before []byte) bool {
	for y := 0; y < plane.Height; y++ {
		row := before[y*plane.Width : (y+1)*plane.Width]
		for x := 0; x < plane.Width; x++ {
			if plane.Pix[y*plane.Stride+x] != row[x] {
				return true
			}
		}
	}
	return false
}

func TestFrameWorkFillCDEFInputSentinelsFillsHaloOnly(t *testing.T) {
	input := make([]uint16, cdef.InputBufferSize)
	for i := range input {
		input[i] = 0x7777
	}
	const unitW = 32
	const unitH = 32
	if err := frameWorkFillCDEFInputSentinels(input, unitW, unitH); err != nil {
		t.Fatalf("frameWorkFillCDEFInputSentinels: %v", err)
	}

	fillW := unitW + 2*cdef.HorizontalBorder
	fillH := unitH + 2*cdef.VerticalBorder
	for row := range fillH {
		for col := range fillW {
			got := input[row*cdef.BStride+col]
			want := uint16(0x7777)
			if row < cdef.VerticalBorder || row >= cdef.VerticalBorder+unitH ||
				col < cdef.HorizontalBorder || col >= cdef.HorizontalBorder+unitW {
				want = cdef.VeryLarge
			}
			if got != want {
				t.Fatalf("input row=%d col=%d got=%d want=%d", row, col, got, want)
			}
		}
	}
	fillLen := (fillH-1)*cdef.BStride + fillW
	if got := input[fillLen]; got != 0x7777 {
		t.Fatalf("inactive next row got=%d want marker", got)
	}
}

// TestFrameWorkCopyCDEFInputBottomEdgeReplicatesLastRow pins libaom's CDEF
// bot_linebuf behavior for chroma planes whose visible height ends inside a
// non-final CDEF unit. The chroma 33-tall plane in the libaom 66x66 vector hits
// CDEF unit row 0 with unit height 32: the kernel's bottom directional taps at
// by=7 (rows 28..31) need samples at rows 32 and 33 of the input buffer. Row 32
// is visible (real sample); row 33 in libaom comes from the YV12 buffer's
// aligned-but-out-of-crop padding (uv_height vs uv_crop_height), which holds
// the last visible row replicated by alignment/decode writes. Our SamplePlane
// is exactly the visible window, so frameWorkCopyCDEFInput synthesizes the
// extension by replicating the last visible row into the bottom border. The
// final CDEF unit row must keep the VeryLarge sentinel just like libaom's
// fill_rect(...CDEF_VERY_LARGE) when fbr == nvfb - 1.
func TestFrameWorkCopyCDEFInputBottomEdgeReplicatesLastRow(t *testing.T) {
	const planeHeight = 33
	const planeWidth = 33
	const unitX = 0
	const unitY = 0
	const unitW = 32
	const unitH = 32

	src := frame.SamplePlane{
		Pix:    make([]uint16, planeWidth*planeHeight),
		Stride: planeWidth,
		Width:  planeWidth,
		Height: planeHeight,
	}
	for y := range planeHeight {
		for x := range planeWidth {
			src.Pix[y*src.Stride+x] = uint16(100 + y)
		}
	}

	input := make([]uint16, cdef.InputBufferSize)
	for i := range input {
		input[i] = cdef.VeryLarge
	}
	if err := frameWorkCopyCDEFInput(input, src, unitX, unitY, unitW, unitH); err != nil {
		t.Fatalf("frameWorkCopyCDEFInput: %v", err)
	}

	originRow := cdef.VerticalBorder
	originCol := cdef.HorizontalBorder
	at := func(row, col int) uint16 {
		return input[(originRow+row)*cdef.BStride+originCol+col]
	}

	// Visible rows 0..32 must hold the real samples (last visible row is 32).
	for y := range planeHeight {
		for x := range planeWidth {
			if got, want := at(y, x), uint16(100+y); got != want {
				t.Fatalf("input row=%d col=%d got=%d want=%d", y, x, got, want)
			}
		}
	}
	// Rows past the visible plane within the CDEF VerticalBorder must replicate
	// the last visible row instead of leaving VeryLarge — otherwise the kernel's
	// directional taps at by=7 would diverge from libaom's bot_linebuf reads.
	const lastVisible = planeHeight - 1
	for y := planeHeight; y < unitY+unitH+cdef.VerticalBorder; y++ {
		for x := range planeWidth {
			if got, want := at(y, x), uint16(100+lastVisible); got != want {
				t.Fatalf("bottom-extension row=%d col=%d got=%d want=%d", y, x, got, want)
			}
		}
	}
}

// TestFrameWorkCopyCDEFInputRightEdgeReplicatesLastColumn pins libaom's CDEF
// av1_cdef_copy_sb8_16 read behavior for non-rightmost superblocks whose visible
// plane width ends inside the central copy's hsize+CDEF_HBORDER reach. The
// chroma 33-wide plane in the libaom 66x66 vector hits unit (0,0) with unit
// width 32: the kernel's directional secondary taps at bx=7 (cols 28..31) need
// samples at cols 32 and 33 of the input buffer. Col 32 is visible (real
// sample); col 33 in libaom comes from the YV12 buffer's
// aligned-but-out-of-crop padding (uv_width vs uv_crop_width), which holds the
// last visible column replicated by alignment/decode writes. Our SamplePlane is
// exactly the visible window, so frameWorkCopyCDEFInput synthesizes the
// extension by replicating the last visible column into the right border. The
// final CDEF unit column must keep the VeryLarge sentinel just like libaom's
// fill_rect(...CDEF_VERY_LARGE) frame_boundary[RIGHT] overlay when fbc ==
// nhfb - 1.
func TestFrameWorkCopyCDEFInputRightEdgeReplicatesLastColumn(t *testing.T) {
	const planeHeight = 33
	const planeWidth = 33
	const unitX = 0
	const unitY = 0
	const unitW = 32
	const unitH = 32

	src := frame.SamplePlane{
		Pix:    make([]uint16, planeWidth*planeHeight),
		Stride: planeWidth,
		Width:  planeWidth,
		Height: planeHeight,
	}
	for y := range planeHeight {
		for x := range planeWidth {
			src.Pix[y*src.Stride+x] = uint16(300 + x)
		}
	}

	input := make([]uint16, cdef.InputBufferSize)
	for i := range input {
		input[i] = cdef.VeryLarge
	}
	if err := frameWorkCopyCDEFInput(input, src, unitX, unitY, unitW, unitH); err != nil {
		t.Fatalf("frameWorkCopyCDEFInput: %v", err)
	}

	originRow := cdef.VerticalBorder
	originCol := cdef.HorizontalBorder
	at := func(row, col int) uint16 {
		return input[(originRow+row)*cdef.BStride+originCol+col]
	}

	// Visible cols 0..32 must hold the real samples (last visible col is 32).
	// Rows past the visible plane within the CDEF VerticalBorder are independently
	// covered by TestFrameWorkCopyCDEFInputBottomEdgeReplicatesLastRow.
	for y := range planeHeight {
		for x := range planeWidth {
			if got, want := at(y, x), uint16(300+x); got != want {
				t.Fatalf("input row=%d col=%d got=%d want=%d", y, x, got, want)
			}
		}
	}
	// Cols past the visible plane within the CDEF HorizontalBorder must replicate
	// the last visible column instead of leaving VeryLarge — otherwise the kernel's
	// directional taps at bx=7 would diverge from libaom's av1_cdef_copy_sb8_16
	// read into out-of-crop frame buffer padding. The bottom-row replication
	// extends the visible rows below the plane into wantSrcY1 first; the column
	// replication then carries the last visible column across both the visible
	// rows and the bottom-extension rows so that the bottom-right corner is also
	// populated with the last visible sample (libaom-equivalent nearest-pixel
	// 2D edge extension for past-visible CDEF input).
	const lastVisible = planeWidth - 1
	wantSrcY1 := unitY + unitH + cdef.VerticalBorder
	for y := range wantSrcY1 {
		for x := planeWidth; x < unitX+unitW+cdef.HorizontalBorder; x++ {
			if got, want := at(y, x), uint16(300+lastVisible); got != want {
				t.Fatalf("right-extension row=%d col=%d got=%d want=%d", y, x, got, want)
			}
		}
	}
}

// TestFrameWorkCopyCDEFInputFinalColumnKeepsVeryLarge pins the libaom branch in
// cdef_prepare_fb that fills the right border with VERY_LARGE when the unit is
// the final superblock column (fbc == nhfb - 1) — i.e., the unit's right edge
// coincides with the plane's right edge. Replicating the last visible column in
// that case would diverge from libaom's frame_boundary[RIGHT] overlay and
// regress vectors whose rightmost CDEF unit has visible content (e.g. SB(1,0)
// of the 66x66 Y plane).
func TestFrameWorkCopyCDEFInputFinalColumnKeepsVeryLarge(t *testing.T) {
	const planeHeight = 66
	const planeWidth = 66
	// Final luma CDEF unit at column 1: unitX=64 unitW=2, ends exactly at plane
	// right edge. This matches the libaom fbc == nhfb - 1 branch.
	const unitX = 64
	const unitY = 0
	const unitW = 2
	const unitH = 64

	src := frame.SamplePlane{
		Pix:    make([]uint16, planeWidth*planeHeight),
		Stride: planeWidth,
		Width:  planeWidth,
		Height: planeHeight,
	}
	for y := range planeHeight {
		for x := range planeWidth {
			src.Pix[y*src.Stride+x] = uint16(400 + x)
		}
	}

	input := make([]uint16, cdef.InputBufferSize)
	for i := range input {
		input[i] = cdef.VeryLarge
	}
	if err := frameWorkCopyCDEFInput(input, src, unitX, unitY, unitW, unitH); err != nil {
		t.Fatalf("frameWorkCopyCDEFInput: %v", err)
	}

	originRow := cdef.VerticalBorder
	originCol := cdef.HorizontalBorder
	at := func(row, col int) uint16 {
		return input[(originRow+row)*cdef.BStride+originCol+col]
	}

	// Right border cols (relative offsets unitW..unitW+HorizontalBorder-1) must
	// remain VeryLarge to match libaom's fill_rect(...VERY_LARGE) branch on the
	// rightmost superblock unit.
	for y := range unitH {
		for x := unitW; x < unitW+cdef.HorizontalBorder; x++ {
			if got := at(y, x); got != cdef.VeryLarge {
				t.Fatalf("final-column right border row=%d col=%d got=%d want VeryLarge", y, x, got)
			}
		}
	}
}

// TestFrameWorkCopyCDEFInputFinalUnitKeepsVeryLarge pins the libaom branch in
// cdef_prepare_fb that fills the bottom border with VERY_LARGE when the unit is
// the final superblock row (fbr == nvfb - 1) — i.e., the unit's bottom edge
// coincides with the plane's bottom edge. Replicating the last visible row in
// that case would diverge from libaom and regress the Y plane on 66x66.
func TestFrameWorkCopyCDEFInputFinalUnitKeepsVeryLarge(t *testing.T) {
	const planeHeight = 66
	const planeWidth = 66
	// Final luma CDEF unit at row 1: unitY=64 unitH=2, ends exactly at plane
	// bottom. This matches the libaom fbr == nvfb - 1 branch.
	const unitX = 0
	const unitY = 64
	const unitW = 64
	const unitH = 2

	src := frame.SamplePlane{
		Pix:    make([]uint16, planeWidth*planeHeight),
		Stride: planeWidth,
		Width:  planeWidth,
		Height: planeHeight,
	}
	for y := range planeHeight {
		for x := range planeWidth {
			src.Pix[y*src.Stride+x] = uint16(200 + y)
		}
	}

	input := make([]uint16, cdef.InputBufferSize)
	for i := range input {
		input[i] = cdef.VeryLarge
	}
	if err := frameWorkCopyCDEFInput(input, src, unitX, unitY, unitW, unitH); err != nil {
		t.Fatalf("frameWorkCopyCDEFInput: %v", err)
	}

	originRow := cdef.VerticalBorder
	originCol := cdef.HorizontalBorder
	at := func(row, col int) uint16 {
		return input[(originRow+row)*cdef.BStride+originCol+col]
	}

	// Bottom border rows (relative offsets unitH..unitH+VerticalBorder-1)
	// must remain VeryLarge to match libaom's fill_rect(...VERY_LARGE) branch.
	for y := unitH; y < unitH+cdef.VerticalBorder; y++ {
		for x := range unitW {
			if got := at(y, x); got != cdef.VeryLarge {
				t.Fatalf("final-unit bottom border row=%d col=%d got=%d want VeryLarge", y, x, got)
			}
		}
	}
}

// TestCDEFPostFilterBandedMatchesWhole proves the unit-row banded apply
// reproduces the whole-frame application exactly: bands over disjoint row
// ranges against one shared sample snapshot (each with its own input and
// unit scratch) write the same output the single-call path does.
func TestCDEFPostFilterBandedMatchesWhole(t *testing.T) {
	const width, height = 256, 192
	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{CodedWidth: width, UpscaledWidth: width, Height: height, SuperResDenominator: 8}
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 2,
			YStrength:     [parser.MaxCDEFStrengths]uint8{9, 31},
			UVStrength:    [parser.MaxCDEFStrengths]uint8{5, 17},
		},
	}
	build := func() (*frame.Frame, FrameWorkPostFilterContext, FrameWorkCDEFPostFilterRequest) {
		output := testFrameWorkCDEFFrame(t, frame.Format{
			Width: width, Height: height, BitDepth: 8,
			SubsamplingX: true, SubsamplingY: true, Align: 64,
		})
		testFillFrameWorkCDEFPlane(output.Y)
		testFillFrameWorkCDEFPlane(output.U)
		testFillFrameWorkCDEFPlane(output.V)
		ctx := FrameWorkPostFilterContext{Event: event, Output: output}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter)
		batch := threading.FrameWorkBatch{
			FrameWorkFrameContext: threading.FrameWorkFrameContext{
				Sequence:  threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader),
				FrameSize: event.FrameSize,
				CDEF:      event.CDEF,
			},
		}
		_, _, length, err := batch.CDEFIndexMapShape()
		if err != nil {
			t.Fatal(err)
		}
		idx := make([]uint8, length)
		for i := range idx {
			idx[i] = uint8(i % 2)
		}
		cdefMap, err := batch.BindCDEFIndexMap(idx, make([]bool, length))
		if err != nil {
			t.Fatal(err)
		}
		for i := range cdefMap.Read {
			cdefMap.Read[i] = true
		}
		scratch, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			t.Fatal(err)
		}
		samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(scratch)
		req, err := scratch.BindRequest(cdefMap, samples, dst, directionGrid, varianceGrid, input, unitDst)
		if err != nil {
			t.Fatal(err)
		}
		return output, ctx, req
	}

	wholeOut, wholeCtx, wholeReq := build()
	if _, err := wholeCtx.ApplyCDEFPostFilter(wholeReq); err != nil {
		t.Fatal(err)
	}

	bandOut, bandCtx, bandReq := build()
	if err := bandCtx.LoadCDEFPostFilterSamples(bandReq); err != nil {
		t.Fatal(err)
	}
	_, rows, err := frameWorkCDEFUnitGrid(event.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	for r0 := 0; r0 < rows; r0 += 1 { // one band per unit row, separate scratch
		band := bandReq
		band.InputScratch = make([]uint16, len(bandReq.InputScratch))
		band.UnitDstScratch = make([]uint16, len(bandReq.UnitDstScratch))
		if _, err := bandCtx.ApplyCDEFPostFilterUnitRows(band, r0, r0+1); err != nil {
			t.Fatalf("band %d: %v", r0, err)
		}
	}
	for _, planes := range [][2]frame.Plane{{wholeOut.Y, bandOut.Y}, {wholeOut.U, bandOut.U}, {wholeOut.V, bandOut.V}} {
		w, b := planes[0], planes[1]
		for y := 0; y < w.Height; y++ {
			for x := 0; x < w.Width; x++ {
				if w.Pix[y*w.Stride+x] != b.Pix[y*b.Stride+x] {
					t.Fatalf("banded differs at (%d,%d)", x, y)
				}
			}
		}
	}
}

func TestCDEFPostFilterBandedU8MatchesSnapshot(t *testing.T) {
	const width, height = 256, 512
	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{CodedWidth: width, UpscaledWidth: width, Height: height, SuperResDenominator: 8}
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 2,
			YStrength:     [parser.MaxCDEFStrengths]uint8{9, 31},
			UVStrength:    [parser.MaxCDEFStrengths]uint8{5, 17},
		},
	}
	build := func() (*frame.Frame, FrameWorkPostFilterContext, FrameWorkCDEFPostFilterRequest) {
		output := testFrameWorkCDEFFrame(t, frame.Format{
			Width: width, Height: height, BitDepth: 8,
			SubsamplingX: true, SubsamplingY: true, Align: 64,
		})
		testFillFrameWorkCDEFPlane(output.Y)
		testFillFrameWorkCDEFPlane(output.U)
		testFillFrameWorkCDEFPlane(output.V)
		ctx := FrameWorkPostFilterContext{Event: event, Output: output}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter)
		batch := threading.FrameWorkBatch{
			FrameWorkFrameContext: threading.FrameWorkFrameContext{
				Sequence:  threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader),
				FrameSize: event.FrameSize,
				CDEF:      event.CDEF,
			},
		}
		_, _, length, err := batch.CDEFIndexMapShape()
		if err != nil {
			t.Fatal(err)
		}
		idx := make([]uint8, length)
		for i := range idx {
			idx[i] = uint8(i % 2)
		}
		cdefMap, err := batch.BindCDEFIndexMap(idx, make([]bool, length))
		if err != nil {
			t.Fatal(err)
		}
		for i := range cdefMap.Read {
			cdefMap.Read[i] = true
		}
		scratch, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			t.Fatal(err)
		}
		samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(scratch)
		req, err := scratch.BindRequest(cdefMap, samples, dst, directionGrid, varianceGrid, input, unitDst)
		if err != nil {
			t.Fatal(err)
		}
		return output, ctx, req
	}

	snapshotOut, snapshotCtx, snapshotReq := build()
	if err := snapshotCtx.LoadCDEFPostFilterSamples(snapshotReq); err != nil {
		t.Fatal(err)
	}
	_, rows, err := frameWorkCDEFUnitGrid(event.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	for r0 := 0; r0 < rows; r0++ {
		band := snapshotReq
		band.InputScratch = make([]uint16, len(snapshotReq.InputScratch))
		band.UnitDstScratch = make([]uint16, len(snapshotReq.UnitDstScratch))
		if _, err := snapshotCtx.ApplyCDEFPostFilterUnitRows(band, r0, r0+1); err != nil {
			t.Fatalf("snapshot band %d: %v", r0, err)
		}
	}

	u8Out, u8Ctx, u8Req := build()
	boundaries := make([]FrameWorkCDEFPostFilterU8BandBoundary, rows)
	for r0 := 0; r0 < rows; r0++ {
		boundaries[r0] = testCDEFU8BandBoundary(width)
		if err := u8Ctx.LoadCDEFPostFilterU8BandBoundary(&boundaries[r0], r0, r0+1); err != nil {
			t.Fatalf("load u8 boundary %d: %v", r0, err)
		}
	}
	for r0 := 0; r0 < rows; r0++ {
		band := u8Req
		band.SampleScratch = testCDEFU8LineScratch(width)
		band.InputScratch = make([]uint16, len(u8Req.InputScratch))
		if _, err := u8Ctx.ApplyCDEFPostFilterUnitRowsU8(band, boundaries[r0], r0, r0+1); err != nil {
			t.Fatalf("u8 band %d: %v", r0, err)
		}
	}
	for _, planes := range [][2]frame.Plane{{snapshotOut.Y, u8Out.Y}, {snapshotOut.U, u8Out.U}, {snapshotOut.V, u8Out.V}} {
		w, b := planes[0], planes[1]
		for y := 0; y < w.Height; y++ {
			for x := 0; x < w.Width; x++ {
				if w.Pix[y*w.Stride+x] != b.Pix[y*b.Stride+x] {
					t.Fatalf("u8 banded differs at (%d,%d)", x, y)
				}
			}
		}
	}
}

func testCDEFU8BandBoundary(width int) FrameWorkCDEFPostFilterU8BandBoundary {
	var boundary FrameWorkCDEFPostFilterU8BandBoundary
	widths := [3]int{width, width / 2, width / 2}
	for plane, planeWidth := range widths {
		need := cdef.VerticalBorder * planeWidth
		boundary.Top[plane] = make([]byte, need)
		boundary.Bottom[plane] = make([]byte, need)
	}
	return boundary
}

func testCDEFU8LineScratch(width int) [3][]uint16 {
	return [3][]uint16{
		make([]uint16, 2*width),
		make([]uint16, width),
		make([]uint16, width),
	}
}

// TestCDEFPostFilterPooledMatchesSerial proves the pooled row-band CDEF apply
// (applyCDEFPostFilterMaybePooled with a multi-lane pool) is byte-identical to
// the serial whole-frame apply, for several worker counts. The frame is tall
// enough (8 CDEF unit rows) that the min-rows-per-band gate lets the pooled path
// engage. This is the worker-count-invariance guarantee the postfilter pool
// rests on.
func TestCDEFPostFilterPooledMatchesSerial(t *testing.T) {
	const width, height = 256, 512 // 4x8 CDEF unit grid
	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{CodedWidth: width, UpscaledWidth: width, Height: height, SuperResDenominator: 8}
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 2,
			YStrength:     [parser.MaxCDEFStrengths]uint8{9, 31},
			UVStrength:    [parser.MaxCDEFStrengths]uint8{5, 17},
		},
	}
	build := func(pool *threading.Pool) (*frame.Frame, FrameWorkPostFilterContext, FrameWorkCDEFPostFilterRequest) {
		output := testFrameWorkCDEFFrame(t, frame.Format{
			Width: width, Height: height, BitDepth: 8,
			SubsamplingX: true, SubsamplingY: true, Align: 64,
		})
		testFillFrameWorkCDEFPlane(output.Y)
		testFillFrameWorkCDEFPlane(output.U)
		testFillFrameWorkCDEFPlane(output.V)
		ctx := FrameWorkPostFilterContext{Event: event, Output: output, pool: pool}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter)
		batch := threading.FrameWorkBatch{
			FrameWorkFrameContext: threading.FrameWorkFrameContext{
				Sequence:  threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader),
				FrameSize: event.FrameSize,
				CDEF:      event.CDEF,
			},
		}
		_, _, length, err := batch.CDEFIndexMapShape()
		if err != nil {
			t.Fatal(err)
		}
		idx := make([]uint8, length)
		for i := range idx {
			idx[i] = uint8(i % 2)
		}
		cdefMap, err := batch.BindCDEFIndexMap(idx, make([]bool, length))
		if err != nil {
			t.Fatal(err)
		}
		for i := range cdefMap.Read {
			cdefMap.Read[i] = true
		}
		scratch, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			t.Fatal(err)
		}
		samples, dst, dg, vg, input, unitDst := testFrameWorkCDEFScratchStorage(scratch)
		req, err := scratch.BindRequest(cdefMap, samples, dst, dg, vg, input, unitDst)
		if err != nil {
			t.Fatal(err)
		}
		return output, ctx, req
	}

	serialPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer serialPool.Close()
	serialOut, serialCtx, serialReq := build(serialPool)
	if _, err := serialCtx.applyCDEFPostFilterMaybePooled(serialReq); err != nil {
		t.Fatal(err)
	}

	for _, workers := range []int{2, 3, 4, 8} {
		pool, err := threading.NewPool(workers)
		if err != nil {
			t.Fatalf("workers=%d: %v", workers, err)
		}
		pooledOut, pooledCtx, pooledReq := build(pool)
		if _, err := pooledCtx.applyCDEFPostFilterMaybePooled(pooledReq); err != nil {
			pool.Close()
			t.Fatalf("workers=%d apply: %v", workers, err)
		}
		for _, planes := range [][2]frame.Plane{{serialOut.Y, pooledOut.Y}, {serialOut.U, pooledOut.U}, {serialOut.V, pooledOut.V}} {
			s, p := planes[0], planes[1]
			for y := 0; y < s.Height; y++ {
				for x := 0; x < s.Width; x++ {
					if s.Pix[y*s.Stride+x] != p.Pix[y*p.Stride+x] {
						pool.Close()
						t.Fatalf("workers=%d pooled differs at (%d,%d): serial=%d pooled=%d", workers, x, y, s.Pix[y*s.Stride+x], p.Pix[y*p.Stride+x])
					}
				}
			}
		}
		pool.Close()
	}
}
