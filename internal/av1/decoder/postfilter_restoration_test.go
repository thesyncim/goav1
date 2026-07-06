package decoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestFrameWorkPostFilterContextApplyLoopRestorationPostFilterAllNoneRecords(t *testing.T) {
	const width = 64
	const height = 64

	pool := testFramePoolForSize(t, width, height, 1)
	_, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	output.Y.Pix[0] = 0x5a

	event := Event{
		SequenceHeader: testSequence(),
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		Restoration: parser.RestorationParams{
			Type:      [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
			UnitSizeY: 64,
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	if got := ctx.ActivePostFilters(); got != FrameWorkPostFilterLoopRestoration {
		t.Fatalf("ActivePostFilters=%b want %b", got, FrameWorkPostFilterLoopRestoration)
	}

	plan, err := ctx.LoopRestorationPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.UnitRecordLen() != 1 {
		t.Fatalf("plan active=%v records=%d", plan.Active, plan.UnitRecordLen())
	}
	records, err := tile.BindRestorationFrameRecordBuffers(plan, make([]tile.RestorationUnitRecord, plan.UnitRecordLen()))
	if err != nil {
		t.Fatal(err)
	}
	for plane := 0; plane < int(plan.Planes); plane++ {
		if plan.Grids[plane].Type == parser.RestorationNone {
			continue
		}
		if err := tile.ResetRestorationPlaneRecords(plan.Grids[plane], records[plane]); err != nil {
			t.Fatal(err)
		}
	}

	size, err := ctx.LoopRestorationPostFilterScratchLen(records, false)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ctx.ApplyLoopRestorationPostFilter(FrameWorkRestorationPostFilterRequest{
		Records:     records,
		DataScratch: make([]uint16, size.Samples.DataLen),
		DstScratch:  make([]uint16, size.Samples.DstLen),
		Scratch:     testFrameWorkRestorationPostFilterScratch(size.Apply),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 1 || result.Records != 1 || result.FilteredRecords != 0 {
		t.Fatalf("result=%+v", result)
	}
	if output.Y.Pix[0] != 0x5a {
		t.Fatalf("output sample=%d want 0x5a", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplyLoopRestorationPostFilterDefaultsBuffersFromContext(t *testing.T) {
	const width = 64
	const height = 64

	pool := testFramePoolForSize(t, width, height, 1)
	_, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	output.Y.Pix[0] = 0x6b

	event := Event{
		SequenceHeader: testSequence(),
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		Restoration: parser.RestorationParams{
			Type:      [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
			UnitSizeY: 64,
		},
	}
	batch := threading.FrameWorkBatch{
		FrameWorkFrameContext: threading.FrameWorkFrameContext{
			Sequence:    threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader),
			FrameSize:   event.FrameSize,
			Restoration: event.Restoration,
		},
	}
	plan, err := batch.RestorationFramePlan()
	if err != nil {
		t.Fatal(err)
	}
	buffers, err := batch.BindRestorationFrameBuffers(
		make([]tile.RestorationUnitRecord, plan.UnitRecordLen()),
		make([]uint16, plan.BoundaryBufferLen()),
		make([]uint16, plan.BoundaryBufferLen()),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := buffers.ResetRecords(); err != nil {
		t.Fatal(err)
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, RestorationFrameBuffers: &buffers}
	pipelineSize, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if pipelineSize.Restoration.Samples.DataLen == 0 || pipelineSize.Restoration.Samples.DstLen != 0 {
		t.Fatalf("pipeline scratch=%+v", pipelineSize.Restoration)
	}
	size, err := ctx.LoopRestorationPostFilterScratchLen(buffers.Records, false)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ctx.ApplyLoopRestorationPostFilter(FrameWorkRestorationPostFilterRequest{
		DataScratch: make([]uint16, size.Samples.DataLen),
		DstScratch:  make([]uint16, size.Samples.DstLen),
		Scratch:     testFrameWorkRestorationPostFilterScratch(size.Apply),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 1 || result.Records != 1 || result.FilteredRecords != 0 {
		t.Fatalf("result=%+v", result)
	}
	if output.Y.Pix[0] != 0x6b {
		t.Fatalf("output sample=%d want 0x6b", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextSaveRestorationBoundariesSkipsOptimized(t *testing.T) {
	const width = 64
	const height = 64

	pool := testFramePoolForSize(t, width, height, 1)
	_, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	event := Event{
		SequenceHeader: testSequence(),
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		Restoration: parser.RestorationParams{
			Type:      [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
			UnitSizeY: 64,
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := FrameWorkRestorationPostFilterRequest{
		Boundaries: [3]tile.RestorationStripeBoundaries{
			{Above: []uint16{1}},
		},
	}
	if err := ctx.saveRestorationBoundariesForRequest(req, false); !errors.Is(err, tile.ErrInvalidPlan) {
		t.Fatalf("saveRestorationBoundariesForRequest err=%v want %v", err, tile.ErrInvalidPlan)
	}
	req.Optimized = true
	if err := ctx.saveRestorationBoundariesForRequest(req, false); err != nil {
		t.Fatalf("optimized saveRestorationBoundariesForRequest err=%v", err)
	}
	if err := ctx.saveRestorationBoundariesForRequest(req, true); err != nil {
		t.Fatalf("optimized after-CDEF saveRestorationBoundariesForRequest err=%v", err)
	}
}

func TestFrameWorkPostFilterContextApplyLoopRestorationPostFilterRejectsEarlierActiveStages(t *testing.T) {
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FrameSize: parser.FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              64,
				SuperResDenominator: 8,
			},
			CDEF: parser.CDEFParams{
				StrengthCount: 1,
				YStrength:     [parser.MaxCDEFStrengths]uint8{1},
			},
			Restoration: parser.RestorationParams{
				Type:      [3]parser.RestorationType{parser.RestorationWiener},
				UnitSizeY: 64,
			},
		},
	}
	_, err := ctx.ApplyLoopRestorationPostFilter(FrameWorkRestorationPostFilterRequest{})
	if !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("ApplyLoopRestorationPostFilter err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func TestFrameWorkPostFilterContextApplyLoopRestorationPostFilterAllowsCompletedEarlierStages(t *testing.T) {
	const width = 64
	const height = 64

	pool := testFramePoolForSize(t, width, height, 1)
	_, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	output.Y.Pix[0] = 0x33

	event := Event{
		SequenceHeader: testSequence(),
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResEnabled:     true,
			SuperResDenominator: 16,
		},
		LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{1}},
		CDEF: parser.CDEFParams{
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{1},
		},
		Restoration: parser.RestorationParams{
			Type:      [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
			UnitSizeY: 64,
		},
		FilmGrain: parser.FilmGrainParams{Apply: true},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}.WithCompletedPostFilters(
		FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF | FrameWorkPostFilterSuperRes,
	)
	if got := ctx.RemainingPostFilters(); got != FrameWorkPostFilterLoopRestoration|FrameWorkPostFilterFilmGrain {
		t.Fatalf("remaining=%b", got)
	}
	plan, err := ctx.LoopRestorationPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	records, err := tile.BindRestorationFrameRecordBuffers(plan, make([]tile.RestorationUnitRecord, plan.UnitRecordLen()))
	if err != nil {
		t.Fatal(err)
	}
	if err := tile.ResetRestorationPlaneRecords(plan.Grids[0], records[0]); err != nil {
		t.Fatal(err)
	}
	size, err := ctx.LoopRestorationPostFilterScratchLen(records, false)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ctx.ApplyLoopRestorationPostFilter(FrameWorkRestorationPostFilterRequest{
		Records:     records,
		DataScratch: make([]uint16, size.Samples.DataLen),
		DstScratch:  make([]uint16, size.Samples.DstLen),
		Scratch:     testFrameWorkRestorationPostFilterScratch(size.Apply),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 1 || result.Records != 1 || result.FilteredRecords != 0 {
		t.Fatalf("result=%+v", result)
	}
	if output.Y.Pix[0] != 0x33 {
		t.Fatalf("output sample=%d want 0x33", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplyCallerPostFiltersSuperResRestorationAllocs(t *testing.T) {
	const codedWidth = 32
	const upscaledWidth = 48
	const height = 64

	seq := testSequence()
	size := parser.FrameSize{
		CodedWidth:          codedWidth,
		UpscaledWidth:       upscaledWidth,
		Height:              height,
		SuperResEnabled:     true,
		SuperResDenominator: 16,
	}
	restoration := parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
		UnitSizeY: 64,
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{
		Width:        codedWidth,
		Height:       height,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	})
	for i := range output.Y.Pix {
		output.Y.Pix[i] = byte(32 + i%96)
	}
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		Restoration:    restoration,
	}
	batch := threading.FrameWorkBatch{
		FrameWorkFrameContext: threading.FrameWorkFrameContext{
			Sequence:    threading.FrameWorkSequenceContextFromHeader(seq),
			FrameSize:   size,
			Restoration: restoration,
		},
	}
	plan, err := batch.RestorationFramePlan()
	if err != nil {
		t.Fatal(err)
	}
	buffers, err := batch.BindRestorationFrameBuffers(
		make([]tile.RestorationUnitRecord, plan.UnitRecordLen()),
		make([]uint16, plan.BoundaryBufferLen()),
		make([]uint16, plan.BoundaryBufferLen()),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := buffers.ResetRecords(); err != nil {
		t.Fatal(err)
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, RestorationFrameBuffers: &buffers}
	var outputView frame.Frame
	options := FrameWorkPostFilterBindOptions{
		SuperResOutputView:    &outputView,
		RestorationRecords:    buffers.Records,
		RestorationBoundaries: buffers.Boundaries,
	}
	first, err := ctx.CallerPostFilterScratchLen(FrameWorkPostFilterRequest{
		Restoration: FrameWorkRestorationPostFilterRequest{Records: buffers.Records},
	})
	if err != nil {
		t.Fatal(err)
	}
	if first.SuperRes.OutputFrame == 0 || first.Restoration.Samples.DataLen != 0 {
		t.Fatalf("bootstrap scratch=%+v", first)
	}
	firstScratch := testFrameWorkPostFilterScratchStorage(first)
	probe, err := first.BindRequest(options, firstScratch)
	if err != nil {
		t.Fatal(err)
	}
	full, err := ctx.CallerPostFilterScratchLen(probe)
	if err != nil {
		t.Fatal(err)
	}
	if full.Restoration.Samples.DataLen < output.Y.Width {
		t.Fatalf("restoration scratch too small for superres boundary row: %+v", full.Restoration)
	}
	scratch := testFrameWorkPostFilterScratchStorage(full)
	req, err := full.BindRequest(options, scratch)
	if err != nil {
		t.Fatal(err)
	}

	next, result, err := ctx.ApplyCallerPostFilters(req)
	if err != nil {
		t.Fatal(err)
	}
	wantCompleted := FrameWorkPostFilterSuperRes | FrameWorkPostFilterLoopRestoration
	if result.Completed != wantCompleted ||
		result.SuperRes.Output.Format.Width != upscaledWidth ||
		result.Restoration.Records != 1 ||
		next.RemainingPostFilters() != 0 ||
		!next.DetachedPostFilterOutput() {
		t.Fatalf("next remaining=%b detached=%v result=%+v", next.RemainingPostFilters(), next.DetachedPostFilterOutput(), result)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if _, _, err := ctx.ApplyCallerPostFilters(req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("superres restoration caller postfilter allocated: %f", allocs)
	}
}

func TestFrameWorkRestorationPostFilterScratchSizeBindRequest(t *testing.T) {
	size := FrameWorkRestorationPostFilterScratchSize{
		Samples: tile.RestorationFrameSampleScratchSize{
			DataLen: 16,
			DstLen:  12,
		},
		Apply: tile.RestorationUnitRecordBoundaryScratchSize{
			Unit:     tile.RestorationUnitScratchSize{Wiener: 8, SGRProj: 6},
			Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 4, Below: 2},
		},
	}
	records := [3][]tile.RestorationUnitRecord{{{Index: 1}}}
	boundaries := [3]tile.RestorationStripeBoundaries{{Stride: 16}}
	data, dst, wiener, sgr, above, below := testFrameWorkRestorationPostFilterScratchStorage(size)
	req, err := size.BindRequest(records, boundaries, data, dst, wiener, sgr, above, below, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.DataScratch) != size.Samples.DataLen || len(req.DstScratch) != size.Samples.DstLen ||
		len(req.Scratch.Unit.Wiener) != size.Apply.Unit.Wiener ||
		len(req.Scratch.Unit.SGRProj) != size.Apply.Unit.SGRProj ||
		len(req.Scratch.Boundary.Above) != size.Apply.Boundary.Above ||
		len(req.Scratch.Boundary.Below) != size.Apply.Boundary.Below ||
		!req.Optimized || len(req.Records[0]) != 1 || req.Boundaries[0].Stride != boundaries[0].Stride {
		t.Fatalf("request data=%d dst=%d wiener=%d sgr=%d above=%d below=%d optimized=%v records=%d boundary=%d",
			len(req.DataScratch), len(req.DstScratch), len(req.Scratch.Unit.Wiener), len(req.Scratch.Unit.SGRProj),
			len(req.Scratch.Boundary.Above), len(req.Scratch.Boundary.Below), req.Optimized, len(req.Records[0]), req.Boundaries[0].Stride)
	}
}

func TestFrameWorkRestorationPostFilterScratchSizeBindRequestRejectsShortBuffers(t *testing.T) {
	size := FrameWorkRestorationPostFilterScratchSize{
		Samples: tile.RestorationFrameSampleScratchSize{
			DataLen: 16,
			DstLen:  12,
		},
		Apply: tile.RestorationUnitRecordBoundaryScratchSize{
			Unit:     tile.RestorationUnitScratchSize{Wiener: 8, SGRProj: 6},
			Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 4, Below: 2},
		},
	}
	data, dst, wiener, sgr, above, below := testFrameWorkRestorationPostFilterScratchStorage(size)
	tests := []struct {
		name   string
		size   FrameWorkRestorationPostFilterScratchSize
		data   []uint16
		dst    []uint16
		wiener []uint16
		sgr    []int32
		above  []uint16
		below  []uint16
		want   error
	}{
		{name: "data", size: size, data: data[:15], dst: dst, wiener: wiener, sgr: sgr, above: above, below: below, want: tile.ErrJobBufferTooSmall},
		{name: "dst", size: size, data: data, dst: dst[:11], wiener: wiener, sgr: sgr, above: above, below: below, want: tile.ErrJobBufferTooSmall},
		{name: "wiener", size: size, data: data, dst: dst, wiener: wiener[:7], sgr: sgr, above: above, below: below, want: tile.ErrInvalidPlan},
		{name: "sgr", size: size, data: data, dst: dst, wiener: wiener, sgr: sgr[:5], above: above, below: below, want: tile.ErrInvalidPlan},
		{name: "above", size: size, data: data, dst: dst, wiener: wiener, sgr: sgr, above: above[:3], below: below, want: tile.ErrInvalidPlan},
		{name: "below", size: size, data: data, dst: dst, wiener: wiener, sgr: sgr, above: above, below: below[:1], want: tile.ErrInvalidPlan},
		{name: "negative", size: FrameWorkRestorationPostFilterScratchSize{Samples: tile.RestorationFrameSampleScratchSize{DataLen: -1}}, data: data, dst: dst, wiener: wiener, sgr: sgr, above: above, below: below, want: tile.ErrInvalidPlan},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.size.BindRequest([3][]tile.RestorationUnitRecord{}, [3]tile.RestorationStripeBoundaries{}, tt.data, tt.dst, tt.wiener, tt.sgr, tt.above, tt.below, false)
			if !errors.Is(err, tt.want) {
				t.Fatalf("BindRequest err=%v want %v", err, tt.want)
			}
		})
	}
}

func TestFrameWorkRestorationPostFilterScratchSizeBindRequestAllocs(t *testing.T) {
	size := FrameWorkRestorationPostFilterScratchSize{
		Samples: tile.RestorationFrameSampleScratchSize{
			DataLen: 1920 * 1080,
			DstLen:  1920 * 1080,
		},
		Apply: tile.RestorationUnitRecordBoundaryScratchSize{
			Unit:     tile.RestorationUnitScratchSize{Wiener: 4096, SGRProj: 4096},
			Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 1920, Below: 1920},
		},
	}
	data, dst, wiener, sgr, above, below := testFrameWorkRestorationPostFilterScratchStorage(size)
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := size.BindRequest([3][]tile.RestorationUnitRecord{}, [3]tile.RestorationStripeBoundaries{}, data, dst, wiener, sgr, above, below, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("BindRequest allocated: %f", allocs)
	}
}

func BenchmarkFrameWorkRestorationPostFilterScratchSizeBindRequest(b *testing.B) {
	size := FrameWorkRestorationPostFilterScratchSize{
		Samples: tile.RestorationFrameSampleScratchSize{
			DataLen: 1920 * 1080,
			DstLen:  1920 * 1080,
		},
		Apply: tile.RestorationUnitRecordBoundaryScratchSize{
			Unit:     tile.RestorationUnitScratchSize{Wiener: 4096, SGRProj: 4096},
			Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 1920, Below: 1920},
		},
	}
	data, dst, wiener, sgr, above, below := testFrameWorkRestorationPostFilterScratchStorage(size)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := size.BindRequest([3][]tile.RestorationUnitRecord{}, [3]tile.RestorationStripeBoundaries{}, data, dst, wiener, sgr, above, below, false); err != nil {
			b.Fatal(err)
		}
	}
}

func testFrameWorkRestorationPostFilterScratch(size tile.RestorationUnitRecordBoundaryScratchSize) tile.RestorationUnitRecordBoundaryScratch {
	return tile.RestorationUnitRecordBoundaryScratch{
		Unit: tile.RestorationUnitScratch{
			Wiener:  make([]uint16, size.Unit.Wiener),
			SGRProj: make([]int32, size.Unit.SGRProj),
		},
		Boundary: tile.RestorationStripeBoundaryScratch{
			Above: make([]uint16, size.Boundary.Above),
			Below: make([]uint16, size.Boundary.Below),
		},
	}
}

func testFrameWorkRestorationPostFilterScratchStorage(size FrameWorkRestorationPostFilterScratchSize) ([]uint16, []uint16, []uint16, []int32, []uint16, []uint16) {
	return make([]uint16, maxInt(size.Samples.DataLen, 0)),
		make([]uint16, maxInt(size.Samples.DstLen, 0)),
		make([]uint16, maxInt(size.Apply.Unit.Wiener, 0)),
		make([]int32, maxInt(size.Apply.Unit.SGRProj, 0)),
		make([]uint16, maxInt(size.Apply.Boundary.Above, 0)),
		make([]uint16, maxInt(size.Apply.Boundary.Below, 0))
}
