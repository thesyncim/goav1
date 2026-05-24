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

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRejectsUnsupportedBeforeMutation(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 32})
	output.Y.Pix[0] = 0x44
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

func testFrameWorkCDEFFrame(t *testing.T, format frame.Format) *frame.Frame {
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

func testFrameWorkCDEFPostFilterRequest(t *testing.T, ctx FrameWorkPostFilterContext, event Event) FrameWorkCDEFPostFilterRequest {
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
	req := FrameWorkCDEFPostFilterRequest{
		IndexMap:       cdefMap,
		DirectionGrid:  make([]cdef.DirectionGrid, size.DirectionGrid),
		VarianceGrid:   make([]cdef.VarianceGrid, size.VarianceGrid),
		InputScratch:   make([]uint16, size.Input),
		UnitDstScratch: make([]uint16, size.UnitDst),
	}
	for plane := 0; plane < 3; plane++ {
		req.SampleScratch[plane] = make([]uint16, size.Samples[plane])
		req.DstScratch[plane] = make([]uint16, size.Dst[plane])
	}
	return req
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
