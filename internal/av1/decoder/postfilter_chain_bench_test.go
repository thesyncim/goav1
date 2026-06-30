package decoder

// End-to-end post-filter chain benchmarks. These exercise each frame-level
// post-filter stage (loop filter, CDEF, loop restoration) in isolation and as
// a chain so users can see where decode time is spent in the post-filter
// pipeline.
//
// The benchmarks all share a single synthetic 128x128 8-bit 4:2:0 frame
// pre-populated with a deterministic pattern, plus pre-built side data
// (loop-filter block map, CDEF index map, restoration records). Each
// iteration refills the frame so the per-iteration cost reflects a clean
// decode-then-filter sweep rather than amplified filtering of already
// idempotent data.
//
// All scratch buffers are allocated once during fixture build; the
// per-iteration hot path is expected to stay at zero allocations.
//
// To make the numbers easy to consume the benchmarks report ms/frame and
// MB/sec via b.ReportMetric and b.SetBytes. The MB/sec figure uses the
// frame's pixel byte count (Y+U+V) as the throughput denominator.

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

const (
	// postFilterBenchWidth and postFilterBenchHeight pick the largest
	// luma size that fits in a single super-block mode context (32 MI
	// slots = 128 px) while still being a clean multiple of every
	// per-stage unit (8 px deblock, 64 px CDEF, 64 px LR unit). The
	// resulting 4:2:0 frame fits well inside L1 cache so the benchmark
	// numbers are stable run-to-run.
	postFilterBenchWidth  = 128
	postFilterBenchHeight = 128
)

// postFilterChainFixture bundles everything an Apply*PostFilter benchmark
// needs: a frame populated with a deterministic pattern, the post-filter
// context, and pre-built per-stage request scratch.
type postFilterChainFixture struct {
	ctx        FrameWorkPostFilterContext
	output     *frame.Frame
	loopFilter FrameWorkLoopFilterPostFilterRequest
	cdef       FrameWorkCDEFPostFilterRequest
	rest       FrameWorkRestorationPostFilterRequest
	pixelBytes int
}

// frameBytes is the raw decoded pixel byte count (Y+U+V). It is used as the
// b.SetBytes denominator so `go test -bench` automatically surfaces a
// MB/sec column comparable to the end-to-end BenchmarkDecodeFullVector
// number.
func (f postFilterChainFixture) frameBytes() int64 { return int64(f.pixelBytes) }

// refill restores the frame samples to the deterministic pre-filter pattern
// so each benchmark iteration measures the same amount of work.
func (f postFilterChainFixture) refill() {
	testFillFrameWorkCDEFPlane(f.output.Y)
	testFillFrameWorkCDEFPlane(f.output.U)
	testFillFrameWorkCDEFPlane(f.output.V)
}

// reportPostFilterMetrics adds the human-friendly ms/frame metric used in
// the README post-filter performance table.
func reportPostFilterMetrics(b *testing.B) {
	b.Helper()
	if b.N == 0 {
		return
	}
	nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
	b.ReportMetric(nsPerOp/1e6, "ms/frame")
}

// buildPostFilterChainFixture wires up the synthetic frame and side data
// shared by every Apply*PostFilter benchmark in this file.
//
// All stages are configured with non-trivial filter strengths so the
// benchmark exercises real work in every kernel:
//   - loop filter levels are clamped to 63 (max) so every edge is filtered
//   - CDEF primary strength is 63 (max luma) so every 64x64 unit triggers
//   - loop restoration uses Wiener on all three planes with non-identity
//     filter coefficients so every unit invokes the Wiener kernel
//
// The shared synthetic pattern produces enough sample variance that the
// per-block CDEF direction search and Wiener filter cannot short-circuit.
func buildPostFilterChainFixture(b testing.TB) postFilterChainFixture {
	b.Helper()
	const width = postFilterBenchWidth
	const height = postFilterBenchHeight

	seq := testSequence()
	seq.EnableCDEF = true
	seq.EnableRestoration = true

	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       width,
		Height:              height,
		SuperResDenominator: 8,
	}
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		LoopFilter: parser.LoopFilterParams{
			LevelY:    [2]uint8{63, 63},
			LevelU:    63,
			LevelV:    63,
			Sharpness: 1,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
			UVStrength:    [parser.MaxCDEFStrengths]uint8{63},
		},
		Restoration: parser.RestorationParams{
			Type: [3]parser.RestorationType{
				parser.RestorationWiener,
				parser.RestorationWiener,
				parser.RestorationWiener,
			},
			UnitSizeY:  64,
			UnitSizeUV: 64,
		},
	}

	output := testFrameWorkCDEFFrame(b, frame.Format{
		Width:        width,
		Height:       height,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	})
	testFillFrameWorkCDEFPlane(output.Y)
	testFillFrameWorkCDEFPlane(output.U)
	testFillFrameWorkCDEFPlane(output.V)

	// Loop-filter map: a full grid of 16x16 records covering the frame so
	// every internal block edge is in scope for ApplyLoopFilterEdges.
	records := buildPostFilterChainLoopFilterRecords(width, height)
	filterMap := testFrameWorkLoopFilterPostFilterMap(b, size, records...)

	ctx := FrameWorkPostFilterContext{
		Event:         event,
		Output:        output,
		LoopFilterMap: &filterMap,
	}

	// Loop-filter edge scratch sized to the worst case for this filter map.
	lfSize, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		b.Fatal(err)
	}
	edges, err := lfSize.BindEdges(make([]FrameWorkLoopFilterPostFilterEdge, lfSize.Edges))
	if err != nil {
		b.Fatal(err)
	}

	// CDEF request: a fully-readable index map so every 64x64 unit runs
	// the filter kernel.
	cdefReq := buildPostFilterChainCDEFRequest(b, ctx, event)

	// Restoration request: Wiener on all three planes with a non-identity
	// filter so ApplyLoopRestorationPostFilter exercises the full kernel.
	restReq := buildPostFilterChainRestorationRequest(b, ctx)

	pixelBytes := output.Y.Width*output.Y.Height + output.U.Width*output.U.Height + output.V.Width*output.V.Height

	return postFilterChainFixture{
		ctx:        ctx,
		output:     output,
		loopFilter: FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges},
		cdef:       cdefReq,
		rest:       restReq,
		pixelBytes: pixelBytes,
	}
}

// buildPostFilterChainLoopFilterRecords returns one 16x16-block record for
// every aligned tile position inside the frame so the loop-filter map is
// fully populated.
func buildPostFilterChainLoopFilterRecords(width int, height int) []threading.FrameWorkLoopFilterBlockRecord {
	// 16x16 block -> 4 MI units per dimension.
	const blockMI = 4
	cols := width / 4
	rows := height / 4
	records := make([]threading.FrameWorkLoopFilterBlockRecord, 0, (cols/blockMI)*(rows/blockMI))
	for row := 0; row < rows; row += blockMI {
		for col := 0; col < cols; col += blockMI {
			records = append(records, testFrameWorkLoopFilterPostFilterRecordAt(col, row, col+blockMI, row+blockMI))
		}
	}
	return records
}

// buildPostFilterChainCDEFRequest sizes scratch from the context, marks every
// 64x64 unit as readable, and returns a fully-bound CDEF request ready for
// repeated ApplyCDEFPostFilter calls.
func buildPostFilterChainCDEFRequest(b testing.TB, ctx FrameWorkPostFilterContext, event Event) FrameWorkCDEFPostFilterRequest {
	b.Helper()
	batch := threading.FrameWorkBatch{
		FrameWorkFrameContext: threading.FrameWorkFrameContext{
			Sequence:  threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader),
			FrameSize: event.FrameSize,
			CDEF:      event.CDEF,
		},
	}
	_, _, length, err := batch.CDEFIndexMapShape()
	if err != nil {
		b.Fatal(err)
	}
	cdefMap, err := batch.BindCDEFIndexMap(make([]uint8, length), make([]bool, length))
	if err != nil {
		b.Fatal(err)
	}
	for i := range cdefMap.Read {
		cdefMap.Read[i] = true
	}
	size, err := ctx.CDEFPostFilterScratchLen()
	if err != nil {
		b.Fatal(err)
	}
	samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(size)
	req, err := size.BindRequest(cdefMap, samples, dst, directionGrid, varianceGrid, input, unitDst)
	if err != nil {
		b.Fatal(err)
	}
	return req
}

// buildPostFilterChainRestorationRequest pre-builds the records / boundary
// buffers / scratch for the Wiener restoration pass so ApplyLoopRestoration
// can run with zero per-iteration allocations.
func buildPostFilterChainRestorationRequest(b testing.TB, ctx FrameWorkPostFilterContext) FrameWorkRestorationPostFilterRequest {
	b.Helper()
	batch := threading.FrameWorkBatch{
		FrameWorkFrameContext: threading.FrameWorkFrameContext{
			Sequence:    threading.FrameWorkSequenceContextFromHeader(ctx.Event.SequenceHeader),
			FrameSize:   ctx.Event.FrameSize,
			Restoration: ctx.Event.Restoration,
		},
	}
	plan, err := batch.RestorationFramePlan()
	if err != nil {
		b.Fatal(err)
	}
	buffers, err := batch.BindRestorationFrameBuffers(
		make([]tile.RestorationUnitRecord, plan.UnitRecordLen()),
		make([]uint16, plan.BoundaryBufferLen()),
		make([]uint16, plan.BoundaryBufferLen()),
	)
	if err != nil {
		b.Fatal(err)
	}
	if err := buffers.ResetRecords(); err != nil {
		b.Fatal(err)
	}
	// Default records carry Type=RestorationNone which short-circuits the
	// Wiener kernel. Override each unit with the default Wiener filter so
	// every record actually invokes the per-pixel filter routine.
	defaultWiener := av1restoration.DefaultWienerInfo()
	for plane := 0; plane < int(plan.Planes); plane++ {
		if plan.Grids[plane].Type != parser.RestorationWiener {
			continue
		}
		records := buffers.Records[plane]
		for i := range records {
			records[i].Unit.Type = parser.RestorationWiener
			records[i].Unit.Wiener = defaultWiener
		}
	}
	size, err := ctx.LoopRestorationPostFilterScratchLen(buffers.Records, false)
	if err != nil {
		b.Fatal(err)
	}
	return FrameWorkRestorationPostFilterRequest{
		Records:     buffers.Records,
		Boundaries:  buffers.Boundaries,
		DataScratch: make([]uint16, size.Samples.DataLen),
		DstScratch:  make([]uint16, size.Samples.DstLen),
		Scratch:     testFrameWorkRestorationPostFilterScratch(size.Apply),
	}
}

func TestApplySupportedPostFiltersBandedMatchesWholeChain(t *testing.T) {
	whole := buildPostFilterChainFixture(t)
	banded := buildPostFilterChainFixture(t)
	whole.loopFilter.TrustedCoverage = true
	banded.loopFilter.TrustedCoverage = true

	wholeReq := FrameWorkPostFilterRequest{
		LoopFilter:  whole.loopFilter,
		CDEF:        whole.cdef,
		Restoration: whole.rest,
	}
	bandedReq := FrameWorkPostFilterRequest{
		LoopFilter:  banded.loopFilter,
		CDEF:        banded.cdef,
		Restoration: banded.rest,
	}

	wholeNext, wholeResult, err := whole.ctx.ApplySupportedPostFilters(wholeReq)
	if err != nil {
		t.Fatal(err)
	}
	bandedNext, bandedResult, err := banded.ctx.ApplySupportedPostFiltersBanded(bandedReq, frameWorkDav1dPostFilterBanding(banded.ctx))
	if err != nil {
		t.Fatal(err)
	}
	wantCompleted := FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF | FrameWorkPostFilterLoopRestoration
	if wholeResult.Completed != wantCompleted || bandedResult.Completed != wantCompleted ||
		wholeNext.RemainingPostFilters() != 0 || bandedNext.RemainingPostFilters() != 0 {
		t.Fatalf("whole remaining=%b result=%+v banded remaining=%b result=%+v",
			wholeNext.RemainingPostFilters(), wholeResult, bandedNext.RemainingPostFilters(), bandedResult)
	}
	if wholeResult.LoopFilter.Applied != bandedResult.LoopFilter.Applied ||
		wholeResult.LoopFilter.Edges != bandedResult.LoopFilter.Edges ||
		wholeResult.LoopFilter.Plan.EdgeCandidates != bandedResult.LoopFilter.Plan.EdgeCandidates ||
		wholeResult.LoopFilter.Plan.StoredEdges != bandedResult.LoopFilter.Plan.StoredEdges ||
		wholeResult.CDEF != bandedResult.CDEF {
		t.Fatalf("whole result=%+v banded result=%+v", wholeResult, bandedResult)
	}
	assertPostFilterFramesEqual(t, whole.output, banded.output)
}

func assertPostFilterFramesEqual(t testing.TB, want *frame.Frame, got *frame.Frame) {
	t.Helper()
	if want == nil || got == nil {
		t.Fatalf("nil frame want=%p got=%p", want, got)
	}
	for _, planes := range [][2]frame.Plane{{want.Y, got.Y}, {want.U, got.U}, {want.V, got.V}} {
		w, g := planes[0], planes[1]
		if w.Width != g.Width || w.Height != g.Height || w.Stride != g.Stride {
			t.Fatalf("plane shape want=%dx%d/%d got=%dx%d/%d", w.Width, w.Height, w.Stride, g.Width, g.Height, g.Stride)
		}
		for y := 0; y < w.Height; y++ {
			wRow := w.Pix[y*w.Stride : y*w.Stride+w.Width]
			gRow := g.Pix[y*g.Stride : y*g.Stride+g.Width]
			if !bytes.Equal(wRow, gRow) {
				t.Fatalf("plane differs at row %d", y)
			}
		}
	}
}

// BenchmarkApplyLoopFilter measures only the deblocking pass on the
// fully populated 4:2:0 fixture frame. The per-iteration sequence is
// refill-the-frame -> ApplyLoopFilterEdges. The reported ms/frame and
// MB/sec numbers feed the README post-filter performance table.
func BenchmarkApplyLoopFilter(b *testing.B) {
	f := buildPostFilterChainFixture(b)
	// CDEF/restoration must be marked inactive on this ctx so the
	// LoopFilter-only Apply can run without triggering chained stages.
	ctx := f.ctx
	ctx.Event.SequenceHeader.EnableCDEF = false
	ctx.Event.SequenceHeader.EnableRestoration = false
	ctx.Event.CDEF = parser.CDEFParams{}
	ctx.Event.Restoration = parser.RestorationParams{}

	b.SetBytes(f.frameBytes())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.refill()
		if _, err := ctx.ApplyLoopFilterEdges(f.loopFilter); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	reportPostFilterMetrics(b)
}

func BenchmarkApplyLoopFilterBanded(b *testing.B) {
	f := buildPostFilterChainFixture(b)
	ctx := f.ctx
	ctx.Event.SequenceHeader.EnableCDEF = false
	ctx.Event.SequenceHeader.EnableRestoration = false
	ctx.Event.CDEF = parser.CDEFParams{}
	ctx.Event.Restoration = parser.RestorationParams{}
	req := f.loopFilter
	req.TrustedCoverage = true
	banding := frameWorkDav1dPostFilterBanding(ctx)

	b.SetBytes(f.frameBytes())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.refill()
		if _, err := ctx.ApplyLoopFilterEdgesBanded(req, banding.LoopFilterMIRows); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	reportPostFilterMetrics(b)
}

// BenchmarkApplyCDEF measures only the CDEF pass on the same shared frame
// fixture. The loop-filter and restoration stages are marked inactive so
// ApplyCDEFPostFilter runs in isolation.
func BenchmarkApplyCDEF(b *testing.B) {
	f := buildPostFilterChainFixture(b)
	ctx := f.ctx
	ctx.Event.LoopFilter = parser.LoopFilterParams{}
	ctx.Event.SequenceHeader.EnableRestoration = false
	ctx.Event.Restoration = parser.RestorationParams{}

	b.SetBytes(f.frameBytes())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.refill()
		if _, err := ctx.ApplyCDEFPostFilter(f.cdef); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	reportPostFilterMetrics(b)
}

func BenchmarkApplyCDEFBanded(b *testing.B) {
	f := buildPostFilterChainFixture(b)
	ctx := f.ctx
	ctx.Event.LoopFilter = parser.LoopFilterParams{}
	ctx.Event.SequenceHeader.EnableRestoration = false
	ctx.Event.Restoration = parser.RestorationParams{}
	banding := frameWorkDav1dPostFilterBanding(ctx)

	b.SetBytes(f.frameBytes())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.refill()
		if _, err := ctx.ApplyCDEFPostFilterBanded(f.cdef, banding.CDEFUnitRows); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	reportPostFilterMetrics(b)
}

// BenchmarkApplyLoopRestoration measures only the loop-restoration pass.
// Earlier post-filter stages are marked complete so ApplyLoopRestoration
// can run without triggering them.
func BenchmarkApplyLoopRestoration(b *testing.B) {
	f := buildPostFilterChainFixture(b)
	ctx := f.ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF)

	b.SetBytes(f.frameBytes())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.refill()
		if _, err := ctx.ApplyLoopRestorationPostFilter(f.rest); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	reportPostFilterMetrics(b)
}

// BenchmarkApplyFullPostFilter exercises the supported frame-pool publication
// chain end-to-end: loop filter -> CDEF -> loop restoration. This is the
// number that bounds the post-filter cost a real decoder will see for any
// frame that signals all three stages.
//
// Film grain and super-res are not in the supported chain on this fixture
// (no AR coefficients / no superres scaling configured) so this captures
// the steady-state cost of the three stages that are always run together.
func BenchmarkApplyFullPostFilter(b *testing.B) {
	f := buildPostFilterChainFixture(b)
	ctx := f.ctx
	req := FrameWorkPostFilterRequest{
		LoopFilter:  f.loopFilter,
		CDEF:        f.cdef,
		Restoration: f.rest,
	}

	b.SetBytes(f.frameBytes())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.refill()
		if _, _, err := ctx.ApplySupportedPostFilters(req); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	reportPostFilterMetrics(b)
}

func BenchmarkApplyFullPostFilterBanded(b *testing.B) {
	f := buildPostFilterChainFixture(b)
	ctx := f.ctx
	loopFilter := f.loopFilter
	loopFilter.TrustedCoverage = true
	req := FrameWorkPostFilterRequest{
		LoopFilter:  loopFilter,
		CDEF:        f.cdef,
		Restoration: f.rest,
	}
	banding := frameWorkDav1dPostFilterBanding(ctx)

	b.SetBytes(f.frameBytes())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.refill()
		if _, _, err := ctx.ApplySupportedPostFiltersBanded(req, banding); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	reportPostFilterMetrics(b)
}
