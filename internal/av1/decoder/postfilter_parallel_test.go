package decoder

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

func TestPostFilterParallelRunnerCDEFNoPoolAllocsAndParity(t *testing.T) {
	const width, height = 256, 512 // eight CDEF unit rows; two worker ranges are active
	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth: width, UpscaledWidth: width, Height: height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping: 5, StrengthCount: 2,
			YStrength:  [parser.MaxCDEFStrengths]uint8{9, 31},
			UVStrength: [parser.MaxCDEFStrengths]uint8{5, 17},
		},
	}
	format := frame.Format{
		Width: width, Height: height, BitDepth: 8,
		SubsamplingX: true, SubsamplingY: true, Align: 64,
	}
	build := func(parallel *FrameWorkPostFilterParallel) (*frame.Frame, FrameWorkPostFilterContext, FrameWorkCDEFPostFilterRequest) {
		out := testFrameWorkCDEFFrame(t, format)
		testFillFrameWorkCDEFPlane(out.Y)
		testFillFrameWorkCDEFPlane(out.U)
		testFillFrameWorkCDEFPlane(out.V)
		ctx := FrameWorkPostFilterContext{Event: event, Output: out, Parallel: parallel}
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
		indices := make([]uint8, length)
		for i := range indices {
			indices[i] = uint8(i % 2)
		}
		indexMap, err := batch.BindCDEFIndexMap(indices, make([]bool, length))
		if err != nil {
			t.Fatal(err)
		}
		for i := range indexMap.Read {
			indexMap.Read[i] = true
		}
		scratchSize, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			t.Fatal(err)
		}
		samples, dst, directions, variances, input, unitDst := testFrameWorkCDEFScratchStorage(scratchSize)
		req, err := scratchSize.BindRequest(indexMap, samples, dst, directions, variances, input, unitDst)
		if err != nil {
			t.Fatal(err)
		}
		return out, ctx, req
	}

	serialOut, serialCtx, serialReq := build(nil)
	serialResult, err := serialCtx.ApplyCDEFPostFilter(serialReq)
	if err != nil {
		t.Fatal(err)
	}
	if serialResult.Units == 0 || serialResult.Blocks == 0 {
		t.Fatalf("serial CDEF did no filtering: %+v", serialResult)
	}

	parallel := &FrameWorkPostFilterParallel{Workers: 2}
	defer parallel.Close()
	parallelOut, parallelCtx, parallelReq := build(parallel)
	initialY := append([]byte(nil), parallelOut.Y.Pix...)
	initialU := append([]byte(nil), parallelOut.U.Pix...)
	initialV := append([]byte(nil), parallelOut.V.Pix...)
	apply := func() {
		copy(parallelOut.Y.Pix, initialY)
		copy(parallelOut.U.Pix, initialU)
		copy(parallelOut.V.Pix, initialV)
		result, handled, err := parallelCtx.applyCDEFPostFilterParallel(parallelReq, 1)
		if err != nil {
			t.Fatalf("parallel CDEF: %v", err)
		}
		if !handled || result.Units == 0 || result.Blocks == 0 {
			t.Fatalf("parallel CDEF did not process the active bands: handled=%v result=%+v", handled, result)
		}
	}
	apply() // allocate runner pool and scratch once before measuring
	if parallel.ownedPool == nil || parallel.ownedWorkers != 2 {
		t.Fatalf("no-pool apply did not retain a two-worker owned pool: pool=%p workers=%d", parallel.ownedPool, parallel.ownedWorkers)
	}
	allocs := testing.AllocsPerRun(5, apply)
	if allocs != 0 {
		t.Fatalf("warmed two-worker CDEF runner allocated %g times per apply", allocs)
	}
	if len(parallel.cdef) < 2 || parallel.cdef[0].result.Blocks == 0 || parallel.cdef[1].result.Blocks == 0 {
		t.Fatalf("8-bit CDEF did not use both worker ranges: %+v", parallel.cdef)
	}
	lfMaskApplyAssertFramesEqual(t, serialOut, parallelOut)

	oldPool := parallel.ownedPool
	parallel.Workers = 3
	apply()
	if parallel.ownedPool == oldPool || parallel.ownedWorkers != 3 {
		t.Fatalf("changing worker count did not rebuild the owned pool: old=%p new=%p workers=%d", oldPool, parallel.ownedPool, parallel.ownedWorkers)
	}
	if err := oldPool.RunRangesRunner(1, 1, postFilterNoopRangeRunner{}); err != threading.ErrPoolClosed {
		t.Fatalf("replaced pool RunRangesRunner error=%v, want ErrPoolClosed", err)
	}
	lfMaskApplyAssertFramesEqual(t, serialOut, parallelOut)

	parallel.Close()
	parallel.Close()
	if parallel.ownedPool != nil {
		t.Fatal("Close retained the owned pool")
	}
}

func TestPostFilterParallelRunnerBorrowsContextPool(t *testing.T) {
	parallel := &FrameWorkPostFilterParallel{Workers: 2}
	ctx, req, serialOut, gotOut := buildPostFilterParallelCDEFPair(t, parallel, 8)
	borrowed, err := threading.NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer borrowed.Close()
	ctx.pool = borrowed
	if _, handled, err := ctx.applyCDEFPostFilterParallel(req, 1); err != nil || !handled {
		t.Fatalf("borrowed-pool CDEF handled=%v err=%v", handled, err)
	}
	lfMaskApplyAssertFramesEqual(t, serialOut, gotOut)
	parallel.Close()
	if parallel.ownedPool != nil {
		t.Fatalf("borrowed context pool was captured as owned: %p", parallel.ownedPool)
	}
	if err := borrowed.RunRangesRunner(1, 1, postFilterNoopRangeRunner{}); err != nil {
		t.Fatalf("parallel Close closed borrowed pool: %v", err)
	}
}

func TestPostFilterParallelRunnerCDEFSnapshotMatchesSerial(t *testing.T) {
	parallel := &FrameWorkPostFilterParallel{Workers: 2}
	defer parallel.Close()
	ctx, req, serialOut, gotOut := buildPostFilterParallelCDEFPair(t, parallel, 10)
	result, handled, err := ctx.applyCDEFPostFilterParallel(req, 1)
	if err != nil || !handled {
		t.Fatalf("HBD snapshot parallel CDEF handled=%v err=%v", handled, err)
	}
	if result.Units == 0 || result.Blocks == 0 || len(parallel.cdef) < 2 ||
		parallel.cdef[0].result.Blocks == 0 || parallel.cdef[1].result.Blocks == 0 {
		t.Fatalf("HBD CDEF did not use two active worker ranges: result=%+v workers=%+v", result, parallel.cdef)
	}
	lfMaskApplyAssertFramesEqual(t, serialOut, gotOut)
}

func buildPostFilterParallelCDEFPair(t *testing.T, parallel *FrameWorkPostFilterParallel, bitDepth int) (FrameWorkPostFilterContext, FrameWorkCDEFPostFilterRequest, *frame.Frame, *frame.Frame) {
	t.Helper()
	const width, height = 256, 512
	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth: width, UpscaledWidth: width, Height: height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping: 5, StrengthCount: 2,
			YStrength:  [parser.MaxCDEFStrengths]uint8{9, 31},
			UVStrength: [parser.MaxCDEFStrengths]uint8{5, 17},
		},
	}
	format := frame.Format{Width: width, Height: height, BitDepth: uint8(bitDepth), SubsamplingX: true, SubsamplingY: true, Align: 64}
	makeContext := func(par *FrameWorkPostFilterParallel) (*frame.Frame, FrameWorkPostFilterContext, FrameWorkCDEFPostFilterRequest) {
		out := testFrameWorkCDEFFrame(t, format)
		fillPostFilterParallelCDEFPlane(out.Y, bitDepth)
		fillPostFilterParallelCDEFPlane(out.U, bitDepth)
		fillPostFilterParallelCDEFPlane(out.V, bitDepth)
		ctx := (FrameWorkPostFilterContext{Event: event, Output: out, Parallel: par}).WithCompletedPostFilters(FrameWorkPostFilterLoopFilter)
		batch := threading.FrameWorkBatch{FrameWorkFrameContext: threading.FrameWorkFrameContext{
			Sequence:  threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader),
			FrameSize: event.FrameSize, CDEF: event.CDEF,
		}}
		_, _, length, err := batch.CDEFIndexMapShape()
		if err != nil {
			t.Fatal(err)
		}
		indices := make([]uint8, length)
		for i := range indices {
			indices[i] = uint8(i % 2)
		}
		indexMap, err := batch.BindCDEFIndexMap(indices, make([]bool, length))
		if err != nil {
			t.Fatal(err)
		}
		for i := range indexMap.Read {
			indexMap.Read[i] = true
		}
		scratchSize, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			t.Fatal(err)
		}
		samples, dst, directions, variances, input, unitDst := testFrameWorkCDEFScratchStorage(scratchSize)
		req, err := scratchSize.BindRequest(indexMap, samples, dst, directions, variances, input, unitDst)
		if err != nil {
			t.Fatal(err)
		}
		return out, ctx, req
	}
	serialOut, serialCtx, serialReq := makeContext(nil)
	if _, err := serialCtx.ApplyCDEFPostFilter(serialReq); err != nil {
		t.Fatal(err)
	}
	gotOut, gotCtx, gotReq := makeContext(parallel)
	return gotCtx, gotReq, serialOut, gotOut
}

func fillPostFilterParallelCDEFPlane(plane frame.Plane, bitDepth int) {
	bytesPerSample := 1
	if bitDepth > 8 {
		bytesPerSample = 2
	}
	maxSample := (1 << bitDepth) - 1
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			value := (x*37 + y*53 + (x^y)*17) & maxSample
			offset := y*plane.Stride + x*bytesPerSample
			plane.Pix[offset] = byte(value)
			if bytesPerSample == 2 {
				plane.Pix[offset+1] = byte(value >> 8)
			}
		}
	}
}

func TestPostFilterParallelRunnerLoopFilterNoPoolAllocsAndParity(t *testing.T) {
	const width, height = 256, 512
	size := parser.FrameSize{CodedWidth: width, UpscaledWidth: width, Height: height, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	lf := parser.LoopFilterParams{LevelY: [2]uint8{20, 20}, LevelU: 16, LevelV: 16, Sharpness: 1}
	event := lfMaskApplyEvent(seq, size, lf)
	format := frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	var records []lfMaskApplyRecord
	for row := 0; row < height/4; row += 4 {
		for col := 0; col < width/4; col += 4 {
			records = append(records, lfMaskApplyRecord{rec: testFrameWorkLoopFilterPostFilterRecordAt(col, row, col+4, row+4)})
		}
	}
	plainRecords := make([]threading.FrameWorkLoopFilterBlockRecord, len(records))
	for i := range records {
		plainRecords[i] = records[i].rec
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, plainRecords...)
	build := func(parallel *FrameWorkPostFilterParallel) (*frame.Frame, FrameWorkPostFilterContext, *threading.FrameWorkLoopFilterMasks) {
		out := testFrameWorkCDEFFrame(t, format)
		lfMaskApplyFillFrame(out)
		masks := buildLoopFilterMasksFromRecords(t, event, size, records, 4)
		ctx := FrameWorkPostFilterContext{Event: event, Output: out, LoopFilterMap: &filterMap, LoopFilterMasks: masks, Parallel: parallel}
		return out, ctx, masks
	}
	serialOut, serialCtx, serialMasks := build(nil)
	if _, err := serialCtx.ApplyLoopFilterEdgesFromMasks(serialMasks, filterMap); err != nil {
		t.Fatal(err)
	}

	parallel := &FrameWorkPostFilterParallel{Workers: 2}
	defer parallel.Close()
	parallelOut, parallelCtx, _ := build(parallel)
	initialY := append([]byte(nil), parallelOut.Y.Pix...)
	initialU := append([]byte(nil), parallelOut.U.Pix...)
	initialV := append([]byte(nil), parallelOut.V.Pix...)
	apply := func() {
		copy(parallelOut.Y.Pix, initialY)
		copy(parallelOut.U.Pix, initialU)
		copy(parallelOut.V.Pix, initialV)
		result, handled, err := parallelCtx.applyLoopFilterMaskBandsParallel(filterMap)
		if err != nil {
			t.Fatalf("parallel loop filter: %v", err)
		}
		if !handled || !result.Active {
			t.Fatalf("parallel loop filter did no work: handled=%v result=%+v", handled, result)
		}
	}
	apply()
	if parallel.ownedPool == nil || parallel.ownedWorkers != 2 {
		t.Fatalf("loop-filter path did not retain an owned two-worker pool: pool=%p workers=%d", parallel.ownedPool, parallel.ownedWorkers)
	}
	allocs := testing.AllocsPerRun(5, apply)
	if allocs != 0 {
		t.Fatalf("warmed two-worker loop-filter runner allocated %g times per apply", allocs)
	}
	if bytes.Equal(initialY, parallelOut.Y.Pix) {
		t.Fatal("parallel loop filter left the active luma plane unchanged")
	}
	lfMaskApplyAssertFramesEqual(t, serialOut, parallelOut)
}

type postFilterNoopRangeRunner struct{}

func (postFilterNoopRangeRunner) RunRange(int, int, int) error { return nil }
