package decoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// TestLoopRestorationPooledMatchesSerial proves the pooled 8-bit in-place
// restoration apply (applyLoopRestorationPostFilterMaybePooled with a multi-lane
// pool) is byte-identical to the serial whole-plane apply, for several worker
// counts, under -race. The frame is tall enough (8 luma RU rows) that the RU-row
// band split engages, and every plane carries a Wiener filter so the kernels run.
// Identical pixels across worker counts is the worker-invariance proof (mirrors
// TestCDEFPostFilterPooledMatchesSerial).
func TestLoopRestorationPooledMatchesSerial(t *testing.T) {
	const width, height = 256, 512
	seq := testSequence()
	seq.EnableRestoration = true
	seq.ColorConfig.BitDepth = 8
	seq.ColorConfig.SubsamplingX = true
	seq.ColorConfig.SubsamplingY = true
	size := parser.FrameSize{CodedWidth: width, UpscaledWidth: width, Height: height, SuperResDenominator: 8}
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		Restoration: parser.RestorationParams{
			Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationWiener, parser.RestorationWiener},
			UnitSizeY:  64,
			UnitSizeUV: 64,
		},
	}
	format := frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}

	// Records (Wiener on all planes) and zero boundary buffers shared read-only.
	batch := threading.FrameWorkBatch{
		FrameWorkFrameContext: threading.FrameWorkFrameContext{
			Sequence:    threading.FrameWorkSequenceContextFromHeader(seq),
			FrameSize:   size,
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
	defaultWiener := av1restoration.DefaultWienerInfo()
	for plane := 0; plane < int(plan.Planes); plane++ {
		if plan.Grids[plane].Type != parser.RestorationWiener {
			continue
		}
		for i := range buffers.Records[plane] {
			buffers.Records[plane][i].Unit.Type = parser.RestorationWiener
			buffers.Records[plane][i].Unit.Wiener = defaultWiener
		}
	}

	build := func(pool *threading.Pool) (*frame.Frame, FrameWorkPostFilterContext, FrameWorkRestorationPostFilterRequest) {
		output := testFrameWorkCDEFFrame(t, format)
		testFillFrameWorkCDEFPlane(output.Y)
		testFillFrameWorkCDEFPlane(output.U)
		testFillFrameWorkCDEFPlane(output.V)
		ctx := FrameWorkPostFilterContext{Event: event, Output: output, pool: pool}
		scratchSize, err := ctx.LoopRestorationPostFilterScratchLen(buffers.Records, false)
		if err != nil {
			t.Fatal(err)
		}
		req := FrameWorkRestorationPostFilterRequest{
			Records:     buffers.Records,
			Boundaries:  buffers.Boundaries,
			DataScratch: make([]uint16, scratchSize.Samples.DataLen),
			DstScratch:  make([]uint16, scratchSize.Samples.DstLen),
			Scratch:     testFrameWorkRestorationPostFilterScratch(scratchSize.Apply),
			Pool: FrameWorkRestorationPoolScratch{
				Bands:      scratchSize.Pool.Bands,
				BandData:   make([]uint16, scratchSize.Pool.TotalData()),
				BandWiener: make([]uint16, scratchSize.Pool.TotalWiener()),
				BandSGR:    make([]int32, scratchSize.Pool.TotalSGR()),
			},
		}
		return output, ctx, req
	}

	serialOut, serialCtx, serialReq := build(nil)
	if _, err := serialCtx.ApplyLoopRestorationPostFilter(serialReq); err != nil {
		t.Fatal(err)
	}

	for _, workers := range []int{2, 3, 4, 8} {
		pool, err := threading.NewPool(workers)
		if err != nil {
			t.Fatalf("workers=%d: %v", workers, err)
		}
		pooledOut, pooledCtx, pooledReq := build(pool)
		if pooledReq.Pool.Bands < 2 {
			pool.Close()
			t.Fatalf("workers=%d: pool scratch not sized (Bands=%d); pooled path would not engage", workers, pooledReq.Pool.Bands)
		}
		res, err := pooledCtx.applyLoopRestorationPostFilterMaybePooled(pooledReq)
		if err != nil {
			pool.Close()
			t.Fatalf("workers=%d apply: %v", workers, err)
		}
		if res.FilteredRecords == 0 {
			pool.Close()
			t.Fatalf("workers=%d degenerate: pooled apply filtered no records", workers)
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
