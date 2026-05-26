package main

import (
	"errors"
	"fmt"
	"io"
	"time"

	av1 "github.com/thesyncim/goav1"
)

// decodeStats accumulates the per-temporal-unit numbers dump_svc prints to
// stderr at the end of a successful run, plus the partial counts the run()
// helper reports when a payload mid-stream fails to decode.
type decodeStats struct {
	PayloadsConsumed int
	TemporalUnits    int
	CompletedFrames  int
	BytesOut         int64
}

// decoder owns the caller-supplied scratch needed to drive an SVC IVF
// through the goav1 public residual stream runner. The layout mirrors
// cmd/aom-go-dec/decoder.go: this command differs only in how it picks
// which completed-frame surface to emit per temporal unit (the highest
// spatial layer the temporal unit publishes).
//
// The runner is bound around a single frame.Pool, which constrains us
// to SVC streams whose spatial layers share one CodedWidth x Height
// (the L1T2 case). Truly multi-spatial-resolution streams need the
// multi-pool decoder.FrameSurfaceProvider path documented in
// docs/svc.md; that path is exercised by the framework dry-run tests
// in internal/av1/testvector but not by this CLI.
type decoder struct {
	pool       av1.FramePool
	workerPool *av1.TileWorkerPool
	stream     av1.DecoderStream

	refs       av1.DecoderSurfaceReferences
	state      av1.DecoderFrameWorkState
	refSurface []int
	refFrames  []*av1.Frame
	releases   []int
	stats      av1.DecoderFrameWorkTileResidualStats
	sideData   av1.DecoderFrameWorkSideData
	batch      av1.DecoderFrameWorkBatchResidualRunner

	scratch av1.DecoderFrameWorkResidualStreamScratch
	runner  av1.DecoderFrameWorkResidualStreamRunner

	payloads [][]byte
	format   av1.FrameFormat
}

func newDecoder(payloads [][]byte, workers int) (*decoder, error) {
	if workers < 1 {
		return nil, fmt.Errorf("workers must be >= 1, got %d", workers)
	}
	if len(payloads) == 0 {
		return nil, errors.New("no payloads to decode")
	}

	workerPool, err := av1.NewTileWorkerPool(workers)
	if err != nil {
		return nil, fmt.Errorf("worker pool: %w", err)
	}

	// Probe the stream to learn how much scratch the residual runner
	// needs. The probe also returns the bind event whose FrameSize
	// determines the frame.Pool format.
	var probeStream av1.DecoderStream
	probeEvents := make([]av1.DecoderEvent, probeEventBudget(payloads))
	probeSpans := make([]av1.TileSpan, av1.MaxTiles)
	probeJobs := make([]av1.TileJob, av1.MaxTiles)
	probeBatches := make([]av1.TileBatch, av1.MaxTiles)

	plan, err := av1.DecoderFrameWorkResidualLowOverheadStreamsPlan(
		probeStream, payloads, workers,
		probeEvents, probeSpans, probeJobs, probeBatches,
	)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("stream plan: %w", err)
	}
	if !plan.HasEvent() {
		workerPool.Close()
		return nil, errors.New("stream plan did not identify a bind event")
	}

	format := av1.FrameFormat{
		Width:        int(plan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(plan.Bind.Event.FrameSize.Height),
		BitDepth:     plan.Bind.Sequence.ColorConfig.BitDepth,
		MonoChrome:   plan.Bind.Sequence.ColorConfig.MonoChrome,
		SubsamplingX: plan.Bind.Sequence.ColorConfig.SubsamplingX,
		SubsamplingY: plan.Bind.Sequence.ColorConfig.SubsamplingY,
		Align:        64,
	}
	if format.BitDepth == 0 {
		format.BitDepth = 8
	}
	if format.Width <= 0 || format.Height <= 0 {
		workerPool.Close()
		return nil, fmt.Errorf("invalid frame format from stream plan: %dx%d", format.Width, format.Height)
	}

	const surfaceCount = av1.RefFrames + 1
	pool, err := bindFramePool(format, surfaceCount)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("frame pool: %w", err)
	}

	d := &decoder{
		pool:       pool,
		workerPool: workerPool,
		refSurface: make([]int, av1.InterRefsPerFrame),
		refFrames:  make([]*av1.Frame, av1.InterRefsPerFrame),
		releases:   make([]int, av1.RefFrames),
		payloads:   payloads,
		format:     format,
	}
	d.scratch = newStreamScratch(plan.Size)

	runner, _, err := av1.BindDecoderFrameWorkResidualStreamPlanRunner(plan, &d.stream,
		av1.DecoderFrameWorkResidualEventRuntime{
			State:             &d.state,
			Refs:              &d.refs,
			FramePool:         &d.pool,
			Align:             64,
			ReferenceSurfaces: d.refSurface,
			ReferenceFrames:   d.refFrames,
			Releases:          d.releases,
			WorkerPool:        d.workerPool,
			SideData:          &d.sideData,
			Stats:             &d.stats,
		}, d.scratch, &d.batch)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("bind runner: %w", err)
	}
	d.runner = runner
	return d, nil
}

// Close releases the worker goroutine pool. It is safe to call more than
// once.
func (d *decoder) Close() {
	if d.workerPool != nil {
		d.workerPool.Close()
		d.workerPool = nil
	}
}

// Decode drives the bound payloads one temporal unit at a time. After
// each payload it inspects the residual runner's Outputs slice and
// writes only the highest-SpatialID completed frame for that temporal
// unit, matching libaom aomdec's output_all_layers=0 convention.
//
// The runner's Outputs slice aliases caller-owned storage that gets
// reused on the next call into the runner, so the *Frame samples must
// be copied (or in our case, written to the io.Writer) before the next
// Decode iteration.
func (d *decoder) Decode(dst io.Writer, quiet bool, log io.Writer) (decodeStats, error) {
	var stats decodeStats

	d.pool.Reset()
	d.refs.Reset()
	d.state.Reset()
	d.stats = av1.DecoderFrameWorkTileResidualStats{}
	if err := d.runner.Reset(); err != nil {
		return stats, fmt.Errorf("runner reset: %w", err)
	}

	var result av1.DecoderFrameWorkResidualStreamResult
	for i, payload := range d.payloads {
		start := time.Now()
		result = av1.DecoderFrameWorkResidualStreamResult{}
		if err := d.runner.RunLowOverheadInto(&result, payload, nil); err != nil {
			return stats, fmt.Errorf("payload %d (%d bytes): %w", i, len(payload), err)
		}
		stats.PayloadsConsumed++
		stats.TemporalUnits++
		elapsed := time.Since(start)

		// The residual runner can publish multiple completed frames per
		// payload (e.g. an SVC temporal unit that publishes one frame
		// per spatial layer). Walk the outputs to find the last
		// non-nil frame (the runner appends outputs in event order, so
		// the last one corresponds to the highest spatial layer the
		// temporal unit reached).
		var emit *av1.Frame
		for _, frame := range result.Run.Outputs {
			if frame != nil {
				emit = frame
				stats.CompletedFrames++
			}
		}
		if emit != nil {
			n, werr := writeYUVFrame(dst, emit)
			stats.BytesOut += n
			if werr != nil {
				return stats, fmt.Errorf("payload %d write: %w", i, werr)
			}
		}
		if !quiet {
			fmt.Fprintf(log, "tu=%d payload=%d ms=%.3f outputs=%d completed_frames=%d\n",
				i, len(payload), float64(elapsed.Microseconds())/1000.0,
				result.Run.OutputCount, result.Run.CompletedFrames)
		}
	}
	return stats, nil
}

// writeYUVFrame writes a decoded frame's Y, U, V planes to dst stripping
// any per-plane stride padding. For monochrome streams only Y is
// written. 10/12-bit samples are written little-endian as two bytes per
// sample, matching the canonical FFmpeg "yuv420p10le" layout.
func writeYUVFrame(dst io.Writer, frame *av1.Frame) (int64, error) {
	var written int64
	for _, plane := range []av1.FramePlane{frame.Y, frame.U, frame.V} {
		if plane.Width == 0 || plane.Height == 0 || len(plane.Pix) == 0 {
			continue
		}
		rowBytes := plane.Width * frame.Layout.BytesPerSample
		for row := 0; row < plane.Height; row++ {
			off := row * plane.Stride
			n, err := dst.Write(plane.Pix[off : off+rowBytes])
			if err != nil {
				return written + int64(n), err
			}
			written += int64(n)
		}
	}
	return written, nil
}

func bindFramePool(format av1.FrameFormat, count int) (av1.FramePool, error) {
	_, backingSize, err := av1.FramePoolRequiredSize(format, count)
	if err != nil {
		return av1.FramePool{}, err
	}
	frames := make([]av1.Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	return av1.BindFramePool(make([]byte, backingSize), format, frames, free, used)
}

func newStreamScratch(size av1.DecoderFrameWorkResidualStreamScratchSize) av1.DecoderFrameWorkResidualStreamScratch {
	return av1.DecoderFrameWorkResidualStreamScratch{
		Events:    make([]av1.DecoderEvent, size.Events),
		Event:     newEventScratch(size.Event),
		SideData:  newSideDataScratch(size.Event.SideData),
		Outputs:   make([]*av1.Frame, size.Event.Outputs),
		RTPBuffer: make([]byte, size.RTPBuffer),
		RTPSpans:  make([]av1.RTPObuSpan, size.RTPSpans),
	}
}

func newEventScratch(size av1.DecoderFrameWorkResidualEventScratchSize) av1.DecoderFrameWorkResidualEventScratch {
	return av1.DecoderFrameWorkResidualEventScratch{
		Runner:   newBatchRunnerScratch(size.Runner),
		SideData: newSideDataScratch(size.SideData),
		Spans:    make([]av1.TileSpan, size.Plan.SpanCount),
		Jobs:     make([]av1.TileJob, size.Plan.JobCount),
		Batches:  make([]av1.TileBatch, size.Plan.BatchCount),
	}
}

func newBatchRunnerScratch(size av1.DecoderFrameWorkBatchResidualRunnerScratchSize) av1.DecoderFrameWorkBatchResidualRunnerScratch {
	return av1.DecoderFrameWorkBatchResidualRunnerScratch{
		States:                  make([]av1.TileDecodeState, size.Workers),
		Storages:                make([]av1.DecoderFrameWorkTileResidualCDFStorage, size.Workers),
		TileScratch:             make([]av1.DecoderFrameWorkTileResidualScratch, size.Workers),
		RestorationRequests:     make([]av1.DecoderFrameWorkTileRestorationRequest, size.RestorationRequests),
		PredictionScratch:       make([]av1.DecoderFrameWorkPredictionScratch, size.Workers),
		InterPredictionScratch:  make([]av1.DecoderFrameWorkInterPredictionScratch, size.Workers),
		Stats:                   make([]av1.DecoderFrameWorkTileResidualStats, size.Workers),
		Int32Scratch:            make([]int32, size.Int32Scratch),
		ResidualScratch:         make([]int16, size.ResidualScratch),
		LoopContextAboveScratch: make([]av1.TileBlockLoopRootAboveContext, size.LoopContextAbove),
	}
}

func newSideDataScratch(size av1.DecoderFrameWorkSideDataScratchSize) av1.DecoderFrameWorkSideDataScratch {
	return av1.DecoderFrameWorkSideDataScratch{
		CDEFIndexMap:             make([]uint8, size.CDEFIndexMap),
		CDEFReadMap:              make([]bool, size.CDEFReadMap),
		LoopFilterMap:            make([]av1.DecoderFrameWorkLoopFilterBlockRecord, size.LoopFilterMap),
		RestorationRecords:       make([]av1.TileRestorationUnitRecord, size.RestorationRecords),
		RestorationBoundaryAbove: make([]uint16, size.RestorationBoundaryAbove),
		RestorationBoundaryBelow: make([]uint16, size.RestorationBoundaryBelow),
	}
}

func probeEventBudget(payloads [][]byte) int {
	const eventsPerFrame = 16
	n := len(payloads) * eventsPerFrame
	if n < 64 {
		n = 64
	}
	return n
}
