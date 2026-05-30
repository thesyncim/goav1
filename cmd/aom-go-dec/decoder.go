package main

import (
	"errors"
	"fmt"
	"io"
	"time"

	av1 "github.com/thesyncim/goav1"
)

// decoder owns all caller-supplied scratch the public residual stream runner
// needs to decode a sequence of IVF frame payloads end-to-end. It mirrors the
// layout used by bench_test.go's decodeBenchmarkHarness so the CLI exercises
// the same public-API path a production integration would take.
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

	postFilter av1.DecoderFrameWorkReusableSupportedPostFilterRunner

	payloads [][]byte
	format   av1.FrameFormat
}

// newDecoder probes the supplied frame payloads to size all scratch arenas
// the public stream runner needs, allocates them once, and binds the runner
// against caller-owned state. Once bound, decode iterations are zero-alloc.
func newDecoder(frames []av1.IVFFrame, workers int) (*decoder, error) {
	if workers < 1 {
		return nil, fmt.Errorf("workers must be >= 1, got %d", workers)
	}
	payloads := make([][]byte, len(frames))
	for i, f := range frames {
		payloads[i] = f.Payload
	}

	workerPool, err := av1.NewTileWorkerPool(workers)
	if err != nil {
		return nil, fmt.Errorf("worker pool: %w", err)
	}

	// Probe the stream to learn how much scratch the runner will need.
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

	// Derive the pool format from the bound headers so the superblock alignment
	// (64 vs 128) matches the surface the decoder reconstructs into.
	format, err := av1.FrameCodedFormatFromHeaders(plan.Bind.Sequence, plan.Bind.Event.FrameSize, 64)
	if err != nil {
		workerPool.Close()
		return nil, fmt.Errorf("frame format from stream plan: %w", err)
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

// Close releases the worker goroutine pool. It is safe to call more than once.
func (d *decoder) Close() {
	if d.workerPool != nil {
		d.workerPool.Close()
		d.workerPool = nil
	}
}

// Decode iterates the bound payloads, drives them through the runner one
// payload at a time so per-frame timing is meaningful, writes each completed
// frame's planes to dst in I420 / I400 order, and reports total bytes written
// plus completed-frame count.
func (d *decoder) Decode(dst io.Writer, quiet bool, log io.Writer) (int64, int, error) {
	d.pool.Reset()
	d.refs.Reset()
	d.state.Reset()
	d.stats = av1.DecoderFrameWorkTileResidualStats{}
	if err := d.runner.Reset(); err != nil {
		return 0, 0, fmt.Errorf("runner reset: %w", err)
	}

	var (
		totalBytes int64
		completed  int
		result     av1.DecoderFrameWorkResidualStreamResult
	)

	for i, payload := range d.payloads {
		start := time.Now()
		result = av1.DecoderFrameWorkResidualStreamResult{}
		if err := d.runner.RunLowOverheadIntoWithPostFilterRunner(&result, payload, &d.postFilter); err != nil {
			return totalBytes, completed, fmt.Errorf("frame %d: %w", i, err)
		}
		elapsed := time.Since(start)
		// result.Run.Outputs aliases the stream runner's caller-owned output
		// arena, so the *Frame pointers are valid until the next Run* call.

		for _, frame := range result.Run.Outputs {
			if frame == nil {
				continue
			}
			n, err := writeYUVFrame(dst, frame)
			if err != nil {
				return totalBytes, completed, fmt.Errorf("frame %d write: %w", i, err)
			}
			totalBytes += n
			completed++
		}
		if !quiet {
			fmt.Fprintf(log, "frame %d payload=%d ms=%.3f outputs=%d completed_frames=%d\n",
				i, len(payload), float64(elapsed.Microseconds())/1000.0,
				result.Run.OutputCount, result.Run.CompletedFrames)
		}
	}
	return totalBytes, completed, nil
}

// writeYUVFrame writes a decoded frame's Y, U, V planes to dst stripping any
// per-plane stride padding. For monochrome streams only Y is written. 10/12-bit
// samples are written little-endian as two bytes per sample, matching the
// canonical FFmpeg "yuv420p10le" layout.
func writeYUVFrame(dst io.Writer, frame *av1.Frame) (int64, error) {
	var written int64
	for _, plane := range []av1.FramePlane{frame.Y, frame.U, frame.V} {
		// MonoChrome streams leave U/V zero-sized; skip them so the output
		// matches the canonical I400 byte layout.
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
	// Each frame typically expands to a handful of events. Reserve a generous
	// upper bound so the streams planner never returns ErrEventBufferTooSmall.
	const eventsPerFrame = 16
	n := max(len(payloads)*eventsPerFrame, 64)
	return n
}
