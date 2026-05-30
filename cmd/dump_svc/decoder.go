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

	// FailureEvents captures the (TemporalID, SpatialID) layer
	// composition of the payload that triggered a decode error so the
	// CLI can surface "which layer was in flight when we died" together
	// with the proximate decoder error string.
	FailureEvents string
}

// decodeOptions bundles the per-Decode tuning knobs the CLI surfaces.
// SpatialLayer == spatialLayerHighest (-1) follows libaom aomdec's
// output_all_layers=0 convention: emit the highest-SpatialID frame the
// temporal unit publishes. A non-negative value pins emission to that
// SpatialID, skipping temporal units that did not publish at that layer.
type decodeOptions struct {
	Quiet        bool
	Log          io.Writer
	SpatialLayer int8
}

// decoder owns the caller-supplied scratch needed to drive an SVC IVF
// through the goav1 public residual stream runner. The layout mirrors
// cmd/aom-go-dec/decoder.go: this command differs only in how it picks
// which completed-frame surface to emit per temporal unit (the highest
// spatial layer the temporal unit publishes, or the pinned layer ID
// when -spatial-layer is set).
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
// writes the selected spatial layer's completed frame for that temporal
// unit. With opts.SpatialLayer == spatialLayerHighest (-1) it matches
// libaom aomdec's output_all_layers=0 convention (last non-nil output);
// with a non-negative value it pins emission to that SpatialID.
//
// The runner's Outputs slice aliases caller-owned storage that gets
// reused on the next call into the runner, so the *Frame samples must
// be copied (or in our case, written to the io.Writer) before the next
// Decode iteration.
func (d *decoder) Decode(dst io.Writer, opts decodeOptions) (decodeStats, error) {
	var stats decodeStats
	if opts.Log == nil {
		opts.Log = io.Discard
	}

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
			// Annotate the failure with the layer composition of the
			// failing payload so the caller knows which (T, S) pair was
			// in flight at the moment of the proximate decoder error.
			stats.FailureEvents = describePayloadLayers(payload)
			return stats, fmt.Errorf("payload %d (%d bytes, layers=%s): %w",
				i, len(payload), stats.FailureEvents, err)
		}
		stats.PayloadsConsumed++
		stats.TemporalUnits++
		elapsed := time.Since(start)

		// The residual runner can publish multiple completed frames per
		// payload (e.g. an SVC temporal unit that publishes one frame
		// per spatial layer). Walk the parsed events (which carry the
		// SVC (T, S) tagging) in lock-step with the output slots to
		// pick either the highest spatial layer (default) or the frame
		// whose SpatialID matches opts.SpatialLayer.
		events := d.runner.Events[:result.EventCount]
		emit, emitSpatialID, completedInUnit := selectOutput(events, result.Run.Outputs, opts.SpatialLayer)
		stats.CompletedFrames += completedInUnit
		if emit != nil {
			n, werr := writeYUVFrame(dst, emit)
			stats.BytesOut += n
			if werr != nil {
				stats.FailureEvents = describePayloadLayers(payload)
				return stats, fmt.Errorf("payload %d write: %w", i, werr)
			}
		}
		if !opts.Quiet {
			fmt.Fprintf(opts.Log,
				"tu=%d payload=%d ms=%.3f outputs=%d completed_frames=%d emit=%v emit_spatial_id=%d\n",
				i, len(payload), float64(elapsed.Microseconds())/1000.0,
				result.Run.OutputCount, result.Run.CompletedFrames, emit != nil, emitSpatialID)
		}
	}
	return stats, nil
}

// selectOutput pairs the parsed-event slice with the output slice to
// pick the *Frame the CLI should emit for one payload. Output slots are
// filled in event order (one slot per frame-publishing event, in the
// order the runner processed those events), so by walking events and
// outputs in lock-step we can correlate each output to the SpatialID
// the event carried.
//
// Layer selection rules:
//
//   - spatialLayer == spatialLayerHighest (-1): emit the last non-nil
//     entry. The residual stream runner processes events in OBU order
//     and SVC OBUs are emitted in ascending SpatialID per temporal
//     unit, so the last non-nil frame corresponds to the highest
//     spatial layer the temporal unit reached. This matches libaom
//     aomdec's default output_all_layers=0 behavior.
//
//   - spatialLayer >= 0: emit only the frame whose event SpatialID
//     matches the pinned layer. If the temporal unit did not publish
//     at that layer we return nil (and the caller writes no bytes for
//     this temporal unit), which matches an SFU that drops everything
//     above its operating point.
//
// The returned int is the total number of non-nil outputs in this
// payload regardless of which one we pick, so the CLI's summary
// reflects the total decode work the runner actually performed (not
// just the bytes we serialised). The returned SpatialID is the layer
// of the *Frame we emit, or -1 if no frame is emitted.
func selectOutput(events []av1.DecoderEvent, outputs []*av1.Frame, spatialLayer int8) (*av1.Frame, int8, int) {
	completed := 0
	var last *av1.Frame
	var lastSpatial int8 = -1
	var pinned *av1.Frame
	var pinnedSpatial int8 = -1
	outIdx := 0
	for i := range events {
		if !av1.DecoderEventOutputsFrame(events[i]) {
			continue
		}
		if outIdx >= len(outputs) {
			break
		}
		frame := outputs[outIdx]
		outIdx++
		if frame == nil {
			// Slot reserved for this event but the runner did not
			// publish a surface (e.g. a hidden frame). Don't count
			// it toward completed frames.
			continue
		}
		completed++
		last = frame
		lastSpatial = int8(events[i].SpatialID)
		if spatialLayer >= 0 && int8(events[i].SpatialID) == spatialLayer {
			pinned = frame
			pinnedSpatial = int8(events[i].SpatialID)
		}
	}
	if spatialLayer >= 0 {
		return pinned, pinnedSpatial, completed
	}
	return last, lastSpatial, completed
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
	n := max(len(payloads)*eventsPerFrame, 64)
	return n
}
