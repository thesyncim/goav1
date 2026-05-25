package goav1

// DecoderFrameWorkResidualEventRequest carries caller-owned state for running
// one decoder frame-work event with DecoderFrameWorkBatchResidualRunner.
type DecoderFrameWorkResidualEventRequest struct {
	State     *DecoderFrameWorkState
	Refs      *DecoderSurfaceReferences
	FramePool *FramePool

	Sequence SequenceHeader
	Event    DecoderEvent
	Align    int

	ReferenceSurfaces []int
	ReferenceFrames   []*Frame

	Workers int
	Spans   []TileSpan
	Jobs    []TileJob
	Batches []TileBatch

	Releases   []int
	WorkerPool *TileWorkerPool

	Runner   *DecoderFrameWorkBatchResidualRunner
	SideData *DecoderFrameWorkSideData
	Post     DecoderFrameWorkPostFilterFunc

	PostRunner DecoderFrameWorkPostFilterRunner
	Stats      *DecoderFrameWorkTileResidualStats
}

// DecoderFrameWorkResidualEventRunner stores the stable caller-owned buffers
// and pools used to run decoder frame-work events with a residual batch runner.
type DecoderFrameWorkResidualEventRunner struct {
	State     *DecoderFrameWorkState
	Refs      *DecoderSurfaceReferences
	FramePool *FramePool
	Align     int

	ReferenceSurfaces []int
	ReferenceFrames   []*Frame

	Workers int
	Spans   []TileSpan
	Jobs    []TileJob
	Batches []TileBatch

	Releases   []int
	WorkerPool *TileWorkerPool

	BatchRunner *DecoderFrameWorkBatchResidualRunner
	Stats       *DecoderFrameWorkTileResidualStats
}

// DecoderFrameWorkResidualEventRuntime groups stable decoder state that is not
// carved out of residual-event scratch.
type DecoderFrameWorkResidualEventRuntime struct {
	State     *DecoderFrameWorkState
	Refs      *DecoderSurfaceReferences
	FramePool *FramePool
	Align     int

	ReferenceSurfaces []int
	ReferenceFrames   []*Frame
	Releases          []int

	WorkerPool *TileWorkerPool
	Stats      *DecoderFrameWorkTileResidualStats
}

// DecoderFrameWorkResidualEventScratchSize reports the caller-owned scratch
// needed to run one residual event and capture its postfilter side data.
type DecoderFrameWorkResidualEventScratchSize struct {
	Runner   DecoderFrameWorkBatchResidualRunnerScratchSize
	SideData DecoderFrameWorkSideDataScratchSize
	Plan     DecoderTileWorkPlan
}

// DecoderFrameWorkResidualEventScratch carries typed caller-owned arenas for
// BindDecoderFrameWorkResidualEventRunner.
type DecoderFrameWorkResidualEventScratch struct {
	Runner   DecoderFrameWorkBatchResidualRunnerScratch
	SideData DecoderFrameWorkSideDataScratch

	Spans   []TileSpan
	Jobs    []TileJob
	Batches []TileBatch
}

// Max returns per-arena maximum lengths and plan counts for reusable residual
// event scratch.
func (s DecoderFrameWorkResidualEventScratchSize) Max(other DecoderFrameWorkResidualEventScratchSize) DecoderFrameWorkResidualEventScratchSize {
	return DecoderFrameWorkResidualEventScratchSize{
		Runner:   s.Runner.Max(other.Runner),
		SideData: s.SideData.Max(other.SideData),
		Plan: DecoderTileWorkPlan{
			SpanCount:  max(s.Plan.SpanCount, other.Plan.SpanCount),
			JobCount:   max(s.Plan.JobCount, other.Plan.JobCount),
			BatchCount: max(s.Plan.BatchCount, other.Plan.BatchCount),
		},
	}
}

// BindDecoderFrameWorkResidualEventRunner binds a reusable event runner and
// frame-level side data from caller-owned scratch. batchRunner is caller-owned
// storage for the nested residual batch runner; the returned event runner keeps
// a pointer to it.
func BindDecoderFrameWorkResidualEventRunner(size DecoderFrameWorkResidualEventScratchSize, sequence SequenceHeader, event DecoderEvent, runtime DecoderFrameWorkResidualEventRuntime, scratch DecoderFrameWorkResidualEventScratch, batchRunner *DecoderFrameWorkBatchResidualRunner) (DecoderFrameWorkResidualEventRunner, DecoderFrameWorkSideData, error) {
	if batchRunner == nil {
		return DecoderFrameWorkResidualEventRunner{}, DecoderFrameWorkSideData{}, ErrThreadingInvalidBatch
	}
	if decoderFrameWorkResidualScratchTooShort(scratch.Spans, size.Plan.SpanCount) ||
		decoderFrameWorkResidualScratchTooShort(scratch.Jobs, size.Plan.JobCount) ||
		decoderFrameWorkResidualScratchTooShort(scratch.Batches, size.Plan.BatchCount) {
		return DecoderFrameWorkResidualEventRunner{}, DecoderFrameWorkSideData{}, ErrFrameShortBuffer
	}

	boundRunner, err := BindDecoderFrameWorkBatchResidualRunner(size.Runner, scratch.Runner)
	if err != nil {
		return DecoderFrameWorkResidualEventRunner{}, DecoderFrameWorkSideData{}, err
	}
	side, err := BindDecoderFrameWorkResidualEventSideData(sequence, event, scratch.SideData)
	if err != nil {
		return DecoderFrameWorkResidualEventRunner{}, DecoderFrameWorkSideData{}, err
	}
	*batchRunner = boundRunner
	return DecoderFrameWorkResidualEventRunner{
		State:             runtime.State,
		Refs:              runtime.Refs,
		FramePool:         runtime.FramePool,
		Align:             runtime.Align,
		ReferenceSurfaces: runtime.ReferenceSurfaces,
		ReferenceFrames:   runtime.ReferenceFrames,
		Workers:           size.Runner.Workers,
		Spans:             scratch.Spans[:size.Plan.SpanCount],
		Jobs:              scratch.Jobs[:size.Plan.JobCount],
		Batches:           scratch.Batches[:size.Plan.BatchCount],
		Releases:          runtime.Releases,
		WorkerPool:        runtime.WorkerPool,
		BatchRunner:       batchRunner,
		Stats:             runtime.Stats,
	}, side, nil
}

// Run plans and executes one decoder frame-work event using the runner's
// stable caller-owned state. The sequence, event, side-data, and postfilter
// callback remain per-event inputs.
func (r DecoderFrameWorkResidualEventRunner) Run(sequence SequenceHeader, event DecoderEvent, side *DecoderFrameWorkSideData, post DecoderFrameWorkPostFilterFunc) (DecoderFrameWorkEventResult, error) {
	return RunDecoderFrameWorkEventWithResidualRunner(DecoderFrameWorkResidualEventRequest{
		State:             r.State,
		Refs:              r.Refs,
		FramePool:         r.FramePool,
		Sequence:          sequence,
		Event:             event,
		Align:             r.Align,
		ReferenceSurfaces: r.ReferenceSurfaces,
		ReferenceFrames:   r.ReferenceFrames,
		Workers:           r.Workers,
		Spans:             r.Spans,
		Jobs:              r.Jobs,
		Batches:           r.Batches,
		Releases:          r.Releases,
		WorkerPool:        r.WorkerPool,
		Runner:            r.BatchRunner,
		SideData:          side,
		Post:              post,
		Stats:             r.Stats,
	})
}

// RunWithPostFilterRunner plans and executes one decoder frame-work event using
// direct residual and postfilter runners, avoiding callback method values in the
// final-frame path.
func (r DecoderFrameWorkResidualEventRunner) RunWithPostFilterRunner(sequence SequenceHeader, event DecoderEvent, side *DecoderFrameWorkSideData, post DecoderFrameWorkPostFilterRunner) (DecoderFrameWorkEventResult, error) {
	return RunDecoderFrameWorkEventWithResidualRunner(DecoderFrameWorkResidualEventRequest{
		State:             r.State,
		Refs:              r.Refs,
		FramePool:         r.FramePool,
		Sequence:          sequence,
		Event:             event,
		Align:             r.Align,
		ReferenceSurfaces: r.ReferenceSurfaces,
		ReferenceFrames:   r.ReferenceFrames,
		Workers:           r.Workers,
		Spans:             r.Spans,
		Jobs:              r.Jobs,
		Batches:           r.Batches,
		Releases:          r.Releases,
		WorkerPool:        r.WorkerPool,
		Runner:            r.BatchRunner,
		SideData:          side,
		PostRunner:        post,
		Stats:             r.Stats,
	})
}

// DecoderFrameWorkResidualEventScratchLen plans event tile work into
// caller-owned spans/jobs/batches and reports both residual-runner scratch and
// frame-level side-data scratch for the event.
func DecoderFrameWorkResidualEventScratchLen(sequence SequenceHeader, event DecoderEvent, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkResidualEventScratchSize, error) {
	runner, plan, err := DecoderFrameWorkResidualEventRunnerScratchLen(sequence, event, workers, spans, jobs, batches)
	if err != nil {
		return DecoderFrameWorkResidualEventScratchSize{}, err
	}
	sideData, err := DecoderFrameWorkResidualEventSideDataScratchLen(sequence, event)
	if err != nil {
		return DecoderFrameWorkResidualEventScratchSize{}, err
	}
	return DecoderFrameWorkResidualEventScratchSize{
		Runner:   runner,
		SideData: sideData,
		Plan:     plan,
	}, nil
}

// DecoderFrameWorkResidualEventRunnerScratchLen plans event tile work into
// caller-owned spans/jobs/batches and reports the residual runner scratch
// required to execute the planned event.
func DecoderFrameWorkResidualEventRunnerScratchLen(sequence SequenceHeader, event DecoderEvent, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkBatchResidualRunnerScratchSize, DecoderTileWorkPlan, error) {
	plan, err := PlanDecoderTileWork(event, workers, spans, jobs, batches)
	if err != nil {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, DecoderTileWorkPlan{}, err
	}
	batch := DecoderFrameWorkBatch{
		FrameWorkFrameContext: decoderFrameWorkResidualEventContext(sequence, event),
		Jobs:                  jobs[:plan.JobCount],
	}
	size, err := DecoderFrameWorkBatchResidualRunnerScratchLen(batch, workers)
	if err != nil {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, DecoderTileWorkPlan{}, err
	}
	return size, plan, nil
}

// DecoderFrameWorkResidualEventSideDataScratchLen reports frame-level side-data
// scratch for the event's frame size, CDEF, and restoration parameters.
func DecoderFrameWorkResidualEventSideDataScratchLen(sequence SequenceHeader, event DecoderEvent) (DecoderFrameWorkSideDataScratchSize, error) {
	return DecoderFrameWorkSideDataScratchLen(sequence, event.FrameSize, event.CDEF, event.Restoration)
}

// BindDecoderFrameWorkResidualEventSideData binds frame-level side data using
// the event's frame size, CDEF, and restoration parameters.
func BindDecoderFrameWorkResidualEventSideData(sequence SequenceHeader, event DecoderEvent, scratch DecoderFrameWorkSideDataScratch) (DecoderFrameWorkSideData, error) {
	return BindDecoderFrameWorkSideData(sequence, event.FrameSize, event.CDEF, event.Restoration, scratch)
}

// RunDecoderFrameWorkEventWithResidualRunner plans and executes one decoder
// frame-work event using a caller-owned residual runner. If SideData is set, it
// is attached to both the active frame-work state and the residual runner after
// planning and before tile work runs.
func RunDecoderFrameWorkEventWithResidualRunner(req DecoderFrameWorkResidualEventRequest) (DecoderFrameWorkEventResult, error) {
	if req.Runner == nil {
		return DecoderFrameWorkEventResult{}, ErrThreadingInvalidBatch
	}
	if req.Post != nil && req.PostRunner != nil {
		return DecoderFrameWorkEventResult{}, ErrDecoderInvalidFrameWorkState
	}
	event := req.Event
	event.SequenceHeader = req.Sequence

	if err := req.Runner.ResetStats(); err != nil {
		return DecoderFrameWorkEventResult{}, err
	}
	if req.Stats != nil {
		*req.Stats = DecoderFrameWorkTileResidualStats{}
	}

	step, output, err := req.State.PlanEvent(req.Refs, req.FramePool, req.Sequence, event, req.Align, req.ReferenceSurfaces, req.Workers, req.Spans, req.Jobs, req.Batches, req.Releases)
	if err != nil {
		return DecoderFrameWorkEventResult{}, err
	}

	tile, referenceCount, hasTile, err := decoderFrameWorkEventStepTile(step)
	if err != nil {
		return DecoderFrameWorkEventResult{}, err
	}
	if hasTile && req.SideData != nil {
		if err := SetDecoderFrameWorkBatchResidualRunnerSideData(req.Runner, *req.SideData); err != nil {
			return DecoderFrameWorkEventResult{}, err
		}
		if err := SetDecoderFrameWorkSideData(req.State, *req.SideData); err != nil {
			return DecoderFrameWorkEventResult{}, err
		}
	}

	if step.Kind == DecoderFrameWorkStepShowExisting {
		output, err = req.FramePool.Frame(step.ShowExisting.Surface)
		if err != nil {
			return DecoderFrameWorkEventResult{}, err
		}
	}

	var references []*Frame
	if hasTile && tile.JobCount != 0 {
		if output == nil {
			surface, err := decoderFrameWorkEventStepSurface(step)
			if err != nil {
				return DecoderFrameWorkEventResult{}, err
			}
			output, err = req.FramePool.Frame(surface)
			if err != nil {
				return DecoderFrameWorkEventResult{}, err
			}
		}
		if referenceCount != 0 {
			if len(req.ReferenceSurfaces) < referenceCount {
				return DecoderFrameWorkEventResult{}, ErrDecoderSurfaceReferenceBufferTooSmall
			}
			if step.Kind == DecoderFrameWorkStepTile {
				count, err := req.Refs.FrameReferences(event, req.ReferenceSurfaces)
				if err != nil {
					return DecoderFrameWorkEventResult{}, err
				}
				if count != referenceCount {
					return DecoderFrameWorkEventResult{}, ErrDecoderInvalidFrameWorkStep
				}
			}
			count, err := ResolveDecoderFrameReferences(req.FramePool, req.ReferenceSurfaces[:referenceCount], req.ReferenceFrames)
			if err != nil {
				return DecoderFrameWorkEventResult{}, err
			}
			if count != referenceCount {
				return DecoderFrameWorkEventResult{}, ErrDecoderInvalidFrameWorkStep
			}
			references = req.ReferenceFrames[:referenceCount]
		}
	}

	var run DecoderFrameWorkStepResult
	if req.PostRunner != nil {
		run, err = req.State.RunStepWithPayloadContextAndPostFilterRunners(req.Refs, req.FramePool, event, step, req.WorkerPool, output, references, event.Unit.Payload, req.Jobs, req.Batches, req.Releases, req.Runner, req.PostRunner)
	} else {
		run, err = req.State.RunStepWithPayloadContextAndPostFilterRunner(req.Refs, req.FramePool, event, step, req.WorkerPool, output, references, event.Unit.Payload, req.Jobs, req.Batches, req.Releases, req.Runner, req.Post)
	}
	stats, statsErr := req.Runner.TotalStats()
	if req.Stats != nil {
		*req.Stats = stats
	}
	if err != nil {
		return DecoderFrameWorkEventResult{}, err
	}
	if statsErr != nil {
		return DecoderFrameWorkEventResult{}, statsErr
	}
	return DecoderFrameWorkEventResult{
		Step:           step,
		Output:         output,
		ReferenceCount: referenceCount,
		Run:            run,
	}, nil
}

func decoderFrameWorkEventStepTile(step DecoderFrameWorkStep) (DecoderTileWorkPlan, int, bool, error) {
	switch step.Kind {
	case DecoderFrameWorkStepIgnored, DecoderFrameWorkStepDropped, DecoderFrameWorkStepShowExisting:
		return DecoderTileWorkPlan{}, 0, false, nil
	case DecoderFrameWorkStepBegin:
		return step.Begin.Tile, step.Begin.ReferenceCount, true, nil
	case DecoderFrameWorkStepTile:
		return step.Tile.Tile, step.Tile.ReferenceCount, true, nil
	default:
		return DecoderTileWorkPlan{}, 0, false, ErrDecoderInvalidTileWork
	}
}

func decoderFrameWorkResidualEventContext(sequence SequenceHeader, event DecoderEvent) DecoderFrameWorkFrameContext {
	return DecoderFrameWorkFrameContext{
		Sequence:            DecoderFrameWorkSequenceContextFromHeader(sequence),
		FrameHeader:         event.FrameHeader,
		FrameSize:           event.FrameSize,
		TileInfo:            event.TileInfo,
		Quantization:        event.Quantization,
		Segmentation:        event.Segmentation,
		Delta:               event.Delta,
		LoopFilter:          event.LoopFilter,
		CDEF:                event.CDEF,
		Restoration:         event.Restoration,
		TransformRef:        event.TransformRef,
		SkipMode:            event.SkipMode,
		FrameMode:           event.FrameMode,
		GlobalMotion:        event.GlobalMotion,
		FilmGrain:           event.FilmGrain,
		ReferenceOrderHints: event.ReferenceOrderHints,
	}
}

func decoderFrameWorkEventStepSurface(step DecoderFrameWorkStep) (int, error) {
	switch step.Kind {
	case DecoderFrameWorkStepBegin:
		return step.Begin.Surface, nil
	case DecoderFrameWorkStepTile:
		return step.Tile.Surface, nil
	case DecoderFrameWorkStepShowExisting:
		return step.ShowExisting.Surface, nil
	default:
		return -1, ErrDecoderInvalidFrameWorkStep
	}
}
