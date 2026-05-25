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
}

// RunDecoderFrameWorkEventWithResidualRunner plans and executes one decoder
// frame-work event using a caller-owned residual runner. If SideData is set, it
// is attached to both the active frame-work state and the residual runner after
// planning and before tile work runs.
func RunDecoderFrameWorkEventWithResidualRunner(req DecoderFrameWorkResidualEventRequest) (DecoderFrameWorkEventResult, error) {
	if req.Runner == nil {
		return DecoderFrameWorkEventResult{}, ErrThreadingInvalidBatch
	}
	step, output, err := req.State.PlanEvent(req.Refs, req.FramePool, req.Sequence, req.Event, req.Align, req.ReferenceSurfaces, req.Workers, req.Spans, req.Jobs, req.Batches, req.Releases)
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
				count, err := req.Refs.FrameReferences(req.Event, req.ReferenceSurfaces)
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

	run, err := req.State.RunStepWithPayloadContextAndPostFilter(req.Refs, req.FramePool, req.Event, step, req.WorkerPool, output, references, req.Event.Unit.Payload, req.Jobs, req.Batches, req.Releases, req.Runner.Run, req.Post)
	if err != nil {
		return DecoderFrameWorkEventResult{}, err
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
