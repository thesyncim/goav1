package decoder

import (
	"errors"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

var ErrInvalidTileWork = errors.New("decoder: invalid tile work")

// TileWorkPlan is the caller-buffer work description derived from one tile
// group event.
type TileWorkPlan struct {
	SpanCount  int
	JobCount   int
	BatchCount int
}

// FrameWorkPlan is the caller-buffer work description for one frame-begin
// event. Frame-header events have no tile work yet; frame events may carry an
// implicit final tile group.
type FrameWorkPlan struct {
	Surface        int
	ReferenceCount int
	Tile           TileWorkPlan
}

// FrameTileWorkPlan binds a tile-group work plan to the output surface and
// resolved reference count chosen when the frame began.
type FrameTileWorkPlan struct {
	Surface        int
	ReferenceCount int
	Tile           TileWorkPlan
}

// ShowExistingFrameWorkPlan is the caller-buffer work result for a
// show-existing-frame event. Surface is the frame-pool slot to output.
type ShowExistingFrameWorkPlan struct {
	Surface          int
	ReleaseCount     int
	DroppedFrameWork bool
}

// FrameWorkStepKind identifies the action produced by PlanEvent.
type FrameWorkStepKind uint8

const (
	// FrameWorkStepIgnored means the event did not affect frame work.
	FrameWorkStepIgnored FrameWorkStepKind = iota
	// FrameWorkStepDropped means active incomplete frame work was aborted.
	FrameWorkStepDropped
	// FrameWorkStepBegin means a new output surface was acquired.
	FrameWorkStepBegin
	// FrameWorkStepTile means continuation tile work was planned.
	FrameWorkStepTile
	// FrameWorkStepShowExisting means an existing reference surface is output.
	FrameWorkStepShowExisting
)

// FrameWorkStep is the caller-buffer result of applying one decoder event to
// frame-work state. The active frame should be finished separately after the
// caller executes any tile work reported by Begin or Tile.
type FrameWorkStep struct {
	Kind             FrameWorkStepKind
	DroppedFrameWork bool

	Begin        FrameWorkPlan
	Tile         FrameTileWorkPlan
	ShowExisting ShowExistingFrameWorkPlan
}

// FrameWorkState is caller-owned lifecycle state for one in-flight frame. It
// records the acquired output surface between the frame begin event, any later
// tile-group continuation events, and the final reference publication step.
type FrameWorkState struct {
	Surface        int
	ReferenceCount int

	active bool
}

// EventDropsFrameWork reports whether a stream event implies that any active
// incomplete frame work should be discarded. This matches Stream's pending
// frame reset points.
func EventDropsFrameWork(event Event) bool {
	return event.NewCodedVideoSequence || event.NewTemporalUnit || event.Kind == EventExistingFrame
}

// EventCompletesFrameWork reports whether event carries the final tile group
// for an in-flight frame and is therefore ready for reference publication
// after tile decoding succeeds.
func EventCompletesFrameWork(event Event) bool {
	return (event.Kind == EventFrame || event.Kind == EventTileGroup) && event.TileGroup.Final
}

// Active reports whether a frame has begun and not yet been finished, aborted,
// or reset.
func (s *FrameWorkState) Active() bool {
	return s != nil && s.active
}

// Reset clears any in-flight frame work state without touching frame-pool or
// reference ownership. Callers should normally use Finish for successfully
// decoded final tile groups and Abort for incomplete or discarded frames.
func (s *FrameWorkState) Reset() {
	if s == nil {
		return
	}
	*s = FrameWorkState{}
}

// Begin acquires the output surface and records the active frame work state.
func (s *FrameWorkState) Begin(refs *SurfaceReferences, pool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, references []int, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch) (FrameWorkPlan, *frame.Frame, error) {
	if s == nil || s.active {
		return FrameWorkPlan{}, nil, ErrInvalidFrameWorkState
	}
	plan, output, err := BeginFrameWork(refs, pool, sequence, event, align, references, workers, spans, jobs, batches)
	if err != nil {
		return FrameWorkPlan{}, nil, err
	}
	s.Surface = plan.Surface
	s.ReferenceCount = plan.ReferenceCount
	s.active = true
	return plan, output, nil
}

// PlanTile plans tile work for a continuation tile group using the active
// frame's output surface and resolved reference count.
func (s *FrameWorkState) PlanTile(event Event, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch) (FrameTileWorkPlan, error) {
	if s == nil || !s.active {
		return FrameTileWorkPlan{}, ErrInvalidFrameWorkState
	}
	return PlanFrameTileWork(event, s.Surface, s.ReferenceCount, workers, spans, jobs, batches)
}

// Finish applies the final frame event to surface references and releases any
// overwritten frame-pool slots atomically. State is cleared only after a
// successful finish so callers can recover or retry after transient release
// failures.
func (s *FrameWorkState) Finish(refs *SurfaceReferences, pool *frame.Pool, event Event, releases []int) (int, error) {
	if s == nil || !s.active {
		return 0, ErrInvalidFrameWorkState
	}
	count, err := FinishFrameSurface(refs, pool, event, s.Surface, releases)
	if err != nil {
		return 0, err
	}
	s.Reset()
	return count, nil
}

// Abort releases the active output surface without publishing it to reference
// slots. State is cleared only after the pool accepts the release.
func (s *FrameWorkState) Abort(pool *frame.Pool) error {
	if s == nil || !s.active {
		return ErrInvalidFrameWorkState
	}
	if err := pool.Release(s.Surface); err != nil {
		return err
	}
	s.Reset()
	return nil
}

// AbortIfEventDropsFrameWork aborts active frame work when event marks a
// stream boundary that invalidates incomplete frame data. It is a no-op for
// inactive state and non-boundary events.
func (s *FrameWorkState) AbortIfEventDropsFrameWork(pool *frame.Pool, event Event) (bool, error) {
	if !EventDropsFrameWork(event) || s == nil || !s.active {
		return false, nil
	}
	if err := s.Abort(pool); err != nil {
		return false, err
	}
	return true, nil
}

// FinishIfEventCompletesFrameWork publishes active frame work only when event
// carries a final tile group. It is a no-op for non-final events.
func (s *FrameWorkState) FinishIfEventCompletesFrameWork(refs *SurfaceReferences, pool *frame.Pool, event Event, releases []int) (bool, int, error) {
	if !EventCompletesFrameWork(event) {
		return false, 0, nil
	}
	if s == nil || !s.active {
		return false, 0, ErrInvalidFrameWorkState
	}
	count, err := s.Finish(refs, pool, event, releases)
	if err != nil {
		return false, 0, err
	}
	return true, count, nil
}

// ShowExisting resolves a show-existing-frame event to an output surface,
// aborts any active incomplete frame work, and applies any AV1 key-frame
// reference reset. Active frame work is dropped only after the show-existing
// event validates and the abort succeeds.
func (s *FrameWorkState) ShowExisting(refs *SurfaceReferences, pool *frame.Pool, event Event, releases []int) (ShowExistingFrameWorkPlan, error) {
	if event.Kind != EventExistingFrame || !event.FrameHeader.ShowExistingFrame {
		return ShowExistingFrameWorkPlan{}, ErrInvalidSurfaceEvent
	}
	if refs == nil {
		return ShowExistingFrameWorkPlan{}, ErrInvalidSurfaceReference
	}

	next := *refs
	surface, releaseCount, err := next.ShowExistingFrame(event, releases)
	if err != nil {
		return ShowExistingFrameWorkPlan{}, err
	}
	if releaseCount != 0 && pool == nil {
		return ShowExistingFrameWorkPlan{}, frame.ErrInvalidPool
	}

	dropped := false
	if s != nil && s.active {
		if err := s.Abort(pool); err != nil {
			return ShowExistingFrameWorkPlan{}, err
		}
		dropped = true
	}
	if releaseCount != 0 {
		if err := pool.ReleaseMany(releases[:releaseCount]); err != nil {
			return ShowExistingFrameWorkPlan{}, err
		}
	}
	*refs = next
	return ShowExistingFrameWorkPlan{
		Surface:          surface,
		ReleaseCount:     releaseCount,
		DroppedFrameWork: dropped,
	}, nil
}

// PlanEvent applies one parsed decoder event to frame-work state and returns
// the work that should be executed by the caller. It does not finish final tile
// groups; callers should execute the planned tile work first, then call
// FinishIfEventCompletesFrameWork with the same event.
func (s *FrameWorkState) PlanEvent(refs *SurfaceReferences, pool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, references []int, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch, releases []int) (FrameWorkStep, *frame.Frame, error) {
	if event.Kind == EventExistingFrame {
		plan, err := s.ShowExisting(refs, pool, event, releases)
		if err != nil {
			return FrameWorkStep{}, nil, err
		}
		return FrameWorkStep{
			Kind:             FrameWorkStepShowExisting,
			DroppedFrameWork: plan.DroppedFrameWork,
			ShowExisting:     plan,
		}, nil, nil
	}

	dropped := false
	if EventDropsFrameWork(event) {
		var err error
		dropped, err = s.AbortIfEventDropsFrameWork(pool, event)
		if err != nil {
			return FrameWorkStep{}, nil, err
		}
		if event.Kind != EventFrameHeader && event.Kind != EventFrame {
			if dropped {
				return FrameWorkStep{Kind: FrameWorkStepDropped, DroppedFrameWork: true}, nil, nil
			}
			return FrameWorkStep{}, nil, nil
		}
	}

	switch event.Kind {
	case EventFrameHeader, EventFrame:
		plan, output, err := s.Begin(refs, pool, sequence, event, align, references, workers, spans, jobs, batches)
		if err != nil {
			return FrameWorkStep{}, nil, err
		}
		return FrameWorkStep{
			Kind:             FrameWorkStepBegin,
			DroppedFrameWork: dropped,
			Begin:            plan,
		}, output, nil

	case EventTileGroup:
		plan, err := s.PlanTile(event, workers, spans, jobs, batches)
		if err != nil {
			return FrameWorkStep{}, nil, err
		}
		return FrameWorkStep{
			Kind: FrameWorkStepTile,
			Tile: plan,
		}, nil, nil
	}

	return FrameWorkStep{}, nil, nil
}

// ExecuteTileWork dispatches the planned tile jobs through a reusable worker
// pool. Only the caller-owned job and batch ranges named by plan are used.
func ExecuteTileWork(plan TileWorkPlan, pool *threading.Pool, jobs []tile.Job, batches []threading.Batch, fn threading.BatchFunc) error {
	if err := validateTileWorkPlan(plan, jobs, batches); err != nil {
		return err
	}
	if plan.JobCount == 0 {
		return nil
	}
	return pool.Execute(batches[:plan.BatchCount], jobs[:plan.JobCount], fn)
}

// ExecuteFrameWorkStep dispatches any tile work carried by step. Steps that do
// not carry tile work are successful no-ops; the boolean reports whether tile
// work actually ran.
func ExecuteFrameWorkStep(step FrameWorkStep, pool *threading.Pool, jobs []tile.Job, batches []threading.Batch, fn threading.BatchFunc) (bool, error) {
	var plan TileWorkPlan
	switch step.Kind {
	case FrameWorkStepIgnored, FrameWorkStepDropped, FrameWorkStepShowExisting:
		return false, nil
	case FrameWorkStepBegin:
		plan = step.Begin.Tile
	case FrameWorkStepTile:
		plan = step.Tile.Tile
	default:
		return false, ErrInvalidTileWork
	}

	if err := ExecuteTileWork(plan, pool, jobs, batches, fn); err != nil {
		return false, err
	}
	return plan.JobCount != 0, nil
}

// BeginFrameWork resolves frame references, plans any inline tile work, and
// acquires the output frame surface. For frame OBUs, tile work is validated
// before acquiring a pool slot so malformed tile data leaves surface ownership
// untouched.
func BeginFrameWork(refs *SurfaceReferences, pool *frame.Pool, sequence parser.SequenceHeader, event Event, align int, references []int, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch) (FrameWorkPlan, *frame.Frame, error) {
	if event.Kind != EventFrameHeader && event.Kind != EventFrame {
		return FrameWorkPlan{}, nil, ErrInvalidSurfaceEvent
	}

	var tilePlan TileWorkPlan
	var err error
	if event.Kind == EventFrame {
		tilePlan, err = PlanTileWork(event, workers, spans, jobs, batches)
		if err != nil {
			return FrameWorkPlan{}, nil, err
		}
	}

	surface, output, refCount, err := BeginFrameSurface(refs, pool, sequence, event, align, references)
	if err != nil {
		return FrameWorkPlan{}, nil, err
	}

	return FrameWorkPlan{
		Surface:        surface,
		ReferenceCount: refCount,
		Tile:           tilePlan,
	}, output, nil
}

// PlanFrameTileWork plans a later tile-group event for an already-begun frame.
// The caller carries surface and reference count from BeginFrameWork.
func PlanFrameTileWork(event Event, surface int, referenceCount int, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch) (FrameTileWorkPlan, error) {
	if event.Kind != EventTileGroup {
		return FrameTileWorkPlan{}, ErrInvalidTileWork
	}
	if surface < 0 || referenceCount < 0 || referenceCount > parser.InterRefsPerFrame {
		return FrameTileWorkPlan{}, ErrInvalidTileWork
	}
	tilePlan, err := PlanTileWork(event, workers, spans, jobs, batches)
	if err != nil {
		return FrameTileWorkPlan{}, err
	}
	return FrameTileWorkPlan{
		Surface:        surface,
		ReferenceCount: referenceCount,
		Tile:           tilePlan,
	}, nil
}

// PlanTileWork turns an EventFrame or EventTileGroup into tile payload spans,
// tile jobs, and deterministic worker batches. All output storage is
// caller-owned.
func PlanTileWork(event Event, workers int, spans []parser.TileSpan, jobs []tile.Job, batches []threading.Batch) (TileWorkPlan, error) {
	if event.Kind != EventFrame && event.Kind != EventTileGroup {
		return TileWorkPlan{}, ErrInvalidTileWork
	}

	spanCount, err := parser.SplitTileGroup(event.Unit.Payload, event.TileInfo, event.TileGroup, spans)
	if err != nil {
		return TileWorkPlan{}, err
	}
	jobCount, err := tile.BuildJobs(jobs, event.TileInfo, spans[:spanCount])
	if err != nil {
		return TileWorkPlan{}, err
	}
	batchCount, err := threading.BuildBatches(batches, jobs[:jobCount], workers)
	if err != nil {
		return TileWorkPlan{}, err
	}
	return TileWorkPlan{
		SpanCount:  spanCount,
		JobCount:   jobCount,
		BatchCount: batchCount,
	}, nil
}

func validateTileWorkPlan(plan TileWorkPlan, jobs []tile.Job, batches []threading.Batch) error {
	if plan.SpanCount < 0 ||
		plan.JobCount < 0 ||
		plan.BatchCount < 0 ||
		plan.SpanCount != plan.JobCount ||
		plan.JobCount > len(jobs) ||
		plan.BatchCount > len(batches) ||
		plan.BatchCount > plan.JobCount {
		return ErrInvalidTileWork
	}
	if plan.JobCount == 0 {
		if plan.BatchCount != 0 {
			return ErrInvalidTileWork
		}
		return nil
	}
	if plan.BatchCount == 0 {
		return ErrInvalidTileWork
	}
	return nil
}
