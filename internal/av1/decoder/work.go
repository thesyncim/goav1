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

// FrameWorkState is caller-owned lifecycle state for one in-flight frame. It
// records the acquired output surface between the frame begin event, any later
// tile-group continuation events, and the final reference publication step.
type FrameWorkState struct {
	Surface        int
	ReferenceCount int

	active bool
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
