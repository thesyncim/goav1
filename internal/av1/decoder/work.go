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
