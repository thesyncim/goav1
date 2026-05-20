package decoder

import (
	"errors"

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
