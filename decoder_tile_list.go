package goav1

import (
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
)

// decoderTileListEntryDecodeRunnerScratchLen reports residual-runner scratch
// for the synthetic tile-group event produced by PlanDecoderTileListEntryDecode.
// It also returns the tile work plan written into spans, jobs, and batches.
func decoderTileListEntryDecodeRunnerScratchLen(event *internaldecoder.Event, entryIndex int, workers int, spans []internalparser.TileSpan, jobs []internaltile.Job, batches []internalthreading.Batch) (decoderFrameWorkBatchResidualRunnerScratchSize, internaldecoder.TileWorkPlan, error) {
	if event.Kind != internaldecoder.EventTileList {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, internaldecoder.TileWorkPlan{}, internaldecoder.ErrInvalidSurfaceEvent
	}
	if event.TileListErr != nil {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, internaldecoder.TileWorkPlan{}, event.TileListErr
	}
	plan, err := internaldecoder.PlanTileListEntryWork(event.TileList, event.TileInfo, entryIndex, workers, spans, jobs, batches)
	if err != nil {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, internaldecoder.TileWorkPlan{}, err
	}
	batch := internaldecoder.FrameWorkBatch{
		FrameWorkFrameContext: decoderFrameWorkResidualEventContext(event.SequenceHeader, event),
		Jobs:                  jobs[:plan.JobCount],
	}
	size, err := decoderFrameWorkBatchResidualRunnerScratchLen(batch, workers)
	if err != nil {
		return decoderFrameWorkBatchResidualRunnerScratchSize{}, internaldecoder.TileWorkPlan{}, err
	}
	return size, plan, nil
}
