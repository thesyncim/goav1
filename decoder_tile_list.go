package goav1

import internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"

// DecoderTileListEntryDecodePlan is the residual-runner contract for decoding
// one tile_list_obu() entry. Event is a synthetic tile-group event whose payload
// is the entry's raw coded tile bytes; Step binds the single-tile work plan and
// exposes exactly one reference, LAST_FRAME, backed by the entry's external
// anchor frame.
type DecoderTileListEntryDecodePlan = internaldecoder.TileListEntryDecodePlan

// PlanDecoderTileListEntryDecode prepares one tile-list entry for residual
// decoding. references receives the libaom-shaped reference table for the
// synthetic entry decode: references[0] is LAST_FRAME and points at
// anchors[anchor_frame_idx]. The returned Event and Step can be supplied to the
// frame-work residual runner with output set to a caller-owned scratch
// reconstruction frame; after the entry decodes, copy Region into the tile-list
// output mosaic with CopyTileListEntryToOutputFrame.
func PlanDecoderTileListEntryDecode(event DecoderEvent, entryIndex int, surface int, anchors []*Frame, references []*Frame, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderTileListEntryDecodePlan, error) {
	return internaldecoder.PlanTileListEntryDecode(event, entryIndex, surface, anchors, references, workers, spans, jobs, batches)
}

// DecoderTileListEntryDecodeRunnerScratchLen reports residual-runner scratch
// for the synthetic tile-group event produced by PlanDecoderTileListEntryDecode.
// It also returns the tile work plan written into spans, jobs, and batches.
func DecoderTileListEntryDecodeRunnerScratchLen(event DecoderEvent, entryIndex int, workers int, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkBatchResidualRunnerScratchSize, DecoderTileWorkPlan, error) {
	if event.Kind != DecoderEventTileList {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, DecoderTileWorkPlan{}, ErrDecoderInvalidSurfaceEvent
	}
	if event.TileListErr != nil {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, DecoderTileWorkPlan{}, event.TileListErr
	}
	plan, err := PlanDecoderTileListEntryWork(event.TileList, event.TileInfo, entryIndex, workers, spans, jobs, batches)
	if err != nil {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, DecoderTileWorkPlan{}, err
	}
	batch := DecoderFrameWorkBatch{
		FrameWorkFrameContext: decoderFrameWorkResidualEventContext(event.SequenceHeader, event),
		Jobs:                  jobs[:plan.JobCount],
	}
	size, err := DecoderFrameWorkBatchResidualRunnerScratchLen(batch, workers)
	if err != nil {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, DecoderTileWorkPlan{}, err
	}
	return size, plan, nil
}
