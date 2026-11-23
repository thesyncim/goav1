package goav1

import (
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

// decoderFrameWorkReferenceMVFrameShape returns the column count, row count,
// and total entry count required to back a TileReferenceMVFrame for the given
// sequence and frame size. Callers use the returned length to size the
// entries slice passed to bindDecoderFrameWorkReferenceMVFrame.
func decoderFrameWorkReferenceMVFrameShape(sequence SequenceHeader, size FrameSize) (cols int, rows int, length int, err error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.ReferenceMVFrameShape()
}

// bindDecoderFrameWorkReferenceMVFrame wires entries into a
// TileReferenceMVFrame matching (sequence, size). The slice must be at least
// the length reported by decoderFrameWorkReferenceMVFrameShape.
func bindDecoderFrameWorkReferenceMVFrame(sequence SequenceHeader, size FrameSize, entries []internaltile.ReferenceMVEntry) (internaltile.ReferenceMVFrame, error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.BindReferenceMVFrame(entries)
}

// bindDecoderFrameWorkTemporalMotionField wires entries into a
// TileTemporalMotionField matching (sequence, size) for temporal motion-vector
// projection.
func bindDecoderFrameWorkTemporalMotionField(sequence SequenceHeader, size FrameSize, entries []internaltile.TemporalMotionEntry) (internaltile.TemporalMotionField, error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.BindTemporalMotionField(entries)
}
