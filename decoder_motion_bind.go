package goav1

// DecoderFrameWorkReferenceMVFrameShape returns the column count, row count,
// and total entry count required to back a TileReferenceMVFrame for the given
// sequence and frame size. Callers use the returned length to size the
// entries slice passed to BindDecoderFrameWorkReferenceMVFrame.
func DecoderFrameWorkReferenceMVFrameShape(sequence SequenceHeader, size FrameSize) (cols int, rows int, length int, err error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.ReferenceMVFrameShape()
}

// BindDecoderFrameWorkReferenceMVFrame wires entries into a
// TileReferenceMVFrame matching (sequence, size). The slice must be at least
// the length reported by DecoderFrameWorkReferenceMVFrameShape.
func BindDecoderFrameWorkReferenceMVFrame(sequence SequenceHeader, size FrameSize, entries []TileReferenceMVEntry) (TileReferenceMVFrame, error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.BindReferenceMVFrame(entries)
}

// BindDecoderFrameWorkTemporalMotionField wires entries into a
// TileTemporalMotionField matching (sequence, size) for temporal motion-vector
// projection.
func BindDecoderFrameWorkTemporalMotionField(sequence SequenceHeader, size FrameSize, entries []TileTemporalMotionEntry) (TileTemporalMotionField, error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.BindTemporalMotionField(entries)
}
