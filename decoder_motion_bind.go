package goav1

func DecoderFrameWorkReferenceMVFrameShape(sequence SequenceHeader, size FrameSize) (cols int, rows int, length int, err error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.ReferenceMVFrameShape()
}

func BindDecoderFrameWorkReferenceMVFrame(sequence SequenceHeader, size FrameSize, entries []TileReferenceMVEntry) (TileReferenceMVFrame, error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.BindReferenceMVFrame(entries)
}

func BindDecoderFrameWorkTemporalMotionField(sequence SequenceHeader, size FrameSize, entries []TileTemporalMotionEntry) (TileTemporalMotionField, error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.BindTemporalMotionField(entries)
}
