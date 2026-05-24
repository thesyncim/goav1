package threading

import "github.com/thesyncim/goav1/internal/av1/tile"

// ReferenceMVFrameShape returns the frame-level half-MI MV_REF grid dimensions
// used by libaom for current-frame ref-frame-MVS side data.
func (b FrameWorkBatch) ReferenceMVFrameShape() (cols int, rows int, length int, err error) {
	miRows, miCols, err := b.referenceMVFrameMIExtents()
	if err != nil {
		return 0, 0, 0, err
	}
	rows = int((miRows + 1) >> 1)
	cols = int((miCols + 1) >> 1)
	length, err = tile.ReferenceMVFrameEntries(miRows, miCols)
	if err != nil {
		return 0, 0, 0, ErrInvalidBatch
	}
	return cols, rows, length, nil
}

// BindReferenceMVFrame validates and clears caller-owned MV_REF storage for the
// current frame. Callers can assign the returned frame to FrameWorkBatch's
// CurrentMVFrame before deriving job block-loop requests.
func (b FrameWorkBatch) BindReferenceMVFrame(entries []tile.ReferenceMVEntry) (tile.ReferenceMVFrame, error) {
	miRows, miCols, err := b.referenceMVFrameMIExtents()
	if err != nil {
		return tile.ReferenceMVFrame{}, err
	}
	need, err := tile.ReferenceMVFrameEntries(miRows, miCols)
	if err != nil {
		return tile.ReferenceMVFrame{}, ErrInvalidBatch
	}
	if len(entries) < need {
		return tile.ReferenceMVFrame{}, ErrInvalidBatch
	}
	var frame tile.ReferenceMVFrame
	if err := frame.Init(miRows, miCols, entries); err != nil {
		return tile.ReferenceMVFrame{}, ErrInvalidBatch
	}
	return frame, nil
}

func (b FrameWorkBatch) referenceMVFrameMIExtents() (miRows uint32, miCols uint32, err error) {
	if !b.Sequence.Valid() || b.FrameSize.CodedWidth == 0 || b.FrameSize.Height == 0 {
		return 0, 0, ErrInvalidBatch
	}
	miCols, ok := frameWorkMIExtent(b.FrameSize.CodedWidth)
	if !ok {
		return 0, 0, ErrInvalidBatch
	}
	miRows, ok = frameWorkMIExtent(b.FrameSize.Height)
	if !ok {
		return 0, 0, ErrInvalidBatch
	}
	return miRows, miCols, nil
}
