package tile

import "github.com/thesyncim/goav1/internal/av1/motion"

const refMVSLimit = (1 << 12) - 1

// ReferenceMVFrame stores the current frame's ref-frame-MVS side data at the
// half-MI granularity used by libaom's MV_REF array. Entries is caller-owned.
type ReferenceMVFrame struct {
	Rows    int
	Cols    int
	Stride  int
	Entries []ReferenceMVEntry
}

// ReferenceMVEntry mirrors libaom's MV_REF payload.
type ReferenceMVEntry struct {
	Ref   ReferenceFrame
	MV    motion.Vector
	Valid bool
}

// ReferenceMVFrameBlockRequest describes one decoded block to copy into the
// current-frame ref-MV side data.
type ReferenceMVFrameBlockRequest struct {
	MICol uint32
	MIRow uint32

	VisibleW4 uint8
	VisibleH4 uint8

	Prediction   BlockPredictionModeResult
	RefFrameSide [referenceFrameCount]int8
}

// ReferenceMVFrameEntries returns the number of entries needed for a frame with
// miRows x miCols 4x4 units.
func ReferenceMVFrameEntries(miRows uint32, miCols uint32) (int, error) {
	if miRows == 0 || miCols == 0 {
		return 0, ErrInvalidDecodeState
	}
	rows := int((miRows + 1) >> 1)
	cols := int((miCols + 1) >> 1)
	return rows * cols, nil
}

// Init attaches caller-owned storage and clears it to NONE, matching libaom's
// per-frame MV_REF side data initialization.
func (f *ReferenceMVFrame) Init(miRows uint32, miCols uint32, entries []ReferenceMVEntry) error {
	if f == nil {
		return ErrInvalidDecodeState
	}
	need, err := ReferenceMVFrameEntries(miRows, miCols)
	if err != nil {
		return err
	}
	if len(entries) < need {
		return ErrInvalidDecodeState
	}
	rows := int((miRows + 1) >> 1)
	cols := int((miCols + 1) >> 1)
	f.Rows = rows
	f.Cols = cols
	f.Stride = cols
	f.Entries = entries[:need]
	for i := range f.Entries {
		f.Entries[i] = ReferenceMVEntry{Ref: ReferenceFrameNone}
	}
	return nil
}

// MarkBlock ports libaom's intra_copy_frame_mvs()/av1_copy_frame_mvs() update
// for one decoded block.
func (f *ReferenceMVFrame) MarkBlock(req ReferenceMVFrameBlockRequest) error {
	if err := f.Validate(); err != nil {
		return err
	}
	if req.VisibleW4 == 0 || req.VisibleH4 == 0 {
		return ErrInvalidDecodeState
	}
	col := int(req.MICol >> 1)
	row := int(req.MIRow >> 1)
	w := (int(req.VisibleW4) + 1) >> 1
	h := (int(req.VisibleH4) + 1) >> 1
	if col < 0 || row < 0 || w <= 0 || h <= 0 || col+w > f.Cols || row+h > f.Rows {
		return ErrInvalidDecodeState
	}

	entry := ReferenceMVEntry{Ref: ReferenceFrameNone}
	if req.Prediction.Valid && !req.Prediction.Intra && req.Prediction.InterMotionValid {
		entry = referenceMVEntryForInter(req.Prediction.InterMotion, req.RefFrameSide)
	}
	for y := 0; y < h; y++ {
		line := f.Entries[(row+y)*f.Stride+col : (row+y)*f.Stride+col+w]
		for x := range line {
			line[x] = entry
		}
	}
	return nil
}

// Validate checks that f carries a well-formed caller-owned MV_REF grid.
func (f *ReferenceMVFrame) Validate() error {
	if f == nil || f.Rows <= 0 || f.Cols <= 0 || f.Stride < f.Cols ||
		len(f.Entries) < (f.Rows-1)*f.Stride+f.Cols {
		return ErrInvalidDecodeState
	}
	return nil
}

func referenceMVEntryForInter(result InterMotionResult, refFrameSide [referenceFrameCount]int8) ReferenceMVEntry {
	entry := ReferenceMVEntry{Ref: ReferenceFrameNone}
	for i, ref := range result.References.Ref {
		if !ref.Valid() || refFrameSide[ref] != 0 {
			continue
		}
		mv := result.MV[i]
		if refMVOutOfRange(mv.Row) || refMVOutOfRange(mv.Col) {
			continue
		}
		entry = ReferenceMVEntry{Ref: ref, MV: mv, Valid: true}
	}
	return entry
}

func refMVOutOfRange(v int32) bool {
	if v < 0 {
		return -int64(v) > refMVSLimit
	}
	return v > refMVSLimit
}
