package tile

import "github.com/thesyncim/goav1/internal/av1/motion"

const refMVSLimit = (1 << 12) - 1

// ReferenceMVFrame stores the current frame's ref-frame-MVS side data at the
// half-MI granularity used by libaom's MV_REF array. Entries is caller-owned.
type ReferenceMVFrame struct {
	Rows    uint16
	Cols    uint16
	Stride  uint16
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
	MICol uint16
	MIRow uint16

	VisibleW4 uint8
	VisibleH4 uint8

	Prediction   BlockPredictionModeResult
	RefFrameSide [referenceFrameCount]int8
}

// ReferenceMVFrameEntries returns the number of entries needed for a frame with
// miRows x miCols 4x4 units.
func ReferenceMVFrameEntries(miRows uint32, miCols uint32) (int, error) {
	_, _, need, err := referenceMVHalfMIGridShape(miRows, miCols)
	return need, err
}

func referenceMVHalfMIGridShape(miRows uint32, miCols uint32) (uint16, uint16, int, error) {
	if miRows == 0 || miCols == 0 {
		return 0, 0, 0, ErrInvalidDecodeState
	}
	rows32 := (miRows >> 1) + (miRows & 1)
	cols32 := (miCols >> 1) + (miCols & 1)
	if rows32 == 0 || cols32 == 0 || rows32 > uint32(^uint16(0)) || cols32 > uint32(^uint16(0)) {
		return 0, 0, 0, ErrInvalidDecodeState
	}
	need64 := uint64(rows32) * uint64(cols32)
	if need64 > uint64(^uint(0)>>1) {
		return 0, 0, 0, ErrInvalidDecodeState
	}
	return uint16(rows32), uint16(cols32), int(need64), nil
}

// Init attaches caller-owned storage and clears it to NONE, matching libaom's
// per-frame MV_REF side data initialization.
func (f *ReferenceMVFrame) Init(miRows uint32, miCols uint32, entries []ReferenceMVEntry) error {
	if f == nil {
		return ErrInvalidDecodeState
	}
	rows, cols, need, err := referenceMVHalfMIGridShape(miRows, miCols)
	if err != nil {
		return err
	}
	if len(entries) < need {
		return ErrInvalidDecodeState
	}
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
	return f.MarkBlockPtr(req.MICol, req.MIRow, req.VisibleW4, req.VisibleH4, &req.Prediction, req.RefFrameSide)
}

// MarkBlockPtr is MarkBlock without copying the large block prediction result.
func (f *ReferenceMVFrame) MarkBlockPtr(miCol uint16, miRow uint16, visibleW4 uint8, visibleH4 uint8, prediction *BlockPredictionModeResult, refFrameSide [referenceFrameCount]int8) error {
	if err := f.Validate(); err != nil {
		return err
	}
	if visibleW4 == 0 || visibleH4 == 0 {
		return ErrInvalidDecodeState
	}
	col := int(miCol >> 1)
	row := int(miRow >> 1)
	w := (int(visibleW4) + 1) >> 1
	h := (int(visibleH4) + 1) >> 1
	cols := int(f.Cols)
	rows := int(f.Rows)
	stride := int(f.Stride)
	if w <= 0 || h <= 0 || col+w > cols || row+h > rows {
		return ErrInvalidDecodeState
	}

	entry := ReferenceMVEntry{Ref: ReferenceFrameNone}
	if prediction != nil && prediction.Valid && !prediction.Intra && prediction.InterMotionValid {
		entry = referenceMVEntryForInter(prediction.InterMotion, refFrameSide)
	}
	for y := range h {
		line := f.Entries[(row+y)*stride+col : (row+y)*stride+col+w]
		for x := range line {
			line[x] = entry
		}
	}
	return nil
}

// Validate checks that f carries a well-formed caller-owned MV_REF grid.
func (f *ReferenceMVFrame) Validate() error {
	if f == nil || f.Rows <= 0 || f.Cols <= 0 || f.Stride < f.Cols ||
		len(f.Entries) < (int(f.Rows)-1)*int(f.Stride)+int(f.Cols) {
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

func refMVOutOfRange(v int16) bool {
	if v < 0 {
		return -int(v) > refMVSLimit
	}
	return int(v) > refMVSLimit
}
