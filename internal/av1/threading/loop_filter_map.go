package threading

import (
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// FrameWorkLoopFilterBlockRecord stores the block-local syntax needed to build
// loop-filter edge masks after tile reconstruction.
type FrameWorkLoopFilterBlockRecord struct {
	Valid bool

	Block         tile.BlockVisit
	TransformTree tile.TransformTreeResult

	SegmentID     uint8
	SkipTransform bool
	Intra         bool
	RefFrame      int
	Mode          loopfilter.ModeDeltaClass

	DeltaLFFromBase int8
	DeltaLF         [tile.FrameLoopFilterCount]int8
}

// FrameWorkLoopFilterMap stores block-local loop-filter data in frame-level MI
// grid order. Records is caller-owned so tile workers can fill the map without
// allocating in the block loop.
type FrameWorkLoopFilterMap struct {
	Records []FrameWorkLoopFilterBlockRecord
	Stride  int
	Rows    int
}

// FrameWorkLoopFilterMapStats summarizes decoded loop-filter metadata coverage
// for a frame-level MI grid.
type FrameWorkLoopFilterMapStats struct {
	Cells   int
	Blocks  int
	Missing int
}

// LoopFilterMapShape returns the frame-level MI grid dimensions.
func (b FrameWorkBatch) LoopFilterMapShape() (cols int, rows int, length int, err error) {
	if !b.Sequence.Valid() || b.FrameSize.CodedWidth == 0 || b.FrameSize.Height == 0 {
		return 0, 0, 0, ErrInvalidBatch
	}
	miCols, ok := frameWorkMIExtent(b.FrameSize.CodedWidth)
	if !ok {
		return 0, 0, 0, ErrInvalidBatch
	}
	miRows, ok := frameWorkMIExtent(b.FrameSize.Height)
	if !ok {
		return 0, 0, 0, ErrInvalidBatch
	}
	length64 := uint64(miCols) * uint64(miRows)
	maxInt := uint64(^uint(0) >> 1)
	if miCols == 0 || miRows == 0 || length64 > maxInt {
		return 0, 0, 0, ErrInvalidBatch
	}
	return int(miCols), int(miRows), int(length64), nil
}

// BindLoopFilterMap validates and slices caller-owned storage for this frame.
func (b FrameWorkBatch) BindLoopFilterMap(records []FrameWorkLoopFilterBlockRecord) (FrameWorkLoopFilterMap, error) {
	cols, rows, length, err := b.LoopFilterMapShape()
	if err != nil {
		return FrameWorkLoopFilterMap{}, err
	}
	if len(records) < length {
		return FrameWorkLoopFilterMap{}, ErrInvalidBatch
	}
	out := FrameWorkLoopFilterMap{
		Records: records[:length],
		Stride:  cols,
		Rows:    rows,
	}
	if err := out.Reset(); err != nil {
		return FrameWorkLoopFilterMap{}, err
	}
	return out, nil
}

// Reset clears all recorded loop-filter metadata while preserving caller-owned
// storage and shape.
func (m FrameWorkLoopFilterMap) Reset() error {
	if err := m.validate(); err != nil {
		return err
	}
	clear(m.Records[:m.Stride*m.Rows])
	return nil
}

// MarkBlock records block-local loop-filter metadata for every MI cell covered
// by visit.
func (m FrameWorkLoopFilterMap) MarkBlock(visit tile.BlockLoopVisit, state *tile.DecodeState) error {
	if err := m.validate(); err != nil {
		return err
	}
	if state == nil {
		return ErrInvalidBatch
	}
	block := visit.Block
	if block.MIColEnd <= block.MICol || block.MIRowEnd <= block.MIRow {
		return ErrInvalidBatch
	}
	if block.MIColEnd > uint32(m.Stride) || block.MIRowEnd > uint32(m.Rows) {
		return ErrInvalidBatch
	}
	refFrame, mode, err := frameWorkLoopFilterRefMode(visit)
	if err != nil {
		return err
	}
	record := FrameWorkLoopFilterBlockRecord{
		Valid:           true,
		Block:           block,
		TransformTree:   visit.Coefficients.Tree,
		SegmentID:       visit.SegmentID,
		SkipTransform:   visit.Prefix.SkipTransform,
		Intra:           !visit.Prediction.Valid || visit.Prediction.Intra,
		RefFrame:        refFrame,
		Mode:            mode,
		DeltaLFFromBase: state.DeltaLFFromBase,
		DeltaLF:         state.DeltaLF,
	}
	for miRow := block.MIRow; miRow < block.MIRowEnd; miRow++ {
		row := int(miRow) * m.Stride
		for miCol := block.MICol; miCol < block.MIColEnd; miCol++ {
			m.Records[row+int(miCol)] = record
		}
	}
	return nil
}

// RecordAt returns the decoded loop-filter metadata for one frame-level MI
// cell. Missing cells return ok=false; out-of-map coordinates are invalid.
func (m FrameWorkLoopFilterMap) RecordAt(miCol uint32, miRow uint32) (FrameWorkLoopFilterBlockRecord, bool, error) {
	if err := m.validate(); err != nil {
		return FrameWorkLoopFilterBlockRecord{}, false, err
	}
	if miCol >= uint32(m.Stride) || miRow >= uint32(m.Rows) {
		return FrameWorkLoopFilterBlockRecord{}, false, ErrInvalidBatch
	}
	record := m.Records[int(miRow)*m.Stride+int(miCol)]
	if !record.Valid {
		return FrameWorkLoopFilterBlockRecord{}, false, nil
	}
	return record, true, nil
}

// CoverageStats validates and summarizes records in the requested frame-level
// MI grid. It permits maps with a wider stride than cols so callers can reuse
// padded storage across frames.
func (m FrameWorkLoopFilterMap) CoverageStats(cols int, rows int) (FrameWorkLoopFilterMapStats, error) {
	if err := m.validateGrid(cols, rows); err != nil {
		return FrameWorkLoopFilterMapStats{}, err
	}
	var stats FrameWorkLoopFilterMapStats
	for row := 0; row < rows; row++ {
		base := row * m.Stride
		for col := 0; col < cols; col++ {
			record := m.Records[base+col]
			if !record.Valid {
				stats.Missing++
				continue
			}
			if !frameWorkLoopFilterRecordCovers(record, col, row, cols, rows) {
				return stats, ErrInvalidBatch
			}
			stats.Cells++
			if record.Block.MICol == uint32(col) && record.Block.MIRow == uint32(row) {
				stats.Blocks++
			}
		}
	}
	return stats, nil
}

// ForEachBlock visits each unique block record once in raster MI order. Cells
// covered by the same block are skipped after the block's top-left cell.
func (m FrameWorkLoopFilterMap) ForEachBlock(visit func(FrameWorkLoopFilterBlockRecord) error) error {
	return m.ForEachBlockInGrid(m.Stride, m.Rows, visit)
}

// ForEachBlockInGrid visits each unique block record once inside the requested
// frame-level MI grid. It ignores any padded cells beyond cols/rows.
func (m FrameWorkLoopFilterMap) ForEachBlockInGrid(cols int, rows int, visit func(FrameWorkLoopFilterBlockRecord) error) error {
	if visit == nil {
		return ErrInvalidBatch
	}
	if err := m.validateGrid(cols, rows); err != nil {
		return err
	}
	for row := 0; row < rows; row++ {
		base := row * m.Stride
		for col := 0; col < cols; col++ {
			record := m.Records[base+col]
			if !record.Valid {
				continue
			}
			if !frameWorkLoopFilterRecordCovers(record, col, row, cols, rows) {
				return ErrInvalidBatch
			}
			if record.Block.MICol != uint32(col) || record.Block.MIRow != uint32(row) {
				continue
			}
			if err := visit(record); err != nil {
				return err
			}
		}
	}
	return nil
}

func (m FrameWorkLoopFilterMap) validate() error {
	if m.Stride <= 0 || m.Rows <= 0 {
		return ErrInvalidBatch
	}
	maxInt := int(^uint(0) >> 1)
	if m.Stride > maxInt/m.Rows ||
		uint64(m.Stride) > uint64(^uint32(0)) ||
		uint64(m.Rows) > uint64(^uint32(0)) {
		return ErrInvalidBatch
	}
	length := m.Stride * m.Rows
	if len(m.Records) < length {
		return ErrInvalidBatch
	}
	return nil
}

func (m FrameWorkLoopFilterMap) validateGrid(cols int, rows int) error {
	if err := m.validate(); err != nil {
		return err
	}
	if cols <= 0 || rows <= 0 || cols > m.Stride || rows > m.Rows {
		return ErrInvalidBatch
	}
	return nil
}

func frameWorkLoopFilterRecordCovers(record FrameWorkLoopFilterBlockRecord, col int, row int, cols int, rows int) bool {
	block := record.Block
	if block.MIColEnd <= block.MICol || block.MIRowEnd <= block.MIRow ||
		block.MIColEnd > uint32(cols) || block.MIRowEnd > uint32(rows) {
		return false
	}
	return uint32(col) >= block.MICol && uint32(col) < block.MIColEnd &&
		uint32(row) >= block.MIRow && uint32(row) < block.MIRowEnd
}

func frameWorkLoopFilterRefMode(visit tile.BlockLoopVisit) (int, loopfilter.ModeDeltaClass, error) {
	if !visit.Prediction.Valid || visit.Prediction.Intra || !visit.Prediction.InterReferencesValid {
		return 0, loopfilter.ModeDeltaClassZero, nil
	}
	ref := visit.Prediction.InterReferences.Ref[0]
	if !ref.Valid() {
		return 0, loopfilter.ModeDeltaClassZero, ErrInvalidBatch
	}
	// libaom (av1/common/av1_loopfilter.c mode_lf_lut) classifies every inter
	// mode as ModeDeltaClassMotion (1) except GLOBALMV (single) and
	// GLOBAL_GLOBALMV (compound), which use ModeDeltaClassZero. Restricting
	// Motion to NEW_MV alone (the previous behaviour) under-applied the
	// loop-filter mode delta for NEAREST_MV / NEAR_MV / NEAR_NEAR and the
	// other non-GLOBAL compound modes.
	mode := loopfilter.ModeDeltaClassZero
	if visit.Prediction.InterModeValid {
		interMode := visit.Prediction.InterMode
		if interMode.Compound {
			if !interMode.CompoundMode.Valid() {
				return 0, loopfilter.ModeDeltaClassZero, ErrInvalidBatch
			}
			if interMode.CompoundMode != tile.CompoundInterModeGlobalGlobal {
				mode = loopfilter.ModeDeltaClassMotion
			}
		} else {
			if !interMode.Mode.Valid() {
				return 0, loopfilter.ModeDeltaClassZero, ErrInvalidBatch
			}
			if interMode.Mode != tile.InterModeGlobalMV {
				mode = loopfilter.ModeDeltaClassMotion
			}
		}
	}
	return int(ref) + 1, mode, nil
}
