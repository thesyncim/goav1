package threading

import (
	"encoding/binary"

	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// FrameWorkLoopFilterCompactState is an opaque, decoder-owned capability for
// the private single-tile decode/postfilter path. Its fields deliberately stay
// unexported: ordinary copied mask handles cannot manufacture readiness.
type FrameWorkLoopFilterCompactState struct {
	active    bool
	ready     bool
	cols      int
	rows      int
	unitCols  int
	unitRows  int
	layout    lfmask.Layout
	hasChroma bool
	cells     int32
	blocks    int32
	noskip    []byte

	// levels is dav1d's frame-level loop-filter level table (f->lf.lvl).
	levels FrameWorkLoopFilterLevelTable
}

// BindFrameWorkLoopFilterCompactState configures an owned compact state for a
// new frame and attaches it to h. noSkip holds eight row bytes per 64x64 CDEF
// unit. The caller must be the private exact-owner binder; public APIs never
// call this function.
func BindFrameWorkLoopFilterCompactState(h *FrameWorkLoopFilterMasks, state *FrameWorkLoopFilterCompactState, noSkip []byte, loopFilter parser.LoopFilterParams, segmentation parser.SegmentationParams, delta parser.DeltaParams, monoChrome bool) error {
	if h == nil || state == nil || h.Masks == nil || h.Cols <= 0 || h.Rows <= 0 || h.SB128W <= 0 ||
		h.Cols&1 != 0 || h.Rows&1 != 0 || len(h.LevelCache) < h.Cols*h.Rows {
		return ErrInvalidBatch
	}
	unitCols := (h.Cols + 15) >> 4
	unitRows := (h.Rows + 15) >> 4
	unitBytes := unitCols * unitRows * 8
	if unitCols <= 0 || unitRows <= 0 || len(noSkip) < unitBytes {
		return ErrInvalidBatch
	}
	state.active = true
	state.ready = false
	state.cols = h.Cols
	state.rows = h.Rows
	state.unitCols = unitCols
	state.unitRows = unitRows
	state.layout = h.Layout
	state.hasChroma = h.HasChroma
	state.cells = 0
	state.blocks = 0
	state.noskip = noSkip[:unitBytes]
	_ = monoChrome
	state.levels.Bind(loopFilter, segmentation, delta)
	initFrameWorkLoopFilterNoSkip(state.noskip, h.Cols, h.Rows, unitCols, unitRows)
	return nil
}

// InvalidateFrameWorkLoopFilterCompactState invalidates readiness but retains
// the active marker so an incomplete owned frame cannot fall back to the
// intentionally absent record map.
func InvalidateFrameWorkLoopFilterCompactState(state *FrameWorkLoopFilterCompactState) {
	if state != nil {
		state.ready = false
	}
}

// DeactivateFrameWorkLoopFilterCompactState invalidates and clears the private
// frame binding. Use it when the exact owner transitions to an ineligible path
// or is reset/closed.
func DeactivateFrameWorkLoopFilterCompactState(state *FrameWorkLoopFilterCompactState) {
	if state != nil {
		state.active = false
		state.ready = false
		state.cols = 0
		state.rows = 0
		state.unitCols = 0
		state.unitRows = 0
		state.cells = 0
		state.blocks = 0
		state.noskip = nil
	}
}

// FrameWorkLoopFilterCompactActive reports whether h belongs to an in-progress
// or completed compact frame. An active-but-not-ready frame must fail closed.
func FrameWorkLoopFilterCompactActive(state *FrameWorkLoopFilterCompactState) bool {
	return state != nil && state.active
}

// FrameWorkLoopFilterCompactStateActive reports the active lifecycle state
// independently of any copied mask handle. Exact owners use it to reject a
// missing or detached capability instead of falling back to an absent map.
func FrameWorkLoopFilterCompactStateActive(state *FrameWorkLoopFilterCompactState) bool {
	return state != nil && state.active
}

// FrameWorkLoopFilterCompactMatchesMasks confirms that a private owner is
// carrying the mask geometry and chroma layout for the exact prepared frame.
// The capability itself never rides on the exported mask value.
func FrameWorkLoopFilterCompactMatchesMasks(h *FrameWorkLoopFilterMasks, state *FrameWorkLoopFilterCompactState) bool {
	return h != nil && state != nil && state.active && h.Valid() &&
		h.Cols == state.cols && h.Rows == state.rows && h.Layout == state.layout &&
		h.HasChroma == state.hasChroma && h.SB128W > 0 && h.SB128H > 0 &&
		len(h.LevelCache) >= state.cols*state.rows &&
		state.unitCols == (state.cols+15)>>4 && state.unitRows == (state.rows+15)>>4 &&
		len(state.noskip) >= state.unitCols*state.unitRows*8
}

// FrameWorkLoopFilterCompactActiveShape reports whether h carries an active
// compact frame with the requested exact MI dimensions. Unlike Ready, this is
// used before the block walk to validate the incoming batch shape.
func FrameWorkLoopFilterCompactActiveShape(h *FrameWorkLoopFilterMasks, state *FrameWorkLoopFilterCompactState, cols, rows int) bool {
	if state == nil || state.ready || state.cols != cols || state.rows != rows ||
		!FrameWorkLoopFilterCompactMatchesMasks(h, state) {
		return false
	}
	return true
}

// FrameWorkLoopFilterCompactReady reports whether h carries a successfully
// completed compact frame with the requested dimensions.
func FrameWorkLoopFilterCompactReady(h *FrameWorkLoopFilterMasks, state *FrameWorkLoopFilterCompactState, cols, rows int) bool {
	return FrameWorkLoopFilterCompactMatchesMasks(h, state) && state.ready &&
		state.cols == cols && state.rows == rows &&
		state.cells == int32(cols*rows) && state.blocks > 0 &&
		len(h.LevelCache) >= cols*rows
}

// FrameWorkLoopFilterCompactUnitGridMatches reports whether the ready compact
// frame owns exactly the requested CDEF 64x64-unit grid.
func FrameWorkLoopFilterCompactUnitGridMatches(h *FrameWorkLoopFilterMasks, state *FrameWorkLoopFilterCompactState, unitCols, unitRows int) bool {
	return FrameWorkLoopFilterCompactMatchesMasks(h, state) && state.ready &&
		state.unitCols == unitCols && state.unitRows == unitRows &&
		len(state.noskip) >= unitCols*unitRows*8
}

// FrameWorkLoopFilterCompactCoverage reports decoded cells and blocks only for
// a ready, exact-shape compact frame.
func FrameWorkLoopFilterCompactCoverage(h *FrameWorkLoopFilterMasks, state *FrameWorkLoopFilterCompactState, cols, rows int) (cells, blocks int32, ok bool) {
	if !FrameWorkLoopFilterCompactReady(h, state, cols, rows) {
		return 0, 0, false
	}
	return state.cells, state.blocks, true
}

// FrameWorkLoopFilterCompactUnitMask returns the direct CDEF no-skip word for
// one 64x64 unit. Bytes are interpreted in little-endian row-major order.
func FrameWorkLoopFilterCompactUnitMask(h *FrameWorkLoopFilterMasks, state *FrameWorkLoopFilterCompactState, unitRow, unitCol int) (uint64, bool) {
	if !FrameWorkLoopFilterCompactMatchesMasks(h, state) || !state.ready ||
		unitRow < 0 || unitRow >= state.unitRows || unitCol < 0 || unitCol >= state.unitCols {
		return 0, false
	}
	off := (unitRow*state.unitCols + unitCol) * 8
	if off < 0 || off+8 > len(state.noskip) {
		return 0, false
	}
	return binary.LittleEndian.Uint64(state.noskip[off : off+8]), true
}

// CompleteFrameWorkLoopFilterCompactState publishes readiness only after the
// full block walk, reconstruction, and retained-CDF step have succeeded.
func CompleteFrameWorkLoopFilterCompactState(state *FrameWorkLoopFilterCompactState) error {
	if state == nil || !state.active || state.cols <= 0 || state.rows <= 0 || state.cells != int32(state.cols*state.rows) || state.blocks <= 0 {
		if state != nil {
			state.ready = false
		}
		return ErrInvalidBatch
	}
	state.ready = true
	return nil
}

// buildCompact is the private decode-order producer. It validates the same
// geometry/ref-mode inputs as MarkBlockPtr but writes compact level/no-skip
// data instead of replicating a record over every MI cell.
func (h *FrameWorkLoopFilterMasks) buildCompact(compact *FrameWorkLoopFilterCompactState, scratch *frameWorkLoopFilterMaskScratch, visit *tile.BlockLoopVisit, state *tile.DecodeState, sbLog2 int) error {
	if h == nil || compact == nil || !compact.active || compact.ready || visit == nil || state == nil ||
		compact.cols != h.Cols || compact.rows != h.Rows {
		return ErrInvalidBatch
	}
	block := visit.Block
	if block.MIColEnd <= block.MICol || block.MIRowEnd <= block.MIRow ||
		int(block.MIColEnd) > h.Cols || int(block.MIRowEnd) > h.Rows {
		return ErrInvalidBatch
	}
	if _, ok := block.Size.Dimensions(); !ok {
		return ErrInvalidBatch
	}
	if !compact.levels.off {
		levelValues, err := compact.levels.blockLevels(scratch, visit, state)
		if err != nil {
			return err
		}
		intra := !visit.Prediction.Valid || visit.Prediction.Intra
		lc := lfmask.LevelCache{Cells: h.LevelCache, Stride: h.Cols}
		h.buildBlockWithLevels(scratch, int(block.MICol), int(block.MIRow), block.Size,
			visit.Coefficients.Tree, visit.Prefix.SkipTransform, intra, sbLog2, lc, levelValues)
	}
	if !visit.Prefix.SkipTransform {
		compact.markNoSkip(int(block.MICol), int(block.MIRow), int(block.MIColEnd), int(block.MIRowEnd))
	}
	bw := int(block.MIColEnd - block.MICol)
	bh := int(block.MIRowEnd - block.MIRow)
	if bw <= 0 || bh <= 0 {
		return ErrInvalidBatch
	}
	compact.cells += int32(bw * bh)
	compact.blocks++
	return nil
}

func (s *FrameWorkLoopFilterCompactState) markNoSkip(miColStart, miRowStart, miColEnd, miRowEnd int) {
	markFrameWorkLoopFilterNoSkip(s.noskip, s.unitCols, miColStart, miRowStart, miColEnd, miRowEnd)
}
