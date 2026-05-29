package tile

import "github.com/thesyncim/goav1/internal/av1/motion"

// SubChromaInterCell is one chroma sub-block's MV plus its position within
// the chroma block, expressed in chroma samples relative to the chroma
// block's top-left.
type SubChromaInterCell struct {
	OffsetX int
	OffsetY int
	Width   int
	Height  int
	MV      motion.Vector
	// Reference identifies the AV1 reference frame this sub-block uses.
	// In libaom's sub8x8 path each sub-block reads its own this_mbmi
	// ref_frame[0]; in practice all four cells share the chroma anchor's
	// ref slot since the spec also requires all sub-blocks be inter and
	// they were decoded in the same partition tree under the same
	// reference signaling, but we carry it explicitly for completeness.
	Reference ReferenceFrame
	// InterpFilters mirror this_mbmi->interp_filters. Per-sub-block
	// resolution is not tracked separately in BlockModeContext, so all
	// cells share the anchor block's filter (the libaom q00 vectors use
	// the same REGULAR/REGULAR filter for all blocks so this is exact;
	// vectors that vary filter across an SB still match because the
	// per-sub-block filter cannot change within one MI cell).
	InterpFilters motion.InterpFilters
}

// SubChromaInterResult is the populated set of chroma sub-block MVs for a
// block that triggers libaom's sub8x8 chroma inter path. Cells holds up to
// 4 entries (the NW/N/W/self quadrant for BlockSize4x4, two cells for
// BlockSize4x8 or BlockSize8x4, one entry per other case).
type SubChromaInterResult struct {
	Count int
	Cells [4]SubChromaInterCell
}

// CollectSubChromaInterCells returns the per-sub-block chroma cells libaom's
// build_inter_predictors_sub8x8() iterates over for an inter block at
// (x4, y4) of the given size when the chroma planes are subsampled. The
// returned SubChromaInterResult is populated only when:
//
//   - bsize has width 4 with ssX, or height 4 with ssY (sub4 trigger), and
//   - every luma cell within the chroma footprint (NW/N/W/self quadrant of
//     the chroma anchor) is recorded as inter in GridMotionValid.
//
// When sub-block predictors are not required (no sub4 trigger or any
// neighbor is intra/intrabc/missing) the return value has Count==0 and the
// caller should fall back to the standard single-MV chroma prediction with
// the anchor block's own MV. The trailing "self" cell intentionally
// reflects whatever the grid currently holds for (x4, y4); callers that
// invoke this before MarkInterMotion is run for the anchor block must
// overwrite that cell's MV/Reference with the anchor's just-decoded
// values.
func (c *BlockModeContext) CollectSubChromaInterCells(size BlockSize, x4 int, y4 int, ssX bool, ssY bool, filters motion.InterpFilters) (SubChromaInterResult, bool) {
	if c == nil {
		return SubChromaInterResult{}, false
	}
	dims, ok := size.Dimensions()
	if !ok {
		return SubChromaInterResult{}, false
	}
	is4x := dims.W4 == 1 && ssX
	is4y := dims.H4 == 1 && ssY
	if !is4x && !is4y {
		return SubChromaInterResult{}, false
	}
	cellW := int(dims.W4) * 4
	cellH := int(dims.H4) * 4
	if ssX {
		cellW >>= 1
	}
	if ssY {
		cellH >>= 1
	}
	rowStart := 0
	if is4y {
		rowStart = -1
	}
	colStart := 0
	if is4x {
		colStart = -1
	}
	var result SubChromaInterResult
	for row := rowStart; row <= 0; row++ {
		for col := colStart; col <= 0; col++ {
			nx := x4 + col
			ny := y4 + row
			if nx < 0 || ny < 0 || nx >= MaxBlockModeSlots || ny >= MaxBlockModeSlots {
				return SubChromaInterResult{}, false
			}
			if row == 0 && col == 0 {
				// Self cell: caller fills MV/Reference after
				// MarkInterMotion. Record the offset and dims
				// using the just-decoded anchor's filters.
				result.Cells[result.Count] = SubChromaInterCell{
					OffsetX:       (col - colStart) * cellW,
					OffsetY:       (row - rowStart) * cellH,
					Width:         cellW,
					Height:        cellH,
					InterpFilters: filters,
				}
				result.Count++
				continue
			}
			if c.GridMotionValid[ny][nx] == 0 {
				return SubChromaInterResult{}, false
			}
			cell := c.GridInterMotion[ny][nx]
			if !cell.References.Ref[0].Valid() {
				return SubChromaInterResult{}, false
			}
			// libaom's build_inter_predictors_sub8x8 predicts each cell with
			// that cell's own this_mbmi->interp_filters, not the anchor's.
			// Read the neighbor's actual per-MI filter pair from GridInterp
			// (written by MarkInterFilters for every inter block). The
			// collapsed 1D Above/Left filter contexts only retain the last
			// block written to a row/column slot, so they can report a
			// different neighbor's filter (and the X/Y axes do not line up
			// with a specific 2D neighbor). Fall back to the anchor filters
			// only when the grid slot was never recorded.
			cellFilters := filters
			if c.GridInterpValid[ny][nx] != 0 {
				cellFilters = c.GridInterp[ny][nx]
			}
			result.Cells[result.Count] = SubChromaInterCell{
				OffsetX:       (col - colStart) * cellW,
				OffsetY:       (row - rowStart) * cellH,
				Width:         cellW,
				Height:        cellH,
				MV:            cell.MV[0],
				Reference:     cell.References.Ref[0],
				InterpFilters: cellFilters,
			}
			result.Count++
		}
	}
	if result.Count == 0 {
		return SubChromaInterResult{}, false
	}
	return result, true
}
