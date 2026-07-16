package decoder

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// This file consumes the dav1d-style per-superblock deblocking edge bitmasks
// (internal/av1/lfmask, built at decode time by threading.FrameWorkLoopFilterMasks)
// to apply the loop filter, replacing the run-merged edge-list walk in
// postfilter_loopfilter.go. It is byte-identical to ApplyLoopFilterEdges: the
// lfmask build+scan reproduces production's per-4x4 (edge, width class, level)
// decisions (proven by lfmask_diff_test.go), the loop-filter kernels process
// each line perpendicular to the edge independently (so a per-4x4 apply equals
// the run-merged apply), and AV1's filter-width-vs-transform-spacing rule keeps
// same-direction edges from overlapping (so the intra-pass scan order is free).
// The frame-boundary width clamps mirror the edge-list path exactly.

// frameWorkLoopFilterMaskLumaWidth maps a dav1d luma strength class 0/1/2 to the
// pixel filter width the kernels dispatch on (dav1d 4<<idx = 4/8/16 is the same
// selection as production's 4/8/14 filter set).
var frameWorkLoopFilterMaskLumaWidth = [3]uint8{4, 8, 14}

// frameWorkLoopFilterMaskChromaWidth maps a dav1d chroma strength class 0/1 to
// the production chroma pixel filter width (4/6).
var frameWorkLoopFilterMaskChromaWidth = [2]uint8{4, 6}

// ApplyLoopFilterEdgesFromMasks applies the decode-time deblocking bitmasks to
// ctx.Output, resolving per-4x4 levels from filterMap. It is the mask-driven
// counterpart of ApplyLoopFilterEdges and produces byte-identical output.
func (ctx FrameWorkPostFilterContext) ApplyLoopFilterEdgesFromMasks(masks *threading.FrameWorkLoopFilterMasks, filterMap FrameWorkLoopFilterMap) (FrameWorkLoopFilterPostFilterApplyResult, error) {
	var result FrameWorkLoopFilterPostFilterApplyResult
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterLoopFilter) {
		return FrameWorkLoopFilterPostFilterApplyResult{}, nil
	}
	result.Active = true
	if ctx.Output == nil {
		return result, frame.ErrInvalidSlot
	}
	if masks == nil || !masks.Valid() {
		return result, loopfilter.ErrInvalidFilter
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(ctx.Event.FrameSize)
	if err != nil {
		return result, err
	}
	if cols != masks.Cols || rows != masks.Rows {
		return result, threading.ErrInvalidBatch
	}
	if frameWorkLoopFilterMapEmpty(filterMap) && ctx.LoopFilterMap != nil {
		filterMap = *ctx.LoopFilterMap
	}
	if frameWorkLoopFilterMapEmpty(filterMap) {
		return result, threading.ErrInvalidBatch
	}
	if err := frameWorkValidateLoopFilterMap(filterMap, cols, rows); err != nil {
		return result, err
	}
	result.Plan.Active = true
	result.Plan.MICols = uint16(cols)
	result.Plan.MIRows = uint16(rows)
	levelCtx := frameWorkLoopFilterLevelContextFor(&ctx.Event)
	if err := ctx.populateLoopFilterLevelCacheFromMap(masks, filterMap, levelCtx, &result.Plan); err != nil {
		return result, err
	}
	maxPlane := loopfilter.PlaneV
	if masks.Layout.Mono || ctx.Output.Format.MonoChrome || !masks.HasChroma {
		maxPlane = loopfilter.PlaneY
	}
	if err := ctx.applyLoopFilterEdgesFromMasksInPlanePassOrder(&result, masks, levelCtx, loopfilter.PlaneY, maxPlane); err != nil {
		return result, err
	}
	return result, nil
}

// populateLoopFilterLevelCacheFromMap fills the frame-wide per-4x4 level grid
// (masks.LevelCache) from the decoded loop-filter map, mirroring the level
// writes dav1d's create_lf_mask_{inter,intra} would perform (lfmask build.go).
// The level cells are order-independent, so a raster walk over block origins is
// sufficient.
func (ctx FrameWorkPostFilterContext) populateLoopFilterLevelCacheFromMap(masks *threading.FrameWorkLoopFilterMasks, filterMap FrameWorkLoopFilterMap, levelCtx frameWorkLoopFilterLevelContext, plan *FrameWorkLoopFilterPostFilterPlan) error {
	if len(masks.LevelCache) < masks.Cols*masks.Rows {
		return threading.ErrInvalidBatch
	}
	clear(masks.LevelCache[:masks.Cols*masks.Rows])
	return ctx.populateLoopFilterLevelCacheRange(masks, filterMap, levelCtx, plan, 0, masks.Rows)
}

// populateLoopFilterLevelCacheRange fills the level cache from block origins whose
// MI row is in [rowStart,rowEnd). It does not clear (the caller clears once). Block
// coverage partitions the frame, so distinct origin-row bands write disjoint cells
// -- a block originating in this band may extend its (disjoint) coverage into a
// neighbour band's rows, but never overwrites another block's cells -- so bands run
// concurrently without a barrier between them. The caller must clear the whole
// cache and complete every band before any level read (the mask apply scan).
func (ctx FrameWorkPostFilterContext) populateLoopFilterLevelCacheRange(masks *threading.FrameWorkLoopFilterMasks, filterMap FrameWorkLoopFilterMap, levelCtx frameWorkLoopFilterLevelContext, plan *FrameWorkLoopFilterPostFilterPlan, rowStart, rowEnd int) error {
	cols := masks.Cols
	rows := masks.Rows
	if len(masks.LevelCache) < cols*rows {
		return threading.ErrInvalidBatch
	}
	if rowStart < 0 || rowStart > rowEnd || rowEnd > rows {
		return threading.ErrInvalidBatch
	}
	ssHor := masks.Layout.SSHor
	ssVer := masks.Layout.SSVer
	hasChroma := masks.HasChroma
	stride := int(filterMap.Stride)
	for row := rowStart; row < rowEnd; row++ {
		base := row * stride
		for col := 0; col < cols; {
			record := &filterMap.Records[base+col]
			if !record.Valid {
				plan.Missing++
				col++
				continue
			}
			block := record.Block
			nextCol := int(block.MIColEnd)
			if nextCol <= col || nextCol > cols {
				return threading.ErrInvalidBatch
			}
			plan.Cells += int32(nextCol - col)
			if int(block.MICol) != col || int(block.MIRow) != row {
				col = nextCol
				continue
			}
			plan.Blocks++
			levels, err := frameWorkResolveLoopFilterRecordLevels(levelCtx, record)
			if err != nil {
				return err
			}
			dims, ok := record.Block.Size.Dimensions()
			if !ok {
				return threading.ErrInvalidBatch
			}
			bx := col
			by := row
			bw4 := minInt(cols-bx, int(dims.W4))
			bh4 := minInt(rows-by, int(dims.H4))
			yVert := levels[loopfilter.PlaneY][loopfilter.EdgeVertical]
			yHorz := levels[loopfilter.PlaneY][loopfilter.EdgeHorizontal]
			for y := 0; y < bh4; y++ {
				rowBase := (by + y) * cols
				for x := 0; x < bw4; x++ {
					cell := &masks.LevelCache[rowBase+bx+x]
					cell[0] = yVert
					cell[1] = yHorz
				}
			}
			if !hasChroma {
				col = nextCol
				continue
			}
			cbw4 := minInt(((cols+ssHor)>>ssHor)-(bx>>ssHor), (int(dims.W4)+ssHor)>>ssHor)
			cbh4 := minInt(((rows+ssVer)>>ssVer)-(by>>ssVer), (int(dims.H4)+ssVer)>>ssVer)
			if cbw4 <= 0 || cbh4 <= 0 {
				col = nextCol
				continue
			}
			uLevel := levels[loopfilter.PlaneU][loopfilter.EdgeVertical]
			vLevel := levels[loopfilter.PlaneV][loopfilter.EdgeVertical]
			cbx := bx >> ssHor
			cby := by >> ssVer
			for y := 0; y < cbh4; y++ {
				rowBase := (cby + y) * cols
				for x := 0; x < cbw4; x++ {
					cell := &masks.LevelCache[rowBase+cbx+x]
					cell[2] = uLevel
					cell[3] = vLevel
				}
			}
			col = nextCol
		}
	}
	return nil
}

// maskDir bits select which deblock edge directions a plane-range apply scans.
// The bit index matches loopfilter.Edge (EdgeVertical==0, EdgeHorizontal==1) so a
// single direction maps to 1<<edge. The whole-frame single-shot apply passes
// maskDirBoth (vertical scan then horizontal scan, per plane); the banded parallel
// apply passes one bit per phase so a frame-wide barrier separates the directions.
const (
	maskDirVertical   = 1 << loopfilter.EdgeVertical
	maskDirHorizontal = 1 << loopfilter.EdgeHorizontal
	maskDirBoth       = maskDirVertical | maskDirHorizontal
)

type frameWorkLoopFilterMaskRun struct {
	active bool
	edge   loopfilter.Edge
	x4, y4 int
	len4   int
	width  uint8
	th     loopfilter.Thresholds
}

// frameWorkLoopFilterMaskApplyState is the concrete trusted mask consumer.
// Keeping the run builder and threshold cache behind direct method calls lets
// the scan loops avoid allocating closures or calling through EdgeVisitor
// function values for every filtered cell.
type frameWorkLoopFilterMaskApplyState struct {
	plane     loopfilter.Plane
	bounds    frameWorkLoopFilterBounds
	sharpness uint8

	thresholds     [loopfilter.MaxLevel + 1]loopfilter.Thresholds
	thresholdReady [loopfilter.MaxLevel + 1]bool
	firstErr       error
	run            frameWorkLoopFilterMaskRun
}

func (s *frameWorkLoopFilterMaskApplyState) thresholdFor(level uint8) (loopfilter.Thresholds, bool) {
	if level == 0 || level > loopfilter.MaxLevel {
		s.firstErr = loopfilter.ErrInvalidFilter
		return loopfilter.Thresholds{}, false
	}
	if !s.thresholdReady[level] {
		th, err := loopfilter.ThresholdsForLevel(level, s.sharpness)
		if err != nil {
			s.firstErr = err
			return loopfilter.Thresholds{}, false
		}
		s.thresholds[level] = th
		s.thresholdReady[level] = true
	}
	return s.thresholds[level], true
}

func frameWorkFlushLoopFilterMaskRun(s *frameWorkLoopFilterMaskApplyState, dst frame.Plane, bytesPerSample int, bitDepth uint8) {
	if !s.run.active {
		return
	}
	s.run.active = false
	if s.firstErr != nil {
		return
	}
	x := int32(s.run.x4) * 4
	y := int32(s.run.y4) * 4
	length := int32(s.run.len4) * 4
	var err error
	switch s.run.width {
	case 4:
		err = loopfilter.Filter4Edge(dst, bytesPerSample, bitDepth, s.run.edge, x, y, length, s.run.th)
	case 6:
		err = loopfilter.Filter6Edge(dst, bytesPerSample, bitDepth, s.run.edge, x, y, length, s.run.th)
	case 8:
		err = loopfilter.Filter8Edge(dst, bytesPerSample, bitDepth, s.run.edge, x, y, length, s.run.th)
	case 14:
		err = loopfilter.Filter14Edge(dst, bytesPerSample, bitDepth, s.run.edge, x, y, length, s.run.th)
	default:
		s.firstErr = loopfilter.ErrInvalidFilter
		return
	}
	if err != nil {
		s.firstErr = err
	}
}

// frameWorkLoopFilterMaskCellWidth resolves one trusted mask cell's effective
// filter width. Mask coordinates are non-negative 4x4 positions and every call
// covers exactly one cell, so the general edge-list bounds path's overflow and
// arbitrary-length checks reduce to the two crop bounds plus the perpendicular
// tap radius. The backing buffer covers at least the MI-aligned crop extent, so
// a crop-valid four-sample run always fits along the edge.
func frameWorkLoopFilterMaskCellWidth(bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4, y4 int, width uint8) (uint8, bool) {
	x := int32(x4) << 2
	y := int32(y4) << 2
	if x >= bounds.posWidth || y >= bounds.posHeight {
		return 0, false
	}
	var position, limit int32
	if edge == loopfilter.EdgeVertical {
		position = x
		limit = bounds.bufWidth
	} else {
		position = y
		limit = bounds.bufHeight
	}
	for width != 0 {
		radius := int32(width) >> 1
		if position >= radius && position+radius-1 < limit {
			return width, true
		}
		width = frameWorkLoopFilterDowngradeWidth(width)
	}
	return 0, false
}

func (s *frameWorkLoopFilterMaskApplyState) addCell(result *FrameWorkLoopFilterPostFilterApplyResult, dst frame.Plane, bytesPerSample int, bitDepth uint8, edge loopfilter.Edge, x4, y4 int, level, width uint8) {
	if s.firstErr != nil {
		return
	}
	w, ok := frameWorkLoopFilterMaskCellWidth(s.bounds, edge, x4, y4, width)
	if !ok {
		return
	}
	th, ok := s.thresholdFor(level)
	if !ok {
		return
	}
	frameWorkCountAppliedLoopFilterMaskEdge(result, s.plane, level)
	if s.run.active && s.run.edge == edge && s.run.width == w && s.run.th == th {
		if edge == loopfilter.EdgeVertical {
			if x4 == s.run.x4 && y4 == s.run.y4+s.run.len4 {
				s.run.len4++
				return
			}
		} else if y4 == s.run.y4 && x4 == s.run.x4+s.run.len4 {
			s.run.len4++
			return
		}
	}
	frameWorkFlushLoopFilterMaskRun(s, dst, bytesPerSample, bitDepth)
	if s.firstErr != nil {
		return
	}
	s.run = frameWorkLoopFilterMaskRun{
		active: true,
		edge:   edge,
		x4:     x4,
		y4:     y4,
		len4:   1,
		width:  w,
		th:     th,
	}
}

func (ctx FrameWorkPostFilterContext) applyLoopFilterEdgesFromMasksInPlanePassOrder(result *FrameWorkLoopFilterPostFilterApplyResult, masks *threading.FrameWorkLoopFilterMasks, levelCtx frameWorkLoopFilterLevelContext, minPlane, maxPlane loopfilter.Plane) error {
	sharpness := ctx.Event.LoopFilter.Sharpness
	lc := lfmask.LevelCache{Cells: masks.LevelCache, Stride: masks.Cols}
	// Match the edge-list chroma gating: AV1 deblocks chroma only when the frame
	// carries a non-zero luma level, and skips a chroma plane whose frame base
	// level is zero regardless of per-block segment deltas (postfilter_loopfilter
	// frameWorkAppendLoopFilterChromaEdges). Luma keeps its per-cell level with
	// the level-from-previous fallback.
	chromaGated := levelCtx.lumaZero
	for plane := minPlane; plane <= maxPlane; plane++ {
		if plane != loopfilter.PlaneY {
			if chromaGated || levelCtx.base[plane][loopfilter.EdgeVertical] == 0 {
				continue
			}
		}
		if err := ctx.applyLoopFilterMaskPlaneRange(result, masks, lc, sharpness, plane, maskDirBoth, 0, masks.SB128H, 0, masks.SB128W); err != nil {
			return err
		}
	}
	return nil
}

// applyLoopFilterMaskPlaneRange scans and filters one plane over a contiguous
// range of 128x128 mask region-rows [rRow0,rRow1), for the directions selected by
// dirMask (vertical scan before horizontal when both are set). The per-plane setup
// (aligned destination, bounds, threshold cache, apply closure) stays local so the
// closure is stack-allocated; result accumulates this call's counts. Filtering a
// region-row band writes only pixels within that band's rows (perpendicular filter
// taps are bounded by the adjacent transform sizes), so disjoint bands can run on
// separate goroutines with their own result once a direction barrier is in place.
func (ctx FrameWorkPostFilterContext) applyLoopFilterMaskPlaneRange(result *FrameWorkLoopFilterPostFilterApplyResult, masks *threading.FrameWorkLoopFilterMasks, lc lfmask.LevelCache, sharpness uint8, plane loopfilter.Plane, dirMask, rRow0, rRow1, rCol0, rCol1 int) error {
	bytesPerSample := ctx.Output.Layout.BytesPerSample
	bitDepth := ctx.Output.Format.BitDepth
	dst, ok := frameWorkLoopFilterOutputPlane(*ctx.Output, plane)
	if !ok {
		if plane == loopfilter.PlaneY {
			return loopfilter.ErrInvalidFilter
		}
		return nil
	}
	planeW, planeH, err := frameWorkLoopFilterBufferSize(ctx, plane)
	if err != nil {
		return err
	}
	dst = frameWorkLoopFilterAlignedPlane(dst, planeW, planeH, bytesPerSample)
	bounds, err := frameWorkLoopFilterPlaneBounds(ctx, plane)
	if err != nil {
		return err
	}

	// Run batcher. The per-4x4 scan emits one addCell per set edge cell, but the
	// deblock kernels only reach their NEON fast path when handed a run of >= 8
	// sample positions (two 4x4 cells) along the edge; a lone cell (length 4)
	// always falls back to scalar. Merging geometrically contiguous cells that
	// resolve to the same (edge, filter width, thresholds) into a single
	// Filter*Edge call over length=4*cells is byte-identical to filtering each
	// cell separately -- the kernels loop over the positions along the edge
	// independently -- while letting NEON process the run. The scan visits cells
	// in increasing position order along the merge axis (rows for a vertical edge,
	// columns for a horizontal edge), so an extending cell is always the immediate
	// successor of the pending run. State stays concrete and stack-owned so the
	// scan calls addCell directly without a closure or function-value dispatch.
	state := frameWorkLoopFilterMaskApplyState{
		plane:     plane,
		bounds:    bounds,
		sharpness: sharpness,
	}

	if plane == loopfilter.PlaneY {
		if dirMask&maskDirVertical != 0 {
			ctx.scanLoopFilterMaskLumaBand(masks, lc, loopfilter.EdgeVertical, rRow0, rRow1, rCol0, rCol1, &state, result, dst, bytesPerSample, bitDepth)
			frameWorkFlushLoopFilterMaskRun(&state, dst, bytesPerSample, bitDepth)
		}
		if state.firstErr == nil && dirMask&maskDirHorizontal != 0 {
			ctx.scanLoopFilterMaskLumaBand(masks, lc, loopfilter.EdgeHorizontal, rRow0, rRow1, rCol0, rCol1, &state, result, dst, bytesPerSample, bitDepth)
			frameWorkFlushLoopFilterMaskRun(&state, dst, bytesPerSample, bitDepth)
		}
	} else {
		if dirMask&maskDirVertical != 0 {
			ctx.scanLoopFilterMaskChromaBand(masks, lc, plane, loopfilter.EdgeVertical, rRow0, rRow1, rCol0, rCol1, &state, result, dst, bytesPerSample, bitDepth)
			frameWorkFlushLoopFilterMaskRun(&state, dst, bytesPerSample, bitDepth)
		}
		if state.firstErr == nil && dirMask&maskDirHorizontal != 0 {
			ctx.scanLoopFilterMaskChromaBand(masks, lc, plane, loopfilter.EdgeHorizontal, rRow0, rRow1, rCol0, rCol1, &state, result, dst, bytesPerSample, bitDepth)
			frameWorkFlushLoopFilterMaskRun(&state, dst, bytesPerSample, bitDepth)
		}
	}
	return state.firstErr
}

// scanLoopFilterMaskLumaBand scans the luma bitmasks for one edge direction over
// the 128x128 region-row range [rRow0,rRow1) and dispatches each set 4x4 edge
// through the concrete apply state. The frame-left column (x4==0) and frame-top row (y4==0) are
// skipped to match the edge-list apply's have_left / have_top boundary skip.
// Passing rRow0=0, rRow1=SB128H reproduces the whole-frame scan; a sub-range is
// one band of the parallel apply.
func (ctx FrameWorkPostFilterContext) scanLoopFilterMaskLumaBand(masks *threading.FrameWorkLoopFilterMasks, lc lfmask.LevelCache, dir loopfilter.Edge, rRow0, rRow1, rCol0, rCol1 int, state *frameWorkLoopFilterMaskApplyState, result *FrameWorkLoopFilterPostFilterApplyResult, dst frame.Plane, bytesPerSample int, bitDepth uint8) {
	cols := masks.Cols
	rows := masks.Rows
	if dir == loopfilter.EdgeVertical {
		// Vertical edges (dir 0): position = column, scanned bits = rows.
		for rRow := rRow0; rRow < rRow1; rRow++ {
			baseRow := rRow * 32
			extentRows := minInt(32, rows-baseRow)
			if extentRows <= 0 {
				continue
			}
			for rCol := rCol0; rCol < rCol1; rCol++ {
				baseCol := rCol * 32
				extentCols := minInt(32, cols-baseCol)
				if extentCols <= 0 {
					continue
				}
				region := &masks.Masks[rRow*masks.SB128W+rCol]
				for pos := 0; pos < extentCols; pos++ {
					col := baseCol + pos
					if col == 0 {
						continue
					}
					m0, m1, m2 := lfmask.LumaMaskWords(&region.Y[0][pos], 0, extentRows)
					for vm := m0 | m1 | m2; vm != 0; vm &= vm - 1 {
						off := bits.TrailingZeros32(vm)
						bit := uint32(1) << off
						widthClass := 0
						if m2&bit != 0 {
							widthClass = 2
						} else if m1&bit != 0 {
							widthClass = 1
						}
						y4 := baseRow + off
						level := lc.Cells[y4*cols+col][loopfilter.EdgeVertical]
						if level == 0 {
							level = lc.Cells[y4*cols+col-1][loopfilter.EdgeVertical]
						}
						if level == 0 {
							continue
						}
						state.addCell(result, dst, bytesPerSample, bitDepth, loopfilter.EdgeVertical, col, y4, level, frameWorkLoopFilterMaskLumaWidth[widthClass])
					}
				}
			}
		}
		return
	}
	// Horizontal edges (dir 1): position = row, scanned bits = columns.
	for rRow := rRow0; rRow < rRow1; rRow++ {
		baseRow := rRow * 32
		extentRows := minInt(32, rows-baseRow)
		if extentRows <= 0 {
			continue
		}
		for rCol := rCol0; rCol < rCol1; rCol++ {
			baseCol := rCol * 32
			extentCols := minInt(32, cols-baseCol)
			if extentCols <= 0 {
				continue
			}
			region := &masks.Masks[rRow*masks.SB128W+rCol]
			for pos := 0; pos < extentRows; pos++ {
				row := baseRow + pos
				if row == 0 {
					continue
				}
				m0, m1, m2 := lfmask.LumaMaskWords(&region.Y[1][pos], 0, extentCols)
				for vm := m0 | m1 | m2; vm != 0; vm &= vm - 1 {
					off := bits.TrailingZeros32(vm)
					bit := uint32(1) << off
					widthClass := 0
					if m2&bit != 0 {
						widthClass = 2
					} else if m1&bit != 0 {
						widthClass = 1
					}
					x4 := baseCol + off
					level := lc.Cells[row*cols+x4][loopfilter.EdgeHorizontal]
					if level == 0 {
						level = lc.Cells[(row-1)*cols+x4][loopfilter.EdgeHorizontal]
					}
					if level == 0 {
						continue
					}
					state.addCell(result, dst, bytesPerSample, bitDepth, loopfilter.EdgeHorizontal, x4, row, level, frameWorkLoopFilterMaskLumaWidth[widthClass])
				}
			}
		}
	}
}

// scanLoopFilterMaskChromaBand scans the chroma bitmasks for one chroma plane
// (U or V) and one edge direction over the 128x128 region-row range [rRow0,rRow1),
// dispatching each set 4x4 chroma edge through the concrete apply state in chroma-plane 4x4
// coordinates. Passing rRow0=0, rRow1=SB128H reproduces the whole-frame scan.
func (ctx FrameWorkPostFilterContext) scanLoopFilterMaskChromaBand(masks *threading.FrameWorkLoopFilterMasks, lc lfmask.LevelCache, plane loopfilter.Plane, dir loopfilter.Edge, rRow0, rRow1, rCol0, rCol1 int, state *frameWorkLoopFilterMaskApplyState, result *FrameWorkLoopFilterPostFilterApplyResult, dst frame.Plane, bytesPerSample int, bitDepth uint8) {
	cols := masks.Cols
	rows := masks.Rows
	ssHor := masks.Layout.SSHor
	ssVer := masks.Layout.SSVer
	ccols := (cols + ssHor) >> ssHor
	crows := (rows + ssVer) >> ssVer
	regionCW := 32 >> ssHor
	regionCH := 32 >> ssVer
	laneBitsV := 16 >> ssVer
	laneBitsH := 16 >> ssHor
	levelIdx := 2
	if plane == loopfilter.PlaneV {
		levelIdx = 3
	}
	if dir == loopfilter.EdgeVertical {
		// Vertical edges (dir 0): position = chroma column, scanned bits = chroma rows.
		for rRow := rRow0; rRow < rRow1; rRow++ {
			baseCRow := rRow * regionCH
			extentCRows := minInt(regionCH, crows-baseCRow)
			if extentCRows <= 0 {
				continue
			}
			for rCol := rCol0; rCol < rCol1; rCol++ {
				baseCCol := rCol * regionCW
				extentCCols := minInt(regionCW, ccols-baseCCol)
				if extentCCols <= 0 {
					continue
				}
				region := &masks.Masks[rRow*masks.SB128W+rCol]
				for pos := 0; pos < extentCCols; pos++ {
					ccol := baseCCol + pos
					if ccol == 0 {
						continue
					}
					m0, m1 := lfmask.ChromaMaskWords(&region.UV[0][pos], 0, extentCRows, laneBitsV)
					for vm := m0 | m1; vm != 0; vm &= vm - 1 {
						off := bits.TrailingZeros32(vm)
						bit := uint32(1) << off
						widthClass := 0
						if m1&bit != 0 {
							widthClass = 1
						}
						crow := baseCRow + off
						level := lc.Cells[crow*cols+ccol][levelIdx]
						if level == 0 {
							level = lc.Cells[crow*cols+ccol-1][levelIdx]
						}
						if level == 0 {
							continue
						}
						state.addCell(result, dst, bytesPerSample, bitDepth, loopfilter.EdgeVertical, ccol, crow, level, frameWorkLoopFilterMaskChromaWidth[widthClass])
					}
				}
			}
		}
		return
	}
	// Horizontal edges (dir 1): position = chroma row, scanned bits = chroma columns.
	for rRow := rRow0; rRow < rRow1; rRow++ {
		baseCRow := rRow * regionCH
		extentCRows := minInt(regionCH, crows-baseCRow)
		if extentCRows <= 0 {
			continue
		}
		for rCol := rCol0; rCol < rCol1; rCol++ {
			baseCCol := rCol * regionCW
			extentCCols := minInt(regionCW, ccols-baseCCol)
			if extentCCols <= 0 {
				continue
			}
			region := &masks.Masks[rRow*masks.SB128W+rCol]
			for pos := 0; pos < extentCRows; pos++ {
				crow := baseCRow + pos
				if crow == 0 {
					continue
				}
				m0, m1 := lfmask.ChromaMaskWords(&region.UV[1][pos], 0, extentCCols, laneBitsH)
				for vm := m0 | m1; vm != 0; vm &= vm - 1 {
					off := bits.TrailingZeros32(vm)
					bit := uint32(1) << off
					widthClass := 0
					if m1&bit != 0 {
						widthClass = 1
					}
					ccol := baseCCol + off
					level := lc.Cells[crow*cols+ccol][levelIdx]
					if level == 0 {
						level = lc.Cells[(crow-1)*cols+ccol][levelIdx]
					}
					if level == 0 {
						continue
					}
					state.addCell(result, dst, bytesPerSample, bitDepth, loopfilter.EdgeHorizontal, ccol, crow, level, frameWorkLoopFilterMaskChromaWidth[widthClass])
				}
			}
		}
	}
}

// FrameWorkLoopFilterMaskBands is the validated, frame-level state a banded
// parallel mask apply shares across its worker goroutines. Build it once per frame
// with PrepareLoopFilterMaskBands (which validates the map, populates the frame-wide
// per-4x4 level cache, and resolves which planes deblock), then dispatch
// ApplyBand jobs -- one per (plane, direction, region-row band) -- across workers.
// After Prepare, the level cache and masks are read-only during ApplyBand, so bands
// carry no shared mutable state and each accumulates its own local counts.
//
// The direction is the barrier axis: all vertical-edge bands of the frame must
// complete before any horizontal-edge band starts (the horizontal filter reads
// pixels the vertical pass modified). Within one direction, region-row bands write
// disjoint pixel rows (perpendicular filter taps are bounded by the adjacent
// transform sizes), and distinct planes write disjoint surfaces, so all jobs of a
// direction run concurrently. Applying every (plane, direction, band) in this order
// is byte-identical to the single-shot ApplyLoopFilterEdgesFromMasks.
type FrameWorkLoopFilterMaskBands struct {
	ctx       FrameWorkPostFilterContext
	masks     *threading.FrameWorkLoopFilterMasks
	filterMap FrameWorkLoopFilterMap
	// levelCtx is NOT stored: it holds pointers into ctx.Event, and stashing it
	// in this (persistent, goroutine-shared) value would force ctx.Event onto the
	// heap every frame. PopulateBand recomputes it locally from b.ctx.Event so the
	// pointers stay on the worker stack; Prepare uses a local copy for gating.
	sharpness   uint8
	minPlane    loopfilter.Plane
	maxPlane    loopfilter.Plane
	planeActive [3]bool
	regionRows  int
	rows        int
	active      bool
}

// Active reports whether the frame owes a loop-filter pass; when false the caller
// applies nothing.
func (b FrameWorkLoopFilterMaskBands) Active() bool { return b.active }

// RegionRows is the number of 128x128 mask region-rows (the apply banding axis: a
// band is a contiguous sub-range of [0,RegionRows)).
func (b FrameWorkLoopFilterMaskBands) RegionRows() int { return b.regionRows }

// MIRows is the number of 4x4 MI rows (the level-cache populate banding axis: a
// populate band is a contiguous sub-range of block-origin MI rows [0,MIRows)).
func (b FrameWorkLoopFilterMaskBands) MIRows() int { return b.rows }

// PlaneActive reports whether the given plane deblocks this frame, mirroring the
// single-shot chroma gating (chroma is skipped when the frame luma level is zero or
// the chroma base level is zero). Luma is always active on an active frame.
func (b FrameWorkLoopFilterMaskBands) PlaneActive(plane loopfilter.Plane) bool {
	// loopfilter.Plane is unsigned, so only the upper bound can be out of range.
	if int(plane) >= len(b.planeActive) {
		return false
	}
	return b.planeActive[plane]
}

// MaxPlane is the highest plane index that may deblock (PlaneY for mono/chroma-less
// frames, otherwise PlaneV).
func (b FrameWorkLoopFilterMaskBands) MaxPlane() loopfilter.Plane { return b.maxPlane }

// PrepareLoopFilterMaskBands validates the deblocking masks and map and clears the
// frame-wide per-4x4 level cache, returning the shared band state. Unlike the
// single-shot ApplyLoopFilterEdgesFromMasks it does NOT populate the level cache or
// scan edges: the caller fans the O(frame) populate out across PopulateBand jobs
// (barrier), then the per-plane edge scan+filter across ApplyBand jobs (two
// direction phases). Deferring both is what lets the parallel apply match the
// edge-list parallel apply, which likewise plans in bands before filtering.
func (ctx FrameWorkPostFilterContext) PrepareLoopFilterMaskBands(masks *threading.FrameWorkLoopFilterMasks, filterMap FrameWorkLoopFilterMap) (FrameWorkLoopFilterMaskBands, error) {
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterLoopFilter) {
		return FrameWorkLoopFilterMaskBands{}, nil
	}
	if ctx.Output == nil {
		return FrameWorkLoopFilterMaskBands{}, frame.ErrInvalidSlot
	}
	if masks == nil || !masks.Valid() {
		return FrameWorkLoopFilterMaskBands{}, loopfilter.ErrInvalidFilter
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(ctx.Event.FrameSize)
	if err != nil {
		return FrameWorkLoopFilterMaskBands{}, err
	}
	if cols != masks.Cols || rows != masks.Rows {
		return FrameWorkLoopFilterMaskBands{}, threading.ErrInvalidBatch
	}
	if frameWorkLoopFilterMapEmpty(filterMap) && ctx.LoopFilterMap != nil {
		filterMap = *ctx.LoopFilterMap
	}
	if frameWorkLoopFilterMapEmpty(filterMap) {
		return FrameWorkLoopFilterMaskBands{}, threading.ErrInvalidBatch
	}
	if err := frameWorkValidateLoopFilterMap(filterMap, cols, rows); err != nil {
		return FrameWorkLoopFilterMaskBands{}, err
	}
	if len(masks.LevelCache) < cols*rows {
		return FrameWorkLoopFilterMaskBands{}, threading.ErrInvalidBatch
	}
	clear(masks.LevelCache[:cols*rows])
	levelCtx := frameWorkLoopFilterLevelContextFor(&ctx.Event)
	maxPlane := loopfilter.PlaneV
	if masks.Layout.Mono || ctx.Output.Format.MonoChrome || !masks.HasChroma {
		maxPlane = loopfilter.PlaneY
	}
	bands := FrameWorkLoopFilterMaskBands{
		ctx:        ctx,
		masks:      masks,
		filterMap:  filterMap,
		sharpness:  ctx.Event.LoopFilter.Sharpness,
		minPlane:   loopfilter.PlaneY,
		maxPlane:   maxPlane,
		regionRows: masks.SB128H,
		rows:       rows,
		active:     true,
	}
	chromaGated := levelCtx.lumaZero
	for plane := loopfilter.PlaneY; plane <= maxPlane; plane++ {
		if plane == loopfilter.PlaneY {
			bands.planeActive[plane] = true
			continue
		}
		bands.planeActive[plane] = !chromaGated && levelCtx.base[plane][loopfilter.EdgeVertical] != 0
	}
	return bands, nil
}

// PopulateBand fills the per-4x4 level cache from block origins whose MI row is in
// [rowStart,rowEnd). Prepare already cleared the whole cache; block coverage
// partitions the frame, so distinct origin-row bands write disjoint cells and run
// concurrently. Every PopulateBand must complete (barrier) before any ApplyBand,
// which reads the level cache.
func (b FrameWorkLoopFilterMaskBands) PopulateBand(rowStart, rowEnd int) error {
	if !b.active {
		return nil
	}
	// Recompute the level context locally (it holds pointers into b.ctx.Event);
	// keeping it off the shared struct keeps b.ctx.Event on this worker's stack.
	levelCtx := frameWorkLoopFilterLevelContextFor(&b.ctx.Event)
	var plan FrameWorkLoopFilterPostFilterPlan
	return b.ctx.populateLoopFilterLevelCacheRange(b.masks, b.filterMap, levelCtx, &plan, rowStart, rowEnd)
}

// RegionCols is the number of 128x128 mask region-columns (the horizontal-pass
// banding axis: an ApplyBandCols band is a contiguous sub-range of [0,RegionCols)).
func (b FrameWorkLoopFilterMaskBands) RegionCols() int { return b.masks.SB128W }

// ApplyBand scans and filters one plane's edges for one direction over the mask
// region-ROW range [rRow0,rRow1), all columns. Row-banding is race-free ONLY for
// VERTICAL edges: a vertical-edge filter modifies pixels left/right of the edge
// within the edge's own rows, so disjoint region-row bands (and distinct planes)
// write disjoint pixels. Horizontal edges modify pixels above/below the edge and
// straddle region-row seams, so they must be column-banded via ApplyBandCols
// instead. Callers must complete every vertical band before any horizontal band.
func (b FrameWorkLoopFilterMaskBands) ApplyBand(plane loopfilter.Plane, dir loopfilter.Edge, rRow0, rRow1 int) error {
	return b.applyBandRange(plane, dir, rRow0, rRow1, 0, b.masks.SB128W)
}

// ApplyBandCols scans and filters one plane's edges for one direction over the
// mask region-COLUMN range [rCol0,rCol1), all rows. This is the horizontal-pass
// dual of ApplyBand: a horizontal-edge filter modifies pixels above/below the
// edge within the edge's own COLUMNS, so disjoint region-column bands (and
// distinct planes) write disjoint pixels and run concurrently without straddling
// the region-row seams that make horizontal row-banding unsafe.
func (b FrameWorkLoopFilterMaskBands) ApplyBandCols(plane loopfilter.Plane, dir loopfilter.Edge, rCol0, rCol1 int) error {
	return b.applyBandRange(plane, dir, 0, b.masks.SB128H, rCol0, rCol1)
}

func (b FrameWorkLoopFilterMaskBands) applyBandRange(plane loopfilter.Plane, dir loopfilter.Edge, rRow0, rRow1, rCol0, rCol1 int) error {
	if !b.active {
		return nil
	}
	if !b.PlaneActive(plane) {
		return nil
	}
	dirMask := maskDirVertical
	if dir == loopfilter.EdgeHorizontal {
		dirMask = maskDirHorizontal
	}
	lc := lfmask.LevelCache{Cells: b.masks.LevelCache, Stride: b.masks.Cols}
	var result FrameWorkLoopFilterPostFilterApplyResult
	return b.ctx.applyLoopFilterMaskPlaneRange(&result, b.masks, lc, b.sharpness, plane, dirMask, rRow0, rRow1, rCol0, rCol1)
}

func frameWorkCountAppliedLoopFilterMaskEdge(result *FrameWorkLoopFilterPostFilterApplyResult, plane loopfilter.Plane, level uint8) {
	result.Edges++
	result.PlaneEdges[plane]++
	if level == 0 {
		return
	}
	result.Applied++
	result.PlaneApplied[plane]++
	if level > result.PlaneMaxLevel[plane] {
		result.PlaneMaxLevel[plane] = level
	}
	if level > result.MaxLevel {
		result.MaxLevel = level
	}
}
