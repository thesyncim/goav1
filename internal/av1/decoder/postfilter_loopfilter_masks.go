package decoder

import (
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
	levelCtx := frameWorkLoopFilterLevelContextFor(&ctx.Event)
	if err := ctx.populateLoopFilterLevelCacheFromMap(masks, filterMap, levelCtx); err != nil {
		return result, err
	}
	maxPlane := loopfilter.PlaneV
	if masks.Layout.Mono || ctx.Output.Format.MonoChrome || !masks.HasChroma {
		maxPlane = loopfilter.PlaneY
	}
	if err := ctx.applyLoopFilterEdgesFromMasksInPlanePassOrder(&result, masks, loopfilter.PlaneY, maxPlane); err != nil {
		return result, err
	}
	return result, nil
}

// populateLoopFilterLevelCacheFromMap fills the frame-wide per-4x4 level grid
// (masks.LevelCache) from the decoded loop-filter map, mirroring the level
// writes dav1d's create_lf_mask_{inter,intra} would perform (lfmask build.go).
// The level cells are order-independent, so a raster walk over block origins is
// sufficient.
func (ctx FrameWorkPostFilterContext) populateLoopFilterLevelCacheFromMap(masks *threading.FrameWorkLoopFilterMasks, filterMap FrameWorkLoopFilterMap, levelCtx frameWorkLoopFilterLevelContext) error {
	cols := masks.Cols
	rows := masks.Rows
	if len(masks.LevelCache) < cols*rows {
		return threading.ErrInvalidBatch
	}
	clear(masks.LevelCache[:cols*rows])
	ssHor := masks.Layout.SSHor
	ssVer := masks.Layout.SSVer
	hasChroma := masks.HasChroma
	stride := int(filterMap.Stride)
	for row := 0; row < rows; row++ {
		base := row * stride
		for col := 0; col < cols; col++ {
			record := &filterMap.Records[base+col]
			if !record.Valid {
				continue
			}
			if int(record.Block.MICol) != col || int(record.Block.MIRow) != row {
				continue
			}
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
				continue
			}
			cbw4 := minInt(((cols+ssHor)>>ssHor)-(bx>>ssHor), (int(dims.W4)+ssHor)>>ssHor)
			cbh4 := minInt(((rows+ssVer)>>ssVer)-(by>>ssVer), (int(dims.H4)+ssVer)>>ssVer)
			if cbw4 <= 0 || cbh4 <= 0 {
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
		}
	}
	return nil
}

func (ctx FrameWorkPostFilterContext) applyLoopFilterEdgesFromMasksInPlanePassOrder(result *FrameWorkLoopFilterPostFilterApplyResult, masks *threading.FrameWorkLoopFilterMasks, minPlane, maxPlane loopfilter.Plane) error {
	bytesPerSample := ctx.Output.Layout.BytesPerSample
	bitDepth := ctx.Output.Format.BitDepth
	sharpness := ctx.Event.LoopFilter.Sharpness
	lc := lfmask.LevelCache{Cells: masks.LevelCache, Stride: masks.Cols}

	var thresholds [loopfilter.MaxLevel + 1]loopfilter.Thresholds
	var thresholdReady [loopfilter.MaxLevel + 1]bool
	var firstErr error
	thresholdFor := func(level uint8) (loopfilter.Thresholds, bool) {
		if level == 0 || level > loopfilter.MaxLevel {
			firstErr = loopfilter.ErrInvalidFilter
			return loopfilter.Thresholds{}, false
		}
		if !thresholdReady[level] {
			th, err := loopfilter.ThresholdsForLevel(level, sharpness)
			if err != nil {
				firstErr = err
				return loopfilter.Thresholds{}, false
			}
			thresholds[level] = th
			thresholdReady[level] = true
		}
		return thresholds[level], true
	}

	for plane := minPlane; plane <= maxPlane && firstErr == nil; plane++ {
		dst, ok := frameWorkLoopFilterOutputPlane(*ctx.Output, plane)
		if !ok {
			if plane == loopfilter.PlaneY {
				return loopfilter.ErrInvalidFilter
			}
			continue
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

		applyCell := func(edge loopfilter.Edge, x4, y4 int, level, width uint8) {
			if firstErr != nil {
				return
			}
			clamped, err := frameWorkLoopFilterClampEdgeLengthInBounds(bounds, edge, x4, y4, 1)
			if err != nil {
				firstErr = err
				return
			}
			if clamped <= 0 {
				return
			}
			w, ok, err := frameWorkLoopFilterScheduledWidthInBounds(bounds, edge, x4, y4, 1, width)
			if err != nil {
				firstErr = err
				return
			}
			if !ok {
				return
			}
			th, ok := thresholdFor(level)
			if !ok {
				return
			}
			x := int32(x4) * 4
			y := int32(y4) * 4
			const length = int32(4)
			switch w {
			case 4:
				err = loopfilter.Filter4Edge(dst, bytesPerSample, bitDepth, edge, x, y, length, th)
			case 6:
				err = loopfilter.Filter6Edge(dst, bytesPerSample, bitDepth, edge, x, y, length, th)
			case 8:
				err = loopfilter.Filter8Edge(dst, bytesPerSample, bitDepth, edge, x, y, length, th)
			case 14:
				err = loopfilter.Filter14Edge(dst, bytesPerSample, bitDepth, edge, x, y, length, th)
			default:
				firstErr = loopfilter.ErrInvalidFilter
				return
			}
			if err != nil {
				firstErr = err
				return
			}
			frameWorkCountAppliedLoopFilterMaskEdge(result, plane, level)
		}

		if plane == loopfilter.PlaneY {
			ctx.applyLoopFilterMaskLumaPlane(masks, lc, applyCell)
		} else {
			ctx.applyLoopFilterMaskChromaPlane(masks, lc, plane, applyCell)
		}
	}
	return firstErr
}

// applyLoopFilterMaskLumaPlane scans the luma bitmasks (vertical edges first,
// then horizontal) and dispatches each set 4x4 edge through applyCell. The
// frame-left column (x4==0) and frame-top row (y4==0) are skipped to match the
// edge-list apply's have_left / have_top boundary skip.
func (ctx FrameWorkPostFilterContext) applyLoopFilterMaskLumaPlane(masks *threading.FrameWorkLoopFilterMasks, lc lfmask.LevelCache, applyCell func(edge loopfilter.Edge, x4, y4 int, level, width uint8)) {
	cols := masks.Cols
	rows := masks.Rows
	// Vertical edges (dir 0): position = column, scanned bits = rows.
	for rRow := 0; rRow < masks.SB128H; rRow++ {
		baseRow := rRow * 32
		extentRows := minInt(32, rows-baseRow)
		if extentRows <= 0 {
			continue
		}
		for rCol := 0; rCol < masks.SB128W; rCol++ {
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
				lfmask.ScanLuma(&region.Y[0][pos], 0, extentRows, func(off, widthClass int) {
					y4 := baseRow + off
					level := lc.Cells[y4*cols+col][loopfilter.EdgeVertical]
					if level == 0 {
						level = lc.Cells[y4*cols+col-1][loopfilter.EdgeVertical]
					}
					if level == 0 {
						return
					}
					applyCell(loopfilter.EdgeVertical, col, y4, level, frameWorkLoopFilterMaskLumaWidth[widthClass])
				})
			}
		}
	}
	// Horizontal edges (dir 1): position = row, scanned bits = columns.
	for rRow := 0; rRow < masks.SB128H; rRow++ {
		baseRow := rRow * 32
		extentRows := minInt(32, rows-baseRow)
		if extentRows <= 0 {
			continue
		}
		for rCol := 0; rCol < masks.SB128W; rCol++ {
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
				lfmask.ScanLuma(&region.Y[1][pos], 0, extentCols, func(off, widthClass int) {
					x4 := baseCol + off
					level := lc.Cells[row*cols+x4][loopfilter.EdgeHorizontal]
					if level == 0 {
						level = lc.Cells[(row-1)*cols+x4][loopfilter.EdgeHorizontal]
					}
					if level == 0 {
						return
					}
					applyCell(loopfilter.EdgeHorizontal, x4, row, level, frameWorkLoopFilterMaskLumaWidth[widthClass])
				})
			}
		}
	}
}

// applyLoopFilterMaskChromaPlane scans the chroma bitmasks for one chroma plane
// (U or V), vertical edges then horizontal, dispatching each set 4x4 chroma edge
// through applyCell in chroma-plane 4x4 coordinates.
func (ctx FrameWorkPostFilterContext) applyLoopFilterMaskChromaPlane(masks *threading.FrameWorkLoopFilterMasks, lc lfmask.LevelCache, plane loopfilter.Plane, applyCell func(edge loopfilter.Edge, x4, y4 int, level, width uint8)) {
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
	// Vertical edges (dir 0): position = chroma column, scanned bits = chroma rows.
	for rRow := 0; rRow < masks.SB128H; rRow++ {
		baseCRow := rRow * regionCH
		extentCRows := minInt(regionCH, crows-baseCRow)
		if extentCRows <= 0 {
			continue
		}
		for rCol := 0; rCol < masks.SB128W; rCol++ {
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
				lfmask.ScanChroma(&region.UV[0][pos], 0, extentCRows, laneBitsV, func(off, widthClass int) {
					crow := baseCRow + off
					level := lc.Cells[crow*cols+ccol][levelIdx]
					if level == 0 {
						level = lc.Cells[crow*cols+ccol-1][levelIdx]
					}
					if level == 0 {
						return
					}
					applyCell(loopfilter.EdgeVertical, ccol, crow, level, frameWorkLoopFilterMaskChromaWidth[widthClass])
				})
			}
		}
	}
	// Horizontal edges (dir 1): position = chroma row, scanned bits = chroma columns.
	for rRow := 0; rRow < masks.SB128H; rRow++ {
		baseCRow := rRow * regionCH
		extentCRows := minInt(regionCH, crows-baseCRow)
		if extentCRows <= 0 {
			continue
		}
		for rCol := 0; rCol < masks.SB128W; rCol++ {
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
				lfmask.ScanChroma(&region.UV[1][pos], 0, extentCCols, laneBitsH, func(off, widthClass int) {
					ccol := baseCCol + off
					level := lc.Cells[crow*cols+ccol][levelIdx]
					if level == 0 {
						level = lc.Cells[(crow-1)*cols+ccol][levelIdx]
					}
					if level == 0 {
						return
					}
					applyCell(loopfilter.EdgeHorizontal, ccol, crow, level, frameWorkLoopFilterMaskChromaWidth[widthClass])
				})
			}
		}
	}
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
