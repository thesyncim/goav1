package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// frameWorkCDEFHBDPlaneViewAvailable reports whether the snapshot-free path
// can filter this byte plane in place: a valid plane with an even byte stride.
func frameWorkCDEFHBDPlaneViewAvailable(plane frame.Plane) bool {
	_, err := frameWorkCDEFPlaneHBDStride(plane)
	return err == nil
}

func frameWorkApplyCDEFPlaneRowsHBD(params parser.CDEFParams, indexMap FrameWorkCDEFIndexMap, skipMap *FrameWorkLoopFilterMap, cols, rows, rowStart, rowEnd int, dst frame.Plane, lineScratch []uint16, boundary FrameWorkCDEFPostFilterHBD16BandBoundary, input, unitDst []uint16, blockStorage []cdef.BlockPosition, directions *cdef.DirectionGrid, variances *cdef.VarianceGrid, directionGrid []cdef.DirectionGrid, varianceGrid []cdef.VarianceGrid, plane, xDec, yDec, coeffShift int, forceLumaDirections bool, unitMasks *frameWorkCDEFUnitMaskCache) (uint32, uint32, error) {
	return frameWorkApplyCDEFPlaneRowsHBDLE(params, indexMap, skipMap, cols, rows, rowStart, rowEnd, dst, lineScratch, boundary, input, unitDst, blockStorage, directions, variances, directionGrid, varianceGrid, plane, xDec, yDec, coeffShift, forceLumaDirections, unitMasks)
}

// frameWorkApplyCDEFPlaneRowsHBDLE is the 10/12-bit in-place counterpart of
// frameWorkApplyCDEFPlaneRows. It follows dav1d's cdef_brow ownership: before
// a unit row is filtered, it backs up the two pre-filter rows needed by the
// next row; before a unit is filtered, it backs up the pre-filter eight-column
// left halo needed by the next unit. Top/bottom band boundaries are supplied
// by the caller before any bands mutate the frame. Samples are decoded from
// the little-endian byte plane into the tap buffer and backups; the CDEF
// kernel writes a unit into unitDst and only the filtered blocks are encoded
// back, so skipped blocks stay in place and need no UnitDst prefill.
func frameWorkApplyCDEFPlaneRowsHBDLE(params parser.CDEFParams, indexMap FrameWorkCDEFIndexMap, skipMap *FrameWorkLoopFilterMap, cols, rows, rowStart, rowEnd int, dst frame.Plane, lineScratch []uint16, boundary FrameWorkCDEFPostFilterHBD16BandBoundary, input, unitDst []uint16, blockStorage []cdef.BlockPosition, directions *cdef.DirectionGrid, variances *cdef.VarianceGrid, directionGrid []cdef.DirectionGrid, varianceGrid []cdef.VarianceGrid, plane, xDec, yDec, coeffShift int, forceLumaDirections bool, unitMasks *frameWorkCDEFUnitMaskCache) (uint32, uint32, error) {
	var units uint32
	var blocksTotal uint32
	if coeffShift <= 0 || coeffShift > 4 || dst.Stride%2 != 0 {
		return 0, 0, frame.ErrInvalidFormat
	}
	stride, err := frameWorkCDEFPlaneHBDStride(dst)
	if err != nil {
		return 0, 0, err
	}
	pix, strideB := dst.Pix, dst.Stride
	width := dst.Width
	height := dst.Height
	if width <= 0 || height <= 0 || stride < width ||
		(height-1)*strideB+2*width > len(pix) {
		return 0, 0, frame.ErrInvalidPlane
	}
	if rowEnd < 0 || rowEnd > rows {
		rowEnd = rows
	}
	if rowStart < 0 || rowStart >= rowEnd {
		if rowStart == rowEnd {
			return 0, 0, nil
		}
		return 0, 0, threading.ErrInvalidBatch
	}
	if len(input) < cdef.InputBufferSize || len(lineScratch) < 4*width {
		return 0, 0, frame.ErrShortBuffer
	}
	if rowStart > 0 && len(boundary.Top[plane]) < cdef.VerticalBorder*width {
		return 0, 0, frame.ErrShortBuffer
	}
	if rowEnd < rows && len(boundary.Bottom[plane]) < cdef.VerticalBorder*width {
		return 0, 0, frame.ErrShortBuffer
	}

	unitSizeX := cdef.BlockSize >> xDec
	unitSizeY := cdef.BlockSize >> yDec
	blockWidth := 8 >> xDec
	blockHeight := 8 >> yDec
	indexStride := int(indexMap.Stride)
	lineBuf := lineScratch[:4*width]
	if rowStart > 0 {
		need := cdef.VerticalBorder * width
		topSlot := ((rowStart ^ 1) & 1) * need
		copy(lineBuf[topSlot:topSlot+need], boundary.Top[plane][:need])
	}
	var bottomBoundary []uint16
	if rowEnd < rows {
		bottomBoundary = boundary.Bottom[plane]
	}
	var leftStorage [cdef.HorizontalBorder * cdef.BlockSize]uint16

	for unitRow := rowStart; unitRow < rowEnd; unitRow++ {
		unitY := (unitRow * cdef.BlockSize) >> yDec
		// Save the two pre-filter lines the next unit row reads as its top
		// halo before this row writes any blocks into those samples.
		if unitRow+1 < rowEnd {
			nextTopY := unitY + unitSizeY
			save := lineBuf[(unitRow&1)*2*width:]
			for r := 0; r < cdef.VerticalBorder; r++ {
				y := nextTopY - cdef.VerticalBorder + r
				if y < 0 || y >= height {
					continue
				}
				frame.LoadSampleRows16Trusted(save[r*width:r*width+width], width, pix[y*strideB:], strideB, width, 1)
			}
		}
		topBuf := lineBuf[((unitRow^1)&1)*2*width:]
		bottomBuf := []uint16(nil)
		if unitRow+1 == rowEnd {
			bottomBuf = bottomBoundary
		}
		prevFiltered := false
		for unitCol := range cols {
			mapOffset := unitRow*indexStride + unitCol
			if !indexMap.Read[mapOffset] {
				prevFiltered = false
				continue
			}
			strengthIndex := indexMap.Index[mapOffset]
			if int(strengthIndex) >= int(params.StrengthCount) || int(strengthIndex) >= parser.MaxCDEFStrengths {
				return units, blocksTotal, threading.ErrInvalidBatch
			}
			index := int(strengthIndex)
			packed := params.YStrength[index]
			if plane != 0 {
				packed = params.UVStrength[index]
			}
			needPrimaryDirection := frameWorkCDEFPlaneNeedsPrimaryDirection(params, index, plane, forceLumaDirections)
			directionOnly := plane == 0 && forceLumaDirections && packed == 0 && params.UVStrength[index]>>2 != 0
			if packed == 0 && !directionOnly {
				prevFiltered = false
				continue
			}
			unitX := (unitCol * cdef.BlockSize) >> xDec
			unitW := minInt(unitSizeX, width-unitX)
			unitH := minInt(unitSizeY, height-unitY)
			if unitW <= 0 || unitH <= 0 {
				prevFiltered = false
				continue
			}
			blocks := frameWorkCDEFUnitBlockList(blockStorage, unitW, unitH, blockWidth, blockHeight, skipMap, unitMasks, unitRow, unitCol)
			if len(blocks) == 0 {
				prevFiltered = false
				continue
			}

			unitIndex := unitRow*cols + unitCol
			unitDirections := directions
			unitVariances := variances
			if unitIndex < len(directionGrid) && unitIndex < len(varianceGrid) {
				unitDirections = &directionGrid[unitIndex]
				unitVariances = &varianceGrid[unitIndex]
			}
			cdefPlane := cdef.PlaneY
			if plane == 1 {
				cdefPlane = cdef.PlaneU
			} else if plane == 2 {
				cdefPlane = cdef.PlaneV
			}
			filterParams, err := cdef.FrameFilterParamsFromStrength(cdefPlane, xDec, yDec, packed, int(params.Damping), coeffShift)
			if err != nil {
				return units, blocksTotal, err
			}
			// The assembler writes sentinels to every uncovered ring cell itself;
			// pre-filling the full ring here would write those cells twice.
			noSentinels, err := frameWorkAssembleCDEFInputHBDLE(input, pix, strideB, width, height, topBuf, bottomBuf, leftStorage[:], prevFiltered, unitRow, unitX, unitY, unitW, unitH)
			if err != nil {
				return units, blocksTotal, err
			}

			// Preserve the source columns for the next unit before this kernel
			// overwrites the current unit in place.
			nextUnitX := unitX + unitSizeX
			if nextUnitX < width {
				frame.LoadSampleRows16Trusted(leftStorage[:], cdef.HorizontalBorder, pix[unitY*strideB+2*(nextUnitX-cdef.HorizontalBorder):], strideB, cdef.HorizontalBorder, unitH)
			}
			if cdefDebugUnit(plane, unitRow, unitCol) {
				cdefDebugLogUnit(plane, unitRow, unitCol, packed, filterParams, *unitDirections, *unitVariances, input, len(blocks))
			}

			unitPix := pix[unitY*strideB+2*unitX:]
			if needPrimaryDirection {
				err = cdef.FilterFrameBlocksU16LETrusted(unitPix, strideB, input, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, blocks, unitDirections, unitVariances, filterParams, noSentinels)
			} else {
				err = cdef.FilterFrameBlocksU16LETrustedNoPrimaryDirection(unitPix, strideB, input, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, blocks, unitDirections, unitVariances, filterParams, noSentinels)
			}
			if err != nil {
				return units, blocksTotal, err
			}
			if directionOnly {
				// Zero strengths leave every sample as it is in the frame.
				prevFiltered = false
				continue
			}
			prevFiltered = true
			units++
			blocksTotal += uint32(len(blocks))
		}
	}
	return units, blocksTotal, nil
}

// frameWorkCDEFPlaneHBDStride validates a 16-bit frame plane for the in-place
// walk and returns its stride in samples.
func frameWorkCDEFPlaneHBDStride(plane frame.Plane) (int, error) {
	need, err := frame.SamplePlaneLen(plane, 2)
	if err != nil || plane.Stride%2 != 0 || need <= 0 || need > int(^uint(0)>>1)/2 || len(plane.Pix) < need*2 {
		return 0, frame.ErrInvalidPlane
	}
	return plane.Stride / 2, nil
}

// frameWorkAssembleCDEFInputHBDLE builds the padded uint16 tap buffer from the
// little-endian frame plane bytes (stride strideB) and the same pre-filter backups as the
// 8-bit walk. Frame and crop edge extension intentionally mirrors
// frameWorkCopyCDEFInput; the full aligned plane extent is passed as width and
// height so allocated MI padding remains real input.
func frameWorkAssembleCDEFInputHBDLE(input []uint16, pix []byte, strideB, width, height int, topBuf, bottomBuf, leftBuf []uint16, leftFromBackup bool, unitRow, unitX, unitY, unitW, unitH int) (bool, error) {
	fillW := unitW + 2*cdef.HorizontalBorder
	fillH := unitH + 2*cdef.VerticalBorder
	if unitW <= 0 || unitH <= 0 || fillW > cdef.BStride || !cdefInputRectFits(len(input), fillW, fillH) {
		return false, threading.ErrInvalidBatch
	}
	srcX0 := maxInt(unitX-cdef.HorizontalBorder, 0)
	srcY0 := maxInt(unitY-cdef.VerticalBorder, 0)
	srcX1 := minInt(unitX+unitW+cdef.HorizontalBorder, width)
	srcY1 := minInt(unitY+unitH+cdef.VerticalBorder, height)
	wantSrcX1 := unitX + unitW + cdef.HorizontalBorder
	wantSrcY1 := unitY + unitH + cdef.VerticalBorder
	rowExtend := srcY1 < wantSrcY1 && srcY1 > 0 && unitY+unitH < height
	colExtend := srcX1 < wantSrcX1 && srcX1 > 0 && unitX+unitW < width
	xLo := cdef.HorizontalBorder + srcX0 - unitX
	xHiEff := cdef.HorizontalBorder + srcX1 - unitX
	if colExtend {
		xHiEff = fillW
	}
	rowCovStart := cdef.VerticalBorder + srcY0 - unitY
	rowCovEnd := cdef.VerticalBorder + srcY1 - unitY
	if rowExtend {
		rowCovEnd = fillH
	}
	noSentinels := rowCovStart == 0 && rowCovEnd == fillH && xLo == 0 && xHiEff == fillW
	if !noSentinels {
		for row := 0; row < fillH; row++ {
			rowBuf := input[row*cdef.BStride:]
			if row < rowCovStart || row >= rowCovEnd {
				frameWorkFillCDEFInputSentinelRow(rowBuf, fillW)
				continue
			}
			if xLo > 0 {
				frameWorkFillCDEFInputSentinelRow(rowBuf, xLo)
			}
			if xHiEff < fillW {
				frameWorkFillCDEFInputSentinelRow(rowBuf[xHiEff:], fillW-xHiEff)
			}
		}
	}
	copyW := srcX1 - srcX0
	dstOffset := cdef.VerticalBorder*cdef.BStride + cdef.HorizontalBorder +
		(srcY0-unitY)*cdef.BStride + (srcX0 - unitX)
	if topRows := unitY - srcY0; topRows > 0 {
		if unitRow == 0 || len(topBuf) < cdef.VerticalBorder*width {
			return false, threading.ErrInvalidBatch
		}
		for row := 0; row < topRows; row++ {
			srcOff := row*width + srcX0
			dstOff := dstOffset + row*cdef.BStride
			copy(input[dstOff:dstOff+copyW], topBuf[srcOff:srcOff+copyW])
		}
	}
	midOffset := dstOffset + (unitY-srcY0)*cdef.BStride
	leftW := unitX - srcX0
	midW := srcX1 - unitX
	if leftW > 0 && leftFromBackup {
		if len(leftBuf) < cdef.HorizontalBorder*unitH {
			return false, threading.ErrInvalidBatch
		}
		for row := 0; row < unitH; row++ {
			dstOff := midOffset + row*cdef.BStride
			if leftW == cdef.HorizontalBorder {
				// A fixed-size move, not a memmove call.
				*(*[cdef.HorizontalBorder]uint16)(input[dstOff:]) = *(*[cdef.HorizontalBorder]uint16)(leftBuf[row*cdef.HorizontalBorder:])
			} else {
				copy(input[dstOff:dstOff+leftW], leftBuf[row*cdef.HorizontalBorder:row*cdef.HorizontalBorder+leftW])
			}
		}
		frame.LoadSampleRows16Trusted(input[midOffset+leftW:], cdef.BStride, pix[unitY*strideB+2*unitX:], strideB, midW, unitH)
	} else {
		// The left halo, when present, is the plane's own samples, contiguous
		// with the unit's: one copy per row covers both.
		frame.LoadSampleRows16Trusted(input[midOffset:], cdef.BStride, pix[unitY*strideB+2*srcX0:], strideB, leftW+midW, unitH)
	}
	if botRows := srcY1 - (unitY + unitH); botRows > 0 {
		if bottomBuf != nil {
			if len(bottomBuf) < cdef.VerticalBorder*width {
				return false, threading.ErrInvalidBatch
			}
			for row := 0; row < botRows; row++ {
				srcOff := row*width + srcX0
				dstOff := dstOffset + (unitY+unitH-srcY0+row)*cdef.BStride
				copy(input[dstOff:dstOff+copyW], bottomBuf[srcOff:srcOff+copyW])
			}
		} else {
			frame.LoadSampleRows16Trusted(input[dstOffset+(unitY+unitH-srcY0)*cdef.BStride:], cdef.BStride, pix[(unitY+unitH)*strideB+2*srcX0:], strideB, copyW, botRows)
		}
	}
	copyH := srcY1 - srcY0
	if rowExtend {
		lastRowOffset := dstOffset + (copyH-1)*cdef.BStride
		for row := srcY1; row < wantSrcY1; row++ {
			extOffset := dstOffset + (row-srcY0)*cdef.BStride
			copy(input[extOffset:extOffset+copyW], input[lastRowOffset:lastRowOffset+copyW])
		}
	}
	if colExtend {
		extRows := copyH
		if rowExtend {
			extRows = wantSrcY1 - srcY0
		}
		for row := 0; row < extRows; row++ {
			rowOffset := dstOffset + row*cdef.BStride
			last := input[rowOffset+copyW-1]
			for i := rowOffset + copyW; i < rowOffset+wantSrcX1-srcX0; i++ {
				input[i] = last
			}
		}
	}
	return noSentinels, nil
}

// LoadCDEFPostFilterHBD16BandBoundary snapshots only the two top and two bottom
// pre-CDEF rows an in-place 10/12-bit unit-row band needs across band seams.
// It must run before any parallel ApplyCDEFPostFilterUnitRowsHBD calls.
func (ctx *FrameWorkPostFilterContext) LoadCDEFPostFilterHBD16BandBoundary(boundary *FrameWorkCDEFPostFilterHBD16BandBoundary, rowStart, rowEnd int) error {
	if boundary == nil {
		return frame.ErrShortBuffer
	}
	remaining := ctx.RemainingPostFilters()
	if remaining.Has(FrameWorkPostFilterLoopFilter) {
		return ErrUnsupportedPostFilter
	}
	if !remaining.Has(FrameWorkPostFilterCDEF) {
		return nil
	}
	if ctx.Output == nil {
		return frame.ErrInvalidSlot
	}
	if !frameWorkCDEFHasFiltering(ctx.Event.CDEF, ctx.Output.Format.MonoChrome) {
		return nil
	}
	coeffShift := int(ctx.Output.Format.BitDepth) - 8
	if ctx.Output.Layout.BytesPerSample != 2 || coeffShift <= 0 || coeffShift > 4 {
		return frame.ErrInvalidFormat
	}
	_, rows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
	if err != nil {
		return err
	}
	if rowEnd < 0 || rowEnd > rows {
		rowEnd = rows
	}
	if rowStart < 0 || rowStart >= rowEnd {
		if rowStart == rowEnd {
			return nil
		}
		return threading.ErrInvalidBatch
	}
	chromaFiltering := !ctx.Output.Format.MonoChrome && frameWorkCDEFChromaHasFiltering(ctx.Event.CDEF)
	for plane := range 3 {
		planeFrame, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		processPlane := frameWorkCDEFPlaneHasFiltering(ctx.Event.CDEF, plane)
		if plane == 0 && !processPlane {
			processPlane = chromaFiltering
		}
		if !ok || !processPlane {
			continue
		}
		xDec, yDec := frameWorkCDEFPlaneDecimation(ctx.Output.Format, plane)
		planeFrame = frameWorkCDEFAlignedPlane(planeFrame, ctx.Event.FrameSize, xDec, yDec, ctx.Output.Layout.BytesPerSample)
		if err := loadCDEFHBDBandBoundaryPlane(boundary, plane, planeFrame, rowStart, rowEnd, rows, yDec); err != nil {
			return err
		}
	}
	return nil
}

func loadCDEFHBDBandBoundaryPlane(boundary *FrameWorkCDEFPostFilterHBD16BandBoundary, plane int, planeFrame frame.Plane, rowStart, rowEnd, rows, yDec int) error {
	width := planeFrame.Width
	height := planeFrame.Height
	strideB := planeFrame.Stride
	if width <= 0 || height <= 0 || strideB%2 != 0 || strideB < 2*width ||
		(height-1)*strideB+2*width > len(planeFrame.Pix) {
		return frame.ErrInvalidPlane
	}
	need := cdef.VerticalBorder * width
	if rowStart > 0 {
		top := boundary.Top[plane]
		if len(top) < need {
			return frame.ErrShortBuffer
		}
		y0 := (rowStart*cdef.BlockSize)>>yDec - cdef.VerticalBorder
		for r := 0; r < cdef.VerticalBorder; r++ {
			y := y0 + r
			if y < 0 || y >= height {
				return frame.ErrInvalidPlane
			}
			frame.LoadSampleRows16Trusted(top[r*width:(r+1)*width], width, planeFrame.Pix[y*strideB:], strideB, width, 1)
		}
	}
	if rowEnd < rows {
		bottom := boundary.Bottom[plane]
		if len(bottom) < need {
			return frame.ErrShortBuffer
		}
		y0 := (rowEnd * cdef.BlockSize) >> yDec
		for r := 0; r < cdef.VerticalBorder; r++ {
			y := y0 + r
			if y < 0 || y >= height {
				return frame.ErrInvalidPlane
			}
			frame.LoadSampleRows16Trusted(bottom[r*width:(r+1)*width], width, planeFrame.Pix[y*strideB:], strideB, width, 1)
		}
	}
	return nil
}

// ApplyCDEFPostFilterUnitRowsHBD applies a 10/12-bit unit-row band in place
// directly to the little-endian picture planes using immutable boundary rows
// loaded by LoadCDEFPostFilterHBD16BandBoundary.
func (ctx *FrameWorkPostFilterContext) ApplyCDEFPostFilterUnitRowsHBD(req FrameWorkCDEFPostFilterRequest, boundary FrameWorkCDEFPostFilterHBD16BandBoundary, rowStart, rowEnd int) (FrameWorkCDEFPostFilterResult, error) {
	remaining := ctx.RemainingPostFilters()
	if remaining.Has(FrameWorkPostFilterLoopFilter) {
		return FrameWorkCDEFPostFilterResult{}, ErrUnsupportedPostFilter
	}
	if !remaining.Has(FrameWorkPostFilterCDEF) {
		return FrameWorkCDEFPostFilterResult{}, nil
	}
	if ctx.Output == nil {
		return FrameWorkCDEFPostFilterResult{}, frame.ErrInvalidSlot
	}
	if !frameWorkCDEFHasFiltering(ctx.Event.CDEF, ctx.Output.Format.MonoChrome) {
		return FrameWorkCDEFPostFilterResult{}, nil
	}
	cols, rows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
	if err != nil {
		return FrameWorkCDEFPostFilterResult{}, err
	}
	if rowEnd < 0 || rowEnd > rows {
		rowEnd = rows
	}
	if rowStart < 0 || rowStart >= rowEnd {
		if rowStart == rowEnd {
			return FrameWorkCDEFPostFilterResult{}, nil
		}
		return FrameWorkCDEFPostFilterResult{}, threading.ErrInvalidBatch
	}
	indexMap := req.IndexMap
	if frameWorkCDEFIndexMapEmpty(indexMap) && ctx.CDEFIndexMap != nil {
		indexMap = *ctx.CDEFIndexMap
	}
	if err := frameWorkValidateCDEFIndexMap(indexMap, cols, rows); err != nil {
		return FrameWorkCDEFPostFilterResult{}, err
	}
	chromaFiltering := !ctx.Output.Format.MonoChrome && frameWorkCDEFChromaHasFiltering(ctx.Event.CDEF)
	coeffShift := int(ctx.Output.Format.BitDepth) - 8
	if ctx.Output.Layout.BytesPerSample != 2 || coeffShift <= 0 || coeffShift > 4 {
		return FrameWorkCDEFPostFilterResult{}, frame.ErrInvalidFormat
	}
	if len(req.InputScratch) < cdef.InputBufferSize {
		return FrameWorkCDEFPostFilterResult{}, frame.ErrShortBuffer
	}
	if chromaFiltering && (len(req.DirectionGrid) < cols*rows || len(req.VarianceGrid) < cols*rows) {
		return FrameWorkCDEFPostFilterResult{}, frame.ErrShortBuffer
	}
	skipMap := ctx.LoopFilterMap
	var result FrameWorkCDEFPostFilterResult
	var directions cdef.DirectionGrid
	var variances cdef.VarianceGrid
	var blockStorage [cdef.NBlocks * cdef.NBlocks]cdef.BlockPosition
	var unitMaskCache frameWorkCDEFUnitMaskCache
	var unitMasks *frameWorkCDEFUnitMaskCache
	if ctx.LoopFilterMasks.NoSkipUnitGrid(cols, rows) {
		unitMaskCache.resetNoSkip(ctx.LoopFilterMasks, rowStart, cols)
		unitMasks = &unitMaskCache
	}
	for plane := range 3 {
		planeFrame, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		processPlane := frameWorkCDEFPlaneHasFiltering(ctx.Event.CDEF, plane)
		if plane == 0 && !processPlane {
			processPlane = chromaFiltering
		}
		if !ok || !processPlane {
			continue
		}
		xDec, yDec := frameWorkCDEFPlaneDecimation(ctx.Output.Format, plane)
		planeFrame = frameWorkCDEFAlignedPlane(planeFrame, ctx.Event.FrameSize, xDec, yDec, ctx.Output.Layout.BytesPerSample)
		lineNeed := 4 * planeFrame.Width
		if len(req.SampleScratch[plane]) < lineNeed {
			return FrameWorkCDEFPostFilterResult{}, frame.ErrShortBuffer
		}
		planeUnits, planeBlocks, err := frameWorkApplyCDEFPlaneRowsHBD(ctx.Event.CDEF, indexMap, skipMap, cols, rows, rowStart, rowEnd, planeFrame, req.SampleScratch[plane][:lineNeed], boundary, req.InputScratch[:cdef.InputBufferSize], nil, blockStorage[:], &directions, &variances, req.DirectionGrid, req.VarianceGrid, plane, xDec, yDec, coeffShift, chromaFiltering, unitMasks)
		if err != nil {
			return FrameWorkCDEFPostFilterResult{}, err
		}
		result.Units += planeUnits
		result.Blocks += planeBlocks
		result.Planes++
	}
	return result, nil
}

func (ctx *FrameWorkPostFilterContext) cdefHBDPlanesAvailable() bool {
	if ctx.Output == nil || ctx.Output.Layout.BytesPerSample != 2 {
		return false
	}
	coeffShift := int(ctx.Output.Format.BitDepth) - 8
	if coeffShift <= 0 || coeffShift > 4 {
		return false
	}
	chromaFiltering := !ctx.Output.Format.MonoChrome && frameWorkCDEFChromaHasFiltering(ctx.Event.CDEF)
	for plane := range 3 {
		planeFrame, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		processPlane := frameWorkCDEFPlaneHasFiltering(ctx.Event.CDEF, plane)
		if plane == 0 && !processPlane {
			processPlane = chromaFiltering
		}
		if !ok || !processPlane {
			continue
		}
		xDec, yDec := frameWorkCDEFPlaneDecimation(ctx.Output.Format, plane)
		planeFrame = frameWorkCDEFAlignedPlane(planeFrame, ctx.Event.FrameSize, xDec, yDec, ctx.Output.Layout.BytesPerSample)
		if !frameWorkCDEFHBDPlaneViewAvailable(planeFrame) {
			return false
		}
	}
	return true
}
