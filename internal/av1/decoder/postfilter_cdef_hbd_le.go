//go:build 386 || amd64 || arm || arm64 || loong64 || mips64le || mipsle || ppc64le || riscv64 || wasm

package decoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// frameWorkCDEFHBDInPlaceAvailable is true only on architectures where the
// frame package treats its 16-bit pixel bytes as native little-endian uint16s.
// The generic-endian implementation keeps using the portable snapshot path.
func frameWorkCDEFHBDInPlaceAvailable() bool { return true }

// frameWorkCDEFHBDPlaneViewAvailable reports whether the snapshot-free path can
// view this valid byte plane as native uint16 samples. Misaligned but otherwise
// valid planes remain supported by the portable snapshot route.
func frameWorkCDEFHBDPlaneViewAvailable(plane frame.Plane) bool {
	_, _, err := frameWorkCDEFPlaneU16ViewLE(plane)
	return err == nil
}

func frameWorkApplyCDEFPlaneRowsHBD(params parser.CDEFParams, indexMap FrameWorkCDEFIndexMap, skipMap *FrameWorkLoopFilterMap, cols, rows, rowStart, rowEnd int, dst frame.Plane, lineScratch []uint16, boundary FrameWorkCDEFPostFilterHBD16BandBoundary, input []uint16, blockStorage []cdef.BlockPosition, directions *cdef.DirectionGrid, variances *cdef.VarianceGrid, directionGrid []cdef.DirectionGrid, varianceGrid []cdef.VarianceGrid, plane, xDec, yDec, coeffShift int, forceLumaDirections bool) (uint32, uint32, error) {
	return frameWorkApplyCDEFPlaneRowsHBDLE(params, indexMap, skipMap, cols, rows, rowStart, rowEnd, dst, lineScratch, boundary, input, blockStorage, directions, variances, directionGrid, varianceGrid, plane, xDec, yDec, coeffShift, forceLumaDirections)
}

// frameWorkApplyCDEFPlaneRowsHBDLE is the 10/12-bit in-place counterpart of
// frameWorkApplyCDEFPlaneRows. It follows dav1d's cdef_brow ownership: before
// a unit row is filtered, it backs up the two pre-filter rows needed by the
// next row; before a unit is filtered, it backs up the pre-filter eight-column
// left halo needed by the next unit. Top/bottom band boundaries are supplied
// by the caller before any bands mutate the frame. The CDEF kernel writes to a
// short-lived LE uint16 view of the byte plane, so skipped blocks remain in
// place and need neither UnitDst prefill nor a store-back pass.
func frameWorkApplyCDEFPlaneRowsHBDLE(params parser.CDEFParams, indexMap FrameWorkCDEFIndexMap, skipMap *FrameWorkLoopFilterMap, cols, rows, rowStart, rowEnd int, dst frame.Plane, lineScratch []uint16, boundary FrameWorkCDEFPostFilterHBD16BandBoundary, input []uint16, blockStorage []cdef.BlockPosition, directions *cdef.DirectionGrid, variances *cdef.VarianceGrid, directionGrid []cdef.DirectionGrid, varianceGrid []cdef.VarianceGrid, plane, xDec, yDec, coeffShift int, forceLumaDirections bool) (uint32, uint32, error) {
	var units uint32
	var blocksTotal uint32
	if coeffShift <= 0 || coeffShift > 4 || dst.Stride%2 != 0 {
		return 0, 0, frame.ErrInvalidFormat
	}
	pixels, stride, err := frameWorkCDEFPlaneU16ViewLE(dst)
	if err != nil {
		return 0, 0, err
	}
	width := dst.Width
	height := dst.Height
	if width <= 0 || height <= 0 || stride < width ||
		(height-1)*stride+width > len(pixels) {
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
				copy(save[r*width:r*width+width], pixels[y*stride:y*stride+width])
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
			blocks := frameWorkCDEFBlockPositionsFiltered(blockStorage, unitW, unitH, blockWidth, blockHeight, skipMap, unitRow, unitCol)
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
			if err := frameWorkAssembleCDEFInputHBDLE(input, pixels, stride, width, height, topBuf, bottomBuf, leftStorage[:], prevFiltered, unitRow, unitX, unitY, unitW, unitH); err != nil {
				return units, blocksTotal, err
			}

			// Preserve the source columns for the next unit before this kernel
			// overwrites the current unit in place.
			nextUnitX := unitX + unitSizeX
			if nextUnitX < width {
				for r := 0; r < unitH; r++ {
					rowOff := (unitY+r)*stride + nextUnitX - cdef.HorizontalBorder
					copy(leftStorage[r*cdef.HorizontalBorder:(r+1)*cdef.HorizontalBorder], pixels[rowOff:rowOff+cdef.HorizontalBorder])
				}
			}
			if cdefDebugUnit(plane, unitRow, unitCol) {
				cdefDebugLogUnit(plane, unitRow, unitCol, packed, filterParams, *unitDirections, *unitVariances, input, len(blocks))
			}

			unitOrigin := unitY*stride + unitX
			if needPrimaryDirection {
				err = cdef.FilterFrameBlocksTrusted(pixels[unitOrigin:], stride, input, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, blocks, unitDirections, unitVariances, filterParams)
			} else {
				err = cdef.FilterFrameBlocksTrustedNoPrimaryDirection(pixels[unitOrigin:], stride, input, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, blocks, unitDirections, unitVariances, filterParams)
			}
			if err != nil {
				return units, blocksTotal, err
			}
			if directionOnly {
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

// frameWorkCDEFPlaneU16ViewLE returns a short-lived uint16 view of a validated
// little-endian frame plane. The caller must retain plane.Pix and use the view
// synchronously; it must not store the view in long-lived state.
func frameWorkCDEFPlaneU16ViewLE(plane frame.Plane) ([]uint16, int, error) {
	need, err := frame.SamplePlaneLen(plane, 2)
	if err != nil || plane.Stride%2 != 0 || need <= 0 || need > int(^uint(0)>>1)/2 || len(plane.Pix) < need*2 {
		return nil, 0, frame.ErrInvalidPlane
	}
	base := unsafe.SliceData(plane.Pix)
	if base == nil || uintptr(unsafe.Pointer(base))%unsafe.Alignof(uint16(0)) != 0 {
		return nil, 0, frame.ErrInvalidPlane
	}
	return unsafe.Slice((*uint16)(unsafe.Pointer(base)), need), plane.Stride / 2, nil
}

// frameWorkAssembleCDEFInputHBDLE builds the padded uint16 tap buffer from the
// in-place little-endian frame plane and the same pre-filter backups as the
// 8-bit walk. Frame and crop edge extension intentionally mirrors
// frameWorkCopyCDEFInput; the full aligned plane extent is passed as width and
// height so allocated MI padding remains real input.
func frameWorkAssembleCDEFInputHBDLE(input, pixels []uint16, stride, width, height int, topBuf, bottomBuf, leftBuf []uint16, leftFromBackup bool, unitRow, unitX, unitY, unitW, unitH int) error {
	fillW := unitW + 2*cdef.HorizontalBorder
	fillH := unitH + 2*cdef.VerticalBorder
	if unitW <= 0 || unitH <= 0 || fillW > cdef.BStride || !cdefInputRectFits(len(input), fillW, fillH) {
		return threading.ErrInvalidBatch
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
	copyW := srcX1 - srcX0
	dstOffset := cdef.VerticalBorder*cdef.BStride + cdef.HorizontalBorder +
		(srcY0-unitY)*cdef.BStride + (srcX0 - unitX)
	if topRows := unitY - srcY0; topRows > 0 {
		if unitRow == 0 || len(topBuf) < cdef.VerticalBorder*width {
			return threading.ErrInvalidBatch
		}
		for row := 0; row < topRows; row++ {
			srcOff := row*width + srcX0
			dstOff := dstOffset + row*cdef.BStride
			copy(input[dstOff:dstOff+copyW], topBuf[srcOff:srcOff+copyW])
		}
	}
	midOffset := dstOffset + (unitY-srcY0)*cdef.BStride
	if leftW := unitX - srcX0; leftW > 0 {
		if leftFromBackup {
			if len(leftBuf) < cdef.HorizontalBorder*unitH {
				return threading.ErrInvalidBatch
			}
			for row := 0; row < unitH; row++ {
				dstOff := midOffset + row*cdef.BStride
				copy(input[dstOff:dstOff+leftW], leftBuf[row*cdef.HorizontalBorder:row*cdef.HorizontalBorder+leftW])
			}
		} else {
			for row := 0; row < unitH; row++ {
				srcOff := (unitY+row)*stride + srcX0
				dstOff := midOffset + row*cdef.BStride
				copy(input[dstOff:dstOff+leftW], pixels[srcOff:srcOff+leftW])
			}
		}
	}
	for row := 0; row < unitH; row++ {
		srcOff := (unitY+row)*stride + unitX
		dstOff := midOffset + (unitX - srcX0) + row*cdef.BStride
		copy(input[dstOff:dstOff+srcX1-unitX], pixels[srcOff:srcOff+srcX1-unitX])
	}
	if botRows := srcY1 - (unitY + unitH); botRows > 0 {
		if bottomBuf != nil {
			if len(bottomBuf) < cdef.VerticalBorder*width {
				return threading.ErrInvalidBatch
			}
			for row := 0; row < botRows; row++ {
				srcOff := row*width + srcX0
				dstOff := dstOffset + (unitY+unitH-srcY0+row)*cdef.BStride
				copy(input[dstOff:dstOff+copyW], bottomBuf[srcOff:srcOff+copyW])
			}
		} else {
			for row := 0; row < botRows; row++ {
				srcOff := (unitY+unitH+row)*stride + srcX0
				dstOff := dstOffset + (unitY+unitH-srcY0+row)*cdef.BStride
				copy(input[dstOff:dstOff+copyW], pixels[srcOff:srcOff+copyW])
			}
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
	return nil
}
