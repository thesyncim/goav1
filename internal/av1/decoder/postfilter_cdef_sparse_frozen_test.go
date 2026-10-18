package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// Frozen original complete unit staging walk from 5484d71c. Assembly and
// filter helpers used here retain their original implementation.
func frameWorkApplyCDEFPlaneRowsU8FrozenSparse(params parser.CDEFParams, indexMap FrameWorkCDEFIndexMap, skipMap *FrameWorkLoopFilterMap, cols int, rows int, rowStart int, rowEnd int, dst frame.Plane, byteScratch []uint16, boundary FrameWorkCDEFPostFilterU8BandBoundary, input []uint16, blockStorage []cdef.BlockPosition, directions *cdef.DirectionGrid, variances *cdef.VarianceGrid, directionGrid []cdef.DirectionGrid, varianceGrid []cdef.VarianceGrid, plane int, xDec int, yDec int, forceLumaDirections bool, unitMasks *frameWorkCDEFUnitMaskCache) (uint32, uint32, error) {
	var units uint32
	var blocksTotal uint32
	width := dst.Width
	height := dst.Height
	stride := dst.Stride
	if width <= 0 || height <= 0 || stride < width ||
		(height-1)*stride+width > len(dst.Pix) {
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
	unitSizeX := cdef.BlockSize >> xDec
	unitSizeY := cdef.BlockSize >> yDec
	blockWidth := 8 >> xDec
	blockHeight := 8 >> yDec
	indexStride := int(indexMap.Stride)

	// Pre-filter backups (dav1d cdef_apply_tmpl.c): two ping-pong line
	// buffers of 2 rows x plane width (backup2lines / cdef_line[tf]) carved
	// from the sample scratch arena, and an 8-column x unit-height left
	// backup (backup2x8 / lr_bak) on the stack.
	lineBuf, ok := cdefByteScratchView(byteScratch, 4*width)
	if !ok {
		return 0, 0, threading.ErrInvalidBatch
	}
	if rowStart > 0 {
		top := boundary.Top[plane]
		need := cdef.VerticalBorder * width
		if len(top) < need {
			return 0, 0, frame.ErrShortBuffer
		}
		topSlot := ((rowStart ^ 1) & 1) * cdef.VerticalBorder * width
		copy(lineBuf[topSlot:topSlot+need], top[:need])
	}
	var bottomBoundary []byte
	if rowEnd < rows {
		bottomBoundary = boundary.Bottom[plane]
		if len(bottomBoundary) < cdef.VerticalBorder*width {
			return 0, 0, frame.ErrShortBuffer
		}
	}
	var leftStorage [cdef.HorizontalBorder * cdef.BlockSize]byte

	for unitRow := rowStart; unitRow < rowEnd; unitRow++ {
		unitY := (unitRow * cdef.BlockSize) >> yDec
		// Back up the 2 pre-filter rows the NEXT unit row reads as its top
		// halo before any unit in this row overwrites them.
		if unitRow+1 < rowEnd {
			nextTopY := unitY + unitSizeY
			save := lineBuf[(unitRow&1)*2*width:]
			for r := 0; r < cdef.VerticalBorder; r++ {
				y := nextTopY - cdef.VerticalBorder + r
				if y < 0 || y >= height {
					continue
				}
				copy(save[r*width:r*width+width], dst.Pix[y*stride:y*stride+width])
			}
		}
		topBuf := lineBuf[((unitRow^1)&1)*2*width:]
		bottomBuf := []byte(nil)
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
			var blocks []cdef.BlockPosition
			if unitMasks != nil {
				blocks = frameWorkCDEFBlockPositionsFromMask(blockStorage, unitMasks.mask(unitRow, unitCol), unitW, unitH, blockWidth, blockHeight)
			} else {
				blocks = frameWorkCDEFBlockPositionsFiltered(blockStorage, unitW, unitH, blockWidth, blockHeight, skipMap, unitRow, unitCol)
			}
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
			filterParams, err := cdef.FrameFilterParamsFromStrength(cdefPlane, xDec, yDec, packed, int(params.Damping), 0)
			if err != nil {
				return units, blocksTotal, err
			}
			unitOrigin := unitY*stride + unitX
			if directionOnly {
				// Directions come from the uint8 frame; no tap buffer needed.
				if err := cdef.FilterFrameBlocksU8Trusted(dst.Pix[unitOrigin:], stride, input[:0], 0, blocks, unitDirections, unitVariances, filterParams); err != nil {
					return units, blocksTotal, err
				}
				prevFiltered = false
				continue
			}
			if frameWorkCDEFCanUseSecondaryByteUnit(cdef.U8SecondaryOnlyByteInteriorAvailable(), needPrimaryDirection, filterParams, unitX, unitY, unitW, unitH, width, height, blockWidth, blockHeight, cdefDebugUnit(plane, unitRow, unitCol)) {
				byteInput, byteOK := cdefByteScratchView(input, 2*len(input))
				if byteOK && frameWorkAssembleCDEFSecondaryByteInputU8(byteInput, dst, topBuf, bottomBuf, leftStorage[:], prevFiltered, unitX, unitY, unitW, unitH) == nil {
					// Preserve the previous left halo until the byte filter accepts
					// the unit. Its entry validates the complete read window before
					// writing, so a decline can safely fall through to the wide path.
					oldLeftStorage := leftStorage
					nextUnitX := unitX + unitSizeX
					if nextUnitX < width {
						for r := 0; r < unitH; r++ {
							rowOff := (unitY+r)*stride + nextUnitX - cdef.HorizontalBorder
							copy(leftStorage[r*cdef.HorizontalBorder:(r+1)*cdef.HorizontalBorder], dst.Pix[rowOff:rowOff+cdef.HorizontalBorder])
						}
					}
					if cdef.FilterFrameBlocksU8SecondaryOnlyByteInteriorTrusted(dst.Pix[unitOrigin:], stride, byteInput, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, unitW, unitH, blocks, filterParams) {
						prevFiltered = true
						units++
						blocksTotal += uint32(len(blocks))
						continue
					}
					leftStorage = oldLeftStorage
				}
			}
			if err := frameWorkAssembleCDEFInputU8(input, dst, topBuf, bottomBuf, leftStorage[:], prevFiltered, unitRow, unitX, unitY, unitW, unitH); err != nil {
				return units, blocksTotal, err
			}
			// Back up the pre-filter columns the NEXT unit reads as its left
			// halo (dav1d backup2x8 / lr_bak). This must happen after this
			// unit's tap assembly consumed the PREVIOUS backup (dav1d
			// ping-pongs lr_bak[bit] for the same reason) and before the
			// kernels overwrite the frame.
			nextUnitX := unitX + unitSizeX
			if nextUnitX < width {
				for r := 0; r < unitH; r++ {
					rowOff := (unitY+r)*stride + nextUnitX - cdef.HorizontalBorder
					copy(leftStorage[r*cdef.HorizontalBorder:(r+1)*cdef.HorizontalBorder], dst.Pix[rowOff:rowOff+cdef.HorizontalBorder])
				}
			}
			if cdefDebugUnit(plane, unitRow, unitCol) {
				cdefDebugLogUnit(plane, unitRow, unitCol, packed, filterParams, *unitDirections, *unitVariances, input, len(blocks))
			}
			var filterErr error
			if needPrimaryDirection {
				filterErr = cdef.FilterFrameBlocksU8Trusted(dst.Pix[unitOrigin:], stride, input, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, blocks, unitDirections, unitVariances, filterParams)
			} else {
				filterErr = cdef.FilterFrameBlocksU8TrustedNoPrimaryDirection(dst.Pix[unitOrigin:], stride, input, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, blocks, unitDirections, unitVariances, filterParams)
			}
			if filterErr != nil {
				return units, blocksTotal, filterErr
			}
			prevFiltered = true
			units++
			blocksTotal += uint32(len(blocks))
		}
	}
	return units, blocksTotal, nil
}
