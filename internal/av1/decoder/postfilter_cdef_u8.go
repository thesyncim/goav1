package decoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// 8-bit CDEF frame walk. dav1d's 8bpc CDEF (src/cdef_apply_tmpl.c
// dav1d_cdef_brow) filters uint8 pixels IN PLACE on the frame: the walk is
// strictly top-down/left-to-right, the 2-row halo above each superblock row
// comes from a pre-filter line backup saved before that row was overwritten
// (backup2lines + the cdef_line ping-pong), the left halo into an
// already-filtered neighbour comes from a pre-filter column backup taken
// before overwrite (backup2x8 / lr_bak), and the right/bottom halos read the
// not-yet-filtered frame directly. This file ports that walk onto goav1's
// per-64x64-unit staging shape and the uint8 kernel family in
// internal/av1/cdef:
//
//   - the per-unit uint16 tap buffer (libaom cdef_prepare_fb geometry,
//     0x4000 VeryLarge sentinels, dav1d padding() role) is assembled straight
//     from uint8 sources — the frame plane, the 2-row line backup, and the
//     8-column left backup — with one strided widen per region;
//   - sentinels are written only where the assembly leaves no real pixels
//     (frame edges), exactly the final state the uint16 path reaches by
//     pre-filling the whole ring and copying over it;
//   - directions are searched on the uint8 frame directly (dav1d
//     cdef_find_dir_8bpc), so direction-only luma units skip the tap-buffer
//     assembly entirely;
//   - the kernels write filtered blocks straight into the frame plane.
//
// This removes the whole-plane u8->u16 snapshot widen, the per-unit
// uint16->uint16 copy, most sentinel fills, the uint16 unit dst scratch, and
// the store-back narrowing pass for 8-bit frames. Unit-row-banded callers use
// the same walk with immutable two-row top/bottom boundary backups for every
// cross-band halo and private line/input scratch for every worker. 10/12-bit
// frames keep the uint16 path unchanged.

// cdefByteScratchView reinterprets a segment of the uint16 sample scratch
// arena as byte storage for the pre-filter line backups. The 8-bit walk needs
// only 4 plane-width rows, far below the arena's plane-sized allocation; the
// arena is otherwise unused on the u8 path (no snapshot).
func cdefByteScratchView(seg []uint16, n int) ([]byte, bool) {
	if n <= 0 || len(seg) == 0 || n > 2*len(seg) {
		return nil, false
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(seg))), n), true
}

// frameWorkApplyCDEFPlaneRowsU8 is the 8-bit in-place counterpart of
// frameWorkApplyCDEFPlaneRows. dst is the MI-aligned plane view
// (frameWorkCDEFAlignedPlane) of an 8-bit frame (stride in bytes, one byte
// per sample); byteScratch is the caller's private sample scratch for the
// plane, reused as the line-backup arena. Whole-frame callers pass
// rowStart=0,rowEnd=rows and an empty boundary; banded callers pass immutable
// boundary strips for any cross-band top/bottom halos.
func frameWorkApplyCDEFPlaneRowsU8(params parser.CDEFParams, indexMap FrameWorkCDEFIndexMap, skipMap *FrameWorkLoopFilterMap, skipLevels *threading.FrameWorkLoopFilterMasks, cols int, rows int, rowStart int, rowEnd int, dst frame.Plane, byteScratch []uint16, boundary FrameWorkCDEFPostFilterU8BandBoundary, input []uint16, blockStorage []cdef.BlockPosition, directions *cdef.DirectionGrid, variances *cdef.VarianceGrid, directionGrid []cdef.DirectionGrid, varianceGrid []cdef.VarianceGrid, plane int, xDec int, yDec int, forceLumaDirections bool) (uint32, uint32, error) {
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
			directionOnly := plane == 0 && forceLumaDirections && packed == 0 && params.UVStrength[index] != 0
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
			unitIndex := unitRow*cols + unitCol
			unitDirections := directions
			unitVariances := variances
			if unitIndex < len(directionGrid) && unitIndex < len(varianceGrid) {
				unitDirections = &directionGrid[unitIndex]
				unitVariances = &varianceGrid[unitIndex]
			}
			var blocks []cdef.BlockPosition
			if plane == 0 {
				frameWorkCDEFMarkDirectionsUnavailable(unitDirections, unitW, unitH, blockWidth, blockHeight)
				blocks = frameWorkCDEFBlockPositionsFilteredSources(blockStorage, unitW, unitH, blockWidth, blockHeight, skipMap, skipLevels, unitRow, unitCol)
			} else {
				blocks = frameWorkCDEFBlockPositionsFromDirections(blockStorage, unitW, unitH, blockWidth, blockHeight, unitDirections)
			}
			if len(blocks) == 0 {
				prevFiltered = false
				continue
			}
			skipDirectionSearch := plane == 0 && packed>>2 == 0 && (!forceLumaDirections || params.UVStrength[index]>>2 == 0)
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
				// A needed direction comes from the uint8 frame; when both
				// primary strengths are zero, canonical grid zeroes suffice.
				// Neither shape needs a tap buffer.
				var err error
				if skipDirectionSearch {
					err = cdef.FilterFrameBlocksU8TrustedNoDirection(dst.Pix[unitOrigin:], stride, input[:0], 0, blocks, unitDirections, unitVariances, filterParams)
				} else {
					err = cdef.FilterFrameBlocksU8Trusted(dst.Pix[unitOrigin:], stride, input[:0], 0, blocks, unitDirections, unitVariances, filterParams)
				}
				if err != nil {
					return units, blocksTotal, err
				}
				prevFiltered = false
				continue
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
			if skipDirectionSearch {
				filterErr = cdef.FilterFrameBlocksU8TrustedNoDirection(dst.Pix[unitOrigin:], stride, input, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, blocks, unitDirections, unitVariances, filterParams)
			} else {
				filterErr = cdef.FilterFrameBlocksU8Trusted(dst.Pix[unitOrigin:], stride, input, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, blocks, unitDirections, unitVariances, filterParams)
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

// frameWorkAssembleCDEFInputU8 builds the unit's padded uint16 tap buffer
// from uint8 sources, reproducing bit-for-bit the state the uint16 path
// reaches with frameWorkFillCDEFInputSentinels + frameWorkCopyCDEFInput (see
// those functions for the libaom cdef_prepare_fb grounding of the read
// window, the row/column extensions, and the VeryLarge frame-boundary fill):
//
//   - rows above the unit come from topBuf (the pre-filter line backup;
//     row 0 of the frame keeps the sentinel exactly like the snapshot's
//     srcY0 clamp);
//   - the left halo comes from leftBuf when the left neighbour was filtered
//     this pass, else from the frame (both hold the pre-filter bytes);
//   - the unit interior, right halo, and bottom halo read the frame, which
//     is not yet filtered there;
//   - sentinels are written only to the ring regions the copies and
//     extensions leave untouched.
func frameWorkAssembleCDEFInputU8(input []uint16, dst frame.Plane, topBuf []byte, bottomBuf []byte, leftBuf []byte, leftFromBackup bool, unitRow int, unitX int, unitY int, unitW int, unitH int) error {
	width := dst.Width
	height := dst.Height
	stride := dst.Stride
	fillW := unitW + 2*cdef.HorizontalBorder
	fillH := unitH + 2*cdef.VerticalBorder
	if unitW <= 0 || unitH <= 0 || fillW > cdef.BStride ||
		!cdefInputRectFits(len(input), fillW, fillH) {
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

	// Effective coverage after the copies and the row/column extensions; the
	// remaining ring stays VeryLarge (libaom's frame_boundary fill).
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
	// Rows above the unit: the pre-filter line backup holds plane rows
	// [unitY-VerticalBorder, unitY). At unit row 0 there are none (srcY0
	// clamps to unitY) and the sentinel rows above remain.
	if topRows := unitY - srcY0; topRows > 0 {
		if unitRow == 0 {
			return threading.ErrInvalidBatch
		}
		// topBuf row r holds plane row unitY-VerticalBorder+r; topRows > 0
		// implies unitY >= VerticalBorder so srcY0 == unitY-VerticalBorder
		// and the read starts at topBuf row 0, column srcX0.
		frame.LoadSampleRows8Trusted(input[dstOffset:], cdef.BStride, topBuf[srcX0:], width, copyW, topRows)
	}
	midOffset := dstOffset + (unitY-srcY0)*cdef.BStride
	// Left halo columns [srcX0, unitX).
	if leftW := unitX - srcX0; leftW > 0 {
		if leftFromBackup {
			// leftBuf rows are HorizontalBorder wide and hold plane columns
			// [unitX-HorizontalBorder, unitX); the halo is always the full
			// HorizontalBorder here (unitX >= unit size >= border).
			frame.LoadSampleRows8Trusted(input[midOffset:], cdef.BStride, leftBuf, cdef.HorizontalBorder, leftW, unitH)
		} else {
			frame.LoadSampleRows8Trusted(input[midOffset:], cdef.BStride, dst.Pix[unitY*stride+srcX0:], stride, leftW, unitH)
		}
	}
	// Unit interior + right halo columns [unitX, srcX1).
	frame.LoadSampleRows8Trusted(input[midOffset+(unitX-srcX0):], cdef.BStride, dst.Pix[unitY*stride+unitX:], stride, srcX1-unitX, unitH)
	// Rows below the unit (not yet filtered).
	if botRows := srcY1 - (unitY + unitH); botRows > 0 {
		botOffset := dstOffset + (unitY+unitH-srcY0)*cdef.BStride
		if bottomBuf != nil {
			frame.LoadSampleRows8Trusted(input[botOffset:], cdef.BStride, bottomBuf[srcX0:], width, copyW, botRows)
		} else {
			frame.LoadSampleRows8Trusted(input[botOffset:], cdef.BStride, dst.Pix[(unitY+unitH)*stride+srcX0:], stride, copyW, botRows)
		}
	}

	// Row/column extensions: identical to frameWorkCopyCDEFInput's replication
	// of the last visible row/column into the halo (libaom's
	// aligned-but-out-of-crop padding reads).
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
			extStart := rowOffset + copyW
			extEnd := rowOffset + (wantSrcX1 - srcX0)
			for i := extStart; i < extEnd; i++ {
				input[i] = last
			}
		}
	}
	return nil
}
