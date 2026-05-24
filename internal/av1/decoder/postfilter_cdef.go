package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// FrameWorkCDEFIndexMap is the decoded frame-level CDEF unit index map.
type FrameWorkCDEFIndexMap = threading.FrameWorkCDEFIndexMap

// FrameWorkCDEFPostFilterScratchSize reports caller-owned scratch needed by
// ApplyCDEFPostFilter.
type FrameWorkCDEFPostFilterScratchSize struct {
	Samples       [3]int
	Dst           [3]int
	DirectionGrid int
	VarianceGrid  int
	Input         int
	UnitDst       int
}

// FrameWorkCDEFPostFilterRequest carries decoded CDEF state and caller-owned
// scratch for ApplyCDEFPostFilter.
type FrameWorkCDEFPostFilterRequest struct {
	IndexMap FrameWorkCDEFIndexMap

	SampleScratch  [3][]uint16
	DstScratch     [3][]uint16
	DirectionGrid  []cdef.DirectionGrid
	VarianceGrid   []cdef.VarianceGrid
	InputScratch   []uint16
	UnitDstScratch []uint16
}

// FrameWorkCDEFPostFilterResult summarizes CDEF frame filtering work.
type FrameWorkCDEFPostFilterResult struct {
	Units  int
	Planes int
	Blocks int
}

// CDEFPostFilterScratchLen reports scratch lengths needed to apply CDEF to
// ctx.Output.
func (ctx FrameWorkPostFilterContext) CDEFPostFilterScratchLen() (FrameWorkCDEFPostFilterScratchSize, error) {
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterCDEF) {
		return FrameWorkCDEFPostFilterScratchSize{}, nil
	}
	if ctx.Output == nil {
		return FrameWorkCDEFPostFilterScratchSize{}, frame.ErrInvalidSlot
	}
	var size FrameWorkCDEFPostFilterScratchSize
	for plane := 0; plane < 3; plane++ {
		planeFrame, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		if !ok {
			continue
		}
		need, err := frame.SamplePlaneLen(planeFrame, ctx.Output.Layout.BytesPerSample)
		if err != nil {
			return FrameWorkCDEFPostFilterScratchSize{}, err
		}
		size.Samples[plane] = need
		size.Dst[plane] = need
	}
	if !ctx.Output.Format.MonoChrome && frameWorkCDEFChromaHasFiltering(ctx.Event.CDEF) {
		cols, rows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
		if err != nil {
			return FrameWorkCDEFPostFilterScratchSize{}, err
		}
		size.DirectionGrid = cols * rows
		size.VarianceGrid = cols * rows
	}
	size.Input = cdef.InputBufferSize
	size.UnitDst = cdef.InputBufferSize
	return size, nil
}

// ApplyCDEFPostFilter applies CDEF to ctx.Output. Loop filter must already be
// inactive or marked complete with WithCompletedPostFilters.
func (ctx FrameWorkPostFilterContext) ApplyCDEFPostFilter(req FrameWorkCDEFPostFilterRequest) (FrameWorkCDEFPostFilterResult, error) {
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
	if err := frameWorkValidateCDEFIndexMap(req.IndexMap, cols, rows); err != nil {
		return FrameWorkCDEFPostFilterResult{}, err
	}
	chromaFiltering := !ctx.Output.Format.MonoChrome && frameWorkCDEFChromaHasFiltering(ctx.Event.CDEF)
	unitCount := cols * rows
	if chromaFiltering && (len(req.DirectionGrid) < unitCount || len(req.VarianceGrid) < unitCount) {
		return FrameWorkCDEFPostFilterResult{}, frame.ErrShortBuffer
	}
	if len(req.InputScratch) < cdef.InputBufferSize || len(req.UnitDstScratch) < cdef.InputBufferSize {
		return FrameWorkCDEFPostFilterResult{}, frame.ErrShortBuffer
	}
	coeffShift := int(ctx.Output.Format.BitDepth) - 8
	if coeffShift < 0 || coeffShift > 4 {
		return FrameWorkCDEFPostFilterResult{}, frame.ErrInvalidFormat
	}

	var result FrameWorkCDEFPostFilterResult
	var directions cdef.DirectionGrid
	var variances cdef.VarianceGrid
	var blockStorage [cdef.NBlocks * cdef.NBlocks]cdef.BlockPosition
	for plane := 0; plane < 3; plane++ {
		planeFrame, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		processPlane := frameWorkCDEFPlaneHasFiltering(ctx.Event.CDEF, plane)
		if plane == 0 && !processPlane {
			processPlane = chromaFiltering
		}
		if !ok || !processPlane {
			continue
		}
		src, err := frame.LoadSamplePlane(req.SampleScratch[plane], planeFrame, ctx.Output.Layout.BytesPerSample)
		if err != nil {
			return FrameWorkCDEFPostFilterResult{}, err
		}
		dst, err := frame.LoadSamplePlane(req.DstScratch[plane], planeFrame, ctx.Output.Layout.BytesPerSample)
		if err != nil {
			return FrameWorkCDEFPostFilterResult{}, err
		}
		xDec, yDec := frameWorkCDEFPlaneDecimation(ctx.Output.Format, plane)
		planeUnits, planeBlocks, err := frameWorkApplyCDEFPlane(ctx.Event.CDEF, req.IndexMap, cols, rows, src, dst, req.InputScratch[:cdef.InputBufferSize], req.UnitDstScratch[:cdef.InputBufferSize], blockStorage[:], &directions, &variances, req.DirectionGrid, req.VarianceGrid, plane, xDec, yDec, coeffShift, chromaFiltering)
		if err != nil {
			return FrameWorkCDEFPostFilterResult{}, err
		}
		if err := frame.StoreSamplePlane(planeFrame, ctx.Output.Layout.BytesPerSample, dst); err != nil {
			return FrameWorkCDEFPostFilterResult{}, err
		}
		result.Units += planeUnits
		result.Blocks += planeBlocks
		result.Planes++
	}
	return result, nil
}

func frameWorkApplyCDEFPlane(params parser.CDEFParams, indexMap FrameWorkCDEFIndexMap, cols int, rows int, src frame.SamplePlane, dst frame.SamplePlane, input []uint16, unitDst []uint16, blockStorage []cdef.BlockPosition, directions *cdef.DirectionGrid, variances *cdef.VarianceGrid, directionGrid []cdef.DirectionGrid, varianceGrid []cdef.VarianceGrid, plane int, xDec int, yDec int, coeffShift int, forceLumaDirections bool) (int, int, error) {
	units := 0
	blocksTotal := 0
	unitSizeX := cdef.BlockSize >> xDec
	unitSizeY := cdef.BlockSize >> yDec
	blockWidth := 8 >> xDec
	blockHeight := 8 >> yDec
	for unitRow := 0; unitRow < rows; unitRow++ {
		for unitCol := 0; unitCol < cols; unitCol++ {
			mapOffset := unitRow*indexMap.Stride + unitCol
			if !indexMap.Read[mapOffset] {
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
				continue
			}
			unitX := (unitCol * cdef.BlockSize) >> xDec
			unitY := (unitRow * cdef.BlockSize) >> yDec
			unitW := minInt(unitSizeX, src.Width-unitX)
			unitH := minInt(unitSizeY, src.Height-unitY)
			if unitW <= 0 || unitH <= 0 {
				continue
			}
			for i := range input {
				input[i] = cdef.VeryLarge
			}
			if err := frameWorkCopyCDEFInput(input, src, unitX, unitY, unitW, unitH); err != nil {
				return units, blocksTotal, err
			}
			blocks := frameWorkCDEFBlockPositions(blockStorage, unitW, unitH, blockWidth, blockHeight)
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
			if err := cdef.FilterFrameBlocks(unitDst, cdef.BStride, input, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, blocks, unitDirections, unitVariances, filterParams); err != nil {
				return units, blocksTotal, err
			}
			if directionOnly {
				continue
			}
			dstOffset := unitY*dst.Stride + unitX
			if err := cdef.CopyRect16To16(dst.Pix[dstOffset:], dst.Stride, unitDst, cdef.BStride, unitW, unitH); err != nil {
				return units, blocksTotal, err
			}
			units++
			blocksTotal += len(blocks)
		}
	}
	return units, blocksTotal, nil
}

func frameWorkCopyCDEFInput(input []uint16, src frame.SamplePlane, unitX int, unitY int, unitW int, unitH int) error {
	srcX0 := maxInt(unitX-cdef.HorizontalBorder, 0)
	srcY0 := maxInt(unitY-cdef.VerticalBorder, 0)
	srcX1 := minInt(unitX+unitW+cdef.HorizontalBorder, src.Width)
	srcY1 := minInt(unitY+unitH+cdef.VerticalBorder, src.Height)
	copyW := srcX1 - srcX0
	copyH := srcY1 - srcY0
	if copyW <= 0 || copyH <= 0 {
		return nil
	}
	dstOffset := cdef.VerticalBorder*cdef.BStride + cdef.HorizontalBorder +
		(srcY0-unitY)*cdef.BStride + (srcX0 - unitX)
	srcOffset := srcY0*src.Stride + srcX0
	return cdef.CopyRect16To16(input[dstOffset:], cdef.BStride, src.Pix[srcOffset:], src.Stride, copyW, copyH)
}

func frameWorkCDEFBlockPositions(storage []cdef.BlockPosition, unitW int, unitH int, blockW int, blockH int) []cdef.BlockPosition {
	cols := (unitW + blockW - 1) / blockW
	rows := (unitH + blockH - 1) / blockH
	out := storage[:0]
	for by := 0; by < rows; by++ {
		for bx := 0; bx < cols; bx++ {
			out = append(out, cdef.BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	return out
}

func frameWorkCDEFUnitGrid(size parser.FrameSize) (int, int, error) {
	if size.CodedWidth == 0 || size.Height == 0 {
		return 0, 0, threading.ErrInvalidBatch
	}
	cols := int((size.CodedWidth + cdef.BlockSize - 1) / cdef.BlockSize)
	rows := int((size.Height + cdef.BlockSize - 1) / cdef.BlockSize)
	if cols <= 0 || rows <= 0 {
		return 0, 0, threading.ErrInvalidBatch
	}
	return cols, rows, nil
}

func frameWorkValidateCDEFIndexMap(indexMap FrameWorkCDEFIndexMap, cols int, rows int) error {
	if indexMap.Stride < cols || indexMap.Rows < rows || indexMap.Stride <= 0 || indexMap.Rows <= 0 {
		return threading.ErrInvalidBatch
	}
	if indexMap.Stride > int(^uint(0)>>1)/indexMap.Rows {
		return threading.ErrInvalidBatch
	}
	length := indexMap.Stride * indexMap.Rows
	if len(indexMap.Index) < length || len(indexMap.Read) < length {
		return threading.ErrInvalidBatch
	}
	return nil
}

func frameWorkCDEFHasFiltering(params parser.CDEFParams, mono bool) bool {
	limit := int(params.StrengthCount)
	if limit > parser.MaxCDEFStrengths {
		limit = parser.MaxCDEFStrengths
	}
	for i := 0; i < limit; i++ {
		if params.YStrength[i] != 0 || (!mono && params.UVStrength[i] != 0) {
			return true
		}
	}
	return false
}

func frameWorkCDEFChromaHasFiltering(params parser.CDEFParams) bool {
	limit := int(params.StrengthCount)
	if limit > parser.MaxCDEFStrengths {
		limit = parser.MaxCDEFStrengths
	}
	for i := 0; i < limit; i++ {
		if params.UVStrength[i] != 0 {
			return true
		}
	}
	return false
}

func frameWorkCDEFPlaneHasFiltering(params parser.CDEFParams, plane int) bool {
	limit := int(params.StrengthCount)
	if limit > parser.MaxCDEFStrengths {
		limit = parser.MaxCDEFStrengths
	}
	for i := 0; i < limit; i++ {
		if plane == 0 && params.YStrength[i] != 0 {
			return true
		}
		if plane != 0 && params.UVStrength[i] != 0 {
			return true
		}
	}
	return false
}

func frameWorkCDEFPlane(output frame.Frame, plane int) (frame.Plane, bool) {
	switch plane {
	case 0:
		return output.Y, true
	case 1:
		return output.U, !output.Format.MonoChrome
	case 2:
		return output.V, !output.Format.MonoChrome
	default:
		return frame.Plane{}, false
	}
}

func frameWorkCDEFPlaneDecimation(format frame.Format, plane int) (int, int) {
	if plane == 0 {
		return 0, 0
	}
	xDec := 0
	yDec := 0
	if format.SubsamplingX {
		xDec = 1
	}
	if format.SubsamplingY {
		yDec = 1
	}
	return xDec, yDec
}

func minInt(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a int, b int) int {
	if a > b {
		return a
	}
	return b
}
