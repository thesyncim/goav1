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

// BindRequest validates and slices caller-owned scratch for CDEF postfiltering.
func (s FrameWorkCDEFPostFilterScratchSize) BindRequest(indexMap FrameWorkCDEFIndexMap, samples [3][]uint16, dst [3][]uint16, directionGrid []cdef.DirectionGrid, varianceGrid []cdef.VarianceGrid, input []uint16, unitDst []uint16) (FrameWorkCDEFPostFilterRequest, error) {
	if len(directionGrid) < s.DirectionGrid || len(varianceGrid) < s.VarianceGrid ||
		len(input) < s.Input || len(unitDst) < s.UnitDst ||
		s.DirectionGrid < 0 || s.VarianceGrid < 0 || s.Input < 0 || s.UnitDst < 0 {
		return FrameWorkCDEFPostFilterRequest{}, frame.ErrShortBuffer
	}
	req := FrameWorkCDEFPostFilterRequest{
		IndexMap:       indexMap,
		DirectionGrid:  directionGrid[:s.DirectionGrid],
		VarianceGrid:   varianceGrid[:s.VarianceGrid],
		InputScratch:   input[:s.Input],
		UnitDstScratch: unitDst[:s.UnitDst],
	}
	for plane := 0; plane < len(req.SampleScratch); plane++ {
		if s.Samples[plane] < 0 || s.Dst[plane] < 0 ||
			len(samples[plane]) < s.Samples[plane] || len(dst[plane]) < s.Dst[plane] {
			return FrameWorkCDEFPostFilterRequest{}, frame.ErrShortBuffer
		}
		req.SampleScratch[plane] = samples[plane][:s.Samples[plane]]
		req.DstScratch[plane] = dst[plane][:s.Dst[plane]]
	}
	return req, nil
}

// Max returns the per-field maximum CDEF scratch size.
func (s FrameWorkCDEFPostFilterScratchSize) Max(other FrameWorkCDEFPostFilterScratchSize) FrameWorkCDEFPostFilterScratchSize {
	result := FrameWorkCDEFPostFilterScratchSize{
		DirectionGrid: maxInt(s.DirectionGrid, other.DirectionGrid),
		VarianceGrid:  maxInt(s.VarianceGrid, other.VarianceGrid),
		Input:         maxInt(s.Input, other.Input),
		UnitDst:       maxInt(s.UnitDst, other.UnitDst),
	}
	for plane := 0; plane < len(result.Samples); plane++ {
		result.Samples[plane] = maxInt(s.Samples[plane], other.Samples[plane])
		result.Dst[plane] = maxInt(s.Dst[plane], other.Dst[plane])
	}
	return result
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
	if err := ctx.validateCDEFPostFilterRequest(req); err != nil {
		return FrameWorkCDEFPostFilterResult{}, err
	}
	cols, rows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
	if err != nil {
		return FrameWorkCDEFPostFilterResult{}, err
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
	skipMap := ctx.LoopFilterMap

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
		planeUnits, planeBlocks, err := frameWorkApplyCDEFPlane(ctx.Event.CDEF, indexMap, skipMap, cols, rows, src, dst, req.InputScratch[:cdef.InputBufferSize], req.UnitDstScratch[:cdef.InputBufferSize], blockStorage[:], &directions, &variances, req.DirectionGrid, req.VarianceGrid, plane, xDec, yDec, coeffShift, chromaFiltering)
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

func (ctx FrameWorkPostFilterContext) validateCDEFPostFilterRequest(req FrameWorkCDEFPostFilterRequest) error {
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterCDEF) {
		return nil
	}
	if ctx.Output == nil {
		return frame.ErrInvalidSlot
	}
	if !frameWorkCDEFHasFiltering(ctx.Event.CDEF, ctx.Output.Format.MonoChrome) {
		return nil
	}
	cols, rows, err := frameWorkCDEFUnitGrid(ctx.Event.FrameSize)
	if err != nil {
		return err
	}
	indexMap := req.IndexMap
	if frameWorkCDEFIndexMapEmpty(indexMap) && ctx.CDEFIndexMap != nil {
		indexMap = *ctx.CDEFIndexMap
	}
	if err := frameWorkValidateCDEFIndexMap(indexMap, cols, rows); err != nil {
		return err
	}
	chromaFiltering := !ctx.Output.Format.MonoChrome && frameWorkCDEFChromaHasFiltering(ctx.Event.CDEF)
	unitCount := cols * rows
	if chromaFiltering && (len(req.DirectionGrid) < unitCount || len(req.VarianceGrid) < unitCount) {
		return frame.ErrShortBuffer
	}
	if len(req.InputScratch) < cdef.InputBufferSize || len(req.UnitDstScratch) < cdef.InputBufferSize {
		return frame.ErrShortBuffer
	}
	coeffShift := int(ctx.Output.Format.BitDepth) - 8
	if coeffShift < 0 || coeffShift > 4 {
		return frame.ErrInvalidFormat
	}
	for plane := 0; plane < 3; plane++ {
		planeFrame, ok := frameWorkCDEFPlane(*ctx.Output, plane)
		processPlane := frameWorkCDEFPlaneHasFiltering(ctx.Event.CDEF, plane)
		if plane == 0 && !processPlane {
			processPlane = chromaFiltering
		}
		if !ok || !processPlane {
			continue
		}
		need, err := frame.SamplePlaneLen(planeFrame, ctx.Output.Layout.BytesPerSample)
		if err != nil {
			return err
		}
		if len(req.SampleScratch[plane]) < need || len(req.DstScratch[plane]) < need {
			return frame.ErrShortBuffer
		}
	}
	return nil
}

func frameWorkCDEFIndexMapEmpty(indexMap FrameWorkCDEFIndexMap) bool {
	return indexMap.Stride == 0 && indexMap.Rows == 0 &&
		len(indexMap.Index) == 0 && len(indexMap.Read) == 0
}

func frameWorkApplyCDEFPlane(params parser.CDEFParams, indexMap FrameWorkCDEFIndexMap, skipMap *FrameWorkLoopFilterMap, cols int, rows int, src frame.SamplePlane, dst frame.SamplePlane, input []uint16, unitDst []uint16, blockStorage []cdef.BlockPosition, directions *cdef.DirectionGrid, variances *cdef.VarianceGrid, directionGrid []cdef.DirectionGrid, varianceGrid []cdef.VarianceGrid, plane int, xDec int, yDec int, coeffShift int, forceLumaDirections bool) (int, int, error) {
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
			blocks := frameWorkCDEFBlockPositionsFiltered(blockStorage, unitW, unitH, blockWidth, blockHeight, skipMap, unitRow, unitCol)
			if len(blocks) == 0 {
				continue
			}
			for i := range input {
				input[i] = cdef.VeryLarge
			}
			if err := frameWorkCopyCDEFInput(input, src, unitX, unitY, unitW, unitH); err != nil {
				return units, blocksTotal, err
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
			if cdefDebugUnit(plane, unitRow, unitCol) {
				cdefDebugLogUnit(plane, unitRow, unitCol, packed, filterParams, *unitDirections, *unitVariances, input, len(blocks))
			}
			if err := cdef.FilterFrameBlocks(unitDst, cdef.BStride, input, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, blocks, unitDirections, unitVariances, filterParams); err != nil {
				return units, blocksTotal, err
			}
			cdefDebugLogUnitDst(plane, unitRow, unitCol, unitDst)
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
	if err := cdef.CopyRect16To16(input[dstOffset:], cdef.BStride, src.Pix[srcOffset:], src.Stride, copyW, copyH); err != nil {
		return err
	}
	// libaom's CDEF bot_linebuf for non-final superblock rows is fed from the
	// frame buffer at rows past the current 64x64 superblock; for chroma planes
	// whose visible height isn't a multiple of the CDEF unit height that read
	// lands in the plane's aligned-but-out-of-crop padding (uv_height vs
	// uv_crop_height), which libaom populates with the last visible row via
	// implicit alignment/decode writes. Our SamplePlane is exactly the visible
	// window, so we synthesize the equivalent extension by replicating the last
	// visible row into the CDEF input bottom border whenever the unit is not the
	// final unit row but the visible plane ends inside the unit's
	// VerticalBorder reach. Without it the kernel's directional taps hit the
	// VeryLarge sentinel even when libaom had real data, producing the V plane
	// row-31 -1 divergence on 66x66 (chroma height 33, CDEF unit height 32).
	//
	// The final superblock unit row keeps the VeryLarge sentinel just like
	// libaom's fill_rect(...CDEF_VERY_LARGE) branch when fbr == nvfb - 1.
	wantSrcY1 := unitY + unitH + cdef.VerticalBorder
	rowExtend := srcY1 < wantSrcY1 && srcY1 > 0 && unitY+unitH < src.Height
	if rowExtend {
		lastRowOffset := dstOffset + (copyH-1)*cdef.BStride
		for row := srcY1; row < wantSrcY1; row++ {
			extOffset := dstOffset + (row-srcY0)*cdef.BStride
			copy(input[extOffset:extOffset+copyW], input[lastRowOffset:lastRowOffset+copyW])
		}
	}
	// Same scenario, but symmetric on the horizontal axis: libaom's central
	// av1_cdef_copy_sb8_16 read for a non-rightmost superblock pulls in
	// hsize + CDEF_HBORDER columns from the dst frame buffer. When the visible
	// plane width ends inside that read (e.g. 33-wide chroma in the 66x66
	// vector at SB(0,0): visible cols 0..32 vs the read of cols 0..39 for
	// hsize=32 + HBORDER=8), libaom reads from the YV12 buffer's
	// aligned-but-out-of-crop padding which holds the last visible column via
	// implicit alignment/decode writes. Our SamplePlane is exactly the visible
	// window, so the right border stays at VeryLarge and the kernel's
	// directional secondary taps at the right edge of the unit diverge —
	// producing the V plane col-31 -1 divergence on 66x66 once the bottom-edge
	// fix already covered the row direction. Y plane SBs whose hsize=64 only
	// reach +2 into the right halo with visible content, so the read never
	// hits the VeryLarge sentinel and luma stays unaffected by this branch.
	//
	// Synthesize the equivalent extension by replicating the last visible
	// column into the CDEF input right border whenever the unit is not the
	// final unit column but the visible plane ends inside the unit's
	// HorizontalBorder reach. The final superblock unit column keeps the
	// VeryLarge sentinel just like libaom's fill_rect(...CDEF_VERY_LARGE)
	// frame_boundary[RIGHT] overlay when fbc == nhfb - 1.
	//
	// Apply column extension after row extension so that the bottom-right
	// corner of the input buffer (which is doubly past-visible for chroma SBs
	// on 66x66) is also populated with the last visible pixel.
	wantSrcX1 := unitX + unitW + cdef.HorizontalBorder
	if srcX1 < wantSrcX1 && srcX1 > 0 && unitX+unitW < src.Width {
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

func frameWorkCDEFBlockPositions(storage []cdef.BlockPosition, unitW int, unitH int, blockW int, blockH int) []cdef.BlockPosition {
	return frameWorkCDEFBlockPositionsFiltered(storage, unitW, unitH, blockW, blockH, nil, 0, 0)
}

// frameWorkCDEFBlockPositionsFiltered enumerates 8x8 CDEF blocks inside the
// current 64x64 unit, skipping those whose luma MI cells all report
// SkipTransform=true. Block indexing follows libaom: by/bx are luma 8x8
// indices reused across planes, so the skip lookup is in luma MI coordinates
// regardless of plane subsampling. When skipMap is nil every block is kept,
// matching the previous behaviour.
func frameWorkCDEFBlockPositionsFiltered(storage []cdef.BlockPosition, unitW int, unitH int, blockW int, blockH int, skipMap *FrameWorkLoopFilterMap, unitRow int, unitCol int) []cdef.BlockPosition {
	cols := (unitW + blockW - 1) / blockW
	rows := (unitH + blockH - 1) / blockH
	out := storage[:0]
	for by := 0; by < rows; by++ {
		for bx := 0; bx < cols; bx++ {
			if frameWorkCDEFBlockAllSkipped(skipMap, unitRow, unitCol, by, bx) {
				continue
			}
			out = append(out, cdef.BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	return out
}

// frameWorkCDEFBlockAllSkipped reports whether every luma MI cell covered by
// the 8x8 luma CDEF block at (by, bx) inside the unit at (unitRow, unitCol)
// has SkipTransform=true. Missing or invalid coverage falls back to false so
// CDEF still runs on those positions (matches the legacy unfiltered behaviour).
func frameWorkCDEFBlockAllSkipped(skipMap *FrameWorkLoopFilterMap, unitRow int, unitCol int, by int, bx int) bool {
	if skipMap == nil {
		return false
	}
	const miPerUnit = 16 // 64-pixel luma unit / 4-pixel MI cell
	const miPerBlock = 2 // 8-pixel block / 4-pixel MI cell
	miColStart := unitCol*miPerUnit + bx*miPerBlock
	miRowStart := unitRow*miPerUnit + by*miPerBlock
	if miColStart < 0 || miRowStart < 0 {
		return false
	}
	if miColStart+miPerBlock > skipMap.Stride || miRowStart+miPerBlock > skipMap.Rows {
		return false
	}
	for dy := 0; dy < miPerBlock; dy++ {
		row := (miRowStart + dy) * skipMap.Stride
		for dx := 0; dx < miPerBlock; dx++ {
			record := skipMap.Records[row+miColStart+dx]
			if !record.Valid {
				return false
			}
			if !record.SkipTransform {
				return false
			}
		}
	}
	return true
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
