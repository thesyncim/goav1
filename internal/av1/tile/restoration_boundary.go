package tile

import (
	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

const (
	restorationBorder    = 3
	restorationCtxVert   = 2
	restorationExtraHorz = 4
)

// RestorationStripeBoundaries contains the saved two-row context above and
// below each restoration processing stripe. The boundary buffers include
// restorationExtraHorz extended samples on both horizontal sides.
type RestorationStripeBoundaries struct {
	Above  []uint16
	Below  []uint16
	Stride int
}

// RestorationFrameBoundaryPlane binds one plane's source and boundary buffers
// for libaom's frame-level loop-restoration boundary save pass.
type RestorationFrameBoundaryPlane struct {
	Grid       RestorationPlaneGrid
	Src        []uint16
	SrcStride  int
	SrcOrigin  int
	Boundaries RestorationStripeBoundaries
}

// RestorationStripeBoundaryScratchSize reports caller-owned temporary storage
// required while overwriting a processing stripe's top and bottom borders.
type RestorationStripeBoundaryScratchSize struct {
	Above int
	Below int
}

// RestorationStripeBoundaryScratch is caller-owned temporary storage used to
// save rows overwritten by SetupRestorationStripeBoundary.
type RestorationStripeBoundaryScratch struct {
	Above []uint16
	Below []uint16
}

// RestorationStripeBoundaryBufferSize reports the caller-owned boundary buffer
// layout for one whole-frame restoration plane.
type RestorationStripeBoundaryBufferSize struct {
	Stride int
	Rows   int
	Len    int
}

func RestorationStripeBoundaryBufferLen(grid RestorationPlaneGrid) (RestorationStripeBoundaryBufferSize, error) {
	if !grid.validGeometry() {
		return RestorationStripeBoundaryBufferSize{}, ErrInvalidPlan
	}
	maxInt := uint64(^uint(0) >> 1)
	if uint64(grid.PlaneWidth) > maxInt || uint64(grid.PlaneHeight) > maxInt {
		return RestorationStripeBoundaryBufferSize{}, ErrInvalidPlan
	}
	stripeHeight := int(av1restoration.ProcUnitSize >> boolToShift(grid.SubsamplingY))
	stripeOff := int(restorationUnitOffset >> boolToShift(grid.SubsamplingY))
	if stripeHeight <= 0 {
		return RestorationStripeBoundaryBufferSize{}, ErrInvalidPlan
	}
	planeWidth := int(grid.PlaneWidth)
	planeHeight := int(grid.PlaneHeight)
	stride, ok := checkedAddInt(planeWidth, 2*restorationExtraHorz)
	if !ok {
		return RestorationStripeBoundaryBufferSize{}, ErrInvalidPlan
	}
	heightWithOffset, ok := checkedAddInt(planeHeight, stripeOff)
	if !ok {
		return RestorationStripeBoundaryBufferSize{}, ErrInvalidPlan
	}
	stripes := ceilDivInt(heightWithOffset, stripeHeight)
	rows, ok := checkedMulInt(stripes, restorationCtxVert)
	if !ok {
		return RestorationStripeBoundaryBufferSize{}, ErrInvalidPlan
	}
	n, ok := checkedMulInt(rows, stride)
	if !ok {
		return RestorationStripeBoundaryBufferSize{}, ErrInvalidPlan
	}
	return RestorationStripeBoundaryBufferSize{Stride: stride, Rows: rows, Len: n}, nil
}

func RestorationStripeBoundaryScratchLen(stripe RestorationProcessingStripe, optimized bool) (RestorationStripeBoundaryScratchSize, error) {
	lineWidth, err := restorationStripeBoundaryLineWidth(stripe)
	if err != nil {
		return RestorationStripeBoundaryScratchSize{}, err
	}
	rows := restorationBorder
	if optimized {
		rows = 1
	}
	var size RestorationStripeBoundaryScratchSize
	if stripe.CopyAbove {
		var ok bool
		size.Above, ok = checkedMulInt(lineWidth, rows)
		if !ok {
			return RestorationStripeBoundaryScratchSize{}, ErrInvalidPlan
		}
	}
	if stripe.CopyBelow {
		var ok bool
		size.Below, ok = checkedMulInt(lineWidth, rows)
		if !ok {
			return RestorationStripeBoundaryScratchSize{}, ErrInvalidPlan
		}
	}
	return size, nil
}

// ExtendRestorationFrame ports av1_extend_frame for uint16 sample planes.
// origin identifies plane coordinate (0,0); the caller-owned buffer must include
// borderHorz columns and borderVert rows around the visible rectangle.
func ExtendRestorationFrame(data []uint16, stride int, origin int, width int, height int, borderHorz int, borderVert int) error {
	if stride <= 0 || origin < 0 || width <= 0 || height <= 0 || borderHorz < 0 || borderVert < 0 {
		return ErrInvalidPlan
	}
	lineWidth, ok := checkedMulInt(borderHorz, 2)
	if !ok {
		return ErrInvalidPlan
	}
	lineWidth, ok = checkedAddInt(lineWidth, width)
	if !ok {
		return ErrInvalidPlan
	}

	for y := 0; y < height; y++ {
		line, ok := restorationFrameLine(data, stride, origin, -borderHorz, y, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		first := line[borderHorz]
		last := line[borderHorz+width-1]
		for x := 0; x < borderHorz; x++ {
			line[x] = first
			line[borderHorz+width+x] = last
		}
	}

	top, ok := restorationFrameLine(data, stride, origin, -borderHorz, 0, lineWidth)
	if !ok {
		return ErrInvalidPlan
	}
	for y := -borderVert; y < 0; y++ {
		line, ok := restorationFrameLine(data, stride, origin, -borderHorz, y, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		copy(line, top)
	}

	bottom, ok := restorationFrameLine(data, stride, origin, -borderHorz, height-1, lineWidth)
	if !ok {
		return ErrInvalidPlan
	}
	for y := height; y < height+borderVert; y++ {
		line, ok := restorationFrameLine(data, stride, origin, -borderHorz, y, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		copy(line, bottom)
	}
	return nil
}

// SaveRestorationBoundaryLines ports save_tile_row_boundary_lines for an
// already-upscaled whole-frame restoration plane.
func SaveRestorationBoundaryLines(grid RestorationPlaneGrid, src []uint16, srcStride int, srcOrigin int, boundaries RestorationStripeBoundaries, afterCDEF bool) error {
	if !grid.validGeometry() || srcStride <= 0 || srcOrigin < 0 {
		return ErrInvalidPlan
	}
	if err := validateRestorationStripeBoundaries(grid, boundaries); err != nil {
		return err
	}
	planeHeight := int(grid.PlaneHeight)
	stripeHeight := int(av1restoration.ProcUnitSize >> boolToShift(grid.SubsamplingY))
	stripeOff := int(restorationUnitOffset >> boolToShift(grid.SubsamplingY))
	if stripeHeight <= 0 {
		return ErrInvalidPlan
	}
	for tileStripe := 0; ; tileStripe++ {
		relY0 := maxInt(0, tileStripe*stripeHeight-stripeOff)
		if relY0 >= planeHeight {
			break
		}
		relY1 := minInt((tileStripe+1)*stripeHeight-stripeOff, planeHeight)
		useDeblockAbove := tileStripe > 0
		useDeblockBelow := relY1 < planeHeight

		switch {
		case !afterCDEF:
			if useDeblockAbove {
				if err := saveRestorationDeblockBoundaryLines(grid, src, srcStride, srcOrigin, relY0-restorationCtxVert, tileStripe, true, boundaries); err != nil {
					return err
				}
			}
			if useDeblockBelow {
				if err := saveRestorationDeblockBoundaryLines(grid, src, srcStride, srcOrigin, relY1, tileStripe, false, boundaries); err != nil {
					return err
				}
			}
		default:
			if !useDeblockAbove {
				if err := saveRestorationCDEFBoundaryLines(grid, src, srcStride, srcOrigin, relY0, tileStripe, true, boundaries); err != nil {
					return err
				}
			}
			if !useDeblockBelow {
				if err := saveRestorationCDEFBoundaryLines(grid, src, srcStride, srcOrigin, relY1-1, tileStripe, false, boundaries); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// SaveRestorationFrameBoundaryLines ports av1_loop_restoration_save_boundary_lines
// for a one-plane monochrome frame or a three-plane color frame.
func SaveRestorationFrameBoundaryLines(planes []RestorationFrameBoundaryPlane, afterCDEF bool) error {
	if len(planes) != 1 && len(planes) != 3 {
		return ErrInvalidPlan
	}
	for i := range planes {
		if planes[i].Grid.Plane != uint8(i) {
			return ErrInvalidPlan
		}
		if planes[i].Grid.Type == parser.RestorationNone {
			continue
		}
		if err := SaveRestorationBoundaryLines(planes[i].Grid, planes[i].Src, planes[i].SrcStride, planes[i].SrcOrigin, planes[i].Boundaries, afterCDEF); err != nil {
			return err
		}
	}
	return nil
}

// SetupRestorationStripeBoundary ports setup_processing_stripe_boundary for
// sample slices. unitRect is the enclosing restoration-unit rectangle; dataOrigin
// identifies plane coordinate (0,0) in data.
func SetupRestorationStripeBoundary(unitRect RestorationUnitRect, stripe RestorationProcessingStripe, boundaries RestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, scratch RestorationStripeBoundaryScratch, optimized bool) error {
	if err := validateRestorationStripeBoundaryInputs(unitRect, stripe, dataStride, dataOrigin); err != nil {
		return err
	}
	lineWidth, err := restorationStripeBoundaryLineWidth(stripe)
	if err != nil {
		return err
	}
	if optimized {
		return setupRestorationStripeBoundaryOptimized(unitRect, stripe, data, dataStride, dataOrigin, scratch, lineWidth)
	}
	return setupRestorationStripeBoundaryFromSaved(stripe, boundaries, data, dataStride, dataOrigin, scratch, lineWidth)
}

// RestoreRestorationStripeBoundary restores rows saved by
// SetupRestorationStripeBoundary.
func RestoreRestorationStripeBoundary(unitRect RestorationUnitRect, stripe RestorationProcessingStripe, data []uint16, dataStride int, dataOrigin int, scratch RestorationStripeBoundaryScratch, optimized bool) error {
	if err := validateRestorationStripeBoundaryInputs(unitRect, stripe, dataStride, dataOrigin); err != nil {
		return err
	}
	lineWidth, err := restorationStripeBoundaryLineWidth(stripe)
	if err != nil {
		return err
	}
	if optimized {
		return restoreRestorationStripeBoundaryOptimized(unitRect, stripe, data, dataStride, dataOrigin, scratch, lineWidth)
	}
	return restoreRestorationStripeBoundarySaved(unitRect, stripe, data, dataStride, dataOrigin, scratch, lineWidth)
}

func setupRestorationStripeBoundaryFromSaved(stripe RestorationProcessingStripe, boundaries RestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, scratch RestorationStripeBoundaryScratch, lineWidth int) error {
	rsbRow := int(stripe.TileStripe) * restorationCtxVert
	if stripe.CopyAbove {
		if len(scratch.Above) < lineWidth*restorationBorder {
			return ErrInvalidPlan
		}
		for i := -restorationBorder; i < 0; i++ {
			dst, ok := restorationStripeDataLine(data, dataStride, dataOrigin, stripe, i, lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			bufRow := rsbRow + maxInt(i+restorationCtxVert, 0)
			src, ok := restorationBoundaryLine(boundaries.Above, boundaries.Stride, bufRow, int(stripe.Rect.X0), lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			save := scratch.Above[(i+restorationBorder)*lineWidth : (i+restorationBorder+1)*lineWidth]
			copy(save, dst)
			copy(dst, src)
		}
	}
	if stripe.CopyBelow {
		if len(scratch.Below) < lineWidth*restorationBorder {
			return ErrInvalidPlan
		}
		for i := 0; i < restorationBorder; i++ {
			dst, ok := restorationStripeDataLine(data, dataStride, dataOrigin, stripe, int(stripe.Rect.Height())+i, lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			bufRow := rsbRow + minInt(i, restorationCtxVert-1)
			src, ok := restorationBoundaryLine(boundaries.Below, boundaries.Stride, bufRow, int(stripe.Rect.X0), lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			save := scratch.Below[i*lineWidth : (i+1)*lineWidth]
			copy(save, dst)
			copy(dst, src)
		}
	}
	return nil
}

func setupRestorationStripeBoundaryOptimized(unitRect RestorationUnitRect, stripe RestorationProcessingStripe, data []uint16, dataStride int, dataOrigin int, scratch RestorationStripeBoundaryScratch, lineWidth int) error {
	if stripe.CopyAbove {
		if len(scratch.Above) < lineWidth {
			return ErrInvalidPlan
		}
		dst, ok := restorationStripeDataLine(data, dataStride, dataOrigin, stripe, -restorationBorder, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		src, ok := restorationStripeDataLine(data, dataStride, dataOrigin, stripe, -restorationBorder+1, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		copy(scratch.Above[:lineWidth], dst)
		copy(dst, src)
	}
	if stripe.CopyBelow {
		if len(scratch.Below) < lineWidth {
			return ErrInvalidPlan
		}
		stripeHeight := int(stripe.Rect.Height())
		dst, ok := restorationStripeDataLine(data, dataStride, dataOrigin, stripe, stripeHeight+2, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		src, ok := restorationStripeDataLine(data, dataStride, dataOrigin, stripe, stripeHeight+1, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		copy(scratch.Below[:lineWidth], dst)
		copy(dst, src)
	}
	_ = unitRect
	return nil
}

func restoreRestorationStripeBoundarySaved(unitRect RestorationUnitRect, stripe RestorationProcessingStripe, data []uint16, dataStride int, dataOrigin int, scratch RestorationStripeBoundaryScratch, lineWidth int) error {
	if stripe.CopyAbove {
		if len(scratch.Above) < lineWidth*restorationBorder {
			return ErrInvalidPlan
		}
		for i := -restorationBorder; i < 0; i++ {
			dst, ok := restorationStripeDataLine(data, dataStride, dataOrigin, stripe, i, lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			save := scratch.Above[(i+restorationBorder)*lineWidth : (i+restorationBorder+1)*lineWidth]
			copy(dst, save)
		}
	}
	if stripe.CopyBelow {
		if len(scratch.Below) < lineWidth*restorationBorder {
			return ErrInvalidPlan
		}
		stripeBottom := int(stripe.Rect.Y1)
		unitEnd := int(unitRect.Y1)
		for i := 0; i < restorationBorder; i++ {
			if stripeBottom+i >= unitEnd+restorationBorder {
				break
			}
			dst, ok := restorationStripeDataLine(data, dataStride, dataOrigin, stripe, int(stripe.Rect.Height())+i, lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			save := scratch.Below[i*lineWidth : (i+1)*lineWidth]
			copy(dst, save)
		}
	}
	return nil
}

func restoreRestorationStripeBoundaryOptimized(unitRect RestorationUnitRect, stripe RestorationProcessingStripe, data []uint16, dataStride int, dataOrigin int, scratch RestorationStripeBoundaryScratch, lineWidth int) error {
	if stripe.CopyAbove {
		if len(scratch.Above) < lineWidth {
			return ErrInvalidPlan
		}
		dst, ok := restorationStripeDataLine(data, dataStride, dataOrigin, stripe, -restorationBorder, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		copy(dst, scratch.Above[:lineWidth])
	}
	if stripe.CopyBelow {
		if len(scratch.Below) < lineWidth {
			return ErrInvalidPlan
		}
		stripeBottom := int(stripe.Rect.Y1)
		unitEnd := int(unitRect.Y1)
		if stripeBottom+2 < unitEnd+restorationBorder {
			dst, ok := restorationStripeDataLine(data, dataStride, dataOrigin, stripe, int(stripe.Rect.Height())+2, lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			copy(dst, scratch.Below[:lineWidth])
		}
	}
	return nil
}

func saveRestorationDeblockBoundaryLines(grid RestorationPlaneGrid, src []uint16, srcStride int, srcOrigin int, row int, stripe int, above bool, boundaries RestorationStripeBoundaries) error {
	planeHeight := int(grid.PlaneHeight)
	linesToSave := minInt(restorationCtxVert, planeHeight-row)
	if row < 0 || linesToSave <= 0 || linesToSave > restorationCtxVert {
		return ErrInvalidPlan
	}
	for i := 0; i < linesToSave; i++ {
		if err := saveRestorationBoundaryLine(grid, src, srcStride, srcOrigin, row+i, stripe, i, above, boundaries); err != nil {
			return err
		}
	}
	if linesToSave == 1 {
		dst, ok := restorationBoundaryPlaneLine(boundariesForSide(boundaries, above), boundaries.Stride, stripe*restorationCtxVert+1, int(grid.PlaneWidth))
		if !ok {
			return ErrInvalidPlan
		}
		srcLine, ok := restorationBoundaryPlaneLine(boundariesForSide(boundaries, above), boundaries.Stride, stripe*restorationCtxVert, int(grid.PlaneWidth))
		if !ok {
			return ErrInvalidPlan
		}
		copy(dst, srcLine)
	}
	return extendRestorationBoundaryLines(grid, stripe, above, boundaries)
}

func saveRestorationCDEFBoundaryLines(grid RestorationPlaneGrid, src []uint16, srcStride int, srcOrigin int, row int, stripe int, above bool, boundaries RestorationStripeBoundaries) error {
	if row < 0 || row >= int(grid.PlaneHeight) {
		return ErrInvalidPlan
	}
	for i := 0; i < restorationCtxVert; i++ {
		if err := saveRestorationBoundaryLine(grid, src, srcStride, srcOrigin, row, stripe, i, above, boundaries); err != nil {
			return err
		}
	}
	return extendRestorationBoundaryLines(grid, stripe, above, boundaries)
}

func saveRestorationBoundaryLine(grid RestorationPlaneGrid, src []uint16, srcStride int, srcOrigin int, srcRow int, stripe int, ctxRow int, above bool, boundaries RestorationStripeBoundaries) error {
	width := int(grid.PlaneWidth)
	srcOffset, ok := restorationSignedPlaneOffset(srcOrigin, srcStride, 0, srcRow)
	if !ok || !restorationLineFits(len(src), srcOffset, width) {
		return ErrInvalidPlan
	}
	dst, ok := restorationBoundaryPlaneLine(boundariesForSide(boundaries, above), boundaries.Stride, stripe*restorationCtxVert+ctxRow, width)
	if !ok {
		return ErrInvalidPlan
	}
	copy(dst[restorationExtraHorz:restorationExtraHorz+width], src[srcOffset:srcOffset+width])
	return nil
}

func extendRestorationBoundaryLines(grid RestorationPlaneGrid, stripe int, above bool, boundaries RestorationStripeBoundaries) error {
	width := int(grid.PlaneWidth)
	for i := 0; i < restorationCtxVert; i++ {
		line, ok := restorationBoundaryPlaneLine(boundariesForSide(boundaries, above), boundaries.Stride, stripe*restorationCtxVert+i, width)
		if !ok {
			return ErrInvalidPlan
		}
		first := line[restorationExtraHorz]
		last := line[restorationExtraHorz+width-1]
		for j := 0; j < restorationExtraHorz; j++ {
			line[j] = first
			line[restorationExtraHorz+width+j] = last
		}
	}
	return nil
}

func validateRestorationStripeBoundaries(grid RestorationPlaneGrid, boundaries RestorationStripeBoundaries) error {
	size, err := RestorationStripeBoundaryBufferLen(grid)
	if err != nil {
		return err
	}
	need, ok := checkedMulInt(boundaries.Stride, size.Rows)
	if !ok {
		return ErrInvalidPlan
	}
	if boundaries.Stride < size.Stride ||
		len(boundaries.Above) < need ||
		len(boundaries.Below) < need {
		return ErrInvalidPlan
	}
	return nil
}

func boundariesForSide(boundaries RestorationStripeBoundaries, above bool) []uint16 {
	if above {
		return boundaries.Above
	}
	return boundaries.Below
}

func restorationBoundaryPlaneLine(buf []uint16, stride int, row int, planeWidth int) ([]uint16, bool) {
	lineWidth := planeWidth + 2*restorationExtraHorz
	return restorationBoundaryLine(buf, stride, row, 0, lineWidth)
}

func restorationFrameLine(data []uint16, stride int, origin int, x int, y int, lineWidth int) ([]uint16, bool) {
	offset, ok := restorationSignedPlaneOffset(origin, stride, x, y)
	if !ok || !restorationLineFits(len(data), offset, lineWidth) {
		return nil, false
	}
	return data[offset : offset+lineWidth], true
}

func validateRestorationStripeBoundaryInputs(unitRect RestorationUnitRect, stripe RestorationProcessingStripe, dataStride int, dataOrigin int) error {
	if !unitRect.valid() || !stripe.Rect.valid() ||
		stripe.Rect.X0 != unitRect.X0 ||
		stripe.Rect.X1 != unitRect.X1 ||
		stripe.Rect.Y0 < unitRect.Y0 ||
		stripe.Rect.Y1 > unitRect.Y1 ||
		dataStride <= 0 ||
		dataOrigin < 0 {
		return ErrInvalidPlan
	}
	return nil
}

func restorationStripeBoundaryLineWidth(stripe RestorationProcessingStripe) (int, error) {
	if !stripe.Rect.valid() {
		return 0, ErrInvalidPlan
	}
	width := uint64(stripe.Rect.Width()) + 2*restorationExtraHorz
	maxInt := uint64(^uint(0) >> 1)
	if width == 0 || width > maxInt {
		return 0, ErrInvalidPlan
	}
	return int(width), nil
}

func restorationStripeDataLine(data []uint16, stride int, origin int, stripe RestorationProcessingStripe, rowOffset int, lineWidth int) ([]uint16, bool) {
	x := int(stripe.Rect.X0) - restorationExtraHorz
	y := int(stripe.Rect.Y0) + rowOffset
	offset, ok := restorationSignedPlaneOffset(origin, stride, x, y)
	if !ok || !restorationLineFits(len(data), offset, lineWidth) {
		return nil, false
	}
	return data[offset : offset+lineWidth], true
}

func restorationBoundaryLine(buf []uint16, stride int, row int, x0 int, lineWidth int) ([]uint16, bool) {
	if stride <= 0 || row < 0 || x0 < 0 {
		return nil, false
	}
	offset, ok := checkedMulInt(row, stride)
	if !ok {
		return nil, false
	}
	offset, ok = checkedAddInt(offset, x0)
	if !ok || !restorationLineFits(len(buf), offset, lineWidth) {
		return nil, false
	}
	return buf[offset : offset+lineWidth], true
}

func restorationSignedPlaneOffset(origin int, stride int, x int, y int) (int, bool) {
	if stride <= 0 {
		return 0, false
	}
	offset := int64(origin) + int64(y)*int64(stride) + int64(x)
	if offset < 0 || offset > int64(^uint(0)>>1) {
		return 0, false
	}
	return int(offset), true
}

func restorationLineFits(length int, offset int, lineWidth int) bool {
	return offset >= 0 && lineWidth > 0 && offset <= length-lineWidth
}

func checkedAddInt(a int, b int) (int, bool) {
	if b > 0 && a > int(^uint(0)>>1)-b {
		return 0, false
	}
	if b < 0 && a < -int(^uint(0)>>1)-1-b {
		return 0, false
	}
	return a + b, true
}

func checkedMulInt(a int, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > int(^uint(0)>>1)/a {
		return 0, false
	}
	return a * b, true
}

func ceilDivInt(n int, d int) int {
	if n <= 0 || d <= 0 {
		return 0
	}
	return 1 + (n-1)/d
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
