package tile

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
