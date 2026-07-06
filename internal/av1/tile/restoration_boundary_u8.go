package tile

// 8-bit-pixel counterparts of the loop-restoration frame-extension and
// stripe-boundary save/restore flow. dav1d's 8bpc restoration
// (src/lr_apply_tmpl.c) runs these row juggles directly on uint8 pixel rows;
// these are the same mechanical ports as the uint16 versions in
// restoration_boundary.go with only the sample domain changed:
//
//   - The plane snapshot rows are uint8 (plain byte copies).
//   - The saved boundary context (RestorationStripeBoundaries) stays uint16 —
//     it is captured from an 8-bit plane, so every value is <= 255 and the
//     narrowing copy into the u8 snapshot is an exact byte cast.
//   - The overwritten-row scratch (RestorationStripeBoundaryScratch) also
//     stays uint16: rows are widened on save and narrowed on restore, an
//     exact round-trip for 8-bit samples.

// extendRestorationFrameU8 ports av1_extend_frame for uint8 sample planes;
// see ExtendRestorationFrame for the uint16 contract.
func extendRestorationFrameU8(data []uint8, stride int, origin int, width int, height int, borderHorz int, borderVert int) error {
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

	for y := range height {
		line, ok := restorationFrameLineU8(data, stride, origin, -borderHorz, y, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		first := line[borderHorz]
		last := line[borderHorz+width-1]
		for x := range borderHorz {
			line[x] = first
			line[borderHorz+width+x] = last
		}
	}

	top, ok := restorationFrameLineU8(data, stride, origin, -borderHorz, 0, lineWidth)
	if !ok {
		return ErrInvalidPlan
	}
	for y := -borderVert; y < 0; y++ {
		line, ok := restorationFrameLineU8(data, stride, origin, -borderHorz, y, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		copy(line, top)
	}

	bottom, ok := restorationFrameLineU8(data, stride, origin, -borderHorz, height-1, lineWidth)
	if !ok {
		return ErrInvalidPlan
	}
	for y := height; y < height+borderVert; y++ {
		line, ok := restorationFrameLineU8(data, stride, origin, -borderHorz, y, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		copy(line, bottom)
	}
	return nil
}

// setupRestorationStripeBoundaryU8 ports setup_processing_stripe_boundary for
// a uint8 plane snapshot; see SetupRestorationStripeBoundary.
func setupRestorationStripeBoundaryU8(unitRect RestorationUnitRect, stripe RestorationProcessingStripe, boundaries RestorationStripeBoundaries, data []uint8, dataStride int, dataOrigin int, scratch RestorationStripeBoundaryScratch, optimized bool) error {
	if err := validateRestorationStripeBoundaryInputs(unitRect, stripe, dataStride, dataOrigin); err != nil {
		return err
	}
	lineWidth, err := restorationStripeBoundaryLineWidth(stripe)
	if err != nil {
		return err
	}
	if optimized {
		return setupRestorationStripeBoundaryOptimizedU8(unitRect, stripe, data, dataStride, dataOrigin, scratch, lineWidth)
	}
	return setupRestorationStripeBoundaryFromSavedU8(stripe, boundaries, data, dataStride, dataOrigin, scratch, lineWidth)
}

// restoreRestorationStripeBoundaryU8 restores rows saved by
// setupRestorationStripeBoundaryU8; see RestoreRestorationStripeBoundary.
func restoreRestorationStripeBoundaryU8(unitRect RestorationUnitRect, stripe RestorationProcessingStripe, data []uint8, dataStride int, dataOrigin int, scratch RestorationStripeBoundaryScratch, optimized bool) error {
	if err := validateRestorationStripeBoundaryInputs(unitRect, stripe, dataStride, dataOrigin); err != nil {
		return err
	}
	lineWidth, err := restorationStripeBoundaryLineWidth(stripe)
	if err != nil {
		return err
	}
	if optimized {
		return restoreRestorationStripeBoundaryOptimizedU8(unitRect, stripe, data, dataStride, dataOrigin, scratch, lineWidth)
	}
	return restoreRestorationStripeBoundarySavedU8(unitRect, stripe, data, dataStride, dataOrigin, scratch, lineWidth)
}

func setupRestorationStripeBoundaryFromSavedU8(stripe RestorationProcessingStripe, boundaries RestorationStripeBoundaries, data []uint8, dataStride int, dataOrigin int, scratch RestorationStripeBoundaryScratch, lineWidth int) error {
	rsbRow := int(stripe.TileStripe) * restorationCtxVert
	if stripe.CopyAbove {
		if len(scratch.Above) < lineWidth*restorationBorder {
			return ErrInvalidPlan
		}
		for i := -restorationBorder; i < 0; i++ {
			dst, ok := restorationStripeDataLineU8(data, dataStride, dataOrigin, stripe, i, lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			bufRow := rsbRow + maxInt(i+restorationCtxVert, 0)
			src, ok := restorationBoundaryLine(boundaries.Above, boundaries.Stride, bufRow, int(stripe.Rect.X0), lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			save := scratch.Above[(i+restorationBorder)*lineWidth : (i+restorationBorder+1)*lineWidth]
			widenRestorationLineU8(save, dst)
			narrowRestorationLineU16(dst, src)
		}
	}
	if stripe.CopyBelow {
		if len(scratch.Below) < lineWidth*restorationBorder {
			return ErrInvalidPlan
		}
		for i := range restorationBorder {
			dst, ok := restorationStripeDataLineU8(data, dataStride, dataOrigin, stripe, int(stripe.Rect.Height())+i, lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			bufRow := rsbRow + minInt(i, restorationCtxVert-1)
			src, ok := restorationBoundaryLine(boundaries.Below, boundaries.Stride, bufRow, int(stripe.Rect.X0), lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			save := scratch.Below[i*lineWidth : (i+1)*lineWidth]
			widenRestorationLineU8(save, dst)
			narrowRestorationLineU16(dst, src)
		}
	}
	return nil
}

func setupRestorationStripeBoundaryOptimizedU8(unitRect RestorationUnitRect, stripe RestorationProcessingStripe, data []uint8, dataStride int, dataOrigin int, scratch RestorationStripeBoundaryScratch, lineWidth int) error {
	if stripe.CopyAbove {
		if len(scratch.Above) < lineWidth {
			return ErrInvalidPlan
		}
		dst, ok := restorationStripeDataLineU8(data, dataStride, dataOrigin, stripe, -restorationBorder, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		src, ok := restorationStripeDataLineU8(data, dataStride, dataOrigin, stripe, -restorationBorder+1, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		widenRestorationLineU8(scratch.Above[:lineWidth], dst)
		copy(dst, src)
	}
	if stripe.CopyBelow {
		if len(scratch.Below) < lineWidth {
			return ErrInvalidPlan
		}
		stripeHeight := int(stripe.Rect.Height())
		dst, ok := restorationStripeDataLineU8(data, dataStride, dataOrigin, stripe, stripeHeight+2, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		src, ok := restorationStripeDataLineU8(data, dataStride, dataOrigin, stripe, stripeHeight+1, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		widenRestorationLineU8(scratch.Below[:lineWidth], dst)
		copy(dst, src)
	}
	_ = unitRect
	return nil
}

func restoreRestorationStripeBoundarySavedU8(unitRect RestorationUnitRect, stripe RestorationProcessingStripe, data []uint8, dataStride int, dataOrigin int, scratch RestorationStripeBoundaryScratch, lineWidth int) error {
	if stripe.CopyAbove {
		if len(scratch.Above) < lineWidth*restorationBorder {
			return ErrInvalidPlan
		}
		for i := -restorationBorder; i < 0; i++ {
			dst, ok := restorationStripeDataLineU8(data, dataStride, dataOrigin, stripe, i, lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			save := scratch.Above[(i+restorationBorder)*lineWidth : (i+restorationBorder+1)*lineWidth]
			narrowRestorationLineU16(dst, save)
		}
	}
	if stripe.CopyBelow {
		if len(scratch.Below) < lineWidth*restorationBorder {
			return ErrInvalidPlan
		}
		stripeBottom := int(stripe.Rect.Y1)
		unitEnd := int(unitRect.Y1)
		for i := range restorationBorder {
			if stripeBottom+i >= unitEnd+restorationBorder {
				break
			}
			dst, ok := restorationStripeDataLineU8(data, dataStride, dataOrigin, stripe, int(stripe.Rect.Height())+i, lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			save := scratch.Below[i*lineWidth : (i+1)*lineWidth]
			narrowRestorationLineU16(dst, save)
		}
	}
	return nil
}

func restoreRestorationStripeBoundaryOptimizedU8(unitRect RestorationUnitRect, stripe RestorationProcessingStripe, data []uint8, dataStride int, dataOrigin int, scratch RestorationStripeBoundaryScratch, lineWidth int) error {
	if stripe.CopyAbove {
		if len(scratch.Above) < lineWidth {
			return ErrInvalidPlan
		}
		dst, ok := restorationStripeDataLineU8(data, dataStride, dataOrigin, stripe, -restorationBorder, lineWidth)
		if !ok {
			return ErrInvalidPlan
		}
		narrowRestorationLineU16(dst, scratch.Above[:lineWidth])
	}
	if stripe.CopyBelow {
		if len(scratch.Below) < lineWidth {
			return ErrInvalidPlan
		}
		stripeBottom := int(stripe.Rect.Y1)
		unitEnd := int(unitRect.Y1)
		if stripeBottom+2 < unitEnd+restorationBorder {
			dst, ok := restorationStripeDataLineU8(data, dataStride, dataOrigin, stripe, int(stripe.Rect.Height())+2, lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			narrowRestorationLineU16(dst, scratch.Below[:lineWidth])
		}
	}
	return nil
}

func restorationFrameLineU8(data []uint8, stride int, origin int, x int, y int, lineWidth int) ([]uint8, bool) {
	offset, ok := restorationSignedPlaneOffset(origin, stride, x, y)
	if !ok || !restorationLineFits(len(data), offset, lineWidth) {
		return nil, false
	}
	return data[offset : offset+lineWidth], true
}

func restorationStripeDataLineU8(data []uint8, stride int, origin int, stripe RestorationProcessingStripe, rowOffset int, lineWidth int) ([]uint8, bool) {
	x := int(stripe.Rect.X0) - restorationExtraHorz
	y := int(stripe.Rect.Y0) + rowOffset
	offset, ok := restorationSignedPlaneOffset(origin, stride, x, y)
	if !ok || !restorationLineFits(len(data), offset, lineWidth) {
		return nil, false
	}
	return data[offset : offset+lineWidth], true
}

// widenRestorationLineU8 saves one uint8 row into uint16 scratch (exact for
// all 8-bit samples).
func widenRestorationLineU8(dst []uint16, src []uint8) {
	_ = dst[len(src)-1]
	for i, s := range src {
		dst[i] = uint16(s)
	}
}

// narrowRestorationLineU16 writes one uint16 row into a uint8 row with a
// trusted byte cast. Callers only pass rows captured from 8-bit planes
// (boundary context or the widened save above), so every value is <= 255 and
// the cast is exact.
func narrowRestorationLineU16(dst []uint8, src []uint16) {
	_ = dst[len(src)-1]
	for i, s := range src {
		dst[i] = uint8(s)
	}
}
