package tile

import (
	av1frame "github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

// In-place 8-bit loop-restoration walk, ported from dav1d's restoration
// architecture (src/lr_apply_tmpl.c):
//
//   - lr_sbrow walks restoration units left-to-right and filters the frame
//     surface IN PLACE; before a unit is overwritten, backup4xU saves its
//     last 4 pre-filter columns into pre_lr_border so the next unit's left
//     halo still reads pre-filter pixels. goav1's record walk is strictly
//     row-major (validateRestorationPlaneRecords pins records[i].Index == i),
//     so the same single-neighbour backup discipline applies.
//   - lr_stripe feeds the vertical halo across every stripe seam from the
//     backed-up loop-filter lines (lpf / f->lf.lr_lpf_line), never from the
//     surface; goav1's equivalent is RestorationStripeBoundaries
//     (Above/Below), already saved pre-restoration by
//     SaveRestorationBoundaryLines*. CopyAbove/CopyBelow are false only at
//     the plane's top/bottom edge, where the halo is edge replication of the
//     unit's own (still pre-filter) rows — so no vertical surface read ever
//     sees another unit's filtered output.
//   - dav1d's C filters assemble a small padded source block per stripe
//     (padding() into tmp in src/looprestoration_tmpl.c) because the surface
//     has no left border to read through; goav1's uint8 kernels likewise
//     require a contiguous bordered source, so assembleRestorationStripeBandU8
//     plays the padding() role: interior rows from the frame, left halo from
//     the neighbour backup, seam rows from the boundary buffers, plane edges
//     replicated. The kernels then write the frame plane directly.
//
// Byte-exactness contract: the band holds exactly the pixels the whole-plane
// pre-restoration snapshot walk (applyRestorationPlaneSnapshotU8) would
// expose to the kernels — pre-filter surface pixels for the unit and its
// unfiltered neighbours, backup pixels for the filtered left neighbour,
// boundary-buffer rows at stripe seams, and edge replication at plane
// borders — so the filtered output is bit-identical.

// restorationInPlaceU8Layout carves the per-plane in-place scratch out of the
// caller's uint16 sample arena viewed as bytes: one stripe band plus two
// ping-pong left-halo column backups.
type restorationInPlaceU8Layout struct {
	Band       []uint8
	BandStride int
	Backup     [2][]uint8
}

// bindRestorationInPlaceU8Layout sizes and slices the band and backup buffers
// for one plane. The band covers the widest unit plus restorationExtraHorz
// columns on both sides and the tallest processing stripe plus
// restorationBorder rows above and below; each backup holds
// restorationExtraHorz pre-filter columns over the tallest unit.
func bindRestorationInPlaceU8Layout(grid RestorationPlaneGrid, seg []uint16) (restorationInPlaceU8Layout, error) {
	if !grid.validGeometry() {
		return restorationInPlaceU8Layout{}, ErrInvalidPlan
	}
	maxUnitW := int(grid.PlaneWidth)
	if grid.HorzUnits > 1 {
		lastW := int(grid.PlaneWidth) - (int(grid.HorzUnits)-1)*int(grid.UnitSize)
		maxUnitW = maxInt(int(grid.UnitSize), lastW)
	}
	offset := int(restorationUnitOffset >> boolToShift(grid.SubsamplingY))
	maxUnitH := int(grid.PlaneHeight)
	if grid.VertUnits > 1 {
		lastH := int(grid.PlaneHeight) + offset - (int(grid.VertUnits)-1)*int(grid.UnitSize)
		maxUnitH = maxInt(int(grid.UnitSize), lastH)
	}
	if maxUnitW <= 0 || maxUnitH <= 0 {
		return restorationInPlaceU8Layout{}, ErrInvalidPlan
	}
	stripeH := int(av1restoration.ProcUnitSize >> boolToShift(grid.SubsamplingY))
	if stripeH <= 0 {
		return restorationInPlaceU8Layout{}, ErrInvalidPlan
	}
	bandStride, ok := checkedAddInt(maxUnitW, 2*restorationExtraHorz)
	if !ok {
		return restorationInPlaceU8Layout{}, ErrInvalidPlan
	}
	bandStride, ok = alignPowerOfTwoInt(bandStride, 4)
	if !ok {
		return restorationInPlaceU8Layout{}, ErrInvalidPlan
	}
	bandRows := stripeH + 2*restorationBorder
	bandLen, ok := checkedMulInt(bandStride, bandRows)
	if !ok {
		return restorationInPlaceU8Layout{}, ErrInvalidPlan
	}
	bakLen, ok := checkedMulInt(maxUnitH, restorationExtraHorz)
	if !ok {
		return restorationInPlaceU8Layout{}, ErrInvalidPlan
	}
	need, ok := checkedAddInt(bandLen, 2*bakLen)
	if !ok {
		return restorationInPlaceU8Layout{}, ErrInvalidPlan
	}
	bytes, ok := restorationByteScratchView(seg, need)
	if !ok {
		return restorationInPlaceU8Layout{}, ErrInvalidPlan
	}
	return restorationInPlaceU8Layout{
		Band:       bytes[:bandLen],
		BandStride: bandStride,
		Backup: [2][]uint8{
			bytes[bandLen : bandLen+bakLen],
			bytes[bandLen+bakLen : bandLen+2*bakLen],
		},
	}, nil
}

// applyRestorationPlaneRecordsInPlaceU8 filters one 8-bit plane's restoration
// records in place on the frame surface (dav1d lr_sbrow walk; see the file
// comment). RESTORATION_NONE records are validated and counted but never
// touched; result accounting matches applyRestorationPlaneRecordsFilteredU8
// exactly.
func applyRestorationPlaneRecordsInPlaceU8(grid RestorationPlaneGrid, records []RestorationUnitRecord, boundaries RestorationStripeBoundaries, buffer av1frame.Plane, dataSeg []uint16, scratch RestorationUnitRecordBoundaryScratch) (RestorationPlaneApplyResult, error) {
	if err := validateRestorationPlaneRecords(grid, records); err != nil {
		return RestorationPlaneApplyResult{}, err
	}
	// The production walk always saves stripe boundary lines pre-restoration
	// (optimized_lr is never set); validate them like the snapshot walk does.
	if err := validateRestorationStripeBoundaries(grid, boundaries); err != nil {
		return RestorationPlaneApplyResult{}, err
	}
	if buffer.Stride <= 0 || buffer.Width != int(grid.PlaneWidth) || buffer.Height != int(grid.PlaneHeight) {
		return RestorationPlaneApplyResult{}, ErrInvalidPlan
	}
	layout, err := bindRestorationInPlaceU8Layout(grid, dataSeg)
	if err != nil {
		return RestorationPlaneApplyResult{}, err
	}

	var result RestorationPlaneApplyResult
	// pre_lr_border ping-pong: prev holds the filtered left neighbour's
	// pre-filter right-edge columns, cur receives the current unit's.
	prevBak, curBak := layout.Backup[0], layout.Backup[1]
	leftFiltered := false
	for i := range records {
		record := &records[i]
		if record.Col == 0 {
			leftFiltered = false
		}
		if record.Unit.Type == parser.RestorationNone {
			if err := validateRestorationUnitRecord(grid, *record); err != nil {
				return RestorationPlaneApplyResult{}, err
			}
			result.Records++
			// An unfiltered unit's pixels stay pre-filter in the frame, so
			// the next unit's halo reads it directly (dav1d gates backup4xU
			// on restore_next only because its backup runs one unit early;
			// the values are identical either way).
			leftFiltered = false
			continue
		}
		// dav1d lr_sbrow: back up the unit's pre-filter right-edge columns
		// before its filter overwrites them, iff a right neighbour will read
		// them as its left halo.
		if int(record.Col)+1 < int(grid.HorzUnits) && records[i+1].Unit.Type != parser.RestorationNone {
			if err := backupRestorationRightColsU8(buffer.Pix, buffer.Stride, record.Rect, curBak); err != nil {
				return RestorationPlaneApplyResult{}, err
			}
		}
		var leftBak []uint8
		if record.Col > 0 && leftFiltered {
			leftBak = prevBak
		}
		recordResult, err := applyRestorationUnitRecordInPlaceU8(grid, *record, boundaries, buffer, layout.Band, layout.BandStride, leftBak, scratch)
		if err != nil {
			return RestorationPlaneApplyResult{}, err
		}
		result.Records++
		if recordResult.Filtered {
			result.FilteredRecords++
		}
		result.Stripes += uint32(recordResult.Stripes)
		result.ProcessingUnits += uint32(recordResult.ProcessingUnits)
		prevBak, curBak = curBak, prevBak
		leftFiltered = true
	}
	return result, nil
}

// applyRestorationUnitRecordInPlaceU8 filters one restoration unit stripe by
// stripe (dav1d lr_stripe): each stripe's bordered source band is assembled
// from the frame, the left-neighbour backup, and the boundary line buffers,
// then the kernels write the frame plane directly.
func applyRestorationUnitRecordInPlaceU8(grid RestorationPlaneGrid, record RestorationUnitRecord, boundaries RestorationStripeBoundaries, buffer av1frame.Plane, band []uint8, bandStride int, leftBak []uint8, scratch RestorationUnitRecordBoundaryScratch) (RestorationUnitRecordApplyResult, error) {
	if err := validateRestorationUnitRecord(grid, record); err != nil {
		return RestorationUnitRecordApplyResult{}, err
	}
	if record.Unit.Type == parser.RestorationNone {
		return RestorationUnitRecordApplyResult{}, ErrInvalidPlan
	}

	result := RestorationUnitRecordApplyResult{Type: record.Unit.Type, Filtered: true}
	for stripeIndex := 0; stripeIndex < int(record.StripeCount); stripeIndex++ {
		stripe, ok, err := grid.ProcessingStripe(record.Rect, stripeIndex)
		if err != nil {
			return RestorationUnitRecordApplyResult{}, err
		}
		if !ok {
			return RestorationUnitRecordApplyResult{}, ErrInvalidPlan
		}
		if err := assembleRestorationStripeBandU8(grid, record.Rect, stripe, boundaries, buffer, band, bandStride, leftBak); err != nil {
			return RestorationUnitRecordApplyResult{}, err
		}
		unitCount, err := applyRestorationStripeUnitsInPlaceU8(grid, record, stripe, band, bandStride, buffer, scratch.Unit)
		if err != nil {
			return RestorationUnitRecordApplyResult{}, err
		}
		if unitCount > int(^uint16(0)) || result.ProcessingUnits > ^uint16(0)-uint16(unitCount) {
			return RestorationUnitRecordApplyResult{}, ErrInvalidPlan
		}
		result.Stripes++
		result.ProcessingUnits += uint16(unitCount)
	}
	return result, nil
}

// assembleRestorationStripeBandU8 builds the bordered uint8 source band for
// one processing stripe (the role of dav1d's padding() into the padded tmp
// block in src/looprestoration_tmpl.c). Band column 0 is plane column
// stripe.X0-restorationExtraHorz; band row 0 is plane row
// stripe.Y0-restorationBorder. The band reproduces exactly what the
// whole-plane pre-restoration snapshot would hold after boundary setup:
//
//	interior rows  <- frame surface (pre-filter: the walk is row-major and
//	                  each pixel is written at most once), with the left halo
//	                  overridden from the filtered left neighbour's
//	                  pre-filter column backup;
//	seam rows      <- saved deblock/CDEF boundary lines
//	                  (setup_processing_stripe_boundary's non-optimized
//	                  copy, bufRow = rsbRow + max(i+2,0) / rsbRow + min(i,1));
//	plane edges    <- edge replication (av1_extend_frame).
func assembleRestorationStripeBandU8(grid RestorationPlaneGrid, unitRect RestorationUnitRect, stripe RestorationProcessingStripe, boundaries RestorationStripeBoundaries, buffer av1frame.Plane, band []uint8, bandStride int, leftBak []uint8) error {
	if err := validateRestorationStripeBoundaryInputs(unitRect, stripe, bandStride, 0); err != nil {
		return err
	}
	lineWidth, err := restorationStripeBoundaryLineWidth(stripe)
	if err != nil {
		return err
	}
	height := int(stripe.Rect.Height())
	if bandStride < lineWidth || len(band) < (height+2*restorationBorder)*bandStride {
		return ErrInvalidPlan
	}
	x0 := int(stripe.Rect.X0)
	y0 := int(stripe.Rect.Y0)
	planeWidth := int(grid.PlaneWidth)
	srcL := x0 - restorationExtraHorz
	srcR := srcL + lineWidth
	clipL := maxInt(srcL, 0)
	clipR := minInt(srcR, planeWidth)
	if clipL >= clipR || y0 < 0 || (y0+height)*buffer.Stride > len(buffer.Pix) {
		return ErrInvalidPlan
	}
	if leftBak != nil {
		bakRows := (y0 + height - int(unitRect.Y0)) * restorationExtraHorz
		if y0 < int(unitRect.Y0) || len(leftBak) < bakRows {
			return ErrInvalidPlan
		}
	}

	// Interior stripe rows.
	for y := 0; y < height; y++ {
		bandRow := band[(restorationBorder+y)*bandStride:][:lineWidth]
		rowOff := (y0 + y) * buffer.Stride
		copy(bandRow[clipL-srcL:clipR-srcL], buffer.Pix[rowOff+clipL:rowOff+clipR])
		if leftBak != nil {
			// Filtered left neighbour: halo columns from its pre-filter
			// backup (dav1d pre_lr_border).
			bak := leftBak[(y0+y-int(unitRect.Y0))*restorationExtraHorz:][:restorationExtraHorz]
			copy(bandRow[:restorationExtraHorz], bak)
		} else if srcL < 0 {
			// Plane left edge: replicate column 0.
			v := bandRow[restorationExtraHorz]
			for x := range restorationExtraHorz {
				bandRow[x] = v
			}
		}
		if srcR > planeWidth {
			// Plane right edge: replicate the last plane column.
			v := bandRow[clipR-srcL-1]
			for x := clipR - srcL; x < lineWidth; x++ {
				bandRow[x] = v
			}
		}
	}

	// Stripe-seam context rows above and below (or plane-edge replication).
	rsbRow := int(stripe.TileStripe) * restorationCtxVert
	if stripe.CopyAbove {
		for i := -restorationBorder; i < 0; i++ {
			bandRow := band[(restorationBorder+i)*bandStride:][:lineWidth]
			bufRow := rsbRow + maxInt(i+restorationCtxVert, 0)
			src, ok := restorationBoundaryLine(boundaries.Above, boundaries.Stride, bufRow, x0, lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			narrowRestorationLineU16(bandRow, src)
		}
	} else {
		top := band[restorationBorder*bandStride:][:lineWidth]
		for i := range restorationBorder {
			copy(band[i*bandStride:][:lineWidth], top)
		}
	}
	if stripe.CopyBelow {
		for i := range restorationBorder {
			bandRow := band[(restorationBorder+height+i)*bandStride:][:lineWidth]
			bufRow := rsbRow + minInt(i, restorationCtxVert-1)
			src, ok := restorationBoundaryLine(boundaries.Below, boundaries.Stride, bufRow, x0, lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			narrowRestorationLineU16(bandRow, src)
		}
	} else {
		bottom := band[(restorationBorder+height-1)*bandStride:][:lineWidth]
		for i := range restorationBorder {
			copy(band[(restorationBorder+height+i)*bandStride:][:lineWidth], bottom)
		}
	}
	return nil
}

// applyRestorationStripeUnitsInPlaceU8 mirrors applyRestorationStripeUnitsU8
// with the band as the bordered source and the frame plane as the in-place
// destination.
func applyRestorationStripeUnitsInPlaceU8(grid RestorationPlaneGrid, record RestorationUnitRecord, stripe RestorationProcessingStripe, band []uint8, bandStride int, buffer av1frame.Plane, scratch RestorationUnitScratch) (int, error) {
	unitCount, err := grid.ProcessingUnitCount(stripe, record.Unit.Type)
	if err != nil {
		return 0, err
	}
	for unitIndex := range unitCount {
		unit, ok, err := grid.ProcessingUnit(stripe, record.Unit.Type, unitIndex)
		if err != nil {
			return 0, err
		}
		if !ok {
			return 0, ErrInvalidPlan
		}
		// Same VisibleRect-only filtering rationale as the snapshot walk in
		// applyRestorationStripeUnitsU8: FilterRect only exceeds VisibleRect
		// on the right and the visible columns' bytes are identical.
		if unit.VisibleRect.X0 < stripe.Rect.X0 || unit.VisibleRect.Y0 != stripe.Rect.Y0 {
			return 0, ErrInvalidPlan
		}
		srcBlock := restorationBorder*bandStride + restorationExtraHorz + int(unit.VisibleRect.X0-stripe.Rect.X0)
		dstBlock, ok := restorationPlaneOffset(0, buffer.Stride, unit.VisibleRect.X0, unit.VisibleRect.Y0)
		if !ok || dstBlock > len(buffer.Pix) {
			return 0, ErrInvalidPlan
		}
		if err := applyRestorationUnitU8(band, bandStride, srcBlock, buffer.Pix[dstBlock:], buffer.Stride, int(unit.VisibleRect.Width()), int(unit.VisibleRect.Height()), record.Unit, scratch); err != nil {
			return 0, err
		}
	}
	return unitCount, nil
}

// backupRestorationRightColsU8 saves a unit's pre-filter right-edge columns
// before its filter overwrites them (dav1d lr_apply_tmpl.c backup4xU): the
// right neighbour's band assembly reads them as its left halo. bak is packed
// as rect-height rows of restorationExtraHorz bytes.
func backupRestorationRightColsU8(pix []uint8, stride int, rect RestorationUnitRect, bak []uint8) error {
	width := int(rect.Width())
	height := int(rect.Height())
	if stride <= 0 || width < restorationExtraHorz || height <= 0 || len(bak) < height*restorationExtraHorz {
		return ErrInvalidPlan
	}
	x := int(rect.X1) - restorationExtraHorz
	y0 := int(rect.Y0)
	if (y0+height-1)*stride+x+restorationExtraHorz > len(pix) {
		return ErrInvalidPlan
	}
	for y := 0; y < height; y++ {
		copy(bak[y*restorationExtraHorz:][:restorationExtraHorz], pix[(y0+y)*stride+x:][:restorationExtraHorz])
	}
	return nil
}
