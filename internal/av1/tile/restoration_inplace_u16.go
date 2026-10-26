package tile

import (
	av1frame "github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

// In-place 10/12-bit loop-restoration stripe walk: the 16-bit counterpart of
// the 8-bit lr_sbrow walk in restoration_inplace_u8.go (dav1d's
// lr_apply_tmpl.c, templated for 16bpc). Each processing stripe of a unit is
// assembled into a bordered uint16 band exactly as the 8-bit walk assembles
// its byte band (interior rows from the surface, the left halo from the
// filtered left neighbour's pre-filter column backup, stripe-seam rows from
// the saved boundary lines, plane edges replicated), filtered with the same
// wide-sample unit kernels the whole-plane snapshot walk uses, and stored
// back into the little-endian 16-bit surface. The band holds exactly the
// samples the whole-plane snapshot would expose to the kernels, so the
// output is bit-identical to ApplyRestorationFrameToFrame.

// restorationInPlaceU16Layout carves one plane's in-place scratch out of a
// uint16 arena: the stripe band, two ping-pong left-halo column backups and
// the filtered stripe rows before they are stored.
type restorationInPlaceU16Layout struct {
	Band       []uint16
	BandStride int
	Backup     [2][]uint16
	Dst        []uint16
}

type restorationInPlaceU16ScratchSize struct {
	BandStride int
	BandLen    int
	BackupLen  int
	DstLen     int
	Len        int
}

func restorationInPlaceU16ScratchSizeForGrid(grid RestorationPlaneGrid) (restorationInPlaceU16ScratchSize, error) {
	u8, err := restorationInPlaceU8ScratchSizeForGrid(grid)
	if err != nil {
		return restorationInPlaceU16ScratchSize{}, err
	}
	stripeH := int(av1restoration.ProcUnitSize >> boolToShift(grid.SubsamplingY))
	dstLen, ok := checkedMulInt(u8.BandStride, stripeH)
	if !ok {
		return restorationInPlaceU16ScratchSize{}, ErrInvalidPlan
	}
	n := u8.BandLen
	for _, add := range [...]int{u8.BackupLen, u8.BackupLen, dstLen} {
		if n, ok = checkedAddInt(n, add); !ok {
			return restorationInPlaceU16ScratchSize{}, ErrInvalidPlan
		}
	}
	return restorationInPlaceU16ScratchSize{
		BandStride: u8.BandStride,
		BandLen:    u8.BandLen,
		BackupLen:  u8.BackupLen,
		DstLen:     dstLen,
		Len:        n,
	}, nil
}

// RestorationFramePlaneInPlaceU16ScratchLen reports the uint16 data scratch
// one worker needs to restore stripes of grid's 10/12-bit plane with
// ApplyRestorationFramePlaneInPlaceU16StripesFollowing.
func RestorationFramePlaneInPlaceU16ScratchLen(grid RestorationPlaneGrid) (int, error) {
	size, err := restorationInPlaceU16ScratchSizeForGrid(grid)
	if err != nil {
		return 0, err
	}
	return size.Len, nil
}

func bindRestorationInPlaceU16Layout(grid RestorationPlaneGrid, seg []uint16) (restorationInPlaceU16Layout, error) {
	size, err := restorationInPlaceU16ScratchSizeForGrid(grid)
	if err != nil {
		return restorationInPlaceU16Layout{}, err
	}
	if len(seg) < size.Len {
		return restorationInPlaceU16Layout{}, ErrInvalidPlan
	}
	b := size.BandLen
	k := size.BackupLen
	return restorationInPlaceU16Layout{
		Band:       seg[:b],
		BandStride: size.BandStride,
		Backup:     [2][]uint16{seg[b : b+k], seg[b+k : b+2*k]},
		Dst:        seg[b+2*k : b+2*k+size.DstLen],
	}, nil
}

// ApplyRestorationFramePlaneInPlaceU16StripesFollowing is
// ApplyRestorationFramePlaneInPlaceU8StripesFollowing for a 10/12-bit plane
// (buffer holds little-endian 16-bit samples): it restores processing stripes
// [stripeStart, stripeEnd) in place, reading only the records of the units the
// stripes cover. Stripes may run in any order and concurrently with other
// stripes, each with its own dataSeg and scratch.
func ApplyRestorationFramePlaneInPlaceU16StripesFollowing(grid RestorationPlaneGrid, records []RestorationUnitRecord, boundaries RestorationStripeBoundaries, buffer av1frame.Plane, bitDepth uint8, dataSeg []uint16, scratch RestorationUnitRecordBoundaryScratch, stripeStart, stripeEnd int) (RestorationPlaneApplyResult, error) {
	if !grid.validGeometry() || len(records) != int(grid.HorzUnits)*int(grid.VertUnits) || boundaries.Bytes {
		return RestorationPlaneApplyResult{}, ErrInvalidPlan
	}
	if bitDepth != 10 && bitDepth != 12 {
		return RestorationPlaneApplyResult{}, ErrInvalidPlan
	}
	if err := validateRestorationStripeBoundaries(grid, boundaries); err != nil {
		return RestorationPlaneApplyResult{}, err
	}
	if buffer.Stride <= 0 || buffer.Stride&1 != 0 || buffer.Width != int(grid.PlaneWidth) || buffer.Height != int(grid.PlaneHeight) ||
		buffer.Height*buffer.Stride > len(buffer.Pix) || 2*buffer.Width > buffer.Stride {
		return RestorationPlaneApplyResult{}, ErrInvalidPlan
	}
	if stripeStart < 0 || stripeEnd > RestorationTileStripeCount(grid) || stripeStart > stripeEnd {
		return RestorationPlaneApplyResult{}, ErrInvalidPlan
	}
	layout, err := bindRestorationInPlaceU16Layout(grid, dataSeg)
	if err != nil {
		return RestorationPlaneApplyResult{}, err
	}
	return walkRestorationPlaneStripesInPlaceU16(grid, records, boundaries, buffer, bitDepth, layout, scratch, stripeStart, stripeEnd)
}

func walkRestorationPlaneStripesInPlaceU16(grid RestorationPlaneGrid, records []RestorationUnitRecord, boundaries RestorationStripeBoundaries, buffer av1frame.Plane, bitDepth uint8, layout restorationInPlaceU16Layout, scratch RestorationUnitRecordBoundaryScratch, stripeStart, stripeEnd int) (RestorationPlaneApplyResult, error) {
	var result RestorationPlaneApplyResult
	stripeH := int(av1restoration.ProcUnitSize >> boolToShift(grid.SubsamplingY))
	off := int(restorationUnitOffset >> boolToShift(grid.SubsamplingY))
	unitSize := int(grid.UnitSize)
	horz := int(grid.HorzUnits)
	vert := int(grid.VertUnits)
	if stripeH <= 0 || unitSize <= 0 || horz <= 0 || vert <= 0 {
		return RestorationPlaneApplyResult{}, ErrInvalidPlan
	}
	for t := stripeStart; t < stripeEnd; t++ {
		y0 := max(0, t*stripeH-off)
		row := min((y0+off)/unitSize, vert-1)
		prevBak, curBak := layout.Backup[0], layout.Backup[1]
		leftFiltered := false
		for col := 0; col < horz; col++ {
			i := row*horz + col
			record := &records[i]
			firstStripe := (int(record.Rect.Y0) + off) / stripeH
			local := t - firstStripe
			if local < 0 || local >= int(record.StripeCount) {
				return RestorationPlaneApplyResult{}, ErrInvalidPlan
			}
			if local == 0 {
				// A unit is validated with its first stripe.
				if err := validateRestorationUnitRecord(grid, *record); err != nil {
					return RestorationPlaneApplyResult{}, err
				}
				result.Records++
				if record.Unit.Type != parser.RestorationNone {
					result.FilteredRecords++
				}
			}
			if record.Unit.Type == parser.RestorationNone {
				leftFiltered = false
				continue
			}
			stripe, ok, err := grid.ProcessingStripe(record.Rect, local)
			if err != nil {
				return RestorationPlaneApplyResult{}, err
			}
			if !ok || int(stripe.TileStripe) != t {
				return RestorationPlaneApplyResult{}, ErrInvalidPlan
			}
			stripeRect := record.Rect
			stripeRect.Y0, stripeRect.Y1 = stripe.Rect.Y0, stripe.Rect.Y1
			if col+1 < horz && records[i+1].Unit.Type != parser.RestorationNone {
				if err := backupRestorationRightColsU16(buffer.Pix, buffer.Stride, stripeRect, curBak); err != nil {
					return RestorationPlaneApplyResult{}, err
				}
			}
			var leftBak []uint16
			if col > 0 && leftFiltered {
				leftBak = prevBak
			}
			if err := assembleRestorationStripeBandU16(grid, stripeRect, stripe, boundaries, buffer, layout.Band, layout.BandStride, leftBak); err != nil {
				return RestorationPlaneApplyResult{}, err
			}
			unitCount, err := applyRestorationStripeUnitsInPlaceU16(grid, *record, stripe, layout, buffer, bitDepth, scratch.Unit)
			if err != nil {
				return RestorationPlaneApplyResult{}, err
			}
			result.Stripes++
			result.ProcessingUnits += uint32(unitCount)
			prevBak, curBak = curBak, prevBak
			leftFiltered = true
		}
	}
	return result, nil
}

func loadSampleU16LE(pix []byte, off int) uint16 {
	return uint16(pix[off]) | uint16(pix[off+1])<<8
}

// assembleRestorationStripeBandU16 is assembleRestorationStripeBandU8 for a
// 16-bit surface.
func assembleRestorationStripeBandU16(grid RestorationPlaneGrid, unitRect RestorationUnitRect, stripe RestorationProcessingStripe, boundaries RestorationStripeBoundaries, buffer av1frame.Plane, band []uint16, bandStride int, leftBak []uint16) error {
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

	for y := 0; y < height; y++ {
		bandRow := band[(restorationBorder+y)*bandStride:][:lineWidth]
		rowOff := (y0 + y) * buffer.Stride
		dst := bandRow[clipL-srcL : clipR-srcL]
		src := buffer.Pix[rowOff+2*clipL : rowOff+2*clipR]
		for x := range dst {
			dst[x] = uint16(src[2*x]) | uint16(src[2*x+1])<<8
		}
		if leftBak != nil {
			bak := leftBak[(y0+y-int(unitRect.Y0))*restorationExtraHorz:][:restorationExtraHorz]
			copy(bandRow[:restorationExtraHorz], bak)
		} else if srcL < 0 {
			v := bandRow[restorationExtraHorz]
			for x := range restorationExtraHorz {
				bandRow[x] = v
			}
		}
		if srcR > planeWidth {
			v := bandRow[clipR-srcL-1]
			for x := clipR - srcL; x < lineWidth; x++ {
				bandRow[x] = v
			}
		}
	}

	rsbRow := int(stripe.TileStripe) * restorationCtxVert
	if stripe.CopyAbove {
		for i := -restorationBorder; i < 0; i++ {
			bandRow := band[(restorationBorder+i)*bandStride:][:lineWidth]
			bufRow := rsbRow + maxInt(i+restorationCtxVert, 0)
			src, ok := restorationBoundaryLine(boundaries.Above, boundaries.Stride, bufRow, x0, lineWidth)
			if !ok {
				return ErrInvalidPlan
			}
			copy(bandRow, src)
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
			copy(bandRow, src)
		}
	} else {
		bottom := band[(restorationBorder+height-1)*bandStride:][:lineWidth]
		for i := range restorationBorder {
			copy(band[(restorationBorder+height+i)*bandStride:][:lineWidth], bottom)
		}
	}
	return nil
}

// applyRestorationStripeUnitsInPlaceU16 filters one stripe's processing
// units from the band into layout.Dst with the wide-sample unit kernels and
// stores the visible samples back into the 16-bit surface.
func applyRestorationStripeUnitsInPlaceU16(grid RestorationPlaneGrid, record RestorationUnitRecord, stripe RestorationProcessingStripe, layout restorationInPlaceU16Layout, buffer av1frame.Plane, bitDepth uint8, scratch RestorationUnitScratch) (int, error) {
	unitCount, err := grid.ProcessingUnitCount(stripe, record.Unit.Type)
	if err != nil {
		return 0, err
	}
	bandStride := layout.BandStride
	for unitIndex := range unitCount {
		unit, ok, err := grid.ProcessingUnit(stripe, record.Unit.Type, unitIndex)
		if err != nil {
			return 0, err
		}
		if !ok {
			return 0, ErrInvalidPlan
		}
		// Same VisibleRect-only filtering as the 8-bit in-place walk.
		if unit.VisibleRect.X0 < stripe.Rect.X0 || unit.VisibleRect.Y0 != stripe.Rect.Y0 {
			return 0, ErrInvalidPlan
		}
		w := int(unit.VisibleRect.Width())
		h := int(unit.VisibleRect.Height())
		if w <= 0 || h <= 0 || w > bandStride || h*bandStride > len(layout.Dst) {
			return 0, ErrInvalidPlan
		}
		srcBlock := restorationBorder*bandStride + restorationExtraHorz + int(unit.VisibleRect.X0-stripe.Rect.X0)
		if _, err := ApplyRestorationUnit(layout.Band, bandStride, srcBlock, layout.Dst, bandStride, w, h, record.Unit, bitDepth, scratch); err != nil {
			return 0, err
		}
		x := int(unit.VisibleRect.X0)
		y := int(unit.VisibleRect.Y0)
		if (y+h-1)*buffer.Stride+2*(x+w) > len(buffer.Pix) {
			return 0, ErrInvalidPlan
		}
		for r := range h {
			src := layout.Dst[r*bandStride:][:w]
			dst := buffer.Pix[(y+r)*buffer.Stride+2*x:][:2*w]
			for i, v := range src {
				dst[2*i] = byte(v)
				dst[2*i+1] = byte(v >> 8)
			}
		}
	}
	return unitCount, nil
}

// backupRestorationRightColsU16 is backupRestorationRightColsU8 for a 16-bit
// surface.
func backupRestorationRightColsU16(pix []byte, stride int, rect RestorationUnitRect, bak []uint16) error {
	width := int(rect.Width())
	height := int(rect.Height())
	if stride <= 0 || width < restorationExtraHorz || height <= 0 || len(bak) < height*restorationExtraHorz {
		return ErrInvalidPlan
	}
	x := int(rect.X1) - restorationExtraHorz
	y0 := int(rect.Y0)
	if (y0+height-1)*stride+2*(x+restorationExtraHorz) > len(pix) {
		return ErrInvalidPlan
	}
	for y := 0; y < height; y++ {
		row := (y0 + y) * stride
		for i := range restorationExtraHorz {
			bak[y*restorationExtraHorz+i] = loadSampleU16LE(pix, row+2*(x+i))
		}
	}
	return nil
}
