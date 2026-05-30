package tile

import (
	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

// RestorationUnitScratchSize reports the caller-owned scratch required by
// ApplyRestorationUnit for one processing unit.
type RestorationUnitScratchSize struct {
	Wiener  int
	SGRProj int
}

// RestorationUnitScratch is caller-owned temporary storage. The disabled
// RESTORE_NONE path does not require scratch.
type RestorationUnitScratch struct {
	Wiener  []uint16
	SGRProj []int32
}

// RestorationUnitApplyResult describes which concrete restoration path ran.
type RestorationUnitApplyResult struct {
	Type     parser.RestorationType
	Filtered bool
}

// RestorationUnitRecordApplyResult describes the loop-restoration work applied
// for one decoded restoration unit record.
type RestorationUnitRecordApplyResult struct {
	Type            parser.RestorationType
	Stripes         uint8
	ProcessingUnits uint16
	Filtered        bool
}

// RestorationPlaneApplyResult summarizes restoration work over one plane.
type RestorationPlaneApplyResult struct {
	Records         uint32
	FilteredRecords uint32
	Stripes         uint32
	ProcessingUnits uint32
}

// RestorationFramePlane binds the caller-owned buffers and decoded records for
// one plane in libaom's frame-level loop-restoration pass.
type RestorationFramePlane struct {
	Grid       RestorationPlaneGrid
	Records    []RestorationUnitRecord
	Boundaries RestorationStripeBoundaries

	Data       []uint16
	DataStride int
	DataOrigin int

	Dst       []uint16
	DstStride int
	DstOrigin int
}

// RestorationFrameApplyResult summarizes restoration work over a one- or
// three-plane AV1 frame.
type RestorationFrameApplyResult struct {
	Planes       uint8
	PlaneResults [3]RestorationPlaneApplyResult

	Records         uint32
	FilteredRecords uint32
	Stripes         uint32
	ProcessingUnits uint32
}

// RestorationUnitRecordBoundaryScratchSize reports caller-owned scratch for a
// full striped restoration-unit apply, including temporary boundary rows.
type RestorationUnitRecordBoundaryScratchSize struct {
	Unit     RestorationUnitScratchSize
	Boundary RestorationStripeBoundaryScratchSize
}

// RestorationUnitRecordBoundaryScratch is caller-owned scratch used by
// ApplyRestorationUnitRecordWithBoundaries.
type RestorationUnitRecordBoundaryScratch struct {
	Unit     RestorationUnitScratch
	Boundary RestorationStripeBoundaryScratch
}

func RestorationUnitScratchLen(width int, height int) (RestorationUnitScratchSize, error) {
	wiener, err := av1restoration.WienerScratchLen(width, height)
	if err != nil {
		return RestorationUnitScratchSize{}, err
	}
	sgr, err := av1restoration.SelfguidedScratchLen(width, height)
	if err != nil {
		return RestorationUnitScratchSize{}, err
	}
	return RestorationUnitScratchSize{Wiener: wiener, SGRProj: sgr}, nil
}

// RestorationUnitRecordScratchLen reports the active caller-owned scratch
// required to apply one restoration unit record over all of its processing
// stripes. Disabled units return a zero size.
func RestorationUnitRecordScratchLen(grid RestorationPlaneGrid, record RestorationUnitRecord) (RestorationUnitScratchSize, error) {
	if err := validateRestorationUnitRecord(grid, record); err != nil {
		return RestorationUnitScratchSize{}, err
	}
	if record.Unit.Type == parser.RestorationNone {
		return RestorationUnitScratchSize{}, nil
	}

	var size RestorationUnitScratchSize
	for stripeIndex := 0; stripeIndex < int(record.StripeCount); stripeIndex++ {
		stripe, ok, err := grid.ProcessingStripe(record.Rect, stripeIndex)
		if err != nil {
			return RestorationUnitScratchSize{}, err
		}
		if !ok {
			return RestorationUnitScratchSize{}, ErrInvalidPlan
		}
		unitCount, err := grid.ProcessingUnitCount(stripe, record.Unit.Type)
		if err != nil {
			return RestorationUnitScratchSize{}, err
		}
		for unitIndex := range unitCount {
			unit, ok, err := grid.ProcessingUnit(stripe, record.Unit.Type, unitIndex)
			if err != nil {
				return RestorationUnitScratchSize{}, err
			}
			if !ok {
				return RestorationUnitScratchSize{}, ErrInvalidPlan
			}
			blockSize, err := RestorationUnitScratchLen(int(unit.FilterRect.Width()), int(unit.FilterRect.Height()))
			if err != nil {
				return RestorationUnitScratchSize{}, err
			}
			switch record.Unit.Type {
			case parser.RestorationWiener:
				if blockSize.Wiener > size.Wiener {
					size.Wiener = blockSize.Wiener
				}
			case parser.RestorationSGRProj:
				if blockSize.SGRProj > size.SGRProj {
					size.SGRProj = blockSize.SGRProj
				}
			default:
				return RestorationUnitScratchSize{}, ErrInvalidPlan
			}
		}
	}
	return size, nil
}

// RestorationUnitRecordBoundaryScratchLen reports scratch for the full
// libaom-style striped unit flow: boundary setup/restore plus filtering.
// Disabled units return a zero size.
func RestorationUnitRecordBoundaryScratchLen(grid RestorationPlaneGrid, record RestorationUnitRecord, optimized bool) (RestorationUnitRecordBoundaryScratchSize, error) {
	unitSize, err := RestorationUnitRecordScratchLen(grid, record)
	if err != nil {
		return RestorationUnitRecordBoundaryScratchSize{}, err
	}
	if record.Unit.Type == parser.RestorationNone {
		return RestorationUnitRecordBoundaryScratchSize{}, nil
	}

	var boundarySize RestorationStripeBoundaryScratchSize
	for stripeIndex := 0; stripeIndex < int(record.StripeCount); stripeIndex++ {
		stripe, ok, err := grid.ProcessingStripe(record.Rect, stripeIndex)
		if err != nil {
			return RestorationUnitRecordBoundaryScratchSize{}, err
		}
		if !ok {
			return RestorationUnitRecordBoundaryScratchSize{}, ErrInvalidPlan
		}
		stripeSize, err := RestorationStripeBoundaryScratchLen(stripe, optimized)
		if err != nil {
			return RestorationUnitRecordBoundaryScratchSize{}, err
		}
		if stripeSize.Above > boundarySize.Above {
			boundarySize.Above = stripeSize.Above
		}
		if stripeSize.Below > boundarySize.Below {
			boundarySize.Below = stripeSize.Below
		}
	}
	return RestorationUnitRecordBoundaryScratchSize{Unit: unitSize, Boundary: boundarySize}, nil
}

// RestorationPlaneApplyScratchLen reports scratch for applying all decoded
// restoration records in one plane. records must be complete and row-major.
func RestorationPlaneApplyScratchLen(grid RestorationPlaneGrid, records []RestorationUnitRecord, optimized bool) (RestorationUnitRecordBoundaryScratchSize, error) {
	if err := validateRestorationPlaneRecords(grid, records); err != nil {
		return RestorationUnitRecordBoundaryScratchSize{}, err
	}
	var size RestorationUnitRecordBoundaryScratchSize
	for i := range records {
		recordSize, err := RestorationUnitRecordBoundaryScratchLen(grid, records[i], optimized)
		if err != nil {
			return RestorationUnitRecordBoundaryScratchSize{}, err
		}
		if recordSize.Unit.Wiener > size.Unit.Wiener {
			size.Unit.Wiener = recordSize.Unit.Wiener
		}
		if recordSize.Unit.SGRProj > size.Unit.SGRProj {
			size.Unit.SGRProj = recordSize.Unit.SGRProj
		}
		if recordSize.Boundary.Above > size.Boundary.Above {
			size.Boundary.Above = recordSize.Boundary.Above
		}
		if recordSize.Boundary.Below > size.Boundary.Below {
			size.Boundary.Below = recordSize.Boundary.Below
		}
	}
	return size, nil
}

// RestorationFramePlaneScratchLen reports scratch for
// ApplyRestorationFramePlane. Disabled restoration planes do not need scratch.
func RestorationFramePlaneScratchLen(grid RestorationPlaneGrid, records []RestorationUnitRecord, optimized bool) (RestorationUnitRecordBoundaryScratchSize, error) {
	if grid.Type == parser.RestorationNone {
		if grid.Plane > 2 {
			return RestorationUnitRecordBoundaryScratchSize{}, ErrInvalidPlan
		}
		return RestorationUnitRecordBoundaryScratchSize{}, nil
	}
	return RestorationPlaneApplyScratchLen(grid, records, optimized)
}

// RestorationFrameScratchLen reports the max scratch required to restore a
// whole one- or three-plane frame with ApplyRestorationFrame.
func RestorationFrameScratchLen(planes []RestorationFramePlane, optimized bool) (RestorationUnitRecordBoundaryScratchSize, error) {
	if err := validateRestorationFramePlanes(planes); err != nil {
		return RestorationUnitRecordBoundaryScratchSize{}, err
	}
	var size RestorationUnitRecordBoundaryScratchSize
	for i := range planes {
		planeSize, err := RestorationFramePlaneScratchLen(planes[i].Grid, planes[i].Records, optimized)
		if err != nil {
			return RestorationUnitRecordBoundaryScratchSize{}, err
		}
		if planeSize.Unit.Wiener > size.Unit.Wiener {
			size.Unit.Wiener = planeSize.Unit.Wiener
		}
		if planeSize.Unit.SGRProj > size.Unit.SGRProj {
			size.Unit.SGRProj = planeSize.Unit.SGRProj
		}
		if planeSize.Boundary.Above > size.Boundary.Above {
			size.Boundary.Above = planeSize.Boundary.Above
		}
		if planeSize.Boundary.Below > size.Boundary.Below {
			size.Boundary.Below = planeSize.Boundary.Below
		}
	}
	return size, nil
}

// ApplyRestorationUnit dispatches one decoded restoration unit to the matching
// libaom-ported primitive. srcOrigin identifies the top-left pixel of the unit;
// filtered units require the primitive-specific border around that origin.
func ApplyRestorationUnit(src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, width int, height int, unit RestorationUnit, bitDepth uint8, scratch RestorationUnitScratch) (RestorationUnitApplyResult, error) {
	switch unit.Type {
	case parser.RestorationNone:
		if err := copyRestorationUnit(src, srcStride, srcOrigin, dst, dstStride, width, height, bitDepth); err != nil {
			return RestorationUnitApplyResult{}, err
		}
		return RestorationUnitApplyResult{Type: parser.RestorationNone}, nil
	case parser.RestorationWiener:
		wienerDebugPreApply(src, srcStride, srcOrigin, width, height, unit.Wiener)
		err := av1restoration.ApplyWienerRestoration(src, srcStride, srcOrigin, dst, dstStride, width, height, unit.Wiener, bitDepth, scratch.Wiener)
		if err != nil {
			return RestorationUnitApplyResult{}, err
		}
		wienerDebugPostApply(dst, dstStride, width, height)
		return RestorationUnitApplyResult{Type: parser.RestorationWiener, Filtered: true}, nil
	case parser.RestorationSGRProj:
		err := av1restoration.ApplySelfguidedRestoration(src, srcStride, srcOrigin, dst, dstStride, width, height, int(unit.SGRProj.ParamsIndex), unit.SGRProj.XQD, bitDepth, scratch.SGRProj)
		if err != nil {
			return RestorationUnitApplyResult{}, err
		}
		return RestorationUnitApplyResult{Type: parser.RestorationSGRProj, Filtered: true}, nil
	default:
		return RestorationUnitApplyResult{}, ErrInvalidPlan
	}
}

// ApplyRestorationUnitRecord applies one decoded restoration unit record. The
// origins identify plane coordinate (0,0) inside caller-owned sample buffers;
// filtered paths require the caller to provide the restoration border around
// each processing unit, matching libaom's boundary-prepared buffers.
func ApplyRestorationUnitRecord(grid RestorationPlaneGrid, record RestorationUnitRecord, src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch RestorationUnitScratch) (RestorationUnitRecordApplyResult, error) {
	if err := validateRestorationUnitRecord(grid, record); err != nil {
		return RestorationUnitRecordApplyResult{}, err
	}
	if record.Unit.Type == parser.RestorationNone {
		srcBlock, ok := restorationPlaneOffset(srcOrigin, srcStride, record.Rect.X0, record.Rect.Y0)
		if !ok {
			return RestorationUnitRecordApplyResult{}, ErrInvalidPlan
		}
		dstBlock, ok := restorationPlaneOffset(dstOrigin, dstStride, record.Rect.X0, record.Rect.Y0)
		if !ok || dstBlock > len(dst) {
			return RestorationUnitRecordApplyResult{}, ErrInvalidPlan
		}
		result, err := ApplyRestorationUnit(src, srcStride, srcBlock, dst[dstBlock:], dstStride, int(record.Rect.Width()), int(record.Rect.Height()), record.Unit, bitDepth, scratch)
		if err != nil {
			return RestorationUnitRecordApplyResult{}, err
		}
		return RestorationUnitRecordApplyResult{Type: result.Type, Filtered: result.Filtered}, nil
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
		unitCount, err := grid.ProcessingUnitCount(stripe, record.Unit.Type)
		if err != nil {
			return RestorationUnitRecordApplyResult{}, err
		}
		result.Stripes++
		for unitIndex := range unitCount {
			unit, ok, err := grid.ProcessingUnit(stripe, record.Unit.Type, unitIndex)
			if err != nil {
				return RestorationUnitRecordApplyResult{}, err
			}
			if !ok || result.ProcessingUnits == ^uint16(0) {
				return RestorationUnitRecordApplyResult{}, ErrInvalidPlan
			}
			srcBlock, ok := restorationPlaneOffset(srcOrigin, srcStride, unit.FilterRect.X0, unit.FilterRect.Y0)
			if !ok {
				return RestorationUnitRecordApplyResult{}, ErrInvalidPlan
			}
			dstBlock, ok := restorationPlaneOffset(dstOrigin, dstStride, unit.FilterRect.X0, unit.FilterRect.Y0)
			if !ok || dstBlock > len(dst) {
				return RestorationUnitRecordApplyResult{}, ErrInvalidPlan
			}
			_, err = ApplyRestorationUnit(src, srcStride, srcBlock, dst[dstBlock:], dstStride, int(unit.FilterRect.Width()), int(unit.FilterRect.Height()), record.Unit, bitDepth, scratch)
			if err != nil {
				return RestorationUnitRecordApplyResult{}, err
			}
			result.ProcessingUnits++
		}
	}
	return result, nil
}

// ApplyRestorationPlaneRecords applies a complete row-major set of restoration
// unit records over one plane, matching libaom's plane walk once unit syntax and
// boundary lines have already been prepared.
func ApplyRestorationPlaneRecords(grid RestorationPlaneGrid, records []RestorationUnitRecord, boundaries RestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch RestorationUnitRecordBoundaryScratch, optimized bool) (RestorationPlaneApplyResult, error) {
	if err := validateRestorationPlaneRecords(grid, records); err != nil {
		return RestorationPlaneApplyResult{}, err
	}
	var result RestorationPlaneApplyResult
	for i := range records {
		recordResult, err := ApplyRestorationUnitRecordWithBoundaries(grid, records[i], boundaries, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, bitDepth, scratch, optimized)
		if err != nil {
			return RestorationPlaneApplyResult{}, err
		}
		result.Records++
		if recordResult.Filtered {
			result.FilteredRecords++
		}
		result.Stripes += uint32(recordResult.Stripes)
		result.ProcessingUnits += uint32(recordResult.ProcessingUnits)
	}
	return result, nil
}

// ApplyRestorationFramePlane ports libaom's one-plane frame restoration flow:
// extend the input frame, filter every restoration unit into dst, then copy the
// restored visible plane back into data. data is the mutable source frame and
// dst is the caller-owned restoration output buffer, matching libaom's
// cm->rst_frame.
func ApplyRestorationFramePlane(grid RestorationPlaneGrid, records []RestorationUnitRecord, boundaries RestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch RestorationUnitRecordBoundaryScratch, optimized bool) (RestorationPlaneApplyResult, error) {
	if grid.Type == parser.RestorationNone {
		if grid.Plane > 2 {
			return RestorationPlaneApplyResult{}, ErrInvalidPlan
		}
		return RestorationPlaneApplyResult{}, nil
	}
	if err := ExtendRestorationFrame(data, dataStride, dataOrigin, int(grid.PlaneWidth), int(grid.PlaneHeight), restorationBorder, restorationBorder); err != nil {
		return RestorationPlaneApplyResult{}, err
	}
	result, err := ApplyRestorationPlaneRecords(grid, records, boundaries, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, bitDepth, scratch, optimized)
	if err != nil {
		return RestorationPlaneApplyResult{}, err
	}
	if err := copyRestoredPlane(dst, dstStride, dstOrigin, data, dataStride, dataOrigin, int(grid.PlaneWidth), int(grid.PlaneHeight)); err != nil {
		return RestorationPlaneApplyResult{}, err
	}
	return result, nil
}

// ApplyRestorationFrame ports libaom's frame-level loop-restoration
// orchestration for a one-plane monochrome frame or a three-plane color frame.
// The scratch buffer is reused across planes.
func ApplyRestorationFrame(planes []RestorationFramePlane, bitDepth uint8, scratch RestorationUnitRecordBoundaryScratch, optimized bool) (RestorationFrameApplyResult, error) {
	if err := validateRestorationFramePlanes(planes); err != nil {
		return RestorationFrameApplyResult{}, err
	}
	var result RestorationFrameApplyResult
	for i := range planes {
		planeResult, err := ApplyRestorationFramePlane(planes[i].Grid, planes[i].Records, planes[i].Boundaries, planes[i].Data, planes[i].DataStride, planes[i].DataOrigin, planes[i].Dst, planes[i].DstStride, planes[i].DstOrigin, bitDepth, scratch, optimized)
		if err != nil {
			return RestorationFrameApplyResult{}, err
		}
		result.PlaneResults[i] = planeResult
		if planes[i].Grid.Type == parser.RestorationNone {
			continue
		}
		result.Planes++
		if err := accumulateRestorationFrameResult(&result, planeResult); err != nil {
			return RestorationFrameApplyResult{}, err
		}
	}
	return result, nil
}

// ApplyRestorationUnitRecordWithBoundaries ports libaom's striped
// av1_loop_restoration_filter_unit() orchestration for one decoded restoration
// unit. data is the mutable, frame-extended source plane; its overwritten stripe
// borders are restored before return. The optimized mode follows libaom's
// optimized_lr path and does not read boundaries.
func ApplyRestorationUnitRecordWithBoundaries(grid RestorationPlaneGrid, record RestorationUnitRecord, boundaries RestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch RestorationUnitRecordBoundaryScratch, optimized bool) (RestorationUnitRecordApplyResult, error) {
	if err := validateRestorationUnitRecord(grid, record); err != nil {
		return RestorationUnitRecordApplyResult{}, err
	}
	if record.Unit.Type == parser.RestorationNone {
		return ApplyRestorationUnitRecord(grid, record, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, bitDepth, scratch.Unit)
	}
	if !optimized {
		if err := validateRestorationStripeBoundaries(grid, boundaries); err != nil {
			return RestorationUnitRecordApplyResult{}, err
		}
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
		if err := SetupRestorationStripeBoundary(record.Rect, stripe, boundaries, data, dataStride, dataOrigin, scratch.Boundary, optimized); err != nil {
			return RestorationUnitRecordApplyResult{}, err
		}

		unitCount, applyErr := applyRestorationStripeUnits(grid, record, stripe, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, bitDepth, scratch.Unit)
		restoreErr := RestoreRestorationStripeBoundary(record.Rect, stripe, data, dataStride, dataOrigin, scratch.Boundary, optimized)
		if applyErr != nil {
			if restoreErr != nil {
				return RestorationUnitRecordApplyResult{}, restoreErr
			}
			return RestorationUnitRecordApplyResult{}, applyErr
		}
		if restoreErr != nil {
			return RestorationUnitRecordApplyResult{}, restoreErr
		}
		if unitCount > int(^uint16(0)) || result.ProcessingUnits > ^uint16(0)-uint16(unitCount) {
			return RestorationUnitRecordApplyResult{}, ErrInvalidPlan
		}
		result.Stripes++
		result.ProcessingUnits += uint16(unitCount)
	}
	return result, nil
}

func copyRestorationUnit(src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, width int, height int, bitDepth uint8) error {
	max, ok := restorationApplyMaxSample(bitDepth)
	if !ok ||
		!restorationSampleBlockFitsAt(len(src), srcOrigin, srcStride, width, height) ||
		!restorationSampleBlockFitsAt(len(dst), 0, dstStride, width, height) {
		return ErrInvalidPlan
	}
	for row := range height {
		srcRow := srcOrigin + row*srcStride
		dstRow := row * dstStride
		for col := range width {
			sample := src[srcRow+col]
			if sample > max {
				return ErrInvalidPlan
			}
			dst[dstRow+col] = sample
		}
	}
	return nil
}

func copyRestoredPlane(src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, dstOrigin int, width int, height int) error {
	if !restorationSampleBlockFitsAt(len(src), srcOrigin, srcStride, width, height) ||
		!restorationSampleBlockFitsAt(len(dst), dstOrigin, dstStride, width, height) {
		return ErrInvalidPlan
	}
	for row := range height {
		srcRow := srcOrigin + row*srcStride
		dstRow := dstOrigin + row*dstStride
		copy(dst[dstRow:dstRow+width], src[srcRow:srcRow+width])
	}
	return nil
}

func validateRestorationFramePlanes(planes []RestorationFramePlane) error {
	if len(planes) != 1 && len(planes) != 3 {
		return ErrInvalidPlan
	}
	for i := range planes {
		if planes[i].Grid.Plane != uint8(i) {
			return ErrInvalidPlan
		}
	}
	return nil
}

func accumulateRestorationFrameResult(dst *RestorationFrameApplyResult, src RestorationPlaneApplyResult) error {
	var ok bool
	if dst.Records, ok = checkedAddUint32(dst.Records, src.Records); !ok {
		return ErrInvalidPlan
	}
	if dst.FilteredRecords, ok = checkedAddUint32(dst.FilteredRecords, src.FilteredRecords); !ok {
		return ErrInvalidPlan
	}
	if dst.Stripes, ok = checkedAddUint32(dst.Stripes, src.Stripes); !ok {
		return ErrInvalidPlan
	}
	if dst.ProcessingUnits, ok = checkedAddUint32(dst.ProcessingUnits, src.ProcessingUnits); !ok {
		return ErrInvalidPlan
	}
	return nil
}

func applyRestorationStripeUnits(grid RestorationPlaneGrid, record RestorationUnitRecord, stripe RestorationProcessingStripe, src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch RestorationUnitScratch) (int, error) {
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
		srcBlock, ok := restorationPlaneOffset(srcOrigin, srcStride, unit.FilterRect.X0, unit.FilterRect.Y0)
		if !ok {
			return 0, ErrInvalidPlan
		}
		dstBlock, ok := restorationPlaneOffset(dstOrigin, dstStride, unit.FilterRect.X0, unit.FilterRect.Y0)
		if !ok || dstBlock > len(dst) {
			return 0, ErrInvalidPlan
		}
		if _, err := ApplyRestorationUnit(src, srcStride, srcBlock, dst[dstBlock:], dstStride, int(unit.FilterRect.Width()), int(unit.FilterRect.Height()), record.Unit, bitDepth, scratch); err != nil {
			return 0, err
		}
	}
	return unitCount, nil
}

func restorationApplyMaxSample(bitDepth uint8) (uint16, bool) {
	switch bitDepth {
	case 8, 10, 12:
		return uint16((1 << bitDepth) - 1), true
	default:
		return 0, false
	}
}

func validateRestorationUnitRecord(grid RestorationPlaneGrid, record RestorationUnitRecord) error {
	if !grid.validGeometry() ||
		record.Col >= grid.HorzUnits ||
		record.Row >= grid.VertUnits ||
		record.Index != uint32(record.Row)*uint32(grid.HorzUnits)+uint32(record.Col) ||
		!restorationUnitTypeAllowed(grid.Type, record.Unit.Type) {
		return ErrInvalidPlan
	}
	rect, err := grid.UnitRect(record.Col, record.Row)
	if err != nil {
		return err
	}
	if record.Rect != rect {
		return ErrInvalidPlan
	}
	stripeCount, err := grid.ProcessingStripeCount(record.Rect)
	if err != nil || stripeCount > int(^uint8(0)) {
		if err != nil {
			return err
		}
		return ErrInvalidPlan
	}
	if record.StripeCount != uint8(stripeCount) {
		return ErrInvalidPlan
	}
	return nil
}

func checkedAddUint32(a uint32, b uint32) (uint32, bool) {
	c := a + b
	return c, c >= a
}

func validateRestorationPlaneRecords(grid RestorationPlaneGrid, records []RestorationUnitRecord) error {
	if !grid.validGeometry() {
		return ErrInvalidPlan
	}
	need, err := restorationPlaneRecordCount(grid)
	if err != nil {
		return err
	}
	if len(records) != need {
		return ErrInvalidPlan
	}
	for i := range records {
		if records[i].Index != uint32(i) {
			return ErrInvalidPlan
		}
		if err := validateRestorationUnitRecord(grid, records[i]); err != nil {
			return err
		}
	}
	return nil
}

func restorationPlaneRecordCount(grid RestorationPlaneGrid) (int, error) {
	if !grid.validGeometry() {
		return 0, ErrInvalidPlan
	}
	maxInt := uint64(^uint(0) >> 1)
	n := uint64(grid.HorzUnits) * uint64(grid.VertUnits)
	if n == 0 || n > maxInt {
		return 0, ErrInvalidPlan
	}
	return int(n), nil
}

func restorationUnitTypeAllowed(frameType parser.RestorationType, unitType parser.RestorationType) bool {
	switch frameType {
	case parser.RestorationSwitchable:
		return unitType == parser.RestorationNone ||
			unitType == parser.RestorationWiener ||
			unitType == parser.RestorationSGRProj
	case parser.RestorationWiener:
		return unitType == parser.RestorationNone || unitType == parser.RestorationWiener
	case parser.RestorationSGRProj:
		return unitType == parser.RestorationNone || unitType == parser.RestorationSGRProj
	default:
		return false
	}
}

func restorationPlaneOffset(origin int, stride int, x uint32, y uint32) (int, bool) {
	if origin < 0 || stride <= 0 {
		return 0, false
	}
	maxInt := uint64(^uint(0) >> 1)
	offset := uint64(origin) + uint64(y)*uint64(stride) + uint64(x)
	if offset > maxInt {
		return 0, false
	}
	return int(offset), true
}

func restorationSampleBlockFitsAt(length int, origin int, stride int, width int, height int) bool {
	if origin < 0 || stride < width || width <= 0 || height <= 0 {
		return false
	}
	lastRow := origin + (height-1)*stride
	if lastRow < origin {
		return false
	}
	end := lastRow + width
	return end >= lastRow && end <= length
}
