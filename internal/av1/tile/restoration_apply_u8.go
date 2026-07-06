package tile

import (
	"unsafe"

	av1frame "github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

// 8-bit-pixel loop-restoration frame walk. dav1d's 8bpc restoration
// (src/lr_apply_tmpl.c lr_sbrow + src/looprestoration_tmpl.c) filters uint8
// pixels IN PLACE on the frame: the walk is strictly top-down/left-to-right,
// the vertical halo across stripe seams comes from the saved deblock/CDEF
// boundary line buffers (dav1d's lpf lines), and the horizontal halo into an
// already-filtered left neighbour comes from a small pre-filter column backup
// taken before that neighbour was overwritten (dav1d lr_apply_tmpl.c
// backup4xU + pre_lr_border). This file ports that walk onto goav1's uint8
// kernel family in internal/av1/restoration:
//
//   - goav1's kernels read a contiguous bordered source (libaom shape), and
//     goav1 frame planes have no left/top border allocation, so each
//     processing stripe is assembled into a small bordered band buffer (the
//     role of dav1d's padded tmp in src/looprestoration_tmpl.c: interior rows
//     from the frame, halo columns from the left-neighbour backup, seam rows
//     from the boundary buffers, plane edges replicated)...
//   - ...and filter output is written DIRECTLY into the frame's 8-bit plane.
//     Filtered rects are disjoint and every cross-unit read is served by the
//     band, so in-frame writes cannot pollute later units' inputs. No
//     whole-plane snapshot, no uint16 dst scratch, no store-back pass.
//
// The libaom "optimized" boundary mode (optimized_lr, never used by the
// production decoder) reads actual pre-filter rows above/below each stripe
// seam from the surface, which top-down in-place filtering destroys; that
// mode keeps the whole-plane snapshot walk below.
//
// 10/12-bit frames keep the uint16 path in restoration_frame_bridge.go
// unchanged.

// applyRestorationFrameToFrameU8 is the 8-bit driver behind
// ApplyRestorationFrameToFrame. The caller has already validated the plan,
// scratch sizes, and sample format. Result accounting matches the uint16
// filtered walk exactly.
func applyRestorationFrameToFrameU8(plan RestorationFramePlan, frm av1frame.Frame, records [3][]RestorationUnitRecord, boundaries [3]RestorationStripeBoundaries, dataScratch []uint16, scratch RestorationUnitRecordBoundaryScratch, optimized bool, size RestorationFrameSampleScratchSize, align int) (RestorationFrameApplyResult, error) {
	var result RestorationFrameApplyResult
	dataOffset := 0
	for plane := 0; plane < int(plan.Planes); plane++ {
		grid := plan.Grids[plane]
		if grid.Plane != uint8(plane) {
			return RestorationFrameApplyResult{}, ErrInvalidPlan
		}
		if grid.Type == parser.RestorationNone {
			continue
		}
		buffer, err := restorationFramePlaneBuffer(frm, plane)
		if err != nil {
			return RestorationFrameApplyResult{}, err
		}
		dataLayout := size.Data[plane]
		var planeResult RestorationPlaneApplyResult
		switch {
		case !restorationRecordsAnyFiltered(records[plane]):
			// No record filters anything: nothing reads or writes samples
			// (mirrors applyRestorationFramePlaneToDstFiltered's all-NONE
			// branch), but the records are still validated and counted.
			if err := validateRestorationPlaneRecords(grid, records[plane]); err != nil {
				return RestorationFrameApplyResult{}, err
			}
			for i := range records[plane] {
				if err := validateRestorationUnitRecord(grid, records[plane][i]); err != nil {
					return RestorationFrameApplyResult{}, err
				}
				planeResult.Records++
			}
		case optimized:
			planeResult, err = applyRestorationPlaneSnapshotU8(grid, records[plane], boundaries[plane], buffer, dataScratch[dataOffset:dataOffset+dataLayout.Len], dataLayout.Len, scratch, optimized, align)
			if err != nil {
				return RestorationFrameApplyResult{}, err
			}
		default:
			planeResult, err = applyRestorationPlaneRecordsInPlaceU8(grid, records[plane], boundaries[plane], buffer, dataScratch[dataOffset:dataOffset+dataLayout.Len], scratch)
			if err != nil {
				return RestorationFrameApplyResult{}, err
			}
		}
		dataOffset += dataLayout.Len
		result.PlaneResults[plane] = planeResult
		result.Planes++
		if err := accumulateRestorationFrameResult(&result, planeResult); err != nil {
			return RestorationFrameApplyResult{}, err
		}
	}
	return result, nil
}

// applyRestorationPlaneSnapshotU8 is the whole-plane bordered-snapshot walk
// (libaom av1_loop_restoration_filter_frame shape): the pre-restoration read
// source is a bordered uint8 snapshot of the plane and the kernels write the
// frame directly. It remains the reference walk for the in-place differential
// tests and the production walk for the libaom optimized_lr boundary mode,
// whose stripe-seam context reads actual pre-filter surface rows that the
// in-place walk overwrites.
func applyRestorationPlaneSnapshotU8(grid RestorationPlaneGrid, records []RestorationUnitRecord, boundaries RestorationStripeBoundaries, buffer av1frame.Plane, dataSeg []uint16, dataLen int, scratch RestorationUnitRecordBoundaryScratch, optimized bool, align int) (RestorationPlaneApplyResult, error) {
	snapshot, ok := restorationByteScratchView(dataSeg, dataLen)
	if !ok {
		return RestorationPlaneApplyResult{}, ErrInvalidPlan
	}
	dataView, err := av1frame.LoadBorderedBytePlane(snapshot, buffer, RestorationFrameBorder, RestorationFrameBorder, align)
	if err != nil {
		return RestorationPlaneApplyResult{}, err
	}
	if err := extendRestorationFrameU8(dataView.Pix, dataView.Stride, dataView.Origin, int(grid.PlaneWidth), int(grid.PlaneHeight), restorationBorder, restorationBorder); err != nil {
		return RestorationPlaneApplyResult{}, err
	}
	return applyRestorationPlaneRecordsFilteredU8(grid, records, boundaries, dataView.Pix, dataView.Stride, dataView.Origin, buffer.Pix, buffer.Stride, 0, scratch, optimized)
}

// restorationByteScratchView reinterprets a segment of the uint16 sample
// scratch arena as byte storage for the uint8 plane snapshot. The snapshot
// needs n bytes and the segment holds len(seg) uint16s (2*len(seg) bytes), so
// the view always fits inside the arena's allocation; the 8-bit path simply
// uses half the arena's footprint.
func restorationByteScratchView(seg []uint16, n int) ([]uint8, bool) {
	if n <= 0 || len(seg) == 0 || n > 2*len(seg) {
		return nil, false
	}
	return unsafe.Slice((*uint8)(unsafe.Pointer(unsafe.SliceData(seg))), n), true
}

// applyRestorationPlaneRecordsFilteredU8 is the uint8 counterpart of
// applyRestorationPlaneRecordsFiltered: RESTORATION_NONE records are
// validated and counted but never touched; filtered records read the uint8
// snapshot and write the frame plane directly.
func applyRestorationPlaneRecordsFilteredU8(grid RestorationPlaneGrid, records []RestorationUnitRecord, boundaries RestorationStripeBoundaries, data []uint8, dataStride int, dataOrigin int, dst []uint8, dstStride int, dstOrigin int, scratch RestorationUnitRecordBoundaryScratch, optimized bool) (RestorationPlaneApplyResult, error) {
	if err := validateRestorationPlaneRecords(grid, records); err != nil {
		return RestorationPlaneApplyResult{}, err
	}
	var result RestorationPlaneApplyResult
	for i := range records {
		if records[i].Unit.Type == parser.RestorationNone {
			if err := validateRestorationUnitRecord(grid, records[i]); err != nil {
				return RestorationPlaneApplyResult{}, err
			}
			result.Records++
			continue
		}
		recordResult, err := applyRestorationUnitRecordWithBoundariesU8(grid, records[i], boundaries, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, scratch, optimized)
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

// applyRestorationUnitRecordWithBoundariesU8 mirrors
// ApplyRestorationUnitRecordWithBoundaries on the uint8 snapshot: the
// snapshot's overwritten stripe borders are saved, replaced with boundary
// context, filtered, and restored before return. Callers never route
// RESTORATION_NONE records here.
func applyRestorationUnitRecordWithBoundariesU8(grid RestorationPlaneGrid, record RestorationUnitRecord, boundaries RestorationStripeBoundaries, data []uint8, dataStride int, dataOrigin int, dst []uint8, dstStride int, dstOrigin int, scratch RestorationUnitRecordBoundaryScratch, optimized bool) (RestorationUnitRecordApplyResult, error) {
	if err := validateRestorationUnitRecord(grid, record); err != nil {
		return RestorationUnitRecordApplyResult{}, err
	}
	if record.Unit.Type == parser.RestorationNone {
		return RestorationUnitRecordApplyResult{}, ErrInvalidPlan
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
		if err := setupRestorationStripeBoundaryU8(record.Rect, stripe, boundaries, data, dataStride, dataOrigin, scratch.Boundary, optimized); err != nil {
			return RestorationUnitRecordApplyResult{}, err
		}

		unitCount, applyErr := applyRestorationStripeUnitsU8(grid, record, stripe, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, scratch.Unit)
		restoreErr := restoreRestorationStripeBoundaryU8(record.Rect, stripe, data, dataStride, dataOrigin, scratch.Boundary, optimized)
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

func applyRestorationStripeUnitsU8(grid RestorationPlaneGrid, record RestorationUnitRecord, stripe RestorationProcessingStripe, src []uint8, srcStride int, srcOrigin int, dst []uint8, dstStride int, dstOrigin int, scratch RestorationUnitScratch) (int, error) {
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
		// The uint16 path filters the full FilterRect into bordered scratch
		// and the store-back clips to the visible record rect. Here the
		// kernels write the frame directly, so filter only the VisibleRect:
		// FilterRect only ever exceeds it on the right (libaom rounds the
		// final Wiener block width up to 16), and the separable Wiener output
		// is per-column independent, so the visible columns' bytes are
		// identical either way while nothing spills past the plane width.
		srcBlock, ok := restorationPlaneOffset(srcOrigin, srcStride, unit.VisibleRect.X0, unit.VisibleRect.Y0)
		if !ok {
			return 0, ErrInvalidPlan
		}
		dstBlock, ok := restorationPlaneOffset(dstOrigin, dstStride, unit.VisibleRect.X0, unit.VisibleRect.Y0)
		if !ok || dstBlock > len(dst) {
			return 0, ErrInvalidPlan
		}
		if err := applyRestorationUnitU8(src, srcStride, srcBlock, dst[dstBlock:], dstStride, int(unit.VisibleRect.Width()), int(unit.VisibleRect.Height()), record.Unit, scratch); err != nil {
			return 0, err
		}
	}
	return unitCount, nil
}

// applyRestorationUnitU8 dispatches one filtered restoration unit to the
// uint8 kernel family. RESTORATION_NONE never reaches this path (the
// filtered walk skips those records), so it is rejected.
func applyRestorationUnitU8(src []uint8, srcStride int, srcOrigin int, dst []uint8, dstStride int, width int, height int, unit RestorationUnit, scratch RestorationUnitScratch) error {
	switch unit.Type {
	case parser.RestorationWiener:
		return av1restoration.ApplyWienerRestorationU8(src, srcStride, srcOrigin, dst, dstStride, width, height, unit.Wiener, scratch.Wiener)
	case parser.RestorationSGRProj:
		return av1restoration.ApplySelfguidedRestorationU8(src, srcStride, srcOrigin, dst, dstStride, width, height, int(unit.SGRProj.ParamsIndex), unit.SGRProj.XQD, scratch.SGRProj)
	default:
		return ErrInvalidPlan
	}
}
