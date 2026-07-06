package tile

import (
	av1frame "github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// RestorationFrameBorder is libaom's AOM_RESTORATION_FRAME_BORDER, the border
// allocated around the temporary loop-restoration destination frame.
const RestorationFrameBorder = 32

// RestorationFrameSampleScratchSize reports caller-owned uint16 scratch for
// ApplyRestorationFrameToFrame's sample staging. All-none restoration frames
// report zero lengths. For normal 8-bit restoration, Data holds the small
// in-place stripe band and left-neighbour backups viewed as bytes and Dst is
// empty; optimized 8-bit and 10/12-bit restoration keep the full bordered
// sample layouts.
type RestorationFrameSampleScratchSize struct {
	Data [3]av1frame.BorderedSamplePlaneLayout
	Dst  [3]av1frame.BorderedSamplePlaneLayout

	DataLen int
	DstLen  int
}

// RestorationFrameSampleScratchLen reports the uint16 frame sample scratch
// required by ApplyRestorationFrameToFrame.
func RestorationFrameSampleScratchLen(plan RestorationFramePlan, frm av1frame.Frame, optimized bool) (RestorationFrameSampleScratchSize, error) {
	if err := validateRestorationFramePlan(plan); err != nil {
		return RestorationFrameSampleScratchSize{}, err
	}
	if restorationFramePlaneCount(frm) != int(plan.Planes) {
		return RestorationFrameSampleScratchSize{}, ErrInvalidPlan
	}
	bytesPerSample, _, err := restorationFrameSampleFormat(frm)
	if err != nil {
		return RestorationFrameSampleScratchSize{}, err
	}
	align, err := restorationFrameSampleAlign(frm, bytesPerSample)
	if err != nil {
		return RestorationFrameSampleScratchSize{}, err
	}

	var size RestorationFrameSampleScratchSize
	for plane := 0; plane < int(plan.Planes); plane++ {
		grid := plan.Grids[plane]
		if grid.Type == parser.RestorationNone {
			continue
		}
		buffer, err := restorationFramePlaneBuffer(frm, plane)
		if err != nil {
			return RestorationFrameSampleScratchSize{}, err
		}
		if buffer.Width != int(grid.PlaneWidth) || buffer.Height != int(grid.PlaneHeight) {
			return RestorationFrameSampleScratchSize{}, ErrInvalidPlan
		}
		if bytesPerSample == 1 && !optimized {
			need, err := restorationInPlaceU8SampleScratchLen(grid)
			if err != nil {
				return RestorationFrameSampleScratchSize{}, err
			}
			size.Data[plane].Len = need
			var ok bool
			if size.DataLen, ok = checkedAddInt(size.DataLen, need); !ok {
				return RestorationFrameSampleScratchSize{}, ErrInvalidPlan
			}
			continue
		}
		layout, err := av1frame.BorderedSamplePlaneLen(buffer, bytesPerSample, RestorationFrameBorder, RestorationFrameBorder, align)
		if err != nil {
			return RestorationFrameSampleScratchSize{}, err
		}
		size.Data[plane] = layout
		size.Dst[plane] = layout
		var ok bool
		if size.DataLen, ok = checkedAddInt(size.DataLen, layout.Len); !ok {
			return RestorationFrameSampleScratchSize{}, ErrInvalidPlan
		}
		if size.DstLen, ok = checkedAddInt(size.DstLen, layout.Len); !ok {
			return RestorationFrameSampleScratchSize{}, ErrInvalidPlan
		}
	}
	return size, nil
}

// ApplyRestorationFrameToFrame applies decoded restoration records to a
// byte-backed frame. Following dav1d's restoration walk
// (src/lr_apply_tmpl.c lr_sbrow(): restore = lr->type !=
// DAV1D_RESTORATION_NONE), only filtered restoration units are touched:
// RESTORATION_NONE records are never copied through the sample scratch — their
// pixels stay resident in the frame — and all-NONE planes skip the
// load/extend/store round-trip entirely.
//
// 8-bit frames stay in the pixel domain end to end (dav1d's 8bpc shape): the
// read source is a bordered uint8 snapshot of the plane (byte copies into the
// data scratch arena, no widening) and the uint8 kernels write filtered units
// directly into the frame plane, so there is no dst staging and no store-back.
// 10/12-bit frames load active planes into caller-owned bordered uint16
// scratch, filter into a bordered dst scratch, and store only filtered record
// rects back to frm.
func ApplyRestorationFrameToFrame(plan RestorationFramePlan, frm av1frame.Frame, records [3][]RestorationUnitRecord, boundaries [3]RestorationStripeBoundaries, dataScratch []uint16, dstScratch []uint16, scratch RestorationUnitRecordBoundaryScratch, optimized bool) (RestorationFrameApplyResult, error) {
	size, err := RestorationFrameSampleScratchLen(plan, frm, optimized)
	if err != nil {
		return RestorationFrameApplyResult{}, err
	}
	if len(dataScratch) < size.DataLen || len(dstScratch) < size.DstLen {
		return RestorationFrameApplyResult{}, ErrJobBufferTooSmall
	}
	if len(dataScratch) != size.DataLen || len(dstScratch) != size.DstLen {
		return RestorationFrameApplyResult{}, ErrInvalidPlan
	}
	bytesPerSample, bitDepth, err := restorationFrameSampleFormat(frm)
	if err != nil {
		return RestorationFrameApplyResult{}, err
	}
	align, err := restorationFrameSampleAlign(frm, bytesPerSample)
	if err != nil {
		return RestorationFrameApplyResult{}, err
	}
	if bytesPerSample == 1 {
		return applyRestorationFrameToFrameU8(plan, frm, records, boundaries, dataScratch, scratch, optimized, size, align)
	}
	return applyRestorationFrameToFrameWide(plan, frm, records, boundaries, dataScratch, dstScratch, scratch, optimized, size, bytesPerSample, bitDepth, align)
}

// applyRestorationFrameToFrameWide is the uint16 sample-scratch round-trip
// driver behind ApplyRestorationFrameToFrame, used for 10/12-bit frames (and
// callable on 8-bit frames by tests as the widened-path reference).
func applyRestorationFrameToFrameWide(plan RestorationFramePlan, frm av1frame.Frame, records [3][]RestorationUnitRecord, boundaries [3]RestorationStripeBoundaries, dataScratch []uint16, dstScratch []uint16, scratch RestorationUnitRecordBoundaryScratch, optimized bool, size RestorationFrameSampleScratchSize, bytesPerSample int, bitDepth uint8, align int) (RestorationFrameApplyResult, error) {
	var planes [3]RestorationFramePlane
	var dstViews [3]av1frame.BorderedSamplePlane
	var anyFiltered [3]bool
	dataOffset := 0
	dstOffset := 0
	for plane := 0; plane < int(plan.Planes); plane++ {
		grid := plan.Grids[plane]
		planes[plane] = RestorationFramePlane{
			Grid:       grid,
			Records:    records[plane],
			Boundaries: boundaries[plane],
		}
		if grid.Type == parser.RestorationNone {
			continue
		}
		anyFiltered[plane] = restorationRecordsAnyFiltered(records[plane])
		buffer, err := restorationFramePlaneBuffer(frm, plane)
		if err != nil {
			return RestorationFrameApplyResult{}, err
		}
		dataLayout := size.Data[plane]
		dstLayout := size.Dst[plane]
		var dataView av1frame.BorderedSamplePlane
		if anyFiltered[plane] {
			dataView, err = av1frame.LoadBorderedSamplePlane(dataScratch[dataOffset:dataOffset+dataLayout.Len], buffer, bytesPerSample, RestorationFrameBorder, RestorationFrameBorder, align)
		} else {
			// No record filters anything: the plane walk reads no samples,
			// so bind the scratch without the whole-plane load.
			dataView, err = av1frame.BindBorderedSamplePlane(dataScratch[dataOffset:dataOffset+dataLayout.Len], buffer, bytesPerSample, RestorationFrameBorder, RestorationFrameBorder, align)
		}
		if err != nil {
			return RestorationFrameApplyResult{}, err
		}
		dstView, err := av1frame.BindBorderedSamplePlane(dstScratch[dstOffset:dstOffset+dstLayout.Len], buffer, bytesPerSample, RestorationFrameBorder, RestorationFrameBorder, align)
		if err != nil {
			return RestorationFrameApplyResult{}, err
		}
		dstViews[plane] = dstView
		planes[plane].Data = dataView.Pix
		planes[plane].DataStride = dataView.Stride
		planes[plane].DataOrigin = dataView.Origin
		planes[plane].Dst = dstView.Pix
		planes[plane].DstStride = dstView.Stride
		planes[plane].DstOrigin = dstView.Origin
		dataOffset += dataLayout.Len
		dstOffset += dstLayout.Len
	}

	result, err := applyRestorationFrameToDstFiltered(planes[:int(plan.Planes)], bitDepth, scratch, optimized)
	if err != nil {
		return RestorationFrameApplyResult{}, err
	}
	for plane := 0; plane < int(plan.Planes); plane++ {
		if plan.Grids[plane].Type == parser.RestorationNone || !anyFiltered[plane] {
			continue
		}
		buffer, err := restorationFramePlaneBuffer(frm, plane)
		if err != nil {
			return RestorationFrameApplyResult{}, err
		}
		for i := range records[plane] {
			record := &records[plane][i]
			if record.Unit.Type == parser.RestorationNone {
				continue
			}
			rect := record.Rect
			if err := av1frame.StoreBorderedSamplePlaneRectTrusted(buffer, bytesPerSample, dstViews[plane], int(rect.X0), int(rect.Y0), int(rect.Width()), int(rect.Height())); err != nil {
				return RestorationFrameApplyResult{}, err
			}
		}
	}
	return result, nil
}

func restorationFrameSampleFormat(frm av1frame.Frame) (int, uint8, error) {
	bitDepth := frm.Format.BitDepth
	bytesPerSample := 1
	switch bitDepth {
	case 8:
	case 10, 12:
		bytesPerSample = 2
	default:
		return 0, 0, ErrInvalidPlan
	}
	if frm.Layout.BytesPerSample != 0 && frm.Layout.BytesPerSample != bytesPerSample {
		return 0, 0, ErrInvalidPlan
	}
	return bytesPerSample, bitDepth, nil
}

func restorationFrameSampleAlign(frm av1frame.Frame, bytesPerSample int) (int, error) {
	alignBytes := frm.Format.Align
	if alignBytes <= 0 {
		alignBytes = 1
	}
	if alignBytes&(alignBytes-1) != 0 {
		return 0, ErrInvalidPlan
	}
	align := alignBytes / bytesPerSample
	if align == 0 {
		align = 1
	}
	return align, nil
}

func restorationFramePlaneCount(frm av1frame.Frame) int {
	if frm.Format.MonoChrome {
		return 1
	}
	return 3
}

func restorationFramePlaneBuffer(frm av1frame.Frame, plane int) (av1frame.Plane, error) {
	switch plane {
	case 0:
		return frm.Y, nil
	case 1:
		return frm.U, nil
	case 2:
		return frm.V, nil
	default:
		return av1frame.Plane{}, ErrInvalidPlan
	}
}
