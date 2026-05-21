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
		for unitIndex := 0; unitIndex < unitCount; unitIndex++ {
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
		err := av1restoration.ApplyWienerRestoration(src, srcStride, srcOrigin, dst, dstStride, width, height, unit.Wiener, bitDepth, scratch.Wiener)
		if err != nil {
			return RestorationUnitApplyResult{}, err
		}
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
		for unitIndex := 0; unitIndex < unitCount; unitIndex++ {
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

func copyRestorationUnit(src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, width int, height int, bitDepth uint8) error {
	max, ok := restorationApplyMaxSample(bitDepth)
	if !ok ||
		!restorationSampleBlockFitsAt(len(src), srcOrigin, srcStride, width, height) ||
		!restorationSampleBlockFitsAt(len(dst), 0, dstStride, width, height) {
		return ErrInvalidPlan
	}
	for row := 0; row < height; row++ {
		srcRow := srcOrigin + row*srcStride
		dstRow := row * dstStride
		for col := 0; col < width; col++ {
			sample := src[srcRow+col]
			if sample > max {
				return ErrInvalidPlan
			}
			dst[dstRow+col] = sample
		}
	}
	return nil
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
