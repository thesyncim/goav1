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
