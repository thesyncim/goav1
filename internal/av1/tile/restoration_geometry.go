package tile

import (
	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

const restorationUnitOffset = 8

// RestorationUnitRect is the plane-space rectangle covered by one restoration
// unit after AV1's vertical restoration-unit offset is applied.
type RestorationUnitRect struct {
	X0 uint32
	Y0 uint32
	X1 uint32
	Y1 uint32
}

// RestorationProcessingStripe is one processing stripe inside a restoration
// unit, matching libaom's RESTORATION_PROC_UNIT_SIZE stripe walk.
type RestorationProcessingStripe struct {
	Rect          RestorationUnitRect
	ProcUnitWidth uint16
	TileStripe    uint16
	CopyAbove     bool
	CopyBelow     bool
}

// RestorationProcessingUnit is one horizontal processing unit inside a stripe.
// FilterRect is the primitive's exact block. For Wiener, libaom rounds the
// final block width up to a 16-sample boundary, so FilterRect may extend past
// VisibleRect at the right edge.
type RestorationProcessingUnit struct {
	FilterRect  RestorationUnitRect
	VisibleRect RestorationUnitRect
}

func (r RestorationUnitRect) Width() uint32 {
	if r.X1 <= r.X0 {
		return 0
	}
	return r.X1 - r.X0
}

func (r RestorationUnitRect) Height() uint32 {
	if r.Y1 <= r.Y0 {
		return 0
	}
	return r.Y1 - r.Y0
}

// UnitRect returns the plane-space limits for one restoration unit. This ports
// libaom's foreach_rest_unit_in_tile() geometry for the whole-frame tile case.
func (g RestorationPlaneGrid) UnitRect(col uint16, row uint16) (RestorationUnitRect, error) {
	if !g.validGeometry() || col >= g.HorzUnits || row >= g.VertUnits {
		return RestorationUnitRect{}, ErrInvalidPlan
	}

	unitSize := uint32(g.UnitSize)
	x0 := uint32(col) * unitSize
	x1 := x0 + unitSize
	if col+1 == g.HorzUnits {
		x1 = g.PlaneWidth
	}
	y0 := uint32(row) * unitSize
	y1 := y0 + unitSize
	if row+1 == g.VertUnits {
		y1 = g.PlaneHeight
	}
	if x1 > g.PlaneWidth || y1 > g.PlaneHeight {
		return RestorationUnitRect{}, ErrInvalidPlan
	}

	offset := uint32(restorationUnitOffset >> boolToShift(g.SubsamplingY))
	if y0 > 0 {
		if y0 <= offset {
			return RestorationUnitRect{}, ErrInvalidPlan
		}
		y0 -= offset
	}
	if y1 < g.PlaneHeight {
		if y1 <= offset {
			return RestorationUnitRect{}, ErrInvalidPlan
		}
		y1 -= offset
	}

	rect := RestorationUnitRect{X0: x0, Y0: y0, X1: x1, Y1: y1}
	if !rect.valid() {
		return RestorationUnitRect{}, ErrInvalidPlan
	}
	return rect, nil
}

// ProcessingStripe returns the index'th 64-pixel processing stripe inside rect.
// The boolean is false when index is past the last stripe.
func (g RestorationPlaneGrid) ProcessingStripe(rect RestorationUnitRect, index int) (RestorationProcessingStripe, bool, error) {
	if index < 0 || !g.validGeometry() || !rect.valid() ||
		rect.X1 > g.PlaneWidth || rect.Y1 > g.PlaneHeight {
		return RestorationProcessingStripe{}, false, ErrInvalidPlan
	}

	fullStripeHeight := uint32(av1restoration.ProcUnitSize >> boolToShift(g.SubsamplingY))
	procUnitWidth := uint32(av1restoration.ProcUnitSize >> boolToShift(g.SubsamplingX))
	runitOffset := uint32(restorationUnitOffset >> boolToShift(g.SubsamplingY))
	if fullStripeHeight == 0 || procUnitWidth == 0 {
		return RestorationProcessingStripe{}, false, ErrInvalidPlan
	}

	y := rect.Y0
	for stripeIndex := 0; y < rect.Y1; stripeIndex++ {
		tileStripe := (y + runitOffset) / fullStripeHeight
		nominalHeight := fullStripeHeight
		if y == 0 {
			nominalHeight -= runitOffset
		}
		if nominalHeight == 0 {
			return RestorationProcessingStripe{}, false, ErrInvalidPlan
		}
		height := minUint32(nominalHeight, rect.Y1-y)
		if stripeIndex == index {
			if tileStripe > uint32(^uint16(0)) || procUnitWidth > uint32(^uint16(0)) {
				return RestorationProcessingStripe{}, false, ErrInvalidPlan
			}
			return RestorationProcessingStripe{
				Rect: RestorationUnitRect{
					X0: rect.X0,
					Y0: y,
					X1: rect.X1,
					Y1: y + height,
				},
				ProcUnitWidth: uint16(procUnitWidth),
				TileStripe:    uint16(tileStripe),
				CopyAbove:     y != 0,
				CopyBelow:     y+nominalHeight < g.PlaneHeight,
			}, true, nil
		}
		y += height
	}
	return RestorationProcessingStripe{}, false, nil
}

func (g RestorationPlaneGrid) ProcessingStripeCount(rect RestorationUnitRect) (int, error) {
	if !g.validGeometry() || !rect.valid() ||
		rect.X1 > g.PlaneWidth || rect.Y1 > g.PlaneHeight {
		return 0, ErrInvalidPlan
	}
	count := 0
	for {
		stripe, ok, err := g.ProcessingStripe(rect, count)
		if err != nil || !ok {
			return count, err
		}
		if !stripe.Rect.valid() {
			return 0, ErrInvalidPlan
		}
		count++
	}
}

// ProcessingUnit returns the index'th horizontal primitive block inside stripe,
// using libaom's per-filter width rules.
func (g RestorationPlaneGrid) ProcessingUnit(stripe RestorationProcessingStripe, typ parser.RestorationType, index int) (RestorationProcessingUnit, bool, error) {
	if index < 0 || stripe.ProcUnitWidth == 0 || !stripe.Rect.valid() ||
		stripe.Rect.X1 > g.PlaneWidth || stripe.Rect.Y1 > g.PlaneHeight {
		return RestorationProcessingUnit{}, false, ErrInvalidPlan
	}
	procWidth := uint32(stripe.ProcUnitWidth)
	if procWidth == 0 {
		return RestorationProcessingUnit{}, false, ErrInvalidPlan
	}
	x := stripe.Rect.X0
	for unitIndex := 0; x < stripe.Rect.X1; unitIndex++ {
		remaining := stripe.Rect.X1 - x
		filterWidth, err := restorationProcessingUnitWidth(typ, remaining, procWidth)
		if err != nil {
			return RestorationProcessingUnit{}, false, err
		}
		visibleWidth := minUint32(filterWidth, remaining)
		if unitIndex == index {
			return RestorationProcessingUnit{
				FilterRect: RestorationUnitRect{
					X0: x,
					Y0: stripe.Rect.Y0,
					X1: x + filterWidth,
					Y1: stripe.Rect.Y1,
				},
				VisibleRect: RestorationUnitRect{
					X0: x,
					Y0: stripe.Rect.Y0,
					X1: x + visibleWidth,
					Y1: stripe.Rect.Y1,
				},
			}, true, nil
		}
		x += procWidth
	}
	return RestorationProcessingUnit{}, false, nil
}

func (g RestorationPlaneGrid) ProcessingUnitCount(stripe RestorationProcessingStripe, typ parser.RestorationType) (int, error) {
	if stripe.ProcUnitWidth == 0 || !stripe.Rect.valid() ||
		stripe.Rect.X1 > g.PlaneWidth || stripe.Rect.Y1 > g.PlaneHeight {
		return 0, ErrInvalidPlan
	}
	count := 0
	for {
		unit, ok, err := g.ProcessingUnit(stripe, typ, count)
		if err != nil || !ok {
			return count, err
		}
		if !unit.FilterRect.valid() || !unit.VisibleRect.valid() ||
			unit.VisibleRect.X0 != unit.FilterRect.X0 ||
			unit.VisibleRect.Y0 != unit.FilterRect.Y0 ||
			unit.VisibleRect.Y1 != unit.FilterRect.Y1 ||
			unit.VisibleRect.X1 > stripe.Rect.X1 ||
			unit.FilterRect.X1 < unit.VisibleRect.X1 {
			return 0, ErrInvalidPlan
		}
		count++
	}
}

func restorationProcessingUnitWidth(typ parser.RestorationType, remaining uint32, procWidth uint32) (uint32, error) {
	if remaining == 0 || procWidth == 0 {
		return 0, ErrInvalidPlan
	}
	switch typ {
	case parser.RestorationWiener:
		return minUint32(procWidth, roundUpUint32(remaining, 16)), nil
	case parser.RestorationSGRProj:
		return minUint32(procWidth, remaining), nil
	default:
		return 0, ErrInvalidPlan
	}
}

func (g RestorationPlaneGrid) validGeometry() bool {
	return g.valid() && g.PlaneWidth > 0 && g.PlaneHeight > 0
}

func (r RestorationUnitRect) valid() bool {
	return r.X0 < r.X1 && r.Y0 < r.Y1
}

func roundUpUint32(v uint32, align uint32) uint32 {
	if align == 0 {
		return v
	}
	return (v + align - 1) &^ (align - 1)
}
