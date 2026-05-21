package tile

import "github.com/thesyncim/goav1/internal/av1/parser"

const (
	miSizePixels   = 4
	scaleNumerator = 8
)

// RestorationPlaneGrid describes the restoration-unit grid for one frame plane.
type RestorationPlaneGrid struct {
	Plane uint8
	Type  parser.RestorationType

	UnitSize  uint16
	HorzUnits uint16
	VertUnits uint16

	PlaneWidth  uint32
	PlaneHeight uint32

	SubsamplingX bool
	SubsamplingY bool

	SuperResEnabled     bool
	SuperResDenominator uint8
}

// RestorationUnitRange is the rectangular range of restoration units whose
// top-left corners are decoded while processing one superblock.
type RestorationUnitRange struct {
	Col0 uint16
	Col1 uint16
	Row0 uint16
	Row1 uint16
}

// RestorationUnitRecord carries one decoded restoration unit plus its position
// in the plane restoration grid.
type RestorationUnitRecord struct {
	Index uint32
	Col   uint16
	Row   uint16
	Unit  RestorationUnit
}

func BuildRestorationPlaneGrid(params parser.RestorationParams, size parser.FrameSize, color parser.ColorConfig, plane int) (RestorationPlaneGrid, error) {
	if plane < 0 || plane > 2 || size.UpscaledWidth == 0 || size.Height == 0 {
		return RestorationPlaneGrid{}, ErrInvalidPlan
	}
	if color.MonoChrome && plane > 0 {
		return RestorationPlaneGrid{Plane: uint8(plane), Type: parser.RestorationNone}, nil
	}

	unitSize := params.UnitSizeY
	if plane > 0 {
		unitSize = params.UnitSizeUV
	}
	typ := params.Type[plane]
	planeW := roundPowerOfTwoUint32(size.UpscaledWidth, boolToShift(plane > 0 && color.SubsamplingX))
	planeH := roundPowerOfTwoUint32(size.Height, boolToShift(plane > 0 && color.SubsamplingY))
	if planeW == 0 || planeH == 0 {
		return RestorationPlaneGrid{}, ErrInvalidPlan
	}
	grid := RestorationPlaneGrid{
		Plane:               uint8(plane),
		Type:                typ,
		UnitSize:            unitSize,
		PlaneWidth:          planeW,
		PlaneHeight:         planeH,
		SubsamplingX:        plane > 0 && color.SubsamplingX,
		SubsamplingY:        plane > 0 && color.SubsamplingY,
		SuperResEnabled:     size.SuperResEnabled,
		SuperResDenominator: size.SuperResDenominator,
	}
	if typ == parser.RestorationNone {
		return grid, nil
	}
	if !validRestorationGridType(typ) {
		return RestorationPlaneGrid{}, ErrInvalidPlan
	}
	if unitSize == 0 {
		return RestorationPlaneGrid{}, ErrInvalidPlan
	}
	if size.SuperResEnabled && size.SuperResDenominator < scaleNumerator+1 {
		return RestorationPlaneGrid{}, ErrInvalidPlan
	}

	grid.HorzUnits = countRestorationUnits(uint32(unitSize), planeW)
	grid.VertUnits = countRestorationUnits(uint32(unitSize), planeH)
	return grid, nil
}

func (g RestorationPlaneGrid) UnitsInSuperblock(miCol uint32, miRow uint32, sbSizeMIB uint8) (RestorationUnitRange, bool, error) {
	if g.Type == parser.RestorationNone {
		return RestorationUnitRange{}, false, nil
	}
	if !g.valid() || sbSizeMIB == 0 {
		return RestorationUnitRange{}, false, ErrInvalidPlan
	}

	miCol1 := miCol + uint32(sbSizeMIB)
	miRow1 := miRow + uint32(sbSizeMIB)
	if miCol1 < miCol || miRow1 < miRow {
		return RestorationUnitRange{}, false, ErrInvalidPlan
	}

	miToNumX := miSizePixels >> boolToShift(g.SubsamplingX)
	miToNumY := miSizePixels >> boolToShift(g.SubsamplingY)
	denomX := int(g.UnitSize)
	denomY := int(g.UnitSize)
	if g.SuperResEnabled {
		miToNumX *= int(g.SuperResDenominator)
		denomX *= scaleNumerator
	}

	col0 := ceilDivScaled(miCol, miToNumX, denomX)
	row0 := ceilDivScaled(miRow, miToNumY, denomY)
	col1 := minUint32(ceilDivScaled(miCol1, miToNumX, denomX), uint32(g.HorzUnits))
	row1 := minUint32(ceilDivScaled(miRow1, miToNumY, denomY), uint32(g.VertUnits))
	if col0 >= col1 || row0 >= row1 {
		return RestorationUnitRange{}, false, nil
	}
	return RestorationUnitRange{
		Col0: uint16(col0),
		Col1: uint16(col1),
		Row0: uint16(row0),
		Row1: uint16(row1),
	}, true, nil
}

func (r RestorationUnitRange) Count() int {
	if r.Col1 <= r.Col0 || r.Row1 <= r.Row0 {
		return 0
	}
	return int(r.Col1-r.Col0) * int(r.Row1-r.Row0)
}

// ReadRestorationUnitsForSuperblock decodes every restoration unit whose
// top-left corner belongs to one superblock. dst is caller-owned and receives
// records in row-major restoration-unit order.
func (s *DecodeState) ReadRestorationUnitsForSuperblock(grid RestorationPlaneGrid, miCol uint32, miRow uint32, sbSizeMIB uint8, dst []RestorationUnitRecord, refs *RestorationReferences, cdfs RestorationCDFs) (int, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	unitRange, ok, err := grid.UnitsInSuperblock(miCol, miRow, sbSizeMIB)
	if err != nil || !ok {
		return 0, err
	}
	need := unitRange.Count()
	if len(dst) < need {
		return 0, ErrJobBufferTooSmall
	}

	count := 0
	for row := unitRange.Row0; row < unitRange.Row1; row++ {
		for col := unitRange.Col0; col < unitRange.Col1; col++ {
			unit, err := s.ReadRestorationUnit(grid.Type, int(grid.Plane), refs, cdfs)
			if err != nil {
				return 0, err
			}
			dst[count] = RestorationUnitRecord{
				Index: uint32(row)*uint32(grid.HorzUnits) + uint32(col),
				Col:   col,
				Row:   row,
				Unit:  unit,
			}
			count++
		}
	}
	return count, nil
}

func (g RestorationPlaneGrid) valid() bool {
	if g.Plane > 2 || g.UnitSize == 0 || g.HorzUnits == 0 || g.VertUnits == 0 {
		return false
	}
	if !validRestorationGridType(g.Type) {
		return false
	}
	return !g.SuperResEnabled || g.SuperResDenominator >= scaleNumerator+1
}

func validRestorationGridType(typ parser.RestorationType) bool {
	switch typ {
	case parser.RestorationSwitchable, parser.RestorationWiener, parser.RestorationSGRProj:
		return true
	default:
		return false
	}
}

func countRestorationUnits(unitSize uint32, planeSize uint32) uint16 {
	units := (planeSize + (unitSize >> 1)) / unitSize
	if units == 0 {
		units = 1
	}
	if units > uint32(^uint16(0)) {
		return ^uint16(0)
	}
	return uint16(units)
}

func ceilDivScaled(v uint32, multiplier int, denominator int) uint32 {
	num := uint64(v)*uint64(multiplier) + uint64(denominator-1)
	return uint32(num / uint64(denominator))
}

func roundPowerOfTwoUint32(v uint32, bits uint8) uint32 {
	if bits == 0 {
		return v
	}
	return (v + (1 << (bits - 1))) >> bits
}

func boolToShift(v bool) uint8 {
	if v {
		return 1
	}
	return 0
}

func minUint32(a uint32, b uint32) uint32 {
	if a < b {
		return a
	}
	return b
}
