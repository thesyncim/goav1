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
	Index       uint32
	Col         uint16
	Row         uint16
	Rect        RestorationUnitRect
	StripeCount uint8
	Unit        RestorationUnit
}

// RestorationFramePlan is the caller-owned allocation plan for loop
// restoration, matching libaom's per-frame RestorationInfo setup without
// allocating unit-info or stripe-boundary buffers.
type RestorationFramePlan struct {
	Planes uint8
	Active bool

	Grids       [3]RestorationPlaneGrid
	UnitRecords [3]int
	Boundaries  [3]RestorationStripeBoundaryBufferSize
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

// BuildRestorationFramePlan ports the frame-level restoration allocation pass:
// active planes get libaom-matched unit-info counts and stripe-boundary buffer
// sizes, while all-none frames report zero caller-owned storage.
func BuildRestorationFramePlan(params parser.RestorationParams, size parser.FrameSize, color parser.ColorConfig) (RestorationFramePlan, error) {
	planes := 3
	if color.MonoChrome {
		planes = 1
	}
	var plan RestorationFramePlan
	plan.Planes = uint8(planes)
	for plane := 0; plane < planes; plane++ {
		grid, err := BuildRestorationPlaneGrid(params, size, color, plane)
		if err != nil {
			return RestorationFramePlan{}, err
		}
		plan.Grids[plane] = grid
		if grid.Type == parser.RestorationNone {
			continue
		}
		records, err := grid.UnitRecordLen()
		if err != nil {
			return RestorationFramePlan{}, err
		}
		boundaries, err := RestorationStripeBoundaryBufferLen(grid)
		if err != nil {
			return RestorationFramePlan{}, err
		}
		plan.UnitRecords[plane] = records
		plan.Boundaries[plane] = boundaries
		plan.Active = true
	}
	return plan, nil
}

// UnitRecordLen reports the number of RestorationUnitRecord slots needed for
// this plane's unit_info-equivalent storage.
func (g RestorationPlaneGrid) UnitRecordLen() (int, error) {
	if g.Type == parser.RestorationNone {
		return 0, nil
	}
	if !g.valid() {
		return 0, ErrInvalidPlan
	}
	n, ok := checkedMulInt(int(g.HorzUnits), int(g.VertUnits))
	if !ok {
		return 0, ErrInvalidPlan
	}
	return n, nil
}

// UnitRecordLen reports the total unit-info-equivalent record slots required
// by all active planes in the frame plan.
func (p RestorationFramePlan) UnitRecordLen() int {
	total := 0
	for i := 0; i < int(p.Planes) && i < len(p.UnitRecords); i++ {
		total += p.UnitRecords[i]
	}
	return total
}

// BoundaryBufferLen reports the sum of per-plane stripe-boundary lengths for
// one boundary side. Callers allocate that amount separately for Above and Below.
func (p RestorationFramePlan) BoundaryBufferLen() int {
	total := 0
	for i := 0; i < int(p.Planes) && i < len(p.Boundaries); i++ {
		total += p.Boundaries[i].Len
	}
	return total
}

// BindRestorationFrameRecordBuffers partitions caller-owned storage into the
// per-plane unit_info-equivalent record slices described by plan.
func BindRestorationFrameRecordBuffers(plan RestorationFramePlan, backing []RestorationUnitRecord) ([3][]RestorationUnitRecord, error) {
	if err := validateRestorationFramePlan(plan); err != nil {
		return [3][]RestorationUnitRecord{}, err
	}
	need := plan.UnitRecordLen()
	if len(backing) < need {
		return [3][]RestorationUnitRecord{}, ErrJobBufferTooSmall
	}
	if len(backing) != need {
		return [3][]RestorationUnitRecord{}, ErrInvalidPlan
	}
	var records [3][]RestorationUnitRecord
	offset := 0
	for plane := 0; plane < int(plan.Planes); plane++ {
		n := plan.UnitRecords[plane]
		if n == 0 {
			continue
		}
		records[plane] = backing[offset : offset+n]
		offset += n
	}
	return records, nil
}

// BindRestorationFrameBoundaryBuffers partitions caller-owned storage into
// per-plane stripe-boundary buffers. above and below are separate one-side
// backing buffers with length BoundaryBufferLen().
func BindRestorationFrameBoundaryBuffers(plan RestorationFramePlan, above []uint16, below []uint16) ([3]RestorationStripeBoundaries, error) {
	if err := validateRestorationFramePlan(plan); err != nil {
		return [3]RestorationStripeBoundaries{}, err
	}
	need := plan.BoundaryBufferLen()
	if len(above) < need || len(below) < need {
		return [3]RestorationStripeBoundaries{}, ErrJobBufferTooSmall
	}
	if len(above) != need || len(below) != need {
		return [3]RestorationStripeBoundaries{}, ErrInvalidPlan
	}
	var boundaries [3]RestorationStripeBoundaries
	offset := 0
	for plane := 0; plane < int(plan.Planes); plane++ {
		size := plan.Boundaries[plane]
		if size.Len == 0 {
			continue
		}
		boundaries[plane] = RestorationStripeBoundaries{
			Above:  above[offset : offset+size.Len],
			Below:  below[offset : offset+size.Len],
			Stride: size.Stride,
		}
		offset += size.Len
	}
	return boundaries, nil
}

// ResetRestorationPlaneRecords initializes a frame-wide unit_info-equivalent
// record buffer for one active restoration plane. Every record starts as
// RESTORE_NONE with geometry and stripe counts precomputed.
func ResetRestorationPlaneRecords(grid RestorationPlaneGrid, dst []RestorationUnitRecord) error {
	need, err := restorationPlaneRecordBufferLen(grid)
	if err != nil {
		return err
	}
	if len(dst) < need {
		return ErrJobBufferTooSmall
	}
	if len(dst) != need {
		return ErrInvalidPlan
	}
	index := 0
	for row := uint16(0); row < grid.VertUnits; row++ {
		for col := uint16(0); col < grid.HorzUnits; col++ {
			record, err := grid.defaultRestorationUnitRecord(col, row)
			if err != nil {
				return err
			}
			dst[index] = record
			index++
		}
	}
	return nil
}

// StoreRestorationUnitRecords writes decoded superblock records into their
// frame-wide unit_info-equivalent slots, matching libaom's runit_idx indexing.
func StoreRestorationUnitRecords(grid RestorationPlaneGrid, dst []RestorationUnitRecord, records []RestorationUnitRecord) error {
	need, err := restorationPlaneRecordBufferLen(grid)
	if err != nil {
		return err
	}
	if len(dst) < need {
		return ErrJobBufferTooSmall
	}
	if len(dst) != need {
		return ErrInvalidPlan
	}
	for i := range records {
		if err := validateRestorationUnitRecord(grid, records[i]); err != nil {
			return err
		}
		if int(records[i].Index) >= len(dst) {
			return ErrInvalidPlan
		}
		dst[records[i].Index] = records[i]
	}
	return nil
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
			record, err := s.readRestorationUnitRecord(grid, col, row, refs, cdfs)
			if err != nil {
				return 0, err
			}
			dst[count] = record
			count++
		}
	}
	return count, nil
}

// ReadRestorationUnitsForSuperblockInto decodes the restoration units whose
// top-left corners belong to one superblock and writes them into a frame-wide
// unit_info-equivalent buffer by record index.
func (s *DecodeState) ReadRestorationUnitsForSuperblockInto(grid RestorationPlaneGrid, miCol uint32, miRow uint32, sbSizeMIB uint8, dst []RestorationUnitRecord, refs *RestorationReferences, cdfs RestorationCDFs) (int, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	unitRange, ok, err := grid.UnitsInSuperblock(miCol, miRow, sbSizeMIB)
	if err != nil || !ok {
		return 0, err
	}
	need, err := restorationPlaneRecordBufferLen(grid)
	if err != nil {
		return 0, err
	}
	if len(dst) < need {
		return 0, ErrJobBufferTooSmall
	}
	if len(dst) != need {
		return 0, ErrInvalidPlan
	}

	count := 0
	for row := unitRange.Row0; row < unitRange.Row1; row++ {
		for col := unitRange.Col0; col < unitRange.Col1; col++ {
			record, err := s.readRestorationUnitRecord(grid, col, row, refs, cdfs)
			if err != nil {
				return 0, err
			}
			if int(record.Index) >= len(dst) {
				return 0, ErrInvalidPlan
			}
			dst[record.Index] = record
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

func (s *DecodeState) readRestorationUnitRecord(grid RestorationPlaneGrid, col uint16, row uint16, refs *RestorationReferences, cdfs RestorationCDFs) (RestorationUnitRecord, error) {
	record, err := grid.defaultRestorationUnitRecord(col, row)
	if err != nil {
		return RestorationUnitRecord{}, err
	}
	record.Unit, err = s.ReadRestorationUnit(grid.Type, int(grid.Plane), refs, cdfs)
	if err != nil {
		return RestorationUnitRecord{}, err
	}
	return record, nil
}

func (g RestorationPlaneGrid) defaultRestorationUnitRecord(col uint16, row uint16) (RestorationUnitRecord, error) {
	if !g.validGeometry() || col >= g.HorzUnits || row >= g.VertUnits {
		return RestorationUnitRecord{}, ErrInvalidPlan
	}
	rect, err := g.UnitRect(col, row)
	if err != nil {
		return RestorationUnitRecord{}, err
	}
	stripeCount, err := g.ProcessingStripeCount(rect)
	if err != nil || stripeCount > int(^uint8(0)) {
		if err != nil {
			return RestorationUnitRecord{}, err
		}
		return RestorationUnitRecord{}, ErrInvalidPlan
	}
	return RestorationUnitRecord{
		Index:       uint32(row)*uint32(g.HorzUnits) + uint32(col),
		Col:         col,
		Row:         row,
		Rect:        rect,
		StripeCount: uint8(stripeCount),
		Unit:        RestorationUnit{Type: parser.RestorationNone},
	}, nil
}

func restorationPlaneRecordBufferLen(grid RestorationPlaneGrid) (int, error) {
	if grid.Type == parser.RestorationNone {
		if grid.Plane > 2 {
			return 0, ErrInvalidPlan
		}
		return 0, nil
	}
	return grid.UnitRecordLen()
}

func validateRestorationFramePlan(plan RestorationFramePlan) error {
	if plan.Planes != 1 && plan.Planes != 3 {
		return ErrInvalidPlan
	}
	active := false
	for plane := range len(plan.Grids) {
		if plane >= int(plan.Planes) {
			if plan.Grids[plane] != (RestorationPlaneGrid{}) ||
				plan.UnitRecords[plane] != 0 ||
				plan.Boundaries[plane] != (RestorationStripeBoundaryBufferSize{}) {
				return ErrInvalidPlan
			}
			continue
		}
		grid := plan.Grids[plane]
		if grid.Type == parser.RestorationNone {
			if plan.UnitRecords[plane] != 0 || plan.Boundaries[plane] != (RestorationStripeBoundaryBufferSize{}) {
				return ErrInvalidPlan
			}
			continue
		}
		active = true
		if grid.Plane != uint8(plane) || !grid.validGeometry() {
			return ErrInvalidPlan
		}
		records, err := grid.UnitRecordLen()
		if err != nil {
			return err
		}
		if plan.UnitRecords[plane] != records {
			return ErrInvalidPlan
		}
		boundaries, err := RestorationStripeBoundaryBufferLen(grid)
		if err != nil {
			return err
		}
		if plan.Boundaries[plane] != boundaries {
			return ErrInvalidPlan
		}
	}
	if plan.Active != active {
		return ErrInvalidPlan
	}
	return nil
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

func alignPowerOfTwoInt(v int, bits uint) (int, bool) {
	if v < 0 || bits >= 30 {
		return 0, false
	}
	mask := (1 << bits) - 1
	maxInt := int(^uint(0) >> 1)
	if v > maxInt-mask {
		return 0, false
	}
	return (v + mask) &^ mask, true
}
