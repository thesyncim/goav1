package goav1

import internaltile "github.com/thesyncim/goav1/internal/av1/tile"

// BuildTileRestorationPlaneGrid returns the per-plane restoration grid (unit
// count, unit size, offsets) for the given (params, size, color, plane).
func BuildTileRestorationPlaneGrid(params RestorationParams, size FrameSize, color ColorConfig, plane int) (TileRestorationPlaneGrid, error) {
	return internaltile.BuildRestorationPlaneGrid(params, size, color, plane)
}

// BuildTileRestorationFramePlan returns the frame-level restoration plan
// covering all three planes for (params, size, color).
func BuildTileRestorationFramePlan(params RestorationParams, size FrameSize, color ColorConfig) (TileRestorationFramePlan, error) {
	return internaltile.BuildRestorationFramePlan(params, size, color)
}

// BindTileRestorationFrameRecordBuffers carves the caller-owned backing slice
// into three per-plane TileRestorationUnitRecord views sized by plan.
func BindTileRestorationFrameRecordBuffers(plan TileRestorationFramePlan, backing []TileRestorationUnitRecord) ([3][]TileRestorationUnitRecord, error) {
	return internaltile.BindRestorationFrameRecordBuffers(plan, backing)
}

// BindTileRestorationFrameBoundaryBuffers carves the caller-owned above/below
// uint16 backing slices into three per-plane stripe-boundary buffers sized
// by plan.
func BindTileRestorationFrameBoundaryBuffers(plan TileRestorationFramePlan, above []uint16, below []uint16) ([3]TileRestorationStripeBoundaries, error) {
	return internaltile.BindRestorationFrameBoundaryBuffers(plan, above, below)
}

// ResetTileRestorationPlaneRecords clears every unit record in dst to the
// "no restoration" sentinel for the supplied grid.
func ResetTileRestorationPlaneRecords(grid TileRestorationPlaneGrid, dst []TileRestorationUnitRecord) error {
	return internaltile.ResetRestorationPlaneRecords(grid, dst)
}

// StoreTileRestorationUnitRecords copies records into dst at the positions
// implied by grid (validating geometry).
func StoreTileRestorationUnitRecords(grid TileRestorationPlaneGrid, dst []TileRestorationUnitRecord, records []TileRestorationUnitRecord) error {
	return internaltile.StoreRestorationUnitRecords(grid, dst, records)
}

// TileRestorationStripeBoundaryBufferLen reports the above/below sample
// counts required to back the stripe boundary buffers for grid.
func TileRestorationStripeBoundaryBufferLen(grid TileRestorationPlaneGrid) (TileRestorationStripeBoundaryBufferSize, error) {
	return internaltile.RestorationStripeBoundaryBufferLen(grid)
}

// TileRestorationStripeBoundaryScratchLen reports the scratch sample counts
// required by SetupTileRestorationStripeBoundary for one stripe, optimized
// for the SIMD-friendly layout when optimized is true.
func TileRestorationStripeBoundaryScratchLen(stripe TileRestorationProcessingStripe, optimized bool) (TileRestorationStripeBoundaryScratchSize, error) {
	return internaltile.RestorationStripeBoundaryScratchLen(stripe, optimized)
}

// ExtendTileRestorationFrame replicates the border samples of a restoration
// input plane out to (borderHorz, borderVert) so the unit application
// kernels can read across the frame edges.
func ExtendTileRestorationFrame(data []uint16, stride int, origin int, width int, height int, borderHorz int, borderVert int) error {
	return internaltile.ExtendRestorationFrame(data, stride, origin, width, height, borderHorz, borderVert)
}

// SaveTileRestorationBoundaryLines copies the rows that the loop-restoration
// stage will need from src into boundaries. afterCDEF selects whether the
// pre- or post-CDEF lines are being saved.
func SaveTileRestorationBoundaryLines(grid TileRestorationPlaneGrid, src []uint16, srcStride int, srcOrigin int, boundaries TileRestorationStripeBoundaries, afterCDEF bool) error {
	return internaltile.SaveRestorationBoundaryLines(grid, src, srcStride, srcOrigin, boundaries, afterCDEF)
}

// SaveTileRestorationFrameBoundaryLines is the multi-plane version of
// SaveTileRestorationBoundaryLines that processes every plane in planes.
func SaveTileRestorationFrameBoundaryLines(planes []TileRestorationFrameBoundaryPlane, afterCDEF bool) error {
	return internaltile.SaveRestorationFrameBoundaryLines(planes, afterCDEF)
}

// SetupTileRestorationStripeBoundary swaps the stripe-boundary rows in data
// with the saved boundaries for the supplied unitRect/stripe combination,
// preserving the displaced rows in scratch for a subsequent restore call.
func SetupTileRestorationStripeBoundary(unitRect TileRestorationUnitRect, stripe TileRestorationProcessingStripe, boundaries TileRestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, scratch TileRestorationStripeBoundaryScratch, optimized bool) error {
	return internaltile.SetupRestorationStripeBoundary(unitRect, stripe, boundaries, data, dataStride, dataOrigin, scratch, optimized)
}

// RestoreTileRestorationStripeBoundary undoes SetupTileRestorationStripeBoundary,
// putting the original rows back from scratch.
func RestoreTileRestorationStripeBoundary(unitRect TileRestorationUnitRect, stripe TileRestorationProcessingStripe, data []uint16, dataStride int, dataOrigin int, scratch TileRestorationStripeBoundaryScratch, optimized bool) error {
	return internaltile.RestoreRestorationStripeBoundary(unitRect, stripe, data, dataStride, dataOrigin, scratch, optimized)
}

// TileRestorationUnitScratchLen reports the scratch buffer sizes required by
// ApplyTileRestorationUnit for a unit of the given dimensions.
func TileRestorationUnitScratchLen(width int, height int) (TileRestorationUnitScratchSize, error) {
	return internaltile.RestorationUnitScratchLen(width, height)
}

// TileRestorationUnitRecordScratchLen reports the scratch sizes required by
// ApplyTileRestorationUnitRecord for the supplied (grid, record).
func TileRestorationUnitRecordScratchLen(grid TileRestorationPlaneGrid, record TileRestorationUnitRecord) (TileRestorationUnitScratchSize, error) {
	return internaltile.RestorationUnitRecordScratchLen(grid, record)
}

// TileRestorationUnitRecordBoundaryScratchLen is
// TileRestorationUnitRecordScratchLen extended to include the stripe-
// boundary scratches required to apply the unit with caller-owned
// stripe-boundary swap state.
func TileRestorationUnitRecordBoundaryScratchLen(grid TileRestorationPlaneGrid, record TileRestorationUnitRecord, optimized bool) (TileRestorationUnitRecordBoundaryScratchSize, error) {
	return internaltile.RestorationUnitRecordBoundaryScratchLen(grid, record, optimized)
}

// TileRestorationPlaneApplyScratchLen reports the worst-case scratch sizes
// required by ApplyTileRestorationPlaneRecords for the supplied records.
func TileRestorationPlaneApplyScratchLen(grid TileRestorationPlaneGrid, records []TileRestorationUnitRecord, optimized bool) (TileRestorationUnitRecordBoundaryScratchSize, error) {
	return internaltile.RestorationPlaneApplyScratchLen(grid, records, optimized)
}

// TileRestorationFramePlaneScratchLen reports the worst-case scratch sizes
// required by ApplyTileRestorationFramePlane for one plane's records.
func TileRestorationFramePlaneScratchLen(grid TileRestorationPlaneGrid, records []TileRestorationUnitRecord, optimized bool) (TileRestorationUnitRecordBoundaryScratchSize, error) {
	return internaltile.RestorationFramePlaneScratchLen(grid, records, optimized)
}

// TileRestorationFrameScratchLen reports the worst-case scratch sizes
// required by ApplyTileRestorationFrame across every plane in planes.
func TileRestorationFrameScratchLen(planes []TileRestorationFramePlane, optimized bool) (TileRestorationUnitRecordBoundaryScratchSize, error) {
	return internaltile.RestorationFrameScratchLen(planes, optimized)
}

// ApplyTileRestorationUnit applies one restoration unit (Wiener or SGR) to
// the (width, height) input region at srcOrigin, writing into dst.
func ApplyTileRestorationUnit(src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, width int, height int, unit TileRestorationUnit, bitDepth uint8, scratch TileRestorationUnitScratch) (TileRestorationUnitApplyResult, error) {
	return internaltile.ApplyRestorationUnit(src, srcStride, srcOrigin, dst, dstStride, width, height, unit, bitDepth, scratch)
}

// ApplyTileRestorationUnitRecord is ApplyTileRestorationUnit driven from a
// stored record (rect + unit) inside grid.
func ApplyTileRestorationUnitRecord(grid TileRestorationPlaneGrid, record TileRestorationUnitRecord, src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch TileRestorationUnitScratch) (TileRestorationUnitRecordApplyResult, error) {
	return internaltile.ApplyRestorationUnitRecord(grid, record, src, srcStride, srcOrigin, dst, dstStride, dstOrigin, bitDepth, scratch)
}

// ApplyTileRestorationPlaneRecords applies every record in records to data,
// writing the restored samples into dst. boundaries provides the saved
// stripe-boundary rows for cross-stripe filtering.
func ApplyTileRestorationPlaneRecords(grid TileRestorationPlaneGrid, records []TileRestorationUnitRecord, boundaries TileRestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch TileRestorationUnitRecordBoundaryScratch, optimized bool) (TileRestorationPlaneApplyResult, error) {
	return internaltile.ApplyRestorationPlaneRecords(grid, records, boundaries, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, bitDepth, scratch, optimized)
}

// ApplyTileRestorationFramePlane is ApplyTileRestorationPlaneRecords driven
// from a TileRestorationFramePlane (records + grid + data buffer).
func ApplyTileRestorationFramePlane(grid TileRestorationPlaneGrid, records []TileRestorationUnitRecord, boundaries TileRestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch TileRestorationUnitRecordBoundaryScratch, optimized bool) (TileRestorationPlaneApplyResult, error) {
	return internaltile.ApplyRestorationFramePlane(grid, records, boundaries, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, bitDepth, scratch, optimized)
}

// ApplyTileRestorationFrame applies the restoration pipeline across every
// plane in planes.
func ApplyTileRestorationFrame(planes []TileRestorationFramePlane, bitDepth uint8, scratch TileRestorationUnitRecordBoundaryScratch, optimized bool) (TileRestorationFrameApplyResult, error) {
	return internaltile.ApplyRestorationFrame(planes, bitDepth, scratch, optimized)
}

// ApplyTileRestorationUnitRecordWithBoundaries is ApplyTileRestorationUnitRecord
// that also performs the stripe-boundary swap/restore using boundaries and
// scratch.
func ApplyTileRestorationUnitRecordWithBoundaries(grid TileRestorationPlaneGrid, record TileRestorationUnitRecord, boundaries TileRestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch TileRestorationUnitRecordBoundaryScratch, optimized bool) (TileRestorationUnitRecordApplyResult, error) {
	return internaltile.ApplyRestorationUnitRecordWithBoundaries(grid, record, boundaries, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, bitDepth, scratch, optimized)
}
