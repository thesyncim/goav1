package goav1

import internaltile "github.com/thesyncim/goav1/internal/av1/tile"

func BuildTileRestorationPlaneGrid(params RestorationParams, size FrameSize, color ColorConfig, plane int) (TileRestorationPlaneGrid, error) {
	return internaltile.BuildRestorationPlaneGrid(params, size, color, plane)
}

func BuildTileRestorationFramePlan(params RestorationParams, size FrameSize, color ColorConfig) (TileRestorationFramePlan, error) {
	return internaltile.BuildRestorationFramePlan(params, size, color)
}

func BindTileRestorationFrameRecordBuffers(plan TileRestorationFramePlan, backing []TileRestorationUnitRecord) ([3][]TileRestorationUnitRecord, error) {
	return internaltile.BindRestorationFrameRecordBuffers(plan, backing)
}

func BindTileRestorationFrameBoundaryBuffers(plan TileRestorationFramePlan, above []uint16, below []uint16) ([3]TileRestorationStripeBoundaries, error) {
	return internaltile.BindRestorationFrameBoundaryBuffers(plan, above, below)
}

func ResetTileRestorationPlaneRecords(grid TileRestorationPlaneGrid, dst []TileRestorationUnitRecord) error {
	return internaltile.ResetRestorationPlaneRecords(grid, dst)
}

func StoreTileRestorationUnitRecords(grid TileRestorationPlaneGrid, dst []TileRestorationUnitRecord, records []TileRestorationUnitRecord) error {
	return internaltile.StoreRestorationUnitRecords(grid, dst, records)
}

func TileRestorationStripeBoundaryBufferLen(grid TileRestorationPlaneGrid) (TileRestorationStripeBoundaryBufferSize, error) {
	return internaltile.RestorationStripeBoundaryBufferLen(grid)
}

func TileRestorationStripeBoundaryScratchLen(stripe TileRestorationProcessingStripe, optimized bool) (TileRestorationStripeBoundaryScratchSize, error) {
	return internaltile.RestorationStripeBoundaryScratchLen(stripe, optimized)
}

func ExtendTileRestorationFrame(data []uint16, stride int, origin int, width int, height int, borderHorz int, borderVert int) error {
	return internaltile.ExtendRestorationFrame(data, stride, origin, width, height, borderHorz, borderVert)
}

func SaveTileRestorationBoundaryLines(grid TileRestorationPlaneGrid, src []uint16, srcStride int, srcOrigin int, boundaries TileRestorationStripeBoundaries, afterCDEF bool) error {
	return internaltile.SaveRestorationBoundaryLines(grid, src, srcStride, srcOrigin, boundaries, afterCDEF)
}

func SaveTileRestorationFrameBoundaryLines(planes []TileRestorationFrameBoundaryPlane, afterCDEF bool) error {
	return internaltile.SaveRestorationFrameBoundaryLines(planes, afterCDEF)
}

func SetupTileRestorationStripeBoundary(unitRect TileRestorationUnitRect, stripe TileRestorationProcessingStripe, boundaries TileRestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, scratch TileRestorationStripeBoundaryScratch, optimized bool) error {
	return internaltile.SetupRestorationStripeBoundary(unitRect, stripe, boundaries, data, dataStride, dataOrigin, scratch, optimized)
}

func RestoreTileRestorationStripeBoundary(unitRect TileRestorationUnitRect, stripe TileRestorationProcessingStripe, data []uint16, dataStride int, dataOrigin int, scratch TileRestorationStripeBoundaryScratch, optimized bool) error {
	return internaltile.RestoreRestorationStripeBoundary(unitRect, stripe, data, dataStride, dataOrigin, scratch, optimized)
}

func TileRestorationUnitScratchLen(width int, height int) (TileRestorationUnitScratchSize, error) {
	return internaltile.RestorationUnitScratchLen(width, height)
}

func TileRestorationUnitRecordScratchLen(grid TileRestorationPlaneGrid, record TileRestorationUnitRecord) (TileRestorationUnitScratchSize, error) {
	return internaltile.RestorationUnitRecordScratchLen(grid, record)
}

func TileRestorationUnitRecordBoundaryScratchLen(grid TileRestorationPlaneGrid, record TileRestorationUnitRecord, optimized bool) (TileRestorationUnitRecordBoundaryScratchSize, error) {
	return internaltile.RestorationUnitRecordBoundaryScratchLen(grid, record, optimized)
}

func TileRestorationPlaneApplyScratchLen(grid TileRestorationPlaneGrid, records []TileRestorationUnitRecord, optimized bool) (TileRestorationUnitRecordBoundaryScratchSize, error) {
	return internaltile.RestorationPlaneApplyScratchLen(grid, records, optimized)
}

func TileRestorationFramePlaneScratchLen(grid TileRestorationPlaneGrid, records []TileRestorationUnitRecord, optimized bool) (TileRestorationUnitRecordBoundaryScratchSize, error) {
	return internaltile.RestorationFramePlaneScratchLen(grid, records, optimized)
}

func TileRestorationFrameScratchLen(planes []TileRestorationFramePlane, optimized bool) (TileRestorationUnitRecordBoundaryScratchSize, error) {
	return internaltile.RestorationFrameScratchLen(planes, optimized)
}

func ApplyTileRestorationUnit(src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, width int, height int, unit TileRestorationUnit, bitDepth uint8, scratch TileRestorationUnitScratch) (TileRestorationUnitApplyResult, error) {
	return internaltile.ApplyRestorationUnit(src, srcStride, srcOrigin, dst, dstStride, width, height, unit, bitDepth, scratch)
}

func ApplyTileRestorationUnitRecord(grid TileRestorationPlaneGrid, record TileRestorationUnitRecord, src []uint16, srcStride int, srcOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch TileRestorationUnitScratch) (TileRestorationUnitRecordApplyResult, error) {
	return internaltile.ApplyRestorationUnitRecord(grid, record, src, srcStride, srcOrigin, dst, dstStride, dstOrigin, bitDepth, scratch)
}

func ApplyTileRestorationPlaneRecords(grid TileRestorationPlaneGrid, records []TileRestorationUnitRecord, boundaries TileRestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch TileRestorationUnitRecordBoundaryScratch, optimized bool) (TileRestorationPlaneApplyResult, error) {
	return internaltile.ApplyRestorationPlaneRecords(grid, records, boundaries, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, bitDepth, scratch, optimized)
}

func ApplyTileRestorationFramePlane(grid TileRestorationPlaneGrid, records []TileRestorationUnitRecord, boundaries TileRestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch TileRestorationUnitRecordBoundaryScratch, optimized bool) (TileRestorationPlaneApplyResult, error) {
	return internaltile.ApplyRestorationFramePlane(grid, records, boundaries, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, bitDepth, scratch, optimized)
}

func ApplyTileRestorationFrame(planes []TileRestorationFramePlane, bitDepth uint8, scratch TileRestorationUnitRecordBoundaryScratch, optimized bool) (TileRestorationFrameApplyResult, error) {
	return internaltile.ApplyRestorationFrame(planes, bitDepth, scratch, optimized)
}

func ApplyTileRestorationUnitRecordWithBoundaries(grid TileRestorationPlaneGrid, record TileRestorationUnitRecord, boundaries TileRestorationStripeBoundaries, data []uint16, dataStride int, dataOrigin int, dst []uint16, dstStride int, dstOrigin int, bitDepth uint8, scratch TileRestorationUnitRecordBoundaryScratch, optimized bool) (TileRestorationUnitRecordApplyResult, error) {
	return internaltile.ApplyRestorationUnitRecordWithBoundaries(grid, record, boundaries, data, dataStride, dataOrigin, dst, dstStride, dstOrigin, bitDepth, scratch, optimized)
}
