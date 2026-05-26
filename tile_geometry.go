package goav1

import internaltile "github.com/thesyncim/goav1/internal/av1/tile"

// TileRootBlockLevel returns the root block partitioning level for a sequence
// that uses 128x128 (true) or 64x64 (false) superblocks.
func TileRootBlockLevel(use128x128Superblock bool) TileBlockLevel {
	return internaltile.RootBlockLevel(use128x128Superblock)
}

// TileBlockLevelHalfSize4x4 returns the half-block side, in units of 4x4
// samples, for the supplied block level.
func TileBlockLevelHalfSize4x4(level TileBlockLevel) uint8 {
	return level.HalfSize4x4()
}

// TileBlockLevelSize4x4 returns the full-block side, in units of 4x4 samples,
// for the supplied block level.
func TileBlockLevelSize4x4(level TileBlockLevel) uint8 {
	return level.Size4x4()
}

// TileBlockDimensionsOf returns the (width, height, valid) tuple for size.
func TileBlockDimensionsOf(size TileBlockSize) (TileBlockDimensions, bool) {
	return size.Dimensions()
}

// TilePartitionValidForLevel reports whether partition is allowed by the AV1
// partition rules at level.
func TilePartitionValidForLevel(partition TilePartition, level TileBlockLevel) bool {
	return partition.ValidForLevel(level)
}

// TilePartitionBlockSizes returns the two child block sizes produced by
// partition at level (the second size is only meaningful for asymmetric
// partitions).
func TilePartitionBlockSizes(partition TilePartition, level TileBlockLevel) (TileBlockSize, TileBlockSize, bool) {
	return partition.BlockSizes(level)
}

// TileTransformDimensionsOf returns the (width, height, valid) tuple for the
// supplied transform size.
func TileTransformDimensionsOf(size TileTransformSize) (TileTransformDimensions, bool) {
	return size.Dimensions()
}

// TileTransformSizePixels converts a TileTransformSize into the corresponding
// canonical TransformSize used by the DSP layer.
func TileTransformSizePixels(size TileTransformSize) (TransformSize, error) {
	return size.TransformSize()
}

// MaxTileTransformSize returns the largest transform size allowed in block at
// (color, plane).
func MaxTileTransformSize(block TileBlockSize, color ColorConfig, plane int) (TileTransformSize, error) {
	return internaltile.MaxTransformSize(block, color, plane)
}

// TileTransformSizeSquare returns the square transform size with the same
// area as size, or an error if no such size exists.
func TileTransformSizeSquare(size TileTransformSize) (TileTransformSize, error) {
	return internaltile.TransformSizeSquare(size)
}

// TileTransformSizeSquareUp returns the next-larger square transform size
// covering size.
func TileTransformSizeSquareUp(size TileTransformSize) (TileTransformSize, error) {
	return internaltile.TransformSizeSquareUp(size)
}

// TileExtTXSetTypeFor returns the AV1 ext-tx set type implied by the
// (size, inter, reduced) tuple.
func TileExtTXSetTypeFor(size TileTransformSize, inter bool, reduced bool) (TileExtTXSetType, error) {
	return internaltile.ExtTXSetTypeFor(size, inter, reduced)
}

// TileExtTXSetIndex returns the AV1 ext-tx set index for the (size, inter,
// reduced) tuple used by the entropy tables.
func TileExtTXSetIndex(size TileTransformSize, inter bool, reduced bool) (int, error) {
	return internaltile.ExtTXSetIndex(size, inter, reduced)
}

// TileExtTXTypeCount returns the number of transform types in the ext-tx set
// identified by set.
func TileExtTXTypeCount(set TileExtTXSetType) (int, error) {
	return internaltile.ExtTXTypeCount(set)
}

// TileExtTXTypeFromSymbol decodes the ext-tx symbol into a TransformType for
// the given set.
func TileExtTXTypeFromSymbol(set TileExtTXSetType, symbol int) (TransformType, error) {
	return internaltile.ExtTXTypeFromSymbol(set, symbol)
}

// TileExtTXTypeAllowed reports whether typ is permitted in the ext-tx set
// identified by set.
func TileExtTXTypeAllowed(set TileExtTXSetType, typ TransformType) (bool, error) {
	return internaltile.ExtTXTypeAllowed(set, typ)
}

// TileExtTXSymbolForType returns the ext-tx symbol used to signal typ within
// the ext-tx set identified by set.
func TileExtTXSymbolForType(set TileExtTXSetType, typ TransformType) (int, error) {
	return internaltile.ExtTXSymbolForType(set, typ)
}
