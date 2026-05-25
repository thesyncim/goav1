package goav1

import internaltile "github.com/thesyncim/goav1/internal/av1/tile"

func TileRootBlockLevel(use128x128Superblock bool) TileBlockLevel {
	return internaltile.RootBlockLevel(use128x128Superblock)
}

func TileBlockLevelHalfSize4x4(level TileBlockLevel) uint8 {
	return level.HalfSize4x4()
}

func TileBlockLevelSize4x4(level TileBlockLevel) uint8 {
	return level.Size4x4()
}

func TileBlockDimensionsOf(size TileBlockSize) (TileBlockDimensions, bool) {
	return size.Dimensions()
}

func TilePartitionValidForLevel(partition TilePartition, level TileBlockLevel) bool {
	return partition.ValidForLevel(level)
}

func TilePartitionBlockSizes(partition TilePartition, level TileBlockLevel) (TileBlockSize, TileBlockSize, bool) {
	return partition.BlockSizes(level)
}

func TileTransformDimensionsOf(size TileTransformSize) (TileTransformDimensions, bool) {
	return size.Dimensions()
}

func TileTransformSizePixels(size TileTransformSize) (TransformSize, error) {
	return size.TransformSize()
}

func MaxTileTransformSize(block TileBlockSize, color ColorConfig, plane int) (TileTransformSize, error) {
	return internaltile.MaxTransformSize(block, color, plane)
}

func TileTransformSizeSquare(size TileTransformSize) (TileTransformSize, error) {
	return internaltile.TransformSizeSquare(size)
}

func TileTransformSizeSquareUp(size TileTransformSize) (TileTransformSize, error) {
	return internaltile.TransformSizeSquareUp(size)
}

func TileExtTXSetTypeFor(size TileTransformSize, inter bool, reduced bool) (TileExtTXSetType, error) {
	return internaltile.ExtTXSetTypeFor(size, inter, reduced)
}

func TileExtTXSetIndex(size TileTransformSize, inter bool, reduced bool) (int, error) {
	return internaltile.ExtTXSetIndex(size, inter, reduced)
}

func TileExtTXTypeCount(set TileExtTXSetType) (int, error) {
	return internaltile.ExtTXTypeCount(set)
}

func TileExtTXTypeFromSymbol(set TileExtTXSetType, symbol int) (TransformType, error) {
	return internaltile.ExtTXTypeFromSymbol(set, symbol)
}

func TileExtTXTypeAllowed(set TileExtTXSetType, typ TransformType) (bool, error) {
	return internaltile.ExtTXTypeAllowed(set, typ)
}

func TileExtTXSymbolForType(set TileExtTXSetType, typ TransformType) (int, error) {
	return internaltile.ExtTXSymbolForType(set, typ)
}
