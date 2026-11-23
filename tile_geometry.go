package goav1

import (
	internaltransform "github.com/thesyncim/goav1/internal/av1/transform"

	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

// tileBlockLevelHalfSize4x4 returns the half-block side, in units of 4x4
// samples, for the supplied block level.
func tileBlockLevelHalfSize4x4(level internaltile.BlockLevel) uint8 {
	return level.HalfSize4x4()
}

// tileBlockLevelSize4x4 returns the full-block side, in units of 4x4 samples,
// for the supplied block level.
func tileBlockLevelSize4x4(level internaltile.BlockLevel) uint8 {
	return level.Size4x4()
}

// tileBlockDimensionsOf returns the (width, height, valid) tuple for size.
func tileBlockDimensionsOf(size internaltile.BlockSize) (internaltile.BlockDimensions, bool) {
	return size.Dimensions()
}

// tilePartitionValidForLevel reports whether partition is allowed by the AV1
// partition rules at level.
func tilePartitionValidForLevel(partition internaltile.Partition, level internaltile.BlockLevel) bool {
	return partition.ValidForLevel(level)
}

// tilePartitionBlockSizes returns the two child block sizes produced by
// partition at level (the second size is only meaningful for asymmetric
// partitions).
func tilePartitionBlockSizes(partition internaltile.Partition, level internaltile.BlockLevel) (internaltile.BlockSize, internaltile.BlockSize, bool) {
	return partition.BlockSizes(level)
}

// tileTransformDimensionsOf returns the (width, height, valid) tuple for the
// supplied transform size.
func tileTransformDimensionsOf(size internaltile.TransformSize) (internaltile.TransformDimensions, bool) {
	return size.Dimensions()
}

// tileTransformSizePixels converts a TileTransformSize into the corresponding
// canonical TransformSize used by the DSP layer.
func tileTransformSizePixels(size internaltile.TransformSize) (internaltransform.Size, error) {
	return size.TransformSize()
}
