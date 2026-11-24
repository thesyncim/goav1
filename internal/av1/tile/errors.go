package tile

import "errors"

var (
	ErrInvalidDecodeState = errors.New("tile: invalid decode state")
	ErrInvalidPlan        = errors.New("tile: invalid work plan")
	ErrJobBufferTooSmall  = errors.New("tile: job buffer too small")
	// ErrTileDataOverread reports a tile whose symbols ran past the end of its
	// payload (damaged or truncated tile data), which libaom and dav1d reject.
	ErrTileDataOverread = errors.New("tile: symbol decode ran past the tile data")
	// ErrInvalid422Partition reports a 4:2:2 partition whose chroma block
	// size is invalid (blocks taller than wide), which dav1d rejects.
	ErrInvalid422Partition = errors.New("tile: partition has no 4:2:2 chroma block size")
)
