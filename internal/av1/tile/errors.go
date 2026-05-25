package tile

import "errors"

var (
	ErrInvalidDecodeState = errors.New("tile: invalid decode state")
	ErrInvalidPlan        = errors.New("tile: invalid work plan")
	ErrJobBufferTooSmall  = errors.New("tile: job buffer too small")
	ErrUnsupportedSyntax  = errors.New("tile: unsupported syntax")
)
