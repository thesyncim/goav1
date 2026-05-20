package tile

import "errors"

var (
	ErrInvalidPlan       = errors.New("tile: invalid work plan")
	ErrJobBufferTooSmall = errors.New("tile: job buffer too small")
)
