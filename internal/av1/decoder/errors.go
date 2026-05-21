package decoder

import "errors"

var (
	ErrMissingSequenceHeader        = errors.New("decoder: missing sequence header")
	ErrMissingFrameHeader           = errors.New("decoder: missing frame header")
	ErrEventBufferTooSmall          = errors.New("decoder: event buffer too small")
	ErrInvalidSurfaceEvent          = errors.New("decoder: invalid surface event")
	ErrInvalidSurfaceReference      = errors.New("decoder: invalid surface reference")
	ErrSurfaceReleaseBufferTooSmall = errors.New("decoder: surface release buffer too small")
)
