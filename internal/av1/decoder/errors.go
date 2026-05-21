package decoder

import "errors"

var (
	ErrMissingSequenceHeader          = errors.New("decoder: missing sequence header")
	ErrMissingFrameHeader             = errors.New("decoder: missing frame header")
	ErrEventBufferTooSmall            = errors.New("decoder: event buffer too small")
	ErrInvalidFrameWorkState          = errors.New("decoder: invalid frame work state")
	ErrInvalidFrameWorkStep           = errors.New("decoder: invalid frame work step")
	ErrInvalidSurfaceEvent            = errors.New("decoder: invalid surface event")
	ErrInvalidSurfaceReference        = errors.New("decoder: invalid surface reference")
	ErrSurfaceReferenceBufferTooSmall = errors.New("decoder: surface reference buffer too small")
	ErrSurfaceReleaseBufferTooSmall   = errors.New("decoder: surface release buffer too small")
)
