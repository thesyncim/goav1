package decoder

import "errors"

var (
	ErrMissingSequenceHeader = errors.New("decoder: missing sequence header")
	ErrMissingFrameHeader    = errors.New("decoder: missing frame header")
	ErrEventBufferTooSmall   = errors.New("decoder: event buffer too small")
)
