package parser

import "errors"

var (
	ErrInvalidSequenceHeader = errors.New("parser: invalid sequence header")
	ErrInvalidFrameHeader    = errors.New("parser: invalid frame header")
	ErrReferenceFrameNeeded  = errors.New("parser: reference frame state required")
)
