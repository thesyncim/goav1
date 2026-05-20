package parser

import "errors"

var (
	ErrInvalidSequenceHeader = errors.New("parser: invalid sequence header")
	ErrInvalidFrameHeader    = errors.New("parser: invalid frame header")
	ErrInvalidTileGroup      = errors.New("parser: invalid tile group")
	ErrReferenceFrameNeeded  = errors.New("parser: reference frame state required")
)
