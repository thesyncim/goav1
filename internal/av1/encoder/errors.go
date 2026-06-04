package encoder

import "errors"

var (
	ErrInvalidConfig = errors.New("encoder: invalid config")
	ErrInvalidFrame  = errors.New("encoder: invalid frame")
	ErrUnsupported   = errors.New("encoder: unsupported configuration")
)
