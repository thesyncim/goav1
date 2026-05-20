package threading

import "errors"

var (
	ErrInvalidWorkerCount  = errors.New("threading: invalid worker count")
	ErrBatchBufferTooSmall = errors.New("threading: batch buffer too small")
	ErrInvalidJobs         = errors.New("threading: invalid jobs")
)
