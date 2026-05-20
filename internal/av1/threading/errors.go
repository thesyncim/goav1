package threading

import "errors"

var (
	ErrInvalidWorkerCount  = errors.New("threading: invalid worker count")
	ErrBatchBufferTooSmall = errors.New("threading: batch buffer too small")
	ErrInvalidBatch        = errors.New("threading: invalid batch")
	ErrInvalidJobs         = errors.New("threading: invalid jobs")
	ErrInvalidCallback     = errors.New("threading: invalid callback")
	ErrPoolClosed          = errors.New("threading: worker pool closed")
)
