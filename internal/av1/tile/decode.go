package tile

import "github.com/thesyncim/goav1/internal/av1/entropy"

// DecodeOptions contains frame-level controls that affect tile symbol decode.
type DecodeOptions struct {
	DisableCDFUpdate bool
}

// NewEntropyReader returns an AV1 entropy reader over job's exact tile payload.
// The returned reader aliases payload and does not allocate.
func NewEntropyReader(payload []byte, job Job, options DecodeOptions) (entropy.Reader, error) {
	data, err := job.Payload(payload)
	if err != nil {
		return entropy.Reader{}, err
	}
	return entropy.NewReaderWithCDFUpdate(data, !options.DisableCDFUpdate), nil
}
