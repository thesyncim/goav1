package tile

import "github.com/thesyncim/goav1/internal/av1/entropy"

// DecodeOptions contains frame-level controls that affect tile symbol decode.
type DecodeOptions struct {
	DisableCDFUpdate bool
}

// DecodeState is caller-owned per-tile decode state. It binds a scheduled job
// to an entropy reader and records whether this tile's adapted context should
// be retained as the next frame context after decode succeeds.
type DecodeState struct {
	Job                Job
	Reader             entropy.Reader
	RetainFrameContext bool
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

// Reset initializes s for job's exact tile payload without allocating.
func (s *DecodeState) Reset(payload []byte, job Job, options DecodeOptions) error {
	if s == nil {
		return ErrInvalidDecodeState
	}
	reader, err := NewEntropyReader(payload, job, options)
	if err != nil {
		return err
	}
	*s = DecodeState{
		Job:                job,
		Reader:             reader,
		RetainFrameContext: job.UpdatesFrameContext && !options.DisableCDFUpdate,
	}
	return nil
}

// ReadSymbol decodes one tile symbol from caller-owned CDF state.
func (s *DecodeState) ReadSymbol(cdf *entropy.CDF) (int, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	return s.Reader.ReadCDF(cdf)
}

// ReadSignedDelta decodes the signed delta core used by AV1 delta-q and
// delta-loopfilter tile syntax.
func (s *DecodeState) ReadSignedDelta(cdf *entropy.CDF, small int) (int, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	return s.Reader.ReadSignedDelta(cdf, small)
}
