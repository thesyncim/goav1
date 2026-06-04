package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

const (
	MaxQIndex            = 255
	MaxLoopFilter        = 63
	FrameLoopFilterCount = 4
)

// DecodeOptions contains frame-level controls that affect tile symbol decode.
type DecodeOptions struct {
	DisableCDFUpdate bool
	BaseQIdx         uint8
}

// DecodeState is caller-owned per-tile decode state. It binds a scheduled job
// to an entropy reader and records whether this tile's adapted context should
// be retained as the next frame context after decode succeeds.
type DecodeState struct {
	Job                Job
	Reader             entropy.Reader
	RetainFrameContext bool

	CurrentBaseQIdx uint8
	DeltaLFFromBase int8
	DeltaLF         [FrameLoopFilterCount]int8
}

// BlockDeltaContext identifies the block currently being decoded for AV1
// delta-q and delta-loopfilter syntax.
type BlockDeltaContext struct {
	MICol uint16
	MIRow uint16

	SBSizeMIB uint8

	FullSuperblock bool
	SkipTransform  bool
	Monochrome     bool
}

// DeltaCDFs contains the caller-owned CDFs used by block delta syntax.
type DeltaCDFs struct {
	Q  *entropy.CDF
	LF *entropy.CDF

	LFMulti [FrameLoopFilterCount]*entropy.CDF
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
		CurrentBaseQIdx:    options.BaseQIdx,
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
func (s *DecodeState) ReadSignedDelta(cdf *entropy.CDF, small uint8) (int, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	return s.Reader.ReadSignedDelta(cdf, small)
}

// ReadBlockDeltas decodes AV1 delta_qindex and delta_lflevel syntax for one
// block and updates the tile's current q/loopfilter state.
func (s *DecodeState) ReadBlockDeltas(params parser.DeltaParams, block BlockDeltaContext, cdfs DeltaCDFs) error {
	if s == nil {
		return ErrInvalidDecodeState
	}
	if !params.DeltaQPresent {
		return nil
	}
	read, err := shouldReadBlockDelta(block)
	if err != nil {
		return err
	}
	if !read {
		return nil
	}

	deltaQ, err := s.Reader.ReadSignedDelta(cdfs.Q, entropy.DeltaSmall)
	if err != nil {
		return err
	}
	s.CurrentBaseQIdx = clampQIndex(int(s.CurrentBaseQIdx) + (deltaQ << params.DeltaQResLog2))

	if !params.DeltaLFPresent {
		return nil
	}
	if params.DeltaLFMulti {
		count := FrameLoopFilterCount
		if block.Monochrome {
			count = FrameLoopFilterCount - 2
		}
		for i := 0; i < count; i++ {
			deltaLF, err := s.Reader.ReadSignedDelta(cdfs.LFMulti[i], entropy.DeltaSmall)
			if err != nil {
				return err
			}
			s.DeltaLF[i] = clampLoopFilterDelta(int(s.DeltaLF[i]) + (deltaLF << params.DeltaLFResLog2))
		}
		return nil
	}

	deltaLF, err := s.Reader.ReadSignedDelta(cdfs.LF, entropy.DeltaSmall)
	if err != nil {
		return err
	}
	s.DeltaLFFromBase = clampLoopFilterDelta(int(s.DeltaLFFromBase) + (deltaLF << params.DeltaLFResLog2))
	return nil
}

func shouldReadBlockDelta(block BlockDeltaContext) (bool, error) {
	if block.SBSizeMIB == 0 {
		return false, ErrInvalidDecodeState
	}
	if block.MICol%uint16(block.SBSizeMIB) != 0 || block.MIRow%uint16(block.SBSizeMIB) != 0 {
		return false, nil
	}
	return !block.FullSuperblock || !block.SkipTransform, nil
}

func clampQIndex(v int) uint8 {
	if v < 1 {
		return 1
	}
	if v > MaxQIndex {
		return MaxQIndex
	}
	return uint8(v)
}

func clampLoopFilterDelta(v int) int8 {
	if v < -MaxLoopFilter {
		return -MaxLoopFilter
	}
	if v > MaxLoopFilter {
		return MaxLoopFilter
	}
	return int8(v)
}
