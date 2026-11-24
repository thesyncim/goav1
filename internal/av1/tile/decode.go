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

	// motionExtras backs slot 1 of BlockPredictionModeResult.ExtrasSlot for
	// the block being decoded.
	motionExtras [1]BlockPredictionMotionExtras
	// subChroma backs slot 1 of BlockPredictionModeResult.SubChromaSlot.
	subChroma [1]SubChromaInterResult
}

// blockMotionExtras points result at the state's out-of-line payload slot.
func (s *DecodeState) blockMotionExtras(result *BlockPredictionModeResult) *BlockPredictionMotionExtras {
	result.ExtrasSlot = 1
	return &s.motionExtras[0]
}

// Payloads returns the single-slot payloads the block loop's visits name
// while they are current: the state's motion extras and sub8x8 chroma cells
// and the loop scratch's palette maps.
func (s *DecodeState) Payloads(scratch *BlockLoopScratch) BlockPayloads {
	return BlockPayloads{Extras: s.motionExtras[:], SubChroma: s.subChroma[:], Palette: scratch.Palette[:]}
}

// BlockPayloads holds the out-of-line payloads that BlockPredictionModeResult
// names by slot: slot n is element n-1, slot 0 is none. The block loop's own
// payloads have one slot each (DecodeState.Payloads); buffered visits are
// renumbered into their owner's arenas. Slots keep BlockLoopVisit free of
// pointers, so buffers of visits are never scanned by the garbage collector
// and copying one runs no write barrier.
type BlockPayloads struct {
	Extras    []BlockPredictionMotionExtras
	SubChroma []SubChromaInterResult
	Palette   []PaletteModeScratch
}

// MotionExtras returns r's warp/OBMC payload, or nil.
func (p *BlockPayloads) MotionExtras(r *BlockPredictionModeResult) *BlockPredictionMotionExtras {
	if r.ExtrasSlot == 0 || p == nil || int(r.ExtrasSlot) > len(p.Extras) {
		return nil
	}
	return &p.Extras[r.ExtrasSlot-1]
}

// SubChromaInter returns r's sub8x8 chroma cells, or nil.
func (p *BlockPayloads) SubChromaInter(r *BlockPredictionModeResult) *SubChromaInterResult {
	if r.SubChromaSlot == 0 || p == nil || int(r.SubChromaSlot) > len(p.SubChroma) {
		return nil
	}
	return &p.SubChroma[r.SubChromaSlot-1]
}

// PaletteYMap returns r's luma colour index map, or nil.
func (p *BlockPayloads) PaletteYMap(r *PaletteModeResult) *[MaxPalettePixels]uint8 {
	if r.YMapSlot == 0 || p == nil || int(r.YMapSlot) > len(p.Palette) {
		return nil
	}
	return &p.Palette[r.YMapSlot-1].Y
}

// PaletteUVMap returns r's chroma colour index map, or nil.
func (p *BlockPayloads) PaletteUVMap(r *PaletteModeResult) *[MaxPalettePixels]uint8 {
	if r.UVMapSlot == 0 || p == nil || int(r.UVMapSlot) > len(p.Palette) {
		return nil
	}
	return &p.Palette[r.UVMapSlot-1].UV
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
	return s.readBlockDeltas(&params, &cdfs, block.Monochrome)
}

// readBlockDeltas reads one block's delta_qindex and delta_lf symbols once the
// caller has decided they are coded (dav1d decode_b's delta_q block).
func (s *DecodeState) readBlockDeltas(params *parser.DeltaParams, cdfs *DeltaCDFs, monochrome bool) error {
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
		if monochrome {
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
