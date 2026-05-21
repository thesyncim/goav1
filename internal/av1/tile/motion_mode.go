package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// MotionMode identifies AV1's motion variation mode in libaom/dav1d order.
type MotionMode uint8

const (
	MotionModeTranslation MotionMode = iota
	MotionModeOBMC
	MotionModeWarp
	motionModeCount
)

// MotionModeCDFs contains caller-owned CDFs for motion mode syntax.
type MotionModeCDFs struct {
	MotionMode [blockSizeCount]entropy.CDF
	OBMC       [blockSizeCount]entropy.CDF
}

// MotionModeRequest describes the already-derived block/frame conditions used
// by libaom's motion_mode_allowed() gate.
type MotionModeRequest struct {
	Size BlockSize
	Mode InterMode

	Compound   bool
	SkipMode   bool
	InterIntra bool

	SwitchableMotionMode bool
	AllowWarpedMotion    bool
	ForceIntegerMV       bool
	GlobalMotionType     parser.GlobalMotionType
	ScaledReference      bool

	OverlappableNeighbors int
	NumProjRef            int
}

var motionModeDefaultCDF = [blockSizeCount][2]uint16{
	BlockSize128x128: {32507, 32558},
	BlockSize128x64:  {30878, 31335},
	BlockSize64x128:  {28898, 30397},
	BlockSize64x64:   {29516, 30701},
	BlockSize64x32:   {21679, 26830},
	BlockSize64x16:   {29742, 31203},
	BlockSize32x64:   {20360, 28062},
	BlockSize32x32:   {26260, 29116},
	BlockSize32x16:   {11606, 24308},
	BlockSize32x8:    {28799, 31390},
	BlockSize16x64:   {28973, 31594},
	BlockSize16x32:   {5123, 23606},
	BlockSize16x16:   {19419, 26810},
	BlockSize16x8:    {5391, 25528},
	BlockSize8x32:    {26431, 30774},
	BlockSize8x16:    {4738, 24765},
	BlockSize8x8:     {7651, 24760},
}

var obmcDefaultCDF = [blockSizeCount]uint16{
	BlockSize128x128: 32638,
	BlockSize128x64:  31560,
	BlockSize64x128:  31014,
	BlockSize64x64:   30128,
	BlockSize64x32:   22083,
	BlockSize64x16:   26879,
	BlockSize32x64:   22823,
	BlockSize32x32:   25817,
	BlockSize32x16:   15142,
	BlockSize32x8:    23664,
	BlockSize16x64:   24008,
	BlockSize16x32:   14423,
	BlockSize16x16:   17432,
	BlockSize16x8:    9301,
	BlockSize8x32:    20901,
	BlockSize8x16:    9371,
	BlockSize8x8:     10437,
}

func (mode MotionMode) Valid() bool {
	return mode < motionModeCount
}

// InitDefault seeds c with dav1d/libaom's default motion-mode CDFs.
func (c *MotionModeCDFs) InitDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	var next MotionModeCDFs
	for size := BlockSize(0); size < blockSizeCount; size++ {
		if !motionModeBlockSizeAllowed(size) {
			continue
		}
		if err := next.MotionMode[size].Init(motionModeDefaultCDF[size][:]); err != nil {
			return err
		}
		if err := next.OBMC[size].Init([]uint16{obmcDefaultCDF[size]}); err != nil {
			return err
		}
	}
	*c = next
	return nil
}

func (c *MotionModeCDFs) MotionModeCDF(size BlockSize) (*entropy.CDF, error) {
	if c == nil || !motionModeBlockSizeAllowed(size) {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.MotionMode[size]
	if cdf.Symbols() != int(motionModeCount) {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

func (c *MotionModeCDFs) OBMCCDF(size BlockSize) (*entropy.CDF, error) {
	if c == nil || !motionModeBlockSizeAllowed(size) {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.OBMC[size])
}

// LastMotionModeAllowed ports libaom's motion_mode_allowed() result. It returns
// the largest syntax mode that may be read for this block.
func LastMotionModeAllowed(req MotionModeRequest) (MotionMode, error) {
	if _, ok := req.Size.Dimensions(); !ok {
		return 0, ErrInvalidDecodeState
	}
	if !req.Compound && !req.Mode.Valid() {
		return 0, ErrInvalidDecodeState
	}
	if !globalMotionTypeValid(req.GlobalMotionType) || req.OverlappableNeighbors < 0 || req.NumProjRef < 0 {
		return 0, ErrInvalidDecodeState
	}
	if !req.SwitchableMotionMode || req.SkipMode || req.InterIntra || req.OverlappableNeighbors == 0 {
		return MotionModeTranslation, nil
	}
	if !motionModeBlockSizeAllowed(req.Size) || req.Compound {
		return MotionModeTranslation, nil
	}
	if !req.ForceIntegerMV && req.Mode == InterModeGlobalMV && req.GlobalMotionType > parser.GlobalMotionTranslation {
		return MotionModeTranslation, nil
	}
	if req.NumProjRef >= 1 && req.AllowWarpedMotion && !req.ForceIntegerMV && !req.ScaledReference {
		return MotionModeWarp, nil
	}
	return MotionModeOBMC, nil
}

func (s *DecodeState) ReadMotionMode(cdfs *MotionModeCDFs, req MotionModeRequest) (MotionMode, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	last, err := LastMotionModeAllowed(req)
	if err != nil {
		return 0, err
	}
	switch last {
	case MotionModeTranslation:
		return MotionModeTranslation, nil
	case MotionModeOBMC:
		cdf, err := cdfs.OBMCCDF(req.Size)
		if err != nil {
			return 0, err
		}
		obmc, err := readBoolCDF(s, cdf)
		if err != nil {
			return 0, err
		}
		if obmc {
			return MotionModeOBMC, nil
		}
		return MotionModeTranslation, nil
	case MotionModeWarp:
		cdf, err := cdfs.MotionModeCDF(req.Size)
		if err != nil {
			return 0, err
		}
		symbol, err := s.Reader.ReadCDF(cdf)
		if err != nil {
			return 0, err
		}
		mode := MotionMode(symbol)
		if !mode.Valid() {
			return 0, ErrInvalidDecodeState
		}
		return mode, nil
	default:
		return 0, ErrInvalidDecodeState
	}
}

func motionModeBlockSizeAllowed(size BlockSize) bool {
	dims, ok := size.Dimensions()
	return ok && dims.W4 >= 2 && dims.H4 >= 2
}

func globalMotionTypeValid(t parser.GlobalMotionType) bool {
	return t <= parser.GlobalMotionAffine
}
