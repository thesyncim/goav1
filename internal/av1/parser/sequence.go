package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const (
	ColorPrimariesBT709         = 1
	TransferCharacteristicsSRGB = 13
	MatrixCoefficientsIdentity  = 0

	SelectScreenContentTools = 2
	SelectIntegerMV          = 2
)

// SequenceHeader is a compact parsed view of sequence_header_obu().
type SequenceHeader struct {
	SeqProfile                uint8
	StillPicture              bool
	ReducedStillPictureHeader bool

	TimingInfoPresent          bool
	DecoderModelInfoPresent    bool
	InitialDisplayDelayPresent bool
	TimingInfo                 TimingInfo
	DecoderModelInfo           DecoderModelInfo
	OperatingPointsCount       uint8
	OperatingPoints            [32]OperatingPoint
	FrameWidthBits             uint8
	FrameHeightBits            uint8
	MaxFrameWidth              uint32
	MaxFrameHeight             uint32
	FrameIDNumbersPresent      bool
	DeltaFrameIDLength         uint8
	AdditionalFrameIDLength    uint8
	Use128x128Superblock       bool
	EnableFilterIntra          bool
	EnableIntraEdgeFilter      bool
	EnableInterIntraCompound   bool
	EnableMaskedCompound       bool
	EnableWarpedMotion         bool
	EnableDualFilter           bool
	EnableOrderHint            bool
	EnableJNTComp              bool
	EnableRefFrameMVS          bool
	SeqForceScreenContentTools uint8
	SeqForceIntegerMV          uint8
	OrderHintBits              uint8
	EnableSuperRes             bool
	EnableCDEF                 bool
	EnableRestoration          bool
	ColorConfig                ColorConfig
	FilmGrainParamsPresent     bool
}

type TimingInfo struct {
	NumUnitsInDisplayTick    uint32
	TimeScale                uint32
	EqualPictureInterval     bool
	NumTicksPerPictureMinus1 uint32
}

type DecoderModelInfo struct {
	BufferDelayLength           uint8
	NumUnitsInDecodingTick      uint32
	BufferRemovalTimeLength     uint8
	FramePresentationTimeLength uint8
}

type OperatingPoint struct {
	IDC                        uint16
	SeqLevelIdx                uint8
	SeqTier                    uint8
	DecoderModelPresent        bool
	DecoderBufferDelay         uint32
	EncoderBufferDelay         uint32
	LowDelayMode               bool
	InitialDisplayDelayPresent bool
	InitialDisplayDelayMinus1  uint8
}

type ColorConfig struct {
	HighBitdepth            bool
	TwelveBit               bool
	BitDepth                uint8
	MonoChrome              bool
	ColorDescriptionPresent bool
	ColorPrimaries          uint8
	TransferCharacteristics uint8
	MatrixCoefficients      uint8
	ColorRange              bool
	SubsamplingX            bool
	SubsamplingY            bool
	ChromaSamplePosition    uint8
	SeparateUVDeltaQ        bool
}

// ParseSequenceHeader parses sequence_header_obu() payload bytes, including
// required AV1 trailing bits.
func ParseSequenceHeader(payload []byte) (SequenceHeader, error) {
	return parseSequenceHeader(payload, true)
}

// PeekSequenceHeader parses sequence_header_obu() fields without validating
// trailing bits. This matches libaom's stream-info/config peek path; callers
// that consume a full sequence header should use ParseSequenceHeader.
func PeekSequenceHeader(payload []byte) (SequenceHeader, error) {
	return parseSequenceHeader(payload, false)
}

func parseSequenceHeader(payload []byte, validateTrailing bool) (SequenceHeader, error) {
	r := bitstream.NewReader(payload)
	var sh SequenceHeader

	v, err := r.ReadBits(3)
	if err != nil {
		return SequenceHeader{}, err
	}
	sh.SeqProfile = uint8(v)
	if sh.SeqProfile > 2 {
		return SequenceHeader{}, ErrInvalidSequenceHeader
	}

	if sh.StillPicture, err = r.ReadBool(); err != nil {
		return SequenceHeader{}, err
	}
	if sh.ReducedStillPictureHeader, err = r.ReadBool(); err != nil {
		return SequenceHeader{}, err
	}
	if sh.ReducedStillPictureHeader && !sh.StillPicture {
		return SequenceHeader{}, ErrInvalidSequenceHeader
	}

	if sh.ReducedStillPictureHeader {
		sh.OperatingPointsCount = 1
		level, err := r.ReadBits(5)
		if err != nil {
			return SequenceHeader{}, err
		}
		sh.OperatingPoints[0].SeqLevelIdx = uint8(level)
	} else {
		if err := parseSequenceHeaderTiming(&r, &sh); err != nil {
			return SequenceHeader{}, err
		}
	}

	if err := parseSequenceFrameSize(&r, &sh); err != nil {
		return SequenceHeader{}, err
	}
	if err := parseSequenceTools(&r, &sh); err != nil {
		return SequenceHeader{}, err
	}
	if err := parseColorConfig(&r, &sh); err != nil {
		return SequenceHeader{}, err
	}
	if sh.FilmGrainParamsPresent, err = r.ReadBool(); err != nil {
		return SequenceHeader{}, err
	}
	if validateTrailing {
		if err := r.ReadTrailingBits(); err != nil {
			return SequenceHeader{}, err
		}
	}

	return sh, nil
}

func parseSequenceHeaderTiming(r *bitstream.Reader, sh *SequenceHeader) error {
	var err error
	if sh.TimingInfoPresent, err = r.ReadBool(); err != nil {
		return err
	}
	if sh.TimingInfoPresent {
		if err := parseTimingInfo(r, &sh.TimingInfo); err != nil {
			return err
		}
		if sh.DecoderModelInfoPresent, err = r.ReadBool(); err != nil {
			return err
		}
		if sh.DecoderModelInfoPresent {
			if err := parseDecoderModelInfo(r, &sh.DecoderModelInfo); err != nil {
				return err
			}
		}
	}

	if sh.InitialDisplayDelayPresent, err = r.ReadBool(); err != nil {
		return err
	}
	opsMinus1, err := r.ReadBits(5)
	if err != nil {
		return err
	}
	sh.OperatingPointsCount = uint8(opsMinus1 + 1)
	for i := uint8(0); i < sh.OperatingPointsCount; i++ {
		op := &sh.OperatingPoints[i]
		idc, err := r.ReadBits(12)
		if err != nil {
			return err
		}
		op.IDC = uint16(idc)
		level, err := r.ReadBits(5)
		if err != nil {
			return err
		}
		op.SeqLevelIdx = uint8(level)
		if op.SeqLevelIdx > 7 {
			tier, err := r.ReadBits(1)
			if err != nil {
				return err
			}
			op.SeqTier = uint8(tier)
		}
		if sh.DecoderModelInfoPresent {
			if op.DecoderModelPresent, err = r.ReadBool(); err != nil {
				return err
			}
			if op.DecoderModelPresent {
				if err := parseOperatingParametersInfo(r, sh.DecoderModelInfo, op); err != nil {
					return err
				}
			}
		}
		if sh.InitialDisplayDelayPresent {
			if op.InitialDisplayDelayPresent, err = r.ReadBool(); err != nil {
				return err
			}
			if op.InitialDisplayDelayPresent {
				delay, err := r.ReadBits(4)
				if err != nil {
					return err
				}
				op.InitialDisplayDelayMinus1 = uint8(delay)
			}
		}
	}
	return nil
}

func parseTimingInfo(r *bitstream.Reader, timing *TimingInfo) error {
	v, err := r.ReadBits(32)
	if err != nil {
		return err
	}
	timing.NumUnitsInDisplayTick = uint32(v)
	v, err = r.ReadBits(32)
	if err != nil {
		return err
	}
	timing.TimeScale = uint32(v)
	if timing.EqualPictureInterval, err = r.ReadBool(); err != nil {
		return err
	}
	if timing.EqualPictureInterval {
		timing.NumTicksPerPictureMinus1, err = r.ReadUVLC()
	}
	return err
}

func parseDecoderModelInfo(r *bitstream.Reader, info *DecoderModelInfo) error {
	v, err := r.ReadBits(5)
	if err != nil {
		return err
	}
	info.BufferDelayLength = uint8(v + 1)
	v, err = r.ReadBits(32)
	if err != nil {
		return err
	}
	info.NumUnitsInDecodingTick = uint32(v)
	v, err = r.ReadBits(5)
	if err != nil {
		return err
	}
	info.BufferRemovalTimeLength = uint8(v + 1)
	v, err = r.ReadBits(5)
	if err != nil {
		return err
	}
	info.FramePresentationTimeLength = uint8(v + 1)
	return nil
}

func parseOperatingParametersInfo(r *bitstream.Reader, info DecoderModelInfo, op *OperatingPoint) error {
	v, err := r.ReadBits(info.BufferDelayLength)
	if err != nil {
		return err
	}
	op.DecoderBufferDelay = uint32(v)
	v, err = r.ReadBits(info.BufferDelayLength)
	if err != nil {
		return err
	}
	op.EncoderBufferDelay = uint32(v)
	op.LowDelayMode, err = r.ReadBool()
	return err
}

func parseSequenceFrameSize(r *bitstream.Reader, sh *SequenceHeader) error {
	v, err := r.ReadBits(4)
	if err != nil {
		return err
	}
	sh.FrameWidthBits = uint8(v + 1)
	v, err = r.ReadBits(4)
	if err != nil {
		return err
	}
	sh.FrameHeightBits = uint8(v + 1)
	v, err = r.ReadBits(sh.FrameWidthBits)
	if err != nil {
		return err
	}
	sh.MaxFrameWidth = uint32(v + 1)
	v, err = r.ReadBits(sh.FrameHeightBits)
	if err != nil {
		return err
	}
	sh.MaxFrameHeight = uint32(v + 1)

	if !sh.ReducedStillPictureHeader {
		if sh.FrameIDNumbersPresent, err = r.ReadBool(); err != nil {
			return err
		}
	}
	if sh.FrameIDNumbersPresent {
		v, err = r.ReadBits(4)
		if err != nil {
			return err
		}
		sh.DeltaFrameIDLength = uint8(v + 2)
		v, err = r.ReadBits(3)
		if err != nil {
			return err
		}
		sh.AdditionalFrameIDLength = uint8(v + 1)
	}
	return nil
}

func parseSequenceTools(r *bitstream.Reader, sh *SequenceHeader) error {
	var err error
	if sh.Use128x128Superblock, err = r.ReadBool(); err != nil {
		return err
	}
	if sh.EnableFilterIntra, err = r.ReadBool(); err != nil {
		return err
	}
	if sh.EnableIntraEdgeFilter, err = r.ReadBool(); err != nil {
		return err
	}

	if !sh.ReducedStillPictureHeader {
		if sh.EnableInterIntraCompound, err = r.ReadBool(); err != nil {
			return err
		}
		if sh.EnableMaskedCompound, err = r.ReadBool(); err != nil {
			return err
		}
		if sh.EnableWarpedMotion, err = r.ReadBool(); err != nil {
			return err
		}
		if sh.EnableDualFilter, err = r.ReadBool(); err != nil {
			return err
		}
		if sh.EnableOrderHint, err = r.ReadBool(); err != nil {
			return err
		}
		if sh.EnableOrderHint {
			if sh.EnableJNTComp, err = r.ReadBool(); err != nil {
				return err
			}
			if sh.EnableRefFrameMVS, err = r.ReadBool(); err != nil {
				return err
			}
		}
		chooseScreen, err := r.ReadBool()
		if err != nil {
			return err
		}
		if chooseScreen {
			sh.SeqForceScreenContentTools = SelectScreenContentTools
		} else {
			b, err := r.ReadBits(1)
			if err != nil {
				return err
			}
			sh.SeqForceScreenContentTools = uint8(b)
		}
		if sh.SeqForceScreenContentTools > 0 {
			chooseInteger, err := r.ReadBool()
			if err != nil {
				return err
			}
			if chooseInteger {
				sh.SeqForceIntegerMV = SelectIntegerMV
			} else {
				b, err := r.ReadBits(1)
				if err != nil {
					return err
				}
				sh.SeqForceIntegerMV = uint8(b)
			}
		}
		if sh.EnableOrderHint {
			v, err := r.ReadBits(3)
			if err != nil {
				return err
			}
			sh.OrderHintBits = uint8(v + 1)
		}
	}

	if sh.EnableSuperRes, err = r.ReadBool(); err != nil {
		return err
	}
	if sh.EnableCDEF, err = r.ReadBool(); err != nil {
		return err
	}
	sh.EnableRestoration, err = r.ReadBool()
	return err
}

func parseColorConfig(r *bitstream.Reader, sh *SequenceHeader) error {
	cfg := &sh.ColorConfig
	var err error
	if cfg.HighBitdepth, err = r.ReadBool(); err != nil {
		return err
	}

	if sh.SeqProfile == 2 && cfg.HighBitdepth {
		if cfg.TwelveBit, err = r.ReadBool(); err != nil {
			return err
		}
		if cfg.TwelveBit {
			cfg.BitDepth = 12
		} else {
			cfg.BitDepth = 10
		}
	} else if cfg.HighBitdepth {
		cfg.BitDepth = 10
	} else {
		cfg.BitDepth = 8
	}

	if sh.SeqProfile == 1 {
		cfg.MonoChrome = false
	} else if cfg.MonoChrome, err = r.ReadBool(); err != nil {
		return err
	}

	if cfg.ColorDescriptionPresent, err = r.ReadBool(); err != nil {
		return err
	}
	if cfg.ColorDescriptionPresent {
		v, err := r.ReadBits(8)
		if err != nil {
			return err
		}
		cfg.ColorPrimaries = uint8(v)
		v, err = r.ReadBits(8)
		if err != nil {
			return err
		}
		cfg.TransferCharacteristics = uint8(v)
		v, err = r.ReadBits(8)
		if err != nil {
			return err
		}
		cfg.MatrixCoefficients = uint8(v)
	} else {
		cfg.ColorPrimaries = 2
		cfg.TransferCharacteristics = 2
		cfg.MatrixCoefficients = 2
	}

	if cfg.MonoChrome {
		cfg.ColorRange, err = r.ReadBool()
		cfg.SubsamplingX = true
		cfg.SubsamplingY = true
		return err
	}

	if cfg.ColorPrimaries == ColorPrimariesBT709 &&
		cfg.TransferCharacteristics == TransferCharacteristicsSRGB &&
		cfg.MatrixCoefficients == MatrixCoefficientsIdentity {
		cfg.ColorRange = true
		cfg.SubsamplingX = false
		cfg.SubsamplingY = false
	} else {
		if cfg.ColorRange, err = r.ReadBool(); err != nil {
			return err
		}
		switch sh.SeqProfile {
		case 0:
			cfg.SubsamplingX = true
			cfg.SubsamplingY = true
		case 1:
			cfg.SubsamplingX = false
			cfg.SubsamplingY = false
		case 2:
			if cfg.BitDepth == 12 {
				if cfg.SubsamplingX, err = r.ReadBool(); err != nil {
					return err
				}
				if cfg.SubsamplingX {
					if cfg.SubsamplingY, err = r.ReadBool(); err != nil {
						return err
					}
				}
			} else {
				cfg.SubsamplingX = true
			}
		}
		if cfg.SubsamplingX && cfg.SubsamplingY {
			v, err := r.ReadBits(2)
			if err != nil {
				return err
			}
			cfg.ChromaSamplePosition = uint8(v)
		}
	}

	cfg.SeparateUVDeltaQ, err = r.ReadBool()
	return err
}
