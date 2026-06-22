package encoder

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

const (
	SequenceColorPrimariesBT709         = parser.ColorPrimariesBT709
	SequenceTransferCharacteristicsSRGB = parser.TransferCharacteristicsSRGB
	SequenceMatrixCoefficientsIdentity  = parser.MatrixCoefficientsIdentity

	SequenceSelectScreenContentTools = parser.SelectScreenContentTools
	SequenceSelectIntegerMV          = parser.SelectIntegerMV
	SequenceLevelMax                 = parser.SeqLevelMax
)

// SequenceHeader describes the encoder-owned fields for sequence_header_obu().
// It intentionally mirrors libaom/parser syntax order and keeps caller-owned
// buffers outside the struct.
type SequenceHeader struct {
	Profile                   Profile
	StillPicture              bool
	ReducedStillPictureHeader bool

	TimingInfoPresent          bool
	DecoderModelInfoPresent    bool
	InitialDisplayDelayPresent bool
	TimingInfo                 SequenceTimingInfo
	DecoderModelInfo           SequenceDecoderModelInfo
	OperatingPointsCount       uint8
	OperatingPoints            [32]SequenceOperatingPoint

	MaxFrameWidth           uint32
	MaxFrameHeight          uint32
	FrameIDNumbersPresent   bool
	DeltaFrameIDLength      uint8
	AdditionalFrameIDLength uint8

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

	ColorConfig            SequenceColorConfig
	FilmGrainParamsPresent bool
}

type SequenceOperatingPoint struct {
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

type SequenceTimingInfo struct {
	NumUnitsInDisplayTick    uint32
	TimeScale                uint32
	EqualPictureInterval     bool
	NumTicksPerPictureMinus1 uint32
}

type SequenceDecoderModelInfo struct {
	BufferDelayLength           uint8
	NumUnitsInDecodingTick      uint32
	BufferRemovalTimeLength     uint8
	FramePresentationTimeLength uint8
}

type SequenceColorConfig struct {
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

// SequenceHeaderPayloadSize returns the exact sequence_header_obu() payload
// byte count, including AV1 trailing bits.
func SequenceHeaderPayloadSize(seq SequenceHeader) (int, error) {
	w := newSizingBitWriter()
	if err := writeSequenceHeaderPayload(&w, seq); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

// AppendSequenceHeaderPayload appends sequence_header_obu() payload bytes into
// dst without growing it. Callers must reserve capacity up front.
func AppendSequenceHeaderPayload(dst []byte, seq SequenceHeader) ([]byte, error) {
	size, err := SequenceHeaderPayloadSize(seq)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, bitstream.ErrShortBuffer
	}

	off := len(dst)
	out := dst[:off+size]
	w := newBitWriter(out[off:])
	if err := writeSequenceHeaderPayload(&w, seq); err != nil {
		return dst, err
	}
	return out, nil
}

// LowOverheadSequenceHeaderOBUSize returns the exact size of a low-overhead
// sequence-header OBU with obu_has_size_field set.
func LowOverheadSequenceHeaderOBUSize(seq SequenceHeader) (int, error) {
	payloadSize, err := SequenceHeaderPayloadSize(seq)
	if err != nil {
		return 0, err
	}
	return 1 + bitstream.LEB128Len(uint32(payloadSize)) + payloadSize, nil
}

// AppendLowOverheadSequenceHeaderOBU appends a sequence header as one
// low-overhead OBU. It validates and sizes before writing, so errors leave dst
// length unchanged.
func AppendLowOverheadSequenceHeaderOBU(dst []byte, seq SequenceHeader) ([]byte, error) {
	payloadSize, err := SequenceHeaderPayloadSize(seq)
	if err != nil {
		return dst, err
	}
	size := 1 + bitstream.LEB128Len(uint32(payloadSize)) + payloadSize
	if cap(dst)-len(dst) < size {
		return dst, bitstream.ErrShortBuffer
	}

	off := len(dst)
	out := dst[:off+size]
	n, err := obu.PutHeader(out[off:], obu.Header{
		Type:         obu.TypeSequenceHeader,
		HasSizeField: true,
	})
	if err != nil {
		return dst, err
	}
	off += n
	n, err = bitstream.PutLEB128(out[off:], uint32(payloadSize))
	if err != nil {
		return dst, err
	}
	off += n
	w := newBitWriter(out[off:])
	if err := writeSequenceHeaderPayload(&w, seq); err != nil {
		return dst, err
	}
	return out, nil
}

func writeSequenceHeaderPayload(w *bitWriter, seq SequenceHeader) error {
	if err := validateSequenceHeader(seq); err != nil {
		return err
	}

	if err := w.writeBits(uint64(seq.Profile), 3); err != nil {
		return err
	}
	if err := w.writeBool(seq.StillPicture); err != nil {
		return err
	}
	if err := w.writeBool(seq.ReducedStillPictureHeader); err != nil {
		return err
	}

	opCount := sequenceOperatingPointCount(seq)
	if seq.ReducedStillPictureHeader {
		if err := writeSequenceLevel(w, seq.OperatingPoints[0].SeqLevelIdx); err != nil {
			return err
		}
	} else {
		if err := w.writeBool(seq.TimingInfoPresent); err != nil {
			return err
		}
		if seq.TimingInfoPresent {
			if err := writeSequenceTimingInfo(w, seq.TimingInfo); err != nil {
				return err
			}
			if err := w.writeBool(seq.DecoderModelInfoPresent); err != nil {
				return err
			}
			if seq.DecoderModelInfoPresent {
				if err := writeSequenceDecoderModelInfo(w, seq.DecoderModelInfo); err != nil {
					return err
				}
			}
		}
		if err := w.writeBool(seq.InitialDisplayDelayPresent); err != nil {
			return err
		}
		if err := w.writeBits(uint64(opCount-1), 5); err != nil {
			return err
		}
		for i := uint8(0); i < opCount; i++ {
			op := seq.OperatingPoints[i]
			if err := w.writeBits(uint64(op.IDC), 12); err != nil {
				return err
			}
			if err := writeSequenceLevel(w, op.SeqLevelIdx); err != nil {
				return err
			}
			if op.SeqLevelIdx > 7 {
				if err := w.writeBits(uint64(op.SeqTier), 1); err != nil {
					return err
				}
			}
			if seq.DecoderModelInfoPresent {
				if err := w.writeBool(op.DecoderModelPresent); err != nil {
					return err
				}
				if op.DecoderModelPresent {
					if err := writeSequenceOperatingParametersInfo(w, seq.DecoderModelInfo, op); err != nil {
						return err
					}
				}
			}
			if seq.InitialDisplayDelayPresent {
				if err := w.writeBool(op.InitialDisplayDelayPresent); err != nil {
					return err
				}
				if op.InitialDisplayDelayPresent {
					if err := w.writeBits(uint64(op.InitialDisplayDelayMinus1), 4); err != nil {
						return err
					}
				}
			}
		}
	}

	if err := writeSequenceFrameSize(w, seq); err != nil {
		return err
	}
	if err := writeSequenceTools(w, seq); err != nil {
		return err
	}
	if err := writeSequenceColorConfig(w, seq.Profile, seq.ColorConfig); err != nil {
		return err
	}
	if err := w.writeBool(seq.FilmGrainParamsPresent); err != nil {
		return err
	}
	return w.writeTrailingBits()
}

func writeSequenceLevel(w *bitWriter, level uint8) error {
	return w.writeBits(uint64(level), 5)
}

func writeSequenceTimingInfo(w *bitWriter, timing SequenceTimingInfo) error {
	if err := w.writeBits(uint64(timing.NumUnitsInDisplayTick), 32); err != nil {
		return err
	}
	if err := w.writeBits(uint64(timing.TimeScale), 32); err != nil {
		return err
	}
	if err := w.writeBool(timing.EqualPictureInterval); err != nil {
		return err
	}
	if timing.EqualPictureInterval {
		return w.writeUVLC(timing.NumTicksPerPictureMinus1)
	}
	return nil
}

func writeSequenceDecoderModelInfo(w *bitWriter, info SequenceDecoderModelInfo) error {
	if err := w.writeBits(uint64(info.BufferDelayLength-1), 5); err != nil {
		return err
	}
	if err := w.writeBits(uint64(info.NumUnitsInDecodingTick), 32); err != nil {
		return err
	}
	if err := w.writeBits(uint64(info.BufferRemovalTimeLength-1), 5); err != nil {
		return err
	}
	return w.writeBits(uint64(info.FramePresentationTimeLength-1), 5)
}

func writeSequenceOperatingParametersInfo(w *bitWriter, info SequenceDecoderModelInfo, op SequenceOperatingPoint) error {
	if err := w.writeBits(uint64(op.DecoderBufferDelay), info.BufferDelayLength); err != nil {
		return err
	}
	if err := w.writeBits(uint64(op.EncoderBufferDelay), info.BufferDelayLength); err != nil {
		return err
	}
	return w.writeBool(op.LowDelayMode)
}

func writeSequenceFrameSize(w *bitWriter, seq SequenceHeader) error {
	widthBits := sequenceFrameDimensionBits(seq.MaxFrameWidth)
	heightBits := sequenceFrameDimensionBits(seq.MaxFrameHeight)
	if err := w.writeBits(uint64(widthBits-1), 4); err != nil {
		return err
	}
	if err := w.writeBits(uint64(heightBits-1), 4); err != nil {
		return err
	}
	if err := w.writeBits(uint64(seq.MaxFrameWidth-1), widthBits); err != nil {
		return err
	}
	if err := w.writeBits(uint64(seq.MaxFrameHeight-1), heightBits); err != nil {
		return err
	}
	if !seq.ReducedStillPictureHeader {
		if err := w.writeBool(seq.FrameIDNumbersPresent); err != nil {
			return err
		}
	}
	if seq.FrameIDNumbersPresent {
		if err := w.writeBits(uint64(seq.DeltaFrameIDLength-2), 4); err != nil {
			return err
		}
		if err := w.writeBits(uint64(seq.AdditionalFrameIDLength-1), 3); err != nil {
			return err
		}
	}
	return nil
}

func writeSequenceTools(w *bitWriter, seq SequenceHeader) error {
	if err := w.writeBool(seq.Use128x128Superblock); err != nil {
		return err
	}
	if err := w.writeBool(seq.EnableFilterIntra); err != nil {
		return err
	}
	if err := w.writeBool(seq.EnableIntraEdgeFilter); err != nil {
		return err
	}

	if !seq.ReducedStillPictureHeader {
		if err := w.writeBool(seq.EnableInterIntraCompound); err != nil {
			return err
		}
		if err := w.writeBool(seq.EnableMaskedCompound); err != nil {
			return err
		}
		if err := w.writeBool(seq.EnableWarpedMotion); err != nil {
			return err
		}
		if err := w.writeBool(seq.EnableDualFilter); err != nil {
			return err
		}
		if err := w.writeBool(seq.EnableOrderHint); err != nil {
			return err
		}
		if seq.EnableOrderHint {
			if err := w.writeBool(seq.EnableJNTComp); err != nil {
				return err
			}
			if err := w.writeBool(seq.EnableRefFrameMVS); err != nil {
				return err
			}
		}
		if err := writeSequenceSelectOrBit(w, seq.SeqForceScreenContentTools); err != nil {
			return err
		}
		if seq.SeqForceScreenContentTools > 0 {
			if err := writeSequenceSelectOrBit(w, seq.SeqForceIntegerMV); err != nil {
				return err
			}
		}
		if seq.EnableOrderHint {
			if err := w.writeBits(uint64(seq.OrderHintBits-1), 3); err != nil {
				return err
			}
		}
	}

	if err := w.writeBool(seq.EnableSuperRes); err != nil {
		return err
	}
	if err := w.writeBool(seq.EnableCDEF); err != nil {
		return err
	}
	return w.writeBool(seq.EnableRestoration)
}

func writeSequenceSelectOrBit(w *bitWriter, value uint8) error {
	if value == SequenceSelectScreenContentTools {
		return w.writeBool(true)
	}
	if err := w.writeBool(false); err != nil {
		return err
	}
	return w.writeBits(uint64(value), 1)
}

func writeSequenceColorConfig(w *bitWriter, profile Profile, cfg SequenceColorConfig) error {
	bitDepth := sequenceColorBitDepth(cfg)
	highBitdepth := bitDepth > 8
	if err := w.writeBool(highBitdepth); err != nil {
		return err
	}
	if profile == Profile2 && highBitdepth {
		if err := w.writeBool(bitDepth == 12); err != nil {
			return err
		}
	}

	if profile != Profile1 {
		if err := w.writeBool(cfg.MonoChrome); err != nil {
			return err
		}
	}

	if err := w.writeBool(cfg.ColorDescriptionPresent); err != nil {
		return err
	}
	if cfg.ColorDescriptionPresent {
		if err := w.writeBits(uint64(cfg.ColorPrimaries), 8); err != nil {
			return err
		}
		if err := w.writeBits(uint64(cfg.TransferCharacteristics), 8); err != nil {
			return err
		}
		if err := w.writeBits(uint64(cfg.MatrixCoefficients), 8); err != nil {
			return err
		}
	}

	if cfg.MonoChrome {
		return w.writeBool(cfg.ColorRange)
	}

	if sequenceColorIsIdentity(cfg) {
		return w.writeBool(cfg.SeparateUVDeltaQ)
	}

	if err := w.writeBool(cfg.ColorRange); err != nil {
		return err
	}
	switch profile {
	case Profile2:
		if bitDepth == 12 {
			if err := w.writeBool(cfg.SubsamplingX); err != nil {
				return err
			}
			if cfg.SubsamplingX {
				if err := w.writeBool(cfg.SubsamplingY); err != nil {
					return err
				}
			}
		}
	}
	if cfg.SubsamplingX && cfg.SubsamplingY {
		if err := w.writeBits(uint64(cfg.ChromaSamplePosition), 2); err != nil {
			return err
		}
	}
	return w.writeBool(cfg.SeparateUVDeltaQ)
}

func validateSequenceHeader(seq SequenceHeader) error {
	if !seq.Profile.Valid() {
		return ErrInvalidConfig
	}
	if seq.ReducedStillPictureHeader && !seq.StillPicture {
		return ErrInvalidConfig
	}
	if seq.ReducedStillPictureHeader && (seq.TimingInfoPresent || seq.DecoderModelInfoPresent || seq.InitialDisplayDelayPresent) {
		return ErrInvalidConfig
	}
	if !seq.TimingInfoPresent && seq.TimingInfo != (SequenceTimingInfo{}) {
		return ErrInvalidConfig
	}
	if seq.TimingInfoPresent && !seq.TimingInfo.EqualPictureInterval && seq.TimingInfo.NumTicksPerPictureMinus1 != 0 {
		return ErrInvalidConfig
	}
	if seq.DecoderModelInfoPresent && !seq.TimingInfoPresent {
		return ErrInvalidConfig
	}
	if seq.DecoderModelInfoPresent && !validSequenceDecoderModelInfo(seq.DecoderModelInfo) {
		return ErrInvalidConfig
	}
	if !seq.DecoderModelInfoPresent && seq.DecoderModelInfo != (SequenceDecoderModelInfo{}) {
		return ErrInvalidConfig
	}
	opCount := sequenceOperatingPointCount(seq)
	if opCount == 0 || opCount > 32 {
		return ErrInvalidConfig
	}
	if seq.ReducedStillPictureHeader && opCount != 1 {
		return ErrInvalidConfig
	}
	for i := uint8(0); i < opCount; i++ {
		op := seq.OperatingPoints[i]
		if op.IDC > 0x0fff || !parser.ValidSequenceLevelIndex(op.SeqLevelIdx) || op.SeqTier > 1 {
			return ErrInvalidConfig
		}
		if op.SeqLevelIdx <= 7 && op.SeqTier != 0 {
			return ErrInvalidConfig
		}
		if op.DecoderModelPresent && !seq.DecoderModelInfoPresent {
			return ErrInvalidConfig
		}
		if op.DecoderModelPresent {
			if !valueFitsBits32(op.DecoderBufferDelay, seq.DecoderModelInfo.BufferDelayLength) ||
				!valueFitsBits32(op.EncoderBufferDelay, seq.DecoderModelInfo.BufferDelayLength) {
				return ErrInvalidConfig
			}
		} else if op.DecoderBufferDelay != 0 || op.EncoderBufferDelay != 0 || op.LowDelayMode {
			return ErrInvalidConfig
		}
		if op.InitialDisplayDelayPresent && !seq.InitialDisplayDelayPresent {
			return ErrInvalidConfig
		}
		if op.InitialDisplayDelayMinus1 > 0x0f {
			return ErrInvalidConfig
		}
		if !op.InitialDisplayDelayPresent && op.InitialDisplayDelayMinus1 != 0 {
			return ErrInvalidConfig
		}
	}
	if seq.MaxFrameWidth == 0 || seq.MaxFrameHeight == 0 || seq.MaxFrameWidth > 1<<16 || seq.MaxFrameHeight > 1<<16 {
		return ErrInvalidConfig
	}
	if seq.ReducedStillPictureHeader && seq.FrameIDNumbersPresent {
		return ErrInvalidConfig
	}
	if seq.FrameIDNumbersPresent {
		if seq.DeltaFrameIDLength < 2 || seq.DeltaFrameIDLength > 16 || seq.AdditionalFrameIDLength < 1 || seq.AdditionalFrameIDLength > 8 {
			return ErrInvalidConfig
		}
		if seq.DeltaFrameIDLength+seq.AdditionalFrameIDLength > 16 {
			return ErrInvalidConfig
		}
	} else if seq.DeltaFrameIDLength != 0 || seq.AdditionalFrameIDLength != 0 {
		return ErrInvalidConfig
	}
	if seq.SeqForceScreenContentTools > SequenceSelectScreenContentTools {
		return ErrInvalidConfig
	}
	if seq.SeqForceScreenContentTools == 0 && seq.SeqForceIntegerMV != 0 {
		return ErrInvalidConfig
	}
	if seq.SeqForceScreenContentTools > 0 && seq.SeqForceIntegerMV > SequenceSelectIntegerMV {
		return ErrInvalidConfig
	}
	if seq.EnableOrderHint {
		if seq.OrderHintBits == 0 || seq.OrderHintBits > 8 {
			return ErrInvalidConfig
		}
	} else if seq.EnableJNTComp || seq.EnableRefFrameMVS || seq.OrderHintBits != 0 {
		return ErrInvalidConfig
	}
	return validateSequenceColorConfig(seq.Profile, seq.ColorConfig)
}

func validSequenceDecoderModelInfo(info SequenceDecoderModelInfo) bool {
	return info.BufferDelayLength >= 1 && info.BufferDelayLength <= 32 &&
		info.BufferRemovalTimeLength >= 1 && info.BufferRemovalTimeLength <= 32 &&
		info.FramePresentationTimeLength >= 1 && info.FramePresentationTimeLength <= 32
}

func valueFitsBits32(value uint32, n uint8) bool {
	if n == 0 || n > 32 {
		return false
	}
	if n == 32 {
		return true
	}
	return value < (uint32(1) << n)
}

func validateSequenceColorConfig(profile Profile, cfg SequenceColorConfig) error {
	bitDepth := sequenceColorBitDepth(cfg)
	switch bitDepth {
	case 8, 10:
	case 12:
		if profile != Profile2 {
			return ErrInvalidConfig
		}
	default:
		return ErrInvalidConfig
	}
	if profile == Profile1 && cfg.MonoChrome {
		return ErrInvalidConfig
	}
	if cfg.MonoChrome {
		if cfg.ChromaSamplePosition != 0 || cfg.SeparateUVDeltaQ {
			return ErrInvalidConfig
		}
		return nil
	}
	if cfg.ChromaSamplePosition > 3 {
		return ErrInvalidConfig
	}
	if sequenceColorIsIdentity(cfg) {
		if !cfg.ColorRange || cfg.SubsamplingX || cfg.SubsamplingY || cfg.ChromaSamplePosition != 0 {
			return ErrInvalidConfig
		}
		return nil
	}
	switch profile {
	case Profile0:
		if !cfg.SubsamplingX || !cfg.SubsamplingY {
			return ErrInvalidConfig
		}
	case Profile1:
		if cfg.SubsamplingX || cfg.SubsamplingY {
			return ErrInvalidConfig
		}
	case Profile2:
		if bitDepth == 12 {
			if !cfg.SubsamplingX && cfg.SubsamplingY {
				return ErrInvalidConfig
			}
		} else if !cfg.SubsamplingX || cfg.SubsamplingY {
			return ErrInvalidConfig
		}
	}
	if !cfg.SubsamplingX || !cfg.SubsamplingY {
		if cfg.ChromaSamplePosition != 0 {
			return ErrInvalidConfig
		}
	}
	return nil
}

func sequenceColorBitDepth(cfg SequenceColorConfig) uint8 {
	if cfg.BitDepth == 0 {
		return 8
	}
	return cfg.BitDepth
}

func sequenceColorIsIdentity(cfg SequenceColorConfig) bool {
	return cfg.ColorDescriptionPresent &&
		cfg.ColorPrimaries == SequenceColorPrimariesBT709 &&
		cfg.TransferCharacteristics == SequenceTransferCharacteristicsSRGB &&
		cfg.MatrixCoefficients == SequenceMatrixCoefficientsIdentity
}

func sequenceOperatingPointCount(seq SequenceHeader) uint8 {
	if seq.OperatingPointsCount == 0 {
		return 1
	}
	return seq.OperatingPointsCount
}

func sequenceFrameDimensionBits(value uint32) uint8 {
	n := bits.Len32(value - 1)
	if n == 0 {
		return 1
	}
	return uint8(n)
}
