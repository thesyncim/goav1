package encoder

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const EncoderPrimaryRefNone uint8 = 7

type FrameHeaderType uint8

const (
	FrameHeaderTypeKey FrameHeaderType = iota
	FrameHeaderTypeInter
	FrameHeaderTypeIntraOnly
	FrameHeaderTypeSwitch
)

func (t FrameHeaderType) valid() bool {
	return t <= FrameHeaderTypeSwitch
}

func (t FrameHeaderType) keyOrIntra() bool {
	return t == FrameHeaderTypeKey || t == FrameHeaderTypeIntraOnly
}

func (t FrameHeaderType) interOrSwitch() bool {
	return t == FrameHeaderTypeInter || t == FrameHeaderTypeSwitch
}

// FrameHeaderPrefix is the encoder-owned, syntax-width view of the initial
// uncompressed_header() fields up to primary_ref_frame.
type FrameHeaderPrefix struct {
	ShowExistingFrame bool
	ExistingFrameIdx  uint8

	FrameType     FrameHeaderType
	ShowFrame     bool
	ShowableFrame bool

	ErrorResilientMode bool
	DisableCDFUpdate   bool

	AllowScreenContentTools bool
	ForceIntegerMV          bool

	FramePresentationDelay uint32
	FrameID                uint16
	FrameSizeOverride      bool
	OrderHint              uint8
	PrimaryRefFrame        uint8
}

// FrameHeaderPrefixPayloadSize returns the byte count for the emitted
// uncompressed_header() prefix. The payload is intentionally not byte-aligned:
// later frame-header syntax continues from this bit position.
func FrameHeaderPrefixPayloadSize(seq SequenceHeader, prefix FrameHeaderPrefix) (int, error) {
	w := newSizingBitWriter()
	if err := writeFrameHeaderPrefixPayload(&w, seq, prefix); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

// AppendFrameHeaderPrefixPayload appends the initial uncompressed_header()
// fields into dst without growing it. The result is a frame-header prefix, not
// a complete frame_header_obu payload.
func AppendFrameHeaderPrefixPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix) ([]byte, error) {
	size, err := FrameHeaderPrefixPayloadSize(seq, prefix)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	w := newBitWriter(out[off:])
	if err := writeFrameHeaderPrefixPayload(&w, seq, prefix); err != nil {
		return dst, err
	}
	return out, nil
}

func writeFrameHeaderPrefixPayload(w *bitWriter, seq SequenceHeader, prefix FrameHeaderPrefix) error {
	if err := validateFrameHeaderPrefix(seq, prefix); err != nil {
		return err
	}

	if !seq.ReducedStillPictureHeader {
		if err := w.writeBool(prefix.ShowExistingFrame); err != nil {
			return err
		}
		if prefix.ShowExistingFrame {
			return writeShowExistingFrameHeaderPrefix(w, seq, prefix)
		}

		if err := w.writeBits(uint64(prefix.FrameType), 2); err != nil {
			return err
		}
		if err := w.writeBool(prefix.ShowFrame); err != nil {
			return err
		}
		if prefix.ShowFrame {
			if err := writeFramePresentationDelay(w, seq, prefix); err != nil {
				return err
			}
		} else {
			if err := w.writeBool(prefix.ShowableFrame); err != nil {
				return err
			}
		}
		if prefix.FrameType != FrameHeaderTypeSwitch && !(prefix.FrameType == FrameHeaderTypeKey && prefix.ShowFrame) {
			if err := w.writeBool(prefix.ErrorResilientMode); err != nil {
				return err
			}
		}
	}

	if err := writeFrameHeaderFeatureFlags(w, seq, prefix); err != nil {
		return err
	}
	return writeFrameHeaderReferencePrefix(w, seq, prefix)
}

func writeShowExistingFrameHeaderPrefix(w *bitWriter, seq SequenceHeader, prefix FrameHeaderPrefix) error {
	if err := w.writeBits(uint64(prefix.ExistingFrameIdx), 3); err != nil {
		return err
	}
	if err := writeFramePresentationDelay(w, seq, prefix); err != nil {
		return err
	}
	if seq.FrameIDNumbersPresent {
		return w.writeBits(uint64(prefix.FrameID), sequenceFrameIDBits(seq))
	}
	return nil
}

func writeFrameHeaderFeatureFlags(w *bitWriter, seq SequenceHeader, prefix FrameHeaderPrefix) error {
	if err := w.writeBool(prefix.DisableCDFUpdate); err != nil {
		return err
	}
	if seq.SeqForceScreenContentTools == SequenceSelectScreenContentTools {
		if err := w.writeBool(prefix.AllowScreenContentTools); err != nil {
			return err
		}
	}
	if prefix.AllowScreenContentTools && seq.SeqForceIntegerMV == SequenceSelectIntegerMV {
		return w.writeBool(prefix.ForceIntegerMV)
	}
	return nil
}

func writeFramePresentationDelay(w *bitWriter, seq SequenceHeader, prefix FrameHeaderPrefix) error {
	if !seq.DecoderModelInfoPresent || seq.TimingInfo.EqualPictureInterval {
		return nil
	}
	return w.writeBits(uint64(prefix.FramePresentationDelay), seq.DecoderModelInfo.FramePresentationTimeLength)
}

func writeFrameHeaderReferencePrefix(w *bitWriter, seq SequenceHeader, prefix FrameHeaderPrefix) error {
	if seq.ReducedStillPictureHeader {
		return nil
	}
	if seq.FrameIDNumbersPresent {
		if err := w.writeBits(uint64(prefix.FrameID), sequenceFrameIDBits(seq)); err != nil {
			return err
		}
	}
	if prefix.FrameType != FrameHeaderTypeSwitch {
		if err := w.writeBool(prefix.FrameSizeOverride); err != nil {
			return err
		}
	}
	if seq.EnableOrderHint {
		if err := w.writeBits(uint64(prefix.OrderHint), seq.OrderHintBits); err != nil {
			return err
		}
	}
	if !prefix.ErrorResilientMode && prefix.FrameType.interOrSwitch() {
		return w.writeBits(uint64(prefix.PrimaryRefFrame), 3)
	}
	return nil
}

func validateFrameHeaderPrefix(seq SequenceHeader, prefix FrameHeaderPrefix) error {
	if err := validateSequenceHeader(seq); err != nil {
		return err
	}
	if !prefix.FrameType.valid() || prefix.ExistingFrameIdx > 7 || prefix.PrimaryRefFrame > EncoderPrimaryRefNone {
		return ErrInvalidFrame
	}
	presentationDelayEmitted := prefix.ShowExistingFrame || prefix.ShowFrame
	if seq.DecoderModelInfoPresent && !seq.TimingInfo.EqualPictureInterval && presentationDelayEmitted {
		if !valueFitsBits32(prefix.FramePresentationDelay, seq.DecoderModelInfo.FramePresentationTimeLength) {
			return ErrInvalidFrame
		}
	} else if prefix.FramePresentationDelay != 0 {
		return ErrInvalidFrame
	}
	if seq.ReducedStillPictureHeader && prefix.ShowExistingFrame {
		return ErrInvalidFrame
	}
	if seq.StillPicture && !seq.ReducedStillPictureHeader && (prefix.FrameType != FrameHeaderTypeKey || !prefix.ShowFrame) {
		return ErrInvalidFrame
	}
	if prefix.ShowExistingFrame {
		return validateShowExistingFrameHeaderPrefix(seq, prefix)
	}
	if seq.ReducedStillPictureHeader {
		if prefix.FrameType != FrameHeaderTypeKey || !prefix.ShowFrame || prefix.ShowableFrame || !prefix.ErrorResilientMode {
			return ErrInvalidFrame
		}
	} else {
		if prefix.ShowFrame && prefix.ShowableFrame != (prefix.FrameType != FrameHeaderTypeKey) {
			return ErrInvalidFrame
		}
		if (prefix.FrameType == FrameHeaderTypeSwitch || (prefix.FrameType == FrameHeaderTypeKey && prefix.ShowFrame)) && !prefix.ErrorResilientMode {
			return ErrInvalidFrame
		}
	}

	if err := validateFrameHeaderFeatureFlags(seq, prefix); err != nil {
		return err
	}
	if seq.FrameIDNumbersPresent && uint32(prefix.FrameID) >= uint32(1)<<sequenceFrameIDBits(seq) {
		return ErrInvalidFrame
	}
	if prefix.FrameType == FrameHeaderTypeSwitch {
		if !prefix.FrameSizeOverride {
			return ErrInvalidFrame
		}
	} else if seq.ReducedStillPictureHeader && prefix.FrameSizeOverride {
		return ErrInvalidFrame
	}
	if seq.EnableOrderHint {
		if uint16(prefix.OrderHint) >= uint16(1)<<seq.OrderHintBits {
			return ErrInvalidFrame
		}
	} else if prefix.OrderHint != 0 {
		return ErrInvalidFrame
	}
	if prefix.ErrorResilientMode || !prefix.FrameType.interOrSwitch() {
		if prefix.PrimaryRefFrame != EncoderPrimaryRefNone {
			return ErrInvalidFrame
		}
	}
	return nil
}

func validateShowExistingFrameHeaderPrefix(seq SequenceHeader, prefix FrameHeaderPrefix) error {
	if prefix.FrameType != 0 || prefix.ShowFrame || prefix.ShowableFrame || prefix.ErrorResilientMode ||
		prefix.DisableCDFUpdate || prefix.AllowScreenContentTools || prefix.ForceIntegerMV ||
		prefix.FrameSizeOverride || prefix.OrderHint != 0 || prefix.PrimaryRefFrame != EncoderPrimaryRefNone {
		return ErrInvalidFrame
	}
	if seq.FrameIDNumbersPresent && uint32(prefix.FrameID) >= uint32(1)<<sequenceFrameIDBits(seq) {
		return ErrInvalidFrame
	}
	if !seq.FrameIDNumbersPresent && prefix.FrameID != 0 {
		return ErrInvalidFrame
	}
	return nil
}

func validateFrameHeaderFeatureFlags(seq SequenceHeader, prefix FrameHeaderPrefix) error {
	if seq.SeqForceScreenContentTools != SequenceSelectScreenContentTools &&
		prefix.AllowScreenContentTools != (seq.SeqForceScreenContentTools != 0) {
		return ErrInvalidFrame
	}
	forceInteger := prefix.ForceIntegerMV
	if prefix.FrameType.keyOrIntra() {
		forceInteger = true
	}
	if prefix.FrameType.keyOrIntra() && !prefix.ForceIntegerMV {
		return ErrInvalidFrame
	}
	if !prefix.AllowScreenContentTools && !prefix.FrameType.keyOrIntra() && prefix.ForceIntegerMV {
		return ErrInvalidFrame
	}
	if prefix.AllowScreenContentTools && seq.SeqForceIntegerMV != SequenceSelectIntegerMV &&
		forceInteger != (seq.SeqForceIntegerMV != 0) {
		return ErrInvalidFrame
	}
	return nil
}

func sequenceFrameIDBits(seq SequenceHeader) uint8 {
	if !seq.FrameIDNumbersPresent {
		return 0
	}
	return seq.DeltaFrameIDLength + seq.AdditionalFrameIDLength
}
