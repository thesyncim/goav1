package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// SkipModeParams is the encoder-owned view of skip_mode_params().
type SkipModeParams struct {
	Allowed     bool
	Enabled     bool
	RefFrameIdx [2]uint8
}

func SkipModeParamsPayloadSize(seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, transformRef TransformReferenceParams, params SkipModeParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeSkipModeParamsPayload(&w, seq, prefix, size, refs, transformRef, params); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendSkipModeParamsPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, transformRef TransformReferenceParams, params SkipModeParams) ([]byte, error) {
	payloadSize, err := SkipModeParamsPayloadSize(seq, prefix, size, refs, transformRef, params)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeSkipModeParamsPayload(&w, seq, prefix, size, refs, transformRef, params); err != nil {
		return dst, err
	}
	return out, nil
}

func writeSkipModeParamsPayload(w *bitWriter, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, transformRef TransformReferenceParams, params SkipModeParams) error {
	if err := validateSkipModeParams(seq, prefix, size, refs, transformRef, params); err != nil {
		return err
	}
	if params.Allowed {
		return w.writeBool(params.Enabled)
	}
	return nil
}

func validateSkipModeParams(seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, transformRef TransformReferenceParams, params SkipModeParams) error {
	if err := validateSequenceHeader(seq); err != nil {
		return err
	}
	if !prefix.FrameType.valid() || transformRef.ReferenceMode > ReferenceModeSelect {
		return ErrInvalidFrame
	}
	if !skipModeMayBeAllowed(seq, prefix, transformRef) {
		if params.Allowed || params.Enabled || params.RefFrameIdx != [2]uint8{} {
			return ErrInvalidFrame
		}
		return nil
	}
	if refs == nil {
		return ErrInvalidFrame
	}
	allowed, refFrameIdx, err := deriveEncoderSkipModeRefs(seq, prefix, size, refs)
	if err != nil {
		return err
	}
	if !allowed {
		if params.Allowed || params.Enabled || params.RefFrameIdx != [2]uint8{} {
			return ErrInvalidFrame
		}
		return nil
	}
	if !params.Allowed || params.RefFrameIdx != refFrameIdx {
		return ErrInvalidFrame
	}
	return nil
}

func skipModeMayBeAllowed(seq SequenceHeader, prefix FrameHeaderPrefix, transformRef TransformReferenceParams) bool {
	return seq.EnableOrderHint && prefix.FrameType.interOrSwitch() && transformRef.ReferenceMode != ReferenceModeSingle
}

func deriveEncoderSkipModeRefs(seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState) (bool, [2]uint8, error) {
	beforeIdx := int8(-1)
	afterIdx := int8(-1)
	beforeDiff := int16(0)
	afterDiff := int16(0)

	for i := uint8(0); i < parser.InterRefsPerFrame; i++ {
		slot := size.RefFrameIdx[i]
		if slot >= parser.RefFrames {
			return false, [2]uint8{}, ErrInvalidFrame
		}
		ref := refs.Frames[slot]
		if !ref.Valid {
			return false, [2]uint8{}, ErrInvalidFrame
		}
		diff := encoderRelativeOrderHint(seq.OrderHintBits, ref.OrderHint, prefix.OrderHint)
		if diff > 0 {
			if afterIdx < 0 || diff < afterDiff {
				afterIdx = int8(i)
				afterDiff = diff
			}
			continue
		}
		if diff < 0 && (beforeIdx < 0 || diff > beforeDiff) {
			beforeIdx = int8(i)
			beforeDiff = diff
		}
	}

	if beforeIdx >= 0 && afterIdx >= 0 {
		return true, sortedSkipModeRefIdx(uint8(beforeIdx), uint8(afterIdx)), nil
	}
	if beforeIdx < 0 {
		return false, [2]uint8{}, nil
	}

	before2Idx := int8(-1)
	before2Diff := int16(0)
	for i := uint8(0); i < parser.InterRefsPerFrame; i++ {
		ref := refs.Frames[size.RefFrameIdx[i]]
		diff := encoderRelativeOrderHint(seq.OrderHintBits, ref.OrderHint, prefix.OrderHint)
		if diff >= beforeDiff {
			continue
		}
		if before2Idx < 0 || diff > before2Diff {
			before2Idx = int8(i)
			before2Diff = diff
		}
	}
	if before2Idx < 0 {
		return false, [2]uint8{}, nil
	}
	return true, sortedSkipModeRefIdx(uint8(beforeIdx), uint8(before2Idx)), nil
}

func sortedSkipModeRefIdx(a uint8, b uint8) [2]uint8 {
	if a < b {
		return [2]uint8{a, b}
	}
	return [2]uint8{b, a}
}

func encoderRelativeOrderHint(bits uint8, a uint8, b uint8) int16 {
	if bits == 0 {
		return 0
	}
	mask := int32(1 << uint(bits-1))
	diff := int32(uint32(a)) - int32(uint32(b))
	return int16((diff & (mask - 1)) - (diff & mask))
}
