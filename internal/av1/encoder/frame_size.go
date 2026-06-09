package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

const encoderRefFrames uint8 = 8

// IntraFrameSize is the encoder-owned, syntax-width view of the frame-size
// fields reachable from key and intra-only frames.
type IntraFrameSize struct {
	UpscaledWidth uint32
	Height        uint32
	RenderWidth   uint32
	RenderHeight  uint32

	SuperResDenominator uint8
	HaveRenderSize      bool
	AllowIntrabc        bool

	RefreshFrameFlags uint8
	RefOrderHints     [8]uint8
}

// InterFrameSize is the encoder-owned, syntax-width view of the frame-size and
// reference-index fields reachable from inter and switch frames.
type InterFrameSize struct {
	UpscaledWidth uint32
	Height        uint32
	RenderWidth   uint32
	RenderHeight  uint32

	SuperResDenominator uint8
	HaveRenderSize      bool

	RefreshFrameFlags uint8
	RefOrderHints     [8]uint8
	RefFrameIdx       [7]uint8
}

// FrameHeaderIntraPayloadSize returns the byte count for a frame-header payload
// through frame_size() on the key/intra-only path.
func FrameHeaderIntraPayloadSize(seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize) (int, error) {
	w := newSizingBitWriter()
	if err := writeFrameHeaderPrefixPayload(&w, seq, prefix); err != nil {
		return 0, err
	}
	if err := writeIntraFrameSizePayload(&w, seq, prefix, size); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

// AppendFrameHeaderIntraPayload appends the frame-header payload through
// frame_size() for key/intra-only frames. It writes bits contiguously from the
// frame-header prefix into the frame-size syntax and never grows dst.
func AppendFrameHeaderIntraPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize) ([]byte, error) {
	payloadSize, err := FrameHeaderIntraPayloadSize(seq, prefix, size)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeFrameHeaderPrefixPayload(&w, seq, prefix); err != nil {
		return dst, err
	}
	if err := writeIntraFrameSizePayload(&w, seq, prefix, size); err != nil {
		return dst, err
	}
	return out, nil
}

// FrameHeaderInterPayloadSize returns the byte count for a frame-header payload
// through frame_size() on the inter/switch path. It currently emits full
// reference-index signaling and direct frame dimensions.
func FrameHeaderInterPayloadSize(seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize) (int, error) {
	w := newSizingBitWriter()
	if err := writeFrameHeaderPrefixPayload(&w, seq, prefix); err != nil {
		return 0, err
	}
	if err := writeInterFrameSizePayload(&w, seq, prefix, size); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

// AppendFrameHeaderInterPayload appends the frame-header payload through
// reference-index and frame_size() syntax for inter/switch frames. It writes
// bits contiguously from the prefix and never grows dst.
func AppendFrameHeaderInterPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize) ([]byte, error) {
	payloadSize, err := FrameHeaderInterPayloadSize(seq, prefix, size)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeFrameHeaderPrefixPayload(&w, seq, prefix); err != nil {
		return dst, err
	}
	if err := writeInterFrameSizePayload(&w, seq, prefix, size); err != nil {
		return dst, err
	}
	return out, nil
}

// LowOverheadFrameHeaderIntraOBUSize returns the exact size of a low-overhead
// frame-header OBU for the key/intra-only path through frame_size().
func LowOverheadFrameHeaderIntraOBUSize(seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize) (int, error) {
	payloadSize, err := FrameHeaderIntraPayloadSize(seq, prefix, size)
	if err != nil {
		return 0, err
	}
	return 1 + bitstream.LEB128Len(uint32(payloadSize)) + payloadSize, nil
}

// LowOverheadFrameHeaderInterOBUSize returns the exact size of a low-overhead
// frame-header OBU for the inter/switch path through frame_size().
func LowOverheadFrameHeaderInterOBUSize(seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize) (int, error) {
	payloadSize, err := FrameHeaderInterPayloadSize(seq, prefix, size)
	if err != nil {
		return 0, err
	}
	return 1 + bitstream.LEB128Len(uint32(payloadSize)) + payloadSize, nil
}

// AppendLowOverheadFrameHeaderIntraOBU appends one low-overhead frame-header
// OBU for the key/intra-only path through frame_size(). It validates and sizes
// before writing, so errors leave dst length unchanged.
func AppendLowOverheadFrameHeaderIntraOBU(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize) ([]byte, error) {
	payloadSize, err := FrameHeaderIntraPayloadSize(seq, prefix, size)
	if err != nil {
		return dst, err
	}
	obuSize := 1 + bitstream.LEB128Len(uint32(payloadSize)) + payloadSize
	if cap(dst)-len(dst) < obuSize {
		return dst, bitstream.ErrShortBuffer
	}

	off := len(dst)
	out := dst[:off+obuSize]
	n, err := obu.PutHeader(out[off:], obu.Header{
		Type:         obu.TypeFrameHeader,
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
	if err := writeFrameHeaderPrefixPayload(&w, seq, prefix); err != nil {
		return dst, err
	}
	if err := writeIntraFrameSizePayload(&w, seq, prefix, size); err != nil {
		return dst, err
	}
	return out, nil
}

// AppendLowOverheadFrameHeaderInterOBU appends one low-overhead frame-header
// OBU for the inter/switch path through frame_size(). It validates and sizes
// before writing, so errors leave dst length unchanged.
func AppendLowOverheadFrameHeaderInterOBU(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize) ([]byte, error) {
	payloadSize, err := FrameHeaderInterPayloadSize(seq, prefix, size)
	if err != nil {
		return dst, err
	}
	obuSize := 1 + bitstream.LEB128Len(uint32(payloadSize)) + payloadSize
	if cap(dst)-len(dst) < obuSize {
		return dst, bitstream.ErrShortBuffer
	}

	off := len(dst)
	out := dst[:off+obuSize]
	n, err := obu.PutHeader(out[off:], obu.Header{
		Type:         obu.TypeFrameHeader,
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
	if err := writeFrameHeaderPrefixPayload(&w, seq, prefix); err != nil {
		return dst, err
	}
	if err := writeInterFrameSizePayload(&w, seq, prefix, size); err != nil {
		return dst, err
	}
	return out, nil
}

// LowOverheadIntraHeaderTemporalUnitSize returns the exact size of one
// low-overhead temporal unit carrying a temporal delimiter, sequence header,
// and key/intra-only frame-header OBU through frame_size().
func LowOverheadIntraHeaderTemporalUnitSize(seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize) (int, error) {
	seqSize, err := LowOverheadSequenceHeaderOBUSize(seq)
	if err != nil {
		return 0, err
	}
	frameSize, err := LowOverheadFrameHeaderIntraOBUSize(seq, prefix, size)
	if err != nil {
		return 0, err
	}
	return lowOverheadOBUSizeUnchecked(OBU{Type: obu.TypeTemporalDelimiter}) + seqSize + frameSize, nil
}

// AppendLowOverheadIntraHeaderTemporalUnit appends one low-overhead temporal
// unit carrying a temporal delimiter, sequence header, and key/intra-only
// frame-header OBU. It validates and sizes before writing, so errors leave dst
// length unchanged.
func AppendLowOverheadIntraHeaderTemporalUnit(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize) ([]byte, error) {
	unitSize, err := LowOverheadIntraHeaderTemporalUnitSize(seq, prefix, size)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < unitSize {
		return dst, bitstream.ErrShortBuffer
	}

	out, err := AppendLowOverheadOBU(dst, OBU{Type: obu.TypeTemporalDelimiter})
	if err != nil {
		return dst, err
	}
	out, err = AppendLowOverheadSequenceHeaderOBU(out, seq)
	if err != nil {
		return dst, err
	}
	out, err = AppendLowOverheadFrameHeaderIntraOBU(out, seq, prefix, size)
	if err != nil {
		return dst, err
	}
	return out, nil
}

func writeIntraFrameSizePayload(w *bitWriter, seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize) error {
	if err := validateIntraFrameSize(seq, prefix, size); err != nil {
		return err
	}

	if (prefix.FrameType != FrameHeaderTypeKey || !prefix.ShowFrame) && prefix.FrameType != FrameHeaderTypeSwitch {
		if err := w.writeBits(uint64(size.RefreshFrameFlags), 8); err != nil {
			return err
		}
	}
	if frameSizeReadsRefOrderHints(seq, prefix, size) {
		for i := uint8(0); i < encoderRefFrames; i++ {
			if err := w.writeBits(uint64(size.RefOrderHints[i]), seq.OrderHintBits); err != nil {
				return err
			}
		}
	}

	if prefix.FrameSizeOverride {
		if err := w.writeBits(uint64(size.UpscaledWidth-1), sequenceFrameDimensionBits(seq.MaxFrameWidth)); err != nil {
			return err
		}
		if err := w.writeBits(uint64(size.Height-1), sequenceFrameDimensionBits(seq.MaxFrameHeight)); err != nil {
			return err
		}
	}
	if err := writeFrameSuperResScale(w, seq, size); err != nil {
		return err
	}
	if err := writeFrameRenderSize(w, size); err != nil {
		return err
	}
	if prefix.AllowScreenContentTools && size.SuperResDenominator == 8 {
		return w.writeBool(size.AllowIntrabc)
	}
	return nil
}

func writeInterFrameSizePayload(w *bitWriter, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize) error {
	if err := validateInterFrameSize(seq, prefix, size); err != nil {
		return err
	}

	if prefix.FrameType != FrameHeaderTypeSwitch {
		if err := w.writeBits(uint64(size.RefreshFrameFlags), 8); err != nil {
			return err
		}
	}
	if frameInterSizeReadsRefOrderHints(seq, prefix) {
		for i := uint8(0); i < encoderRefFrames; i++ {
			if err := w.writeBits(uint64(size.RefOrderHints[i]), seq.OrderHintBits); err != nil {
				return err
			}
		}
	}
	if seq.EnableOrderHint {
		if err := w.writeBool(false); err != nil {
			return err
		}
	}
	for i := range size.RefFrameIdx {
		if err := w.writeBits(uint64(size.RefFrameIdx[i]), 3); err != nil {
			return err
		}
	}
	if prefix.FrameSizeOverride {
		if err := w.writeBits(uint64(size.UpscaledWidth-1), sequenceFrameDimensionBits(seq.MaxFrameWidth)); err != nil {
			return err
		}
		if err := w.writeBits(uint64(size.Height-1), sequenceFrameDimensionBits(seq.MaxFrameHeight)); err != nil {
			return err
		}
	}
	if err := writeInterFrameSuperResScale(w, seq, size); err != nil {
		return err
	}
	return writeInterFrameRenderSize(w, size)
}

func writeFrameSuperResScale(w *bitWriter, seq SequenceHeader, size IntraFrameSize) error {
	if !seq.EnableSuperRes {
		return nil
	}
	if size.SuperResDenominator == 8 {
		return w.writeBool(false)
	}
	if err := w.writeBool(true); err != nil {
		return err
	}
	return w.writeBits(uint64(size.SuperResDenominator-9), 3)
}

func writeInterFrameSuperResScale(w *bitWriter, seq SequenceHeader, size InterFrameSize) error {
	if !seq.EnableSuperRes {
		return nil
	}
	if size.SuperResDenominator == 8 {
		return w.writeBool(false)
	}
	if err := w.writeBool(true); err != nil {
		return err
	}
	return w.writeBits(uint64(size.SuperResDenominator-9), 3)
}

func writeFrameRenderSize(w *bitWriter, size IntraFrameSize) error {
	if err := w.writeBool(size.HaveRenderSize); err != nil {
		return err
	}
	if !size.HaveRenderSize {
		return nil
	}
	if err := w.writeBits(uint64(size.RenderWidth-1), 16); err != nil {
		return err
	}
	return w.writeBits(uint64(size.RenderHeight-1), 16)
}

func writeInterFrameRenderSize(w *bitWriter, size InterFrameSize) error {
	if err := w.writeBool(size.HaveRenderSize); err != nil {
		return err
	}
	if !size.HaveRenderSize {
		return nil
	}
	if err := w.writeBits(uint64(size.RenderWidth-1), 16); err != nil {
		return err
	}
	return w.writeBits(uint64(size.RenderHeight-1), 16)
}

func validateIntraFrameSize(seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize) error {
	if err := validateFrameHeaderPrefix(seq, prefix); err != nil {
		return err
	}
	if prefix.ShowExistingFrame || !prefix.FrameType.keyOrIntra() {
		return ErrInvalidFrame
	}
	if size.UpscaledWidth == 0 || size.Height == 0 || size.UpscaledWidth > seq.MaxFrameWidth || size.Height > seq.MaxFrameHeight {
		return ErrInvalidFrame
	}
	if !prefix.FrameSizeOverride && (size.UpscaledWidth != seq.MaxFrameWidth || size.Height != seq.MaxFrameHeight) {
		return ErrInvalidFrame
	}
	if prefix.FrameType == FrameHeaderTypeKey && prefix.ShowFrame {
		if size.RefreshFrameFlags != 0xff {
			return ErrInvalidFrame
		}
	} else {
		if prefix.FrameType == FrameHeaderTypeIntraOnly && size.RefreshFrameFlags == 0xff {
			return ErrInvalidFrame
		}
	}
	if !frameSizeReadsRefOrderHints(seq, prefix, size) {
		for i := uint8(0); i < encoderRefFrames; i++ {
			if size.RefOrderHints[i] != 0 {
				return ErrInvalidFrame
			}
		}
	}
	if err := validateFrameSuperRes(seq, size); err != nil {
		return err
	}
	if err := validateFrameRenderSize(size); err != nil {
		return err
	}
	if (!prefix.AllowScreenContentTools || size.SuperResDenominator != 8) && size.AllowIntrabc {
		return ErrInvalidFrame
	}
	return nil
}

func validateInterFrameSize(seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize) error {
	if err := validateFrameHeaderPrefix(seq, prefix); err != nil {
		return err
	}
	if prefix.ShowExistingFrame || !prefix.FrameType.interOrSwitch() {
		return ErrInvalidFrame
	}
	if prefix.FrameType == FrameHeaderTypeSwitch {
		if size.RefreshFrameFlags != 0xff {
			return ErrInvalidFrame
		}
	}
	// refresh_frame_flags may be zero for inter frames that no later frame
	// references (temporal-layer droppable frames).
	if !frameInterSizeReadsRefOrderHints(seq, prefix) {
		for i := uint8(0); i < encoderRefFrames; i++ {
			if size.RefOrderHints[i] != 0 {
				return ErrInvalidFrame
			}
		}
	}
	for i := range size.RefFrameIdx {
		if size.RefFrameIdx[i] >= encoderRefFrames {
			return ErrInvalidFrame
		}
	}
	if size.UpscaledWidth == 0 || size.Height == 0 || size.UpscaledWidth > seq.MaxFrameWidth || size.Height > seq.MaxFrameHeight {
		return ErrInvalidFrame
	}
	if !prefix.FrameSizeOverride && (size.UpscaledWidth != seq.MaxFrameWidth || size.Height != seq.MaxFrameHeight) {
		return ErrInvalidFrame
	}
	if err := validateInterFrameSuperRes(seq, size); err != nil {
		return err
	}
	return validateInterFrameRenderSize(size)
}

func validateFrameSuperRes(seq SequenceHeader, size IntraFrameSize) error {
	if size.SuperResDenominator == 0 {
		return ErrInvalidFrame
	}
	if !seq.EnableSuperRes && size.SuperResDenominator != 8 {
		return ErrInvalidFrame
	}
	if size.SuperResDenominator != 8 && (size.SuperResDenominator < 9 || size.SuperResDenominator > 16) {
		return ErrInvalidFrame
	}
	return nil
}

func validateInterFrameSuperRes(seq SequenceHeader, size InterFrameSize) error {
	if size.SuperResDenominator == 0 {
		return ErrInvalidFrame
	}
	if !seq.EnableSuperRes && size.SuperResDenominator != 8 {
		return ErrInvalidFrame
	}
	if size.SuperResDenominator != 8 && (size.SuperResDenominator < 9 || size.SuperResDenominator > 16) {
		return ErrInvalidFrame
	}
	return nil
}

func validateFrameRenderSize(size IntraFrameSize) error {
	if size.HaveRenderSize {
		if size.RenderWidth == 0 || size.RenderHeight == 0 || size.RenderWidth > 1<<16 || size.RenderHeight > 1<<16 {
			return ErrInvalidFrame
		}
		return nil
	}
	if (size.RenderWidth != 0 && size.RenderWidth != size.UpscaledWidth) ||
		(size.RenderHeight != 0 && size.RenderHeight != size.Height) {
		return ErrInvalidFrame
	}
	return nil
}

func validateInterFrameRenderSize(size InterFrameSize) error {
	if size.HaveRenderSize {
		if size.RenderWidth == 0 || size.RenderHeight == 0 || size.RenderWidth > 1<<16 || size.RenderHeight > 1<<16 {
			return ErrInvalidFrame
		}
		return nil
	}
	if (size.RenderWidth != 0 && size.RenderWidth != size.UpscaledWidth) ||
		(size.RenderHeight != 0 && size.RenderHeight != size.Height) {
		return ErrInvalidFrame
	}
	return nil
}

func frameSizeReadsRefOrderHints(seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize) bool {
	return prefix.ErrorResilientMode && seq.EnableOrderHint &&
		(prefix.FrameType.interOrSwitch() || size.RefreshFrameFlags != 0xff)
}

func frameInterSizeReadsRefOrderHints(seq SequenceHeader, prefix FrameHeaderPrefix) bool {
	return prefix.ErrorResilientMode && seq.EnableOrderHint && prefix.FrameType.interOrSwitch()
}
