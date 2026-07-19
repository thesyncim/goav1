package encoder

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const MaxSegments = 8

// SegmentData is one segment's frame-header feature data.
type SegmentData struct {
	DeltaQ    int16
	DeltaLFYV int16
	DeltaLFYH int16
	DeltaLFU  int16
	DeltaLFV  int16
	// RefFrame is -1 when the SEG_LVL_REF_FRAME feature is disabled.
	RefFrame int8
	Skip     bool
	GlobalMV bool
}

// SegmentationData is the reference-copyable segmentation feature state.
type SegmentationData struct {
	Segments [MaxSegments]SegmentData
}

// SegmentationParams is the encoder-owned view of segmentation_params().
type SegmentationParams struct {
	Enabled        bool
	UpdateMap      bool
	TemporalUpdate bool
	UpdateData     bool
	Data           SegmentationData
}

func SegmentationParamsPayloadSize(prefix FrameHeaderPrefix, seg SegmentationParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeSegmentationParamsPayload(&w, prefix, seg); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendSegmentationParamsPayload(dst []byte, prefix FrameHeaderPrefix, seg SegmentationParams) ([]byte, error) {
	payloadSize, err := SegmentationParamsPayloadSize(prefix, seg)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeSegmentationParamsPayload(&w, prefix, seg); err != nil {
		return dst, err
	}
	return out, nil
}

func writeSegmentationParamsPayload(w *bitWriter, prefix FrameHeaderPrefix, seg SegmentationParams) error {
	// When the per-segment feature data is not emitted (segmentation disabled or
	// carried over without an update), the encoder's inactive state is the
	// SEG_LVL_REF_FRAME disabled sentinel (RefFrame == -1), not the raw Go
	// zero value. Normalising the default here lets a zero-value SegmentationData
	// stand in for a disabled segmentation while segmentationDataNonZero treats
	// any enabled ref (RefFrame >= 0) as active. Data is never written in this
	// branch, so the produced bitstream is unchanged.
	if !seg.Enabled || !seg.UpdateData {
		disableInactiveSegmentRefFrames(&seg.Data)
	}
	if err := validateSegmentationParams(prefix, seg); err != nil {
		return err
	}
	if err := w.writeBool(seg.Enabled); err != nil {
		return err
	}
	if !seg.Enabled {
		return nil
	}
	if prefix.PrimaryRefFrame != EncoderPrimaryRefNone {
		if err := w.writeBool(seg.UpdateMap); err != nil {
			return err
		}
		if seg.UpdateMap {
			if err := w.writeBool(seg.TemporalUpdate); err != nil {
				return err
			}
		}
		if err := w.writeBool(seg.UpdateData); err != nil {
			return err
		}
	}
	if !seg.UpdateData {
		return nil
	}
	return writeSegmentationData(w, seg.Data)
}

func writeSegmentationData(w *bitWriter, data SegmentationData) error {
	for i := uint8(0); i < MaxSegments; i++ {
		seg := data.Segments[i]
		if err := writeSegmentSignedFeature(w, seg.DeltaQ, 9); err != nil {
			return err
		}
		if err := writeSegmentSignedFeature(w, seg.DeltaLFYV, 7); err != nil {
			return err
		}
		if err := writeSegmentSignedFeature(w, seg.DeltaLFYH, 7); err != nil {
			return err
		}
		if err := writeSegmentSignedFeature(w, seg.DeltaLFU, 7); err != nil {
			return err
		}
		if err := writeSegmentSignedFeature(w, seg.DeltaLFV, 7); err != nil {
			return err
		}

		if seg.RefFrame < 0 {
			if err := w.writeBool(false); err != nil {
				return err
			}
		} else {
			if err := w.writeBool(true); err != nil {
				return err
			}
			if err := w.writeBits(uint64(uint8(seg.RefFrame)), 3); err != nil {
				return err
			}
		}
		if err := w.writeBool(seg.Skip); err != nil {
			return err
		}
		if err := w.writeBool(seg.GlobalMV); err != nil {
			return err
		}
	}
	return nil
}

func writeSegmentSignedFeature(w *bitWriter, value int16, bits uint8) error {
	if value == 0 {
		return w.writeBool(false)
	}
	if err := w.writeBool(true); err != nil {
		return err
	}
	mask := uint16((uint32(1) << bits) - 1)
	return w.writeBits(uint64(uint16(value)&mask), bits)
}

func validateSegmentationParams(prefix FrameHeaderPrefix, seg SegmentationParams) error {
	if !seg.Enabled {
		if seg.UpdateMap || seg.TemporalUpdate || seg.UpdateData || segmentationDataNonZero(seg.Data) {
			return ErrInvalidFrame
		}
		return nil
	}
	if prefix.PrimaryRefFrame == EncoderPrimaryRefNone {
		if !seg.UpdateMap || !seg.UpdateData {
			return ErrInvalidFrame
		}
	} else if !seg.UpdateMap && seg.TemporalUpdate {
		return ErrInvalidFrame
	}
	if !seg.UpdateData {
		if segmentationDataNonZero(seg.Data) {
			return ErrInvalidFrame
		}
		return nil
	}
	for i := uint8(0); i < MaxSegments; i++ {
		if err := validateSegmentData(seg.Data.Segments[i]); err != nil {
			return err
		}
	}
	return nil
}

func validateSegmentData(seg SegmentData) error {
	if seg.RefFrame < -1 || seg.RefFrame > 7 {
		return ErrInvalidFrame
	}
	if err := validateSignedFeature(seg.DeltaQ, 9); err != nil {
		return err
	}
	if err := validateSignedFeature(seg.DeltaLFYV, 7); err != nil {
		return err
	}
	if err := validateSignedFeature(seg.DeltaLFYH, 7); err != nil {
		return err
	}
	if err := validateSignedFeature(seg.DeltaLFU, 7); err != nil {
		return err
	}
	return validateSignedFeature(seg.DeltaLFV, 7)
}

func validateSignedFeature(value int16, bits uint8) error {
	min := int16(-(1 << (bits - 1)))
	max := int16((1 << (bits - 1)) - 1)
	if value < min || value > max {
		return ErrInvalidFrame
	}
	return nil
}

// disableInactiveSegmentRefFrames rewrites the Go zero-value RefFrame (0) to the
// SEG_LVL_REF_FRAME disabled sentinel (-1). Callers use it only on segment data
// that will not be written, so a default (all-zero) SegmentationData reads as a
// disabled segmentation rather than one selecting reference index 0. An already
// selected ref (>= 1) or an explicit sentinel (-1) is left untouched.
func disableInactiveSegmentRefFrames(data *SegmentationData) {
	for i := range data.Segments {
		if data.Segments[i].RefFrame == 0 {
			data.Segments[i].RefFrame = -1
		}
	}
}

func segmentationDataNonZero(data SegmentationData) bool {
	for i := uint8(0); i < MaxSegments; i++ {
		seg := data.Segments[i]
		if seg.DeltaQ != 0 || seg.DeltaLFYV != 0 || seg.DeltaLFYH != 0 ||
			seg.DeltaLFU != 0 || seg.DeltaLFV != 0 ||
			seg.RefFrame >= 0 || seg.Skip || seg.GlobalMV {
			return true
		}
	}
	return false
}
