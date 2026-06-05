package encoder

import "github.com/thesyncim/goav1/internal/av1/bitstream"

// QuantizationParams is the encoder-owned, syntax-width view of
// quantization_params() from uncompressed_header().
type QuantizationParams struct {
	BaseQIdx uint8

	YDCDelta int16
	UDCDelta int16
	UACDelta int16
	VDCDelta int16
	VACDelta int16

	DiffUVDeltas bool

	UsingQMatrix  bool
	QMatrixLevelY uint8
	QMatrixLevelU uint8
	QMatrixLevelV uint8
}

func QuantizationParamsPayloadSize(seq SequenceHeader, quant QuantizationParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeQuantizationParamsPayload(&w, seq, quant); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendQuantizationParamsPayload(dst []byte, seq SequenceHeader, quant QuantizationParams) ([]byte, error) {
	payloadSize, err := QuantizationParamsPayloadSize(seq, quant)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeQuantizationParamsPayload(&w, seq, quant); err != nil {
		return dst, err
	}
	return out, nil
}

func writeQuantizationParamsPayload(w *bitWriter, seq SequenceHeader, quant QuantizationParams) error {
	if err := validateQuantizationParams(seq, quant); err != nil {
		return err
	}
	if err := w.writeBits(uint64(quant.BaseQIdx), 8); err != nil {
		return err
	}
	if err := writeDeltaQ(w, quant.YDCDelta); err != nil {
		return err
	}
	if !seq.ColorConfig.MonoChrome {
		if seq.ColorConfig.SeparateUVDeltaQ {
			if err := w.writeBool(quant.DiffUVDeltas); err != nil {
				return err
			}
		}
		if err := writeDeltaQ(w, quant.UDCDelta); err != nil {
			return err
		}
		if err := writeDeltaQ(w, quant.UACDelta); err != nil {
			return err
		}
		if quant.DiffUVDeltas {
			if err := writeDeltaQ(w, quant.VDCDelta); err != nil {
				return err
			}
			if err := writeDeltaQ(w, quant.VACDelta); err != nil {
				return err
			}
		}
	}

	if err := w.writeBool(quant.UsingQMatrix); err != nil {
		return err
	}
	if !quant.UsingQMatrix {
		return nil
	}
	if err := w.writeBits(uint64(quant.QMatrixLevelY), 4); err != nil {
		return err
	}
	if err := w.writeBits(uint64(quant.QMatrixLevelU), 4); err != nil {
		return err
	}
	if seq.ColorConfig.SeparateUVDeltaQ {
		return w.writeBits(uint64(quant.QMatrixLevelV), 4)
	}
	return nil
}

func writeDeltaQ(w *bitWriter, delta int16) error {
	if delta == 0 {
		return w.writeBool(false)
	}
	if err := w.writeBool(true); err != nil {
		return err
	}
	return w.writeBits(uint64(uint16(delta)&0x7f), 7)
}

func validateQuantizationParams(seq SequenceHeader, quant QuantizationParams) error {
	if err := validateDeltaQ(quant.YDCDelta); err != nil {
		return err
	}
	if err := validateDeltaQ(quant.UDCDelta); err != nil {
		return err
	}
	if err := validateDeltaQ(quant.UACDelta); err != nil {
		return err
	}
	if err := validateDeltaQ(quant.VDCDelta); err != nil {
		return err
	}
	if err := validateDeltaQ(quant.VACDelta); err != nil {
		return err
	}

	if seq.ColorConfig.MonoChrome {
		if quant.DiffUVDeltas || quant.UDCDelta != 0 || quant.UACDelta != 0 ||
			quant.VDCDelta != 0 || quant.VACDelta != 0 {
			return ErrInvalidFrame
		}
	} else if !seq.ColorConfig.SeparateUVDeltaQ {
		if quant.DiffUVDeltas || quant.VDCDelta != quant.UDCDelta || quant.VACDelta != quant.UACDelta {
			return ErrInvalidFrame
		}
	}

	if !quant.UsingQMatrix {
		if quant.QMatrixLevelY != 0 || quant.QMatrixLevelU != 0 || quant.QMatrixLevelV != 0 {
			return ErrInvalidFrame
		}
		return nil
	}
	if quant.QMatrixLevelY > 15 || quant.QMatrixLevelU > 15 || quant.QMatrixLevelV > 15 {
		return ErrInvalidFrame
	}
	if !seq.ColorConfig.SeparateUVDeltaQ && quant.QMatrixLevelV != quant.QMatrixLevelU {
		return ErrInvalidFrame
	}
	return nil
}

func validateDeltaQ(delta int16) error {
	if delta < -64 || delta > 63 {
		return ErrInvalidFrame
	}
	return nil
}
