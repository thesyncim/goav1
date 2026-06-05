package encoder

import "github.com/thesyncim/goav1/internal/av1/bitstream"

// DeltaParams is the encoder-owned view of delta_q_params() plus
// delta_lf_params().
type DeltaParams struct {
	DeltaQPresent  bool
	DeltaQResLog2  uint8
	DeltaLFPresent bool
	DeltaLFResLog2 uint8
	DeltaLFMulti   bool
}

func DeltaParamsPayloadSize(size IntraFrameSize, quant QuantizationParams, delta DeltaParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeDeltaParamsPayload(&w, size.AllowIntrabc, quant, delta); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendDeltaParamsPayload(dst []byte, size IntraFrameSize, quant QuantizationParams, delta DeltaParams) ([]byte, error) {
	payloadSize, err := DeltaParamsPayloadSize(size, quant, delta)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeDeltaParamsPayload(&w, size.AllowIntrabc, quant, delta); err != nil {
		return dst, err
	}
	return out, nil
}

func DeltaParamsInterPayloadSize(size InterFrameSize, quant QuantizationParams, delta DeltaParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeDeltaParamsPayload(&w, false, quant, delta); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendDeltaParamsInterPayload(dst []byte, size InterFrameSize, quant QuantizationParams, delta DeltaParams) ([]byte, error) {
	payloadSize, err := DeltaParamsInterPayloadSize(size, quant, delta)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeDeltaParamsPayload(&w, false, quant, delta); err != nil {
		return dst, err
	}
	return out, nil
}

func writeDeltaParamsPayload(w *bitWriter, allowIntrabc bool, quant QuantizationParams, delta DeltaParams) error {
	if err := validateDeltaParams(allowIntrabc, quant, delta); err != nil {
		return err
	}
	if quant.BaseQIdx == 0 {
		return nil
	}
	if err := w.writeBool(delta.DeltaQPresent); err != nil {
		return err
	}
	if !delta.DeltaQPresent {
		return nil
	}
	if err := w.writeBits(uint64(delta.DeltaQResLog2), 2); err != nil {
		return err
	}
	if !allowIntrabc {
		if err := w.writeBool(delta.DeltaLFPresent); err != nil {
			return err
		}
	}
	if !delta.DeltaLFPresent {
		return nil
	}
	if err := w.writeBits(uint64(delta.DeltaLFResLog2), 2); err != nil {
		return err
	}
	return w.writeBool(delta.DeltaLFMulti)
}

func validateDeltaParams(allowIntrabc bool, quant QuantizationParams, delta DeltaParams) error {
	if delta.DeltaQResLog2 > 3 || delta.DeltaLFResLog2 > 3 {
		return ErrInvalidFrame
	}
	if quant.BaseQIdx == 0 {
		if delta.DeltaQPresent || delta.DeltaQResLog2 != 0 ||
			delta.DeltaLFPresent || delta.DeltaLFResLog2 != 0 || delta.DeltaLFMulti {
			return ErrInvalidFrame
		}
		return nil
	}
	if !delta.DeltaQPresent {
		if delta.DeltaQResLog2 != 0 || delta.DeltaLFPresent ||
			delta.DeltaLFResLog2 != 0 || delta.DeltaLFMulti {
			return ErrInvalidFrame
		}
		return nil
	}
	if allowIntrabc && (delta.DeltaLFPresent || delta.DeltaLFResLog2 != 0 || delta.DeltaLFMulti) {
		return ErrInvalidFrame
	}
	if !delta.DeltaLFPresent && (delta.DeltaLFResLog2 != 0 || delta.DeltaLFMulti) {
		return ErrInvalidFrame
	}
	return nil
}
