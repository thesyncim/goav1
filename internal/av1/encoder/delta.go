package encoder

// DeltaParams is the encoder-owned view of delta_q_params() plus
// delta_lf_params().
type DeltaParams struct {
	DeltaQPresent  bool
	DeltaQResLog2  uint8
	DeltaLFPresent bool
	DeltaLFResLog2 uint8
	DeltaLFMulti   bool
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
