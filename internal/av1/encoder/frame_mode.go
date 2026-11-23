package encoder

// FrameModeParams is the encoder-owned view of frame-level motion/transform
// feature bits after skip_mode_params().
type FrameModeParams struct {
	AllowWarpedMotion bool
	ReducedTxSet      bool
}

func writeFrameModeParamsPayload(w *bitWriter, seq SequenceHeader, prefix FrameHeaderPrefix, params FrameModeParams) error {
	if err := validateFrameModeParams(seq, prefix, params); err != nil {
		return err
	}
	if frameModeWritesAllowWarpedMotion(seq, prefix) {
		if err := w.writeBool(params.AllowWarpedMotion); err != nil {
			return err
		}
	}
	return w.writeBool(params.ReducedTxSet)
}

func validateFrameModeParams(seq SequenceHeader, prefix FrameHeaderPrefix, params FrameModeParams) error {
	if err := validateSequenceHeader(seq); err != nil {
		return err
	}
	if !prefix.FrameType.valid() {
		return ErrInvalidFrame
	}
	if params.AllowWarpedMotion && !frameModeWritesAllowWarpedMotion(seq, prefix) {
		return ErrInvalidFrame
	}
	return nil
}

func frameModeWritesAllowWarpedMotion(seq SequenceHeader, prefix FrameHeaderPrefix) bool {
	return !prefix.ErrorResilientMode && prefix.FrameType.interOrSwitch() && seq.EnableWarpedMotion
}
