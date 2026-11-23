package encoder

type TransformMode uint8

const (
	TransformMode4x4Only TransformMode = iota
	TransformModeLargest
	TransformModeSwitchable
)

type ReferenceMode uint8

const (
	ReferenceModeSingle ReferenceMode = iota
	ReferenceModeCompound
	ReferenceModeSelect
)

// TransformReferenceParams is the encoder-owned view of read_tx_mode() plus
// reference_select syntax.
type TransformReferenceParams struct {
	TransformMode TransformMode
	ReferenceMode ReferenceMode
}

func writeTransformReferenceParamsPayload(w *bitWriter, prefix FrameHeaderPrefix, allLossless bool, params TransformReferenceParams) error {
	if err := validateTransformReferenceParams(prefix, allLossless, params); err != nil {
		return err
	}
	if !allLossless {
		if err := w.writeBool(params.TransformMode == TransformModeSwitchable); err != nil {
			return err
		}
	}
	if prefix.FrameType.interOrSwitch() {
		return w.writeBool(params.ReferenceMode == ReferenceModeSelect)
	}
	return nil
}

func validateTransformReferenceParams(prefix FrameHeaderPrefix, allLossless bool, params TransformReferenceParams) error {
	if params.TransformMode > TransformModeSwitchable || params.ReferenceMode > ReferenceModeSelect {
		return ErrInvalidFrame
	}
	if allLossless {
		if params.TransformMode != TransformMode4x4Only {
			return ErrInvalidFrame
		}
	} else if params.TransformMode != TransformModeLargest && params.TransformMode != TransformModeSwitchable {
		return ErrInvalidFrame
	}
	if prefix.FrameType.interOrSwitch() {
		if params.ReferenceMode != ReferenceModeSingle && params.ReferenceMode != ReferenceModeSelect {
			return ErrInvalidFrame
		}
	} else if params.ReferenceMode != ReferenceModeSingle {
		return ErrInvalidFrame
	}
	return nil
}
