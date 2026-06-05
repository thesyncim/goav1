package encoder

// IntraHeaderTemporalUnit describes the syntax descriptors used to emit the
// initial WebRTC keyframe-header temporal unit for a normalized config.
type IntraHeaderTemporalUnit struct {
	Sequence SequenceHeader
	Prefix   FrameHeaderPrefix
	Size     IntraFrameSize
}

// IntraHeaderTemporalUnitForConfig maps a WebRTC encoder config into the
// sequence header, shown-key frame-header prefix, and frame-size syntax used by
// the first low-overhead temporal unit.
func IntraHeaderTemporalUnitForConfig(config Config, orderHint uint8) (IntraHeaderTemporalUnit, error) {
	seq, err := SequenceHeaderForConfig(config)
	if err != nil {
		return IntraHeaderTemporalUnit{}, err
	}
	if seq.EnableOrderHint && uint16(orderHint) >= uint16(1)<<seq.OrderHintBits {
		return IntraHeaderTemporalUnit{}, ErrInvalidFrame
	}
	if !seq.EnableOrderHint && orderHint != 0 {
		return IntraHeaderTemporalUnit{}, ErrInvalidFrame
	}

	unit := IntraHeaderTemporalUnit{
		Sequence: seq,
		Prefix: FrameHeaderPrefix{
			FrameType:          FrameHeaderTypeKey,
			ShowFrame:          true,
			ErrorResilientMode: true,
			ForceIntegerMV:     true,
			OrderHint:          orderHint,
			PrimaryRefFrame:    EncoderPrimaryRefNone,
		},
		Size: IntraFrameSize{
			UpscaledWidth:       seq.MaxFrameWidth,
			Height:              seq.MaxFrameHeight,
			SuperResDenominator: 8,
			RefreshFrameFlags:   0xff,
		},
	}
	if err := validateFrameHeaderPrefix(seq, unit.Prefix); err != nil {
		return IntraHeaderTemporalUnit{}, err
	}
	if err := validateIntraFrameSize(seq, unit.Prefix, unit.Size); err != nil {
		return IntraHeaderTemporalUnit{}, err
	}
	return unit, nil
}

// LowOverheadIntraHeaderTemporalUnitForConfigSize returns the exact size of the
// config-derived initial keyframe-header temporal unit.
func LowOverheadIntraHeaderTemporalUnitForConfigSize(config Config, orderHint uint8) (int, IntraHeaderTemporalUnit, error) {
	unit, err := IntraHeaderTemporalUnitForConfig(config, orderHint)
	if err != nil {
		return 0, IntraHeaderTemporalUnit{}, err
	}
	size, err := LowOverheadIntraHeaderTemporalUnitSize(unit.Sequence, unit.Prefix, unit.Size)
	if err != nil {
		return 0, IntraHeaderTemporalUnit{}, err
	}
	return size, unit, nil
}

// AppendLowOverheadIntraHeaderTemporalUnitForConfig appends the config-derived
// initial keyframe-header temporal unit into dst without growing it.
func AppendLowOverheadIntraHeaderTemporalUnitForConfig(dst []byte, config Config, orderHint uint8) ([]byte, IntraHeaderTemporalUnit, error) {
	_, unit, err := LowOverheadIntraHeaderTemporalUnitForConfigSize(config, orderHint)
	if err != nil {
		return dst, IntraHeaderTemporalUnit{}, err
	}
	out, err := AppendLowOverheadIntraHeaderTemporalUnit(dst, unit.Sequence, unit.Prefix, unit.Size)
	if err != nil {
		return dst, IntraHeaderTemporalUnit{}, err
	}
	return out, unit, nil
}
