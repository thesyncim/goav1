package encoder

// IntraHeaderTemporalUnit describes the syntax descriptors used to emit the
// initial WebRTC keyframe-header temporal unit for a normalized config.
type IntraHeaderTemporalUnit struct {
	Sequence SequenceHeader
	Prefix   FrameHeaderPrefix
	Size     IntraFrameSize
}

// WebRTCKeyFrameTemporalUnit describes the config-derived initial key temporal
// unit: emitted AV1 headers plus WebRTC frame-control/dependency metadata.
type WebRTCKeyFrameTemporalUnit struct {
	Header   IntraHeaderTemporalUnit
	Frames   [WebRTCMaxSpatialLayers]FrameEncodeSettings
	FrameNum uint8
	Control  WebRTCTemporalUnitControl
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

// WebRTCKeyFrameTemporalUnitForConfig maps a WebRTC config into the initial
// key temporal-unit header descriptors and per-spatial-layer frame controls.
func WebRTCKeyFrameTemporalUnitForConfig(config Config, orderHint uint8, firstFrameID uint64) (WebRTCKeyFrameTemporalUnit, error) {
	config, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return WebRTCKeyFrameTemporalUnit{}, err
	}
	header, err := IntraHeaderTemporalUnitForConfig(config, orderHint)
	if err != nil {
		return WebRTCKeyFrameTemporalUnit{}, err
	}
	if config.SpatialLayerCount == 0 || config.SpatialLayerCount > WebRTCMaxSpatialLayers {
		return WebRTCKeyFrameTemporalUnit{}, ErrInvalidConfig
	}

	var unit WebRTCKeyFrameTemporalUnit
	unit.Header = header
	unit.FrameNum = config.SpatialLayerCount
	for i := uint8(0); i < unit.FrameNum; i++ {
		layer := config.SpatialLayers[i]
		settings := FrameEncodeSettings{
			Type:            FrameTypeKey,
			Resolution:      layer.Resolution,
			SpatialID:       i,
			TemporalID:      0,
			UpdateBuffer:    i,
			UpdateBufferSet: true,
			EffortLevel:     config.Speed,
			RateControl:     config.RateControl,
			Output:          true,
		}
		if i > 0 {
			if config.Scalability.IsSimulcast() {
				settings.Type = FrameTypeStart
			} else {
				settings.Type = FrameTypeDelta
				settings.ReferenceBuffers[0] = i - 1
				settings.ReferenceCount = 1
			}
		}
		unit.Frames[i] = settings
	}
	control, err := WebRTCTemporalUnitControlForFrames(config, unit.Frames[:unit.FrameNum], ReferenceBufferState{}, FrameIDBufferState{}, firstFrameID)
	if err != nil {
		return WebRTCKeyFrameTemporalUnit{}, err
	}
	unit.Control = control
	return unit, nil
}

// LowOverheadWebRTCKeyFrameTemporalUnitForConfigSize returns the exact emitted
// header-byte size plus the derived WebRTC frame controls.
func LowOverheadWebRTCKeyFrameTemporalUnitForConfigSize(config Config, orderHint uint8, firstFrameID uint64) (int, WebRTCKeyFrameTemporalUnit, error) {
	unit, err := WebRTCKeyFrameTemporalUnitForConfig(config, orderHint, firstFrameID)
	if err != nil {
		return 0, WebRTCKeyFrameTemporalUnit{}, err
	}
	size, err := LowOverheadIntraHeaderTemporalUnitSize(unit.Header.Sequence, unit.Header.Prefix, unit.Header.Size)
	if err != nil {
		return 0, WebRTCKeyFrameTemporalUnit{}, err
	}
	return size, unit, nil
}

// AppendLowOverheadWebRTCKeyFrameTemporalUnitForConfig appends the config-
// derived initial key temporal-unit headers into dst without growing it and
// returns the matching WebRTC control metadata.
func AppendLowOverheadWebRTCKeyFrameTemporalUnitForConfig(dst []byte, config Config, orderHint uint8, firstFrameID uint64) ([]byte, WebRTCKeyFrameTemporalUnit, error) {
	_, unit, err := LowOverheadWebRTCKeyFrameTemporalUnitForConfigSize(config, orderHint, firstFrameID)
	if err != nil {
		return dst, WebRTCKeyFrameTemporalUnit{}, err
	}
	out, err := AppendLowOverheadIntraHeaderTemporalUnit(dst, unit.Header.Sequence, unit.Header.Prefix, unit.Header.Size)
	if err != nil {
		return dst, WebRTCKeyFrameTemporalUnit{}, err
	}
	return out, unit, nil
}
