package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

// IntraHeaderTemporalUnit describes the syntax descriptors used to emit the
// initial WebRTC keyframe-header temporal unit for a normalized config.
type IntraHeaderTemporalUnit struct {
	Sequence SequenceHeader
	Prefix   FrameHeaderPrefix
	Size     IntraFrameSize
}

// InterHeaderFrame describes the syntax descriptors used to emit one WebRTC
// delta frame-header OBU for a normalized config.
type InterHeaderFrame struct {
	Sequence SequenceHeader
	Prefix   FrameHeaderPrefix
	Size     InterFrameSize

	TemporalID uint8
	SpatialID  uint8
}

// WebRTCKeyFrameTemporalUnit describes the config-derived initial key temporal
// unit: emitted AV1 headers plus WebRTC frame-control/dependency metadata.
type WebRTCKeyFrameTemporalUnit struct {
	Header   IntraHeaderTemporalUnit
	Frames   [WebRTCMaxSpatialLayers]FrameEncodeSettings
	FrameNum uint8
	Control  WebRTCTemporalUnitControl
}

// WebRTCDeltaFrameTemporalUnit describes a config-derived steady-state delta
// temporal unit plus WebRTC frame-control/dependency metadata.
type WebRTCDeltaFrameTemporalUnit struct {
	Headers  [WebRTCMaxSpatialLayers]InterHeaderFrame
	Frames   [WebRTCMaxSpatialLayers]FrameEncodeSettings
	FrameNum uint8
	Control  WebRTCTemporalUnitControl
}

type WebRTCPictureTemporalUnit struct {
	Key       bool
	KeyUnit   WebRTCKeyFrameTemporalUnit
	Delta     bool
	DeltaUnit WebRTCDeltaFrameTemporalUnit
}

type WebRTCEncoderState struct {
	ReferenceState           ReferenceBufferState
	FrameIDState             FrameIDBufferState
	DependencyStructureState WebRTCDependencyStructureState
	NextOrderHint            uint8
	NextFrameID              uint64
	DeltaPictureIndex        uint64
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

func WebRTCKeyFrameTemporalUnitForState(config Config, state WebRTCEncoderState) (WebRTCKeyFrameTemporalUnit, WebRTCEncoderState, error) {
	unit, err := WebRTCKeyFrameTemporalUnitForConfig(config, state.NextOrderHint, state.NextFrameID)
	if err != nil {
		return WebRTCKeyFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	next, err := advanceWebRTCEncoderState(config, state, unit.FrameNum, unit.Control, 1)
	if err != nil {
		return WebRTCKeyFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	return unit, next, nil
}

func LowOverheadWebRTCKeyFrameTemporalUnitForStateSize(config Config, state WebRTCEncoderState) (int, WebRTCKeyFrameTemporalUnit, WebRTCEncoderState, error) {
	unit, next, err := WebRTCKeyFrameTemporalUnitForState(config, state)
	if err != nil {
		return 0, WebRTCKeyFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	size, err := LowOverheadIntraHeaderTemporalUnitSize(unit.Header.Sequence, unit.Header.Prefix, unit.Header.Size)
	if err != nil {
		return 0, WebRTCKeyFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	return size, unit, next, nil
}

func AppendLowOverheadWebRTCKeyFrameTemporalUnitForState(dst []byte, config Config, state WebRTCEncoderState) ([]byte, WebRTCKeyFrameTemporalUnit, WebRTCEncoderState, error) {
	_, unit, next, err := LowOverheadWebRTCKeyFrameTemporalUnitForStateSize(config, state)
	if err != nil {
		return dst, WebRTCKeyFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	out, err := AppendLowOverheadIntraHeaderTemporalUnit(dst, unit.Header.Sequence, unit.Header.Prefix, unit.Header.Size)
	if err != nil {
		return dst, WebRTCKeyFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	return out, unit, next, nil
}

func LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(config Config, state WebRTCEncoderState, forceKeyFrame bool) (int, WebRTCPictureTemporalUnit, WebRTCEncoderState, error) {
	unit, next, err := WebRTCNextTemporalUnitForState(config, state, forceKeyFrame)
	if err != nil {
		return 0, WebRTCPictureTemporalUnit{}, WebRTCEncoderState{}, err
	}
	if unit.Key {
		size, err := LowOverheadIntraHeaderTemporalUnitSize(unit.KeyUnit.Header.Sequence, unit.KeyUnit.Header.Prefix, unit.KeyUnit.Header.Size)
		if err != nil {
			return 0, WebRTCPictureTemporalUnit{}, WebRTCEncoderState{}, err
		}
		return size, unit, next, nil
	}
	if unit.Delta {
		size, err := LowOverheadWebRTCDeltaHeaderTemporalUnitSize(unit.DeltaUnit)
		if err != nil {
			return 0, WebRTCPictureTemporalUnit{}, WebRTCEncoderState{}, err
		}
		return size, unit, next, nil
	}
	return 0, WebRTCPictureTemporalUnit{}, WebRTCEncoderState{}, ErrInvalidFrame
}

func AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState(dst []byte, config Config, state WebRTCEncoderState, forceKeyFrame bool) ([]byte, WebRTCPictureTemporalUnit, WebRTCEncoderState, error) {
	_, unit, next, err := LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(config, state, forceKeyFrame)
	if err != nil {
		return dst, WebRTCPictureTemporalUnit{}, WebRTCEncoderState{}, err
	}
	if unit.Key {
		out, err := AppendLowOverheadIntraHeaderTemporalUnit(dst, unit.KeyUnit.Header.Sequence, unit.KeyUnit.Header.Prefix, unit.KeyUnit.Header.Size)
		if err != nil {
			return dst, WebRTCPictureTemporalUnit{}, WebRTCEncoderState{}, err
		}
		return out, unit, next, nil
	}
	if unit.Delta {
		out, err := AppendLowOverheadWebRTCDeltaHeaderTemporalUnit(dst, unit.DeltaUnit)
		if err != nil {
			return dst, WebRTCPictureTemporalUnit{}, WebRTCEncoderState{}, err
		}
		return out, unit, next, nil
	}
	return dst, WebRTCPictureTemporalUnit{}, WebRTCEncoderState{}, ErrInvalidFrame
}

func WebRTCNextTemporalUnitForState(config Config, state WebRTCEncoderState, forceKeyFrame bool) (WebRTCPictureTemporalUnit, WebRTCEncoderState, error) {
	config, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return WebRTCPictureTemporalUnit{}, WebRTCEncoderState{}, err
	}
	if forceKeyFrame || webRTCEncoderStateNeedsKey(config, state) {
		key, next, err := WebRTCKeyFrameTemporalUnitForState(config, state)
		if err != nil {
			return WebRTCPictureTemporalUnit{}, WebRTCEncoderState{}, err
		}
		return WebRTCPictureTemporalUnit{Key: true, KeyUnit: key}, next, nil
	}
	delta, next, err := WebRTCDeltaFrameTemporalUnitForState(config, state)
	if err != nil {
		return WebRTCPictureTemporalUnit{}, WebRTCEncoderState{}, err
	}
	return WebRTCPictureTemporalUnit{Delta: true, DeltaUnit: delta}, next, nil
}

func webRTCEncoderStateNeedsKey(config Config, state WebRTCEncoderState) bool {
	if !state.DependencyStructureState.Valid || state.DeltaPictureIndex == 0 {
		return true
	}
	return config.KeyFrameInterval > 0 && state.DeltaPictureIndex >= uint64(config.KeyFrameInterval)
}

func WebRTCDeltaFrameTemporalUnitForState(config Config, state WebRTCEncoderState) (WebRTCDeltaFrameTemporalUnit, WebRTCEncoderState, error) {
	config, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return WebRTCDeltaFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	temporalID, err := WebRTCTemporalIDForDeltaPicture(config.TemporalLayerCount, state.DeltaPictureIndex)
	if err != nil {
		return WebRTCDeltaFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	unit, err := WebRTCDeltaFrameTemporalUnitForConfigWithOrderHint(config, state.ReferenceState, state.FrameIDState, temporalID, state.NextFrameID, state.NextOrderHint)
	if err != nil {
		return WebRTCDeltaFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	next, err := advanceWebRTCEncoderState(config, state, unit.FrameNum, unit.Control, state.DeltaPictureIndex+1)
	if err != nil {
		return WebRTCDeltaFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	return unit, next, nil
}

func WebRTCTemporalIDForDeltaPicture(temporalLayers uint8, deltaPictureIndex uint64) (uint8, error) {
	if temporalLayers == 0 || temporalLayers > WebRTCMaxTemporalLayers || deltaPictureIndex == 0 {
		return 0, ErrInvalidConfig
	}
	var trailingZeroCount uint8
	for value := deltaPictureIndex; value&1 == 0 && trailingZeroCount < temporalLayers-1; value >>= 1 {
		trailingZeroCount++
	}
	return temporalLayers - 1 - trailingZeroCount, nil
}

func advanceWebRTCEncoderState(config Config, state WebRTCEncoderState, frameNum uint8, control WebRTCTemporalUnitControl, nextDeltaPictureIndex uint64) (WebRTCEncoderState, error) {
	config, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return WebRTCEncoderState{}, err
	}
	structureState, _, err := WebRTCDependencyStructureStateForTemporalUnit(control, state.DependencyStructureState)
	if err != nil {
		return WebRTCEncoderState{}, err
	}
	orderHint, err := advanceWebRTCOrderHint(config, state.NextOrderHint)
	if err != nil {
		return WebRTCEncoderState{}, err
	}
	state.ReferenceState = control.ReferenceState
	state.FrameIDState = control.FrameIDState
	state.DependencyStructureState = structureState
	state.NextOrderHint = orderHint
	state.NextFrameID += uint64(frameNum)
	state.DeltaPictureIndex = nextDeltaPictureIndex
	return state, nil
}

func advanceWebRTCOrderHint(config Config, orderHint uint8) (uint8, error) {
	seq, err := SequenceHeaderForConfig(config)
	if err != nil {
		return 0, err
	}
	if !seq.EnableOrderHint {
		if orderHint != 0 {
			return 0, ErrInvalidFrame
		}
		return 0, nil
	}
	if seq.OrderHintBits == 0 || seq.OrderHintBits > 8 || uint16(orderHint) >= uint16(1)<<seq.OrderHintBits {
		return 0, ErrInvalidFrame
	}
	return (orderHint + 1) & uint8((1<<seq.OrderHintBits)-1), nil
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

// LowOverheadInterHeaderFrameOBUSize returns the exact size of one low-
// overhead frame-header OBU, including temporal/spatial extension fields.
func LowOverheadInterHeaderFrameOBUSize(header InterHeaderFrame) (int, error) {
	payloadSize, err := FrameHeaderInterPayloadSize(header.Sequence, header.Prefix, header.Size)
	if err != nil {
		return 0, err
	}
	if header.TemporalID > 7 || header.SpatialID > 3 {
		return 0, ErrInvalidFrame
	}
	obuHeaderSize := 1
	if header.TemporalID != 0 || header.SpatialID != 0 {
		obuHeaderSize = 2
	}
	return obuHeaderSize + bitstream.LEB128Len(uint32(payloadSize)) + payloadSize, nil
}

func AppendLowOverheadInterHeaderFrameOBU(dst []byte, header InterHeaderFrame) ([]byte, error) {
	payloadSize, err := FrameHeaderInterPayloadSize(header.Sequence, header.Prefix, header.Size)
	if err != nil {
		return dst, err
	}
	if header.TemporalID > 7 || header.SpatialID > 3 {
		return dst, ErrInvalidFrame
	}
	obuHeaderSize := 1
	if header.TemporalID != 0 || header.SpatialID != 0 {
		obuHeaderSize = 2
	}
	obuSize := obuHeaderSize + bitstream.LEB128Len(uint32(payloadSize)) + payloadSize
	if cap(dst)-len(dst) < obuSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+obuSize]
	n, err := obu.PutHeader(out[off:], obu.Header{
		Type:         obu.TypeFrameHeader,
		Extension:    header.TemporalID != 0 || header.SpatialID != 0,
		HasSizeField: true,
		TemporalID:   header.TemporalID,
		SpatialID:    header.SpatialID,
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
	if err := writeFrameHeaderPrefixPayload(&w, header.Sequence, header.Prefix); err != nil {
		return dst, err
	}
	if err := writeInterFrameSizePayload(&w, header.Sequence, header.Prefix, header.Size); err != nil {
		return dst, err
	}
	return out, nil
}

// LowOverheadWebRTCDeltaHeaderTemporalUnitSize returns the exact byte size of a
// low-overhead temporal unit carrying the scheduled delta frame-header OBUs.
func LowOverheadWebRTCDeltaHeaderTemporalUnitSize(unit WebRTCDeltaFrameTemporalUnit) (int, error) {
	if unit.FrameNum == 0 || unit.FrameNum > WebRTCMaxSpatialLayers {
		return 0, ErrInvalidFrame
	}
	size := lowOverheadOBUSizeUnchecked(OBU{Type: obu.TypeTemporalDelimiter})
	for i := uint8(0); i < unit.FrameNum; i++ {
		headerSize, err := LowOverheadInterHeaderFrameOBUSize(unit.Headers[i])
		if err != nil {
			return 0, err
		}
		size += headerSize
	}
	return size, nil
}

// AppendLowOverheadWebRTCDeltaHeaderTemporalUnit appends one temporal
// delimiter followed by the scheduled delta frame-header OBUs. It never grows
// dst and validates/sizes before writing so errors leave dst length unchanged.
func AppendLowOverheadWebRTCDeltaHeaderTemporalUnit(dst []byte, unit WebRTCDeltaFrameTemporalUnit) ([]byte, error) {
	size, err := LowOverheadWebRTCDeltaHeaderTemporalUnitSize(unit)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, bitstream.ErrShortBuffer
	}
	out, err := AppendLowOverheadOBU(dst, OBU{Type: obu.TypeTemporalDelimiter})
	if err != nil {
		return dst, err
	}
	for i := uint8(0); i < unit.FrameNum; i++ {
		out, err = AppendLowOverheadInterHeaderFrameOBU(out, unit.Headers[i])
		if err != nil {
			return dst, err
		}
	}
	return out, nil
}

func LowOverheadWebRTCDeltaHeaderTemporalUnitForStateSize(config Config, state WebRTCEncoderState) (int, WebRTCDeltaFrameTemporalUnit, WebRTCEncoderState, error) {
	unit, next, err := WebRTCDeltaFrameTemporalUnitForState(config, state)
	if err != nil {
		return 0, WebRTCDeltaFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	size, err := LowOverheadWebRTCDeltaHeaderTemporalUnitSize(unit)
	if err != nil {
		return 0, WebRTCDeltaFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	return size, unit, next, nil
}

func AppendLowOverheadWebRTCDeltaHeaderTemporalUnitForState(dst []byte, config Config, state WebRTCEncoderState) ([]byte, WebRTCDeltaFrameTemporalUnit, WebRTCEncoderState, error) {
	_, unit, next, err := LowOverheadWebRTCDeltaHeaderTemporalUnitForStateSize(config, state)
	if err != nil {
		return dst, WebRTCDeltaFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	out, err := AppendLowOverheadWebRTCDeltaHeaderTemporalUnit(dst, unit)
	if err != nil {
		return dst, WebRTCDeltaFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	return out, unit, next, nil
}

// WebRTCDeltaFrameTemporalUnitForConfig maps a WebRTC config and existing
// reference state into the next refreshing delta temporal-unit controls.
func WebRTCDeltaFrameTemporalUnitForConfig(config Config, referenceState ReferenceBufferState, frameIDState FrameIDBufferState, temporalID uint8, firstFrameID uint64) (WebRTCDeltaFrameTemporalUnit, error) {
	return WebRTCDeltaFrameTemporalUnitForConfigWithOrderHint(config, referenceState, frameIDState, temporalID, firstFrameID, 0)
}

func WebRTCDeltaFrameTemporalUnitForConfigWithOrderHint(config Config, referenceState ReferenceBufferState, frameIDState FrameIDBufferState, temporalID uint8, firstFrameID uint64, orderHint uint8) (WebRTCDeltaFrameTemporalUnit, error) {
	config, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return WebRTCDeltaFrameTemporalUnit{}, err
	}
	if config.SpatialLayerCount == 0 || config.SpatialLayerCount > WebRTCMaxSpatialLayers ||
		temporalID >= config.TemporalLayerCount {
		return WebRTCDeltaFrameTemporalUnit{}, ErrInvalidConfig
	}

	var unit WebRTCDeltaFrameTemporalUnit
	unit.FrameNum = config.SpatialLayerCount
	for i := uint8(0); i < unit.FrameNum; i++ {
		layer := config.SpatialLayers[i]
		settings := FrameEncodeSettings{
			Type:            FrameTypeDelta,
			Resolution:      layer.Resolution,
			SpatialID:       i,
			TemporalID:      temporalID,
			UpdateBuffer:    i,
			UpdateBufferSet: true,
			EffortLevel:     config.Speed,
			RateControl:     config.RateControl,
			Output:          true,
		}
		settings.ReferenceBuffers[settings.ReferenceCount] = i
		settings.ReferenceCount++
		if i > 0 && !config.Scalability.IsSimulcast() {
			settings.ReferenceBuffers[settings.ReferenceCount] = i - 1
			settings.ReferenceCount++
		}
		unit.Frames[i] = settings
	}
	control, err := WebRTCTemporalUnitControlForFrames(config, unit.Frames[:unit.FrameNum], referenceState, frameIDState, firstFrameID)
	if err != nil {
		return WebRTCDeltaFrameTemporalUnit{}, err
	}
	unit.Control = control
	for i := uint8(0); i < unit.FrameNum; i++ {
		header, err := interHeaderFrameForSettings(config, unit.Frames[i], firstFrameID+uint64(i), orderHint)
		if err != nil {
			return WebRTCDeltaFrameTemporalUnit{}, err
		}
		unit.Headers[i] = header
	}
	return unit, nil
}

func interHeaderFrameForSettings(config Config, settings FrameEncodeSettings, frameID uint64, orderHint uint8) (InterHeaderFrame, error) {
	seq, err := SequenceHeaderForConfig(config)
	if err != nil {
		return InterHeaderFrame{}, err
	}
	if seq.EnableOrderHint && uint16(orderHint) >= uint16(1)<<seq.OrderHintBits {
		return InterHeaderFrame{}, ErrInvalidFrame
	}
	if !seq.EnableOrderHint && orderHint != 0 {
		return InterHeaderFrame{}, ErrInvalidFrame
	}
	if settings.ReferenceCount == 0 || settings.ReferenceCount > WebRTCMaxFrameReferences ||
		!settings.UpdateBufferSet || settings.UpdateBuffer >= WebRTCReferenceBuffers {
		return InterHeaderFrame{}, ErrInvalidFrame
	}

	prefix := FrameHeaderPrefix{
		FrameType:          FrameHeaderTypeInter,
		ShowFrame:          true,
		ShowableFrame:      true,
		ErrorResilientMode: true,
		FrameSizeOverride:  true,
		OrderHint:          orderHint,
		PrimaryRefFrame:    EncoderPrimaryRefNone,
	}
	if seq.FrameIDNumbersPresent {
		prefix.FrameID = uint16(frameID & uint64((uint32(1)<<sequenceFrameIDBits(seq))-1))
	}
	size := InterFrameSize{
		UpscaledWidth:       uint32(settings.Resolution.Width),
		Height:              uint32(settings.Resolution.Height),
		SuperResDenominator: 8,
		RefreshFrameFlags:   1 << settings.UpdateBuffer,
	}
	firstRef := settings.ReferenceBuffers[0]
	for i := range size.RefFrameIdx {
		size.RefFrameIdx[i] = firstRef
	}
	for i := uint8(0); i < settings.ReferenceCount; i++ {
		size.RefFrameIdx[i] = settings.ReferenceBuffers[i]
	}
	if err := validateFrameHeaderPrefix(seq, prefix); err != nil {
		return InterHeaderFrame{}, err
	}
	if err := validateInterFrameSize(seq, prefix, size); err != nil {
		return InterHeaderFrame{}, err
	}
	return InterHeaderFrame{
		Sequence:   seq,
		Prefix:     prefix,
		Size:       size,
		TemporalID: settings.TemporalID,
		SpatialID:  settings.SpatialID,
	}, nil
}
