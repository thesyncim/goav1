package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
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
	Header      IntraHeaderTemporalUnit
	Scalability ScalabilityMode
	Frames      [WebRTCMaxSpatialLayers]FrameEncodeSettings
	FrameNum    uint8
	Control     WebRTCTemporalUnitControl
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
			FrameType:               FrameHeaderTypeKey,
			ShowFrame:               true,
			ErrorResilientMode:      true,
			AllowScreenContentTools: config.Content == ContentScreen,
			ForceIntegerMV:          true,
			OrderHint:               orderHint,
			PrimaryRefFrame:         EncoderPrimaryRefNone,
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
	unit.Scalability = config.Scalability
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
			Quantizer:       config.Quantizer,
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
	size, err := LowOverheadWebRTCKeyFrameHeaderTemporalUnitSize(unit)
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
	out, err := AppendLowOverheadWebRTCKeyFrameHeaderTemporalUnit(dst, unit)
	if err != nil {
		return dst, WebRTCKeyFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	return out, unit, next, nil
}

// LowOverheadWebRTCKeyFrameHeaderTemporalUnitSize returns the byte size for a
// WebRTC key temporal unit carrying sequence header, optional scalability
// metadata, and key/intra-only frame header.
func LowOverheadWebRTCKeyFrameHeaderTemporalUnitSize(unit WebRTCKeyFrameTemporalUnit) (int, error) {
	metadataFrames, err := webRTCKeyUnitScalabilityFrames(unit)
	if err != nil {
		return 0, err
	}
	seqSize, err := LowOverheadSequenceHeaderOBUSize(unit.Header.Sequence)
	if err != nil {
		return 0, err
	}
	frameSize, err := LowOverheadFrameHeaderIntraOBUSize(unit.Header.Sequence, unit.Header.Prefix, unit.Header.Size)
	if err != nil {
		return 0, err
	}
	size := lowOverheadOBUSizeUnchecked(OBU{Type: obu.TypeTemporalDelimiter}) + seqSize + frameSize
	if metadataSize, ok, err := lowOverheadWebRTCScalabilityMetadataOBUSizeForFrames(unit.Scalability, metadataFrames); err != nil {
		return 0, err
	} else if ok {
		size += metadataSize
	}
	return size, nil
}

// AppendLowOverheadWebRTCKeyFrameHeaderTemporalUnit appends one WebRTC key
// temporal unit with optional scalability metadata immediately after the
// sequence header.
func AppendLowOverheadWebRTCKeyFrameHeaderTemporalUnit(dst []byte, unit WebRTCKeyFrameTemporalUnit) ([]byte, error) {
	size, err := LowOverheadWebRTCKeyFrameHeaderTemporalUnitSize(unit)
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
	out, err = AppendLowOverheadSequenceHeaderOBU(out, unit.Header.Sequence)
	if err != nil {
		return dst, err
	}
	metadataFrames, err := webRTCKeyUnitScalabilityFrames(unit)
	if err != nil {
		return dst, err
	}
	if next, _, err := appendLowOverheadWebRTCScalabilityMetadataOBUForFrames(out, unit.Scalability, metadataFrames); err != nil {
		return dst, err
	} else {
		out = next
	}
	out, err = AppendLowOverheadFrameHeaderIntraOBU(out, unit.Header.Sequence, unit.Header.Prefix, unit.Header.Size)
	if err != nil {
		return dst, err
	}
	return out, nil
}

func LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(config Config, state WebRTCEncoderState, forceKeyFrame bool) (int, WebRTCPictureTemporalUnit, WebRTCEncoderState, error) {
	unit, next, err := WebRTCNextTemporalUnitForState(config, state, forceKeyFrame)
	if err != nil {
		return 0, WebRTCPictureTemporalUnit{}, WebRTCEncoderState{}, err
	}
	if unit.Key {
		size, err := LowOverheadWebRTCKeyFrameHeaderTemporalUnitSize(unit.KeyUnit)
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
		out, err := AppendLowOverheadWebRTCKeyFrameHeaderTemporalUnit(dst, unit.KeyUnit)
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

// LowOverheadCompleteIntraHeaderTemporalUnitSize returns the exact size of one
// temporal unit containing a temporal delimiter, sequence header, and complete
// intra frame-header OBU.
func LowOverheadCompleteIntraHeaderTemporalUnitSize(seq SequenceHeader, header IntraFrameHeaderParams) (int, error) {
	size := lowOverheadOBUSizeUnchecked(OBU{Type: obu.TypeTemporalDelimiter})
	seqSize, err := LowOverheadSequenceHeaderOBUSize(seq)
	if err != nil {
		return 0, err
	}
	frameSize, err := LowOverheadIntraFrameHeaderOBUSize(seq, header, 0, 0)
	if err != nil {
		return 0, err
	}
	return size + seqSize + frameSize, nil
}

// AppendLowOverheadCompleteIntraHeaderTemporalUnit appends one temporal unit
// containing a temporal delimiter, sequence header, and complete intra
// frame-header OBU. It validates and sizes before writing, so errors leave dst
// length unchanged.
func AppendLowOverheadCompleteIntraHeaderTemporalUnit(dst []byte, seq SequenceHeader, header IntraFrameHeaderParams) ([]byte, error) {
	size, err := LowOverheadCompleteIntraHeaderTemporalUnitSize(seq, header)
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
	out, err = AppendLowOverheadSequenceHeaderOBU(out, seq)
	if err != nil {
		return dst, err
	}
	out, err = AppendLowOverheadIntraFrameHeaderOBU(out, seq, header, 0, 0)
	if err != nil {
		return dst, err
	}
	return out, nil
}

func LowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitSize(unit WebRTCKeyFrameTemporalUnit, header IntraFrameHeaderParams) (int, error) {
	metadataFrames, err := webRTCKeyUnitScalabilityFrames(unit)
	if err != nil {
		return 0, err
	}
	seqSize, err := LowOverheadSequenceHeaderOBUSize(unit.Header.Sequence)
	if err != nil {
		return 0, err
	}
	frameSize, err := LowOverheadIntraFrameHeaderOBUSize(unit.Header.Sequence, header, 0, 0)
	if err != nil {
		return 0, err
	}
	size := lowOverheadOBUSizeUnchecked(OBU{Type: obu.TypeTemporalDelimiter}) + seqSize + frameSize
	if metadataSize, ok, err := lowOverheadWebRTCScalabilityMetadataOBUSizeForFrames(unit.Scalability, metadataFrames); err != nil {
		return 0, err
	} else if ok {
		size += metadataSize
	}
	return size, nil
}

func AppendLowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnit(dst []byte, unit WebRTCKeyFrameTemporalUnit, header IntraFrameHeaderParams) ([]byte, error) {
	size, err := LowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitSize(unit, header)
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
	out, err = AppendLowOverheadSequenceHeaderOBU(out, unit.Header.Sequence)
	if err != nil {
		return dst, err
	}
	metadataFrames, err := webRTCKeyUnitScalabilityFrames(unit)
	if err != nil {
		return dst, err
	}
	if next, _, err := appendLowOverheadWebRTCScalabilityMetadataOBUForFrames(out, unit.Scalability, metadataFrames); err != nil {
		return dst, err
	} else {
		out = next
	}
	out, err = AppendLowOverheadIntraFrameHeaderOBU(out, unit.Header.Sequence, header, 0, 0)
	if err != nil {
		return dst, err
	}
	return out, nil
}

// LowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitForStateSize returns the
// exact emitted byte size for the next config/state-derived WebRTC key temporal
// unit with a complete frame-header OBU.
func LowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitForStateSize(config Config, state WebRTCEncoderState) (int, WebRTCKeyFrameTemporalUnit, WebRTCEncoderState, IntraFrameHeaderParams, error) {
	unit, next, err := WebRTCKeyFrameTemporalUnitForState(config, state)
	if err != nil {
		return 0, WebRTCKeyFrameTemporalUnit{}, WebRTCEncoderState{}, IntraFrameHeaderParams{}, err
	}
	header, err := completeIntraFrameHeaderParams(unit.Header.Sequence, unit.Header.Prefix, unit.Header.Size, unit.Frames[0].RateControl, unit.Frames[0].Quantizer)
	if err != nil {
		return 0, WebRTCKeyFrameTemporalUnit{}, WebRTCEncoderState{}, IntraFrameHeaderParams{}, err
	}
	size, err := LowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitSize(unit, header)
	if err != nil {
		return 0, WebRTCKeyFrameTemporalUnit{}, WebRTCEncoderState{}, IntraFrameHeaderParams{}, err
	}
	return size, unit, next, header, nil
}

// AppendLowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitForState appends the
// next config/state-derived WebRTC key temporal unit with a complete frame-
// header OBU.
func AppendLowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitForState(dst []byte, config Config, state WebRTCEncoderState) ([]byte, WebRTCKeyFrameTemporalUnit, WebRTCEncoderState, IntraFrameHeaderParams, error) {
	_, unit, next, header, err := LowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnitForStateSize(config, state)
	if err != nil {
		return dst, WebRTCKeyFrameTemporalUnit{}, WebRTCEncoderState{}, IntraFrameHeaderParams{}, err
	}
	out, err := AppendLowOverheadWebRTCCompleteKeyFrameHeaderTemporalUnit(dst, unit, header)
	if err != nil {
		return dst, WebRTCKeyFrameTemporalUnit{}, WebRTCEncoderState{}, IntraFrameHeaderParams{}, err
	}
	return out, unit, next, header, nil
}

func webRTCKeyUnitScalabilityFrames(unit WebRTCKeyFrameTemporalUnit) ([]FrameEncodeSettings, error) {
	if unit.FrameNum > WebRTCMaxSpatialLayers {
		return nil, ErrInvalidFrame
	}
	return unit.Frames[:unit.FrameNum], nil
}

// LowOverheadWebRTCCompleteDeltaHeaderTemporalUnitSize returns the exact byte
// size of a low-overhead temporal unit carrying complete delta frame-header
// OBUs for each scheduled spatial layer.
func LowOverheadWebRTCCompleteDeltaHeaderTemporalUnitSize(unit WebRTCDeltaFrameTemporalUnit) (int, error) {
	if unit.FrameNum == 0 || unit.FrameNum > WebRTCMaxSpatialLayers {
		return 0, ErrInvalidFrame
	}
	size := lowOverheadOBUSizeUnchecked(OBU{Type: obu.TypeTemporalDelimiter})
	for i := uint8(0); i < unit.FrameNum; i++ {
		refs := completeHeaderReferenceStateForBuffers(unit.Headers[i], unit.Control.ReferenceState)
		header, err := completeInterFrameHeaderParams(unit.Headers[i], unit.Frames[i], &refs)
		if err != nil {
			return 0, err
		}
		headerSize, err := LowOverheadInterFrameHeaderOBUSize(unit.Headers[i].Sequence, header, unit.Headers[i].TemporalID, unit.Headers[i].SpatialID)
		if err != nil {
			return 0, err
		}
		size += headerSize
	}
	return size, nil
}

// AppendLowOverheadWebRTCCompleteDeltaHeaderTemporalUnit appends one temporal
// delimiter followed by complete delta frame-header OBUs for each scheduled
// spatial layer.
func AppendLowOverheadWebRTCCompleteDeltaHeaderTemporalUnit(dst []byte, unit WebRTCDeltaFrameTemporalUnit) ([]byte, error) {
	size, err := LowOverheadWebRTCCompleteDeltaHeaderTemporalUnitSize(unit)
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
		refs := completeHeaderReferenceStateForBuffers(unit.Headers[i], unit.Control.ReferenceState)
		header, err := completeInterFrameHeaderParams(unit.Headers[i], unit.Frames[i], &refs)
		if err != nil {
			return dst, err
		}
		out, err = AppendLowOverheadInterFrameHeaderOBU(out, unit.Headers[i].Sequence, header, unit.Headers[i].TemporalID, unit.Headers[i].SpatialID)
		if err != nil {
			return dst, err
		}
	}
	return out, nil
}

func LowOverheadWebRTCCompleteDeltaHeaderTemporalUnitForStateSize(config Config, state WebRTCEncoderState) (int, WebRTCDeltaFrameTemporalUnit, WebRTCEncoderState, error) {
	unit, next, err := WebRTCDeltaFrameTemporalUnitForState(config, state)
	if err != nil {
		return 0, WebRTCDeltaFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	size, err := LowOverheadWebRTCCompleteDeltaHeaderTemporalUnitSize(unit)
	if err != nil {
		return 0, WebRTCDeltaFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	return size, unit, next, nil
}

func AppendLowOverheadWebRTCCompleteDeltaHeaderTemporalUnitForState(dst []byte, config Config, state WebRTCEncoderState) ([]byte, WebRTCDeltaFrameTemporalUnit, WebRTCEncoderState, error) {
	_, unit, next, err := LowOverheadWebRTCCompleteDeltaHeaderTemporalUnitForStateSize(config, state)
	if err != nil {
		return dst, WebRTCDeltaFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	out, err := AppendLowOverheadWebRTCCompleteDeltaHeaderTemporalUnit(dst, unit)
	if err != nil {
		return dst, WebRTCDeltaFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	return out, unit, next, nil
}

func WebRTCNextTemporalUnitForState(config Config, state WebRTCEncoderState, forceKeyFrame bool) (WebRTCPictureTemporalUnit, WebRTCEncoderState, error) {
	config, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return WebRTCPictureTemporalUnit{}, WebRTCEncoderState{}, err
	}
	needsKey, err := webRTCEncoderStateNeedsKey(config, state)
	if err != nil {
		return WebRTCPictureTemporalUnit{}, WebRTCEncoderState{}, err
	}
	if forceKeyFrame || needsKey {
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

func webRTCEncoderStateNeedsKey(config Config, state WebRTCEncoderState) (bool, error) {
	if !state.DependencyStructureState.Valid || state.DeltaPictureIndex == 0 {
		return true, nil
	}
	if config.KeyFrameInterval > 0 && state.DeltaPictureIndex >= uint64(config.KeyFrameInterval) {
		return true, nil
	}
	structure, err := WebRTCFrameDependencyStructureForConfig(config)
	if err != nil {
		return false, err
	}
	return state.DependencyStructureState.Structure != structure, nil
}

func WebRTCDeltaFrameTemporalUnitForState(config Config, state WebRTCEncoderState) (WebRTCDeltaFrameTemporalUnit, WebRTCEncoderState, error) {
	config, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return WebRTCDeltaFrameTemporalUnit{}, WebRTCEncoderState{}, err
	}
	unit, err := WebRTCDeltaFrameTemporalUnitForConfigWithDeltaPictureIndex(config, state.ReferenceState, state.FrameIDState, state.DeltaPictureIndex, state.NextFrameID, state.NextOrderHint)
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
	size, err := LowOverheadWebRTCKeyFrameHeaderTemporalUnitSize(unit)
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
	out, err := AppendLowOverheadWebRTCKeyFrameHeaderTemporalUnit(dst, unit)
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
	return webRTCDeltaFrameTemporalUnitForConfigWithOrderHint(config, referenceState, frameIDState, temporalID, firstFrameID, orderHint, 0, false)
}

func WebRTCDeltaFrameTemporalUnitForConfigWithDeltaPictureIndex(config Config, referenceState ReferenceBufferState, frameIDState FrameIDBufferState, deltaPictureIndex uint64, firstFrameID uint64, orderHint uint8) (WebRTCDeltaFrameTemporalUnit, error) {
	config, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return WebRTCDeltaFrameTemporalUnit{}, err
	}
	temporalID, err := WebRTCTemporalIDForDeltaPicture(config.TemporalLayerCount, deltaPictureIndex)
	if err != nil {
		return WebRTCDeltaFrameTemporalUnit{}, err
	}
	return webRTCDeltaFrameTemporalUnitForConfigWithOrderHint(config, referenceState, frameIDState, temporalID, firstFrameID, orderHint, deltaPictureIndex, true)
}

func webRTCDeltaFrameTemporalUnitForConfigWithOrderHint(config Config, referenceState ReferenceBufferState, frameIDState FrameIDBufferState, temporalID uint8, firstFrameID uint64, orderHint uint8, deltaPictureIndex uint64, hasDeltaPictureIndex bool) (WebRTCDeltaFrameTemporalUnit, error) {
	config, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return WebRTCDeltaFrameTemporalUnit{}, err
	}
	if config.SpatialLayerCount == 0 || config.SpatialLayerCount > WebRTCMaxSpatialLayers ||
		temporalID >= config.TemporalLayerCount {
		return WebRTCDeltaFrameTemporalUnit{}, ErrInvalidConfig
	}
	if config.Scalability == ScalabilityModeL2T2_KEY_SHIFT {
		return webRTCL2T2KeyShiftDeltaFrameTemporalUnitForConfigWithOrderHint(config, referenceState, frameIDState, temporalID, firstFrameID, orderHint)
	}

	var unit WebRTCDeltaFrameTemporalUnit
	unit.FrameNum = config.SpatialLayerCount
	for i := uint8(0); i < unit.FrameNum; i++ {
		layerTemporalID := webRTCLayerTemporalIDForDeltaPicture(config, i, temporalID, deltaPictureIndex, hasDeltaPictureIndex)
		settings := webRTCFrameEncodeSettingsForLayer(config, i, layerTemporalID)

		refBuffer, updateBuffer, update, err := webRTCTemporalReferenceBufferForLayer(config, i, layerTemporalID, deltaPictureIndex, hasDeltaPictureIndex)
		if err != nil {
			return WebRTCDeltaFrameTemporalUnit{}, err
		}
		settings.ReferenceBuffers[settings.ReferenceCount] = refBuffer
		settings.ReferenceCount++
		if update {
			settings.UpdateBuffer = updateBuffer
			settings.UpdateBufferSet = true
		}
		if webRTCUsesDeltaInterLayerReference(config) {
			if i+1 < unit.FrameNum && !settings.UpdateBufferSet {
				currentBuffer, ok := webRTCInterLayerCurrentBuffer(config, i)
				if !ok {
					return WebRTCDeltaFrameTemporalUnit{}, ErrInvalidConfig
				}
				settings.UpdateBuffer = currentBuffer
				settings.UpdateBufferSet = true
			}
			if i > 0 {
				lower := unit.Frames[i-1]
				if !lower.UpdateBufferSet {
					return WebRTCDeltaFrameTemporalUnit{}, ErrInvalidFrame
				}
				settings.ReferenceBuffers[settings.ReferenceCount] = lower.UpdateBuffer
				settings.ReferenceCount++
			}
		}
		if settings.ReferenceCount > WebRTCMaxFrameReferences {
			return WebRTCDeltaFrameTemporalUnit{}, ErrInvalidFrame
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

func webRTCLayerTemporalIDForDeltaPicture(config Config, spatialID uint8, fallbackTemporalID uint8, deltaPictureIndex uint64, hasDeltaPictureIndex bool) uint8 {
	if !hasDeltaPictureIndex {
		return fallbackTemporalID
	}
	temporalID, err := WebRTCTemporalIDForDeltaPicture(config.TemporalLayerCount, deltaPictureIndex)
	if err != nil {
		return fallbackTemporalID
	}
	if !config.Scalability.UsesKeyFrameInterLayerDependencyShift() {
		return temporalID
	}
	switch config.Scalability {
	case ScalabilityModeL2T2_KEY_SHIFT:
		table := [2][2]uint8{{0, 1}, {1, 0}}
		return table[(deltaPictureIndex-1)%2][spatialID]
	case ScalabilityModeL2T3_KEY_SHIFT:
		table := [4][2]uint8{{2, 2}, {0, 1}, {2, 2}, {1, 0}}
		return table[(deltaPictureIndex-1)%4][spatialID]
	case ScalabilityModeL3T2_KEY_SHIFT:
		table := [2][3]uint8{{0, 0, 1}, {1, 1, 0}}
		return table[(deltaPictureIndex-1)%2][spatialID]
	case ScalabilityModeL3T3_KEY_SHIFT:
		table := [4][3]uint8{{0, 2, 2}, {2, 0, 1}, {1, 2, 2}, {2, 1, 0}}
		return table[(deltaPictureIndex-1)%4][spatialID]
	default:
		return temporalID
	}
}

func webRTCTemporalReferenceBufferForLayer(config Config, spatialID uint8, temporalID uint8, deltaPictureIndex uint64, hasDeltaPictureIndex bool) (referenceBuffer uint8, updateBuffer uint8, update bool, err error) {
	if spatialID >= config.SpatialLayerCount || temporalID >= config.TemporalLayerCount {
		return 0, 0, false, ErrInvalidConfig
	}
	baseBuffer := spatialID
	switch config.TemporalLayerCount {
	case 1:
		return baseBuffer, baseBuffer, true, nil
	case 2:
		if temporalID == 0 {
			return baseBuffer, baseBuffer, true, nil
		}
		return baseBuffer, 0, false, nil
	case 3:
		middleBuffer := config.SpatialLayerCount + spatialID
		if middleBuffer >= WebRTCReferenceBuffers {
			return 0, 0, false, ErrInvalidConfig
		}
		switch temporalID {
		case 0:
			return baseBuffer, baseBuffer, true, nil
		case 1:
			return baseBuffer, middleBuffer, true, nil
		case 2:
			referenceBuffer := baseBuffer
			if hasDeltaPictureIndex && deltaPictureIndex > 1 {
				previousTemporalID := webRTCLayerTemporalIDForDeltaPicture(config, spatialID, 0, deltaPictureIndex-1, true)
				if previousTemporalID == 1 {
					referenceBuffer = middleBuffer
				}
			}
			return referenceBuffer, 0, false, nil
		}
	}
	return 0, 0, false, ErrInvalidConfig
}

func webRTCUsesDeltaInterLayerReference(config Config) bool {
	return config.SpatialLayerCount > 1 &&
		!config.Scalability.IsSimulcast() &&
		!config.Scalability.UsesKeyFrameInterLayerDependency()
}

func webRTCInterLayerCurrentBuffer(config Config, spatialID uint8) (uint8, bool) {
	if !webRTCUsesDeltaInterLayerReference(config) || spatialID+1 >= config.SpatialLayerCount {
		return 0, false
	}
	start := config.SpatialLayerCount
	if config.TemporalLayerCount == 3 {
		start = config.SpatialLayerCount * 2
	}
	buffer := start + spatialID
	return buffer, buffer < WebRTCReferenceBuffers
}

// Ported from libwebrtc:
// modules/video_coding/svc/scalability_structure_l2t2_key_shift.cc
func webRTCL2T2KeyShiftDeltaFrameTemporalUnitForConfigWithOrderHint(config Config, referenceState ReferenceBufferState, frameIDState FrameIDBufferState, temporalID uint8, firstFrameID uint64, orderHint uint8) (WebRTCDeltaFrameTemporalUnit, error) {
	if config.SpatialLayerCount != 2 || config.TemporalLayerCount != 2 {
		return WebRTCDeltaFrameTemporalUnit{}, ErrInvalidConfig
	}
	var unit WebRTCDeltaFrameTemporalUnit
	unit.FrameNum = 2
	switch temporalID {
	case 1:
		unit.Frames[0] = webRTCFrameEncodeSettingsForLayer(config, 0, 0)
		unit.Frames[0].ReferenceBuffers[0] = 0
		unit.Frames[0].ReferenceCount = 1
		unit.Frames[0].UpdateBuffer = 0
		unit.Frames[0].UpdateBufferSet = true

		unit.Frames[1] = webRTCFrameEncodeSettingsForLayer(config, 1, 1)
		unit.Frames[1].ReferenceBuffers[0] = 1
		unit.Frames[1].ReferenceCount = 1
	case 0:
		unit.Frames[0] = webRTCFrameEncodeSettingsForLayer(config, 0, 1)
		unit.Frames[0].ReferenceBuffers[0] = 0
		unit.Frames[0].ReferenceCount = 1

		unit.Frames[1] = webRTCFrameEncodeSettingsForLayer(config, 1, 0)
		unit.Frames[1].ReferenceBuffers[0] = 1
		unit.Frames[1].ReferenceCount = 1
		unit.Frames[1].UpdateBuffer = 1
		unit.Frames[1].UpdateBufferSet = true
	default:
		return WebRTCDeltaFrameTemporalUnit{}, ErrInvalidConfig
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

func webRTCFrameEncodeSettingsForLayer(config Config, spatialID uint8, temporalID uint8) FrameEncodeSettings {
	layer := config.SpatialLayers[spatialID]
	return FrameEncodeSettings{
		Type:        FrameTypeDelta,
		Resolution:  layer.Resolution,
		SpatialID:   spatialID,
		TemporalID:  temporalID,
		EffortLevel: config.Speed,
		RateControl: config.RateControl,
		Quantizer:   config.Quantizer,
		Output:      true,
	}
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
		(settings.UpdateBufferSet && settings.UpdateBuffer >= WebRTCReferenceBuffers) {
		return InterHeaderFrame{}, ErrInvalidFrame
	}

	prefix := FrameHeaderPrefix{
		FrameType:               FrameHeaderTypeInter,
		ShowFrame:               true,
		ShowableFrame:           true,
		ErrorResilientMode:      true,
		AllowScreenContentTools: config.Content == ContentScreen,
		ForceIntegerMV:          config.Content == ContentScreen,
		FrameSizeOverride:       true,
		OrderHint:               orderHint,
		PrimaryRefFrame:         EncoderPrimaryRefNone,
	}
	if seq.FrameIDNumbersPresent {
		prefix.FrameID = uint16(frameID & uint64((uint32(1)<<sequenceFrameIDBits(seq))-1))
	}
	size := InterFrameSize{
		UpscaledWidth:       uint32(settings.Resolution.Width),
		Height:              uint32(settings.Resolution.Height),
		SuperResDenominator: 8,
	}
	if settings.UpdateBufferSet {
		size.RefreshFrameFlags = 1 << settings.UpdateBuffer
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

func completeIntraFrameHeaderParams(seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize, rc RateControlMode, quantizer uint8) (IntraFrameHeaderParams, error) {
	quant, allLossless, err := defaultCompleteHeaderQuantization(rc, quantizer)
	if err != nil {
		return IntraFrameHeaderParams{}, err
	}
	tiles, err := defaultCompleteHeaderTileInfo(seq, prefix, codedWidthFromIntraFrameSize(size), size.Height)
	if err != nil {
		return IntraFrameHeaderParams{}, err
	}
	header := IntraFrameHeaderParams{
		Prefix:       prefix,
		Size:         size,
		Tile:         tiles,
		Quantization: quant,
		LoopFilter:   defaultCompleteHeaderLoopFilter(allLossless),
		CDEF:         defaultCompleteHeaderCDEF(seq, allLossless, size.AllowIntrabc),
		Restoration:  RestorationParams{},
		TransformRef: defaultCompleteHeaderTransform(prefix, allLossless),
		FrameMode:    FrameModeParams{},
		FilmGrain:    FilmGrainParams{},
		AllLossless:  allLossless,
	}
	if _, err := IntraFrameHeaderPayloadSize(seq, header); err != nil {
		return IntraFrameHeaderParams{}, err
	}
	return header, nil
}

func completeInterFrameHeaderParams(frame InterHeaderFrame, settings FrameEncodeSettings, refs *parser.ReferenceState) (InterFrameHeaderParams, error) {
	if refs == nil {
		return InterFrameHeaderParams{}, ErrInvalidFrame
	}
	if !referenceStateHasValidFrame(refs) {
		*refs = completeHeaderReferenceState(frame)
	}
	for i := uint8(0); i < settings.ReferenceCount; i++ {
		ref := settings.ReferenceBuffers[i]
		if ref >= parser.RefFrames || !refs.Frames[ref].Valid {
			return InterFrameHeaderParams{}, ErrInvalidFrame
		}
	}
	quant, allLossless, err := defaultCompleteHeaderQuantization(settings.RateControl, settings.Quantizer)
	if err != nil {
		return InterFrameHeaderParams{}, err
	}
	tiles, err := defaultCompleteHeaderTileInfo(frame.Sequence, frame.Prefix, codedWidthFromInterFrameSize(frame.Size), frame.Size.Height)
	if err != nil {
		return InterFrameHeaderParams{}, err
	}
	header := InterFrameHeaderParams{
		Prefix:       frame.Prefix,
		Size:         frame.Size,
		Tile:         tiles,
		Quantization: quant,
		LoopFilter:   defaultCompleteHeaderLoopFilter(allLossless),
		CDEF:         defaultCompleteHeaderCDEF(frame.Sequence, allLossless, false),
		Restoration:  RestorationParams{},
		TransformRef: defaultCompleteHeaderTransform(frame.Prefix, allLossless),
		SkipMode:     SkipModeParams{},
		FrameMode:    FrameModeParams{},
		GlobalMotion: DefaultGlobalMotionParams(),
		FilmGrain:    FilmGrainParams{},
		AllLossless:  allLossless,
		References:   refs,
	}
	if _, err := InterFrameHeaderPayloadSize(frame.Sequence, header); err != nil {
		return InterFrameHeaderParams{}, err
	}
	return header, nil
}

func defaultCompleteHeaderQuantization(rc RateControlMode, quantizer uint8) (QuantizationParams, bool, error) {
	if !rc.Valid() {
		return QuantizationParams{}, false, ErrInvalidFrame
	}
	baseQ := uint8(32)
	if rc == RateControlCQP {
		if quantizer > WebRTCMaxQuantizer {
			return QuantizationParams{}, false, ErrInvalidFrame
		}
		baseQ = quantizer
	}
	quant := QuantizationParams{BaseQIdx: baseQ}
	return quant, baseQ == 0, nil
}

func defaultCompleteHeaderLoopFilter(allLossless bool) LoopFilterParams {
	lf := LoopFilterParams{Deltas: defaultLoopFilterDeltas()}
	if allLossless {
		lf.ModeRefDeltaEnabled = true
		lf.ModeRefDeltaUpdate = true
	}
	return lf
}

func defaultCompleteHeaderCDEF(seq SequenceHeader, allLossless bool, allowIntrabc bool) CDEFParams {
	if allLossless || !seq.EnableCDEF || allowIntrabc {
		return CDEFParams{}
	}
	return CDEFParams{Damping: 3}
}

func defaultCompleteHeaderTransform(prefix FrameHeaderPrefix, allLossless bool) TransformReferenceParams {
	if allLossless {
		return TransformReferenceParams{TransformMode: TransformMode4x4Only, ReferenceMode: ReferenceModeSingle}
	}
	return TransformReferenceParams{TransformMode: TransformModeLargest, ReferenceMode: ReferenceModeSingle}
}

func defaultCompleteHeaderTileInfo(seq SequenceHeader, prefix FrameHeaderPrefix, codedWidth uint32, height uint32) (TileInfo, error) {
	derived, err := deriveEncoderTileInfo(seq, codedWidth, height, TileInfo{})
	if err != nil {
		return TileInfo{}, err
	}
	tiles := TileInfo{
		RefreshContext:      !seq.ReducedStillPictureHeader && !prefix.DisableCDFUpdate,
		UniformSpacing:      true,
		SBCols:              derived.SBCols,
		SBRows:              derived.SBRows,
		MinLog2Cols:         derived.MinLog2Cols,
		MaxLog2Cols:         derived.MaxLog2Cols,
		MaxLog2Rows:         derived.MaxLog2Rows,
		MinLog2Tiles:        derived.MinLog2Tiles,
		Log2Cols:            derived.MinLog2Cols,
		InterpolationFilter: InterpolationEightTap,
	}
	if !prefix.FrameType.interOrSwitch() {
		tiles.InterpolationFilter = 0
	}
	minRows := uint8(0)
	if derived.MinLog2Tiles > tiles.Log2Cols {
		minRows = derived.MinLog2Tiles - tiles.Log2Cols
	}
	tiles.MinLog2Rows = minRows
	tiles.Log2Rows = minRows
	fillUniformTileStarts(&tiles, derived)
	if tiles.Log2Cols != 0 || tiles.Log2Rows != 0 {
		tiles.TileSizeBytes = 4
	}
	if err := validateTileInfo(seq, prefix, tiles, derived); err != nil {
		return TileInfo{}, err
	}
	return tiles, nil
}

func fillUniformTileStarts(tiles *TileInfo, derived encoderTileDerived) {
	tileW := uint16(1 + ((uint32(derived.SBCols) - 1) >> tiles.Log2Cols))
	cols := uint8(0)
	for sbx := uint16(0); sbx < derived.SBCols && cols < MaxTileCols; sbx += tileW {
		tiles.ColStartSB[cols] = sbx
		cols++
	}
	tiles.ColStartSB[cols] = derived.SBCols
	tiles.Cols = cols

	tileH := uint16(1 + ((uint32(derived.SBRows) - 1) >> tiles.Log2Rows))
	rows := uint8(0)
	for sby := uint16(0); sby < derived.SBRows && rows < MaxTileRows; sby += tileH {
		tiles.RowStartSB[rows] = sby
		rows++
	}
	tiles.RowStartSB[rows] = derived.SBRows
	tiles.Rows = rows
}

func completeHeaderReferenceState(frame InterHeaderFrame) parser.ReferenceState {
	var refs parser.ReferenceState
	defaultGlobal := parser.DefaultGlobalMotionParams()
	for i := uint8(0); i < parser.RefFrames; i++ {
		refs.Frames[i] = parser.ReferenceFrame{
			Valid:        true,
			OrderHint:    frame.Prefix.OrderHint,
			GlobalMotion: defaultGlobal,
			Size: parser.FrameSize{
				CodedWidth:          codedWidthFromInterFrameSize(frame.Size),
				UpscaledWidth:       frame.Size.UpscaledWidth,
				Height:              frame.Size.Height,
				RenderWidth:         frame.Size.RenderWidth,
				RenderHeight:        frame.Size.RenderHeight,
				SuperResDenominator: frame.Size.SuperResDenominator,
			},
		}
	}
	return refs
}

func completeHeaderReferenceStateForBuffers(frame InterHeaderFrame, state ReferenceBufferState) parser.ReferenceState {
	var refs parser.ReferenceState
	defaultGlobal := parser.DefaultGlobalMotionParams()
	for i := uint8(0); i < parser.RefFrames; i++ {
		if !state.Valid[i] || !state.Resolutions[i].Valid() {
			continue
		}
		resolution := state.Resolutions[i]
		refs.Frames[i] = parser.ReferenceFrame{
			Valid:        true,
			OrderHint:    frame.Prefix.OrderHint,
			GlobalMotion: defaultGlobal,
			Size: parser.FrameSize{
				CodedWidth:          uint32(resolution.Width),
				UpscaledWidth:       uint32(resolution.Width),
				Height:              uint32(resolution.Height),
				RenderWidth:         uint32(resolution.Width),
				RenderHeight:        uint32(resolution.Height),
				SuperResDenominator: 8,
			},
		}
	}
	return refs
}

func referenceStateHasValidFrame(refs *parser.ReferenceState) bool {
	if refs == nil {
		return false
	}
	for i := uint8(0); i < parser.RefFrames; i++ {
		if refs.Frames[i].Valid {
			return true
		}
	}
	return false
}
