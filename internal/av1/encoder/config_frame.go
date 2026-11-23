package encoder

import (
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
	applyWebRTCReferenceTemplateHints(config, &control, 0, false)
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
	applyWebRTCReferenceTemplateHints(config, &control, deltaPictureIndex, hasDeltaPictureIndex)
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
	applyWebRTCReferenceTemplateHints(config, &control, 0, false)
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
		FrameType:          FrameHeaderTypeInter,
		ShowFrame:          true,
		ShowableFrame:      true,
		ErrorResilientMode: true,
		// SVT rtc: screen tools on I-slices only, force_integer_mv 0.
		FrameSizeOverride: true,
		OrderHint:         orderHint,
		PrimaryRefFrame:   EncoderPrimaryRefNone,
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
