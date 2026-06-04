package encoder

const (
	WebRTCRtpDependencyMaxDecodeTargets = MaxLayers
	WebRTCRtpDependencyMaxTemplates     = MaxLayers
	LibaomSVCReferenceSlots             = 7
)

// LibaomSVCRefFrameConfig is the Go, fixed-width equivalent of libaom's
// aom_svc_ref_frame_config_t control payload.
type LibaomSVCRefFrameConfig struct {
	Reference [LibaomSVCReferenceSlots]int32
	RefIdx    [LibaomSVCReferenceSlots]int32
	Refresh   [WebRTCReferenceBuffers]int32
}

func LibaomSVCRefFrameConfigForFrame(settings FrameEncodeSettings) (LibaomSVCRefFrameConfig, error) {
	if !settings.Type.Valid() || !settings.Resolution.Valid() ||
		settings.ReferenceCount > WebRTCMaxFrameReferences {
		return LibaomSVCRefFrameConfig{}, ErrInvalidFrame
	}
	var config LibaomSVCRefFrameConfig
	firstRef := int32(0)
	if settings.ReferenceCount > 0 {
		first := settings.ReferenceBuffers[0]
		if first >= WebRTCReferenceBuffers {
			return LibaomSVCRefFrameConfig{}, ErrInvalidFrame
		}
		firstRef = int32(first)
	}
	for i := range config.RefIdx {
		config.RefIdx[i] = firstRef
	}
	for i := uint8(0); i < settings.ReferenceCount; i++ {
		ref := settings.ReferenceBuffers[i]
		if ref >= WebRTCReferenceBuffers || duplicateReference(settings.ReferenceBuffers, settings.ReferenceCount, i) {
			return LibaomSVCRefFrameConfig{}, ErrInvalidFrame
		}
		config.Reference[i] = 1
		config.RefIdx[i] = int32(ref)
	}
	if settings.UpdateBufferSet {
		if settings.UpdateBuffer >= WebRTCReferenceBuffers {
			return LibaomSVCRefFrameConfig{}, ErrInvalidFrame
		}
		config.Refresh[settings.UpdateBuffer] = 1
	}
	return config, nil
}

type DecodeTargetIndication uint8

const (
	DecodeTargetNotPresent DecodeTargetIndication = iota
	DecodeTargetDiscardable
	DecodeTargetSwitch
	DecodeTargetRequired
)

func (d DecodeTargetIndication) Valid() bool {
	return d <= DecodeTargetRequired
}

type FrameIDBufferState struct {
	Valid    [WebRTCReferenceBuffers]bool
	FrameIDs [WebRTCReferenceBuffers]uint64
}

type WebRTCGenericFrameInfo struct {
	FrameID       uint64
	SpatialID     uint8
	TemporalID    uint8
	Dependencies  [WebRTCMaxFrameReferences]uint64
	DependencyNum uint8
	DTIs          [WebRTCRtpDependencyMaxDecodeTargets]DecodeTargetIndication
	DTINum        uint8
}

func WebRTCGenericFrameInfoForFrame(settings FrameEncodeSettings, frameID uint64, state FrameIDBufferState, spatialLayers uint8, temporalLayers uint8) (WebRTCGenericFrameInfo, FrameIDBufferState, error) {
	if spatialLayers == 0 || spatialLayers > WebRTCMaxSpatialLayers ||
		temporalLayers == 0 || temporalLayers > WebRTCMaxTemporalLayers ||
		settings.SpatialID >= spatialLayers || settings.TemporalID >= temporalLayers ||
		settings.ReferenceCount > WebRTCMaxFrameReferences {
		return WebRTCGenericFrameInfo{}, FrameIDBufferState{}, ErrInvalidFrame
	}

	info := WebRTCGenericFrameInfo{
		FrameID:    frameID,
		SpatialID:  settings.SpatialID,
		TemporalID: settings.TemporalID,
		DTINum:     spatialLayers,
	}
	fillWebRTCDTIs(&info, settings, spatialLayers)

	for i := uint8(0); i < settings.ReferenceCount; i++ {
		ref := settings.ReferenceBuffers[i]
		if ref >= WebRTCReferenceBuffers || duplicateReference(settings.ReferenceBuffers, settings.ReferenceCount, i) || !state.Valid[ref] {
			return WebRTCGenericFrameInfo{}, FrameIDBufferState{}, ErrInvalidFrame
		}
		info.Dependencies[i] = state.FrameIDs[ref]
		info.DependencyNum++
	}

	out := state
	if settings.Type.resetsReferences() {
		out = FrameIDBufferState{}
	}
	if settings.UpdateBufferSet {
		if settings.UpdateBuffer >= WebRTCReferenceBuffers {
			return WebRTCGenericFrameInfo{}, FrameIDBufferState{}, ErrInvalidFrame
		}
		out.Valid[settings.UpdateBuffer] = true
		out.FrameIDs[settings.UpdateBuffer] = frameID
	}
	return info, out, nil
}

type WebRTCFrameDependencyTemplate struct {
	SpatialID    uint8
	TemporalID   uint8
	DTIs         [WebRTCRtpDependencyMaxDecodeTargets]DecodeTargetIndication
	DTINum       uint8
	FrameDiffs   [WebRTCMaxFrameReferences]uint16
	FrameDiffNum uint8
	ChainDiffs   [WebRTCRtpDependencyMaxDecodeTargets]uint8
	ChainDiffNum uint8
}

type WebRTCFrameDependencyStructure struct {
	StructureID                  uint8
	NumDecodeTargets             uint8
	NumChains                    uint8
	DecodeTargetProtectedByChain [WebRTCRtpDependencyMaxDecodeTargets]uint8
	Templates                    [WebRTCRtpDependencyMaxTemplates]WebRTCFrameDependencyTemplate
	TemplateNum                  uint8
	Resolutions                  [WebRTCMaxSpatialLayers]Resolution
	ResolutionNum                uint8
}

func WebRTCFrameDependencyStructureForConfig(config Config) (WebRTCFrameDependencyStructure, error) {
	config, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return WebRTCFrameDependencyStructure{}, err
	}
	if config.SpatialLayerCount == 0 || config.TemporalLayerCount == 0 ||
		uint16(config.SpatialLayerCount)*uint16(config.TemporalLayerCount) > WebRTCRtpDependencyMaxTemplates {
		return WebRTCFrameDependencyStructure{}, ErrInvalidConfig
	}

	var structure WebRTCFrameDependencyStructure
	structure.NumDecodeTargets = config.SpatialLayerCount
	structure.NumChains = config.SpatialLayerCount
	structure.ResolutionNum = config.SpatialLayerCount
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		structure.DecodeTargetProtectedByChain[i] = i
		structure.Resolutions[i] = config.SpatialLayers[i].Resolution
	}

	for spatial := uint8(0); spatial < config.SpatialLayerCount; spatial++ {
		for temporal := uint8(0); temporal < config.TemporalLayerCount; temporal++ {
			template := &structure.Templates[structure.TemplateNum]
			template.SpatialID = spatial
			template.TemporalID = temporal
			template.DTINum = config.SpatialLayerCount
			template.ChainDiffNum = config.SpatialLayerCount
			fillTemplateDTIs(template, spatial, temporal, config.SpatialLayerCount)
			structure.TemplateNum++
		}
	}
	return structure, nil
}

func fillWebRTCDTIs(info *WebRTCGenericFrameInfo, settings FrameEncodeSettings, spatialLayers uint8) {
	for target := uint8(0); target < spatialLayers; target++ {
		info.DTIs[target] = webRTCDTIForLayer(settings.Type, settings.SpatialID, settings.TemporalID, target)
	}
}

func fillTemplateDTIs(template *WebRTCFrameDependencyTemplate, spatialID uint8, temporalID uint8, spatialLayers uint8) {
	for target := uint8(0); target < spatialLayers; target++ {
		template.DTIs[target] = webRTCDTIForLayer(FrameTypeDelta, spatialID, temporalID, target)
	}
}

func webRTCDTIForLayer(frameType FrameType, spatialID uint8, temporalID uint8, target uint8) DecodeTargetIndication {
	if target < spatialID {
		return DecodeTargetNotPresent
	}
	if target > spatialID {
		if frameType.resetsReferences() && temporalID == 0 {
			return DecodeTargetSwitch
		}
		return DecodeTargetNotPresent
	}
	if temporalID == 0 || frameType.resetsReferences() {
		return DecodeTargetSwitch
	}
	return DecodeTargetDiscardable
}
