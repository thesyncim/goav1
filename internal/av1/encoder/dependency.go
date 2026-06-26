package encoder

const (
	WebRTCRtpDependencyMaxDecodeTargets = MaxLayers
	WebRTCRtpDependencyMaxTemplates     = 64
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
	FrameID              uint64
	SpatialID            uint8
	TemporalID           uint8
	Dependencies         [WebRTCMaxFrameReferences]uint64
	DependencyNum        uint8
	DTIs                 [WebRTCRtpDependencyMaxDecodeTargets]DecodeTargetIndication
	DTINum               uint8
	TemplateIndexHint    uint8
	TemplateIndexHintSet bool
}

func WebRTCGenericFrameInfoForFrame(settings FrameEncodeSettings, frameID uint64, state FrameIDBufferState, spatialLayers uint8, temporalLayers uint8) (WebRTCGenericFrameInfo, FrameIDBufferState, error) {
	if spatialLayers == 0 || spatialLayers > WebRTCMaxSpatialLayers ||
		temporalLayers == 0 || temporalLayers > WebRTCMaxTemporalLayers ||
		uint16(spatialLayers)*uint16(temporalLayers) > WebRTCRtpDependencyMaxDecodeTargets ||
		settings.SpatialID >= spatialLayers || settings.TemporalID >= temporalLayers ||
		settings.ReferenceCount > WebRTCMaxFrameReferences {
		return WebRTCGenericFrameInfo{}, FrameIDBufferState{}, ErrInvalidFrame
	}

	info := WebRTCGenericFrameInfo{
		FrameID:    frameID,
		SpatialID:  settings.SpatialID,
		TemporalID: settings.TemporalID,
		DTINum:     spatialLayers * temporalLayers,
	}
	fillWebRTCDTIs(&info, settings, spatialLayers, temporalLayers, ScalabilityModeL1T1, false)

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

type WebRTCDependencyStructureState struct {
	Valid     bool
	Structure WebRTCFrameDependencyStructure
}

type WebRTCFrameControl struct {
	Settings                  FrameEncodeSettings
	LibaomSVCRefFrameConfig   LibaomSVCRefFrameConfig
	GenericFrameInfo          WebRTCGenericFrameInfo
	AttachDependencyStructure bool
}

type WebRTCTemporalUnitControl struct {
	Frames                 [WebRTCMaxSpatialLayers]WebRTCFrameControl
	FrameNum               uint8
	ReferenceState         ReferenceBufferState
	FrameIDState           FrameIDBufferState
	DependencyStructure    WebRTCFrameDependencyStructure
	HasDependencyStructure bool
}

func WebRTCTemporalUnitControlForFrames(config Config, frames []FrameEncodeSettings, referenceState ReferenceBufferState, frameIDState FrameIDBufferState, firstFrameID uint64) (WebRTCTemporalUnitControl, error) {
	config, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return WebRTCTemporalUnitControl{}, err
	}
	if len(frames) == 0 || len(frames) > int(config.SpatialLayerCount) || len(frames) > WebRTCMaxSpatialLayers {
		return WebRTCTemporalUnitControl{}, ErrInvalidFrame
	}

	nextReferenceState, err := ValidateTemporalUnitFrames(frames, referenceState, config.RateControl)
	if err != nil {
		return WebRTCTemporalUnitControl{}, err
	}

	var control WebRTCTemporalUnitControl
	control.FrameNum = uint8(len(frames))
	control.ReferenceState = nextReferenceState
	nextFrameIDState := frameIDState
	keyTemporalUnit := false
	for i := range frames {
		if frames[i].Type == FrameTypeKey {
			keyTemporalUnit = true
			break
		}
	}
	for i := range frames {
		settings := frames[i]
		if settings.SpatialID >= config.SpatialLayerCount || settings.TemporalID >= config.TemporalLayerCount {
			return WebRTCTemporalUnitControl{}, ErrInvalidFrame
		}

		frameID := firstFrameID + uint64(i)
		frameControl := &control.Frames[i]
		frameControl.Settings = settings
		frameControl.LibaomSVCRefFrameConfig, err = LibaomSVCRefFrameConfigForFrame(settings)
		if err != nil {
			return WebRTCTemporalUnitControl{}, err
		}
		frameControl.GenericFrameInfo, nextFrameIDState, err = webRTCGenericFrameInfoForFrameMode(settings, frameID, nextFrameIDState, config, keyTemporalUnit)
		if err != nil {
			return WebRTCTemporalUnitControl{}, err
		}
		if settings.Type == FrameTypeKey {
			frameControl.AttachDependencyStructure = true
			control.HasDependencyStructure = true
		}
	}
	control.FrameIDState = nextFrameIDState
	if control.HasDependencyStructure {
		control.DependencyStructure, err = WebRTCFrameDependencyStructureForConfig(config)
		if err != nil {
			return WebRTCTemporalUnitControl{}, err
		}
	}
	return control, nil
}

func WebRTCDependencyStructureStateForTemporalUnit(control WebRTCTemporalUnitControl, state WebRTCDependencyStructureState) (WebRTCDependencyStructureState, WebRTCFrameDependencyStructure, error) {
	if control.FrameNum == 0 || control.FrameNum > WebRTCMaxSpatialLayers {
		return WebRTCDependencyStructureState{}, WebRTCFrameDependencyStructure{}, ErrInvalidFrame
	}

	structure := state.Structure
	if control.HasDependencyStructure {
		structure = control.DependencyStructure
	} else if !state.Valid {
		return WebRTCDependencyStructureState{}, WebRTCFrameDependencyStructure{}, ErrInvalidFrame
	}

	if err := validateWebRTCDependencyStructure(structure); err != nil {
		return WebRTCDependencyStructureState{}, WebRTCFrameDependencyStructure{}, err
	}
	for i := uint8(0); i < control.FrameNum; i++ {
		if _, err := webRTCDependencyDescriptorMatchFrame(structure, control.Frames[i].GenericFrameInfo); err != nil {
			return WebRTCDependencyStructureState{}, WebRTCFrameDependencyStructure{}, err
		}
	}

	next := WebRTCDependencyStructureState{
		Valid:     true,
		Structure: structure,
	}
	return next, structure, nil
}

func WebRTCTemplateIDForFrame(structure WebRTCFrameDependencyStructure, info WebRTCGenericFrameInfo) (uint8, error) {
	match, err := webRTCDependencyDescriptorMatchFrame(structure, info)
	if err != nil {
		return 0, err
	}
	return (structure.StructureID + match.templateIndex) % WebRTCRtpDependencyMaxTemplates, nil
}

func WebRTCFrameDependencyStructureForConfig(config Config) (WebRTCFrameDependencyStructure, error) {
	config, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return WebRTCFrameDependencyStructure{}, err
	}
	if config.SpatialLayerCount == 0 || config.TemporalLayerCount == 0 ||
		uint16(config.SpatialLayerCount)*uint16(config.TemporalLayerCount) > WebRTCRtpDependencyMaxTemplates ||
		uint16(config.SpatialLayerCount)*uint16(config.TemporalLayerCount) > WebRTCRtpDependencyMaxDecodeTargets {
		return WebRTCFrameDependencyStructure{}, ErrInvalidConfig
	}

	var structure WebRTCFrameDependencyStructure
	structure.NumDecodeTargets = config.SpatialLayerCount * config.TemporalLayerCount
	structure.NumChains = config.SpatialLayerCount
	structure.ResolutionNum = config.SpatialLayerCount
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		structure.Resolutions[i] = config.SpatialLayers[i].Resolution
	}
	for target := uint8(0); target < structure.NumDecodeTargets; target++ {
		spatial, _, ok := webRTCDecodeTargetLayer(target, config.SpatialLayerCount, config.TemporalLayerCount)
		if !ok {
			return WebRTCFrameDependencyStructure{}, ErrInvalidConfig
		}
		structure.DecodeTargetProtectedByChain[target] = spatial
	}

	if fillWebRTCReferenceDependencyTemplates(&structure, config.Scalability) {
		return structure, nil
	}

	for spatial := uint8(0); spatial < config.SpatialLayerCount; spatial++ {
		for temporal := uint8(0); temporal < config.TemporalLayerCount; temporal++ {
			template := &structure.Templates[structure.TemplateNum]
			template.SpatialID = spatial
			template.TemporalID = temporal
			template.DTINum = structure.NumDecodeTargets
			template.ChainDiffNum = config.SpatialLayerCount
			fillTemplateDTIs(template, spatial, temporal, config)
			structure.TemplateNum++
		}
	}
	return structure, nil
}

// Ported from libwebrtc:
// modules/video_coding/svc/scalability_structure_full_svc.cc
// modules/video_coding/svc/scalability_structure_key_svc.cc
// modules/video_coding/svc/scalability_structure_l2t2_key_shift.cc
// modules/video_coding/svc/scalability_structure_simulcast.cc
func fillWebRTCReferenceDependencyTemplates(structure *WebRTCFrameDependencyStructure, mode ScalabilityMode) bool {
	switch mode {
	case ScalabilityModeL1T2:
		fillWebRTCL1T2Templates(structure)
	case ScalabilityModeL1T3:
		fillWebRTCL1T3Templates(structure)
	case ScalabilityModeL2T1, ScalabilityModeL2T1h:
		fillWebRTCL2T1Templates(structure)
	case ScalabilityModeL2T2, ScalabilityModeL2T2h:
		fillWebRTCL2T2Templates(structure)
	case ScalabilityModeL2T3, ScalabilityModeL2T3h:
		fillWebRTCL2T3Templates(structure)
	case ScalabilityModeL3T1, ScalabilityModeL3T1h:
		fillWebRTCL3T1Templates(structure)
	case ScalabilityModeL3T2, ScalabilityModeL3T2h:
		fillWebRTCL3T2Templates(structure)
	case ScalabilityModeL3T3, ScalabilityModeL3T3h:
		fillWebRTCL3T3Templates(structure)
	case ScalabilityModeL2T1_KEY:
		fillWebRTCL2T1KeyTemplates(structure)
	case ScalabilityModeL2T2_KEY:
		fillWebRTCL2T2KeyTemplates(structure)
	case ScalabilityModeL2T2_KEY_SHIFT:
		fillWebRTCL2T2KeyShiftTemplates(structure)
	case ScalabilityModeL2T3_KEY:
		fillWebRTCL2T3KeyTemplates(structure)
	case ScalabilityModeL3T1_KEY:
		fillWebRTCL3T1KeyTemplates(structure)
	case ScalabilityModeL3T2_KEY:
		fillWebRTCL3T2KeyTemplates(structure)
	case ScalabilityModeL3T3_KEY:
		fillWebRTCL3T3KeyTemplates(structure)
	case ScalabilityModeS2T1, ScalabilityModeS2T1h:
		fillWebRTCS2T1Templates(structure)
	case ScalabilityModeS2T2, ScalabilityModeS2T2h:
		fillWebRTCS2T2Templates(structure)
	case ScalabilityModeS2T3, ScalabilityModeS2T3h:
		fillWebRTCS2T3Templates(structure)
	case ScalabilityModeS3T1, ScalabilityModeS3T1h:
		fillWebRTCS3T1Templates(structure)
	case ScalabilityModeS3T2, ScalabilityModeS3T2h:
		fillWebRTCS3T2Templates(structure)
	case ScalabilityModeS3T3, ScalabilityModeS3T3h:
		fillWebRTCS3T3Templates(structure)
	default:
		return false
	}
	return true
}

func applyWebRTCReferenceTemplateHints(config Config, control *WebRTCTemporalUnitControl, deltaPictureIndex uint64, hasDeltaPictureIndex bool) {
	if control == nil || !hasWebRTCReferenceDependencyTemplates(config.Scalability) {
		return
	}
	for i := uint8(0); i < control.FrameNum; i++ {
		hint, ok := webRTCReferenceTemplateHint(config, control.Frames[i].Settings, deltaPictureIndex, hasDeltaPictureIndex)
		if !ok {
			continue
		}
		info := &control.Frames[i].GenericFrameInfo
		info.TemplateIndexHint = hint
		info.TemplateIndexHintSet = true
	}
}

func hasWebRTCReferenceDependencyTemplates(mode ScalabilityMode) bool {
	switch mode {
	case ScalabilityModeL1T2, ScalabilityModeL1T3,
		ScalabilityModeL2T1, ScalabilityModeL2T1h,
		ScalabilityModeL2T2, ScalabilityModeL2T2h,
		ScalabilityModeL2T3, ScalabilityModeL2T3h,
		ScalabilityModeL3T1, ScalabilityModeL3T1h,
		ScalabilityModeL3T2, ScalabilityModeL3T2h,
		ScalabilityModeL3T3, ScalabilityModeL3T3h,
		ScalabilityModeL2T1_KEY, ScalabilityModeL2T2_KEY,
		ScalabilityModeL2T2_KEY_SHIFT, ScalabilityModeL2T3_KEY,
		ScalabilityModeL3T1_KEY, ScalabilityModeL3T2_KEY,
		ScalabilityModeL3T3_KEY,
		ScalabilityModeS2T1, ScalabilityModeS2T1h,
		ScalabilityModeS2T2, ScalabilityModeS2T2h,
		ScalabilityModeS2T3, ScalabilityModeS2T3h,
		ScalabilityModeS3T1, ScalabilityModeS3T1h,
		ScalabilityModeS3T2, ScalabilityModeS3T2h,
		ScalabilityModeS3T3, ScalabilityModeS3T3h:
		return true
	default:
		return false
	}
}

func webRTCReferenceTemplateHint(config Config, settings FrameEncodeSettings, deltaPictureIndex uint64, hasDeltaPictureIndex bool) (uint8, bool) {
	if settings.TemporalID != 2 || config.TemporalLayerCount != 3 || !hasDeltaPictureIndex || deltaPictureIndex == 0 {
		return 0, false
	}
	switch config.Scalability {
	case ScalabilityModeL1T3,
		ScalabilityModeL2T3, ScalabilityModeL2T3h, ScalabilityModeL2T3_KEY,
		ScalabilityModeL3T3, ScalabilityModeL3T3h, ScalabilityModeL3T3_KEY,
		ScalabilityModeS2T3, ScalabilityModeS2T3h,
		ScalabilityModeS3T3, ScalabilityModeS3T3h:
	default:
		return 0, false
	}
	phase := (deltaPictureIndex - 1) % 4
	switch phase {
	case 0:
		return settings.SpatialID*5 + 3, true
	case 2:
		return settings.SpatialID*5 + 4, true
	default:
		return 0, false
	}
}

func fillWebRTCL1T2Templates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 3
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "SS", nil, []uint8{0})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SS", []uint16{2}, []uint8{2})
	setWebRTCTemplate(&structure.Templates[2], 0, 1, "-D", []uint16{1}, []uint8{1})
}

func fillWebRTCL1T3Templates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 5
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "SSS", nil, []uint8{0})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SSS", []uint16{4}, []uint8{4})
	setWebRTCTemplate(&structure.Templates[2], 0, 1, "-DS", []uint16{2}, []uint8{2})
	setWebRTCTemplate(&structure.Templates[3], 0, 2, "--D", []uint16{1}, []uint8{1})
	setWebRTCTemplate(&structure.Templates[4], 0, 2, "--D", []uint16{1}, []uint8{3})
}

func fillWebRTCL2T1Templates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 4
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "SR", []uint16{2}, []uint8{2, 1})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SS", nil, []uint8{0, 0})
	setWebRTCTemplate(&structure.Templates[2], 1, 0, "-S", []uint16{2, 1}, []uint8{1, 1})
	setWebRTCTemplate(&structure.Templates[3], 1, 0, "-S", []uint16{1}, []uint8{1, 1})
}

func fillWebRTCL2T2Templates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 6
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "SSSS", nil, []uint8{0, 0})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SSRR", []uint16{4}, []uint8{4, 3})
	setWebRTCTemplate(&structure.Templates[2], 0, 1, "-D-R", []uint16{2}, []uint8{2, 1})
	setWebRTCTemplate(&structure.Templates[3], 1, 0, "--SS", []uint16{1}, []uint8{1, 1})
	setWebRTCTemplate(&structure.Templates[4], 1, 0, "--SS", []uint16{4, 1}, []uint8{1, 1})
	setWebRTCTemplate(&structure.Templates[5], 1, 1, "---D", []uint16{2, 1}, []uint8{3, 2})
}

func fillWebRTCL2T3Templates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 10
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "SSSRRR", []uint16{8}, []uint8{8, 7})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SSSSSS", nil, []uint8{0, 0})
	setWebRTCTemplate(&structure.Templates[2], 0, 1, "-DS-RR", []uint16{4}, []uint8{4, 3})
	setWebRTCTemplate(&structure.Templates[3], 0, 2, "--D--R", []uint16{2}, []uint8{2, 1})
	setWebRTCTemplate(&structure.Templates[4], 0, 2, "--D--R", []uint16{2}, []uint8{6, 5})
	setWebRTCTemplate(&structure.Templates[5], 1, 0, "---SSS", []uint16{8, 1}, []uint8{1, 1})
	setWebRTCTemplate(&structure.Templates[6], 1, 0, "---SSS", []uint16{1}, []uint8{1, 1})
	setWebRTCTemplate(&structure.Templates[7], 1, 1, "----DS", []uint16{4, 1}, []uint8{5, 4})
	setWebRTCTemplate(&structure.Templates[8], 1, 2, "-----D", []uint16{2, 1}, []uint8{3, 2})
	setWebRTCTemplate(&structure.Templates[9], 1, 2, "-----D", []uint16{2, 1}, []uint8{7, 6})
}

func fillWebRTCL3T1Templates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 6
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "SRR", []uint16{3}, []uint8{3, 2, 1})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SSS", nil, []uint8{0, 0, 0})
	setWebRTCTemplate(&structure.Templates[2], 1, 0, "-SR", []uint16{3, 1}, []uint8{1, 1, 1})
	setWebRTCTemplate(&structure.Templates[3], 1, 0, "-SS", []uint16{1}, []uint8{1, 1, 1})
	setWebRTCTemplate(&structure.Templates[4], 2, 0, "--S", []uint16{3, 1}, []uint8{2, 1, 1})
	setWebRTCTemplate(&structure.Templates[5], 2, 0, "--S", []uint16{1}, []uint8{2, 1, 1})
}

func fillWebRTCL3T2Templates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 9
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "SSRRRR", []uint16{6}, []uint8{6, 5, 4})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SSSSSS", nil, []uint8{0, 0, 0})
	setWebRTCTemplate(&structure.Templates[2], 0, 1, "-D-R-R", []uint16{3}, []uint8{3, 2, 1})
	setWebRTCTemplate(&structure.Templates[3], 1, 0, "--SSRR", []uint16{6, 1}, []uint8{1, 1, 1})
	setWebRTCTemplate(&structure.Templates[4], 1, 0, "--SSSS", []uint16{1}, []uint8{1, 1, 1})
	setWebRTCTemplate(&structure.Templates[5], 1, 1, "---D-R", []uint16{3, 1}, []uint8{4, 3, 2})
	setWebRTCTemplate(&structure.Templates[6], 2, 0, "----SS", []uint16{6, 1}, []uint8{2, 1, 1})
	setWebRTCTemplate(&structure.Templates[7], 2, 0, "----SS", []uint16{1}, []uint8{2, 1, 1})
	setWebRTCTemplate(&structure.Templates[8], 2, 1, "-----D", []uint16{3, 1}, []uint8{5, 4, 3})
}

func fillWebRTCL3T3Templates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 15
	setWebRTCTemplate(&structure.Templates[0x0], 0, 0, "SSSRRRRRR", []uint16{12}, []uint8{12, 11, 10})
	setWebRTCTemplate(&structure.Templates[0x1], 0, 0, "SSSSSSSSS", nil, []uint8{0, 0, 0})
	setWebRTCTemplate(&structure.Templates[0x2], 0, 1, "-DS-RR-RR", []uint16{6}, []uint8{6, 5, 4})
	setWebRTCTemplate(&structure.Templates[0x3], 0, 2, "--D--R--R", []uint16{3}, []uint8{3, 2, 1})
	setWebRTCTemplate(&structure.Templates[0x4], 0, 2, "--D--R--R", []uint16{3}, []uint8{9, 8, 7})
	setWebRTCTemplate(&structure.Templates[0x5], 1, 0, "---SSSRRR", []uint16{12, 1}, []uint8{1, 1, 1})
	setWebRTCTemplate(&structure.Templates[0x6], 1, 0, "---SSSSSS", []uint16{1}, []uint8{1, 1, 1})
	setWebRTCTemplate(&structure.Templates[0x7], 1, 1, "----DS-RR", []uint16{6, 1}, []uint8{7, 6, 5})
	setWebRTCTemplate(&structure.Templates[0x8], 1, 2, "-----D--R", []uint16{3, 1}, []uint8{4, 3, 2})
	setWebRTCTemplate(&structure.Templates[0x9], 1, 2, "-----D--R", []uint16{3, 1}, []uint8{10, 9, 8})
	setWebRTCTemplate(&structure.Templates[0xA], 2, 0, "------SSS", []uint16{12, 1}, []uint8{2, 1, 1})
	setWebRTCTemplate(&structure.Templates[0xB], 2, 0, "------SSS", []uint16{1}, []uint8{2, 1, 1})
	setWebRTCTemplate(&structure.Templates[0xC], 2, 1, "-------DS", []uint16{6, 1}, []uint8{8, 7, 6})
	setWebRTCTemplate(&structure.Templates[0xD], 2, 2, "--------D", []uint16{3, 1}, []uint8{5, 4, 3})
	setWebRTCTemplate(&structure.Templates[0xE], 2, 2, "--------D", []uint16{3, 1}, []uint8{11, 10, 9})
}

func fillWebRTCL2T1KeyTemplates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 4
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "S-", []uint16{2}, []uint8{2, 1})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SS", nil, []uint8{0, 0})
	setWebRTCTemplate(&structure.Templates[2], 1, 0, "-S", []uint16{2}, []uint8{1, 2})
	setWebRTCTemplate(&structure.Templates[3], 1, 0, "-S", []uint16{1}, []uint8{1, 1})
}

func fillWebRTCL2T2KeyTemplates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 6
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "SSSS", nil, []uint8{0, 0})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SS--", []uint16{4}, []uint8{4, 3})
	setWebRTCTemplate(&structure.Templates[2], 0, 1, "-D--", []uint16{2}, []uint8{2, 1})
	setWebRTCTemplate(&structure.Templates[3], 1, 0, "--SS", []uint16{1}, []uint8{1, 1})
	setWebRTCTemplate(&structure.Templates[4], 1, 0, "--SS", []uint16{4}, []uint8{1, 4})
	setWebRTCTemplate(&structure.Templates[5], 1, 1, "---D", []uint16{2}, []uint8{3, 2})
}

// Ported from libwebrtc:
// modules/video_coding/svc/scalability_structure_l2t2_key_shift.cc
func fillWebRTCL2T2KeyShiftTemplates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 7
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "SSSS", nil, []uint8{0, 0})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SS--", []uint16{2}, []uint8{2, 1})
	setWebRTCTemplate(&structure.Templates[2], 0, 0, "SS--", []uint16{4}, []uint8{4, 1})
	setWebRTCTemplate(&structure.Templates[3], 0, 1, "-D--", []uint16{2}, []uint8{2, 3})
	setWebRTCTemplate(&structure.Templates[4], 1, 0, "--SS", []uint16{1}, []uint8{1, 1})
	setWebRTCTemplate(&structure.Templates[5], 1, 0, "--SS", []uint16{4}, []uint8{3, 4})
	setWebRTCTemplate(&structure.Templates[6], 1, 1, "---D", []uint16{2}, []uint8{1, 2})
}

func fillWebRTCL2T3KeyTemplates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 10
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "SSSSSS", nil, []uint8{0, 0})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SSS---", []uint16{8}, []uint8{8, 7})
	setWebRTCTemplate(&structure.Templates[2], 0, 1, "-DS---", []uint16{4}, []uint8{4, 3})
	setWebRTCTemplate(&structure.Templates[3], 0, 2, "--D---", []uint16{2}, []uint8{2, 1})
	setWebRTCTemplate(&structure.Templates[4], 0, 2, "--D---", []uint16{2}, []uint8{6, 5})
	setWebRTCTemplate(&structure.Templates[5], 1, 0, "---SSS", []uint16{1}, []uint8{1, 1})
	setWebRTCTemplate(&structure.Templates[6], 1, 0, "---SSS", []uint16{8}, []uint8{1, 8})
	setWebRTCTemplate(&structure.Templates[7], 1, 1, "----DS", []uint16{4}, []uint8{5, 4})
	setWebRTCTemplate(&structure.Templates[8], 1, 2, "-----D", []uint16{2}, []uint8{3, 2})
	setWebRTCTemplate(&structure.Templates[9], 1, 2, "-----D", []uint16{2}, []uint8{7, 6})
}

func fillWebRTCL3T1KeyTemplates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 6
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "S--", []uint16{3}, []uint8{3, 2, 1})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SSS", nil, []uint8{0, 0, 0})
	setWebRTCTemplate(&structure.Templates[2], 1, 0, "-S-", []uint16{3}, []uint8{1, 3, 2})
	setWebRTCTemplate(&structure.Templates[3], 1, 0, "-SS", []uint16{1}, []uint8{1, 1, 1})
	setWebRTCTemplate(&structure.Templates[4], 2, 0, "--S", []uint16{3}, []uint8{2, 1, 3})
	setWebRTCTemplate(&structure.Templates[5], 2, 0, "--S", []uint16{1}, []uint8{2, 1, 1})
}

func fillWebRTCL3T2KeyTemplates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 9
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "SS----", []uint16{6}, []uint8{6, 5, 4})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SSSSSS", nil, []uint8{0, 0, 0})
	setWebRTCTemplate(&structure.Templates[2], 0, 1, "-D----", []uint16{3}, []uint8{3, 2, 1})
	setWebRTCTemplate(&structure.Templates[3], 1, 0, "--SS--", []uint16{6}, []uint8{1, 6, 5})
	setWebRTCTemplate(&structure.Templates[4], 1, 0, "--SSSS", []uint16{1}, []uint8{1, 1, 1})
	setWebRTCTemplate(&structure.Templates[5], 1, 1, "---D--", []uint16{3}, []uint8{4, 3, 2})
	setWebRTCTemplate(&structure.Templates[6], 2, 0, "----SS", []uint16{6}, []uint8{2, 1, 6})
	setWebRTCTemplate(&structure.Templates[7], 2, 0, "----SS", []uint16{1}, []uint8{2, 1, 1})
	setWebRTCTemplate(&structure.Templates[8], 2, 1, "-----D", []uint16{3}, []uint8{5, 4, 3})
}

func fillWebRTCL3T3KeyTemplates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 15
	setWebRTCTemplate(&structure.Templates[0x0], 0, 0, "SSSSSSSSS", nil, []uint8{0, 0, 0})
	setWebRTCTemplate(&structure.Templates[0x1], 0, 0, "SSS------", []uint16{12}, []uint8{12, 11, 10})
	setWebRTCTemplate(&structure.Templates[0x2], 0, 1, "-DS------", []uint16{6}, []uint8{6, 5, 4})
	setWebRTCTemplate(&structure.Templates[0x3], 0, 2, "--D------", []uint16{3}, []uint8{3, 2, 1})
	setWebRTCTemplate(&structure.Templates[0x4], 0, 2, "--D------", []uint16{3}, []uint8{9, 8, 7})
	setWebRTCTemplate(&structure.Templates[0x5], 1, 0, "---SSSSSS", []uint16{1}, []uint8{1, 1, 1})
	setWebRTCTemplate(&structure.Templates[0x6], 1, 0, "---SSS---", []uint16{12}, []uint8{1, 12, 11})
	setWebRTCTemplate(&structure.Templates[0x7], 1, 1, "----DS---", []uint16{6}, []uint8{7, 6, 5})
	setWebRTCTemplate(&structure.Templates[0x8], 1, 2, "-----D---", []uint16{3}, []uint8{4, 3, 2})
	setWebRTCTemplate(&structure.Templates[0x9], 1, 2, "-----D---", []uint16{3}, []uint8{10, 9, 8})
	setWebRTCTemplate(&structure.Templates[0xA], 2, 0, "------SSS", []uint16{1}, []uint8{2, 1, 1})
	setWebRTCTemplate(&structure.Templates[0xB], 2, 0, "------SSS", []uint16{12}, []uint8{2, 1, 12})
	setWebRTCTemplate(&structure.Templates[0xC], 2, 1, "-------DS", []uint16{6}, []uint8{8, 7, 6})
	setWebRTCTemplate(&structure.Templates[0xD], 2, 2, "--------D", []uint16{3}, []uint8{5, 4, 3})
	setWebRTCTemplate(&structure.Templates[0xE], 2, 2, "--------D", []uint16{3}, []uint8{11, 10, 9})
}

func fillWebRTCS2T1Templates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 4
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "S-", []uint16{2}, []uint8{2, 1})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "S-", nil, []uint8{0, 0})
	setWebRTCTemplate(&structure.Templates[2], 1, 0, "-S", []uint16{2}, []uint8{1, 2})
	setWebRTCTemplate(&structure.Templates[3], 1, 0, "-S", nil, []uint8{1, 0})
}

func fillWebRTCS2T2Templates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 6
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "SS--", []uint16{4}, []uint8{4, 3})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SS--", nil, []uint8{0, 0})
	setWebRTCTemplate(&structure.Templates[2], 0, 1, "-D--", []uint16{2}, []uint8{2, 1})
	setWebRTCTemplate(&structure.Templates[3], 1, 0, "--SS", []uint16{4}, []uint8{1, 4})
	setWebRTCTemplate(&structure.Templates[4], 1, 0, "--SS", nil, []uint8{1, 0})
	setWebRTCTemplate(&structure.Templates[5], 1, 1, "---D", []uint16{2}, []uint8{3, 2})
}

func fillWebRTCS2T3Templates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 10
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "SSS---", []uint16{8}, []uint8{8, 7})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SSS---", nil, []uint8{0, 0})
	setWebRTCTemplate(&structure.Templates[2], 0, 1, "-DS---", []uint16{4}, []uint8{4, 3})
	setWebRTCTemplate(&structure.Templates[3], 0, 2, "--D---", []uint16{2}, []uint8{2, 1})
	setWebRTCTemplate(&structure.Templates[4], 0, 2, "--D---", []uint16{2}, []uint8{6, 5})
	setWebRTCTemplate(&structure.Templates[5], 1, 0, "---SSS", []uint16{8}, []uint8{1, 8})
	setWebRTCTemplate(&structure.Templates[6], 1, 0, "---SSS", nil, []uint8{1, 0})
	setWebRTCTemplate(&structure.Templates[7], 1, 1, "----DS", []uint16{4}, []uint8{5, 4})
	setWebRTCTemplate(&structure.Templates[8], 1, 2, "-----D", []uint16{2}, []uint8{3, 2})
	setWebRTCTemplate(&structure.Templates[9], 1, 2, "-----D", []uint16{2}, []uint8{7, 6})
}

func fillWebRTCS3T1Templates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 6
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "S--", []uint16{3}, []uint8{3, 2, 1})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "S--", nil, []uint8{0, 0, 0})
	setWebRTCTemplate(&structure.Templates[2], 1, 0, "-S-", []uint16{3}, []uint8{1, 3, 2})
	setWebRTCTemplate(&structure.Templates[3], 1, 0, "-S-", nil, []uint8{1, 0, 0})
	setWebRTCTemplate(&structure.Templates[4], 2, 0, "--S", []uint16{3}, []uint8{2, 1, 3})
	setWebRTCTemplate(&structure.Templates[5], 2, 0, "--S", nil, []uint8{2, 1, 0})
}

func fillWebRTCS3T2Templates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 9
	setWebRTCTemplate(&structure.Templates[0], 0, 0, "SS----", []uint16{6}, []uint8{6, 5, 4})
	setWebRTCTemplate(&structure.Templates[1], 0, 0, "SS----", nil, []uint8{0, 0, 0})
	setWebRTCTemplate(&structure.Templates[2], 0, 1, "-D----", []uint16{3}, []uint8{3, 2, 1})
	setWebRTCTemplate(&structure.Templates[3], 1, 0, "--SS--", []uint16{6}, []uint8{1, 6, 5})
	setWebRTCTemplate(&structure.Templates[4], 1, 0, "--SS--", nil, []uint8{1, 0, 0})
	setWebRTCTemplate(&structure.Templates[5], 1, 1, "---D--", []uint16{3}, []uint8{4, 3, 2})
	setWebRTCTemplate(&structure.Templates[6], 2, 0, "----SS", []uint16{6}, []uint8{2, 1, 6})
	setWebRTCTemplate(&structure.Templates[7], 2, 0, "----SS", nil, []uint8{2, 1, 0})
	setWebRTCTemplate(&structure.Templates[8], 2, 1, "-----D", []uint16{3}, []uint8{5, 4, 3})
}

func fillWebRTCS3T3Templates(structure *WebRTCFrameDependencyStructure) {
	structure.TemplateNum = 15
	setWebRTCTemplate(&structure.Templates[0x0], 0, 0, "SSS------", []uint16{12}, []uint8{12, 11, 10})
	setWebRTCTemplate(&structure.Templates[0x1], 0, 0, "SSS------", nil, []uint8{0, 0, 0})
	setWebRTCTemplate(&structure.Templates[0x2], 0, 1, "-DS------", []uint16{6}, []uint8{6, 5, 4})
	setWebRTCTemplate(&structure.Templates[0x3], 0, 2, "--D------", []uint16{3}, []uint8{3, 2, 1})
	setWebRTCTemplate(&structure.Templates[0x4], 0, 2, "--D------", []uint16{3}, []uint8{9, 8, 7})
	setWebRTCTemplate(&structure.Templates[0x5], 1, 0, "---SSS---", []uint16{12}, []uint8{1, 12, 11})
	setWebRTCTemplate(&structure.Templates[0x6], 1, 0, "---SSS---", nil, []uint8{1, 0, 0})
	setWebRTCTemplate(&structure.Templates[0x7], 1, 1, "----DS---", []uint16{6}, []uint8{7, 6, 5})
	setWebRTCTemplate(&structure.Templates[0x8], 1, 2, "-----D---", []uint16{3}, []uint8{4, 3, 2})
	setWebRTCTemplate(&structure.Templates[0x9], 1, 2, "-----D---", []uint16{3}, []uint8{10, 9, 8})
	setWebRTCTemplate(&structure.Templates[0xA], 2, 0, "------SSS", []uint16{12}, []uint8{2, 1, 12})
	setWebRTCTemplate(&structure.Templates[0xB], 2, 0, "------SSS", nil, []uint8{2, 1, 0})
	setWebRTCTemplate(&structure.Templates[0xC], 2, 1, "-------DS", []uint16{6}, []uint8{8, 7, 6})
	setWebRTCTemplate(&structure.Templates[0xD], 2, 2, "--------D", []uint16{3}, []uint8{5, 4, 3})
	setWebRTCTemplate(&structure.Templates[0xE], 2, 2, "--------D", []uint16{3}, []uint8{11, 10, 9})
}

func setWebRTCTemplate(template *WebRTCFrameDependencyTemplate, spatialID uint8, temporalID uint8, dtis string, frameDiffs []uint16, chainDiffs []uint8) {
	template.SpatialID = spatialID
	template.TemporalID = temporalID
	template.DTINum = uint8(len(dtis))
	for i := 0; i < len(dtis) && i < len(template.DTIs); i++ {
		switch dtis[i] {
		case 'D':
			template.DTIs[i] = DecodeTargetDiscardable
		case 'S':
			template.DTIs[i] = DecodeTargetSwitch
		case 'R':
			template.DTIs[i] = DecodeTargetRequired
		default:
			template.DTIs[i] = DecodeTargetNotPresent
		}
	}
	template.FrameDiffNum = uint8(len(frameDiffs))
	for i := 0; i < len(frameDiffs) && i < len(template.FrameDiffs); i++ {
		template.FrameDiffs[i] = frameDiffs[i]
	}
	template.ChainDiffNum = uint8(len(chainDiffs))
	for i := 0; i < len(chainDiffs) && i < len(template.ChainDiffs); i++ {
		template.ChainDiffs[i] = chainDiffs[i]
	}
}

func webRTCGenericFrameInfoForFrameMode(settings FrameEncodeSettings, frameID uint64, state FrameIDBufferState, config Config, keyTemporalUnit bool) (WebRTCGenericFrameInfo, FrameIDBufferState, error) {
	if config.SpatialLayerCount == 0 || config.SpatialLayerCount > WebRTCMaxSpatialLayers ||
		config.TemporalLayerCount == 0 || config.TemporalLayerCount > WebRTCMaxTemporalLayers ||
		uint16(config.SpatialLayerCount)*uint16(config.TemporalLayerCount) > WebRTCRtpDependencyMaxDecodeTargets {
		return WebRTCGenericFrameInfo{}, FrameIDBufferState{}, ErrInvalidFrame
	}
	info, out, err := WebRTCGenericFrameInfoForFrame(settings, frameID, state, config.SpatialLayerCount, config.TemporalLayerCount)
	if err != nil {
		return WebRTCGenericFrameInfo{}, FrameIDBufferState{}, err
	}
	fillWebRTCDTIs(&info, settings, config.SpatialLayerCount, config.TemporalLayerCount, config.Scalability, keyTemporalUnit)
	return info, out, nil
}

func fillWebRTCDTIs(info *WebRTCGenericFrameInfo, settings FrameEncodeSettings, spatialLayers uint8, temporalLayers uint8, mode ScalabilityMode, keyTemporalUnit bool) {
	count := spatialLayers * temporalLayers
	for target := uint8(0); target < count; target++ {
		targetSpatial, targetTemporal, ok := webRTCDecodeTargetLayer(target, spatialLayers, temporalLayers)
		if !ok {
			info.DTIs[target] = DecodeTargetNotPresent
			continue
		}
		info.DTIs[target] = webRTCDTIForTarget(settings, targetSpatial, targetTemporal, mode, keyTemporalUnit)
	}
}

func fillTemplateDTIs(template *WebRTCFrameDependencyTemplate, spatialID uint8, temporalID uint8, config Config) {
	settings := FrameEncodeSettings{
		Type:       FrameTypeDelta,
		SpatialID:  spatialID,
		TemporalID: temporalID,
	}
	count := config.SpatialLayerCount * config.TemporalLayerCount
	for target := uint8(0); target < count; target++ {
		targetSpatial, targetTemporal, ok := webRTCDecodeTargetLayer(target, config.SpatialLayerCount, config.TemporalLayerCount)
		if !ok {
			template.DTIs[target] = DecodeTargetNotPresent
			continue
		}
		template.DTIs[target] = webRTCDTIForTarget(settings, targetSpatial, targetTemporal, config.Scalability, false)
	}
}

func webRTCDecodeTargetLayer(target uint8, spatialLayers uint8, temporalLayers uint8) (spatialID uint8, temporalID uint8, ok bool) {
	if spatialLayers == 0 || temporalLayers == 0 || target >= spatialLayers*temporalLayers {
		return 0, 0, false
	}
	return target / temporalLayers, target % temporalLayers, true
}

func webRTCDTIForTarget(settings FrameEncodeSettings, targetSpatialID uint8, targetTemporalID uint8, mode ScalabilityMode, keyTemporalUnit bool) DecodeTargetIndication {
	switch {
	case mode.IsSimulcast():
		return webRTCSimulcastDTI(settings, targetSpatialID, targetTemporalID)
	case mode.UsesKeyFrameInterLayerDependency():
		return webRTCKeySVCDTI(settings, targetSpatialID, targetTemporalID, keyTemporalUnit)
	default:
		return webRTCFullSVCDTI(settings, targetSpatialID, targetTemporalID, keyTemporalUnit)
	}
}

func webRTCFullSVCDTI(settings FrameEncodeSettings, targetSpatialID uint8, targetTemporalID uint8, keyTemporalUnit bool) DecodeTargetIndication {
	if targetSpatialID < settings.SpatialID || targetTemporalID < settings.TemporalID {
		return DecodeTargetNotPresent
	}
	if targetSpatialID == settings.SpatialID {
		if targetTemporalID == 0 {
			return DecodeTargetSwitch
		}
		if targetTemporalID == settings.TemporalID {
			return DecodeTargetDiscardable
		}
		return DecodeTargetSwitch
	}
	if keyTemporalUnit || settings.Type.resetsReferences() {
		return DecodeTargetSwitch
	}
	return DecodeTargetRequired
}

func webRTCKeySVCDTI(settings FrameEncodeSettings, targetSpatialID uint8, targetTemporalID uint8, keyTemporalUnit bool) DecodeTargetIndication {
	if keyTemporalUnit || settings.Type.resetsReferences() {
		if targetSpatialID < settings.SpatialID {
			return DecodeTargetNotPresent
		}
		return DecodeTargetSwitch
	}
	return webRTCSimulcastDTI(settings, targetSpatialID, targetTemporalID)
}

func webRTCSimulcastDTI(settings FrameEncodeSettings, targetSpatialID uint8, targetTemporalID uint8) DecodeTargetIndication {
	if targetSpatialID != settings.SpatialID || targetTemporalID < settings.TemporalID {
		return DecodeTargetNotPresent
	}
	if targetTemporalID == 0 {
		return DecodeTargetSwitch
	}
	if targetTemporalID == settings.TemporalID {
		return DecodeTargetDiscardable
	}
	return DecodeTargetSwitch
}
