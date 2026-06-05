package goav1

import (
	internalbitstream "github.com/thesyncim/goav1/internal/av1/bitstream"
	internalencoder "github.com/thesyncim/goav1/internal/av1/encoder"
)

const (
	EncoderMaxLayers                           = internalencoder.MaxLayers
	EncoderMaxTemporalLayers                   = internalencoder.MaxTemporalLayers
	EncoderMaxSpatialLayers                    = internalencoder.MaxSpatialLayers
	EncoderWebRTCMaxSpatialLayers              = internalencoder.WebRTCMaxSpatialLayers
	EncoderWebRTCMaxTemporalLayers             = internalencoder.WebRTCMaxTemporalLayers
	EncoderWebRTCReferenceBuffers              = internalencoder.WebRTCReferenceBuffers
	EncoderWebRTCMaxReferences                 = internalencoder.WebRTCMaxFrameReferences
	EncoderWebRTCRtpTicksPerSecond             = internalencoder.WebRTCRtpTicksPerSecond
	EncoderWebRTCMinEffortLevel                = internalencoder.WebRTCMinEffortLevel
	EncoderWebRTCMaxEffortLevel                = internalencoder.WebRTCMaxEffortLevel
	EncoderWebRTCMaxQuantizer                  = internalencoder.WebRTCMaxQuantizer
	EncoderWebRTCMaxDimension                  = internalencoder.WebRTCMaxDimension
	EncoderWebRTCMaxFramePixels                = internalencoder.WebRTCMaxFramePixels
	EncoderWebRTCMaxBitrateKbps                = internalencoder.WebRTCMaxBitrateKbps
	EncoderWebRTCRtpDependencyMaxDecodeTargets = internalencoder.WebRTCRtpDependencyMaxDecodeTargets
	EncoderWebRTCRtpDependencyMaxTemplates     = internalencoder.WebRTCRtpDependencyMaxTemplates
	EncoderLibaomSVCReferenceSlots             = internalencoder.LibaomSVCReferenceSlots
)

type EncoderProfile = internalencoder.Profile

const (
	EncoderProfile0 = internalencoder.Profile0
	EncoderProfile1 = internalencoder.Profile1
	EncoderProfile2 = internalencoder.Profile2
)

type EncoderScalabilityMode = internalencoder.ScalabilityMode

const (
	EncoderScalabilityModeL1T1           = internalencoder.ScalabilityModeL1T1
	EncoderScalabilityModeL1T2           = internalencoder.ScalabilityModeL1T2
	EncoderScalabilityModeL1T3           = internalencoder.ScalabilityModeL1T3
	EncoderScalabilityModeL2T1           = internalencoder.ScalabilityModeL2T1
	EncoderScalabilityModeL2T1h          = internalencoder.ScalabilityModeL2T1h
	EncoderScalabilityModeL2T1_KEY       = internalencoder.ScalabilityModeL2T1_KEY
	EncoderScalabilityModeL2T2           = internalencoder.ScalabilityModeL2T2
	EncoderScalabilityModeL2T2h          = internalencoder.ScalabilityModeL2T2h
	EncoderScalabilityModeL2T2_KEY       = internalencoder.ScalabilityModeL2T2_KEY
	EncoderScalabilityModeL2T2_KEY_SHIFT = internalencoder.ScalabilityModeL2T2_KEY_SHIFT
	EncoderScalabilityModeL2T3           = internalencoder.ScalabilityModeL2T3
	EncoderScalabilityModeL2T3h          = internalencoder.ScalabilityModeL2T3h
	EncoderScalabilityModeL2T3_KEY       = internalencoder.ScalabilityModeL2T3_KEY
	EncoderScalabilityModeL3T1           = internalencoder.ScalabilityModeL3T1
	EncoderScalabilityModeL3T1h          = internalencoder.ScalabilityModeL3T1h
	EncoderScalabilityModeL3T1_KEY       = internalencoder.ScalabilityModeL3T1_KEY
	EncoderScalabilityModeL3T2           = internalencoder.ScalabilityModeL3T2
	EncoderScalabilityModeL3T2h          = internalencoder.ScalabilityModeL3T2h
	EncoderScalabilityModeL3T2_KEY       = internalencoder.ScalabilityModeL3T2_KEY
	EncoderScalabilityModeL3T3           = internalencoder.ScalabilityModeL3T3
	EncoderScalabilityModeL3T3h          = internalencoder.ScalabilityModeL3T3h
	EncoderScalabilityModeL3T3_KEY       = internalencoder.ScalabilityModeL3T3_KEY
	EncoderScalabilityModeS2T1           = internalencoder.ScalabilityModeS2T1
	EncoderScalabilityModeS2T1h          = internalencoder.ScalabilityModeS2T1h
	EncoderScalabilityModeS2T2           = internalencoder.ScalabilityModeS2T2
	EncoderScalabilityModeS2T2h          = internalencoder.ScalabilityModeS2T2h
	EncoderScalabilityModeS2T3           = internalencoder.ScalabilityModeS2T3
	EncoderScalabilityModeS2T3h          = internalencoder.ScalabilityModeS2T3h
	EncoderScalabilityModeS3T1           = internalencoder.ScalabilityModeS3T1
	EncoderScalabilityModeS3T1h          = internalencoder.ScalabilityModeS3T1h
	EncoderScalabilityModeS3T2           = internalencoder.ScalabilityModeS3T2
	EncoderScalabilityModeS3T2h          = internalencoder.ScalabilityModeS3T2h
	EncoderScalabilityModeS3T3           = internalencoder.ScalabilityModeS3T3
	EncoderScalabilityModeS3T3h          = internalencoder.ScalabilityModeS3T3h
)

type EncoderRational = internalencoder.Rational
type EncoderResolution = internalencoder.Resolution
type EncoderRateControlMode = internalencoder.RateControlMode
type EncoderContentHint = internalencoder.ContentHint
type EncoderSpatialLayer = internalencoder.SpatialLayer
type EncoderConfig = internalencoder.Config
type EncoderFrameType = internalencoder.FrameType
type EncoderFrameEncodeSettings = internalencoder.FrameEncodeSettings
type EncoderReferenceBufferState = internalencoder.ReferenceBufferState
type EncoderLibaomSVCRefFrameConfig = internalencoder.LibaomSVCRefFrameConfig
type EncoderDecodeTargetIndication = internalencoder.DecodeTargetIndication
type EncoderFrameIDBufferState = internalencoder.FrameIDBufferState
type EncoderWebRTCGenericFrameInfo = internalencoder.WebRTCGenericFrameInfo
type EncoderWebRTCFrameDependencyTemplate = internalencoder.WebRTCFrameDependencyTemplate
type EncoderWebRTCFrameDependencyStructure = internalencoder.WebRTCFrameDependencyStructure
type EncoderWebRTCDependencyStructureState = internalencoder.WebRTCDependencyStructureState
type EncoderWebRTCFrameControl = internalencoder.WebRTCFrameControl
type EncoderWebRTCTemporalUnitControl = internalencoder.WebRTCTemporalUnitControl
type EncoderOBU = internalencoder.OBU
type EncoderSequenceHeader = internalencoder.SequenceHeader
type EncoderSequenceOperatingPoint = internalencoder.SequenceOperatingPoint
type EncoderSequenceColorConfig = internalencoder.SequenceColorConfig
type EncoderFrameHeaderType = internalencoder.FrameHeaderType
type EncoderFrameHeaderPrefix = internalencoder.FrameHeaderPrefix
type EncoderIntraFrameSize = internalencoder.IntraFrameSize
type EncoderIntraHeaderTemporalUnit = internalencoder.IntraHeaderTemporalUnit
type EncoderWebRTCKeyFrameTemporalUnit = internalencoder.WebRTCKeyFrameTemporalUnit
type EncoderWebRTCDeltaFrameTemporalUnit = internalencoder.WebRTCDeltaFrameTemporalUnit
type EncoderWebRTCPictureTemporalUnit = internalencoder.WebRTCPictureTemporalUnit
type EncoderWebRTCState = internalencoder.WebRTCEncoderState

type EncoderWebRTCPictureTemporalUnitRTPScratchSize struct {
	Packetizer         RTPPacketizerScratchSize
	MaxPayloadBytes    int
	MaxDescriptorBytes int
}

type EncoderWebRTCPictureTemporalUnitFramesRTPScratchSize struct {
	FrameOBUBytes      int
	FrameSpans         int
	PacketSpans        int
	Packetizer         RTPPacketizerScratchSize
	MaxPayloadBytes    int
	MaxDescriptorBytes int
}

type EncoderWebRTCPictureTemporalUnitFramesRTPScratch struct {
	FrameOBU    []byte
	FrameSpans  []EncoderWebRTCFrameRTPPacketSpan
	PacketSpans []EncoderWebRTCRTPPacketSpan
	OBUs        []RTPPacketizerOBU
	Packets     []RTPPacketPlan
	Work        []RTPPacketPlan
}

type EncoderWebRTCPictureTemporalUnitRTPPacketsSizeInfo struct {
	PacketCount     int
	PayloadBytes    int
	DescriptorBytes int
}

type EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo struct {
	FrameOBUBytes int
	RTP           EncoderWebRTCPictureTemporalUnitRTPPacketsSizeInfo
}

type EncoderWebRTCFrameOBUSpan struct {
	Offset int
	Length int
}

type EncoderWebRTCFrameRTPPacketSpan struct {
	FrameOBUOffset int
	FrameOBULength int
	PacketOffset   int
	PacketCount    int
}

type EncoderWebRTCRTPPacketSpan struct {
	PayloadOffset    int
	PayloadLength    int
	DescriptorOffset int
	DescriptorLength int
	Marker           bool
}

const (
	EncoderRateControlCBR = internalencoder.RateControlCBR
	EncoderRateControlCQP = internalencoder.RateControlCQP

	EncoderContentCamera = internalencoder.ContentCamera
	EncoderContentScreen = internalencoder.ContentScreen

	EncoderFrameTypeKey   = internalencoder.FrameTypeKey
	EncoderFrameTypeStart = internalencoder.FrameTypeStart
	EncoderFrameTypeDelta = internalencoder.FrameTypeDelta

	EncoderFrameHeaderTypeKey       = internalencoder.FrameHeaderTypeKey
	EncoderFrameHeaderTypeInter     = internalencoder.FrameHeaderTypeInter
	EncoderFrameHeaderTypeIntraOnly = internalencoder.FrameHeaderTypeIntraOnly
	EncoderFrameHeaderTypeSwitch    = internalencoder.FrameHeaderTypeSwitch
	EncoderPrimaryRefNone           = internalencoder.EncoderPrimaryRefNone

	EncoderSequenceColorPrimariesBT709         = internalencoder.SequenceColorPrimariesBT709
	EncoderSequenceTransferCharacteristicsSRGB = internalencoder.SequenceTransferCharacteristicsSRGB
	EncoderSequenceMatrixCoefficientsIdentity  = internalencoder.SequenceMatrixCoefficientsIdentity
	EncoderSequenceSelectScreenContentTools    = internalencoder.SequenceSelectScreenContentTools
	EncoderSequenceSelectIntegerMV             = internalencoder.SequenceSelectIntegerMV
	EncoderSequenceLevelMax                    = internalencoder.SequenceLevelMax

	EncoderDecodeTargetNotPresent  = internalencoder.DecodeTargetNotPresent
	EncoderDecodeTargetDiscardable = internalencoder.DecodeTargetDiscardable
	EncoderDecodeTargetSwitch      = internalencoder.DecodeTargetSwitch
	EncoderDecodeTargetRequired    = internalencoder.DecodeTargetRequired
)

var (
	ErrEncoderInvalidConfig = internalencoder.ErrInvalidConfig
	ErrEncoderInvalidFrame  = internalencoder.ErrInvalidFrame
	ErrEncoderUnsupported   = internalencoder.ErrUnsupported
	ErrEncoderShortBuffer   = internalbitstream.ErrShortBuffer
)

func ParseEncoderProfile(profile string) (EncoderProfile, bool) {
	return internalencoder.ParseProfile(profile)
}

func ParseEncoderScalabilityMode(mode string) (EncoderScalabilityMode, bool) {
	return internalencoder.ParseScalabilityMode(mode)
}

func DefaultEncoderScalabilityMode(temporalLayers uint8, spatialLayers uint8) (EncoderScalabilityMode, bool) {
	return internalencoder.DefaultScalabilityMode(temporalLayers, spatialLayers)
}

func LimitEncoderScalabilityModeSpatialLayers(mode EncoderScalabilityMode, limit uint8) (EncoderScalabilityMode, bool) {
	return internalencoder.LimitScalabilityModeSpatialLayers(mode, limit)
}

func EncoderLimitedSpatialLayers(resolution EncoderResolution) uint8 {
	return internalencoder.LimitedSpatialLayers(resolution)
}

func SetWebRTCEncoderSVCConfig(config EncoderConfig, requestedTemporalLayers uint8, requestedSpatialLayers uint8) (EncoderConfig, error) {
	return internalencoder.SetWebRTCSVCConfig(config, requestedTemporalLayers, requestedSpatialLayers)
}

func ValidateEncoderTemporalUnitFrames(frames []EncoderFrameEncodeSettings, state EncoderReferenceBufferState, rcMode EncoderRateControlMode) (EncoderReferenceBufferState, error) {
	return internalencoder.ValidateTemporalUnitFrames(frames, state, rcMode)
}

func EncoderLibaomSVCRefFrameConfigForFrame(settings EncoderFrameEncodeSettings) (EncoderLibaomSVCRefFrameConfig, error) {
	return internalencoder.LibaomSVCRefFrameConfigForFrame(settings)
}

func EncoderWebRTCGenericFrameInfoForFrame(settings EncoderFrameEncodeSettings, frameID uint64, state EncoderFrameIDBufferState, spatialLayers uint8, temporalLayers uint8) (EncoderWebRTCGenericFrameInfo, EncoderFrameIDBufferState, error) {
	return internalencoder.WebRTCGenericFrameInfoForFrame(settings, frameID, state, spatialLayers, temporalLayers)
}

func EncoderWebRTCFrameDependencyStructureForConfig(config EncoderConfig) (EncoderWebRTCFrameDependencyStructure, error) {
	return internalencoder.WebRTCFrameDependencyStructureForConfig(config)
}

func EncoderWebRTCTemplateIDForFrame(structure EncoderWebRTCFrameDependencyStructure, info EncoderWebRTCGenericFrameInfo) (uint8, error) {
	return internalencoder.WebRTCTemplateIDForFrame(structure, info)
}

func EncoderWebRTCRTPDependencyDescriptorMandatoryForFrame(structure EncoderWebRTCFrameDependencyStructure, info EncoderWebRTCGenericFrameInfo, firstPacketInFrame bool, lastPacketInFrame bool) (RTPDependencyDescriptorMandatory, error) {
	templateID, err := internalencoder.WebRTCTemplateIDForFrame(structure, info)
	if err != nil {
		return RTPDependencyDescriptorMandatory{}, err
	}
	return RTPDependencyDescriptorMandatory{
		FirstPacketInFrame: firstPacketInFrame,
		LastPacketInFrame:  lastPacketInFrame,
		TemplateID:         templateID,
		FrameNumber:        uint16(info.FrameID),
	}, nil
}

func EncoderWebRTCDependencyDescriptorSize(structure EncoderWebRTCFrameDependencyStructure, info EncoderWebRTCGenericFrameInfo, attachStructure bool) (int, error) {
	return internalencoder.WebRTCDependencyDescriptorSize(structure, info, attachStructure)
}

func AppendEncoderWebRTCDependencyDescriptor(dst []byte, structure EncoderWebRTCFrameDependencyStructure, info EncoderWebRTCGenericFrameInfo, firstPacketInFrame bool, lastPacketInFrame bool, attachStructure bool) ([]byte, error) {
	return internalencoder.AppendWebRTCDependencyDescriptor(dst, structure, info, firstPacketInFrame, lastPacketInFrame, attachStructure)
}

func EncoderWebRTCRTPPacketDependencyDescriptorSize(packetizer *RTPPacketizer, structure EncoderWebRTCFrameDependencyStructure, info EncoderWebRTCGenericFrameInfo, attachStructureOnFirstPacket bool) (int, bool, error) {
	flags, ok := NextRTPPacketDependencyDescriptorFlags(packetizer)
	if !ok {
		return 0, false, nil
	}
	size, err := internalencoder.WebRTCDependencyDescriptorSize(structure, info, attachStructureOnFirstPacket && flags.FirstPacketInFrame)
	if err != nil {
		return 0, true, err
	}
	return size, true, nil
}

func AppendEncoderWebRTCRTPPacketDependencyDescriptor(dst []byte, packetizer *RTPPacketizer, structure EncoderWebRTCFrameDependencyStructure, info EncoderWebRTCGenericFrameInfo, attachStructureOnFirstPacket bool) ([]byte, bool, error) {
	flags, ok := NextRTPPacketDependencyDescriptorFlags(packetizer)
	if !ok {
		return dst, false, nil
	}
	out, err := internalencoder.AppendWebRTCDependencyDescriptor(dst, structure, info, flags.FirstPacketInFrame, flags.LastPacketInFrame, attachStructureOnFirstPacket && flags.FirstPacketInFrame)
	if err != nil {
		return dst, true, err
	}
	return out, true, nil
}

func EncoderWebRTCFrameControlRTPPacketDependencyDescriptorSize(packetizer *RTPPacketizer, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure) (int, bool, error) {
	return EncoderWebRTCRTPPacketDependencyDescriptorSize(packetizer, structure, control.GenericFrameInfo, control.AttachDependencyStructure)
}

func AppendEncoderWebRTCFrameControlRTPPacketDependencyDescriptor(dst []byte, packetizer *RTPPacketizer, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure) ([]byte, bool, error) {
	return AppendEncoderWebRTCRTPPacketDependencyDescriptor(dst, packetizer, structure, control.GenericFrameInfo, control.AttachDependencyStructure)
}

func EncoderWebRTCFrameControlRTPPacketSize(packetizer *RTPPacketizer, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure) (payloadSize int, descriptorSize int, ok bool, err error) {
	payloadSize, ok = packetizer.NextPacketSize()
	if !ok {
		return 0, 0, false, nil
	}
	descriptorSize, ok, err = EncoderWebRTCFrameControlRTPPacketDependencyDescriptorSize(packetizer, control, structure)
	if err != nil {
		return 0, 0, true, err
	}
	if !ok {
		return 0, 0, false, nil
	}
	return payloadSize, descriptorSize, true, nil
}

func AppendEncoderWebRTCFrameControlRTPPacket(payloadDst []byte, descriptorDst []byte, packetizer *RTPPacketizer, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure) (payload []byte, descriptor []byte, marker bool, ok bool, err error) {
	descriptor, ok, err = AppendEncoderWebRTCFrameControlRTPPacketDependencyDescriptor(descriptorDst, packetizer, control, structure)
	if err != nil || !ok {
		return payloadDst, descriptorDst, false, ok, err
	}
	payloadSize, sizeOK := packetizer.NextPacketSize()
	if !sizeOK {
		return payloadDst, descriptorDst, false, false, nil
	}
	if cap(payloadDst)-len(payloadDst) < payloadSize {
		return payloadDst, descriptorDst, false, true, ErrRTPShortBuffer
	}
	off := len(payloadDst)
	out := payloadDst[:off+payloadSize]
	n, marker, ok, err := packetizer.NextPacket(out[off:])
	if err != nil {
		return payloadDst, descriptorDst, false, true, err
	}
	if !ok {
		return payloadDst, descriptorDst, false, false, nil
	}
	return out[:off+n], descriptor, marker, true, nil
}

func EncoderWebRTCPictureTemporalUnitFrameNum(unit EncoderWebRTCPictureTemporalUnit) uint8 {
	if unit.Key {
		return unit.KeyUnit.FrameNum
	}
	if unit.Delta {
		return unit.DeltaUnit.FrameNum
	}
	return 0
}

func EncoderWebRTCPictureTemporalUnitFrameControl(unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8) (EncoderWebRTCFrameControl, EncoderWebRTCFrameDependencyStructure, error) {
	if unit.Key == unit.Delta {
		return EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
	}
	if unit.Key {
		if frameIndex >= unit.KeyUnit.Control.FrameNum {
			return EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
		}
		return unit.KeyUnit.Control.Frames[frameIndex], unit.KeyUnit.Control.DependencyStructure, nil
	}
	if frameIndex >= unit.DeltaUnit.Control.FrameNum || !state.DependencyStructureState.Valid {
		return EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
	}
	return unit.DeltaUnit.Control.Frames[frameIndex], state.DependencyStructureState.Structure, nil
}

func EncoderWebRTCPictureTemporalUnitDependencyDescriptorSize(unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8, attachStructure bool) (int, error) {
	control, structure, err := EncoderWebRTCPictureTemporalUnitFrameControl(unit, state, frameIndex)
	if err != nil {
		return 0, err
	}
	return internalencoder.WebRTCDependencyDescriptorSize(structure, control.GenericFrameInfo, attachStructure)
}

func EncoderWebRTCPictureTemporalUnitMaxDependencyDescriptorSize(unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8) (int, error) {
	control, structure, err := EncoderWebRTCPictureTemporalUnitFrameControl(unit, state, frameIndex)
	if err != nil {
		return 0, err
	}
	return internalencoder.WebRTCDependencyDescriptorSize(structure, control.GenericFrameInfo, control.AttachDependencyStructure)
}

func AppendEncoderWebRTCPictureTemporalUnitDependencyDescriptor(dst []byte, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8, firstPacketInFrame bool, lastPacketInFrame bool, attachStructure bool) ([]byte, error) {
	control, structure, err := EncoderWebRTCPictureTemporalUnitFrameControl(unit, state, frameIndex)
	if err != nil {
		return dst, err
	}
	return internalencoder.AppendWebRTCDependencyDescriptor(dst, structure, control.GenericFrameInfo, firstPacketInFrame, lastPacketInFrame, attachStructure)
}

func EncoderWebRTCPictureTemporalUnitFrameOBUSize(payload []byte, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8) (int, EncoderWebRTCFrameControl, EncoderWebRTCFrameDependencyStructure, error) {
	control, structure, err := EncoderWebRTCPictureTemporalUnitFrameControl(unit, state, frameIndex)
	if err != nil {
		return 0, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	size, err := internalencoder.LowOverheadOBUSize(internalencoder.OBU{
		Type:       OBUFrame,
		TemporalID: control.Settings.TemporalID,
		SpatialID:  control.Settings.SpatialID,
		Payload:    payload,
	})
	if err != nil {
		return 0, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	return size, control, structure, nil
}

func AppendEncoderWebRTCPictureTemporalUnitFrameOBU(dst []byte, payload []byte, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8) ([]byte, EncoderWebRTCFrameControl, EncoderWebRTCFrameDependencyStructure, error) {
	_, control, structure, err := EncoderWebRTCPictureTemporalUnitFrameOBUSize(payload, unit, state, frameIndex)
	if err != nil {
		return dst, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	out, err := internalencoder.AppendLowOverheadOBU(dst, internalencoder.OBU{
		Type:       OBUFrame,
		TemporalID: control.Settings.TemporalID,
		SpatialID:  control.Settings.SpatialID,
		Payload:    payload,
	})
	if err != nil {
		return dst, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	return out, control, structure, nil
}

func EncoderWebRTCPictureTemporalUnitFramesOBUSize(framePayloads [][]byte, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState) (int, error) {
	frameNum := EncoderWebRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || len(framePayloads) != int(frameNum) {
		return 0, ErrEncoderInvalidFrame
	}
	size := 0
	for i := uint8(0); i < frameNum; i++ {
		frameSize, _, _, err := EncoderWebRTCPictureTemporalUnitFrameOBUSize(framePayloads[i], unit, state, i)
		if err != nil {
			return 0, err
		}
		size += frameSize
	}
	return size, nil
}

func AppendEncoderWebRTCPictureTemporalUnitFramesOBU(dst []byte, spans []EncoderWebRTCFrameOBUSpan, framePayloads [][]byte, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState) ([]byte, int, error) {
	frameNum := EncoderWebRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || len(framePayloads) != int(frameNum) {
		return dst, 0, ErrEncoderInvalidFrame
	}
	if len(spans) < int(frameNum) {
		return dst, 0, ErrEncoderShortBuffer
	}
	out := dst
	for i := uint8(0); i < frameNum; i++ {
		start := len(out)
		next, _, _, err := AppendEncoderWebRTCPictureTemporalUnitFrameOBU(out, framePayloads[i], unit, state, i)
		if err != nil {
			return dst, 0, err
		}
		out = next
		spans[i] = EncoderWebRTCFrameOBUSpan{
			Offset: start,
			Length: len(out) - start,
		}
	}
	return out, int(frameNum), nil
}

func EncoderWebRTCPictureTemporalUnitFramesRTPScratchLen(framePayloads [][]byte, limits RTPPayloadSizeLimits, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameOBUScratch []byte, obuScratch []RTPPacketizerOBU) (EncoderWebRTCPictureTemporalUnitFramesRTPScratchSize, error) {
	frameNum := EncoderWebRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || len(framePayloads) != int(frameNum) {
		return EncoderWebRTCPictureTemporalUnitFramesRTPScratchSize{}, ErrEncoderInvalidFrame
	}
	frameOBUBytes, err := EncoderWebRTCPictureTemporalUnitFramesOBUSize(framePayloads, unit, state)
	size := EncoderWebRTCPictureTemporalUnitFramesRTPScratchSize{
		FrameOBUBytes: frameOBUBytes,
		FrameSpans:    int(frameNum),
	}
	if err != nil {
		return size, err
	}
	if cap(frameOBUScratch)-len(frameOBUScratch) < frameOBUBytes {
		return size, ErrEncoderShortBuffer
	}

	frameOBUs := frameOBUScratch[:0]
	for i := uint8(0); i < frameNum; i++ {
		frameStart := len(frameOBUs)
		nextFrameOBUs, _, _, err := AppendEncoderWebRTCPictureTemporalUnitFrameOBU(frameOBUs, framePayloads[i], unit, state, i)
		if err != nil {
			return size, err
		}
		frameOBUs = nextFrameOBUs
		scratch, err := EncoderWebRTCPictureTemporalUnitRTPScratchLen(frameOBUs[frameStart:], limits, unit, state, i, obuScratch)
		if err != nil {
			return size, err
		}
		size.PacketSpans += scratch.Packetizer.Packets
		if scratch.Packetizer.OBUs > size.Packetizer.OBUs {
			size.Packetizer.OBUs = scratch.Packetizer.OBUs
		}
		if scratch.Packetizer.Packets > size.Packetizer.Packets {
			size.Packetizer.Packets = scratch.Packetizer.Packets
		}
		if scratch.Packetizer.Work > size.Packetizer.Work {
			size.Packetizer.Work = scratch.Packetizer.Work
		}
		if scratch.MaxPayloadBytes > size.MaxPayloadBytes {
			size.MaxPayloadBytes = scratch.MaxPayloadBytes
		}
		if scratch.MaxDescriptorBytes > size.MaxDescriptorBytes {
			size.MaxDescriptorBytes = scratch.MaxDescriptorBytes
		}
	}
	return size, nil
}

func BindEncoderWebRTCPictureTemporalUnitFramesRTPScratch(size EncoderWebRTCPictureTemporalUnitFramesRTPScratchSize, scratch EncoderWebRTCPictureTemporalUnitFramesRTPScratch) (EncoderWebRTCPictureTemporalUnitFramesRTPScratch, error) {
	if size.FrameOBUBytes < 0 || size.FrameSpans < 0 || size.PacketSpans < 0 ||
		size.Packetizer.OBUs < 0 || size.Packetizer.Packets < 0 || size.Packetizer.Work < 0 {
		return EncoderWebRTCPictureTemporalUnitFramesRTPScratch{}, ErrEncoderShortBuffer
	}
	if cap(scratch.FrameOBU) < size.FrameOBUBytes || len(scratch.FrameSpans) < size.FrameSpans ||
		len(scratch.PacketSpans) < size.PacketSpans || len(scratch.OBUs) < size.Packetizer.OBUs ||
		len(scratch.Packets) < size.Packetizer.Packets || len(scratch.Work) < size.Packetizer.Work {
		return EncoderWebRTCPictureTemporalUnitFramesRTPScratch{}, ErrEncoderShortBuffer
	}
	scratch.FrameOBU = scratch.FrameOBU[:0]
	scratch.FrameSpans = scratch.FrameSpans[:size.FrameSpans]
	scratch.PacketSpans = scratch.PacketSpans[:size.PacketSpans]
	scratch.OBUs = scratch.OBUs[:size.Packetizer.OBUs]
	scratch.Packets = scratch.Packets[:size.Packetizer.Packets]
	scratch.Work = scratch.Work[:size.Packetizer.Work]
	return scratch, nil
}

func EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSize(framePayload []byte, limits RTPPayloadSizeLimits, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8, frameOBUScratch []byte, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo, EncoderWebRTCFrameControl, EncoderWebRTCFrameDependencyStructure, error) {
	frameOBUSize, control, structure, err := EncoderWebRTCPictureTemporalUnitFrameOBUSize(framePayload, unit, state, frameIndex)
	if err != nil {
		return EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	frameOBU, _, _, err := AppendEncoderWebRTCPictureTemporalUnitFrameOBU(frameOBUScratch[:0], framePayload, unit, state, frameIndex)
	if err != nil {
		return EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	rtpSize, rtpControl, rtpStructure, err := EncoderWebRTCPictureTemporalUnitRTPPacketsSize(frameOBU, limits, unit, state, frameIndex, obuScratch, packetScratch, workScratch)
	if err != nil {
		return EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	if rtpControl != control || rtpStructure != structure {
		return EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
	}
	return EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{
		FrameOBUBytes: frameOBUSize,
		RTP:           rtpSize,
	}, control, structure, nil
}

func AppendEncoderWebRTCPictureTemporalUnitFrameRTPPackets(payloadDst []byte, descriptorDst []byte, spans []EncoderWebRTCRTPPacketSpan, frameOBUScratch []byte, framePayload []byte, limits RTPPayloadSizeLimits, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (frameOBU []byte, rtpPayloads []byte, descriptors []byte, packetCount int, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure, err error) {
	frameOBU, control, structure, err = AppendEncoderWebRTCPictureTemporalUnitFrameOBU(frameOBUScratch[:0], framePayload, unit, state, frameIndex)
	if err != nil {
		return frameOBUScratch[:0], payloadDst, descriptorDst, 0, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	rtpPayloads, descriptors, packetCount, rtpControl, rtpStructure, err := AppendEncoderWebRTCPictureTemporalUnitRTPPackets(payloadDst, descriptorDst, spans, frameOBU, limits, unit, state, frameIndex, obuScratch, packetScratch, workScratch)
	if err != nil {
		return frameOBU, payloadDst, descriptorDst, 0, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	if rtpControl != control || rtpStructure != structure {
		return frameOBU, payloadDst, descriptorDst, 0, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
	}
	return frameOBU, rtpPayloads, descriptors, packetCount, control, structure, nil
}

func EncoderWebRTCPictureTemporalUnitFramesRTPPacketsSize(framePayloads [][]byte, limits RTPPayloadSizeLimits, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameOBUScratch []byte, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo, error) {
	frameNum := EncoderWebRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || len(framePayloads) != int(frameNum) {
		return EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, ErrEncoderInvalidFrame
	}
	frameOBUBytes, err := EncoderWebRTCPictureTemporalUnitFramesOBUSize(framePayloads, unit, state)
	if err != nil {
		return EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, err
	}
	frameOBUs := frameOBUScratch[:0]
	size := EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{FrameOBUBytes: frameOBUBytes}
	for i := uint8(0); i < frameNum; i++ {
		frameStart := len(frameOBUs)
		nextFrameOBUs, _, _, err := AppendEncoderWebRTCPictureTemporalUnitFrameOBU(frameOBUs, framePayloads[i], unit, state, i)
		if err != nil {
			return EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, err
		}
		frameOBUs = nextFrameOBUs
		rtpSize, _, _, err := EncoderWebRTCPictureTemporalUnitRTPPacketsSize(frameOBUs[frameStart:], limits, unit, state, i, obuScratch, packetScratch, workScratch)
		if err != nil {
			return EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, err
		}
		size.RTP.PacketCount += rtpSize.PacketCount
		size.RTP.PayloadBytes += rtpSize.PayloadBytes
		size.RTP.DescriptorBytes += rtpSize.DescriptorBytes
	}
	return size, nil
}

func AppendEncoderWebRTCPictureTemporalUnitFramesRTPPackets(frameOBUDst []byte, payloadDst []byte, descriptorDst []byte, frameSpans []EncoderWebRTCFrameRTPPacketSpan, packetSpans []EncoderWebRTCRTPPacketSpan, framePayloads [][]byte, limits RTPPayloadSizeLimits, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (frameOBUs []byte, rtpPayloads []byte, descriptors []byte, frameCount int, packetCount int, err error) {
	frameNum := EncoderWebRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || len(framePayloads) != int(frameNum) {
		return frameOBUDst, payloadDst, descriptorDst, 0, 0, ErrEncoderInvalidFrame
	}
	if len(frameSpans) < int(frameNum) {
		return frameOBUDst, payloadDst, descriptorDst, 0, 0, ErrRTPPacketPlanTooSmall
	}

	frameOBUs = frameOBUDst
	rtpPayloads = payloadDst
	descriptors = descriptorDst
	for i := uint8(0); i < frameNum; i++ {
		frameStart := len(frameOBUs)
		packetStart := packetCount
		nextFrameOBUs, _, _, err := AppendEncoderWebRTCPictureTemporalUnitFrameOBU(frameOBUs, framePayloads[i], unit, state, i)
		if err != nil {
			return frameOBUDst, payloadDst, descriptorDst, 0, 0, err
		}
		frameOBUs = nextFrameOBUs
		nextPayloads, nextDescriptors, wrotePackets, _, _, err := AppendEncoderWebRTCPictureTemporalUnitRTPPackets(rtpPayloads, descriptors, packetSpans[packetCount:], frameOBUs[frameStart:], limits, unit, state, i, obuScratch, packetScratch, workScratch)
		if err != nil {
			return frameOBUDst, payloadDst, descriptorDst, 0, 0, err
		}
		frameSpans[i] = EncoderWebRTCFrameRTPPacketSpan{
			FrameOBUOffset: frameStart,
			FrameOBULength: len(frameOBUs) - frameStart,
			PacketOffset:   packetStart,
			PacketCount:    wrotePackets,
		}
		rtpPayloads = nextPayloads
		descriptors = nextDescriptors
		packetCount += wrotePackets
		frameCount++
	}
	return frameOBUs, rtpPayloads, descriptors, frameCount, packetCount, nil
}

func EncoderWebRTCPictureTemporalUnitRTPScratchLen(payload []byte, limits RTPPayloadSizeLimits, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8, obuScratch []RTPPacketizerOBU) (EncoderWebRTCPictureTemporalUnitRTPScratchSize, error) {
	packetizer, err := RTPPacketizerScratchLen(payload, limits, obuScratch)
	size := EncoderWebRTCPictureTemporalUnitRTPScratchSize{Packetizer: packetizer}
	if err != nil {
		return size, err
	}
	descriptor, err := EncoderWebRTCPictureTemporalUnitMaxDependencyDescriptorSize(unit, state, frameIndex)
	if err != nil {
		return size, err
	}
	size.MaxDescriptorBytes = descriptor
	if packetizer.OBUs != 0 {
		size.MaxPayloadBytes = limits.MaxPayloadLen
	}
	return size, nil
}

func NewEncoderWebRTCPictureTemporalUnitRTPPacketizer(payload []byte, limits RTPPayloadSizeLimits, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (RTPPacketizer, EncoderWebRTCFrameControl, EncoderWebRTCFrameDependencyStructure, error) {
	control, structure, err := EncoderWebRTCPictureTemporalUnitFrameControl(unit, state, frameIndex)
	if err != nil {
		return RTPPacketizer{}, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	packetizer, err := NewRTPPacketizer(
		payload,
		limits,
		control.Settings.Type == EncoderFrameTypeKey,
		frameIndex+1 == EncoderWebRTCPictureTemporalUnitFrameNum(unit),
		obuScratch,
		packetScratch,
		workScratch,
	)
	if err != nil {
		return RTPPacketizer{}, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	return packetizer, control, structure, nil
}

func EncoderWebRTCPictureTemporalUnitRTPPacketsSize(payload []byte, limits RTPPayloadSizeLimits, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (EncoderWebRTCPictureTemporalUnitRTPPacketsSizeInfo, EncoderWebRTCFrameControl, EncoderWebRTCFrameDependencyStructure, error) {
	packetizer, control, structure, err := NewEncoderWebRTCPictureTemporalUnitRTPPacketizer(payload, limits, unit, state, frameIndex, obuScratch, packetScratch, workScratch)
	if err != nil {
		return EncoderWebRTCPictureTemporalUnitRTPPacketsSizeInfo{}, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	size := EncoderWebRTCPictureTemporalUnitRTPPacketsSizeInfo{
		PacketCount:  packetizer.NumPackets(),
		PayloadBytes: RTPPacketizerRemainingPayloadSize(&packetizer),
	}
	if size.PacketCount == 0 {
		return size, control, structure, nil
	}
	firstDescriptor, err := internalencoder.WebRTCDependencyDescriptorSize(structure, control.GenericFrameInfo, control.AttachDependencyStructure)
	if err != nil {
		return EncoderWebRTCPictureTemporalUnitRTPPacketsSizeInfo{}, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	size.DescriptorBytes = firstDescriptor
	if size.PacketCount > 1 {
		descriptor, err := internalencoder.WebRTCDependencyDescriptorSize(structure, control.GenericFrameInfo, false)
		if err != nil {
			return EncoderWebRTCPictureTemporalUnitRTPPacketsSizeInfo{}, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
		}
		size.DescriptorBytes += descriptor * (size.PacketCount - 1)
	}
	return size, control, structure, nil
}

func AppendEncoderWebRTCPictureTemporalUnitFirstRTPPacket(payloadDst []byte, descriptorDst []byte, payload []byte, limits RTPPayloadSizeLimits, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (rtpPayload []byte, descriptor []byte, marker bool, ok bool, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure, err error) {
	packetizer, control, structure, err := NewEncoderWebRTCPictureTemporalUnitRTPPacketizer(payload, limits, unit, state, frameIndex, obuScratch, packetScratch, workScratch)
	if err != nil {
		return payloadDst, descriptorDst, false, false, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	rtpPayload, descriptor, marker, ok, err = AppendEncoderWebRTCFrameControlRTPPacket(payloadDst, descriptorDst, &packetizer, control, structure)
	if err != nil {
		return payloadDst, descriptorDst, false, ok, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	return rtpPayload, descriptor, marker, ok, control, structure, nil
}

func AppendEncoderWebRTCPictureTemporalUnitRTPPackets(payloadDst []byte, descriptorDst []byte, spans []EncoderWebRTCRTPPacketSpan, payload []byte, limits RTPPayloadSizeLimits, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (rtpPayloads []byte, descriptors []byte, packetCount int, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure, err error) {
	packetizer, control, structure, err := NewEncoderWebRTCPictureTemporalUnitRTPPacketizer(payload, limits, unit, state, frameIndex, obuScratch, packetScratch, workScratch)
	if err != nil {
		return payloadDst, descriptorDst, 0, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
	}
	rtpPayloads = payloadDst
	descriptors = descriptorDst
	for {
		if packetCount >= len(spans) {
			if packetizer.NumPackets() == 0 {
				return rtpPayloads, descriptors, packetCount, control, structure, nil
			}
			return payloadDst, descriptorDst, 0, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, ErrRTPPacketPlanTooSmall
		}
		payloadStart := len(rtpPayloads)
		descriptorStart := len(descriptors)
		nextPayloads, nextDescriptors, marker, ok, err := AppendEncoderWebRTCFrameControlRTPPacket(rtpPayloads, descriptors, &packetizer, control, structure)
		if err != nil {
			return payloadDst, descriptorDst, 0, EncoderWebRTCFrameControl{}, EncoderWebRTCFrameDependencyStructure{}, err
		}
		if !ok {
			return rtpPayloads, descriptors, packetCount, control, structure, nil
		}
		spans[packetCount] = EncoderWebRTCRTPPacketSpan{
			PayloadOffset:    payloadStart,
			PayloadLength:    len(nextPayloads) - payloadStart,
			DescriptorOffset: descriptorStart,
			DescriptorLength: len(nextDescriptors) - descriptorStart,
			Marker:           marker,
		}
		rtpPayloads = nextPayloads
		descriptors = nextDescriptors
		packetCount++
	}
}

func EncoderWebRTCPictureTemporalUnitRTPPacketSize(packetizer *RTPPacketizer, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8) (payloadSize int, descriptorSize int, ok bool, err error) {
	control, structure, err := EncoderWebRTCPictureTemporalUnitFrameControl(unit, state, frameIndex)
	if err != nil {
		return 0, 0, false, err
	}
	return EncoderWebRTCFrameControlRTPPacketSize(packetizer, control, structure)
}

func AppendEncoderWebRTCPictureTemporalUnitRTPPacket(payloadDst []byte, descriptorDst []byte, packetizer *RTPPacketizer, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState, frameIndex uint8) (payload []byte, descriptor []byte, marker bool, ok bool, err error) {
	control, structure, err := EncoderWebRTCPictureTemporalUnitFrameControl(unit, state, frameIndex)
	if err != nil {
		return payloadDst, descriptorDst, false, false, err
	}
	return AppendEncoderWebRTCFrameControlRTPPacket(payloadDst, descriptorDst, packetizer, control, structure)
}

func EncoderWebRTCTemporalUnitControlForFrames(config EncoderConfig, frames []EncoderFrameEncodeSettings, referenceState EncoderReferenceBufferState, frameIDState EncoderFrameIDBufferState, firstFrameID uint64) (EncoderWebRTCTemporalUnitControl, error) {
	return internalencoder.WebRTCTemporalUnitControlForFrames(config, frames, referenceState, frameIDState, firstFrameID)
}

func EncoderWebRTCDependencyStructureStateForTemporalUnit(control EncoderWebRTCTemporalUnitControl, state EncoderWebRTCDependencyStructureState) (EncoderWebRTCDependencyStructureState, EncoderWebRTCFrameDependencyStructure, error) {
	return internalencoder.WebRTCDependencyStructureStateForTemporalUnit(control, state)
}

func EncoderSupportedResolutionScaling(from EncoderResolution, to EncoderResolution) (EncoderRational, bool) {
	return internalencoder.SupportedResolutionScaling(from, to)
}

func EncoderSequenceHeaderForConfig(config EncoderConfig) (EncoderSequenceHeader, error) {
	return internalencoder.SequenceHeaderForConfig(config)
}

func EncoderIntraHeaderTemporalUnitForConfig(config EncoderConfig, orderHint uint8) (EncoderIntraHeaderTemporalUnit, error) {
	return internalencoder.IntraHeaderTemporalUnitForConfig(config, orderHint)
}

func EncoderWebRTCKeyFrameTemporalUnitForConfig(config EncoderConfig, orderHint uint8, firstFrameID uint64) (EncoderWebRTCKeyFrameTemporalUnit, error) {
	return internalencoder.WebRTCKeyFrameTemporalUnitForConfig(config, orderHint, firstFrameID)
}

func EncoderWebRTCDeltaFrameTemporalUnitForConfig(config EncoderConfig, referenceState EncoderReferenceBufferState, frameIDState EncoderFrameIDBufferState, temporalID uint8, firstFrameID uint64) (EncoderWebRTCDeltaFrameTemporalUnit, error) {
	return internalencoder.WebRTCDeltaFrameTemporalUnitForConfig(config, referenceState, frameIDState, temporalID, firstFrameID)
}

func EncoderWebRTCKeyFrameTemporalUnitForState(config EncoderConfig, state EncoderWebRTCState) (EncoderWebRTCKeyFrameTemporalUnit, EncoderWebRTCState, error) {
	return internalencoder.WebRTCKeyFrameTemporalUnitForState(config, state)
}

func EncoderWebRTCDeltaFrameTemporalUnitForState(config EncoderConfig, state EncoderWebRTCState) (EncoderWebRTCDeltaFrameTemporalUnit, EncoderWebRTCState, error) {
	return internalencoder.WebRTCDeltaFrameTemporalUnitForState(config, state)
}

func EncoderWebRTCNextTemporalUnitForState(config EncoderConfig, state EncoderWebRTCState, forceKeyFrame bool) (EncoderWebRTCPictureTemporalUnit, EncoderWebRTCState, error) {
	return internalencoder.WebRTCNextTemporalUnitForState(config, state, forceKeyFrame)
}

func EncoderWebRTCTemporalIDForDeltaPicture(temporalLayers uint8, deltaPictureIndex uint64) (uint8, error) {
	return internalencoder.WebRTCTemporalIDForDeltaPicture(temporalLayers, deltaPictureIndex)
}

func EncoderLowOverheadOBUSize(unit EncoderOBU) (int, error) {
	return internalencoder.LowOverheadOBUSize(unit)
}

func AppendEncoderLowOverheadOBU(dst []byte, unit EncoderOBU) ([]byte, error) {
	return internalencoder.AppendLowOverheadOBU(dst, unit)
}

func EncoderLowOverheadTemporalUnitSize(obus []EncoderOBU) (int, error) {
	return internalencoder.LowOverheadTemporalUnitSize(obus)
}

func AppendEncoderLowOverheadTemporalUnit(dst []byte, obus []EncoderOBU) ([]byte, error) {
	return internalencoder.AppendLowOverheadTemporalUnit(dst, obus)
}

func EncoderSequenceHeaderPayloadSize(seq EncoderSequenceHeader) (int, error) {
	return internalencoder.SequenceHeaderPayloadSize(seq)
}

func AppendEncoderSequenceHeaderPayload(dst []byte, seq EncoderSequenceHeader) ([]byte, error) {
	return internalencoder.AppendSequenceHeaderPayload(dst, seq)
}

func EncoderLowOverheadSequenceHeaderOBUSize(seq EncoderSequenceHeader) (int, error) {
	return internalencoder.LowOverheadSequenceHeaderOBUSize(seq)
}

func AppendEncoderLowOverheadSequenceHeaderOBU(dst []byte, seq EncoderSequenceHeader) ([]byte, error) {
	return internalencoder.AppendLowOverheadSequenceHeaderOBU(dst, seq)
}

func EncoderFrameHeaderPrefixPayloadSize(seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix) (int, error) {
	return internalencoder.FrameHeaderPrefixPayloadSize(seq, prefix)
}

func AppendEncoderFrameHeaderPrefixPayload(dst []byte, seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix) ([]byte, error) {
	return internalencoder.AppendFrameHeaderPrefixPayload(dst, seq, prefix)
}

func EncoderFrameHeaderIntraPayloadSize(seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderIntraFrameSize) (int, error) {
	return internalencoder.FrameHeaderIntraPayloadSize(seq, prefix, size)
}

func AppendEncoderFrameHeaderIntraPayload(dst []byte, seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderIntraFrameSize) ([]byte, error) {
	return internalencoder.AppendFrameHeaderIntraPayload(dst, seq, prefix, size)
}

func EncoderLowOverheadFrameHeaderIntraOBUSize(seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderIntraFrameSize) (int, error) {
	return internalencoder.LowOverheadFrameHeaderIntraOBUSize(seq, prefix, size)
}

func AppendEncoderLowOverheadFrameHeaderIntraOBU(dst []byte, seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderIntraFrameSize) ([]byte, error) {
	return internalencoder.AppendLowOverheadFrameHeaderIntraOBU(dst, seq, prefix, size)
}

func EncoderLowOverheadIntraHeaderTemporalUnitSize(seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderIntraFrameSize) (int, error) {
	return internalencoder.LowOverheadIntraHeaderTemporalUnitSize(seq, prefix, size)
}

func AppendEncoderLowOverheadIntraHeaderTemporalUnit(dst []byte, seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderIntraFrameSize) ([]byte, error) {
	return internalencoder.AppendLowOverheadIntraHeaderTemporalUnit(dst, seq, prefix, size)
}

func EncoderLowOverheadIntraHeaderTemporalUnitForConfigSize(config EncoderConfig, orderHint uint8) (int, EncoderIntraHeaderTemporalUnit, error) {
	return internalencoder.LowOverheadIntraHeaderTemporalUnitForConfigSize(config, orderHint)
}

func AppendEncoderLowOverheadIntraHeaderTemporalUnitForConfig(dst []byte, config EncoderConfig, orderHint uint8) ([]byte, EncoderIntraHeaderTemporalUnit, error) {
	return internalencoder.AppendLowOverheadIntraHeaderTemporalUnitForConfig(dst, config, orderHint)
}

func EncoderLowOverheadWebRTCKeyFrameTemporalUnitForConfigSize(config EncoderConfig, orderHint uint8, firstFrameID uint64) (int, EncoderWebRTCKeyFrameTemporalUnit, error) {
	return internalencoder.LowOverheadWebRTCKeyFrameTemporalUnitForConfigSize(config, orderHint, firstFrameID)
}

func AppendEncoderLowOverheadWebRTCKeyFrameTemporalUnitForConfig(dst []byte, config EncoderConfig, orderHint uint8, firstFrameID uint64) ([]byte, EncoderWebRTCKeyFrameTemporalUnit, error) {
	return internalencoder.AppendLowOverheadWebRTCKeyFrameTemporalUnitForConfig(dst, config, orderHint, firstFrameID)
}

func EncoderLowOverheadWebRTCKeyFrameTemporalUnitForStateSize(config EncoderConfig, state EncoderWebRTCState) (int, EncoderWebRTCKeyFrameTemporalUnit, EncoderWebRTCState, error) {
	return internalencoder.LowOverheadWebRTCKeyFrameTemporalUnitForStateSize(config, state)
}

func AppendEncoderLowOverheadWebRTCKeyFrameTemporalUnitForState(dst []byte, config EncoderConfig, state EncoderWebRTCState) ([]byte, EncoderWebRTCKeyFrameTemporalUnit, EncoderWebRTCState, error) {
	return internalencoder.AppendLowOverheadWebRTCKeyFrameTemporalUnitForState(dst, config, state)
}
