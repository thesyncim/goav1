package goav1

import internalencoder "github.com/thesyncim/goav1/internal/av1/encoder"

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
type EncoderWebRTCFrameControl = internalencoder.WebRTCFrameControl
type EncoderWebRTCTemporalUnitControl = internalencoder.WebRTCTemporalUnitControl
type EncoderOBU = internalencoder.OBU
type EncoderSequenceHeader = internalencoder.SequenceHeader
type EncoderSequenceOperatingPoint = internalencoder.SequenceOperatingPoint
type EncoderSequenceColorConfig = internalencoder.SequenceColorConfig
type EncoderFrameHeaderType = internalencoder.FrameHeaderType
type EncoderFrameHeaderPrefix = internalencoder.FrameHeaderPrefix
type EncoderIntraFrameSize = internalencoder.IntraFrameSize

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

func EncoderWebRTCTemporalUnitControlForFrames(config EncoderConfig, frames []EncoderFrameEncodeSettings, referenceState EncoderReferenceBufferState, frameIDState EncoderFrameIDBufferState, firstFrameID uint64) (EncoderWebRTCTemporalUnitControl, error) {
	return internalencoder.WebRTCTemporalUnitControlForFrames(config, frames, referenceState, frameIDState, firstFrameID)
}

func EncoderSupportedResolutionScaling(from EncoderResolution, to EncoderResolution) (EncoderRational, bool) {
	return internalencoder.SupportedResolutionScaling(from, to)
}

func EncoderSequenceHeaderForConfig(config EncoderConfig) (EncoderSequenceHeader, error) {
	return internalencoder.SequenceHeaderForConfig(config)
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
