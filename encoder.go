package goav1

import internalencoder "github.com/thesyncim/goav1/internal/av1/encoder"

const (
	EncoderMaxLayers               = internalencoder.MaxLayers
	EncoderMaxTemporalLayers       = internalencoder.MaxTemporalLayers
	EncoderMaxSpatialLayers        = internalencoder.MaxSpatialLayers
	EncoderWebRTCMaxSpatialLayers  = internalencoder.WebRTCMaxSpatialLayers
	EncoderWebRTCMaxTemporalLayers = internalencoder.WebRTCMaxTemporalLayers
	EncoderWebRTCReferenceBuffers  = internalencoder.WebRTCReferenceBuffers
	EncoderWebRTCMaxReferences     = internalencoder.WebRTCMaxFrameReferences
	EncoderWebRTCRtpTicksPerSecond = internalencoder.WebRTCRtpTicksPerSecond
	EncoderWebRTCMinEffortLevel    = internalencoder.WebRTCMinEffortLevel
	EncoderWebRTCMaxEffortLevel    = internalencoder.WebRTCMaxEffortLevel
	EncoderWebRTCMaxQuantizer      = internalencoder.WebRTCMaxQuantizer
	EncoderWebRTCMaxDimension      = internalencoder.WebRTCMaxDimension
	EncoderWebRTCMaxFramePixels    = internalencoder.WebRTCMaxFramePixels
	EncoderWebRTCMaxBitrateKbps    = internalencoder.WebRTCMaxBitrateKbps
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
