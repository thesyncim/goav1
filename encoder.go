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
	EncoderScalabilityModeL2T3_KEY_SHIFT = internalencoder.ScalabilityModeL2T3_KEY_SHIFT
	EncoderScalabilityModeL3T1           = internalencoder.ScalabilityModeL3T1
	EncoderScalabilityModeL3T1h          = internalencoder.ScalabilityModeL3T1h
	EncoderScalabilityModeL3T1_KEY       = internalencoder.ScalabilityModeL3T1_KEY
	EncoderScalabilityModeL3T2           = internalencoder.ScalabilityModeL3T2
	EncoderScalabilityModeL3T2h          = internalencoder.ScalabilityModeL3T2h
	EncoderScalabilityModeL3T2_KEY       = internalencoder.ScalabilityModeL3T2_KEY
	EncoderScalabilityModeL3T2_KEY_SHIFT = internalencoder.ScalabilityModeL3T2_KEY_SHIFT
	EncoderScalabilityModeL3T3           = internalencoder.ScalabilityModeL3T3
	EncoderScalabilityModeL3T3h          = internalencoder.ScalabilityModeL3T3h
	EncoderScalabilityModeL3T3_KEY       = internalencoder.ScalabilityModeL3T3_KEY
	EncoderScalabilityModeL3T3_KEY_SHIFT = internalencoder.ScalabilityModeL3T3_KEY_SHIFT
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
type EncoderWebRTCDependencyDescriptorOptions = internalencoder.WebRTCDependencyDescriptorOptions
type EncoderWebRTCDependencyStructureState = internalencoder.WebRTCDependencyStructureState
type EncoderWebRTCFrameControl = internalencoder.WebRTCFrameControl
type EncoderWebRTCTemporalUnitControl = internalencoder.WebRTCTemporalUnitControl
type EncoderOBU = internalencoder.OBU
type EncoderSequenceHeader = internalencoder.SequenceHeader
type EncoderSequenceOperatingPoint = internalencoder.SequenceOperatingPoint
type EncoderSequenceColorConfig = internalencoder.SequenceColorConfig
type EncoderFrameHeaderType = internalencoder.FrameHeaderType
type EncoderFrameHeaderPrefix = internalencoder.FrameHeaderPrefix
type EncoderIntraFrameHeaderParams = internalencoder.IntraFrameHeaderParams
type EncoderInterFrameHeaderParams = internalencoder.InterFrameHeaderParams
type EncoderIntraFrameSize = internalencoder.IntraFrameSize
type EncoderInterFrameSize = internalencoder.InterFrameSize
type EncoderQuantizationParams = internalencoder.QuantizationParams
type EncoderSegmentData = internalencoder.SegmentData
type EncoderSegmentationData = internalencoder.SegmentationData
type EncoderSegmentationParams = internalencoder.SegmentationParams
type EncoderDeltaParams = internalencoder.DeltaParams
type EncoderLoopFilterDeltas = internalencoder.LoopFilterDeltas
type EncoderLoopFilterParams = internalencoder.LoopFilterParams
type EncoderCDEFParams = internalencoder.CDEFParams
type EncoderRestorationType = internalencoder.RestorationType
type EncoderRestorationParams = internalencoder.RestorationParams
type EncoderInterpolationFilter = internalencoder.InterpolationFilter
type EncoderTileInfo = internalencoder.TileInfo
type EncoderTilePayload = internalencoder.TilePayload
type EncoderTransformMode = internalencoder.TransformMode
type EncoderReferenceMode = internalencoder.ReferenceMode
type EncoderTransformReferenceParams = internalencoder.TransformReferenceParams
type EncoderSkipModeParams = internalencoder.SkipModeParams
type EncoderFrameModeParams = internalencoder.FrameModeParams
type EncoderGlobalMotionType = internalencoder.GlobalMotionType
type EncoderWarpedMotionParams = internalencoder.WarpedMotionParams
type EncoderGlobalMotionParams = internalencoder.GlobalMotionParams
type EncoderFilmGrainParams = internalencoder.FilmGrainParams
type EncoderIntraHeaderTemporalUnit = internalencoder.IntraHeaderTemporalUnit
type EncoderInterHeaderFrame = internalencoder.InterHeaderFrame
type EncoderWebRTCKeyFrameTemporalUnit = internalencoder.WebRTCKeyFrameTemporalUnit
type EncoderWebRTCDeltaFrameTemporalUnit = internalencoder.WebRTCDeltaFrameTemporalUnit
type EncoderWebRTCPictureTemporalUnit = internalencoder.WebRTCPictureTemporalUnit
type EncoderWebRTCState = internalencoder.WebRTCEncoderState

type WebRTCEncoder struct {
	config EncoderConfig
	state  EncoderWebRTCState
}

type EncoderWebRTCRTPPacketDependencyDescriptorOptions struct {
	AttachStructureOnFirstPacket            bool
	ActiveDecodeTargetsPresentOnFirstPacket bool
	ActiveDecodeTargetsMask                 uint32
}

// EncoderWebRTCVideoLayersAllocationConfig controls how encoder configuration
// is converted into WebRTC's video-layers-allocation RTP header extension.
//
// The zero value describes one RTP stream, all configured spatial layers active,
// and resolution/framerate omitted. For multi-temporal modes, callers must set
// TargetBitratesKbps[spatial][temporal] to the cumulative bitrate ladder in
// kbps; goav1 does not invent a temporal bitrate split from a spatial target.
type EncoderWebRTCVideoLayersAllocationConfig struct {
	RTPStreamID    int
	RTPStreamCount int

	// SpatialLayerRTPStreamIDs maps each configured spatial layer to an RTP
	// stream ID. The zero value maps every layer to stream 0.
	SpatialLayerRTPStreamIDs [EncoderWebRTCMaxSpatialLayers]int

	// ActiveSpatialLayerMaskSet makes ActiveSpatialLayerMask authoritative. When
	// false, all configured spatial layers are active. When true, bit N selects
	// spatial layer N; a zero mask writes the paused one-byte allocation form.
	ActiveSpatialLayerMaskSet bool
	ActiveSpatialLayerMask    uint8

	// TargetBitratesKbps holds cumulative target bitrates in kbps by spatial
	// layer then temporal layer. Single-temporal layers may omit this and use the
	// normalized encoder spatial-layer target bitrate.
	TargetBitratesKbps [EncoderWebRTCMaxSpatialLayers][EncoderWebRTCMaxTemporalLayers]uint32

	IncludeResolutionAndFramerate bool
}

// EncoderWebRTCVideoLayersAllocationScratch owns the backing storage used by
// EncoderWebRTCVideoLayersAllocationInto. The returned RTPVideoLayersAllocation
// aliases this scratch until the next call that reuses it.
type EncoderWebRTCVideoLayersAllocationScratch struct {
	ActiveSpatialLayers [EncoderWebRTCMaxSpatialLayers]RTPVideoLayersAllocationLayer
	TargetBitratesKbps  [EncoderWebRTCMaxSpatialLayers][EncoderWebRTCMaxTemporalLayers]uint32
}

type EncoderFrameSamplePlanes struct {
	Y FrameSamplePlane
	U FrameSamplePlane
	V FrameSamplePlane
}

type EncoderWebRTCPictureSampleScratchSize struct {
	Samples  int
	FrameNum int
}

type EncoderWebRTCPictureSampleScratch struct {
	Samples []uint16
}

type EncoderWebRTCPictureSamplePlanes struct {
	Frames   [EncoderWebRTCMaxSpatialLayers]EncoderFrameSamplePlanes
	FrameNum uint8
}

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

	EncoderRestorationNone       = internalencoder.RestorationNone
	EncoderRestorationSwitchable = internalencoder.RestorationSwitchable
	EncoderRestorationWiener     = internalencoder.RestorationWiener
	EncoderRestorationSGRProj    = internalencoder.RestorationSGRProj

	EncoderInterpolationEightTap   = internalencoder.InterpolationEightTap
	EncoderInterpolationSmooth     = internalencoder.InterpolationSmooth
	EncoderInterpolationSharp      = internalencoder.InterpolationSharp
	EncoderInterpolationBilinear   = internalencoder.InterpolationBilinear
	EncoderInterpolationSwitchable = internalencoder.InterpolationSwitchable

	EncoderTransformMode4x4Only    = internalencoder.TransformMode4x4Only
	EncoderTransformModeLargest    = internalencoder.TransformModeLargest
	EncoderTransformModeSwitchable = internalencoder.TransformModeSwitchable

	EncoderReferenceModeSingle   = internalencoder.ReferenceModeSingle
	EncoderReferenceModeCompound = internalencoder.ReferenceModeCompound
	EncoderReferenceModeSelect   = internalencoder.ReferenceModeSelect

	EncoderGlobalMotionIdentity    = internalencoder.GlobalMotionIdentity
	EncoderGlobalMotionTranslation = internalencoder.GlobalMotionTranslation
	EncoderGlobalMotionRotZoom     = internalencoder.GlobalMotionRotZoom
	EncoderGlobalMotionAffine      = internalencoder.GlobalMotionAffine

	EncoderMaxFilmGrainYPoints  = internalencoder.MaxFilmGrainYPoints
	EncoderMaxFilmGrainUVPoints = internalencoder.MaxFilmGrainUVPoints
	EncoderMaxFilmGrainYCoeffs  = internalencoder.MaxFilmGrainYCoeffs
	EncoderMaxFilmGrainUVCoeffs = internalencoder.MaxFilmGrainUVCoeffs

	EncoderMaxTileRows = internalencoder.MaxTileRows
	EncoderMaxTileCols = internalencoder.MaxTileCols
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

// EncoderWebRTCScalabilityModes returns the W3C WebRTC SVC scalabilityMode
// values supported by the WebRTC encoder/control surfaces.
func EncoderWebRTCScalabilityModes() []EncoderScalabilityMode {
	out := make([]EncoderScalabilityMode, 0, internalencoder.WebRTCScalabilityModeCount())
	return internalencoder.AppendWebRTCScalabilityModes(out)
}

// AppendEncoderWebRTCScalabilityModes appends the W3C WebRTC SVC
// scalabilityMode values supported by the WebRTC encoder/control surfaces to
// dst and returns the extended slice.
func AppendEncoderWebRTCScalabilityModes(dst []EncoderScalabilityMode) []EncoderScalabilityMode {
	return internalencoder.AppendWebRTCScalabilityModes(dst)
}

// ValidateEncoderWebRTCActiveScalabilityModes validates the active
// scalabilityMode values for one WebRTC sender. Pass only active encodings; a
// sender with no active encodings is accepted. Per the WebRTC-SVC setParameters
// rule, multiple active encodings cannot include a single-SSRC simulcast S mode
// such as S2T1 or S3T3h.
func ValidateEncoderWebRTCActiveScalabilityModes(modes ...EncoderScalabilityMode) error {
	return internalencoder.ValidateWebRTCActiveScalabilityModes(modes)
}

// EncoderWebRTCScalabilityModeIDC maps a W3C WebRTC scalabilityMode value to
// its predefined AV1 metadata_scalability() scalability_mode_idc value. ok is
// false when AV1 has no predefined IDC for the WebRTC mode. The metadata OBU
// helpers still support those modes by writing MetadataScalabilityModeSS with
// an explicit scalability_structure().
func EncoderWebRTCScalabilityModeIDC(mode EncoderScalabilityMode) (idc uint8, ok bool) {
	return internalencoder.WebRTCScalabilityModeIDC(mode)
}

func EncoderWebRTCScalabilityMetadataPayloadSize(mode EncoderScalabilityMode) (size int, ok bool) {
	return internalencoder.WebRTCScalabilityMetadataPayloadSize(mode)
}

func AppendEncoderWebRTCScalabilityMetadataPayload(dst []byte, mode EncoderScalabilityMode) ([]byte, bool, error) {
	return internalencoder.AppendWebRTCScalabilityMetadataPayload(dst, mode)
}

func EncoderLowOverheadWebRTCScalabilityMetadataOBUSize(mode EncoderScalabilityMode) (size int, ok bool, err error) {
	return internalencoder.LowOverheadWebRTCScalabilityMetadataOBUSize(mode)
}

func AppendEncoderLowOverheadWebRTCScalabilityMetadataOBU(dst []byte, mode EncoderScalabilityMode) ([]byte, bool, error) {
	return internalencoder.AppendLowOverheadWebRTCScalabilityMetadataOBU(dst, mode)
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

// EncoderWebRTCRTPFrameDuration returns the exact RTP timestamp duration of
// one encoded WebRTC picture after applying the same config normalization used
// by the WebRTC encoder. The returned rational is expressed in RTP timestamp
// ticks, not seconds. For the default 90 kHz RTP timebase, 30 fps returns
// 3000/1 and 30000/1001 fps returns 3003/1.
func EncoderWebRTCRTPFrameDuration(config EncoderConfig) (EncoderRational, error) {
	normalized, err := internalencoder.SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return EncoderRational{}, err
	}
	num := int64(normalized.MaxFramerate.Den) * int64(normalized.RTPTimebase.Den)
	den := int64(normalized.MaxFramerate.Num) * int64(normalized.RTPTimebase.Num)
	div := encoderWebRTCRTPDurationGCD(num, den)
	num /= div
	den /= div
	if num > 1<<31-1 || den > 1<<31-1 {
		return EncoderRational{}, internalencoder.ErrInvalidConfig
	}
	return EncoderRational{Num: int32(num), Den: int32(den)}, nil
}

// EncoderWebRTCVideoLayersAllocation builds WebRTC's video-layers-allocation
// RTP header-extension payload model from an encoder config and explicit
// allocation controls. The result is ready for RTPVideoLayersAllocationSize,
// PutRTPVideoLayersAllocationHeaderExtension, or
// EncoderWebRTCRTPPacketHeaderConfig.VideoLayersAllocation.
func EncoderWebRTCVideoLayersAllocation(config EncoderConfig, allocationConfig EncoderWebRTCVideoLayersAllocationConfig) (RTPVideoLayersAllocation, error) {
	var scratch EncoderWebRTCVideoLayersAllocationScratch
	allocation, err := EncoderWebRTCVideoLayersAllocationInto(config, allocationConfig, &scratch)
	if err != nil {
		return RTPVideoLayersAllocation{}, err
	}
	return cloneRTPVideoLayersAllocation(allocation), nil
}

// EncoderWebRTCVideoLayersAllocationInto is EncoderWebRTCVideoLayersAllocation
// with caller-owned scratch for production control loops that refresh layer
// allocation metadata when bitrate, framerate, active layers, or scalability
// settings change.
func EncoderWebRTCVideoLayersAllocationInto(config EncoderConfig, allocationConfig EncoderWebRTCVideoLayersAllocationConfig, scratch *EncoderWebRTCVideoLayersAllocationScratch) (RTPVideoLayersAllocation, error) {
	if scratch == nil {
		return RTPVideoLayersAllocation{}, ErrEncoderInvalidConfig
	}
	normalized, err := internalencoder.SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return RTPVideoLayersAllocation{}, err
	}
	spatialLayers := int(normalized.SpatialLayerCount)
	temporalLayers := int(normalized.TemporalLayerCount)
	if spatialLayers <= 0 || spatialLayers > EncoderWebRTCMaxSpatialLayers ||
		temporalLayers <= 0 || temporalLayers > EncoderWebRTCMaxTemporalLayers {
		return RTPVideoLayersAllocation{}, ErrEncoderInvalidConfig
	}

	rtpStreamCount := allocationConfig.RTPStreamCount
	if rtpStreamCount == 0 {
		rtpStreamCount = 1
	}
	if rtpStreamCount < 1 || rtpStreamCount > 4 ||
		allocationConfig.RTPStreamID < 0 || allocationConfig.RTPStreamID >= rtpStreamCount {
		return RTPVideoLayersAllocation{}, ErrEncoderInvalidConfig
	}

	allSpatialMask := uint8((1 << uint(spatialLayers)) - 1)
	activeMask := allSpatialMask
	if allocationConfig.ActiveSpatialLayerMaskSet {
		activeMask = allocationConfig.ActiveSpatialLayerMask
		if activeMask&^allSpatialMask != 0 {
			return RTPVideoLayersAllocation{}, ErrEncoderInvalidConfig
		}
	}
	if activeMask == 0 {
		if allocationConfig.RTPStreamID != 0 || rtpStreamCount != 1 {
			return RTPVideoLayersAllocation{}, ErrEncoderInvalidConfig
		}
		return RTPVideoLayersAllocation{RTPStreamCount: 1}, nil
	}

	framerate := 0
	if allocationConfig.IncludeResolutionAndFramerate {
		framerate, err = encoderWebRTCVideoLayersAllocationFramerate(normalized.MaxFramerate)
		if err != nil {
			return RTPVideoLayersAllocation{}, err
		}
	}

	allocation := RTPVideoLayersAllocation{
		RTPStreamID:               allocationConfig.RTPStreamID,
		RTPStreamCount:            rtpStreamCount,
		ActiveSpatialLayers:       scratch.ActiveSpatialLayers[:0],
		HasResolutionAndFramerate: allocationConfig.IncludeResolutionAndFramerate,
	}
	for streamID := 0; streamID < rtpStreamCount; streamID++ {
		for spatialID := 0; spatialID < spatialLayers; spatialID++ {
			if activeMask&(1<<uint(spatialID)) == 0 {
				continue
			}
			layerStreamID := allocationConfig.SpatialLayerRTPStreamIDs[spatialID]
			if layerStreamID < 0 || layerStreamID >= rtpStreamCount {
				return RTPVideoLayersAllocation{}, ErrEncoderInvalidConfig
			}
			if layerStreamID != streamID {
				continue
			}
			layer, err := encoderWebRTCVideoLayersAllocationLayer(normalized, allocationConfig, spatialID, temporalLayers, layerStreamID, framerate, scratch.TargetBitratesKbps[spatialID][:])
			if err != nil {
				return RTPVideoLayersAllocation{}, err
			}
			allocation.ActiveSpatialLayers = append(allocation.ActiveSpatialLayers, layer)
		}
	}
	if len(allocation.ActiveSpatialLayers) == 0 {
		return RTPVideoLayersAllocation{}, ErrEncoderInvalidConfig
	}
	if err := ValidateRTPVideoLayersAllocation(allocation); err != nil {
		return RTPVideoLayersAllocation{}, err
	}
	return allocation, nil
}

func cloneRTPVideoLayersAllocation(allocation RTPVideoLayersAllocation) RTPVideoLayersAllocation {
	out := allocation
	if len(allocation.ActiveSpatialLayers) == 0 {
		return out
	}
	out.ActiveSpatialLayers = make([]RTPVideoLayersAllocationLayer, len(allocation.ActiveSpatialLayers))
	copy(out.ActiveSpatialLayers, allocation.ActiveSpatialLayers)
	for i := range out.ActiveSpatialLayers {
		targets := allocation.ActiveSpatialLayers[i].TargetBitratesKbps
		if len(targets) != 0 {
			out.ActiveSpatialLayers[i].TargetBitratesKbps = append([]uint32(nil), targets...)
		}
	}
	return out
}

func encoderWebRTCVideoLayersAllocationLayer(config EncoderConfig, allocationConfig EncoderWebRTCVideoLayersAllocationConfig, spatialID int, temporalLayers int, rtpStreamID int, framerate int, targetScratch []uint32) (RTPVideoLayersAllocationLayer, error) {
	targets, err := encoderWebRTCVideoLayersAllocationTargets(config, allocationConfig, spatialID, temporalLayers, targetScratch)
	if err != nil {
		return RTPVideoLayersAllocationLayer{}, err
	}
	layerConfig := config.SpatialLayers[spatialID]
	layer := RTPVideoLayersAllocationLayer{
		RTPStreamID:        rtpStreamID,
		SpatialID:          spatialID,
		TargetBitratesKbps: targets,
	}
	if allocationConfig.IncludeResolutionAndFramerate {
		layer.Width = int(layerConfig.Resolution.Width)
		layer.Height = int(layerConfig.Resolution.Height)
		layer.Framerate = framerate
	}
	return layer, nil
}

func encoderWebRTCVideoLayersAllocationTargets(config EncoderConfig, allocationConfig EncoderWebRTCVideoLayersAllocationConfig, spatialID int, temporalLayers int, targetScratch []uint32) ([]uint32, error) {
	if len(targetScratch) < temporalLayers {
		return nil, ErrEncoderInvalidConfig
	}
	targets := targetScratch[:temporalLayers]
	var previous uint32
	for temporalID := 0; temporalID < temporalLayers; temporalID++ {
		target := allocationConfig.TargetBitratesKbps[spatialID][temporalID]
		if target == 0 && temporalLayers == 1 {
			layerTarget := config.SpatialLayers[spatialID].TargetBitrateKbps
			if layerTarget <= 0 {
				layerTarget = config.TargetBitrateKbps
			}
			if layerTarget > 0 {
				target = uint32(layerTarget)
			}
		}
		if target == 0 ||
			target > uint32(EncoderWebRTCMaxBitrateKbps) ||
			(temporalID > 0 && target < previous) {
			return nil, ErrEncoderInvalidConfig
		}
		targets[temporalID] = target
		previous = target
	}
	return targets, nil
}

func encoderWebRTCVideoLayersAllocationFramerate(rate EncoderRational) (int, error) {
	if !rate.Valid() {
		return 0, ErrEncoderInvalidConfig
	}
	fps := int((int64(rate.Num) + int64(rate.Den)/2) / int64(rate.Den))
	if fps < 1 || fps > 0xff {
		return 0, ErrEncoderInvalidConfig
	}
	return fps, nil
}

func encoderWebRTCRTPDurationGCD(a int64, b int64) int64 {
	for b != 0 {
		a, b = b, a%b
	}
	if a < 0 {
		return -a
	}
	return a
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

func EncoderWebRTCAllDecodeTargetsMask(structure EncoderWebRTCFrameDependencyStructure) (uint32, error) {
	return internalencoder.WebRTCAllDecodeTargetsMask(structure)
}

func EncoderWebRTCActiveDecodeTargetsMask(structure EncoderWebRTCFrameDependencyStructure, maxSpatialID uint8, maxTemporalID uint8) (uint32, error) {
	return internalencoder.WebRTCActiveDecodeTargetsMask(structure, maxSpatialID, maxTemporalID)
}

func EncoderWebRTCSpatialDecodeTargetsMask(structure EncoderWebRTCFrameDependencyStructure, spatialID uint8, maxTemporalID uint8) (uint32, error) {
	return internalencoder.WebRTCSpatialDecodeTargetsMask(structure, spatialID, maxTemporalID)
}

func EncoderWebRTCDependencyDescriptorSizeWithOptions(structure EncoderWebRTCFrameDependencyStructure, info EncoderWebRTCGenericFrameInfo, options EncoderWebRTCDependencyDescriptorOptions) (int, error) {
	return internalencoder.WebRTCDependencyDescriptorSizeWithOptions(structure, info, options)
}

func AppendEncoderWebRTCDependencyDescriptor(dst []byte, structure EncoderWebRTCFrameDependencyStructure, info EncoderWebRTCGenericFrameInfo, firstPacketInFrame bool, lastPacketInFrame bool, attachStructure bool) ([]byte, error) {
	return internalencoder.AppendWebRTCDependencyDescriptor(dst, structure, info, firstPacketInFrame, lastPacketInFrame, attachStructure)
}

func AppendEncoderWebRTCDependencyDescriptorWithOptions(dst []byte, structure EncoderWebRTCFrameDependencyStructure, info EncoderWebRTCGenericFrameInfo, options EncoderWebRTCDependencyDescriptorOptions) ([]byte, error) {
	return internalencoder.AppendWebRTCDependencyDescriptorWithOptions(dst, structure, info, options)
}

func EncoderWebRTCRTPPacketDependencyDescriptorSize(packetizer *RTPPacketizer, structure EncoderWebRTCFrameDependencyStructure, info EncoderWebRTCGenericFrameInfo, attachStructureOnFirstPacket bool) (int, bool, error) {
	return EncoderWebRTCRTPPacketDependencyDescriptorSizeWithOptions(packetizer, structure, info, EncoderWebRTCRTPPacketDependencyDescriptorOptions{
		AttachStructureOnFirstPacket: attachStructureOnFirstPacket,
	})
}

func EncoderWebRTCRTPPacketDependencyDescriptorSizeWithOptions(packetizer *RTPPacketizer, structure EncoderWebRTCFrameDependencyStructure, info EncoderWebRTCGenericFrameInfo, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) (int, bool, error) {
	flags, ok := NextRTPPacketDependencyDescriptorFlags(packetizer)
	if !ok {
		return 0, false, nil
	}
	size, err := internalencoder.WebRTCDependencyDescriptorSizeWithOptions(structure, info, encoderWebRTCRTPPacketDependencyDescriptorOptions(flags, options))
	if err != nil {
		return 0, true, err
	}
	return size, true, nil
}

func AppendEncoderWebRTCRTPPacketDependencyDescriptor(dst []byte, packetizer *RTPPacketizer, structure EncoderWebRTCFrameDependencyStructure, info EncoderWebRTCGenericFrameInfo, attachStructureOnFirstPacket bool) ([]byte, bool, error) {
	return AppendEncoderWebRTCRTPPacketDependencyDescriptorWithOptions(dst, packetizer, structure, info, EncoderWebRTCRTPPacketDependencyDescriptorOptions{
		AttachStructureOnFirstPacket: attachStructureOnFirstPacket,
	})
}

func AppendEncoderWebRTCRTPPacketDependencyDescriptorWithOptions(dst []byte, packetizer *RTPPacketizer, structure EncoderWebRTCFrameDependencyStructure, info EncoderWebRTCGenericFrameInfo, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) ([]byte, bool, error) {
	flags, ok := NextRTPPacketDependencyDescriptorFlags(packetizer)
	if !ok {
		return dst, false, nil
	}
	out, err := internalencoder.AppendWebRTCDependencyDescriptorWithOptions(dst, structure, info, encoderWebRTCRTPPacketDependencyDescriptorOptions(flags, options))
	if err != nil {
		return dst, true, err
	}
	return out, true, nil
}

func EncoderWebRTCFrameControlRTPPacketDependencyDescriptorSize(packetizer *RTPPacketizer, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure) (int, bool, error) {
	return EncoderWebRTCFrameControlRTPPacketDependencyDescriptorSizeWithOptions(packetizer, control, structure, EncoderWebRTCRTPPacketDependencyDescriptorOptions{
		AttachStructureOnFirstPacket: control.AttachDependencyStructure,
	})
}

func EncoderWebRTCFrameControlRTPPacketDependencyDescriptorSizeWithOptions(packetizer *RTPPacketizer, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) (int, bool, error) {
	if control.AttachDependencyStructure {
		options.AttachStructureOnFirstPacket = true
	}
	return EncoderWebRTCRTPPacketDependencyDescriptorSizeWithOptions(packetizer, structure, control.GenericFrameInfo, options)
}

func AppendEncoderWebRTCFrameControlRTPPacketDependencyDescriptor(dst []byte, packetizer *RTPPacketizer, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure) ([]byte, bool, error) {
	return AppendEncoderWebRTCFrameControlRTPPacketDependencyDescriptorWithOptions(dst, packetizer, control, structure, EncoderWebRTCRTPPacketDependencyDescriptorOptions{
		AttachStructureOnFirstPacket: control.AttachDependencyStructure,
	})
}

func AppendEncoderWebRTCFrameControlRTPPacketDependencyDescriptorWithOptions(dst []byte, packetizer *RTPPacketizer, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) ([]byte, bool, error) {
	if control.AttachDependencyStructure {
		options.AttachStructureOnFirstPacket = true
	}
	return AppendEncoderWebRTCRTPPacketDependencyDescriptorWithOptions(dst, packetizer, structure, control.GenericFrameInfo, options)
}

func EncoderWebRTCFrameControlRTPPacketSize(packetizer *RTPPacketizer, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure) (payloadSize int, descriptorSize int, ok bool, err error) {
	return EncoderWebRTCFrameControlRTPPacketSizeWithOptions(packetizer, control, structure, EncoderWebRTCRTPPacketDependencyDescriptorOptions{
		AttachStructureOnFirstPacket: control.AttachDependencyStructure,
	})
}

func EncoderWebRTCFrameControlRTPPacketSizeWithOptions(packetizer *RTPPacketizer, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) (payloadSize int, descriptorSize int, ok bool, err error) {
	payloadSize, ok = packetizer.NextPacketSize()
	if !ok {
		return 0, 0, false, nil
	}
	descriptorSize, ok, err = EncoderWebRTCFrameControlRTPPacketDependencyDescriptorSizeWithOptions(packetizer, control, structure, options)
	if err != nil {
		return 0, 0, true, err
	}
	if !ok {
		return 0, 0, false, nil
	}
	return payloadSize, descriptorSize, true, nil
}

func AppendEncoderWebRTCFrameControlRTPPacket(payloadDst []byte, descriptorDst []byte, packetizer *RTPPacketizer, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure) (payload []byte, descriptor []byte, marker bool, ok bool, err error) {
	return AppendEncoderWebRTCFrameControlRTPPacketWithOptions(payloadDst, descriptorDst, packetizer, control, structure, EncoderWebRTCRTPPacketDependencyDescriptorOptions{
		AttachStructureOnFirstPacket: control.AttachDependencyStructure,
	})
}

func AppendEncoderWebRTCFrameControlRTPPacketWithOptions(payloadDst []byte, descriptorDst []byte, packetizer *RTPPacketizer, control EncoderWebRTCFrameControl, structure EncoderWebRTCFrameDependencyStructure, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) (payload []byte, descriptor []byte, marker bool, ok bool, err error) {
	descriptor, ok, err = AppendEncoderWebRTCFrameControlRTPPacketDependencyDescriptorWithOptions(descriptorDst, packetizer, control, structure, options)
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

func encoderWebRTCRTPPacketDependencyDescriptorOptions(flags RTPPacketDependencyDescriptorFlags, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) EncoderWebRTCDependencyDescriptorOptions {
	return EncoderWebRTCDependencyDescriptorOptions{
		FirstPacketInFrame:         flags.FirstPacketInFrame,
		LastPacketInFrame:          flags.LastPacketInFrame,
		AttachStructure:            options.AttachStructureOnFirstPacket && flags.FirstPacketInFrame,
		ActiveDecodeTargetsPresent: options.ActiveDecodeTargetsPresentOnFirstPacket && flags.FirstPacketInFrame,
		ActiveDecodeTargetsMask:    options.ActiveDecodeTargetsMask,
	}
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

func AppendEncoderWebRTCPictureTemporalUnitFramesRTPPacketsWithScratch(payloadDst []byte, descriptorDst []byte, scratch EncoderWebRTCPictureTemporalUnitFramesRTPScratch, framePayloads [][]byte, limits RTPPayloadSizeLimits, unit EncoderWebRTCPictureTemporalUnit, state EncoderWebRTCState) (frameOBUs []byte, rtpPayloads []byte, descriptors []byte, frameCount int, packetCount int, err error) {
	return AppendEncoderWebRTCPictureTemporalUnitFramesRTPPackets(
		scratch.FrameOBU,
		payloadDst,
		descriptorDst,
		scratch.FrameSpans,
		scratch.PacketSpans,
		framePayloads,
		limits,
		unit,
		state,
		scratch.OBUs,
		scratch.Packets,
		scratch.Work,
	)
}

func EncoderWebRTCNextTemporalUnitFramesRTPScratchLen(framePayloads [][]byte, limits RTPPayloadSizeLimits, config EncoderConfig, state EncoderWebRTCState, forceKeyFrame bool, frameOBUScratch []byte, obuScratch []RTPPacketizerOBU) (EncoderWebRTCPictureTemporalUnitFramesRTPScratchSize, EncoderWebRTCPictureTemporalUnit, EncoderWebRTCState, error) {
	unit, next, err := EncoderWebRTCNextTemporalUnitForState(config, state, forceKeyFrame)
	if err != nil {
		return EncoderWebRTCPictureTemporalUnitFramesRTPScratchSize{}, EncoderWebRTCPictureTemporalUnit{}, EncoderWebRTCState{}, err
	}
	size, err := EncoderWebRTCPictureTemporalUnitFramesRTPScratchLen(framePayloads, limits, unit, next, frameOBUScratch, obuScratch)
	if err != nil {
		return size, unit, next, err
	}
	return size, unit, next, nil
}

func EncoderWebRTCNextTemporalUnitFramesRTPPacketsSize(framePayloads [][]byte, limits RTPPayloadSizeLimits, config EncoderConfig, state EncoderWebRTCState, forceKeyFrame bool, frameOBUScratch []byte, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo, EncoderWebRTCPictureTemporalUnit, EncoderWebRTCState, error) {
	unit, next, err := EncoderWebRTCNextTemporalUnitForState(config, state, forceKeyFrame)
	if err != nil {
		return EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, EncoderWebRTCPictureTemporalUnit{}, EncoderWebRTCState{}, err
	}
	size, err := EncoderWebRTCPictureTemporalUnitFramesRTPPacketsSize(framePayloads, limits, unit, next, frameOBUScratch, obuScratch, packetScratch, workScratch)
	if err != nil {
		return EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, unit, next, err
	}
	return size, unit, next, nil
}

func AppendEncoderWebRTCNextTemporalUnitFramesRTPPacketsWithScratch(payloadDst []byte, descriptorDst []byte, scratch EncoderWebRTCPictureTemporalUnitFramesRTPScratch, framePayloads [][]byte, limits RTPPayloadSizeLimits, config EncoderConfig, state EncoderWebRTCState, forceKeyFrame bool) (frameOBUs []byte, rtpPayloads []byte, descriptors []byte, frameCount int, packetCount int, unit EncoderWebRTCPictureTemporalUnit, next EncoderWebRTCState, err error) {
	unit, next, err = EncoderWebRTCNextTemporalUnitForState(config, state, forceKeyFrame)
	if err != nil {
		return scratch.FrameOBU, payloadDst, descriptorDst, 0, 0, EncoderWebRTCPictureTemporalUnit{}, EncoderWebRTCState{}, err
	}
	frameOBUs, rtpPayloads, descriptors, frameCount, packetCount, err = AppendEncoderWebRTCPictureTemporalUnitFramesRTPPacketsWithScratch(
		payloadDst,
		descriptorDst,
		scratch,
		framePayloads,
		limits,
		unit,
		next,
	)
	if err != nil {
		return scratch.FrameOBU, payloadDst, descriptorDst, 0, 0, EncoderWebRTCPictureTemporalUnit{}, EncoderWebRTCState{}, err
	}
	return frameOBUs, rtpPayloads, descriptors, frameCount, packetCount, unit, next, nil
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

func EncoderWebRTCDeltaFrameTemporalUnitForConfigWithOrderHint(config EncoderConfig, referenceState EncoderReferenceBufferState, frameIDState EncoderFrameIDBufferState, temporalID uint8, firstFrameID uint64, orderHint uint8) (EncoderWebRTCDeltaFrameTemporalUnit, error) {
	return internalencoder.WebRTCDeltaFrameTemporalUnitForConfigWithOrderHint(config, referenceState, frameIDState, temporalID, firstFrameID, orderHint)
}

func EncoderWebRTCDeltaFrameTemporalUnitForConfigWithDeltaPictureIndex(config EncoderConfig, referenceState EncoderReferenceBufferState, frameIDState EncoderFrameIDBufferState, deltaPictureIndex uint64, firstFrameID uint64, orderHint uint8) (EncoderWebRTCDeltaFrameTemporalUnit, error) {
	return internalencoder.WebRTCDeltaFrameTemporalUnitForConfigWithDeltaPictureIndex(config, referenceState, frameIDState, deltaPictureIndex, firstFrameID, orderHint)
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

func NewWebRTCEncoder(config EncoderConfig, state EncoderWebRTCState) (WebRTCEncoder, error) {
	normalized, err := internalencoder.SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return WebRTCEncoder{}, err
	}
	if _, err := internalencoder.SequenceHeaderForConfig(normalized); err != nil {
		return WebRTCEncoder{}, err
	}
	return WebRTCEncoder{config: normalized, state: state}, nil
}

func (e *WebRTCEncoder) Config() EncoderConfig {
	if e == nil {
		return EncoderConfig{}
	}
	return e.config
}

// RTPFrameDuration returns the exact RTP timestamp duration of one encoded
// picture under the encoder's current normalized config.
func (e *WebRTCEncoder) RTPFrameDuration() (EncoderRational, error) {
	if e == nil {
		return EncoderRational{}, ErrEncoderInvalidConfig
	}
	return EncoderWebRTCRTPFrameDuration(e.config)
}

// SetConfig atomically updates the WebRTC control config. Changes that alter
// the AV1 sequence header or dependency structure reset reference/dependency
// state so the next temporal unit is a key unit while preserving frame number
// and order-hint continuity.
func (e *WebRTCEncoder) SetConfig(config EncoderConfig) error {
	if e == nil {
		return ErrEncoderInvalidConfig
	}
	normalized, err := internalencoder.SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return err
	}
	seq, err := internalencoder.SequenceHeaderForConfig(normalized)
	if err != nil {
		return err
	}
	prevSeq, prevSeqErr := internalencoder.SequenceHeaderForConfig(e.config)
	newStructure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(normalized)
	if err != nil {
		return err
	}
	prevStructure, prevStructureErr := internalencoder.WebRTCFrameDependencyStructureForConfig(e.config)
	if prevSeqErr != nil || prevSeq != seq ||
		prevStructureErr != nil || prevStructure != newStructure {
		e.state = encoderWebRTCStateForNextKey(e.state)
	}
	e.config = normalized
	return nil
}

func encoderWebRTCStateForNextKey(state EncoderWebRTCState) EncoderWebRTCState {
	return EncoderWebRTCState{
		NextOrderHint: state.NextOrderHint,
		NextFrameID:   state.NextFrameID,
	}
}

func (e *WebRTCEncoder) State() EncoderWebRTCState {
	if e == nil {
		return EncoderWebRTCState{}
	}
	return e.state
}

func (e *WebRTCEncoder) ResetState(state EncoderWebRTCState) error {
	if e == nil {
		return ErrEncoderInvalidConfig
	}
	e.state = state
	return nil
}

func (e *WebRTCEncoder) LowOverheadPictureHeaderTemporalUnitSize(forceKeyFrame bool) (int, EncoderWebRTCPictureTemporalUnit, error) {
	if e == nil {
		return 0, EncoderWebRTCPictureTemporalUnit{}, ErrEncoderInvalidConfig
	}
	size, unit, _, err := internalencoder.LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(e.config, e.state, forceKeyFrame)
	return size, unit, err
}

func (e *WebRTCEncoder) AppendLowOverheadPictureHeaderTemporalUnit(dst []byte, forceKeyFrame bool) ([]byte, EncoderWebRTCPictureTemporalUnit, error) {
	if e == nil {
		return dst, EncoderWebRTCPictureTemporalUnit{}, ErrEncoderInvalidConfig
	}
	out, unit, next, err := internalencoder.AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState(dst, e.config, e.state, forceKeyFrame)
	if err != nil {
		return dst, EncoderWebRTCPictureTemporalUnit{}, err
	}
	e.state = next
	return out, unit, nil
}

func (e *WebRTCEncoder) LowOverheadPictureHeaderTemporalUnitForFramesSize(frames []Frame, forceKeyFrame bool) (int, EncoderWebRTCPictureTemporalUnit, error) {
	if e == nil {
		return 0, EncoderWebRTCPictureTemporalUnit{}, ErrEncoderInvalidConfig
	}
	size, unit, _, err := internalencoder.LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(e.config, e.state, forceKeyFrame)
	if err != nil {
		return 0, EncoderWebRTCPictureTemporalUnit{}, err
	}
	if err := validateEncoderPictureFrames(frames, unit, e.config); err != nil {
		return 0, EncoderWebRTCPictureTemporalUnit{}, err
	}
	return size, unit, nil
}

func (e *WebRTCEncoder) AppendLowOverheadPictureHeaderTemporalUnitForFrames(dst []byte, frames []Frame, forceKeyFrame bool) ([]byte, EncoderWebRTCPictureTemporalUnit, error) {
	if e == nil {
		return dst, EncoderWebRTCPictureTemporalUnit{}, ErrEncoderInvalidConfig
	}
	_, unit, _, err := internalencoder.LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(e.config, e.state, forceKeyFrame)
	if err != nil {
		return dst, EncoderWebRTCPictureTemporalUnit{}, err
	}
	if err := validateEncoderPictureFrames(frames, unit, e.config); err != nil {
		return dst, EncoderWebRTCPictureTemporalUnit{}, err
	}
	out, appended, next, err := internalencoder.AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState(dst, e.config, e.state, forceKeyFrame)
	if err != nil {
		return dst, EncoderWebRTCPictureTemporalUnit{}, err
	}
	if appended != unit {
		return dst, EncoderWebRTCPictureTemporalUnit{}, ErrEncoderInvalidFrame
	}
	e.state = next
	return out, unit, nil
}

func (e *WebRTCEncoder) PictureSampleScratchSize(frames []Frame, forceKeyFrame bool) (EncoderWebRTCPictureSampleScratchSize, EncoderWebRTCPictureTemporalUnit, error) {
	if e == nil {
		return EncoderWebRTCPictureSampleScratchSize{}, EncoderWebRTCPictureTemporalUnit{}, ErrEncoderInvalidConfig
	}
	_, unit, _, err := internalencoder.LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(e.config, e.state, forceKeyFrame)
	if err != nil {
		return EncoderWebRTCPictureSampleScratchSize{}, EncoderWebRTCPictureTemporalUnit{}, err
	}
	if err := validateEncoderPictureFrames(frames, unit, e.config); err != nil {
		return EncoderWebRTCPictureSampleScratchSize{}, EncoderWebRTCPictureTemporalUnit{}, err
	}
	size, err := encoderPictureSampleScratchSize(frames)
	if err != nil {
		return EncoderWebRTCPictureSampleScratchSize{}, EncoderWebRTCPictureTemporalUnit{}, err
	}
	return size, unit, nil
}

func (e *WebRTCEncoder) LoadPictureSamplePlanes(scratch EncoderWebRTCPictureSampleScratch, frames []Frame, forceKeyFrame bool) (EncoderWebRTCPictureSamplePlanes, EncoderWebRTCPictureTemporalUnit, error) {
	if e == nil {
		return EncoderWebRTCPictureSamplePlanes{}, EncoderWebRTCPictureTemporalUnit{}, ErrEncoderInvalidConfig
	}
	size, unit, err := e.PictureSampleScratchSize(frames, forceKeyFrame)
	if err != nil {
		return EncoderWebRTCPictureSamplePlanes{}, EncoderWebRTCPictureTemporalUnit{}, err
	}
	if len(scratch.Samples) < size.Samples {
		return EncoderWebRTCPictureSamplePlanes{}, EncoderWebRTCPictureTemporalUnit{}, ErrFrameShortBuffer
	}
	planes, err := loadEncoderPictureSamplePlanes(scratch.Samples[:size.Samples], frames)
	if err != nil {
		return EncoderWebRTCPictureSamplePlanes{}, EncoderWebRTCPictureTemporalUnit{}, err
	}
	return planes, unit, nil
}

func (e *WebRTCEncoder) PictureTemporalUnitFramesRTPScratchSize(framePayloads [][]byte, limits RTPPayloadSizeLimits, forceKeyFrame bool, frameOBUScratch []byte, obuScratch []RTPPacketizerOBU) (EncoderWebRTCPictureTemporalUnitFramesRTPScratchSize, EncoderWebRTCPictureTemporalUnit, error) {
	if e == nil {
		return EncoderWebRTCPictureTemporalUnitFramesRTPScratchSize{}, EncoderWebRTCPictureTemporalUnit{}, ErrEncoderInvalidConfig
	}
	size, unit, _, err := EncoderWebRTCNextTemporalUnitFramesRTPScratchLen(framePayloads, limits, e.config, e.state, forceKeyFrame, frameOBUScratch, obuScratch)
	if err != nil {
		return size, EncoderWebRTCPictureTemporalUnit{}, err
	}
	return size, unit, nil
}

func (e *WebRTCEncoder) AppendPictureTemporalUnitFramesRTPPacketsWithScratch(payloadDst []byte, descriptorDst []byte, scratch EncoderWebRTCPictureTemporalUnitFramesRTPScratch, framePayloads [][]byte, limits RTPPayloadSizeLimits, forceKeyFrame bool) (frameOBUs []byte, rtpPayloads []byte, descriptors []byte, frameCount int, packetCount int, unit EncoderWebRTCPictureTemporalUnit, err error) {
	if e == nil {
		return scratch.FrameOBU, payloadDst, descriptorDst, 0, 0, EncoderWebRTCPictureTemporalUnit{}, ErrEncoderInvalidConfig
	}
	frameOBUs, rtpPayloads, descriptors, frameCount, packetCount, unit, next, err := AppendEncoderWebRTCNextTemporalUnitFramesRTPPacketsWithScratch(
		payloadDst,
		descriptorDst,
		scratch,
		framePayloads,
		limits,
		e.config,
		e.state,
		forceKeyFrame,
	)
	if err != nil {
		return scratch.FrameOBU, payloadDst, descriptorDst, 0, 0, EncoderWebRTCPictureTemporalUnit{}, err
	}
	e.state = next
	return frameOBUs, rtpPayloads, descriptors, frameCount, packetCount, unit, nil
}

func (e *WebRTCEncoder) NextTemporalUnit(forceKeyFrame bool) (EncoderWebRTCPictureTemporalUnit, error) {
	if e == nil {
		return EncoderWebRTCPictureTemporalUnit{}, ErrEncoderInvalidConfig
	}
	unit, next, err := internalencoder.WebRTCNextTemporalUnitForState(e.config, e.state, forceKeyFrame)
	if err != nil {
		return EncoderWebRTCPictureTemporalUnit{}, err
	}
	e.state = next
	return unit, nil
}

func validateEncoderPictureFrames(frames []Frame, unit EncoderWebRTCPictureTemporalUnit, config EncoderConfig) error {
	frameNum := EncoderWebRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || len(frames) != int(frameNum) {
		return ErrEncoderInvalidFrame
	}
	seq, err := internalencoder.SequenceHeaderForConfig(config)
	if err != nil {
		return err
	}
	for i := uint8(0); i < frameNum; i++ {
		settings, ok := encoderPictureUnitFrameSettings(unit, i)
		if !ok || !encoderFrameMatchesSettings(frames[i], settings, seq) {
			return ErrEncoderInvalidFrame
		}
	}
	return nil
}

func encoderPictureSampleScratchSize(frames []Frame) (EncoderWebRTCPictureSampleScratchSize, error) {
	total := 0
	for i := range frames {
		frame := frames[i]
		layout, err := FrameRequiredSize(frame.Format)
		if err != nil {
			return EncoderWebRTCPictureSampleScratchSize{}, err
		}
		need, err := FrameSamplePlaneLen(frame.Y, layout.BytesPerSample)
		if err != nil {
			return EncoderWebRTCPictureSampleScratchSize{}, err
		}
		if !checkedAddEncoderInt(&total, need) {
			return EncoderWebRTCPictureSampleScratchSize{}, ErrEncoderInvalidFrame
		}
		if frame.Format.MonoChrome {
			continue
		}
		need, err = FrameSamplePlaneLen(frame.U, layout.BytesPerSample)
		if err != nil {
			return EncoderWebRTCPictureSampleScratchSize{}, err
		}
		if !checkedAddEncoderInt(&total, need) {
			return EncoderWebRTCPictureSampleScratchSize{}, ErrEncoderInvalidFrame
		}
		need, err = FrameSamplePlaneLen(frame.V, layout.BytesPerSample)
		if err != nil {
			return EncoderWebRTCPictureSampleScratchSize{}, err
		}
		if !checkedAddEncoderInt(&total, need) {
			return EncoderWebRTCPictureSampleScratchSize{}, ErrEncoderInvalidFrame
		}
	}
	return EncoderWebRTCPictureSampleScratchSize{Samples: total, FrameNum: len(frames)}, nil
}

func loadEncoderPictureSamplePlanes(samples []uint16, frames []Frame) (EncoderWebRTCPictureSamplePlanes, error) {
	var out EncoderWebRTCPictureSamplePlanes
	if len(frames) > EncoderWebRTCMaxSpatialLayers {
		return EncoderWebRTCPictureSamplePlanes{}, ErrEncoderInvalidFrame
	}
	off := 0
	for i := range frames {
		frame := frames[i]
		layout, err := FrameRequiredSize(frame.Format)
		if err != nil {
			return EncoderWebRTCPictureSamplePlanes{}, err
		}
		yNeed, err := FrameSamplePlaneLen(frame.Y, layout.BytesPerSample)
		if err != nil {
			return EncoderWebRTCPictureSamplePlanes{}, err
		}
		y, err := LoadFrameSamplePlane(samples[off:off+yNeed], frame.Y, layout.BytesPerSample)
		if err != nil {
			return EncoderWebRTCPictureSamplePlanes{}, err
		}
		off += yNeed
		out.Frames[i].Y = y
		if !frame.Format.MonoChrome {
			uNeed, err := FrameSamplePlaneLen(frame.U, layout.BytesPerSample)
			if err != nil {
				return EncoderWebRTCPictureSamplePlanes{}, err
			}
			u, err := LoadFrameSamplePlane(samples[off:off+uNeed], frame.U, layout.BytesPerSample)
			if err != nil {
				return EncoderWebRTCPictureSamplePlanes{}, err
			}
			off += uNeed
			vNeed, err := FrameSamplePlaneLen(frame.V, layout.BytesPerSample)
			if err != nil {
				return EncoderWebRTCPictureSamplePlanes{}, err
			}
			v, err := LoadFrameSamplePlane(samples[off:off+vNeed], frame.V, layout.BytesPerSample)
			if err != nil {
				return EncoderWebRTCPictureSamplePlanes{}, err
			}
			off += vNeed
			out.Frames[i].U = u
			out.Frames[i].V = v
		}
	}
	out.FrameNum = uint8(len(frames))
	return out, nil
}

func checkedAddEncoderInt(dst *int, add int) bool {
	if add < 0 {
		return false
	}
	maxInt := int(^uint(0) >> 1)
	if *dst > maxInt-add {
		return false
	}
	*dst += add
	return true
}

func encoderPictureUnitFrameSettings(unit EncoderWebRTCPictureTemporalUnit, frameIndex uint8) (EncoderFrameEncodeSettings, bool) {
	if unit.Key == unit.Delta {
		return EncoderFrameEncodeSettings{}, false
	}
	if unit.Key {
		if frameIndex >= unit.KeyUnit.FrameNum {
			return EncoderFrameEncodeSettings{}, false
		}
		return unit.KeyUnit.Frames[frameIndex], true
	}
	if frameIndex >= unit.DeltaUnit.FrameNum {
		return EncoderFrameEncodeSettings{}, false
	}
	return unit.DeltaUnit.Frames[frameIndex], true
}

func encoderFrameMatchesSettings(f Frame, settings EncoderFrameEncodeSettings, seq EncoderSequenceHeader) bool {
	if !settings.Output || settings.Resolution.Width <= 0 || settings.Resolution.Height <= 0 {
		return false
	}
	if f.Format.Width != int(settings.Resolution.Width) || f.Format.Height != int(settings.Resolution.Height) ||
		f.Format.BitDepth != seq.ColorConfig.BitDepth || f.Format.MonoChrome != seq.ColorConfig.MonoChrome ||
		f.Format.SubsamplingX != seq.ColorConfig.SubsamplingX || f.Format.SubsamplingY != seq.ColorConfig.SubsamplingY {
		return false
	}
	layout, err := FrameRequiredSize(f.Format)
	if err != nil || layout.BytesPerSample <= 0 {
		return false
	}
	if f.Y.Width != f.Format.Width || f.Y.Height != f.Format.Height ||
		!encoderPlaneFits(f.Y, layout.BytesPerSample) {
		return false
	}
	if f.Format.MonoChrome {
		return len(f.U.Pix) == 0 && len(f.V.Pix) == 0 && f.U.Stride == 0 && f.V.Stride == 0
	}
	return f.U.Width == layout.ChromaWidth && f.U.Height == layout.ChromaHeight &&
		f.V.Width == layout.ChromaWidth && f.V.Height == layout.ChromaHeight &&
		encoderPlaneFits(f.U, layout.BytesPerSample) && encoderPlaneFits(f.V, layout.BytesPerSample)
}

func encoderPlaneFits(p FramePlane, bytesPerSample int) bool {
	if bytesPerSample != 1 && bytesPerSample != 2 {
		return false
	}
	if p.Width < 0 || p.Height < 0 || p.Stride < 0 {
		return false
	}
	if p.Width == 0 || p.Height == 0 {
		return len(p.Pix) == 0
	}
	maxInt := int(^uint(0) >> 1)
	if p.Width > maxInt/bytesPerSample {
		return false
	}
	rowBytes := p.Width * bytesPerSample
	if p.Stride < rowBytes {
		return false
	}
	if p.Height-1 > maxInt/p.Stride {
		return false
	}
	lastRow := (p.Height - 1) * p.Stride
	if lastRow > maxInt-rowBytes {
		return false
	}
	need := lastRow + rowBytes
	return need >= rowBytes && len(p.Pix) >= need
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

func EncoderFrameHeaderInterPayloadSize(seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderInterFrameSize) (int, error) {
	return internalencoder.FrameHeaderInterPayloadSize(seq, prefix, size)
}

func AppendEncoderFrameHeaderInterPayload(dst []byte, seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderInterFrameSize) ([]byte, error) {
	return internalencoder.AppendFrameHeaderInterPayload(dst, seq, prefix, size)
}

func EncoderIntraFrameHeaderPayloadSize(seq EncoderSequenceHeader, header EncoderIntraFrameHeaderParams) (int, error) {
	return internalencoder.IntraFrameHeaderPayloadSize(seq, header)
}

func AppendEncoderIntraFrameHeaderPayload(dst []byte, seq EncoderSequenceHeader, header EncoderIntraFrameHeaderParams) ([]byte, error) {
	return internalencoder.AppendIntraFrameHeaderPayload(dst, seq, header)
}

func EncoderInterFrameHeaderPayloadSize(seq EncoderSequenceHeader, header EncoderInterFrameHeaderParams) (int, error) {
	return internalencoder.InterFrameHeaderPayloadSize(seq, header)
}

func AppendEncoderInterFrameHeaderPayload(dst []byte, seq EncoderSequenceHeader, header EncoderInterFrameHeaderParams) ([]byte, error) {
	return internalencoder.AppendInterFrameHeaderPayload(dst, seq, header)
}

func EncoderTileInfoPayloadSize(seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, codedWidth uint32, height uint32, tiles EncoderTileInfo) (int, error) {
	return internalencoder.TileInfoPayloadSize(seq, prefix, codedWidth, height, tiles)
}

func AppendEncoderTileInfoPayload(dst []byte, seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, codedWidth uint32, height uint32, tiles EncoderTileInfo) ([]byte, error) {
	return internalencoder.AppendTileInfoPayload(dst, seq, prefix, codedWidth, height, tiles)
}

func EncoderTileGroupPayloadSize(tiles EncoderTileInfo, startTile uint16, endTile uint16, payloads []EncoderTilePayload) (int, error) {
	return internalencoder.TileGroupPayloadSize(tiles, startTile, endTile, payloads)
}

func AppendEncoderTileGroupPayload(dst []byte, tiles EncoderTileInfo, startTile uint16, endTile uint16, payloads []EncoderTilePayload) ([]byte, error) {
	return internalencoder.AppendTileGroupPayload(dst, tiles, startTile, endTile, payloads)
}

func EncoderQuantizationParamsPayloadSize(seq EncoderSequenceHeader, quant EncoderQuantizationParams) (int, error) {
	return internalencoder.QuantizationParamsPayloadSize(seq, quant)
}

func AppendEncoderQuantizationParamsPayload(dst []byte, seq EncoderSequenceHeader, quant EncoderQuantizationParams) ([]byte, error) {
	return internalencoder.AppendQuantizationParamsPayload(dst, seq, quant)
}

func EncoderSegmentationParamsPayloadSize(prefix EncoderFrameHeaderPrefix, seg EncoderSegmentationParams) (int, error) {
	return internalencoder.SegmentationParamsPayloadSize(prefix, seg)
}

func AppendEncoderSegmentationParamsPayload(dst []byte, prefix EncoderFrameHeaderPrefix, seg EncoderSegmentationParams) ([]byte, error) {
	return internalencoder.AppendSegmentationParamsPayload(dst, prefix, seg)
}

func EncoderDeltaParamsPayloadSize(size EncoderIntraFrameSize, quant EncoderQuantizationParams, delta EncoderDeltaParams) (int, error) {
	return internalencoder.DeltaParamsPayloadSize(size, quant, delta)
}

func AppendEncoderDeltaParamsPayload(dst []byte, size EncoderIntraFrameSize, quant EncoderQuantizationParams, delta EncoderDeltaParams) ([]byte, error) {
	return internalencoder.AppendDeltaParamsPayload(dst, size, quant, delta)
}

func EncoderDeltaParamsInterPayloadSize(size EncoderInterFrameSize, quant EncoderQuantizationParams, delta EncoderDeltaParams) (int, error) {
	return internalencoder.DeltaParamsInterPayloadSize(size, quant, delta)
}

func AppendEncoderDeltaParamsInterPayload(dst []byte, size EncoderInterFrameSize, quant EncoderQuantizationParams, delta EncoderDeltaParams) ([]byte, error) {
	return internalencoder.AppendDeltaParamsInterPayload(dst, size, quant, delta)
}

func EncoderLoopFilterParamsPayloadSize(seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderIntraFrameSize, allLossless bool, lf EncoderLoopFilterParams, previous *EncoderLoopFilterDeltas) (int, error) {
	return internalencoder.LoopFilterParamsPayloadSize(seq, prefix, size, allLossless, lf, previous)
}

func AppendEncoderLoopFilterParamsPayload(dst []byte, seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderIntraFrameSize, allLossless bool, lf EncoderLoopFilterParams, previous *EncoderLoopFilterDeltas) ([]byte, error) {
	return internalencoder.AppendLoopFilterParamsPayload(dst, seq, prefix, size, allLossless, lf, previous)
}

func EncoderLoopFilterParamsInterPayloadSize(seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderInterFrameSize, allLossless bool, lf EncoderLoopFilterParams, previous *EncoderLoopFilterDeltas) (int, error) {
	return internalencoder.LoopFilterParamsInterPayloadSize(seq, prefix, size, allLossless, lf, previous)
}

func AppendEncoderLoopFilterParamsInterPayload(dst []byte, seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderInterFrameSize, allLossless bool, lf EncoderLoopFilterParams, previous *EncoderLoopFilterDeltas) ([]byte, error) {
	return internalencoder.AppendLoopFilterParamsInterPayload(dst, seq, prefix, size, allLossless, lf, previous)
}

func EncoderCDEFParamsPayloadSize(seq EncoderSequenceHeader, size EncoderIntraFrameSize, allLossless bool, cdef EncoderCDEFParams) (int, error) {
	return internalencoder.CDEFParamsPayloadSize(seq, size, allLossless, cdef)
}

func AppendEncoderCDEFParamsPayload(dst []byte, seq EncoderSequenceHeader, size EncoderIntraFrameSize, allLossless bool, cdef EncoderCDEFParams) ([]byte, error) {
	return internalencoder.AppendCDEFParamsPayload(dst, seq, size, allLossless, cdef)
}

func EncoderCDEFParamsInterPayloadSize(seq EncoderSequenceHeader, size EncoderInterFrameSize, allLossless bool, cdef EncoderCDEFParams) (int, error) {
	return internalencoder.CDEFParamsInterPayloadSize(seq, size, allLossless, cdef)
}

func AppendEncoderCDEFParamsInterPayload(dst []byte, seq EncoderSequenceHeader, size EncoderInterFrameSize, allLossless bool, cdef EncoderCDEFParams) ([]byte, error) {
	return internalencoder.AppendCDEFParamsInterPayload(dst, seq, size, allLossless, cdef)
}

func EncoderRestorationParamsPayloadSize(seq EncoderSequenceHeader, size EncoderIntraFrameSize, allLossless bool, restoration EncoderRestorationParams) (int, error) {
	return internalencoder.RestorationParamsPayloadSize(seq, size, allLossless, restoration)
}

func AppendEncoderRestorationParamsPayload(dst []byte, seq EncoderSequenceHeader, size EncoderIntraFrameSize, allLossless bool, restoration EncoderRestorationParams) ([]byte, error) {
	return internalencoder.AppendRestorationParamsPayload(dst, seq, size, allLossless, restoration)
}

func EncoderRestorationParamsInterPayloadSize(seq EncoderSequenceHeader, size EncoderInterFrameSize, allLossless bool, restoration EncoderRestorationParams) (int, error) {
	return internalencoder.RestorationParamsInterPayloadSize(seq, size, allLossless, restoration)
}

func AppendEncoderRestorationParamsInterPayload(dst []byte, seq EncoderSequenceHeader, size EncoderInterFrameSize, allLossless bool, restoration EncoderRestorationParams) ([]byte, error) {
	return internalencoder.AppendRestorationParamsInterPayload(dst, seq, size, allLossless, restoration)
}

func EncoderTransformReferenceParamsPayloadSize(prefix EncoderFrameHeaderPrefix, allLossless bool, params EncoderTransformReferenceParams) (int, error) {
	return internalencoder.TransformReferenceParamsPayloadSize(prefix, allLossless, params)
}

func AppendEncoderTransformReferenceParamsPayload(dst []byte, prefix EncoderFrameHeaderPrefix, allLossless bool, params EncoderTransformReferenceParams) ([]byte, error) {
	return internalencoder.AppendTransformReferenceParamsPayload(dst, prefix, allLossless, params)
}

func EncoderSkipModeParamsPayloadSize(seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderInterFrameSize, refs *ReferenceState, transformRef EncoderTransformReferenceParams, params EncoderSkipModeParams) (int, error) {
	return internalencoder.SkipModeParamsPayloadSize(seq, prefix, size, refs, transformRef, params)
}

func AppendEncoderSkipModeParamsPayload(dst []byte, seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderInterFrameSize, refs *ReferenceState, transformRef EncoderTransformReferenceParams, params EncoderSkipModeParams) ([]byte, error) {
	return internalencoder.AppendSkipModeParamsPayload(dst, seq, prefix, size, refs, transformRef, params)
}

func EncoderFrameModeParamsPayloadSize(seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, params EncoderFrameModeParams) (int, error) {
	return internalencoder.FrameModeParamsPayloadSize(seq, prefix, params)
}

func AppendEncoderFrameModeParamsPayload(dst []byte, seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, params EncoderFrameModeParams) ([]byte, error) {
	return internalencoder.AppendFrameModeParamsPayload(dst, seq, prefix, params)
}

func EncoderDefaultWarpedMotionParams() EncoderWarpedMotionParams {
	return internalencoder.DefaultWarpedMotionParams()
}

func EncoderDefaultGlobalMotionParams() EncoderGlobalMotionParams {
	return internalencoder.DefaultGlobalMotionParams()
}

func EncoderGlobalMotionParamsPayloadSize(prefix EncoderFrameHeaderPrefix, size EncoderInterFrameSize, tiles TileInfo, refs *ReferenceState, params EncoderGlobalMotionParams) (int, error) {
	return internalencoder.GlobalMotionParamsPayloadSize(prefix, size, tiles, refs, params)
}

func AppendEncoderGlobalMotionParamsPayload(dst []byte, prefix EncoderFrameHeaderPrefix, size EncoderInterFrameSize, tiles TileInfo, refs *ReferenceState, params EncoderGlobalMotionParams) ([]byte, error) {
	return internalencoder.AppendGlobalMotionParamsPayload(dst, prefix, size, tiles, refs, params)
}

func EncoderFilmGrainParamsPayloadSize(seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderInterFrameSize, refs *ReferenceState, params EncoderFilmGrainParams) (int, error) {
	return internalencoder.FilmGrainParamsPayloadSize(seq, prefix, size, refs, params)
}

func AppendEncoderFilmGrainParamsPayload(dst []byte, seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderInterFrameSize, refs *ReferenceState, params EncoderFilmGrainParams) ([]byte, error) {
	return internalencoder.AppendFilmGrainParamsPayload(dst, seq, prefix, size, refs, params)
}

func EncoderLowOverheadFrameHeaderIntraOBUSize(seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderIntraFrameSize) (int, error) {
	return internalencoder.LowOverheadFrameHeaderIntraOBUSize(seq, prefix, size)
}

func AppendEncoderLowOverheadFrameHeaderIntraOBU(dst []byte, seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderIntraFrameSize) ([]byte, error) {
	return internalencoder.AppendLowOverheadFrameHeaderIntraOBU(dst, seq, prefix, size)
}

func EncoderLowOverheadFrameHeaderInterOBUSize(seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderInterFrameSize) (int, error) {
	return internalencoder.LowOverheadFrameHeaderInterOBUSize(seq, prefix, size)
}

func AppendEncoderLowOverheadFrameHeaderInterOBU(dst []byte, seq EncoderSequenceHeader, prefix EncoderFrameHeaderPrefix, size EncoderInterFrameSize) ([]byte, error) {
	return internalencoder.AppendLowOverheadFrameHeaderInterOBU(dst, seq, prefix, size)
}

func EncoderLowOverheadInterHeaderFrameOBUSize(header EncoderInterHeaderFrame) (int, error) {
	return internalencoder.LowOverheadInterHeaderFrameOBUSize(header)
}

func AppendEncoderLowOverheadInterHeaderFrameOBU(dst []byte, header EncoderInterHeaderFrame) ([]byte, error) {
	return internalencoder.AppendLowOverheadInterHeaderFrameOBU(dst, header)
}

func EncoderLowOverheadWebRTCDeltaHeaderTemporalUnitSize(unit EncoderWebRTCDeltaFrameTemporalUnit) (int, error) {
	return internalencoder.LowOverheadWebRTCDeltaHeaderTemporalUnitSize(unit)
}

func AppendEncoderLowOverheadWebRTCDeltaHeaderTemporalUnit(dst []byte, unit EncoderWebRTCDeltaFrameTemporalUnit) ([]byte, error) {
	return internalencoder.AppendLowOverheadWebRTCDeltaHeaderTemporalUnit(dst, unit)
}

func EncoderLowOverheadWebRTCDeltaHeaderTemporalUnitForStateSize(config EncoderConfig, state EncoderWebRTCState) (int, EncoderWebRTCDeltaFrameTemporalUnit, EncoderWebRTCState, error) {
	return internalencoder.LowOverheadWebRTCDeltaHeaderTemporalUnitForStateSize(config, state)
}

func AppendEncoderLowOverheadWebRTCDeltaHeaderTemporalUnitForState(dst []byte, config EncoderConfig, state EncoderWebRTCState) ([]byte, EncoderWebRTCDeltaFrameTemporalUnit, EncoderWebRTCState, error) {
	return internalencoder.AppendLowOverheadWebRTCDeltaHeaderTemporalUnitForState(dst, config, state)
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

func EncoderLowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(config EncoderConfig, state EncoderWebRTCState, forceKeyFrame bool) (int, EncoderWebRTCPictureTemporalUnit, EncoderWebRTCState, error) {
	return internalencoder.LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(config, state, forceKeyFrame)
}

func AppendEncoderLowOverheadWebRTCPictureHeaderTemporalUnitForState(dst []byte, config EncoderConfig, state EncoderWebRTCState, forceKeyFrame bool) ([]byte, EncoderWebRTCPictureTemporalUnit, EncoderWebRTCState, error) {
	return internalencoder.AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState(dst, config, state, forceKeyFrame)
}
