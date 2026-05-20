package goav1

import internalparser "github.com/thesyncim/goav1/internal/av1/parser"

type SequenceHeader = internalparser.SequenceHeader
type TimingInfo = internalparser.TimingInfo
type DecoderModelInfo = internalparser.DecoderModelInfo
type OperatingPoint = internalparser.OperatingPoint
type ColorConfig = internalparser.ColorConfig
type FrameType = internalparser.FrameType
type FrameHeaderPrefix = internalparser.FrameHeaderPrefix
type FrameSize = internalparser.FrameSize
type TileInfo = internalparser.TileInfo
type QuantizationParams = internalparser.QuantizationParams
type SegmentationParams = internalparser.SegmentationParams
type SegmentationData = internalparser.SegmentationData
type SegmentData = internalparser.SegmentData
type DeltaParams = internalparser.DeltaParams
type LoopFilterParams = internalparser.LoopFilterParams
type LoopFilterDeltas = internalparser.LoopFilterDeltas
type CDEFParams = internalparser.CDEFParams
type RestorationParams = internalparser.RestorationParams
type RestorationType = internalparser.RestorationType
type TransformReferenceParams = internalparser.TransformReferenceParams
type TransformMode = internalparser.TransformMode
type ReferenceMode = internalparser.ReferenceMode
type SkipModeParams = internalparser.SkipModeParams
type FrameModeParams = internalparser.FrameModeParams
type GlobalMotionParams = internalparser.GlobalMotionParams
type GlobalMotionType = internalparser.GlobalMotionType
type WarpedMotionParams = internalparser.WarpedMotionParams
type InterpolationFilter = internalparser.InterpolationFilter
type ReferenceFrame = internalparser.ReferenceFrame
type ReferenceState = internalparser.ReferenceState

const (
	FrameTypeKey         = internalparser.FrameTypeKey
	FrameTypeInter       = internalparser.FrameTypeInter
	FrameTypeIntraOnly   = internalparser.FrameTypeIntraOnly
	FrameTypeSwitch      = internalparser.FrameTypeSwitch
	PrimaryRefNone       = internalparser.PrimaryRefNone
	RefFrames            = internalparser.RefFrames
	InterRefsPerFrame    = internalparser.InterRefsPerFrame
	MaxTileRows          = internalparser.MaxTileRows
	MaxTileCols          = internalparser.MaxTileCols
	MaxSegments          = internalparser.MaxSegments
	LoopFilterModeDeltas = internalparser.LoopFilterModeDeltas
	MaxCDEFStrengths     = internalparser.MaxCDEFStrengths
	RestorationUnitMax   = internalparser.RestorationUnitMax

	InterpolationEightTap   = internalparser.InterpolationEightTap
	InterpolationSmooth     = internalparser.InterpolationSmooth
	InterpolationSharp      = internalparser.InterpolationSharp
	InterpolationBilinear   = internalparser.InterpolationBilinear
	InterpolationSwitchable = internalparser.InterpolationSwitchable

	RestorationNone       = internalparser.RestorationNone
	RestorationSwitchable = internalparser.RestorationSwitchable
	RestorationWiener     = internalparser.RestorationWiener
	RestorationSGRProj    = internalparser.RestorationSGRProj

	TransformMode4x4Only    = internalparser.TransformMode4x4Only
	TransformModeLargest    = internalparser.TransformModeLargest
	TransformModeSwitchable = internalparser.TransformModeSwitchable

	ReferenceModeSingle   = internalparser.ReferenceModeSingle
	ReferenceModeCompound = internalparser.ReferenceModeCompound
	ReferenceModeSelect   = internalparser.ReferenceModeSelect

	GlobalMotionIdentity    = internalparser.GlobalMotionIdentity
	GlobalMotionTranslation = internalparser.GlobalMotionTranslation
	GlobalMotionRotZoom     = internalparser.GlobalMotionRotZoom
	GlobalMotionAffine      = internalparser.GlobalMotionAffine
)

var (
	ErrInvalidSequenceHeader = internalparser.ErrInvalidSequenceHeader
	ErrInvalidFrameHeader    = internalparser.ErrInvalidFrameHeader
	ErrReferenceFrameNeeded  = internalparser.ErrReferenceFrameNeeded
)

func ParseSequenceHeader(payload []byte) (SequenceHeader, error) {
	return internalparser.ParseSequenceHeader(payload)
}

func ParseFrameHeaderPrefix(payload []byte, sequence SequenceHeader) (FrameHeaderPrefix, error) {
	return internalparser.ParseFrameHeaderPrefix(payload, sequence)
}

func ParseIntraFrameSize(payload []byte, sequence SequenceHeader, prefix FrameHeaderPrefix, temporalID uint8, spatialID uint8) (FrameSize, error) {
	return internalparser.ParseIntraFrameSize(payload, sequence, prefix, temporalID, spatialID)
}

func ParseFrameSize(payload []byte, sequence SequenceHeader, prefix FrameHeaderPrefix, references *ReferenceState, temporalID uint8, spatialID uint8) (FrameSize, error) {
	return internalparser.ParseFrameSize(payload, sequence, prefix, references, temporalID, spatialID)
}

func ParseTileInfo(payload []byte, sequence SequenceHeader, prefix FrameHeaderPrefix, size FrameSize) (TileInfo, error) {
	return internalparser.ParseTileInfo(payload, sequence, prefix, size)
}

func ParseQuantizationParams(payload []byte, sequence SequenceHeader, tiles TileInfo) (QuantizationParams, error) {
	return internalparser.ParseQuantizationParams(payload, sequence, tiles)
}

func ParseSegmentationParams(payload []byte, prefix FrameHeaderPrefix, quant QuantizationParams, previous *SegmentationData) (SegmentationParams, error) {
	return internalparser.ParseSegmentationParams(payload, prefix, quant, previous)
}

func ParseDeltaParams(payload []byte, size FrameSize, quant QuantizationParams, seg SegmentationParams) (DeltaParams, error) {
	return internalparser.ParseDeltaParams(payload, size, quant, seg)
}

func ParseLoopFilterParams(payload []byte, sequence SequenceHeader, prefix FrameHeaderPrefix, size FrameSize, seg SegmentationParams, delta DeltaParams, previous *LoopFilterDeltas) (LoopFilterParams, error) {
	return internalparser.ParseLoopFilterParams(payload, sequence, prefix, size, seg, delta, previous)
}

func ParseCDEFParams(payload []byte, sequence SequenceHeader, size FrameSize, seg SegmentationParams, lf LoopFilterParams) (CDEFParams, error) {
	return internalparser.ParseCDEFParams(payload, sequence, size, seg, lf)
}

func ParseRestorationParams(payload []byte, sequence SequenceHeader, size FrameSize, seg SegmentationParams, cdef CDEFParams) (RestorationParams, error) {
	return internalparser.ParseRestorationParams(payload, sequence, size, seg, cdef)
}

func ParseTransformReferenceParams(payload []byte, prefix FrameHeaderPrefix, seg SegmentationParams, restoration RestorationParams) (TransformReferenceParams, error) {
	return internalparser.ParseTransformReferenceParams(payload, prefix, seg, restoration)
}

func ParseSkipModeParams(payload []byte, sequence SequenceHeader, prefix FrameHeaderPrefix, size FrameSize, references *ReferenceState, transformRef TransformReferenceParams) (SkipModeParams, error) {
	return internalparser.ParseSkipModeParams(payload, sequence, prefix, size, references, transformRef)
}

func ParseFrameModeParams(payload []byte, sequence SequenceHeader, prefix FrameHeaderPrefix, skip SkipModeParams) (FrameModeParams, error) {
	return internalparser.ParseFrameModeParams(payload, sequence, prefix, skip)
}

func ParseGlobalMotionParams(payload []byte, prefix FrameHeaderPrefix, size FrameSize, tiles TileInfo, references *ReferenceState, frameMode FrameModeParams) (GlobalMotionParams, error) {
	return internalparser.ParseGlobalMotionParams(payload, prefix, size, tiles, references, frameMode)
}
