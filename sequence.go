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
type InterpolationFilter = internalparser.InterpolationFilter
type ReferenceFrame = internalparser.ReferenceFrame
type ReferenceState = internalparser.ReferenceState

const (
	FrameTypeKey       = internalparser.FrameTypeKey
	FrameTypeInter     = internalparser.FrameTypeInter
	FrameTypeIntraOnly = internalparser.FrameTypeIntraOnly
	FrameTypeSwitch    = internalparser.FrameTypeSwitch
	PrimaryRefNone     = internalparser.PrimaryRefNone
	RefFrames          = internalparser.RefFrames
	InterRefsPerFrame  = internalparser.InterRefsPerFrame
	MaxTileRows        = internalparser.MaxTileRows
	MaxTileCols        = internalparser.MaxTileCols

	InterpolationEightTap   = internalparser.InterpolationEightTap
	InterpolationSmooth     = internalparser.InterpolationSmooth
	InterpolationSharp      = internalparser.InterpolationSharp
	InterpolationBilinear   = internalparser.InterpolationBilinear
	InterpolationSwitchable = internalparser.InterpolationSwitchable
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
