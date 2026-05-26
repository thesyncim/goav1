package goav1

import internalparser "github.com/thesyncim/goav1/internal/av1/parser"

// SequenceHeader is the parsed AV1 sequence_header_obu(): profile, level,
// color config, operating points, decoder model, and feature enable bits.
// It is the long-lived per-coded-video-sequence parameter set.
type SequenceHeader = internalparser.SequenceHeader

// TimingInfo holds the AV1 sequence-level timing information (numerator/
// denominator, equal-picture-interval flag, num_ticks_per_picture).
type TimingInfo = internalparser.TimingInfo

// DecoderModelInfo is the AV1 sequence-level decoder model description used
// for hypothetical-reference-decoder (HRD) computations.
type DecoderModelInfo = internalparser.DecoderModelInfo

// OperatingPoint is one entry in the AV1 sequence-header operating-point
// list, describing the scalable layer combinations a decoder may select.
type OperatingPoint = internalparser.OperatingPoint

// AV1 scalability constants mirroring the operating_point_idc() syntax
// element (AV1 spec section 6.4.1).
const (
	// MaxOperatingPointTemporalLayers is the maximum temporal_id count a
	// single operating_point_idc value can describe.
	MaxOperatingPointTemporalLayers = internalparser.MaxOperatingPointTemporalLayers

	// MaxOperatingPointSpatialLayers is the maximum spatial_id count a
	// single operating_point_idc value can describe.
	MaxOperatingPointSpatialLayers = internalparser.MaxOperatingPointSpatialLayers

	// OperatingPointIDCBits is the width of the operating_point_idc
	// syntax element in bits.
	OperatingPointIDCBits = internalparser.OperatingPointIDCBits

	// OperatingPointIDCAnyLayer is the operating_point_idc sentinel that
	// selects every (temporal_id, spatial_id) combination.
	OperatingPointIDCAnyLayer = internalparser.OperatingPointIDCAnyLayer
)

// OperatingPointIDCMatches reports whether a layer-tagged OBU with the
// supplied (temporal_id, spatial_id) participates in the operating point
// described by idc. idc == OperatingPointIDCAnyLayer matches every layer;
// otherwise the OBU participates only if both layer-bit positions are set
// in idc, matching libaom's is_obu_in_current_operating_point().
func OperatingPointIDCMatches(idc uint16, temporalID uint8, spatialID uint8) bool {
	return internalparser.OperatingPointIDCMatches(idc, temporalID, spatialID)
}

// OperatingPointTemporalLayerCount reports the number of distinct
// temporal layers selected by idc per AV1 spec section 6.4.1. idc == 0
// yields 1.
func OperatingPointTemporalLayerCount(idc uint16) uint8 {
	return internalparser.OperatingPointTemporalLayerCount(idc)
}

// OperatingPointSpatialLayerCount reports the number of distinct spatial
// layers selected by idc per AV1 spec section 6.4.1. idc == 0 yields 1.
func OperatingPointSpatialLayerCount(idc uint16) uint8 {
	return internalparser.OperatingPointSpatialLayerCount(idc)
}

// SelectOperatingPoint returns the first index into
// seq.OperatingPoints[:seq.OperatingPointsCount] whose IDC covers the
// supplied (temporal_id, spatial_id), matching libaom's "first match wins"
// convention. ok is false when no operating point covers the layer pair.
func SelectOperatingPoint(seq SequenceHeader, temporalID uint8, spatialID uint8) (index int, ok bool) {
	return internalparser.SelectOperatingPoint(seq, temporalID, spatialID)
}

// ColorConfig is the AV1 sequence-level color description: bit depth,
// monochrome flag, chroma subsampling, color primaries, transfer function,
// matrix coefficients, and full-range flag.
type ColorConfig = internalparser.ColorConfig

// FrameType is the AV1 frame_type code (key, inter, intra-only, switch).
type FrameType = internalparser.FrameType

// FrameHeaderPrefix is the parsed AV1 uncompressed-frame-header prefix up to
// and including show_existing/ref_frame_idx. Subsequent parse helpers consume
// the rest of the frame header relative to this prefix.
type FrameHeaderPrefix = internalparser.FrameHeaderPrefix

// FrameSize holds the AV1 frame_size() result: coded, upscaled, and render
// dimensions in samples.
type FrameSize = internalparser.FrameSize

// TileInfo holds the AV1 tile_info(): tile column/row counts, per-tile sizes,
// and context-update tile id.
type TileInfo = internalparser.TileInfo

// QuantizationParams holds the AV1 quantization_params(): base q-index, per-
// plane DC/AC deltas, qmatrix flag and per-plane qmatrix levels, lossless
// flag, and CDEF/loop-filter applicability flags derived from the q-index.
type QuantizationParams = internalparser.QuantizationParams

// SegmentationParams holds the AV1 segmentation_params() fields: enable,
// update_map/data flags, and the active per-segment feature data.
type SegmentationParams = internalparser.SegmentationParams

// SegmentationData is the per-segment feature table referenced by
// SegmentationParams.
type SegmentationData = internalparser.SegmentationData

// SegmentData is one row of SegmentationData: feature mask plus the per-
// feature signed delta value.
type SegmentData = internalparser.SegmentData

// DeltaParams holds the AV1 delta_q/delta_lf control flags parsed after
// segmentation.
type DeltaParams = internalparser.DeltaParams

// LoopFilterParams holds the AV1 loop_filter_params(): per-plane filter
// levels, sharpness, mode-deltas state, and ref-deltas state.
type LoopFilterParams = internalparser.LoopFilterParams

// LoopFilterDeltas holds the persistent loop-filter ref/mode delta tables
// carried across frames.
type LoopFilterDeltas = internalparser.LoopFilterDeltas

// CDEFParams holds the AV1 cdef_params(): bits/strengths for primary and
// secondary CDEF filters per CDEF region.
type CDEFParams = internalparser.CDEFParams

// RestorationParams holds the AV1 lr_params(): per-plane restoration type
// and unit-size.
type RestorationParams = internalparser.RestorationParams

// RestorationType selects between the three restoration modes (none, Wiener,
// SGR proj) and the switchable mode used during entropy decode.
type RestorationType = internalparser.RestorationType

// TransformReferenceParams holds the AV1 read_tx_mode and reference-mode
// fields parsed after restoration.
type TransformReferenceParams = internalparser.TransformReferenceParams

// TransformMode is the AV1 frame-level tx_mode value (4x4-only, largest,
// switchable).
type TransformMode = internalparser.TransformMode

// ReferenceMode is the AV1 frame-level reference_mode value (single,
// compound, select).
type ReferenceMode = internalparser.ReferenceMode

// SkipModeParams holds the AV1 skip-mode eligibility, allowed flag, and
// chosen reference indices.
type SkipModeParams = internalparser.SkipModeParams

// FrameModeParams bundles the AV1 frame-level warped-motion and reduced-tx-set
// flags.
type FrameModeParams = internalparser.FrameModeParams

// GlobalMotionParams holds the per-reference global-motion model selections
// and matrices for the current frame.
type GlobalMotionParams = internalparser.GlobalMotionParams

// GlobalMotionType selects between identity, translation, rotzoom, and
// affine global-motion models.
type GlobalMotionType = internalparser.GlobalMotionType

// WarpedMotionParams holds one warped-motion model (matrix and validity).
type WarpedMotionParams = internalparser.WarpedMotionParams

// FilmGrainParams holds the AV1 film_grain_params(): scaling points, AR
// coefficients, scaling shift, seed, overlap and clipping flags.
type FilmGrainParams = internalparser.FilmGrainParams

// TileGroup is the parsed AV1 tile_group_obu() header: start/end tile indices
// and per-tile sizes (excluding payload).
type TileGroup = internalparser.TileGroup

// TileSpan is one decoded tile span: tile index, byte offset, and length
// inside the tile-group payload.
type TileSpan = internalparser.TileSpan

// InterpolationFilter is the AV1 sequence/frame interpolation_filter value
// (eight-tap, smooth, sharp, bilinear, switchable).
type InterpolationFilter = internalparser.InterpolationFilter

// ReferenceFrame is the per-slot AV1 reference-frame descriptor (frame id,
// size, order hint, color config) carried in ReferenceState.
type ReferenceFrame = internalparser.ReferenceFrame

// ReferenceState is the caller-owned eight-slot reference-frame table that
// the frame-header parsers read and update.
type ReferenceState = internalparser.ReferenceState

// AV1 sequence/frame-header constants. They mirror the corresponding
// specification limits and enumerations: frame types, primary-reference
// sentinel, reference-slot count, max tile/segment counts, loop-filter and
// CDEF/restoration limits, film-grain table sizes, and the interpolation,
// restoration, transform, reference, and global-motion mode enumerations.
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
	MaxFilmGrainYPoints  = internalparser.MaxFilmGrainYPoints
	MaxFilmGrainUVPoints = internalparser.MaxFilmGrainUVPoints
	MaxFilmGrainYCoeffs  = internalparser.MaxFilmGrainYCoeffs
	MaxFilmGrainUVCoeffs = internalparser.MaxFilmGrainUVCoeffs
	MaxTiles             = internalparser.MaxTiles

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

// Sequence/frame-header parse errors. Each Err* value is returned by the
// corresponding Parse* helper when the input violates the AV1 syntax or
// constraint rules.
var (
	ErrInvalidSequenceHeader = internalparser.ErrInvalidSequenceHeader
	ErrInvalidFrameHeader    = internalparser.ErrInvalidFrameHeader
	ErrInvalidTileGroup      = internalparser.ErrInvalidTileGroup
	ErrReferenceFrameNeeded  = internalparser.ErrReferenceFrameNeeded
)

// ParseSequenceHeader parses an AV1 sequence_header_obu() from payload.
func ParseSequenceHeader(payload []byte) (SequenceHeader, error) {
	return internalparser.ParseSequenceHeader(payload)
}

// ParseFrameHeaderPrefix parses the AV1 uncompressed-header prefix
// (show_existing_frame / show_frame, frame_type, ref_order_hint, etc.) from
// payload. Subsequent ParseFrame* helpers consume the rest of the frame
// header relative to this prefix.
func ParseFrameHeaderPrefix(payload []byte, sequence SequenceHeader) (FrameHeaderPrefix, error) {
	return internalparser.ParseFrameHeaderPrefix(payload, sequence)
}

// ParseIntraFrameSize parses the AV1 intra-frame frame_size() and
// render_size() syntax for key and intra-only frames.
func ParseIntraFrameSize(payload []byte, sequence SequenceHeader, prefix FrameHeaderPrefix, temporalID uint8, spatialID uint8) (FrameSize, error) {
	return internalparser.ParseIntraFrameSize(payload, sequence, prefix, temporalID, spatialID)
}

// ParseFrameSize parses the AV1 inter-frame frame_size_with_refs() syntax,
// resolving sizes from references when found_ref flags are set.
func ParseFrameSize(payload []byte, sequence SequenceHeader, prefix FrameHeaderPrefix, references *ReferenceState, temporalID uint8, spatialID uint8) (FrameSize, error) {
	return internalparser.ParseFrameSize(payload, sequence, prefix, references, temporalID, spatialID)
}

// ParseTileInfo parses the AV1 tile_info() syntax (column/row partitioning,
// context-update tile id).
func ParseTileInfo(payload []byte, sequence SequenceHeader, prefix FrameHeaderPrefix, size FrameSize) (TileInfo, error) {
	return internalparser.ParseTileInfo(payload, sequence, prefix, size)
}

// ParseQuantizationParams parses the AV1 quantization_params() syntax.
func ParseQuantizationParams(payload []byte, sequence SequenceHeader, tiles TileInfo) (QuantizationParams, error) {
	return internalparser.ParseQuantizationParams(payload, sequence, tiles)
}

// ParseSegmentationParams parses the AV1 segmentation_params() syntax,
// using previous to carry segmentation_data when update_data is 0.
func ParseSegmentationParams(payload []byte, prefix FrameHeaderPrefix, quant QuantizationParams, previous *SegmentationData) (SegmentationParams, error) {
	return internalparser.ParseSegmentationParams(payload, prefix, quant, previous)
}

// ParseDeltaParams parses the AV1 delta_q_params() and delta_lf_params()
// flags.
func ParseDeltaParams(payload []byte, size FrameSize, quant QuantizationParams, seg SegmentationParams) (DeltaParams, error) {
	return internalparser.ParseDeltaParams(payload, size, quant, seg)
}

// ParseLoopFilterParams parses the AV1 loop_filter_params() syntax, carrying
// over previous ref/mode deltas via previous.
func ParseLoopFilterParams(payload []byte, sequence SequenceHeader, prefix FrameHeaderPrefix, size FrameSize, seg SegmentationParams, delta DeltaParams, previous *LoopFilterDeltas) (LoopFilterParams, error) {
	return internalparser.ParseLoopFilterParams(payload, sequence, prefix, size, seg, delta, previous)
}

// ParseCDEFParams parses the AV1 cdef_params() syntax.
func ParseCDEFParams(payload []byte, sequence SequenceHeader, size FrameSize, seg SegmentationParams, lf LoopFilterParams) (CDEFParams, error) {
	return internalparser.ParseCDEFParams(payload, sequence, size, seg, lf)
}

// ParseRestorationParams parses the AV1 lr_params() syntax.
func ParseRestorationParams(payload []byte, sequence SequenceHeader, size FrameSize, seg SegmentationParams, cdef CDEFParams) (RestorationParams, error) {
	return internalparser.ParseRestorationParams(payload, sequence, size, seg, cdef)
}

// ParseTransformReferenceParams parses the AV1 read_tx_mode() and
// frame_reference_mode() syntax.
func ParseTransformReferenceParams(payload []byte, prefix FrameHeaderPrefix, seg SegmentationParams, restoration RestorationParams) (TransformReferenceParams, error) {
	return internalparser.ParseTransformReferenceParams(payload, prefix, seg, restoration)
}

// ParseSkipModeParams parses the AV1 skip_mode_params() syntax, resolving
// allowable reference pairs from references.
func ParseSkipModeParams(payload []byte, sequence SequenceHeader, prefix FrameHeaderPrefix, size FrameSize, references *ReferenceState, transformRef TransformReferenceParams) (SkipModeParams, error) {
	return internalparser.ParseSkipModeParams(payload, sequence, prefix, size, references, transformRef)
}

// ParseFrameModeParams parses the AV1 frame-level warped-motion and
// reduced-tx-set flags.
func ParseFrameModeParams(payload []byte, sequence SequenceHeader, prefix FrameHeaderPrefix, skip SkipModeParams) (FrameModeParams, error) {
	return internalparser.ParseFrameModeParams(payload, sequence, prefix, skip)
}

// ParseGlobalMotionParams parses the AV1 global_motion_params() syntax,
// using references to carry over previous global-motion warp parameters.
func ParseGlobalMotionParams(payload []byte, prefix FrameHeaderPrefix, size FrameSize, tiles TileInfo, references *ReferenceState, frameMode FrameModeParams) (GlobalMotionParams, error) {
	return internalparser.ParseGlobalMotionParams(payload, prefix, size, tiles, references, frameMode)
}

// ParseFilmGrainParams parses the AV1 film_grain_params() syntax, using
// references to carry over the previously decoded film-grain table when
// update_grain is 0.
func ParseFilmGrainParams(payload []byte, sequence SequenceHeader, prefix FrameHeaderPrefix, size FrameSize, references *ReferenceState, globalMotion GlobalMotionParams) (FilmGrainParams, error) {
	return internalparser.ParseFilmGrainParams(payload, sequence, prefix, size, references, globalMotion)
}

// ParseTileGroupHeader parses the AV1 tile_group_obu() header. startBits is
// the number of bits used to encode the tile start/end markers (derived from
// the active TileInfo); frameOBU is true when the tile-group is embedded in
// a frame-OBU rather than a standalone tile-group-OBU.
func ParseTileGroupHeader(payload []byte, tiles TileInfo, startBits int, expectedStart uint16, frameOBU bool) (TileGroup, error) {
	return internalparser.ParseTileGroupHeader(payload, tiles, startBits, expectedStart, frameOBU)
}

// SplitTileGroup populates spans with the (offset, length) of each tile
// payload inside the tile-group payload and returns the number of spans
// written.
func SplitTileGroup(payload []byte, tiles TileInfo, group TileGroup, spans []TileSpan) (int, error) {
	return internalparser.SplitTileGroup(payload, tiles, group, spans)
}
