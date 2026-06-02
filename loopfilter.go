package goav1

import internalloopfilter "github.com/thesyncim/goav1/internal/av1/loopfilter"

// LoopFilterModeDeltaClass selects between the AV1 zero-prediction and motion
// mode-delta classes used by the loop filter.
type LoopFilterModeDeltaClass = internalloopfilter.ModeDeltaClass

// LoopFilterLevelRequest carries the per-block information (segment id,
// reference frame, mode-delta class, plane, edge) needed to resolve the
// effective loop-filter level.
type LoopFilterLevelRequest = internalloopfilter.LevelRequest

// LoopFilterThresholds bundles the per-edge filter thresholds (limit, blimit,
// hev_thresh) derived from a loop-filter level.
type LoopFilterThresholds = internalloopfilter.Thresholds

// LoopFilterDeltaState caches the most recently applied per-block loop-filter
// deltas so the next block can apply incremental updates.
type LoopFilterDeltaState = internalloopfilter.DeltaState

// LoopFilterEdgeRequest carries the geometry and per-edge state required to
// invoke ApplyLoopFilter6/8/14BlockEdge.
//
// See also: LoopFilterResult.
type LoopFilterEdgeRequest = internalloopfilter.FilterEdgeRequest

// LoopFilter4Request is the LoopFilterEdgeRequest specialization for 4-tap
// edges (chroma at 4:2:0 and small block boundaries).
//
// See also: LoopFilterResult, ApplyLoopFilter4BlockEdge.
type LoopFilter4Request = internalloopfilter.Filter4Request

// LoopFilterBlockRequest carries the per-block parameters for the dispatched
// ApplyLoopFilterBlockEdge entry point.
//
// See also: LoopFilterResult, ApplyLoopFilterBlockEdge.
type LoopFilterBlockRequest = internalloopfilter.FilterBlockRequest

// LoopFilterResult reports whether a loop-filter edge was applied and at
// what effective level.
//
// See also: LoopFilterEdgeRequest, LoopFilterBlockRequest.
type LoopFilterResult = internalloopfilter.FilterResult

// Loop-filter constants.
//
//   - LoopFilterMaxLevel is the maximum filter level (63) defined by AV1.
//   - LoopFilterMaxSharpness is the maximum sharpness value (7).
//   - LoopFilterDeltaCount is the number of per-frame loop-filter deltas
//     (one per ref frame class).
//   - LoopFilterModeDeltaClassZero / Motion enumerate the mode-delta classes.
const (
	LoopFilterMaxLevel     = internalloopfilter.MaxLevel
	LoopFilterMaxSharpness = internalloopfilter.MaxSharpness
	LoopFilterDeltaCount   = internalloopfilter.DeltaCount

	LoopFilterModeDeltaClassZero   LoopFilterModeDeltaClass = internalloopfilter.ModeDeltaClassZero
	LoopFilterModeDeltaClassMotion LoopFilterModeDeltaClass = internalloopfilter.ModeDeltaClassMotion
)

// ErrLoopFilterInvalidFilter is returned by the loop-filter helpers when the
// requested parameters fall outside the AV1 spec range.
var ErrLoopFilterInvalidFilter = internalloopfilter.ErrInvalidFilter

// LoopFilterClampLevel clamps v into the [0, LoopFilterMaxLevel] range used
// by the AV1 loop filter.
func LoopFilterClampLevel(v int) uint8 {
	return internalloopfilter.ClampLevel(v)
}

// LoopFilterBaseLevel returns the base (pre-delta) loop-filter level for the
// (plane, edge) combination according to the frame loop-filter parameters.
func LoopFilterBaseLevel(params LoopFilterParams, plane LoopFilterPlane, edge LoopFilterEdge) (uint8, error) {
	return internalloopfilter.BaseLevel(params, plane, edge)
}

// LoopFilterDeltaIndex returns the per-frame loop-filter delta index for the
// (plane, edge) combination.
func LoopFilterDeltaIndex(plane LoopFilterPlane, edge LoopFilterEdge) (int, error) {
	return internalloopfilter.DeltaIndex(plane, edge)
}

// SelectLoopFilterDelta picks the per-block delta to apply on top of fromBase
// using the AV1 multi-delta and mode-delta tables.
func SelectLoopFilterDelta(params DeltaParams, fromBase int8, multi [LoopFilterDeltaCount]int8, plane LoopFilterPlane, edge LoopFilterEdge) (int8, error) {
	return internalloopfilter.SelectDelta(params, fromBase, multi, plane, edge)
}

// LoopFilterSegmentDelta returns the per-segment delta (in level units) for
// the (segmentID, plane, edge) combination from seg.
func LoopFilterSegmentDelta(seg SegmentationParams, segmentID int, plane LoopFilterPlane, edge LoopFilterEdge) (int16, error) {
	return internalloopfilter.SegmentDelta(seg, segmentID, plane, edge)
}

// ResolveLoopFilterLevel resolves the effective loop-filter level for req by
// combining the frame parameters and per-segment overrides.
func ResolveLoopFilterLevel(params LoopFilterParams, seg SegmentationParams, req LoopFilterLevelRequest) (uint8, error) {
	return internalloopfilter.ResolveLevel(params, seg, req)
}

// LoopFilterThresholdsForLevel returns the (limit, blimit, hev_thresh)
// triplet implied by the AV1 spec for a given filter level and sharpness.
func LoopFilterThresholdsForLevel(level uint8, sharpness uint8) (LoopFilterThresholds, error) {
	return internalloopfilter.ThresholdsForLevel(level, sharpness)
}

// ResolveLoopFilterBlockLevel resolves the per-block loop-filter level taking
// the previous block's LoopFilterDeltaState into account.
func ResolveLoopFilterBlockLevel(params LoopFilterParams, seg SegmentationParams, delta DeltaParams, state LoopFilterDeltaState, req LoopFilterLevelRequest) (uint8, error) {
	return internalloopfilter.ResolveBlockLevel(params, seg, delta, state, req)
}

// ApplyLoopFilter4Edge applies the 4-tap loop filter to one edge in dst.
func ApplyLoopFilter4Edge(dst FramePlane, bytesPerSample int, bitDepth uint8, edge LoopFilterEdge, x int, y int, length int, thresholds LoopFilterThresholds) error {
	return internalloopfilter.Filter4Edge(dst, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), thresholds)
}

// ApplyLoopFilter6Edge applies the 6-tap loop filter to one edge in dst.
func ApplyLoopFilter6Edge(dst FramePlane, bytesPerSample int, bitDepth uint8, edge LoopFilterEdge, x int, y int, length int, thresholds LoopFilterThresholds) error {
	return internalloopfilter.Filter6Edge(dst, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), thresholds)
}

// ApplyLoopFilter8Edge applies the 8-tap loop filter to one edge in dst.
func ApplyLoopFilter8Edge(dst FramePlane, bytesPerSample int, bitDepth uint8, edge LoopFilterEdge, x int, y int, length int, thresholds LoopFilterThresholds) error {
	return internalloopfilter.Filter8Edge(dst, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), thresholds)
}

// ApplyLoopFilter14Edge applies the 14-tap loop filter to one edge in dst.
func ApplyLoopFilter14Edge(dst FramePlane, bytesPerSample int, bitDepth uint8, edge LoopFilterEdge, x int, y int, length int, thresholds LoopFilterThresholds) error {
	return internalloopfilter.Filter14Edge(dst, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), thresholds)
}

// ApplyLoopFilterEdgeByWidth dispatches to ApplyLoopFilter4/6/8/14Edge based
// on width (4, 6, 8, or 14).
func ApplyLoopFilterEdgeByWidth(width int, dst FramePlane, bytesPerSample int, bitDepth uint8, edge LoopFilterEdge, x int, y int, length int, thresholds LoopFilterThresholds) error {
	return internalloopfilter.FilterEdgeByWidth(int32(width), dst, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), thresholds)
}

// ApplyLoopFilter4BlockEdge resolves the 4-tap edge filter parameters for
// req and applies them to dst, returning the resolved level.
func ApplyLoopFilter4BlockEdge(dst FramePlane, bytesPerSample int, bitDepth uint8, params LoopFilterParams, seg SegmentationParams, delta DeltaParams, state LoopFilterDeltaState, req LoopFilter4Request) (LoopFilterResult, error) {
	return internalloopfilter.Filter4BlockEdge(dst, bytesPerSample, bitDepth, params, seg, delta, state, req)
}

// ApplyLoopFilter6BlockEdge resolves the 6-tap edge filter parameters for
// req and applies them to dst, returning the resolved level.
func ApplyLoopFilter6BlockEdge(dst FramePlane, bytesPerSample int, bitDepth uint8, params LoopFilterParams, seg SegmentationParams, delta DeltaParams, state LoopFilterDeltaState, req LoopFilterEdgeRequest) (LoopFilterResult, error) {
	return internalloopfilter.Filter6BlockEdge(dst, bytesPerSample, bitDepth, params, seg, delta, state, req)
}

// ApplyLoopFilter8BlockEdge resolves the 8-tap edge filter parameters for
// req and applies them to dst, returning the resolved level.
func ApplyLoopFilter8BlockEdge(dst FramePlane, bytesPerSample int, bitDepth uint8, params LoopFilterParams, seg SegmentationParams, delta DeltaParams, state LoopFilterDeltaState, req LoopFilterEdgeRequest) (LoopFilterResult, error) {
	return internalloopfilter.Filter8BlockEdge(dst, bytesPerSample, bitDepth, params, seg, delta, state, req)
}

// ApplyLoopFilter14BlockEdge resolves the 14-tap edge filter parameters for
// req and applies them to dst, returning the resolved level.
func ApplyLoopFilter14BlockEdge(dst FramePlane, bytesPerSample int, bitDepth uint8, params LoopFilterParams, seg SegmentationParams, delta DeltaParams, state LoopFilterDeltaState, req LoopFilterEdgeRequest) (LoopFilterResult, error) {
	return internalloopfilter.Filter14BlockEdge(dst, bytesPerSample, bitDepth, params, seg, delta, state, req)
}

// ApplyLoopFilterBlockEdge dispatches to the appropriate
// ApplyLoopFilter[N]BlockEdge based on the edge width encoded in req.
func ApplyLoopFilterBlockEdge(dst FramePlane, bytesPerSample int, bitDepth uint8, params LoopFilterParams, seg SegmentationParams, delta DeltaParams, state LoopFilterDeltaState, req LoopFilterBlockRequest) (LoopFilterResult, error) {
	return internalloopfilter.FilterBlockEdge(dst, bytesPerSample, bitDepth, params, seg, delta, state, req)
}
