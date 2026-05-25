package goav1

import internalloopfilter "github.com/thesyncim/goav1/internal/av1/loopfilter"

type LoopFilterModeDeltaClass = internalloopfilter.ModeDeltaClass
type LoopFilterLevelRequest = internalloopfilter.LevelRequest
type LoopFilterThresholds = internalloopfilter.Thresholds
type LoopFilterDeltaState = internalloopfilter.DeltaState
type LoopFilterEdgeRequest = internalloopfilter.FilterEdgeRequest
type LoopFilter4Request = internalloopfilter.Filter4Request
type LoopFilterBlockRequest = internalloopfilter.FilterBlockRequest
type LoopFilterResult = internalloopfilter.FilterResult

const (
	LoopFilterMaxLevel     = internalloopfilter.MaxLevel
	LoopFilterMaxSharpness = internalloopfilter.MaxSharpness
	LoopFilterDeltaCount   = internalloopfilter.DeltaCount

	LoopFilterModeDeltaClassZero   LoopFilterModeDeltaClass = internalloopfilter.ModeDeltaClassZero
	LoopFilterModeDeltaClassMotion LoopFilterModeDeltaClass = internalloopfilter.ModeDeltaClassMotion
)

var ErrLoopFilterInvalidFilter = internalloopfilter.ErrInvalidFilter

func LoopFilterClampLevel(v int) uint8 {
	return internalloopfilter.ClampLevel(v)
}

func LoopFilterBaseLevel(params LoopFilterParams, plane LoopFilterPlane, edge LoopFilterEdge) (uint8, error) {
	return internalloopfilter.BaseLevel(params, plane, edge)
}

func LoopFilterDeltaIndex(plane LoopFilterPlane, edge LoopFilterEdge) (int, error) {
	return internalloopfilter.DeltaIndex(plane, edge)
}

func SelectLoopFilterDelta(params DeltaParams, fromBase int8, multi [LoopFilterDeltaCount]int8, plane LoopFilterPlane, edge LoopFilterEdge) (int8, error) {
	return internalloopfilter.SelectDelta(params, fromBase, multi, plane, edge)
}

func LoopFilterSegmentDelta(seg SegmentationParams, segmentID int, plane LoopFilterPlane, edge LoopFilterEdge) (int16, error) {
	return internalloopfilter.SegmentDelta(seg, segmentID, plane, edge)
}

func ResolveLoopFilterLevel(params LoopFilterParams, seg SegmentationParams, req LoopFilterLevelRequest) (uint8, error) {
	return internalloopfilter.ResolveLevel(params, seg, req)
}

func LoopFilterThresholdsForLevel(level uint8, sharpness uint8) (LoopFilterThresholds, error) {
	return internalloopfilter.ThresholdsForLevel(level, sharpness)
}

func ResolveLoopFilterBlockLevel(params LoopFilterParams, seg SegmentationParams, delta DeltaParams, state LoopFilterDeltaState, req LoopFilterLevelRequest) (uint8, error) {
	return internalloopfilter.ResolveBlockLevel(params, seg, delta, state, req)
}

func ApplyLoopFilter4Edge(dst FramePlane, bytesPerSample int, bitDepth uint8, edge LoopFilterEdge, x int, y int, length int, thresholds LoopFilterThresholds) error {
	return internalloopfilter.Filter4Edge(dst, bytesPerSample, bitDepth, edge, x, y, length, thresholds)
}

func ApplyLoopFilter6Edge(dst FramePlane, bytesPerSample int, bitDepth uint8, edge LoopFilterEdge, x int, y int, length int, thresholds LoopFilterThresholds) error {
	return internalloopfilter.Filter6Edge(dst, bytesPerSample, bitDepth, edge, x, y, length, thresholds)
}

func ApplyLoopFilter8Edge(dst FramePlane, bytesPerSample int, bitDepth uint8, edge LoopFilterEdge, x int, y int, length int, thresholds LoopFilterThresholds) error {
	return internalloopfilter.Filter8Edge(dst, bytesPerSample, bitDepth, edge, x, y, length, thresholds)
}

func ApplyLoopFilter14Edge(dst FramePlane, bytesPerSample int, bitDepth uint8, edge LoopFilterEdge, x int, y int, length int, thresholds LoopFilterThresholds) error {
	return internalloopfilter.Filter14Edge(dst, bytesPerSample, bitDepth, edge, x, y, length, thresholds)
}

func ApplyLoopFilterEdgeByWidth(width int, dst FramePlane, bytesPerSample int, bitDepth uint8, edge LoopFilterEdge, x int, y int, length int, thresholds LoopFilterThresholds) error {
	return internalloopfilter.FilterEdgeByWidth(width, dst, bytesPerSample, bitDepth, edge, x, y, length, thresholds)
}

func ApplyLoopFilter4BlockEdge(dst FramePlane, bytesPerSample int, bitDepth uint8, params LoopFilterParams, seg SegmentationParams, delta DeltaParams, state LoopFilterDeltaState, req LoopFilter4Request) (LoopFilterResult, error) {
	return internalloopfilter.Filter4BlockEdge(dst, bytesPerSample, bitDepth, params, seg, delta, state, req)
}

func ApplyLoopFilter6BlockEdge(dst FramePlane, bytesPerSample int, bitDepth uint8, params LoopFilterParams, seg SegmentationParams, delta DeltaParams, state LoopFilterDeltaState, req LoopFilterEdgeRequest) (LoopFilterResult, error) {
	return internalloopfilter.Filter6BlockEdge(dst, bytesPerSample, bitDepth, params, seg, delta, state, req)
}

func ApplyLoopFilter8BlockEdge(dst FramePlane, bytesPerSample int, bitDepth uint8, params LoopFilterParams, seg SegmentationParams, delta DeltaParams, state LoopFilterDeltaState, req LoopFilterEdgeRequest) (LoopFilterResult, error) {
	return internalloopfilter.Filter8BlockEdge(dst, bytesPerSample, bitDepth, params, seg, delta, state, req)
}

func ApplyLoopFilter14BlockEdge(dst FramePlane, bytesPerSample int, bitDepth uint8, params LoopFilterParams, seg SegmentationParams, delta DeltaParams, state LoopFilterDeltaState, req LoopFilterEdgeRequest) (LoopFilterResult, error) {
	return internalloopfilter.Filter14BlockEdge(dst, bytesPerSample, bitDepth, params, seg, delta, state, req)
}

func ApplyLoopFilterBlockEdge(dst FramePlane, bytesPerSample int, bitDepth uint8, params LoopFilterParams, seg SegmentationParams, delta DeltaParams, state LoopFilterDeltaState, req LoopFilterBlockRequest) (LoopFilterResult, error) {
	return internalloopfilter.FilterBlockEdge(dst, bytesPerSample, bitDepth, params, seg, delta, state, req)
}
