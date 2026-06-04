// Ported from libaom: av1/common/av1_loopfilter.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package loopfilter

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// DeltaState is the block-local delta-lf state carried by tile decode.
type DeltaState struct {
	FromBase int8
	Multi    [DeltaCount]int8
}

// FilterEdgeRequest describes one deblocking edge in plane coordinates.
type FilterEdgeRequest struct {
	LevelRequest

	X      uint16
	Y      uint16
	Length uint16
}

// Filter4Request describes one narrow deblocking edge in plane coordinates.
type Filter4Request = FilterEdgeRequest

// FilterBlockRequest describes one resolved-width deblocking edge.
type FilterBlockRequest struct {
	FilterEdgeRequest

	Width uint8
}

// FilterResult reports the resolved setup used for one edge filter call.
type FilterResult struct {
	Level      uint8
	Thresholds Thresholds
	Applied    bool
}

// ResolveBlockLevel selects the tile-local delta-lf value and resolves the
// final AV1 loop-filter level for one block edge.
func ResolveBlockLevel(params parser.LoopFilterParams, seg parser.SegmentationParams, delta parser.DeltaParams, state DeltaState, req LevelRequest) (uint8, error) {
	deltaLF, err := SelectDelta(delta, state.FromBase, state.Multi, req.Plane, req.Edge)
	if err != nil {
		return 0, err
	}
	req.DeltaLF = deltaLF
	return ResolveLevel(params, seg, req)
}

// Filter4BlockEdge resolves loop-filter level and thresholds, then applies
// AV1's narrow four-sample deblocking filter when the resolved level is nonzero.
func Filter4BlockEdge(dst frame.Plane, bytesPerSample int, bitDepth uint8, params parser.LoopFilterParams, seg parser.SegmentationParams, delta parser.DeltaParams, state DeltaState, req Filter4Request) (FilterResult, error) {
	return filterBlockEdgeWidth(dst, bytesPerSample, bitDepth, params, seg, delta, state, FilterBlockRequest{
		FilterEdgeRequest: req,
		Width:             4,
	})
}

// Filter6BlockEdge resolves loop-filter level and thresholds, then applies
// AV1's six-sample deblocking filter when the resolved level is nonzero.
func Filter6BlockEdge(dst frame.Plane, bytesPerSample int, bitDepth uint8, params parser.LoopFilterParams, seg parser.SegmentationParams, delta parser.DeltaParams, state DeltaState, req FilterEdgeRequest) (FilterResult, error) {
	return filterBlockEdgeWidth(dst, bytesPerSample, bitDepth, params, seg, delta, state, FilterBlockRequest{
		FilterEdgeRequest: req,
		Width:             6,
	})
}

// Filter8BlockEdge resolves loop-filter level and thresholds, then applies
// AV1's eight-sample deblocking filter when the resolved level is nonzero.
func Filter8BlockEdge(dst frame.Plane, bytesPerSample int, bitDepth uint8, params parser.LoopFilterParams, seg parser.SegmentationParams, delta parser.DeltaParams, state DeltaState, req FilterEdgeRequest) (FilterResult, error) {
	return filterBlockEdgeWidth(dst, bytesPerSample, bitDepth, params, seg, delta, state, FilterBlockRequest{
		FilterEdgeRequest: req,
		Width:             8,
	})
}

// Filter14BlockEdge resolves loop-filter level and thresholds, then applies
// AV1's fourteen-sample deblocking filter when the resolved level is nonzero.
func Filter14BlockEdge(dst frame.Plane, bytesPerSample int, bitDepth uint8, params parser.LoopFilterParams, seg parser.SegmentationParams, delta parser.DeltaParams, state DeltaState, req FilterEdgeRequest) (FilterResult, error) {
	return filterBlockEdgeWidth(dst, bytesPerSample, bitDepth, params, seg, delta, state, FilterBlockRequest{
		FilterEdgeRequest: req,
		Width:             14,
	})
}

// FilterBlockEdge resolves loop-filter level and thresholds, then dispatches to
// the requested AV1 deblocking width when the resolved level is nonzero.
func FilterBlockEdge(dst frame.Plane, bytesPerSample int, bitDepth uint8, params parser.LoopFilterParams, seg parser.SegmentationParams, delta parser.DeltaParams, state DeltaState, req FilterBlockRequest) (FilterResult, error) {
	return filterBlockEdgeWidth(dst, bytesPerSample, bitDepth, params, seg, delta, state, req)
}

func filterBlockEdgeWidth(dst frame.Plane, bytesPerSample int, bitDepth uint8, params parser.LoopFilterParams, seg parser.SegmentationParams, delta parser.DeltaParams, state DeltaState, req FilterBlockRequest) (FilterResult, error) {
	if !validFilterWidth(req.Width) {
		return FilterResult{}, ErrInvalidFilter
	}
	if params.Sharpness > MaxSharpness {
		return FilterResult{}, ErrInvalidFilter
	}
	level, err := ResolveBlockLevel(params, seg, delta, state, req.LevelRequest)
	if err != nil {
		return FilterResult{}, err
	}
	if level == 0 {
		return FilterResult{}, nil
	}
	thresholds, err := ThresholdsForLevel(level, params.Sharpness)
	if err != nil {
		return FilterResult{}, err
	}
	if err := FilterEdgeByWidth(req.Width, dst, bytesPerSample, bitDepth, req.Edge, int32(req.X), int32(req.Y), int32(req.Length), thresholds); err != nil {
		return FilterResult{}, err
	}
	return FilterResult{
		Level:      level,
		Thresholds: thresholds,
		Applied:    true,
	}, nil
}

// FilterEdgeByWidth applies one AV1 deblocking edge with precomputed thresholds.
func FilterEdgeByWidth(width uint8, dst frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int32, y int32, length int32, thresholds Thresholds) error {
	switch width {
	case 4:
		return Filter4Edge(dst, bytesPerSample, bitDepth, edge, x, y, length, thresholds)
	case 6:
		return Filter6Edge(dst, bytesPerSample, bitDepth, edge, x, y, length, thresholds)
	case 8:
		return Filter8Edge(dst, bytesPerSample, bitDepth, edge, x, y, length, thresholds)
	case 14:
		return Filter14Edge(dst, bytesPerSample, bitDepth, edge, x, y, length, thresholds)
	default:
		return ErrInvalidFilter
	}
}

func validFilterWidth(width uint8) bool {
	return width == 4 || width == 6 || width == 8 || width == 14
}
