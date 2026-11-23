package goav1

import (
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"

	internalloopfilter "github.com/thesyncim/goav1/internal/av1/loopfilter"
)

// loopFilterSegmentDelta returns the per-segment delta (in level units) for
// the (segmentID, plane, edge) combination from seg.
func loopFilterSegmentDelta(seg internalparser.SegmentationParams, segmentID int, plane internalloopfilter.Plane, edge internalloopfilter.Edge) (int16, error) {
	if segmentID < 0 || segmentID >= internalparser.MaxSegments {
		return 0, internalloopfilter.ErrInvalidFilter
	}
	return internalloopfilter.SegmentDelta(seg, uint8(segmentID), plane, edge)
}

// applyLoopFilter4Edge applies the 4-tap loop filter to one edge in dst.
func applyLoopFilter4Edge(dst FramePlane, bytesPerSample int, bitDepth uint8, edge internalloopfilter.Edge, x int, y int, length int, thresholds internalloopfilter.Thresholds) error {
	return internalloopfilter.Filter4Edge(dst, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), thresholds)
}

// applyLoopFilterEdgeByWidth dispatches to ApplyLoopFilter4/6/8/14Edge based
// on width (4, 6, 8, or 14).
func applyLoopFilterEdgeByWidth(width int, dst FramePlane, bytesPerSample int, bitDepth uint8, edge internalloopfilter.Edge, x int, y int, length int, thresholds internalloopfilter.Thresholds) error {
	switch width {
	case 4, 6, 8, 14:
		return internalloopfilter.FilterEdgeByWidth(uint8(width), dst, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), thresholds)
	default:
		return internalloopfilter.ErrInvalidFilter
	}
}
