// Ported from libaom: av1/common/av1_loopfilter.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package loopfilter

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
)

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
