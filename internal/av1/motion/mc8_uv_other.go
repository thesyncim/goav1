// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && arm64 && !purego)

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

// Put8UV is Put8 for the U and V blocks of one chroma prediction (same block,
// reference origin, phase and filters). Without the shared-check arm64 entry
// it is Put8 per plane; it returns false when some plane was not written, and
// the caller then predicts both planes plane by plane.
func Put8UV(dstU, dstV, refU, refV *frame.Plane, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY int, filters InterpFilters, scratch *ConvolveScratch) bool {
	return Put8(dstU, refU, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY, filters, scratch) &&
		Put8(dstV, refV, dstX, dstY, refX, refY, w, h, filterW, filterH, subX, subY, filters, scratch)
}
