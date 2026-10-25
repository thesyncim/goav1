// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && arm64 && !purego)

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

// Targets without the vector 16bpc copy always run dsp.CopyPlaneBlock.
func mc16CopyFast(dst *frame.Plane, ref *frame.Plane, dstX, dstY, refX, refY, w, h int) bool {
	return false
}
