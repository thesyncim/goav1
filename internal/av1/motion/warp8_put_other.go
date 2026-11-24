// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && arm64 && !purego)

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

// warpPut8x8Dispatch has no int16-intermediate kernel here; the int32 warp
// path runs.
func warpPut8x8Dispatch(dst []byte, stride int, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, gamma, delta int) bool {
	return false
}
