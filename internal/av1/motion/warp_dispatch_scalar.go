// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !arm64 || purego

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

func warpHorizontal8ResidentDispatch(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	return warpHorizontal8Resident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
}
