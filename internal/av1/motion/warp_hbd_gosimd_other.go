// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goexperiment.simd || !arm64 || purego

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

func warpHorizontalHighBDResidentDispatch(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, ix4 int, sx4 int, iy4 int, sy4 int, alpha int, beta int, reduceBitsHoriz int, offsetBitsHoriz int) int {
	return warpHorizontalHighBDResident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
}
