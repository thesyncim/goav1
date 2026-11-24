// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && arm64 && !purego)

package restoration

// sgrU8Passes has no vector form on this target; the int32 passes run.
func sgrU8Passes(src []uint8, srcStride, srcOrigin, width, height int, params SGRParams, flt0, flt1 []int32) bool {
	return false
}
