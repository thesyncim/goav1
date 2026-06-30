// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build !arm64 || purego

package tile

func coeffCulLevelArch(coeffs []int16, scan []int16, eob int) (uint8, bool) {
	return 0, false
}
