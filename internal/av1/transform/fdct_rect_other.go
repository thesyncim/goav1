// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goexperiment.simd || !arm64 || purego

package transform

// forwardDCTRect8BitTrusted has no vector kernel in this build; the scalar
// rectangles run instead.
func forwardDCTRect8BitTrusted(coeff []int32, coeffStride int, residual []int16, residualStride int, size Size) bool {
	return false
}
