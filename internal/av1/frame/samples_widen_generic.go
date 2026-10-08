// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build purego || !goexperiment.simd || (!arm64 && !amd64)

package frame

// loadSampleRows8 binds the pure-Go 8-bit sample staging widen on targets
// without a tuned variant, on builds without GOEXPERIMENT=simd, and on purego
// builds, where the Go SIMD kernels are excluded.
// See the dispatch note on loadSampleRows8PureGo for why this is a build-tag
// binding instead of a func variable.
func loadSampleRows8(dst []uint16, dstStride int, src []byte, srcStride int, width int, height int) {
	loadSampleRows8PureGo(dst, dstStride, src, srcStride, width, height)
}
