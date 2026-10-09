// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && goexperiment.simd

package cdef

// init binds the Go SIMD CDEF block filters on arm64. Both are bit-exact with
// the pure-Go references (TestFilterBlockSIMDMatchesPureGo and
// TestFilterBlockU8SIMDMatchesPureGo). The unit-level loops bind statically in
// filter_simd_arm64.go.
func init() {
	filterBlockImpl = filterBlockSIMD
	filterBlockU8Impl = filterBlockU8SIMD
}
