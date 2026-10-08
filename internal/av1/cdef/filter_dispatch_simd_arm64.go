// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && goexperiment.simd

package cdef

// init binds the Go SIMD CDEF block filter on arm64. It is bit-exact with the
// pure-Go reference (TestFilterBlockSIMDMatchesPureGo). The unit-level loop
// binds statically in filter_simd_arm64.go.
func init() {
	filterBlockImpl = filterBlockSIMD
}
