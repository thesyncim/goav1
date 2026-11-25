// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goexperiment.simd || purego || (!amd64 && !arm64)

package cdef

// init binds the pure-Go CDEF block filter on architectures the dispatcher
// does not special-case, on purego builds, and on arm64/amd64 builds without
// the SIMD experiment, where the Go SIMD kernels are excluded. This file only
// keeps the dispatch wiring symmetric across builds.
func init() {
	filterBlockImpl = filterBlockPureGo
}

// init binds the pure-Go 8-bit-dst CDEF block filter on architectures the
// dispatcher does not special-case, and on the arm64/amd64 purego builds.
// Kept only so the dispatch wiring stays symmetric across builds.
func init() {
	filterBlockU8Impl = filterBlockU8PureGo
}
