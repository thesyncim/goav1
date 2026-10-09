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
