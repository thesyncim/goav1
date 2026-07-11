// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for upstream attribution.

//go:build arm64 && !purego && !goexperiment.simd

package cdef

// init binds the NEON CDEF direction-search kernels; they are bit-exact with the
// scalar references (TestFindDirectionNEONMatchesScalar). Under the
// goexperiment.simd build this file is excluded and the Go-native SIMD kernels
// bind instead (direction_gosimd_arm64.go); the NEON wrappers stay compiled
// there for the differential test and the head-to-head benchmark.
func init() {
	findDirectionImpl = findDirectionNEON
	findDirectionDualImpl = findDirectionDualNEON
	findDirectionU8Impl = findDirectionU8NEON
	findDirectionDualU8Impl = findDirectionDualU8NEON
}
