// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package cdef

import "simd/archsimd"

// init binds the AVX2 Go SIMD CDEF block filters when the CPU advertises AVX2.
// Otherwise the pure-Go references stay in place. The 256-bit word kernels
// (filter_word_amd64.go) take every shape they cover and hand the rest to the
// 128-bit kernels. The assignment happens once, before any decoder goroutine
// starts, so the steady-state cost is a single indirect call.
func init() {
	if archsimd.X86.AVX2() {
		filterBlockImpl = filterBlockAVX2
		filterBlockU8Impl = filterBlockU8AVX2
	}
}
