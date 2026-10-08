// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package loopfilter

import "simd/archsimd"

// init binds the Go-native SIMD narrow deblocking kernels on amd64 when the CPU
// advertises AVX2. Without AVX2 the slots keep the pure-Go reference. The
// assignment happens once, before any decoder goroutine starts, so the
// steady-state cost is a single indirect call.
func init() {
	if archsimd.X86.AVX2() {
		filter4EdgeImpl = filter4EdgeSIMD
		filter4Edge16Impl = filter4Edge16SIMD
	}
}
