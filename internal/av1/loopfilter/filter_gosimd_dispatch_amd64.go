// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package loopfilter

import "simd/archsimd"

// init binds the Go-native SIMD 8-bit deblocking kernels on amd64 when the CPU
// advertises AVX2: the narrow four-tap kernels (8-bit and 10/12-bit) and the
// six/eight/fourteen-tap 8-bit kernels. The 10/12-bit wide kernels keep the
// pure-Go reference on amd64, as before. Without AVX2 every slot keeps the
// pure-Go reference. The assignment happens once, before any decoder goroutine
// starts, so the steady-state cost is a single indirect call.
func init() {
	if archsimd.X86.AVX2() {
		filter4EdgeImpl = filter4EdgeSIMD
		filter4Edge16Impl = filter4Edge16SIMD
		filter6EdgeImpl = filter6EdgeSIMD
		filter8EdgeImpl = filter8EdgeSIMD
		filter14EdgeImpl = filter14EdgeSIMD
	}
}
