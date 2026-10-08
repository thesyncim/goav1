// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package encoder

import "simd/archsimd"

// sadSIMDBound reports whether init binds the SAD dispatch variables to the
// AVX2 Go-native-SIMD kernels. It gates on archsimd's own feature report rather
// than cpu.Detected.AVX2, which Rosetta reports as false although AVX2 executes.
// Without AVX2 the portable Go references stay in place, so the lowercase
// wrappers in sad_dispatch_default.go keep their scalar behavior.
var sadSIMDBound = archsimd.X86.AVX2()

// init binds the SAD dispatch variables to the AVX2 Go-native-SIMD kernels.
func init() {
	if sadSIMDBound {
		bindSIMDSAD()
	}
}
