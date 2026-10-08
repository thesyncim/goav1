// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego && goexperiment.simd

package dsp

import "simd/archsimd"

// init binds the Go-native SIMD MinMaxAbsDiff8x8 on amd64 SIMD builds. AVX2 is
// not part of the GOAMD64=v1 baseline, so the SIMD variant is selected only when
// archsimd reports AVX2; the pure-Go reference is the fallback.
func init() {
	if archsimd.X86.AVX2() {
		minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8SIMD
		return
	}
	minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8PureGo
}
