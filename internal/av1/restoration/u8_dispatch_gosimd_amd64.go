// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package restoration

import "simd/archsimd"

// init binds the Go-native AVX2 8-bit-pixel self-guided projection on amd64 when
// AVX2 is available. Without AVX2 the dispatch keeps the pure-Go reference. The
// 8-bit Wiener passes are bound in wiener_dispatch_amd64.go.
func init() {
	if archsimd.X86.AVX2() {
		sgrWeightedRowU8Impl = sgrWeightedRowU8SIMD
		return
	}
	sgrWeightedRowU8Impl = sgrWeightedRowU8
}
