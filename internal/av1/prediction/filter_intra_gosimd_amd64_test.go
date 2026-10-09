// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

import (
	"simd/archsimd"
	"testing"
)

// TestFilterIntra8SIMDBinding asserts that on an AVX2 machine the amd64 SIMD
// build binds the Go-native 8-bit filter-intra kernel to the dispatch slot.
func TestFilterIntra8SIMDBinding(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 not available; the SIMD kernel is not bound")
	}
	assertDispatchTarget(t, "predictFilterIntra8Impl", predictFilterIntra8Impl, "predictFilterIntraBlockDirect8SIMD")
}
