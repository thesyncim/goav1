// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && amd64 && !purego

package tile

import (
	"simd/archsimd"
	"unsafe"
)

// coeffNZMapSIMDAvailable gates the AVX2 Go SIMD nz-map pass on OS-enabled
// AVX2 as reported by archsimd.
func coeffNZMapSIMDAvailable() bool {
	return archsimd.X86.AVX2()
}

// coeffNZMapCompress gathers the bytes of v selected by pick (index with the
// high bit set yields zero) using a per-lane byte shuffle (VPSHUFB). The index
// table is shared with the arm64 lookup, so the byte view is reinterpreted.
func coeffNZMapCompress(v archsimd.Uint8x16, pick *[16]uint8) archsimd.Uint8x16 {
	return v.PermuteOrZero(archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(pick))))
}
