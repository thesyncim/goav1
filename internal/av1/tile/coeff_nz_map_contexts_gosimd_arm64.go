// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && arm64 && !purego

package tile

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// coeffNZMapSIMDAvailable gates the NEON Go SIMD nz-map pass.
func coeffNZMapSIMDAvailable() bool {
	return cpu.Detected.NEON
}

// coeffNZMapCompress gathers the bytes of v selected by pick (index 0x80 or
// any index >= 16 yields zero) using a NEON table lookup.
func coeffNZMapCompress(v archsimd.Uint8x16, pick *[16]uint8) archsimd.Uint8x16 {
	return v.LookupOrZero(archsimd.LoadUint8x16Array(pick))
}
