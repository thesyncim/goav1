// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && !goexperiment.simd

package cdef

const filterBlockU8SplitSIMDEnabled = false

// dispatchFilterBlockU8NEON handles the fused block kernels in ordinary
// builds. Primary-only and secondary-only blocks use the scalar reference
// before reaching this router.
func dispatchFilterBlockU8NEON(ctx *filterBlockU8NEONCtx, width int, primaryStrength int, secondaryStrength int) {
	if width == 8 {
		cdefFilterBlock8U8NEON(ctx)
		return
	}
	cdefFilterBlock4U8NEON(ctx)
}
