// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package cdef

import "simd/archsimd"

// cdefShiftCount is the per-block constrain() shift in the form AVX2 takes
// for a uniform per-block count: a scalar that VPSRLW reads from an xmm.
type cdefShiftCount = uint64

func cdefShiftCountOf(n int) cdefShiftCount {
	return uint64(n)
}

// cdefShrU16 is a logical right shift of each 16-bit lane by the per-block
// count. AVX2 has no per-lane 16-bit variable shift, so the uniform count
// rides in the scalar form.
func cdefShrU16(v archsimd.Uint16x8, sh cdefShiftCount) archsimd.Uint16x8 {
	return v.ShiftAllRight(sh)
}
