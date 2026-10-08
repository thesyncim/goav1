// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import "simd/archsimd"

// cdefAbsDiffInt16x8 computes signed absolute difference in each 16-bit lane,
// preserving the low 16 bits when the full-range difference wraps.
func cdefAbsDiffInt16x8(a, b archsimd.Int16x8) archsimd.Int16x8 {
	return a.Max(b).Sub(a.Min(b))
}
