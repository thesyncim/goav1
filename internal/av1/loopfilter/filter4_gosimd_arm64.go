// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"simd/archsimd"
	"unsafe"
)

// lf16LoadP loads eight contiguous uint16 samples as signed 16-bit lanes.
// Loopfilter samples are at most 4095, so the sign bit stays clear.
func lf16LoadP(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p))
}

// lf16StoreP writes eight filtered 16-bit samples back to a contiguous row.
func lf16StoreP(p unsafe.Pointer, v archsimd.Int16x8) {
	v.StoreArray((*[8]int16)(p))
}
