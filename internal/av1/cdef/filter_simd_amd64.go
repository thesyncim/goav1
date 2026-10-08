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

// cdefLowByteIdx gathers the low byte of each 16-bit lane into bytes 0..7
// (PSHUFB, AVX2-era; the lanes 8..15 index with the zeroing high bit).
var cdefLowByteIdx = archsimd.LoadInt8x16Array(&[16]int8{0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128})

// cdefNarrowU8 keeps the low byte of each lane, the same bits as byte(y) in
// the pure-Go reference. AVX2 has no 16-to-8 truncation without AVX-512, so
// the bytes are gathered with PermuteOrZero (VPSHUFB) instead.
func cdefNarrowU8(y archsimd.Int16x8) archsimd.Uint8x16 {
	return y.ToBits().ReshapeToUint8s().PermuteOrZero(cdefLowByteIdx)
}
