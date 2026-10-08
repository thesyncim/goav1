// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package dsp

import "simd/archsimd"

// The AVX2 forms of the three narrowing operations are not available in archsimd
// (VPACKUSWB and the 128-bit VPACKUSDW/VPACKSSDW pair with zeros are not exposed,
// and the saturating narrows that are exposed need AVX-512), so the lanes are
// narrowed through a stored array. These helpers are correctness-only on amd64.

// planeNarrowWord returns the low byte of each u16 lane of v, lane i in byte i of
// a little-endian word. The lanes must hold values of at most 255.
func planeNarrowWord(v archsimd.Uint16x8) uint64 {
	var t [8]uint16
	v.StoreArray(&t)
	var w uint64
	for i, x := range t {
		w |= uint64(byte(x)) << (8 * i)
	}
	return w
}

// planeNarrow16to8 returns the low bytes of the eight u16 lanes of lo followed by
// those of hi, sixteen bytes in all. The lanes must hold values of at most 255.
func planeNarrow16to8(lo, hi archsimd.Uint16x8) archsimd.Uint8x16 {
	var a, b [8]uint16
	var t [16]uint8
	lo.StoreArray(&a)
	hi.StoreArray(&b)
	for i := range 8 {
		t[i] = byte(a[i])
		t[8+i] = byte(b[i])
	}
	return archsimd.LoadUint8x16Array(&t)
}

// planePackPairI16 saturates the int32 lanes of lo to int16 in lanes 0-3 and the
// int32 lanes of hi in lanes 4-7.
func planePackPairI16(lo, hi archsimd.Int32x4) archsimd.Int16x8 {
	var a, b [4]int32
	var t [8]int16
	lo.StoreArray(&a)
	hi.StoreArray(&b)
	for i := range 4 {
		t[i] = planeSatInt16(a[i])
		t[4+i] = planeSatInt16(b[i])
	}
	return archsimd.LoadInt16x8Array(&t)
}

// planeSatInt16 saturates v to the int16 range.
func planeSatInt16(v int32) int16 {
	switch {
	case v > 32767:
		return 32767
	case v < -32768:
		return -32768
	default:
		return int16(v)
	}
}

// planeRawResidual returns the int16-range rounding of each int32 lane,
// floor((v+8)/16), without any intermediate overflow: (v>>4) plus the carry
// ((v&15)+8)>>4. c15 and c8 are the constants 15 and 8; the shifts are
// immediate arithmetic shifts (VPSRAD), so sh4 is not needed here.
func planeRawResidual(v, c15, c8, _ archsimd.Int32x4) archsimd.Int32x4 {
	carry := v.And(c15).Add(c8).ShiftAllRight(4)
	return v.ShiftAllRight(4).Add(carry)
}
