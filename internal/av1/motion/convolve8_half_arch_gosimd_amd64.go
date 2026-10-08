// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import "simd/archsimd"

func u8HalfMAC(x, coefficient, sum archsimd.Int16x8) archsimd.Int16x8 {
	return x.Mul(coefficient).Add(sum)
}

func u8HalfSaturate(x archsimd.Int16x8) archsimd.Uint8x16 {
	var lanes [8]int16
	var bytes [16]byte
	x.StoreArray(&lanes)
	for i, v := range lanes {
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		bytes[i] = byte(v)
	}
	return archsimd.LoadUint8x16(bytes[:])
}

func u8WideMAC8(lo, hi archsimd.Int32x4, samples, coefficient archsimd.Int16x8) (archsimd.Int32x4, archsimd.Int32x4) {
	return hbdMAC8(lo, hi, samples, coefficient)
}

func u8PackClipped32(lo, hi archsimd.Int32x4) archsimd.Uint8x16 {
	var a, b [4]int32
	var bytes [16]byte
	lo.StoreArray(&a)
	hi.StoreArray(&b)
	for i := range a {
		bytes[i] = byte(a[i])
		bytes[i+4] = byte(b[i])
	}
	return archsimd.LoadUint8x16(bytes[:])
}
