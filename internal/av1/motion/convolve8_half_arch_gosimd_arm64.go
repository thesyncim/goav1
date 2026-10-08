// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import "simd/archsimd"

func u8HalfMAC(x, coefficient, sum archsimd.Int16x8) archsimd.Int16x8 {
	return x.MulAdd(coefficient, sum)
}

func u8HalfSaturate(x archsimd.Int16x8) archsimd.Uint8x16 {
	return x.SaturateToUint8()
}

func u8WideMAC8(lo, hi archsimd.Int32x4, samples, coefficient archsimd.Int16x8) (archsimd.Int32x4, archsimd.Int32x4) {
	return lo.Add(samples.MulWidenLo(coefficient)), hi.Add(samples.HiToLo().MulWidenLo(coefficient))
}

func u8PackClipped32(lo, hi archsimd.Int32x4) archsimd.Uint8x16 {
	return lo.TruncToInt16().ToBits().ReshapeToUint64s().InterleaveLo(hi.TruncToInt16().ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16().SaturateToUint8()
}
