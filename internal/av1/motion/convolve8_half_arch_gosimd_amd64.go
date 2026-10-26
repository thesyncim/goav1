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
	x = x.Max(archsimd.Int16x8{}).Min(archsimd.BroadcastInt16x8(255))
	return x.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&u8EvenBytesIdx))
}

// VPSHUFB indices: the low byte of each word into bytes 0..7, and the low
// byte of each dword into bytes 0..3 (lo) or 4..7 (hi); -1 zeroes the rest.
var (
	u8EvenBytesIdx = [16]int8{0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1}
	u8LoDwordBytes = [16]int8{0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
	u8HiDwordBytes = [16]int8{-1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1}
)

func u8WideMAC8(lo, hi archsimd.Int32x4, samples, coefficient archsimd.Int16x8) (archsimd.Int32x4, archsimd.Int32x4) {
	return hbdMAC8(lo, hi, samples, coefficient)
}

// u8PackClipped32 truncates the eight int32 lanes of lo and hi to bytes 0..7
// (the scalar byte cast), zeroing bytes 8..15.
func u8PackClipped32(lo, hi archsimd.Int32x4) archsimd.Uint8x16 {
	a := lo.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&u8LoDwordBytes))
	b := hi.ToBits().ReshapeToUint8s().PermuteOrZero(archsimd.LoadInt8x16Array(&u8HiDwordBytes))
	return a.Or(b)
}

func u8PackU16From32(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	return hbdJoinU16(lo, hi)
}

func u8MulAdd32(x, coefficient, sum archsimd.Int32x4) archsimd.Int32x4 {
	return x.Mul(coefficient).Add(sum)
}

func u8WidenU16(x archsimd.Uint16x8) (archsimd.Int32x4, archsimd.Int32x4) {
	return hbdWidenU16(x)
}

func u8PackSaturate32(lo, hi archsimd.Int32x4) archsimd.Uint8x16 {
	zero, c255 := archsimd.Int32x4{}, archsimd.BroadcastInt32x4(255)
	return u8PackClipped32(lo.Max(zero).Min(c255), hi.Max(zero).Min(c255))
}

func u8Right16(x, amount archsimd.Int16x8, count uint64) archsimd.Int16x8 {
	return x.ShiftAllRight(count)
}

func u8Right32(x, amount archsimd.Int32x4, count uint64) archsimd.Int32x4 {
	return x.ShiftAllRight(count)
}
