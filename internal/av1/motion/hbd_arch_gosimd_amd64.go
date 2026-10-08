// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"simd/archsimd"
)

// hbdSIMDAvailable reports whether the Go SIMD high-bit-depth kernels may run.
// The amd64 kernels are AVX2-only; AVX-512 is never required.
func hbdSIMDAvailable() bool {
	return archsimd.X86.AVX2()
}

// hbdJoinU16 packs the four low lanes of lo and hi into eight uint16 lanes
// (lo first) with the scalar uint16 cast. The AVX2 pack-with-truncation forms
// lower to AVX-512 on this target, so the narrowing stays scalar.
func hbdJoinU16(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	var a [8]int32
	lo.Store(a[:4])
	hi.Store(a[4:])
	var u [8]uint16
	for i := range u {
		u[i] = uint16(a[i])
	}
	return archsimd.LoadUint16x8(u[:])
}

// hbdMAC8 accumulates the products of eight int16 samples s with the broadcast
// coefficient c into lo (lanes 0..3) and hi (lanes 4..7). The samples are
// zero-extended, which equals sign extension for HBD sample values (<= 4095).
func hbdMAC8(lo, hi archsimd.Int32x4, s, c archsimd.Int16x8) (archsimd.Int32x4, archsimd.Int32x4) {
	zero := archsimd.BroadcastInt16x8(0)
	c32 := c.ExtendLo4ToInt32()
	lo32 := s.InterleaveLo(zero).ToBits().ReshapeToUint32s().BitsToInt32()
	hi32 := s.InterleaveHi(zero).ToBits().ReshapeToUint32s().BitsToInt32()
	return lo.Add(lo32.Mul(c32)), hi.Add(hi32.Mul(c32))
}

// hbdMulAdd32 returns x*y + z for int32 lanes.
func hbdMulAdd32(x, y, z archsimd.Int32x4) archsimd.Int32x4 {
	return x.Mul(y).Add(z)
}

// amd64 archsimd has no signed per-lane Shift operation. The scalar amount is
// invariant for this kernel and retains the same arithmetic right shift.
func hbdShiftRight(v, _ archsimd.Int32x4, n int) archsimd.Int32x4 {
	return v.ShiftAllRight(uint64(n))
}

func hbdShiftRightU16(v archsimd.Uint16x8, _ archsimd.Int16x8, n int) archsimd.Uint16x8 {
	return v.ShiftAllRight(uint64(n))
}

// hbdWidenU16 zero-extends eight uint16 lanes to two int32 vectors (lanes 0..3
// and 4..7) by interleaving with zero.
func hbdWidenU16(u archsimd.Uint16x8) (lo, hi archsimd.Int32x4) {
	zero := archsimd.BroadcastUint16x8(0)
	return u.InterleaveLo(zero).ReshapeToUint32s().BitsToInt32(), u.InterleaveHi(zero).ReshapeToUint32s().BitsToInt32()
}

// narrowU8x8 packs the eight int32 lanes of lo and hi (already clipped to
// [0, 255]) into the low eight bytes of the result. The narrowing is scalar for
// the same reason as hbdJoinU16.
func narrowU8x8(lo, hi archsimd.Int32x4) archsimd.Uint8x16 {
	var a [8]int32
	lo.Store(a[:4])
	hi.Store(a[4:])
	var b [16]byte
	for i := range a {
		b[i] = byte(a[i])
	}
	return archsimd.LoadUint8x16(b[:])
}
