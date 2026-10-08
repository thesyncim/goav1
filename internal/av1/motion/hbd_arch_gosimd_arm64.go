// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// hbdSIMDAvailable reports whether the Go SIMD high-bit-depth kernels may run.
// NEON is mandatory on every arm64 CPU Go supports.
func hbdSIMDAvailable() bool {
	return cpu.Detected.NEON
}

// hbdJoinU16 packs the four low lanes of lo and hi into eight uint16 lanes
// (lo first). The narrowing truncation keeps the scalar uint16 cast of the
// CONV_BUF; the clipped pixel values are already in range.
func hbdJoinU16(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	lo16 := lo.TruncToInt16().ToBits().ReshapeToUint64s()
	hi16 := hi.TruncToInt16().ToBits().ReshapeToUint64s()
	return lo16.InterleaveLo(hi16).ReshapeToUint16s()
}

// hbdMAC8 accumulates the widening products of eight int16 samples s with the
// broadcast coefficient c into lo (lanes 0..3) and hi (lanes 4..7).
func hbdMAC8(lo, hi archsimd.Int32x4, s, c archsimd.Int16x8) (archsimd.Int32x4, archsimd.Int32x4) {
	return lo.Add(s.MulWidenLo(c)), hi.Add(s.HiToLo().MulWidenLo(c))
}

// hbdMulAdd32 returns x*y + z for int32 lanes (one fused multiply-accumulate).
func hbdMulAdd32(x, y, z archsimd.Int32x4) archsimd.Int32x4 {
	return x.MulAdd(y, z)
}

// hbdShiftRight uses arm64's signed per-lane shift with a hoisted amount.
func hbdShiftRight(v, amount archsimd.Int32x4, _ int) archsimd.Int32x4 {
	return v.Shift(amount)
}

func hbdShiftRightU16(v archsimd.Uint16x8, amount archsimd.Int16x8, _ int) archsimd.Uint16x8 {
	return v.Shift(amount)
}

// hbdWidenU16 zero-extends eight uint16 lanes to two int32 vectors (lanes 0..3
// and 4..7).
func hbdWidenU16(u archsimd.Uint16x8) (lo, hi archsimd.Int32x4) {
	return u.ExtendLo4ToUint32().BitsToInt32(), u.HiToLo().ExtendLo4ToUint32().BitsToInt32()
}

// narrowU8x8 packs the sixteen int32 lanes of lo and hi (already clipped to
// [0, 255]) into the low eight bytes of the result.
func narrowU8x8(lo, hi archsimd.Int32x4) archsimd.Uint8x16 {
	return hbdJoinU16(lo, hi).TruncToUint8()
}
