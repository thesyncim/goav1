// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package dsp

import "simd/archsimd"

// planeNarrowWord returns the low byte of each u16 lane of v, lane i in byte i of
// a little-endian word. The lanes must hold values of at most 255.
func planeNarrowWord(v archsimd.Uint16x8) uint64 {
	return v.SaturateToUint8().ReshapeToUint64s().GetElem(0)
}

// planeNarrow16to8 returns the low bytes of the eight u16 lanes of lo followed by
// those of hi, sixteen bytes in all. The lanes must hold values of at most 255.
// VUQXTN narrows each half to the low 64 bits, which InterleaveLo then joins.
func planeNarrow16to8(lo, hi archsimd.Uint16x8) archsimd.Uint8x16 {
	l := lo.SaturateToUint8().ReshapeToUint64s()
	h := hi.SaturateToUint8().ReshapeToUint64s()
	return l.InterleaveLo(h).ReshapeToUint8s()
}

// planePackPairI16 saturates the int32 lanes of lo to int16 in lanes 0-3 and the
// int32 lanes of hi in lanes 4-7. SQXTN narrows each half to the low 64 bits.
func planePackPairI16(lo, hi archsimd.Int32x4) archsimd.Int16x8 {
	l := lo.SaturateToInt16().ToBits().ReshapeToUint64s()
	h := hi.SaturateToInt16().ToBits().ReshapeToUint64s()
	return l.InterleaveLo(h).ReshapeToUint16s().BitsToInt16()
}

// planeRawResidual returns the int16-range rounding of each int32 lane,
// floor((v+8)/16), without any intermediate overflow: (v>>4) plus the carry
// ((v&15)+8)>>4. c15, c8 and sh4 are the loop-invariant constants 15, 8 and -4;
// a negative per-lane shift count is an arithmetic right shift on arm64.
func planeRawResidual(v, c15, c8, sh4 archsimd.Int32x4) archsimd.Int32x4 {
	carry := v.And(c15).Add(c8).Shift(sh4)
	return v.Shift(sh4).Add(carry)
}
