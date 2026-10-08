// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package restoration

import (
	"encoding/binary"
	"simd/archsimd"
)

// AVX2 Go-native SIMD vertical Wiener passes. Eight output columns are computed
// per iteration in 32-bit lanes as a plain seven-tap MAC, which is exact for any
// filter (the vertical taps are not assumed symmetric) because every product and
// the sum stay far inside int32 for the whole temp range (samples <= 32767). The
// rounding bias and -offset seed the accumulator so the trailing arithmetic
// shift reproduces roundPowerOfTwo(sum, round1), and the [0,max] clamp matches
// clampInt32 bit for bit.

// wienerVerticalSIMD is the AVX2 form of wienerVertical for the uint16 output
// domain. Widths that are not a multiple of 8 use the scalar reference.
func wienerVerticalSIMD(temp []uint16, tempStride int, dst []uint16, dstStride int, width int, height int, filter WienerFilter, bitDepth int, round1 int, max uint16) {
	if width < 8 || width%8 != 0 {
		wienerVertical(temp, tempStride, dst, dstStride, width, height, filter, bitDepth, round1, max)
		return
	}
	offset := int32(1) << (bitDepth + round1 - 1)
	seedV := archsimd.BroadcastInt32x8(-offset + roundBias(round1))
	zero := archsimd.BroadcastInt32x8(0)
	maxV := archsimd.BroadcastInt32x8(int32(max))
	f0 := archsimd.BroadcastInt32x8(int32(filter[0]))
	f1 := archsimd.BroadcastInt32x8(int32(filter[1]))
	f2 := archsimd.BroadcastInt32x8(int32(filter[2]))
	// The center tap absorbs the s3<<WienerFilterBits center reapplication; the
	// vertical taps are not assumed symmetric, so all seven are kept.
	f3 := archsimd.BroadcastInt32x8(int32(filter[3]) + (1 << WienerFilterBits))
	f4 := archsimd.BroadcastInt32x8(int32(filter[4]))
	f5 := archsimd.BroadcastInt32x8(int32(filter[5]))
	f6 := archsimd.BroadcastInt32x8(int32(filter[6]))
	shift := uint64(round1)
	for row := 0; row < height; row++ {
		dstRow := dst[row*dstStride : row*dstStride+width]
		for col := 0; col < width; col += 8 {
			r0 := archsimd.LoadUint16x8(temp[(row+0)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r1 := archsimd.LoadUint16x8(temp[(row+1)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r2 := archsimd.LoadUint16x8(temp[(row+2)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r3 := archsimd.LoadUint16x8(temp[(row+3)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r4 := archsimd.LoadUint16x8(temp[(row+4)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r5 := archsimd.LoadUint16x8(temp[(row+5)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r6 := archsimd.LoadUint16x8(temp[(row+6)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			sum := seedV.Add(r0.Mul(f0)).Add(r1.Mul(f1)).Add(r2.Mul(f2)).Add(r3.Mul(f3)).
				Add(r4.Mul(f4)).Add(r5.Mul(f5)).Add(r6.Mul(f6))
			out := sum.ShiftAllRight(shift).Max(zero).Min(maxV)
			packClampedInt32x8ToUint16(out).Store(dstRow[col:])
		}
	}
}

// wienerVerticalU8SIMD is the AVX2 form of wienerVerticalU8: the same vertical
// pass with the output clamped to [0,255] and packed to uint8. Eight bytes are
// stored per group; widths that are not a multiple of 8 use the scalar reference.
func wienerVerticalU8SIMD(temp []uint16, tempStride int, dst []uint8, dstStride int, width int, height int, filter WienerFilter, round1 int) {
	if width < 8 || width%8 != 0 {
		wienerVerticalU8(temp, tempStride, dst, dstStride, width, height, filter, round1)
		return
	}
	const bitDepth = 8
	offset := int32(1) << (bitDepth + round1 - 1)
	seedV := archsimd.BroadcastInt32x8(-offset + roundBias(round1))
	zero := archsimd.BroadcastInt32x8(0)
	maxV := archsimd.BroadcastInt32x8(255)
	// PSHUFB index vector: the low byte of each 16-bit lane. The clamped values
	// are already in [0,255], so the low bytes of the first eight lanes are the
	// exact uint8 results (VPMOVUSWB, the saturating narrow, needs AVX-512).
	lowBytes := archsimd.LoadInt8x16Array(&u8LowByteIdx)
	f0 := archsimd.BroadcastInt32x8(int32(filter[0]))
	f1 := archsimd.BroadcastInt32x8(int32(filter[1]))
	f2 := archsimd.BroadcastInt32x8(int32(filter[2]))
	// The center tap absorbs the s3<<WienerFilterBits center reapplication; the
	// vertical taps are not assumed symmetric, so all seven are kept.
	f3 := archsimd.BroadcastInt32x8(int32(filter[3]) + (1 << WienerFilterBits))
	f4 := archsimd.BroadcastInt32x8(int32(filter[4]))
	f5 := archsimd.BroadcastInt32x8(int32(filter[5]))
	f6 := archsimd.BroadcastInt32x8(int32(filter[6]))
	shift := uint64(round1)
	for row := 0; row < height; row++ {
		dstRow := dst[row*dstStride : row*dstStride+width]
		for col := 0; col < width; col += 8 {
			r0 := archsimd.LoadUint16x8(temp[(row+0)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r1 := archsimd.LoadUint16x8(temp[(row+1)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r2 := archsimd.LoadUint16x8(temp[(row+2)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r3 := archsimd.LoadUint16x8(temp[(row+3)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r4 := archsimd.LoadUint16x8(temp[(row+4)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r5 := archsimd.LoadUint16x8(temp[(row+5)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r6 := archsimd.LoadUint16x8(temp[(row+6)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			sum := seedV.Add(r0.Mul(f0)).Add(r1.Mul(f1)).Add(r2.Mul(f2)).Add(r3.Mul(f3)).
				Add(r4.Mul(f4)).Add(r5.Mul(f5)).Add(r6.Mul(f6))
			out := packClampedInt32x8ToUint16(sum.ShiftAllRight(shift).Max(zero).Min(maxV)).ReshapeToUint8s().PermuteOrZero(lowBytes)
			binary.LittleEndian.PutUint64(dstRow[col:col+8], out.ReshapeToUint64s().GetElem(0))
		}
	}
}

// u8LowByteIdx is the PSHUFB selector that gathers the low byte of each of the
// first eight 16-bit lanes into the low eight bytes of the result.
var u8LowByteIdx = [16]int8{0, 2, 4, 6, 8, 10, 12, 14}
