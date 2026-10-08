// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import (
	"encoding/binary"
	"simd/archsimd"
)

// Go-native SIMD CfL apply, DC sum kernels for arm64. They mirror the scalar
// contracts in intra_kernels.go and cfl.go:
//   - applyCFL adds roundPowerOfTwoSigned(alpha*ac, 6) to the chroma DC sample and
//     clamps to [0,max]. The product is formed in int32 lanes (MulWidenLo) and the
//     signed round is (|x|+32)>>6 with the sign restored by (r^s)-s.
//   - sumSamples accumulates uint16 samples into uint32 lanes; up to 65536 samples
//     the lane sums cannot exceed 32 bits, so the total is exact.

// cflRoundBias32 is the rounding bias of roundPowerOfTwoSigned(x, 6).
var cflRoundBias32 = archsimd.BroadcastInt32x4(1 << 5)

// cflShr6I32 and cflShr31I32 are per-lane shift vectors: arithmetic right shift
// by 6 for the magnitude and by 31 for the sign mask (see shr8U16 for why these
// are hoisted rather than ShiftAllRight).
var (
	cflShr6I32  = archsimd.BroadcastInt32x4(-6)
	cflShr31I32 = archsimd.BroadcastInt32x4(-31)
)

// maxSIMDSumSamples bounds the sample count for which the uint32 lane sums in
// sumSamplesSIMD are exact (65536 * 65535 < 2^32).
const maxSIMDSumSamples = 1 << 16

// cflRound6 returns roundPowerOfTwoSigned(x, 6) for four int32 lanes.
func cflRound6(x archsimd.Int32x4) archsimd.Int32x4 {
	mag := x.Abs().Add(cflRoundBias32).Shift(cflShr6I32)
	sign := x.Shift(cflShr31I32)
	return mag.Xor(sign).Sub(sign)
}

// cflClampInt32Pair clamps two int32 vectors to [0,max] and packs them to int16.
func cflClampInt32Pair(lo, hi archsimd.Int32x4, zero, max archsimd.Int32x4) archsimd.Int16x8 {
	return cflTruncateInt32PairToInt16(lo.Max(zero).Min(max), hi.Max(zero).Min(max))
}

// loadBytes16 gathers the 16 bytes b[0:16] as eight little-endian uint16 lanes.
func loadBytes16(b []byte) archsimd.Uint16x8 {
	return archsimd.LoadUint64x2Array(&[2]uint64{
		binary.LittleEndian.Uint64(b[0:]),
		binary.LittleEndian.Uint64(b[8:]),
	}).ReshapeToUint16s()
}

// sumSamplesSIMD is the Go-native SIMD form of sumSamplesPureGo.
func sumSamplesSIMD(samples []uint16) int {
	n := len(samples)
	if n > maxSIMDSumSamples {
		return sumSamplesPureGo(samples)
	}
	chunks := n &^ 7
	total := 0
	if chunks > 0 {
		acc := archsimd.BroadcastUint32x4(0)
		for i := 0; i < chunks; i += 8 {
			v := archsimd.LoadUint16x8Array((*[8]uint16)(samples[i:]))
			acc = acc.Add(v.ExtendLo4ToUint32()).Add(v.HiToLo().ExtendLo4ToUint32())
		}
		total = int(acc.ReduceSum())
	}
	for i := chunks; i < n; i++ {
		total += int(samples[i])
	}
	return total
}

// applyCFL8SIMD is the 8-bit CfL apply for widths that are a multiple of 8 with
// max 255 (the saturating clamp is exactly [0,255]).
func applyCFL8SIMD(block planeBlock, visibleWidth int, visibleHeight int, acQ3 []int16, alphaQ3 int) {
	alphaV := archsimd.BroadcastInt16x8(int16(alphaQ3))
	zeroV := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(0xff)
	for row := 0; row < visibleHeight; row++ {
		line := block.pix[row*block.stride:][:visibleWidth]
		ac := acQ3[row*CFLBufLine:][:visibleWidth]
		for col := 0; col < visibleWidth; col += 8 {
			v := archsimd.LoadInt16x8Array((*[8]int16)(ac[col:]))
			loR := cflRound6(alphaV.MulWidenLo(v))
			hiR := cflRound6(alphaV.MulWidenLo(v.HiToLo()))
			cur := loadBytes8(line[col:])
			lo := cur.ExtendLo4ToInt32().Add(loR)
			hi := cur.HiToLo().ExtendLo4ToInt32().Add(hiR)
			packed := cflClampInt32Pair(lo, hi, zeroV, maxV).SaturateToUint8().ReshapeToUint64s()
			binary.LittleEndian.PutUint64(line[col:], packed.GetElem(0))
		}
	}
}

// loadBytes8 gathers the eight bytes b[0:8] as eight uint16 lanes.
func loadBytes8(b []byte) archsimd.Int16x8 {
	return archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(b)).ReshapeToUint8s().ExtendLo8ToUint16().ConvertToInt16()
}

// applyCFL16SIMD is the high-bit-depth CfL apply for widths that are a multiple
// of 8 and any max.
func applyCFL16SIMD(block planeBlock, visibleWidth int, visibleHeight int, acQ3 []int16, alphaQ3 int, max uint16) {
	alphaV := archsimd.BroadcastInt16x8(int16(alphaQ3))
	zeroV := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(int32(max))
	for row := 0; row < visibleHeight; row++ {
		line := block.pix[row*block.stride:][:visibleWidth*2]
		ac := acQ3[row*CFLBufLine:][:visibleWidth]
		for col := 0; col < visibleWidth; col += 8 {
			v := archsimd.LoadInt16x8Array((*[8]int16)(ac[col:]))
			loR := cflRound6(alphaV.MulWidenLo(v))
			hiR := cflRound6(alphaV.MulWidenLo(v.HiToLo()))
			cur := loadBytes16(line[col*2:])
			lo := cur.ExtendLo4ToUint32().ConvertToInt32().Add(loR)
			hi := cur.HiToLo().ExtendLo4ToUint32().ConvertToInt32().Add(hiR)
			storeU16x8(line[col*2:], cflClampInt32Pair(lo, hi, zeroV, maxV))
		}
	}
}

// applyCFL4SIMD is the 8-bit CfL apply for width 4 with even height and max 255.
// Lanes 0..3 hold row r and lanes 4..7 hold row r+1, so one vector covers two rows.
func applyCFL4SIMD(block planeBlock, visibleHeight int, acQ3 []int16, alphaQ3 int) {
	alphaV := archsimd.BroadcastInt16x8(int16(alphaQ3))
	zeroV := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(0xff)
	for row := 0; row < visibleHeight; row += 2 {
		ac0 := acQ3[row*CFLBufLine:]
		ac1 := acQ3[(row+1)*CFLBufLine:]
		v := archsimd.LoadInt16x8Array(&[8]int16{ac0[0], ac0[1], ac0[2], ac0[3], ac1[0], ac1[1], ac1[2], ac1[3]})
		loR := cflRound6(alphaV.MulWidenLo(v))
		hiR := cflRound6(alphaV.MulWidenLo(v.HiToLo()))
		l0 := block.pix[row*block.stride:]
		l1 := block.pix[(row+1)*block.stride:]
		cur := archsimd.LoadInt16x8Array(&[8]int16{
			int16(l0[0]), int16(l0[1]), int16(l0[2]), int16(l0[3]),
			int16(l1[0]), int16(l1[1]), int16(l1[2]), int16(l1[3]),
		})
		lo := cur.ExtendLo4ToInt32().Add(loR)
		hi := cur.HiToLo().ExtendLo4ToInt32().Add(hiR)
		packed := cflClampInt32Pair(lo, hi, zeroV, maxV).SaturateToUint8().ReshapeToUint32s()
		putU32(block.pix, row*block.stride, packed.GetElem(0))
		putU32(block.pix, (row+1)*block.stride, packed.GetElem(1))
	}
}

// applyCFLSIMD routes a CfL apply to the kernel for its shape; shapes no kernel
// covers use the pure-Go reference.
func applyCFLSIMD(block planeBlock, bytesPerSample int, visibleWidth int, visibleHeight int, acQ3 []int16, alphaQ3 int, max uint16) {
	switch {
	case bytesPerSample == 1 && visibleWidth%8 == 0 && max == 0xff:
		applyCFL8SIMD(block, visibleWidth, visibleHeight, acQ3, alphaQ3)
	case bytesPerSample == 2 && visibleWidth%8 == 0:
		applyCFL16SIMD(block, visibleWidth, visibleHeight, acQ3, alphaQ3, max)
	case bytesPerSample == 1 && visibleWidth == 4 && visibleHeight%2 == 0 && max == 0xff:
		applyCFL4SIMD(block, visibleHeight, acQ3, alphaQ3)
	default:
		applyCFLPureGo(block, bytesPerSample, visibleWidth, visibleHeight, acQ3, alphaQ3, max)
	}
}
