// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

import (
	"encoding/binary"
	"simd/archsimd"
)

// Go-native SIMD CfL luma subsampling for amd64 (AVX2). It mirrors the arm64
// kernel and the scalar contract in subsampleLuma8PureGo. Even and odd input
// bytes are separated with VPSHUFB, widened to uint16 and summed.

// evenBytesFullIdx and oddBytesFullIdx pick the even and odd bytes of a 16-byte row
// into the low eight lanes.
var (
	evenBytesFullIdx = archsimd.LoadInt8x16Array(&[16]int8{0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128})
	oddBytesFullIdx  = archsimd.LoadInt8x16Array(&[16]int8{1, 3, 5, 7, 9, 11, 13, 15, -128, -128, -128, -128, -128, -128, -128, -128})
)

// cflPairSumBytes returns the per-column pair sums a[2c]+a[2c+1] of the 16-byte
// row v, for the eight output columns.
func cflPairSumBytes(v archsimd.Uint8x16) archsimd.Uint16x8 {
	even := v.PermuteOrZero(evenBytesFullIdx).ExtendLo8ToUint16()
	odd := v.PermuteOrZero(oddBytesFullIdx).ExtendLo8ToUint16()
	return even.Add(odd)
}

// subsampleLuma8SIMD is the amd64 form of subsampleLuma8PureGo.
func subsampleLuma8SIMD(outputQ3 []uint16, input []uint8, inputStride int, width int, height int, outW int, outH int, subX bool, subY bool) {
	switch {
	case subX && subY:
		if outW%8 != 0 {
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		for row := 0; row < height; row += 2 {
			outBase := (row >> 1) * CFLBufLine
			top := input[row*inputStride:]
			bot := input[(row+1)*inputStride:]
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				t := cflPairSumBytes(loadRow16(top[ic:]))
				b := cflPairSumBytes(loadRow16(bot[ic:]))
				t.Add(b).ShiftAllLeft(1).StoreArray((*[8]uint16)(outputQ3[outBase+oc:]))
			}
		}
	case subX:
		if outW%8 != 0 {
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		for row := 0; row < outH; row++ {
			outBase := row * CFLBufLine
			in := input[row*inputStride:]
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				cflPairSumBytes(loadRow16(in[ic:])).ShiftAllLeft(2).StoreArray((*[8]uint16)(outputQ3[outBase+oc:]))
			}
		}
	default:
		if outW%8 != 0 {
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		for row := 0; row < outH; row++ {
			outBase := row * CFLBufLine
			in := input[row*inputStride:]
			for oc := 0; oc < outW; oc += 8 {
				archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(in[oc:])).ReshapeToUint8s().ExtendLo8ToUint16().ShiftAllLeft(3).StoreArray((*[8]uint16)(outputQ3[outBase+oc:]))
			}
		}
	}
}

// loadRow16 returns the sixteen bytes at the start of s. Near the end of the
// input buffer fewer bytes may remain; those lanes are never read by the scalar
// reference, so they are zero-filled from a local copy instead of over-reading.
func loadRow16(s []byte) archsimd.Uint8x16 {
	if len(s) >= 16 {
		return archsimd.LoadUint8x16Array((*[16]uint8)(s))
	}
	var buf [16]uint8
	copy(buf[:], s)
	return archsimd.LoadUint8x16Array(&buf)
}
