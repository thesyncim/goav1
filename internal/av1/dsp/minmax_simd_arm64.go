// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package dsp

import "simd/archsimd"

// minMaxAbsDiff8x8SIMD is the Go-native-SIMD analogue of minMaxAbsDiff8x8PureGo
// for the 8-bit path: the per-pixel absolute difference |a-b| is Max(a,b)-
// Min(a,b) (exact for unsigned), accumulated as a per-column running min/max
// across the 8 rows, then reduced over the 8 columns. The 16-bit path falls
// back to the scalar reference.
//
// Partial loads zero-fill lanes 8..15, so those lanes carry absdiff 0; the
// final reduction reads only lanes 0..7, so the padding never affects the
// result. Byte-identical to the scalar reference.
func minMaxAbsDiff8x8SIMD(a []byte, aStride int, b []byte, bStride int, bytesPerSample int) (uint16, uint16, error) {
	if bytesPerSample != 1 {
		return minMaxAbsDiff8x8PureGo(a, aStride, b, bStride, bytesPerSample)
	}
	const rowBytes = 8
	if aStride < rowBytes || bStride < rowBytes ||
		!byteBlockFits(len(a), aStride, rowBytes, 8) ||
		!byteBlockFits(len(b), bStride, rowBytes, 8) {
		return 0, 0, ErrInvalidBlock
	}
	minV := archsimd.BroadcastUint8x16(255)
	maxV := archsimd.BroadcastUint8x16(0)
	for row := 0; row < 8; row++ {
		av, _ := archsimd.LoadUint8x16Part(a[row*aStride:])
		bv, _ := archsimd.LoadUint8x16Part(b[row*bStride:])
		absd := av.Max(bv).Sub(av.Min(bv))
		minV = minV.Min(absd)
		maxV = maxV.Max(absd)
	}
	var minArr, maxArr [16]uint8
	minV.StoreArray(&minArr)
	maxV.StoreArray(&maxArr)
	minDiff := uint16(255)
	var maxDiff uint16
	for i := 0; i < 8; i++ {
		if d := uint16(minArr[i]); d < minDiff {
			minDiff = d
		}
		if d := uint16(maxArr[i]); d > maxDiff {
			maxDiff = d
		}
	}
	return minDiff, maxDiff, nil
}
