// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package dsp

import (
	"simd/archsimd"
	"unsafe"
)

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
	var aTail, bTail [16]uint8
	aTailOffset, bTailOffset := 7*aStride, 7*bStride
	aLastHas16 := len(a)-aTailOffset >= 16
	bLastHas16 := len(b)-bTailOffset >= 16
	ap := unsafe.Pointer(&a[0])
	bp := unsafe.Pointer(&b[0])
	for row := 0; row < 8; row++ {
		var av, bv archsimd.Uint8x16
		if row == 7 && !aLastHas16 {
			// The final row may have only eight to fifteen accessible bytes.
			// byteBlockFits guarantees the low eight; the zeroed high half is
			// discarded by InterleaveLo below.
			copy(aTail[:8], a[aTailOffset:aTailOffset+8])
			av = archsimd.LoadUint8x16Array(&aTail)
		} else {
			// For rows 0..6, the validated 8x8 extent plus stride >= 8 leaves at
			// least 16 bytes in the slice. Row 7 takes this path only after the
			// explicit remaining-length check above. InterleaveLo discards bytes
			// 8..15, even when they are from the next row.
			av = archsimd.LoadUint8x16Array((*[16]uint8)(ap))
		}
		if row == 7 && !bLastHas16 {
			copy(bTail[:8], b[bTailOffset:bTailOffset+8])
			bv = archsimd.LoadUint8x16Array(&bTail)
		} else {
			bv = archsimd.LoadUint8x16Array((*[16]uint8)(bp))
		}
		// Duplicate the first eight loaded bytes so every lane participates in
		// the official 16-lane reductions without changing the 8x8 result.
		a := av.InterleaveLo(av)
		b := bv.InterleaveLo(bv)
		absd := a.Max(b).Sub(a.Min(b))
		minV = minV.Min(absd)
		maxV = maxV.Max(absd)
		if row < 7 {
			ap = unsafe.Add(ap, aStride)
			bp = unsafe.Add(bp, bStride)
		}
	}
	return uint16(minV.ReduceMin()), uint16(maxV.ReduceMax()), nil
}
