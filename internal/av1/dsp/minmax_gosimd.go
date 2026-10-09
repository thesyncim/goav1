// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package dsp

// Go-native SIMD (simd/archsimd) MinMaxAbsDiff8x8 shared by arm64 and amd64.
// It is the vector form of minMaxAbsDiff8x8PureGo: the per-lane absolute
// difference is max(a,b)-min(a,b), the running per-lane minimum and maximum
// fold over the eight rows, and a final horizontal reduction gives the block
// extremes. Validation is identical to the reference.
//
// Bit-exactness with minMaxAbsDiff8x8PureGo:
//   - absDiff8/absDiff16 are max-min in unsigned lanes, with no wrap.
//   - the 8-bit row is broadcast into both halves of a 128-bit vector. The
//     duplicate lanes repeat the same eight differences, so the minimum and
//     maximum over sixteen lanes equal those over the eight real samples.
//   - the min/max fold is order-independent and covers all 64 samples.

import (
	"encoding/binary"
	"simd/archsimd"
	"unsafe"
)

// minMaxAbsDiff8x8SIMD is the Go SIMD variant selected by the SIMD dispatchers.
func minMaxAbsDiff8x8SIMD(a []byte, aStride int, b []byte, bStride int, bytesPerSample int) (uint16, uint16, error) {
	if bytesPerSample != 1 && bytesPerSample != 2 {
		return 0, 0, ErrInvalidBlock
	}
	rowBytes := 8 * bytesPerSample
	if aStride < rowBytes || bStride < rowBytes ||
		!byteBlockFits(len(a), aStride, rowBytes, 8) ||
		!byteBlockFits(len(b), bStride, rowBytes, 8) {
		return 0, 0, ErrInvalidBlock
	}
	if bytesPerSample == 1 {
		minv, maxv := minMaxAbs8SIMD(a, aStride, b, bStride)
		return minv, maxv, nil
	}
	minv, maxv := minMaxAbs16SIMD(a, aStride, b, bStride)
	return minv, maxv, nil
}

// minMaxAbs8SIMD handles 8-bit samples. Two rows share one 128-bit vector: row
// 2p fills the low 64-bit half and row 2p+1 the high half, each read as one
// 64-bit word. The caller has validated that all eight rows lie inside a and b,
// so the walk advances raw offsets rather than re-slicing per row.
func minMaxAbs8SIMD(a []byte, aStride int, b []byte, bStride int) (uint16, uint16) {
	ap := unsafe.Pointer(unsafe.SliceData(a))
	bp := unsafe.Pointer(unsafe.SliceData(b))
	vmin := archsimd.BroadcastUint8x16(0xff)
	vmax := archsimd.BroadcastUint8x16(0)
	aOff, bOff := 0, 0
	for range 4 {
		av := archsimd.BroadcastUint64x2(minmaxLoad64(ap, aOff)).
			SetElem(1, minmaxLoad64(ap, aOff+aStride)).ReshapeToUint8s()
		bv := archsimd.BroadcastUint64x2(minmaxLoad64(bp, bOff)).
			SetElem(1, minmaxLoad64(bp, bOff+bStride)).ReshapeToUint8s()
		d := av.Max(bv).Sub(av.Min(bv))
		vmin = vmin.Min(d)
		vmax = vmax.Max(d)
		aOff += 2 * aStride
		bOff += 2 * bStride
	}
	return uint16(minmaxReduceMinU8(vmin)), uint16(minmaxReduceMaxU8(vmax))
}

// minMaxAbs16SIMD handles 16-bit little-endian samples. Each row is one 128-bit
// load of eight u16 lanes; the caller has validated that every row lies inside
// a and b.
func minMaxAbs16SIMD(a []byte, aStride int, b []byte, bStride int) (uint16, uint16) {
	ap := unsafe.Pointer(unsafe.SliceData(a))
	bp := unsafe.Pointer(unsafe.SliceData(b))
	vmin := archsimd.BroadcastUint16x8(0xffff)
	vmax := archsimd.BroadcastUint16x8(0)
	aOff, bOff := 0, 0
	for range 8 {
		av := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(ap, aOff))).ReshapeToUint16s()
		bv := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(bp, bOff))).ReshapeToUint16s()
		d := av.Max(bv).Sub(av.Min(bv))
		vmin = vmin.Min(d)
		vmax = vmax.Max(d)
		aOff += aStride
		bOff += bStride
	}
	return minmaxReduceMinU16(vmin), minmaxReduceMaxU16(vmax)
}

// minmaxLoad64 reads the eight bytes at byte offset off from p, little-endian.
func minmaxLoad64(p unsafe.Pointer, off int) uint64 {
	return binary.LittleEndian.Uint64(unsafe.Slice((*byte)(unsafe.Add(p, off)), 8))
}
