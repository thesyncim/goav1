// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package dsp

// Go-native SIMD (simd/archsimd) AddResidualPlaneBlock and AddRawTransform-
// PlaneBlock kernels shared by arm64 and amd64. The architecture files provide
// the three operations whose instruction forms differ: the signed 32-to-16 pack
// of the raw path, and the two unsigned narrowings that store 8-bit samples.
//
// Shapes: blocks whose width is a multiple of eight run eight samples per
// vector (sixteen for 8-bit residuals with width a multiple of sixteen), and
// width four runs two rows per vector. Any other width routes to
// addResidualPlaneBlockPureGo / addRawTransformPlaneBlockPureGo, as the retired
// assembly did.
//
// Bit-exactness with the scalar references, for the full input domain:
//   - the output is clamp(d + r, 0, max) with d the u16 sample (any value) and
//     r an int16 residual. planeClampAddU16 computes it with saturating u16
//     arithmetic: add the positive part of r saturating at 0xffff, subtract the
//     magnitude of the negative part saturating at zero, then take the minimum
//     with max. Each saturation step leaves the clamped result unchanged.
//   - the raw path rounds with rawTransformResidual, which is int64-exact. The
//     Go SIMD form computes floor((v+8)/16) as (v>>4) + (((v&15)+8)>>4), which
//     cannot overflow int32 for any v, and then saturates to int16.

import (
	"encoding/binary"
	"simd/archsimd"
	"unsafe"
)

// addResidualPlaneBlockSIMD is the Go SIMD residual add for 8-bit and 16-bit
// samples.
func addResidualPlaneBlockSIMD(block planeBlock, bytesPerSample int, max uint16, width int, residual []int16, residualStride int) {
	if bytesPerSample != 1 && bytesPerSample != 2 || (width%8 != 0 && width != 4) {
		addResidualPlaneBlockPureGo(block, bytesPerSample, max, width, residual, residualStride)
		return
	}
	rowBytes := width * bytesPerSample
	zero := archsimd.BroadcastInt16x8(0)
	zeroU := archsimd.BroadcastUint16x8(0)
	maxv := archsimd.BroadcastUint16x8(max)
	if width == 4 {
		rows := block.height
		row := 0
		for ; row+1 < rows; row += 2 {
			dstA := block.pix[row*block.stride : row*block.stride+rowBytes]
			dstB := block.pix[(row+1)*block.stride : (row+1)*block.stride+rowBytes]
			r := planeLoadResidualPair(residual[row*residualStride:], residual[(row+1)*residualStride:])
			d := planeLoadSamplePair(dstA, dstB, bytesPerSample)
			planeStoreSamplePair(planeClampAddU16(d, r, zero, zeroU, maxv), dstA, dstB, bytesPerSample)
		}
		if row < rows {
			addResidualPlaneBlockPureGo(planeBlock{
				pix: block.pix[row*block.stride:], stride: block.stride,
				width: width, height: 1, rowBytes: rowBytes,
			}, bytesPerSample, max, width, residual[row*residualStride:], residualStride)
		}
		return
	}
	for row := range block.height {
		dst := block.pix[row*block.stride : row*block.stride+rowBytes]
		res := residual[row*residualStride : row*residualStride+width]
		// The row slices were validated by the caller, so the loops walk raw
		// pointers: the vector loads and stores stay inside dst and res.
		dp := unsafe.Pointer(&dst[0])
		rp := unsafe.Pointer(&res[0])
		if bytesPerSample == 2 {
			for x := 0; x < width; x += 8 {
				d := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(dp, 2*x))).ReshapeToUint16s()
				r := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rp, 2*x)))
				planeClampAddU16(d, r, zero, zeroU, maxv).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(dp, 2*x)))
			}
			continue
		}
		x := 0
		for ; x+16 <= width; x += 16 {
			v := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(dp, x)))
			lo := planeClampAddU16(v.ExtendLo8ToUint16(), archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rp, 2*x))), zero, zeroU, maxv)
			hi := planeClampAddU16(v.ConcatShiftBytesRight(v, 8).ExtendLo8ToUint16(), archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rp, 2*x+16))), zero, zeroU, maxv)
			planeNarrow16to8(lo, hi).StoreArray((*[16]uint8)(unsafe.Add(dp, x)))
		}
		for ; x < width; x += 8 {
			d := planeLoadSamples8(dst[x:])
			r := archsimd.LoadInt16x8Array((*[8]int16)(res[x:]))
			planeStoreSamples8(planeClampAddU16(d, r, zero, zeroU, maxv), dst[x:])
		}
	}
}

// addRawTransformPlaneBlockSIMD is the Go SIMD fused raw-transform add for 8-bit
// and 16-bit samples.
func addRawTransformPlaneBlockSIMD(block planeBlock, bytesPerSample int, max uint16, width int, raw []int32, rawStride int) {
	if bytesPerSample != 1 && bytesPerSample != 2 || (width%8 != 0 && width != 4) {
		addRawTransformPlaneBlockPureGo(block, bytesPerSample, max, width, raw, rawStride)
		return
	}
	rowBytes := width * bytesPerSample
	zero := archsimd.BroadcastInt16x8(0)
	zeroU := archsimd.BroadcastUint16x8(0)
	maxv := archsimd.BroadcastUint16x8(max)
	c15 := archsimd.BroadcastInt32x4(15)
	c8 := archsimd.BroadcastInt32x4(8)
	sh4 := archsimd.BroadcastInt32x4(-4)
	if width == 4 {
		rows := block.height
		row := 0
		for ; row+1 < rows; row += 2 {
			dstA := block.pix[row*block.stride : row*block.stride+rowBytes]
			dstB := block.pix[(row+1)*block.stride : (row+1)*block.stride+rowBytes]
			r := planePackPairI16(
				planeRawResidual(archsimd.LoadInt32x4Array((*[4]int32)(raw[row*rawStride:])), c15, c8, sh4),
				planeRawResidual(archsimd.LoadInt32x4Array((*[4]int32)(raw[(row+1)*rawStride:])), c15, c8, sh4))
			d := planeLoadSamplePair(dstA, dstB, bytesPerSample)
			planeStoreSamplePair(planeClampAddU16(d, r, zero, zeroU, maxv), dstA, dstB, bytesPerSample)
		}
		if row < rows {
			addRawTransformPlaneBlockPureGo(planeBlock{
				pix: block.pix[row*block.stride:], stride: block.stride,
				width: width, height: 1, rowBytes: rowBytes,
			}, bytesPerSample, max, width, raw[row*rawStride:], rawStride)
		}
		return
	}
	for row := range block.height {
		dst := block.pix[row*block.stride : row*block.stride+rowBytes]
		src := raw[row*rawStride : row*rawStride+width]
		dp := unsafe.Pointer(&dst[0])
		sp := unsafe.Pointer(&src[0])
		for x := 0; x < width; x += 8 {
			r := planePackPairI16(
				planeRawResidual(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(sp, 4*x))), c15, c8, sh4),
				planeRawResidual(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(sp, 4*x+16))), c15, c8, sh4))
			if bytesPerSample == 2 {
				d := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(dp, 2*x))).ReshapeToUint16s()
				planeClampAddU16(d, r, zero, zeroU, maxv).ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(dp, 2*x)))
				continue
			}
			d := planeLoadSamples8(unsafe.Slice((*byte)(unsafe.Add(dp, x)), 8))
			planeStoreSamples8(planeClampAddU16(d, r, zero, zeroU, maxv), unsafe.Slice((*byte)(unsafe.Add(dp, x)), 8))
		}
	}
}

// planeClampAddU16 returns clamp(d + r, 0, maxv) lane by lane for u16 samples d
// and int16 residuals r. zero and zeroU are zero vectors of the two lane types.
// The positive part of r is added with unsigned saturation at 0xffff, the
// magnitude of the negative part is subtracted with saturation at zero, and the
// result is limited to maxv; each step keeps the clamped value exact.
func planeClampAddU16(d archsimd.Uint16x8, r archsimd.Int16x8, zero archsimd.Int16x8, zeroU archsimd.Uint16x8, maxv archsimd.Uint16x8) archsimd.Uint16x8 {
	neg := zero.Greater(r).ToInt16x8().ToBits()
	ru := r.ToBits()
	pos := ru.AndNot(neg)
	mag := zeroU.Sub(ru).And(neg)
	return d.AddSaturated(pos).SubSaturated(mag).Min(maxv)
}

// planeLoadSamples8 loads eight 8-bit samples from the start of src as u16
// lanes. Only the first eight bytes are read.
func planeLoadSamples8(src []byte) archsimd.Uint16x8 {
	return archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(src[:8])).ReshapeToUint8s().ExtendLo8ToUint16()
}

// planeStoreSamples8 stores the low byte of each u16 lane of v to the first
// eight bytes of dst.
func planeStoreSamples8(v archsimd.Uint16x8, dst []byte) {
	binary.LittleEndian.PutUint64(dst[:8], planeNarrowWord(v))
}

// planeLoadSamplePair loads the four samples of two consecutive rows, A then B,
// as eight lanes (A in lanes 0-3, B in lanes 4-7). bytesPerSample selects the
// 8-bit or 16-bit lane type.
func planeLoadSamplePair(a []byte, b []byte, bytesPerSample int) archsimd.Uint16x8 {
	if bytesPerSample == 2 {
		return archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(a[:8])).
			SetElem(1, binary.LittleEndian.Uint64(b[:8])).ReshapeToUint16s()
	}
	w := uint64(binary.LittleEndian.Uint32(a[:4])) | uint64(binary.LittleEndian.Uint32(b[:4]))<<32
	return archsimd.BroadcastUint64x2(w).ReshapeToUint8s().ExtendLo8ToUint16()
}

// planeStoreSamplePair stores the lanes of v back to two rows, the layout that
// planeLoadSamplePair reads.
func planeStoreSamplePair(v archsimd.Uint16x8, a []byte, b []byte, bytesPerSample int) {
	if bytesPerSample == 2 {
		w := v.ReshapeToUint64s()
		binary.LittleEndian.PutUint64(a[:8], w.GetElem(0))
		binary.LittleEndian.PutUint64(b[:8], w.GetElem(1))
		return
	}
	w := planeNarrowWord(v)
	binary.LittleEndian.PutUint32(a[:4], uint32(w))
	binary.LittleEndian.PutUint32(b[:4], uint32(w>>32))
}

// planeLoadResidualPair loads four residuals of each of two rows as eight int16
// lanes, A in lanes 0-3 and B in lanes 4-7.
func planeLoadResidualPair(a []int16, b []int16) archsimd.Int16x8 {
	var t [8]int16
	*(*[4]int16)(t[0:4]) = [4]int16(a[:4])
	*(*[4]int16)(t[4:8]) = [4]int16(b[:4])
	return archsimd.LoadInt16x8Array(&t)
}
