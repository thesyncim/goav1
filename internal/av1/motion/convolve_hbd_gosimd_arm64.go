// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD HBD interpolation for resident filters with zero endpoint
// taps. The 2D forms touch six coefficient slots and trim the intermediate row
// range from height+7 to height+5. Sharp filters with nonzero endpoints stay
// on the NEON path.
package motion

import (
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

const hbdSIMDVectorSamples = 8

func bindHBD6TapGoSIMD() {
	convolve2DHighBDImpl = convolve2DHighBDGoSIMD
	convolve2DHighBDClampedImpl = convolve2DHighBDClampedGoSIMD
	convolve2DHighBDWithScratchImpl = convolve2DHighBDGoSIMDWithScratch
	convolve2DHighBDClampedWithScratchImpl = convolve2DHighBDClampedGoSIMDWithScratch
}

func bindCompoundHBD6TapGoSIMD() {
	predictInterCompoundRefHighBDToConvBuf2DResidentImpl = predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD
	predictInterCompoundRefHighBDToConvBuf2DClampedImpl = predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD
}

func hbdSIMDShape(width, height int) bool {
	return width >= hbdSIMDVectorSamples && width <= maxBlockSize && width%hbdSIMDVectorSamples == 0 &&
		height > 0 && height <= maxBlockSize
}

// hbdSIMDMAC8 widens eight unsigned 10/12-bit samples to exact signed 32-bit
// products. AV1 HBD samples are <=4095, so their int16 bit patterns are
// positive; coefficient products and sums remain within int32 for the pinned
// interpolation tables (maximum coefficient L1 norm is 240).
func hbdSIMDMAC8(lo, hi archsimd.Int32x4, p unsafe.Pointer, coeff archsimd.Int16x8) (archsimd.Int32x4, archsimd.Int32x4) {
	samples := archsimd.LoadUint16x8Array((*[8]uint16)(p)).BitsToInt16()
	lo = lo.Add(samples.MulWidenLo(coeff))
	// Go 1.27 lowers this pair of HiToLo operands directly to SMULL2.
	hi = hi.Add(samples.HiToLo().MulWidenLo(coeff.HiToLo()))
	return lo, hi
}

func hbdSIMDClip(v, zero, max archsimd.Int32x4) archsimd.Int32x4 {
	return v.Max(zero).Min(max)
}

// hbdSIMDJoinU16 merges the low four values of lo and hi into eight adjacent
// uint16 lanes. Truncation preserves the scalar uint16 cast used by CONV_BUF;
// clipped pixel outputs call hbdSIMDClip first and use the same join.
func hbdSIMDJoinU16(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	lo16 := lo.TruncToInt16().ToBits().ReshapeToUint64s()
	hi16 := hi.TruncToInt16().ToBits().ReshapeToUint64s()
	return lo16.InterleaveLo(hi16).ReshapeToUint16s()
}

func hbdSIMDStorePixel8(p unsafe.Pointer, lo, hi, zero, max archsimd.Int32x4) {
	packed := hbdSIMDJoinU16(hbdSIMDClip(lo, zero, max), hbdSIMDClip(hi, zero, max))
	packed.StoreArray((*[8]uint16)(p))
}

func hbdSIMDStoreConvBuf8(p unsafe.Pointer, lo, hi archsimd.Int32x4) {
	hbdSIMDJoinU16(lo, hi).StoreArray((*[8]uint16)(p))
}

func convolve2DHighBDGoSIMD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2DHighBDGoSIMDWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

func convolve2DHighBDGoSIMDWithScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	if !hbdSIMDShape(width, height) {
		convolve2DHighBDNEONWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if xKernel[0] != 0 || xKernel[filterTaps-1] != 0 || yKernel[0] != 0 || yKernel[filterTaps-1] != 0 {
		convolve2DHighBDNEONWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if scratch != nil {
		convolve2DHighBD6SIMDWithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &scratch.imHBD[0])
		return
	}
	convolve2DHighBD6SIMDNoScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
}

//go:noinline
func convolve2DHighBD6SIMDNoScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	var im [(maxBlockSize + filterTaps - 1) * maxBlockSize]int32
	convolve2DHighBD6SIMDWithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &im[0])
}

func convolve2DHighBD6SIMDWithIM(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, im *int32) {
	const imStride = maxBlockSize
	round0, round1 := highBDRoundBits(bitDepth)
	// highBDRoundBits preserves round0+round1=2*filterBits, so the scalar
	// final round has a zero-bit shift and is an identity operation.
	xBias := archsimd.BroadcastInt32x4(int32((1 << (int(bitDepth) + filterBits - 1)) + (1 << (round0 - 1))))
	round0Shift := archsimd.BroadcastInt32x4(-int32(round0))
	zero := archsimd.BroadcastInt32x4(0)
	x0, x1, x2 := archsimd.BroadcastInt16x8(xKernel[1]), archsimd.BroadcastInt16x8(xKernel[2]), archsimd.BroadcastInt16x8(xKernel[3])
	x3, x4, x5 := archsimd.BroadcastInt16x8(xKernel[4]), archsimd.BroadcastInt16x8(xKernel[5]), archsimd.BroadcastInt16x8(xKernel[6])
	y0, y1, y2 := archsimd.BroadcastInt32x4(int32(yKernel[1])), archsimd.BroadcastInt32x4(int32(yKernel[2])), archsimd.BroadcastInt32x4(int32(yKernel[3]))
	y3, y4, y5 := archsimd.BroadcastInt32x4(int32(yKernel[4])), archsimd.BroadcastInt32x4(int32(yKernel[5])), archsimd.BroadcastInt32x4(int32(yKernel[6]))

	// Both zero endpoint coefficients trim one source row at either side, so
	// H creates only height+5 rows. im row zero corresponds to reference y-2.
	imBase := unsafe.Pointer(im)
	for y := 0; y < height+5; y++ {
		row := unsafe.Pointer(&ref.Pix[(refY-2+y)*ref.Stride+(refX-2)*2])
		for x := 0; x < width; x += hbdSIMDVectorSamples {
			base := unsafe.Add(row, uintptr(x*2))
			lo, hi := zero, zero
			lo, hi = hbdSIMDMAC8(lo, hi, base, x0)
			lo, hi = hbdSIMDMAC8(lo, hi, unsafe.Add(base, 2), x1)
			lo, hi = hbdSIMDMAC8(lo, hi, unsafe.Add(base, 4), x2)
			lo, hi = hbdSIMDMAC8(lo, hi, unsafe.Add(base, 6), x3)
			lo, hi = hbdSIMDMAC8(lo, hi, unsafe.Add(base, 8), x4)
			lo, hi = hbdSIMDMAC8(lo, hi, unsafe.Add(base, 10), x5)
			lo = lo.Add(xBias).Shift(round0Shift)
			hi = hi.Add(xBias).Shift(round0Shift)
			lo.StoreArray((*[4]int32)(unsafe.Add(imBase, uintptr((y*imStride+x)*4))))
			hi.StoreArray((*[4]int32)(unsafe.Add(imBase, uintptr((y*imStride+x+4)*4))))
		}
	}

	offsetBits := int(bitDepth) + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - round1)) + (1 << (offsetBits - round1 - 1))
	yBias := archsimd.BroadcastInt32x4(int32((1 << offsetBits) + (1 << (round1 - 1))))
	round1Shift := archsimd.BroadcastInt32x4(-int32(round1))
	roundOffsetV := archsimd.BroadcastInt32x4(int32(roundOffset))
	zero = archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(int32(max))
	for y := 0; y < height; y++ {
		dstRow := unsafe.Pointer(&dst.Pix[(dstY+y)*dst.Stride+dstX*2])
		for x := 0; x < width; x += hbdSIMDVectorSamples {
			lo, hi := zero, zero
			row0 := unsafe.Add(imBase, uintptr((y*imStride+x)*4))
			row1 := unsafe.Add(row0, uintptr(imStride*4))
			row2 := unsafe.Add(row1, uintptr(imStride*4))
			row3 := unsafe.Add(row2, uintptr(imStride*4))
			row4 := unsafe.Add(row3, uintptr(imStride*4))
			row5 := unsafe.Add(row4, uintptr(imStride*4))
			lo = archsimd.LoadInt32x4Array((*[4]int32)(row0)).MulAdd(y0, lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(row0, 16))).MulAdd(y0, hi)
			lo = archsimd.LoadInt32x4Array((*[4]int32)(row1)).MulAdd(y1, lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(row1, 16))).MulAdd(y1, hi)
			lo = archsimd.LoadInt32x4Array((*[4]int32)(row2)).MulAdd(y2, lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(row2, 16))).MulAdd(y2, hi)
			lo = archsimd.LoadInt32x4Array((*[4]int32)(row3)).MulAdd(y3, lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(row3, 16))).MulAdd(y3, hi)
			lo = archsimd.LoadInt32x4Array((*[4]int32)(row4)).MulAdd(y4, lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(row4, 16))).MulAdd(y4, hi)
			lo = archsimd.LoadInt32x4Array((*[4]int32)(row5)).MulAdd(y5, lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(row5, 16))).MulAdd(y5, hi)
			lo = lo.Add(yBias).Shift(round1Shift).Sub(roundOffsetV)
			hi = hi.Add(yBias).Shift(round1Shift).Sub(roundOffsetV)
			hbdSIMDStorePixel8(unsafe.Add(dstRow, uintptr(x*2)), lo, hi, zero, maxV)
		}
	}
}

func convolve2DHighBDClampedGoSIMD(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	convolve2DHighBDClampedGoSIMDWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, nil)
}

func convolve2DHighBDClampedGoSIMDWithScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, scratch *ConvolveScratch) {
	if !hbdSIMDShape(width, height) {
		convolve2DHighBDClampedNEONWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if xKernel[0] != 0 || xKernel[filterTaps-1] != 0 || yKernel[0] != 0 || yKernel[filterTaps-1] != 0 {
		convolve2DHighBDClampedNEONWithScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, scratch)
		return
	}
	if planeRegionFits(ref, 2, refX-2, refY-2, width+5, height+5) {
		if scratch != nil {
			convolve2DHighBD6SIMDWithIM(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel, &scratch.imHBD[0])
			return
		}
		convolve2DHighBD6SIMDNoScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
		return
	}
	if scratch != nil {
		emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &scratch.edge16)
		convolve2DHighBD6SIMDWithIM(dst, emu, bitDepth, max, dstX, dstY, emuX, emuY, width, height, xKernel, yKernel, &scratch.imHBD[0])
		return
	}
	convolve2DHighBD6SIMDClampedNoScratch(dst, ref, bitDepth, max, dstX, dstY, refX, refY, width, height, xKernel, yKernel)
}

//go:noinline
func convolve2DHighBD6SIMDClampedNoScratch(dst frame.Plane, ref frame.Plane, bitDepth uint8, max uint16, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	var im [(maxBlockSize + filterTaps - 1) * maxBlockSize]int32
	var edge emuEdge16Buf
	emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &edge)
	convolve2DHighBD6SIMDWithIM(dst, emu, bitDepth, max, dstX, dstY, emuX, emuY, width, height, xKernel, yKernel, &im[0])
}

func predictInterCompoundRefHighBDToConvBuf2DResidentGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, bitDepth int, im *compoundIM) {
	if !hbdSIMDShape(width, height) {
		predictInterCompoundRefHighBDToConvBuf2DResidentNEON(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	if xKernel[0] != 0 || xKernel[filterTaps-1] != 0 || yKernel[0] != 0 || yKernel[filterTaps-1] != 0 {
		predictInterCompoundRefHighBDToConvBuf2DResidentNEON(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	predictInterCompoundRefHighBDToConvBuf2D6SIMD(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
}

func predictInterCompoundRefHighBDToConvBuf2D6SIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, bitDepth int, im *compoundIM) {
	const imStride = maxBlockSize
	xBias := archsimd.BroadcastInt32x4(int32((1 << (bitDepth + filterBits - 1)) + (1 << (round0 - 1))))
	round0Shift := archsimd.BroadcastInt32x4(-int32(round0))
	zero := archsimd.BroadcastInt32x4(0)
	x0, x1, x2 := archsimd.BroadcastInt16x8(xKernel[1]), archsimd.BroadcastInt16x8(xKernel[2]), archsimd.BroadcastInt16x8(xKernel[3])
	x3, x4, x5 := archsimd.BroadcastInt16x8(xKernel[4]), archsimd.BroadcastInt16x8(xKernel[5]), archsimd.BroadcastInt16x8(xKernel[6])
	y0, y1, y2 := archsimd.BroadcastInt32x4(int32(yKernel[1])), archsimd.BroadcastInt32x4(int32(yKernel[2])), archsimd.BroadcastInt32x4(int32(yKernel[3]))
	y3, y4, y5 := archsimd.BroadcastInt32x4(int32(yKernel[4])), archsimd.BroadcastInt32x4(int32(yKernel[5])), archsimd.BroadcastInt32x4(int32(yKernel[6]))
	imBase := unsafe.Pointer(&im[0])
	for y := 0; y < height+5; y++ {
		row := unsafe.Pointer(&ref.Pix[(refY-2+y)*ref.Stride+(refX-2)*2])
		for x := 0; x < width; x += hbdSIMDVectorSamples {
			base := unsafe.Add(row, uintptr(x*2))
			lo, hi := zero, zero
			lo, hi = hbdSIMDMAC8(lo, hi, base, x0)
			lo, hi = hbdSIMDMAC8(lo, hi, unsafe.Add(base, 2), x1)
			lo, hi = hbdSIMDMAC8(lo, hi, unsafe.Add(base, 4), x2)
			lo, hi = hbdSIMDMAC8(lo, hi, unsafe.Add(base, 6), x3)
			lo, hi = hbdSIMDMAC8(lo, hi, unsafe.Add(base, 8), x4)
			lo, hi = hbdSIMDMAC8(lo, hi, unsafe.Add(base, 10), x5)
			lo = lo.Add(xBias).Shift(round0Shift)
			hi = hi.Add(xBias).Shift(round0Shift)
			lo.StoreArray((*[4]int32)(unsafe.Add(imBase, uintptr((y*imStride+x)*4))))
			hi.StoreArray((*[4]int32)(unsafe.Add(imBase, uintptr((y*imStride+x+4)*4))))
		}
	}

	yBias := archsimd.BroadcastInt32x4(int32((1 << offsetBits) + (1 << (compoundRound1Bits - 1))))
	round1Shift := archsimd.BroadcastInt32x4(-int32(compoundRound1Bits))
	for y := 0; y < height; y++ {
		outRow := unsafe.Pointer(&out[y*width])
		for x := 0; x < width; x += hbdSIMDVectorSamples {
			lo, hi := zero, zero
			row0 := unsafe.Add(imBase, uintptr((y*imStride+x)*4))
			row1 := unsafe.Add(row0, uintptr(imStride*4))
			row2 := unsafe.Add(row1, uintptr(imStride*4))
			row3 := unsafe.Add(row2, uintptr(imStride*4))
			row4 := unsafe.Add(row3, uintptr(imStride*4))
			row5 := unsafe.Add(row4, uintptr(imStride*4))
			lo = archsimd.LoadInt32x4Array((*[4]int32)(row0)).MulAdd(y0, lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(row0, 16))).MulAdd(y0, hi)
			lo = archsimd.LoadInt32x4Array((*[4]int32)(row1)).MulAdd(y1, lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(row1, 16))).MulAdd(y1, hi)
			lo = archsimd.LoadInt32x4Array((*[4]int32)(row2)).MulAdd(y2, lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(row2, 16))).MulAdd(y2, hi)
			lo = archsimd.LoadInt32x4Array((*[4]int32)(row3)).MulAdd(y3, lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(row3, 16))).MulAdd(y3, hi)
			lo = archsimd.LoadInt32x4Array((*[4]int32)(row4)).MulAdd(y4, lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(row4, 16))).MulAdd(y4, hi)
			lo = archsimd.LoadInt32x4Array((*[4]int32)(row5)).MulAdd(y5, lo)
			hi = archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(row5, 16))).MulAdd(y5, hi)
			lo = lo.Add(yBias).Shift(round1Shift)
			hi = hi.Add(yBias).Shift(round1Shift)
			hbdSIMDStoreConvBuf8(unsafe.Add(outRow, uintptr(x*2)), lo, hi)
		}
	}
}

func predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMD(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, bitDepth int, im *compoundIM, edge *emuEdge16Buf) {
	if !hbdSIMDShape(width, height) {
		predictInterCompoundRefHighBDToConvBuf2DClampedNEON(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im, edge)
		return
	}
	if xKernel[0] != 0 || xKernel[filterTaps-1] != 0 || yKernel[0] != 0 || yKernel[filterTaps-1] != 0 {
		predictInterCompoundRefHighBDToConvBuf2DClampedNEON(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im, edge)
		return
	}
	if planeRegionFits(ref, 2, refX-2, refY-2, width+5, height+5) {
		predictInterCompoundRefHighBDToConvBuf2D6SIMD(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	if edge == nil {
		predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMDNoEdge(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
		return
	}
	emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, edge)
	predictInterCompoundRefHighBDToConvBuf2D6SIMD(out, emu, emuX, emuY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
}

//go:noinline
func predictInterCompoundRefHighBDToConvBuf2DClampedGoSIMDNoEdge(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, bitDepth int, im *compoundIM) {
	var edge emuEdge16Buf
	emu, emuX, emuY := emuEdgeWindow16(ref, refX, refY, width, height, &edge)
	predictInterCompoundRefHighBDToConvBuf2D6SIMD(out, emu, emuX, emuY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, im)
}
