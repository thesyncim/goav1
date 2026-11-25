// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import (
	"encoding/binary"
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"simd/archsimd"
)

// Go-native SIMD CfL apply, DC sum kernels for arm64. They mirror the scalar
// contracts in intra_kernels.go and cfl.go:
//   - applyCFL adds roundPowerOfTwoSigned(alpha*ac, 6) to the chroma DC sample and
//     clamps to [0,max]. The product is formed in int32 lanes (MulWidenLo) and the
//     signed round is (|x|+32)>>6 with the sign restored by (r^s)-s.
//   - sumSamples accumulates uint16 samples into uint32 lanes; up to 65536 samples
//     the lane sums cannot exceed 32 bits, so the total is exact.

// loadBytes16 gathers the 16 bytes b[0:16] as eight little-endian uint16 lanes.
func loadBytes16(b []byte) archsimd.Uint16x8 {
	return archsimd.LoadUint64x2Array(&[2]uint64{
		binary.LittleEndian.Uint64(b[0:]),
		binary.LittleEndian.Uint64(b[8:]),
	}).ReshapeToUint16s()
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

// predictFilterIntraBlockDirect10TBL is the 10-bit filter-intra predictor in
// dav1d's ipred_filter_10bpc_neon shape: at 10 bits the seven-tap sums of
// every filter-intra mode fit int16 (filterIntra10SumFitsInt16), so, as in
// dav1d, each 4x2 batch is seven int16 multiply-accumulates over the eight
// outputs with the neighbours broadcast from vector lanes (TBL), the rounding
// bias in the accumulator, an arithmetic shift and the [0, 1023] clamp. The
// serial dependency from one batch's outputs to the next batch's p5/p6 and to
// the next row pair's top taps stays in vector registers, as in the 8-bit
// predictFilterIntraBlockDirect8TBL. The int16 sums wrap only in partial
// sums; the final value is exact, so the result equals the scalar
// roundPowerOfTwo(sum, 4) clamped to [0, max]. Other depths and shapes take
// the 32-bit kernel.
func predictFilterIntraBlockDirect10TBL(block planeBlock, width int, height int, mode FilterIntraMode, edges IntraEdges, max int) {
	if width%4 != 0 || width > 32 || height%2 != 0 || max != 1023 || len(edges.Above) < width || len(edges.Left) < height {
		predictFilterIntraBlockDirect16SIMD(block, width, height, mode, edges, max)
		return
	}
	taps := &filterIntraTapsSIMD[mode]
	t0 := archsimd.LoadInt16x8Array(&taps[0])
	t1 := archsimd.LoadInt16x8Array(&taps[1])
	t2 := archsimd.LoadInt16x8Array(&taps[2])
	t3 := archsimd.LoadInt16x8Array(&taps[3])
	t4 := archsimd.LoadInt16x8Array(&taps[4])
	t5 := archsimd.LoadInt16x8Array(&taps[5])
	t6 := archsimd.LoadInt16x8Array(&taps[6])
	roundV := archsimd.BroadcastInt16x8(1 << (filterIntraScaleBits - 1))
	zeroV := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(1023)
	shr := filterIntraShr4
	l0 := archsimd.LoadUint8x16Array(&filterIntraLane[0])
	l1 := archsimd.LoadUint8x16Array(&filterIntraLane[1])
	l2 := archsimd.LoadUint8x16Array(&filterIntraLane[2])
	l3 := archsimd.LoadUint8x16Array(&filterIntraLane[3])
	l4 := archsimd.LoadUint8x16Array(&filterIntraLane[4])
	l5 := archsimd.LoadUint8x16Array(&filterIntraLane[5])
	l6 := archsimd.LoadUint8x16Array(&filterIntraLane[6])
	l7 := archsimd.LoadUint8x16Array(&filterIntraLane[7])

	// top0[0] is the top-left sample and top0[1+i] the above row, so batch b
	// reads p0..p4 as lanes 0..4 of top0[4b : 4b+8].
	var top0 [40]int16
	top0[0] = int16(edges.AboveLeft)
	for i, v := range edges.Above[:width] {
		top0[1+i] = int16(v)
	}
	// prev[b] is batch b's output of the previous row pair: lanes 4..7 are the
	// row above the next pair at columns 4b..4b+3.
	var prev [8]archsimd.Int16x8
	blocks := width / 4
	for row := 0; row < height; row += 2 {
		row0 := block.pix[row*block.stride:][:2*width]
		row1 := block.pix[(row+1)*block.stride:][:2*width]
		p5 := archsimd.BroadcastInt16x8(int16(edges.Left[row]))
		p6 := archsimd.BroadcastInt16x8(int16(edges.Left[row+1]))
		var left archsimd.Int16x8 // previous row pair's output of batch b-1
		if row > 0 {
			left = archsimd.BroadcastInt16x8(int16(edges.Left[row-1]))
		}
		for b := range prev[:blocks] {
			var q0, q1, q2, q3, q4 archsimd.Int16x8
			if row == 0 {
				v := archsimd.LoadInt16x8Array((*[8]int16)(top0[4*b : 4*b+8]))
				q0, q1, q2, q3, q4 = filterIntraBcast(v, l0), filterIntraBcast(v, l1), filterIntraBcast(v, l2), filterIntraBcast(v, l3), filterIntraBcast(v, l4)
			} else {
				above := prev[b]
				q1, q2, q3, q4 = filterIntraBcast(above, l4), filterIntraBcast(above, l5), filterIntraBcast(above, l6), filterIntraBcast(above, l7)
				if b == 0 {
					q0 = left
				} else {
					q0 = filterIntraBcast(left, l7)
				}
				left = above
			}
			base := t4.MulAdd(q4, t3.MulAdd(q3, t2.MulAdd(q2, t1.MulAdd(q1, t0.MulAdd(q0, roundV)))))
			out := t6.MulAdd(p6, t5.MulAdd(p5, base)).Shift(shr).Max(zeroV).Min(maxV)
			prev[b] = out
			p5, p6 = filterIntraBcast(out, l3), filterIntraBcast(out, l7)
			packed := out.ToBits().ReshapeToUint64s()
			binary.LittleEndian.PutUint64(row0[8*b:], packed.GetElem(0))
			binary.LittleEndian.PutUint64(row1[8*b:], packed.GetElem(1))
		}
	}
}

// predictFilterIntraBlockDirect16SIMD is the Go-native SIMD form of
// predictFilterIntraBlockDirect16 for high-bit-depth samples. It mirrors the
// 8-bit kernel but accumulates each output in Int32x4 lanes (MulWidenLo over the
// low and high halves of the tap vector) because 12-bit tap sums overflow int16.
// The rounding shift and the [0,max] clamp run in 32-bit lanes before the
// narrowing pack, so the result is the scalar roundPowerOfTwo(sum, 4) clamped to
// [0,max] sample for sample.
//
// As in the 8-bit kernel, the taps and neighbours are individual locals so the
// batch loop keeps them in vector registers.
func predictFilterIntraBlockDirect16SIMD(block planeBlock, width int, height int, mode FilterIntraMode, edges IntraEdges, max int) {
	if width%4 != 0 {
		predictFilterIntraBlockDirect16(block, width, height, mode, edges, max)
		return
	}
	taps := &filterIntraTapsSIMD[mode]
	t0 := archsimd.LoadInt16x8Array(&taps[0])
	t1 := archsimd.LoadInt16x8Array(&taps[1])
	t2 := archsimd.LoadInt16x8Array(&taps[2])
	t3 := archsimd.LoadInt16x8Array(&taps[3])
	t4 := archsimd.LoadInt16x8Array(&taps[4])
	t5 := archsimd.LoadInt16x8Array(&taps[5])
	t6 := archsimd.LoadInt16x8Array(&taps[6])
	// The high halves of the taps feed outputs 4..7 through the same widening
	// multiply once moved into the low lanes.
	h0, h1, h2, h3 := t0.HiToLo(), t1.HiToLo(), t2.HiToLo(), t3.HiToLo()
	h4, h5, h6 := t4.HiToLo(), t5.HiToLo(), t6.HiToLo()
	roundV := archsimd.BroadcastInt32x4(1 << (filterIntraScaleBits - 1))
	zeroV := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(int32(max))

	// above follows predictFilterIntraBlockDirect16: above[0] is p0 of the
	// first batch and above[i+1] is the top row sample i.
	var above [33]uint16
	for row := 0; row < height; row += 2 {
		if row == 0 {
			above[0] = edges.AboveLeft
			for i := 0; i < width; i++ {
				above[i+1] = edges.Above[i]
			}
		} else {
			above[0] = edges.Left[row-1]
			top := block.pix[(row-1)*block.stride:]
			for i := 0; i < width; i++ {
				above[i+1] = binary.LittleEndian.Uint16(top[i<<1:])
			}
		}
		row0 := block.pix[row*block.stride:][:width*2]
		row1 := block.pix[(row+1)*block.stride:][:width*2]
		p5 := archsimd.BroadcastInt16x8(int16(edges.Left[row]))
		p6 := archsimd.BroadcastInt16x8(int16(edges.Left[row+1]))
		for col := 0; col < width; col += 4 {
			a := above[col:]
			_ = a[4]
			p0 := archsimd.BroadcastInt16x8(int16(a[0]))
			p1 := archsimd.BroadcastInt16x8(int16(a[1]))
			p2 := archsimd.BroadcastInt16x8(int16(a[2]))
			p3 := archsimd.BroadcastInt16x8(int16(a[3]))
			p4 := archsimd.BroadcastInt16x8(int16(a[4]))
			// Top-row terms first, off the chain; the left-column terms follow.
			lo := t0.MulWidenLo(p0).Add(t1.MulWidenLo(p1)).Add(t2.MulWidenLo(p2)).
				Add(t3.MulWidenLo(p3)).Add(t4.MulWidenLo(p4)).Add(roundV)
			hi := h0.MulWidenLo(p0).Add(h1.MulWidenLo(p1)).Add(h2.MulWidenLo(p2)).
				Add(h3.MulWidenLo(p3)).Add(h4.MulWidenLo(p4)).Add(roundV)
			lo = lo.Add(t5.MulWidenLo(p5)).Add(t6.MulWidenLo(p6))
			hi = hi.Add(h5.MulWidenLo(p5)).Add(h6.MulWidenLo(p6))
			loOut := lo.Shift(filterIntraShr4I32).Max(zeroV).Min(maxV)
			hiOut := hi.Shift(filterIntraShr4I32).Max(zeroV).Min(maxV)
			out := cflTruncateInt32PairToInt16(loOut, hiOut)
			packed := out.ToBits().ReshapeToUint64s()
			binary.LittleEndian.PutUint64(row0[col<<1:], packed.GetElem(0))
			binary.LittleEndian.PutUint64(row1[col<<1:], packed.GetElem(1))
			p5 = archsimd.BroadcastInt16x8(out.GetElem(3))
			p6 = archsimd.BroadcastInt16x8(out.GetElem(7))
		}
	}
}

// filterIntraShr4I32 is the 32-bit rounding shift as a per-lane shift vector.
var filterIntraShr4I32 = archsimd.BroadcastInt32x4(-filterIntraScaleBits)

// filterIntraLane[L] is the TBL index vector that broadcasts int16 lane L of a
// vector to all eight lanes (bytes 2L, 2L+1 repeated).
var filterIntraLane [8][16]uint8

func init() {
	for l := range filterIntraLane {
		for k := 0; k < 8; k++ {
			filterIntraLane[l][2*k] = uint8(2 * l)
			filterIntraLane[l][2*k+1] = uint8(2*l + 1)
		}
	}
}

// filterIntraBcast broadcasts the int16 lane selected by idx to all lanes of v
// (one TBL, no general-register round trip).
func filterIntraBcast(v archsimd.Int16x8, idx archsimd.Uint8x16) archsimd.Int16x8 {
	return v.ToBits().ReshapeToUint8s().LookupOrZero(idx).ReshapeToUint16s().BitsToInt16()
}

// predictFilterIntraBlockDirect8TBL is predictFilterIntraBlockDirect8SIMD in
// dav1d's ipred_filter_8bpc_neon shape: the seven neighbours of each 4x2 batch
// are vector lanes multiplied in place (dav1d's by-element MLA, here a TBL lane
// broadcast feeding MLA), so the serial dependency from one batch's outputs to
// the next batch's p5/p6 never leaves the vector registers. The previous row
// pair's outputs stay as vectors and supply the next pair's top taps the same
// way; only the left column and the first row come from the edge arrays.
func predictFilterIntraBlockDirect8TBL(block planeBlock, width int, height int, mode FilterIntraMode, edges IntraEdges, max int) {
	if width%4 != 0 || width > 32 || height%2 != 0 || max != 0xff || len(edges.Above) < width || len(edges.Left) < height {
		predictFilterIntraBlockDirect8(block, width, height, mode, edges, max)
		return
	}
	taps := &filterIntraTapsSIMD[mode]
	t0 := archsimd.LoadInt16x8Array(&taps[0])
	t1 := archsimd.LoadInt16x8Array(&taps[1])
	t2 := archsimd.LoadInt16x8Array(&taps[2])
	t3 := archsimd.LoadInt16x8Array(&taps[3])
	t4 := archsimd.LoadInt16x8Array(&taps[4])
	t5 := archsimd.LoadInt16x8Array(&taps[5])
	t6 := archsimd.LoadInt16x8Array(&taps[6])
	roundV := archsimd.BroadcastInt16x8(1 << (filterIntraScaleBits - 1))
	zeroV := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(0xff)
	shr := filterIntraShr4
	l0 := archsimd.LoadUint8x16Array(&filterIntraLane[0])
	l1 := archsimd.LoadUint8x16Array(&filterIntraLane[1])
	l2 := archsimd.LoadUint8x16Array(&filterIntraLane[2])
	l3 := archsimd.LoadUint8x16Array(&filterIntraLane[3])
	l4 := archsimd.LoadUint8x16Array(&filterIntraLane[4])
	l5 := archsimd.LoadUint8x16Array(&filterIntraLane[5])
	l6 := archsimd.LoadUint8x16Array(&filterIntraLane[6])
	l7 := archsimd.LoadUint8x16Array(&filterIntraLane[7])

	// top0[0] is the top-left sample and top0[1+i] the above row, so batch b
	// reads p0..p4 as lanes 0..4 of top0[4b : 4b+8].
	var top0 [40]int16
	top0[0] = int16(edges.AboveLeft)
	for i, v := range edges.Above[:width] {
		top0[1+i] = int16(v)
	}
	// prev[b] is batch b's output of the previous row pair: lanes 4..7 are the
	// row above the next pair at columns 4b..4b+3.
	var prev [8]archsimd.Int16x8
	blocks := width / 4
	for row := 0; row < height; row += 2 {
		row0 := block.pix[row*block.stride:][:width]
		row1 := block.pix[(row+1)*block.stride:][:width]
		p5 := archsimd.BroadcastInt16x8(int16(edges.Left[row]))
		p6 := archsimd.BroadcastInt16x8(int16(edges.Left[row+1]))
		var left archsimd.Int16x8 // previous row pair's output of batch b-1
		if row > 0 {
			left = archsimd.BroadcastInt16x8(int16(edges.Left[row-1]))
		}
		for b := range prev[:blocks] {
			var q0, q1, q2, q3, q4 archsimd.Int16x8
			if row == 0 {
				v := archsimd.LoadInt16x8Array((*[8]int16)(top0[4*b : 4*b+8]))
				q0, q1, q2, q3, q4 = filterIntraBcast(v, l0), filterIntraBcast(v, l1), filterIntraBcast(v, l2), filterIntraBcast(v, l3), filterIntraBcast(v, l4)
			} else {
				above := prev[b]
				q1, q2, q3, q4 = filterIntraBcast(above, l4), filterIntraBcast(above, l5), filterIntraBcast(above, l6), filterIntraBcast(above, l7)
				if b == 0 {
					q0 = left
				} else {
					q0 = filterIntraBcast(left, l7)
				}
				left = above
			}
			base := t4.MulAdd(q4, t3.MulAdd(q3, t2.MulAdd(q2, t1.MulAdd(q1, t0.MulAdd(q0, roundV)))))
			out := t6.MulAdd(p6, t5.MulAdd(p5, base)).Shift(shr).Max(zeroV).Min(maxV)
			prev[b] = out
			p5, p6 = filterIntraBcast(out, l3), filterIntraBcast(out, l7)
			packed := filterIntraPackBytes(out).ReshapeToUint32s()
			binary.LittleEndian.PutUint32(row0[4*b:], packed.GetElem(0))
			binary.LittleEndian.PutUint32(row1[4*b:], packed.GetElem(1))
		}
	}
}

// filterIntraPackBytes narrows eight int16 filter outputs, each already clamped
// to [0,255], into the low eight bytes of the result (UQXTN).
func filterIntraPackBytes(out archsimd.Int16x8) archsimd.Uint8x16 {
	return out.ConvertToUint16().SaturateToUint8()
}

// filterIntraShr4 is the rounding shift (by filterIntraScaleBits) as a per-lane
// shift vector: the arm64 ShiftAllRight lowering re-broadcasts the amount for
// every use, whereas a per-lane Shift against a hoisted vector is one VSSHL.
var filterIntraShr4 = archsimd.BroadcastInt16x8(-filterIntraScaleBits)

// filterIntraShift4 arithmetic-shifts each int16 lane right by filterIntraScaleBits.
func filterIntraShift4(v archsimd.Int16x8) archsimd.Int16x8 {
	return v.Shift(filterIntraShr4)
}

// widenEdge8 widens the bytes src[:len(dst)] into dst. It loads sixteen bytes
// per vector while src holds them, so src may extend past len(dst).
func widenEdge8(dst []uint16, src []byte) {
	i := 0
	for ; i+16 <= len(dst); i += 16 {
		v := archsimd.LoadUint8x16Array((*[16]uint8)(src[i:]))
		v.ExtendLo8ToUint16().StoreArray((*[8]uint16)(dst[i:]))
		v.HiToLo().ExtendLo8ToUint16().StoreArray((*[8]uint16)(dst[i+8:]))
	}
	if i+8 <= len(dst) && i+16 <= len(src) {
		archsimd.LoadUint8x16Array((*[16]uint8)(src[i:])).ExtendLo8ToUint16().StoreArray((*[8]uint16)(dst[i:]))
		i += 8
	}
	for ; i < len(dst); i++ {
		dst[i] = uint16(src[i])
	}
}

// fillWide8, verticalWide8 and horizontalWide8 store 16, 32 or 64 wide rows
// sixteen bytes per vector store, as dav1d's st1 {v0.16b, ...} row stores.

func fillWide8(block planeBlock, value byte) {
	v := archsimd.BroadcastUint8x16(value)
	storeRows16(block.pix, block.stride, block.width, block.height, v, v, v, v)
}

func verticalWide8(block planeBlock, top []byte) {
	w := block.width
	top = top[:w]
	v0 := archsimd.LoadUint8x16Array((*[16]uint8)(top))
	v1, v2, v3 := v0, v0, v0
	if w >= 32 {
		v1 = archsimd.LoadUint8x16Array((*[16]uint8)(top[16:]))
	}
	if w == 64 {
		v2 = archsimd.LoadUint8x16Array((*[16]uint8)(top[32:]))
		v3 = archsimd.LoadUint8x16Array((*[16]uint8)(top[48:]))
	}
	storeRows16(block.pix, block.stride, w, block.height, v0, v1, v2, v3)
}

// storeRows16 stores h rows of w bytes (16, 32 or 64), v0..v3 per sixteen
// bytes of each row.
func storeRows16(pix []byte, stride int, w int, h int, v0, v1, v2, v3 archsimd.Uint8x16) {
	switch w {
	case 16:
		for o := 0; h > 0; h-- {
			v0.StoreArray((*[16]uint8)(pix[o : o+16]))
			o += stride
		}
	case 32:
		for o := 0; h > 0; h-- {
			line := pix[o : o+32]
			v0.StoreArray((*[16]uint8)(line[:16]))
			v1.StoreArray((*[16]uint8)(line[16:]))
			o += stride
		}
	default:
		for o := 0; h > 0; h-- {
			line := pix[o : o+64]
			v0.StoreArray((*[16]uint8)(line[:16]))
			v1.StoreArray((*[16]uint8)(line[16:32]))
			v2.StoreArray((*[16]uint8)(line[32:48]))
			v3.StoreArray((*[16]uint8)(line[48:]))
			o += stride
		}
	}
}

func horizontalWide8(block planeBlock, left []byte) {
	pix, stride, w := block.pix, block.stride, block.width
	o := 0
	for _, l := range left {
		v := archsimd.BroadcastUint8x16(l)
		switch w {
		case 16:
			v.StoreArray((*[16]uint8)(pix[o : o+16]))
		case 32:
			line := pix[o : o+32]
			v.StoreArray((*[16]uint8)(line[:16]))
			v.StoreArray((*[16]uint8)(line[16:]))
		default:
			line := pix[o : o+64]
			v.StoreArray((*[16]uint8)(line[:16]))
			v.StoreArray((*[16]uint8)(line[16:32]))
			v.StoreArray((*[16]uint8)(line[32:48]))
			v.StoreArray((*[16]uint8)(line[48:]))
		}
		o += stride
	}
}

// Arm64 implementations of the operations the shared intra kernels need. NEON
// has single instructions for each of these, so the shared kernels map onto the
// native forms directly.

// Right-shift amounts as per-lane shift vectors. The arm64 toolchain lowers
// ShiftAllRight(const) to a per-use DUP of the negated amount, whereas a per-lane
// Shift against a hoisted vector is a single VUSHL/VSSHL.
var (
	shr1U16  = archsimd.BroadcastInt16x8(-1)
	shr5U16  = archsimd.BroadcastInt16x8(-5)
	shr8U16  = archsimd.BroadcastInt16x8(-smoothShift1D)
	shr6I32  = archsimd.BroadcastInt32x4(-6)
	shr31I32 = archsimd.BroadcastInt32x4(-31)
	shr8I32  = archsimd.BroadcastInt32x4(-smoothShift1D)
)

// narrowU16ToU8 narrows eight uint16 lanes already in [0,255] to bytes (UQXTN).
func narrowU16ToU8(v archsimd.Uint16x8) archsimd.Uint8x16 {
	return v.SaturateToUint8()
}

// hiToLoU16 moves the upper four lanes of v into the lower four.
func hiToLoU16(v archsimd.Uint16x8) archsimd.Uint16x8 {
	return v.HiToLo()
}

// hiToLoI16 moves the upper four lanes of v into the lower four.
func hiToLoI16(v archsimd.Int16x8) archsimd.Int16x8 {
	return v.HiToLo()
}

// mulWidenLo16 multiplies the low four lanes of a and b into int32 lanes.
func mulWidenLo16(a, b archsimd.Int16x8) archsimd.Int32x4 {
	return a.MulWidenLo(b)
}

// packInt32PairToInt16 packs eight int32 values (lo's four lanes, then hi's) into
// int16 lanes, truncating; callers guarantee the values fit.
func packInt32PairToInt16(lo, hi archsimd.Int32x4) archsimd.Int16x8 {
	return cflTruncateInt32PairToInt16(lo, hi)
}

// reduceSumU32 returns the sum of the four uint32 lanes.
func reduceSumU32(v archsimd.Uint32x4) uint32 {
	return v.ReduceSum()
}

// shrU16_1, shrU16_5 and shrU16_8 are logical right shifts of uint16 lanes.
func shrU16_1(v archsimd.Uint16x8) archsimd.Uint16x8 { return v.Shift(shr1U16) }
func shrU16_5(v archsimd.Uint16x8) archsimd.Uint16x8 { return v.Shift(shr5U16) }
func shrU16_8(v archsimd.Uint16x8) archsimd.Uint16x8 { return v.Shift(shr8U16) }

// shrI32_6 and shrI32_31 are arithmetic right shifts of int32 lanes.
func shrI32_6(v archsimd.Int32x4) archsimd.Int32x4  { return v.Shift(shr6I32) }
func shrI32_31(v archsimd.Int32x4) archsimd.Int32x4 { return v.Shift(shr31I32) }

// Go-native SIMD CfL kernels. They mirror the scalar contracts in
// intra_kernels.go:
//   - subsampleLuma8SIMD reproduces the lbd Q3 reductions with UXTL/USHLL and
//     UZP even/odd pairing.
//   - subsampleLuma16SIMD preserves the hbd max-sample validation, then performs
//     the same Q3 reductions in uint16 lanes.
//   - subtractCFLAverageSIMD computes the rounded average in uint32 lanes and
//     truncates the final signed difference exactly like int16(int(src)-avg).

func subsampleLuma8SIMD(outputQ3 []uint16, input []uint8, inputStride int, width int, height int, outW int, outH int, subX bool, subY bool) {
	switch {
	case subX && subY:
		if outW < 8 || outW%8 != 0 {
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		sh1 := archsimd.BroadcastInt16x8(1)
		for row := 0; row < height; row += 2 {
			outBase := (row >> 1) * CFLBufLine
			topBase := row * inputStride
			botBase := (row + 1) * inputStride
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				top := archsimd.LoadUint8x16Array((*[16]uint8)(input[topBase+ic:]))
				bot := archsimd.LoadUint8x16Array((*[16]uint8)(input[botBase+ic:]))
				sum := cflPairSumUint8x16(top).Add(cflPairSumUint8x16(bot)).Shift(sh1)
				sum.StoreArray((*[8]uint16)(outputQ3[outBase+oc:]))
			}
		}
	case subX:
		if outW < 8 || outW%8 != 0 {
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		sh2 := archsimd.BroadcastInt16x8(2)
		for row := 0; row < outH; row++ {
			outBase := row * CFLBufLine
			inBase := row * inputStride
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				v := archsimd.LoadUint8x16Array((*[16]uint8)(input[inBase+ic:]))
				sum := cflPairSumUint8x16(v).Shift(sh2)
				sum.StoreArray((*[8]uint16)(outputQ3[outBase+oc:]))
			}
		}
	default:
		if outW < 16 || outW%16 != 0 {
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		sh3 := archsimd.BroadcastInt16x8(3)
		for row := 0; row < outH; row++ {
			outBase := row * CFLBufLine
			inBase := row * inputStride
			for col := 0; col < outW; col += 16 {
				v := archsimd.LoadUint8x16Array((*[16]uint8)(input[inBase+col:]))
				v.ExtendLo8ToUint16().Shift(sh3).StoreArray((*[8]uint16)(outputQ3[outBase+col:]))
				v.HiToLo().ExtendLo8ToUint16().Shift(sh3).StoreArray((*[8]uint16)(outputQ3[outBase+col+8:]))
			}
		}
	}
}

func subsampleLuma16SIMD(outputQ3 []uint16, input []uint16, inputStride int, width int, height int, outW int, outH int, subX bool, subY bool, max uint16) error {
	maxV := archsimd.BroadcastUint16x8(max)
	switch {
	case subX && subY:
		if outW < 8 || outW%8 != 0 {
			return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
		}
		sh1 := archsimd.BroadcastInt16x8(1)
		for row := 0; row < height; row += 2 {
			outBase := (row >> 1) * CFLBufLine
			topBase := row * inputStride
			botBase := (row + 1) * inputStride
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				top0 := archsimd.LoadUint16x8Array((*[8]uint16)(input[topBase+ic:]))
				top1 := archsimd.LoadUint16x8Array((*[8]uint16)(input[topBase+ic+8:]))
				bot0 := archsimd.LoadUint16x8Array((*[8]uint16)(input[botBase+ic:]))
				bot1 := archsimd.LoadUint16x8Array((*[8]uint16)(input[botBase+ic+8:]))
				if cflUint16AnyAbove(top0, maxV) || cflUint16AnyAbove(top1, maxV) ||
					cflUint16AnyAbove(bot0, maxV) || cflUint16AnyAbove(bot1, maxV) {
					return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
				}
				sum := cflPairSumUint16x16(top0, top1).Add(cflPairSumUint16x16(bot0, bot1)).Shift(sh1)
				sum.StoreArray((*[8]uint16)(outputQ3[outBase+oc:]))
			}
		}
	case subX:
		if outW < 8 || outW%8 != 0 {
			return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
		}
		sh2 := archsimd.BroadcastInt16x8(2)
		for row := 0; row < outH; row++ {
			outBase := row * CFLBufLine
			inBase := row * inputStride
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				in0 := archsimd.LoadUint16x8Array((*[8]uint16)(input[inBase+ic:]))
				in1 := archsimd.LoadUint16x8Array((*[8]uint16)(input[inBase+ic+8:]))
				if cflUint16AnyAbove(in0, maxV) || cflUint16AnyAbove(in1, maxV) {
					return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
				}
				sum := cflPairSumUint16x16(in0, in1).Shift(sh2)
				sum.StoreArray((*[8]uint16)(outputQ3[outBase+oc:]))
			}
		}
	default:
		if outW < 8 || outW%8 != 0 {
			return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
		}
		sh3 := archsimd.BroadcastInt16x8(3)
		for row := 0; row < outH; row++ {
			outBase := row * CFLBufLine
			inBase := row * inputStride
			for col := 0; col < outW; col += 8 {
				v := archsimd.LoadUint16x8Array((*[8]uint16)(input[inBase+col:]))
				if cflUint16AnyAbove(v, maxV) {
					return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
				}
				v.Shift(sh3).StoreArray((*[8]uint16)(outputQ3[outBase+col:]))
			}
		}
	}
	return nil
}

func subtractCFLAverageSIMD(srcQ3 []uint16, dstQ3 []int16, width int, height int, numPelLog2 int) {
	if width < 8 || width%8 != 0 {
		subtractCFLAveragePureGo(srcQ3, dstQ3, width, height, numPelLog2)
		return
	}
	acc := archsimd.BroadcastUint32x4(0)
	for row := 0; row < height; row++ {
		base := row * CFLBufLine
		for col := 0; col < width; col += 8 {
			v := archsimd.LoadUint16x8Array((*[8]uint16)(srcQ3[base+col:]))
			acc = acc.Add(v.ExtendLo4ToUint32()).Add(v.HiToLo().ExtendLo4ToUint32())
		}
	}
	sum := int(acc.ReduceSum()) + ((width * height) >> 1)
	avg := sum >> numPelLog2
	avgV := archsimd.BroadcastInt32x4(int32(avg))
	for row := 0; row < height; row++ {
		base := row * CFLBufLine
		for col := 0; col < width; col += 8 {
			v := archsimd.LoadUint16x8Array((*[8]uint16)(srcQ3[base+col:]))
			lo := v.ExtendLo4ToUint32().ConvertToInt32().Sub(avgV)
			hi := v.HiToLo().ExtendLo4ToUint32().ConvertToInt32().Sub(avgV)
			cflTruncateInt32PairToInt16(lo, hi).StoreArray((*[8]int16)(dstQ3[base+col:]))
		}
	}
}

func cflPairSumUint8x16(v archsimd.Uint8x16) archsimd.Uint16x8 {
	return v.ConcatEven(v).ExtendLo8ToUint16().Add(v.ConcatOdd(v).ExtendLo8ToUint16())
}

func cflPairSumUint16x16(lo, hi archsimd.Uint16x8) archsimd.Uint16x8 {
	return lo.ConcatEven(hi).Add(lo.ConcatOdd(hi))
}

func cflUint16AnyAbove(v archsimd.Uint16x8, max archsimd.Uint16x8) bool {
	return v.Greater(max).ToInt16x8().ReduceSum() != 0
}

// cflTruncateInt32PairToInt16 packs the low four lanes of lo followed by the
// low four lanes of hi. ARM64 archsimd exposes single-vector truncation; a
// 64-bit interleave combines the two packed halves in lane order.
func cflTruncateInt32PairToInt16(lo, hi archsimd.Int32x4) archsimd.Int16x8 {
	lo64 := lo.TruncToInt16().ToBits().ReshapeToUint64s()
	hi64 := hi.TruncToInt16().ToBits().ReshapeToUint64s()
	return lo64.InterleaveLo(hi64).ReshapeToUint16s().BitsToInt16()
}

// cflSaturateInt16PairToUint8 packs two groups of eight signed pixels into one
// 16-byte vector, preserving the order expected by the pixel store.

// storeU16x8 writes the eight uint16 lanes of v to dst[0:16], little-endian:
// one vector store (the lanes are already in memory order on arm64).
func storeU16x8(dst []byte, v archsimd.Int16x8) {
	v.ToBits().ReshapeToUint8s().StoreArray((*[16]byte)(dst[:16]))
}

// paethW4SIMD is the 8-bit kernel for width 4 and even height. Lanes 0..3 hold
// row r and lanes 4..7 hold row r+1, so one vector covers two rows.
func paethW4SIMD(block planeBlock, above []uint16, left []uint16, aboveLeft uint16) {
	height := block.height
	aV := archsimd.LoadUint16x8Array(&[8]uint16{above[0], above[1], above[2], above[3], above[0], above[1], above[2], above[3]}).ConvertToInt16()
	tlV := archsimd.BroadcastInt16x8(int16(aboveLeft))
	tl2V := archsimd.BroadcastInt16x8(int16(aboveLeft) << 1)
	for row := 0; row < height; row += 2 {
		lV := archsimd.LoadInt16x8Array(&[8]int16{
			int16(left[row]), int16(left[row]), int16(left[row]), int16(left[row]),
			int16(left[row+1]), int16(left[row+1]), int16(left[row+1]), int16(left[row+1]),
		})
		pTop := lV.Sub(tlV).Abs()
		out := paethInt16(aV, lV, tlV, lV.Sub(tl2V), pTop)
		packed := out.ConvertToUint16().SaturateToUint8().ReshapeToUint32s()
		stride := block.stride
		putU32(block.pix, row*stride, packed.GetElem(0))
		putU32(block.pix, (row+1)*stride, packed.GetElem(1))
	}
}

// smooth8W4SIMD is the 8-bit 2D SMOOTH kernel for width 4 and even height, two
// rows per vector (lanes 0..3 row r, lanes 4..7 row r+1).
func smooth8W4SIMD(block planeBlock, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	height := block.height
	tmb := archsimd.LoadUint16x8Array(&[8]uint16{above[0], above[1], above[2], above[3], above[0], above[1], above[2], above[3]}).Sub(archsimd.BroadcastUint16x8(belowPred))
	wWV := archsimd.LoadUint16x8Array(&[8]uint16{weightsW[0], weightsW[1], weightsW[2], weightsW[3], weightsW[0], weightsW[1], weightsW[2], weightsW[3]})
	vBase := archsimd.BroadcastUint16x8(belowPred<<8 | 0xff)
	hBase := archsimd.BroadcastUint16x8(rightPred << 8)
	for row := 0; row < height; row += 2 {
		wHV := archsimd.LoadUint16x8Array(&[8]uint16{
			weightsH[row], weightsH[row], weightsH[row], weightsH[row],
			weightsH[row+1], weightsH[row+1], weightsH[row+1], weightsH[row+1],
		})
		l0, l1 := left[row]-rightPred, left[row+1]-rightPred
		lmr := archsimd.LoadUint16x8Array(&[8]uint16{l0, l0, l0, l0, l1, l1, l1, l1})
		avg := tmb.MulAdd(wHV, vBase).Average(wWV.MulAdd(lmr, hBase)).ReshapeToUint8s()
		out := avg.ConcatOdd(avg).ReshapeToUint32s()
		putU32(block.pix, row*block.stride, out.GetElem(0))
		putU32(block.pix, (row+1)*block.stride, out.GetElem(1))
	}
}

// smoothSplat holds every smooth weight broadcast to eight lanes, so a row's
// vertical weight is one vector load, as dav1d's ld4r/ld2r of weights_ver.
var smoothSplat = func() (t [len(smoothWeights)][8]uint16) {
	for i, w := range smoothWeights {
		t[i] = [8]uint16{w, w, w, w, w, w, w, w}
	}
	return t
}()

// splat8 broadcasts each lane of v to a whole vector through three zip
// levels: the per-row left broadcasts dav1d's ld4r/ld2r take from memory.
func splat8(v archsimd.Uint16x8) (b0, b1, b2, b3, b4, b5, b6, b7 archsimd.Uint16x8) {
	lo, hi := v.InterleaveLo(v), v.InterleaveHi(v)
	q0, q1, q2, q3 := lo.InterleaveLo(lo), lo.InterleaveHi(lo), hi.InterleaveLo(hi), hi.InterleaveHi(hi)
	return q0.InterleaveLo(q0), q0.InterleaveHi(q0), q1.InterleaveLo(q1), q1.InterleaveHi(q1),
		q2.InterleaveLo(q2), q2.InterleaveHi(q2), q3.InterleaveLo(q3), q3.InterleaveHi(q3)
}

// smoothRow is one 8-pixel smooth row before narrowing: the high byte of each
// lane is the predicted sample.
func smoothRow(tmb, wv, vBase, wh, lmr, hBase archsimd.Uint16x8) archsimd.Uint16x8 {
	return tmb.MulAdd(wv, vBase).Average(wh.MulAdd(lmr, hBase))
}

// putRows8 narrows two smoothRow results and stores them as rows off and
// off+stride.
func putRows8(pix []byte, off int, stride int, a0, a1 archsimd.Uint16x8) {
	out := a0.ReshapeToUint8s().ConcatOdd(a1.ReshapeToUint8s()).ReshapeToUint64s()
	binary.LittleEndian.PutUint64(pix[off:], out.GetElem(0))
	binary.LittleEndian.PutUint64(pix[off+stride:], out.GetElem(1))
}

// smoothW8NEON is the 8-bit 2D SMOOTH kernel for width 8: dav1d's
// ipred_smooth w8 loop, eight rows per iteration with the left samples
// broadcast from one vector and the vertical weights loaded pre-broadcast.
// The pre-broadcast weights are those of the block height, so a block
// clipped below its weight table takes the two-row loop.
func smoothW8NEON(block planeBlock, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	height := block.height
	stride := block.stride
	tmb := archsimd.LoadUint16x8Array((*[8]uint16)(above)).Sub(archsimd.BroadcastUint16x8(belowPred))
	wh := archsimd.LoadUint16x8Array((*[8]uint16)(weightsW))
	vBase := archsimd.BroadcastUint16x8(belowPred<<8 | 0xff)
	hBase := archsimd.BroadcastUint16x8(rightPred << 8)
	pix := block.pix
	if height%8 != 0 || len(weightsH) != height {
		weightsH, left = weightsH[:height], left[:height]
		for row := 0; row+1 < height; row += 2 {
			a0 := smoothRow(tmb, archsimd.BroadcastUint16x8(weightsH[row]), vBase, wh, archsimd.BroadcastUint16x8(left[row]-rightPred), hBase)
			a1 := smoothRow(tmb, archsimd.BroadcastUint16x8(weightsH[row+1]), vBase, wh, archsimd.BroadcastUint16x8(left[row+1]-rightPred), hBase)
			putRows8(pix, row*stride, stride, a0, a1)
		}
		return
	}
	rightV := archsimd.BroadcastUint16x8(rightPred)
	splat := smoothSplat[height-4:][:height]
	left = left[:height]
	for row := 0; row+8 <= height; row += 8 {
		l0, l1, l2, l3, l4, l5, l6, l7 := splat8(archsimd.LoadUint16x8Array((*[8]uint16)(left[row:])).Sub(rightV))
		w := splat[row:][:8]
		off := row * stride
		putRows8(pix, off, stride,
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[0]), vBase, wh, l0, hBase),
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[1]), vBase, wh, l1, hBase))
		putRows8(pix, off+2*stride, stride,
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[2]), vBase, wh, l2, hBase),
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[3]), vBase, wh, l3, hBase))
		putRows8(pix, off+4*stride, stride,
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[4]), vBase, wh, l4, hBase),
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[5]), vBase, wh, l5, hBase))
		putRows8(pix, off+6*stride, stride,
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[6]), vBase, wh, l6, hBase),
			smoothRow(tmb, archsimd.LoadUint16x8Array(&w[7]), vBase, wh, l7, hBase))
	}
}

// smoothWideNEON is the 8-bit 2D SMOOTH kernel for width a multiple of 16:
// per row, sixteen columns per narrowed store. Unclipped blocks whose height
// is a multiple of 8 run smoothW16NEON per sixteen columns.
func smoothWideNEON(block planeBlock, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	width, height := block.width, block.height
	if height%8 == 0 && len(weightsH) == height {
		for c := 0; c+16 <= width; c += 16 {
			smoothW16NEON(block.pix[c:], block.stride, height, weightsW[c:], above[c:], left, belowPred, rightPred)
		}
		return
	}
	blocks := min(width/8, 8)
	bottomV := archsimd.BroadcastUint16x8(belowPred)
	vBase := archsimd.BroadcastUint16x8(belowPred<<8 | 0xff)
	hBase := archsimd.BroadcastUint16x8(rightPred << 8)
	var tmb, wh [8]archsimd.Uint16x8
	for b := 0; b < blocks; b++ {
		tmb[b] = archsimd.LoadUint16x8Array((*[8]uint16)(above[b*8:])).Sub(bottomV)
		wh[b] = archsimd.LoadUint16x8Array((*[8]uint16)(weightsW[b*8:]))
	}
	weightsH, left = weightsH[:height], left[:height]
	for row := range height {
		smoothWideRow(block.pix[row*block.stride:], &tmb, &wh, blocks, archsimd.BroadcastUint16x8(weightsH[row]), vBase, archsimd.BroadcastUint16x8(left[row]-rightPred), hBase)
	}
}

// smoothW16NEON is the 8-bit 2D SMOOTH kernel for sixteen columns of an
// unclipped block whose height is a multiple of 8, the top and weight vectors
// held in registers; wider blocks run it per sixteen columns.
func smoothW16NEON(pix []byte, stride int, height int, weightsW []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	bottomV := archsimd.BroadcastUint16x8(belowPred)
	vBase := archsimd.BroadcastUint16x8(belowPred<<8 | 0xff)
	hBase := archsimd.BroadcastUint16x8(rightPred << 8)
	rightV := archsimd.BroadcastUint16x8(rightPred)
	above, weightsW = above[:16], weightsW[:16]
	tmb0 := archsimd.LoadUint16x8Array((*[8]uint16)(above)).Sub(bottomV)
	tmb1 := archsimd.LoadUint16x8Array((*[8]uint16)(above[8:])).Sub(bottomV)
	wh0 := archsimd.LoadUint16x8Array((*[8]uint16)(weightsW))
	wh1 := archsimd.LoadUint16x8Array((*[8]uint16)(weightsW[8:]))
	splat := smoothSplat[height-4:][:height]
	left = left[:height]
	for row := 0; row+8 <= height; row += 8 {
		l0, l1, l2, l3, l4, l5, l6, l7 := splat8(archsimd.LoadUint16x8Array((*[8]uint16)(left[row:])).Sub(rightV))
		w := splat[row:][:8]
		off := row * stride
		smoothRow16(pix[off:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[0]), vBase, wh0, wh1, l0, hBase)
		smoothRow16(pix[off+stride:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[1]), vBase, wh0, wh1, l1, hBase)
		smoothRow16(pix[off+2*stride:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[2]), vBase, wh0, wh1, l2, hBase)
		smoothRow16(pix[off+3*stride:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[3]), vBase, wh0, wh1, l3, hBase)
		smoothRow16(pix[off+4*stride:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[4]), vBase, wh0, wh1, l4, hBase)
		smoothRow16(pix[off+5*stride:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[5]), vBase, wh0, wh1, l5, hBase)
		smoothRow16(pix[off+6*stride:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[6]), vBase, wh0, wh1, l6, hBase)
		smoothRow16(pix[off+7*stride:], tmb0, tmb1, archsimd.LoadUint16x8Array(&w[7]), vBase, wh0, wh1, l7, hBase)
	}
}

// smoothRow16 writes one 16-pixel row of smoothW16NEON.
func smoothRow16(dst []byte, tmb0, tmb1, wv, vBase, wh0, wh1, lmr, hBase archsimd.Uint16x8) {
	a0 := smoothRow(tmb0, wv, vBase, wh0, lmr, hBase)
	a1 := smoothRow(tmb1, wv, vBase, wh1, lmr, hBase)
	a0.ReshapeToUint8s().ConcatOdd(a1.ReshapeToUint8s()).StoreArray((*[16]uint8)(dst))
}

// smoothWideRow writes one row of smoothWideNEON.
func smoothWideRow(dst []byte, tmb, wh *[8]archsimd.Uint16x8, blocks int, wv, vBase, lmr, hBase archsimd.Uint16x8) {
	for b := 0; b+1 < blocks; b += 2 {
		a0 := smoothRow(tmb[b], wv, vBase, wh[b], lmr, hBase)
		a1 := smoothRow(tmb[b+1], wv, vBase, wh[b+1], lmr, hBase)
		a0.ReshapeToUint8s().ConcatOdd(a1.ReshapeToUint8s()).StoreArray((*[16]uint8)(dst[b*8:]))
	}
}

// smoothVerticalW4SIMD is the 8-bit SMOOTH_V kernel for width 4, even height.
func smoothVerticalW4SIMD(block planeBlock, weights []uint16, above []uint16, belowPred uint16) {
	height := block.height
	aV := archsimd.LoadUint16x8Array(&[8]uint16{above[0], above[1], above[2], above[3], above[0], above[1], above[2], above[3]})
	belowV := archsimd.BroadcastUint16x8(belowPred)
	roundV := archsimd.BroadcastUint16x8(1 << (smoothShift1D - 1))
	for row := 0; row < height; row += 2 {
		wV := archsimd.LoadUint16x8Array(&[8]uint16{
			weights[row], weights[row], weights[row], weights[row],
			weights[row+1], weights[row+1], weights[row+1], weights[row+1],
		})
		cb := archsimd.BroadcastUint16x8(smoothScale).Sub(wV).Mul(belowV).Add(roundV)
		out := wV.Mul(aV).Add(cb).Shift(shr8U16).SaturateToUint8().ReshapeToUint32s()
		putU32(block.pix, row*block.stride, out.GetElem(0))
		putU32(block.pix, (row+1)*block.stride, out.GetElem(1))
	}
}

// smoothHorizontalW4SIMD is the 8-bit SMOOTH_H kernel for width 4, even height.
func smoothHorizontalW4SIMD(block planeBlock, weights []uint16, left []uint16, rightPred uint16) {
	height := block.height
	wWV := archsimd.LoadUint16x8Array(&[8]uint16{weights[0], weights[1], weights[2], weights[3], weights[0], weights[1], weights[2], weights[3]})
	roundV := archsimd.BroadcastUint16x8(1 << (smoothShift1D - 1))
	cwRV := archsimd.BroadcastUint16x8(smoothScale).Sub(wWV).Mul(archsimd.BroadcastUint16x8(rightPred)).Add(roundV)
	for row := 0; row < height; row += 2 {
		lV := archsimd.LoadUint16x8Array(&[8]uint16{
			left[row], left[row], left[row], left[row],
			left[row+1], left[row+1], left[row+1], left[row+1],
		})
		out := wWV.Mul(lV).Add(cwRV).Shift(shr8U16).SaturateToUint8().ReshapeToUint32s()
		putU32(block.pix, row*block.stride, out.GetElem(0))
		putU32(block.pix, (row+1)*block.stride, out.GetElem(1))
	}
}

// paeth16SIMD is the 16-bit PAETH kernel for width a multiple of 8. Samples are
// at most 12 bits, so the int16 lanes hold base and the distances exactly.
func paeth16SIMD(block planeBlock, above []uint16, left []uint16, aboveLeft uint16) {
	width, height := block.width, block.height
	tlV := archsimd.BroadcastInt16x8(int16(aboveLeft))
	for row := 0; row < height; row++ {
		lV := archsimd.BroadcastInt16x8(int16(left[row]))
		pTop := archsimd.BroadcastInt16x8(absDiffInt16(left[row], aboveLeft))
		lmtV := archsimd.BroadcastInt16x8(int16(left[row]) - int16(aboveLeft)*2)
		dst := block.pix[row*block.stride:][:width*2]
		for col := 0; col < width; col += 8 {
			storeU16x8(dst[col*2:], paethInt16(loadU16x8(above[col:]), lV, tlV, lmtV, pTop))
		}
	}
}

// smooth16Cols holds the per-column 16-bit SMOOTH terms of one 8-column
// block as int32 lanes (low and high four columns): top - bottom, and the
// column weight scaled by 2^7.
type smooth16Cols struct {
	tmb [8][2]archsimd.Int32x4
	ww  [8][2]archsimd.Int32x4
}

// smooth16SIMD is the 16-bit 2D SMOOTH kernel for width a multiple of 8, in
// the factored form of the 8-bit kernels:
//
//	pred = (256*bottom + wH*(top - bottom) + 256*right + wW*(left - right) + 256) >> 9
//
// evaluated as int32 multiply-accumulates (MLA) with the constant terms in
// the accumulator seed and every term scaled by 2^7, so the >> 9 is the high
// halves of the int32 sums (one UZP2 for eight samples). The scaled sum is at
// most 2^7 * (512*4095 + 256) < 2^31 and its partial sums are non-negative
// combinations of the same bound, so the arithmetic is exact.
func smooth16SIMD(block planeBlock, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	width, height := block.width, block.height
	blocks := width / 8
	bottomV := archsimd.BroadcastInt16x8(int16(belowPred))
	seed := archsimd.BroadcastInt32x4((256*int32(belowPred) + 256*int32(rightPred) + 256) << 7)
	var cols smooth16Cols
	for b := 0; b < blocks; b++ {
		d := loadU16x8(above[b*8:]).Sub(bottomV)
		cols.tmb[b] = [2]archsimd.Int32x4{d.ExtendLo4ToInt32(), d.HiToLo().ExtendLo4ToInt32()}
		w := loadU16x8(weightsW[b*8:]).ShiftAllLeft(7)
		cols.ww[b] = [2]archsimd.Int32x4{w.ExtendLo4ToInt32(), w.HiToLo().ExtendLo4ToInt32()}
	}
	weightsH, left = weightsH[:height], left[:height]
	for row := range height {
		wH := archsimd.BroadcastInt32x4(int32(weightsH[row]) << 7)
		lmr := archsimd.BroadcastInt32x4(int32(left[row]) - int32(rightPred))
		dst := block.pix[row*block.stride:][:width*2]
		for b := range cols.tmb[:blocks] {
			lo := cols.ww[b][0].MulAdd(lmr, seed)
			hi := cols.ww[b][1].MulAdd(lmr, seed)
			lo = cols.tmb[b][0].MulAdd(wH, lo)
			hi = cols.tmb[b][1].MulAdd(wH, hi)
			lo.ToBits().ReshapeToUint16s().ConcatOdd(hi.ToBits().ReshapeToUint16s()).ReshapeToUint8s().StoreArray((*[16]byte)(dst[b*16:]))
		}
	}
}

// smoothVertical16SIMD is the 16-bit SMOOTH_V kernel for width a multiple of 8.
func smoothVertical16SIMD(block planeBlock, weights []uint16, above []uint16, belowPred uint16) {
	width, height := block.width, block.height
	scale32 := archsimd.BroadcastInt32x4(smoothScale)
	belowV := archsimd.BroadcastInt32x4(int32(belowPred))
	round := archsimd.BroadcastInt32x4(1 << (smoothShift1D - 1))
	for row := 0; row < height; row++ {
		wH := int16(weights[row])
		wHV := archsimd.BroadcastInt16x8(wH)
		cb := scale32.Sub(archsimd.BroadcastInt32x4(int32(wH))).Mul(belowV).Add(round)
		dst := block.pix[row*block.stride:][:width*2]
		for col := 0; col < width; col += 8 {
			aV := loadU16x8(above[col:])
			loOut := wHV.MulWidenLo(aV).Add(cb).Shift(shr8I32)
			hiOut := wHV.MulWidenLo(aV.HiToLo()).Add(cb).Shift(shr8I32)
			storeU16x8(dst[col*2:], cflTruncateInt32PairToInt16(loOut, hiOut))
		}
	}
}

// smoothHorizontal16SIMD is the 16-bit SMOOTH_H kernel for width a multiple of 8.
func smoothHorizontal16SIMD(block planeBlock, weights []uint16, left []uint16, rightPred uint16) {
	width, height := block.width, block.height
	blocks := width / 8
	rightV := archsimd.BroadcastInt32x4(int32(rightPred))
	scale32 := archsimd.BroadcastInt32x4(smoothScale)
	round := archsimd.BroadcastInt32x4(1 << (smoothShift1D - 1))
	var wW [8]archsimd.Int16x8
	var cwRLo, cwRHi [8]archsimd.Int32x4
	for b := 0; b < blocks; b++ {
		w := loadU16x8(weights[b*8:])
		wW[b] = w
		cwRLo[b] = scale32.Sub(w.ExtendLo4ToInt32()).Mul(rightV).Add(round)
		cwRHi[b] = scale32.Sub(w.HiToLo().ExtendLo4ToInt32()).Mul(rightV).Add(round)
	}
	for row := 0; row < height; row++ {
		lV := archsimd.BroadcastInt16x8(int16(left[row]))
		dst := block.pix[row*block.stride:][:width*2]
		for b := 0; b < blocks; b++ {
			loOut := wW[b].MulWidenLo(lV).Add(cwRLo[b]).Shift(shr8I32)
			hiOut := wW[b].HiToLo().MulWidenLo(lV).Add(cwRHi[b]).Shift(shr8I32)
			storeU16x8(dst[b*16:], cflTruncateInt32PairToInt16(loOut, hiOut))
		}
	}
}

// predictPaethSIMD routes a PAETH block to the kernel for its shape; shapes no
// kernel covers use the pure-Go reference.
func predictPaethSIMD(block planeBlock, bytesPerSample int, above []uint16, left []uint16, aboveLeft uint16) {
	switch {
	case bytesPerSample == 1 && block.width%8 == 0:
		paeth8SIMD(block, above, left, aboveLeft)
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
		paethW4SIMD(block, above, left, aboveLeft)
	case bytesPerSample == 2 && block.width%8 == 0:
		paeth16SIMD(block, above, left, aboveLeft)
	default:
		predictPaethPureGo(block, bytesPerSample, above, left, aboveLeft)
	}
}

// predictSmoothSIMD routes a 2D SMOOTH block to the kernel for its shape.
func predictSmoothSIMD(block planeBlock, bytesPerSample int, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	switch {
	case bytesPerSample == 1 && block.width == 8 && block.height%2 == 0:
		smoothW8NEON(block, weightsW, weightsH, above, left, belowPred, rightPred)
	case bytesPerSample == 1 && block.width%16 == 0 && block.width <= 64:
		smoothWideNEON(block, weightsW, weightsH, above, left, belowPred, rightPred)
	case bytesPerSample == 1 && block.width%8 == 0:
		smooth8SIMD(block, weightsW, weightsH, above, left, belowPred, rightPred)
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
		smooth8W4SIMD(block, weightsW, weightsH, above, left, belowPred, rightPred)
	case bytesPerSample == 2 && block.width%8 == 0:
		smooth16SIMD(block, weightsW, weightsH, above, left, belowPred, rightPred)
	default:
		predictSmoothPureGo(block, bytesPerSample, weightsW, weightsH, above, left, belowPred, rightPred)
	}
}

// predictSmoothVerticalSIMD routes a SMOOTH_V block to the kernel for its shape.
func predictSmoothVerticalSIMD(block planeBlock, bytesPerSample int, weights []uint16, above []uint16, belowPred uint16) {
	switch {
	case bytesPerSample == 1 && block.width%8 == 0:
		smoothVertical8SIMD(block, weights, above, belowPred)
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
		smoothVerticalW4SIMD(block, weights, above, belowPred)
	case bytesPerSample == 2 && block.width%8 == 0:
		smoothVertical16SIMD(block, weights, above, belowPred)
	default:
		predictSmoothVerticalPureGo(block, bytesPerSample, weights, above, belowPred)
	}
}

// predictSmoothHorizontalSIMD routes a SMOOTH_H block to the kernel for its shape.
func predictSmoothHorizontalSIMD(block planeBlock, bytesPerSample int, weights []uint16, left []uint16, rightPred uint16) {
	switch {
	case bytesPerSample == 1 && block.width%8 == 0:
		smoothHorizontal8SIMD(block, weights, left, rightPred)
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
		smoothHorizontalW4SIMD(block, weights, left, rightPred)
	case bytesPerSample == 2 && block.width%8 == 0:
		smoothHorizontal16SIMD(block, weights, left, rightPred)
	default:
		predictSmoothHorizontalPureGo(block, bytesPerSample, weights, left, rightPred)
	}
}

// init binds the Go-native SIMD intra kernels under the goexperiment.simd build:
// PAETH and the three SMOOTH predictors (intra_static_gosimd_arm64.go), the DC
// sum, CfL apply and subsampling, the directional interpolation and filter intra.
// Every binding remains byte-identical to its pure-Go reference.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if !cpu.Detected.NEON {
		predictPaethImpl = predictPaethPureGo
		predictSmoothImpl = predictSmoothPureGo
		predictSmoothVerticalImpl = predictSmoothVerticalPureGo
		predictSmoothHorizontalImpl = predictSmoothHorizontalPureGo
		sumSamplesImpl = sumSamplesPureGo
		applyCFLImpl = applyCFLPureGo
		subsampleLuma8Impl = subsampleLuma8PureGo
		subsampleLuma16Impl = subsampleLuma16PureGo
		subtractCFLAverageImpl = subtractCFLAveragePureGo
		dirRowInterp8Impl = dirRowInterp8PureGo
		dirAboveRun8Impl = dirAboveRun8PureGo
		dirLeftCol8Impl = dirLeftCol8PureGo
		return
	}
	predictPaethImpl = predictPaethSIMD
	predictSmoothImpl = predictSmoothSIMD
	predictSmoothVerticalImpl = predictSmoothVerticalSIMD
	predictSmoothHorizontalImpl = predictSmoothHorizontalSIMD
	sumSamplesImpl = sumSamplesSIMD
	applyCFLImpl = applyCFLSIMD
	subsampleLuma8Impl = subsampleLuma8SIMD
	subsampleLuma16Impl = subsampleLuma16SIMD
	subtractCFLAverageImpl = subtractCFLAverageSIMD
	dirRowInterp8Impl = dirRowInterp8SIMD
	dirAboveRun8Impl = dirAboveRun8SIMD
	dirLeftCol8Impl = dirLeftCol8SIMD
	predictFilterIntra8Impl = predictFilterIntraBlockDirect8TBL
	predictFilterIntra16Impl = predictFilterIntraBlockDirect10TBL
}
