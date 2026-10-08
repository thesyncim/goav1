// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import (
	"encoding/binary"
	"simd/archsimd"
)

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
			loOut := lo.ShiftAllRight(filterIntraScaleBits).Max(zeroV).Min(maxV)
			hiOut := hi.ShiftAllRight(filterIntraScaleBits).Max(zeroV).Min(maxV)
			out := cflTruncateInt32PairToInt16(loOut, hiOut)
			packed := out.ToBits().ReshapeToUint64s()
			binary.LittleEndian.PutUint64(row0[col<<1:], packed.GetElem(0))
			binary.LittleEndian.PutUint64(row1[col<<1:], packed.GetElem(1))
			p5 = archsimd.BroadcastInt16x8(out.GetElem(3))
			p6 = archsimd.BroadcastInt16x8(out.GetElem(7))
		}
	}
}
