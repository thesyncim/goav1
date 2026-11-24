// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import (
	"encoding/binary"
	"simd/archsimd"
)

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
