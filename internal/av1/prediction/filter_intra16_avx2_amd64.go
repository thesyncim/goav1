// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

import (
	"encoding/binary"
	"simd/archsimd"
)

// filterIntraPairs16[mode][k] interleaves the taps k and k+1 (k = 0, 2, 4,
// 6, the last paired with zero) of the eight outputs of one 4x2 filter-intra
// batch: lane pair i holds (taps[k][i], taps[k+1][i]) for vpmaddwd.
var filterIntraPairs16 = func() (t [FilterIntraModes][4][16]int16) {
	for mode := range FilterIntraModes {
		for p := range 4 {
			for i := range 8 {
				t[mode][p][2*i] = int16(filterIntraTaps[mode][i][2*p])
				if 2*p+1 < 7 {
					t[mode][p][2*i+1] = int16(filterIntraTaps[mode][i][2*p+1])
				}
			}
		}
	}
	return t
}()

// filterIntraPair broadcasts the neighbour pair (a, b) to every int32 lane.
func filterIntraPair(a, b uint16) archsimd.Int16x16 {
	return archsimd.BroadcastInt32x8(int32(uint32(a) | uint32(b)<<16)).AsInt16x16()
}

// predictFilterIntraBlockDirect16AVX2 is predictFilterIntraBlockDirect16 for
// 10- and 12-bit samples, in the form of dav1d's ipred16_avx2 filter: each
// 4x2 batch is four vpmaddwd of the interleaved tap pairs against the
// broadcast neighbour pairs (p0,p1), (p2,p3), (p4,p5), (p6,0), exact in
// int32, then Round2(., 4) and the [0, max] clamp, which the scalar
// roundPowerOfTwo(sum, 4) clamped to [0, max] equals sample for sample.
func predictFilterIntraBlockDirect16AVX2(block planeBlock, width int, height int, mode FilterIntraMode, edges IntraEdges, max int) {
	if width%4 != 0 || mode >= FilterIntraModes {
		predictFilterIntraBlockDirect16(block, width, height, mode, edges, max)
		return
	}
	taps := &filterIntraPairs16[mode]
	t0 := archsimd.LoadInt16x16Array(&taps[0])
	t1 := archsimd.LoadInt16x16Array(&taps[1])
	t2 := archsimd.LoadInt16x16Array(&taps[2])
	t3 := archsimd.LoadInt16x16Array(&taps[3])
	roundV := archsimd.BroadcastInt32x8(1 << (filterIntraScaleBits - 1))
	maxV := archsimd.BroadcastInt32x8(int32(max))
	var zeroV archsimd.Int32x8

	// above[0] is p0 of the first batch and above[i+1] the top row sample i.
	var above [65]uint16
	for row := 0; row < height; row += 2 {
		if row == 0 {
			above[0] = edges.AboveLeft
			copy(above[1:width+1], edges.Above[:width])
		} else {
			above[0] = edges.Left[row-1]
			top := block.pix[(row-1)*block.stride:][:width*2]
			for i := 0; i < width; i++ {
				above[i+1] = binary.LittleEndian.Uint16(top[i<<1:])
			}
		}
		row0 := block.pix[row*block.stride:][:width*2]
		row1 := block.pix[(row+1)*block.stride:][:width*2]
		p5, p6 := edges.Left[row], edges.Left[row+1]
		for col := 0; col < width; col += 4 {
			a := above[col : col+5]
			acc := t0.DotProductPairs(filterIntraPair(a[0], a[1])).
				Add(t1.DotProductPairs(filterIntraPair(a[2], a[3]))).
				Add(t2.DotProductPairs(filterIntraPair(a[4], p5))).
				Add(t3.DotProductPairs(filterIntraPair(p6, 0))).
				Add(roundV).ShiftAllRight(filterIntraScaleBits).Max(zeroV).Min(maxV)
			// vpackssdw within lanes: lane 0 holds outputs 0..3 (row 0),
			// lane 1 outputs 4..7 (row 1).
			v := acc.SaturateToInt16ConcatGrouped(acc)
			r0 := v.GetLo().AsUint64x2().GetElem(0)
			r1 := v.GetHi().AsUint64x2().GetElem(0)
			binary.LittleEndian.PutUint64(row0[col<<1:], r0)
			binary.LittleEndian.PutUint64(row1[col<<1:], r1)
			p5, p6 = uint16(r0>>48), uint16(r1>>48)
		}
	}
	// dav1d returns through vzeroupper; the Go compiler emits none.
	archsimd.ClearAVXUpperBits()
}
