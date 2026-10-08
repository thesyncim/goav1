// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package prediction

import (
	"encoding/binary"
	"simd/archsimd"
)

// filterIntraTapsSIMD is the transpose of filterIntraTaps: [mode][tap][output],
// so each tap loads as one 8-lane vector whose lanes are the eight outputs of a
// 4x2 filter-intra batch (row0 cols 0-3 in lanes 0-3, row1 cols 0-3 in lanes
// 4-7). Only the first seven taps feed the sum, as in the scalar reference.
var filterIntraTapsSIMD [FilterIntraModes][7][8]int16

func init() {
	for mode := range filterIntraTaps {
		for output := 0; output < 8; output++ {
			for tap := 0; tap < 7; tap++ {
				filterIntraTapsSIMD[mode][tap][output] = int16(filterIntraTaps[mode][output][tap])
			}
		}
	}
}

// predictFilterIntraBlockDirect8SIMD is the Go-native SIMD form of
// predictFilterIntraBlockDirect8 for 8-bit samples. Each 4x2 batch evaluates
// the seven-tap filter for all eight outputs in one Int16x8 accumulation: the
// neighbours p0..p4 are broadcast from the top-row buffer and p5/p6 are the
// previous batch's right-most outputs (lanes 3 and 7). Every tap sum fits int16
// for 8-bit samples (see TestFilterIntraTapSumsFitInt16), so the int16 lanes
// reproduce the scalar int sum exactly; the rounding shift and [0,255] clamp
// match roundPowerOfTwo(sum, filterIntraScaleBits) followed by the clamp.
//
// The seven taps are individual locals rather than an array so they stay
// register-resident across the batch loop.
func predictFilterIntraBlockDirect8SIMD(block planeBlock, width int, height int, mode FilterIntraMode, edges IntraEdges, max int) {
	if width%4 != 0 || max != 0xff {
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
	maxV := archsimd.BroadcastInt16x8(int16(max))

	// above[0] is p0 of the first batch (the top-left sample) and above[i+1]
	// is the top row sample i, so batch col reads p0..p4 at above[col..col+4].
	var above [33]uint8
	for row := 0; row < height; row += 2 {
		if row == 0 {
			above[0] = uint8(edges.AboveLeft)
			for i := 0; i < width; i++ {
				above[i+1] = uint8(edges.Above[i])
			}
		} else {
			above[0] = uint8(edges.Left[row-1])
			top := block.pix[(row-1)*block.stride:]
			for i := 0; i < width; i++ {
				above[i+1] = top[i]
			}
		}
		row0 := block.pix[row*block.stride:][:width]
		row1 := block.pix[(row+1)*block.stride:][:width]
		p5 := archsimd.BroadcastInt16x8(int16(edges.Left[row]))
		p6 := archsimd.BroadcastInt16x8(int16(edges.Left[row+1]))
		for col := 0; col < width; col += 4 {
			a := above[col:]
			_ = a[4]
			// The top-row terms do not depend on the previous batch, so they are
			// accumulated first and only the left-column terms follow the chain.
			base := t0.Mul(archsimd.BroadcastInt16x8(int16(a[0]))).
				Add(t1.Mul(archsimd.BroadcastInt16x8(int16(a[1])))).
				Add(t2.Mul(archsimd.BroadcastInt16x8(int16(a[2])))).
				Add(t3.Mul(archsimd.BroadcastInt16x8(int16(a[3])))).
				Add(t4.Mul(archsimd.BroadcastInt16x8(int16(a[4])))).
				Add(roundV)
			out := base.Add(t5.Mul(p5)).Add(t6.Mul(p6)).ShiftAllRight(filterIntraScaleBits).Max(zeroV).Min(maxV)
			packed := filterIntraPackBytes(out).ReshapeToUint32s()
			binary.LittleEndian.PutUint32(row0[col:], packed.GetElem(0))
			binary.LittleEndian.PutUint32(row1[col:], packed.GetElem(1))
			p5 = archsimd.BroadcastInt16x8(out.GetElem(3))
			p6 = archsimd.BroadcastInt16x8(out.GetElem(7))
		}
	}
}
