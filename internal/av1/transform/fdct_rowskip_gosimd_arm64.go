// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "simd/archsimd"

func init() {
	forwardDCT8x8RowSkipImpl = forwardDCT8x8RowSkipSIMD
	forwardDCT16x16RowSkipImpl = forwardDCT16x16RowSkipSIMD
	forwardDCT32x32RowSkipImpl = forwardDCT32x32RowSkipSIMD
}

var (
	fwdRowSkipShrNeg6  = archsimd.BroadcastInt32x4(-6)
	fwdRowSkipSlack8V  = archsimd.BroadcastInt32x4(fwdRowSkipSlack8)
	fwdRowSkipSlack16V = archsimd.BroadcastInt32x4(fwdRowSkipSlack16)
	fwdRowSkipSlack32V = archsimd.BroadcastInt32x4(fwdRowSkipSlack32)
)

// fwdRowSkipGroupLive takes the per-row partial magnitude sums r0..r3 (lane
// sums of |T| for four consecutive column-pass rows) and reports whether any
// row's S + S>>6 + slack reaches its threshold lane.
func fwdRowSkipGroupLive(r0, r1, r2, r3, thr, slack archsimd.Int32x4) bool {
	s := r0.ConcatAddPairs(r1).ConcatAddPairs(r2.ConcatAddPairs(r3))
	m := s.Add(s.Shift(fwdRowSkipShrNeg6)).Add(slack).Sub(thr)
	return m.ReduceMax() >= 0
}

// fwdAbsSum16 is sum over four vectors of |b[i..i+3]| starting at i.
func fwdAbsSum16(b []int32, i int) archsimd.Int32x4 {
	return fwdLoadI32At(b, i).Abs().Add(fwdLoadI32At(b, i+4).Abs()).
		Add(fwdLoadI32At(b, i+8).Abs().Add(fwdLoadI32At(b, i+12).Abs()))
}

// forwardDCT16x16RowSkipSIMD is forwardDCT16x16RowSkipPureGo over the
// forwardDCT16x16SIMD passes: the column pass runs in full, then each group
// of four column-pass rows is bounded straight from the row-major column
// output and only live groups are transposed and row-transformed.
func forwardDCT16x16RowSkipSIMD(coeff *[256]int32, residual *[256]int16, thr0, thr int32) uint8 {
	var buf, bufT [256]int32
	for g := 0; g < 16; g += fwdLanes {
		fwdColDCT16(buf[:], g, residual[:], 16)
	}
	thrRest := archsimd.BroadcastInt32x4(thr)
	thrFirst := thrRest.SetElem(0, thr0)
	slack := fwdRowSkipSlack16V
	var mask uint8
	for g := range 4 {
		b := buf[g*64:]
		t := thrRest
		if g == 0 {
			t = thrFirst
		}
		if fwdRowSkipGroupLive(fwdAbsSum16(b, 0), fwdAbsSum16(b, 16), fwdAbsSum16(b, 32), fwdAbsSum16(b, 48), t, slack) {
			mask |= 1 << g
		}
	}
	if mask == 0 {
		return 0
	}
	zero := archsimd.BroadcastInt32x4(0)
	for h := 0; h < 16; h += 4 {
		if mask&(1<<(h>>2)) == 0 {
			for l := 0; l < 16; l++ {
				fwdStoreI32At(coeff[:], l*16+h, zero)
			}
			continue
		}
		for bj := 0; bj < 16; bj += 4 {
			fwdTransposeBlock4(bufT[bj*16+h:], 16, buf[h*16+bj:], 16)
		}
		fwdRowDCT16(coeff[:], 16, bufT[:], h)
	}
	return mask
}

// forwardDCT32x32RowSkipSIMD is forwardDCT32x32RowSkipPureGo over the
// forwardDCT32x32SIMD passes (see forwardDCT16x16RowSkipSIMD).
func forwardDCT32x32RowSkipSIMD(coeff *[1024]int32, residual *[1024]int16, thr0, thr int32) uint8 {
	var buf, bufT [1024]int32
	for g := 0; g < 32; g += fwdLanes {
		fwdColDCT32(buf[:], g, residual[:], 32)
	}
	thrRest := archsimd.BroadcastInt32x4(thr)
	thrFirst := thrRest.SetElem(0, thr0)
	slack := fwdRowSkipSlack32V
	var mask uint8
	for g := range 8 {
		b := buf[g*128:]
		t := thrRest
		if g == 0 {
			t = thrFirst
		}
		r0 := fwdAbsSum16(b, 0).Add(fwdAbsSum16(b, 16))
		r1 := fwdAbsSum16(b, 32).Add(fwdAbsSum16(b, 48))
		r2 := fwdAbsSum16(b, 64).Add(fwdAbsSum16(b, 80))
		r3 := fwdAbsSum16(b, 96).Add(fwdAbsSum16(b, 112))
		if fwdRowSkipGroupLive(r0, r1, r2, r3, t, slack) {
			mask |= 1 << g
		}
	}
	if mask == 0 {
		return 0
	}
	zero := archsimd.BroadcastInt32x4(0)
	for h := 0; h < 32; h += 4 {
		if mask&(1<<(h>>2)) == 0 {
			for l := 0; l < 32; l++ {
				fwdStoreI32At(coeff[:], l*32+h, zero)
			}
			continue
		}
		for bj := 0; bj < 32; bj += 4 {
			fwdTransposeBlock4(bufT[bj*32+h:], 32, buf[h*32+bj:], 32)
		}
		fwdRowDCT32(coeff[:], 32, bufT[:], h)
	}
	return mask
}
