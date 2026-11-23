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

// fwdRowSkipSumsLive takes the magnitude sums s of four column-pass rows (one
// row per lane) and reports whether any row's S + S>>6 + slack reaches its
// threshold lane.
func fwdRowSkipSumsLive(s, thr, slack archsimd.Int32x4) bool {
	m := s.Add(s.Shift(fwdRowSkipShrNeg6)).Add(slack).Sub(thr)
	return m.ReduceMax() >= 0
}

// forwardDCT16x16RowSkipSIMD is forwardDCT16x16RowSkipPureGo over the
// forwardDCT16x16SIMD passes: the column pass runs in full into coeff
// (transposed), then each group of four column-pass rows is bounded from it
// and only live groups are row-transformed in place.
func forwardDCT16x16RowSkipSIMD(coeff *[256]int32, residual *[256]int16, thr0, thr int32) uint8 {
	for g := 0; g < 16; g += fwdLanes {
		fwdColDCT16(coeff[:], g, residual[:], 16)
	}
	var sums [4]archsimd.Int32x4
	for h := 0; h < 16; h += 4 {
		r := (*[15*16 + 4]int32)(coeff[h:])
		var s0, s1 archsimd.Int32x4
		for c := 0; c < 16; c += 2 {
			s0 = s0.Add(fwdLoadI32Arr(r[:], c*16).Abs())
			s1 = s1.Add(fwdLoadI32Arr(r[:], c*16+16).Abs())
		}
		sums[h>>2] = s0.Add(s1)
	}
	return fwdRowSkipRows(coeff[:], 16, sums[:], thr0, thr, fwdRowSkipSlack16V, fwdRowDCT16)
}

// forwardDCT32x32RowSkipSIMD is forwardDCT32x32RowSkipPureGo over the
// forwardDCT32x32SIMD passes (see forwardDCT16x16RowSkipSIMD).
func forwardDCT32x32RowSkipSIMD(coeff *[1024]int32, residual *[1024]int16, thr0, thr int32) uint8 {
	for g := 0; g < 32; g += fwdLanes {
		fwdColDCT32(coeff[:], g, residual[:], 32)
	}
	var sums [8]archsimd.Int32x4
	for h := 0; h < 32; h += 4 {
		r := (*[31*32 + 4]int32)(coeff[h:])
		var s0, s1 archsimd.Int32x4
		for c := 0; c < 32; c += 2 {
			s0 = s0.Add(fwdLoadI32Arr(r[:], c*32).Abs())
			s1 = s1.Add(fwdLoadI32Arr(r[:], c*32+32).Abs())
		}
		sums[h>>2] = s0.Add(s1)
	}
	return fwdRowSkipRows(coeff[:], 32, sums[:], thr0, thr, fwdRowSkipSlack32V, fwdRowDCT32)
}

// fwdRowSkipRows takes the row magnitude sums of each group of four rows of
// the transposed n x n column-pass output c (sums[g] covers rows 4g..4g+3),
// row-transforms the live groups in place and zeroes the others, returning
// the live-group mask (c is left as is when none is).
func fwdRowSkipRows(c []int32, n int, sums []archsimd.Int32x4, thr0, thr int32, slack archsimd.Int32x4, row func([]int32, int)) uint8 {
	thrRest := archsimd.BroadcastInt32x4(thr)
	t := thrRest.SetElem(0, thr0)
	var mask uint8
	for g, s := range sums {
		if fwdRowSkipSumsLive(s, t, slack) {
			mask |= 1 << g
		}
		t = thrRest
	}
	if mask == 0 {
		return 0
	}
	zero := archsimd.BroadcastInt32x4(0)
	for h := 0; h < n; h += 4 {
		if mask&(1<<(h>>2)) == 0 {
			for l := 0; l < n; l++ {
				zero.Store(c[l*n+h:])
			}
			continue
		}
		row(c, h)
	}
	return mask
}
