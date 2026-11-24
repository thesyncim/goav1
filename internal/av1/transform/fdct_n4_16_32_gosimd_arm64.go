// Ported from SVT-AV1 v4.1.0:
//   Source/Lib/ASM_NEON/highbd_fwd_txfm_neon.c (svt_av1_fwd_txfm2d_16x16_N4_neon,
//   svt_av1_fwd_txfm2d_32x32_N4_neon)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "simd/archsimd"

// The arm64 N4 kernels run the generated passes (fdct_n4_gen_arm64.go): the
// column pass stores its n/4 kept outputs transposed into buf (buf[c*n/4+k]),
// so each group of four rows is one vector per column for the row pass and
// the row-skip bound.

func forwardDCT16x16N4(coeff *[256]int32, residual *[256]int16) {
	var buf [16 * 4]int32
	fwdN4Cols16(&buf, residual)
	fwdRowDCT16N4T(coeff[:], buf[:], 0)
}

func forwardDCT32x32N4(coeff *[1024]int32, residual *[1024]int16) {
	var buf [32 * 8]int32
	fwdN4Cols32(&buf, residual)
	fwdRowDCT32N4T(coeff[:], buf[:], 0)
	fwdRowDCT32N4T(coeff[:], buf[:], 4)
}

func forwardDCT16x16N4RowSkip(coeff *[256]int32, residual *[256]int16, thr0, thr int32) uint8 {
	var buf [16 * 4]int32
	fwdN4Cols16(&buf, residual)
	var s0, s1 archsimd.Int32x4
	for c := 0; c < 16; c += 2 {
		s0 = s0.Add(fwdLoadI32Arr(buf[:], c*4).Abs())
		s1 = s1.Add(fwdLoadI32Arr(buf[:], c*4+4).Abs())
	}
	t := archsimd.BroadcastInt32x4(thr).SetElem(0, thr0)
	if !fwdRowSkipSumsLive(s0.Add(s1), t, fwdRowSkipSlack16V) {
		return 0
	}
	fwdRowDCT16N4T(coeff[:], buf[:], 0)
	return 1
}

func forwardDCT32x32N4RowSkip(coeff *[1024]int32, residual *[1024]int16, thr0, thr int32) uint8 {
	var buf [32 * 8]int32
	fwdN4Cols32(&buf, residual)
	thrRest := archsimd.BroadcastInt32x4(thr)
	t := thrRest.SetElem(0, thr0)
	var mask uint8
	for h := 0; h < 8; h += 4 {
		r := (*[31*8 + 4]int32)(buf[h:])
		var s0, s1 archsimd.Int32x4
		for c := 0; c < 32; c += 2 {
			s0 = s0.Add(fwdLoadI32Arr(r[:], c*8).Abs())
			s1 = s1.Add(fwdLoadI32Arr(r[:], c*8+8).Abs())
		}
		if fwdRowSkipSumsLive(s0.Add(s1), t, fwdRowSkipSlack32V) {
			mask |= 1 << (h >> 2)
		}
		t = thrRest
	}
	if mask == 0 {
		return 0
	}
	zero := fwdBcast(0)
	for h := 0; h < 8; h += 4 {
		if mask&(1<<(h>>2)) == 0 {
			for k := range 8 {
				fwdStoreI32At(coeff[:], k*32+h, zero)
			}
			continue
		}
		fwdRowDCT32N4T(coeff[:], buf[:], h)
	}
	return mask
}

func fwdN4Cols16(buf *[16 * 4]int32, residual *[256]int16) {
	for g := 0; g < 16; g += fwdLanes {
		fwdColDCT16N4T(buf[:], g, residual)
	}
}

func fwdN4Cols32(buf *[32 * 8]int32, residual *[1024]int16) {
	for g := 0; g < 32; g += fwdLanes {
		fwdColDCT32N4T(buf[:], g, residual)
	}
}
