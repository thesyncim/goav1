// Ported from SVT-AV1 v4.1.0:
//   Source/Lib/ASM_NEON/highbd_fwd_txfm_neon.c (highbd_fdct8_N4_x4_neon,
//   svt_av1_fwd_txfm2d_8x8_N4_neon)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package transform

// forwardDCT8x8N4SIMD is forwardDCT8x8N4PureGo with the column pass, which
// keeps only outputs 0 and 1 of each column's DCT8 (highbd_fdct8_N4), across
// fwdLanes columns per vector; the two kept rows' row pass is scalar.
func forwardDCT8x8N4SIMD(coeff *[64]int32, residual *[64]int16) {
	var t0, t1 [8]int32
	c32, nc32, c8, c56 := fwdConst5793, fwdConstN5793, fwdConst8035, fwdConst1598
	for g := 0; g < 8; g += fwdLanes {
		x0 := fwdLoadResAt(residual[:], 0*8+g&^7, g)
		x1 := fwdLoadResAt(residual[:], 1*8+g&^7, g)
		x2 := fwdLoadResAt(residual[:], 2*8+g&^7, g)
		x3 := fwdLoadResAt(residual[:], 3*8+g&^7, g)
		x4 := fwdLoadResAt(residual[:], 4*8+g&^7, g)
		x5 := fwdLoadResAt(residual[:], 5*8+g&^7, g)
		x6 := fwdLoadResAt(residual[:], 6*8+g&^7, g)
		x7 := fwdLoadResAt(residual[:], 7*8+g&^7, g)
		b0, b1, b2, b3 := x0.Add(x7), x1.Add(x6), x2.Add(x5), x3.Add(x4)
		b4, b5, b6, b7 := x3.Sub(x4), x2.Sub(x5), x1.Sub(x6), x0.Sub(x7)
		o0 := fwdHalfBtf13V(c32, b0.Add(b3), c32, b1.Add(b2))
		s5 := fwdHalfBtf13V(nc32, b5, c32, b6)
		s6 := fwdHalfBtf13V(c32, b6, c32, b5)
		o1 := fwdHalfBtf13V(c56, b4.Add(s5), c8, b7.Add(s6))
		fwdStoreI32At(t0[:], g, fwdRoundShift1V(o0))
		fwdStoreI32At(t1[:], g, fwdRoundShift1V(o1))
	}
	forwardDCT8x8N4RowPass(coeff, &t0, &t1)
}
