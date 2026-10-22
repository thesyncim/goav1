// Ported from SVT-AV1 v4.1.0:
//   Source/Lib/ASM_NEON/highbd_fwd_txfm_neon.c (svt_av1_fwd_txfm2d_8x8_N4_neon,
//   highbd_fdct8_N4_x4_neon)
//   Source/Lib/Codec/transforms.c (av1_tranform_two_d_core_N4_c)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

// ForwardDCT8x8N48BitTrusted is SVT's N4_SHAPE 8x8 DCT_DCT
// (svt_av1_fwd_txfm2d_8x8_N4): only the 2x2 lowest-frequency coefficients
// are computed (with exactly the values of the full transform: the pruned
// network drops the butterflies that feed outputs 2..7 only) and every other
// coefficient is zero, in the decoder layout coeff[col*8+row]. residual is an
// 8-bit residual block at stride 8.
func ForwardDCT8x8N48BitTrusted(coeff *[64]int32, residual *[64]int16) {
	forwardDCT8x8N4Impl(coeff, residual)
}

// fwdDCT8N4Values returns outputs 0 and 1 of the cos-bit-13 DCT8 (fwdDCT8).
func fwdDCT8N4Values(x0, x1, x2, x3, x4, x5, x6, x7 int32) (int32, int32) {
	c := &fwdCospi13
	b0, b1, b2, b3 := x0+x7, x1+x6, x2+x5, x3+x4
	b4, b5, b6, b7 := x3-x4, x2-x5, x1-x6, x0-x7
	out0 := fwdHalfBtf13(c[32], b0+b3, c[32], b1+b2)
	s5 := fwdHalfBtf13(-c[32], b5, c[32], b6)
	s6 := fwdHalfBtf13(c[32], b6, c[32], b5)
	out1 := fwdHalfBtf13(c[56], b4+s5, c[8], b7+s6)
	return out0, out1
}

// forwardDCT8x8N4RowPass finishes the N4 transform from the rounded column
// outputs t0 (vertical frequency 0) and t1 (frequency 1) of the 8 columns.
func forwardDCT8x8N4RowPass(coeff *[64]int32, t0, t1 *[8]int32) {
	*coeff = [64]int32{}
	a0, a1 := fwdDCT8N4Values(t0[0], t0[1], t0[2], t0[3], t0[4], t0[5], t0[6], t0[7])
	b0, b1 := fwdDCT8N4Values(t1[0], t1[1], t1[2], t1[3], t1[4], t1[5], t1[6], t1[7])
	coeff[0], coeff[8] = a0, a1
	coeff[1], coeff[9] = b0, b1
}

func forwardDCT8x8N4PureGo(coeff *[64]int32, residual *[64]int16) {
	var t0, t1 [8]int32
	for c := range 8 {
		var x [8]int32
		for r := range 8 {
			x[r] = int32(residual[r*8+c]) << 2
		}
		o0, o1 := fwdDCT8N4Values(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
		t0[c], t1[c] = fwdRoundShift1Value(o0), fwdRoundShift1Value(o1)
	}
	forwardDCT8x8N4RowPass(coeff, &t0, &t1)
}
