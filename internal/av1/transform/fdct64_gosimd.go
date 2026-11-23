// Ported from SVT-AV1 v4.1.0:
//   Source/Lib/ASM_NEON/highbd_fwd_txfm_neon.c (highbd_fdct64_N2_x4_neon,
//   butterfly_dct_pre/post, butterfly_{0112,2312,0332,0130}_neon,
//   svt_av1_fwd_txfm2d_64x64_neon's column/row pass shape)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package transform

// The 64x64 forward DCT_DCT in SVT-AV1's NEON shape: a column pass of the
// 64-point DCT over fwdLanes columns at a time (cos_bit 13), the shift[1] = -2
// round, a transpose, then the row pass (cos_bit 10) and the shift[2] = -2
// round. AV1 codes only the low 32x32 coefficients of a TX_64X64
// (av1_get_adjusted_tx_size), so each 1-D pass is SVT's half-output variant
// highbd_fdct64_N2_x4_neon: the butterflies that only feed outputs 32..63 are
// dropped and the remaining ones are computed exactly as the full transform
// computes them. The row pass therefore runs over the 32 kept column outputs
// only. Intermediates stay in int32 lanes: with 8-bit residuals every
// product and sum fits (the scalar reference evaluates half_btf in int64,
// and the two agree wherever int32 does not overflow).

// fwd64Twiddles holds one cos_bit's cospi table as broadcast vectors:
// c[i] = cospi[i] and n[i] = -cospi[i] for i in 0..64 (cospi[64] = 0), so
// SVT's (w0, w1) = (cospi[i], cospi[64-i]) pairs are c[i], c[64-i].
type fwd64Twiddles struct {
	c, n [65]fwdVec
	rnd  fwdVec
	shr  fwdShift
}

func newFwd64Twiddles(cospi *[64]int32, bit uint64) *fwd64Twiddles {
	t := &fwd64Twiddles{shr: fwdMakeShift(bit), rnd: fwdBcast(1 << (bit - 1))}
	for i := range 64 {
		t.c[i] = fwdBcast(cospi[i])
		t.n[i] = fwdBcast(-cospi[i])
	}
	t.c[64] = fwdBcast(0)
	t.n[64] = fwdBcast(0)
	return t
}

var (
	fwd64Tw13 = newFwd64Twiddles(&fwdCospi13, 13)
	fwd64Tw10 = newFwd64Twiddles(&fwdCospi10, 10)
)
