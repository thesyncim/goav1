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

// bf is one butterfly_half_neon output: round_shift(a*wa + b*wb, bit), the
// rounding bias folded into the accumulator.
func (t *fwd64Twiddles) bf(a, wa, b, wb fwdVec) fwdVec {
	return fwdShrBy(fwdMulAdd(b, wb, fwdMulAdd(a, wa, t.rnd)), t.shr)
}

// b0112 is butterfly_0112_neon: (n0*w0 + n1*w1, n0*w1 - n1*w0).
func (t *fwd64Twiddles) b0112(w int, n0, n1 fwdVec) (fwdVec, fwdVec) {
	return t.bf(n0, t.c[w], n1, t.c[64-w]), t.bf(n0, t.c[64-w], n1, t.n[w])
}

// b2312 is butterfly_2312_neon: (-n0*w0 - n1*w1, n0*w1 - n1*w0).
func (t *fwd64Twiddles) b2312(w int, n0, n1 fwdVec) (fwdVec, fwdVec) {
	return t.bf(n0, t.n[w], n1, t.n[64-w]), t.bf(n0, t.c[64-w], n1, t.n[w])
}

// b0332 is butterfly_0332_neon: (n0*w0 - n1*w1, -n0*w1 - n1*w0).
func (t *fwd64Twiddles) b0332(w int, n0, n1 fwdVec) (fwdVec, fwdVec) {
	return t.bf(n0, t.c[w], n1, t.n[64-w]), t.bf(n0, t.n[64-w], n1, t.n[w])
}

// b0130 is butterfly_0130_neon: (n0*w0 + n1*w1, -n0*w1 + n1*w0).
func (t *fwd64Twiddles) b0130(w int, n0, n1 fwdVec) (fwdVec, fwdVec) {
	return t.bf(n0, t.c[w], n1, t.c[64-w]), t.bf(n0, t.n[64-w], n1, t.c[w])
}

// h01 is butterfly_half_neon with lanes (0, 1): a*w0 + b*w1.
func (t *fwd64Twiddles) h01(w int, a, b fwdVec) fwdVec {
	return t.bf(a, t.c[w], b, t.c[64-w])
}

// h30 is butterfly_half_neon with lanes (3, 0): -a*w1 + b*w0.
func (t *fwd64Twiddles) h30(w int, a, b fwdVec) fwdVec {
	return t.bf(a, t.n[64-w], b, t.c[w])
}

// fwdCopyVecs copies src into dst one vector register at a time. The builtin
// copy lowers these fixed-size moves to legacy-SSE movups pairs on amd64,
// which stall on Zen 3 while the upper ymm state is dirty.
func fwdCopyVecs(dst, src []fwdVec) {
	src = src[:len(dst)]
	for i := range dst {
		dst[i] = src[i]
	}
}

// fwd64DCTPre is butterfly_dct_pre over in[0:n] into out[0:n].
func fwd64DCTPre(in, out *[64]fwdVec, n int) {
	switch n {
	case 64:
		fwd64Pre64(in, out)
	case 32:
		fwd64Pre32((*[32]fwdVec)(in[:32]), (*[32]fwdVec)(out[:32]))
	case 16:
		fwd64Pre16((*[16]fwdVec)(in[:16]), (*[16]fwdVec)(out[:16]))
	case 8:
		fwd64Pre8((*[8]fwdVec)(in[:8]), (*[8]fwdVec)(out[:8]))
	default:
		fwd64Pre4((*[4]fwdVec)(in[:4]), (*[4]fwdVec)(out[:4]))
	}
}

func fwd64Pre64(a, o *[64]fwdVec) {
	for i := range 32 {
		o[i] = a[i].Add(a[63-i])
		o[32+i] = a[31-i].Sub(a[32+i])
	}
}

func fwd64Pre32(a, o *[32]fwdVec) {
	for i := range 16 {
		o[i] = a[i].Add(a[31-i])
		o[16+i] = a[15-i].Sub(a[16+i])
	}
}

func fwd64Pre16(a, o *[16]fwdVec) {
	for i := range 8 {
		o[i] = a[i].Add(a[15-i])
		o[8+i] = a[7-i].Sub(a[8+i])
	}
}

func fwd64Pre8(a, o *[8]fwdVec) {
	o[0] = a[0].Add(a[7])
	o[1] = a[1].Add(a[6])
	o[2] = a[2].Add(a[5])
	o[3] = a[3].Add(a[4])
	o[4] = a[3].Sub(a[4])
	o[5] = a[2].Sub(a[5])
	o[6] = a[1].Sub(a[6])
	o[7] = a[0].Sub(a[7])
}

func fwd64Pre4(a, o *[4]fwdVec) {
	o[0] = a[0].Add(a[3])
	o[1] = a[1].Add(a[2])
	o[2] = a[1].Sub(a[2])
	o[3] = a[0].Sub(a[3])
}

// fwd64DCTPost is butterfly_dct_post(in+off, in+off, out+off, n).
func fwd64DCTPost(in, out *[64]fwdVec, off, n int) {
	switch n {
	case 32:
		fwd64Post32((*[32]fwdVec)(in[off:off+32]), (*[32]fwdVec)(out[off:off+32]))
	case 16:
		fwd64Post16((*[16]fwdVec)(in[off:off+16]), (*[16]fwdVec)(out[off:off+16]))
	case 8:
		fwd64Post8((*[8]fwdVec)(in[off:off+8]), (*[8]fwdVec)(out[off:off+8]))
	default:
		fwd64Post4((*[4]fwdVec)(in[off:off+4]), (*[4]fwdVec)(out[off:off+4]))
	}
}

func fwd64Post32(a, o *[32]fwdVec) {
	for i := range 8 {
		o[i] = a[i].Add(a[15-i])
		o[8+i] = a[7-i].Sub(a[8+i])
		o[16+i] = a[31-i].Sub(a[16+i])
		o[24+i] = a[24+i].Add(a[23-i])
	}
}

func fwd64Post16(a, o *[16]fwdVec) {
	for i := range 4 {
		o[i] = a[i].Add(a[7-i])
		o[4+i] = a[3-i].Sub(a[4+i])
		o[8+i] = a[15-i].Sub(a[8+i])
		o[12+i] = a[12+i].Add(a[11-i])
	}
}

func fwd64Post8(a, o *[8]fwdVec) {
	o[0] = a[0].Add(a[3])
	o[1] = a[1].Add(a[2])
	o[2] = a[1].Sub(a[2])
	o[3] = a[0].Sub(a[3])
	o[4] = a[7].Sub(a[4])
	o[5] = a[6].Sub(a[5])
	o[6] = a[6].Add(a[5])
	o[7] = a[7].Add(a[4])
}

func fwd64Post4(a, o *[4]fwdVec) {
	o[0] = a[0].Add(a[1])
	o[1] = a[0].Sub(a[1])
	o[2] = a[3].Sub(a[2])
	o[3] = a[3].Add(a[2])
}

// fwdDCT64N2 is highbd_fdct64_N2_x4_neon over fwdLanes lines: in holds the 64
// inputs, out receives the first 32 outputs in natural frequency order.
func fwdDCT64N2(t *fwd64Twiddles, in *[64]fwdVec, out *[32]fwdVec) {
	var x1, x2 [64]fwdVec

	// stage 1
	fwd64DCTPre(in, &x1, 64)

	// stage 2
	fwd64DCTPre(&x1, &x2, 32)
	fwdCopyVecs(x2[32:40], x1[32:40])
	x2[55], x2[40] = t.b0112(32, x1[55], x1[40])
	x2[54], x2[41] = t.b0112(32, x1[54], x1[41])
	x2[53], x2[42] = t.b0112(32, x1[53], x1[42])
	x2[52], x2[43] = t.b0112(32, x1[52], x1[43])
	x2[51], x2[44] = t.b0112(32, x1[51], x1[44])
	x2[50], x2[45] = t.b0112(32, x1[50], x1[45])
	x2[49], x2[46] = t.b0112(32, x1[49], x1[46])
	x2[48], x2[47] = t.b0112(32, x1[48], x1[47])
	fwdCopyVecs(x2[56:64], x1[56:64])

	// stage 3 (into x1)
	x3 := &x1
	fwd64DCTPre(&x2, x3, 16)
	fwdCopyVecs(x3[16:20], x2[16:20])
	x3[27], x3[20] = t.b0112(32, x2[27], x2[20])
	x3[26], x3[21] = t.b0112(32, x2[26], x2[21])
	x3[25], x3[22] = t.b0112(32, x2[25], x2[22])
	x3[24], x3[23] = t.b0112(32, x2[24], x2[23])
	fwdCopyVecs(x3[28:32], x2[28:32])
	fwd64DCTPost(&x2, x3, 32, 32)

	// stage 4 (into x2)
	x4 := &x2
	fwd64DCTPre(x3, x4, 8)
	x4[8] = x3[8]
	x4[9] = x3[9]
	x4[13], x4[10] = t.b0112(32, x3[13], x3[10])
	x4[12], x4[11] = t.b0112(32, x3[12], x3[11])
	x4[14] = x3[14]
	x4[15] = x3[15]
	fwd64DCTPost(x3, x4, 16, 16)
	fwdCopyVecs(x4[32:36], x3[32:36])
	x4[59], x4[36] = t.b0112(16, x3[59], x3[36])
	x4[58], x4[37] = t.b0112(16, x3[58], x3[37])
	x4[57], x4[38] = t.b0112(16, x3[57], x3[38])
	x4[56], x4[39] = t.b0112(16, x3[56], x3[39])
	x4[40], x4[55] = t.b2312(16, x3[55], x3[40])
	x4[41], x4[54] = t.b2312(16, x3[54], x3[41])
	x4[42], x4[53] = t.b2312(16, x3[53], x3[42])
	x4[43], x4[52] = t.b2312(16, x3[52], x3[43])
	fwdCopyVecs(x4[44:52], x3[44:52])
	fwdCopyVecs(x4[60:64], x3[60:64])

	// stage 5 (into x1)
	x5 := &x1
	fwd64DCTPre(x4, x5, 4)
	x5[4] = x4[4]
	x5[6], x5[5] = t.b0112(32, x4[6], x4[5])
	x5[7] = x4[7]
	fwd64DCTPost(x4, x5, 8, 8)
	x5[16] = x4[16]
	x5[17] = x4[17]
	x5[29], x5[18] = t.b0112(16, x4[29], x4[18])
	x5[28], x5[19] = t.b0112(16, x4[28], x4[19])
	x5[20], x5[27] = t.b2312(16, x4[27], x4[20])
	x5[21], x5[26] = t.b2312(16, x4[26], x4[21])
	fwdCopyVecs(x5[22:26], x4[22:26])
	x5[30] = x4[30]
	x5[31] = x4[31]
	fwd64DCTPost(x4, x5, 32, 16)
	fwd64DCTPost(x4, x5, 48, 16)

	// stage 6 (into x2)
	x6 := &x2
	x6[0] = t.h01(32, x5[0], x5[1])
	x6[2] = t.h01(16, x5[3], x5[2])
	fwd64DCTPost(x5, x6, 4, 4)
	x6[8] = x5[8]
	x6[14], x6[9] = t.b0112(16, x5[14], x5[9])
	x6[10], x6[13] = t.b2312(16, x5[13], x5[10])
	x6[11] = x5[11]
	x6[12] = x5[12]
	x6[15] = x5[15]
	fwd64DCTPost(x5, x6, 16, 8)
	fwd64DCTPost(x5, x6, 24, 8)
	x6[32] = x5[32]
	x6[33] = x5[33]
	x6[61], x6[34] = t.b0112(8, x5[61], x5[34])
	x6[60], x6[35] = t.b0112(8, x5[60], x5[35])
	x6[36], x6[59] = t.b2312(8, x5[59], x5[36])
	x6[37], x6[58] = t.b2312(8, x5[58], x5[37])
	fwdCopyVecs(x6[38:42], x5[38:42])
	x6[53], x6[42] = t.b0130(24, x5[42], x5[53])
	x6[52], x6[43] = t.b0130(24, x5[43], x5[52])
	x6[51], x6[44] = t.b0332(24, x5[51], x5[44])
	x6[50], x6[45] = t.b0332(24, x5[50], x5[45])
	fwdCopyVecs(x6[46:50], x5[46:50])
	fwdCopyVecs(x6[54:58], x5[54:58])
	x6[62] = x5[62]
	x6[63] = x5[63]

	// stage 7 (into x1)
	x7 := &x1
	x7[0] = x6[0]
	x7[2] = x6[2]
	x7[4] = t.h01(8, x6[7], x6[4])
	x7[6] = t.h30(24, x6[5], x6[6])
	fwd64DCTPost(x6, x7, 8, 4)
	fwd64DCTPost(x6, x7, 12, 4)
	x7[16] = x6[16]
	x7[30], x7[17] = t.b0112(8, x6[30], x6[17])
	x7[18], x7[29] = t.b2312(8, x6[29], x6[18])
	x7[19] = x6[19]
	x7[20] = x6[20]
	x7[26], x7[21] = t.b0130(24, x6[21], x6[26])
	x7[25], x7[22] = t.b0332(24, x6[25], x6[22])
	x7[23] = x6[23]
	x7[24] = x6[24]
	x7[27] = x6[27]
	x7[28] = x6[28]
	x7[31] = x6[31]
	fwd64DCTPost(x6, x7, 32, 8)
	fwd64DCTPost(x6, x7, 40, 8)
	fwd64DCTPost(x6, x7, 48, 8)
	fwd64DCTPost(x6, x7, 56, 8)

	// stage 8 (into x2)
	x8 := &x2
	x8[0] = x7[0]
	x8[2] = x7[2]
	x8[4] = x7[4]
	x8[6] = x7[6]
	x8[8] = t.h01(4, x7[15], x7[8])
	x8[14] = t.h30(28, x7[9], x7[14])
	x8[10] = t.h01(20, x7[13], x7[10])
	x8[12] = t.h30(12, x7[11], x7[12])
	fwd64DCTPost(x7, x8, 16, 4)
	fwd64DCTPost(x7, x8, 20, 4)
	fwd64DCTPost(x7, x8, 24, 4)
	fwd64DCTPost(x7, x8, 28, 4)
	x8[32] = x7[32]
	x8[62], x8[33] = t.b0112(4, x7[62], x7[33])
	x8[34], x8[61] = t.b2312(4, x7[61], x7[34])
	x8[35] = x7[35]
	x8[36] = x7[36]
	x8[58], x8[37] = t.b0130(28, x7[37], x7[58])
	x8[57], x8[38] = t.b0332(28, x7[57], x7[38])
	x8[39] = x7[39]
	x8[40] = x7[40]
	x8[54], x8[41] = t.b0112(20, x7[54], x7[41])
	x8[42], x8[53] = t.b2312(20, x7[53], x7[42])
	x8[43] = x7[43]
	x8[44] = x7[44]
	x8[50], x8[45] = t.b0130(12, x7[45], x7[50])
	x8[49], x8[46] = t.b0332(12, x7[49], x7[46])
	x8[47] = x7[47]
	x8[48] = x7[48]
	x8[51] = x7[51]
	x8[52] = x7[52]
	x8[55] = x7[55]
	x8[56] = x7[56]
	x8[59] = x7[59]
	x8[60] = x7[60]
	x8[63] = x7[63]

	// stage 9 (into x1)
	x9 := &x1
	x9[0] = x8[0]
	x9[2] = x8[2]
	x9[4] = x8[4]
	x9[6] = x8[6]
	x9[8] = x8[8]
	x9[10] = x8[10]
	x9[12] = x8[12]
	x9[14] = x8[14]
	x9[16] = t.h01(2, x8[31], x8[16])
	x9[30] = t.h30(30, x8[17], x8[30])
	x9[18] = t.h01(18, x8[29], x8[18])
	x9[28] = t.h30(14, x8[19], x8[28])
	x9[20] = t.h01(10, x8[27], x8[20])
	x9[26] = t.h30(22, x8[21], x8[26])
	x9[22] = t.h01(26, x8[25], x8[22])
	x9[24] = t.h30(6, x8[23], x8[24])
	fwd64DCTPost(x8, x9, 32, 4)
	fwd64DCTPost(x8, x9, 36, 4)
	fwd64DCTPost(x8, x9, 40, 4)
	fwd64DCTPost(x8, x9, 44, 4)
	fwd64DCTPost(x8, x9, 48, 4)
	fwd64DCTPost(x8, x9, 52, 4)
	fwd64DCTPost(x8, x9, 56, 4)
	fwd64DCTPost(x8, x9, 60, 4)

	// stage 10 + 11: the odd outputs come from the x9[32..63] butterflies;
	// the even outputs are the x9 entries above in bit-reversed order.
	out[0] = x9[0]
	out[1] = t.h01(1, x9[63], x9[32])
	out[2] = x9[16]
	out[3] = t.h30(3, x9[47], x9[48])
	out[4] = x9[8]
	out[5] = t.h01(5, x9[55], x9[40])
	out[6] = x9[24]
	out[7] = t.h30(7, x9[39], x9[56])
	out[8] = x9[4]
	out[9] = t.h01(9, x9[59], x9[36])
	out[10] = x9[20]
	out[11] = t.h30(11, x9[43], x9[52])
	out[12] = x9[12]
	out[13] = t.h01(13, x9[51], x9[44])
	out[14] = x9[28]
	out[15] = t.h30(15, x9[35], x9[60])
	out[16] = x9[2]
	out[17] = t.h01(17, x9[61], x9[34])
	out[18] = x9[18]
	out[19] = t.h30(19, x9[45], x9[50])
	out[20] = x9[10]
	out[21] = t.h01(21, x9[53], x9[42])
	out[22] = x9[26]
	out[23] = t.h30(23, x9[37], x9[58])
	out[24] = x9[6]
	out[25] = t.h01(25, x9[57], x9[38])
	out[26] = x9[22]
	out[27] = t.h30(27, x9[41], x9[54])
	out[28] = x9[14]
	out[29] = t.h01(29, x9[49], x9[46])
	out[30] = x9[30]
	out[31] = t.h30(31, x9[33], x9[62])
}

// forwardDCT64x64SIMD is svt_av1_fwd_txfm2d_64x64's DCT_DCT case for 8-bit
// residuals, writing the coded 32x32 coefficients in coeff[k*coeffStride+r]
// layout (the frequency column k, row r) like ForwardDCT64x64.
func forwardDCT64x64SIMD(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	// buf holds the column pass outputs row-major (32 kept rows x 64
	// columns); bufT their transpose (64 columns x 32 rows).
	var buf, bufT [32 * 64]int32
	var in [64]fwdVec
	var out [32]fwdVec
	for g := 0; g < 64; g += fwdLanes {
		for r := range 64 {
			in[r] = fwdLoadResRawAt(residual, r*residualStride+g&^7, g)
		}
		fwdDCT64N2(fwd64Tw13, &in, &out)
		for r := range 32 {
			fwdStoreI32At(buf[:], r*64+g, fwdShr(out[r].Add(fwdConst2), 2))
		}
	}
	fwdTransposeRect(bufT[:], buf[:], 32, 64)
	for h := 0; h < 32; h += fwdLanes {
		for c := range 64 {
			in[c] = fwdLoadI32At(bufT[:], c*32+h)
		}
		fwdDCT64N2(fwd64Tw10, &in, &out)
		for k := range 32 {
			fwdStoreI32At(coeff, k*coeffStride+h, fwdShr(out[k].Add(fwdConst2), 2))
		}
	}
}
