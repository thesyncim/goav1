// Ported from libaom:
//   av1/encoder/av1_fwd_txfm1d.c (av1_fdct4, av1_fdct8, av1_fdct16, av1_fdct32)
//   av1/encoder/av1_fwd_txfm2d.c (fwd_txfm2d_c, fwd_shift_4x4, fwd_shift_8x8,
//   fwd_shift_16x16, fwd_shift_32x32, av1_fwd_cos_bit_col/row)
//   av1/common/av1_txfm.c (av1_cospi_arr_data, cos_bit 13 and 12 rows)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package transform

// fwdCospi13 is av1_cospi_arr_data[13-cos_bit_min]: cos(i*pi/64) in Q13. Both
// the 4x4 and 8x8 forward DCT columns and rows use cos_bit 13
// (av1_fwd_cos_bit_col/row).
var fwdCospi13 = [64]int32{
	8192, 8190, 8182, 8170, 8153, 8130, 8103, 8071, 8035, 7993, 7946,
	7895, 7839, 7779, 7713, 7643, 7568, 7489, 7405, 7317, 7225, 7128,
	7027, 6921, 6811, 6698, 6580, 6458, 6333, 6203, 6070, 5933, 5793,
	5649, 5501, 5351, 5197, 5040, 4880, 4717, 4551, 4383, 4212, 4038,
	3862, 3683, 3503, 3320, 3135, 2948, 2760, 2570, 2378, 2185, 1990,
	1795, 1598, 1401, 1202, 1003, 803, 603, 402, 201,
}

// fwdCospi12 is av1_cospi_arr_data[12-cos_bit_min]: cos(i*pi/64) in Q12, the
// row-pass weights of the 16x16 forward DCT (av1_fwd_cos_bit_row[2][2] == 12).
var fwdCospi12 = [64]int32{
	4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973,
	3948, 3920, 3889, 3857, 3822, 3784, 3745, 3703, 3659, 3612, 3564,
	3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967, 2896,
	2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019,
	1931, 1842, 1751, 1660, 1567, 1474, 1380, 1285, 1189, 1092, 995,
	897, 799, 700, 601, 501, 401, 301, 201, 101,
}

// fwdHalfBtf13 is half_btf at cos_bit 13: the rounded Q13 butterfly
// w0*in0 + w1*in1 evaluated in 64-bit like libaom to avoid the 32-bit
// intermediate overflow case.
func fwdHalfBtf13(w0, in0, w1, in1 int32) int32 {
	result := int64(w0)*int64(in0) + int64(w1)*int64(in1)
	return int32((result + (1 << 12)) >> 13)
}

// fwdRoundShift1 is av1_round_shift_array with bit == 1 (the 8x8 shift[1] of
// -1): symmetric rounding right shift by one.
func fwdRoundShift1(arr []int32) {
	for i, v := range arr {
		// Branch-free symmetric rounding: identical to
		// v<0 ? -((-v+1)>>1) : (v+1)>>1 for all int32 inputs.
		arr[i] = (v + 1 + (v >> 31)) >> 1
	}
}

// fwdRoundShift2 is av1_round_shift_array with bit == 2 (the 16x16 shift[1]
// of -2): round_shift(v, 2) = (v + 2) >> 2.
func fwdRoundShift2(arr []int32) {
	for i, v := range arr {
		arr[i] = (v + 2) >> 2
	}
}

// fwdDCT4 is av1_fdct4 at cos_bit 13.
func fwdDCT4(input, output *[4]int32) {
	// stage 1
	s0 := input[0] + input[3]
	s1 := input[1] + input[2]
	s2 := -input[2] + input[1]
	s3 := -input[3] + input[0]
	// stage 2
	t0 := fwdHalfBtf13(fwdCospi13[32], s0, fwdCospi13[32], s1)
	t1 := fwdHalfBtf13(-fwdCospi13[32], s1, fwdCospi13[32], s0)
	t2 := fwdHalfBtf13(fwdCospi13[48], s2, fwdCospi13[16], s3)
	t3 := fwdHalfBtf13(fwdCospi13[48], s3, -fwdCospi13[16], s2)
	// stage 3 (bit-reverse permutation)
	output[0] = t0
	output[1] = t2
	output[2] = t1
	output[3] = t3
}

// fwdDCT8 is av1_fdct8 at cos_bit 13.
func fwdDCT8(input, output *[8]int32) {
	// stage 1
	var b [8]int32
	b[0] = input[0] + input[7]
	b[1] = input[1] + input[6]
	b[2] = input[2] + input[5]
	b[3] = input[3] + input[4]
	b[4] = -input[4] + input[3]
	b[5] = -input[5] + input[2]
	b[6] = -input[6] + input[1]
	b[7] = -input[7] + input[0]
	// stage 2
	var s [8]int32
	s[0] = b[0] + b[3]
	s[1] = b[1] + b[2]
	s[2] = -b[2] + b[1]
	s[3] = -b[3] + b[0]
	s[4] = b[4]
	s[5] = fwdHalfBtf13(-fwdCospi13[32], b[5], fwdCospi13[32], b[6])
	s[6] = fwdHalfBtf13(fwdCospi13[32], b[6], fwdCospi13[32], b[5])
	s[7] = b[7]
	// stage 3
	b[0] = fwdHalfBtf13(fwdCospi13[32], s[0], fwdCospi13[32], s[1])
	b[1] = fwdHalfBtf13(-fwdCospi13[32], s[1], fwdCospi13[32], s[0])
	b[2] = fwdHalfBtf13(fwdCospi13[48], s[2], fwdCospi13[16], s[3])
	b[3] = fwdHalfBtf13(fwdCospi13[48], s[3], -fwdCospi13[16], s[2])
	b[4] = s[4] + s[5]
	b[5] = -s[5] + s[4]
	b[6] = -s[6] + s[7]
	b[7] = s[7] + s[6]
	// stage 4
	s[0] = b[0]
	s[1] = b[1]
	s[2] = b[2]
	s[3] = b[3]
	s[4] = fwdHalfBtf13(fwdCospi13[56], b[4], fwdCospi13[8], b[7])
	s[5] = fwdHalfBtf13(fwdCospi13[24], b[5], fwdCospi13[40], b[6])
	s[6] = fwdHalfBtf13(fwdCospi13[24], b[6], -fwdCospi13[40], b[5])
	s[7] = fwdHalfBtf13(fwdCospi13[56], b[7], -fwdCospi13[8], b[4])
	// stage 5 (bit-reverse permutation)
	output[0] = s[0]
	output[1] = s[4]
	output[2] = s[2]
	output[3] = s[6]
	output[4] = s[1]
	output[5] = s[5]
	output[6] = s[3]
	output[7] = s[7]
}

// fwdHalfBtf is half_btf at an arbitrary cos_bit for the kernels whose row
// pass runs below Q13 (the 16x16 row pass uses cos_bit 12).
func fwdHalfBtf(w0, in0, w1, in1 int32, cosBit uint) int32 {
	result := int64(w0)*int64(in0) + int64(w1)*int64(in1)
	return int32((result + (1 << (cosBit - 1))) >> cosBit)
}

// fwdDCT16 is av1_fdct16 with the cospi table and cos_bit of the active pass.
func fwdDCT16(input, output *[16]int32, cospi *[64]int32, cosBit uint) {
	var step [16]int32
	// stage 1
	bf1 := output
	for i := range 8 {
		bf1[i] = input[i] + input[15-i]
		bf1[8+i] = -input[8+i] + input[7-i]
	}
	// stage 2
	bf0, bf1 := output, &step
	bf1[0] = bf0[0] + bf0[7]
	bf1[1] = bf0[1] + bf0[6]
	bf1[2] = bf0[2] + bf0[5]
	bf1[3] = bf0[3] + bf0[4]
	bf1[4] = -bf0[4] + bf0[3]
	bf1[5] = -bf0[5] + bf0[2]
	bf1[6] = -bf0[6] + bf0[1]
	bf1[7] = -bf0[7] + bf0[0]
	bf1[8] = bf0[8]
	bf1[9] = bf0[9]
	bf1[10] = fwdHalfBtf(-cospi[32], bf0[10], cospi[32], bf0[13], cosBit)
	bf1[11] = fwdHalfBtf(-cospi[32], bf0[11], cospi[32], bf0[12], cosBit)
	bf1[12] = fwdHalfBtf(cospi[32], bf0[12], cospi[32], bf0[11], cosBit)
	bf1[13] = fwdHalfBtf(cospi[32], bf0[13], cospi[32], bf0[10], cosBit)
	bf1[14] = bf0[14]
	bf1[15] = bf0[15]
	// stage 3
	bf0, bf1 = &step, output
	bf1[0] = bf0[0] + bf0[3]
	bf1[1] = bf0[1] + bf0[2]
	bf1[2] = -bf0[2] + bf0[1]
	bf1[3] = -bf0[3] + bf0[0]
	bf1[4] = bf0[4]
	bf1[5] = fwdHalfBtf(-cospi[32], bf0[5], cospi[32], bf0[6], cosBit)
	bf1[6] = fwdHalfBtf(cospi[32], bf0[6], cospi[32], bf0[5], cosBit)
	bf1[7] = bf0[7]
	bf1[8] = bf0[8] + bf0[11]
	bf1[9] = bf0[9] + bf0[10]
	bf1[10] = -bf0[10] + bf0[9]
	bf1[11] = -bf0[11] + bf0[8]
	bf1[12] = -bf0[12] + bf0[15]
	bf1[13] = -bf0[13] + bf0[14]
	bf1[14] = bf0[14] + bf0[13]
	bf1[15] = bf0[15] + bf0[12]
	// stage 4
	bf0, bf1 = output, &step
	bf1[0] = fwdHalfBtf(cospi[32], bf0[0], cospi[32], bf0[1], cosBit)
	bf1[1] = fwdHalfBtf(-cospi[32], bf0[1], cospi[32], bf0[0], cosBit)
	bf1[2] = fwdHalfBtf(cospi[48], bf0[2], cospi[16], bf0[3], cosBit)
	bf1[3] = fwdHalfBtf(cospi[48], bf0[3], -cospi[16], bf0[2], cosBit)
	bf1[4] = bf0[4] + bf0[5]
	bf1[5] = -bf0[5] + bf0[4]
	bf1[6] = -bf0[6] + bf0[7]
	bf1[7] = bf0[7] + bf0[6]
	bf1[8] = bf0[8]
	bf1[9] = fwdHalfBtf(-cospi[16], bf0[9], cospi[48], bf0[14], cosBit)
	bf1[10] = fwdHalfBtf(-cospi[48], bf0[10], -cospi[16], bf0[13], cosBit)
	bf1[11] = bf0[11]
	bf1[12] = bf0[12]
	bf1[13] = fwdHalfBtf(cospi[48], bf0[13], -cospi[16], bf0[10], cosBit)
	bf1[14] = fwdHalfBtf(cospi[16], bf0[14], cospi[48], bf0[9], cosBit)
	bf1[15] = bf0[15]
	// stage 5
	bf0, bf1 = &step, output
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = fwdHalfBtf(cospi[56], bf0[4], cospi[8], bf0[7], cosBit)
	bf1[5] = fwdHalfBtf(cospi[24], bf0[5], cospi[40], bf0[6], cosBit)
	bf1[6] = fwdHalfBtf(cospi[24], bf0[6], -cospi[40], bf0[5], cosBit)
	bf1[7] = fwdHalfBtf(cospi[56], bf0[7], -cospi[8], bf0[4], cosBit)
	bf1[8] = bf0[8] + bf0[9]
	bf1[9] = -bf0[9] + bf0[8]
	bf1[10] = -bf0[10] + bf0[11]
	bf1[11] = bf0[11] + bf0[10]
	bf1[12] = bf0[12] + bf0[13]
	bf1[13] = -bf0[13] + bf0[12]
	bf1[14] = -bf0[14] + bf0[15]
	bf1[15] = bf0[15] + bf0[14]
	// stage 6
	bf0, bf1 = output, &step
	for i := range 8 {
		bf1[i] = bf0[i]
	}
	bf1[8] = fwdHalfBtf(cospi[60], bf0[8], cospi[4], bf0[15], cosBit)
	bf1[9] = fwdHalfBtf(cospi[28], bf0[9], cospi[36], bf0[14], cosBit)
	bf1[10] = fwdHalfBtf(cospi[44], bf0[10], cospi[20], bf0[13], cosBit)
	bf1[11] = fwdHalfBtf(cospi[12], bf0[11], cospi[52], bf0[12], cosBit)
	bf1[12] = fwdHalfBtf(cospi[12], bf0[12], -cospi[52], bf0[11], cosBit)
	bf1[13] = fwdHalfBtf(cospi[44], bf0[13], -cospi[20], bf0[10], cosBit)
	bf1[14] = fwdHalfBtf(cospi[28], bf0[14], -cospi[36], bf0[9], cosBit)
	bf1[15] = fwdHalfBtf(cospi[60], bf0[15], -cospi[4], bf0[8], cosBit)
	// stage 7 (bit-reverse permutation)
	bf0, bf1 = &step, output
	bf1[0] = bf0[0]
	bf1[1] = bf0[8]
	bf1[2] = bf0[4]
	bf1[3] = bf0[12]
	bf1[4] = bf0[2]
	bf1[5] = bf0[10]
	bf1[6] = bf0[6]
	bf1[7] = bf0[14]
	bf1[8] = bf0[1]
	bf1[9] = bf0[9]
	bf1[10] = bf0[5]
	bf1[11] = bf0[13]
	bf1[12] = bf0[3]
	bf1[13] = bf0[11]
	bf1[14] = bf0[7]
	bf1[15] = bf0[15]
}

// ForwardDCT4x4 computes the AV1 forward 4x4 DCT_DCT of a residual block,
// writing coefficients in the decoder's coefficient layout
// (coeff[col*coeffStride + row], the transpose fwd_txfm2d_c emits via
// output[c*rows + r]). residual is read as residual[row*residualStride + col].
// Mirrors fwd_txfm2d_c with fwd_shift_4x4 = {2, 0, 0} and cos_bit 13.
func ForwardDCT4x4(coeff []int32, coeffStride int, residual []int16, residualStride int) error {
	if coeffStride < 4 || residualStride < 4 ||
		!blockFits(len(residual), residualStride, 4, 4) ||
		!coeffBlockFits(len(coeff), coeffStride, 4, 4) {
		return ErrInvalidTransform
	}
	var buf [16]int32 // column-pass output, row-major buf[r*4+c]
	var in, out [4]int32
	for c := range 4 {
		for r := range 4 {
			in[r] = int32(residual[r*residualStride+c]) << 2 // shift[0] = 2
		}
		fwdDCT4(&in, &out)
		for r := range 4 { // shift[1] = 0
			buf[r*4+c] = out[r]
		}
	}
	for r := range 4 {
		for c := range 4 {
			in[c] = buf[r*4+c]
		}
		fwdDCT4(&in, &out)
		for c := range 4 { // shift[2] = 0; output[c*rows + r]
			coeff[c*coeffStride+r] = out[c]
		}
	}
	return nil
}

// ForwardDCT8x8 computes the AV1 forward 8x8 DCT_DCT of a residual block in
// the decoder's coefficient layout. Mirrors fwd_txfm2d_c with fwd_shift_8x8 =
// {2, -1, 0} and cos_bit 13.
func ForwardDCT8x8(coeff []int32, coeffStride int, residual []int16, residualStride int) error {
	if coeffStride < 8 || residualStride < 8 ||
		!blockFits(len(residual), residualStride, 8, 8) ||
		!coeffBlockFits(len(coeff), coeffStride, 8, 8) {
		return ErrInvalidTransform
	}
	var buf [64]int32 // column-pass output, row-major buf[r*8+c]
	var in, out [8]int32
	for c := range 8 {
		for r := range 8 {
			in[r] = int32(residual[r*residualStride+c]) << 2 // shift[0] = 2
		}
		fwdDCT8(&in, &out)
		fwdRoundShift1(out[:]) // shift[1] = -1
		for r := range 8 {
			buf[r*8+c] = out[r]
		}
	}
	for r := range 8 {
		for c := range 8 {
			in[c] = buf[r*8+c]
		}
		fwdDCT8(&in, &out)
		for c := range 8 { // shift[2] = 0; output[c*rows + r]
			coeff[c*coeffStride+r] = out[c]
		}
	}
	return nil
}

// ForwardDCT16x16 computes the AV1 forward 16x16 DCT_DCT of a residual block
// in the decoder's coefficient layout. Mirrors fwd_txfm2d_c with
// fwd_shift_16x16 = {2, -2, 0}, column pass cos_bit 13, row pass cos_bit 12.
func ForwardDCT16x16(coeff []int32, coeffStride int, residual []int16, residualStride int) error {
	if coeffStride < 16 || residualStride < 16 ||
		!blockFits(len(residual), residualStride, 16, 16) ||
		!coeffBlockFits(len(coeff), coeffStride, 16, 16) {
		return ErrInvalidTransform
	}
	var buf [256]int32 // column-pass output, row-major buf[r*16+c]
	var in, out [16]int32
	for c := range 16 {
		for r := range 16 {
			in[r] = int32(residual[r*residualStride+c]) << 2 // shift[0] = 2
		}
		fwdDCT16(&in, &out, &fwdCospi13, 13)
		fwdRoundShift2(out[:]) // shift[1] = -2
		for r := range 16 {
			buf[r*16+c] = out[r]
		}
	}
	for r := range 16 {
		for c := range 16 {
			in[c] = buf[r*16+c]
		}
		fwdDCT16(&in, &out, &fwdCospi12, 12)
		for c := range 16 { // shift[2] = 0; output[c*rows + r]
			coeff[c*coeffStride+r] = out[c]
		}
	}
	return nil
}

// fwdDCT32 is av1_fdct32; both 32x32 passes run at cos_bit 12 so the cospi
// table and shift are fixed (fwdCospi12, fwdHalfBtf at 12).
func fwdDCT32(input, output *[32]int32) {
	var step [32]int32
	bf0, bf1 := (*[32]int32)(nil), (*[32]int32)(nil)
	_ = bf0
	// stage 1
	bf1 = output
	bf1[0] = input[0] + input[31]
	bf1[1] = input[1] + input[30]
	bf1[2] = input[2] + input[29]
	bf1[3] = input[3] + input[28]
	bf1[4] = input[4] + input[27]
	bf1[5] = input[5] + input[26]
	bf1[6] = input[6] + input[25]
	bf1[7] = input[7] + input[24]
	bf1[8] = input[8] + input[23]
	bf1[9] = input[9] + input[22]
	bf1[10] = input[10] + input[21]
	bf1[11] = input[11] + input[20]
	bf1[12] = input[12] + input[19]
	bf1[13] = input[13] + input[18]
	bf1[14] = input[14] + input[17]
	bf1[15] = input[15] + input[16]
	bf1[16] = -input[16] + input[15]
	bf1[17] = -input[17] + input[14]
	bf1[18] = -input[18] + input[13]
	bf1[19] = -input[19] + input[12]
	bf1[20] = -input[20] + input[11]
	bf1[21] = -input[21] + input[10]
	bf1[22] = -input[22] + input[9]
	bf1[23] = -input[23] + input[8]
	bf1[24] = -input[24] + input[7]
	bf1[25] = -input[25] + input[6]
	bf1[26] = -input[26] + input[5]
	bf1[27] = -input[27] + input[4]
	bf1[28] = -input[28] + input[3]
	bf1[29] = -input[29] + input[2]
	bf1[30] = -input[30] + input[1]
	bf1[31] = -input[31] + input[0]
	// stage 2
	bf0 = output
	bf1 = &step
	bf1[0] = bf0[0] + bf0[15]
	bf1[1] = bf0[1] + bf0[14]
	bf1[2] = bf0[2] + bf0[13]
	bf1[3] = bf0[3] + bf0[12]
	bf1[4] = bf0[4] + bf0[11]
	bf1[5] = bf0[5] + bf0[10]
	bf1[6] = bf0[6] + bf0[9]
	bf1[7] = bf0[7] + bf0[8]
	bf1[8] = -bf0[8] + bf0[7]
	bf1[9] = -bf0[9] + bf0[6]
	bf1[10] = -bf0[10] + bf0[5]
	bf1[11] = -bf0[11] + bf0[4]
	bf1[12] = -bf0[12] + bf0[3]
	bf1[13] = -bf0[13] + bf0[2]
	bf1[14] = -bf0[14] + bf0[1]
	bf1[15] = -bf0[15] + bf0[0]
	bf1[16] = bf0[16]
	bf1[17] = bf0[17]
	bf1[18] = bf0[18]
	bf1[19] = bf0[19]
	bf1[20] = fwdHalfBtf(-fwdCospi12[32], bf0[20], fwdCospi12[32], bf0[27], 12)
	bf1[21] = fwdHalfBtf(-fwdCospi12[32], bf0[21], fwdCospi12[32], bf0[26], 12)
	bf1[22] = fwdHalfBtf(-fwdCospi12[32], bf0[22], fwdCospi12[32], bf0[25], 12)
	bf1[23] = fwdHalfBtf(-fwdCospi12[32], bf0[23], fwdCospi12[32], bf0[24], 12)
	bf1[24] = fwdHalfBtf(fwdCospi12[32], bf0[24], fwdCospi12[32], bf0[23], 12)
	bf1[25] = fwdHalfBtf(fwdCospi12[32], bf0[25], fwdCospi12[32], bf0[22], 12)
	bf1[26] = fwdHalfBtf(fwdCospi12[32], bf0[26], fwdCospi12[32], bf0[21], 12)
	bf1[27] = fwdHalfBtf(fwdCospi12[32], bf0[27], fwdCospi12[32], bf0[20], 12)
	bf1[28] = bf0[28]
	bf1[29] = bf0[29]
	bf1[30] = bf0[30]
	bf1[31] = bf0[31]
	// stage 3
	bf0 = &step
	bf1 = output
	bf1[0] = bf0[0] + bf0[7]
	bf1[1] = bf0[1] + bf0[6]
	bf1[2] = bf0[2] + bf0[5]
	bf1[3] = bf0[3] + bf0[4]
	bf1[4] = -bf0[4] + bf0[3]
	bf1[5] = -bf0[5] + bf0[2]
	bf1[6] = -bf0[6] + bf0[1]
	bf1[7] = -bf0[7] + bf0[0]
	bf1[8] = bf0[8]
	bf1[9] = bf0[9]
	bf1[10] = fwdHalfBtf(-fwdCospi12[32], bf0[10], fwdCospi12[32], bf0[13], 12)
	bf1[11] = fwdHalfBtf(-fwdCospi12[32], bf0[11], fwdCospi12[32], bf0[12], 12)
	bf1[12] = fwdHalfBtf(fwdCospi12[32], bf0[12], fwdCospi12[32], bf0[11], 12)
	bf1[13] = fwdHalfBtf(fwdCospi12[32], bf0[13], fwdCospi12[32], bf0[10], 12)
	bf1[14] = bf0[14]
	bf1[15] = bf0[15]
	bf1[16] = bf0[16] + bf0[23]
	bf1[17] = bf0[17] + bf0[22]
	bf1[18] = bf0[18] + bf0[21]
	bf1[19] = bf0[19] + bf0[20]
	bf1[20] = -bf0[20] + bf0[19]
	bf1[21] = -bf0[21] + bf0[18]
	bf1[22] = -bf0[22] + bf0[17]
	bf1[23] = -bf0[23] + bf0[16]
	bf1[24] = -bf0[24] + bf0[31]
	bf1[25] = -bf0[25] + bf0[30]
	bf1[26] = -bf0[26] + bf0[29]
	bf1[27] = -bf0[27] + bf0[28]
	bf1[28] = bf0[28] + bf0[27]
	bf1[29] = bf0[29] + bf0[26]
	bf1[30] = bf0[30] + bf0[25]
	bf1[31] = bf0[31] + bf0[24]
	// stage 4
	bf0 = output
	bf1 = &step
	bf1[0] = bf0[0] + bf0[3]
	bf1[1] = bf0[1] + bf0[2]
	bf1[2] = -bf0[2] + bf0[1]
	bf1[3] = -bf0[3] + bf0[0]
	bf1[4] = bf0[4]
	bf1[5] = fwdHalfBtf(-fwdCospi12[32], bf0[5], fwdCospi12[32], bf0[6], 12)
	bf1[6] = fwdHalfBtf(fwdCospi12[32], bf0[6], fwdCospi12[32], bf0[5], 12)
	bf1[7] = bf0[7]
	bf1[8] = bf0[8] + bf0[11]
	bf1[9] = bf0[9] + bf0[10]
	bf1[10] = -bf0[10] + bf0[9]
	bf1[11] = -bf0[11] + bf0[8]
	bf1[12] = -bf0[12] + bf0[15]
	bf1[13] = -bf0[13] + bf0[14]
	bf1[14] = bf0[14] + bf0[13]
	bf1[15] = bf0[15] + bf0[12]
	bf1[16] = bf0[16]
	bf1[17] = bf0[17]
	bf1[18] = fwdHalfBtf(-fwdCospi12[16], bf0[18], fwdCospi12[48], bf0[29], 12)
	bf1[19] = fwdHalfBtf(-fwdCospi12[16], bf0[19], fwdCospi12[48], bf0[28], 12)
	bf1[20] = fwdHalfBtf(-fwdCospi12[48], bf0[20], -fwdCospi12[16], bf0[27], 12)
	bf1[21] = fwdHalfBtf(-fwdCospi12[48], bf0[21], -fwdCospi12[16], bf0[26], 12)
	bf1[22] = bf0[22]
	bf1[23] = bf0[23]
	bf1[24] = bf0[24]
	bf1[25] = bf0[25]
	bf1[26] = fwdHalfBtf(fwdCospi12[48], bf0[26], -fwdCospi12[16], bf0[21], 12)
	bf1[27] = fwdHalfBtf(fwdCospi12[48], bf0[27], -fwdCospi12[16], bf0[20], 12)
	bf1[28] = fwdHalfBtf(fwdCospi12[16], bf0[28], fwdCospi12[48], bf0[19], 12)
	bf1[29] = fwdHalfBtf(fwdCospi12[16], bf0[29], fwdCospi12[48], bf0[18], 12)
	bf1[30] = bf0[30]
	bf1[31] = bf0[31]
	// stage 5
	bf0 = &step
	bf1 = output
	bf1[0] = fwdHalfBtf(fwdCospi12[32], bf0[0], fwdCospi12[32], bf0[1], 12)
	bf1[1] = fwdHalfBtf(-fwdCospi12[32], bf0[1], fwdCospi12[32], bf0[0], 12)
	bf1[2] = fwdHalfBtf(fwdCospi12[48], bf0[2], fwdCospi12[16], bf0[3], 12)
	bf1[3] = fwdHalfBtf(fwdCospi12[48], bf0[3], -fwdCospi12[16], bf0[2], 12)
	bf1[4] = bf0[4] + bf0[5]
	bf1[5] = -bf0[5] + bf0[4]
	bf1[6] = -bf0[6] + bf0[7]
	bf1[7] = bf0[7] + bf0[6]
	bf1[8] = bf0[8]
	bf1[9] = fwdHalfBtf(-fwdCospi12[16], bf0[9], fwdCospi12[48], bf0[14], 12)
	bf1[10] = fwdHalfBtf(-fwdCospi12[48], bf0[10], -fwdCospi12[16], bf0[13], 12)
	bf1[11] = bf0[11]
	bf1[12] = bf0[12]
	bf1[13] = fwdHalfBtf(fwdCospi12[48], bf0[13], -fwdCospi12[16], bf0[10], 12)
	bf1[14] = fwdHalfBtf(fwdCospi12[16], bf0[14], fwdCospi12[48], bf0[9], 12)
	bf1[15] = bf0[15]
	bf1[16] = bf0[16] + bf0[19]
	bf1[17] = bf0[17] + bf0[18]
	bf1[18] = -bf0[18] + bf0[17]
	bf1[19] = -bf0[19] + bf0[16]
	bf1[20] = -bf0[20] + bf0[23]
	bf1[21] = -bf0[21] + bf0[22]
	bf1[22] = bf0[22] + bf0[21]
	bf1[23] = bf0[23] + bf0[20]
	bf1[24] = bf0[24] + bf0[27]
	bf1[25] = bf0[25] + bf0[26]
	bf1[26] = -bf0[26] + bf0[25]
	bf1[27] = -bf0[27] + bf0[24]
	bf1[28] = -bf0[28] + bf0[31]
	bf1[29] = -bf0[29] + bf0[30]
	bf1[30] = bf0[30] + bf0[29]
	bf1[31] = bf0[31] + bf0[28]
	// stage 6
	bf0 = output
	bf1 = &step
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = fwdHalfBtf(fwdCospi12[56], bf0[4], fwdCospi12[8], bf0[7], 12)
	bf1[5] = fwdHalfBtf(fwdCospi12[24], bf0[5], fwdCospi12[40], bf0[6], 12)
	bf1[6] = fwdHalfBtf(fwdCospi12[24], bf0[6], -fwdCospi12[40], bf0[5], 12)
	bf1[7] = fwdHalfBtf(fwdCospi12[56], bf0[7], -fwdCospi12[8], bf0[4], 12)
	bf1[8] = bf0[8] + bf0[9]
	bf1[9] = -bf0[9] + bf0[8]
	bf1[10] = -bf0[10] + bf0[11]
	bf1[11] = bf0[11] + bf0[10]
	bf1[12] = bf0[12] + bf0[13]
	bf1[13] = -bf0[13] + bf0[12]
	bf1[14] = -bf0[14] + bf0[15]
	bf1[15] = bf0[15] + bf0[14]
	bf1[16] = bf0[16]
	bf1[17] = fwdHalfBtf(-fwdCospi12[8], bf0[17], fwdCospi12[56], bf0[30], 12)
	bf1[18] = fwdHalfBtf(-fwdCospi12[56], bf0[18], -fwdCospi12[8], bf0[29], 12)
	bf1[19] = bf0[19]
	bf1[20] = bf0[20]
	bf1[21] = fwdHalfBtf(-fwdCospi12[40], bf0[21], fwdCospi12[24], bf0[26], 12)
	bf1[22] = fwdHalfBtf(-fwdCospi12[24], bf0[22], -fwdCospi12[40], bf0[25], 12)
	bf1[23] = bf0[23]
	bf1[24] = bf0[24]
	bf1[25] = fwdHalfBtf(fwdCospi12[24], bf0[25], -fwdCospi12[40], bf0[22], 12)
	bf1[26] = fwdHalfBtf(fwdCospi12[40], bf0[26], fwdCospi12[24], bf0[21], 12)
	bf1[27] = bf0[27]
	bf1[28] = bf0[28]
	bf1[29] = fwdHalfBtf(fwdCospi12[56], bf0[29], -fwdCospi12[8], bf0[18], 12)
	bf1[30] = fwdHalfBtf(fwdCospi12[8], bf0[30], fwdCospi12[56], bf0[17], 12)
	bf1[31] = bf0[31]
	// stage 7
	bf0 = &step
	bf1 = output
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = bf0[4]
	bf1[5] = bf0[5]
	bf1[6] = bf0[6]
	bf1[7] = bf0[7]
	bf1[8] = fwdHalfBtf(fwdCospi12[60], bf0[8], fwdCospi12[4], bf0[15], 12)
	bf1[9] = fwdHalfBtf(fwdCospi12[28], bf0[9], fwdCospi12[36], bf0[14], 12)
	bf1[10] = fwdHalfBtf(fwdCospi12[44], bf0[10], fwdCospi12[20], bf0[13], 12)
	bf1[11] = fwdHalfBtf(fwdCospi12[12], bf0[11], fwdCospi12[52], bf0[12], 12)
	bf1[12] = fwdHalfBtf(fwdCospi12[12], bf0[12], -fwdCospi12[52], bf0[11], 12)
	bf1[13] = fwdHalfBtf(fwdCospi12[44], bf0[13], -fwdCospi12[20], bf0[10], 12)
	bf1[14] = fwdHalfBtf(fwdCospi12[28], bf0[14], -fwdCospi12[36], bf0[9], 12)
	bf1[15] = fwdHalfBtf(fwdCospi12[60], bf0[15], -fwdCospi12[4], bf0[8], 12)
	bf1[16] = bf0[16] + bf0[17]
	bf1[17] = -bf0[17] + bf0[16]
	bf1[18] = -bf0[18] + bf0[19]
	bf1[19] = bf0[19] + bf0[18]
	bf1[20] = bf0[20] + bf0[21]
	bf1[21] = -bf0[21] + bf0[20]
	bf1[22] = -bf0[22] + bf0[23]
	bf1[23] = bf0[23] + bf0[22]
	bf1[24] = bf0[24] + bf0[25]
	bf1[25] = -bf0[25] + bf0[24]
	bf1[26] = -bf0[26] + bf0[27]
	bf1[27] = bf0[27] + bf0[26]
	bf1[28] = bf0[28] + bf0[29]
	bf1[29] = -bf0[29] + bf0[28]
	bf1[30] = -bf0[30] + bf0[31]
	bf1[31] = bf0[31] + bf0[30]
	// stage 8
	bf0 = output
	bf1 = &step
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = bf0[4]
	bf1[5] = bf0[5]
	bf1[6] = bf0[6]
	bf1[7] = bf0[7]
	bf1[8] = bf0[8]
	bf1[9] = bf0[9]
	bf1[10] = bf0[10]
	bf1[11] = bf0[11]
	bf1[12] = bf0[12]
	bf1[13] = bf0[13]
	bf1[14] = bf0[14]
	bf1[15] = bf0[15]
	bf1[16] = fwdHalfBtf(fwdCospi12[62], bf0[16], fwdCospi12[2], bf0[31], 12)
	bf1[17] = fwdHalfBtf(fwdCospi12[30], bf0[17], fwdCospi12[34], bf0[30], 12)
	bf1[18] = fwdHalfBtf(fwdCospi12[46], bf0[18], fwdCospi12[18], bf0[29], 12)
	bf1[19] = fwdHalfBtf(fwdCospi12[14], bf0[19], fwdCospi12[50], bf0[28], 12)
	bf1[20] = fwdHalfBtf(fwdCospi12[54], bf0[20], fwdCospi12[10], bf0[27], 12)
	bf1[21] = fwdHalfBtf(fwdCospi12[22], bf0[21], fwdCospi12[42], bf0[26], 12)
	bf1[22] = fwdHalfBtf(fwdCospi12[38], bf0[22], fwdCospi12[26], bf0[25], 12)
	bf1[23] = fwdHalfBtf(fwdCospi12[6], bf0[23], fwdCospi12[58], bf0[24], 12)
	bf1[24] = fwdHalfBtf(fwdCospi12[6], bf0[24], -fwdCospi12[58], bf0[23], 12)
	bf1[25] = fwdHalfBtf(fwdCospi12[38], bf0[25], -fwdCospi12[26], bf0[22], 12)
	bf1[26] = fwdHalfBtf(fwdCospi12[22], bf0[26], -fwdCospi12[42], bf0[21], 12)
	bf1[27] = fwdHalfBtf(fwdCospi12[54], bf0[27], -fwdCospi12[10], bf0[20], 12)
	bf1[28] = fwdHalfBtf(fwdCospi12[14], bf0[28], -fwdCospi12[50], bf0[19], 12)
	bf1[29] = fwdHalfBtf(fwdCospi12[46], bf0[29], -fwdCospi12[18], bf0[18], 12)
	bf1[30] = fwdHalfBtf(fwdCospi12[30], bf0[30], -fwdCospi12[34], bf0[17], 12)
	bf1[31] = fwdHalfBtf(fwdCospi12[62], bf0[31], -fwdCospi12[2], bf0[16], 12)
	// stage 9
	bf0 = &step
	bf1 = output
	bf1[0] = bf0[0]
	bf1[1] = bf0[16]
	bf1[2] = bf0[8]
	bf1[3] = bf0[24]
	bf1[4] = bf0[4]
	bf1[5] = bf0[20]
	bf1[6] = bf0[12]
	bf1[7] = bf0[28]
	bf1[8] = bf0[2]
	bf1[9] = bf0[18]
	bf1[10] = bf0[10]
	bf1[11] = bf0[26]
	bf1[12] = bf0[6]
	bf1[13] = bf0[22]
	bf1[14] = bf0[14]
	bf1[15] = bf0[30]
	bf1[16] = bf0[1]
	bf1[17] = bf0[17]
	bf1[18] = bf0[9]
	bf1[19] = bf0[25]
	bf1[20] = bf0[5]
	bf1[21] = bf0[21]
	bf1[22] = bf0[13]
	bf1[23] = bf0[29]
	bf1[24] = bf0[3]
	bf1[25] = bf0[19]
	bf1[26] = bf0[11]
	bf1[27] = bf0[27]
	bf1[28] = bf0[7]
	bf1[29] = bf0[23]
	bf1[30] = bf0[15]
	bf1[31] = bf0[31]
}

// fwdRoundShift4 is av1_round_shift_array with bit == 4 (the 32x32 shift[1]
// of -4): round_shift(v, 4) = (v + 8) >> 4.
func fwdRoundShift4(arr []int32) {
	for i, v := range arr {
		arr[i] = (v + 8) >> 4
	}
}

// ForwardDCT32x32 computes the AV1 forward 32x32 DCT_DCT of a residual block
// in the decoder's coefficient layout. Mirrors fwd_txfm2d_c with
// fwd_shift_32x32 = {2, -4, 0} and cos_bit 12 on both passes.
func ForwardDCT32x32(coeff []int32, coeffStride int, residual []int16, residualStride int) error {
	if coeffStride < 32 || residualStride < 32 ||
		!blockFits(len(residual), residualStride, 32, 32) ||
		!coeffBlockFits(len(coeff), coeffStride, 32, 32) {
		return ErrInvalidTransform
	}
	var buf [1024]int32 // column-pass output, row-major buf[r*32+c]
	var in, out [32]int32
	for c := range 32 {
		for r := range 32 {
			in[r] = int32(residual[r*residualStride+c]) << 2 // shift[0] = 2
		}
		fwdDCT32(&in, &out)
		fwdRoundShift4(out[:]) // shift[1] = -4
		for r := range 32 {
			buf[r*32+c] = out[r]
		}
	}
	for r := range 32 {
		for c := range 32 {
			in[c] = buf[r*32+c]
		}
		fwdDCT32(&in, &out)
		for c := range 32 { // shift[2] = 0; output[c*rows + r]
			coeff[c*coeffStride+r] = out[c]
		}
	}
	return nil
}
