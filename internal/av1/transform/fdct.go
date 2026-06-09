// Ported from libaom:
//   av1/encoder/av1_fwd_txfm1d.c (av1_fdct4, av1_fdct8)
//   av1/encoder/av1_fwd_txfm2d.c (fwd_txfm2d_c, fwd_shift_4x4, fwd_shift_8x8)
//   av1/common/av1_txfm.c (av1_cospi_arr_data, cos_bit 13 row)
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
		if v < 0 {
			arr[i] = -((-v + 1) >> 1)
		} else {
			arr[i] = (v + 1) >> 1
		}
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
