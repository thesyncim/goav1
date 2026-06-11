// Ported from libaom:
//   av1/encoder/av1_fwd_txfm1d.c (av1_fadst4, av1_fadst8,
//   av1_fidentity4_c, av1_fidentity8_c)
//   av1/encoder/av1_fwd_txfm2d.c (fwd_txfm2d_c for square 4x4/8x8)
//   av1/common/av1_txfm.c (av1_sinpi_arr_data, cos_bit 13 row)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package transform

// fwdSinpi13 is av1_sinpi_arr_data[13-cos_bit_min]. It is used by the 4-point
// forward ADST kernel.
var fwdSinpi13 = [5]int32{0, 2642, 4964, 6689, 7606}

// ForwardBlock computes the AV1 forward transform for a residual block and
// writes coefficients in decoder coefficient layout (coeff[col*stride + row]).
// This generic path currently covers the square 4x4/8x8 ADST/DCT and IDTX
// families needed before the encoder can make tx_type decisions. Larger DCT_DCT
// sizes continue to dispatch through the existing specialized DCT entry points.
func ForwardBlock(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32, size Size, typ Type) error {
	if typ == TypeDCTDCT {
		return forwardDCTBySize(coeff, coeffStride, residual, residualStride, size)
	}
	if !forwardHybridSupported(size, typ) {
		return ErrInvalidTransform
	}
	n := int(size.Width)
	if coeffStride < n || residualStride < n ||
		!blockFits(len(residual), residualStride, n, n) ||
		!coeffBlockFits(len(coeff), coeffStride, n, n) ||
		len(scratch) < n*n {
		return ErrInvalidTransform
	}
	vertical, horizontal, _ := typ.tx1DTypes()
	var in, out [8]int32
	for c := range n {
		for r := range n {
			in[r] = int32(residual[r*residualStride+c]) << 2
		}
		fwd1D(in[:n], out[:n], vertical)
		if n == 8 {
			fwdRoundShift1(out[:n])
		}
		for r := range n {
			scratch[r*n+c] = out[r]
		}
	}
	for r := range n {
		for c := range n {
			in[c] = scratch[r*n+c]
		}
		fwd1D(in[:n], out[:n], horizontal)
		for c := range n {
			coeff[c*coeffStride+r] = out[c]
		}
	}
	return nil
}

func forwardHybridSupported(size Size, typ Type) bool {
	if size.Width != size.Height || (size.Width != 4 && size.Width != 8) {
		return false
	}
	switch typ {
	case TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeIDTX:
		return typ.Supported(size)
	default:
		return false
	}
}

func forwardDCTBySize(coeff []int32, coeffStride int, residual []int16, residualStride int, size Size) error {
	switch size {
	case Size{Width: 4, Height: 4}:
		return ForwardDCT4x4(coeff, coeffStride, residual, residualStride)
	case Size{Width: 8, Height: 8}:
		return ForwardDCT8x8(coeff, coeffStride, residual, residualStride)
	case Size{Width: 16, Height: 16}:
		return ForwardDCT16x16(coeff, coeffStride, residual, residualStride)
	case Size{Width: 32, Height: 32}:
		return ForwardDCT32x32(coeff, coeffStride, residual, residualStride)
	case Size{Width: 16, Height: 8}:
		return ForwardDCT16x8(coeff, coeffStride, residual, residualStride)
	case Size{Width: 8, Height: 16}:
		return ForwardDCT8x16(coeff, coeffStride, residual, residualStride)
	case Size{Width: 8, Height: 4}:
		return ForwardDCT8x4(coeff, coeffStride, residual, residualStride)
	case Size{Width: 4, Height: 8}:
		return ForwardDCT4x8(coeff, coeffStride, residual, residualStride)
	case Size{Width: 32, Height: 16}:
		return ForwardDCT32x16(coeff, coeffStride, residual, residualStride)
	case Size{Width: 16, Height: 32}:
		return ForwardDCT16x32(coeff, coeffStride, residual, residualStride)
	default:
		return ErrInvalidTransform
	}
}

func fwd1D(input []int32, output []int32, typ tx1DType) {
	switch typ {
	case tx1DDCT:
		switch len(input) {
		case 4:
			var in, out [4]int32
			copy(in[:], input)
			fwdDCT4(&in, &out)
			copy(output, out[:])
		case 8:
			var in, out [8]int32
			copy(in[:], input)
			fwdDCT8(&in, &out)
			copy(output, out[:])
		}
	case tx1DADST:
		switch len(input) {
		case 4:
			var in, out [4]int32
			copy(in[:], input)
			fwdADST4(&in, &out)
			copy(output, out[:])
		case 8:
			var in, out [8]int32
			copy(in[:], input)
			fwdADST8(&in, &out)
			copy(output, out[:])
		}
	case tx1DIdentity:
		fwdIdentity1D(input, output)
	}
}

func fwdADST4(input, output *[4]int32) {
	x0, x1, x2, x3 := input[0], input[1], input[2], input[3]
	if x0|x1|x2|x3 == 0 {
		clear(output[:])
		return
	}
	s0 := fwdSinpi13[1] * x0
	s1 := fwdSinpi13[4] * x0
	s2 := fwdSinpi13[2] * x1
	s3 := fwdSinpi13[1] * x1
	s4 := fwdSinpi13[3] * x2
	s5 := fwdSinpi13[4] * x3
	s6 := fwdSinpi13[2] * x3
	s7 := x0 + x1

	s7 -= x3

	x0 = s0 + s2
	x1 = fwdSinpi13[3] * s7
	x2 = s1 - s3
	x3 = s4

	x0 += s5
	x2 += s6

	s0 = x0 + x3
	s1 = x1
	s2 = x2 - x3
	s3 = x2 - x0

	s3 += x3

	output[0] = int32(roundShift(int64(s0), 13))
	output[1] = int32(roundShift(int64(s1), 13))
	output[2] = int32(roundShift(int64(s2), 13))
	output[3] = int32(roundShift(int64(s3), 13))
}

func fwdADST8(input, output *[8]int32) {
	var step [8]int32
	output[0] = input[0]
	output[1] = -input[7]
	output[2] = -input[3]
	output[3] = input[4]
	output[4] = -input[1]
	output[5] = input[6]
	output[6] = input[2]
	output[7] = -input[5]

	step[0] = output[0]
	step[1] = output[1]
	step[2] = fwdHalfBtf13(fwdCospi13[32], output[2], fwdCospi13[32], output[3])
	step[3] = fwdHalfBtf13(fwdCospi13[32], output[2], -fwdCospi13[32], output[3])
	step[4] = output[4]
	step[5] = output[5]
	step[6] = fwdHalfBtf13(fwdCospi13[32], output[6], fwdCospi13[32], output[7])
	step[7] = fwdHalfBtf13(fwdCospi13[32], output[6], -fwdCospi13[32], output[7])

	output[0] = step[0] + step[2]
	output[1] = step[1] + step[3]
	output[2] = step[0] - step[2]
	output[3] = step[1] - step[3]
	output[4] = step[4] + step[6]
	output[5] = step[5] + step[7]
	output[6] = step[4] - step[6]
	output[7] = step[5] - step[7]

	step[0] = output[0]
	step[1] = output[1]
	step[2] = output[2]
	step[3] = output[3]
	step[4] = fwdHalfBtf13(fwdCospi13[16], output[4], fwdCospi13[48], output[5])
	step[5] = fwdHalfBtf13(fwdCospi13[48], output[4], -fwdCospi13[16], output[5])
	step[6] = fwdHalfBtf13(-fwdCospi13[48], output[6], fwdCospi13[16], output[7])
	step[7] = fwdHalfBtf13(fwdCospi13[16], output[6], fwdCospi13[48], output[7])

	output[0] = step[0] + step[4]
	output[1] = step[1] + step[5]
	output[2] = step[2] + step[6]
	output[3] = step[3] + step[7]
	output[4] = step[0] - step[4]
	output[5] = step[1] - step[5]
	output[6] = step[2] - step[6]
	output[7] = step[3] - step[7]

	step[0] = fwdHalfBtf13(fwdCospi13[4], output[0], fwdCospi13[60], output[1])
	step[1] = fwdHalfBtf13(fwdCospi13[60], output[0], -fwdCospi13[4], output[1])
	step[2] = fwdHalfBtf13(fwdCospi13[20], output[2], fwdCospi13[44], output[3])
	step[3] = fwdHalfBtf13(fwdCospi13[44], output[2], -fwdCospi13[20], output[3])
	step[4] = fwdHalfBtf13(fwdCospi13[36], output[4], fwdCospi13[28], output[5])
	step[5] = fwdHalfBtf13(fwdCospi13[28], output[4], -fwdCospi13[36], output[5])
	step[6] = fwdHalfBtf13(fwdCospi13[52], output[6], fwdCospi13[12], output[7])
	step[7] = fwdHalfBtf13(fwdCospi13[12], output[6], -fwdCospi13[52], output[7])

	output[0] = step[1]
	output[1] = step[6]
	output[2] = step[3]
	output[3] = step[4]
	output[4] = step[5]
	output[5] = step[2]
	output[6] = step[7]
	output[7] = step[0]
}

func fwdIdentity1D(input []int32, output []int32) {
	switch len(input) {
	case 4:
		for i, v := range input {
			output[i] = int32(roundShift(int64(v)*5793, 12))
		}
	case 8:
		for i, v := range input {
			output[i] = v * 2
		}
	}
}
