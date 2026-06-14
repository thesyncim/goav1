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
	if size.Width == 8 && size.Height == 8 {
		if !forwardBlock8x8HybridSupported(typ) ||
			coeffStride < 8 || residualStride < 8 ||
			!blockFits(len(residual), residualStride, 8, 8) ||
			!coeffBlockFits(len(coeff), coeffStride, 8, 8) ||
			len(scratch) < 64 {
			return ErrInvalidTransform
		}
		forwardBlock8x8HybridTyped(coeff, coeffStride, residual, residualStride, scratch, typ)
		return nil
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

// ForwardBlock8x8HybridTrusted computes the 8x8 non-DCT_DCT forward transform
// for callers that already proved the block shape and scratch sizes. The
// coefficient and residual strides must be at least 8, and coeff/residual/scratch
// must contain the addressed 8x8 region.
func ForwardBlock8x8HybridTrusted(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32, typ Type) error {
	if !forwardBlock8x8HybridTyped(coeff, coeffStride, residual, residualStride, scratch, typ) {
		return ErrInvalidTransform
	}
	return nil
}

// ForwardBlock8x8ADSTDCTTrusted computes the trusted 8x8 ADST_DCT transform.
func ForwardBlock8x8ADSTDCTTrusted(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	forwardBlock8x8ADSTDCT(coeff, coeffStride, residual, residualStride, scratch)
}

// ForwardBlock8x8DCTADSTTrusted computes the trusted 8x8 DCT_ADST transform.
func ForwardBlock8x8DCTADSTTrusted(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	forwardBlock8x8DCTADST(coeff, coeffStride, residual, residualStride, scratch)
}

// ForwardBlock8x8ADSTADSTTrusted computes the trusted 8x8 ADST_ADST transform.
func ForwardBlock8x8ADSTADSTTrusted(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	forwardBlock8x8ADSTADST(coeff, coeffStride, residual, residualStride, scratch)
}

// ForwardBlock8x8IDTXTrusted computes the trusted 8x8 IDTX transform.
func ForwardBlock8x8IDTXTrusted(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	forwardBlock8x8IDTX(coeff, coeffStride, residual, residualStride, scratch)
}

func forwardBlock8x8HybridSupported(typ Type) bool {
	switch typ {
	case TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeIDTX:
		return true
	default:
		return false
	}
}

func forwardBlock8x8HybridTyped(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32, typ Type) bool {
	switch typ {
	case TypeADSTDCT:
		forwardBlock8x8ADSTDCT(coeff, coeffStride, residual, residualStride, scratch)
	case TypeDCTADST:
		forwardBlock8x8DCTADST(coeff, coeffStride, residual, residualStride, scratch)
	case TypeADSTADST:
		forwardBlock8x8ADSTADST(coeff, coeffStride, residual, residualStride, scratch)
	case TypeIDTX:
		forwardBlock8x8IDTX(coeff, coeffStride, residual, residualStride, scratch)
	default:
		return false
	}
	return true
}

func forwardBlock8x8ADSTDCT(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = coeff[7*coeffStride+7]
	_ = residual[7*residualStride+7]
	_ = scratch[63]
	var in, out [8]int32
	for c := range 8 {
		o0, o1, o2, o3, o4, o5, o6, o7 := fwdADST8Values(
			int32(residual[0*residualStride+c])<<2,
			int32(residual[1*residualStride+c])<<2,
			int32(residual[2*residualStride+c])<<2,
			int32(residual[3*residualStride+c])<<2,
			int32(residual[4*residualStride+c])<<2,
			int32(residual[5*residualStride+c])<<2,
			int32(residual[6*residualStride+c])<<2,
			int32(residual[7*residualStride+c])<<2,
		)
		scratch[0*8+c] = fwdRoundShift1Value(o0)
		scratch[1*8+c] = fwdRoundShift1Value(o1)
		scratch[2*8+c] = fwdRoundShift1Value(o2)
		scratch[3*8+c] = fwdRoundShift1Value(o3)
		scratch[4*8+c] = fwdRoundShift1Value(o4)
		scratch[5*8+c] = fwdRoundShift1Value(o5)
		scratch[6*8+c] = fwdRoundShift1Value(o6)
		scratch[7*8+c] = fwdRoundShift1Value(o7)
	}
	for r := range 8 {
		row := r * 8
		for c := range 8 {
			in[c] = scratch[row+c]
		}
		fwdDCT8(&in, &out)
		for c := range 8 {
			coeff[c*coeffStride+r] = out[c]
		}
	}
}

func forwardBlock8x8DCTADST(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = coeff[7*coeffStride+7]
	_ = residual[7*residualStride+7]
	_ = scratch[63]
	var in, out [8]int32
	for c := range 8 {
		for r := range 8 {
			in[r] = int32(residual[r*residualStride+c]) << 2
		}
		fwdDCT8(&in, &out)
		fwdRoundShift1x8(&out)
		for r := range 8 {
			scratch[r*8+c] = out[r]
		}
	}
	for r := range 8 {
		row := r * 8
		o0, o1, o2, o3, o4, o5, o6, o7 := fwdADST8Values(
			scratch[row+0],
			scratch[row+1],
			scratch[row+2],
			scratch[row+3],
			scratch[row+4],
			scratch[row+5],
			scratch[row+6],
			scratch[row+7],
		)
		coeff[0*coeffStride+r] = o0
		coeff[1*coeffStride+r] = o1
		coeff[2*coeffStride+r] = o2
		coeff[3*coeffStride+r] = o3
		coeff[4*coeffStride+r] = o4
		coeff[5*coeffStride+r] = o5
		coeff[6*coeffStride+r] = o6
		coeff[7*coeffStride+r] = o7
	}
}

func forwardBlock8x8ADSTADST(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = coeff[7*coeffStride+7]
	_ = residual[7*residualStride+7]
	_ = scratch[63]
	for c := range 8 {
		o0, o1, o2, o3, o4, o5, o6, o7 := fwdADST8Values(
			int32(residual[0*residualStride+c])<<2,
			int32(residual[1*residualStride+c])<<2,
			int32(residual[2*residualStride+c])<<2,
			int32(residual[3*residualStride+c])<<2,
			int32(residual[4*residualStride+c])<<2,
			int32(residual[5*residualStride+c])<<2,
			int32(residual[6*residualStride+c])<<2,
			int32(residual[7*residualStride+c])<<2,
		)
		scratch[0*8+c] = fwdRoundShift1Value(o0)
		scratch[1*8+c] = fwdRoundShift1Value(o1)
		scratch[2*8+c] = fwdRoundShift1Value(o2)
		scratch[3*8+c] = fwdRoundShift1Value(o3)
		scratch[4*8+c] = fwdRoundShift1Value(o4)
		scratch[5*8+c] = fwdRoundShift1Value(o5)
		scratch[6*8+c] = fwdRoundShift1Value(o6)
		scratch[7*8+c] = fwdRoundShift1Value(o7)
	}
	for r := range 8 {
		row := r * 8
		o0, o1, o2, o3, o4, o5, o6, o7 := fwdADST8Values(
			scratch[row+0],
			scratch[row+1],
			scratch[row+2],
			scratch[row+3],
			scratch[row+4],
			scratch[row+5],
			scratch[row+6],
			scratch[row+7],
		)
		coeff[0*coeffStride+r] = o0
		coeff[1*coeffStride+r] = o1
		coeff[2*coeffStride+r] = o2
		coeff[3*coeffStride+r] = o3
		coeff[4*coeffStride+r] = o4
		coeff[5*coeffStride+r] = o5
		coeff[6*coeffStride+r] = o6
		coeff[7*coeffStride+r] = o7
	}
}

func forwardBlock8x8IDTX(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = coeff[7*coeffStride+7]
	_ = residual[7*residualStride+7]
	_ = scratch[63]
	for r := range 8 {
		srcRow := r * residualStride
		for c := range 8 {
			coeff[c*coeffStride+r] = int32(residual[srcRow+c]) << 3
		}
	}
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

func fwdIdentity8(input, output *[8]int32) {
	for i := range 8 {
		output[i] = input[i] * 2
	}
}

func fwdRoundShift1x8(arr *[8]int32) {
	for i := range 8 {
		v := arr[i]
		arr[i] = fwdRoundShift1Value(v)
	}
}

func fwdRoundShift1Value(v int32) int32 {
	return (v + 1 + (v >> 31)) >> 1
}

// fwdADST8Values is the same cos-bit-13 ADST8 as fwdADST8, shaped for the
// 8x8 hybrid block paths so they can pass scalar row/column values without
// filling temporary input and output arrays around every 1-D pass.
func fwdADST8Values(x0, x1, x2, x3, x4, x5, x6, x7 int32) (int32, int32, int32, int32, int32, int32, int32, int32) {
	const (
		c4  int32 = 8153
		c12 int32 = 7839
		c16 int32 = 7568
		c20 int32 = 7225
		c28 int32 = 6333
		c32 int32 = 5793
		c36 int32 = 5197
		c44 int32 = 3862
		c48 int32 = 3135
		c52 int32 = 2378
		c60 int32 = 803
	)

	s0 := x0
	s1 := -x7
	s2 := fwdHalfBtf13(c32, -x3, c32, x4)
	s3 := fwdHalfBtf13(c32, -x3, -c32, x4)
	s4 := -x1
	s5 := x6
	s6 := fwdHalfBtf13(c32, x2, c32, -x5)
	s7 := fwdHalfBtf13(c32, x2, -c32, -x5)

	t0 := s0 + s2
	t1 := s1 + s3
	t2 := s0 - s2
	t3 := s1 - s3
	t4 := s4 + s6
	t5 := s5 + s7
	t6 := s4 - s6
	t7 := s5 - s7

	s4 = fwdHalfBtf13(c16, t4, c48, t5)
	s5 = fwdHalfBtf13(c48, t4, -c16, t5)
	s6 = fwdHalfBtf13(-c48, t6, c16, t7)
	s7 = fwdHalfBtf13(c16, t6, c48, t7)

	t4 = t0 - s4
	t5 = t1 - s5
	t6 = t2 - s6
	t7 = t3 - s7
	t0 += s4
	t1 += s5
	t2 += s6
	t3 += s7

	s0 = fwdHalfBtf13(c4, t0, c60, t1)
	s1 = fwdHalfBtf13(c60, t0, -c4, t1)
	s2 = fwdHalfBtf13(c20, t2, c44, t3)
	s3 = fwdHalfBtf13(c44, t2, -c20, t3)
	s4 = fwdHalfBtf13(c36, t4, c28, t5)
	s5 = fwdHalfBtf13(c28, t4, -c36, t5)
	s6 = fwdHalfBtf13(c52, t6, c12, t7)
	s7 = fwdHalfBtf13(c12, t6, -c52, t7)

	return s1, s6, s3, s4, s5, s2, s7, s0
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
	const (
		c4  int32 = 8153
		c12 int32 = 7839
		c16 int32 = 7568
		c20 int32 = 7225
		c28 int32 = 6333
		c32 int32 = 5793
		c36 int32 = 5197
		c44 int32 = 3862
		c48 int32 = 3135
		c52 int32 = 2378
		c60 int32 = 803
	)

	s0 := input[0]
	s1 := -input[7]
	s2 := fwdHalfBtf13(c32, -input[3], c32, input[4])
	s3 := fwdHalfBtf13(c32, -input[3], -c32, input[4])
	s4 := -input[1]
	s5 := input[6]
	s6 := fwdHalfBtf13(c32, input[2], c32, -input[5])
	s7 := fwdHalfBtf13(c32, input[2], -c32, -input[5])

	t0 := s0 + s2
	t1 := s1 + s3
	t2 := s0 - s2
	t3 := s1 - s3
	t4 := s4 + s6
	t5 := s5 + s7
	t6 := s4 - s6
	t7 := s5 - s7

	s4 = fwdHalfBtf13(c16, t4, c48, t5)
	s5 = fwdHalfBtf13(c48, t4, -c16, t5)
	s6 = fwdHalfBtf13(-c48, t6, c16, t7)
	s7 = fwdHalfBtf13(c16, t6, c48, t7)

	t4 = t0 - s4
	t5 = t1 - s5
	t6 = t2 - s6
	t7 = t3 - s7
	t0 += s4
	t1 += s5
	t2 += s6
	t3 += s7

	s0 = fwdHalfBtf13(c4, t0, c60, t1)
	s1 = fwdHalfBtf13(c60, t0, -c4, t1)
	s2 = fwdHalfBtf13(c20, t2, c44, t3)
	s3 = fwdHalfBtf13(c44, t2, -c20, t3)
	s4 = fwdHalfBtf13(c36, t4, c28, t5)
	s5 = fwdHalfBtf13(c28, t4, -c36, t5)
	s6 = fwdHalfBtf13(c52, t6, c12, t7)
	s7 = fwdHalfBtf13(c12, t6, -c52, t7)

	output[0] = s1
	output[1] = s6
	output[2] = s3
	output[3] = s4
	output[4] = s5
	output[5] = s2
	output[6] = s7
	output[7] = s0
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
