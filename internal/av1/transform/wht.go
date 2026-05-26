package transform

const unitQuantShift = 2

// InverseWHT4x4Block writes AV1's lossless 4x4 inverse Walsh-Hadamard
// residual. The source coefficients use AV1 coefficient order:
// coeff_idx = col * coeffStride + row. This is selected by reconstruction for
// lossless DCT_DCT 4x4 blocks; WHT is not a bitstream tx_type.
func InverseWHT4x4Block(dst []int16, dstStride int, coeff []int32, coeffStride int, eob int) error {
	if dstStride < 4 ||
		coeffStride < 4 ||
		eob < 0 ||
		!blockFits(len(dst), dstStride, 4, 4) ||
		!coeffBlockFits(len(coeff), coeffStride, 4, 4) {
		return ErrInvalidTransform
	}
	if eob > 1 {
		inverseWHT4x4EOB16(dst, dstStride, coeff, coeffStride)
		return nil
	}
	inverseWHT4x4EOB1(dst, dstStride, coeff, coeffStride)
	return nil
}

func inverseWHT4x4EOB16(dst []int16, dstStride int, coeff []int32, coeffStride int) {
	var output [16]int32
	for i := range 4 {
		a1 := coeff[0*coeffStride+i] >> unitQuantShift
		c1 := coeff[1*coeffStride+i] >> unitQuantShift
		d1 := coeff[2*coeffStride+i] >> unitQuantShift
		b1 := coeff[3*coeffStride+i] >> unitQuantShift

		a1 += c1
		d1 -= b1
		e1 := (a1 - d1) >> 1
		b1 = e1 - b1
		c1 = e1 - c1
		a1 -= b1
		d1 += c1

		output[0*4+i] = a1
		output[1*4+i] = b1
		output[2*4+i] = c1
		output[3*4+i] = d1
	}

	for i := range 4 {
		a1 := output[i*4+0]
		c1 := output[i*4+1]
		d1 := output[i*4+2]
		b1 := output[i*4+3]

		a1 += c1
		d1 -= b1
		e1 := (a1 - d1) >> 1
		b1 = e1 - b1
		c1 = e1 - c1
		a1 -= b1
		d1 += c1

		dst[0*dstStride+i] = clipInt16(a1)
		dst[1*dstStride+i] = clipInt16(b1)
		dst[2*dstStride+i] = clipInt16(c1)
		dst[3*dstStride+i] = clipInt16(d1)
	}
}

func inverseWHT4x4EOB1(dst []int16, dstStride int, coeff []int32, coeffStride int) {
	a1 := coeff[0*coeffStride] >> unitQuantShift
	e1 := a1 >> 1
	a1 -= e1
	tmp := [4]int32{a1, e1, e1, e1}
	for i := range 4 {
		e1 = tmp[i] >> 1
		a1 = tmp[i] - e1
		dst[0*dstStride+i] = clipInt16(a1)
		dst[1*dstStride+i] = clipInt16(e1)
		dst[2*dstStride+i] = clipInt16(e1)
		dst[3*dstStride+i] = clipInt16(e1)
	}
}
