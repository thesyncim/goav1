package transform

const dct4Size = 4

// ScratchLen returns the number of int32 values needed by inverse transforms
// that use caller-provided temporary storage.
func ScratchLen(size Size) (int, error) {
	if !size.Valid() {
		return 0, ErrInvalidTransform
	}
	return size.Width * size.Height, nil
}

// InverseDCTBlock writes an AV1 DCT_DCT residual block to dst. The source
// coefficients are dequantized transform coefficients in row-major order.
// Currently the pure-Go DCT path supports 4x4 blocks.
func InverseDCTBlock(dst []int16, dstStride int, coeff []int32, coeffStride int, scratch []int32, size Size) error {
	if size != (Size{Width: dct4Size, Height: dct4Size}) ||
		dstStride < dct4Size ||
		coeffStride < dct4Size ||
		len(scratch) < dct4Size*dct4Size ||
		!blockFits(len(dst), dstStride, dct4Size, dct4Size) ||
		!blockFits(len(coeff), coeffStride, dct4Size, dct4Size) {
		return ErrInvalidTransform
	}

	for row := 0; row < dct4Size; row++ {
		srcLine := coeff[row*coeffStride : row*coeffStride+dct4Size]
		tmpLine := scratch[row*dct4Size : row*dct4Size+dct4Size]
		copy(tmpLine, srcLine)
		inverseDCT4(tmpLine, 1, minInt16, maxInt16)
	}

	for col := 0; col < dct4Size; col++ {
		inverseDCT4(scratch[col:], dct4Size, minInt16, maxInt16)
	}

	for row := 0; row < dct4Size; row++ {
		dstLine := dst[row*dstStride : row*dstStride+dct4Size]
		tmpLine := scratch[row*dct4Size : row*dct4Size+dct4Size]
		for col := 0; col < dct4Size; col++ {
			dstLine[col] = clipInt16(clipInt32(roundShift(int64(tmpLine[col]), 4)))
		}
	}
	return nil
}

func inverseDCT4(c []int32, stride int, min int32, max int32) {
	in0 := int64(c[0*stride])
	in1 := int64(c[1*stride])
	in2 := int64(c[2*stride])
	in3 := int64(c[3*stride])

	t0 := roundShift((in0+in2)*181, 8)
	t1 := roundShift((in0-in2)*181, 8)
	t2 := roundShift(in1*1567-in3*(3784-4096), 12) - in3
	t3 := roundShift(in1*(3784-4096)+in3*1567, 12) + in1

	c[0*stride] = clipRange(t0+t3, min, max)
	c[1*stride] = clipRange(t1+t2, min, max)
	c[2*stride] = clipRange(t1-t2, min, max)
	c[3*stride] = clipRange(t0-t3, min, max)
}

func clipRange(v int64, min int32, max int32) int32 {
	if v < int64(min) {
		return min
	}
	if v > int64(max) {
		return max
	}
	return int32(v)
}
