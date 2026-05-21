package transform

const dct4Size = 4
const dct8Size = 8

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
// Currently the pure-Go DCT path supports 4x4 and 8x8 blocks.
func InverseDCTBlock(dst []int16, dstStride int, coeff []int32, coeffStride int, scratch []int32, size Size) error {
	shift, ok := size.shift()
	scratchLen := size.Width * size.Height
	if !ok ||
		!dctBlockSupported(size) ||
		dstStride < size.Width ||
		coeffStride < size.Width ||
		len(scratch) < scratchLen ||
		!blockFits(len(dst), dstStride, size.Width, size.Height) ||
		!blockFits(len(coeff), coeffStride, size.Width, size.Height) {
		return ErrInvalidTransform
	}

	for row := 0; row < size.Height; row++ {
		srcLine := coeff[row*coeffStride : row*coeffStride+size.Width]
		tmpLine := scratch[row*size.Width : row*size.Width+size.Width]
		copy(tmpLine, srcLine)
		inverseDCT1D(tmpLine, 1, size.Width, minInt16, maxInt16)
	}

	if shift > 0 {
		for i := 0; i < scratchLen; i++ {
			scratch[i] = clipRange(roundShift(int64(scratch[i]), shift), minInt16, maxInt16)
		}
	}

	for col := 0; col < size.Width; col++ {
		inverseDCT1D(scratch[col:], size.Width, size.Height, minInt16, maxInt16)
	}

	for row := 0; row < size.Height; row++ {
		dstLine := dst[row*dstStride : row*dstStride+size.Width]
		tmpLine := scratch[row*size.Width : row*size.Width+size.Width]
		for col := 0; col < size.Width; col++ {
			dstLine[col] = clipInt16(clipInt32(roundShift(int64(tmpLine[col]), 4)))
		}
	}
	return nil
}

func dctBlockSupported(size Size) bool {
	return size == (Size{Width: dct4Size, Height: dct4Size}) ||
		size == (Size{Width: dct8Size, Height: dct8Size})
}

func inverseDCT1D(c []int32, stride int, length int, min int32, max int32) {
	switch length {
	case dct4Size:
		inverseDCT4(c, stride, min, max)
	case dct8Size:
		inverseDCT8(c, stride, min, max)
	}
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

func inverseDCT8(c []int32, stride int, min int32, max int32) {
	inverseDCT4(c, stride<<1, min, max)

	in1 := int64(c[1*stride])
	in3 := int64(c[3*stride])
	in5 := int64(c[5*stride])
	in7 := int64(c[7*stride])

	t4a := roundShift(in1*799-in7*(4017-4096), 12) - in7
	t5a := roundShift(in5*1703-in3*1138, 11)
	t6a := roundShift(in5*1138+in3*1703, 11)
	t7a := roundShift(in1*(4017-4096)+in7*799, 12) + in1

	t4 := int64(clipRange(t4a+t5a, min, max))
	t5a = int64(clipRange(t4a-t5a, min, max))
	t7 := int64(clipRange(t7a+t6a, min, max))
	t6a = int64(clipRange(t7a-t6a, min, max))

	t5 := roundShift((t6a-t5a)*181, 8)
	t6 := roundShift((t6a+t5a)*181, 8)

	t0 := int64(c[0*stride])
	t1 := int64(c[2*stride])
	t2 := int64(c[4*stride])
	t3 := int64(c[6*stride])

	c[0*stride] = clipRange(t0+t7, min, max)
	c[1*stride] = clipRange(t1+t6, min, max)
	c[2*stride] = clipRange(t2+t5, min, max)
	c[3*stride] = clipRange(t3+t4, min, max)
	c[4*stride] = clipRange(t3-t4, min, max)
	c[5*stride] = clipRange(t2-t5, min, max)
	c[6*stride] = clipRange(t1-t6, min, max)
	c[7*stride] = clipRange(t0-t7, min, max)
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
