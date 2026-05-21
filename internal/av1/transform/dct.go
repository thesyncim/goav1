package transform

const dct4Size = 4
const dct8Size = 8
const dct16Size = 16
const dct32Size = 32

// ScratchLen returns the generic transform scratch length for size. For exact
// dispatch scratch sizing, use ScratchLenForType.
func ScratchLen(size Size) (int, error) {
	if !size.Valid() {
		return 0, ErrInvalidTransform
	}
	return size.Width * size.Height, nil
}

// InverseDCTBlock writes an AV1 DCT_DCT residual block to dst. The source
// coefficients are dequantized transform coefficients in AV1 coefficient order:
// coeff_idx = col * coeffStride + row.
// The pure-Go DCT path supports AV1 transform sizes with 4, 8, 16, or 32
// samples on each axis. AV1's reduced 64-wide transforms are handled by the
// separate 64-point path once that is ported.
func InverseDCTBlock(dst []int16, dstStride int, coeff []int32, coeffStride int, scratch []int32, size Size) error {
	shift, ok := size.shift()
	coeffSize := dctCoeffSize(size)
	scratchLen := size.Width * size.Height
	if !ok ||
		!dctBlockSupported(size) ||
		dstStride < size.Width ||
		coeffStride < coeffSize.Height ||
		len(scratch) < scratchLen ||
		!blockFits(len(dst), dstStride, size.Width, size.Height) ||
		!coeffBlockFits(len(coeff), coeffStride, coeffSize.Width, coeffSize.Height) {
		return ErrInvalidTransform
	}

	for row := 0; row < size.Height; row++ {
		tmpLine := scratch[row*size.Width : row*size.Width+size.Width]
		for col := 0; col < size.Width; col++ {
			if col < coeffSize.Width && row < coeffSize.Height {
				tmpLine[col] = coeff[col*coeffStride+row]
			} else {
				tmpLine[col] = 0
			}
		}
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
	return size.Valid() && dctLengthSupported(size.Width) && dctLengthSupported(size.Height)
}

func dctCoeffSize(size Size) Size {
	return adjustedScanSize(size)
}

func dctLengthSupported(length int) bool {
	switch length {
	case dct4Size, dct8Size, dct16Size, dct32Size:
		return true
	default:
		return false
	}
}

func inverseDCT1D(c []int32, stride int, length int, min int32, max int32) {
	switch length {
	case dct4Size:
		inverseDCT4(c, stride, min, max)
	case dct8Size:
		inverseDCT8(c, stride, min, max)
	case dct16Size:
		inverseDCT16(c, stride, min, max)
	case dct32Size:
		inverseDCT32(c, stride, min, max)
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

func inverseDCT16(c []int32, stride int, min int32, max int32) {
	inverseDCT8(c, stride<<1, min, max)

	in1 := int64(c[1*stride])
	in3 := int64(c[3*stride])
	in5 := int64(c[5*stride])
	in7 := int64(c[7*stride])
	in9 := int64(c[9*stride])
	in11 := int64(c[11*stride])
	in13 := int64(c[13*stride])
	in15 := int64(c[15*stride])

	t8a := roundShift(in1*401-in15*(4076-4096), 12) - in15
	t9a := roundShift(in9*1583-in7*1299, 11)
	t10a := roundShift(in5*1931-in11*(3612-4096), 12) - in11
	t11a := roundShift(in13*(3920-4096)-in3*1189, 12) + in13
	t12a := roundShift(in13*1189+in3*(3920-4096), 12) + in3
	t13a := roundShift(in5*(3612-4096)+in11*1931, 12) + in5
	t14a := roundShift(in9*1299+in7*1583, 11)
	t15a := roundShift(in1*(4076-4096)+in15*401, 12) + in1

	t8 := int64(clipRange(t8a+t9a, min, max))
	t9 := int64(clipRange(t8a-t9a, min, max))
	t10 := int64(clipRange(t11a-t10a, min, max))
	t11 := int64(clipRange(t11a+t10a, min, max))
	t12 := int64(clipRange(t12a+t13a, min, max))
	t13 := int64(clipRange(t12a-t13a, min, max))
	t14 := int64(clipRange(t15a-t14a, min, max))
	t15 := int64(clipRange(t15a+t14a, min, max))

	t9a = roundShift(t14*1567-t9*(3784-4096), 12) - t9
	t14a = roundShift(t14*(3784-4096)+t9*1567, 12) + t14
	t10a = roundShift(-(t13*(3784-4096)+t10*1567), 12) - t13
	t13a = roundShift(t13*1567-t10*(3784-4096), 12) - t10

	t8a = int64(clipRange(t8+t11, min, max))
	t9 = int64(clipRange(t9a+t10a, min, max))
	t10 = int64(clipRange(t9a-t10a, min, max))
	t11a = int64(clipRange(t8-t11, min, max))
	t12a = int64(clipRange(t15-t12, min, max))
	t13 = int64(clipRange(t14a-t13a, min, max))
	t14 = int64(clipRange(t14a+t13a, min, max))
	t15a = int64(clipRange(t15+t12, min, max))

	t10a = roundShift((t13-t10)*181, 8)
	t13a = roundShift((t13+t10)*181, 8)
	t11 = roundShift((t12a-t11a)*181, 8)
	t12 = roundShift((t12a+t11a)*181, 8)

	t0 := int64(c[0*stride])
	t1 := int64(c[2*stride])
	t2 := int64(c[4*stride])
	t3 := int64(c[6*stride])
	t4 := int64(c[8*stride])
	t5 := int64(c[10*stride])
	t6 := int64(c[12*stride])
	t7 := int64(c[14*stride])

	c[0*stride] = clipRange(t0+t15a, min, max)
	c[1*stride] = clipRange(t1+t14, min, max)
	c[2*stride] = clipRange(t2+t13a, min, max)
	c[3*stride] = clipRange(t3+t12, min, max)
	c[4*stride] = clipRange(t4+t11, min, max)
	c[5*stride] = clipRange(t5+t10a, min, max)
	c[6*stride] = clipRange(t6+t9, min, max)
	c[7*stride] = clipRange(t7+t8a, min, max)
	c[8*stride] = clipRange(t7-t8a, min, max)
	c[9*stride] = clipRange(t6-t9, min, max)
	c[10*stride] = clipRange(t5-t10a, min, max)
	c[11*stride] = clipRange(t4-t11, min, max)
	c[12*stride] = clipRange(t3-t12, min, max)
	c[13*stride] = clipRange(t2-t13a, min, max)
	c[14*stride] = clipRange(t1-t14, min, max)
	c[15*stride] = clipRange(t0-t15a, min, max)
}

func inverseDCT32(c []int32, stride int, min int32, max int32) {
	inverseDCT16(c, stride<<1, min, max)

	in1 := int64(c[1*stride])
	in3 := int64(c[3*stride])
	in5 := int64(c[5*stride])
	in7 := int64(c[7*stride])
	in9 := int64(c[9*stride])
	in11 := int64(c[11*stride])
	in13 := int64(c[13*stride])
	in15 := int64(c[15*stride])
	in17 := int64(c[17*stride])
	in19 := int64(c[19*stride])
	in21 := int64(c[21*stride])
	in23 := int64(c[23*stride])
	in25 := int64(c[25*stride])
	in27 := int64(c[27*stride])
	in29 := int64(c[29*stride])
	in31 := int64(c[31*stride])

	t16a := roundShift(in1*201-in31*(4091-4096), 12) - in31
	t17a := roundShift(in17*(3035-4096)-in15*2751, 12) + in17
	t18a := roundShift(in9*1751-in23*(3703-4096), 12) - in23
	t19a := roundShift(in25*(3857-4096)-in7*1380, 12) + in25
	t20a := roundShift(in5*995-in27*(3973-4096), 12) - in27
	t21a := roundShift(in21*(3513-4096)-in11*2106, 12) + in21
	t22a := roundShift(in13*1220-in19*1645, 11)
	t23a := roundShift(in29*(4052-4096)-in3*601, 12) + in29
	t24a := roundShift(in29*601+in3*(4052-4096), 12) + in3
	t25a := roundShift(in13*1645+in19*1220, 11)
	t26a := roundShift(in21*2106+in11*(3513-4096), 12) + in11
	t27a := roundShift(in5*(3973-4096)+in27*995, 12) + in5
	t28a := roundShift(in25*1380+in7*(3857-4096), 12) + in7
	t29a := roundShift(in9*(3703-4096)+in23*1751, 12) + in9
	t30a := roundShift(in17*2751+in15*(3035-4096), 12) + in15
	t31a := roundShift(in1*(4091-4096)+in31*201, 12) + in1

	t16 := int64(clipRange(t16a+t17a, min, max))
	t17 := int64(clipRange(t16a-t17a, min, max))
	t18 := int64(clipRange(t19a-t18a, min, max))
	t19 := int64(clipRange(t19a+t18a, min, max))
	t20 := int64(clipRange(t20a+t21a, min, max))
	t21 := int64(clipRange(t20a-t21a, min, max))
	t22 := int64(clipRange(t23a-t22a, min, max))
	t23 := int64(clipRange(t23a+t22a, min, max))
	t24 := int64(clipRange(t24a+t25a, min, max))
	t25 := int64(clipRange(t24a-t25a, min, max))
	t26 := int64(clipRange(t27a-t26a, min, max))
	t27 := int64(clipRange(t27a+t26a, min, max))
	t28 := int64(clipRange(t28a+t29a, min, max))
	t29 := int64(clipRange(t28a-t29a, min, max))
	t30 := int64(clipRange(t31a-t30a, min, max))
	t31 := int64(clipRange(t31a+t30a, min, max))

	t17a = roundShift(t30*799-t17*(4017-4096), 12) - t17
	t30a = roundShift(t30*(4017-4096)+t17*799, 12) + t30
	t18a = roundShift(-(t29*(4017-4096)+t18*799), 12) - t29
	t29a = roundShift(t29*799-t18*(4017-4096), 12) - t18
	t21a = roundShift(t26*1703-t21*1138, 11)
	t26a = roundShift(t26*1138+t21*1703, 11)
	t22a = roundShift(-(t25*1138 + t22*1703), 11)
	t25a = roundShift(t25*1703-t22*1138, 11)

	t16a = int64(clipRange(t16+t19, min, max))
	t17 = int64(clipRange(t17a+t18a, min, max))
	t18 = int64(clipRange(t17a-t18a, min, max))
	t19a = int64(clipRange(t16-t19, min, max))
	t20a = int64(clipRange(t23-t20, min, max))
	t21 = int64(clipRange(t22a-t21a, min, max))
	t22 = int64(clipRange(t22a+t21a, min, max))
	t23a = int64(clipRange(t23+t20, min, max))
	t24a = int64(clipRange(t24+t27, min, max))
	t25 = int64(clipRange(t25a+t26a, min, max))
	t26 = int64(clipRange(t25a-t26a, min, max))
	t27a = int64(clipRange(t24-t27, min, max))
	t28a = int64(clipRange(t31-t28, min, max))
	t29 = int64(clipRange(t30a-t29a, min, max))
	t30 = int64(clipRange(t30a+t29a, min, max))
	t31a = int64(clipRange(t31+t28, min, max))

	t18a = roundShift(t29*1567-t18*(3784-4096), 12) - t18
	t29a = roundShift(t29*(3784-4096)+t18*1567, 12) + t29
	t19 = roundShift(t28a*1567-t19a*(3784-4096), 12) - t19a
	t28 = roundShift(t28a*(3784-4096)+t19a*1567, 12) + t28a
	t20 = roundShift(-(t27a*(3784-4096)+t20a*1567), 12) - t27a
	t27 = roundShift(t27a*1567-t20a*(3784-4096), 12) - t20a
	t21a = roundShift(-(t26*(3784-4096)+t21*1567), 12) - t26
	t26a = roundShift(t26*1567-t21*(3784-4096), 12) - t21

	t16 = int64(clipRange(t16a+t23a, min, max))
	t17a = int64(clipRange(t17+t22, min, max))
	t18 = int64(clipRange(t18a+t21a, min, max))
	t19a = int64(clipRange(t19+t20, min, max))
	t20a = int64(clipRange(t19-t20, min, max))
	t21 = int64(clipRange(t18a-t21a, min, max))
	t22a = int64(clipRange(t17-t22, min, max))
	t23 = int64(clipRange(t16a-t23a, min, max))
	t24 = int64(clipRange(t31a-t24a, min, max))
	t25a = int64(clipRange(t30-t25, min, max))
	t26 = int64(clipRange(t29a-t26a, min, max))
	t27a = int64(clipRange(t28-t27, min, max))
	t28a = int64(clipRange(t28+t27, min, max))
	t29 = int64(clipRange(t29a+t26a, min, max))
	t30a = int64(clipRange(t30+t25, min, max))
	t31 = int64(clipRange(t31a+t24a, min, max))

	t20 = roundShift((t27a-t20a)*181, 8)
	t27 = roundShift((t27a+t20a)*181, 8)
	t21a = roundShift((t26-t21)*181, 8)
	t26a = roundShift((t26+t21)*181, 8)
	t22 = roundShift((t25a-t22a)*181, 8)
	t25 = roundShift((t25a+t22a)*181, 8)
	t23a = roundShift((t24-t23)*181, 8)
	t24a = roundShift((t24+t23)*181, 8)

	t0 := int64(c[0*stride])
	t1 := int64(c[2*stride])
	t2 := int64(c[4*stride])
	t3 := int64(c[6*stride])
	t4 := int64(c[8*stride])
	t5 := int64(c[10*stride])
	t6 := int64(c[12*stride])
	t7 := int64(c[14*stride])
	t8 := int64(c[16*stride])
	t9 := int64(c[18*stride])
	t10 := int64(c[20*stride])
	t11 := int64(c[22*stride])
	t12 := int64(c[24*stride])
	t13 := int64(c[26*stride])
	t14 := int64(c[28*stride])
	t15 := int64(c[30*stride])

	c[0*stride] = clipRange(t0+t31, min, max)
	c[1*stride] = clipRange(t1+t30a, min, max)
	c[2*stride] = clipRange(t2+t29, min, max)
	c[3*stride] = clipRange(t3+t28a, min, max)
	c[4*stride] = clipRange(t4+t27, min, max)
	c[5*stride] = clipRange(t5+t26a, min, max)
	c[6*stride] = clipRange(t6+t25, min, max)
	c[7*stride] = clipRange(t7+t24a, min, max)
	c[8*stride] = clipRange(t8+t23a, min, max)
	c[9*stride] = clipRange(t9+t22, min, max)
	c[10*stride] = clipRange(t10+t21a, min, max)
	c[11*stride] = clipRange(t11+t20, min, max)
	c[12*stride] = clipRange(t12+t19a, min, max)
	c[13*stride] = clipRange(t13+t18, min, max)
	c[14*stride] = clipRange(t14+t17a, min, max)
	c[15*stride] = clipRange(t15+t16, min, max)
	c[16*stride] = clipRange(t15-t16, min, max)
	c[17*stride] = clipRange(t14-t17a, min, max)
	c[18*stride] = clipRange(t13-t18, min, max)
	c[19*stride] = clipRange(t12-t19a, min, max)
	c[20*stride] = clipRange(t11-t20, min, max)
	c[21*stride] = clipRange(t10-t21a, min, max)
	c[22*stride] = clipRange(t9-t22, min, max)
	c[23*stride] = clipRange(t8-t23a, min, max)
	c[24*stride] = clipRange(t7-t24a, min, max)
	c[25*stride] = clipRange(t6-t25, min, max)
	c[26*stride] = clipRange(t5-t26a, min, max)
	c[27*stride] = clipRange(t4-t27, min, max)
	c[28*stride] = clipRange(t3-t28a, min, max)
	c[29*stride] = clipRange(t2-t29, min, max)
	c[30*stride] = clipRange(t1-t30a, min, max)
	c[31*stride] = clipRange(t0-t31, min, max)
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
