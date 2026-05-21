package transform

const dct4Size = 4
const dct8Size = 8
const dct16Size = 16
const dct32Size = 32
const dct64Size = 64

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
// The pure-Go DCT path supports AV1 transform sizes with 4, 8, 16, 32, or
// reduced-coefficient 64 samples on each axis.
func InverseDCTBlock(dst []int16, dstStride int, coeff []int32, coeffStride int, scratch []int32, size Size) error {
	return inverseSeparableBlock(dst, dstStride, coeff, coeffStride, scratch, size, TypeDCTDCT)
}

func dctLengthSupported(length int) bool {
	switch length {
	case dct4Size, dct8Size, dct16Size, dct32Size, dct64Size:
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
	case dct64Size:
		inverseDCT64(c, stride, min, max)
	}
}

func inverseDCT4(c []int32, stride int, min int32, max int32) {
	inverseDCT4Internal(c, stride, min, max, false)
}

func inverseDCT4Internal(c []int32, stride int, min int32, max int32, tx64 bool) {
	in0 := int64(c[0*stride])
	in1 := int64(c[1*stride])

	var t0, t1, t2, t3 int64
	if tx64 {
		t0 = roundShift(in0*181, 8)
		t1 = t0
		t2 = roundShift(in1*1567, 12)
		t3 = roundShift(in1*3784, 12)
	} else {
		in2 := int64(c[2*stride])
		in3 := int64(c[3*stride])
		t0 = roundShift((in0+in2)*181, 8)
		t1 = roundShift((in0-in2)*181, 8)
		t2 = roundShift(in1*1567-in3*(3784-4096), 12) - in3
		t3 = roundShift(in1*(3784-4096)+in3*1567, 12) + in1
	}

	c[0*stride] = clipRange(t0+t3, min, max)
	c[1*stride] = clipRange(t1+t2, min, max)
	c[2*stride] = clipRange(t1-t2, min, max)
	c[3*stride] = clipRange(t0-t3, min, max)
}

func inverseDCT8(c []int32, stride int, min int32, max int32) {
	inverseDCT8Internal(c, stride, min, max, false)
}

func inverseDCT8Internal(c []int32, stride int, min int32, max int32, tx64 bool) {
	inverseDCT4Internal(c, stride<<1, min, max, tx64)

	in1 := int64(c[1*stride])
	in3 := int64(c[3*stride])

	var t4a, t5a, t6a, t7a int64
	if tx64 {
		t4a = roundShift(in1*799, 12)
		t5a = roundShift(in3*-2276, 12)
		t6a = roundShift(in3*3406, 12)
		t7a = roundShift(in1*4017, 12)
	} else {
		in5 := int64(c[5*stride])
		in7 := int64(c[7*stride])
		t4a = roundShift(in1*799-in7*(4017-4096), 12) - in7
		t5a = roundShift(in5*1703-in3*1138, 11)
		t6a = roundShift(in5*1138+in3*1703, 11)
		t7a = roundShift(in1*(4017-4096)+in7*799, 12) + in1
	}

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
	inverseDCT16Internal(c, stride, min, max, false)
}

func inverseDCT16Internal(c []int32, stride int, min int32, max int32, tx64 bool) {
	inverseDCT8Internal(c, stride<<1, min, max, tx64)

	in1 := int64(c[1*stride])
	in3 := int64(c[3*stride])
	in5 := int64(c[5*stride])
	in7 := int64(c[7*stride])

	var t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a int64
	if tx64 {
		t8a = roundShift(in1*401, 12)
		t9a = roundShift(in7*-2598, 12)
		t10a = roundShift(in5*1931, 12)
		t11a = roundShift(in3*-1189, 12)
		t12a = roundShift(in3*3920, 12)
		t13a = roundShift(in5*3612, 12)
		t14a = roundShift(in7*3166, 12)
		t15a = roundShift(in1*4076, 12)
	} else {
		in9 := int64(c[9*stride])
		in11 := int64(c[11*stride])
		in13 := int64(c[13*stride])
		in15 := int64(c[15*stride])
		t8a = roundShift(in1*401-in15*(4076-4096), 12) - in15
		t9a = roundShift(in9*1583-in7*1299, 11)
		t10a = roundShift(in5*1931-in11*(3612-4096), 12) - in11
		t11a = roundShift(in13*(3920-4096)-in3*1189, 12) + in13
		t12a = roundShift(in13*1189+in3*(3920-4096), 12) + in3
		t13a = roundShift(in5*(3612-4096)+in11*1931, 12) + in5
		t14a = roundShift(in9*1299+in7*1583, 11)
		t15a = roundShift(in1*(4076-4096)+in15*401, 12) + in1
	}

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
	inverseDCT32Internal(c, stride, min, max, false)
}

func inverseDCT32Internal(c []int32, stride int, min int32, max int32, tx64 bool) {
	inverseDCT16Internal(c, stride<<1, min, max, tx64)

	in1 := int64(c[1*stride])
	in3 := int64(c[3*stride])
	in5 := int64(c[5*stride])
	in7 := int64(c[7*stride])
	in9 := int64(c[9*stride])
	in11 := int64(c[11*stride])
	in13 := int64(c[13*stride])
	in15 := int64(c[15*stride])

	var t16a, t17a, t18a, t19a, t20a, t21a, t22a, t23a int64
	var t24a, t25a, t26a, t27a, t28a, t29a, t30a, t31a int64
	if tx64 {
		t16a = roundShift(in1*201, 12)
		t17a = roundShift(in15*-2751, 12)
		t18a = roundShift(in9*1751, 12)
		t19a = roundShift(in7*-1380, 12)
		t20a = roundShift(in5*995, 12)
		t21a = roundShift(in11*-2106, 12)
		t22a = roundShift(in13*2440, 12)
		t23a = roundShift(in3*-601, 12)
		t24a = roundShift(in3*4052, 12)
		t25a = roundShift(in13*3290, 12)
		t26a = roundShift(in11*3513, 12)
		t27a = roundShift(in5*3973, 12)
		t28a = roundShift(in7*3857, 12)
		t29a = roundShift(in9*3703, 12)
		t30a = roundShift(in15*3035, 12)
		t31a = roundShift(in1*4091, 12)
	} else {
		in17 := int64(c[17*stride])
		in19 := int64(c[19*stride])
		in21 := int64(c[21*stride])
		in23 := int64(c[23*stride])
		in25 := int64(c[25*stride])
		in27 := int64(c[27*stride])
		in29 := int64(c[29*stride])
		in31 := int64(c[31*stride])
		t16a = roundShift(in1*201-in31*(4091-4096), 12) - in31
		t17a = roundShift(in17*(3035-4096)-in15*2751, 12) + in17
		t18a = roundShift(in9*1751-in23*(3703-4096), 12) - in23
		t19a = roundShift(in25*(3857-4096)-in7*1380, 12) + in25
		t20a = roundShift(in5*995-in27*(3973-4096), 12) - in27
		t21a = roundShift(in21*(3513-4096)-in11*2106, 12) + in21
		t22a = roundShift(in13*1220-in19*1645, 11)
		t23a = roundShift(in29*(4052-4096)-in3*601, 12) + in29
		t24a = roundShift(in29*601+in3*(4052-4096), 12) + in3
		t25a = roundShift(in13*1645+in19*1220, 11)
		t26a = roundShift(in21*2106+in11*(3513-4096), 12) + in11
		t27a = roundShift(in5*(3973-4096)+in27*995, 12) + in5
		t28a = roundShift(in25*1380+in7*(3857-4096), 12) + in7
		t29a = roundShift(in9*(3703-4096)+in23*1751, 12) + in9
		t30a = roundShift(in17*2751+in15*(3035-4096), 12) + in15
		t31a = roundShift(in1*(4091-4096)+in31*201, 12) + in1
	}

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

func inverseDCT64(c []int32, stride int, min int32, max int32) {
	inverseDCT32Internal(c, stride<<1, min, max, true)

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

	t32a := roundShift(in1*101, 12)
	t33a := roundShift(in31*-2824, 12)
	t34a := roundShift(in17*1660, 12)
	t35a := roundShift(in15*-1474, 12)
	t36a := roundShift(in9*897, 12)
	t37a := roundShift(in23*-2191, 12)
	t38a := roundShift(in25*2359, 12)
	t39a := roundShift(in7*-700, 12)
	t40a := roundShift(in5*501, 12)
	t41a := roundShift(in27*-2520, 12)
	t42a := roundShift(in21*2019, 12)
	t43a := roundShift(in11*-1092, 12)
	t44a := roundShift(in13*1285, 12)
	t45a := roundShift(in19*-1842, 12)
	t46a := roundShift(in29*2675, 12)
	t47a := roundShift(in3*-301, 12)
	t48a := roundShift(in3*4085, 12)
	t49a := roundShift(in29*3102, 12)
	t50a := roundShift(in19*3659, 12)
	t51a := roundShift(in13*3889, 12)
	t52a := roundShift(in11*3948, 12)
	t53a := roundShift(in21*3564, 12)
	t54a := roundShift(in27*3229, 12)
	t55a := roundShift(in5*4065, 12)
	t56a := roundShift(in7*4036, 12)
	t57a := roundShift(in25*3349, 12)
	t58a := roundShift(in23*3461, 12)
	t59a := roundShift(in9*3996, 12)
	t60a := roundShift(in15*3822, 12)
	t61a := roundShift(in17*3745, 12)
	t62a := roundShift(in31*2967, 12)
	t63a := roundShift(in1*4095, 12)

	t32 := int64(clipRange(t32a+t33a, min, max))
	t33 := int64(clipRange(t32a-t33a, min, max))
	t34 := int64(clipRange(t35a-t34a, min, max))
	t35 := int64(clipRange(t35a+t34a, min, max))
	t36 := int64(clipRange(t36a+t37a, min, max))
	t37 := int64(clipRange(t36a-t37a, min, max))
	t38 := int64(clipRange(t39a-t38a, min, max))
	t39 := int64(clipRange(t39a+t38a, min, max))
	t40 := int64(clipRange(t40a+t41a, min, max))
	t41 := int64(clipRange(t40a-t41a, min, max))
	t42 := int64(clipRange(t43a-t42a, min, max))
	t43 := int64(clipRange(t43a+t42a, min, max))
	t44 := int64(clipRange(t44a+t45a, min, max))
	t45 := int64(clipRange(t44a-t45a, min, max))
	t46 := int64(clipRange(t47a-t46a, min, max))
	t47 := int64(clipRange(t47a+t46a, min, max))
	t48 := int64(clipRange(t48a+t49a, min, max))
	t49 := int64(clipRange(t48a-t49a, min, max))
	t50 := int64(clipRange(t51a-t50a, min, max))
	t51 := int64(clipRange(t51a+t50a, min, max))
	t52 := int64(clipRange(t52a+t53a, min, max))
	t53 := int64(clipRange(t52a-t53a, min, max))
	t54 := int64(clipRange(t55a-t54a, min, max))
	t55 := int64(clipRange(t55a+t54a, min, max))
	t56 := int64(clipRange(t56a+t57a, min, max))
	t57 := int64(clipRange(t56a-t57a, min, max))
	t58 := int64(clipRange(t59a-t58a, min, max))
	t59 := int64(clipRange(t59a+t58a, min, max))
	t60 := int64(clipRange(t60a+t61a, min, max))
	t61 := int64(clipRange(t60a-t61a, min, max))
	t62 := int64(clipRange(t63a-t62a, min, max))
	t63 := int64(clipRange(t63a+t62a, min, max))

	t33a = roundShift(t33*(4096-4076)+t62*401, 12) - t33
	t34a = roundShift(t34*-401+t61*(4096-4076), 12) - t61
	t37a = roundShift(t37*-1299+t58*1583, 11)
	t38a = roundShift(t38*-1583+t57*-1299, 11)
	t41a = roundShift(t41*(4096-3612)+t54*1931, 12) - t41
	t42a = roundShift(t42*-1931+t53*(4096-3612), 12) - t53
	t45a = roundShift(t45*-1189+t50*(3920-4096), 12) + t50
	t46a = roundShift(t46*(4096-3920)+t49*-1189, 12) - t46
	t49a = roundShift(t46*-1189+t49*(3920-4096), 12) + t49
	t50a = roundShift(t45*(3920-4096)+t50*1189, 12) + t45
	t53a = roundShift(t42*(4096-3612)+t53*1931, 12) - t42
	t54a = roundShift(t41*1931+t54*(3612-4096), 12) + t54
	t57a = roundShift(t38*-1299+t57*1583, 11)
	t58a = roundShift(t37*1583+t58*1299, 11)
	t61a = roundShift(t34*(4096-4076)+t61*401, 12) - t34
	t62a = roundShift(t33*401+t62*(4076-4096), 12) + t62

	t32a = int64(clipRange(t32+t35, min, max))
	t33 = int64(clipRange(t33a+t34a, min, max))
	t34 = int64(clipRange(t33a-t34a, min, max))
	t35a = int64(clipRange(t32-t35, min, max))
	t36a = int64(clipRange(t39-t36, min, max))
	t37 = int64(clipRange(t38a-t37a, min, max))
	t38 = int64(clipRange(t38a+t37a, min, max))
	t39a = int64(clipRange(t39+t36, min, max))
	t40a = int64(clipRange(t40+t43, min, max))
	t41 = int64(clipRange(t41a+t42a, min, max))
	t42 = int64(clipRange(t41a-t42a, min, max))
	t43a = int64(clipRange(t40-t43, min, max))
	t44a = int64(clipRange(t47-t44, min, max))
	t45 = int64(clipRange(t46a-t45a, min, max))
	t46 = int64(clipRange(t46a+t45a, min, max))
	t47a = int64(clipRange(t47+t44, min, max))
	t48a = int64(clipRange(t48+t51, min, max))
	t49 = int64(clipRange(t49a+t50a, min, max))
	t50 = int64(clipRange(t49a-t50a, min, max))
	t51a = int64(clipRange(t48-t51, min, max))
	t52a = int64(clipRange(t55-t52, min, max))
	t53 = int64(clipRange(t54a-t53a, min, max))
	t54 = int64(clipRange(t54a+t53a, min, max))
	t55a = int64(clipRange(t55+t52, min, max))
	t56a = int64(clipRange(t56+t59, min, max))
	t57 = int64(clipRange(t57a+t58a, min, max))
	t58 = int64(clipRange(t57a-t58a, min, max))
	t59a = int64(clipRange(t56-t59, min, max))
	t60a = int64(clipRange(t63-t60, min, max))
	t61 = int64(clipRange(t62a-t61a, min, max))
	t62 = int64(clipRange(t62a+t61a, min, max))
	t63a = int64(clipRange(t63+t60, min, max))

	t34a = roundShift(t34*(4096-4017)+t61*799, 12) - t34
	t35 = roundShift(t35a*(4096-4017)+t60a*799, 12) - t35a
	t36 = roundShift(t36a*-799+t59a*(4096-4017), 12) - t59a
	t37a = roundShift(t37*-799+t58*(4096-4017), 12) - t58
	t42a = roundShift(t42*-1138+t53*1703, 11)
	t43 = roundShift(t43a*-1138+t52a*1703, 11)
	t44 = roundShift(t44a*-1703+t51a*-1138, 11)
	t45a = roundShift(t45*-1703+t50*-1138, 11)
	t50a = roundShift(t45*-1138+t50*1703, 11)
	t51 = roundShift(t44a*-1138+t51a*1703, 11)
	t52 = roundShift(t43a*1703+t52a*1138, 11)
	t53a = roundShift(t42*1703+t53*1138, 11)
	t58a = roundShift(t37*(4096-4017)+t58*799, 12) - t37
	t59 = roundShift(t36a*(4096-4017)+t59a*799, 12) - t36a
	t60 = roundShift(t35a*799+t60a*(4017-4096), 12) + t60a
	t61a = roundShift(t34*799+t61*(4017-4096), 12) + t61

	t32 = int64(clipRange(t32a+t39a, min, max))
	t33a = int64(clipRange(t33+t38, min, max))
	t34 = int64(clipRange(t34a+t37a, min, max))
	t35a = int64(clipRange(t35+t36, min, max))
	t36a = int64(clipRange(t35-t36, min, max))
	t37 = int64(clipRange(t34a-t37a, min, max))
	t38a = int64(clipRange(t33-t38, min, max))
	t39 = int64(clipRange(t32a-t39a, min, max))
	t40 = int64(clipRange(t47a-t40a, min, max))
	t41a = int64(clipRange(t46-t41, min, max))
	t42 = int64(clipRange(t45a-t42a, min, max))
	t43a = int64(clipRange(t44-t43, min, max))
	t44a = int64(clipRange(t44+t43, min, max))
	t45 = int64(clipRange(t45a+t42a, min, max))
	t46a = int64(clipRange(t46+t41, min, max))
	t47 = int64(clipRange(t47a+t40a, min, max))
	t48 = int64(clipRange(t48a+t55a, min, max))
	t49a = int64(clipRange(t49+t54, min, max))
	t50 = int64(clipRange(t50a+t53a, min, max))
	t51a = int64(clipRange(t51+t52, min, max))
	t52a = int64(clipRange(t51-t52, min, max))
	t53 = int64(clipRange(t50a-t53a, min, max))
	t54a = int64(clipRange(t49-t54, min, max))
	t55 = int64(clipRange(t48a-t55a, min, max))
	t56 = int64(clipRange(t63a-t56a, min, max))
	t57a = int64(clipRange(t62-t57, min, max))
	t58 = int64(clipRange(t61a-t58a, min, max))
	t59a = int64(clipRange(t60-t59, min, max))
	t60a = int64(clipRange(t60+t59, min, max))
	t61 = int64(clipRange(t61a+t58a, min, max))
	t62a = int64(clipRange(t62+t57, min, max))
	t63 = int64(clipRange(t63a+t56a, min, max))

	t36 = roundShift(t36a*(4096-3784)+t59a*1567, 12) - t36a
	t37a = roundShift(t37*(4096-3784)+t58*1567, 12) - t37
	t38 = roundShift(t38a*(4096-3784)+t57a*1567, 12) - t38a
	t39a = roundShift(t39*(4096-3784)+t56*1567, 12) - t39
	t40a = roundShift(t40*-1567+t55*(4096-3784), 12) - t55
	t41 = roundShift(t41a*-1567+t54a*(4096-3784), 12) - t54a
	t42a = roundShift(t42*-1567+t53*(4096-3784), 12) - t53
	t43 = roundShift(t43a*-1567+t52a*(4096-3784), 12) - t52a
	t52 = roundShift(t43a*(4096-3784)+t52a*1567, 12) - t43a
	t53a = roundShift(t42*(4096-3784)+t53*1567, 12) - t42
	t54 = roundShift(t41a*(4096-3784)+t54a*1567, 12) - t41a
	t55a = roundShift(t40*(4096-3784)+t55*1567, 12) - t40
	t56a = roundShift(t39*1567+t56*(3784-4096), 12) + t56
	t57 = roundShift(t38a*1567+t57a*(3784-4096), 12) + t57a
	t58a = roundShift(t37*1567+t58*(3784-4096), 12) + t58
	t59 = roundShift(t36a*1567+t59a*(3784-4096), 12) + t59a

	t32a = int64(clipRange(t32+t47, min, max))
	t33 = int64(clipRange(t33a+t46a, min, max))
	t34a = int64(clipRange(t34+t45, min, max))
	t35 = int64(clipRange(t35a+t44a, min, max))
	t36a = int64(clipRange(t36+t43, min, max))
	t37 = int64(clipRange(t37a+t42a, min, max))
	t38a = int64(clipRange(t38+t41, min, max))
	t39 = int64(clipRange(t39a+t40a, min, max))
	t40 = int64(clipRange(t39a-t40a, min, max))
	t41a = int64(clipRange(t38-t41, min, max))
	t42 = int64(clipRange(t37a-t42a, min, max))
	t43a = int64(clipRange(t36-t43, min, max))
	t44 = int64(clipRange(t35a-t44a, min, max))
	t45a = int64(clipRange(t34-t45, min, max))
	t46 = int64(clipRange(t33a-t46a, min, max))
	t47a = int64(clipRange(t32-t47, min, max))
	t48a = int64(clipRange(t63-t48, min, max))
	t49 = int64(clipRange(t62a-t49a, min, max))
	t50a = int64(clipRange(t61-t50, min, max))
	t51 = int64(clipRange(t60a-t51a, min, max))
	t52a = int64(clipRange(t59-t52, min, max))
	t53 = int64(clipRange(t58a-t53a, min, max))
	t54a = int64(clipRange(t57-t54, min, max))
	t55 = int64(clipRange(t56a-t55a, min, max))
	t56 = int64(clipRange(t56a+t55a, min, max))
	t57a = int64(clipRange(t57+t54, min, max))
	t58 = int64(clipRange(t58a+t53a, min, max))
	t59a = int64(clipRange(t59+t52, min, max))
	t60 = int64(clipRange(t60a+t51a, min, max))
	t61a = int64(clipRange(t61+t50, min, max))
	t62 = int64(clipRange(t62a+t49a, min, max))
	t63a = int64(clipRange(t63+t48, min, max))

	t40a = roundShift((t55-t40)*181, 8)
	t41 = roundShift((t54a-t41a)*181, 8)
	t42a = roundShift((t53-t42)*181, 8)
	t43 = roundShift((t52a-t43a)*181, 8)
	t44a = roundShift((t51-t44)*181, 8)
	t45 = roundShift((t50a-t45a)*181, 8)
	t46a = roundShift((t49-t46)*181, 8)
	t47 = roundShift((t48a-t47a)*181, 8)
	t48 = roundShift((t47a+t48a)*181, 8)
	t49a = roundShift((t46+t49)*181, 8)
	t50 = roundShift((t45a+t50a)*181, 8)
	t51a = roundShift((t44+t51)*181, 8)
	t52 = roundShift((t43a+t52a)*181, 8)
	t53a = roundShift((t42+t53)*181, 8)
	t54 = roundShift((t41a+t54a)*181, 8)
	t55a = roundShift((t40+t55)*181, 8)

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
	t16 := int64(c[32*stride])
	t17 := int64(c[34*stride])
	t18 := int64(c[36*stride])
	t19 := int64(c[38*stride])
	t20 := int64(c[40*stride])
	t21 := int64(c[42*stride])
	t22 := int64(c[44*stride])
	t23 := int64(c[46*stride])
	t24 := int64(c[48*stride])
	t25 := int64(c[50*stride])
	t26 := int64(c[52*stride])
	t27 := int64(c[54*stride])
	t28 := int64(c[56*stride])
	t29 := int64(c[58*stride])
	t30 := int64(c[60*stride])
	t31 := int64(c[62*stride])

	c[0*stride] = clipRange(t0+t63a, min, max)
	c[1*stride] = clipRange(t1+t62, min, max)
	c[2*stride] = clipRange(t2+t61a, min, max)
	c[3*stride] = clipRange(t3+t60, min, max)
	c[4*stride] = clipRange(t4+t59a, min, max)
	c[5*stride] = clipRange(t5+t58, min, max)
	c[6*stride] = clipRange(t6+t57a, min, max)
	c[7*stride] = clipRange(t7+t56, min, max)
	c[8*stride] = clipRange(t8+t55a, min, max)
	c[9*stride] = clipRange(t9+t54, min, max)
	c[10*stride] = clipRange(t10+t53a, min, max)
	c[11*stride] = clipRange(t11+t52, min, max)
	c[12*stride] = clipRange(t12+t51a, min, max)
	c[13*stride] = clipRange(t13+t50, min, max)
	c[14*stride] = clipRange(t14+t49a, min, max)
	c[15*stride] = clipRange(t15+t48, min, max)
	c[16*stride] = clipRange(t16+t47, min, max)
	c[17*stride] = clipRange(t17+t46a, min, max)
	c[18*stride] = clipRange(t18+t45, min, max)
	c[19*stride] = clipRange(t19+t44a, min, max)
	c[20*stride] = clipRange(t20+t43, min, max)
	c[21*stride] = clipRange(t21+t42a, min, max)
	c[22*stride] = clipRange(t22+t41, min, max)
	c[23*stride] = clipRange(t23+t40a, min, max)
	c[24*stride] = clipRange(t24+t39, min, max)
	c[25*stride] = clipRange(t25+t38a, min, max)
	c[26*stride] = clipRange(t26+t37, min, max)
	c[27*stride] = clipRange(t27+t36a, min, max)
	c[28*stride] = clipRange(t28+t35, min, max)
	c[29*stride] = clipRange(t29+t34a, min, max)
	c[30*stride] = clipRange(t30+t33, min, max)
	c[31*stride] = clipRange(t31+t32a, min, max)
	c[32*stride] = clipRange(t31-t32a, min, max)
	c[33*stride] = clipRange(t30-t33, min, max)
	c[34*stride] = clipRange(t29-t34a, min, max)
	c[35*stride] = clipRange(t28-t35, min, max)
	c[36*stride] = clipRange(t27-t36a, min, max)
	c[37*stride] = clipRange(t26-t37, min, max)
	c[38*stride] = clipRange(t25-t38a, min, max)
	c[39*stride] = clipRange(t24-t39, min, max)
	c[40*stride] = clipRange(t23-t40a, min, max)
	c[41*stride] = clipRange(t22-t41, min, max)
	c[42*stride] = clipRange(t21-t42a, min, max)
	c[43*stride] = clipRange(t20-t43, min, max)
	c[44*stride] = clipRange(t19-t44a, min, max)
	c[45*stride] = clipRange(t18-t45, min, max)
	c[46*stride] = clipRange(t17-t46a, min, max)
	c[47*stride] = clipRange(t16-t47, min, max)
	c[48*stride] = clipRange(t15-t48, min, max)
	c[49*stride] = clipRange(t14-t49a, min, max)
	c[50*stride] = clipRange(t13-t50, min, max)
	c[51*stride] = clipRange(t12-t51a, min, max)
	c[52*stride] = clipRange(t11-t52, min, max)
	c[53*stride] = clipRange(t10-t53a, min, max)
	c[54*stride] = clipRange(t9-t54, min, max)
	c[55*stride] = clipRange(t8-t55a, min, max)
	c[56*stride] = clipRange(t7-t56, min, max)
	c[57*stride] = clipRange(t6-t57a, min, max)
	c[58*stride] = clipRange(t5-t58, min, max)
	c[59*stride] = clipRange(t4-t59a, min, max)
	c[60*stride] = clipRange(t3-t60, min, max)
	c[61*stride] = clipRange(t2-t61a, min, max)
	c[62*stride] = clipRange(t1-t62, min, max)
	c[63*stride] = clipRange(t0-t63a, min, max)
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
