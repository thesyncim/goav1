// Ported from libaom: av1/common/av1_inv_txfm1d.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

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
	return int(size.Width) * int(size.Height), nil
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

func inverseDCT1D[T txElem](c []T, stride int, length int, min int32, max int32) {
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

func inverseDCT4[T txElem](c []T, stride int, min int32, max int32) {
	in0 := int64(c[0*stride])
	in1 := int64(c[1*stride])
	in2 := int64(c[2*stride])
	in3 := int64(c[3*stride])

	t0 := roundShift((in0+in2)*181, 8)
	t1 := roundShift((in0-in2)*181, 8)
	t2 := roundShift(in1*1567-in3*(3784-4096), 12) - in3
	t3 := roundShift(in1*(3784-4096)+in3*1567, 12) + in1

	c[0*stride] = clipRangeT[T](t0+t3, min, max)
	c[1*stride] = clipRangeT[T](t1+t2, min, max)
	c[2*stride] = clipRangeT[T](t1-t2, min, max)
	c[3*stride] = clipRangeT[T](t0-t3, min, max)
}

func inverseDCT8[T txElem](c []T, stride int, min int32, max int32) {
	inverseDCT4(c, stride<<1, min, max)

	in1 := int64(c[1*stride])
	in3 := int64(c[3*stride])
	in5 := int64(c[5*stride])
	in7 := int64(c[7*stride])

	t4a := roundShift(in1*799-in7*(4017-4096), 12) - in7
	t5a := roundShift(in5*1703-in3*1138, 11)
	t6a := roundShift(in5*1138+in3*1703, 11)
	t7a := roundShift(in1*(4017-4096)+in7*799, 12) + in1

	t4 := int64(clipRangeT[T](t4a+t5a, min, max))
	t5a = int64(clipRangeT[T](t4a-t5a, min, max))
	t7 := int64(clipRangeT[T](t7a+t6a, min, max))
	t6a = int64(clipRangeT[T](t7a-t6a, min, max))

	t5 := roundShift((t6a-t5a)*181, 8)
	t6 := roundShift((t6a+t5a)*181, 8)

	t0 := int64(c[0*stride])
	t1 := int64(c[2*stride])
	t2 := int64(c[4*stride])
	t3 := int64(c[6*stride])

	c[0*stride] = clipRangeT[T](t0+t7, min, max)
	c[1*stride] = clipRangeT[T](t1+t6, min, max)
	c[2*stride] = clipRangeT[T](t2+t5, min, max)
	c[3*stride] = clipRangeT[T](t3+t4, min, max)
	c[4*stride] = clipRangeT[T](t3-t4, min, max)
	c[5*stride] = clipRangeT[T](t2-t5, min, max)
	c[6*stride] = clipRangeT[T](t1-t6, min, max)
	c[7*stride] = clipRangeT[T](t0-t7, min, max)
}

func inverseDCT16[T txElem](c []T, stride int, min int32, max int32) {
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

	t8 := int64(clipRangeT[T](t8a+t9a, min, max))
	t9 := int64(clipRangeT[T](t8a-t9a, min, max))
	t10 := int64(clipRangeT[T](t11a-t10a, min, max))
	t11 := int64(clipRangeT[T](t11a+t10a, min, max))
	t12 := int64(clipRangeT[T](t12a+t13a, min, max))
	t13 := int64(clipRangeT[T](t12a-t13a, min, max))
	t14 := int64(clipRangeT[T](t15a-t14a, min, max))
	t15 := int64(clipRangeT[T](t15a+t14a, min, max))

	t9a = roundShift(t14*1567-t9*(3784-4096), 12) - t9
	t14a = roundShift(t14*(3784-4096)+t9*1567, 12) + t14
	t10a = roundShift(-(t13*(3784-4096)+t10*1567), 12) - t13
	t13a = roundShift(t13*1567-t10*(3784-4096), 12) - t10

	t8a = int64(clipRangeT[T](t8+t11, min, max))
	t9 = int64(clipRangeT[T](t9a+t10a, min, max))
	t10 = int64(clipRangeT[T](t9a-t10a, min, max))
	t11a = int64(clipRangeT[T](t8-t11, min, max))
	t12a = int64(clipRangeT[T](t15-t12, min, max))
	t13 = int64(clipRangeT[T](t14a-t13a, min, max))
	t14 = int64(clipRangeT[T](t14a+t13a, min, max))
	t15a = int64(clipRangeT[T](t15+t12, min, max))

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

	c[0*stride] = clipRangeT[T](t0+t15a, min, max)
	c[1*stride] = clipRangeT[T](t1+t14, min, max)
	c[2*stride] = clipRangeT[T](t2+t13a, min, max)
	c[3*stride] = clipRangeT[T](t3+t12, min, max)
	c[4*stride] = clipRangeT[T](t4+t11, min, max)
	c[5*stride] = clipRangeT[T](t5+t10a, min, max)
	c[6*stride] = clipRangeT[T](t6+t9, min, max)
	c[7*stride] = clipRangeT[T](t7+t8a, min, max)
	c[8*stride] = clipRangeT[T](t7-t8a, min, max)
	c[9*stride] = clipRangeT[T](t6-t9, min, max)
	c[10*stride] = clipRangeT[T](t5-t10a, min, max)
	c[11*stride] = clipRangeT[T](t4-t11, min, max)
	c[12*stride] = clipRangeT[T](t3-t12, min, max)
	c[13*stride] = clipRangeT[T](t2-t13a, min, max)
	c[14*stride] = clipRangeT[T](t1-t14, min, max)
	c[15*stride] = clipRangeT[T](t0-t15a, min, max)
}

func inverseDCT32[T txElem](c []T, stride int, min int32, max int32) {
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

	t16 := int64(clipRangeT[T](t16a+t17a, min, max))
	t17 := int64(clipRangeT[T](t16a-t17a, min, max))
	t18 := int64(clipRangeT[T](t19a-t18a, min, max))
	t19 := int64(clipRangeT[T](t19a+t18a, min, max))
	t20 := int64(clipRangeT[T](t20a+t21a, min, max))
	t21 := int64(clipRangeT[T](t20a-t21a, min, max))
	t22 := int64(clipRangeT[T](t23a-t22a, min, max))
	t23 := int64(clipRangeT[T](t23a+t22a, min, max))
	t24 := int64(clipRangeT[T](t24a+t25a, min, max))
	t25 := int64(clipRangeT[T](t24a-t25a, min, max))
	t26 := int64(clipRangeT[T](t27a-t26a, min, max))
	t27 := int64(clipRangeT[T](t27a+t26a, min, max))
	t28 := int64(clipRangeT[T](t28a+t29a, min, max))
	t29 := int64(clipRangeT[T](t28a-t29a, min, max))
	t30 := int64(clipRangeT[T](t31a-t30a, min, max))
	t31 := int64(clipRangeT[T](t31a+t30a, min, max))

	t17a = roundShift(t30*799-t17*(4017-4096), 12) - t17
	t30a = roundShift(t30*(4017-4096)+t17*799, 12) + t30
	t18a = roundShift(-(t29*(4017-4096)+t18*799), 12) - t29
	t29a = roundShift(t29*799-t18*(4017-4096), 12) - t18
	t21a = roundShift(t26*1703-t21*1138, 11)
	t26a = roundShift(t26*1138+t21*1703, 11)
	t22a = roundShift(-(t25*1138 + t22*1703), 11)
	t25a = roundShift(t25*1703-t22*1138, 11)

	t16a = int64(clipRangeT[T](t16+t19, min, max))
	t17 = int64(clipRangeT[T](t17a+t18a, min, max))
	t18 = int64(clipRangeT[T](t17a-t18a, min, max))
	t19a = int64(clipRangeT[T](t16-t19, min, max))
	t20a = int64(clipRangeT[T](t23-t20, min, max))
	t21 = int64(clipRangeT[T](t22a-t21a, min, max))
	t22 = int64(clipRangeT[T](t22a+t21a, min, max))
	t23a = int64(clipRangeT[T](t23+t20, min, max))
	t24a = int64(clipRangeT[T](t24+t27, min, max))
	t25 = int64(clipRangeT[T](t25a+t26a, min, max))
	t26 = int64(clipRangeT[T](t25a-t26a, min, max))
	t27a = int64(clipRangeT[T](t24-t27, min, max))
	t28a = int64(clipRangeT[T](t31-t28, min, max))
	t29 = int64(clipRangeT[T](t30a-t29a, min, max))
	t30 = int64(clipRangeT[T](t30a+t29a, min, max))
	t31a = int64(clipRangeT[T](t31+t28, min, max))

	t18a = roundShift(t29*1567-t18*(3784-4096), 12) - t18
	t29a = roundShift(t29*(3784-4096)+t18*1567, 12) + t29
	t19 = roundShift(t28a*1567-t19a*(3784-4096), 12) - t19a
	t28 = roundShift(t28a*(3784-4096)+t19a*1567, 12) + t28a
	t20 = roundShift(-(t27a*(3784-4096)+t20a*1567), 12) - t27a
	t27 = roundShift(t27a*1567-t20a*(3784-4096), 12) - t20a
	t21a = roundShift(-(t26*(3784-4096)+t21*1567), 12) - t26
	t26a = roundShift(t26*1567-t21*(3784-4096), 12) - t21

	t16 = int64(clipRangeT[T](t16a+t23a, min, max))
	t17a = int64(clipRangeT[T](t17+t22, min, max))
	t18 = int64(clipRangeT[T](t18a+t21a, min, max))
	t19a = int64(clipRangeT[T](t19+t20, min, max))
	t20a = int64(clipRangeT[T](t19-t20, min, max))
	t21 = int64(clipRangeT[T](t18a-t21a, min, max))
	t22a = int64(clipRangeT[T](t17-t22, min, max))
	t23 = int64(clipRangeT[T](t16a-t23a, min, max))
	t24 = int64(clipRangeT[T](t31a-t24a, min, max))
	t25a = int64(clipRangeT[T](t30-t25, min, max))
	t26 = int64(clipRangeT[T](t29a-t26a, min, max))
	t27a = int64(clipRangeT[T](t28-t27, min, max))
	t28a = int64(clipRangeT[T](t28+t27, min, max))
	t29 = int64(clipRangeT[T](t29a+t26a, min, max))
	t30a = int64(clipRangeT[T](t30+t25, min, max))
	t31 = int64(clipRangeT[T](t31a+t24a, min, max))

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

	c[0*stride] = clipRangeT[T](t0+t31, min, max)
	c[1*stride] = clipRangeT[T](t1+t30a, min, max)
	c[2*stride] = clipRangeT[T](t2+t29, min, max)
	c[3*stride] = clipRangeT[T](t3+t28a, min, max)
	c[4*stride] = clipRangeT[T](t4+t27, min, max)
	c[5*stride] = clipRangeT[T](t5+t26a, min, max)
	c[6*stride] = clipRangeT[T](t6+t25, min, max)
	c[7*stride] = clipRangeT[T](t7+t24a, min, max)
	c[8*stride] = clipRangeT[T](t8+t23a, min, max)
	c[9*stride] = clipRangeT[T](t9+t22, min, max)
	c[10*stride] = clipRangeT[T](t10+t21a, min, max)
	c[11*stride] = clipRangeT[T](t11+t20, min, max)
	c[12*stride] = clipRangeT[T](t12+t19a, min, max)
	c[13*stride] = clipRangeT[T](t13+t18, min, max)
	c[14*stride] = clipRangeT[T](t14+t17a, min, max)
	c[15*stride] = clipRangeT[T](t15+t16, min, max)
	c[16*stride] = clipRangeT[T](t15-t16, min, max)
	c[17*stride] = clipRangeT[T](t14-t17a, min, max)
	c[18*stride] = clipRangeT[T](t13-t18, min, max)
	c[19*stride] = clipRangeT[T](t12-t19a, min, max)
	c[20*stride] = clipRangeT[T](t11-t20, min, max)
	c[21*stride] = clipRangeT[T](t10-t21a, min, max)
	c[22*stride] = clipRangeT[T](t9-t22, min, max)
	c[23*stride] = clipRangeT[T](t8-t23a, min, max)
	c[24*stride] = clipRangeT[T](t7-t24a, min, max)
	c[25*stride] = clipRangeT[T](t6-t25, min, max)
	c[26*stride] = clipRangeT[T](t5-t26a, min, max)
	c[27*stride] = clipRangeT[T](t4-t27, min, max)
	c[28*stride] = clipRangeT[T](t3-t28a, min, max)
	c[29*stride] = clipRangeT[T](t2-t29, min, max)
	c[30*stride] = clipRangeT[T](t1-t30a, min, max)
	c[31*stride] = clipRangeT[T](t0-t31, min, max)
}

// inverseDCT64 is a libaom-faithful port of av1_idct64
// (av1/common/av1_inv_txfm1d.c) for cos_bit=12. Callers supply the
// per-pass stage-range clamp via (min, max), matching libaom's
// av1_gen_inv_stage_range() output. Unlike the smaller-length DCT inverses,
// this implementation reads the full 64-element input via stride and writes
// the 64-element output back through stride; it does NOT assume input
// positions 33, 35, ..., 63 are zero. (The prior reduced-input form was
// only correct when the input was the "reduced 32-coeff" 64x64 row pass;
// it produced growing-with-column residual error on the column pass.)
func inverseDCT64[T txElem](c []T, stride int, min int32, max int32) {
	clamp := func(v int64) T { return clipRangeT[T](v, min, max) }
	hbtf := func(w0, in0, w1, in1 int64) int64 {
		return roundShift(w0*in0+w1*in1, 12)
	}
	const (
		cospi1  = 4095
		cospi2  = 4091
		cospi3  = 4085
		cospi4  = 4076
		cospi5  = 4065
		cospi6  = 4052
		cospi7  = 4036
		cospi8  = 4017
		cospi9  = 3996
		cospi10 = 3973
		cospi11 = 3948
		cospi12 = 3920
		cospi13 = 3889
		cospi14 = 3857
		cospi15 = 3822
		cospi16 = 3784
		cospi17 = 3745
		cospi18 = 3703
		cospi19 = 3659
		cospi20 = 3612
		cospi21 = 3564
		cospi22 = 3513
		cospi23 = 3461
		cospi24 = 3406
		cospi25 = 3349
		cospi26 = 3290
		cospi27 = 3229
		cospi28 = 3166
		cospi29 = 3102
		cospi30 = 3035
		cospi31 = 2967
		cospi32 = 2896
		cospi33 = 2824
		cospi34 = 2751
		cospi35 = 2675
		cospi36 = 2598
		cospi37 = 2520
		cospi38 = 2440
		cospi39 = 2359
		cospi40 = 2276
		cospi41 = 2191
		cospi42 = 2106
		cospi43 = 2019
		cospi44 = 1931
		cospi45 = 1842
		cospi46 = 1751
		cospi47 = 1660
		cospi48 = 1567
		cospi49 = 1474
		cospi50 = 1380
		cospi51 = 1285
		cospi52 = 1189
		cospi53 = 1092
		cospi54 = 995
		cospi55 = 897
		cospi56 = 799
		cospi57 = 700
		cospi58 = 601
		cospi59 = 501
		cospi60 = 401
		cospi61 = 301
		cospi62 = 201
		cospi63 = 101
	)

	var bf0, bf1 [64]int64

	// stage 1: bit reversal of input
	bf1[0] = int64(c[0*stride])
	bf1[1] = int64(c[32*stride])
	bf1[2] = int64(c[16*stride])
	bf1[3] = int64(c[48*stride])
	bf1[4] = int64(c[8*stride])
	bf1[5] = int64(c[40*stride])
	bf1[6] = int64(c[24*stride])
	bf1[7] = int64(c[56*stride])
	bf1[8] = int64(c[4*stride])
	bf1[9] = int64(c[36*stride])
	bf1[10] = int64(c[20*stride])
	bf1[11] = int64(c[52*stride])
	bf1[12] = int64(c[12*stride])
	bf1[13] = int64(c[44*stride])
	bf1[14] = int64(c[28*stride])
	bf1[15] = int64(c[60*stride])
	bf1[16] = int64(c[2*stride])
	bf1[17] = int64(c[34*stride])
	bf1[18] = int64(c[18*stride])
	bf1[19] = int64(c[50*stride])
	bf1[20] = int64(c[10*stride])
	bf1[21] = int64(c[42*stride])
	bf1[22] = int64(c[26*stride])
	bf1[23] = int64(c[58*stride])
	bf1[24] = int64(c[6*stride])
	bf1[25] = int64(c[38*stride])
	bf1[26] = int64(c[22*stride])
	bf1[27] = int64(c[54*stride])
	bf1[28] = int64(c[14*stride])
	bf1[29] = int64(c[46*stride])
	bf1[30] = int64(c[30*stride])
	bf1[31] = int64(c[62*stride])
	bf1[32] = int64(c[1*stride])
	bf1[33] = int64(c[33*stride])
	bf1[34] = int64(c[17*stride])
	bf1[35] = int64(c[49*stride])
	bf1[36] = int64(c[9*stride])
	bf1[37] = int64(c[41*stride])
	bf1[38] = int64(c[25*stride])
	bf1[39] = int64(c[57*stride])
	bf1[40] = int64(c[5*stride])
	bf1[41] = int64(c[37*stride])
	bf1[42] = int64(c[21*stride])
	bf1[43] = int64(c[53*stride])
	bf1[44] = int64(c[13*stride])
	bf1[45] = int64(c[45*stride])
	bf1[46] = int64(c[29*stride])
	bf1[47] = int64(c[61*stride])
	bf1[48] = int64(c[3*stride])
	bf1[49] = int64(c[35*stride])
	bf1[50] = int64(c[19*stride])
	bf1[51] = int64(c[51*stride])
	bf1[52] = int64(c[11*stride])
	bf1[53] = int64(c[43*stride])
	bf1[54] = int64(c[27*stride])
	bf1[55] = int64(c[59*stride])
	bf1[56] = int64(c[7*stride])
	bf1[57] = int64(c[39*stride])
	bf1[58] = int64(c[23*stride])
	bf1[59] = int64(c[55*stride])
	bf1[60] = int64(c[15*stride])
	bf1[61] = int64(c[47*stride])
	bf1[62] = int64(c[31*stride])
	bf1[63] = int64(c[63*stride])

	// stage 2: bf0 := bf1 (low 32 copy; high 32 half_btf with cospi(63..1))
	bf0 = bf1
	bf1[32] = hbtf(cospi63, bf0[32], -cospi1, bf0[63])
	bf1[33] = hbtf(cospi31, bf0[33], -cospi33, bf0[62])
	bf1[34] = hbtf(cospi47, bf0[34], -cospi17, bf0[61])
	bf1[35] = hbtf(cospi15, bf0[35], -cospi49, bf0[60])
	bf1[36] = hbtf(cospi55, bf0[36], -cospi9, bf0[59])
	bf1[37] = hbtf(cospi23, bf0[37], -cospi41, bf0[58])
	bf1[38] = hbtf(cospi39, bf0[38], -cospi25, bf0[57])
	bf1[39] = hbtf(cospi7, bf0[39], -cospi57, bf0[56])
	bf1[40] = hbtf(cospi59, bf0[40], -cospi5, bf0[55])
	bf1[41] = hbtf(cospi27, bf0[41], -cospi37, bf0[54])
	bf1[42] = hbtf(cospi43, bf0[42], -cospi21, bf0[53])
	bf1[43] = hbtf(cospi11, bf0[43], -cospi53, bf0[52])
	bf1[44] = hbtf(cospi51, bf0[44], -cospi13, bf0[51])
	bf1[45] = hbtf(cospi19, bf0[45], -cospi45, bf0[50])
	bf1[46] = hbtf(cospi35, bf0[46], -cospi29, bf0[49])
	bf1[47] = hbtf(cospi3, bf0[47], -cospi61, bf0[48])
	bf1[48] = hbtf(cospi61, bf0[47], cospi3, bf0[48])
	bf1[49] = hbtf(cospi29, bf0[46], cospi35, bf0[49])
	bf1[50] = hbtf(cospi45, bf0[45], cospi19, bf0[50])
	bf1[51] = hbtf(cospi13, bf0[44], cospi51, bf0[51])
	bf1[52] = hbtf(cospi53, bf0[43], cospi11, bf0[52])
	bf1[53] = hbtf(cospi21, bf0[42], cospi43, bf0[53])
	bf1[54] = hbtf(cospi37, bf0[41], cospi27, bf0[54])
	bf1[55] = hbtf(cospi5, bf0[40], cospi59, bf0[55])
	bf1[56] = hbtf(cospi57, bf0[39], cospi7, bf0[56])
	bf1[57] = hbtf(cospi25, bf0[38], cospi39, bf0[57])
	bf1[58] = hbtf(cospi41, bf0[37], cospi23, bf0[58])
	bf1[59] = hbtf(cospi9, bf0[36], cospi55, bf0[59])
	bf1[60] = hbtf(cospi49, bf0[35], cospi15, bf0[60])
	bf1[61] = hbtf(cospi17, bf0[34], cospi47, bf0[61])
	bf1[62] = hbtf(cospi33, bf0[33], cospi31, bf0[62])
	bf1[63] = hbtf(cospi1, bf0[32], cospi63, bf0[63])

	// stage 3
	bf0 = bf1
	bf1[16] = hbtf(cospi62, bf0[16], -cospi2, bf0[31])
	bf1[17] = hbtf(cospi30, bf0[17], -cospi34, bf0[30])
	bf1[18] = hbtf(cospi46, bf0[18], -cospi18, bf0[29])
	bf1[19] = hbtf(cospi14, bf0[19], -cospi50, bf0[28])
	bf1[20] = hbtf(cospi54, bf0[20], -cospi10, bf0[27])
	bf1[21] = hbtf(cospi22, bf0[21], -cospi42, bf0[26])
	bf1[22] = hbtf(cospi38, bf0[22], -cospi26, bf0[25])
	bf1[23] = hbtf(cospi6, bf0[23], -cospi58, bf0[24])
	bf1[24] = hbtf(cospi58, bf0[23], cospi6, bf0[24])
	bf1[25] = hbtf(cospi26, bf0[22], cospi38, bf0[25])
	bf1[26] = hbtf(cospi42, bf0[21], cospi22, bf0[26])
	bf1[27] = hbtf(cospi10, bf0[20], cospi54, bf0[27])
	bf1[28] = hbtf(cospi50, bf0[19], cospi14, bf0[28])
	bf1[29] = hbtf(cospi18, bf0[18], cospi46, bf0[29])
	bf1[30] = hbtf(cospi34, bf0[17], cospi30, bf0[30])
	bf1[31] = hbtf(cospi2, bf0[16], cospi62, bf0[31])
	bf1[32] = int64(clamp(bf0[32] + bf0[33]))
	bf1[33] = int64(clamp(bf0[32] - bf0[33]))
	bf1[34] = int64(clamp(-bf0[34] + bf0[35]))
	bf1[35] = int64(clamp(bf0[34] + bf0[35]))
	bf1[36] = int64(clamp(bf0[36] + bf0[37]))
	bf1[37] = int64(clamp(bf0[36] - bf0[37]))
	bf1[38] = int64(clamp(-bf0[38] + bf0[39]))
	bf1[39] = int64(clamp(bf0[38] + bf0[39]))
	bf1[40] = int64(clamp(bf0[40] + bf0[41]))
	bf1[41] = int64(clamp(bf0[40] - bf0[41]))
	bf1[42] = int64(clamp(-bf0[42] + bf0[43]))
	bf1[43] = int64(clamp(bf0[42] + bf0[43]))
	bf1[44] = int64(clamp(bf0[44] + bf0[45]))
	bf1[45] = int64(clamp(bf0[44] - bf0[45]))
	bf1[46] = int64(clamp(-bf0[46] + bf0[47]))
	bf1[47] = int64(clamp(bf0[46] + bf0[47]))
	bf1[48] = int64(clamp(bf0[48] + bf0[49]))
	bf1[49] = int64(clamp(bf0[48] - bf0[49]))
	bf1[50] = int64(clamp(-bf0[50] + bf0[51]))
	bf1[51] = int64(clamp(bf0[50] + bf0[51]))
	bf1[52] = int64(clamp(bf0[52] + bf0[53]))
	bf1[53] = int64(clamp(bf0[52] - bf0[53]))
	bf1[54] = int64(clamp(-bf0[54] + bf0[55]))
	bf1[55] = int64(clamp(bf0[54] + bf0[55]))
	bf1[56] = int64(clamp(bf0[56] + bf0[57]))
	bf1[57] = int64(clamp(bf0[56] - bf0[57]))
	bf1[58] = int64(clamp(-bf0[58] + bf0[59]))
	bf1[59] = int64(clamp(bf0[58] + bf0[59]))
	bf1[60] = int64(clamp(bf0[60] + bf0[61]))
	bf1[61] = int64(clamp(bf0[60] - bf0[61]))
	bf1[62] = int64(clamp(-bf0[62] + bf0[63]))
	bf1[63] = int64(clamp(bf0[62] + bf0[63]))

	// stage 4
	bf0 = bf1
	bf1[8] = hbtf(cospi60, bf0[8], -cospi4, bf0[15])
	bf1[9] = hbtf(cospi28, bf0[9], -cospi36, bf0[14])
	bf1[10] = hbtf(cospi44, bf0[10], -cospi20, bf0[13])
	bf1[11] = hbtf(cospi12, bf0[11], -cospi52, bf0[12])
	bf1[12] = hbtf(cospi52, bf0[11], cospi12, bf0[12])
	bf1[13] = hbtf(cospi20, bf0[10], cospi44, bf0[13])
	bf1[14] = hbtf(cospi36, bf0[9], cospi28, bf0[14])
	bf1[15] = hbtf(cospi4, bf0[8], cospi60, bf0[15])
	bf1[16] = int64(clamp(bf0[16] + bf0[17]))
	bf1[17] = int64(clamp(bf0[16] - bf0[17]))
	bf1[18] = int64(clamp(-bf0[18] + bf0[19]))
	bf1[19] = int64(clamp(bf0[18] + bf0[19]))
	bf1[20] = int64(clamp(bf0[20] + bf0[21]))
	bf1[21] = int64(clamp(bf0[20] - bf0[21]))
	bf1[22] = int64(clamp(-bf0[22] + bf0[23]))
	bf1[23] = int64(clamp(bf0[22] + bf0[23]))
	bf1[24] = int64(clamp(bf0[24] + bf0[25]))
	bf1[25] = int64(clamp(bf0[24] - bf0[25]))
	bf1[26] = int64(clamp(-bf0[26] + bf0[27]))
	bf1[27] = int64(clamp(bf0[26] + bf0[27]))
	bf1[28] = int64(clamp(bf0[28] + bf0[29]))
	bf1[29] = int64(clamp(bf0[28] - bf0[29]))
	bf1[30] = int64(clamp(-bf0[30] + bf0[31]))
	bf1[31] = int64(clamp(bf0[30] + bf0[31]))
	bf1[33] = hbtf(-cospi4, bf0[33], cospi60, bf0[62])
	bf1[34] = hbtf(-cospi60, bf0[34], -cospi4, bf0[61])
	bf1[37] = hbtf(-cospi36, bf0[37], cospi28, bf0[58])
	bf1[38] = hbtf(-cospi28, bf0[38], -cospi36, bf0[57])
	bf1[41] = hbtf(-cospi20, bf0[41], cospi44, bf0[54])
	bf1[42] = hbtf(-cospi44, bf0[42], -cospi20, bf0[53])
	bf1[45] = hbtf(-cospi52, bf0[45], cospi12, bf0[50])
	bf1[46] = hbtf(-cospi12, bf0[46], -cospi52, bf0[49])
	bf1[49] = hbtf(-cospi52, bf0[46], cospi12, bf0[49])
	bf1[50] = hbtf(cospi12, bf0[45], cospi52, bf0[50])
	bf1[53] = hbtf(-cospi20, bf0[42], cospi44, bf0[53])
	bf1[54] = hbtf(cospi44, bf0[41], cospi20, bf0[54])
	bf1[57] = hbtf(-cospi36, bf0[38], cospi28, bf0[57])
	bf1[58] = hbtf(cospi28, bf0[37], cospi36, bf0[58])
	bf1[61] = hbtf(-cospi4, bf0[34], cospi60, bf0[61])
	bf1[62] = hbtf(cospi60, bf0[33], cospi4, bf0[62])

	// stage 5
	bf0 = bf1
	bf1[4] = hbtf(cospi56, bf0[4], -cospi8, bf0[7])
	bf1[5] = hbtf(cospi24, bf0[5], -cospi40, bf0[6])
	bf1[6] = hbtf(cospi40, bf0[5], cospi24, bf0[6])
	bf1[7] = hbtf(cospi8, bf0[4], cospi56, bf0[7])
	bf1[8] = int64(clamp(bf0[8] + bf0[9]))
	bf1[9] = int64(clamp(bf0[8] - bf0[9]))
	bf1[10] = int64(clamp(-bf0[10] + bf0[11]))
	bf1[11] = int64(clamp(bf0[10] + bf0[11]))
	bf1[12] = int64(clamp(bf0[12] + bf0[13]))
	bf1[13] = int64(clamp(bf0[12] - bf0[13]))
	bf1[14] = int64(clamp(-bf0[14] + bf0[15]))
	bf1[15] = int64(clamp(bf0[14] + bf0[15]))
	bf1[17] = hbtf(-cospi8, bf0[17], cospi56, bf0[30])
	bf1[18] = hbtf(-cospi56, bf0[18], -cospi8, bf0[29])
	bf1[21] = hbtf(-cospi40, bf0[21], cospi24, bf0[26])
	bf1[22] = hbtf(-cospi24, bf0[22], -cospi40, bf0[25])
	bf1[25] = hbtf(-cospi40, bf0[22], cospi24, bf0[25])
	bf1[26] = hbtf(cospi24, bf0[21], cospi40, bf0[26])
	bf1[29] = hbtf(-cospi8, bf0[18], cospi56, bf0[29])
	bf1[30] = hbtf(cospi56, bf0[17], cospi8, bf0[30])
	bf1[32] = int64(clamp(bf0[32] + bf0[35]))
	bf1[33] = int64(clamp(bf0[33] + bf0[34]))
	bf1[34] = int64(clamp(bf0[33] - bf0[34]))
	bf1[35] = int64(clamp(bf0[32] - bf0[35]))
	bf1[36] = int64(clamp(-bf0[36] + bf0[39]))
	bf1[37] = int64(clamp(-bf0[37] + bf0[38]))
	bf1[38] = int64(clamp(bf0[37] + bf0[38]))
	bf1[39] = int64(clamp(bf0[36] + bf0[39]))
	bf1[40] = int64(clamp(bf0[40] + bf0[43]))
	bf1[41] = int64(clamp(bf0[41] + bf0[42]))
	bf1[42] = int64(clamp(bf0[41] - bf0[42]))
	bf1[43] = int64(clamp(bf0[40] - bf0[43]))
	bf1[44] = int64(clamp(-bf0[44] + bf0[47]))
	bf1[45] = int64(clamp(-bf0[45] + bf0[46]))
	bf1[46] = int64(clamp(bf0[45] + bf0[46]))
	bf1[47] = int64(clamp(bf0[44] + bf0[47]))
	bf1[48] = int64(clamp(bf0[48] + bf0[51]))
	bf1[49] = int64(clamp(bf0[49] + bf0[50]))
	bf1[50] = int64(clamp(bf0[49] - bf0[50]))
	bf1[51] = int64(clamp(bf0[48] - bf0[51]))
	bf1[52] = int64(clamp(-bf0[52] + bf0[55]))
	bf1[53] = int64(clamp(-bf0[53] + bf0[54]))
	bf1[54] = int64(clamp(bf0[53] + bf0[54]))
	bf1[55] = int64(clamp(bf0[52] + bf0[55]))
	bf1[56] = int64(clamp(bf0[56] + bf0[59]))
	bf1[57] = int64(clamp(bf0[57] + bf0[58]))
	bf1[58] = int64(clamp(bf0[57] - bf0[58]))
	bf1[59] = int64(clamp(bf0[56] - bf0[59]))
	bf1[60] = int64(clamp(-bf0[60] + bf0[63]))
	bf1[61] = int64(clamp(-bf0[61] + bf0[62]))
	bf1[62] = int64(clamp(bf0[61] + bf0[62]))
	bf1[63] = int64(clamp(bf0[60] + bf0[63]))

	// stage 6
	bf0 = bf1
	bf1[0] = hbtf(cospi32, bf0[0], cospi32, bf0[1])
	bf1[1] = hbtf(cospi32, bf0[0], -cospi32, bf0[1])
	bf1[2] = hbtf(cospi48, bf0[2], -cospi16, bf0[3])
	bf1[3] = hbtf(cospi16, bf0[2], cospi48, bf0[3])
	bf1[4] = int64(clamp(bf0[4] + bf0[5]))
	bf1[5] = int64(clamp(bf0[4] - bf0[5]))
	bf1[6] = int64(clamp(-bf0[6] + bf0[7]))
	bf1[7] = int64(clamp(bf0[6] + bf0[7]))
	bf1[9] = hbtf(-cospi16, bf0[9], cospi48, bf0[14])
	bf1[10] = hbtf(-cospi48, bf0[10], -cospi16, bf0[13])
	bf1[13] = hbtf(-cospi16, bf0[10], cospi48, bf0[13])
	bf1[14] = hbtf(cospi48, bf0[9], cospi16, bf0[14])
	bf1[16] = int64(clamp(bf0[16] + bf0[19]))
	bf1[17] = int64(clamp(bf0[17] + bf0[18]))
	bf1[18] = int64(clamp(bf0[17] - bf0[18]))
	bf1[19] = int64(clamp(bf0[16] - bf0[19]))
	bf1[20] = int64(clamp(-bf0[20] + bf0[23]))
	bf1[21] = int64(clamp(-bf0[21] + bf0[22]))
	bf1[22] = int64(clamp(bf0[21] + bf0[22]))
	bf1[23] = int64(clamp(bf0[20] + bf0[23]))
	bf1[24] = int64(clamp(bf0[24] + bf0[27]))
	bf1[25] = int64(clamp(bf0[25] + bf0[26]))
	bf1[26] = int64(clamp(bf0[25] - bf0[26]))
	bf1[27] = int64(clamp(bf0[24] - bf0[27]))
	bf1[28] = int64(clamp(-bf0[28] + bf0[31]))
	bf1[29] = int64(clamp(-bf0[29] + bf0[30]))
	bf1[30] = int64(clamp(bf0[29] + bf0[30]))
	bf1[31] = int64(clamp(bf0[28] + bf0[31]))
	bf1[34] = hbtf(-cospi8, bf0[34], cospi56, bf0[61])
	bf1[35] = hbtf(-cospi8, bf0[35], cospi56, bf0[60])
	bf1[36] = hbtf(-cospi56, bf0[36], -cospi8, bf0[59])
	bf1[37] = hbtf(-cospi56, bf0[37], -cospi8, bf0[58])
	bf1[42] = hbtf(-cospi40, bf0[42], cospi24, bf0[53])
	bf1[43] = hbtf(-cospi40, bf0[43], cospi24, bf0[52])
	bf1[44] = hbtf(-cospi24, bf0[44], -cospi40, bf0[51])
	bf1[45] = hbtf(-cospi24, bf0[45], -cospi40, bf0[50])
	bf1[50] = hbtf(-cospi40, bf0[45], cospi24, bf0[50])
	bf1[51] = hbtf(-cospi40, bf0[44], cospi24, bf0[51])
	bf1[52] = hbtf(cospi24, bf0[43], cospi40, bf0[52])
	bf1[53] = hbtf(cospi24, bf0[42], cospi40, bf0[53])
	bf1[58] = hbtf(-cospi8, bf0[37], cospi56, bf0[58])
	bf1[59] = hbtf(-cospi8, bf0[36], cospi56, bf0[59])
	bf1[60] = hbtf(cospi56, bf0[35], cospi8, bf0[60])
	bf1[61] = hbtf(cospi56, bf0[34], cospi8, bf0[61])

	// stage 7
	bf0 = bf1
	bf1[0] = int64(clamp(bf0[0] + bf0[3]))
	bf1[1] = int64(clamp(bf0[1] + bf0[2]))
	bf1[2] = int64(clamp(bf0[1] - bf0[2]))
	bf1[3] = int64(clamp(bf0[0] - bf0[3]))
	bf1[5] = hbtf(-cospi32, bf0[5], cospi32, bf0[6])
	bf1[6] = hbtf(cospi32, bf0[5], cospi32, bf0[6])
	bf1[8] = int64(clamp(bf0[8] + bf0[11]))
	bf1[9] = int64(clamp(bf0[9] + bf0[10]))
	bf1[10] = int64(clamp(bf0[9] - bf0[10]))
	bf1[11] = int64(clamp(bf0[8] - bf0[11]))
	bf1[12] = int64(clamp(-bf0[12] + bf0[15]))
	bf1[13] = int64(clamp(-bf0[13] + bf0[14]))
	bf1[14] = int64(clamp(bf0[13] + bf0[14]))
	bf1[15] = int64(clamp(bf0[12] + bf0[15]))
	bf1[18] = hbtf(-cospi16, bf0[18], cospi48, bf0[29])
	bf1[19] = hbtf(-cospi16, bf0[19], cospi48, bf0[28])
	bf1[20] = hbtf(-cospi48, bf0[20], -cospi16, bf0[27])
	bf1[21] = hbtf(-cospi48, bf0[21], -cospi16, bf0[26])
	bf1[26] = hbtf(-cospi16, bf0[21], cospi48, bf0[26])
	bf1[27] = hbtf(-cospi16, bf0[20], cospi48, bf0[27])
	bf1[28] = hbtf(cospi48, bf0[19], cospi16, bf0[28])
	bf1[29] = hbtf(cospi48, bf0[18], cospi16, bf0[29])
	bf1[32] = int64(clamp(bf0[32] + bf0[39]))
	bf1[33] = int64(clamp(bf0[33] + bf0[38]))
	bf1[34] = int64(clamp(bf0[34] + bf0[37]))
	bf1[35] = int64(clamp(bf0[35] + bf0[36]))
	bf1[36] = int64(clamp(bf0[35] - bf0[36]))
	bf1[37] = int64(clamp(bf0[34] - bf0[37]))
	bf1[38] = int64(clamp(bf0[33] - bf0[38]))
	bf1[39] = int64(clamp(bf0[32] - bf0[39]))
	bf1[40] = int64(clamp(-bf0[40] + bf0[47]))
	bf1[41] = int64(clamp(-bf0[41] + bf0[46]))
	bf1[42] = int64(clamp(-bf0[42] + bf0[45]))
	bf1[43] = int64(clamp(-bf0[43] + bf0[44]))
	bf1[44] = int64(clamp(bf0[43] + bf0[44]))
	bf1[45] = int64(clamp(bf0[42] + bf0[45]))
	bf1[46] = int64(clamp(bf0[41] + bf0[46]))
	bf1[47] = int64(clamp(bf0[40] + bf0[47]))
	bf1[48] = int64(clamp(bf0[48] + bf0[55]))
	bf1[49] = int64(clamp(bf0[49] + bf0[54]))
	bf1[50] = int64(clamp(bf0[50] + bf0[53]))
	bf1[51] = int64(clamp(bf0[51] + bf0[52]))
	bf1[52] = int64(clamp(bf0[51] - bf0[52]))
	bf1[53] = int64(clamp(bf0[50] - bf0[53]))
	bf1[54] = int64(clamp(bf0[49] - bf0[54]))
	bf1[55] = int64(clamp(bf0[48] - bf0[55]))
	bf1[56] = int64(clamp(-bf0[56] + bf0[63]))
	bf1[57] = int64(clamp(-bf0[57] + bf0[62]))
	bf1[58] = int64(clamp(-bf0[58] + bf0[61]))
	bf1[59] = int64(clamp(-bf0[59] + bf0[60]))
	bf1[60] = int64(clamp(bf0[59] + bf0[60]))
	bf1[61] = int64(clamp(bf0[58] + bf0[61]))
	bf1[62] = int64(clamp(bf0[57] + bf0[62]))
	bf1[63] = int64(clamp(bf0[56] + bf0[63]))

	// stage 8
	bf0 = bf1
	bf1[0] = int64(clamp(bf0[0] + bf0[7]))
	bf1[1] = int64(clamp(bf0[1] + bf0[6]))
	bf1[2] = int64(clamp(bf0[2] + bf0[5]))
	bf1[3] = int64(clamp(bf0[3] + bf0[4]))
	bf1[4] = int64(clamp(bf0[3] - bf0[4]))
	bf1[5] = int64(clamp(bf0[2] - bf0[5]))
	bf1[6] = int64(clamp(bf0[1] - bf0[6]))
	bf1[7] = int64(clamp(bf0[0] - bf0[7]))
	bf1[10] = hbtf(-cospi32, bf0[10], cospi32, bf0[13])
	bf1[11] = hbtf(-cospi32, bf0[11], cospi32, bf0[12])
	bf1[12] = hbtf(cospi32, bf0[11], cospi32, bf0[12])
	bf1[13] = hbtf(cospi32, bf0[10], cospi32, bf0[13])
	bf1[16] = int64(clamp(bf0[16] + bf0[23]))
	bf1[17] = int64(clamp(bf0[17] + bf0[22]))
	bf1[18] = int64(clamp(bf0[18] + bf0[21]))
	bf1[19] = int64(clamp(bf0[19] + bf0[20]))
	bf1[20] = int64(clamp(bf0[19] - bf0[20]))
	bf1[21] = int64(clamp(bf0[18] - bf0[21]))
	bf1[22] = int64(clamp(bf0[17] - bf0[22]))
	bf1[23] = int64(clamp(bf0[16] - bf0[23]))
	bf1[24] = int64(clamp(-bf0[24] + bf0[31]))
	bf1[25] = int64(clamp(-bf0[25] + bf0[30]))
	bf1[26] = int64(clamp(-bf0[26] + bf0[29]))
	bf1[27] = int64(clamp(-bf0[27] + bf0[28]))
	bf1[28] = int64(clamp(bf0[27] + bf0[28]))
	bf1[29] = int64(clamp(bf0[26] + bf0[29]))
	bf1[30] = int64(clamp(bf0[25] + bf0[30]))
	bf1[31] = int64(clamp(bf0[24] + bf0[31]))
	bf1[36] = hbtf(-cospi16, bf0[36], cospi48, bf0[59])
	bf1[37] = hbtf(-cospi16, bf0[37], cospi48, bf0[58])
	bf1[38] = hbtf(-cospi16, bf0[38], cospi48, bf0[57])
	bf1[39] = hbtf(-cospi16, bf0[39], cospi48, bf0[56])
	bf1[40] = hbtf(-cospi48, bf0[40], -cospi16, bf0[55])
	bf1[41] = hbtf(-cospi48, bf0[41], -cospi16, bf0[54])
	bf1[42] = hbtf(-cospi48, bf0[42], -cospi16, bf0[53])
	bf1[43] = hbtf(-cospi48, bf0[43], -cospi16, bf0[52])
	bf1[52] = hbtf(-cospi16, bf0[43], cospi48, bf0[52])
	bf1[53] = hbtf(-cospi16, bf0[42], cospi48, bf0[53])
	bf1[54] = hbtf(-cospi16, bf0[41], cospi48, bf0[54])
	bf1[55] = hbtf(-cospi16, bf0[40], cospi48, bf0[55])
	bf1[56] = hbtf(cospi48, bf0[39], cospi16, bf0[56])
	bf1[57] = hbtf(cospi48, bf0[38], cospi16, bf0[57])
	bf1[58] = hbtf(cospi48, bf0[37], cospi16, bf0[58])
	bf1[59] = hbtf(cospi48, bf0[36], cospi16, bf0[59])

	// stage 9
	bf0 = bf1
	bf1[0] = int64(clamp(bf0[0] + bf0[15]))
	bf1[1] = int64(clamp(bf0[1] + bf0[14]))
	bf1[2] = int64(clamp(bf0[2] + bf0[13]))
	bf1[3] = int64(clamp(bf0[3] + bf0[12]))
	bf1[4] = int64(clamp(bf0[4] + bf0[11]))
	bf1[5] = int64(clamp(bf0[5] + bf0[10]))
	bf1[6] = int64(clamp(bf0[6] + bf0[9]))
	bf1[7] = int64(clamp(bf0[7] + bf0[8]))
	bf1[8] = int64(clamp(bf0[7] - bf0[8]))
	bf1[9] = int64(clamp(bf0[6] - bf0[9]))
	bf1[10] = int64(clamp(bf0[5] - bf0[10]))
	bf1[11] = int64(clamp(bf0[4] - bf0[11]))
	bf1[12] = int64(clamp(bf0[3] - bf0[12]))
	bf1[13] = int64(clamp(bf0[2] - bf0[13]))
	bf1[14] = int64(clamp(bf0[1] - bf0[14]))
	bf1[15] = int64(clamp(bf0[0] - bf0[15]))
	bf1[20] = hbtf(-cospi32, bf0[20], cospi32, bf0[27])
	bf1[21] = hbtf(-cospi32, bf0[21], cospi32, bf0[26])
	bf1[22] = hbtf(-cospi32, bf0[22], cospi32, bf0[25])
	bf1[23] = hbtf(-cospi32, bf0[23], cospi32, bf0[24])
	bf1[24] = hbtf(cospi32, bf0[23], cospi32, bf0[24])
	bf1[25] = hbtf(cospi32, bf0[22], cospi32, bf0[25])
	bf1[26] = hbtf(cospi32, bf0[21], cospi32, bf0[26])
	bf1[27] = hbtf(cospi32, bf0[20], cospi32, bf0[27])
	bf1[32] = int64(clamp(bf0[32] + bf0[47]))
	bf1[33] = int64(clamp(bf0[33] + bf0[46]))
	bf1[34] = int64(clamp(bf0[34] + bf0[45]))
	bf1[35] = int64(clamp(bf0[35] + bf0[44]))
	bf1[36] = int64(clamp(bf0[36] + bf0[43]))
	bf1[37] = int64(clamp(bf0[37] + bf0[42]))
	bf1[38] = int64(clamp(bf0[38] + bf0[41]))
	bf1[39] = int64(clamp(bf0[39] + bf0[40]))
	bf1[40] = int64(clamp(bf0[39] - bf0[40]))
	bf1[41] = int64(clamp(bf0[38] - bf0[41]))
	bf1[42] = int64(clamp(bf0[37] - bf0[42]))
	bf1[43] = int64(clamp(bf0[36] - bf0[43]))
	bf1[44] = int64(clamp(bf0[35] - bf0[44]))
	bf1[45] = int64(clamp(bf0[34] - bf0[45]))
	bf1[46] = int64(clamp(bf0[33] - bf0[46]))
	bf1[47] = int64(clamp(bf0[32] - bf0[47]))
	bf1[48] = int64(clamp(-bf0[48] + bf0[63]))
	bf1[49] = int64(clamp(-bf0[49] + bf0[62]))
	bf1[50] = int64(clamp(-bf0[50] + bf0[61]))
	bf1[51] = int64(clamp(-bf0[51] + bf0[60]))
	bf1[52] = int64(clamp(-bf0[52] + bf0[59]))
	bf1[53] = int64(clamp(-bf0[53] + bf0[58]))
	bf1[54] = int64(clamp(-bf0[54] + bf0[57]))
	bf1[55] = int64(clamp(-bf0[55] + bf0[56]))
	bf1[56] = int64(clamp(bf0[55] + bf0[56]))
	bf1[57] = int64(clamp(bf0[54] + bf0[57]))
	bf1[58] = int64(clamp(bf0[53] + bf0[58]))
	bf1[59] = int64(clamp(bf0[52] + bf0[59]))
	bf1[60] = int64(clamp(bf0[51] + bf0[60]))
	bf1[61] = int64(clamp(bf0[50] + bf0[61]))
	bf1[62] = int64(clamp(bf0[49] + bf0[62]))
	bf1[63] = int64(clamp(bf0[48] + bf0[63]))

	// stage 10
	bf0 = bf1
	bf1[0] = int64(clamp(bf0[0] + bf0[31]))
	bf1[1] = int64(clamp(bf0[1] + bf0[30]))
	bf1[2] = int64(clamp(bf0[2] + bf0[29]))
	bf1[3] = int64(clamp(bf0[3] + bf0[28]))
	bf1[4] = int64(clamp(bf0[4] + bf0[27]))
	bf1[5] = int64(clamp(bf0[5] + bf0[26]))
	bf1[6] = int64(clamp(bf0[6] + bf0[25]))
	bf1[7] = int64(clamp(bf0[7] + bf0[24]))
	bf1[8] = int64(clamp(bf0[8] + bf0[23]))
	bf1[9] = int64(clamp(bf0[9] + bf0[22]))
	bf1[10] = int64(clamp(bf0[10] + bf0[21]))
	bf1[11] = int64(clamp(bf0[11] + bf0[20]))
	bf1[12] = int64(clamp(bf0[12] + bf0[19]))
	bf1[13] = int64(clamp(bf0[13] + bf0[18]))
	bf1[14] = int64(clamp(bf0[14] + bf0[17]))
	bf1[15] = int64(clamp(bf0[15] + bf0[16]))
	bf1[16] = int64(clamp(bf0[15] - bf0[16]))
	bf1[17] = int64(clamp(bf0[14] - bf0[17]))
	bf1[18] = int64(clamp(bf0[13] - bf0[18]))
	bf1[19] = int64(clamp(bf0[12] - bf0[19]))
	bf1[20] = int64(clamp(bf0[11] - bf0[20]))
	bf1[21] = int64(clamp(bf0[10] - bf0[21]))
	bf1[22] = int64(clamp(bf0[9] - bf0[22]))
	bf1[23] = int64(clamp(bf0[8] - bf0[23]))
	bf1[24] = int64(clamp(bf0[7] - bf0[24]))
	bf1[25] = int64(clamp(bf0[6] - bf0[25]))
	bf1[26] = int64(clamp(bf0[5] - bf0[26]))
	bf1[27] = int64(clamp(bf0[4] - bf0[27]))
	bf1[28] = int64(clamp(bf0[3] - bf0[28]))
	bf1[29] = int64(clamp(bf0[2] - bf0[29]))
	bf1[30] = int64(clamp(bf0[1] - bf0[30]))
	bf1[31] = int64(clamp(bf0[0] - bf0[31]))
	bf1[40] = hbtf(-cospi32, bf0[40], cospi32, bf0[55])
	bf1[41] = hbtf(-cospi32, bf0[41], cospi32, bf0[54])
	bf1[42] = hbtf(-cospi32, bf0[42], cospi32, bf0[53])
	bf1[43] = hbtf(-cospi32, bf0[43], cospi32, bf0[52])
	bf1[44] = hbtf(-cospi32, bf0[44], cospi32, bf0[51])
	bf1[45] = hbtf(-cospi32, bf0[45], cospi32, bf0[50])
	bf1[46] = hbtf(-cospi32, bf0[46], cospi32, bf0[49])
	bf1[47] = hbtf(-cospi32, bf0[47], cospi32, bf0[48])
	bf1[48] = hbtf(cospi32, bf0[47], cospi32, bf0[48])
	bf1[49] = hbtf(cospi32, bf0[46], cospi32, bf0[49])
	bf1[50] = hbtf(cospi32, bf0[45], cospi32, bf0[50])
	bf1[51] = hbtf(cospi32, bf0[44], cospi32, bf0[51])
	bf1[52] = hbtf(cospi32, bf0[43], cospi32, bf0[52])
	bf1[53] = hbtf(cospi32, bf0[42], cospi32, bf0[53])
	bf1[54] = hbtf(cospi32, bf0[41], cospi32, bf0[54])
	bf1[55] = hbtf(cospi32, bf0[40], cospi32, bf0[55])

	// stage 11 — final outputs back to c[i*stride]
	bf0 = bf1
	c[0*stride] = clamp(bf0[0] + bf0[63])
	c[1*stride] = clamp(bf0[1] + bf0[62])
	c[2*stride] = clamp(bf0[2] + bf0[61])
	c[3*stride] = clamp(bf0[3] + bf0[60])
	c[4*stride] = clamp(bf0[4] + bf0[59])
	c[5*stride] = clamp(bf0[5] + bf0[58])
	c[6*stride] = clamp(bf0[6] + bf0[57])
	c[7*stride] = clamp(bf0[7] + bf0[56])
	c[8*stride] = clamp(bf0[8] + bf0[55])
	c[9*stride] = clamp(bf0[9] + bf0[54])
	c[10*stride] = clamp(bf0[10] + bf0[53])
	c[11*stride] = clamp(bf0[11] + bf0[52])
	c[12*stride] = clamp(bf0[12] + bf0[51])
	c[13*stride] = clamp(bf0[13] + bf0[50])
	c[14*stride] = clamp(bf0[14] + bf0[49])
	c[15*stride] = clamp(bf0[15] + bf0[48])
	c[16*stride] = clamp(bf0[16] + bf0[47])
	c[17*stride] = clamp(bf0[17] + bf0[46])
	c[18*stride] = clamp(bf0[18] + bf0[45])
	c[19*stride] = clamp(bf0[19] + bf0[44])
	c[20*stride] = clamp(bf0[20] + bf0[43])
	c[21*stride] = clamp(bf0[21] + bf0[42])
	c[22*stride] = clamp(bf0[22] + bf0[41])
	c[23*stride] = clamp(bf0[23] + bf0[40])
	c[24*stride] = clamp(bf0[24] + bf0[39])
	c[25*stride] = clamp(bf0[25] + bf0[38])
	c[26*stride] = clamp(bf0[26] + bf0[37])
	c[27*stride] = clamp(bf0[27] + bf0[36])
	c[28*stride] = clamp(bf0[28] + bf0[35])
	c[29*stride] = clamp(bf0[29] + bf0[34])
	c[30*stride] = clamp(bf0[30] + bf0[33])
	c[31*stride] = clamp(bf0[31] + bf0[32])
	c[32*stride] = clamp(bf0[31] - bf0[32])
	c[33*stride] = clamp(bf0[30] - bf0[33])
	c[34*stride] = clamp(bf0[29] - bf0[34])
	c[35*stride] = clamp(bf0[28] - bf0[35])
	c[36*stride] = clamp(bf0[27] - bf0[36])
	c[37*stride] = clamp(bf0[26] - bf0[37])
	c[38*stride] = clamp(bf0[25] - bf0[38])
	c[39*stride] = clamp(bf0[24] - bf0[39])
	c[40*stride] = clamp(bf0[23] - bf0[40])
	c[41*stride] = clamp(bf0[22] - bf0[41])
	c[42*stride] = clamp(bf0[21] - bf0[42])
	c[43*stride] = clamp(bf0[20] - bf0[43])
	c[44*stride] = clamp(bf0[19] - bf0[44])
	c[45*stride] = clamp(bf0[18] - bf0[45])
	c[46*stride] = clamp(bf0[17] - bf0[46])
	c[47*stride] = clamp(bf0[16] - bf0[47])
	c[48*stride] = clamp(bf0[15] - bf0[48])
	c[49*stride] = clamp(bf0[14] - bf0[49])
	c[50*stride] = clamp(bf0[13] - bf0[50])
	c[51*stride] = clamp(bf0[12] - bf0[51])
	c[52*stride] = clamp(bf0[11] - bf0[52])
	c[53*stride] = clamp(bf0[10] - bf0[53])
	c[54*stride] = clamp(bf0[9] - bf0[54])
	c[55*stride] = clamp(bf0[8] - bf0[55])
	c[56*stride] = clamp(bf0[7] - bf0[56])
	c[57*stride] = clamp(bf0[6] - bf0[57])
	c[58*stride] = clamp(bf0[5] - bf0[58])
	c[59*stride] = clamp(bf0[4] - bf0[59])
	c[60*stride] = clamp(bf0[3] - bf0[60])
	c[61*stride] = clamp(bf0[2] - bf0[61])
	c[62*stride] = clamp(bf0[1] - bf0[62])
	c[63*stride] = clamp(bf0[0] - bf0[63])
}

// txElem is the transform scratch element type. The int16 column pipeline
// (8/10-bit, dav1d-style) reuses the exact same scalar kernels as the int32
// pipeline; only the storage width changes. Intermediate math stays int64
// (reference-equivalent), and every stored value is clamped into T's range.
type txElem interface{ ~int16 | ~int32 }

// clipRangeT clamps v to [min,max] and narrows to the element type T.
func clipRangeT[T txElem](v int64, lo int32, hi int32) T {
	return T(min(max(v, int64(lo)), int64(hi)))
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
