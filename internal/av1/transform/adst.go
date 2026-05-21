package transform

const (
	adst4Size  = 4
	adst8Size  = 8
	adst16Size = 16
)

func adstLengthSupported(length int) bool {
	switch length {
	case adst4Size, adst8Size, adst16Size:
		return true
	default:
		return false
	}
}

func inverseADST1D(c []int32, stride int, length int, min int32, max int32) {
	switch length {
	case adst4Size:
		inverseADST4(c, stride)
	case adst8Size:
		inverseADST8(c, stride, min, max)
	case adst16Size:
		inverseADST16(c, stride, min, max)
	}
}

func inverseFlipADST1D(c []int32, stride int, length int, min int32, max int32) {
	switch length {
	case adst4Size:
		inverseADST4To(c, stride, c, (adst4Size-1)*stride, -stride)
	case adst8Size:
		inverseADST8To(c, stride, c, (adst8Size-1)*stride, -stride, min, max)
	case adst16Size:
		inverseADST16To(c, stride, c, (adst16Size-1)*stride, -stride, min, max)
	}
}

func inverseADST4(c []int32, stride int) {
	inverseADST4To(c, stride, c, 0, stride)
}

func inverseADST4To(in []int32, inStride int, out []int32, outOffset int, outStride int) {
	in0 := int64(in[0*inStride])
	in1 := int64(in[1*inStride])
	in2 := int64(in[2*inStride])
	in3 := int64(in[3*inStride])

	out[outOffset+0*outStride] = clipInt32(roundShift(1321*in0+(3803-4096)*in2+(2482-4096)*in3+(3344-4096)*in1, 12) + in2 + in3 + in1)
	out[outOffset+1*outStride] = clipInt32(roundShift((2482-4096)*in0-1321*in2-(3803-4096)*in3+(3344-4096)*in1, 12) + in0 - in3 + in1)
	out[outOffset+2*outStride] = clipInt32(roundShift(209*(in0-in2+in3), 8))
	out[outOffset+3*outStride] = clipInt32(roundShift((3803-4096)*in0+(2482-4096)*in2-1321*in3-(3344-4096)*in1, 12) + in0 + in2 - in1)
}

func inverseADST8(c []int32, stride int, min int32, max int32) {
	inverseADST8To(c, stride, c, 0, stride, min, max)
}

func inverseADST8To(in []int32, inStride int, out []int32, outOffset int, outStride int, min int32, max int32) {
	in0 := int64(in[0*inStride])
	in1 := int64(in[1*inStride])
	in2 := int64(in[2*inStride])
	in3 := int64(in[3*inStride])
	in4 := int64(in[4*inStride])
	in5 := int64(in[5*inStride])
	in6 := int64(in[6*inStride])
	in7 := int64(in[7*inStride])

	t0a := roundShift((4076-4096)*in7+401*in0, 12) + in7
	t1a := roundShift(401*in7-(4076-4096)*in0, 12) - in0
	t2a := roundShift((3612-4096)*in5+1931*in2, 12) + in5
	t3a := roundShift(1931*in5-(3612-4096)*in2, 12) - in2
	t4a := roundShift(1299*in3+1583*in4, 11)
	t5a := roundShift(1583*in3-1299*in4, 11)
	t6a := roundShift(1189*in1+(3920-4096)*in6, 12) + in6
	t7a := roundShift((3920-4096)*in1-1189*in6, 12) + in1

	t0 := int64(clipRange(t0a+t4a, min, max))
	t1 := int64(clipRange(t1a+t5a, min, max))
	t2 := int64(clipRange(t2a+t6a, min, max))
	t3 := int64(clipRange(t3a+t7a, min, max))
	t4 := int64(clipRange(t0a-t4a, min, max))
	t5 := int64(clipRange(t1a-t5a, min, max))
	t6 := int64(clipRange(t2a-t6a, min, max))
	t7 := int64(clipRange(t3a-t7a, min, max))

	t4a = roundShift((3784-4096)*t4+1567*t5, 12) + t4
	t5a = roundShift(1567*t4-(3784-4096)*t5, 12) - t5
	t6a = roundShift((3784-4096)*t7-1567*t6, 12) + t7
	t7a = roundShift(1567*t7+(3784-4096)*t6, 12) + t6

	out[outOffset+0*outStride] = clipRange(t0+t2, min, max)
	out[outOffset+7*outStride] = clipInt32(-int64(clipRange(t1+t3, min, max)))
	t2 = int64(clipRange(t0-t2, min, max))
	t3 = int64(clipRange(t1-t3, min, max))
	out[outOffset+1*outStride] = clipInt32(-int64(clipRange(t4a+t6a, min, max)))
	out[outOffset+6*outStride] = clipRange(t5a+t7a, min, max)
	t6 = int64(clipRange(t4a-t6a, min, max))
	t7 = int64(clipRange(t5a-t7a, min, max))

	out[outOffset+3*outStride] = clipInt32(-roundShift((t2+t3)*181, 8))
	out[outOffset+4*outStride] = clipInt32(roundShift((t2-t3)*181, 8))
	out[outOffset+2*outStride] = clipInt32(roundShift((t6+t7)*181, 8))
	out[outOffset+5*outStride] = clipInt32(-roundShift((t6-t7)*181, 8))
}

func inverseADST16(c []int32, stride int, min int32, max int32) {
	inverseADST16To(c, stride, c, 0, stride, min, max)
}

func inverseADST16To(in []int32, inStride int, out []int32, outOffset int, outStride int, min int32, max int32) {
	in0 := int64(in[0*inStride])
	in1 := int64(in[1*inStride])
	in2 := int64(in[2*inStride])
	in3 := int64(in[3*inStride])
	in4 := int64(in[4*inStride])
	in5 := int64(in[5*inStride])
	in6 := int64(in[6*inStride])
	in7 := int64(in[7*inStride])
	in8 := int64(in[8*inStride])
	in9 := int64(in[9*inStride])
	in10 := int64(in[10*inStride])
	in11 := int64(in[11*inStride])
	in12 := int64(in[12*inStride])
	in13 := int64(in[13*inStride])
	in14 := int64(in[14*inStride])
	in15 := int64(in[15*inStride])

	t0 := roundShift(in15*(4091-4096)+in0*201, 12) + in15
	t1 := roundShift(in15*201-in0*(4091-4096), 12) - in0
	t2 := roundShift(in13*(3973-4096)+in2*995, 12) + in13
	t3 := roundShift(in13*995-in2*(3973-4096), 12) - in2
	t4 := roundShift(in11*(3703-4096)+in4*1751, 12) + in11
	t5 := roundShift(in11*1751-in4*(3703-4096), 12) - in4
	t6 := roundShift(in9*1645+in6*1220, 11)
	t7 := roundShift(in9*1220-in6*1645, 11)
	t8 := roundShift(in7*2751+in8*(3035-4096), 12) + in8
	t9 := roundShift(in7*(3035-4096)-in8*2751, 12) + in7
	t10 := roundShift(in5*2106+in10*(3513-4096), 12) + in10
	t11 := roundShift(in5*(3513-4096)-in10*2106, 12) + in5
	t12 := roundShift(in3*1380+in12*(3857-4096), 12) + in12
	t13 := roundShift(in3*(3857-4096)-in12*1380, 12) + in3
	t14 := roundShift(in1*601+in14*(4052-4096), 12) + in14
	t15 := roundShift(in1*(4052-4096)-in14*601, 12) + in1

	t0a := int64(clipRange(t0+t8, min, max))
	t1a := int64(clipRange(t1+t9, min, max))
	t2a := int64(clipRange(t2+t10, min, max))
	t3a := int64(clipRange(t3+t11, min, max))
	t4a := int64(clipRange(t4+t12, min, max))
	t5a := int64(clipRange(t5+t13, min, max))
	t6a := int64(clipRange(t6+t14, min, max))
	t7a := int64(clipRange(t7+t15, min, max))
	t8a := int64(clipRange(t0-t8, min, max))
	t9a := int64(clipRange(t1-t9, min, max))
	t10a := int64(clipRange(t2-t10, min, max))
	t11a := int64(clipRange(t3-t11, min, max))
	t12a := int64(clipRange(t4-t12, min, max))
	t13a := int64(clipRange(t5-t13, min, max))
	t14a := int64(clipRange(t6-t14, min, max))
	t15a := int64(clipRange(t7-t15, min, max))

	t8 = roundShift(t8a*(4017-4096)+t9a*799, 12) + t8a
	t9 = roundShift(t8a*799-t9a*(4017-4096), 12) - t9a
	t10 = roundShift(t10a*2276+t11a*(3406-4096), 12) + t11a
	t11 = roundShift(t10a*(3406-4096)-t11a*2276, 12) + t10a
	t12 = roundShift(t13a*(4017-4096)-t12a*799, 12) + t13a
	t13 = roundShift(t13a*799+t12a*(4017-4096), 12) + t12a
	t14 = roundShift(t15a*2276-t14a*(3406-4096), 12) - t14a
	t15 = roundShift(t15a*(3406-4096)+t14a*2276, 12) + t15a

	t0 = int64(clipRange(t0a+t4a, min, max))
	t1 = int64(clipRange(t1a+t5a, min, max))
	t2 = int64(clipRange(t2a+t6a, min, max))
	t3 = int64(clipRange(t3a+t7a, min, max))
	t4 = int64(clipRange(t0a-t4a, min, max))
	t5 = int64(clipRange(t1a-t5a, min, max))
	t6 = int64(clipRange(t2a-t6a, min, max))
	t7 = int64(clipRange(t3a-t7a, min, max))
	t8a = int64(clipRange(t8+t12, min, max))
	t9a = int64(clipRange(t9+t13, min, max))
	t10a = int64(clipRange(t10+t14, min, max))
	t11a = int64(clipRange(t11+t15, min, max))
	t12a = int64(clipRange(t8-t12, min, max))
	t13a = int64(clipRange(t9-t13, min, max))
	t14a = int64(clipRange(t10-t14, min, max))
	t15a = int64(clipRange(t11-t15, min, max))

	t4a = roundShift(t4*(3784-4096)+t5*1567, 12) + t4
	t5a = roundShift(t4*1567-t5*(3784-4096), 12) - t5
	t6a = roundShift(t7*(3784-4096)-t6*1567, 12) + t7
	t7a = roundShift(t7*1567+t6*(3784-4096), 12) + t6
	t12 = roundShift(t12a*(3784-4096)+t13a*1567, 12) + t12a
	t13 = roundShift(t12a*1567-t13a*(3784-4096), 12) - t13a
	t14 = roundShift(t15a*(3784-4096)-t14a*1567, 12) + t15a
	t15 = roundShift(t15a*1567+t14a*(3784-4096), 12) + t14a

	out[outOffset+0*outStride] = clipRange(t0+t2, min, max)
	out[outOffset+15*outStride] = clipInt32(-int64(clipRange(t1+t3, min, max)))
	t2a = int64(clipRange(t0-t2, min, max))
	t3a = int64(clipRange(t1-t3, min, max))
	out[outOffset+3*outStride] = clipInt32(-int64(clipRange(t4a+t6a, min, max)))
	out[outOffset+12*outStride] = clipRange(t5a+t7a, min, max)
	t6 = int64(clipRange(t4a-t6a, min, max))
	t7 = int64(clipRange(t5a-t7a, min, max))
	out[outOffset+1*outStride] = clipInt32(-int64(clipRange(t8a+t10a, min, max)))
	out[outOffset+14*outStride] = clipRange(t9a+t11a, min, max)
	t10 = int64(clipRange(t8a-t10a, min, max))
	t11 = int64(clipRange(t9a-t11a, min, max))
	out[outOffset+2*outStride] = clipRange(t12+t14, min, max)
	out[outOffset+13*outStride] = clipInt32(-int64(clipRange(t13+t15, min, max)))
	t14a = int64(clipRange(t12-t14, min, max))
	t15a = int64(clipRange(t13-t15, min, max))

	out[outOffset+7*outStride] = clipInt32(-roundShift((t2a+t3a)*181, 8))
	out[outOffset+8*outStride] = clipInt32(roundShift((t2a-t3a)*181, 8))
	out[outOffset+4*outStride] = clipInt32(roundShift((t6+t7)*181, 8))
	out[outOffset+11*outStride] = clipInt32(-roundShift((t6-t7)*181, 8))
	out[outOffset+6*outStride] = clipInt32(roundShift((t10+t11)*181, 8))
	out[outOffset+9*outStride] = clipInt32(-roundShift((t10-t11)*181, 8))
	out[outOffset+5*outStride] = clipInt32(-roundShift((t14a+t15a)*181, 8))
	out[outOffset+10*outStride] = clipInt32(roundShift((t14a-t15a)*181, 8))
}
