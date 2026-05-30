// Stride-1 inverse 1D kernels for the row pass.
//
// The separable inverse transform applies its first 1D pass along contiguous
// rows (stride == 1). Specialising stride==1 lets each kernel reslice its
// input to the exact transform length once, which lets the compiler prove
// every subsequent element access in-bounds and drop the per-element bounds
// checks the generic strided kernels carry. The arithmetic is a verbatim
// copy of the strided kernels (same cospi/sinpi constants, rounding shifts and
// clamp ranges) so the result is byte-for-byte identical.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package transform

// inverse1DRow applies the 1D inverse transform to the contiguous slice c
// (stride 1, length len(c)). It mirrors inverse1D with stride==1, but routes
// the lengths that measurably benefit (DCT 8/16/64, ADST 8) through dedicated
// bounds-check-free stride-1 kernels. The length-4 kernels do so little work
// that the reslice overhead outweighs the saved bounds checks, and DCT/ADST 32
// have no stride-1 specialisation, so those lengths fall through to the shared
// strided kernels.
func inverse1DRow(c []int32, length int, typ tx1DType, min int32, max int32) {
	switch typ {
	case tx1DDCT:
		switch length {
		case dct8Size:
			inverseDCT8Row(c, min, max)
		case dct16Size:
			inverseDCT16Row(c, min, max)
		case dct32Size:
			inverseDCT32Row(c, min, max)
		case dct64Size:
			inverseDCT64Row(c, min, max)
		default:
			inverseDCT1D(c, 1, length, min, max)
		}
	case tx1DADST:
		if length == adst8Size {
			inverseADST8Row(c, min, max)
		} else {
			inverseADST1D(c, 1, length, min, max)
		}
	case tx1DFlipADST:
		if length == adst8Size {
			inverseADST8To(c, 1, c, adst8Size-1, -1, min, max)
		} else {
			inverseFlipADST1D(c, 1, length, min, max)
		}
	case tx1DIdentity:
		inverseIdentity1DRow(c, length)
	}
}

func inverseIdentity1DRow(c []int32, length int) {
	if length <= 0 || length > len(c) {
		return
	}
	c = c[:length]
	for i := range c {
		c[i], _ = identity1DValue(c[i], length)
	}
}

func inverseDCT8Row(c []int32, min int32, max int32) {
	if len(c) < dct8Size {
		return
	}
	c = c[:dct8Size]

	// Even half: DCT4 over indices 0,2,4,6.
	e0 := int64(c[0])
	e1 := int64(c[2])
	e2 := int64(c[4])
	e3 := int64(c[6])
	// DCT4 over the even indices. Its intermediate t0..t3 are unclamped; only
	// the four DCT4 outputs are clipped before the DCT8 stage consumes them
	// (the strided kernel routes those through the int32 coefficient buffer).
	t0 := roundShift((e0+e2)*181, 8)
	t1 := roundShift((e0-e2)*181, 8)
	t2 := roundShift(e1*1567-e3*(3784-4096), 12) - e3
	t3 := roundShift(e1*(3784-4096)+e3*1567, 12) + e1
	even0 := int64(clipRange(t0+t3, min, max))
	even1 := int64(clipRange(t1+t2, min, max))
	even2 := int64(clipRange(t1-t2, min, max))
	even3 := int64(clipRange(t0-t3, min, max))

	in1 := int64(c[1])
	in3 := int64(c[3])
	in5 := int64(c[5])
	in7 := int64(c[7])

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

	c[0] = clipRange(even0+t7, min, max)
	c[1] = clipRange(even1+t6, min, max)
	c[2] = clipRange(even2+t5, min, max)
	c[3] = clipRange(even3+t4, min, max)
	c[4] = clipRange(even3-t4, min, max)
	c[5] = clipRange(even2-t5, min, max)
	c[6] = clipRange(even1-t6, min, max)
	c[7] = clipRange(even0-t7, min, max)
}

func inverseDCT16Row(c []int32, min int32, max int32) {
	if len(c) < dct16Size {
		return
	}
	c = c[:dct16Size]

	// Even half: DCT8 over indices 0,2,4,6,8,10,12,14.
	e0 := int64(c[0])
	e1 := int64(c[2])
	e2 := int64(c[4])
	e3 := int64(c[6])
	e4 := int64(c[8])
	e5 := int64(c[10])
	e6 := int64(c[12])
	e7 := int64(c[14])

	// DCT4 over even-of-even (0,4,8,12 -> e0,e2,e4,e6). The DCT4 intermediates
	// q0..q3 are unclamped; only the four DCT4 outputs r0..r3 are clipped
	// before the surrounding DCT8 stage consumes them (matching the strided
	// kernel's int32 buffer round-trip).
	q0 := roundShift((e0+e4)*181, 8)
	q1 := roundShift((e0-e4)*181, 8)
	q2 := roundShift(e2*1567-e6*(3784-4096), 12) - e6
	q3 := roundShift(e2*(3784-4096)+e6*1567, 12) + e2
	r0 := int64(clipRange(q0+q3, min, max))
	r1 := int64(clipRange(q1+q2, min, max))
	r2 := int64(clipRange(q1-q2, min, max))
	r3 := int64(clipRange(q0-q3, min, max))

	g4a := roundShift(e1*799-e7*(4017-4096), 12) - e7
	g5a := roundShift(e5*1703-e3*1138, 11)
	g6a := roundShift(e5*1138+e3*1703, 11)
	g7a := roundShift(e1*(4017-4096)+e7*799, 12) + e1

	g4 := int64(clipRange(g4a+g5a, min, max))
	g5a = int64(clipRange(g4a-g5a, min, max))
	g7 := int64(clipRange(g7a+g6a, min, max))
	g6a = int64(clipRange(g7a-g6a, min, max))

	g5 := roundShift((g6a-g5a)*181, 8)
	g6 := roundShift((g6a+g5a)*181, 8)

	even0 := clipRange(r0+g7, min, max)
	even1 := clipRange(r1+g6, min, max)
	even2 := clipRange(r2+g5, min, max)
	even3 := clipRange(r3+g4, min, max)
	even4 := clipRange(r3-g4, min, max)
	even5 := clipRange(r2-g5, min, max)
	even6 := clipRange(r1-g6, min, max)
	even7 := clipRange(r0-g7, min, max)

	in1 := int64(c[1])
	in3 := int64(c[3])
	in5 := int64(c[5])
	in7 := int64(c[7])
	in9 := int64(c[9])
	in11 := int64(c[11])
	in13 := int64(c[13])
	in15 := int64(c[15])

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

	c[0] = clipRange(int64(even0)+t15a, min, max)
	c[1] = clipRange(int64(even1)+t14, min, max)
	c[2] = clipRange(int64(even2)+t13a, min, max)
	c[3] = clipRange(int64(even3)+t12, min, max)
	c[4] = clipRange(int64(even4)+t11, min, max)
	c[5] = clipRange(int64(even5)+t10a, min, max)
	c[6] = clipRange(int64(even6)+t9, min, max)
	c[7] = clipRange(int64(even7)+t8a, min, max)
	c[8] = clipRange(int64(even7)-t8a, min, max)
	c[9] = clipRange(int64(even6)-t9, min, max)
	c[10] = clipRange(int64(even5)-t10a, min, max)
	c[11] = clipRange(int64(even4)-t11, min, max)
	c[12] = clipRange(int64(even3)-t12, min, max)
	c[13] = clipRange(int64(even2)-t13a, min, max)
	c[14] = clipRange(int64(even1)-t14, min, max)
	c[15] = clipRange(int64(even0)-t15a, min, max)
}

// inverseDCT32Row applies the inverse DCT32 to the contiguous slice c (stride
// 1). It reslices to the exact transform length so the compiler drops the
// per-element bounds checks, then runs the shared strided butterfly with
// stride 1; the result is byte-for-byte identical to inverseDCT32(c, 1, ...).
func inverseDCT32Row(c []int32, min int32, max int32) {
	if len(c) < dct32Size {
		return
	}
	c = c[:dct32Size]
	inverseDCT32(c, 1, min, max)
}

func inverseADST8Row(c []int32, min int32, max int32) {
	if len(c) < adst8Size {
		return
	}
	c = c[:adst8Size]
	in0 := int64(c[0])
	in1 := int64(c[1])
	in2 := int64(c[2])
	in3 := int64(c[3])
	in4 := int64(c[4])
	in5 := int64(c[5])
	in6 := int64(c[6])
	in7 := int64(c[7])

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

	c[0] = clipRange(t0+t2, min, max)
	c[7] = clipInt32(-int64(clipRange(t1+t3, min, max)))
	t2 = int64(clipRange(t0-t2, min, max))
	t3 = int64(clipRange(t1-t3, min, max))
	c[1] = clipInt32(-int64(clipRange(t4a+t6a, min, max)))
	c[6] = clipRange(t5a+t7a, min, max)
	t6 = int64(clipRange(t4a-t6a, min, max))
	t7 = int64(clipRange(t5a-t7a, min, max))

	c[3] = clipInt32(-roundShift((t2+t3)*181, 8))
	c[4] = clipInt32(roundShift((t2-t3)*181, 8))
	c[2] = clipInt32(roundShift((t6+t7)*181, 8))
	c[5] = clipInt32(-roundShift((t6-t7)*181, 8))
}
