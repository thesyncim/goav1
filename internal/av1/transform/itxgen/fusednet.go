// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package main

// The DCT networks of the fused generator: the stage structure of
// inverseDCT4/8/16/32 (dct.go) and the DCT64 odd half of gen.network, with
// every roundShift written as the rotation Round2(a*ca + b*cb, 12) it
// equals and every clipped sum as a saturating int16 add (the 8-bit row and
// column clamp ranges are the int16 range).

// dct4 is the four-point network of inverseDCT4.
func (g *fgen) dct4(in []fv) []fv {
	t0 := g.rot(in[0], 2896, in[2], 2896)
	t1 := g.rot(in[0], 2896, in[2], -2896)
	t2 := g.rot(in[1], 1567, in[3], -3784)
	t3 := g.rot(in[1], 3784, in[3], 1567)
	return []fv{g.add(t0, t3), g.add(t1, t2), g.sub(t1, t2), g.sub(t0, t3)}
}

// dct8 is inverseDCT8: dct4 on the even inputs, then the odd half.
func (g *fgen) dct8(in []fv) []fv {
	e := g.dct4([]fv{in[0], in[2], in[4], in[6]})
	t4a := g.rot(in[1], 799, in[7], -4017)
	t5a := g.rot(in[5], 3406, in[3], -2276)
	t6a := g.rot(in[5], 2276, in[3], 3406)
	t7a := g.rot(in[1], 4017, in[7], 799)
	t4 := g.add(t4a, t5a)
	t5b := g.sub(t4a, t5a)
	t7 := g.add(t7a, t6a)
	t6b := g.sub(t7a, t6a)
	t5 := g.rot(t6b, 2896, t5b, -2896)
	t6 := g.rot(t6b, 2896, t5b, 2896)
	return []fv{
		g.add(e[0], t7), g.add(e[1], t6), g.add(e[2], t5), g.add(e[3], t4),
		g.sub(e[3], t4), g.sub(e[2], t5), g.sub(e[1], t6), g.sub(e[0], t7),
	}
}

// dct16 is inverseDCT16: dct8 on the even inputs, then the odd half.
func (g *fgen) dct16(in []fv) []fv {
	e := g.dct8([]fv{in[0], in[2], in[4], in[6], in[8], in[10], in[12], in[14]})
	t8a := g.rot(in[1], 401, in[15], -4076)
	t15a := g.rot(in[1], 4076, in[15], 401)
	t9a := g.rot(in[9], 3166, in[7], -2598)
	t14a := g.rot(in[9], 2598, in[7], 3166)
	t10a := g.rot(in[5], 1931, in[11], -3612)
	t13a := g.rot(in[5], 3612, in[11], 1931)
	t11a := g.rot(in[13], 3920, in[3], -1189)
	t12a := g.rot(in[13], 1189, in[3], 3920)

	t8 := g.add(t8a, t9a)
	t9 := g.sub(t8a, t9a)
	t10 := g.sub(t11a, t10a)
	t11 := g.add(t11a, t10a)
	t12 := g.add(t12a, t13a)
	t13 := g.sub(t12a, t13a)
	t14 := g.sub(t15a, t14a)
	t15 := g.add(t15a, t14a)

	t9a = g.rot(t14, 1567, t9, -3784)
	t14a = g.rot(t14, 3784, t9, 1567)
	t10a = g.rot(t13, -3784, t10, -1567)
	t13a = g.rot(t13, 1567, t10, -3784)

	t8a = g.add(t8, t11)
	t9 = g.add(t9a, t10a)
	t10 = g.sub(t9a, t10a)
	t11a = g.sub(t8, t11)
	t12a = g.sub(t15, t12)
	t13 = g.sub(t14a, t13a)
	t14 = g.add(t14a, t13a)
	t15a = g.add(t15, t12)

	t10a = g.rot(t13, 2896, t10, -2896)
	t13a = g.rot(t13, 2896, t10, 2896)
	t11 = g.rot(t12a, 2896, t11a, -2896)
	t12 = g.rot(t12a, 2896, t11a, 2896)

	return []fv{
		g.add(e[0], t15a), g.add(e[1], t14), g.add(e[2], t13a), g.add(e[3], t12),
		g.add(e[4], t11), g.add(e[5], t10a), g.add(e[6], t9), g.add(e[7], t8a),
		g.sub(e[7], t8a), g.sub(e[6], t9), g.sub(e[5], t10a), g.sub(e[4], t11),
		g.sub(e[3], t12), g.sub(e[2], t13a), g.sub(e[1], t14), g.sub(e[0], t15a),
	}
}

// dct32 is inverseDCT32: dct16 on the even inputs, then the odd half.
func (g *fgen) dct32(in []fv) []fv {
	ev := make([]fv, 16)
	for i := range ev {
		ev[i] = in[2*i]
	}
	e := g.dct16(ev)

	t16a := g.rot(in[1], 201, in[31], -4091)
	t17a := g.rot(in[17], 3035, in[15], -2751)
	t18a := g.rot(in[9], 1751, in[23], -3703)
	t19a := g.rot(in[25], 3857, in[7], -1380)
	t20a := g.rot(in[5], 995, in[27], -3973)
	t21a := g.rot(in[21], 3513, in[11], -2106)
	t22a := g.rot(in[13], 2440, in[19], -3290)
	t23a := g.rot(in[29], 4052, in[3], -601)
	t24a := g.rot(in[29], 601, in[3], 4052)
	t25a := g.rot(in[13], 3290, in[19], 2440)
	t26a := g.rot(in[21], 2106, in[11], 3513)
	t27a := g.rot(in[5], 3973, in[27], 995)
	t28a := g.rot(in[25], 1380, in[7], 3857)
	t29a := g.rot(in[9], 3703, in[23], 1751)
	t30a := g.rot(in[17], 2751, in[15], 3035)
	t31a := g.rot(in[1], 4091, in[31], 201)

	t16 := g.add(t16a, t17a)
	t17 := g.sub(t16a, t17a)
	t18 := g.sub(t19a, t18a)
	t19 := g.add(t19a, t18a)
	t20 := g.add(t20a, t21a)
	t21 := g.sub(t20a, t21a)
	t22 := g.sub(t23a, t22a)
	t23 := g.add(t23a, t22a)
	t24 := g.add(t24a, t25a)
	t25 := g.sub(t24a, t25a)
	t26 := g.sub(t27a, t26a)
	t27 := g.add(t27a, t26a)
	t28 := g.add(t28a, t29a)
	t29 := g.sub(t28a, t29a)
	t30 := g.sub(t31a, t30a)
	t31 := g.add(t31a, t30a)

	t17a = g.rot(t30, 799, t17, -4017)
	t30a = g.rot(t30, 4017, t17, 799)
	t18a = g.rot(t29, -4017, t18, -799)
	t29a = g.rot(t29, 799, t18, -4017)
	t21a = g.rot(t26, 3406, t21, -2276)
	t26a = g.rot(t26, 2276, t21, 3406)
	t22a = g.rot(t25, -2276, t22, -3406)
	t25a = g.rot(t25, 3406, t22, -2276)

	t16a = g.add(t16, t19)
	t17 = g.add(t17a, t18a)
	t18 = g.sub(t17a, t18a)
	t19a = g.sub(t16, t19)
	t20a = g.sub(t23, t20)
	t21 = g.sub(t22a, t21a)
	t22 = g.add(t22a, t21a)
	t23a = g.add(t23, t20)
	t24a = g.add(t24, t27)
	t25 = g.add(t25a, t26a)
	t26 = g.sub(t25a, t26a)
	t27a = g.sub(t24, t27)
	t28a = g.sub(t31, t28)
	t29 = g.sub(t30a, t29a)
	t30 = g.add(t30a, t29a)
	t31a = g.add(t31, t28)

	t18a = g.rot(t29, 1567, t18, -3784)
	t29a = g.rot(t29, 3784, t18, 1567)
	t19 = g.rot(t28a, 1567, t19a, -3784)
	t28 = g.rot(t28a, 3784, t19a, 1567)
	t20 = g.rot(t27a, -3784, t20a, -1567)
	t27 = g.rot(t27a, 1567, t20a, -3784)
	t21a = g.rot(t26, -3784, t21, -1567)
	t26a = g.rot(t26, 1567, t21, -3784)

	t16 = g.add(t16a, t23a)
	t17a = g.add(t17, t22)
	t18 = g.add(t18a, t21a)
	t19a = g.add(t19, t20)
	t20a = g.sub(t19, t20)
	t21 = g.sub(t18a, t21a)
	t22a = g.sub(t17, t22)
	t23 = g.sub(t16a, t23a)
	t24 = g.sub(t31a, t24a)
	t25a = g.sub(t30, t25)
	t26 = g.sub(t29a, t26a)
	t27a = g.sub(t28, t27)
	t28a = g.add(t28, t27)
	t29 = g.add(t29a, t26a)
	t30a = g.add(t30, t25)
	t31 = g.add(t31a, t24a)

	t20 = g.rot(t27a, 2896, t20a, -2896)
	t27 = g.rot(t27a, 2896, t20a, 2896)
	t21a = g.rot(t26, 2896, t21, -2896)
	t26a = g.rot(t26, 2896, t21, 2896)
	t22 = g.rot(t25a, 2896, t22a, -2896)
	t25 = g.rot(t25a, 2896, t22a, 2896)
	t23a = g.rot(t24, 2896, t23, -2896)
	t24a = g.rot(t24, 2896, t23, 2896)

	o := []fv{t31, t30a, t29, t28a, t27, t26a, t25, t24a, t23a, t22, t21a, t20, t19a, t18, t17a, t16}
	out := make([]fv, 32)
	for i := 0; i < 16; i++ {
		out[i], out[31-i] = g.combine(e[i], o[i])
	}
	return out
}

// dct64 is inverseDCT64 for inputs 32..63 zero (the AV1 64-point rows and
// columns code only the first 32): dct32 on the even inputs and the odd
// half of gen.network (stage-10 values 32..63), combined as itx64Net does.
func (g *fgen) dct64(in []fv) []fv {
	ev := make([]fv, 32)
	for i := range ev {
		ev[i] = in[2*i]
	}
	e := g.dct32(ev)
	kp := func(i int) int { return cos64[i] }
	kn := func(i int) int { return -cos64[i] }
	var x [64]fv
	// Stage 2: pairs (32+j, 63-j) from inputs (a, 64-a) in bit-reversed order.
	a := [16]int{1, 33, 17, 49, 9, 41, 25, 57, 5, 37, 21, 53, 13, 45, 29, 61}
	for j := 0; j < 16; j++ {
		lo, hi := 32+j, 63-j
		ia, ib := a[j], 64-a[j]
		x[lo] = g.rot(in[ia], kp(ib), in[ib], kn(ia))
		x[hi] = g.rot(in[ia], kp(ia), in[ib], kp(ib))
	}
	// Stage 3.
	var y [64]fv
	for b := 32; b < 64; b += 4 {
		y[b] = g.add(x[b], x[b+1])
		y[b+1] = g.sub(x[b], x[b+1])
		y[b+2] = g.sub(x[b+3], x[b+2])
		y[b+3] = g.add(x[b+2], x[b+3])
	}
	x = y
	// Stage 4.
	y = x
	for m, pq := range [4][2]int{{4, 60}, {36, 28}, {20, 44}, {52, 12}} {
		p, q := pq[0], pq[1]
		lo, hi := 33+4*m, 62-4*m
		y[lo] = g.rot(x[lo], kn(p), x[hi], kp(q))
		y[hi] = g.rot(x[lo], kp(q), x[hi], kp(p))
		y[lo+1] = g.rot(x[lo+1], kn(q), x[hi-1], kn(p))
		y[hi-1] = g.rot(x[lo+1], kn(p), x[hi-1], kp(q))
	}
	x = y
	// Stage 5.
	for b := 32; b < 64; b += 8 {
		y[b] = g.add(x[b], x[b+3])
		y[b+1] = g.add(x[b+1], x[b+2])
		y[b+2] = g.sub(x[b+1], x[b+2])
		y[b+3] = g.sub(x[b], x[b+3])
		y[b+4] = g.sub(x[b+7], x[b+4])
		y[b+5] = g.sub(x[b+6], x[b+5])
		y[b+6] = g.add(x[b+5], x[b+6])
		y[b+7] = g.add(x[b+4], x[b+7])
	}
	x = y
	// Stage 6.
	y = x
	for _, pql := range [2][4]int{{8, 56, 34, 61}, {40, 24, 42, 53}} {
		p, q, lo, hi := pql[0], pql[1], pql[2], pql[3]
		for i := 0; i < 2; i++ {
			y[lo+i] = g.rot(x[lo+i], kn(p), x[hi-i], kp(q))
			y[hi-i] = g.rot(x[lo+i], kp(q), x[hi-i], kp(p))
		}
		for i := 2; i < 4; i++ {
			y[lo+i] = g.rot(x[lo+i], kn(q), x[hi-i], kn(p))
			y[hi-i] = g.rot(x[lo+i], kn(p), x[hi-i], kp(q))
		}
	}
	x = y
	// Stage 7.
	for b := 32; b < 64; b += 16 {
		for i := 0; i < 4; i++ {
			y[b+i] = g.add(x[b+i], x[b+7-i])
			y[b+7-i] = g.sub(x[b+i], x[b+7-i])
			y[b+8+i] = g.sub(x[b+15-i], x[b+8+i])
			y[b+15-i] = g.add(x[b+8+i], x[b+15-i])
		}
	}
	x = y
	// Stage 8.
	y = x
	for i := 0; i < 4; i++ {
		y[36+i] = g.rot(x[36+i], kn(16), x[59-i], kp(48))
		y[59-i] = g.rot(x[36+i], kp(48), x[59-i], kp(16))
		y[40+i] = g.rot(x[40+i], kn(48), x[55-i], kn(16))
		y[55-i] = g.rot(x[40+i], kn(16), x[55-i], kp(48))
	}
	x = y
	// Stage 9.
	for i := 0; i < 8; i++ {
		y[32+i] = g.add(x[32+i], x[47-i])
		y[47-i] = g.sub(x[32+i], x[47-i])
		y[48+i] = g.sub(x[63-i], x[48+i])
		y[63-i] = g.add(x[48+i], x[63-i])
	}
	x = y
	// Stage 10.
	y = x
	for i := 0; i < 8; i++ {
		y[40+i] = g.rot(x[40+i], kn(32), x[55-i], kp(32))
		y[55-i] = g.rot(x[40+i], kp(32), x[55-i], kp(32))
	}
	out := make([]fv, 64)
	for i := 0; i < 32; i++ {
		out[i], out[63-i] = g.combine(e[i], y[32+31-i])
	}
	return out
}

// cos64 is itx64Cos: cospi(i) = round(4096 * cos(i*pi/128)).
var cos64 = [64]int{
	4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
	3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
	2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
	1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101,
}

// adst4 is inverseADST4To (adst.go) with each roundShift(a, 12) + x
// written as the one product sum Round2(a + 4096x, 12) it equals and
// Round2(209s, 8) as Round2(3344s, 12); the sums of inputs are inside the
// products, so nothing is clamped before the final rounding, as in the
// scalar transform.
func (g *fgen) adst4(in []fv) []fv {
	return []fv{
		g.dot([]term{{in[0], 1321}, {in[1], 3344}, {in[2], 3803}, {in[3], 2482}}),
		g.dot([]term{{in[0], 2482}, {in[1], 3344}, {in[2], -1321}, {in[3], -3803}}),
		g.dot([]term{{in[0], 3344}, {in[2], -3344}, {in[3], 3344}}),
		g.dot([]term{{in[0], 3803}, {in[1], -3344}, {in[2], 2482}, {in[3], -1321}}),
	}
}

func (g *fgen) network(n int, in []fv) []fv {
	switch n {
	case 4:
		return g.dct4(in)
	case 8:
		return g.dct8(in)
	case 16:
		return g.dct16(in)
	case 32:
		return g.dct32(in)
	case 64:
		return g.dct64(in)
	}
	panic("network size")
}
