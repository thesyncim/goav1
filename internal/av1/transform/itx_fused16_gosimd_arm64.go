// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// itx_fused16: dav1d's pass structure (itx_fused32) for the 8-bit 16x16
// DCT_DCT block. The row pass runs one or two eight-row strips straight
// from the column-major coefficients with the DCT16 network across vectors
// (its full form, or the Low8 / Low4 forms when the coded columns stop at
// 8 or 4), rounds by 2 in registers and transposes 8x8 into a 16x16 int16
// tmp; the two column strips run the network form that matches the coded
// rows and fuse Round2(., 4), the prediction add and the byte saturation.

// itx16NetFull is inverseDCT16 on eight lanes: in[k] is coefficient k,
// out[r] output r (the network of inverseDCT16Col8SIMD16v2).
func itx16NetFull(in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, p3784, m3784 := k(itx16Kp1567), k(itx16Kp3784), k(itx16Km3784)

	in0, in4, in8, in12 := in[0], in[4], in[8], in[12]
	t0 := itx16Rot(in0, in8, p2896, p2896, s12)
	t1 := itx16Rot(in0, in8, p2896, m2896, s12)
	t2 := itx16Rot(in4, in12, p1567, m3784, s12)
	t3 := itx16Rot(in4, in12, p3784, p1567, s12)
	d0 := t0.AddSaturated(t3)
	d1 := t1.AddSaturated(t2)
	d2 := t1.SubSaturated(t2)
	d3 := t0.SubSaturated(t3)

	in2, in6, in10, in14 := in[2], in[6], in[10], in[14]
	t4a := itx16Rot(in2, in14, k(itx16Kp799), k(itx16Km4017), s12)
	t5a := itx16Rot(in10, in6, k(itx16Kp1703), k(itx16Km1138), s11)
	t6a := itx16Rot(in10, in6, k(itx16Kp1138), k(itx16Kp1703), s11)
	t7a := itx16Rot(in2, in14, k(itx16Kp4017), k(itx16Kp799), s12)
	t4 := t4a.AddSaturated(t5a)
	t5a = t4a.SubSaturated(t5a)
	t7 := t7a.AddSaturated(t6a)
	t6a = t7a.SubSaturated(t6a)
	t5 := itx16Rot(t6a, t5a, p2896, m2896, s12)
	t6 := itx16Rot(t6a, t5a, p2896, p2896, s12)
	e0 := d0.AddSaturated(t7)
	e1 := d1.AddSaturated(t6)
	e2 := d2.AddSaturated(t5)
	e3 := d3.AddSaturated(t4)
	e4 := d3.SubSaturated(t4)
	e5 := d2.SubSaturated(t5)
	e6 := d1.SubSaturated(t6)
	e7 := d0.SubSaturated(t7)

	in1, in15 := in[1], in[15]
	t8a := itx16Rot(in1, in15, k(itx16Kp401), k(itx16Km4076), s12)
	t15a := itx16Rot(in1, in15, k(itx16Kp4076), k(itx16Kp401), s12)
	in9, in7 := in[9], in[7]
	t9a := itx16Rot(in9, in7, k(itx16Kp1583), k(itx16Km1299), s11)
	t14a := itx16Rot(in9, in7, k(itx16Kp1299), k(itx16Kp1583), s11)
	in5, in11 := in[5], in[11]
	t10a := itx16Rot(in5, in11, k(itx16Kp1931), k(itx16Km3612), s12)
	t13a := itx16Rot(in5, in11, k(itx16Kp3612), k(itx16Kp1931), s12)
	in13, in3 := in[13], in[3]
	t11a := itx16Rot(in13, in3, k(itx16Kp3920), k(itx16Km1189), s12)
	t12a := itx16Rot(in13, in3, k(itx16Kp1189), k(itx16Kp3920), s12)

	t8 := t8a.AddSaturated(t9a)
	t9 := t8a.SubSaturated(t9a)
	t10 := t11a.SubSaturated(t10a)
	t11 := t11a.AddSaturated(t10a)
	t12 := t12a.AddSaturated(t13a)
	t13 := t12a.SubSaturated(t13a)
	t14 := t15a.SubSaturated(t14a)
	t15 := t15a.AddSaturated(t14a)

	t9a = itx16Rot(t14, t9, p1567, m3784, s12)
	t14a = itx16Rot(t14, t9, p3784, p1567, s12)
	t10a = itx16Rot(t13, t10, m3784, k(itx16Km1567), s12)
	t13a = itx16Rot(t13, t10, p1567, m3784, s12)

	t8a = t8.AddSaturated(t11)
	t9 = t9a.AddSaturated(t10a)
	t10 = t9a.SubSaturated(t10a)
	t11a = t8.SubSaturated(t11)
	t12a = t15.SubSaturated(t12)
	t13 = t14a.SubSaturated(t13a)
	t14 = t14a.AddSaturated(t13a)
	t15a = t15.AddSaturated(t12)

	t10a = itx16Rot(t13, t10, p2896, m2896, s12)
	t13a = itx16Rot(t13, t10, p2896, p2896, s12)
	t11 = itx16Rot(t12a, t11a, p2896, m2896, s12)
	t12 = itx16Rot(t12a, t11a, p2896, p2896, s12)

	out[0], out[15] = e0.AddSaturated(t15a), e0.SubSaturated(t15a)
	out[1], out[14] = e1.AddSaturated(t14), e1.SubSaturated(t14)
	out[2], out[13] = e2.AddSaturated(t13a), e2.SubSaturated(t13a)
	out[3], out[12] = e3.AddSaturated(t12), e3.SubSaturated(t12)
	out[4], out[11] = e4.AddSaturated(t11), e4.SubSaturated(t11)
	out[5], out[10] = e5.AddSaturated(t10a), e5.SubSaturated(t10a)
	out[6], out[9] = e6.AddSaturated(t9), e6.SubSaturated(t9)
	out[7], out[8] = e7.AddSaturated(t8a), e7.SubSaturated(t8a)
}

// itx16NetLow8 is itx16NetFull for inputs 8..15 zero (the network of
// inverseDCT16Col8RawLow8SIMD16).
func itx16NetLow8(in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, p3784, m3784 := k(itx16Kp1567), k(itx16Kp3784), k(itx16Km3784)
	in0, in1, in2, in3, in4, in5, in6, in7 := in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]

	t0 := itx16Rot1(in0, p2896, s12)
	t2 := itx16Rot1(in4, p1567, s12)
	t3 := itx16Rot1(in4, p3784, s12)
	d0 := t0.AddSaturated(t3)
	d1 := t0.AddSaturated(t2)
	d2 := t0.SubSaturated(t2)
	d3 := t0.SubSaturated(t3)
	t4a := itx16Rot1(in2, k(itx16Kp799), s12)
	t5a := itx16Rot1(in6, k(itx16Km1138), s11)
	t6a := itx16Rot1(in6, k(itx16Kp1703), s11)
	t7a := itx16Rot1(in2, k(itx16Kp4017), s12)
	t4 := t4a.AddSaturated(t5a)
	t5a = t4a.SubSaturated(t5a)
	t7 := t7a.AddSaturated(t6a)
	t6a = t7a.SubSaturated(t6a)
	t5 := itx16Rot(t6a, t5a, p2896, m2896, s12)
	t6 := itx16Rot(t6a, t5a, p2896, p2896, s12)
	e0 := d0.AddSaturated(t7)
	e1 := d1.AddSaturated(t6)
	e2 := d2.AddSaturated(t5)
	e3 := d3.AddSaturated(t4)
	e4 := d3.SubSaturated(t4)
	e5 := d2.SubSaturated(t5)
	e6 := d1.SubSaturated(t6)
	e7 := d0.SubSaturated(t7)

	t8a := itx16Rot1(in1, k(itx16Kp401), s12)
	t15a := itx16Rot1(in1, k(itx16Kp4076), s12)
	t9a := itx16Rot1(in7, k(itx16Km1299), s11)
	t14a := itx16Rot1(in7, k(itx16Kp1583), s11)
	t10a := itx16Rot1(in5, k(itx16Kp1931), s12)
	t13a := itx16Rot1(in5, k(itx16Kp3612), s12)
	t11a := itx16Rot1(in3, k(itx16Km1189), s12)
	t12a := itx16Rot1(in3, k(itx16Kp3920), s12)

	t8 := t8a.AddSaturated(t9a)
	t9 := t8a.SubSaturated(t9a)
	t10 := t11a.SubSaturated(t10a)
	t11 := t11a.AddSaturated(t10a)
	t12 := t12a.AddSaturated(t13a)
	t13 := t12a.SubSaturated(t13a)
	t14 := t15a.SubSaturated(t14a)
	t15 := t15a.AddSaturated(t14a)

	t9a = itx16Rot(t14, t9, p1567, m3784, s12)
	t14a = itx16Rot(t14, t9, p3784, p1567, s12)
	t10a = itx16Rot(t13, t10, m3784, k(itx16Km1567), s12)
	t13a = itx16Rot(t13, t10, p1567, m3784, s12)

	t8a = t8.AddSaturated(t11)
	t9 = t9a.AddSaturated(t10a)
	t10 = t9a.SubSaturated(t10a)
	t11a = t8.SubSaturated(t11)
	t12a = t15.SubSaturated(t12)
	t13 = t14a.SubSaturated(t13a)
	t14 = t14a.AddSaturated(t13a)
	t15a = t15.AddSaturated(t12)

	t10a = itx16Rot(t13, t10, p2896, m2896, s12)
	t13a = itx16Rot(t13, t10, p2896, p2896, s12)
	t11 = itx16Rot(t12a, t11a, p2896, m2896, s12)
	t12 = itx16Rot(t12a, t11a, p2896, p2896, s12)

	out[0], out[15] = e0.AddSaturated(t15a), e0.SubSaturated(t15a)
	out[1], out[14] = e1.AddSaturated(t14), e1.SubSaturated(t14)
	out[2], out[13] = e2.AddSaturated(t13a), e2.SubSaturated(t13a)
	out[3], out[12] = e3.AddSaturated(t12), e3.SubSaturated(t12)
	out[4], out[11] = e4.AddSaturated(t11), e4.SubSaturated(t11)
	out[5], out[10] = e5.AddSaturated(t10a), e5.SubSaturated(t10a)
	out[6], out[9] = e6.AddSaturated(t9), e6.SubSaturated(t9)
	out[7], out[8] = e7.AddSaturated(t8a), e7.SubSaturated(t8a)
}

// itx16NetLow4 is itx16NetFull for inputs 4..15 zero (the network of
// inverseDCT16Col8RawLow4SIMD16).
func itx16NetLow4(in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, p3784, m3784 := k(itx16Kp1567), k(itx16Kp3784), k(itx16Km3784)
	in0, in1, in2, in3 := in[0], in[1], in[2], in[3]

	t0 := itx16Rot1(in0, p2896, s12)
	t4a := itx16Rot1(in2, k(itx16Kp799), s12)
	t7a := itx16Rot1(in2, k(itx16Kp4017), s12)
	t5 := itx16Rot(t7a, t4a, p2896, m2896, s12)
	t6 := itx16Rot(t7a, t4a, p2896, p2896, s12)
	e0 := t0.AddSaturated(t7a)
	e1 := t0.AddSaturated(t6)
	e2 := t0.AddSaturated(t5)
	e3 := t0.AddSaturated(t4a)
	e4 := t0.SubSaturated(t4a)
	e5 := t0.SubSaturated(t5)
	e6 := t0.SubSaturated(t6)
	e7 := t0.SubSaturated(t7a)

	t8a := itx16Rot1(in1, k(itx16Kp401), s12)
	t15a := itx16Rot1(in1, k(itx16Kp4076), s12)
	t11a := itx16Rot1(in3, k(itx16Km1189), s12)
	t12a := itx16Rot1(in3, k(itx16Kp3920), s12)
	t9a := itx16Rot(t15a, t8a, p1567, m3784, s12)
	t14a := itx16Rot(t15a, t8a, p3784, p1567, s12)
	t10a := itx16Rot(t12a, t11a, m3784, k(itx16Km1567), s12)
	t13a := itx16Rot(t12a, t11a, p1567, m3784, s12)

	t8 := t8a.AddSaturated(t11a)
	t9 := t9a.AddSaturated(t10a)
	t10 := t9a.SubSaturated(t10a)
	t11 := t8a.SubSaturated(t11a)
	t12 := t15a.SubSaturated(t12a)
	t13 := t14a.SubSaturated(t13a)
	t14 := t14a.AddSaturated(t13a)
	t15 := t15a.AddSaturated(t12a)

	t10a = itx16Rot(t13, t10, p2896, m2896, s12)
	t13a = itx16Rot(t13, t10, p2896, p2896, s12)
	t11a = itx16Rot(t12, t11, p2896, m2896, s12)
	t12a = itx16Rot(t12, t11, p2896, p2896, s12)

	out[0], out[15] = e0.AddSaturated(t15), e0.SubSaturated(t15)
	out[1], out[14] = e1.AddSaturated(t14), e1.SubSaturated(t14)
	out[2], out[13] = e2.AddSaturated(t13a), e2.SubSaturated(t13a)
	out[3], out[12] = e3.AddSaturated(t12a), e3.SubSaturated(t12a)
	out[4], out[11] = e4.AddSaturated(t11a), e4.SubSaturated(t11a)
	out[5], out[10] = e5.AddSaturated(t10a), e5.SubSaturated(t10a)
	out[6], out[9] = e6.AddSaturated(t9), e6.SubSaturated(t9)
	out[7], out[8] = e7.AddSaturated(t8), e7.SubSaturated(t8)
}

// itx16Net runs the DCT16 network form for n nonzero inputs (4, 8 or 16).
func itx16Net(n int, in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	switch {
	case n <= 4:
		itx16NetLow4(in, out)
	case n <= 8:
		itx16NetLow8(in, out)
	default:
		itx16NetFull(in, out)
	}
}

// itx16RowStrip runs the 16x16 row pass for rows r0..r0+7 with inCols
// coded coefficient columns (4, 8 or 16) into tmp rows r0..r0+7 and adds
// the stored values' magnitudes to the two column guards. It reports false,
// with tmp untouched, when a row fails the L1 guard.
func itx16RowStrip(coeff []int32, stride int, r0 int, inCols int, tmp *[16 * 16]int16, colL1 *[2]archsimd.Uint16x8) bool {
	_ = coeff[(inCols-1)*stride+r0+7]
	var in [16]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < inCols; k++ {
		x := itx16LoadNarrow(coeff[k*stride+r0 : k*stride+r0+8])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[k] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [16]archsimd.Int16x8
	itx16Net(inCols, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	for g := 0; g < 2; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itx32Round2(out[8*g+j], sh1, sh1, one)
		}
		itx16Transpose8x8(&v)
		gl1 := colL1[g]
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[(r0+r)*16+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1[g] = gl1
	}
	return true
}

// itx16ColStripAdd runs the 16x16 column pass for columns c0..c0+7 from
// inRows coded tmp rows (4, 8 or 16) and adds the result to the block.
func itx16ColStripAdd(tmp *[16 * 16]int16, c0 int, inRows int, dst unsafe.Pointer, dstStride int, pred unsafe.Pointer, predStride int) {
	var in [16]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*16+c0:]))
	}
	var out [16]archsimd.Int16x8
	itx16Net(inRows, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	sh3 := archsimd.BroadcastInt16x8(-3)
	for r := 0; r < 16; r++ {
		res := itx32Round2(out[r], sh3, sh1, one)
		rows := [2]uint64{*(*uint64)(unsafe.Add(pred, r*predStride+c0)), 0}
		p := archsimd.LoadUint64x2Array(&rows).ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()
		*(*uint64)(unsafe.Add(dst, r*dstStride+c0)) = p.Add(res).SaturateToUint8().ReshapeToUint64s().GetElem(0)
	}
}

// itx16Form rounds a coded extent up to the network form that covers it.
func itx16Form(n int) int {
	switch {
	case n <= 4:
		return 4
	case n <= 8:
		return 8
	default:
		return 16
	}
}

// InverseAddDCT16x16 reconstructs one 8-bit 16x16 DCT_DCT block with coded
// coefficients in rows 0..activeRows-1 and columns 0..activeCols-1:
// dst = clip(pred + inverse(coeff)), with the dequantized coefficients
// column-major at coeffStride. pred may be dst. It reports false, with dst
// untouched, when the block is outside the kernel's contract or a guard
// declines; the caller then runs the staged path, which produces the same
// bytes.
func InverseAddDCT16x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 16 || activeCols <= 0 || activeCols > 16 ||
		coeffStride < 16 || len(coeff) < 15*coeffStride+16 || dstStride < 16 || predStride < 16 ||
		len(dst) < 15*dstStride+16 || len(pred) < 15*predStride+16 {
		return false
	}
	inCols := itx16Form(activeCols)
	inRows := itx16Form(activeRows)
	var tmp [16 * 16]int16
	var colL1 [2]archsimd.Uint16x8
	// A row strip writes all eight of its rows, so rows past the coded ones
	// come out zero and the column form reads them as such; the second
	// strip runs only when coded rows reach it.
	if !itx16RowStrip(coeff, coeffStride, 0, inCols, &tmp, &colL1) {
		return false
	}
	if activeRows > 8 && !itx16RowStrip(coeff, coeffStride, 8, inCols, &tmp, &colL1) {
		return false
	}
	if colL1[0].Max(colL1[1]).ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	pp := unsafe.Pointer(unsafe.SliceData(pred))
	itx16ColStripAdd(&tmp, 0, inRows, dp, dstStride, pp, predStride)
	itx16ColStripAdd(&tmp, 8, inRows, dp, dstStride, pp, predStride)
	return true
}
