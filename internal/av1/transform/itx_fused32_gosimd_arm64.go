// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// itx_fused32: dav1d's inverse-transform pass structure (src/arm/64/itx.S
// inv_txfm_add_dct_dct_32x32_8bpc_neon) for the 8-bit 32x32 DCT_DCT block
// whose coded coefficients lie in rows and columns 0..7 (the eob-bounded
// shape the decoder meets most):
//
//   - the row pass reads the dequantized coefficients directly. They are
//     stored column-major, so coefficient k of rows 0..7 is one contiguous
//     load; narrowing it to int16 lanes (SQXTN) is the 8-bit row clamp. The
//     DCT32 runs across vectors with the rows in the lanes (the exact int16
//     network of the DCT32 Low8 column kernel, with its per-lane L1 guard),
//     the inter-pass Round2(., 2) is applied in registers, and the outputs
//     are transposed 8x8 at a time (ZIP) into one row-major int16 tmp, the
//     only intermediate dav1d keeps;
//   - each column strip loads its eight columns of tmp (contiguous), runs
//     the same network with the columns in the lanes, and its final stage
//     rounds by 4, adds the prediction row, saturates to bytes and stores:
//     the residual add is fused, with no int32 scratch or separate add pass.
//
// Exactness: both passes are the certified int16 network (itx16_gain_test.go,
// DCT32 bound 0.9989 plus 9.15 slack under the per-lane L1 guard
// itx16ColumnL1Bound), the butterflies saturate exactly as the scalar clip
// to the 8-bit row and column bounds (the int16 range), Round2(x, n) is
// formed as ((x >> (n-1)) + 1) >> 1 so no lane overflows, and the add
// saturates to [0, 255] as the raw add does. The row strip checks its guard
// before writing tmp and gathers the column strips' guards from the
// transposed values, so a declined block is left untouched and the caller
// falls back to the staged path.

// itx16Transpose8x8 transposes the 8x8 int16 matrix v (v[r] = row r) in
// registers: three interleave stages at 16, 32 and 64 bits.
func itx16Transpose8x8(v *[8]archsimd.Int16x8) {
	a0, a1 := v[0].InterleaveLo(v[1]).ToBits().ReshapeToUint32s(), v[0].InterleaveHi(v[1]).ToBits().ReshapeToUint32s()
	a2, a3 := v[2].InterleaveLo(v[3]).ToBits().ReshapeToUint32s(), v[2].InterleaveHi(v[3]).ToBits().ReshapeToUint32s()
	a4, a5 := v[4].InterleaveLo(v[5]).ToBits().ReshapeToUint32s(), v[4].InterleaveHi(v[5]).ToBits().ReshapeToUint32s()
	a6, a7 := v[6].InterleaveLo(v[7]).ToBits().ReshapeToUint32s(), v[6].InterleaveHi(v[7]).ToBits().ReshapeToUint32s()
	b0, b1 := a0.InterleaveLo(a2).ReshapeToUint64s(), a0.InterleaveHi(a2).ReshapeToUint64s()
	b2, b3 := a1.InterleaveLo(a3).ReshapeToUint64s(), a1.InterleaveHi(a3).ReshapeToUint64s()
	b4, b5 := a4.InterleaveLo(a6).ReshapeToUint64s(), a4.InterleaveHi(a6).ReshapeToUint64s()
	b6, b7 := a5.InterleaveLo(a7).ReshapeToUint64s(), a5.InterleaveHi(a7).ReshapeToUint64s()
	v[0] = b0.InterleaveLo(b4).ReshapeToUint16s().BitsToInt16()
	v[1] = b0.InterleaveHi(b4).ReshapeToUint16s().BitsToInt16()
	v[2] = b1.InterleaveLo(b5).ReshapeToUint16s().BitsToInt16()
	v[3] = b1.InterleaveHi(b5).ReshapeToUint16s().BitsToInt16()
	v[4] = b2.InterleaveLo(b6).ReshapeToUint16s().BitsToInt16()
	v[5] = b2.InterleaveHi(b6).ReshapeToUint16s().BitsToInt16()
	v[6] = b3.InterleaveLo(b7).ReshapeToUint16s().BitsToInt16()
	v[7] = b3.InterleaveHi(b7).ReshapeToUint16s().BitsToInt16()
}

// itx32Low8Net is inverseDCT32 on eight lanes for inputs whose indices 8..31
// are zero: in[k] is coefficient k, out[r] is output r. It is the network of
// inverseDCT32Col8RawLow8SIMD16 with the loads and stores lifted out.
func itx32Low8Net(in *[8]archsimd.Int16x8, out *[32]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	k32 := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx32K[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, m1567, p3784, m3784 := k(itx16Kp1567), k(itx16Km1567), k(itx16Kp3784), k(itx16Km3784)

	// Even half: inverseDCT16 on inputs 0, 2, 4, 6 (its Low4 form).
	in0, in2, in4, in6 := in[0], in[2], in[4], in[6]
	t0 := itx16Rot1(in0, p2896, s12)
	t4a := itx16Rot1(in4, k(itx16Kp799), s12)
	t7a := itx16Rot1(in4, k(itx16Kp4017), s12)
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
	t8a := itx16Rot1(in2, k(itx16Kp401), s12)
	t15a := itx16Rot1(in2, k(itx16Kp4076), s12)
	t11a := itx16Rot1(in6, k(itx16Km1189), s12)
	t12a := itx16Rot1(in6, k(itx16Kp3920), s12)
	t9a := itx16Rot(t15a, t8a, p1567, m3784, s12)
	t14a := itx16Rot(t15a, t8a, p3784, p1567, s12)
	t10a := itx16Rot(t12a, t11a, m3784, m1567, s12)
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
	E0, E15 := e0.AddSaturated(t15), e0.SubSaturated(t15)
	E1, E14 := e1.AddSaturated(t14), e1.SubSaturated(t14)
	E2, E13 := e2.AddSaturated(t13a), e2.SubSaturated(t13a)
	E3, E12 := e3.AddSaturated(t12a), e3.SubSaturated(t12a)
	E4, E11 := e4.AddSaturated(t11a), e4.SubSaturated(t11a)
	E5, E10 := e5.AddSaturated(t10a), e5.SubSaturated(t10a)
	E6, E9 := e6.AddSaturated(t9), e6.SubSaturated(t9)
	E7, E8 := e7.AddSaturated(t8), e7.SubSaturated(t8)

	// Odd half from inputs 1, 3, 5, 7: the first-stage rotations are single
	// products and the second-stage butterflies pass them through.
	in1, in3, in5, in7 := in[1], in[3], in[5], in[7]
	t16 := itx16Rot1(in1, k32(itx32Kp201), s12)
	t31 := itx16Rot1(in1, k32(itx32Kp4091), s12)
	t19 := itx16Rot1(in7, k32(itx32Km1380), s12)
	t28 := itx16Rot1(in7, k32(itx32Kp3857), s12)
	t20 := itx16Rot1(in5, k32(itx32Kp995), s12)
	t27 := itx16Rot1(in5, k32(itx32Kp3973), s12)
	t23 := itx16Rot1(in3, k32(itx32Km601), s12)
	t24 := itx16Rot1(in3, k32(itx32Kp4052), s12)
	t17, t18, t21, t22, t25, t26, t29, t30 := t16, t19, t20, t23, t24, t27, t28, t31

	p799, m799, p4017, m4017 := k(itx16Kp799), k32(itx32Km799), k(itx16Kp4017), k(itx16Km4017)
	t17a := itx16Rot(t30, t17, p799, m4017, s12)
	t30a := itx16Rot(t30, t17, p4017, p799, s12)
	t18a := itx16Rot(t29, t18, m4017, m799, s12)
	t29a := itx16Rot(t29, t18, p799, m4017, s12)
	p1703, m1703, p1138, m1138 := k(itx16Kp1703), k32(itx32Km1703), k(itx16Kp1138), k(itx16Km1138)
	t21a := itx16Rot(t26, t21, p1703, m1138, s11)
	t26a := itx16Rot(t26, t21, p1138, p1703, s11)
	t22a := itx16Rot(t25, t22, m1138, m1703, s11)
	t25a := itx16Rot(t25, t22, p1703, m1138, s11)

	t16a := t16.AddSaturated(t19)
	t17 = t17a.AddSaturated(t18a)
	t18 = t17a.SubSaturated(t18a)
	t19a := t16.SubSaturated(t19)
	t20a := t23.SubSaturated(t20)
	t21 = t22a.SubSaturated(t21a)
	t22 = t22a.AddSaturated(t21a)
	t23a := t23.AddSaturated(t20)
	t24a := t24.AddSaturated(t27)
	t25 = t25a.AddSaturated(t26a)
	t26 = t25a.SubSaturated(t26a)
	t27a := t24.SubSaturated(t27)
	t28a := t31.SubSaturated(t28)
	t29 = t30a.SubSaturated(t29a)
	t30 = t30a.AddSaturated(t29a)
	t31a := t31.AddSaturated(t28)

	t18a = itx16Rot(t29, t18, p1567, m3784, s12)
	t29a = itx16Rot(t29, t18, p3784, p1567, s12)
	t19 = itx16Rot(t28a, t19a, p1567, m3784, s12)
	t28 = itx16Rot(t28a, t19a, p3784, p1567, s12)
	t20 = itx16Rot(t27a, t20a, m3784, m1567, s12)
	t27 = itx16Rot(t27a, t20a, p1567, m3784, s12)
	t21a = itx16Rot(t26, t21, m3784, m1567, s12)
	t26a = itx16Rot(t26, t21, p1567, m3784, s12)

	t16 = t16a.AddSaturated(t23a)
	t17a = t17.AddSaturated(t22)
	t18 = t18a.AddSaturated(t21a)
	t19a = t19.AddSaturated(t20)
	t20a = t19.SubSaturated(t20)
	t21 = t18a.SubSaturated(t21a)
	t22a = t17.SubSaturated(t22)
	t23 = t16a.SubSaturated(t23a)
	t24 = t31a.SubSaturated(t24a)
	t25a = t30.SubSaturated(t25)
	t26 = t29a.SubSaturated(t26a)
	t27a = t28.SubSaturated(t27)
	t28a = t28.AddSaturated(t27)
	t29 = t29a.AddSaturated(t26a)
	t30a = t30.AddSaturated(t25)
	t31 = t31a.AddSaturated(t24a)

	t20 = itx16Rot(t27a, t20a, p2896, m2896, s12)
	t27 = itx16Rot(t27a, t20a, p2896, p2896, s12)
	t21a = itx16Rot(t26, t21, p2896, m2896, s12)
	t26a = itx16Rot(t26, t21, p2896, p2896, s12)
	t22 = itx16Rot(t25a, t22a, p2896, m2896, s12)
	t25 = itx16Rot(t25a, t22a, p2896, p2896, s12)
	t23a = itx16Rot(t24, t23, p2896, m2896, s12)
	t24a = itx16Rot(t24, t23, p2896, p2896, s12)

	out[0], out[31] = E0.AddSaturated(t31), E0.SubSaturated(t31)
	out[1], out[30] = E1.AddSaturated(t30a), E1.SubSaturated(t30a)
	out[2], out[29] = E2.AddSaturated(t29), E2.SubSaturated(t29)
	out[3], out[28] = E3.AddSaturated(t28a), E3.SubSaturated(t28a)
	out[4], out[27] = E4.AddSaturated(t27), E4.SubSaturated(t27)
	out[5], out[26] = E5.AddSaturated(t26a), E5.SubSaturated(t26a)
	out[6], out[25] = E6.AddSaturated(t25), E6.SubSaturated(t25)
	out[7], out[24] = E7.AddSaturated(t24a), E7.SubSaturated(t24a)
	out[8], out[23] = E8.AddSaturated(t23a), E8.SubSaturated(t23a)
	out[9], out[22] = E9.AddSaturated(t22), E9.SubSaturated(t22)
	out[10], out[21] = E10.AddSaturated(t21a), E10.SubSaturated(t21a)
	out[11], out[20] = E11.AddSaturated(t20), E11.SubSaturated(t20)
	out[12], out[19] = E12.AddSaturated(t19a), E12.SubSaturated(t19a)
	out[13], out[18] = E13.AddSaturated(t18), E13.SubSaturated(t18)
	out[14], out[17] = E14.AddSaturated(t17a), E14.SubSaturated(t17a)
	out[15], out[16] = E15.AddSaturated(t16), E15.SubSaturated(t16)
}

// itx32Round2 is Round2(x, n) on int16 lanes as ((x >> (n-1)) + 1) >> 1,
// which equals (x + 2^(n-1)) >> n for every lane without overflowing.
func itx32Round2(x archsimd.Int16x8, shiftNm1, shift1 archsimd.Int16x8, one archsimd.Int16x8) archsimd.Int16x8 {
	return x.Shift(shiftNm1).Add(one).Shift(shift1)
}

// itx32RowStripLow8 runs the row pass of the 8-bit 32x32 DCT_DCT block
// whose coefficients lie in rows and columns 0..7: rows 0..7 of the
// column-major coefficients, narrowed to int16, transformed across vectors,
// rounded by 2, transposed and stored into tmp rows 0..7 (32 wide). It
// reports false, with tmp untouched, when a row fails the L1 guard, and
// otherwise also checks the column strips' guards on the stored values.
func itx32RowStripLow8(coeff []int32, stride int, tmp *[8 * 32]int16) bool {
	_ = coeff[7*stride+7]
	var in [8]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < 8; k++ {
		x := itx16LoadNarrow(coeff[k*stride : k*stride+8])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[k] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [32]archsimd.Int16x8
	itx32Low8Net(&in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	var colL1 archsimd.Uint16x8
	for g := 0; g < 4; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itx32Round2(out[8*g+j], sh1, sh1, one)
		}
		itx16Transpose8x8(&v)
		var gl1 archsimd.Uint16x8
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[r*32+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1 = colL1.Max(gl1)
	}
	return colL1.ReduceMax() <= itx16ColumnL1Bound
}

// itx32ColStripLow8Add runs the column pass for columns c0..c0+7 from tmp
// rows 0..7 and adds the result to the block: output row r is
// Round2(., 4) plus the prediction row, saturated to bytes, stored to dst.
func itx32ColStripLow8Add(tmp *[8 * 32]int16, c0 int, dst unsafe.Pointer, dstStride int, pred unsafe.Pointer, predStride int) {
	var in [8]archsimd.Int16x8
	for r := 0; r < 8; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*32+c0:]))
	}
	var out [32]archsimd.Int16x8
	itx32Low8Net(&in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	sh3 := archsimd.BroadcastInt16x8(-3)
	for r := 0; r < 32; r++ {
		res := itx32Round2(out[r], sh3, sh1, one)
		rows := [2]uint64{*(*uint64)(unsafe.Add(pred, r*predStride+c0)), 0}
		p := archsimd.LoadUint64x2Array(&rows).ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()
		*(*uint64)(unsafe.Add(dst, r*dstStride+c0)) = p.Add(res).SaturateToUint8().ReshapeToUint64s().GetElem(0)
	}
}

// InverseAddDCT32x32Low8 reconstructs one 8-bit 32x32 DCT_DCT block whose
// coded coefficients lie in rows and columns 0..7 (the caller's activeRows
// and activeCols are at most 8): dst = clip(pred + inverse(coeff)), with
// the dequantized coefficients column-major at coeffStride. pred may be dst.
// It reports false, with dst untouched, when the block is outside the
// kernel's contract or a guard declines; the caller then runs the staged
// path, which produces the same bytes.
func InverseAddDCT32x32Low8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int) bool {
	if coeffStride < 8 || len(coeff) < 7*coeffStride+8 || dstStride < 32 || predStride < 32 ||
		len(dst) < 31*dstStride+32 || len(pred) < 31*predStride+32 {
		return false
	}
	var tmp [8 * 32]int16
	if !itx32RowStripLow8(coeff, coeffStride, &tmp) {
		return false
	}
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	pp := unsafe.Pointer(unsafe.SliceData(pred))
	for g := 0; g < 4; g++ {
		itx32ColStripLow8Add(&tmp, 8*g, dp, dstStride, pp, predStride)
	}
	return true
}
