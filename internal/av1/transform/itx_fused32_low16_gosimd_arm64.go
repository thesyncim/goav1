// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// The Low16 form of itx_fused32: the 8-bit 32x32 DCT_DCT block whose coded
// coefficients lie in rows and columns 0..15. Two row strips of eight rows
// each read coefficients 0..15, the column strips read sixteen tmp rows, and
// the network's first stage has single-product rotations for every odd
// input (inputs 17..31 are zero); everything else is the Low8 form's.

// itx32K16 holds the odd-half rotation constants the Low16 first stage needs
// beyond itx32K.
var itx32K16 = [...][8]int16{
	{-2751, -2751, -2751, -2751, -2751, -2751, -2751, -2751},
	{3035, 3035, 3035, 3035, 3035, 3035, 3035, 3035},
	{1751, 1751, 1751, 1751, 1751, 1751, 1751, 1751},
	{3703, 3703, 3703, 3703, 3703, 3703, 3703, 3703},
	{-2106, -2106, -2106, -2106, -2106, -2106, -2106, -2106},
	{3513, 3513, 3513, 3513, 3513, 3513, 3513, 3513},
	{1220, 1220, 1220, 1220, 1220, 1220, 1220, 1220},
	{1645, 1645, 1645, 1645, 1645, 1645, 1645, 1645},
}

const (
	itx32K16m2751 = iota
	itx32K16p3035
	itx32K16p1751
	itx32K16p3703
	itx32K16m2106
	itx32K16p3513
	itx32K16p1220
	itx32K16p1645
)

// itx32Low16Net is inverseDCT32 on eight lanes for inputs whose indices
// 16..31 are zero: in[k] is coefficient k, out[r] is output r.
func itx32Low16Net(in *[16]archsimd.Int16x8, out *[32]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	k32 := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx32K[i]) }
	k16 := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx32K16[i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, m1567, p3784, m3784 := k(itx16Kp1567), k(itx16Km1567), k(itx16Kp3784), k(itx16Km3784)

	// Even half: inverseDCT16 on inputs 0, 2, ..., 14 (its Low8 form with
	// DCT16 inputs 0..7 = DCT32 inputs 0, 2, ..., 14).
	e0i, e1i, e2i, e3i := in[0], in[2], in[4], in[6]
	e4i, e5i, e6i, e7i := in[8], in[10], in[12], in[14]
	t0 := itx16Rot1(e0i, p2896, s12)
	t2 := itx16Rot1(e4i, p1567, s12)
	t3 := itx16Rot1(e4i, p3784, s12)
	d0 := t0.AddSaturated(t3)
	d1 := t0.AddSaturated(t2)
	d2 := t0.SubSaturated(t2)
	d3 := t0.SubSaturated(t3)
	t4a := itx16Rot1(e2i, k(itx16Kp799), s12)
	t5a := itx16Rot1(e6i, k(itx16Km1138), s11)
	t6a := itx16Rot1(e6i, k(itx16Kp1703), s11)
	t7a := itx16Rot1(e2i, k(itx16Kp4017), s12)
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
	t8a := itx16Rot1(e1i, k(itx16Kp401), s12)
	t15a := itx16Rot1(e1i, k(itx16Kp4076), s12)
	t9a := itx16Rot1(e7i, k(itx16Km1299), s11)
	t14a := itx16Rot1(e7i, k(itx16Kp1583), s11)
	t10a := itx16Rot1(e5i, k(itx16Kp1931), s12)
	t13a := itx16Rot1(e5i, k(itx16Kp3612), s12)
	t11a := itx16Rot1(e3i, k(itx16Km1189), s12)
	t12a := itx16Rot1(e3i, k(itx16Kp3920), s12)
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
	t10a = itx16Rot(t13, t10, m3784, m1567, s12)
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
	E0, E15 := e0.AddSaturated(t15a), e0.SubSaturated(t15a)
	E1, E14 := e1.AddSaturated(t14), e1.SubSaturated(t14)
	E2, E13 := e2.AddSaturated(t13a), e2.SubSaturated(t13a)
	E3, E12 := e3.AddSaturated(t12), e3.SubSaturated(t12)
	E4, E11 := e4.AddSaturated(t11), e4.SubSaturated(t11)
	E5, E10 := e5.AddSaturated(t10a), e5.SubSaturated(t10a)
	E6, E9 := e6.AddSaturated(t9), e6.SubSaturated(t9)
	E7, E8 := e7.AddSaturated(t8a), e7.SubSaturated(t8a)

	// Odd half from inputs 1, 3, ..., 15: single-product first stage.
	in1, in3, in5, in7 := in[1], in[3], in[5], in[7]
	in9, in11, in13, in15 := in[9], in[11], in[13], in[15]
	t16a := itx16Rot1(in1, k32(itx32Kp201), s12)
	t31a := itx16Rot1(in1, k32(itx32Kp4091), s12)
	t17a := itx16Rot1(in15, k16(itx32K16m2751), s12)
	t30a := itx16Rot1(in15, k16(itx32K16p3035), s12)
	t18a := itx16Rot1(in9, k16(itx32K16p1751), s12)
	t29a := itx16Rot1(in9, k16(itx32K16p3703), s12)
	t19a := itx16Rot1(in7, k32(itx32Km1380), s12)
	t28a := itx16Rot1(in7, k32(itx32Kp3857), s12)
	t20a := itx16Rot1(in5, k32(itx32Kp995), s12)
	t27a := itx16Rot1(in5, k32(itx32Kp3973), s12)
	t21a := itx16Rot1(in11, k16(itx32K16m2106), s12)
	t26a := itx16Rot1(in11, k16(itx32K16p3513), s12)
	t22a := itx16Rot1(in13, k16(itx32K16p1220), s11)
	t25a := itx16Rot1(in13, k16(itx32K16p1645), s11)
	t23a := itx16Rot1(in3, k32(itx32Km601), s12)
	t24a := itx16Rot1(in3, k32(itx32Kp4052), s12)

	t16 := t16a.AddSaturated(t17a)
	t17 := t16a.SubSaturated(t17a)
	t18 := t19a.SubSaturated(t18a)
	t19 := t19a.AddSaturated(t18a)
	t20 := t20a.AddSaturated(t21a)
	t21 := t20a.SubSaturated(t21a)
	t22 := t23a.SubSaturated(t22a)
	t23 := t23a.AddSaturated(t22a)
	t24 := t24a.AddSaturated(t25a)
	t25 := t24a.SubSaturated(t25a)
	t26 := t27a.SubSaturated(t26a)
	t27 := t27a.AddSaturated(t26a)
	t28 := t28a.AddSaturated(t29a)
	t29 := t28a.SubSaturated(t29a)
	t30 := t31a.SubSaturated(t30a)
	t31 := t31a.AddSaturated(t30a)

	p799, m799, p4017, m4017 := k(itx16Kp799), k32(itx32Km799), k(itx16Kp4017), k(itx16Km4017)
	t17a = itx16Rot(t30, t17, p799, m4017, s12)
	t30a = itx16Rot(t30, t17, p4017, p799, s12)
	t18a = itx16Rot(t29, t18, m4017, m799, s12)
	t29a = itx16Rot(t29, t18, p799, m4017, s12)
	p1703, m1703, p1138, m1138 := k(itx16Kp1703), k32(itx32Km1703), k(itx16Kp1138), k(itx16Km1138)
	t21a = itx16Rot(t26, t21, p1703, m1138, s11)
	t26a = itx16Rot(t26, t21, p1138, p1703, s11)
	t22a = itx16Rot(t25, t22, m1138, m1703, s11)
	t25a = itx16Rot(t25, t22, p1703, m1138, s11)

	t16a = t16.AddSaturated(t19)
	t17 = t17a.AddSaturated(t18a)
	t18 = t17a.SubSaturated(t18a)
	t19a = t16.SubSaturated(t19)
	t20a = t23.SubSaturated(t20)
	t21 = t22a.SubSaturated(t21a)
	t22 = t22a.AddSaturated(t21a)
	t23a = t23.AddSaturated(t20)
	t24a = t24.AddSaturated(t27)
	t25 = t25a.AddSaturated(t26a)
	t26 = t25a.SubSaturated(t26a)
	t27a = t24.SubSaturated(t27)
	t28a = t31.SubSaturated(t28)
	t29 = t30a.SubSaturated(t29a)
	t30 = t30a.AddSaturated(t29a)
	t31a = t31.AddSaturated(t28)

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

// itx32RowStripLow16 runs the row pass for rows r0..r0+7 of a block whose
// coefficients lie in rows and columns 0..15, into tmp rows r0..r0+7, and
// adds the stored values' magnitudes to the column guards colL1[g] (strip g
// of eight columns). It reports false, with tmp untouched, when a row
// fails the L1 guard.
func itx32RowStripLow16(coeff []int32, stride int, r0 int, tmp *[16 * 32]int16, colL1 *[4]archsimd.Uint16x8) bool {
	_ = coeff[15*stride+r0+7]
	var in [16]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < 16; k++ {
		x := itx16LoadNarrow(coeff[k*stride+r0 : k*stride+r0+8])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[k] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [32]archsimd.Int16x8
	itx32Low16Net(&in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	for g := 0; g < 4; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itx32Round2(out[8*g+j], sh1, sh1, one)
		}
		itx16Transpose8x8(&v)
		gl1 := colL1[g]
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[(r0+r)*32+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1[g] = gl1
	}
	return true
}

// itx32ColStripLow16Add runs the column pass for columns c0..c0+7 from tmp
// rows 0..15 and adds the result to the block (see itx32ColStripLow8Add).
func itx32ColStripLow16Add(tmp *[16 * 32]int16, c0 int, dst unsafe.Pointer, dstStride int, pred unsafe.Pointer, predStride int) {
	var in [16]archsimd.Int16x8
	for r := 0; r < 16; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*32+c0:]))
	}
	var out [32]archsimd.Int16x8
	itx32Low16Net(&in, &out)
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

// InverseAddDCT32x32Low16 is InverseAddDCT32x32Low8 for a block whose coded
// coefficients lie in rows and columns 0..15 (activeRows and activeCols at
// most 16); activeRows tells it whether coefficient rows 8..15 hold
// anything, so a block with eight coded rows but up to sixteen coded
// columns skips its second row strip.
func InverseAddDCT32x32Low16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows int) bool {
	if coeffStride < 16 || len(coeff) < 15*coeffStride+16 || dstStride < 32 || predStride < 32 ||
		len(dst) < 31*dstStride+32 || len(pred) < 31*predStride+32 {
		return false
	}
	var tmp [16 * 32]int16
	var colL1 [4]archsimd.Uint16x8
	if !itx32RowStripLow16(coeff, coeffStride, 0, &tmp, &colL1) {
		return false
	}
	if activeRows > 8 {
		if !itx32RowStripLow16(coeff, coeffStride, 8, &tmp, &colL1) {
			return false
		}
	} else {
		clear(tmp[8*32:])
	}
	l1 := colL1[0].Max(colL1[1]).Max(colL1[2].Max(colL1[3]))
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	pp := unsafe.Pointer(unsafe.SliceData(pred))
	for g := 0; g < 4; g++ {
		itx32ColStripLow16Add(&tmp, 8*g, dp, dstStride, pp, predStride)
	}
	return true
}
