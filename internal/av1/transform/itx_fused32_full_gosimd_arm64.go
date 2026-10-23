// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// itx_fused32_full: the fused 32x32 path (itx_fused32) for blocks whose
// coded coefficients reach past the 16x16 corner. The row pass runs up to
// four eight-row strips with the DCT32 network form matching the coded
// columns (Low8, Low16 or the full form below) into a 32x32 int16 tmp; the
// four column strips run the form matching the coded rows with the add
// fused.

// itx32FullNet is inverseDCT32 on eight lanes for all 32 inputs: in[k] is
// coefficient k, out[r] is output r. It is itx32Low16Net with the DCT16 even
// half in its full form and the odd first stage as rotation pairs.
func itx32FullNet(in *[32]archsimd.Int16x8, out *[32]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx16K[i]) }
	k32 := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx32K[i]) }
	c := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	cn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	s11 := archsimd.LoadInt32x4Array(&itx16Scale11)
	p2896, m2896 := k(itx16Kp2896), k(itx16Km2896)
	p1567, m1567, p3784, m3784 := k(itx16Kp1567), k(itx16Km1567), k(itx16Kp3784), k(itx16Km3784)

	// Even half: inverseDCT16 on inputs 0, 2, ..., 30.
	var e [16]archsimd.Int16x8
	for i := 0; i < 16; i++ {
		e[i] = in[2*i]
	}
	var E [16]archsimd.Int16x8
	itx16NetFull(&e, &E)

	// Odd half: rotation pairs (cospi index i is cospi(i) = itx64Cos[i]).
	in1, in3, in5, in7 := in[1], in[3], in[5], in[7]
	in9, in11, in13, in15 := in[9], in[11], in[13], in[15]
	in17, in19, in21, in23 := in[17], in[19], in[21], in[23]
	in25, in27, in29, in31 := in[25], in[27], in[29], in[31]
	t16a := itx16Rot(in1, in31, c(62), cn(2), s12)
	t31a := itx16Rot(in1, in31, c(2), c(62), s12)
	t17a := itx16Rot(in17, in15, c(30), cn(34), s12)
	t30a := itx16Rot(in17, in15, c(34), c(30), s12)
	t18a := itx16Rot(in9, in23, c(46), cn(18), s12)
	t29a := itx16Rot(in9, in23, c(18), c(46), s12)
	t19a := itx16Rot(in25, in7, c(14), cn(50), s12)
	t28a := itx16Rot(in25, in7, c(50), c(14), s12)
	t20a := itx16Rot(in5, in27, c(54), cn(10), s12)
	t27a := itx16Rot(in5, in27, c(10), c(54), s12)
	t21a := itx16Rot(in21, in11, c(22), cn(42), s12)
	t26a := itx16Rot(in21, in11, c(42), c(22), s12)
	t22a := itx16Rot(in13, in19, c(38), cn(26), s12)
	t25a := itx16Rot(in13, in19, c(26), c(38), s12)
	t23a := itx16Rot(in29, in3, c(6), cn(58), s12)
	t24a := itx16Rot(in29, in3, c(58), c(6), s12)

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

	out[0], out[31] = E[0].AddSaturated(t31), E[0].SubSaturated(t31)
	out[1], out[30] = E[1].AddSaturated(t30a), E[1].SubSaturated(t30a)
	out[2], out[29] = E[2].AddSaturated(t29), E[2].SubSaturated(t29)
	out[3], out[28] = E[3].AddSaturated(t28a), E[3].SubSaturated(t28a)
	out[4], out[27] = E[4].AddSaturated(t27), E[4].SubSaturated(t27)
	out[5], out[26] = E[5].AddSaturated(t26a), E[5].SubSaturated(t26a)
	out[6], out[25] = E[6].AddSaturated(t25), E[6].SubSaturated(t25)
	out[7], out[24] = E[7].AddSaturated(t24a), E[7].SubSaturated(t24a)
	out[8], out[23] = E[8].AddSaturated(t23a), E[8].SubSaturated(t23a)
	out[9], out[22] = E[9].AddSaturated(t22), E[9].SubSaturated(t22)
	out[10], out[21] = E[10].AddSaturated(t21a), E[10].SubSaturated(t21a)
	out[11], out[20] = E[11].AddSaturated(t20), E[11].SubSaturated(t20)
	out[12], out[19] = E[12].AddSaturated(t19a), E[12].SubSaturated(t19a)
	out[13], out[18] = E[13].AddSaturated(t18), E[13].SubSaturated(t18)
	out[14], out[17] = E[14].AddSaturated(t17a), E[14].SubSaturated(t17a)
	out[15], out[16] = E[15].AddSaturated(t16), E[15].SubSaturated(t16)
}

// itx32Net runs the DCT32 network form for n nonzero inputs (8, 16 or 32).
func itx32Net(n int, in *[32]archsimd.Int16x8, out *[32]archsimd.Int16x8) {
	switch {
	case n <= 8:
		itx32Low8Net((*[8]archsimd.Int16x8)(in[:8]), out)
	case n <= 16:
		itx32Low16Net((*[16]archsimd.Int16x8)(in[:16]), out)
	default:
		itx32FullNet(in, out)
	}
}

// itx32RowStrip runs the 32x32 row pass for rows r0..r0+7 with inCols coded
// coefficient columns (8, 16 or 32) into tmp rows r0..r0+7 and adds the
// stored values' magnitudes to the four column guards. It reports false,
// with tmp untouched, when a row fails the L1 guard.
func itx32RowStrip(coeff []int32, stride int, r0 int, inCols int, tmp *[32 * 32]int16, colL1 *[4]archsimd.Uint16x8) bool {
	_ = coeff[(inCols-1)*stride+r0+7]
	var in [32]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < inCols; k++ {
		x := itx16LoadNarrow(coeff[k*stride+r0 : k*stride+r0+8])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[k] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [32]archsimd.Int16x8
	itx32Net(inCols, &in, &out)
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

// itx32ColStripAdd runs the 32x32 column pass for columns c0..c0+7 from
// inRows coded tmp rows (8, 16 or 32) and adds the result to the block.
func itx32ColStripAdd(tmp *[32 * 32]int16, c0 int, inRows int, dst unsafe.Pointer, dstStride int, pred unsafe.Pointer, predStride int) {
	var in [32]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*32+c0:]))
	}
	var out [32]archsimd.Int16x8
	itx32Net(inRows, &in, &out)
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

// InverseAddDCT32x32Full reconstructs one 8-bit 32x32 DCT_DCT block with
// coded coefficients in rows 0..activeRows-1 and columns 0..activeCols-1,
// whichever network forms that takes: dst = clip(pred + inverse(coeff)),
// with the dequantized coefficients column-major at coeffStride. pred may
// be dst. It reports false, with dst untouched, when the block is outside
// the kernel's contract or a guard declines; the caller then runs the
// staged path, which produces the same bytes. Blocks within the 8x8 or
// 16x16 corner are served faster by InverseAddDCT32x32Low8 and Low16.
func InverseAddDCT32x32Full(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 32 || activeCols <= 0 || activeCols > 32 ||
		coeffStride < 32 || len(coeff) < 31*coeffStride+32 || dstStride < 32 || predStride < 32 ||
		len(dst) < 31*dstStride+32 || len(pred) < 31*predStride+32 {
		return false
	}
	inCols := itx64Form(activeCols)
	inRows := itx64Form(activeRows)
	var tmp [32 * 32]int16
	var colL1 [4]archsimd.Uint16x8
	written := 0
	for ; written < activeRows; written += 8 {
		if !itx32RowStrip(coeff, coeffStride, written, inCols, &tmp, &colL1) {
			return false
		}
	}
	// The column form may read tmp rows past the strips that ran; those
	// rows are the transform of zero coefficient rows, so zero.
	if written < inRows {
		clear(tmp[written*32 : inRows*32])
	}
	if colL1[0].Max(colL1[1]).Max(colL1[2].Max(colL1[3])).ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	pp := unsafe.Pointer(unsafe.SliceData(pred))
	for c0 := 0; c0 < 32; c0 += 8 {
		itx32ColStripAdd(&tmp, c0, inRows, dp, dstStride, pp, predStride)
	}
	return true
}
