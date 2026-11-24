// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import "simd/archsimd"

// 8-bit self-guided filter passes in dav1d's 8bpc arm64 shape
// (src/arm/64/looprestoration_common.S sgr_box3_vert / sgr_box5_vert and
// looprestoration_tmpl.S sgr_finish_filter1/2): the box sums stay in
// 16-bit (sum) and 32-bit (sum of squares) lanes over the uint8 source, the
// x_by_x lookup runs as dav1d's three-table TBL + popcount form on sixteen
// lanes at once, A is kept as int16 and the 3x3 / fast 5x5 stencils use the
// factored weights
//
//	3x3:      4*cross + 3*diag     == 3*(Hp + Hn) + 4*Hc + Ap + An
//	5x5 even: 6*(Ap + An) + 5*diag == 5*(Hp + Hn) + Ap + An
//	5x5 odd:  6*Ac + 5*(Al + Ar)   == 5*Hc + Ac
//
// where H is the horizontal three-sum of a row of A (or B), computed once per
// row and reused by every output row that reads it. The A/B rows stream
// through a three-row ring, as dav1d's row pointers do, instead of a full
// unit-sized buffer.
//
// Every value is the one calculateIntermediateRolling and the selfguided
// blends produce: dav1d's x is SGRProjSgr - xByXPlus1[z] and its B term is
// x*sum*one_by_x, both computed here in the same uint32 arithmetic, and the
// stencil sums are exact in int16 (at most 32*256) and int32.

// sgrU8Cols is the column capacity of every per-row array: the widest box
// window is ProcUnitSize + 2 + 4 = 70 columns, processed in 16-column chunks
// with lane loads reaching at most 13 columns past the last chunk start.
const sgrU8Cols = 96

type sgrU8State struct {
	colSum [sgrU8Cols]uint16
	colSq  [sgrU8Cols]int32
	a      [3][sgrU8Cols]int16
	b      [3][sgrU8Cols]int32
	ha     [3][sgrU8Cols]int16
	hb     [3][sgrU8Cols]int32
	in     [sgrU8Cols]uint8
	out    [sgrU8Cols]uint8
	dgd    [sgrU8Cols]uint8
	flt    [sgrU8Cols]int32
}

// dav1d's x_by_x_tables: x(z) = RefLo[z] (z < 16) + RangeMins[z>>3] +
// popcount(DiffMasks[z>>3] << (z&7)), with x(z) == SGRProjSgr - xByXPlus1[z].
var (
	sgrXRangeMins = [32]uint8{0, 0, 11, 8, 6, 5, 5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0}
	sgrXDiffMasks = [32]uint8{
		0x00, 0x00, 0xD4, 0x44, 0x42, 0x04, 0x00, 0x00,
		0x00, 0x80, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
	}
	sgrXRefLo = [16]uint8{255, 128, 85, 64, 51, 43, 37, 32, 28, 26, 23, 21, 20, 18, 17, 16}
)

// sgrU8Passes runs the active self-guided passes of params over the bordered
// 8-bit unit (the ApplySelfguidedRestorationU8 layout) into flt0 (radius 2,
// fast) and flt1 (radius 1), each width x height at stride width. It reports
// false, writing nothing, for a parameter set outside the AV1 table shapes.
func sgrU8Passes(src []uint8, srcStride, srcOrigin, width, height int, params SGRParams, flt0, flt1 []int32) bool {
	if (params.Radius[0] != 0 && params.Radius[0] != 2) || (params.Radius[1] != 0 && params.Radius[1] != 1) ||
		width > ProcUnitSize || height > ProcUnitSize || len(flt0) < width*height || len(flt1) < width*height {
		return false
	}
	// Finish rows spill up to seven lanes into the next row; the bound keeps
	// the last row's spill inside the width x height block.
	flt0, flt1 = flt0[:width*height], flt1[:width*height]
	var st sgrU8State
	if params.Radius[0] != 0 {
		sgrU8Pass(&st, src, srcStride, srcOrigin, width, height, 2, uint32(params.S[0]), flt0)
	}
	if params.Radius[1] != 0 {
		if params.Radius[0] != 0 {
			st = sgrU8State{}
		}
		sgrU8Pass(&st, src, srcStride, srcOrigin, width, height, 1, uint32(params.S[1]), flt1)
	}
	return true
}

// sgrU8Pass is one filter pass: r == 1 is the full 3x3 filter, r == 2 the
// fast 5x5 filter whose A/B rows exist only at odd offsets (rows -1, 1, 3...).
func sgrU8Pass(st *sgrU8State, src []uint8, srcStride, srcOrigin, w, h, r int, s uint32, flt []int32) {
	span := w + 2 + 2*r
	left := srcOrigin - 1 - r
	// row returns source row row's box-window span, read in place when the
	// 16-byte chunks covering it stay inside src and copied into buf otherwise.
	reach := (span + 15) &^ 15
	row := func(buf *[sgrU8Cols]uint8, row int) []uint8 {
		o := left + row*srcStride
		if o+reach <= len(src) {
			return src[o : o+reach]
		}
		copy(buf[:span], src[o:o+span])
		return buf[:reach]
	}
	// Column sums over source rows y-r..y+r for A/B row y = -1.
	for y := -1 - r; y <= -1+r; y++ {
		sgrU8Roll(st, row(&st.in, y), nil, reach)
	}
	if r == 1 {
		for y := -1; y <= h; y++ {
			sgrU8RowAB(st, (y+1)%3, w, 9, s, uint32(oneByX[8]))
			if y >= 1 {
				sgrU8Finish3(st, src[srcOrigin+(y-1)*srcStride:], (y-1)%3, y%3, (y+1)%3, w, flt[(y-1)*w:])
			}
			if y < h {
				sgrU8Roll(st, row(&st.in, y+1+r), row(&st.out, y-r), reach)
			}
		}
		return
	}
	k := 0
	for y := -1; y < h+1; y += 2 {
		sgrU8RowAB(st, k%3, w, 25, s, uint32(oneByX[24]))
		if k > 0 {
			even := y - 1
			sgrU8Finish5Even(st, src[srcOrigin+even*srcStride:], (k-1)%3, k%3, w, flt[even*w:])
			if even+1 < h {
				sgrU8Finish5Odd(st, src[srcOrigin+(even+1)*srcStride:], k%3, w, flt[(even+1)*w:])
			}
		}
		if y+2 >= h+1 {
			break
		}
		for adv := 0; adv < 2; adv++ {
			sgrU8Roll(st, row(&st.in, y+r+1+adv), row(&st.out, y-r+adv), reach)
		}
		k++
	}
}

// sgrU8Roll adds in (and, when out is non-nil, removes out) to the column
// sums over reach columns: sum += in - out, sumsq += in*in - out*out ==
// (in-out)*(in+out). Lanes past the span carry unused values.
func sgrU8Roll(st *sgrU8State, inRow, outRow []uint8, reach int) {
	inRow = inRow[:reach]
	sub := outRow != nil
	if sub {
		outRow = outRow[:reach]
	}
	for c := 0; c < reach; c += 16 {
		in := archsimd.LoadUint8x16Array((*[16]uint8)(inRow[c : c+16]))
		var out archsimd.Uint8x16
		if sub {
			out = archsimd.LoadUint8x16Array((*[16]uint8)(outRow[c : c+16]))
		}
		inLo, inHi := in.ExtendLo8ToUint16(), in.HiToLo().ExtendLo8ToUint16()
		outLo, outHi := out.ExtendLo8ToUint16(), out.HiToLo().ExtendLo8ToUint16()
		cs := (*[16]uint16)(st.colSum[c : c+16])
		s0 := archsimd.LoadUint16x8Array((*[8]uint16)(cs[0:8]))
		s1 := archsimd.LoadUint16x8Array((*[8]uint16)(cs[8:16]))
		dLo, dHi := inLo.Sub(outLo), inHi.Sub(outHi)
		s0.Add(dLo).StoreArray((*[8]uint16)(cs[0:8]))
		s1.Add(dHi).StoreArray((*[8]uint16)(cs[8:16]))
		d0 := dLo.BitsToInt16()
		d1 := dHi.BitsToInt16()
		p0 := inLo.Add(outLo).BitsToInt16()
		p1 := inHi.Add(outHi).BitsToInt16()
		cq := (*[16]int32)(st.colSq[c : c+16])
		q0 := archsimd.LoadInt32x4Array((*[4]int32)(cq[0:4]))
		q1 := archsimd.LoadInt32x4Array((*[4]int32)(cq[4:8]))
		q2 := archsimd.LoadInt32x4Array((*[4]int32)(cq[8:12]))
		q3 := archsimd.LoadInt32x4Array((*[4]int32)(cq[12:16]))
		q0.Add(d0.MulWidenLo(p0)).StoreArray((*[4]int32)(cq[0:4]))
		q1.Add(d0.HiToLo().MulWidenLo(p0.HiToLo())).StoreArray((*[4]int32)(cq[4:8]))
		q2.Add(d1.MulWidenLo(p1)).StoreArray((*[4]int32)(cq[8:12]))
		q3.Add(d1.HiToLo().MulWidenLo(p1.HiToLo())).StoreArray((*[4]int32)(cq[12:16]))
	}
}

// sgrU8X is dav1d's x_by_x lookup on sixteen z values (each <= 255).
func sgrU8X(z archsimd.Uint8x16) archsimd.Uint8x16 {
	rmLo := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXRangeMins[0:16]))
	rmHi := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXRangeMins[16:32]))
	dmLo := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXDiffMasks[0:16]))
	dmHi := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXDiffMasks[16:32]))
	ref := archsimd.LoadUint8x16Array(&sgrXRefLo)
	hi := z.Shift(archsimd.BroadcastInt8x16(-3))
	hi16 := hi.Sub(archsimd.BroadcastUint8x16(16))
	mins := rmLo.LookupOrZero(hi).Or(rmHi.LookupOrZero(hi16))
	diffs := dmLo.LookupOrZero(hi).Or(dmHi.LookupOrZero(hi16))
	lo := z.And(archsimd.BroadcastUint8x16(7)).BitsToInt8()
	return ref.LookupOrZero(z).Add(mins).Add(diffs.Shift(lo).OnesCount())
}

func sgrU8LoadU16(p []uint16, i int) archsimd.Uint16x8 {
	return archsimd.LoadUint16x8Array((*[8]uint16)(p[i : i+8]))
}

func sgrU8LoadI32(p []int32, i int) archsimd.Int32x4 {
	return archsimd.LoadInt32x4Array((*[4]int32)(p[i : i+4]))
}

func sgrU8LoadI16(p []int16, i int) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p[i : i+8]))
}

// Box-window sums over a 32-column view at a constant offset o: after
// inlining every lane index is constant and in range of the view.
func sgrU8Sum3(cs *[32]uint16, o int) archsimd.Uint16x8 {
	return sgrU8LoadU16(cs[:], o).Add(sgrU8LoadU16(cs[:], o+1)).Add(sgrU8LoadU16(cs[:], o+2))
}

func sgrU8Sum5(cs *[32]uint16, o int) archsimd.Uint16x8 {
	return sgrU8Sum3(cs, o).Add(sgrU8LoadU16(cs[:], o+3)).Add(sgrU8LoadU16(cs[:], o+4))
}

func sgrU8Sq3(cq *[32]int32, o int) archsimd.Uint32x4 {
	return sgrU8LoadI32(cq[:], o).Add(sgrU8LoadI32(cq[:], o+1)).Add(sgrU8LoadI32(cq[:], o+2)).ToBits()
}

func sgrU8Sq5(cq *[32]int32, o int) archsimd.Uint32x4 {
	return sgrU8LoadI32(cq[:], o).Add(sgrU8LoadI32(cq[:], o+1)).Add(sgrU8LoadI32(cq[:], o+2)).
		Add(sgrU8LoadI32(cq[:], o+3)).Add(sgrU8LoadI32(cq[:], o+4)).ToBits()
}

// sgrU8RowAB writes A (int16), B (int32) and their horizontal three-sums for
// the A/B row whose box windows are the current column sums, into ring slot.
// A/B column index 0 is unit column -1. z = min(round((sumsq*n - sum*sum)*s,
// 20), 255) and B use calculateIntermediateRolling's uint32 arithmetic.
func sgrU8RowAB(st *sgrU8State, slot, w int, n, s, recip uint32) {
	r2 := n == 25
	vn := archsimd.BroadcastUint32x4(n)
	vs := archsimd.BroadcastUint32x4(s)
	vRnd := archsimd.BroadcastUint32x4(1 << (SGRProjMTableBits - 1))
	shl4 := archsimd.BroadcastInt16x8(SGRProjMTableBits - 16)
	vRecip := archsimd.BroadcastUint32x4(recip)
	// B carries sgrU8BBias on top of its rounding: every finish stencil
	// weighs B by 32 (or 16 for the fast filter's odd rows), so the bias
	// lands as exactly the finish rounding term.
	vbRnd := archsimd.BroadcastUint32x4(1<<(SGRProjRecipBits-1) + sgrU8BBias<<SGRProjRecipBits)
	vbSh := archsimd.BroadcastInt32x4(-SGRProjRecipBits)
	v256 := archsimd.BroadcastUint16x8(SGRProjSgr)
	rmLo := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXRangeMins[0:16]))
	rmHi := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXRangeMins[16:32]))
	dmLo := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXDiffMasks[0:16]))
	dmHi := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXDiffMasks[16:32]))
	refLo := archsimd.LoadUint8x16Array(&sgrXRefLo)
	shr3 := archsimd.BroadcastInt8x16(-3)
	k16 := archsimd.BroadcastUint8x16(16)
	k7 := archsimd.BroadcastUint8x16(7)
	ar := st.a[slot][:]
	br := st.b[slot][:]
	for j := 0; j < w+2; j += 16 {
		cs := (*[32]uint16)(st.colSum[j : j+32])
		cq := (*[32]int32)(st.colSq[j : j+32])
		var sum0, sum1 archsimd.Uint16x8
		var q0, q1, q2, q3 archsimd.Uint32x4
		if r2 {
			sum0, sum1 = sgrU8Sum5(cs, 0), sgrU8Sum5(cs, 8)
			q0, q1, q2, q3 = sgrU8Sq5(cq, 0), sgrU8Sq5(cq, 4), sgrU8Sq5(cq, 8), sgrU8Sq5(cq, 12)
		} else {
			sum0, sum1 = sgrU8Sum3(cs, 0), sgrU8Sum3(cs, 8)
			q0, q1, q2, q3 = sgrU8Sq3(cq, 0), sgrU8Sq3(cq, 4), sgrU8Sq3(cq, 8), sgrU8Sq3(cq, 12)
		}
		h0, h1 := sum0.HiToLo(), sum1.HiToLo()
		// z = min((p*s + 2^19) >> 20, 255) in uint32 wrap arithmetic, narrowed
		// as dav1d's uqshrn #16 / uqrshrn #4 pair does: the high halves of the
		// biased products (UZP2) are floor(v / 2^16); a saturating << 4 leaves
		// floor(v / 2^20) clamped to 255 in each high byte (UZP2 again).
		z0 := q0.Mul(vn).SubSaturated(sum0.MulWidenLo(sum0)).Mul(vs).Add(vRnd)
		z1 := q1.Mul(vn).SubSaturated(h0.MulWidenLo(h0)).Mul(vs).Add(vRnd)
		z2 := q2.Mul(vn).SubSaturated(sum1.MulWidenLo(sum1)).Mul(vs).Add(vRnd)
		z3 := q3.Mul(vn).SubSaturated(h1.MulWidenLo(h1)).Mul(vs).Add(vRnd)
		zA := z0.ReshapeToUint16s().ConcatOdd(z1.ReshapeToUint16s()).ShiftSaturated(shl4)
		zB := z2.ReshapeToUint16s().ConcatOdd(z3.ReshapeToUint16s()).ShiftSaturated(shl4)
		z := zA.ReshapeToUint8s().ConcatOdd(zB.ReshapeToUint8s())
		// dav1d x_by_x: RefLo[z] + RangeMins[z>>3] + popcount(DiffMasks[z>>3] << (z&7)).
		hi := z.Shift(shr3)
		hi16 := hi.Sub(k16)
		mins := rmLo.LookupOrZero(hi).Or(rmHi.LookupOrZero(hi16))
		diffs := dmLo.LookupOrZero(hi).Or(dmHi.LookupOrZero(hi16))
		x := refLo.LookupOrZero(z).Add(mins).Add(diffs.Shift(z.And(k7).BitsToInt8()).OnesCount())
		x0 := x.ExtendLo8ToUint16()
		x1 := x.HiToLo().ExtendLo8ToUint16()
		ao := (*[16]int16)(ar[j : j+16])
		v256.Sub(x0).BitsToInt16().StoreArray((*[8]int16)(ao[0:8]))
		v256.Sub(x1).BitsToInt16().StoreArray((*[8]int16)(ao[8:16]))
		bo := (*[16]int32)(br[j : j+16])
		x0.MulWidenLo(sum0).MulAdd(vRecip, vbRnd).Shift(vbSh).BitsToInt32().StoreArray((*[4]int32)(bo[0:4]))
		x0.HiToLo().MulWidenLo(h0).MulAdd(vRecip, vbRnd).Shift(vbSh).BitsToInt32().StoreArray((*[4]int32)(bo[4:8]))
		x1.MulWidenLo(sum1).MulAdd(vRecip, vbRnd).Shift(vbSh).BitsToInt32().StoreArray((*[4]int32)(bo[8:12]))
		x1.HiToLo().MulWidenLo(h1).MulAdd(vRecip, vbRnd).Shift(vbSh).BitsToInt32().StoreArray((*[4]int32)(bo[12:16]))
	}
	ha := st.ha[slot][:]
	hb := st.hb[slot][:]
	for c := 0; c < w; c += 8 {
		a := (*[16]int16)(ar[c : c+16])
		b := (*[16]int32)(br[c : c+16])
		sgrU8LoadI16(a[:], 0).Add(sgrU8LoadI16(a[:], 1)).Add(sgrU8LoadI16(a[:], 2)).StoreArray((*[8]int16)(ha[c : c+8]))
		hbo := (*[8]int32)(hb[c : c+8])
		sgrU8LoadI32(b[:], 0).Add(sgrU8LoadI32(b[:], 1)).Add(sgrU8LoadI32(b[:], 2)).StoreArray((*[4]int32)(hbo[0:4]))
		sgrU8LoadI32(b[:], 4).Add(sgrU8LoadI32(b[:], 5)).Add(sgrU8LoadI32(b[:], 6)).StoreArray((*[4]int32)(hbo[4:8]))
	}
}

// sgrU8BBias is added to every stored B value: 8 * 32 == 1 << 8 and
// 8 * 16 == 1 << 7 are the rounding terms of the finish shifts below.
const sgrU8BBias = 8

// sgrU8Out finishes eight output columns: (a*dgd + b) >> shift, the
// rounding bias riding in b.
func sgrU8Out(o *[8]int32, a archsimd.Int16x8, bLo, bHi archsimd.Int32x4, d archsimd.Int16x8, sh archsimd.Int32x4) {
	a.MulWidenLo(d).Add(bLo).Shift(sh).StoreArray((*[4]int32)(o[0:4]))
	a.HiToLo().MulWidenLo(d.HiToLo()).Add(bHi).Shift(sh).StoreArray((*[4]int32)(o[4:8]))
}

// sgrU8Rows prepares a finish row: the source row (read in place when the
// 16-byte loads stay inside it, else copied to st.dgd) and the output row
// (written in place when the 8-lane stores stay inside flt, else staged in
// st.flt and copied by sgrU8Flush). Columns past w spill into the next row,
// which is finished later.
func sgrU8Rows(st *sgrU8State, row []uint8, flt []int32, w int) ([]uint8, []int32) {
	n := (w + 7) &^ 7
	if len(row) < n+8 {
		copy(st.dgd[:w], row[:w])
		row = st.dgd[:]
	}
	if len(flt) < n {
		return row[:n+8], st.flt[:n]
	}
	return row[:n+8], flt[:n]
}

func sgrU8Flush(st *sgrU8State, out, flt []int32, w int) {
	if &out[0] == &st.flt[0] {
		copy(flt[:w], st.flt[:w])
	}
}

func sgrU8Dgd(row []uint8, c int) archsimd.Int16x8 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(row[c : c+16])).ExtendLo8ToUint16().BitsToInt16()
}

// sgrU8Finish3 is the 3x3 output row between ring slots p (above), m and n
// (below); row holds the unit's source row from column 0.
func sgrU8Finish3(st *sgrU8State, row []uint8, p, m, n, w int, flt []int32) {
	row, out := sgrU8Rows(st, row, flt, w)
	sh := archsimd.BroadcastInt32x4(-(SGRProjSgrBits + 5 - SGRProjRstBits))
	three16, three32 := archsimd.BroadcastInt16x8(3), archsimd.BroadcastInt32x4(3)
	haP, haM, haN := st.ha[p][:], st.ha[m][:], st.ha[n][:]
	hbP, hbM, hbN := st.hb[p][:], st.hb[m][:], st.hb[n][:]
	aP, aN := st.a[p][:], st.a[n][:]
	bP, bN := st.b[p][:], st.b[n][:]
	for c := 0; c < len(out); c += 8 {
		// 3*(Hp + Hn) + 4*Hc + Ap + An == 3*(Hp + Hc + Hn) + (Hc + Ap + An).
		haC, hbC0, hbC1 := sgrU8LoadI16(haM, c), sgrU8LoadI32(hbM, c), sgrU8LoadI32(hbM, c+4)
		a := sgrU8LoadI16(haP, c).Add(sgrU8LoadI16(haN, c)).Add(haC).
			MulAdd(three16, haC.Add(sgrU8LoadI16(aP, c+1)).Add(sgrU8LoadI16(aN, c+1)))
		bLo := sgrU8LoadI32(hbP, c).Add(sgrU8LoadI32(hbN, c)).Add(hbC0).
			MulAdd(three32, hbC0.Add(sgrU8LoadI32(bP, c+1)).Add(sgrU8LoadI32(bN, c+1)))
		bHi := sgrU8LoadI32(hbP, c+4).Add(sgrU8LoadI32(hbN, c+4)).Add(hbC1).
			MulAdd(three32, hbC1.Add(sgrU8LoadI32(bP, c+5)).Add(sgrU8LoadI32(bN, c+5)))
		sgrU8Out((*[8]int32)(out[c:c+8]), a, bLo, bHi, sgrU8Dgd(row, c), sh)
	}
	sgrU8Flush(st, out, flt, w)
}

// sgrU8Finish5Even is the fast filter's even output row between the A/B
// rows in slots p (above) and n (below).
func sgrU8Finish5Even(st *sgrU8State, row []uint8, p, n, w int, flt []int32) {
	row, out := sgrU8Rows(st, row, flt, w)
	sh := archsimd.BroadcastInt32x4(-(SGRProjSgrBits + 5 - SGRProjRstBits))
	five16, five32 := archsimd.BroadcastInt16x8(5), archsimd.BroadcastInt32x4(5)
	haP, haN := st.ha[p][:], st.ha[n][:]
	hbP, hbN := st.hb[p][:], st.hb[n][:]
	aP, aN := st.a[p][:], st.a[n][:]
	bP, bN := st.b[p][:], st.b[n][:]
	for c := 0; c < len(out); c += 8 {
		a := sgrU8LoadI16(haP, c).Add(sgrU8LoadI16(haN, c)).
			MulAdd(five16, sgrU8LoadI16(aP, c+1).Add(sgrU8LoadI16(aN, c+1)))
		bLo := sgrU8LoadI32(hbP, c).Add(sgrU8LoadI32(hbN, c)).
			MulAdd(five32, sgrU8LoadI32(bP, c+1).Add(sgrU8LoadI32(bN, c+1)))
		bHi := sgrU8LoadI32(hbP, c+4).Add(sgrU8LoadI32(hbN, c+4)).
			MulAdd(five32, sgrU8LoadI32(bP, c+5).Add(sgrU8LoadI32(bN, c+5)))
		sgrU8Out((*[8]int32)(out[c:c+8]), a, bLo, bHi, sgrU8Dgd(row, c), sh)
	}
	sgrU8Flush(st, out, flt, w)
}

// sgrU8Finish5Odd is the fast filter's odd output row on the A/B row in slot m.
func sgrU8Finish5Odd(st *sgrU8State, row []uint8, m, w int, flt []int32) {
	row, out := sgrU8Rows(st, row, flt, w)
	sh := archsimd.BroadcastInt32x4(-(SGRProjSgrBits + 4 - SGRProjRstBits))
	five16, five32 := archsimd.BroadcastInt16x8(5), archsimd.BroadcastInt32x4(5)
	haM, hbM := st.ha[m][:], st.hb[m][:]
	aM, bM := st.a[m][:], st.b[m][:]
	for c := 0; c < len(out); c += 8 {
		a := sgrU8LoadI16(haM, c).MulAdd(five16, sgrU8LoadI16(aM, c+1))
		bLo := sgrU8LoadI32(hbM, c).MulAdd(five32, sgrU8LoadI32(bM, c+1))
		bHi := sgrU8LoadI32(hbM, c+4).MulAdd(five32, sgrU8LoadI32(bM, c+5))
		sgrU8Out((*[8]int32)(out[c:c+8]), a, bLo, bHi, sgrU8Dgd(row, c), sh)
	}
	sgrU8Flush(st, out, flt, w)
}
