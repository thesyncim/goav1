// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for upstream attribution.

//go:build goexperiment.simd && amd64 && !purego

package cdef

// 8-bit CDEF direction search on AVX2, ported from dav1d's cdef_dir_8bpc
// (src/x86/cdef_avx2.asm). The eight rows are widened to words and paired
// with their mirror rows, m_i = [row i | row 7-i] for i < 4, and the
// half-swapped copies give m_i for i >= 4. Every partial-sum family is then
// built for two directions at once: the same lane-shift aggregation that
// yields an anti-diagonal (or (i>>1)-odd) family in the low 128-bit lane
// yields its mirror family in the high lane, so the low/high lanes produce
// cost0|cost4, cost7|cost5 and cost1|cost3, and one more vector holds the
// column (cost6) and row (cost2) sums. Squares and the libaom div_table
// weights are pmaddwd + pmulld on interleaved (k, 14-k) / (k, 10-k) pairs,
// and three phaddd rounds reduce the eight costs into one ymm.
//
// The mirrored families index their partials k -> 14-k (or 10-k), and every
// cost weight is symmetric under that map, so the costs are exactly the
// scalar ones. The argmax keeps the scalar `>` scan (lowest index wins ties)
// and variance = (best - cost[dir^4]) >> 10, as finishDirection.

import (
	"simd/archsimd"
	"unsafe"
)

var (
	// cdefDirShufW6543210x reverses words 0..6 of each lane and keeps word 7
	// (dav1d shufw_6543210x).
	cdefDirShufW6543210x = [32]int8{
		12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15,
		12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15,
	}
	// cdefDirShufW3012 is pshuflw q3012 on each lane: words 2,1,0,3 then 4..7.
	cdefDirShufW3012 = [32]int8{
		4, 5, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		4, 5, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
	}
	// div_table weights: diagonal pairs (k, 14-k) for k = 0..3 and 4..7, the
	// odd pairs (k, 10-k) for k = 0..2 plus the lone center term.
	cdefDirDivDiagLo = [8]int32{840, 420, 280, 210, 840, 420, 280, 210}
	cdefDirDivDiagHi = [8]int32{168, 140, 120, 105, 168, 140, 120, 105}
	cdefDirDivAlt    = [8]int32{420, 210, 140, 105, 420, 210, 140, 105}
	// cdefDirCostOrder maps the reduced [c0 c6 c1 c7 | c4 c2 c3 c5] to c0..c7.
	cdefDirCostOrder = [8]uint32{0, 2, 5, 6, 4, 7, 1, 3}
)

// cdefDirRowPair loads row i (low 8 bytes of a 16-byte load) and the 8 bytes
// of row j, and widens them to [row i | row j] biased by -128. Only rows
// 0..3 take the 16-byte load, so nothing is read past row 7's last byte.
func cdefDirRowPair(pi, pj unsafe.Pointer, bias archsimd.Int16x16) archsimd.Int16x16 {
	v := archsimd.LoadUint8x16Array((*[16]uint8)(pi)).AsUint64x2().SetElem(1, *(*uint64)(pj))
	return v.AsUint8x16().ExtendToUint16().AsInt16x16().Sub(bias)
}

// cdefDirShl shifts each 128-bit lane left by n words (pslldq 2n).
func cdefDirShl(x, zero archsimd.Int16x16, n uint64) archsimd.Int16x16 {
	return x.AsUint8x32().ConcatShiftBytesRightGrouped(zero.AsUint8x32(), 16-2*n).AsInt16x16()
}

// cdefDirShr shifts each 128-bit lane right by n words (psrldq 2n).
func cdefDirShr(x, zero archsimd.Int16x16, n uint64) archsimd.Int16x16 {
	return zero.AsUint8x32().ConcatShiftBytesRightGrouped(x.AsUint8x32(), 2*n).AsInt16x16()
}

// cdefDirOddCost turns an odd family's left (k = 0..7) and right (k = 8..10
// in words 0..2, zeros above) partials into four int32 cost terms per lane.
func cdefDirOddCost(left, right archsimd.Int16x16, shuf archsimd.Int8x32, divAlt, div105 archsimd.Int32x8) archsimd.Int32x8 {
	rs := right.AsUint8x32().PermuteOrZeroGrouped(shuf).AsInt16x16()
	lo := left.InterleaveLoGrouped(rs)
	hi := right.InterleaveHiGrouped(left)
	return lo.DotProductPairs(lo).Mul(divAlt).Add(hi.DotProductPairs(hi).Mul(div105))
}

func findDirectionU8SIMD(img []byte, stride int) (int, int32) {
	bias := archsimd.BroadcastInt16x16(128)
	p := unsafe.Pointer(unsafe.SliceData(img[:7*stride+8]))
	sb := uintptr(stride)
	m0 := cdefDirRowPair(p, unsafe.Add(p, 7*sb), bias)
	m1 := cdefDirRowPair(unsafe.Add(p, sb), unsafe.Add(p, 6*sb), bias)
	m2 := cdefDirRowPair(unsafe.Add(p, 2*sb), unsafe.Add(p, 5*sb), bias)
	m3 := cdefDirRowPair(unsafe.Add(p, 3*sb), unsafe.Add(p, 4*sb), bias)
	return cdefDirSearchAVX2(m0, m1, m2, m3)
}

// cdefDirSearchAVX2 is cdef_dir's shared body (dav1d's .main label, which
// the 16bpc entry jumps into): m_i = [row i | row 7-i] for i < 4, already
// biased by -128.
func cdefDirSearchAVX2(m0, m1, m2, m3 archsimd.Int16x16) (int, int32) {
	var zero archsimd.Int16x16
	m7 := m0.ConcatPermute128Scalars(1, 0, m0)
	m6 := m1.ConcatPermute128Scalars(1, 0, m1)
	m5 := m2.ConcatPermute128Scalars(1, 0, m2)
	m4 := m3.ConcatPermute128Scalars(1, 0, m3)

	div105 := archsimd.BroadcastInt32x8(105)

	// partial_sum_hv: column sums (cost6) | row sums (cost2).
	s := m0.Add(m1).Add(m2.Add(m3))
	h := m0.ConcatAddPairsGrouped(m1).ConcatAddPairsGrouped(m2.ConcatAddPairsGrouped(m3))
	hv := s.SetHi(h.GetLo().ConcatAddPairs(h.GetHi())).SetLo(s.GetLo().Add(s.GetHi()))
	c62 := hv.DotProductPairs(hv).Mul(div105)

	// partial_sum_diag[0/1]: row i (lane 0) / row 7-i (lane 1) at word shift i.
	dl := m0.Add(cdefDirShl(m1, zero, 1)).Add(cdefDirShl(m2, zero, 2)).Add(cdefDirShl(m3, zero, 3)).
		Add(cdefDirShl(m4, zero, 4)).Add(cdefDirShl(m5, zero, 5)).Add(cdefDirShl(m6, zero, 6)).Add(cdefDirShl(m7, zero, 7))
	dr := cdefDirShr(m1, zero, 7).Add(cdefDirShr(m2, zero, 6)).Add(cdefDirShr(m3, zero, 5)).
		Add(cdefDirShr(m4, zero, 4)).Add(cdefDirShr(m5, zero, 3)).Add(cdefDirShr(m6, zero, 2)).Add(cdefDirShr(m7, zero, 1))
	dr = dr.AsUint8x32().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&cdefDirShufW6543210x)).AsInt16x16()
	dlo := dl.InterleaveLoGrouped(dr)
	dhi := dl.InterleaveHiGrouped(dr)
	c04 := dlo.DotProductPairs(dlo).Mul(archsimd.LoadInt32x8Array(&cdefDirDivDiagLo)).
		Add(dhi.DotProductPairs(dhi).Mul(archsimd.LoadInt32x8Array(&cdefDirDivDiagHi)))

	shuf := archsimd.LoadInt8x32Array(&cdefDirShufW3012)
	divAlt := archsimd.LoadInt32x8Array(&cdefDirDivAlt)

	// partial_sum_alt[3/2] (cost7 | cost5): row pairs at word shift 0..3.
	a0, a1, a2, a3 := m0.Add(m1), m2.Add(m3), m4.Add(m5), m6.Add(m7)
	al := a0.Add(cdefDirShl(a1, zero, 1)).Add(cdefDirShl(a2, zero, 2)).Add(cdefDirShl(a3, zero, 3))
	ar := cdefDirShr(a1, zero, 7).Add(cdefDirShr(a2, zero, 6)).Add(cdefDirShr(a3, zero, 5))
	c75 := cdefDirOddCost(al, ar, shuf, divAlt, div105)

	// partial_sum_alt[0/1] (cost1 | cost3): pairwise row sums at word shift i.
	q0 := m0.ConcatAddPairsGrouped(m4)
	q1 := m1.ConcatAddPairsGrouped(m5)
	q2 := m2.ConcatAddPairsGrouped(m6)
	q3 := m3.ConcatAddPairsGrouped(m7)
	ql := q0.Add(cdefDirShl(q1, zero, 1)).Add(cdefDirShl(q2, zero, 2)).Add(cdefDirShl(q3, zero, 3))
	qr := cdefDirShr(q1, zero, 7).Add(cdefDirShr(q2, zero, 6)).Add(cdefDirShr(q3, zero, 5))
	c13 := cdefDirOddCost(ql, qr, shuf, divAlt, div105)

	// [c0 c6 c1 c7 | c4 c2 c3 c5] -> c0..c7.
	t := c04.ConcatAddPairsGrouped(c62).ConcatAddPairsGrouped(c13.ConcatAddPairsGrouped(c75))
	var cost [8]int32
	t.Permute(archsimd.LoadUint32x8Array(&cdefDirCostOrder)).StoreArray(&cost)
	archsimd.ClearAVXUpperBits()

	bestDir, bestCost := 0, cost[0]
	for d := 1; d < 8; d++ {
		if cost[d] > bestCost {
			bestDir, bestCost = d, cost[d]
		}
	}
	return bestDir, (bestCost - cost[bestDir^4]) >> 10
}

func findDirectionDualU8SIMD(img1 []byte, img2 []byte, stride int) (int, int32, int, int32) {
	dir1, var1 := findDirectionU8SIMD(img1, stride)
	dir2, var2 := findDirectionU8SIMD(img2, stride)
	return dir1, var1, dir2, var2
}
