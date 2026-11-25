// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import (
	"encoding/binary"
	"github.com/thesyncim/goav1/internal/unsafeutil"
	"simd/archsimd"
)

var cdefDirBias128 = [8]int16{128, 128, 128, 128, 128, 128, 128, 128}

var (
	// Diagonal cost weights div[1..8] split lo (k=0..3) / hi (k=4..7).
	cdefDivDiagLo = [4]int32{840, 420, 280, 210}
	cdefDivDiagHi = [4]int32{168, 140, 120, 105}
	// Odd cost paired weights div[2],div[4],div[6] (lane 3 unused).
	cdefDivOdd = [4]int32{420, 210, 140, 0}
	// hi holds p8..p14 in lanes 0..6 -> [p14,p13,p12,p11,p10,p9,p8,0].
	cdefDiagRevTbl = [16]uint8{12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 255, 255}
	// hi holds p8,p9,p10 in lanes 0..2 -> [p10,p9,p8,0,...].
	cdefOddRevTbl = [16]uint8{4, 5, 2, 3, 0, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}
)

func init() {
	findDirectionImpl = findDirectionSIMD
	findDirectionDualImpl = findDirectionDualSIMD
	findDirectionU8Impl = findDirectionU8SIMD
	findDirectionDualU8Impl = findDirectionDualU8SIMD
}

// placeLo returns the low half (indices 0..7) of row r placed at lane offset n
// (0..7): out[k] = r[k-n] for k>=n, else 0.
func placeLo(r, zero archsimd.Int16x8, n uint64) archsimd.Int16x8 {
	if n == 0 {
		return r
	}
	return r.ToBits().ReshapeToUint8s().ConcatShiftBytesRight(zero.ToBits().ReshapeToUint8s(), 16-2*n).ReshapeToUint16s().BitsToInt16()
}

// placeHi returns the high half (indices 8..14, in lanes 0..6) of row r placed at
// lane offset n: the r lanes that overflow past index 7.
func placeHi(r, zero archsimd.Int16x8, n uint64) archsimd.Int16x8 {
	if n == 0 {
		return zero
	}
	return zero.ToBits().ReshapeToUint8s().ConcatShiftBytesRight(r.ToBits().ReshapeToUint8s(), 16-2*n).ReshapeToUint16s().BitsToInt16()
}

// packLo64 combines the low 64 bits (4 int16 lanes) of a and b into one Int16x8.
func packLo64(a, b archsimd.Int16x8) archsimd.Int16x8 {
	return a.ToBits().ReshapeToUint64s().InterleaveLo(b.ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
}

// packHi64 combines the high 64 bits (4 int16 lanes) of a and b into one Int16x8.
func packHi64(a, b archsimd.Int16x8) archsimd.Int16x8 {
	return a.ToBits().ReshapeToUint64s().InterleaveHi(b.ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
}

// cdefStraightCostFromVec = (sum of squares of the 8 int16 lanes)*105.
func cdefStraightCostFromVec(v archsimd.Int16x8) int32 {
	hi := v.HiToLo()
	sq := v.MulWidenLo(v).Add(hi.MulWidenLo(hi))
	return sq.ReduceSum() * 105
}

// cdefDiagCostVec computes finishDirectionDiagonalCost from (lo, hi) using
// hoisted lookup/weight vectors.
func cdefDiagCostVec(lo, hi archsimd.Int16x8, revTbl archsimd.Uint8x16, wLo, wHi archsimd.Int32x4) int32 {
	loHi := lo.HiToLo()
	f0 := lo.MulWidenLo(lo)
	f1 := loHi.MulWidenLo(loHi)
	rev := hi.ToBits().ReshapeToUint8s().LookupOrZero(revTbl).ReshapeToUint16s().BitsToInt16()
	revHi := rev.HiToLo()
	g0 := rev.MulWidenLo(rev)
	g1 := revHi.MulWidenLo(revHi)
	return f0.Add(g0).Mul(wLo).Add(f1.Add(g1).Mul(wHi)).ReduceSum()
}

// cdefOddCostVec computes finishDirectionOddCost from (lo, hi) using hoisted
// lookup/weight vectors, without squaring the low half twice.
func cdefOddCostVec(lo, hi archsimd.Int16x8, oddRevTbl archsimd.Uint8x16, wOdd archsimd.Int32x4) int32 {
	loSq0 := lo.MulWidenLo(lo)
	loHi := lo.HiToLo()
	loSq1 := loHi.MulWidenLo(loHi)
	center := (loSq0.GetElem(3) + loSq1.ReduceSum()) * 105
	rev := hi.ToBits().ReshapeToUint8s().LookupOrZero(oddRevTbl).ReshapeToUint16s().BitsToInt16()
	revSq := rev.MulWidenLo(rev)
	return center + loSq0.Add(revSq).Mul(wOdd).ReduceSum()
}

func findDirectionSIMD(img []uint16, stride int, coeffShift int) (int, int32) {
	if stride < 8 || stride > 1<<24 || len(img) < 7*stride+8 {
		return findDirectionScalar(img, stride, coeffShift)
	}
	bias := archsimd.BroadcastInt16x8(128)
	var zero archsimd.Int16x8
	s := img[:7*stride+8]
	var r0, r1, r2, r3, r4, r5, r6, r7 archsimd.Int16x8
	if coeffShift == 0 {
		r0 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Sub(bias)
		s = s[stride:]
		r1 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Sub(bias)
		s = s[stride:]
		r2 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Sub(bias)
		s = s[stride:]
		r3 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Sub(bias)
		s = s[stride:]
		r4 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Sub(bias)
		s = s[stride:]
		r5 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Sub(bias)
		s = s[stride:]
		r6 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Sub(bias)
		s = s[stride:]
		r7 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Sub(bias)
	} else {
		shiftV := archsimd.BroadcastInt16x8(-int16(coeffShift))
		r0 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Shift(shiftV).Sub(bias)
		s = s[stride:]
		r1 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Shift(shiftV).Sub(bias)
		s = s[stride:]
		r2 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Shift(shiftV).Sub(bias)
		s = s[stride:]
		r3 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Shift(shiftV).Sub(bias)
		s = s[stride:]
		r4 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Shift(shiftV).Sub(bias)
		s = s[stride:]
		r5 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Shift(shiftV).Sub(bias)
		s = s[stride:]
		r6 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Shift(shiftV).Sub(bias)
		s = s[stride:]
		r7 = archsimd.LoadUint16x8(s[0:8]).BitsToInt16().Shift(shiftV).Sub(bias)
	}

	// Row pairs shared by column sums (p6) and odd families (p5, p7).
	r01 := r0.Add(r1)
	r23 := r2.Add(r3)
	r45 := r4.Add(r5)
	r67 := r6.Add(r7)
	c6 := cdefStraightCostFromVec(r01.Add(r23).Add(r45).Add(r67))

	// Pairwise column sums packed two rows per vector: a01 = [s0(4), s1(4)], etc.
	// Three more ADDP steps give all 8 row sums (p2) in-register with zero GPR moves.
	a01 := r0.ConcatAddPairs(r1)
	a23 := r2.ConcatAddPairs(r3)
	a45 := r4.ConcatAddPairs(r5)
	a67 := r6.ConcatAddPairs(r7)
	rowSums := a01.ConcatAddPairs(a23).ConcatAddPairs(a45.ConcatAddPairs(a67))
	c2 := cdefStraightCostFromVec(rowSums)

	diagRevTbl := archsimd.LoadUint8x16Array(&cdefDiagRevTbl)
	wDiagLo := archsimd.LoadInt32x4Array(&cdefDivDiagLo)
	wDiagHi := archsimd.LoadInt32x4Array(&cdefDivDiagHi)

	// --- diagonal family p0 (anti-diagonal): row i at lane offset i ---
	p0lo := r0.
		Add(placeLo(r1, zero, 1)).Add(placeLo(r2, zero, 2)).
		Add(placeLo(r3, zero, 3)).Add(placeLo(r4, zero, 4)).
		Add(placeLo(r5, zero, 5)).Add(placeLo(r6, zero, 6)).Add(placeLo(r7, zero, 7))
	p0hi := placeHi(r1, zero, 1).
		Add(placeHi(r2, zero, 2)).Add(placeHi(r3, zero, 3)).
		Add(placeHi(r4, zero, 4)).Add(placeHi(r5, zero, 5)).
		Add(placeHi(r6, zero, 6)).Add(placeHi(r7, zero, 7))
	c0 := cdefDiagCostVec(p0lo, p0hi, diagRevTbl, wDiagLo, wDiagHi)

	// --- diagonal family p4: symmetric to p0 on reversed rows (r7..r0) ---
	p4lo := r7.
		Add(placeLo(r6, zero, 1)).Add(placeLo(r5, zero, 2)).
		Add(placeLo(r4, zero, 3)).Add(placeLo(r3, zero, 4)).
		Add(placeLo(r2, zero, 5)).Add(placeLo(r1, zero, 6)).Add(placeLo(r0, zero, 7))
	p4hi := placeHi(r6, zero, 1).
		Add(placeHi(r5, zero, 2)).Add(placeHi(r4, zero, 3)).
		Add(placeHi(r3, zero, 4)).Add(placeHi(r2, zero, 5)).
		Add(placeHi(r1, zero, 6)).Add(placeHi(r0, zero, 7))
	c4 := cdefDiagCostVec(p4lo, p4hi, diagRevTbl, wDiagLo, wDiagHi)

	oddRevTbl := archsimd.LoadUint8x16Array(&cdefOddRevTbl)
	wOdd := archsimd.LoadInt32x4Array(&cdefDivOdd)

	// --- odd family p7: paired rows r01,r23,r45,r67 at lane offsets 0,1,2,3 ---
	p7lo := r01.
		Add(placeLo(r23, zero, 1)).Add(placeLo(r45, zero, 2)).Add(placeLo(r67, zero, 3))
	p7hi := placeHi(r23, zero, 1).
		Add(placeHi(r45, zero, 2)).Add(placeHi(r67, zero, 3))
	c7 := cdefOddCostVec(p7lo, p7hi, oddRevTbl, wOdd)

	// --- odd family p5: paired rows r67,r45,r23,r01 at lane offsets 0,1,2,3 ---
	p5lo := r67.
		Add(placeLo(r45, zero, 1)).Add(placeLo(r23, zero, 2)).Add(placeLo(r01, zero, 3))
	p5hi := placeHi(r45, zero, 1).
		Add(placeHi(r23, zero, 2)).Add(placeHi(r01, zero, 3))
	c5 := cdefOddCostVec(p5lo, p5hi, oddRevTbl, wOdd)

	// --- odd family p1: non-overlapping pairs [s0,s4], [s1,s5], [s2,s6], [s3,s7] ---
	s04 := packLo64(a01, a45)
	s15 := packHi64(a01, a45)
	s26 := packLo64(a23, a67)
	s37 := packHi64(a23, a67)
	p1lo := s04.
		Add(placeLo(s15, zero, 1)).Add(placeLo(s26, zero, 2)).Add(placeLo(s37, zero, 3))
	p1hi := placeHi(s15, zero, 1).
		Add(placeHi(s26, zero, 2)).Add(placeHi(s37, zero, 3))
	c1 := cdefOddCostVec(p1lo, p1hi, oddRevTbl, wOdd)

	// --- odd family p3: symmetric to p1 on reversed pairwise sums (s7..s0) ---
	s73 := packHi64(a67, a23)
	s62 := packLo64(a67, a23)
	s51 := packHi64(a45, a01)
	s40 := packLo64(a45, a01)
	p3lo := s73.
		Add(placeLo(s62, zero, 1)).Add(placeLo(s51, zero, 2)).Add(placeLo(s40, zero, 3))
	p3hi := placeHi(s62, zero, 1).
		Add(placeHi(s51, zero, 2)).Add(placeHi(s40, zero, 3))
	c3 := cdefOddCostVec(p3lo, p3hi, oddRevTbl, wOdd)

	bestDir, bestCost, opp := 0, c0, c4
	if c1 > bestCost {
		bestDir, bestCost, opp = 1, c1, c5
	}
	if c2 > bestCost {
		bestDir, bestCost, opp = 2, c2, c6
	}
	if c3 > bestCost {
		bestDir, bestCost, opp = 3, c3, c7
	}
	if c4 > bestCost {
		bestDir, bestCost, opp = 4, c4, c0
	}
	if c5 > bestCost {
		bestDir, bestCost, opp = 5, c5, c1
	}
	if c6 > bestCost {
		bestDir, bestCost, opp = 6, c6, c2
	}
	if c7 > bestCost {
		bestDir, bestCost, opp = 7, c7, c3
	}
	return bestDir, (bestCost - opp) >> 10
}

func findDirectionDualSIMD(img1 []uint16, img2 []uint16, stride int, coeffShift int) (int, int32, int, int32) {
	dir1, var1 := findDirectionSIMD(img1, stride, coeffShift)
	dir2, var2 := findDirectionSIMD(img2, stride, coeffShift)
	return dir1, var1, dir2, var2
}

// findDirectionU8SIMD reads the 8-bit frame plane directly in safe Go.
// Rows 0..6 use 16-byte loads whose 8-byte tail lands inside rows 1..7; row 7
// loads [7*stride-8 : 7*stride+8] and takes the high half so no byte past the
// block's last pixel is read.
func findDirectionU8SIMD(img []byte, stride int) (int, int32) {
	if stride < 8 || stride > 1<<24 || len(img) < 7*stride+8 {
		return findDirectionU8Scalar(img, stride)
	}
	bias := archsimd.LoadInt16x8Array(&cdefDirBias128)
	var zero archsimd.Int16x8
	var r0, r1, r2, r3, r4, r5, r6, r7 archsimd.Int16x8
	if unsafeutil.RowsFit(len(img), 0, stride, 16, 8) {
		b := unsafeutil.BlockAt[cdefR8](unsafeutil.NewRows(img, 0, stride, 16, 8), 0, 0)
		r0 = archsimd.LoadUint8x16Array(b.Row16(0)).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
		r1 = archsimd.LoadUint8x16Array(b.Row16(1)).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
		r2 = archsimd.LoadUint8x16Array(b.Row16(2)).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
		r3 = archsimd.LoadUint8x16Array(b.Row16(3)).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
		r4 = archsimd.LoadUint8x16Array(b.Row16(4)).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
		r5 = archsimd.LoadUint8x16Array(b.Row16(5)).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
		r6 = archsimd.LoadUint8x16Array(b.Row16(6)).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
		r7 = archsimd.LoadUint8x16Array(b.Row16(7)).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
	} else {
		s := img[:7*stride+8]
		r0 = archsimd.LoadUint8x16(s[0:16]).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
		s = s[stride:]
		r1 = archsimd.LoadUint8x16(s[0:16]).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
		s = s[stride:]
		r2 = archsimd.LoadUint8x16(s[0:16]).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
		s = s[stride:]
		r3 = archsimd.LoadUint8x16(s[0:16]).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
		s = s[stride:]
		r4 = archsimd.LoadUint8x16(s[0:16]).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
		s = s[stride:]
		r5 = archsimd.LoadUint8x16(s[0:16]).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
		s = s[stride:]
		r6 = archsimd.LoadUint8x16(s[0:16]).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
		r7 = archsimd.LoadUint8x16(s[stride-8 : stride+8]).HiToLo().ExtendLo8ToUint16().BitsToInt16().Sub(bias)
	}

	r01 := r0.Add(r1)
	r23 := r2.Add(r3)
	r45 := r4.Add(r5)
	r67 := r6.Add(r7)
	c6 := cdefStraightCostFromVec(r01.Add(r23).Add(r45).Add(r67))

	a01 := r0.ConcatAddPairs(r1)
	a23 := r2.ConcatAddPairs(r3)
	a45 := r4.ConcatAddPairs(r5)
	a67 := r6.ConcatAddPairs(r7)
	rowSums := a01.ConcatAddPairs(a23).ConcatAddPairs(a45.ConcatAddPairs(a67))
	c2 := cdefStraightCostFromVec(rowSums)

	diagRevTbl := archsimd.LoadUint8x16Array(&cdefDiagRevTbl)
	wDiagLo := archsimd.LoadInt32x4Array(&cdefDivDiagLo)
	wDiagHi := archsimd.LoadInt32x4Array(&cdefDivDiagHi)

	p0lo := r0.
		Add(placeLo(r1, zero, 1)).Add(placeLo(r2, zero, 2)).
		Add(placeLo(r3, zero, 3)).Add(placeLo(r4, zero, 4)).
		Add(placeLo(r5, zero, 5)).Add(placeLo(r6, zero, 6)).Add(placeLo(r7, zero, 7))
	p0hi := placeHi(r1, zero, 1).
		Add(placeHi(r2, zero, 2)).Add(placeHi(r3, zero, 3)).
		Add(placeHi(r4, zero, 4)).Add(placeHi(r5, zero, 5)).
		Add(placeHi(r6, zero, 6)).Add(placeHi(r7, zero, 7))
	c0 := cdefDiagCostVec(p0lo, p0hi, diagRevTbl, wDiagLo, wDiagHi)

	p4lo := r7.
		Add(placeLo(r6, zero, 1)).Add(placeLo(r5, zero, 2)).
		Add(placeLo(r4, zero, 3)).Add(placeLo(r3, zero, 4)).
		Add(placeLo(r2, zero, 5)).Add(placeLo(r1, zero, 6)).Add(placeLo(r0, zero, 7))
	p4hi := placeHi(r6, zero, 1).
		Add(placeHi(r5, zero, 2)).Add(placeHi(r4, zero, 3)).
		Add(placeHi(r3, zero, 4)).Add(placeHi(r2, zero, 5)).
		Add(placeHi(r1, zero, 6)).Add(placeHi(r0, zero, 7))
	c4 := cdefDiagCostVec(p4lo, p4hi, diagRevTbl, wDiagLo, wDiagHi)

	oddRevTbl := archsimd.LoadUint8x16Array(&cdefOddRevTbl)
	wOdd := archsimd.LoadInt32x4Array(&cdefDivOdd)

	p7lo := r01.
		Add(placeLo(r23, zero, 1)).Add(placeLo(r45, zero, 2)).Add(placeLo(r67, zero, 3))
	p7hi := placeHi(r23, zero, 1).
		Add(placeHi(r45, zero, 2)).Add(placeHi(r67, zero, 3))
	c7 := cdefOddCostVec(p7lo, p7hi, oddRevTbl, wOdd)

	p5lo := r67.
		Add(placeLo(r45, zero, 1)).Add(placeLo(r23, zero, 2)).Add(placeLo(r01, zero, 3))
	p5hi := placeHi(r45, zero, 1).
		Add(placeHi(r23, zero, 2)).Add(placeHi(r01, zero, 3))
	c5 := cdefOddCostVec(p5lo, p5hi, oddRevTbl, wOdd)

	s04 := packLo64(a01, a45)
	s15 := packHi64(a01, a45)
	s26 := packLo64(a23, a67)
	s37 := packHi64(a23, a67)
	p1lo := s04.
		Add(placeLo(s15, zero, 1)).Add(placeLo(s26, zero, 2)).Add(placeLo(s37, zero, 3))
	p1hi := placeHi(s15, zero, 1).
		Add(placeHi(s26, zero, 2)).Add(placeHi(s37, zero, 3))
	c1 := cdefOddCostVec(p1lo, p1hi, oddRevTbl, wOdd)

	s73 := packHi64(a67, a23)
	s62 := packLo64(a67, a23)
	s51 := packHi64(a45, a01)
	s40 := packLo64(a45, a01)
	p3lo := s73.
		Add(placeLo(s62, zero, 1)).Add(placeLo(s51, zero, 2)).Add(placeLo(s40, zero, 3))
	p3hi := placeHi(s62, zero, 1).
		Add(placeHi(s51, zero, 2)).Add(placeHi(s40, zero, 3))
	c3 := cdefOddCostVec(p3lo, p3hi, oddRevTbl, wOdd)

	bestDir, bestCost, opp := 0, c0, c4
	if c1 > bestCost {
		bestDir, bestCost, opp = 1, c1, c5
	}
	if c2 > bestCost {
		bestDir, bestCost, opp = 2, c2, c6
	}
	if c3 > bestCost {
		bestDir, bestCost, opp = 3, c3, c7
	}
	if c4 > bestCost {
		bestDir, bestCost, opp = 4, c4, c0
	}
	if c5 > bestCost {
		bestDir, bestCost, opp = 5, c5, c1
	}
	if c6 > bestCost {
		bestDir, bestCost, opp = 6, c6, c2
	}
	if c7 > bestCost {
		bestDir, bestCost, opp = 7, c7, c3
	}
	return bestDir, (bestCost - opp) >> 10
}

func findDirectionDualU8SIMD(img1 []byte, img2 []byte, stride int) (int, int32, int, int32) {
	dir1, var1 := findDirectionU8SIMD(img1, stride)
	dir2, var2 := findDirectionU8SIMD(img2, stride)
	return dir1, var1, dir2, var2
}

// The uint16-output CDEF block kernels split by strength, as dav1d's
// cdef_filter{4,8}_{pri,sec,pri_sec}_16bpc_neon (src/arm/64/cdef_tmpl.S
// filter_func): each block runs the kernel for the taps it has, so the row
// loops carry no strength branches, the secondary-only and primary-only
// kernels skip the min/max clip (it only applies when both are active), and
// the secondary taps (2, 1) fold to an add instead of two multiplies. The
// arithmetic is cdefFilterBlock8SIMD / cdefFilterBlock4SIMD's, lane for lane.
// The 4-wide block with both strengths keeps the shared kernel.

// cdefFilterBlock16 routes a prepared uint16-output context to the kernel
// for its strengths. At least one strength is nonzero (zero-strength blocks
// are skipped by the caller); width is 8, or 4 with an even height.
func cdefFilterBlock16(ctx *cdefSIMDCtx, width int) {
	switch {
	case !ctx.enablePrimary:
		if width == 8 {
			cdefFilter8Sec16(ctx)
		} else {
			cdefFilter4Sec16(ctx)
		}
	case !ctx.enableSecondary:
		if width == 8 {
			cdefFilter8Pri16(ctx)
		} else {
			cdefFilter4Pri16(ctx)
		}
	case width == 8:
		cdefFilter8PriSec16(ctx)
	default:
		cdefFilterBlockSIMD(ctx, width)
	}
}

// cdefFinish16 is x + ((8 + sum - (sum < 0)) >> 4).
func cdefFinish16(x, sum, eight archsimd.Int16x8, sar15, sar4 cdefShiftCount) archsimd.Int16x8 {
	return x.Add(cdefSarI16(sum.Add(cdefSarI16(sum, sar15)).Add(eight), sar4))
}

// cdefFilter8Sec16 is the 8-wide secondary-only kernel: sum =
// 2 * (c0 + c1 + c2 + c3) + (c4 + c5 + c6 + c7), no clip.
func cdefFilter8Sec16(ctx *cdefSIMDCtx) {
	secStr := archsimd.BroadcastInt16x8(int16(ctx.secStrength)).ToBits()
	secSh := cdefShiftCountOf(ctx.secShift)
	eight := archsimd.BroadcastInt16x8(8)
	sar15 := cdefShiftCountOf(15)
	sar4 := cdefShiftCountOf(4)
	s0p, s0m, s1p, s1m := ctx.secTaps0()
	s2p, s2m, s3p, s3m := ctx.secTaps1()
	in, base, height := ctx.rows()
	out16, _, do, ds := ctx.outRows()
	for r := 0; r < height; r++ {
		w := cdefRow(in, base+r*BStride)
		x := cdefLdW1(w, cdefWinReach)
		a := cdefConstrain(cdefLdW1(w, s0p), x, secStr, secSh).
			Add(cdefConstrain(cdefLdW1(w, s0m), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdW1(w, s1p), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdW1(w, s1m), x, secStr, secSh))
		b := cdefConstrain(cdefLdW1(w, s2p), x, secStr, secSh).
			Add(cdefConstrain(cdefLdW1(w, s2m), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdW1(w, s3p), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdW1(w, s3m), x, secStr, secSh))
		sum := a.Add(a).Add(b)
		cdefFinish16(x, sum, eight, sar15, sar4).ToBits().StoreArray(cdefOut16(out16, do+r*ds))
	}
}

// cdefFilter4Sec16 is cdefFilter8Sec16 for a 4-wide block, two rows per
// vector (rows r and r+1 in lanes 0..3 and 4..7).
func cdefFilter4Sec16(ctx *cdefSIMDCtx) {
	secStr := archsimd.BroadcastInt16x8(int16(ctx.secStrength)).ToBits()
	secSh := cdefShiftCountOf(ctx.secShift)
	eight := archsimd.BroadcastInt16x8(8)
	sar15 := cdefShiftCountOf(15)
	sar4 := cdefShiftCountOf(4)
	s0p, s0m, s1p, s1m := ctx.secTaps0()
	s2p, s2m, s3p, s3m := ctx.secTaps1()
	in, base, height := ctx.rows()
	out16, _, do, ds := ctx.outRows()
	// Counted by row pair: a step-2 counter keeps a second check per window.
	for i := 0; i < height>>1; i++ {
		r := 2 * i
		w0, w1 := cdefRows2(in, base+r*BStride)
		x := cdefLdPair(w0, w1, cdefWinReach)
		a := cdefConstrain(cdefLdPair(w0, w1, s0p), x, secStr, secSh).
			Add(cdefConstrain(cdefLdPair(w0, w1, s0m), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdPair(w0, w1, s1p), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdPair(w0, w1, s1m), x, secStr, secSh))
		b := cdefConstrain(cdefLdPair(w0, w1, s2p), x, secStr, secSh).
			Add(cdefConstrain(cdefLdPair(w0, w1, s2m), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdPair(w0, w1, s3p), x, secStr, secSh)).
			Add(cdefConstrain(cdefLdPair(w0, w1, s3m), x, secStr, secSh))
		sum := a.Add(a).Add(b)
		out := cdefFinish16(x, sum, eight, sar15, sar4).ToBits().ReshapeToUint64s()
		cdefPutU16x4(out16, do+r*ds, out.GetElem(0))
		cdefPutU16x4(out16, do+(r+1)*ds, out.GetElem(1))
	}
}

// cdefFilter8Pri16 is the 8-wide primary-only kernel (no clip).
func cdefFilter8Pri16(ctx *cdefSIMDCtx) {
	priStr := archsimd.BroadcastInt16x8(int16(ctx.priStrength)).ToBits()
	priSh := cdefShiftCountOf(ctx.priShift)
	priTap0 := archsimd.BroadcastInt16x8(int16(ctx.priTap0))
	priTap1 := archsimd.BroadcastInt16x8(int16(ctx.priTap1))
	eight := archsimd.BroadcastInt16x8(8)
	sar15 := cdefShiftCountOf(15)
	sar4 := cdefShiftCountOf(4)
	p0p, p0m, p1p, p1m := ctx.priTaps()
	in, base, height := ctx.rows()
	out16, _, do, ds := ctx.outRows()
	for r := 0; r < height; r++ {
		w := cdefRow(in, base+r*BStride)
		x := cdefLdW1(w, cdefWinReach)
		sum := cdefConstrain(cdefLdW1(w, p0p), x, priStr, priSh).
			Add(cdefConstrain(cdefLdW1(w, p0m), x, priStr, priSh)).Mul(priTap0).
			Add(cdefConstrain(cdefLdW1(w, p1p), x, priStr, priSh).
				Add(cdefConstrain(cdefLdW1(w, p1m), x, priStr, priSh)).Mul(priTap1))
		cdefFinish16(x, sum, eight, sar15, sar4).ToBits().StoreArray(cdefOut16(out16, do+r*ds))
	}
}

// cdefFilter4Pri16 is cdefFilter8Pri16 for a 4-wide block, two rows per
// vector.
func cdefFilter4Pri16(ctx *cdefSIMDCtx) {
	priStr := archsimd.BroadcastInt16x8(int16(ctx.priStrength)).ToBits()
	priSh := cdefShiftCountOf(ctx.priShift)
	priTap0 := archsimd.BroadcastInt16x8(int16(ctx.priTap0))
	priTap1 := archsimd.BroadcastInt16x8(int16(ctx.priTap1))
	eight := archsimd.BroadcastInt16x8(8)
	sar15 := cdefShiftCountOf(15)
	sar4 := cdefShiftCountOf(4)
	p0p, p0m, p1p, p1m := ctx.priTaps()
	in, base, height := ctx.rows()
	out16, _, do, ds := ctx.outRows()
	// Counted by row pair: a step-2 counter keeps a second check per window.
	for i := 0; i < height>>1; i++ {
		r := 2 * i
		w0, w1 := cdefRows2(in, base+r*BStride)
		x := cdefLdPair(w0, w1, cdefWinReach)
		sum := cdefConstrain(cdefLdPair(w0, w1, p0p), x, priStr, priSh).
			Add(cdefConstrain(cdefLdPair(w0, w1, p0m), x, priStr, priSh)).Mul(priTap0).
			Add(cdefConstrain(cdefLdPair(w0, w1, p1p), x, priStr, priSh).
				Add(cdefConstrain(cdefLdPair(w0, w1, p1m), x, priStr, priSh)).Mul(priTap1))
		out := cdefFinish16(x, sum, eight, sar15, sar4).ToBits().ReshapeToUint64s()
		cdefPutU16x4(out16, do+r*ds, out.GetElem(0))
		cdefPutU16x4(out16, do+(r+1)*ds, out.GetElem(1))
	}
}

// cdefFilter8PriSec16 is the 8-wide kernel with both strengths: the primary
// and secondary sums, then the clip to [min, max] over the taps (the
// sentinel masked out of the max, as in cdefFilterBlock8SIMD).
func cdefFilter8PriSec16(ctx *cdefSIMDCtx) {
	priStr := archsimd.BroadcastInt16x8(int16(ctx.priStrength)).ToBits()
	secStr := archsimd.BroadcastInt16x8(int16(ctx.secStrength)).ToBits()
	priSh := cdefShiftCountOf(ctx.priShift)
	secSh := cdefShiftCountOf(ctx.secShift)
	priTap0 := archsimd.BroadcastInt16x8(int16(ctx.priTap0))
	priTap1 := archsimd.BroadcastInt16x8(int16(ctx.priTap1))
	eight := archsimd.BroadcastInt16x8(8)
	noSentinel := archsimd.BroadcastInt16x8(int16(^VeryLarge))
	sar15 := cdefShiftCountOf(15)
	sar4 := cdefShiftCountOf(4)
	p0p, p0m, p1p, p1m := ctx.priTaps()
	s0p, s0m, s1p, s1m := ctx.secTaps0()
	s2p, s2m, s3p, s3m := ctx.secTaps1()
	in, base, height := ctx.rows()
	out16, _, do, ds := ctx.outRows()
	for r := 0; r < height; r++ {
		w := cdefRow(in, base+r*BStride)
		x := cdefLdW1(w, cdefWinReach)
		t0 := cdefLdW1(w, p0p)
		t1 := cdefLdW1(w, p0m)
		t2 := cdefLdW1(w, p1p)
		t3 := cdefLdW1(w, p1m)
		sum := cdefConstrain(t0, x, priStr, priSh).Add(cdefConstrain(t1, x, priStr, priSh)).Mul(priTap0).
			Add(cdefConstrain(t2, x, priStr, priSh).Add(cdefConstrain(t3, x, priStr, priSh)).Mul(priTap1))
		// The min and max fold as trees per tap group so their dependency
		// chains stay short.
		mx := t0.And(noSentinel).Max(t1.And(noSentinel)).Max(t2.And(noSentinel).Max(t3.And(noSentinel)))
		mn := t0.Min(t1).Min(t2.Min(t3))
		s0 := cdefLdW1(w, s0p)
		s1 := cdefLdW1(w, s0m)
		s2 := cdefLdW1(w, s1p)
		s3 := cdefLdW1(w, s1m)
		a := cdefConstrain(s0, x, secStr, secSh).Add(cdefConstrain(s1, x, secStr, secSh)).
			Add(cdefConstrain(s2, x, secStr, secSh)).Add(cdefConstrain(s3, x, secStr, secSh))
		mx1 := s0.And(noSentinel).Max(s1.And(noSentinel)).Max(s2.And(noSentinel).Max(s3.And(noSentinel)))
		mn1 := s0.Min(s1).Min(s2.Min(s3))
		s4 := cdefLdW1(w, s2p)
		s5 := cdefLdW1(w, s2m)
		s6 := cdefLdW1(w, s3p)
		s7 := cdefLdW1(w, s3m)
		b := cdefConstrain(s4, x, secStr, secSh).Add(cdefConstrain(s5, x, secStr, secSh)).
			Add(cdefConstrain(s6, x, secStr, secSh)).Add(cdefConstrain(s7, x, secStr, secSh))
		mx2 := s4.And(noSentinel).Max(s5.And(noSentinel)).Max(s6.And(noSentinel).Max(s7.And(noSentinel)))
		mn2 := s4.Min(s5).Min(s6.Min(s7))
		mx = mx.Max(x).Max(mx1.Max(mx2))
		mn = mn.Min(x).Min(mn1.Min(mn2))
		sum = sum.Add(a.Add(a).Add(b))
		cdefFinish16(x, sum, eight, sar15, sar4).Max(mn).Min(mx).ToBits().StoreArray(cdefOut16(out16, do+r*ds))
	}
}

// searchBlocksU8 binds the batched search kernel on arm64.
func searchBlocksU8(src []byte, srcStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, damping int, trials []SearchTrial, rowStep int, sse []uint64) bool {
	return searchBlocksU8SIMD(src, srcStride, input, inputOrigin, blocks, directions, variances, damping, trials, rowStep, sse)
}

// cdefSearchTrialSIMD is one trial's kernel state for the current block: the
// broadcast strengths and shifts of cdefFilterBlock8SIMD plus which tap sets
// the trial reads.
type cdefSearchTrialSIMD struct {
	priStr  archsimd.Uint16x8
	secStr  archsimd.Uint16x8
	priSh   cdefShiftCount
	secSh   cdefShiftCount
	priTap0 archsimd.Int16x8
	priTap1 archsimd.Int16x8
	priOn   bool
	secOn   bool
	// secAtDir is set when the secondary taps follow the block direction
	// (trial primary strength non-zero); otherwise they use direction 0.
	secAtDir bool
}

// cdefLoadSrc8 widens eight source bytes to int16 lanes.
func cdefLoadSrc8(b *[8]byte) archsimd.Int16x8 {
	v := uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
	rows := [2]uint64{v, 0}
	return archsimd.LoadUint64x2Array(&rows).ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()
}

// cdefSearchTaps returns the window indices cdefWinReach +- t of the tap
// offsets a and b, checked against the window.
func cdefSearchTaps(a, b int16) (uint, uint, uint, uint) {
	ap, am := uint(cdefWinReach+int(a)), uint(cdefWinReach-int(a))
	bp, bm := uint(cdefWinReach+int(b)), uint(cdefWinReach-int(b))
	if ap > cdefTapMax || am > cdefTapMax || bp > cdefTapMax || bm > cdefTapMax {
		panic(errCDEFWindow)
	}
	return ap, am, bp, bm
}

// cdefSquareSum widens the eight int16 errors to their int32 squares.
func cdefSquareSum(e archsimd.Int16x8) archsimd.Int32x4 {
	return e.MulWidenLo(e).Add(e.HiToLo().MulWidenLo(e.HiToLo()))
}

// searchBlocksU8SIMD is the per-trial search with one kernel entry per
// block: each measured row loads the center and tap samples once, derives the
// clamp bounds once, and then loops the trials over the register-resident
// taps, accumulating each trial's squared error against the source in int32
// lanes (at most 64 blocks of 64 samples at 255^2 fits). The per-row
// arithmetic of a trial is cdefFilterBlock8SIMD's, so every filtered row is
// bit-identical to the per-trial path's.
func searchBlocksU8SIMD(src []byte, srcStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, damping int, trials []SearchTrial, rowStep int, sse []uint64) bool {
	n := len(trials)
	if n == 0 || n > MaxSearchTrials || len(sse) < n {
		return false
	}
	if len(blocks) == 0 {
		for t := range sse[:n] {
			sse[t] = 0
		}
		return true
	}
	// Bounds: block rows 0..7 of the last listed block in src, and the
	// two-sample halo around every block in input.
	maxSrc, maxBY, maxBX := 0, 0, 0
	for _, b := range blocks {
		maxSrc = max(maxSrc, (int(b.BY)*8+7)*srcStride+int(b.BX)*8+7)
		maxBY = max(maxBY, int(b.BY))
		maxBX = max(maxBX, int(b.BX))
	}
	// The origin and stride bounds also make each row window below one
	// compare.
	if inputOrigin < 2*BStride+2 || inputOrigin > 1<<24 || uint(srcStride) > 1<<20 ||
		maxBY >= 8 || maxBX >= 8 || maxSrc >= len(src) ||
		inputOrigin+(maxBY*8+7+2)*BStride+maxBX*8+7+2 >= len(input) {
		return false
	}
	_ = src[maxSrc]
	_ = input[inputOrigin+(maxBY*8+7+2)*BStride+maxBX*8+7+2]
	_ = sse[n-1]

	var st [MaxSearchTrials]cdefSearchTrialSIMD
	var acc [MaxSearchTrials]archsimd.Int32x4
	zero := archsimd.BroadcastInt16x8(0)
	secTap0 := archsimd.BroadcastInt16x8(int16(cdefSecondaryTaps[0]))
	secTap1 := archsimd.BroadcastInt16x8(int16(cdefSecondaryTaps[1]))
	eight := archsimd.BroadcastInt16x8(8)
	noSentinel := archsimd.BroadcastInt16x8(int16(^VeryLarge))
	sar15 := cdefShiftCountOf(15)
	sar4 := cdefShiftCountOf(4)
	for t := range trials {
		s := &st[t]
		sec := int(trials[t].Secondary)
		s.secStr = archsimd.BroadcastInt16x8(int16(sec)).ToBits()
		s.secSh = cdefShiftCountOf(constrainShift(sec, damping))
		s.secOn = sec != 0
		s.secAtDir = trials[t].Primary != 0
		acc[t] = archsimd.BroadcastInt32x4(0)
	}
	// Direction-0 secondary tap window indices (trials with no primary).
	z0p, z0m, z2p, z2m := cdefSearchTaps(cdefDirections[4][0], cdefDirections[4][1])
	z1p, z1m, z3p, z3m := cdefSearchTaps(cdefDirections[0][0], cdefDirections[0][1])
	for _, b := range blocks {
		by, bx := int(b.BY), int(b.BX)
		dir := int(directions[by][bx])
		vari := variances[by][bx]
		p0p, p0m, p1p, p1m := cdefSearchTaps(cdefDirections[dir+2][0], cdefDirections[dir+2][1])
		s0p, s0m, s2p, s2m := cdefSearchTaps(cdefDirections[dir+4][0], cdefDirections[dir+4][1])
		s1p, s1m, s3p, s3m := cdefSearchTaps(cdefDirections[dir][0], cdefDirections[dir][1])
		needDir, needClip, needZero, needIdentity := false, false, false, false
		for t := range trials {
			s := &st[t]
			strength := 0
			if primary := int(trials[t].Primary); primary != 0 {
				strength = adjustStrength(primary, vari)
			}
			s.priOn = strength != 0
			if s.priOn {
				taps := cdefPrimaryTaps[strength&1]
				s.priTap0 = archsimd.BroadcastInt16x8(int16(taps[0]))
				s.priTap1 = archsimd.BroadcastInt16x8(int16(taps[1]))
				s.priStr = archsimd.BroadcastInt16x8(int16(strength)).ToBits()
				s.priSh = cdefShiftCountOf(constrainShift(strength, damping))
			}
			switch {
			case !s.priOn && !s.secOn:
				needIdentity = true
			case s.priOn && s.secOn:
				needDir, needClip = true, true
			case s.priOn || s.secAtDir:
				needDir = true
			default:
				needZero = true
			}
		}
		blockIn := inputOrigin + by*8*BStride + bx*8 - cdefWinReach
		blockSrc := by*8*srcStride + bx*8
		for i := 0; i < 8; i += rowStep {
			// i & 7 is i (the loop keeps it in 0..7); the mask gives the row
			// offsets a range the compiler can use with a variable step.
			r := i & 7
			wo, so := blockIn+r*BStride, blockSrc+r*srcStride
			w := (*[cdefWin1]uint16)(input[wo : wo+cdefWin1])
			x := cdefLdW1(w, cdefWinReach)
			srow := cdefLoadSrc8((*[8]byte)(src[so : so+8]))
			var sqIdentity archsimd.Int32x4
			if needIdentity {
				sqIdentity = cdefSquareSum(x.Sub(srow))
			}
			var t0, t1, t2, t3, s0, s1, s2, s3, s4, s5, s6, s7 archsimd.Int16x8
			mx, mn := x, x
			if needDir {
				t0 = cdefLdW1(w, p0p)
				t1 = cdefLdW1(w, p0m)
				t2 = cdefLdW1(w, p1p)
				t3 = cdefLdW1(w, p1m)
				s0 = cdefLdW1(w, s0p)
				s1 = cdefLdW1(w, s0m)
				s2 = cdefLdW1(w, s1p)
				s3 = cdefLdW1(w, s1m)
				s4 = cdefLdW1(w, s2p)
				s5 = cdefLdW1(w, s2m)
				s6 = cdefLdW1(w, s3p)
				s7 = cdefLdW1(w, s3m)
				if needClip {
					mx = mx.Max(t0.And(noSentinel)).Max(t1.And(noSentinel)).Max(t2.And(noSentinel)).Max(t3.And(noSentinel))
					mn = mn.Min(t0).Min(t1).Min(t2).Min(t3)
					mx = mx.Max(s0.And(noSentinel)).Max(s1.And(noSentinel)).Max(s2.And(noSentinel)).Max(s3.And(noSentinel))
					mn = mn.Min(s0).Min(s1).Min(s2).Min(s3)
					mx = mx.Max(s4.And(noSentinel)).Max(s5.And(noSentinel)).Max(s6.And(noSentinel)).Max(s7.And(noSentinel))
					mn = mn.Min(s4).Min(s5).Min(s6).Min(s7)
				}
			}
			var u0, u1, u2, u3, u4, u5, u6, u7 archsimd.Int16x8
			if needZero {
				u0 = cdefLdW1(w, z0p)
				u1 = cdefLdW1(w, z0m)
				u2 = cdefLdW1(w, z1p)
				u3 = cdefLdW1(w, z1m)
				u4 = cdefLdW1(w, z2p)
				u5 = cdefLdW1(w, z2m)
				u6 = cdefLdW1(w, z3p)
				u7 = cdefLdW1(w, z3m)
			}
			for t := 0; t < n; t++ {
				s := &st[t]
				if !s.priOn && !s.secOn {
					acc[t] = acc[t].Add(sqIdentity)
					continue
				}
				sum := zero
				if s.priOn {
					priStr, priSh := s.priStr, s.priSh
					sum = cdefConstrain(t0, x, priStr, priSh).Add(cdefConstrain(t1, x, priStr, priSh)).Mul(s.priTap0).
						Add(cdefConstrain(t2, x, priStr, priSh).Add(cdefConstrain(t3, x, priStr, priSh)).Mul(s.priTap1))
				}
				if s.secOn {
					secStr, secSh := s.secStr, s.secSh
					if s.secAtDir {
						sum = sum.Add(cdefConstrain(s0, x, secStr, secSh).Add(cdefConstrain(s1, x, secStr, secSh)).
							Add(cdefConstrain(s2, x, secStr, secSh)).Add(cdefConstrain(s3, x, secStr, secSh)).Mul(secTap0))
						sum = sum.Add(cdefConstrain(s4, x, secStr, secSh).Add(cdefConstrain(s5, x, secStr, secSh)).
							Add(cdefConstrain(s6, x, secStr, secSh)).Add(cdefConstrain(s7, x, secStr, secSh)).Mul(secTap1))
					} else {
						sum = sum.Add(cdefConstrain(u0, x, secStr, secSh).Add(cdefConstrain(u1, x, secStr, secSh)).
							Add(cdefConstrain(u2, x, secStr, secSh)).Add(cdefConstrain(u3, x, secStr, secSh)).Mul(secTap0))
						sum = sum.Add(cdefConstrain(u4, x, secStr, secSh).Add(cdefConstrain(u5, x, secStr, secSh)).
							Add(cdefConstrain(u6, x, secStr, secSh)).Add(cdefConstrain(u7, x, secStr, secSh)).Mul(secTap1))
					}
				}
				// y = x + ((8 + sum - (sum<0)) >> 4)
				y := x.Add(cdefSarI16(sum.Add(cdefSarI16(sum, sar15)).Add(eight), sar4))
				if s.priOn && s.secOn {
					y = y.Max(mn).Min(mx)
				}
				acc[t] = acc[t].Add(cdefSquareSum(y.Sub(srow)))
			}
		}
	}
	for t := range trials {
		sse[t] = uint64(uint32(acc[t].ReduceSum()))
	}
	return true
}

// cdefShiftCount is the per-block constrain() shift in the form the NEON
// variable shift consumes: a broadcast negative count, so USHL shifts right.
type cdefShiftCount = archsimd.Int16x8

func cdefShiftCountOf(n int) cdefShiftCount {
	return archsimd.BroadcastInt16x8(-int16(n))
}

// cdefShrU16 is a logical right shift of each lane by the per-block count.
func cdefShrU16(v archsimd.Uint16x8, sh cdefShiftCount) archsimd.Uint16x8 {
	return v.Shift(sh)
}

// filterUnitBlocks binds the Go SIMD unit-level loop on arm64 (NEON is
// architecturally mandatory there). See filter_dispatch.go for why this is a
// build-tag binding instead of a func variable.
func filterUnitBlocks(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams, trusted bool) error {
	return filterUnitBlocksSIMD(dst, dstStride, input, inputOrigin, blocks, directions, variances, u, trusted)
}

// filterUnitBlocksSIMD mirrors dav1d's per-superblock cdef apply loop
// (src/cdef_apply_tmpl.c): the per-unit invariants (dst stride, block
// geometry, secondary strength/shift/taps, and for chroma the primary
// constants too) are computed once per filter unit, each block fills only the
// per-block fields, and blocks whose adjusted primary strength and secondary
// strength are both zero skip the filter kernel entirely (identity copy, as
// dav1d skips the fb call for its in-place buffer). Output is bit-identical
// to filterUnitBlocksPureGo.
func filterUnitBlocksSIMD(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams, trusted bool) error {
	if !trusted || (u.blockWidth != 8 && u.blockWidth != 4) {
		return filterUnitBlocksPureGo(dst, dstStride, input, inputOrigin, blocks, directions, variances, u, trusted)
	}
	secondaryStrength := u.secondaryStrength
	ctx := cdefSIMDCtx{
		in:     input,
		out16:  dst,
		dstStr: dstStride,
		height: u.blockHeight,
	}
	ctx.setSecondary(secondaryStrength, u.damping)
	strength := u.primaryStrength
	if !u.lumaAdjust {
		ctx.setPrimary(strength, u.damping, u.coeffShift)
	}
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
			ctx.setPrimary(strength, u.damping, u.coeffShift)
		}
		srcOrigin := inputOrigin + ((by * BStride) << u.bhLog2) + (bx << u.bwLog2)
		dstOrigin := (by<<u.bhLog2)*dstStride + (bx << u.bwLog2)
		if strength == 0 && secondaryStrength == 0 {
			// See filterUnitBlocksPureGo: dav1d's zero-strength skip.
			copyBlockIdentity(dst, dstStride, dstOrigin, input, srcOrigin, u.blockWidth, u.blockHeight)
			continue
		}
		dir := 0
		if u.primaryStrength != 0 {
			dir = int(directions[by][bx])
		}
		ctx.setDirection(dir)
		ctx.do, ctx.io = dstOrigin, srcOrigin
		cdefFilterBlock16(&ctx, u.blockWidth)
	}
	return nil
}

// filterUnitBlocksU8 binds the Go SIMD 8-bit unit-level loop on arm64. See
// filter_u8_dispatch.go for why this is a build-tag binding.
func filterUnitBlocksU8(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) error {
	return filterUnitBlocksU8SIMD(dst, dstStride, input, inputOrigin, blocks, directions, variances, u)
}

// filterUnitBlocksU8SIMD mirrors filterUnitBlocksSIMD with in-place 8-bit
// stores: per-unit invariants hoisted, per-block strength adjust, and blocks
// whose adjusted primary strength and secondary strength are both zero are
// skipped outright (in place there is nothing to copy, dav1d's
// cdef_apply_tmpl.c skip). Output is bit-identical to
// filterUnitBlocksU8PureGo.
func filterUnitBlocksU8SIMD(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) error {
	if u.blockWidth != 8 && u.blockWidth != 4 {
		return filterUnitBlocksU8PureGo(dst, dstStride, input, inputOrigin, blocks, directions, variances, u)
	}
	secondaryStrength := u.secondaryStrength
	ctx := cdefSIMDCtx{
		in:     input,
		out8:   dst,
		dstStr: dstStride,
		height: u.blockHeight,
		u8:     true,
	}
	ctx.setSecondary(secondaryStrength, u.damping)
	strength := u.primaryStrength
	if !u.lumaAdjust {
		ctx.setPrimary(strength, u.damping, 0)
	}
	// Interior routing (dav1d edges == 0xf): the sentinel-free footprint is
	// proven once, at the first fused block, and reused for the unit.
	interiorEligible := u.blockWidth == 8 && u.blockHeight%2 == 0
	interiorChecked := false
	useInterior := false
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
			ctx.setPrimary(strength, u.damping, 0)
		}
		if strength == 0 && secondaryStrength == 0 {
			continue
		}
		if !interiorChecked && interiorEligible && strength != 0 && secondaryStrength != 0 {
			interiorChecked = true
			useInterior = cdefUnitInteriorU8(input, inputOrigin, blocks, u.bwLog2, u.bhLog2)
		}
		srcOrigin := inputOrigin + ((by * BStride) << u.bhLog2) + (bx << u.bwLog2)
		dstOrigin := (by<<u.bhLog2)*dstStride + (bx << u.bwLog2)
		dir := 0
		if u.primaryStrength != 0 {
			dir = int(directions[by][bx])
		}
		ctx.setDirection(dir)
		ctx.do, ctx.io = dstOrigin, srcOrigin
		if useInterior && strength != 0 && secondaryStrength != 0 {
			cdefFilterBlock8InteriorSIMD(&ctx)
		} else {
			cdefFilterBlockSIMD(&ctx, u.blockWidth)
		}
	}
	return nil
}

// cdefNarrowU8 keeps the low byte of each lane, the same bits as byte(y) in
// the pure-Go reference (XTN on arm64).
func cdefNarrowU8(y archsimd.Int16x8) archsimd.Uint8x16 {
	return y.ToBits().TruncToUint8()
}

// cdefSarI16 is an arithmetic right shift of each lane by a count from
// cdefShiftCountOf. The count vector is built once per block: ShiftAllRight
// with a constant would rebuild it (MOVD + INS + DUP) on every row.
func cdefSarI16(v archsimd.Int16x8, sh cdefShiftCount) archsimd.Int16x8 {
	return v.Shift(sh)
}

type cdefR8 = [8]struct{}

const (
	cdefByteCenter = 2*BStride + 2
	cdefByteMaxOff = 4*BStride + 4
	cdefByteWinRow = 4*BStride + 20  // max tap (4*BStride+4) + 16
	cdefByteWin8   = 5*BStride + 20  // max tap + BStride + 16
	cdefByteWin8x8 = 11*BStride + 20 // 8 rows: base + 6*BStride + cdefByteWin8
	cdefByteWin4   = 7*BStride + 20  // max tap + 3*BStride + 16
)

func u8ByteInteriorAvailable() bool { return true }

type cdefByteDirOffs struct {
	p0p, p0m uint
	p1p, p1m uint
	s0p, s0m uint
	s1p, s1m uint
	s2p, s2m uint
	s3p, s3m uint
}

var cdefByteDirTable = func() (t [8]cdefByteDirOffs) {
	for dir := range 8 {
		pri0 := int(cdefDirections[dir+2][0])
		pri1 := int(cdefDirections[dir+2][1])
		sec0, sec1 := int(cdefDirections[dir+4][0]), int(cdefDirections[dir][0])
		sec2, sec3 := int(cdefDirections[dir+4][1]), int(cdefDirections[dir][1])
		t[dir] = cdefByteDirOffs{
			p0p: uint(cdefByteCenter + pri0), p0m: uint(cdefByteCenter - pri0),
			p1p: uint(cdefByteCenter + pri1), p1m: uint(cdefByteCenter - pri1),
			s0p: uint(cdefByteCenter + sec0), s0m: uint(cdefByteCenter - sec0),
			s1p: uint(cdefByteCenter + sec1), s1m: uint(cdefByteCenter - sec1),
			s2p: uint(cdefByteCenter + sec2), s2m: uint(cdefByteCenter - sec2),
			s3p: uint(cdefByteCenter + sec3), s3m: uint(cdefByteCenter - sec3),
		}
	}
	return t
}()

var cdefByteSplatU8 = func() (t [16][16]uint8) {
	for v := range 16 {
		for i := range 16 {
			t[v][i] = uint8(v)
		}
	}
	return t
}()

var cdefByteSplatNegI8 = func() (t [8][16]int8) {
	for v := range 8 {
		for i := range 16 {
			t[v][i] = -int8(v)
		}
	}
	return t
}()

var cdefByteTapVecs = func() (t [2][2][16]int8) {
	for p := range 2 {
		for i := range 16 {
			t[p][0][i] = int8(cdefPrimaryTaps[p][0])
			t[p][1][i] = int8(cdefPrimaryTaps[p][1])
		}
	}
	return t
}()

type cdefByteCtx struct {
	input  []byte
	dst    []byte
	srcOff int
	dstOff int
	dstStr int
	height int

	dir *cdefByteDirOffs

	priStr, secStr archsimd.Uint8x16
	priSh, secSh   archsimd.Int8x16
	tap0, tap1     archsimd.Int8x16
}

// filterUnitBlocksU8Byte is filterUnitBlocksU8PureGo on a byte tap buffer:
// per-block luma strength adjust, blocks with both strengths zero skipped
// (dav1d cdef_apply_tmpl.c), and the pri / sec / pri+sec kernel picked per
// block as dav1d's cdef_filter_block_*_neon does.
func filterUnitBlocksU8Byte(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) {
	sec := u.secondaryStrength
	ctx := cdefByteCtx{
		input:  input,
		dst:    dst,
		dstStr: dstStride,
		height: u.blockHeight,
	}
	if sec != 0 {
		ctx.secStr = archsimd.LoadUint8x16Array(&cdefByteSplatU8[sec&15])
		ctx.secSh = archsimd.LoadInt8x16Array(&cdefByteSplatNegI8[constrainShift(sec, u.damping)&7])
	}

	strength := u.primaryStrength
	prevStrength := -1
	if !u.lumaAdjust {
		prevStrength = strength
		if strength != 0 {
			ctx.priStr = archsimd.LoadUint8x16Array(&cdefByteSplatU8[strength&15])
			ctx.priSh = archsimd.LoadInt8x16Array(&cdefByteSplatNegI8[constrainShift(strength, u.damping)&7])
			ctx.tap0 = archsimd.LoadInt8x16Array(&cdefByteTapVecs[strength&1][0])
			ctx.tap1 = archsimd.LoadInt8x16Array(&cdefByteTapVecs[strength&1][1])
		}
	}

	bwLog2 := u.bwLog2
	bhLog2 := u.bhLog2
	isW8 := u.blockWidth == 8
	hasPri := u.primaryStrength != 0

	if !u.lumaAdjust {
		if strength == 0 && sec == 0 {
			return
		}
		var fn func(*cdefByteCtx)
		if isW8 {
			switch {
			case strength != 0 && sec != 0:
				fn = cdefByteFilter8PriSec
			case strength != 0:
				fn = cdefByteFilter8Pri
			default:
				fn = cdefByteFilter8Sec
			}
		} else {
			switch {
			case strength != 0 && sec != 0:
				fn = cdefByteFilter4PriSec
			case strength != 0:
				fn = cdefByteFilter4Pri
			default:
				fn = cdefByteFilter4Sec
			}
		}
		for _, block := range blocks {
			by := int(block.BY)
			bx := int(block.BX)
			dir := 0
			if hasPri {
				dir = int(directions[by][bx]) & 7
			}
			ctx.srcOff = inputOrigin + ((by * BStride) << bhLog2) + (bx << bwLog2)
			ctx.dstOff = (by<<bhLog2)*dstStride + (bx << bwLog2)
			ctx.dir = &cdefByteDirTable[dir]
			fn(&ctx)
		}
		return
	}

	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
			if strength != prevStrength {
				prevStrength = strength
				if strength != 0 {
					ctx.priStr = archsimd.LoadUint8x16Array(&cdefByteSplatU8[strength&15])
					ctx.priSh = archsimd.LoadInt8x16Array(&cdefByteSplatNegI8[constrainShift(strength, u.damping)&7])
					ctx.tap0 = archsimd.LoadInt8x16Array(&cdefByteTapVecs[strength&1][0])
					ctx.tap1 = archsimd.LoadInt8x16Array(&cdefByteTapVecs[strength&1][1])
				}
			}
		}
		if strength == 0 && sec == 0 {
			continue
		}
		dir := 0
		if hasPri {
			dir = int(directions[by][bx]) & 7
		}
		ctx.srcOff = inputOrigin + ((by * BStride) << bhLog2) + (bx << bwLog2)
		ctx.dstOff = (by<<bhLog2)*dstStride + (bx << bwLog2)
		ctx.dir = &cdefByteDirTable[dir]
		if isW8 {
			switch {
			case strength != 0 && sec != 0:
				cdefByteFilter8PriSec(&ctx)
			case strength != 0:
				cdefByteFilter8Pri(&ctx)
			default:
				cdefByteFilter8Sec(&ctx)
			}
			continue
		}
		switch {
		case strength != 0 && sec != 0:
			cdefByteFilter4PriSec(&ctx)
		case strength != 0:
			cdefByteFilter4Pri(&ctx)
		default:
			cdefByteFilter4Sec(&ctx)
		}
	}
}

// cdefByteLoad8x2Rows packs the 8 bytes at r0[off:] (low half) and r1[off:]
// (high half). Callers bound off <= cdefByteMaxOff so BCE drops all checks.
func cdefByteLoad8x2Rows(r0, r1 *[cdefByteWinRow]byte, off uint) archsimd.Uint8x16 {
	a := archsimd.LoadUint8x16(r0[off : off+16]).ReshapeToUint64s()
	b := archsimd.LoadUint8x16(r1[off : off+16]).ReshapeToUint64s()
	return a.InterleaveLo(b).ReshapeToUint8s()
}

// cdefByteLoad4x4Rows packs the 4 bytes at r0..r3[off:] in row order.
// Callers bound off <= cdefByteMaxOff so BCE drops all checks.
func cdefByteLoad4x4Rows(r0, r1, r2, r3 *[cdefByteWinRow]byte, off uint) archsimd.Uint8x16 {
	a := archsimd.LoadUint8x16(r0[off : off+16]).ReshapeToUint32s()
	b := archsimd.LoadUint8x16(r1[off : off+16]).ReshapeToUint32s()
	c := archsimd.LoadUint8x16(r2[off : off+16]).ReshapeToUint32s()
	d := archsimd.LoadUint8x16(r3[off : off+16]).ReshapeToUint32s()
	ab := a.InterleaveLo(b).ReshapeToUint64s()
	cd := c.InterleaveLo(d).ReshapeToUint64s()
	return ab.InterleaveLo(cd).ReshapeToUint8s()
}

// cdefByteConstrain is handle_pixel_8's constrain(): the signed byte
// min(|t-x|, max(0, str - (|t-x| >> shift))) carrying the sign of t-x.
func cdefByteConstrain(t, x, str archsimd.Uint8x16, sh archsimd.Int8x16) archsimd.Int8x16 {
	pos := t.SubSaturated(x)
	neg := x.SubSaturated(t)
	lim := str.SubSaturated(pos.Or(neg).Shift(sh))
	return pos.Min(lim).BitsToInt8().Sub(neg.Min(lim).BitsToInt8())
}

// cdefByteFinish turns the two accumulators (a seeded with -1) into the
// filtered pixels: x + ((8 + sum - (sum < 0)) >> 4) saturated to [0, 255].
func cdefByteFinish(x archsimd.Uint8x16, a, b archsimd.Int8x16, four archsimd.Int8x16, bias archsimd.Uint8x16) archsimd.Uint8x16 {
	r := a.Average(b)
	odd := a.Xor(b).ShiftAllLeft(7).ShiftAllRight(7)
	h := r.Add(odd.And(r.ShiftAllRight(7)))
	d := h.Add(four).ShiftAllRight(3)
	return x.Xor(bias).BitsToInt8().AddSaturated(d).ToBits().Xor(bias)
}

func cdefByteFilter8PriSec(ctx *cdefByteCtx) {
	d := ctx.dir
	p0p, p0m := d.p0p, d.p0m
	p1p, p1m := d.p1p, d.p1m
	s0p, s0m := d.s0p, d.s0m
	s1p, s1m := d.s1p, d.s1m
	s2p, s2m := d.s2p, d.s2m
	s3p, s3m := d.s3p, d.s3m
	if p0p > cdefByteMaxOff || p0m > cdefByteMaxOff || p1p > cdefByteMaxOff || p1m > cdefByteMaxOff ||
		s0p > cdefByteMaxOff || s0m > cdefByteMaxOff || s1p > cdefByteMaxOff || s1m > cdefByteMaxOff ||
		s2p > cdefByteMaxOff || s2m > cdefByteMaxOff || s3p > cdefByteMaxOff || s3m > cdefByteMaxOff {
		return
	}
	priStr, secStr := ctx.priStr, ctx.secStr
	priSh, secSh := ctx.priSh, ctx.secSh
	tap0, tap1 := ctx.tap0, ctx.tap1
	neg1 := archsimd.BroadcastInt8x16(-1)
	four := archsimd.BroadcastInt8x16(4)
	bias := archsimd.BroadcastUint8x16(0x80)
	input, dst := ctx.input, ctx.dst
	base := ctx.srcOff - cdefByteCenter
	dstOff, dstStr := ctx.dstOff, ctx.dstStr
	if ctx.height == 8 && unsafeutil.RowsFit(len(dst), dstOff, dstStr, 8, 8) {
		in := (*[cdefByteWin8x8]byte)(input[base : base+cdefByteWin8x8])
		db := unsafeutil.Block8At[cdefR8](unsafeutil.NewRows(dst, dstOff, dstStr, 8, 8), 0, 0)
		for s := 0; s < 4; s++ {
			inOff := s * 2 * BStride
			r0 := (*[cdefByteWinRow]byte)(in[inOff : inOff+cdefByteWinRow])
			r1 := (*[cdefByteWinRow]byte)(in[inOff+BStride : inOff+BStride+cdefByteWinRow])
			x := cdefByteLoad8x2Rows(r0, r1, cdefByteCenter)
			mn, mx := x, x
			p, q := cdefByteLoad8x2Rows(r0, r1, p0p), cdefByteLoad8x2Rows(r0, r1, p0m)
			mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
			a := cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap0, neg1)
			b := cdefByteConstrain(q, x, priStr, priSh).Mul(tap0)
			p, q = cdefByteLoad8x2Rows(r0, r1, p1p), cdefByteLoad8x2Rows(r0, r1, p1m)
			mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
			a = cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap1, a)
			b = cdefByteConstrain(q, x, priStr, priSh).MulAdd(tap1, b)
			p, q = cdefByteLoad8x2Rows(r0, r1, s0p), cdefByteLoad8x2Rows(r0, r1, s0m)
			mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
			ka, kb := cdefByteConstrain(p, x, secStr, secSh), cdefByteConstrain(q, x, secStr, secSh)
			p, q = cdefByteLoad8x2Rows(r0, r1, s1p), cdefByteLoad8x2Rows(r0, r1, s1m)
			mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
			ka, kb = ka.Add(cdefByteConstrain(p, x, secStr, secSh)), kb.Add(cdefByteConstrain(q, x, secStr, secSh))
			a, b = a.Add(ka).Add(ka), b.Add(kb).Add(kb)
			p, q = cdefByteLoad8x2Rows(r0, r1, s2p), cdefByteLoad8x2Rows(r0, r1, s2m)
			mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
			a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
			p, q = cdefByteLoad8x2Rows(r0, r1, s3p), cdefByteLoad8x2Rows(r0, r1, s3m)
			mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
			a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
			y := cdefByteFinish(x, a, b, four, bias).Min(mx).Max(mn).ReshapeToUint64s()
			binary.LittleEndian.PutUint64(db.Row8(2 * s)[:], y.GetElem(0))
			binary.LittleEndian.PutUint64(db.Row8(2*s + 1)[:], y.GetElem(1))
		}
		return
	}
	for h := ctx.height; h > 0; h -= 2 {
		w := (*[cdefByteWin8]byte)(input[base : base+cdefByteWin8])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		x := cdefByteLoad8x2Rows(r0, r1, cdefByteCenter)
		mn, mx := x, x
		p, q := cdefByteLoad8x2Rows(r0, r1, p0p), cdefByteLoad8x2Rows(r0, r1, p0m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a := cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap0, neg1)
		b := cdefByteConstrain(q, x, priStr, priSh).Mul(tap0)
		p, q = cdefByteLoad8x2Rows(r0, r1, p1p), cdefByteLoad8x2Rows(r0, r1, p1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a = cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap1, a)
		b = cdefByteConstrain(q, x, priStr, priSh).MulAdd(tap1, b)
		p, q = cdefByteLoad8x2Rows(r0, r1, s0p), cdefByteLoad8x2Rows(r0, r1, s0m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		ka, kb := cdefByteConstrain(p, x, secStr, secSh), cdefByteConstrain(q, x, secStr, secSh)
		p, q = cdefByteLoad8x2Rows(r0, r1, s1p), cdefByteLoad8x2Rows(r0, r1, s1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		ka, kb = ka.Add(cdefByteConstrain(p, x, secStr, secSh)), kb.Add(cdefByteConstrain(q, x, secStr, secSh))
		a, b = a.Add(ka).Add(ka), b.Add(kb).Add(kb)
		p, q = cdefByteLoad8x2Rows(r0, r1, s2p), cdefByteLoad8x2Rows(r0, r1, s2m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
		p, q = cdefByteLoad8x2Rows(r0, r1, s3p), cdefByteLoad8x2Rows(r0, r1, s3m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
		y := cdefByteFinish(x, a, b, four, bias).Min(mx).Max(mn).ReshapeToUint64s()
		binary.LittleEndian.PutUint64(dst[dstOff:dstOff+8], y.GetElem(0))
		binary.LittleEndian.PutUint64(dst[dstOff+dstStr:dstOff+dstStr+8], y.GetElem(1))
		base += 2 * BStride
		dstOff += 2 * dstStr
	}
}

func cdefByteFilter8Pri(ctx *cdefByteCtx) {
	d := ctx.dir
	p0p, p0m := d.p0p, d.p0m
	p1p, p1m := d.p1p, d.p1m
	if p0p > cdefByteMaxOff || p0m > cdefByteMaxOff || p1p > cdefByteMaxOff || p1m > cdefByteMaxOff {
		return
	}
	priStr, priSh := ctx.priStr, ctx.priSh
	tap0, tap1 := ctx.tap0, ctx.tap1
	neg1 := archsimd.BroadcastInt8x16(-1)
	four := archsimd.BroadcastInt8x16(4)
	bias := archsimd.BroadcastUint8x16(0x80)
	input, dst := ctx.input, ctx.dst
	base := ctx.srcOff - cdefByteCenter
	dstOff, dstStr := ctx.dstOff, ctx.dstStr
	if ctx.height == 8 && unsafeutil.RowsFit(len(dst), dstOff, dstStr, 8, 8) {
		in := (*[cdefByteWin8x8]byte)(input[base : base+cdefByteWin8x8])
		db := unsafeutil.Block8At[cdefR8](unsafeutil.NewRows(dst, dstOff, dstStr, 8, 8), 0, 0)
		for s := 0; s < 4; s++ {
			inOff := s * 2 * BStride
			r0 := (*[cdefByteWinRow]byte)(in[inOff : inOff+cdefByteWinRow])
			r1 := (*[cdefByteWinRow]byte)(in[inOff+BStride : inOff+BStride+cdefByteWinRow])
			x := cdefByteLoad8x2Rows(r0, r1, cdefByteCenter)
			a := cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, p0p), x, priStr, priSh).MulAdd(tap0, neg1)
			b := cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, p0m), x, priStr, priSh).Mul(tap0)
			a = cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, p1p), x, priStr, priSh).MulAdd(tap1, a)
			b = cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, p1m), x, priStr, priSh).MulAdd(tap1, b)
			y := cdefByteFinish(x, a, b, four, bias).ReshapeToUint64s()
			binary.LittleEndian.PutUint64(db.Row8(2 * s)[:], y.GetElem(0))
			binary.LittleEndian.PutUint64(db.Row8(2*s + 1)[:], y.GetElem(1))
		}
		return
	}
	for h := ctx.height; h > 0; h -= 2 {
		w := (*[cdefByteWin8]byte)(input[base : base+cdefByteWin8])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		x := cdefByteLoad8x2Rows(r0, r1, cdefByteCenter)
		a := cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, p0p), x, priStr, priSh).MulAdd(tap0, neg1)
		b := cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, p0m), x, priStr, priSh).Mul(tap0)
		a = cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, p1p), x, priStr, priSh).MulAdd(tap1, a)
		b = cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, p1m), x, priStr, priSh).MulAdd(tap1, b)
		y := cdefByteFinish(x, a, b, four, bias).ReshapeToUint64s()
		binary.LittleEndian.PutUint64(dst[dstOff:dstOff+8], y.GetElem(0))
		binary.LittleEndian.PutUint64(dst[dstOff+dstStr:dstOff+dstStr+8], y.GetElem(1))
		base += 2 * BStride
		dstOff += 2 * dstStr
	}
}

func cdefByteFilter8Sec(ctx *cdefByteCtx) {
	d := ctx.dir
	s0p, s0m := d.s0p, d.s0m
	s1p, s1m := d.s1p, d.s1m
	s2p, s2m := d.s2p, d.s2m
	s3p, s3m := d.s3p, d.s3m
	if s0p > cdefByteMaxOff || s0m > cdefByteMaxOff || s1p > cdefByteMaxOff || s1m > cdefByteMaxOff ||
		s2p > cdefByteMaxOff || s2m > cdefByteMaxOff || s3p > cdefByteMaxOff || s3m > cdefByteMaxOff {
		return
	}
	secStr, secSh := ctx.secStr, ctx.secSh
	neg1 := archsimd.BroadcastInt8x16(-1)
	four := archsimd.BroadcastInt8x16(4)
	bias := archsimd.BroadcastUint8x16(0x80)
	input, dst := ctx.input, ctx.dst
	base := ctx.srcOff - cdefByteCenter
	dstOff, dstStr := ctx.dstOff, ctx.dstStr
	if ctx.height == 8 && unsafeutil.RowsFit(len(dst), dstOff, dstStr, 8, 8) {
		in := (*[cdefByteWin8x8]byte)(input[base : base+cdefByteWin8x8])
		db := unsafeutil.Block8At[cdefR8](unsafeutil.NewRows(dst, dstOff, dstStr, 8, 8), 0, 0)
		for s := 0; s < 4; s++ {
			inOff := s * 2 * BStride
			r0 := (*[cdefByteWinRow]byte)(in[inOff : inOff+cdefByteWinRow])
			r1 := (*[cdefByteWinRow]byte)(in[inOff+BStride : inOff+BStride+cdefByteWinRow])
			x := cdefByteLoad8x2Rows(r0, r1, cdefByteCenter)
			ka := cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s0p), x, secStr, secSh).
				Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s1p), x, secStr, secSh))
			kb := cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s0m), x, secStr, secSh).
				Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s1m), x, secStr, secSh))
			a := ka.Add(ka).Add(neg1).
				Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s2p), x, secStr, secSh)).
				Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s3p), x, secStr, secSh))
			b := kb.Add(kb).
				Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s2m), x, secStr, secSh)).
				Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s3m), x, secStr, secSh))
			y := cdefByteFinish(x, a, b, four, bias).ReshapeToUint64s()
			binary.LittleEndian.PutUint64(db.Row8(2 * s)[:], y.GetElem(0))
			binary.LittleEndian.PutUint64(db.Row8(2*s + 1)[:], y.GetElem(1))
		}
		return
	}
	for h := ctx.height; h > 0; h -= 2 {
		w := (*[cdefByteWin8]byte)(input[base : base+cdefByteWin8])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		x := cdefByteLoad8x2Rows(r0, r1, cdefByteCenter)
		ka := cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s0p), x, secStr, secSh).
			Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s1p), x, secStr, secSh))
		kb := cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s0m), x, secStr, secSh).
			Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s1m), x, secStr, secSh))
		a := ka.Add(ka).Add(neg1).
			Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s2p), x, secStr, secSh)).
			Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s3p), x, secStr, secSh))
		b := kb.Add(kb).
			Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s2m), x, secStr, secSh)).
			Add(cdefByteConstrain(cdefByteLoad8x2Rows(r0, r1, s3m), x, secStr, secSh))
		y := cdefByteFinish(x, a, b, four, bias).ReshapeToUint64s()
		binary.LittleEndian.PutUint64(dst[dstOff:dstOff+8], y.GetElem(0))
		binary.LittleEndian.PutUint64(dst[dstOff+dstStr:dstOff+dstStr+8], y.GetElem(1))
		base += 2 * BStride
		dstOff += 2 * dstStr
	}
}

func cdefByteStore4x4(dst []byte, dstOff, dstStr int, v archsimd.Uint8x16) {
	y := v.ReshapeToUint32s()
	if unsafeutil.GridFit(len(dst), dstOff, dstStr, 4, 4) {
		dg := unsafeutil.NewGrid(dst, dstOff, dstStr, 4, 4)
		binary.LittleEndian.PutUint32(dg.Row4(0, 0)[:], y.GetElem(0))
		binary.LittleEndian.PutUint32(dg.Row4(1, 0)[:], y.GetElem(1))
		binary.LittleEndian.PutUint32(dg.Row4(2, 0)[:], y.GetElem(2))
		binary.LittleEndian.PutUint32(dg.Row4(3, 0)[:], y.GetElem(3))
		return
	}
	binary.LittleEndian.PutUint32(dst[dstOff:dstOff+4], y.GetElem(0))
	binary.LittleEndian.PutUint32(dst[dstOff+dstStr:dstOff+dstStr+4], y.GetElem(1))
	binary.LittleEndian.PutUint32(dst[dstOff+2*dstStr:dstOff+2*dstStr+4], y.GetElem(2))
	binary.LittleEndian.PutUint32(dst[dstOff+3*dstStr:dstOff+3*dstStr+4], y.GetElem(3))
}

func cdefByteFilter4PriSec(ctx *cdefByteCtx) {
	d := ctx.dir
	p0p, p0m := d.p0p, d.p0m
	p1p, p1m := d.p1p, d.p1m
	s0p, s0m := d.s0p, d.s0m
	s1p, s1m := d.s1p, d.s1m
	s2p, s2m := d.s2p, d.s2m
	s3p, s3m := d.s3p, d.s3m
	if p0p > cdefByteMaxOff || p0m > cdefByteMaxOff || p1p > cdefByteMaxOff || p1m > cdefByteMaxOff ||
		s0p > cdefByteMaxOff || s0m > cdefByteMaxOff || s1p > cdefByteMaxOff || s1m > cdefByteMaxOff ||
		s2p > cdefByteMaxOff || s2m > cdefByteMaxOff || s3p > cdefByteMaxOff || s3m > cdefByteMaxOff {
		return
	}
	priStr, secStr := ctx.priStr, ctx.secStr
	priSh, secSh := ctx.priSh, ctx.secSh
	tap0, tap1 := ctx.tap0, ctx.tap1
	neg1 := archsimd.BroadcastInt8x16(-1)
	four := archsimd.BroadcastInt8x16(4)
	bias := archsimd.BroadcastUint8x16(0x80)
	input, dst := ctx.input, ctx.dst
	base := ctx.srcOff - cdefByteCenter
	dstOff, dstStr := ctx.dstOff, ctx.dstStr
	if ctx.height == 4 {
		w := (*[cdefByteWin4]byte)(input[base : base+cdefByteWin4])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		r2 := (*[cdefByteWinRow]byte)(w[2*BStride : 2*BStride+cdefByteWinRow])
		r3 := (*[cdefByteWinRow]byte)(w[3*BStride : 3*BStride+cdefByteWinRow])
		x := cdefByteLoad4x4Rows(r0, r1, r2, r3, cdefByteCenter)
		mn, mx := x, x
		p, q := cdefByteLoad4x4Rows(r0, r1, r2, r3, p0p), cdefByteLoad4x4Rows(r0, r1, r2, r3, p0m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a := cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap0, neg1)
		b := cdefByteConstrain(q, x, priStr, priSh).Mul(tap0)
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, p1p), cdefByteLoad4x4Rows(r0, r1, r2, r3, p1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a = cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap1, a)
		b = cdefByteConstrain(q, x, priStr, priSh).MulAdd(tap1, b)
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, s0p), cdefByteLoad4x4Rows(r0, r1, r2, r3, s0m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		ka, kb := cdefByteConstrain(p, x, secStr, secSh), cdefByteConstrain(q, x, secStr, secSh)
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, s1p), cdefByteLoad4x4Rows(r0, r1, r2, r3, s1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		ka, kb = ka.Add(cdefByteConstrain(p, x, secStr, secSh)), kb.Add(cdefByteConstrain(q, x, secStr, secSh))
		a, b = a.Add(ka).Add(ka), b.Add(kb).Add(kb)
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, s2p), cdefByteLoad4x4Rows(r0, r1, r2, r3, s2m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, s3p), cdefByteLoad4x4Rows(r0, r1, r2, r3, s3m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
		cdefByteStore4x4(dst, dstOff, dstStr, cdefByteFinish(x, a, b, four, bias).Min(mx).Max(mn))
		return
	}
	for h := ctx.height; h > 0; h -= 4 {
		w := (*[cdefByteWin4]byte)(input[base : base+cdefByteWin4])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		r2 := (*[cdefByteWinRow]byte)(w[2*BStride : 2*BStride+cdefByteWinRow])
		r3 := (*[cdefByteWinRow]byte)(w[3*BStride : 3*BStride+cdefByteWinRow])
		x := cdefByteLoad4x4Rows(r0, r1, r2, r3, cdefByteCenter)
		mn, mx := x, x
		p, q := cdefByteLoad4x4Rows(r0, r1, r2, r3, p0p), cdefByteLoad4x4Rows(r0, r1, r2, r3, p0m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a := cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap0, neg1)
		b := cdefByteConstrain(q, x, priStr, priSh).Mul(tap0)
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, p1p), cdefByteLoad4x4Rows(r0, r1, r2, r3, p1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a = cdefByteConstrain(p, x, priStr, priSh).MulAdd(tap1, a)
		b = cdefByteConstrain(q, x, priStr, priSh).MulAdd(tap1, b)
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, s0p), cdefByteLoad4x4Rows(r0, r1, r2, r3, s0m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		ka, kb := cdefByteConstrain(p, x, secStr, secSh), cdefByteConstrain(q, x, secStr, secSh)
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, s1p), cdefByteLoad4x4Rows(r0, r1, r2, r3, s1m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		ka, kb = ka.Add(cdefByteConstrain(p, x, secStr, secSh)), kb.Add(cdefByteConstrain(q, x, secStr, secSh))
		a, b = a.Add(ka).Add(ka), b.Add(kb).Add(kb)
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, s2p), cdefByteLoad4x4Rows(r0, r1, r2, r3, s2m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
		p, q = cdefByteLoad4x4Rows(r0, r1, r2, r3, s3p), cdefByteLoad4x4Rows(r0, r1, r2, r3, s3m)
		mn, mx = mn.Min(p).Min(q), mx.Max(p).Max(q)
		a, b = a.Add(cdefByteConstrain(p, x, secStr, secSh)), b.Add(cdefByteConstrain(q, x, secStr, secSh))
		cdefByteStore4x4(dst, dstOff, dstStr, cdefByteFinish(x, a, b, four, bias).Min(mx).Max(mn))
		base += 4 * BStride
		dstOff += 4 * dstStr
	}
}

func cdefByteFilter4Pri(ctx *cdefByteCtx) {
	d := ctx.dir
	p0p, p0m := d.p0p, d.p0m
	p1p, p1m := d.p1p, d.p1m
	if p0p > cdefByteMaxOff || p0m > cdefByteMaxOff || p1p > cdefByteMaxOff || p1m > cdefByteMaxOff {
		return
	}
	priStr, priSh := ctx.priStr, ctx.priSh
	tap0, tap1 := ctx.tap0, ctx.tap1
	neg1 := archsimd.BroadcastInt8x16(-1)
	four := archsimd.BroadcastInt8x16(4)
	bias := archsimd.BroadcastUint8x16(0x80)
	input, dst := ctx.input, ctx.dst
	base := ctx.srcOff - cdefByteCenter
	dstOff, dstStr := ctx.dstOff, ctx.dstStr
	if ctx.height == 4 {
		w := (*[cdefByteWin4]byte)(input[base : base+cdefByteWin4])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		r2 := (*[cdefByteWinRow]byte)(w[2*BStride : 2*BStride+cdefByteWinRow])
		r3 := (*[cdefByteWinRow]byte)(w[3*BStride : 3*BStride+cdefByteWinRow])
		x := cdefByteLoad4x4Rows(r0, r1, r2, r3, cdefByteCenter)
		a := cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, p0p), x, priStr, priSh).MulAdd(tap0, neg1)
		b := cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, p0m), x, priStr, priSh).Mul(tap0)
		a = cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, p1p), x, priStr, priSh).MulAdd(tap1, a)
		b = cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, p1m), x, priStr, priSh).MulAdd(tap1, b)
		cdefByteStore4x4(dst, dstOff, dstStr, cdefByteFinish(x, a, b, four, bias))
		return
	}
	for h := ctx.height; h > 0; h -= 4 {
		w := (*[cdefByteWin4]byte)(input[base : base+cdefByteWin4])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		r2 := (*[cdefByteWinRow]byte)(w[2*BStride : 2*BStride+cdefByteWinRow])
		r3 := (*[cdefByteWinRow]byte)(w[3*BStride : 3*BStride+cdefByteWinRow])
		x := cdefByteLoad4x4Rows(r0, r1, r2, r3, cdefByteCenter)
		a := cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, p0p), x, priStr, priSh).MulAdd(tap0, neg1)
		b := cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, p0m), x, priStr, priSh).Mul(tap0)
		a = cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, p1p), x, priStr, priSh).MulAdd(tap1, a)
		b = cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, p1m), x, priStr, priSh).MulAdd(tap1, b)
		cdefByteStore4x4(dst, dstOff, dstStr, cdefByteFinish(x, a, b, four, bias))
		base += 4 * BStride
		dstOff += 4 * dstStr
	}
}

func cdefByteFilter4Sec(ctx *cdefByteCtx) {
	d := ctx.dir
	s0p, s0m := d.s0p, d.s0m
	s1p, s1m := d.s1p, d.s1m
	s2p, s2m := d.s2p, d.s2m
	s3p, s3m := d.s3p, d.s3m
	if s0p > cdefByteMaxOff || s0m > cdefByteMaxOff || s1p > cdefByteMaxOff || s1m > cdefByteMaxOff ||
		s2p > cdefByteMaxOff || s2m > cdefByteMaxOff || s3p > cdefByteMaxOff || s3m > cdefByteMaxOff {
		return
	}
	secStr, secSh := ctx.secStr, ctx.secSh
	neg1 := archsimd.BroadcastInt8x16(-1)
	four := archsimd.BroadcastInt8x16(4)
	bias := archsimd.BroadcastUint8x16(0x80)
	input, dst := ctx.input, ctx.dst
	base := ctx.srcOff - cdefByteCenter
	dstOff, dstStr := ctx.dstOff, ctx.dstStr
	if ctx.height == 4 {
		w := (*[cdefByteWin4]byte)(input[base : base+cdefByteWin4])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		r2 := (*[cdefByteWinRow]byte)(w[2*BStride : 2*BStride+cdefByteWinRow])
		r3 := (*[cdefByteWinRow]byte)(w[3*BStride : 3*BStride+cdefByteWinRow])
		x := cdefByteLoad4x4Rows(r0, r1, r2, r3, cdefByteCenter)
		ka := cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s0p), x, secStr, secSh).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s1p), x, secStr, secSh))
		kb := cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s0m), x, secStr, secSh).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s1m), x, secStr, secSh))
		a := ka.Add(ka).Add(neg1).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s2p), x, secStr, secSh)).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s3p), x, secStr, secSh))
		b := kb.Add(kb).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s2m), x, secStr, secSh)).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s3m), x, secStr, secSh))
		cdefByteStore4x4(dst, dstOff, dstStr, cdefByteFinish(x, a, b, four, bias))
		return
	}
	for h := ctx.height; h > 0; h -= 4 {
		w := (*[cdefByteWin4]byte)(input[base : base+cdefByteWin4])
		r0 := (*[cdefByteWinRow]byte)(w[0:cdefByteWinRow])
		r1 := (*[cdefByteWinRow]byte)(w[BStride : BStride+cdefByteWinRow])
		r2 := (*[cdefByteWinRow]byte)(w[2*BStride : 2*BStride+cdefByteWinRow])
		r3 := (*[cdefByteWinRow]byte)(w[3*BStride : 3*BStride+cdefByteWinRow])
		x := cdefByteLoad4x4Rows(r0, r1, r2, r3, cdefByteCenter)
		ka := cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s0p), x, secStr, secSh).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s1p), x, secStr, secSh))
		kb := cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s0m), x, secStr, secSh).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s1m), x, secStr, secSh))
		a := ka.Add(ka).Add(neg1).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s2p), x, secStr, secSh)).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s3p), x, secStr, secSh))
		b := kb.Add(kb).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s2m), x, secStr, secSh)).
			Add(cdefByteConstrain(cdefByteLoad4x4Rows(r0, r1, r2, r3, s3m), x, secStr, secSh))
		cdefByteStore4x4(dst, dstOff, dstStr, cdefByteFinish(x, a, b, four, bias))
		base += 4 * BStride
		dstOff += 4 * dstStr
	}
}

// cdefScanU16RowsAtMost255 checks eight samples per official Go SIMD vector.
// It only loads complete vectors inside each row and checks the exact row tail
// with scalar loads, so the final row can end at the slice boundary.
func cdefScanU16RowsAtMost255(input []uint16, start, rows, cols int) bool {
	for r := 0; r < rows; r++ {
		rowStart := start + r*BStride
		row := input[rowStart : rowStart+cols]
		var acc = archsimd.BroadcastUint16x8(0)
		c := 0
		for ; c+8 <= cols; c += 8 {
			chunk := row[c : c+8]
			acc = acc.Or(archsimd.LoadUint16x8Array((*[8]uint16)(chunk)))
		}
		for ; c < cols; c++ {
			if row[c] > 0xFF {
				return false
			}
		}
		if acc.ReduceMax() > 0xFF {
			return false
		}
	}
	return true
}

// cdefTapReach is the maximum |offset|, in samples, that any CDEF tap reads
// beyond the current pixel along either axis (cdefDirections secondary taps
// reach two rows and two columns). A block whose whole footprint expanded by
// this reach holds only real samples has no VeryLarge sentinel in any tap.
const cdefTapReach = 2

// cdefUnitInteriorU8 reports whether every CDEF tap the unit's blocks read
// lands on a real 0..255 sample, i.e. the tap footprint of the block bounding
// box (expanded by cdefTapReach on every side) is free of the VeryLarge border
// sentinel. That is exactly dav1d's edges == 0xf predicate for goav1's tap
// buffer: when it holds, the interior .16b kernels may narrow the uint16 taps
// to uint8 losslessly. The scan is one strided pass over the footprint, run
// once per unit and amortized across all of the unit's blocks.
func cdefUnitInteriorU8(input []uint16, inputOrigin int, blocks []BlockPosition, bwLog2 int, bhLog2 int) bool {
	if len(input) == 0 || len(blocks) == 0 {
		return false
	}
	blockWidth := 1 << bwLog2
	blockHeight := 1 << bhLog2
	minBX := int(blocks[0].BX)
	maxBX := minBX
	minBY := int(blocks[0].BY)
	maxBY := minBY
	for i := 1; i < len(blocks); i++ {
		bx := int(blocks[i].BX)
		by := int(blocks[i].BY)
		if bx < minBX {
			minBX = bx
		}
		if bx > maxBX {
			maxBX = bx
		}
		if by < minBY {
			minBY = by
		}
		if by > maxBY {
			maxBY = by
		}
	}
	// inputOrigin addresses the unit's first interior sample (buffer row
	// VerticalBorder, column HorizontalBorder). Walk out cdefTapReach on each
	// side of the block bounding box.
	topLeft := inputOrigin + ((minBY * BStride) << bhLog2) + (minBX << bwLog2)
	scanStart := topLeft - cdefTapReach*BStride - cdefTapReach
	nRows := (maxBY-minBY+1)*blockHeight + 2*cdefTapReach
	nCols := (maxBX-minBX+1)*blockWidth + 2*cdefTapReach
	if scanStart < 0 {
		return false
	}
	end := scanStart + (nRows-1)*BStride + nCols
	if end > len(input) {
		return false
	}
	return cdefScanU16RowsAtMost255(input, scanStart, nRows, nCols)
}

// cdefPack2Rows packs two rows of eight samples into one 16-byte vector: row
// a in bytes 0..7, row b in bytes 8..15. The samples are 0..255 by the
// interior contract, so the narrowing truncation is exact.
func cdefPack2Rows(a, b archsimd.Int16x8) archsimd.Uint8x16 {
	a8 := a.ToBits().TruncToUint8().ReshapeToUint64s()
	b8 := b.ToBits().TruncToUint8().ReshapeToUint64s()
	return a8.InterleaveLo(b8).ReshapeToUint8s()
}

// cdefConstrainU8 returns the tap value after constrain(), in byte lanes:
// x + clamp(t-x, -lim, +lim) with lim = max(str - (|t-x| >> shift), 0),
// written as min(max(t, x-lim), x+lim) with saturating bounds.
func cdefConstrainU8(t, x, str archsimd.Uint8x16, sh archsimd.Int8x16) archsimd.Uint8x16 {
	ab := t.Max(x).Sub(t.Min(x))
	lim := str.SubSaturated(ab.Shift(sh))
	return t.Max(x.SubSaturated(lim)).Min(x.AddSaturated(lim))
}

// cdefFilterBlock8InteriorSIMD filters a fused 8-wide block whose taps are all
// real 8-bit samples, two rows per vector. Height is even by routing contract.
// Primary and secondary are both enabled (the unit loop routes single-strength
// blocks to cdefFilterBlockSIMD), so the clamp is always active.
func cdefFilterBlock8InteriorSIMD(ctx *cdefSIMDCtx) {
	priStr := archsimd.BroadcastUint8x16(uint8(ctx.priStrength))
	secStr := archsimd.BroadcastUint8x16(uint8(ctx.secStrength))
	priSh := archsimd.BroadcastInt8x16(-int8(ctx.priShift))
	secSh := archsimd.BroadcastInt8x16(-int8(ctx.secShift))
	priTap0 := archsimd.BroadcastInt16x8(int16(ctx.priTap0))
	priTap1 := archsimd.BroadcastInt16x8(int16(ctx.priTap1))
	secTap0 := archsimd.BroadcastInt16x8(int16(ctx.secTap0))
	secTap1 := archsimd.BroadcastInt16x8(int16(ctx.secTap1))
	eight := archsimd.BroadcastInt16x8(8)
	wtot := archsimd.BroadcastInt16x8(int16(2*ctx.priTap0 + 2*ctx.priTap1 + 4*ctx.secTap0 + 4*ctx.secTap1))
	p0p, p0m, p1p, p1m := ctx.priTaps()
	s0p, s0m, s1p, s1m := ctx.secTaps0()
	s2p, s2m, s3p, s3m := ctx.secTaps1()
	in, base, height := ctx.rows()
	_, out8, do, ds := ctx.outRows()
	// Counted by row pair: a step-2 counter keeps a second check per window.
	for i := 0; i < height>>1; i++ {
		r := 2 * i
		w0, w1 := cdefRows2(in, base+r*BStride)
		x := cdefPack2Rows(cdefLdW1(w0, cdefWinReach), cdefLdW1(w1, cdefWinReach))
		p0 := cdefPack2Rows(cdefLdW1(w0, p0p), cdefLdW1(w1, p0p))
		p1 := cdefPack2Rows(cdefLdW1(w0, p0m), cdefLdW1(w1, p0m))
		p2 := cdefPack2Rows(cdefLdW1(w0, p1p), cdefLdW1(w1, p1p))
		p3 := cdefPack2Rows(cdefLdW1(w0, p1m), cdefLdW1(w1, p1m))
		s0 := cdefPack2Rows(cdefLdW1(w0, s0p), cdefLdW1(w1, s0p))
		s1 := cdefPack2Rows(cdefLdW1(w0, s0m), cdefLdW1(w1, s0m))
		s2 := cdefPack2Rows(cdefLdW1(w0, s1p), cdefLdW1(w1, s1p))
		s3 := cdefPack2Rows(cdefLdW1(w0, s1m), cdefLdW1(w1, s1m))
		s4 := cdefPack2Rows(cdefLdW1(w0, s2p), cdefLdW1(w1, s2p))
		s5 := cdefPack2Rows(cdefLdW1(w0, s2m), cdefLdW1(w1, s2m))
		s6 := cdefPack2Rows(cdefLdW1(w0, s3p), cdefLdW1(w1, s3p))
		s7 := cdefPack2Rows(cdefLdW1(w0, s3m), cdefLdW1(w1, s3m))

		// Clamp bounds fold over the raw taps and the centre, as maxClip/min4 do.
		mx := x.Max(p0).Max(p1).Max(p2).Max(p3).Max(s0).Max(s1).Max(s2).Max(s3)
		mx = mx.Max(s4).Max(s5).Max(s6).Max(s7)
		mn := x.Min(p0).Min(p1).Min(p2).Min(p3).Min(s0).Min(s1).Min(s2).Min(s3)
		mn = mn.Min(s4).Min(s5).Min(s6).Min(s7)

		c0 := cdefConstrainU8(p0, x, priStr, priSh)
		c1 := cdefConstrainU8(p1, x, priStr, priSh)
		c2 := cdefConstrainU8(p2, x, priStr, priSh)
		c3 := cdefConstrainU8(p3, x, priStr, priSh)
		d0 := cdefConstrainU8(s0, x, secStr, secSh)
		d1 := cdefConstrainU8(s1, x, secStr, secSh)
		d2 := cdefConstrainU8(s2, x, secStr, secSh)
		d3 := cdefConstrainU8(s3, x, secStr, secSh)
		d4 := cdefConstrainU8(s4, x, secStr, secSh)
		d5 := cdefConstrainU8(s5, x, secStr, secSh)
		d6 := cdefConstrainU8(s6, x, secStr, secSh)
		d7 := cdefConstrainU8(s7, x, secStr, secSh)

		// Widen to 16-bit halves and take the signed differences from x.
		// (UXTL / UXTL2 written out: through a two-result helper each widen
		// left an inline-mark NOP in the row loop.)
		xl, xh := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		ml, mh := mx.ExtendLo8ToUint16().BitsToInt16(), mx.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		nl, nh := mn.ExtendLo8ToUint16().BitsToInt16(), mn.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		c0l, c0h := c0.ExtendLo8ToUint16().BitsToInt16(), c0.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		c1l, c1h := c1.ExtendLo8ToUint16().BitsToInt16(), c1.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		c2l, c2h := c2.ExtendLo8ToUint16().BitsToInt16(), c2.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		c3l, c3h := c3.ExtendLo8ToUint16().BitsToInt16(), c3.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		d0l, d0h := d0.ExtendLo8ToUint16().BitsToInt16(), d0.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		d1l, d1h := d1.ExtendLo8ToUint16().BitsToInt16(), d1.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		d2l, d2h := d2.ExtendLo8ToUint16().BitsToInt16(), d2.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		d3l, d3h := d3.ExtendLo8ToUint16().BitsToInt16(), d3.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		d4l, d4h := d4.ExtendLo8ToUint16().BitsToInt16(), d4.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		d5l, d5h := d5.ExtendLo8ToUint16().BitsToInt16(), d5.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		d6l, d6h := d6.ExtendLo8ToUint16().BitsToInt16(), d6.HiToLo().ExtendLo8ToUint16().BitsToInt16()
		d7l, d7h := d7.ExtendLo8ToUint16().BitsToInt16(), d7.HiToLo().ExtendLo8ToUint16().BitsToInt16()

		// sum = primary taps (weights {p0,p1}) + secondary taps (weights {2,1}).
		// sum = sum_i w_i*(t_i - x) = sum_i w_i*t_i - wtot*x, with wtot the sum of
		// the tap weights (two taps per primary weight, four per secondary weight).
		sumL := c0l.Add(c1l).Mul(priTap0).Add(c2l.Add(c3l).Mul(priTap1))
		sumL = sumL.Add(d0l.Add(d1l).Add(d2l).Add(d3l).Mul(secTap0)).Add(d4l.Add(d5l).Add(d6l).Add(d7l).Mul(secTap1))
		sumL = sumL.Sub(xl.Mul(wtot))
		sumH := c0h.Add(c1h).Mul(priTap0).Add(c2h.Add(c3h).Mul(priTap1))
		sumH = sumH.Add(d0h.Add(d1h).Add(d2h).Add(d3h).Mul(secTap0)).Add(d4h.Add(d5h).Add(d6h).Add(d7h).Mul(secTap1))
		sumH = sumH.Sub(xh.Mul(wtot))

		// y = x + ((8 + sum - (sum<0)) >> 4), clamped to [min, max].
		yL := xl.Add(sumL.Add(sumL.ShiftAllRight(15)).Add(eight).ShiftAllRight(4)).Max(nl).Min(ml)
		yH := xh.Add(sumH.Add(sumH.ShiftAllRight(15)).Add(eight).ShiftAllRight(4)).Max(nh).Min(mh)
		cdefPut8(out8, do+r*ds, yL.ToBits().TruncToUint8().ReshapeToUint64s().GetElem(0))
		cdefPut8(out8, do+(r+1)*ds, yH.ToBits().TruncToUint8().ReshapeToUint64s().GetElem(0))
	}
}

func u8SecondaryOnlyByteInteriorAvailable() bool { return true }

func filterFrameBlocksU8SecondaryOnlyByteInterior(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, blockWidth int, blockHeight int, strength int, shift int) bool {
	str := archsimd.BroadcastUint8x16(uint8(strength))
	sh := archsimd.BroadcastInt8x16(-int8(shift))
	for i := 0; i < len(blocks); {
		block := blocks[i]
		srcOffset := inputOrigin + (int(block.BY)*blockHeight)*BStride + int(block.BX)*blockWidth
		dstOffset := (int(block.BY)*blockHeight)*dstStride + int(block.BX)*blockWidth
		groupEntries := secondaryOnlyByteGroupEntries(blocks, i, blockWidth)
		if groupEntries > 1 {
			cdefFilterSecondaryOnly16xSIMD(dst[dstOffset:], dstStride, input, srcOffset, blockHeight, str, sh)
			i += groupEntries
			continue
		}
		if blockWidth == 8 {
			cdefFilterSecondaryOnly8xSIMD(dst[dstOffset:], dstStride, input, srcOffset, blockHeight, str, sh)
		} else {
			cdefFilterSecondaryOnly4xSIMD(dst[dstOffset:], dstStride, input, srcOffset, blockHeight, str, sh)
		}
		i++
	}
	return true
}

// secondaryOnlyByteGroupEntries returns the exact number of consecutive list
// entries that cover one 16-pixel row strip. It never searches ahead, sorts,
// or fills a missing block; a failure leaves the current entry for the
// original single-block kernel.
func secondaryOnlyByteGroupEntries(blocks []BlockPosition, index int, blockWidth int) int {
	groupEntries := 0
	switch blockWidth {
	case 8:
		groupEntries = 2
	case 4:
		groupEntries = 4
	default:
		return 1
	}
	if index < 0 || index+groupEntries > len(blocks) {
		return 1
	}
	first := blocks[index]
	for offset := 1; offset < groupEntries; offset++ {
		next := blocks[index+offset]
		if next.BY != first.BY || int(next.BX) != int(first.BX)+offset {
			return 1
		}
	}
	return groupEntries
}

func cdefConstrainSecondaryByteSIMD(t, x, strength archsimd.Uint8x16, shift archsimd.Int8x16) archsimd.Int8x16 {
	return cdefByteConstrain(t, x, strength, shift)
}

func cdefRoundSecondaryByteSIMD(sum archsimd.Int8x16) archsimd.Int8x16 {
	return sum.Add(sum.ShiftAllRight(7)).Add(archsimd.BroadcastInt8x16(8)).ShiftAllRight(4)
}

// The secondary-only taps sit at x+-1, x+-2 and y+-1, y+-2 of each output
// sample, so a row group's taps lie in one window of the tap buffer that
// starts two rows and two columns before the group: one bounds check per
// group, every tap a constant offset inside it. cdefSecWindowArgs bounds the
// origin, height and stride so that check is one compare.
const (
	cdefSecReach = 2*BStride + 2
	cdefSecWin16 = 4*BStride + 4 + 16 // one 16-wide row
	cdefSecWin8  = 5*BStride + 4 + 8  // two 8-wide rows
	cdefSecWin4  = 7*BStride + 4 + 4  // four 4-wide rows
)

// cdefSecWindowArgs panics unless the block origin leaves room for the taps
// before it and the height and destination stride are in range.
func cdefSecWindowArgs(so, height, dstStride int) {
	if so < cdefSecReach || so > 1<<24 || uint(height) > 1<<10 || uint(dstStride) > 1<<30 {
		panic("cdef: secondary-only block outside the tap buffer")
	}
}

// cdefSecLoad16 loads the 16 tap bytes at (dx, dy) relative to the window's
// output row.
func cdefSecLoad16(w *[cdefSecWin16]byte, dx, dy int) archsimd.Uint8x16 {
	o := (dy+2)*BStride + dx + 2
	return archsimd.LoadUint8x16Array((*[16]uint8)(w[o : o+16]))
}

// cdefSecLoad8x2 packs the 8 tap bytes at (dx, dy) and (dx, dy+1).
func cdefSecLoad8x2(w *[cdefSecWin8]byte, dx, dy int) archsimd.Uint8x16 {
	o := (dy+2)*BStride + dx + 2
	var packed archsimd.Uint64x2
	packed = packed.SetElem(0, cdefSecLE64(w[o:o+8]))
	packed = packed.SetElem(1, cdefSecLE64(w[o+BStride:o+BStride+8]))
	return packed.ReshapeToUint8s()
}

// cdefSecLoad4x4 packs the 4 tap bytes at (dx, dy) .. (dx, dy+3).
func cdefSecLoad4x4(w *[cdefSecWin4]byte, dx, dy int) archsimd.Uint8x16 {
	o := (dy+2)*BStride + dx + 2
	var packed archsimd.Uint32x4
	packed = packed.SetElem(0, cdefSecLE32(w[o:o+4]))
	packed = packed.SetElem(1, cdefSecLE32(w[o+BStride:o+BStride+4]))
	packed = packed.SetElem(2, cdefSecLE32(w[o+2*BStride:o+2*BStride+4]))
	packed = packed.SetElem(3, cdefSecLE32(w[o+3*BStride:o+3*BStride+4]))
	return packed.ReshapeToUint8s()
}

func cdefSecLE64(b []byte) uint64 {
	_ = b[7]
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

func cdefSecLE32(b []byte) uint32 {
	_ = b[3]
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func cdefSecPut64(b []byte, v uint64) {
	_ = b[7]
	b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24), byte(v>>32), byte(v>>40), byte(v>>48), byte(v>>56)
}

func cdefSecPut32(b []byte, v uint32) {
	_ = b[3]
	b[0], b[1], b[2], b[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
}

func cdefFilterSecondaryOnly16xSIMD(dst []byte, dstStride int, input []byte, so int, height int, strength archsimd.Uint8x16, shift archsimd.Int8x16) {
	cdefSecWindowArgs(so, height, dstStride)
	base := so - cdefSecReach
	for y := 0; y < height; y++ {
		o := base + y*BStride
		w := (*[cdefSecWin16]byte)(input[o : o+cdefSecWin16])
		x := cdefSecLoad16(w, 0, 0)
		// The four weight-2 taps sum before doubling (2a+2b+2c+2d wraps to the
		// same int8 lanes as 2(a+b+c+d)), and every tap is consumed on the line
		// that loads it, so no inlined helper call leaves an inline-mark NOP.
		d := cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, 1, 0), x, strength, shift).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, -1, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, 0, 1), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, 0, -1), x, strength, shift))
		sum := d.Add(d).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, 2, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, -2, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, 0, 2), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad16(w, 0, -2), x, strength, shift))
		pixels := x.Add(cdefRoundSecondaryByteSIMD(sum).ToBits())
		do := y * dstStride
		pixels.StoreArray((*[16]uint8)(dst[do : do+16]))
	}
}

func cdefFilterSecondaryOnly8xSIMD(dst []byte, dstStride int, input []byte, so int, height int, strength archsimd.Uint8x16, shift archsimd.Int8x16) {
	cdefSecWindowArgs(so, height, dstStride)
	base := so - cdefSecReach
	// One iteration per row pair, counted by pair (a step-2 counter keeps a
	// second check per window).
	for i := 0; i < height>>1; i++ {
		y := 2 * i
		o := base + y*BStride
		w := (*[cdefSecWin8]byte)(input[o : o+cdefSecWin8])
		x := cdefSecLoad8x2(w, 0, 0)
		// The four weight-2 taps sum before doubling (2a+2b+2c+2d wraps to the
		// same int8 lanes as 2(a+b+c+d)), and every tap is consumed on the line
		// that loads it, so no inlined helper call leaves an inline-mark NOP.
		d := cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, 1, 0), x, strength, shift).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, -1, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, 0, 1), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, 0, -1), x, strength, shift))
		sum := d.Add(d).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, 2, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, -2, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, 0, 2), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad8x2(w, 0, -2), x, strength, shift))
		packed := x.Add(cdefRoundSecondaryByteSIMD(sum).ToBits()).ReshapeToUint64s()
		do := y * dstStride
		cdefSecPut64(dst[do:do+8], packed.GetElem(0))
		cdefSecPut64(dst[do+dstStride:do+dstStride+8], packed.GetElem(1))
	}
}

func cdefFilterSecondaryOnly4xSIMD(dst []byte, dstStride int, input []byte, so int, height int, strength archsimd.Uint8x16, shift archsimd.Int8x16) {
	cdefSecWindowArgs(so, height, dstStride)
	base := so - cdefSecReach
	for i := 0; i < height>>2; i++ {
		y := 4 * i
		o := base + y*BStride
		w := (*[cdefSecWin4]byte)(input[o : o+cdefSecWin4])
		x := cdefSecLoad4x4(w, 0, 0)
		// The four weight-2 taps sum before doubling (2a+2b+2c+2d wraps to the
		// same int8 lanes as 2(a+b+c+d)), and every tap is consumed on the line
		// that loads it, so no inlined helper call leaves an inline-mark NOP.
		d := cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, 1, 0), x, strength, shift).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, -1, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, 0, 1), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, 0, -1), x, strength, shift))
		sum := d.Add(d).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, 2, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, -2, 0), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, 0, 2), x, strength, shift)).
			Add(cdefConstrainSecondaryByteSIMD(cdefSecLoad4x4(w, 0, -2), x, strength, shift))
		packed := x.Add(cdefRoundSecondaryByteSIMD(sum).ToBits()).ReshapeToUint32s()
		do := y * dstStride
		cdefSecPut32(dst[do:do+4], packed.GetElem(0))
		cdefSecPut32(dst[do+dstStride:do+dstStride+4], packed.GetElem(1))
		cdefSecPut32(dst[do+2*dstStride:do+2*dstStride+4], packed.GetElem(2))
		cdefSecPut32(dst[do+3*dstStride:do+3*dstStride+4], packed.GetElem(3))
	}
}
