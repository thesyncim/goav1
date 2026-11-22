// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for upstream attribution.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD CDEF direction search (simd/archsimd, GOEXPERIMENT=simd).
//
// Byte-exact twin of findDirectionScalar / findDirectionU8Scalar written in
// safe Go with archsimd vector operations.
//
// Partial-array index layout (matches direction.go):
//
//	p0[k] = sum over i+j=k        (anti-diagonal, k=0..14): row i at lane offset i
//	p4[k] = sum over 7+i-j=k      (diagonal,      k=0..14): p0 of rows (r7..r0) reversed
//	p1[k] = sum over i+(j>>1)=k   (k=0..10): pairwise-sum(row i) at lane i
//	p3[k] = sum over 3+i-(j>>1)=k (k=0..10): p1 of pairwise-sums (s7..s0) reversed
//	p5[k] = sum over 3-(i>>1)+j=k (k=0..10): paired rows r67,r45,r23,r01 at 0,1,2,3
//	p7[k] = sum over (i>>1)+j=k   (k=0..10): paired rows r01,r23,r45,r67 at 0,1,2,3
//	p2[i] = row i sum (straight);  p6[j] = column sum (straight)
//
// Because diagonal and odd cost weights are symmetric around the center index
// (w[k] == w[14-k] for p0/p4, w[k] == w[10-k] for p1/p3), p4 has the exact same
// cost as p0 on vertically flipped rows (r7..r0) and p3 has the exact same cost
// as p1 on vertically flipped pairwise sums (s7..s0). That removes all per-row
// VTBL reversals and table loads.

package cdef

import "simd/archsimd"

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
	bias := archsimd.BroadcastInt16x8(128)
	var zero archsimd.Int16x8
	s := img[:7*stride+8]
	r0 := archsimd.LoadUint8x16(s[0:16]).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
	s = s[stride:]
	r1 := archsimd.LoadUint8x16(s[0:16]).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
	s = s[stride:]
	r2 := archsimd.LoadUint8x16(s[0:16]).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
	s = s[stride:]
	r3 := archsimd.LoadUint8x16(s[0:16]).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
	s = s[stride:]
	r4 := archsimd.LoadUint8x16(s[0:16]).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
	s = s[stride:]
	r5 := archsimd.LoadUint8x16(s[0:16]).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
	s = s[stride:]
	r6 := archsimd.LoadUint8x16(s[0:16]).ExtendLo8ToUint16().BitsToInt16().Sub(bias)
	r7 := archsimd.LoadUint8x16(s[stride-8 : stride+8]).HiToLo().ExtendLo8ToUint16().BitsToInt16().Sub(bias)

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

// directionSIMDBound reports whether the Go SIMD direction kernels are the
// dispatch targets on this build: always, on arm64 (NEON is mandatory).
func directionSIMDBound() bool { return true }
