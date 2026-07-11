// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for upstream attribution.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD CDEF direction search (simd/archsimd, GOEXPERIMENT=simd).
//
// Byte-exact twin of findDirectionScalar / cdefFindDirectionNEONAsm. The 8x8
// block is loaded as eight Int16x8 rows (one SIMD load per row), shifted right by
// coeffShift and biased by -128 in-register (matching int32(img>>shift)-128).
//
// All eight AV1 directional partial-sum arrays are built entirely in vector
// registers, exactly like dav1d's AArch64 kernel: each per-row contribution is a
// lane-shifted add, where the lane shift is the byte-lane VEXT exposed here as
// Int16x8.concatShiftRight (archsimd's ConcatShiftBytesRight). Partial-sum
// magnitudes stay in int16 (<= 8*127 = 1016). The finishDirection reduction then
// squares (MulWidenLo/Hi -> int32) and weights against the libaom div-table, and
// the argmax + variance tail matches the scalar `>` scan (lowest index wins ties,
// variance = (best - opposite) >> 10).
//
// Partial-array index layout (matches direction.go):
//
//	p0[k] = sum over i+j=k        (anti-diagonal, k=0..14): row i at lane offset i
//	p4[k] = sum over 7+i-j=k      (diagonal,      k=0..14): rev(row i) at lane i
//	p1[k] = sum over i+(j>>1)=k   (k=0..10): pairwise-sum(row i) at lane i
//	p3[k] = sum over 3+i-(j>>1)=k (k=0..10): rev(pairwise-sum(row i)) at lane i
//	p5[k] = sum over 3-(i>>1)+j=k (k=0..10): row i at lane 3-(i>>1)
//	p7[k] = sum over (i>>1)+j=k   (k=0..10): row i at lane (i>>1)
//	p2[i] = row i sum (straight);  p6[j] = column sum (straight)

package cdef

import (
	"simd/archsimd"
	"unsafe"
)

var (
	// Diagonal cost weights div[1..8] split lo (k=0..3) / hi (k=4..7).
	cdefDivDiagLo = [4]int32{840, 420, 280, 210}
	cdefDivDiagHi = [4]int32{168, 140, 120, 105}
	// Odd cost paired weights div[2],div[4],div[6] (lane 3 unused).
	cdefDivOdd = [4]int32{420, 210, 140, 0}
)

func init() {
	findDirectionImpl = findDirectionSIMD
	findDirectionDualImpl = findDirectionDualSIMD
	findDirectionU8Impl = findDirectionU8SIMD
	findDirectionDualU8Impl = findDirectionDualU8SIMD
}

// cdefDirLoadRowPtr loads eight uint16 samples at raw pointer p as Int16x8 (no
// slice bounds check in the hot loads).
func cdefDirLoadRowPtr(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p))
}

// vext is the raw VEXT on the int16 view: x.ConcatShiftBytesRight(y, s) returns
// bytes [s..s+15] of concat(y, x) — i.e. the receiver x is the HIGH operand and
// y is the LOW operand. s must be a compile-time constant for a single-
// instruction VEXT.
func vext(hi, lo archsimd.Int16x8, shiftBytes uint64) archsimd.Int16x8 {
	hb := hi.ToBits().ReshapeToUint8s()
	lb := lo.ToBits().ReshapeToUint8s()
	return hb.ConcatShiftBytesRight(lb, shiftBytes).ReshapeToUint16s().BitsToInt16()
}

// placeLo returns the low half (indices 0..7) of row r placed at lane offset n
// (0..7): out[k] = r[k-n] for k>=n, else 0.
func placeLo(r, zero archsimd.Int16x8, n uint64) archsimd.Int16x8 {
	if n == 0 {
		return r
	}
	return vext(r, zero, 16-2*n)
}

// placeHi returns the high half (indices 8..14, in lanes 0..6) of row r placed at
// lane offset n: the r lanes that overflow past index 7.
func placeHi(r, zero archsimd.Int16x8, n uint64) archsimd.Int16x8 {
	if n == 0 {
		return zero
	}
	return vext(zero, r, 16-2*n)
}

// revRowLo returns the low half (idx 0..7) of the full 8-lane reverse of row r
// (out[k]=r[7-k]) placed at lane offset n, in one VTBL.
func revRowLo(r archsimd.Int16x8, n int) archsimd.Int16x8 {
	idx := archsimd.LoadUint8x16Array(&cdefP4RevLoTbl[n])
	return r.ToBits().ReshapeToUint8s().LookupOrZero(idx).ReshapeToUint16s().BitsToInt16()
}

// revRowHi returns the high half (idx 8..14) of the same placement (nonzero for
// n>=1).
func revRowHi(r archsimd.Int16x8, n int) archsimd.Int16x8 {
	idx := archsimd.LoadUint8x16Array(&cdefP4RevHiTbl[n])
	return r.ToBits().ReshapeToUint8s().LookupOrZero(idx).ReshapeToUint16s().BitsToInt16()
}

var cdefP4RevLoTbl = [8][16]uint8{
	{14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1},
	{255, 255, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3},
	{255, 255, 255, 255, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5},
	{255, 255, 255, 255, 255, 255, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7},
	{255, 255, 255, 255, 255, 255, 255, 255, 14, 15, 12, 13, 10, 11, 8, 9},
	{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 14, 15, 12, 13, 10, 11},
	{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 14, 15, 12, 13},
	{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 14, 15},
}

var cdefP4RevHiTbl = [8][16]uint8{
	1: {0, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
	2: {2, 3, 0, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
	3: {4, 5, 2, 3, 0, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
	4: {6, 7, 4, 5, 2, 3, 0, 1, 255, 255, 255, 255, 255, 255, 255, 255},
	5: {8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 255, 255, 255, 255, 255, 255},
	6: {10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 255, 255, 255, 255},
	7: {12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 255, 255},
}

func findDirectionSIMD(img []uint16, stride int, coeffShift int) (int, int32) {
	bias := archsimd.BroadcastInt16x8(128)
	shiftV := archsimd.BroadcastInt16x8(-int16(coeffShift)) // negative = right (VSSHL)
	zero := archsimd.BroadcastInt16x8(0)

	// Walk a raw pointer by stride*2 bytes per row; the callers guarantee the 8x8
	// block fits (blockFits + stride check in FindDirection), the same contract
	// the NEON asm relies on. This avoids a per-row slice bounds check.
	p := unsafe.Pointer(&img[0])
	sb := uintptr(stride) * 2
	r0 := cdefDirLoadRowPtr(p).Shift(shiftV).Sub(bias)
	r1 := cdefDirLoadRowPtr(unsafe.Add(p, sb)).Shift(shiftV).Sub(bias)
	r2 := cdefDirLoadRowPtr(unsafe.Add(p, 2*sb)).Shift(shiftV).Sub(bias)
	r3 := cdefDirLoadRowPtr(unsafe.Add(p, 3*sb)).Shift(shiftV).Sub(bias)
	r4 := cdefDirLoadRowPtr(unsafe.Add(p, 4*sb)).Shift(shiftV).Sub(bias)
	r5 := cdefDirLoadRowPtr(unsafe.Add(p, 5*sb)).Shift(shiftV).Sub(bias)
	r6 := cdefDirLoadRowPtr(unsafe.Add(p, 6*sb)).Shift(shiftV).Sub(bias)
	r7 := cdefDirLoadRowPtr(unsafe.Add(p, 7*sb)).Shift(shiftV).Sub(bias)

	// --- straight partials (cost 2 and cost 6) ---
	// partial[6][j] = column sums.
	col := r0.Add(r1).Add(r2).Add(r3).Add(r4).Add(r5).Add(r6).Add(r7)
	c6 := cdefStraightCostFromVec(col)
	// partial[2][i] = row sums; assemble into a Int16x8 then square*105.
	rowSums := zero.
		SetElem(0, r0.ReduceSum()).SetElem(1, r1.ReduceSum()).
		SetElem(2, r2.ReduceSum()).SetElem(3, r3.ReduceSum()).
		SetElem(4, r4.ReduceSum()).SetElem(5, r5.ReduceSum()).
		SetElem(6, r6.ReduceSum()).SetElem(7, r7.ReduceSum())
	c2 := cdefStraightCostFromVec(rowSums)

	// --- diagonal family p0 (anti-diagonal): row i at lane offset i ---
	p0lo := r0.
		Add(placeLo(r1, zero, 1)).Add(placeLo(r2, zero, 2)).
		Add(placeLo(r3, zero, 3)).Add(placeLo(r4, zero, 4)).
		Add(placeLo(r5, zero, 5)).Add(placeLo(r6, zero, 6)).Add(placeLo(r7, zero, 7))
	p0hi := placeHi(r1, zero, 1).
		Add(placeHi(r2, zero, 2)).Add(placeHi(r3, zero, 3)).
		Add(placeHi(r4, zero, 4)).Add(placeHi(r5, zero, 5)).
		Add(placeHi(r6, zero, 6)).Add(placeHi(r7, zero, 7))
	c0 := cdefDiagCostVec(p0lo, p0hi)

	// --- diagonal family p4 (diagonal): rev(row i) at lane offset i ---
	// Fuse the full 8-lane reverse + placement into one VTBL per (row, half).
	p4lo := revRowLo(r0, 0).
		Add(revRowLo(r1, 1)).Add(revRowLo(r2, 2)).
		Add(revRowLo(r3, 3)).Add(revRowLo(r4, 4)).
		Add(revRowLo(r5, 5)).Add(revRowLo(r6, 6)).Add(revRowLo(r7, 7))
	p4hi := revRowHi(r1, 1).
		Add(revRowHi(r2, 2)).Add(revRowHi(r3, 3)).
		Add(revRowHi(r4, 4)).Add(revRowHi(r5, 5)).
		Add(revRowHi(r6, 6)).Add(revRowHi(r7, 7))
	c4 := cdefDiagCostVec(p4lo, p4hi)

	// --- odd family p7: row i at lane (i>>1) (offsets 0,0,1,1,2,2,3,3) ---
	p7lo := r0.Add(r1).
		Add(placeLo(r2, zero, 1)).Add(placeLo(r3, zero, 1)).
		Add(placeLo(r4, zero, 2)).Add(placeLo(r5, zero, 2)).
		Add(placeLo(r6, zero, 3)).Add(placeLo(r7, zero, 3))
	p7hi := placeHi(r2, zero, 1).Add(placeHi(r3, zero, 1)).
		Add(placeHi(r4, zero, 2)).Add(placeHi(r5, zero, 2)).
		Add(placeHi(r6, zero, 3)).Add(placeHi(r7, zero, 3))
	c7 := cdefOddCostVec(p7lo, p7hi)

	// --- odd family p5: row i at lane 3-(i>>1) (offsets 3,3,2,2,1,1,0,0) ---
	p5lo := r6.Add(r7).
		Add(placeLo(r4, zero, 1)).Add(placeLo(r5, zero, 1)).
		Add(placeLo(r2, zero, 2)).Add(placeLo(r3, zero, 2)).
		Add(placeLo(r0, zero, 3)).Add(placeLo(r1, zero, 3))
	p5hi := placeHi(r4, zero, 1).Add(placeHi(r5, zero, 1)).
		Add(placeHi(r2, zero, 2)).Add(placeHi(r3, zero, 2)).
		Add(placeHi(r0, zero, 3)).Add(placeHi(r1, zero, 3))
	c5 := cdefOddCostVec(p5lo, p5hi)

	// --- odd family p1: pairwise-sum(row i) at lane i (offsets 0..7) ---
	// pairwise sum occupies lanes 0..3; placed at lane i -> idx i..i+3 (0..10).
	s0 := r0.ConcatAddPairs(zero)
	s1 := r1.ConcatAddPairs(zero)
	s2 := r2.ConcatAddPairs(zero)
	s3 := r3.ConcatAddPairs(zero)
	s4 := r4.ConcatAddPairs(zero)
	s5 := r5.ConcatAddPairs(zero)
	s6 := r6.ConcatAddPairs(zero)
	s7 := r7.ConcatAddPairs(zero)
	p1lo := s0.
		Add(placeLo(s1, zero, 1)).Add(placeLo(s2, zero, 2)).
		Add(placeLo(s3, zero, 3)).Add(placeLo(s4, zero, 4)).
		Add(placeLo(s5, zero, 5)).Add(placeLo(s6, zero, 6)).Add(placeLo(s7, zero, 7))
	// high half: only s4..s7 reach idx>=8 (s_i occupies i..i+3).
	p1hi := placeHi(s4, zero, 4).
		Add(placeHi(s5, zero, 5)).Add(placeHi(s6, zero, 6)).Add(placeHi(s7, zero, 7))
	c1 := cdefOddCostVec(p1lo, p1hi)

	// --- odd family p3: rev4(pairwise-sum(row i)) at lane i (offsets 0..7) ---
	// Fuse rev4 + placement into a single VTBL per (row, half) so the reversed
	// temporaries never materialize (cuts peak vector liveness and one op/row).
	p3lo := revPlaceLo(s0, 0).
		Add(revPlaceLo(s1, 1)).Add(revPlaceLo(s2, 2)).
		Add(revPlaceLo(s3, 3)).Add(revPlaceLo(s4, 4)).
		Add(revPlaceLo(s5, 5)).Add(revPlaceLo(s6, 6)).Add(revPlaceLo(s7, 7))
	p3hi := revPlaceHi(s5, 5).
		Add(revPlaceHi(s6, 6)).Add(revPlaceHi(s7, 7))
	c3 := cdefOddCostVec(p3lo, p3hi)

	// Argmax with the scalar `>` scan (lowest index wins ties), tracking the
	// opposite cost alongside so the variance needs no second lookup — matches
	// finishDirection exactly. Kept in registers (no cost[] stack array).
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

// revPlaceLo returns the low half (idx 0..7) of rev4(pairwise sums in s lanes
// 0..3) placed at lane offset n, in a single VTBL: out lane k in [n,n+3] = s lane
// 3-(k-n), else 0.
func revPlaceLo(s archsimd.Int16x8, n int) archsimd.Int16x8 {
	idx := archsimd.LoadUint8x16Array(&cdefP3RevLoTbl[n])
	return s.ToBits().ReshapeToUint8s().LookupOrZero(idx).ReshapeToUint16s().BitsToInt16()
}

// revPlaceHi returns the high half (idx 8..14) of the same placement; nonzero
// only for n = 5,6,7 (n<=4 overflows nothing past lane 7).
func revPlaceHi(s archsimd.Int16x8, n int) archsimd.Int16x8 {
	idx := archsimd.LoadUint8x16Array(&cdefP3RevHiTbl[n])
	return s.ToBits().ReshapeToUint8s().LookupOrZero(idx).ReshapeToUint16s().BitsToInt16()
}

// Combined rev4 + place index tables for p3 (out-of-range 255 -> zero lane).
var cdefP3RevLoTbl = [8][16]uint8{
	{6, 7, 4, 5, 2, 3, 0, 1, 255, 255, 255, 255, 255, 255, 255, 255},
	{255, 255, 6, 7, 4, 5, 2, 3, 0, 1, 255, 255, 255, 255, 255, 255},
	{255, 255, 255, 255, 6, 7, 4, 5, 2, 3, 0, 1, 255, 255, 255, 255},
	{255, 255, 255, 255, 255, 255, 6, 7, 4, 5, 2, 3, 0, 1, 255, 255},
	{255, 255, 255, 255, 255, 255, 255, 255, 6, 7, 4, 5, 2, 3, 0, 1},
	{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 6, 7, 4, 5, 2, 3},
	{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 6, 7, 4, 5},
	{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 6, 7},
}

var cdefP3RevHiTbl = [8][16]uint8{
	5: {0, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
	6: {2, 3, 0, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
	7: {4, 5, 2, 3, 0, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
}

// cdefStraightCostFromVec = (sum of squares of the 8 int16 lanes)*105.
func cdefStraightCostFromVec(v archsimd.Int16x8) int32 {
	sq := v.MulWidenLo(v).Add(v.MulWidenHi(v))
	return sq.ReduceSum() * 105
}

// cdefDiagCostVec vectorizes finishDirectionDiagonalCost from the (lo,hi) form:
//
//	sum_{k=0..6} (p[k]^2 + p[14-k]^2)*div[k+1] + p[7]^2*div[8]
//
// lo lanes 0..7 = p[0..7]; hi lanes 0..6 = p[8..14].
func cdefDiagCostVec(lo, hi archsimd.Int16x8) int32 {
	// Forward p[0..7] (lo). Reverse p[14..7] must align lane k with p[14-k].
	// Build rev = [p14,p13,p12,p11,p10,p9,p8,p7]: reverse hi's lanes 0..6 into
	// lanes 0..6 and append p7. Simpler: assemble via SetElem from the scalar
	// squares is costly; instead compute widened squares of lo and of a reversed
	// combination. Do it in int32 to match the scalar exactly.
	// Widen p[0..7] to two Int32x4 (lo0..3, lo4..7).
	f0 := lo.MulWidenLo(lo) // p0^2..p3^2
	f1 := lo.MulWidenHi(lo) // p4^2..p7^2
	// rev vector r = [p14,p13,p12,p11,p10,p9,p8,p7]; then r squared aligns
	// r[k]=p[14-k], so (f[k]+rSq[k]) = p[k]^2+p[14-k]^2 for k=0..7. For k=7 that
	// is p7^2+p7^2; the scalar wants only p7^2*div8, so we halve-correct by using
	// rev's lane7 = 0 instead of p7.
	rev := cdefBuildDiagRev(hi, lo) // [p14..p8, p7-slot=0]
	g0 := rev.MulWidenLo(rev)
	g1 := rev.MulWidenHi(rev)
	sqLo := f0.Add(g0) // (p0^2+p14^2 ... p3^2+p11^2)
	sqHi := f1.Add(g1) // (p4^2+p10^2 ... p7^2+0)
	wLo := archsimd.LoadInt32x4Array(&cdefDivDiagLo)
	wHi := archsimd.LoadInt32x4Array(&cdefDivDiagHi)
	return sqLo.Mul(wLo).Add(sqHi.Mul(wHi)).ReduceSum()
}

// cdefBuildDiagRev builds [p14,p13,p12,p11,p10,p9,p8,0] from hi (=p8..p14 in
// lanes0..6) and lo. Only hi is needed; reverse hi's lanes 0..6 into lanes 0..6
// and zero lane 7.
func cdefBuildDiagRev(hi, lo archsimd.Int16x8) archsimd.Int16x8 {
	_ = lo
	idx := archsimd.LoadUint8x16Array(&cdefDiagRevTbl)
	return hi.ToBits().ReshapeToUint8s().LookupOrZero(idx).ReshapeToUint16s().BitsToInt16()
}

// hi holds p8..p14 in lanes 0..6. We want [p14,p13,p12,p11,p10,p9,p8,0], i.e.
// out lane k = hi lane 6-k for k=0..6, out lane7 = 0. Byte 2k <- 2(6-k).
var cdefDiagRevTbl = [16]uint8{12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 255, 255}

// cdefOddCostVec vectorizes finishDirectionOddCost from (lo,hi):
//
//	(p[3]^2+..+p[7]^2)*div[8] + (p[0]^2+p[10]^2)*div[2]
//	 + (p[1]^2+p[9]^2)*div[4] + (p[2]^2+p[8]^2)*div[6]
//
// lo lanes 0..7 = p[0..7]; hi lanes 0..2 = p[8..10].
func cdefOddCostVec(lo, hi archsimd.Int16x8) int32 {
	loSq0 := lo.MulWidenLo(lo) // p0^2..p3^2
	loSq1 := lo.MulWidenHi(lo) // p4^2..p7^2
	// center = (p3^2+p4^2+p5^2+p6^2+p7^2)*105.
	// p3^2 is loSq0 lane3; p4..7 sq is loSq1 lanes0..3.
	center := (int32(loSq0.GetElem(3)) + loSq1.ReduceSum()) * 105
	// paired: (p0^2+p10^2)*420 + (p1^2+p9^2)*210 + (p2^2+p8^2)*140.
	// fwd = [p0,p1,p2,0]; rev = [p10,p9,p8,0].
	fwdSq := lo.MulWidenLo(lo) // lanes0..2 = p0^2,p1^2,p2^2
	rev := cdefOddRev(hi)      // [p10,p9,p8,0,...]
	revSq := rev.MulWidenLo(rev)
	// zero lane3 of fwdSq so it doesn't contribute (div lane3 = 0 anyway).
	paired := fwdSq.Add(revSq)
	w := archsimd.LoadInt32x4Array(&cdefDivOdd)
	return center + paired.Mul(w).ReduceSum()
}

// cdefOddRev builds [p10,p9,p8,0,...] from hi (=p8,p9,p10 in lanes0..2) via VTBL.
// out lane0=hi lane2, lane1=hi lane1, lane2=hi lane0, lane3=0.
func cdefOddRev(hi archsimd.Int16x8) archsimd.Int16x8 {
	idx := archsimd.LoadUint8x16Array(&cdefOddRevTbl)
	return hi.ToBits().ReshapeToUint8s().LookupOrZero(idx).ReshapeToUint16s().BitsToInt16()
}

var cdefOddRevTbl = [16]uint8{4, 5, 2, 3, 0, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}

func findDirectionDualSIMD(img1 []uint16, img2 []uint16, stride int, coeffShift int) (int, int32, int, int32) {
	dir1, var1 := findDirectionSIMD(img1, stride, coeffShift)
	dir2, var2 := findDirectionSIMD(img2, stride, coeffShift)
	return dir1, var1, dir2, var2
}

// cdefDirLoadRowU8 widens 8 frame bytes at raw pointer p to Int16x8.
func cdefDirLoadRowU8(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(p)).ExtendLo8ToUint16().ConvertToInt16()
}

// cdefDirLoadRowU8Hi widens bytes 8..15 of the 16 bytes at p. The last block
// row uses this with p = rowStart-8 so the 16-byte load ends exactly at the
// row's last needed byte and can never read past the plane allocation (the
// direction search runs on luma only, but monochrome frames end the backing
// buffer right after the Y plane).
func cdefDirLoadRowU8Hi(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(p)).ExtendHi8ToUint16().ConvertToInt16()
}

// findDirectionU8SIMD is findDirectionSIMD reading the 8-bit frame plane
// directly (dav1d cdef_find_dir_8bpc): coeffShift is always 0 at 8 bits, so
// the row prologue is just widen + Sub(128). The partial-sum/cost/argmax body
// is textually identical to findDirectionSIMD — an outlined shared core would
// push eight Int16x8 rows through the stack ABI (see filter4_gosimd_arm64.go).
// Rows 0..6 use 16-byte loads whose 8-byte overread lands in the next row;
// row 7 uses the hi-half load so nothing is read past the block's last byte.
func findDirectionU8SIMD(img []byte, stride int) (int, int32) {
	bias := archsimd.BroadcastInt16x8(128)
	zero := archsimd.BroadcastInt16x8(0)

	p := unsafe.Pointer(&img[0])
	sb := uintptr(stride)
	r0 := cdefDirLoadRowU8(p).Sub(bias)
	r1 := cdefDirLoadRowU8(unsafe.Add(p, sb)).Sub(bias)
	r2 := cdefDirLoadRowU8(unsafe.Add(p, 2*sb)).Sub(bias)
	r3 := cdefDirLoadRowU8(unsafe.Add(p, 3*sb)).Sub(bias)
	r4 := cdefDirLoadRowU8(unsafe.Add(p, 4*sb)).Sub(bias)
	r5 := cdefDirLoadRowU8(unsafe.Add(p, 5*sb)).Sub(bias)
	r6 := cdefDirLoadRowU8(unsafe.Add(p, 6*sb)).Sub(bias)
	r7 := cdefDirLoadRowU8Hi(unsafe.Add(p, 7*sb-8)).Sub(bias)

	// --- straight partials (cost 2 and cost 6) ---
	// partial[6][j] = column sums.
	col := r0.Add(r1).Add(r2).Add(r3).Add(r4).Add(r5).Add(r6).Add(r7)
	c6 := cdefStraightCostFromVec(col)
	// partial[2][i] = row sums; assemble into a Int16x8 then square*105.
	rowSums := zero.
		SetElem(0, r0.ReduceSum()).SetElem(1, r1.ReduceSum()).
		SetElem(2, r2.ReduceSum()).SetElem(3, r3.ReduceSum()).
		SetElem(4, r4.ReduceSum()).SetElem(5, r5.ReduceSum()).
		SetElem(6, r6.ReduceSum()).SetElem(7, r7.ReduceSum())
	c2 := cdefStraightCostFromVec(rowSums)

	// --- diagonal family p0 (anti-diagonal): row i at lane offset i ---
	p0lo := r0.
		Add(placeLo(r1, zero, 1)).Add(placeLo(r2, zero, 2)).
		Add(placeLo(r3, zero, 3)).Add(placeLo(r4, zero, 4)).
		Add(placeLo(r5, zero, 5)).Add(placeLo(r6, zero, 6)).Add(placeLo(r7, zero, 7))
	p0hi := placeHi(r1, zero, 1).
		Add(placeHi(r2, zero, 2)).Add(placeHi(r3, zero, 3)).
		Add(placeHi(r4, zero, 4)).Add(placeHi(r5, zero, 5)).
		Add(placeHi(r6, zero, 6)).Add(placeHi(r7, zero, 7))
	c0 := cdefDiagCostVec(p0lo, p0hi)

	// --- diagonal family p4 (diagonal): rev(row i) at lane offset i ---
	// Fuse the full 8-lane reverse + placement into one VTBL per (row, half).
	p4lo := revRowLo(r0, 0).
		Add(revRowLo(r1, 1)).Add(revRowLo(r2, 2)).
		Add(revRowLo(r3, 3)).Add(revRowLo(r4, 4)).
		Add(revRowLo(r5, 5)).Add(revRowLo(r6, 6)).Add(revRowLo(r7, 7))
	p4hi := revRowHi(r1, 1).
		Add(revRowHi(r2, 2)).Add(revRowHi(r3, 3)).
		Add(revRowHi(r4, 4)).Add(revRowHi(r5, 5)).
		Add(revRowHi(r6, 6)).Add(revRowHi(r7, 7))
	c4 := cdefDiagCostVec(p4lo, p4hi)

	// --- odd family p7: row i at lane (i>>1) (offsets 0,0,1,1,2,2,3,3) ---
	p7lo := r0.Add(r1).
		Add(placeLo(r2, zero, 1)).Add(placeLo(r3, zero, 1)).
		Add(placeLo(r4, zero, 2)).Add(placeLo(r5, zero, 2)).
		Add(placeLo(r6, zero, 3)).Add(placeLo(r7, zero, 3))
	p7hi := placeHi(r2, zero, 1).Add(placeHi(r3, zero, 1)).
		Add(placeHi(r4, zero, 2)).Add(placeHi(r5, zero, 2)).
		Add(placeHi(r6, zero, 3)).Add(placeHi(r7, zero, 3))
	c7 := cdefOddCostVec(p7lo, p7hi)

	// --- odd family p5: row i at lane 3-(i>>1) (offsets 3,3,2,2,1,1,0,0) ---
	p5lo := r6.Add(r7).
		Add(placeLo(r4, zero, 1)).Add(placeLo(r5, zero, 1)).
		Add(placeLo(r2, zero, 2)).Add(placeLo(r3, zero, 2)).
		Add(placeLo(r0, zero, 3)).Add(placeLo(r1, zero, 3))
	p5hi := placeHi(r4, zero, 1).Add(placeHi(r5, zero, 1)).
		Add(placeHi(r2, zero, 2)).Add(placeHi(r3, zero, 2)).
		Add(placeHi(r0, zero, 3)).Add(placeHi(r1, zero, 3))
	c5 := cdefOddCostVec(p5lo, p5hi)

	// --- odd family p1: pairwise-sum(row i) at lane i (offsets 0..7) ---
	// pairwise sum occupies lanes 0..3; placed at lane i -> idx i..i+3 (0..10).
	s0 := r0.ConcatAddPairs(zero)
	s1 := r1.ConcatAddPairs(zero)
	s2 := r2.ConcatAddPairs(zero)
	s3 := r3.ConcatAddPairs(zero)
	s4 := r4.ConcatAddPairs(zero)
	s5 := r5.ConcatAddPairs(zero)
	s6 := r6.ConcatAddPairs(zero)
	s7 := r7.ConcatAddPairs(zero)
	p1lo := s0.
		Add(placeLo(s1, zero, 1)).Add(placeLo(s2, zero, 2)).
		Add(placeLo(s3, zero, 3)).Add(placeLo(s4, zero, 4)).
		Add(placeLo(s5, zero, 5)).Add(placeLo(s6, zero, 6)).Add(placeLo(s7, zero, 7))
	// high half: only s4..s7 reach idx>=8 (s_i occupies i..i+3).
	p1hi := placeHi(s4, zero, 4).
		Add(placeHi(s5, zero, 5)).Add(placeHi(s6, zero, 6)).Add(placeHi(s7, zero, 7))
	c1 := cdefOddCostVec(p1lo, p1hi)

	// --- odd family p3: rev4(pairwise-sum(row i)) at lane i (offsets 0..7) ---
	// Fuse rev4 + placement into a single VTBL per (row, half) so the reversed
	// temporaries never materialize (cuts peak vector liveness and one op/row).
	p3lo := revPlaceLo(s0, 0).
		Add(revPlaceLo(s1, 1)).Add(revPlaceLo(s2, 2)).
		Add(revPlaceLo(s3, 3)).Add(revPlaceLo(s4, 4)).
		Add(revPlaceLo(s5, 5)).Add(revPlaceLo(s6, 6)).Add(revPlaceLo(s7, 7))
	p3hi := revPlaceHi(s5, 5).
		Add(revPlaceHi(s6, 6)).Add(revPlaceHi(s7, 7))
	c3 := cdefOddCostVec(p3lo, p3hi)

	// Argmax with the scalar `>` scan (lowest index wins ties), tracking the
	// opposite cost alongside so the variance needs no second lookup — matches
	// finishDirection exactly. Kept in registers (no cost[] stack array).
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
