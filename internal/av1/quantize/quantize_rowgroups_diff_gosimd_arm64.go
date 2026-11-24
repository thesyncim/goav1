//go:build goexperiment.simd && arm64 && !purego

package quantize

import (
	"math/bits"
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

func init() {
	if cpu.Detected.NEON {
		quantizeBRowGroupsDiffImpl = quantizeBRowGroupsDiffSIMD
	}
}

// quantizeBRowGroupsDiffSIMD retains the quantizer's loaded coefficients and
// levels for the following statistics, instead of storing then reloading them.
// Both arithmetic rules and the AC-first/DC-overwrite order are unchanged.
//
// It walks the block as svt_aom_quantize_b_neon does, eight coefficients at a
// time with the zero-bin test first: a group whose loaded coefficients all
// lie below the AC zero bin stores zero levels and skips the quantizer and
// the statistics, which is exact because a zero level adds nothing to the
// difference, the rate or the end of block. The live four-row groups are
// walked one pair at a time across the live columns, so the pair's liveness
// is decided once per pair rather than per group.
func quantizeBRowGroupsDiffSIMD(qcoeff []int16, coeff []int32, n int, live uint8, cols int, q Quantizer, txScale uint8, iscan []int16) (diff, rate int64, eob int, handled bool) {
	if (n != 8 && n != 16 && n != 32) || txScale > 2 || cols < 1 || cols > n ||
		q.AC <= 0 || q.DC <= 0 || q.AC >= 1<<16 || q.DC >= 1<<16 {
		return 0, 0, 0, false
	}
	count := n * n
	if len(qcoeff) < count || len(coeff) < count {
		return 0, 0, 0, false
	}
	cs, qs, ss := coeff[:count], qcoeff[:count], iscan[:count]
	// unsafe: address-only overlap test of the int32 source and int16
	// destination (different element types, so no safe comparison exists);
	// nothing is dereferenced through these pointers.
	if cp, qp := uintptr(unsafe.Pointer(&cs[0])), uintptr(unsafe.Pointer(&qs[0])); qp < cp+uintptr(4*count) && cp < qp+uintptr(2*count) {
		return 0, 0, 0, false
	}
	l := 31 - int64(bits.LeadingZeros32(uint32(q.AC)))
	if l < int64(txScale) {
		return 0, 0, 0, false
	}
	zbinFactor := int32(80)
	if q.DC < 148 {
		zbinFactor = 84
	}
	acQuant, _ := invertQuant(q.AC)
	zbin := roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.AC, 7), txScale)
	quantV := archsimd.BroadcastInt32x4(acQuant)
	roundV := archsimd.BroadcastInt32x4(roundPowerOfTwo((48*q.AC)>>7, txScale))
	zbinU := archsimd.BroadcastInt32x4(zbin).ToBits()
	maxV := archsimd.BroadcastInt32x4(maxInt16)
	maxU := maxV.ToBits()
	sixteenV := archsimd.BroadcastInt32x4(-16)
	shiftV := archsimd.BroadcastInt32x4(-int32(l - int64(txScale)))
	signShiftV := archsimd.BroadcastInt32x4(-31)
	st := rowGroupDiffConsts{
		step: archsimd.BroadcastInt32x4(q.AC),
		// The dequantization shift and the rate's exponent extraction are
		// hoisted as shift-count vectors: ShiftAllRight rebuilds its count
		// (CSEL, NEG, INS, DUP) at every use.
		txShift:  archsimd.BroadcastInt32x4(-int32(txScale)),
		expShift: archsimd.BroadcastInt32x4(-23),
		rateBias: archsimd.BroadcastUint32x4(124),
		one16:    archsimd.BroadcastInt16x8(1),
	}
	// The zero-bin test compares the group's largest magnitude, as unsigned
	// so that INT32_MIN (|c| = 2^31) passes as the scalar rule keeps it.
	zbinTest := uint32(zbin)
	var acc rowGroupDiffAcc
	var zero archsimd.Int32x4
	var zero16 archsimd.Int16x8
	// Column col's pair p (rows 8p..8p+7) is cs[col*n+8p:][:8]: each pair is
	// walked as a stride-n column loop over its own sub-slice, whose
	// off <= len-8 bound proves every access. The live columns end at
	// lim; the dead columns past it are one contiguous run, zeroed below.
	lim := cols * n
	for p := 0; p < n/8 && 8*p <= len(cs) && 8*p <= len(qs) && 8*p <= len(ss); p++ {
		pc, pq, ps := cs[8*p:], qs[8*p:], ss[8*p:]
		switch (live >> (uint(2*p) & 7)) & 3 {
		case 0:
			for off := 0; off < lim && off <= len(pq)-8; off += n {
				zero16.StoreArray((*[8]int16)(pq[off:]))
			}
		case 1:
			for off := 0; off < lim && off <= len(pq)-8 && off <= len(pc)-8 && off <= len(ps)-8; off += n {
				dst := (*[8]int16)(pq[off:])
				c0 := archsimd.LoadInt32x4Array((*[4]int32)(pc[off : off+4]))
				if c0.Abs().ToBits().ReduceMax() < zbinTest {
					zero16.StoreArray(dst)
					continue
				}
				lo := quantizeBLane4SIMD(c0, quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
				levels := quantizePackInt16x8SIMD(lo, zero)
				levels.StoreArray(dst)
				acc = acc.add(c0, zero, lo, zero, levels, archsimd.LoadInt16x8Array((*[8]int16)(ps[off:])), st)
			}
		case 2:
			for off := 0; off < lim && off <= len(pq)-8 && off <= len(pc)-8 && off <= len(ps)-8; off += n {
				dst := (*[8]int16)(pq[off:])
				c1 := archsimd.LoadInt32x4Array((*[4]int32)(pc[off+4 : off+8]))
				if c1.Abs().ToBits().ReduceMax() < zbinTest {
					zero16.StoreArray(dst)
					continue
				}
				hi := quantizeBLane4SIMD(c1, quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
				levels := quantizePackInt16x8SIMD(zero, hi)
				levels.StoreArray(dst)
				acc = acc.add(zero, c1, zero, hi, levels, archsimd.LoadInt16x8Array((*[8]int16)(ps[off:])), st)
			}
		default:
			for off := 0; off < lim && off <= len(pq)-8 && off <= len(pc)-8 && off <= len(ps)-8; off += n {
				dst := (*[8]int16)(pq[off:])
				c := (*[8]int32)(pc[off:])
				c0 := archsimd.LoadInt32x4Array((*[4]int32)(c[0:4]))
				c1 := archsimd.LoadInt32x4Array((*[4]int32)(c[4:8]))
				if c0.Abs().ToBits().Max(c1.Abs().ToBits()).ReduceMax() < zbinTest {
					zero16.StoreArray(dst)
					continue
				}
				lo := quantizeBLane4SIMD(c0, quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
				hi := quantizeBLane4SIMD(c1, quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
				levels := quantizePackInt16x8SIMD(lo, hi)
				levels.StoreArray(dst)
				acc = acc.add(c0, c1, lo, hi, levels, archsimd.LoadInt16x8Array((*[8]int16)(ps[off:])), st)
			}
		}
	}
	for off := lim; off <= len(qs)-8; off += 8 {
		zero16.StoreArray((*[8]int16)(qs[off:]))
	}
	eob = int(acc.eob.ReduceMax())
	diff = acc.diff.GetElem(0) + acc.diff.GetElem(1)
	rate = int64(acc.rate.ReduceSum())
	if live&1 != 0 {
		// Retain the original last write of DC, then replace its AC-rule
		// contribution. Rate also resolves the AC-only nonzero-DC corner
		// without the old quantizer's possible full-block nonzero rescan.
		dcQuant, dcShift := invertQuant(q.DC)
		dc := quantBParams{
			zbin:  roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.DC, 7), txScale),
			round: roundPowerOfTwo((48*q.DC)>>7, txScale),
			quant: dcQuant,
			shift: dcShift,
		}
		acDC := qcoeff[0]
		dcLevel := quantizeScalarB(coeff[0], &dc, txScale)
		qcoeff[0] = dcLevel
		eob = rowGroupDCEOB(eob, dcLevel)
		oldDQ := (int32(acDC) * q.AC) >> txScale
		newDQ := (int32(dcLevel) * q.AC) >> txScale
		// Match the old SIMD pass's int32 intermediate before widening.
		diff -= int64(oldDQ) * int64(2*coeff[0]-oldDQ)
		diff += int64(newDQ) * int64(2*coeff[0]-newDQ)
		rate += rowGroupLevelRate(dcLevel) - rowGroupLevelRate(acDC)
		if dcLevel != 0 {
			c := int64(coeff[0])
			eAC := c - ((int64(dcLevel) * int64(q.AC)) >> txScale)
			eDC := c - ((int64(dcLevel) * int64(q.DC)) >> txScale)
			diff += eAC*eAC - eDC*eDC
		}
	}
	return diff, rate << 9, eob, true
}

// rowGroupDiffConsts holds the broadcast constants of the statistics.
type rowGroupDiffConsts struct {
	step, txShift, expShift archsimd.Int32x4
	rateBias                archsimd.Uint32x4
	one16                   archsimd.Int16x8
}

// rowGroupDiffAcc accumulates the fused kernel's statistics: the per-lane
// skip-minus-code difference (int64), coarse rate and largest iscan+1 of a
// nonzero level (quantize_b's eob, as svt_aom_quantize_b_neon tracks it).
type rowGroupDiffAcc struct {
	diff archsimd.Int64x2
	rate archsimd.Int32x4
	eob  archsimd.Int16x8
}

// add folds one group's coefficients c0/c1, their levels lo/hi (packed as
// levels) and the group's iscan into the statistics.
func (a rowGroupDiffAcc) add(c0, c1, lo, hi archsimd.Int32x4, levels, iscan archsimd.Int16x8, k rowGroupDiffConsts) rowGroupDiffAcc {
	var zero16 archsimd.Int16x8
	a.eob = a.eob.Max(iscan.Add(k.one16).Masked(levels.NotEqual(zero16)))
	dq0 := lo.Mul(k.step).Shift(k.txShift)
	dq1 := hi.Mul(k.step).Shift(k.txShift)
	d0 := c0.Add(c0).Sub(dq0)
	d1 := c1.Add(c1).Sub(dq1)
	a.diff = a.diff.Add(dq0.MulWidenLo(d0)).Add(dq0.HiToLo().MulWidenLo(d0.HiToLo())).
		Add(dq1.MulWidenLo(d1)).Add(dq1.HiToLo().MulWidenLo(d1.HiToLo()))
	a.rate = a.rate.Add(rowGroupLaneRateSIMD(lo, k.expShift, k.rateBias)).
		Add(rowGroupLaneRateSIMD(hi, k.expShift, k.rateBias))
	return a
}

// rowGroupLaneRateSIMD is rowGroupLevelRate per lane: 2 + bits.Len32(|v|)
// for a nonzero level, 0 for zero. Quantized magnitudes are at most 32767,
// exactly representable in float32, whose biased exponent e is
// bits.Len32 + 126 (and 0 for zero), so the rate is e - 124 saturated at
// zero.
func rowGroupLaneRateSIMD(v, expShift archsimd.Int32x4, bias archsimd.Uint32x4) archsimd.Int32x4 {
	return v.Abs().ConvertToFloat32().ToBits().Shift(expShift).SubSaturated(bias).BitsToInt32()
}
