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
func quantizeBRowGroupsDiffSIMD(qcoeff []int16, coeff []int32, n int, live uint8, q Quantizer, txScale uint8, iscan []int16) (diff, rate int64, eob int, handled bool) {
	if (n != 8 && n != 16 && n != 32) || txScale > 2 ||
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
	quantV := archsimd.BroadcastInt32x4(acQuant)
	roundV := archsimd.BroadcastInt32x4(roundPowerOfTwo((48*q.AC)>>7, txScale))
	zbinU := archsimd.BroadcastInt32x4(roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.AC, 7), txScale)).ToBits()
	maxV := archsimd.BroadcastInt32x4(maxInt16)
	maxU := maxV.ToBits()
	sixteenV := archsimd.BroadcastInt32x4(-16)
	shiftV := archsimd.BroadcastInt32x4(-int32(l - int64(txScale)))
	signShiftV := archsimd.BroadcastInt32x4(-31)
	stepV := archsimd.BroadcastInt32x4(q.AC)
	// The dequantization shift and the rate's exponent extraction are
	// hoisted as shift-count vectors: ShiftAllRight rebuilds its count
	// (CSEL, NEG, INS, DUP) at every use.
	txShiftV := archsimd.BroadcastInt32x4(-int32(txScale))
	expShiftV := archsimd.BroadcastInt32x4(-23)
	rateBiasV := archsimd.BroadcastUint32x4(124)
	zero := archsimd.BroadcastInt32x4(0)
	diffAcc := archsimd.BroadcastInt64x2(0)
	rateAcc := zero
	// eobAcc keeps, per lane, the largest iscan+1 of a nonzero level
	// (quantize_b's eob, as svt_aom_quantize_b_neon tracks it).
	var eobAcc archsimd.Int16x8
	one16 := archsimd.BroadcastInt16x8(1)
	zero16 := archsimd.BroadcastInt16x8(0)
	// One linear walk over the contiguous column-major group pairs (see
	// quantizeBRowGroupsSIMD); the off <= len-8 form proves every access.
	for off, g := 0, 0; off <= len(cs)-8 && off <= len(qs)-8 && off <= len(ss)-8; off, g = off+8, g+2 {
		if 4*g >= n {
			g = 0
		}
		pair := (live >> g) & 3
		dst := (*[8]int16)(qs[off:])
		c := (*[8]int32)(cs[off:])
		if pair == 0 {
			zero.TruncToInt16().StoreArray(dst)
			continue
		}
		c0, c1 := zero, zero
		lo, hi := zero, zero
		if pair&1 != 0 {
			c0 = archsimd.LoadInt32x4Array((*[4]int32)(c[0:4]))
			lo = quantizeBLane4SIMD(c0, quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
		}
		if pair&2 != 0 {
			c1 = archsimd.LoadInt32x4Array((*[4]int32)(c[4:8]))
			hi = quantizeBLane4SIMD(c1, quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
		}
		levels := quantizePackInt16x8SIMD(lo, hi)
		levels.StoreArray(dst)
		u := levels.ToBits().ReshapeToUint64s()
		if u.GetElem(0)|u.GetElem(1) == 0 {
			continue
		}
		pos1 := archsimd.LoadInt16x8Array((*[8]int16)(ss[off:])).Add(one16)
		eobAcc = eobAcc.Max(pos1.Masked(levels.NotEqual(zero16)))
		dq0 := lo.Mul(stepV).Shift(txShiftV)
		dq1 := hi.Mul(stepV).Shift(txShiftV)
		d0 := c0.Add(c0).Sub(dq0)
		d1 := c1.Add(c1).Sub(dq1)
		diffAcc = diffAcc.Add(dq0.MulWidenLo(d0)).Add(dq0.HiToLo().MulWidenLo(d0.HiToLo())).
			Add(dq1.MulWidenLo(d1)).Add(dq1.HiToLo().MulWidenLo(d1.HiToLo()))
		rateAcc = rateAcc.Add(rowGroupLaneRateSIMD(lo, expShiftV, rateBiasV)).
			Add(rowGroupLaneRateSIMD(hi, expShiftV, rateBiasV))
	}
	eob = int(eobAcc.ReduceMax())
	diff = diffAcc.GetElem(0) + diffAcc.GetElem(1)
	for lane := range uint8(4) {
		rate += int64(rateAcc.GetElem(lane))
	}
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

// rowGroupLaneRateSIMD is rowGroupLevelRate per lane: 2 + bits.Len32(|v|)
// for a nonzero level, 0 for zero. Quantized magnitudes are at most 32767,
// exactly representable in float32, whose biased exponent e is
// bits.Len32 + 126 (and 0 for zero), so the rate is e - 124 saturated at
// zero.
func rowGroupLaneRateSIMD(v, expShift archsimd.Int32x4, bias archsimd.Uint32x4) archsimd.Int32x4 {
	return v.Abs().ConvertToFloat32().ToBits().Shift(expShift).SubSaturated(bias).BitsToInt32()
}
