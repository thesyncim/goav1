//go:build goexperiment.simd && arm64 && !purego

package quantize

import (
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
//
// It walks the block as svt_aom_quantize_b_neon does, eight coefficients at a
// time with the zero-bin test first: a group whose loaded coefficients all
// lie below the AC zero bin stores zero levels and skips the quantizer and
// the statistics, which is exact because a zero level adds nothing to the
// difference, the rate or the end of block. The live four-row groups are
// walked one pair at a time across the live columns, so the pair's liveness
// is decided once per pair rather than per group.
//
// The DC coefficient is quantized in the first group with the DC constants
// in lane 0 (SVT's v_zbins0/v_round0/v_quant0 lanes), so its level, rate and
// end of block come out of the vector pass directly; that eob agrees with
// quantize_b's because AV1 scans put DC first (iscan[0] == 0). The
// statistics still dequantize lane 0 with the AC step, as the encoder's
// two-pass SIMD statistics did, and the scalar tail adds their DC
// correction.
func quantizeBRowGroupsDiffSIMD(qcoeff []int16, coeff []int32, n int, live uint8, cols int, bp *BParams, iscan []int16) (diff, rate int64, eob int, handled bool) {
	if !bp.vector || (n != 8 && n != 16 && n != 32) || cols < 1 || cols > n {
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
	ln := &bp.lanes
	quantV := archsimd.LoadInt32x4Array(&ln.quant)
	roundV := archsimd.LoadInt32x4Array(&ln.round)
	zbinU := archsimd.LoadInt32x4Array(&ln.zbin).ToBits()
	maxV := archsimd.LoadInt32x4Array(&ln.max)
	maxU := maxV.ToBits()
	sixteenV := archsimd.LoadInt32x4Array(&ln.sixteen)
	shiftV := archsimd.LoadInt32x4Array(&ln.shift)
	signShiftV := archsimd.LoadInt32x4Array(&ln.signShift)
	// The statistics' constants are loaded where a group is coded (about
	// one group in twenty), not held across the loop.
	st := ln
	// The zero-bin test compares the group's largest magnitude, as unsigned
	// so that INT32_MIN (|c| = 2^31) passes as the scalar rule keeps it.
	zbinTest := bp.acZbin
	var acc rowGroupDiffAcc
	var zero archsimd.Int32x4
	var zero16 archsimd.Int16x8
	// Every level starts zero (SVT's memset of qcoeff), so the dead pairs,
	// the dead columns and the groups below the zero bin store nothing.
	for off := 0; off <= len(qs)-16; off += 16 {
		zero16.StoreArray((*[8]int16)(qs[off:]))
		zero16.StoreArray((*[8]int16)(qs[off+8:]))
	}
	// start is the first column offset of pair 0's walk: column 0 of a live
	// first group is the DC group, coded here.
	start := 0
	var dcLevel int32
	if live&1 != 0 {
		start = n
		c := (*[8]int32)(cs[0:8])
		c0 := archsimd.LoadInt32x4Array((*[4]int32)(c[0:4]))
		var c1 archsimd.Int32x4
		if live&2 != 0 {
			c1 = archsimd.LoadInt32x4Array((*[4]int32)(c[4:8]))
		}
		// A dead second group is not loaded, and its zero lanes stay below
		// both bins (zbin >= 1 when log2(step) >= txScale).
		if c0.Abs().ToBits().Max(c1.Abs().ToBits()).ReduceMax() >= bp.minZbin {
			lo := quantizeBLane4SIMD(c0, archsimd.LoadInt32x4Array(&ln.dcQuant), archsimd.LoadInt32x4Array(&ln.dcRound),
				archsimd.LoadInt32x4Array(&ln.dcZbin).ToBits(), maxU, maxV, sixteenV, archsimd.LoadInt32x4Array(&ln.dcShift), signShiftV)
			hi := zero
			if live&2 != 0 {
				hi = quantizeBLane4SIMD(c1, quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
			}
			levels := quantizePackInt16x8SIMD(lo, hi)
			levels.StoreArray((*[8]int16)(qs[0:8]))
			acc.eob = rowGroupEOBSIMD(acc.eob, levels, archsimd.LoadInt16x8Array((*[8]int16)(ss[0:8])), st)
			acc.diff = rowGroupDiffSIMD(rowGroupDiffSIMD(acc.diff, c0, lo, st), c1, hi, st)
			acc.rate = rowGroupRateSIMD(rowGroupRateSIMD(acc.rate, lo, st), hi, st)
			dcLevel = lo.GetElem(0)
		}
	}
	// Column col's pair p (rows 8p..8p+7) is cs[col*n+8p:][:8]. Each pair is
	// walked as a stride-n column loop over its own sub-slice, cut to end at
	// the last live column's group, so the one off <= len-8 bound both ends
	// the walk and proves every access.
	walk := (cols-1)*n + 8
	for p := 0; p < n/8; p++ {
		pair := (live >> (uint(2*p) & 7)) & 3
		base := 8 * p
		if pair == 0 || base+walk > len(cs) || base+walk > len(qs) || base+walk > len(ss) {
			start = 0
			continue
		}
		pc, pq, ps := cs[base:base+walk], qs[base:base+walk], ss[base:base+walk]
		switch pair {
		case 1:
			for off := start; off <= len(pc)-8; off += n {
				c := (*[8]int32)(pc[off:])
				c0 := archsimd.LoadInt32x4Array((*[4]int32)(c[0:4]))
				if c0.Abs().ToBits().ReduceMax() < zbinTest {
					continue
				}
				lo := quantizeBLane4SIMD(c0, quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
				levels := quantizePackInt16x8SIMD(lo, zero)
				levels.StoreArray((*[8]int16)(pq[off:]))
				acc.eob = rowGroupEOBSIMD(acc.eob, levels, archsimd.LoadInt16x8Array((*[8]int16)(ps[off:])), st)
				acc.diff = rowGroupDiffSIMD(acc.diff, c0, lo, st)
				acc.rate = rowGroupRateSIMD(acc.rate, lo, st)
			}
		case 2:
			for off := 0; off <= len(pc)-8; off += n {
				c := (*[8]int32)(pc[off:])
				c1 := archsimd.LoadInt32x4Array((*[4]int32)(c[4:8]))
				if c1.Abs().ToBits().ReduceMax() < zbinTest {
					continue
				}
				hi := quantizeBLane4SIMD(c1, quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
				levels := quantizePackInt16x8SIMD(zero, hi)
				levels.StoreArray((*[8]int16)(pq[off:]))
				acc.eob = rowGroupEOBSIMD(acc.eob, levels, archsimd.LoadInt16x8Array((*[8]int16)(ps[off:])), st)
				acc.diff = rowGroupDiffSIMD(acc.diff, c1, hi, st)
				acc.rate = rowGroupRateSIMD(acc.rate, hi, st)
			}
		default:
			for off := start; off <= len(pc)-8; off += n {
				c := (*[8]int32)(pc[off:])
				c0 := archsimd.LoadInt32x4Array((*[4]int32)(c[0:4]))
				c1 := archsimd.LoadInt32x4Array((*[4]int32)(c[4:8]))
				if c0.Abs().ToBits().Max(c1.Abs().ToBits()).ReduceMax() < zbinTest {
					continue
				}
				lo := quantizeBLane4SIMD(c0, quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
				hi := quantizeBLane4SIMD(c1, quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
				levels := quantizePackInt16x8SIMD(lo, hi)
				levels.StoreArray((*[8]int16)(pq[off:]))
				acc.eob = rowGroupEOBSIMD(acc.eob, levels, archsimd.LoadInt16x8Array((*[8]int16)(ps[off:])), st)
				acc.diff = rowGroupDiffSIMD(rowGroupDiffSIMD(acc.diff, c0, lo, st), c1, hi, st)
				acc.rate = rowGroupRateSIMD(rowGroupRateSIMD(acc.rate, lo, st), hi, st)
			}
		}
		start = 0
	}
	eob = int(acc.eob.ReduceMax())
	diff = acc.diff.GetElem(0) + acc.diff.GetElem(1)
	rate = int64(acc.rate.ReduceSum())
	if dcLevel != 0 {
		// The statistics dequantized DC with the AC step; replace its error
		// with the DC step's.
		q, txScale := bp.q, bp.txScale
		c := int64(coeff[0])
		eAC := c - ((int64(dcLevel) * int64(q.AC)) >> txScale)
		eDC := c - ((int64(dcLevel) * int64(q.DC)) >> txScale)
		diff += eAC*eAC - eDC*eDC
	}
	return diff, rate << 9, eob, true
}

// rowGroupDiffAcc accumulates the fused kernel's statistics: the per-lane
// skip-minus-code difference (int64), coarse rate and largest iscan+1 of a
// nonzero level (quantize_b's eob, as svt_aom_quantize_b_neon tracks it).
type rowGroupDiffAcc struct {
	diff archsimd.Int64x2
	rate archsimd.Int32x4
	eob  archsimd.Int16x8
}

// rowGroupEOBSIMD keeps, per lane, the largest iscan+1 of a nonzero level.
func rowGroupEOBSIMD(eob, levels, iscan archsimd.Int16x8, k *bParamLanes) archsimd.Int16x8 {
	var zero16 archsimd.Int16x8
	return eob.Max(iscan.Add(archsimd.LoadInt16x8Array(&k.one16)).Masked(levels.NotEqual(zero16)))
}

// rowGroupDiffSIMD adds dq*(2c-dq) for four coefficients c and levels v,
// dq = (v*step) >> txScale, with the int32 intermediates of the encoder's
// two-pass statistics and int64 products.
func rowGroupDiffSIMD(diff archsimd.Int64x2, c, v archsimd.Int32x4, k *bParamLanes) archsimd.Int64x2 {
	dq := v.Mul(archsimd.LoadInt32x4Array(&k.step)).Shift(archsimd.LoadInt32x4Array(&k.txShift))
	d := c.Add(c).Sub(dq)
	return diff.Add(dq.MulWidenLo(d)).Add(dq.HiToLo().MulWidenLo(d.HiToLo()))
}

// rowGroupRateSIMD adds the coarse rate of four levels v.
func rowGroupRateSIMD(rate, v archsimd.Int32x4, k *bParamLanes) archsimd.Int32x4 {
	return rate.Add(rowGroupLaneRateSIMD(v, archsimd.LoadInt32x4Array(&k.expShift), archsimd.LoadUint32x4Array(&k.rateBias)))
}

// rowGroupLaneRateSIMD is rowGroupLevelRate per lane: 2 + bits.Len32(|v|)
// for a nonzero level, 0 for zero. Quantized magnitudes are at most 32767,
// exactly representable in float32, whose biased exponent e is
// bits.Len32 + 126 (and 0 for zero), so the rate is e - 124 saturated at
// zero.
func rowGroupLaneRateSIMD(v, expShift archsimd.Int32x4, bias archsimd.Uint32x4) archsimd.Int32x4 {
	return v.Abs().ConvertToFloat32().ToBits().Shift(expShift).SubSaturated(bias).BitsToInt32()
}
