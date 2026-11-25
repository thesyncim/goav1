// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package quantize

import (
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"math/bits"
	"simd/archsimd"
)

// dequantColumnSIMD is the Go-native SIMD twin of dequantColumnPureGo on NEON.
// It processes eight coefficients per iteration as two Int32x4 halves and
// handles the n%8 tail with the scalar reference.
//
// Bit-exactness with dequantScalar:
//   - the int16 coefficients are sign-extended to int32 lanes, so |c| (and
//     INT16_MIN) is exact in int32.
//   - the product |c|*scale is a 32-bit multiply; only its low 24 bits survive
//     the mask, and the low 24 bits of an int32 product equal those of the
//     int64 product dequantScalar forms.
//   - the masked product is non-negative, so the arithmetic right shift by
//     txScale (SSHL with a negative count, held in a hoisted vector) equals the
//     scalar shift.
//   - the sign is reapplied as multiplication by clamp(c, -1, 1): -1 for
//     negative c, +1 for positive c, and 0 for c == 0 where the magnitude is
//     already 0. The result is then clamped to [dqMin, dqMax] with Min then
//     Max, as the scalar clamp does.
func dequantColumnSIMD(dst []int32, coeff []int16, scale int32, txScale uint8, dqMin int32, dqMax int32) {
	n := len(coeff)
	if n == 0 {
		return
	}
	_ = dst[n-1]
	blocks8 := n &^ 7
	if blocks8 > 0 {
		scaleV := archsimd.BroadcastInt32x4(scale)
		maskV := archsimd.BroadcastInt32x4(0x00ffffff)
		oneV := archsimd.BroadcastInt32x4(1)
		negOneV := archsimd.BroadcastInt32x4(-1)
		// Negative counts make SSHL shift right; hoisted out of the loop.
		shiftV := archsimd.BroadcastInt32x4(-int32(txScale))
		loV := archsimd.BroadcastInt32x4(dqMin)
		hiV := archsimd.BroadcastInt32x4(dqMax)
		cs, ds := coeff[:blocks8], dst[:blocks8]
		for i := 0; i <= len(cs)-8 && i <= len(ds)-8; i += 8 {
			c := archsimd.LoadInt16x8Array((*[8]int16)(cs[i:]))
			lo := dequantLane4SIMD(c.ExtendLo4ToInt32(), scaleV, maskV, oneV, negOneV, shiftV, loV, hiV)
			hi := dequantLane4SIMD(c.HiToLo().ExtendLo4ToInt32(), scaleV, maskV, oneV, negOneV, shiftV, loV, hiV)
			d := (*[8]int32)(ds[i:])
			lo.StoreArray((*[4]int32)(d[0:4]))
			hi.StoreArray((*[4]int32)(d[4:8]))
		}
	}
	if blocks8 < n {
		dequantColumnPureGo(dst[blocks8:], coeff[blocks8:], scale, txScale, dqMin, dqMax)
	}
}

// dequantLane4SIMD dequantizes four sign-extended coefficients (see
// dequantColumnSIMD for the exactness argument).
func dequantLane4SIMD(x, scale, mask, one, negOne, shift, lo, hi archsimd.Int32x4) archsimd.Int32x4 {
	sign := x.Min(one).Max(negOne)
	p := x.Abs().Mul(scale).And(mask).Shift(shift)
	return p.Mul(sign).Min(hi).Max(lo)
}

func init() {
	if cpu.Detected.NEON {
		dequantColumnImpl = dequantColumnSIMD
	}
}

// quantizeFPVectors applies the av1_quantize_fp lane rule to coeff, writing
// qcoeff, eight coefficients per iteration as two Int32x4 halves. len(coeff)
// must be a multiple of 8 and len(qcoeff) >= len(coeff).
//
// Per lane (see quantizeScalarFP):
//   - keep  = (|c| << (1+ts)) >= dequant, with the same int32 wrap as the
//     scalar shift.
//   - a     = min(|c| + round, 32767). The add can wrap only for lanes that
//     keep drops (see quantizeVectorMaxStep), and those are zeroed by the mask.
//   - level = min((a * quant) >> (16-ts), 32767), negated when c < 0, and
//     zeroed when not kept.
func quantizeFPVectors(qcoeff []int16, coeff []int32, quant int32, round int32, dequant int32, txScale uint8) {
	n := len(coeff)
	if n == 0 {
		return
	}
	q := qcoeff[:n]
	quantV := archsimd.BroadcastInt32x4(quant)
	roundV := archsimd.BroadcastInt32x4(round)
	deqV := archsimd.BroadcastInt32x4(dequant)
	maxV := archsimd.BroadcastInt32x4(maxInt16)
	// Shift counts are loop-invariant vectors: a positive count shifts left and
	// a negative count shifts right (SSHL), and the sign mask shifts by 31.
	shlV := archsimd.BroadcastInt32x4(int32(1 + txScale))
	shrV := archsimd.BroadcastInt32x4(-int32(16 - int(txScale)))
	signShiftV := archsimd.BroadcastInt32x4(-31)
	// The i <= len-8 form proves every 8-lane window in bounds, so the loop
	// carries no bounds checks.
	for i := 0; i <= len(coeff)-8 && i <= len(q)-8; i += 8 {
		c := (*[8]int32)(coeff[i:])
		lo := quantizeFPLane4SIMD(archsimd.LoadInt32x4Array((*[4]int32)(c[0:4])), quantV, roundV, deqV, maxV, shlV, shrV, signShiftV)
		hi := quantizeFPLane4SIMD(archsimd.LoadInt32x4Array((*[4]int32)(c[4:8])), quantV, roundV, deqV, maxV, shlV, shrV, signShiftV)
		quantizePackInt16x8SIMD(lo, hi).StoreArray((*[8]int16)(q[i:]))
	}
}

// quantizeFPLane4SIMD applies the fp rule to four coefficients (see
// quantizeFPVectors). The sign is reapplied as (level ^ mask) - mask with
// mask = c >> 31.
func quantizeFPLane4SIMD(c, quant, round, dequant, maxV, shl, shr, signShift archsimd.Int32x4) archsimd.Int32x4 {
	abs := c.Abs()
	keep := abs.Shift(shl).GreaterEqual(dequant)
	a := abs.Add(round).Min(maxV)
	level := a.Mul(quant).Shift(shr).Min(maxV)
	mask := c.Shift(signShift)
	return level.Xor(mask).Sub(mask).Masked(keep)
}

// quantizeBVectors applies the aom_quantize_b lane rule to coeff, writing
// qcoeff, eight coefficients per iteration as two Int32x4 halves. len(coeff)
// must be a multiple of 8 and len(qcoeff) >= len(coeff). shift is l-txScale,
// the right-shift count that folds invert_quant's power-of-two multiply.
//
// Per lane (see quantizeScalarB):
//   - keep  = |c| >= zbin, compared unsigned so that INT32_MIN (|c| = 2^31 in
//     the scalar's int64 magnitude) is kept as the scalar keeps it.
//   - tmp   = min(|c| + round, 32767).
//   - level = min(((tmp*quant)>>16 + tmp) >> shift, 32767), negated when c < 0,
//     and zeroed when not kept.
//
// With a non-nil iscan (len(iscan) >= len(coeff)) it returns the levels' end
// of block, one past the largest iscan of a nonzero level, kept per lane as
// svt_aom_quantize_b_neon keeps it; with a nil iscan it returns 0.
func quantizeBVectors(qcoeff []int16, coeff []int32, quant int32, round int32, zbin int32, shift uint64, iscan []int16) int {
	n := len(coeff)
	if n == 0 {
		return 0
	}
	q := qcoeff[:n]
	quantV := archsimd.BroadcastInt32x4(quant)
	roundV := archsimd.BroadcastInt32x4(round)
	zbinU := archsimd.BroadcastInt32x4(zbin).ToBits()
	maxV := archsimd.BroadcastInt32x4(maxInt16)
	maxU := maxV.ToBits()
	// Loop-invariant shift counts: a negative count shifts right (SSHL).
	sixteenV := archsimd.BroadcastInt32x4(-16)
	shiftV := archsimd.BroadcastInt32x4(-int32(shift))
	signShiftV := archsimd.BroadcastInt32x4(-31)
	// svt_aom_quantize_b_neon's zero-bin test: a group whose magnitudes all
	// lie below the zero bin (compared unsigned, so INT32_MIN passes as the
	// scalar rule keeps it) stores zero levels and skips the quantizer.
	zbinTest := uint32(zbin)
	var zero16 archsimd.Int16x8
	if iscan == nil {
		for i := 0; i <= len(coeff)-8 && i <= len(q)-8; i += 8 {
			c := (*[8]int32)(coeff[i:])
			dst := (*[8]int16)(q[i:])
			c0 := archsimd.LoadInt32x4Array((*[4]int32)(c[0:4]))
			c1 := archsimd.LoadInt32x4Array((*[4]int32)(c[4:8]))
			if c0.Abs().ToBits().Max(c1.Abs().ToBits()).ReduceMax() < zbinTest {
				zero16.StoreArray(dst)
				continue
			}
			lo := quantizeBLane4SIMD(c0, quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
			hi := quantizeBLane4SIMD(c1, quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
			quantizePackInt16x8SIMD(lo, hi).StoreArray(dst)
		}
		return 0
	}
	sc := iscan[:n]
	one16 := archsimd.BroadcastInt16x8(1)
	var eobAcc archsimd.Int16x8
	for i := 0; i <= len(coeff)-8 && i <= len(q)-8 && i <= len(sc)-8; i += 8 {
		c := (*[8]int32)(coeff[i:])
		dst := (*[8]int16)(q[i:])
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
		pos1 := archsimd.LoadInt16x8Array((*[8]int16)(sc[i:])).Add(one16)
		eobAcc = eobAcc.Max(pos1.Masked(levels.NotEqual(zero16)))
	}
	return int(eobAcc.ReduceMax())
}

// quantizeBLane4SIMD applies the zbin rule to four coefficients (see
// quantizeBVectors).
func quantizeBLane4SIMD(c, quant, round archsimd.Int32x4, zbinU, maxU archsimd.Uint32x4, maxV, sixteen, shift, signShift archsimd.Int32x4) archsimd.Int32x4 {
	absU := c.Abs().ToBits()
	keep := absU.GreaterEqual(zbinU)
	tmp := absU.Min(maxU).BitsToInt32().Add(round).Min(maxV)
	t := tmp.Mul(quant).Shift(sixteen)
	level := t.Add(tmp).Shift(shift).Min(maxV)
	mask := c.Shift(signShift)
	return level.Xor(mask).Sub(mask).Masked(keep)
}

// quantizeBBlockEOBTrustedSIMD is svt_aom_quantize_b_neon on count
// contiguous coefficients with bp's prebuilt constants: eight coefficients
// per group, the zero-bin test first, the DC constants in lane 0 of the
// first group and the end of block kept per lane. The coefficients at or
// past live are trusted to quantize to zero; they are stored as zero levels
// without being read.
func quantizeBBlockEOBTrustedSIMD(qcoeff []int16, coeff []int32, count, live int, bp *BParams, iscan []int16) (int, bool) {
	if !bp.vector || count < 8 || count%8 != 0 || live < 8 || live > count || live%8 != 0 {
		return 0, false
	}
	if len(qcoeff) < count || len(coeff) < count || len(iscan) < count {
		return 0, false
	}
	cs, qs, ss := coeff[:count], qcoeff[:count], iscan[:count]
	ln := &bp.lanes
	quantV := archsimd.LoadInt32x4Array(&ln.quant)
	roundV := archsimd.LoadInt32x4Array(&ln.round)
	zbinU := archsimd.LoadInt32x4Array(&ln.zbin).ToBits()
	maxV := archsimd.LoadInt32x4Array(&ln.max)
	maxU := maxV.ToBits()
	sixteenV := archsimd.LoadInt32x4Array(&ln.sixteen)
	shiftV := archsimd.LoadInt32x4Array(&ln.shift)
	signShiftV := archsimd.LoadInt32x4Array(&ln.signShift)
	zbinTest := bp.acZbin
	var zero16, eobAcc archsimd.Int16x8
	// The first group carries DC in lane 0 and tests against the smaller of
	// the two bins.
	{
		c := (*[8]int32)(cs[0:8])
		c0 := archsimd.LoadInt32x4Array((*[4]int32)(c[0:4]))
		c1 := archsimd.LoadInt32x4Array((*[4]int32)(c[4:8]))
		if c0.Abs().ToBits().Max(c1.Abs().ToBits()).ReduceMax() < bp.minZbin {
			zero16.StoreArray((*[8]int16)(qs[0:8]))
		} else {
			lo := quantizeBLane4SIMD(c0, archsimd.LoadInt32x4Array(&ln.dcQuant), archsimd.LoadInt32x4Array(&ln.dcRound),
				archsimd.LoadInt32x4Array(&ln.dcZbin).ToBits(), maxU, maxV, sixteenV, archsimd.LoadInt32x4Array(&ln.dcShift), signShiftV)
			hi := quantizeBLane4SIMD(c1, quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
			levels := quantizePackInt16x8SIMD(lo, hi)
			levels.StoreArray((*[8]int16)(qs[0:8]))
			eobAcc = rowGroupEOBSIMD(eobAcc, levels, archsimd.LoadInt16x8Array((*[8]int16)(ss[0:8])), ln)
		}
	}
	// The three live windows share one length, so one bound proves every
	// access.
	lc, lq, ls := cs[:live], qs[:live], ss[:live]
	for i := 8; i <= len(lc)-8; i += 8 {
		c := (*[8]int32)(lc[i:])
		dst := (*[8]int16)(lq[i:])
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
		eobAcc = rowGroupEOBSIMD(eobAcc, levels, archsimd.LoadInt16x8Array((*[8]int16)(ls[i:])), ln)
	}
	for i := live; i <= len(qs)-8; i += 8 {
		zero16.StoreArray((*[8]int16)(qs[i:]))
	}
	return int(eobAcc.ReduceMax()), true
}

// init binds Go-native SIMD quantizers under GOEXPERIMENT=simd.
func init() {
	if cpu.Detected.NEON {
		quantizeFPBlockImpl = quantizeFPBlockSIMD
		quantizeBBlockImpl = quantizeBBlockSIMD
		quantizeBBlockEOBImpl = quantizeBBlockEOBSIMD
		quantizeBBlockEOBTrustedImpl = quantizeBBlockEOBTrustedSIMD
	}
}

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

func init() {
	if cpu.Detected.NEON {
		quantizeBRowGroupsImpl = quantizeBRowGroupsSIMD
	}
}

// quantizeBRowGroupsSIMD is quantizeBRowGroupsScalar with the
// quantizeBVectors lane rule. Each column is walked as pairs of four-row
// groups: a fully live pair quantizes eight coefficients into one 16-byte
// store, a half-live pair four (the dead half stored as zero), a dead pair
// stores zeros. The DC coefficient is redone with the scalar rule and DC
// constants, as quantizeBBlockSIMD does.
func quantizeBRowGroupsSIMD(qcoeff []int16, coeff []int32, n int, live uint8, q Quantizer, txScale uint8) bool {
	if q.AC > quantizeVectorMaxStep || q.DC > quantizeVectorMaxStep || q.AC <= 0 || q.DC <= 0 || n&7 != 0 {
		return quantizeBRowGroupsScalar(qcoeff, coeff, n, live, q, txScale)
	}
	l := 31 - int64(bits.LeadingZeros32(uint32(q.AC)))
	if l < int64(txScale) {
		return quantizeBRowGroupsScalar(qcoeff, coeff, n, live, q, txScale)
	}
	_ = qcoeff[n*n-1]
	_ = coeff[n*n-1]
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
	zero := archsimd.BroadcastInt32x4(0)
	cs, qs := coeff[:n*n], qcoeff[:n*n]
	orAcc := zero
	// The column-major pairs of four-row groups tile cs contiguously (off =
	// col*n + 4*g steps by eight), so one linear walk with g wrapping at n/4
	// visits them in order and the off <= len-8 form proves every access.
	for off, g := 0, 0; off <= len(cs)-8 && off <= len(qs)-8; off, g = off+8, g+2 {
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
		lo, hi := zero, zero
		if pair&1 != 0 {
			lo = quantizeBLane4SIMD(archsimd.LoadInt32x4Array((*[4]int32)(c[0:4])),
				quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
		}
		if pair&2 != 0 {
			hi = quantizeBLane4SIMD(archsimd.LoadInt32x4Array((*[4]int32)(c[4:8])),
				quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
		}
		orAcc = orAcc.Or(lo).Or(hi)
		quantizePackInt16x8SIMD(lo, hi).StoreArray(dst)
	}
	nonZero := orAcc.ReduceMax() != 0 || orAcc.ReduceMin() != 0
	if live&1 != 0 {
		// The vector pass quantized DC with the AC rule; redo it with the DC
		// constants and rebuild the non-zero report around it.
		dcQuant, dcShift := invertQuant(q.DC)
		dc := quantBParams{
			zbin:  roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.DC, 7), txScale),
			round: roundPowerOfTwo((48*q.DC)>>7, txScale),
			quant: dcQuant,
			shift: dcShift,
		}
		acDC := qcoeff[0]
		qcoeff[0] = quantizeScalarB(coeff[0], &dc, txScale)
		if qcoeff[0] != 0 {
			return true
		}
		if acDC != 0 {
			// The AC-rule DC may have been the only non-zero lane.
			for _, v := range qcoeff[:n*n] {
				if v != 0 {
					return true
				}
			}
			return false
		}
	}
	return nonZero
}

func quantizePackInt16x8SIMD(lo, hi archsimd.Int32x4) archsimd.Int16x8 {
	lo16 := lo.TruncToInt16()
	hi16 := hi.TruncToInt16()
	return lo16.ToBits().ReshapeToUint64s().InterleaveLo(hi16.ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
}
