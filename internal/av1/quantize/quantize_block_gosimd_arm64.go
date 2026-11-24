//go:build goexperiment.simd && arm64 && !purego

package quantize

import "simd/archsimd"

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
