//go:build goexperiment.simd && amd64 && !purego

package quantize

import (
	"math/bits"
	"simd/archsimd"
	"unsafe"
)

func init() {
	if archsimd.X86.AVX2() {
		quantizeBRowGroupsImpl = quantizeBRowGroupsAVX2
		quantizeBRowGroupsDiffImpl = quantizeBRowGroupsDiffAVX2
	}
}

// rowGroupPairMask[pair] keeps the live four-row halves of an eight-row pair:
// bit 0 the low four lanes, bit 1 the high four.
var rowGroupPairMask = [4][8]int32{
	{},
	{-1, -1, -1, -1, 0, 0, 0, 0},
	{0, 0, 0, 0, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1},
}

// rowGroupQuantAVX2 holds the broadcast AC constants of the quantizeBVectors
// lane rule.
type rowGroupQuantAVX2 struct {
	quant, round, maxV archsimd.Int32x8
	zbin, maxU         archsimd.Uint32x8
	shift              archsimd.Uint32x8
}

// init fills k in place. Returning the struct by value would copy its 192
// bytes out of the result slot with legacy-SSE movups pairs right after the
// 256-bit broadcasts, which stalls on Zen 3 while the upper ymm state is
// dirty.
func (k *rowGroupQuantAVX2) init(q Quantizer, txScale uint8, l int64, zbinFactor int32) {
	acQuant, _ := invertQuant(q.AC)
	maxV := archsimd.BroadcastInt32x8(maxInt16)
	k.quant = archsimd.BroadcastInt32x8(acQuant)
	k.round = archsimd.BroadcastInt32x8(roundPowerOfTwo((48*q.AC)>>7, txScale))
	k.maxV = maxV
	k.zbin = archsimd.BroadcastInt32x8(roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.AC, 7), txScale)).ToBits()
	k.maxU = maxV.ToBits()
	k.shift = archsimd.BroadcastUint32x8(uint32(l - int64(txScale)))
}

// lane8 is quantizeBVectors' lane rule on eight coefficients. The level shift
// is a per-lane vector count (VPSRAVD): a variable scalar count would be moved
// into an XMM register at every use.
func (k *rowGroupQuantAVX2) lane8(c archsimd.Int32x8) archsimd.Int32x8 {
	absU := c.Abs().ToBits()
	keep := absU.GreaterEqual(k.zbin)
	tmp := absU.Min(k.maxU).BitsToInt32().Add(k.round).Min(k.maxV)
	t := tmp.Mul(k.quant).ShiftAllRight(16)
	level := t.Add(tmp).ShiftRight(k.shift).Min(k.maxV)
	sign := c.ShiftAllRight(31)
	return level.Xor(sign).Sub(sign).Masked(keep)
}

// quantizeBRowGroupsAVX2 is quantizeBRowGroupsSIMD (the NEON kernel) on AVX2:
// each column is walked as pairs of four-row groups, one eight-lane vector a
// pair, with a dead half's levels masked to zero and a dead pair stored as
// zeros. DC is redone with the scalar rule and the DC constants.
func quantizeBRowGroupsAVX2(qcoeff []int16, coeff []int32, n int, live uint8, q Quantizer, txScale uint8) bool {
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
	var k rowGroupQuantAVX2
	k.init(q, txScale, l, zbinFactor)
	cp := unsafe.Pointer(&coeff[0])
	qp := unsafe.Pointer(&qcoeff[0])
	var orAcc archsimd.Int32x8
	var zero16 archsimd.Int16x8
	for col := 0; col < n; col++ {
		for g := 0; g < n/4; g += 2 {
			off := col*n + 4*g
			pair := (live >> g) & 3
			dst := (*[8]int16)(unsafe.Add(qp, 2*off))
			if pair == 0 {
				zero16.StoreArray(dst)
				continue
			}
			level := k.lane8(archsimd.LoadInt32x8Array((*[8]int32)(unsafe.Add(cp, 4*off))))
			if pair != 3 {
				level = level.And(archsimd.LoadInt32x8Array(&rowGroupPairMask[pair]))
			}
			orAcc = orAcc.Or(level)
			level.GetLo().SaturateToInt16Concat(level.GetHi()).StoreArray(dst)
		}
	}
	nonZero := !orAcc.IsZero()
	archsimd.ClearAVXUpperBits()
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

// quantizeBRowGroupsDiffAVX2 is quantizeBRowGroupsDiffSIMD (the NEON kernel)
// on AVX2: the quantizer's loaded coefficients and levels feed the
// skip-minus-code distortion and the coarse rate directly. The products
// dq*(2c-dq) keep the NEON kernel's int32 operands and widen to int64 with
// VPMULDQ on the even lanes and on the odd lanes shifted down.
func quantizeBRowGroupsDiffAVX2(qcoeff []int16, coeff []int32, n int, live uint8, q Quantizer, txScale uint8) (diff, rate int64, handled bool) {
	if (n != 8 && n != 16 && n != 32) || txScale > 2 ||
		q.AC <= 0 || q.DC <= 0 || q.AC >= 1<<16 || q.DC >= 1<<16 {
		return 0, 0, false
	}
	count := n * n
	if len(qcoeff) < count || len(coeff) < count {
		return 0, 0, false
	}
	cp := unsafe.Pointer(&coeff[0])
	qp := unsafe.Pointer(&qcoeff[0])
	if uintptr(qp) < uintptr(cp)+uintptr(4*count) && uintptr(cp) < uintptr(qp)+uintptr(2*count) {
		return 0, 0, false
	}
	l := 31 - int64(bits.LeadingZeros32(uint32(q.AC)))
	if l < int64(txScale) {
		return 0, 0, false
	}
	zbinFactor := int32(80)
	if q.DC < 148 {
		zbinFactor = 84
	}
	var k rowGroupQuantAVX2
	k.init(q, txScale, l, zbinFactor)
	stepV := archsimd.BroadcastInt32x8(q.AC)
	txShiftV := archsimd.BroadcastUint32x8(uint32(txScale))
	rateBiasV := archsimd.BroadcastInt32x8(124)
	var zero32 archsimd.Int32x8
	var zero16 archsimd.Int16x8
	var diffAcc archsimd.Int64x4
	var rateAcc archsimd.Int32x8
	for col := 0; col < n; col++ {
		for g := 0; g < n/4; g += 2 {
			off := col*n + 4*g
			pair := (live >> g) & 3
			dst := (*[8]int16)(unsafe.Add(qp, 2*off))
			if pair == 0 {
				zero16.StoreArray(dst)
				continue
			}
			c := archsimd.LoadInt32x8Array((*[8]int32)(unsafe.Add(cp, 4*off)))
			level := k.lane8(c)
			if pair != 3 {
				level = level.And(archsimd.LoadInt32x8Array(&rowGroupPairMask[pair]))
			}
			level.GetLo().SaturateToInt16Concat(level.GetHi()).StoreArray(dst)
			if level.IsZero() {
				continue
			}
			// A dead half has zero levels, so its dq is zero and its terms
			// vanish whatever its coefficients.
			dq := level.Mul(stepV).ShiftRight(txShiftV)
			d := c.Add(c).Sub(dq)
			oddDQ := dq.ToBits().ReshapeToUint64s().ShiftAllRight(32).ReshapeToUint32s().BitsToInt32()
			oddD := d.ToBits().ReshapeToUint64s().ShiftAllRight(32).ReshapeToUint32s().BitsToInt32()
			diffAcc = diffAcc.Add(dq.MulWidenEven(d)).Add(oddDQ.MulWidenEven(oddD))
			// 2 + bits.Len32(|v|) for a nonzero level: the biased float32
			// exponent of |v| (exact below 2^24) minus 124, floored at zero.
			e := level.Abs().ConvertToFloat32().ToBits().ShiftAllRight(23).BitsToInt32()
			rateAcc = rateAcc.Add(e.Sub(rateBiasV).Max(zero32))
		}
	}
	dl := diffAcc.GetLo().Add(diffAcc.GetHi())
	diff = dl.GetElem(0) + dl.GetElem(1)
	rl := rateAcc.GetLo().Add(rateAcc.GetHi())
	rate = int64(rl.GetElem(0)) + int64(rl.GetElem(1)) + int64(rl.GetElem(2)) + int64(rl.GetElem(3))
	archsimd.ClearAVXUpperBits()
	if live&1 != 0 {
		// Retain the original last write of DC, then replace its AC-rule
		// contribution (quantizeBRowGroupsDiffSIMD's correction).
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
		oldDQ := (int32(acDC) * q.AC) >> txScale
		newDQ := (int32(dcLevel) * q.AC) >> txScale
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
	return diff, rate << 9, true
}
