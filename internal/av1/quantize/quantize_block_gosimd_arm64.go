//go:build goexperiment.simd && arm64 && !purego

package quantize

import (
	"simd/archsimd"
	"unsafe"
)

// quantizeFPVectors applies the av1_quantize_fp lane rule to coeff, writing
// qcoeff, eight coefficients per iteration as two Int32x4 halves. len(coeff)
// must be a multiple of 8 and len(qcoeff) >= len(coeff).
//
// Per lane (see quantizeScalarFP):
//   - keep  = (|c| << (1+ts)) >= dequant, with the same int32 wrap as the
//     scalar shift.
//   - a     = min(|c| + round, 32767). The add can wrap only for lanes that
//     keep drops (see quantizeFPMaxStep), and those are zeroed by the mask.
//   - level = min((a * quant) >> (16-ts), 32767), negated when c < 0, and
//     zeroed when not kept.
func quantizeFPVectors(qcoeff []int16, coeff []int32, quant int32, round int32, dequant int32, txScale uint8) {
	n := len(coeff)
	if n == 0 {
		return
	}
	_ = qcoeff[n-1]
	quantV := archsimd.BroadcastInt32x4(quant)
	roundV := archsimd.BroadcastInt32x4(round)
	deqV := archsimd.BroadcastInt32x4(dequant)
	maxV := archsimd.BroadcastInt32x4(maxInt16)
	// Shift counts are loop-invariant vectors: a positive count shifts left and
	// a negative count shifts right (SSHL), and the sign mask shifts by 31.
	shlV := archsimd.BroadcastInt32x4(int32(1 + txScale))
	shrV := archsimd.BroadcastInt32x4(-int32(16 - int(txScale)))
	signShiftV := archsimd.BroadcastInt32x4(-31)
	cp := unsafe.Pointer(&coeff[0])
	qp := unsafe.Pointer(&qcoeff[0])
	for i := 0; i < n; i += 8 {
		c0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(cp, 4*i)))
		c1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(cp, 4*i+16)))
		lo := quantizeFPLane4SIMD(c0, quantV, roundV, deqV, maxV, shlV, shrV, signShiftV)
		hi := quantizeFPLane4SIMD(c1, quantV, roundV, deqV, maxV, shlV, shrV, signShiftV)
		quantizePackInt16x8SIMD(lo, hi).StoreArray((*[8]int16)(unsafe.Add(qp, 2*i)))
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
