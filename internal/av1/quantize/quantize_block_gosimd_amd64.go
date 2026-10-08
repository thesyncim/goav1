//go:build goexperiment.simd && amd64 && !purego

package quantize

import (
	"simd/archsimd"
	"unsafe"
)

// quantizeFPVectors applies the av1_quantize_fp lane rule to coeff, writing
// qcoeff, eight coefficients per iteration in one Int32x8 register (AVX2).
// len(coeff) must be a multiple of 8 and len(qcoeff) >= len(coeff). The lane
// rule and its exactness argument are shared with the arm64 kernel (see
// quantize_block_gosimd_arm64.go).
func quantizeFPVectors(qcoeff []int16, coeff []int32, quant int32, round int32, dequant int32, txScale uint8) {
	n := len(coeff)
	if n == 0 {
		return
	}
	_ = qcoeff[n-1]
	quantV := archsimd.BroadcastInt32x8(quant)
	roundV := archsimd.BroadcastInt32x8(round)
	deqV := archsimd.BroadcastInt32x8(dequant)
	maxV := archsimd.BroadcastInt32x8(maxInt16)
	shl := uint64(1 + txScale)
	shr := uint64(16 - txScale)
	cp := unsafe.Pointer(&coeff[0])
	qp := unsafe.Pointer(&qcoeff[0])
	for i := 0; i < n; i += 8 {
		c := archsimd.LoadInt32x8Array((*[8]int32)(unsafe.Add(cp, 4*i)))
		abs := c.Abs()
		keep := abs.ShiftAllLeft(shl).GreaterEqual(deqV)
		a := abs.Add(roundV).Min(maxV)
		level := a.Mul(quantV).ShiftAllRight(shr).Min(maxV)
		level = quantizeApplySignAVX2(level, c).Masked(keep)
		// SaturateToInt16Concat packs the low half then the high half, which
		// is the element order of the eight consecutive outputs.
		level.GetLo().SaturateToInt16Concat(level.GetHi()).StoreArray((*[8]int16)(unsafe.Add(qp, 2*i)))
	}
}

// quantizeApplySignAVX2 is the eight-lane conditional negate used by the
// quantizers: src<0 selects -mag, which is mag^signMask - signMask with
// signMask all-ones in negative lanes.
func quantizeApplySignAVX2(mag archsimd.Int32x8, src archsimd.Int32x8) archsimd.Int32x8 {
	signMask := src.ShiftAllRight(31)
	return mag.Xor(signMask).Sub(signMask)
}
