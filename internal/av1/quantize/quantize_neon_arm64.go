//go:build arm64 && !purego

package quantize

import "unsafe"

// quantizeFPNEONCtx carries the kernel arguments; offsets are mirrored by
// #define in quantize_neon_arm64.s. Count is the coefficient count (a
// multiple of 16). ShiftL is 1+txScale, ShiftR is -(16-txScale) (SSHL with a
// negative count shifts right), Quant is (1<<16)/dequant and Dequant the AC
// step.
type quantizeFPNEONCtx struct {
	Coeff  unsafe.Pointer
	Out    unsafe.Pointer
	Count  int64
	Quant  int64
	Round  int64
	Deq    int64
	ShiftL int64
	ShiftR int64
}

//go:noescape
func quantizeFPNEONAsm(ctx *quantizeFPNEONCtx)

// quantizeFPBlockNEON quantizes a contiguous square block with the fp rule,
// vectorized four coefficients at a time; the DC coefficient is re-done by
// the scalar rule afterwards with the DC constants. The int32 lane math is
// exact: 8-bit quantizer tables keep dequant >= 4, so quant <= 1<<14 and
// (32767 + round) * quant stays inside int32.
func quantizeFPBlockNEON(qcoeff []int16, coeff []int32, n int, q Quantizer, txScale uint8) bool {
	count := n * n
	if count%16 != 0 || int64(1<<16)/int64(q.AC) > 1<<14 || int64(1<<16)/int64(q.DC) > 1<<14 {
		return false
	}
	roundAC := roundPowerOfTwo((64*int32(q.AC))>>7, txScale)
	ctx := quantizeFPNEONCtx{
		Coeff:  unsafe.Pointer(&coeff[0]),
		Out:    unsafe.Pointer(&qcoeff[0]),
		Count:  int64(count),
		Quant:  int64(1<<16) / int64(q.AC),
		Round:  int64(roundAC),
		Deq:    int64(q.AC),
		ShiftL: int64(1 + txScale),
		ShiftR: -int64(16 - int(txScale)),
	}
	quantizeFPNEONAsm(&ctx)
	quantDC := int64(1<<16) / int64(q.DC)
	roundDC := roundPowerOfTwo((64*int32(q.DC))>>7, txScale)
	qcoeff[0] = quantizeScalarFP(coeff[0], q.DC, quantDC, roundDC, txScale)
	return true
}
