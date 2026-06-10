//go:build arm64 && !purego

package quantize

import (
	"math/bits"
	"unsafe"
)

// quantizeBNEONCtx carries the kernel arguments; offsets are mirrored by
// #define in quantize_b_neon_arm64.s. Count is the coefficient count (a
// multiple of 16); Shift is -(l - txScale) where l is the step's MSB index
// (SSHL with a negative count shifts right).
type quantizeBNEONCtx struct {
	Coeff unsafe.Pointer
	Out   unsafe.Pointer
	Count int64
	Quant int64
	Round int64
	Zbin  int64
	Shift int64
}

//go:noescape
func quantizeBNEONAsm(ctx *quantizeBNEONCtx)

func init() {
	quantizeBBlockImpl = quantizeBBlockNEON
}

// quantizeBBlockNEON quantizes a contiguous square block with the zbin rule,
// four coefficients at a time; the DC coefficient is redone with the scalar
// rule and DC constants. The widening multiply keeps the 32x16 product
// exact, and invert_quant's power-of-two second multiply folds into the
// final shift, so the kernel is bit-exact with the scalar rule.
func quantizeBBlockNEON(qcoeff []int16, coeff []int32, n int, q Quantizer, txScale uint8) bool {
	count := n * n
	if count%16 != 0 {
		return false
	}
	l := 31 - int64(bits.LeadingZeros32(uint32(q.AC)))
	if l < int64(txScale) {
		return false
	}
	zbinFactor := int32(80)
	if q.DC < 148 {
		zbinFactor = 84
	}
	acQuant, _ := invertQuant(q.AC)
	ctx := quantizeBNEONCtx{
		Coeff: unsafe.Pointer(&coeff[0]),
		Out:   unsafe.Pointer(&qcoeff[0]),
		Count: int64(count),
		Quant: int64(acQuant),
		Round: int64(roundPowerOfTwo((48*q.AC)>>7, txScale)),
		Zbin:  int64(roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.AC, 7), txScale)),
		Shift: -(l - int64(txScale)),
	}
	quantizeBNEONAsm(&ctx)
	dcQuant, dcShift := invertQuant(q.DC)
	dc := quantBParams{
		zbin:  roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.DC, 7), txScale),
		round: roundPowerOfTwo((48*q.DC)>>7, txScale),
		quant: dcQuant,
		shift: dcShift,
	}
	qcoeff[0] = quantizeScalarB(coeff[0], &dc, txScale)
	return true
}
