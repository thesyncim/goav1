//go:build goexperiment.simd && (amd64 || arm64) && !purego

package quantize

import "math/bits"

// quantizeVectorMaxStep bounds the quantizer steps the vector block kernels
// accept. They do their lane math in int32 on the rounding offset and the zbin
// (about 0.375*step and 0.66*step). Above this cap the scalar rule applies, so
// nothing can wrap in the vector path. AV1 steps top out near 29247.
//
// For fp, the cap also means round << (1+ts) stays far inside int32, so a
// lane whose |c| + round wraps is always one the keep test drops.
const quantizeVectorMaxStep = 1 << 20

// quantizeFPBlockSIMD quantizes a contiguous square block with the fp rule
// using the architecture's vector kernel, which covers every coefficient at
// eight per iteration. The DC coefficient is redone with the scalar rule and
// DC constants. The kernel's int32 lane math is exact: quant <= 1<<14 (the
// dequant >= 4 guard) keeps (32767 + round) * quant inside int32, the step cap
// keeps the rounding add in range for kept lanes, and the keep test reproduces
// quantizeScalarFP's int32 shift.
func quantizeFPBlockSIMD(qcoeff []int16, coeff []int32, n int, q Quantizer, txScale uint8) bool {
	count := n * n
	quantAC := int64(1<<16) / int64(q.AC)
	quantDC := int64(1<<16) / int64(q.DC)
	if count%16 != 0 || quantAC > 1<<14 || quantDC > 1<<14 ||
		q.AC > quantizeVectorMaxStep || q.DC > quantizeVectorMaxStep {
		return false
	}
	quantizeFPVectors(qcoeff[:count], coeff[:count], int32(quantAC), roundPowerOfTwo((64*q.AC)>>7, txScale), q.AC, txScale)
	roundDC := roundPowerOfTwo((64*q.DC)>>7, txScale)
	qcoeff[0] = quantizeScalarFP(coeff[0], q.DC, quantDC, roundDC, txScale)
	return true
}

// quantizeBBlockSIMD quantizes a contiguous square block with the zbin rule
// (quantize_b) using the architecture's vector kernel, eight coefficients per
// iteration, with the DC coefficient redone by the scalar rule. The kernel
// reproduces quantizeScalarB exactly:
//   - the zbin test is an unsigned compare on |c|, so INT32_MIN (|c| = 2^31 in
//     the scalar's int64 math) is kept, as the scalar keeps it.
//   - the rounding add and the (tmp*quant)>>16 product stay inside int32 for
//     tmp <= 32767 and |quant| <= 1<<15.
//   - invert_quant's power-of-two multiply folds into the right shift by
//     l-txScale, which is exact because l >= txScale is required here.
func quantizeBBlockSIMD(qcoeff []int16, coeff []int32, n int, q Quantizer, txScale uint8) bool {
	count := n * n
	if count%16 != 0 || q.AC > quantizeVectorMaxStep || q.DC > quantizeVectorMaxStep {
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
	quantizeBVectors(qcoeff[:count], coeff[:count], acQuant,
		roundPowerOfTwo((48*q.AC)>>7, txScale),
		roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.AC, 7), txScale),
		uint64(l-int64(txScale)))
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
