//go:build goexperiment.simd && (amd64 || arm64) && !purego

package quantize

// quantizeFPMaxStep bounds the quantizer steps the vector fp kernel accepts.
// The kernel adds the rounding offset to |c| in int32 lanes. A lane whose
// |c| + round wraps is always one the keep test drops, because wrapping
// |c| << (1+ts) leaves a negative value below dequant, but only while
// round << (1+ts) stays far inside int32. round is about step/2, so any step
// up to 1<<20 is covered with a wide margin; AV1 steps top out near 29247.
const quantizeFPMaxStep = 1 << 20

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
		q.AC > quantizeFPMaxStep || q.DC > quantizeFPMaxStep {
		return false
	}
	quantizeFPVectors(qcoeff[:count], coeff[:count], int32(quantAC), roundPowerOfTwo((64*q.AC)>>7, txScale), q.AC, txScale)
	roundDC := roundPowerOfTwo((64*q.DC)>>7, txScale)
	qcoeff[0] = quantizeScalarFP(coeff[0], q.DC, quantDC, roundDC, txScale)
	return true
}
