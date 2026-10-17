package quantize

// QuantizeBRowGroupsTrusted quantizes a square n x n coefficient block
// (n = 8, 16 or 32; column-major, both strides n) with the zbin rule of
// QuantizeBlockScaledB, but only over the four-row groups set in live: bit g
// covers rows 4g..4g+3 of every column. Rows of the other groups get zero
// levels; the caller must have proven that every coefficient there lies below
// its zero bin (ZeroBinB), so the levels equal QuantizeBlockScaledB's
// exactly. It reports whether any level is non-zero.
func QuantizeBRowGroupsTrusted(qcoeff []int16, coeff []int32, n int, live uint8, q Quantizer, txScale uint8) bool {
	return quantizeBRowGroupsImpl(qcoeff, coeff, n, live, q, txScale)
}

// quantizeBRowGroupsImpl binds the row-group quantizer; the scalar rule is the
// reference and the fallback for steps the vector kernel does not cover.
var quantizeBRowGroupsImpl = quantizeBRowGroupsScalar

func quantizeBRowGroupsScalar(qcoeff []int16, coeff []int32, n int, live uint8, q Quantizer, txScale uint8) bool {
	_ = qcoeff[n*n-1]
	_ = coeff[n*n-1]
	zbinFactor := int32(80)
	if q.DC < 148 {
		zbinFactor = 84
	}
	dc := quantBParams{
		zbin:  roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.DC, 7), txScale),
		round: roundPowerOfTwo((48*q.DC)>>7, txScale),
	}
	dc.quant, dc.shift = invertQuant(q.DC)
	ac := quantBParams{
		zbin:  roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.AC, 7), txScale),
		round: roundPowerOfTwo((48*q.AC)>>7, txScale),
	}
	ac.quant, ac.shift = invertQuant(q.AC)
	nonZero := false
	for col := range n {
		for g := 0; g < n/4; g++ {
			base := col*n + 4*g
			if live&(1<<g) == 0 {
				qcoeff[base], qcoeff[base+1], qcoeff[base+2], qcoeff[base+3] = 0, 0, 0, 0
				continue
			}
			for r := base; r < base+4; r++ {
				p := &ac
				if r == 0 {
					p = &dc
				}
				v := quantizeScalarB(coeff[r], p, txScale)
				qcoeff[r] = v
				nonZero = nonZero || v != 0
			}
		}
	}
	return nonZero
}
