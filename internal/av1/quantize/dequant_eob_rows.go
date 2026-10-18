package quantize

// DequantizeBlockScaledBitDepthEOBRowsFlatTrusted dequantizes a flat EOB scan
// prefix and returns one past its maximum coefficient row. The caller supplies
// a positive power-of-two coefficient height, a destination covering exactly
// the coefficient block, and the validated buffers, scan, EOB, quantizer,
// transform scale and bit depth required by the other trusted EOB entry points.
//
// The row bound includes every scan-prefix position, even a zero coefficient:
// it replaces a separate traversal of that same prefix, not a nonzero scan.
func DequantizeBlockScaledBitDepthEOBRowsFlatTrusted(dst []int32, coeff []int16, scan []int16, eob, height int, q Quantizer, txScale uint8, bitDepth uint8) int {
	dqMin, dqMax, _ := dqCoeffBounds(bitDepth)
	clear(dst)
	mask := height - 1
	activeRows := 0
	for i := 0; i < eob; i++ {
		pos := int(scan[i])
		activeRows = max(activeRows, (pos&mask)+1)
		scale := q.AC
		if pos == 0 {
			scale = q.DC
		}
		dst[pos] = dequantScalar(int32(coeff[pos]), scale, txScale, dqMin, dqMax)
	}
	return activeRows
}
