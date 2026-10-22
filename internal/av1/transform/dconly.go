package transform

// InverseDCTDCOnlySample is the DC-only branch of dav1d's inv_txfm_add_c
// (src/itx_tmpl.c, eob < has_dconly): the DCT_DCT residual of a block whose
// only nonzero coefficient is dc, in closed form. The row and column DCT DC
// gains are each 181/256 (cospi[32] = 2896 = 181 << 4), with the rect2 gain
// first, the per-size mid-pass shift between them, and the final (x+8)>>4
// folded into the column multiply.
//
// dav1d skips the stage clamps here; the mid-pass clamp to the column range
// is kept because it binds for high-bit-depth DC values near cf_max, where
// libaom (and InverseDCTDCOnlySampleBitDepth) clamp. The caller guarantees dc
// is a dequantized coefficient (|dc| <= cf_max), which the row stage cannot
// overflow.
func InverseDCTDCOnlySample(dc int32, size Size, bitDepth uint8) int16 {
	idx := sizeIndex(size)
	if size.IsRect2() {
		dc = (dc*181 + 128) >> 8
	}
	dc = (dc*181 + 128) >> 8
	shift := sizeShiftTable[idx]
	dc = (dc + (1<<shift)>>1) >> shift
	if bitDepth > 8 {
		colMax := int32(1)<<(max(bitDepth+6, 16)-1) - 1
		dc = min(max(dc, -colMax-1), colMax)
	}
	return int16((dc*181 + 128 + 2048) >> 12)
}
