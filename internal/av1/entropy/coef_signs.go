package entropy

// ReadCoefGolomb is dav1d's read_golomb (src/recon_tmpl.c): an Exp-Golomb
// code with the unary prefix capped at 32 bits.
func (c *Cursor) ReadCoefGolomb() uint32 {
	length := 0
	val := uint32(1)
	for c.ReadBitTrustedInline() == 0 && length < 32 {
		length++
	}
	for ; length > 0; length-- {
		val = val<<1 + uint32(c.ReadBitTrustedInline())
	}
	return val - 1
}
