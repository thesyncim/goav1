package entropy

// DecodeCoefSignsAC is the AC half of dav1d decode_coefs' "residual and sign"
// pass (src/recon_tmpl.c) for a TXB without quantizer matrices, with
// dav1d_msac_decode_bool_equi inlined so the range state stays in registers
// across coefficients. It walks dav1d's token chain from rc (cf[rc] = tok <<
// 11 | next rc, ending at rc 0), replacing each entry with its dequantized,
// signed coefficient. dqShift must be the transform's dequant shift (0..2).
//
// It returns the token sum and the extents of the nonzero AC coefficients
// (rows: max (rc & rowMask) + 1, columns: max (rc >> colShift) + 1) merged
// into the values passed in.
func (c *Cursor) DecodeCoefSignsAC(cf *[1024]int32, rc uint, acDQ uint32, dqShift uint, cfMax uint32, rowMask, colShift uint, culLevel, activeRows, activeCols uint) (uint, uint, uint) {
	dqShift &= 3
	colShift &= 63
	dif := c.dif
	rng := uint32(c.rng)
	cnt := int32(c.cnt)
	for {
		// dav1d_msac_decode_bool_equi. The sign is a coin flip, so both
		// outcomes are computed first and selected rather than branched on.
		split := ((rng >> 8) << 7) + ecMinProb
		window := uint64(split) << (ecWindow - 16)
		difTaken := dif - window
		rngTaken := rng - split
		sign := uint32(1)
		if dif >= window {
			dif, split, sign = difTaken, rngTaken, 0
		}
		if traceEntropyReads {
			traceBoolRead(CDFProbTop/2, difTaken+window, rng, readerTell(int(c.pos), cnt, int32(c.tellOffs)))
		}
		// An equiprobable split leaves shift 0..2.
		sh := uint(normShift16(split)) & 3
		cnt -= int32(sh)
		dif = ((dif + 1) << sh) - 1
		rng = split << sh
		if cnt < 0 {
			var pos int
			var tellOffs int32
			pos, dif, cnt, tellOffs = refillState(c.src, int(c.pos), dif, cnt, int32(c.tellOffs))
			c.pos = uint32(pos)
			c.tellOffs = int16(tellOffs)
		}

		rc &= 1023
		rcTok := uint(uint32(cf[rc]))
		var tok uint
		var v uint32
		if rcTok >= 15<<11 {
			c.dif = dif
			c.rng = uint16(rng)
			c.cnt = int16(cnt)
			tok = uint(c.ReadCoefGolomb()) + 15
			dif = c.dif
			rng = uint32(c.rng)
			cnt = int32(c.cnt)
			tok &= 0xfffff
			v = ((acDQ * uint32(tok)) & 0xffffff) >> dqShift
			v = min(v, cfMax+sign)
		} else {
			tok = rcTok >> 11
			v = (acDQ * uint32(tok)) >> dqShift
		}
		culLevel += tok
		w := int32(v)
		if sign != 0 {
			w = -w
		}
		cf[rc] = w
		activeRows = max(activeRows, (rc&rowMask)+1)
		activeCols = max(activeCols, (rc>>colShift)+1)
		rc = rcTok & 0x3ff
		if rc == 0 {
			break
		}
	}
	c.dif = dif
	c.rng = uint16(rng)
	c.cnt = int16(cnt)
	return culLevel, activeRows, activeCols
}

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
