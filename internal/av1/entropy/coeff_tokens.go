package entropy

// CoeffScanContext is immutable scan metadata shared by the coefficient
// readers and writers. Keep the eight-byte layout stable: position, padded
// position, lower/BR offsets, then the two EOB contexts.
type CoeffScanContext struct {
	Pos           uint16
	Padded        uint16
	Lower2DOffset int8
	BR2DOffset    int8
	LowerEOBCtx   uint8
	BREOBCtx      uint8
}

// ReadCoeffTokens2DUpdateUnchecked decodes the AC base tokens between DC and
// EOB, visiting scan in reverse. It is the tracked-2D coefficient loop from
// tile/coeff.go fused with readCDF4UpdateKnown so the range state stays local
// across ordinary tokens. Symbol selection and adaptation retain the same
// order as dav1d src/recon_tmpl.c DECODE_COEFS_CLASS / src/msac.c.
//
// The caller supplies valid CDF rows, initialized packed levels, trusted scan
// geometry, and separate caller-owned dirty arrays. Slot zero of each dirty
// array must already contain the EOB entry. Each returned length includes it.
// This scope has no error exits; EOB, DC, sign and Golomb handling remain with
// the caller. The cursor is fully synchronized on return.
func (c *Cursor) ReadCoeffTokens2DUpdateUnchecked(scan []CoeffScanContext, stride int, levels []uint8, base *[42]CDF, br *[21]CDF, dirty, levelDirty *[1024]int16) (nonzero, levelDirtyNext int) {
	dif := c.dif
	rng := uint32(c.rng)
	cnt := int32(c.cnt)
	nonzero, levelDirtyNext = 1, 1
	for i := len(scan) - 1; i >= 0; i-- {
		p := &scan[i]
		padded := int(p.Padded)
		s1 := padded + stride
		s1p1 := s1 + 1
		s2 := s1 + stride
		p1 := padded + 1
		p2 := padded + 2
		_ = levels[s2]
		_ = levels[p2]
		// Bits 0..5 hold the token and bits 6..7 hold min(token, 3).
		// The five packed neighbors yield the rounded half-sum of clipped
		// levels; raw-token carries only occur once the context saturates.
		h := int(levels[s1]) + int(levels[p1]) + int(levels[s1p1])
		l := h + int(levels[s2]) + int(levels[p2])
		ctx := (l + 64) >> 7
		if ctx > 4 {
			ctx = 4
		}
		ctx += int(p.Lower2DOffset)
		values := &base[ctx].values
		rangeValue := rng
		rngHi := rangeValue >> 8
		coded := uint32(dif >> (ecWindow - 16))
		upper := rangeValue
		c0 := values[0]
		c1 := values[1]
		c2 := values[2]
		lower := ((rngHi * uint32(c0>>ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
		symbol := 0
		if coded < lower {
			symbol = 1
			upper = lower
			lower = ((rngHi * uint32(c1>>ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
			if coded < lower {
				symbol = 2
				upper = lower
				lower = ((rngHi * uint32(c2>>ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
				if coded < lower {
					symbol = 3
					upper = lower
					lower = 0
				}
			}
		}
		if traceEntropyReads {
			traceCDFRead(c0, 4, dif, rng, readerTell(int(c.pos), cnt, int32(c.tellOffs)))
		}
		dif -= uint64(lower) << (ecWindow - 16)
		rng = upper - lower
		shift := normShift16(rng)
		cnt -= shift
		dif = ((dif + 1) << uint(shift)) - 1
		rng <<= uint(shift)
		if cnt < 0 {
			var pos int
			var tellOffs int32
			pos, dif, cnt, tellOffs = refillState(c.src, int(c.pos), dif, cnt, int32(c.tellOffs))
			c.pos = uint32(pos)
			c.tellOffs = int16(tellOffs)
		}
		count := values[4]
		rate := uint(5 + ((count >> 4) & 3))
		c0u := uint32(c0)
		c1u := uint32(c1)
		c2u := uint32(c2)
		c0u = adaptCDFEntry(c0u, rate, symbol > 0)
		c1u = adaptCDFEntry(c1u, rate, symbol > 1)
		c2u = adaptCDFEntry(c2u, rate, symbol > 2)
		values[0] = uint16(c0u)
		values[1] = uint16(c1u)
		values[2] = uint16(c2u)
		if count < MaxCDFCount {
			values[4] = count + 1
		}

		level := symbol
		packedLevel := uint8(level * 0x41)
		if level == 3 {
			// Only BR chains cross the cursor call boundary. Synchronize the
			// local arithmetic state before that helper and reload afterward.
			c.dif = dif
			c.rng = uint16(rng)
			c.cnt = int16(cnt)
			brH := (int(levels[s1]) + int(levels[p1]) + int(levels[s1p1])) & 63
			mag := (brH + 1) >> 1
			if mag > 6 {
				mag = 6
			}
			brCtx := mag + int(p.BR2DOffset)
			level += int(c.ReadCDF4HighTokenUpdateUnchecked(&br[brCtx]))
			dif = c.dif
			rng = uint32(c.rng)
			cnt = int32(c.cnt)
			packedLevel = uint8(level | 0xc0)
		}
		if level != 0 {
			levels[padded] = packedLevel
			levelDirty[levelDirtyNext] = int16(padded)
			levelDirtyNext++
			dirty[nonzero] = int16((level << 10) | int(p.Pos))
			nonzero++
		}
	}
	c.dif = dif
	c.rng = uint16(rng)
	c.cnt = int16(cnt)
	return nonzero, levelDirtyNext
}
