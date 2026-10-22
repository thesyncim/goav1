package entropy

// CoefLevelsLen sizes dav1d's t->scratch.levels token buffer. A 32x32 2D block
// needs stride 32 times 32+2 columns and the 1D classes stride 16 times 16+2
// rows; the extra slack lets the token loops address rc + 2*stride with rc
// and stride masked to their maxima, so the loads need no bounds checks.
const CoefLevelsLen = 1024 + 2*64

// DecodeCoefsAC2D is the AC loop of dav1d's DECODE_COEFS_CLASS(TX_CLASS_2D)
// (src/recon_tmpl.c decode_coefs) with dav1d_msac_decode_symbol_adapt4
// (src/msac.c) inlined so the range state stays in registers across tokens.
// It visits scan[len(scan)-1] down to scan[1]; scan[0] is DC and the eob
// coefficient has been decoded by the caller, whose rc is passed in.
//
// levels holds dav1d's packed level bytes (bits 0-5 token, 6-7 min(token, 3))
// at rc with stride = scan height and no row padding; cf receives dav1d's
// token chain, cf[rc] = tok << 11 | rc of the previous nonzero. The returned
// rc is the chain head (0 when no AC coefficient is nonzero).
func (c *Cursor) DecodeCoefsAC2D(scan []int16, shift, stride uint, offsets *[5][5]uint8, levels *[CoefLevelsLen]uint8, cf *[1024]int32, lo *[42]CDF, hi *[21]CDF, rc uint) uint {
	stride &= 63
	mask := stride - 1
	update := c.allowCDFUpdate
	dif := c.dif
	rng := uint32(c.rng)
	cnt := int32(c.cnt)
	for i := len(scan) - 1; i > 0; i-- {
		rcI := uint(scan[i]) & 1023
		x, y := rcI>>shift, rcI&mask
		mag := uint(levels[rcI+1]) + uint(levels[rcI+stride]) + uint(levels[rcI+stride+1])
		hiMag := mag
		mag += uint(levels[rcI+2]) + uint(levels[rcI+2*stride])
		ctx := uint(offsets[min(y, 4)][min(x, 4)]) + min((mag+64)>>7, 4)

		// dav1d_msac_decode_symbol_adapt4(lo_cdf[ctx], 3)
		values := &lo[ctx].values
		rngHi := rng >> 8
		coded := uint32(dif >> (ecWindow - 16))
		upper := rng
		c0 := values[0]
		c1 := values[1]
		c2 := values[2]
		lower := ((rngHi * uint32(c0>>ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
		tok := uint(0)
		if coded < lower {
			tok = 1
			upper = lower
			lower = ((rngHi * uint32(c1>>ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
			if coded < lower {
				tok = 2
				upper = lower
				lower = ((rngHi * uint32(c2>>ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
				if coded < lower {
					tok = 3
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
		sh := normShift16(rng)
		cnt -= sh
		dif = ((dif + 1) << uint(sh)) - 1
		rng <<= uint(sh)
		if cnt < 0 {
			var pos int
			var tellOffs int32
			pos, dif, cnt, tellOffs = refillState(c.src, int(c.pos), dif, cnt, int32(c.tellOffs))
			c.pos = uint32(pos)
			c.tellOffs = int16(tellOffs)
		}
		if update {
			count := values[4]
			rate := uint(5 + (count >> 4))
			c0u := uint32(c0)
			c1u := uint32(c1)
			c2u := uint32(c2)
			switch tok {
			case 0:
				c0u -= c0u >> rate
				c1u -= c1u >> rate
				c2u -= c2u >> rate
			case 1:
				c0u += (CDFProbTop - c0u) >> rate
				c1u -= c1u >> rate
				c2u -= c2u >> rate
			case 2:
				c0u += (CDFProbTop - c0u) >> rate
				c1u += (CDFProbTop - c1u) >> rate
				c2u -= c2u >> rate
			default:
				c0u += (CDFProbTop - c0u) >> rate
				c1u += (CDFProbTop - c1u) >> rate
				c2u += (CDFProbTop - c2u) >> rate
			}
			values[0] = uint16(c0u)
			values[1] = uint16(c1u)
			values[2] = uint16(c2u)
			if count < MaxCDFCount {
				values[4] = count + 1
			}
		}

		if tok == 3 {
			hiCtx := uint(7)
			if x|y > 1 {
				hiCtx = 14
			}
			hiMag &= 63
			if hiMag > 12 {
				hiCtx += 6
			} else {
				hiCtx += (hiMag + 1) >> 1
			}
			// dav1d_msac_decode_hi_tok crosses the cursor call boundary;
			// synchronize the local range state around it.
			c.dif = dif
			c.rng = uint16(rng)
			c.cnt = int16(cnt)
			tok = 3 + uint(c.ReadCDF4HighTokenUnchecked(&hi[hiCtx]))
			dif = c.dif
			rng = uint32(c.rng)
			cnt = int32(c.cnt)
			levels[rcI] = uint8(tok + 3<<6)
			cf[rcI] = int32(tok<<11 | rc)
			rc = rcI
		} else {
			// 0x1 for tok, 0x7ff as bitmask for rc, 0x41 for level_tok
			t := uint32(tok) * 0x17ff41
			levels[rcI] = uint8(t)
			// tok ? (tok << 11) | rc : 0
			t = (t >> 9) & (uint32(rc) + ^uint32(0x7ff))
			if t != 0 {
				rc = rcI
			}
			cf[rcI] = int32(t)
		}
	}
	c.dif = dif
	c.rng = uint16(rng)
	c.cnt = int16(cnt)
	return rc
}
