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
	shift &= 63 // log2 of the scan height; masked so rcI >> shift needs no oversize-shift select
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
		if y > 4 {
			y = 4
		}
		if x > 4 {
			x = 4
		}
		ctx := uint(offsets[y][x]) + min((mag+64)>>7, 4)

		// dav1d_msac_decode_symbol_adapt4(lo_cdf[ctx], 3)
		values := &lo[ctx].values
		rngHi := rng >> 8
		coded := uint32(dif >> (ecWindow - 16))
		upper := rng
		// The three probabilities and the zero terminator as one word, the
		// four lanes dav1d's asm loads with ld1 {v0.4h}.
		w := uint64(values[0]) | uint64(values[1])<<16 | uint64(values[2])<<32 | uint64(values[3])<<48
		c0 := uint16(w)
		c1 := uint16(w >> 16)
		c2 := uint16(w >> 32)
		// dav1d's arm64 msac_decode_symbol_adapt4 evaluates every bound at
		// once and counts those above the coded value. The inverse CDF is
		// non-increasing and the EC_MIN_PROB offsets strictly decrease, so
		// l0 > l1 > l2 and each hit implies the earlier ones: the bounds
		// resolve as conditional selects, not a data-dependent branch chain.
		l0 := ((rngHi * uint32(c0>>ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
		l1 := ((rngHi * uint32(c1>>ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
		l2 := ((rngHi * uint32(c2>>ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
		lower := l0
		tok := uint(0)
		if coded < l0 {
			upper, lower, tok = l0, l1, 1
		}
		if coded < l1 {
			upper, lower, tok = l1, l2, 2
		}
		if coded < l2 {
			upper, lower, tok = l2, 0, 3
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
			rate := uint(5+((count>>4)&3)) & 15
			// update_cdf on all lanes at once (dav1d's asm does it in a
			// vector): every lane moves toward 0, c - (c >> rate), or toward
			// 32768, c + ((32768 - c) >> rate), and the lanes below tok take
			// the second. Each lane is below 32768, so neither form borrows
			// or carries across lanes; the masks clear the bits a whole-word
			// shift brings in from the lane above. The terminator lane stays
			// zero.
			const lanes = 0x0000_0001_0001_0001
			m := (0xffff >> rate) * uint64(lanes)
			down := w - (w>>rate)&m
			up := w + ((0x8000*lanes-w)>>rate)&m
			sel := uint64(0)
			if coded < l0 {
				sel = 0xffff
			}
			if coded < l1 {
				sel = 0xffff_ffff
			}
			if coded < l2 {
				sel = 0xffff_ffff_ffff
			}
			w = down ^ (down^up)&sel
			values[0] = uint16(w)
			values[1] = uint16(w >> 16)
			values[2] = uint16(w >> 32)
			values[3] = uint16(w >> 48)
			if count < MaxCDFCount {
				count++
			}
			values[4] = count
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
