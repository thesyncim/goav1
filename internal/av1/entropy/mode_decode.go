package entropy

// This file holds decode_b's (dav1d src/decode.c) mode-info symbol chains
// with the range state in registers: each function loads dif, rng and cnt
// once, writes every read out inline (bound, split, norm, refill check, CDF
// adaptation) and stores the state back once, the way DecodeCoefs does. The
// tile package computes the contexts and passes them in.

// InterModeCDFs is the inter-mode part of dav1d's CdfModeContext:
// newmv_mode, globalmv_mode, refmv_mode, drl_bit and comp_inter_mode, in the
// field order of the tile package's type, which converts to it.
type InterModeCDFs struct {
	NewMV    [6]CDF
	GlobalMV [2]CDF
	RefMV    [6]CDF
	DRL      [3]CDF
	Compound [8]CDF
}

// Inter modes in dav1d's order (enum InterPredMode, enum CompInterPredMode).
const (
	interNearest = 0
	interNear    = 1
	interGlobal  = 2
	interNew     = 3

	compNewNew = 7
)

// compModeDRL is, per compound mode, where its DRL index starts: 1 for the
// modes with a NEARMV half (NEAR_NEAR, NEAR_NEW, NEW_NEAR), 0 for NEW_NEW
// and 2 (no DRL bits) for the rest.
var compModeDRL = [8]uint8{2, 1, 2, 2, 1, 1, 2, 0}

// readBool is one adaptive bool written out for the functions below: it is
// small enough for the inliner, so the state stays in registers.
func readBool(v *[MaxSymbols + 1]uint16, dif uint64, rng uint32) (uint, uint64, uint32) {
	return coefBool(dif, rng, ((rng>>8)*(uint32(v[0])>>ecProbShift))>>(7-ecProbShift)+ecMinProb)
}

// DecodeInterMode reads decode_b's inter mode and DRL index after
// dav1d_refmvs_find: the comp_inter_mode symbol or the single-reference
// newmv/globalmv/refmv bools, then the drl bits. ctx is the refmvs mode
// context; nMVs the candidate count; drlCtx holds get_drl_context for stack
// indices 0, 1 and 2 in bits 0-1, 2-3 and 4-5. forced is the segment
// skip/globalmv feature (single reference only), which selects GLOBALMV
// without a read. It returns the mode (InterPredMode, or CompInterPredMode
// when compound) and the stack index of the chosen candidate; drl reports
// whether the mode carries a DRL index.
func (c *Cursor) DecodeInterMode(cdfs *InterModeCDFs, compound, forced bool, ctx uint, nMVs int, drlCtx uint) (mode uint, idx int, drl bool) {
	update := c.allowCDFUpdate
	dif, rng, cnt := c.dif, uint32(c.rng), int32(c.cnt)
	var bit uint
	if compound {
		v := &cdfs.Compound[ctx&7].values
		if traceEntropyReads {
			traceCDFRead(v[0], 8, dif, rng, c.coefTell(cnt))
		}
		sym, lower, upper := coefSymbol8(v, 7, rng, uint32(dif>>(ecWindow-16)), update)
		dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		mode = uint(sym) & 7
		idx = int(compModeDRL[mode])
		if idx == 2 {
			c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
			return mode, 0, false
		}
		drl = true
	} else {
		if forced {
			c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
			return interGlobal, 0, false
		}
		v := &cdfs.NewMV[ctx&7].values
		if traceEntropyReads {
			traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
		}
		bit, dif, rng = readBool(v, dif, rng)
		dif, rng, cnt = coefNorm(dif, rng, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			updateCDF2(v, int(bit))
		}
		if bit == 0 {
			mode, idx, drl = interNew, 0, true
		} else {
			v = &cdfs.GlobalMV[(ctx>>3)&1].values
			if traceEntropyReads {
				traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
			}
			bit, dif, rng = readBool(v, dif, rng)
			dif, rng, cnt = coefNorm(dif, rng, cnt)
			if cnt < 0 {
				dif, cnt = c.coefRefill(dif, cnt)
			}
			if update {
				updateCDF2(v, int(bit))
			}
			if bit == 0 {
				c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
				return interGlobal, 0, false
			}
			v = &cdfs.RefMV[(ctx>>4)&15].values
			if traceEntropyReads {
				traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
			}
			bit, dif, rng = readBool(v, dif, rng)
			dif, rng, cnt = coefNorm(dif, rng, cnt)
			if cnt < 0 {
				dif, cnt = c.coefRefill(dif, cnt)
			}
			if update {
				updateCDF2(v, int(bit))
			}
			if bit == 0 {
				c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
				return interNearest, 0, false
			}
			mode, idx, drl = interNear, 1, true
		}
	}
	// The drl bits: up to two, each read only while another candidate
	// remains past the current index.
	for end := idx + 2; idx < end && nMVs > idx+1; {
		v := &cdfs.DRL[(drlCtx>>(2*uint(idx&3)))&3].values
		if traceEntropyReads {
			traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
		}
		bit, dif, rng = readBool(v, dif, rng)
		dif, rng, cnt = coefNorm(dif, rng, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			updateCDF2(v, int(bit))
		}
		if bit == 0 {
			break
		}
		idx++
	}
	c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
	return mode, idx, drl
}

// InterRefCDFs is the reference part of dav1d's CdfModeContext: comp,
// comp_dir, ref, comp_fwd_ref, comp_bwd_ref and comp_uni_ref, in the field
// order of the tile package's type, which converts to it.
type InterRefCDFs struct {
	CompInter   [5]CDF
	CompRefType [5]CDF
	SingleRef   [6][3]CDF
	CompFwdRef  [3][3]CDF
	CompBwdRef  [2][3]CDF
	UniCompRef  [3][3]CDF
}

// refCount is byte r of the packed per-reference neighbour counts.
func refCount(n uint64, r uint) uint32 { return uint32(n>>(r*8&63)) & 0xff }

// refCmp is the shared tail of dav1d's av1_get_*_ref_ctx (src/env.h):
// a == b gives 1, a < b 0 and a > b 2.
func refCmp(a, b uint32) uint {
	return coefB2U(a >= b) + coefB2U(a > b)
}

// DecodeRefs reads decode_b's reference frames after the comp flag: for a
// compound block comp_dir, then the bidir (fw, bw) or unidir references;
// otherwise the single-reference tree. n packs the av1_get_*_ref_ctx
// neighbour counts, one byte per reference (LAST = 0 .. ALTREF = 6); dirCtx
// is get_comp_dir_ctx. It returns the references in dav1d's numbering and
// whether a compound pair is unidirectional.
func (c *Cursor) DecodeRefs(cdfs *InterRefCDFs, compound bool, dirCtx uint, n uint64) (ref0, ref1 int, unidir bool) {
	update := c.allowCDFUpdate
	dif, rng, cnt := c.dif, uint32(c.rng), int32(c.cnt)
	n0, n1, n2, n3 := refCount(n, 0), refCount(n, 1), refCount(n, 2), refCount(n, 3)
	n4, n5, n6 := refCount(n, 4), refCount(n, 5), refCount(n, 6)
	refCtx := refCmp(n0+n1+n2+n3, n4+n5+n6) // av1_get_ref_ctx
	fwdCtx := refCmp(n0+n1, n2+n3)          // av1_get_fwd_ref_ctx
	fwd1Ctx := refCmp(n0, n1)               // av1_get_fwd_ref_1_ctx
	fwd2Ctx := refCmp(n2, n3)               // av1_get_fwd_ref_2_ctx
	bwdCtx := refCmp(n4+n5, n6)             // av1_get_bwd_ref_ctx
	bwd1Ctx := refCmp(n4, n5)               // av1_get_bwd_ref_1_ctx
	var v *[MaxSymbols + 1]uint16
	var bit uint
	if compound {
		v = &cdfs.CompRefType[dirCtx].values
		if traceEntropyReads {
			traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
		}
		bit, dif, rng = readBool(v, dif, rng)
		dif, rng, cnt = coefNorm(dif, rng, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			updateCDF2(v, int(bit))
		}
		if bit != 0 {
			// bidir: first reference (fw)
			v = &cdfs.CompFwdRef[0][fwdCtx].values
			if traceEntropyReads {
				traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
			}
			bit, dif, rng = readBool(v, dif, rng)
			dif, rng, cnt = coefNorm(dif, rng, cnt)
			if cnt < 0 {
				dif, cnt = c.coefRefill(dif, cnt)
			}
			if update {
				updateCDF2(v, int(bit))
			}
			if bit != 0 {
				v, ref0 = &cdfs.CompFwdRef[2][fwd2Ctx].values, 2
			} else {
				v, ref0 = &cdfs.CompFwdRef[1][fwd1Ctx].values, 0
			}
			if traceEntropyReads {
				traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
			}
			bit, dif, rng = readBool(v, dif, rng)
			dif, rng, cnt = coefNorm(dif, rng, cnt)
			if cnt < 0 {
				dif, cnt = c.coefRefill(dif, cnt)
			}
			if update {
				updateCDF2(v, int(bit))
			}
			ref0 += int(bit)
			// second reference (bw)
			v = &cdfs.CompBwdRef[0][bwdCtx].values
			if traceEntropyReads {
				traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
			}
			bit, dif, rng = readBool(v, dif, rng)
			dif, rng, cnt = coefNorm(dif, rng, cnt)
			if cnt < 0 {
				dif, cnt = c.coefRefill(dif, cnt)
			}
			if update {
				updateCDF2(v, int(bit))
			}
			ref1 = 6
			if bit == 0 {
				v = &cdfs.CompBwdRef[1][bwd1Ctx].values
				if traceEntropyReads {
					traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
				}
				bit, dif, rng = readBool(v, dif, rng)
				dif, rng, cnt = coefNorm(dif, rng, cnt)
				if cnt < 0 {
					dif, cnt = c.coefRefill(dif, cnt)
				}
				if update {
					updateCDF2(v, int(bit))
				}
				ref1 = 4 + int(bit)
			}
		} else {
			// unidir
			unidir = true
			v = &cdfs.UniCompRef[0][refCtx].values
			if traceEntropyReads {
				traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
			}
			bit, dif, rng = readBool(v, dif, rng)
			dif, rng, cnt = coefNorm(dif, rng, cnt)
			if cnt < 0 {
				dif, cnt = c.coefRefill(dif, cnt)
			}
			if update {
				updateCDF2(v, int(bit))
			}
			ref0, ref1 = 4, 6
			if bit == 0 {
				ref0 = 0
				// av1_get_uni_p1_ctx
				v = &cdfs.UniCompRef[1][refCmp(n1, n2+n3)].values
				if traceEntropyReads {
					traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
				}
				bit, dif, rng = readBool(v, dif, rng)
				dif, rng, cnt = coefNorm(dif, rng, cnt)
				if cnt < 0 {
					dif, cnt = c.coefRefill(dif, cnt)
				}
				if update {
					updateCDF2(v, int(bit))
				}
				ref1 = 1 + int(bit)
				if ref1 == 2 {
					v = &cdfs.UniCompRef[2][fwd2Ctx].values
					if traceEntropyReads {
						traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
					}
					bit, dif, rng = readBool(v, dif, rng)
					dif, rng, cnt = coefNorm(dif, rng, cnt)
					if cnt < 0 {
						dif, cnt = c.coefRefill(dif, cnt)
					}
					if update {
						updateCDF2(v, int(bit))
					}
					ref1 += int(bit)
				}
			}
		}
		c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
		return ref0, ref1, unidir
	}
	v = &cdfs.SingleRef[0][refCtx].values
	if traceEntropyReads {
		traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
	}
	bit, dif, rng = readBool(v, dif, rng)
	dif, rng, cnt = coefNorm(dif, rng, cnt)
	if cnt < 0 {
		dif, cnt = c.coefRefill(dif, cnt)
	}
	if update {
		updateCDF2(v, int(bit))
	}
	if bit != 0 {
		v = &cdfs.SingleRef[1][bwdCtx].values
		if traceEntropyReads {
			traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
		}
		bit, dif, rng = readBool(v, dif, rng)
		dif, rng, cnt = coefNorm(dif, rng, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			updateCDF2(v, int(bit))
		}
		if bit != 0 {
			c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
			return 6, -1, false
		}
		v, ref0 = &cdfs.SingleRef[5][bwd1Ctx].values, 4
	} else {
		v = &cdfs.SingleRef[2][fwdCtx].values
		if traceEntropyReads {
			traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
		}
		bit, dif, rng = readBool(v, dif, rng)
		dif, rng, cnt = coefNorm(dif, rng, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			updateCDF2(v, int(bit))
		}
		if bit != 0 {
			v, ref0 = &cdfs.SingleRef[4][fwd2Ctx].values, 2
		} else {
			v, ref0 = &cdfs.SingleRef[3][fwd1Ctx].values, 0
		}
	}
	if traceEntropyReads {
		traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
	}
	bit, dif, rng = readBool(v, dif, rng)
	dif, rng, cnt = coefNorm(dif, rng, cnt)
	if cnt < 0 {
		dif, cnt = c.coefRefill(dif, cnt)
	}
	if update {
		updateCDF2(v, int(bit))
	}
	c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
	return ref0 + int(bit), -1, false
}

// read3 is the symbol search of dav1d's msac_decode_symbol_adapt4 with
// n_symbols = 2 (a three-symbol CDF): the symbol and its [lower, upper).
func read3(v *[MaxSymbols + 1]uint16, dif uint64, rng uint32) (uint, uint32, uint32) {
	rngHi, coded := rng>>8, uint32(dif>>(ecWindow-16))
	l0, l1 := cdfBound(rngHi, uint32(v[0]), 2*ecMinProb), cdfBound(rngHi, uint32(v[1]), ecMinProb)
	sym, lower, upper := uint(0), l0, rng
	if coded < l0 {
		sym, lower, upper = 1, l1, l0
	}
	if coded < l1 {
		sym, lower, upper = 2, 0, l1
	}
	return sym, lower, upper
}

// adapt3 is update_cdf for a three-symbol CDF.
func adapt3(v *[MaxSymbols + 1]uint16, sym uint) {
	count := v[3]
	rate := uint(4 + ((count >> 4) & 3))
	v[0] = uint16(adaptCDFEntry(uint32(v[0]), rate, sym > 0))
	v[1] = uint16(adaptCDFEntry(uint32(v[1]), rate, sym > 1))
	v[3] = count + coefB2U16(count < MaxCDFCount)
}

// coefB2U16 is b as 0 or 1.
func coefB2U16(b bool) uint16 { return uint16(coefB2U(b)) }

// DecodeFilters reads decode_b's switchable subpel filters: the vertical
// filter with context ctxY and, under dual_filter, the horizontal one with
// ctxX (get_filter_ctx); otherwise the horizontal filter repeats the
// vertical one.
func (c *Cursor) DecodeFilters(cdfs *[16]CDF, dual bool, ctxY, ctxX uint) (y, x uint) {
	update := c.allowCDFUpdate
	dif, rng, cnt := c.dif, uint32(c.rng), int32(c.cnt)
	v := &cdfs[ctxY&15].values
	if traceEntropyReads {
		traceCDFRead(v[0], 3, dif, rng, c.coefTell(cnt))
	}
	y, lower, upper := read3(v, dif, rng)
	dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
	if cnt < 0 {
		dif, cnt = c.coefRefill(dif, cnt)
	}
	if update {
		adapt3(v, y)
	}
	x = y
	if dual {
		v = &cdfs[ctxX&15].values
		if traceEntropyReads {
			traceCDFRead(v[0], 3, dif, rng, c.coefTell(cnt))
		}
		x, lower, upper = read3(v, dif, rng)
		dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			adapt3(v, x)
		}
	}
	c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
	return y, x
}

// IntraCDFs points at the intra-mode CDFs of dav1d's CdfModeContext that
// DecodeIntraModes reads besides the y-mode CDF: angle_delta, uv_mode,
// cfl_sign, cfl_alpha and filter_intra (the mode symbol).
type IntraCDFs struct {
	Angle      *[8]CDF
	UV         *[2][13]CDF
	CFLSign    *CDF
	CFLAlpha   *[6]CDF
	FilterMode *CDF
}

// Flags of DecodeIntraModes.
const (
	IntraAngle  = 1 << iota // b_dim[2] + b_dim[3] >= 2: angle deltas are coded
	IntraChroma             // has_chroma
	IntraCFL                // cfl_allowed
	IntraFilter             // filter_intra is coded for a DC_PRED block (no palette)
)

// IntraModes is what DecodeIntraModes read. Angles are the coded symbols
// (delta + 3); CFLSign is the joint sign symbol and CFLU, CFLV the alpha
// symbols (magnitude - 1), zero when the sign is zero.
type IntraModes struct {
	Y, YAngle, UV, UVAngle uint8
	CFLSign, CFLU, CFLV    uint8
	Filter, FilterMode     uint8
}

// DecodeIntraModes reads the intra arm of decode_b from y_mode to the
// filter-intra flag, the range state in registers: y_mode (from yCDF), its
// angle delta, uv_mode with the CfL alphas or its angle delta, and, when
// flags has IntraFilter, use_filter_intra (filterCDF) with the filter mode.
// The palette syntax, which dav1d reads between uv_mode and filter_intra,
// is not covered: callers with screen content tools read it in between and
// pass IntraFilter only when no palette can be coded.
func (c *Cursor) DecodeIntraModes(out *IntraModes, cdfs *IntraCDFs, yCDF, filterCDF *CDF, flags uint) {
	update := c.allowCDFUpdate
	dif, rng, cnt := c.dif, uint32(c.rng), int32(c.cnt)
	v := &yCDF.values
	if traceEntropyReads {
		traceCDFRead(v[0], 13, dif, rng, c.coefTell(cnt))
	}
	sym, lower, upper := coefSymbol(v, 13, rng, uint32(dif>>(ecWindow-16)), update)
	dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
	if cnt < 0 {
		dif, cnt = c.coefRefill(dif, cnt)
	}
	y := uint(sym)
	out.Y = uint8(y)
	out.YAngle = 3
	// angle delta: directional modes VERT_PRED (1) .. VERT_LEFT_PRED (8)
	if flags&IntraAngle != 0 && y-1 < 8 {
		v = &cdfs.Angle[y-1].values
		if traceEntropyReads {
			traceCDFRead(v[0], 7, dif, rng, c.coefTell(cnt))
		}
		sym, lower, upper = coefSymbol8(v, 6, rng, uint32(dif>>(ecWindow-16)), update)
		dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		out.YAngle = uint8(sym)
	}
	out.UVAngle = 3
	out.CFLSign, out.CFLU, out.CFLV = 0, 0, 0
	if flags&IntraChroma != 0 {
		cfl := (flags / IntraCFL) & 1
		v = &cdfs.UV[cfl][y].values
		n := 13 + int(cfl)
		if traceEntropyReads {
			traceCDFRead(v[0], n, dif, rng, c.coefTell(cnt))
		}
		sym, lower, upper = coefSymbol(v, n, rng, uint32(dif>>(ecWindow-16)), update)
		dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		uv := uint(sym)
		out.UV = uint8(uv)
		if uv == 13 { // CFL_PRED
			v = &cdfs.CFLSign.values
			if traceEntropyReads {
				traceCDFRead(v[0], 8, dif, rng, c.coefTell(cnt))
			}
			sym, lower, upper = coefSymbol8(v, 7, rng, uint32(dif>>(ecWindow-16)), update)
			dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
			if cnt < 0 {
				dif, cnt = c.coefRefill(dif, cnt)
			}
			out.CFLSign = uint8(sym)
			sign := uint(sym) + 1
			signU := sign * 0x56 >> 8
			signV := sign - signU*3
			if signU != 0 {
				v = &cdfs.CFLAlpha[coefB2U(signU == 2)*3+signV].values
				if traceEntropyReads {
					traceCDFRead(v[0], 16, dif, rng, c.coefTell(cnt))
				}
				sym, lower, upper = coefSymbol(v, 16, rng, uint32(dif>>(ecWindow-16)), update)
				dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
				if cnt < 0 {
					dif, cnt = c.coefRefill(dif, cnt)
				}
				out.CFLU = uint8(sym)
			}
			if signV != 0 {
				v = &cdfs.CFLAlpha[coefB2U(signV == 2)*3+signU].values
				if traceEntropyReads {
					traceCDFRead(v[0], 16, dif, rng, c.coefTell(cnt))
				}
				sym, lower, upper = coefSymbol(v, 16, rng, uint32(dif>>(ecWindow-16)), update)
				dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
				if cnt < 0 {
					dif, cnt = c.coefRefill(dif, cnt)
				}
				out.CFLV = uint8(sym)
			}
		} else if flags&IntraAngle != 0 && uv-1 < 8 {
			v = &cdfs.Angle[uv-1].values
			if traceEntropyReads {
				traceCDFRead(v[0], 7, dif, rng, c.coefTell(cnt))
			}
			sym, lower, upper = coefSymbol8(v, 6, rng, uint32(dif>>(ecWindow-16)), update)
			dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
			if cnt < 0 {
				dif, cnt = c.coefRefill(dif, cnt)
			}
			out.UVAngle = uint8(sym)
		}
	}
	out.Filter, out.FilterMode = 0, 0
	if flags&IntraFilter != 0 && y == 0 {
		v = &filterCDF.values
		if traceEntropyReads {
			traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
		}
		var bit uint
		bit, dif, rng = readBool(v, dif, rng)
		dif, rng, cnt = coefNorm(dif, rng, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			updateCDF2(v, int(bit))
		}
		if bit != 0 {
			out.Filter = 1
			v = &cdfs.FilterMode.values
			if traceEntropyReads {
				traceCDFRead(v[0], 5, dif, rng, c.coefTell(cnt))
			}
			sym, lower, upper = coefSymbol8(v, 4, rng, uint32(dif>>(ecWindow-16)), update)
			dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
			if cnt < 0 {
				dif, cnt = c.coefRefill(dif, cnt)
			}
			out.FilterMode = uint8(sym)
		}
	}
	c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
}

// DecodeSkips reads decode_b's skip_mode (when smCDF is non-nil) and skip
// flags back to back. A set skip_mode, or forced (the segment skip
// feature), implies skip without a read.
func (c *Cursor) DecodeSkips(smCDF, skipCDF *CDF, forced bool) (skipMode, skip bool) {
	update := c.allowCDFUpdate
	dif, rng, cnt := c.dif, uint32(c.rng), int32(c.cnt)
	var bit uint
	if smCDF != nil {
		v := &smCDF.values
		if traceEntropyReads {
			traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
		}
		bit, dif, rng = readBool(v, dif, rng)
		dif, rng, cnt = coefNorm(dif, rng, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			updateCDF2(v, int(bit))
		}
		skipMode = bit != 0
	}
	if skipMode || forced {
		c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
		return skipMode, true
	}
	v := &skipCDF.values
	if traceEntropyReads {
		traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
	}
	bit, dif, rng = readBool(v, dif, rng)
	dif, rng, cnt = coefNorm(dif, rng, cnt)
	if cnt < 0 {
		dif, cnt = c.coefRefill(dif, cnt)
	}
	if update {
		updateCDF2(v, int(bit))
	}
	c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
	return false, bit != 0
}
