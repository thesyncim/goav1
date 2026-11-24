package entropy

// MVComponentCDFs is one component of dav1d's CdfMvContext (src/cdf.h),
// in the field order of the tile package's type, which converts to it.
type MVComponentCDFs struct {
	Classes  CDF
	Class0FP [2]CDF
	FP       CDF
	Sign     CDF
	Class0HP CDF
	HP       CDF
	Class0   CDF
	Bits     [10]CDF
}

// mvBool is dav1d's msac_decode_bool_adapt with the range state in
// registers. It is written out at each call site of the MV reads below so
// that dif, rng and cnt stay in registers between the reads.
//
//	bit, dif, rng = coefBool(dif, rng, cdfBound(rng>>8, uint32(v[0]), ecMinProb))
//	dif, rng, cnt = coefNorm(dif, rng, cnt)
//	if cnt < 0 { dif, cnt = c.coefRefill(dif, cnt) }
//	if update { updateCDF2(v, int(bit)) }

// DecodeMVResidual is dav1d's read_mv_residual (src/decode.c) with both
// read_mv_component_diff calls, the range state in registers from the joint
// to the last high-precision bit. mvPrec is allow_high_precision_mv -
// force_integer_mv (-1, 0 or 1). It returns the residual added to row and col.
func (c *Cursor) DecodeMVResidual(joint *CDF, rowCDFs, colCDFs *MVComponentCDFs, row, col int32, mvPrec int) (int32, int32) {
	update := c.allowCDFUpdate
	dif, rng, cnt := c.dif, uint32(c.rng), int32(c.cnt)
	v := &joint.values
	if traceEntropyReads {
		traceCDFRead(v[0], 4, dif, rng, c.coefTell(cnt))
	}
	rngHi := rng >> 8
	coded := uint32(dif >> (ecWindow - 16))
	c0, c1, c2 := uint32(v[0]), uint32(v[1]), uint32(v[2])
	j, lower, upper := selectSymbol4(rng, coded, cdfBound(rngHi, c0, 3*ecMinProb), cdfBound(rngHi, c1, 2*ecMinProb), cdfBound(rngHi, c2, ecMinProb))
	dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
	if cnt < 0 {
		dif, cnt = c.coefRefill(dif, cnt)
	}
	if update {
		count := v[4]
		rate := uint(5 + ((count >> 4) & 3))
		v[0] = uint16(adaptCDFEntry(c0, rate, j > 0))
		v[1] = uint16(adaptCDFEntry(c1, rate, j > 1))
		v[2] = uint16(adaptCDFEntry(c2, rate, j > 2))
		if count < MaxCDFCount {
			v[4] = count + 1
		}
	}
	if j&2 != 0 { // MV_JOINT_V
		var d int32
		d, dif, rng, cnt = c.mvComponentDiff(rowCDFs, mvPrec, update, dif, rng, cnt)
		row += d
	}
	if j&1 != 0 { // MV_JOINT_H
		var d int32
		d, dif, rng, cnt = c.mvComponentDiff(colCDFs, mvPrec, update, dif, rng, cnt)
		col += d
	}
	c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
	return row, col
}

// mvComponentDiff is dav1d's read_mv_component_diff with the range state in
// registers.
func (c *Cursor) mvComponentDiff(comp *MVComponentCDFs, mvPrec int, update bool, dif uint64, rng uint32, cnt int32) (int32, uint64, uint32, int32) {
	var sign uint
	v := &comp.Sign.values
	if traceEntropyReads {
		traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
	}
	sign, dif, rng = coefBool(dif, rng, cdfBound(rng>>8, uint32(v[0]), ecMinProb))
	dif, rng, cnt = coefNorm(dif, rng, cnt)
	if cnt < 0 {
		dif, cnt = c.coefRefill(dif, cnt)
	}
	if update {
		updateCDF2(v, int(sign))
	}

	v = &comp.Classes.values
	if traceEntropyReads {
		traceCDFRead(v[0], 11, dif, rng, c.coefTell(cnt))
	}
	cl, lower, upper := coefSymbol(v, 11, rng, uint32(dif>>(ecWindow-16)), update)
	dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
	if cnt < 0 {
		dif, cnt = c.coefRefill(dif, cnt)
	}

	var up, bit uint
	fp, hp := uint(3), uint(1)
	var fpCDF, hpCDF *[MaxSymbols + 1]uint16
	if cl == 0 {
		v = &comp.Class0.values
		if traceEntropyReads {
			traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
		}
		up, dif, rng = coefBool(dif, rng, cdfBound(rng>>8, uint32(v[0]), ecMinProb))
		dif, rng, cnt = coefNorm(dif, rng, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			updateCDF2(v, int(up))
		}
		fpCDF, hpCDF = &comp.Class0FP[up&1].values, &comp.Class0HP.values
	} else {
		up = 1 << uint(cl)
		for n := range uint(cl) & 15 {
			v = &comp.Bits[n].values
			if traceEntropyReads {
				traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
			}
			bit, dif, rng = coefBool(dif, rng, cdfBound(rng>>8, uint32(v[0]), ecMinProb))
			dif, rng, cnt = coefNorm(dif, rng, cnt)
			if cnt < 0 {
				dif, cnt = c.coefRefill(dif, cnt)
			}
			if update {
				updateCDF2(v, int(bit))
			}
			up |= bit << n
		}
		fpCDF, hpCDF = &comp.FP.values, &comp.HP.values
	}
	if mvPrec >= 0 {
		v = fpCDF
		if traceEntropyReads {
			traceCDFRead(v[0], 4, dif, rng, c.coefTell(cnt))
		}
		rngHi := rng >> 8
		c0, c1, c2 := uint32(v[0]), uint32(v[1]), uint32(v[2])
		fp, lower, upper = selectSymbol4(rng, uint32(dif>>(ecWindow-16)), cdfBound(rngHi, c0, 3*ecMinProb), cdfBound(rngHi, c1, 2*ecMinProb), cdfBound(rngHi, c2, ecMinProb))
		dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			count := v[4]
			rate := uint(5 + ((count >> 4) & 3))
			v[0] = uint16(adaptCDFEntry(c0, rate, fp > 0))
			v[1] = uint16(adaptCDFEntry(c1, rate, fp > 1))
			v[2] = uint16(adaptCDFEntry(c2, rate, fp > 2))
			if count < MaxCDFCount {
				v[4] = count + 1
			}
		}
		if mvPrec > 0 {
			v = hpCDF
			if traceEntropyReads {
				traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
			}
			hp, dif, rng = coefBool(dif, rng, cdfBound(rng>>8, uint32(v[0]), ecMinProb))
			dif, rng, cnt = coefNorm(dif, rng, cnt)
			if cnt < 0 {
				dif, cnt = c.coefRefill(dif, cnt)
			}
			if update {
				updateCDF2(v, int(hp))
			}
		}
	}
	// sign ? -diff : diff without a branch on the sign bit.
	s := -int32(sign)
	return (int32(up<<3|fp<<1|hp) + 1 ^ s) - s, dif, rng, cnt
}
