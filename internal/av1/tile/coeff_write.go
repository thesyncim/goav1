package tile

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// coeff_write.go is the forward of ReadCoefficientsTXB: it codes one transform
// block's quantized coefficients into the bitstream. Source-shaped port of libaom
// av1_write_coeffs_txb (av1/encoder/encodetxb.c) plus write_golomb and
// av1_get_eob_pos_token. It shares this package's coefficient context derivation
// (CoeffLowerLevelsContext, CoeffBRContext, transform.LowerLevelsCtxEOB) and CDF
// state with the decoder, exactly as libaom shares txb_common between enc and dec.

// The EOB position-token decomposition (av1_get_eob_pos_token) is shared with the
// decoder as EOBPositionToken; the eob_to_pos / group-start / offset-bit tables
// live in coeff.go.

// writeGolomb codes a non-negative coefficient-level tail with the AV1 Exp-Golomb
// code as equiprobable bits, the exact inverse of readCoeffGolombCursor. Port of
// write_golomb (encodetxb.c).
func writeGolomb(w *entropy.Writer, level int) {
	x := level + 1
	length := 0
	for i := x; i != 0; i >>= 1 {
		length++
	}
	for range length - 1 {
		w.WriteBit(0)
	}
	for i := length - 1; i >= 0; i-- {
		w.WriteBit((x >> uint(i)) & 1)
	}
}

func writeGolombCounter(w *entropy.BitCounter, level int) {
	x := level + 1
	length := 0
	for i := x; i != 0; i >>= 1 {
		length++
	}
	for range length - 1 {
		w.WriteBit(0)
	}
	for i := length - 1; i >= 0; i-- {
		w.WriteBit((x >> uint(i)) & 1)
	}
}

// TXBEncodeRequest carries the per-block inputs the coefficient writer needs that
// it cannot derive from the coefficients alone, mirroring the corresponding
// fields of TXBDecodeRequest.
type TXBEncodeRequest struct {
	Size           TransformSize
	Plane          CoeffPlaneType
	Class          transform.Class
	TXBSkipContext uint8
	DCSignContext  uint8

	// AfterSkip, when non-nil, runs after the txb_skip symbol for non-skipped
	// blocks and before the eob token — the position av1_write_coeffs_txb
	// emits the luma tx_type symbol (and the decoder's coefficient path
	// invokes its transform-type selector).
	AfterSkip func() error
}

// WriteCoefficientsTXB codes the quantized coefficients of one transform block
// (coeffs in raster order, addressed through scan) into w, the exact inverse of
// ReadCoefficientsTXB. levels is caller-owned scratch of at least
// CoeffLevelsScratchLen(req.Size). cdfs is adapted in place, matching a decoder
// with CDF update enabled. Port of av1_write_coeffs_txb.
func WriteCoefficientsTXB(w *entropy.Writer, cdfs *CoeffCDFs, req TXBEncodeRequest, coeffs []int16, scan []int16, levels []uint8) (TXBDecodeResult, error) {
	if w == nil || cdfs == nil || !req.Plane.Valid() || !req.Class.Valid() {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	geo, ok := coeffGeoPtr(req.Size)
	if !ok {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	maxEOB := int(geo.maxEOB)
	scratchLen := int(geo.scratchLen)
	if len(coeffs) < maxEOB || len(scan) < maxEOB || len(levels) < scratchLen {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	txSize, err := req.Size.TransformSize()
	if err != nil {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	scanSize, err := transform.ScanSize(txSize)
	if err != nil {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	posTable := coeffPosTable[req.Size]
	if len(posTable) < maxEOB {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}

	// eob = scan index of the last non-zero coefficient plus one.
	eob := 0
	for c := maxEOB - 1; c >= 0; c-- {
		if pos := int(scan[c]); coeffs[pos] != 0 {
			eob = c + 1
			break
		}
	}

	txCtx := int(geo.txCtx)
	txBR := int(geo.txBRCtx)

	// 1) txb_skip (all_zero).
	w.WriteBinaryCDFTrusted(&cdfs.TXBSkip[txCtx][req.TXBSkipContext], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}, nil
	}
	if req.AfterSkip != nil {
		// Luma tx_type position (av1_write_coeffs_txb: after txb_skip, before
		// the eob token).
		if err := req.AfterSkip(); err != nil {
			return TXBDecodeResult{}, err
		}
	}

	// 2) eob position token + extra offset bits.
	eobMultiCtx := uint8(0)
	if req.Class != transform.Class2D {
		eobMultiCtx = 1
	}
	eobCDF := eobFlagCDFKnown(cdfs, req.Size, req.Plane, eobMultiCtx)
	token, extra, err := EOBPositionToken(eob)
	if err != nil {
		return TXBDecodeResult{}, err
	}
	if eobMultiSizeTable[req.Size] == 0 {
		w.WriteCDF5(eobCDF, token-1)
	} else if eobMultiSizeTable[req.Size] == 2 && req.Plane == CoeffPlaneY {
		w.WriteCDF7(eobCDF, token-1)
	} else {
		w.WriteCDF(eobCDF, token-1)
	}
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteBinaryCDFTrusted(&cdfs.EOBExtra[txCtx][req.Plane][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	// Build the level magnitude buffer (clamped to INT8_MAX, libaom
	// av1_txb_init_levels) used by the base/br context derivation. The context
	// template only reads already-coded (higher-raster) neighbours, so deriving
	// every context from the full buffer matches the decoder's incremental one.
	clear(levels[:scratchLen])
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		pos := int(scan[c])
		cv := coeffs[pos]
		levels[int(posTable[pos].padded)] = coeffAbsClamp127(cv)
		if cv != 0 {
			if pos > maxScanLine {
				maxScanLine = pos
			}
			level := absInt(int(cv))
			culLevel += level
			if c == 0 {
				dcValue = int(cv)
			}
		}
	}

	// 3) lower-magnitude level pass, reversed scan order. The 2D class (the
	// only one DCT_DCT uses) derives base and br contexts inline from the
	// precomputed position table - the same lower2DOffset/br2DOffset entries
	// the decoder's hot read path consumes - instead of re-deriving geometry
	// per coefficient.
	stride := int(geo.stride)
	baseCDFs := &cdfs.CoeffBase[txCtx][req.Plane]
	brCDFs := &cdfs.CoeffBR[txBR][req.Plane]
	for c := eob - 1; c >= 0; c-- {
		pos := int(scan[c])
		level := absInt(int(coeffs[pos]))
		if c == eob-1 {
			ctx, err := transform.LowerLevelsCtxEOB(scanSize, c)
			if err != nil {
				return TXBDecodeResult{}, ErrInvalidDecodeState
			}
			w.WriteCDF3(&cdfs.CoeffBaseEOB[txCtx][req.Plane][ctx], minInt(level, 3)-1)
		} else if req.Class == transform.Class2D {
			p := &posTable[pos]
			ctx := 0
			if pos != 0 {
				pad := int(p.padded)
				mag := clipMax3(levels[pad+stride]) + clipMax3(levels[pad+1]) +
					clipMax3(levels[pad+stride+1]) + clipMax3(levels[pad+(stride<<1)]) + clipMax3(levels[pad+2])
				ctx = minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
			}
			w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
		} else {
			ctx, err := CoeffLowerLevelsContext(levels, req.Size, req.Class, pos)
			if err != nil {
				return TXBDecodeResult{}, err
			}
			w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
		}
		if level > NumBaseLevels {
			var brCtx int
			if c == eob-1 {
				brCtx, err = CoeffBRContextEOB(req.Size, req.Class, pos)
				if err != nil {
					return TXBDecodeResult{}, err
				}
			} else if req.Class == transform.Class2D && pos != 0 {
				p := &posTable[pos]
				pad := int(p.padded)
				mag := minInt(int(levels[pad+1]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride+1]), MaxBaseBRRange)
				brCtx = minInt((mag+1)>>1, 6) + int(p.br2DOffset)
			} else {
				brCtx, err = CoeffBRContext(levels, req.Size, req.Class, pos)
				if err != nil {
					return TXBDecodeResult{}, err
				}
			}
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	// 4) sign + golomb pass, forward scan order, non-zero coefficients only.
	for c := range eob {
		pos := int(scan[c])
		cv := coeffs[pos]
		if cv == 0 {
			continue
		}
		v := int(cv)
		level := absInt(v)
		sign := int(uint16(cv) >> 15)
		if c == 0 {
			w.WriteBinaryCDFTrusted(&cdfs.DCSign[req.Plane][req.DCSignContext], sign)
		} else {
			w.WriteBit(sign)
		}
		if level >= MaxBaseBRRange {
			writeGolomb(w, level-MaxBaseBRRange)
		}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: uint16(maxScanLine),
		CulLevel:    uint8(culLevel),
	}, nil
}

// WriteCoefficientsTXB8x8Y2DTrusted is the validation-free 8x8 luma/Class2D
// specialization of WriteCoefficientsTXB. It assumes coeffs has 64 entries,
// the default 2D scan is used, txb_skip/dc_sign contexts are zero, and CDF
// updates are enabled. The levels argument is retained for source compatibility
// with the generic writer shape; this hot specialization uses stack scratch.
// If txCDF is non-nil, txSymbol is written after txb_skip and txCDF is restored
// to its input state, matching transform-type trials that price but do not
// adapt the real transform CDFs.
func WriteCoefficientsTXB8x8Y2DTrusted(w *entropy.Writer, cdfs *CoeffCDFs, coeffs []int16, _ []uint8, txCDF *entropy.CDF, txSymbol int) TXBDecodeResult {
	return WriteCoefficientsTXB8x8Y2DTrustedArray(w, cdfs, (*[64]int16)(coeffs), txCDF, txSymbol)
}

// WriteCoefficientsTXB8x8Y2DTrustedArray is the pointer-shaped form for hot
// callers that already own 64-coefficient 8x8 scratch.
func WriteCoefficientsTXB8x8Y2DTrustedArray(w *entropy.Writer, cdfs *CoeffCDFs, coeffs *[64]int16, txCDF *entropy.CDF, txSymbol int) TXBDecodeResult {
	return writeCoefficientsTXB8x8Y2DTrustedArray(w, cdfs, coeffs, 0, 0, txCDF, txSymbol, true)
}

// CountCoefficientsTXB8x8Y2DTrusted is the exact output-free rate-pricing
// variant of WriteCoefficientsTXB8x8Y2DTrusted. It adapts cdfs identically,
// restores txCDF after pricing, and returns the same Tell delta that a byte
// writer would observe.
func CountCoefficientsTXB8x8Y2DTrusted(cdfs *CoeffCDFs, coeffs []int16, txCDF *entropy.CDF, txSymbol int) (TXBDecodeResult, int) {
	return CountCoefficientsTXB8x8Y2DTrustedArray(cdfs, (*[64]int16)(coeffs), txCDF, txSymbol)
}

// CountCoefficientsTXB8x8Y2DTrustedArray is CountCoefficientsTXB8x8Y2DTrusted
// for callers that can prove the 64-coefficient shape before the hot call.
func CountCoefficientsTXB8x8Y2DTrustedArray(cdfs *CoeffCDFs, coeff64 *[64]int16, txCDF *entropy.CDF, txSymbol int) (TXBDecodeResult, int) {
	const (
		maxEOB = 64
		stride = uint8(12)
		txCtx  = 1
		txBR   = 1
	)
	scanHot := &coeffScanHot8x8Y2D
	w := entropy.NewBitCounter()
	base := w.Tell()

	eob := 0
	for c := maxEOB - 1; c >= 0; c-- {
		if coeff64[scanHot[c].pos] != 0 {
			eob = c + 1
			break
		}
	}

	w.WriteBinaryCDFTrusted(&cdfs.TXBSkip[txCtx][0], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}, w.Tell() - base
	}
	if txCDF != nil {
		saved := *txCDF
		w.WriteCDF(txCDF, txSymbol)
		*txCDF = saved
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF7(&cdfs.EOBFlag64[CoeffPlaneY][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteBinaryCDFTrusted(&cdfs.EOBExtra[txCtx][CoeffPlaneY][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [256]uint8
	var absLevels [64]uint16
	prep := transform.PrepTXB8x8Levels2D(coeff64, &levels, &absLevels, eob)

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneY]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneY]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneY]

	c := eob - 1
	p := &scanHot[c]
	level := int(absLevels[c])
	w.WriteCDF3(&baseEOBCDFs[int(p.lowerEOBCtx)], minInt(level, 3)-1)
	if level > NumBaseLevels {
		brCDF := &brCDFs[int(p.brEOBCtx)]
		baseRange := level - 1 - NumBaseLevels
		for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
			k := minInt(baseRange-idx, BRCDFSize-1)
			w.WriteCDF4(brCDF, k)
			if k < BRCDFSize-1 {
				break
			}
		}
	}
	for c = eob - 2; c > 0; c-- {
		p := &scanHot[c]
		level := int(absLevels[c])
		pad := p.padded
		mag := clipMax3(levels[pad+stride]) + clipMax3(levels[pad+1]) +
			clipMax3(levels[pad+stride+1]) + clipMax3(levels[pad+(stride<<1)]) + clipMax3(levels[pad+2])
		ctx := minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
		if level == 0 {
			w.WriteCDF4Zero(&baseCDFs[ctx])
		} else {
			w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
		}
		if level > NumBaseLevels {
			mag := minInt(int(levels[pad+1]), MaxBaseBRRange) +
				minInt(int(levels[pad+stride]), MaxBaseBRRange) +
				minInt(int(levels[pad+stride+1]), MaxBaseBRRange)
			brCtx := minInt((mag+1)>>1, 6) + int(p.br2DOffset)
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}
	if eob > 1 {
		p := &scanHot[0]
		level := int(absLevels[0])
		if level == 0 {
			w.WriteCDF4Zero(&baseCDFs[0])
		} else {
			w.WriteCDF4(&baseCDFs[0], minInt(level, 3))
		}
		if level > NumBaseLevels {
			pad := p.padded
			mag := int(levels[pad+1]) + int(levels[pad+stride]) + int(levels[pad+stride+1])
			brCtx := minInt((mag+1)>>1, 6)
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	for nz := prep.NonZeroBits; nz != 0; nz &= nz - 1 {
		c := bits.TrailingZeros64(nz)
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		sign := int((prep.SignBits >> uint(c)) & 1)
		if pos == 0 {
			w.WriteBinaryCDFTrusted(&cdfs.DCSign[CoeffPlaneY][0], sign)
		} else {
			w.WriteBit(sign)
		}
		if level >= MaxBaseBRRange {
			writeGolombCounter(&w, level-MaxBaseBRRange)
		}
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: prep.MaxScanLine,
		CulLevel:    prep.CulLevel,
	}, w.Tell() - base
}

// CountCoefficientsTXB8x8UV2DTrusted is the exact output-free rate-pricing
// specialization for 8x8 chroma/Class2D trials with zero txb_skip/dc_sign
// contexts. It adapts cdfs identically to WriteCoefficientsTXB and returns the
// same Tell delta that a byte writer would observe.
func CountCoefficientsTXB8x8UV2DTrusted(cdfs *CoeffCDFs, coeffs []int16) (TXBDecodeResult, int) {
	return CountCoefficientsTXB8x8UV2DTrustedArray(cdfs, (*[64]int16)(coeffs))
}

// CountCoefficientsTXB8x8UV2DTrustedArray is
// CountCoefficientsTXB8x8UV2DTrusted for callers that can prove the
// 64-coefficient shape before the hot call.
func CountCoefficientsTXB8x8UV2DTrustedArray(cdfs *CoeffCDFs, coeff64 *[64]int16) (TXBDecodeResult, int) {
	const (
		maxEOB = 64
		stride = uint8(12)
		txCtx  = 1
		txBR   = 1
	)
	scanHot := &coeffScanHot8x8Y2D
	w := entropy.NewBitCounter()
	base := w.Tell()

	eob := 0
	for c := maxEOB - 1; c >= 0; c-- {
		if coeff64[scanHot[c].pos] != 0 {
			eob = c + 1
			break
		}
	}

	w.WriteBinaryCDFTrusted(&cdfs.TXBSkip[txCtx][0], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}, w.Tell() - base
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF(&cdfs.EOBFlag64[CoeffPlaneUV][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteBinaryCDFTrusted(&cdfs.EOBExtra[txCtx][CoeffPlaneUV][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [256]uint8
	var absLevels [64]uint16
	var nonZeroBits uint64
	var signBits uint64
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff64[pos]
		if cv == 0 {
			continue
		}
		levels[p.padded] = coeffAbsClamp127(cv)
		if pos > maxScanLine {
			maxScanLine = pos
		}
		level := absInt(int(cv))
		absLevels[c] = uint16(level)
		nonZeroBits |= 1 << uint(c)
		if cv < 0 {
			signBits |= 1 << uint(c)
		}
		culLevel += level
		if pos == 0 {
			dcValue = int(cv)
		}
	}

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneUV]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneUV]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneUV]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF3(&baseEOBCDFs[ctx], minInt(level, 3)-1)
		} else {
			ctx := 0
			if pos != 0 {
				pad := p.padded
				mag := clipMax3(levels[pad+stride]) + clipMax3(levels[pad+1]) +
					clipMax3(levels[pad+stride+1]) + clipMax3(levels[pad+(stride<<1)]) + clipMax3(levels[pad+2])
				ctx = minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
			}
			if level == 0 {
				w.WriteCDF4Zero(&baseCDFs[ctx])
			} else {
				w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
			}
		}
		if level > NumBaseLevels {
			brCtx := 0
			if c == eob-1 {
				brCtx = int(p.brEOBCtx)
			} else if pos != 0 {
				pad := p.padded
				mag := minInt(int(levels[pad+1]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride+1]), MaxBaseBRRange)
				brCtx = minInt((mag+1)>>1, 6) + int(p.br2DOffset)
			} else {
				pad := p.padded
				mag := int(levels[pad+1]) + int(levels[pad+stride]) + int(levels[pad+stride+1])
				brCtx = minInt((mag+1)>>1, 6)
			}
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	for nz := nonZeroBits; nz != 0; nz &= nz - 1 {
		c := bits.TrailingZeros64(nz)
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		sign := int((signBits >> uint(c)) & 1)
		if pos == 0 {
			w.WriteBinaryCDFTrusted(&cdfs.DCSign[CoeffPlaneUV][0], sign)
		} else {
			w.WriteBit(sign)
		}
		if level >= MaxBaseBRRange {
			writeGolombCounter(&w, level-MaxBaseBRRange)
		}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: uint16(maxScanLine),
		CulLevel:    uint8(culLevel),
	}, w.Tell() - base
}

// CountCoefficientsTXB16x16Y2DTrusted is the exact output-free rate-pricing
// specialization for 16x16 luma/Class2D trials with zero txb_skip/dc_sign
// contexts. It adapts cdfs identically to WriteCoefficientsTXB and returns the
// same Tell delta that a byte writer would observe.
func CountCoefficientsTXB16x16Y2DTrusted(cdfs *CoeffCDFs, coeffs []int16) (TXBDecodeResult, int) {
	return countCoefficientsTXB16x16Y2DTrustedArray(cdfs, (*[256]int16)(coeffs), nil, 0)
}

// CountCoefficientsTXB16x16Y2DTrustedWithTXType is the inter trial-pricing
// variant: it writes the already-derived transform-type symbol after txb_skip,
// then restores txCDF so trial pricing does not adapt the real transform CDFs.
func CountCoefficientsTXB16x16Y2DTrustedWithTXType(cdfs *CoeffCDFs, coeffs []int16, txCDF *entropy.CDF, txSymbol int) (TXBDecodeResult, int) {
	return countCoefficientsTXB16x16Y2DTrustedArray(cdfs, (*[256]int16)(coeffs), txCDF, txSymbol)
}

// CountCoefficientsTXB16x16Y2DTrustedArray is
// CountCoefficientsTXB16x16Y2DTrusted for callers that can prove the
// 256-coefficient shape before the hot call.
func CountCoefficientsTXB16x16Y2DTrustedArray(cdfs *CoeffCDFs, coeff256 *[256]int16) (TXBDecodeResult, int) {
	return countCoefficientsTXB16x16Y2DTrustedArray(cdfs, coeff256, nil, 0)
}

// CountCoefficientsTXB16x16Y2DTrustedWithTXTypeArray is the pointer-shaped
// form for inter transform-type trial pricing.
func CountCoefficientsTXB16x16Y2DTrustedWithTXTypeArray(cdfs *CoeffCDFs, coeff256 *[256]int16, txCDF *entropy.CDF, txSymbol int) (TXBDecodeResult, int) {
	return countCoefficientsTXB16x16Y2DTrustedArray(cdfs, coeff256, txCDF, txSymbol)
}

func countCoefficientsTXB16x16Y2DTrustedArray(cdfs *CoeffCDFs, coeff256 *[256]int16, txCDF *entropy.CDF, txSymbol int) (TXBDecodeResult, int) {
	const (
		maxEOB     = 256
		scratchLen = 400
		stride     = uint16(20)
		txCtx      = 2
		txBR       = 2
	)
	scanHot := &coeffScanHot16x16Y2D
	w := entropy.NewBitCounter()
	base := w.Tell()

	eob := 0
	for c := maxEOB - 1; c >= 0; c-- {
		if coeff256[scanHot[c].pos] != 0 {
			eob = c + 1
			break
		}
	}

	w.WriteBinaryCDFTrusted(&cdfs.TXBSkip[txCtx][0], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}, w.Tell() - base
	}
	if txCDF != nil {
		saved := *txCDF
		w.WriteCDF(txCDF, txSymbol)
		*txCDF = saved
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF(&cdfs.EOBFlag256[CoeffPlaneY][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteBinaryCDFTrusted(&cdfs.EOBExtra[txCtx][CoeffPlaneY][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [scratchLen]uint8
	var absLevels [maxEOB]uint16
	var nonZeroBits [4]uint64
	var signBits [4]uint64
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff256[pos]
		if cv == 0 {
			continue
		}
		levels[p.padded] = coeffAbsClamp127(cv)
		if pos > maxScanLine {
			maxScanLine = pos
		}
		level := absInt(int(cv))
		absLevels[c] = uint16(level)
		word := c >> 6
		bit := uint(c & 63)
		nonZeroBits[word] |= 1 << bit
		if cv < 0 {
			signBits[word] |= 1 << bit
		}
		culLevel += level
		if pos == 0 {
			dcValue = int(cv)
		}
	}
	var lowerContexts [maxEOB]int8
	lowerContextMap := coeffNZMapContexts2DFullArch(levels[:], TransformSize16x16, lowerContexts[:])

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneY]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneY]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneY]

	c := eob - 1
	p := &scanHot[c]
	level := int(absLevels[c])
	w.WriteCDF3(&baseEOBCDFs[int(p.lowerEOBCtx)], minInt(level, 3)-1)
	if level > NumBaseLevels {
		brCDF := &brCDFs[int(p.brEOBCtx)]
		baseRange := level - 1 - NumBaseLevels
		for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
			k := minInt(baseRange-idx, BRCDFSize-1)
			w.WriteCDF4(brCDF, k)
			if k < BRCDFSize-1 {
				break
			}
		}
	}
	for c = eob - 2; c > 0; c-- {
		p := &scanHot[c]
		level := int(absLevels[c])
		pad := p.padded
		ctx := int(lowerContexts[p.pos])
		if !lowerContextMap {
			mag := clipMax3(levels[pad+stride]) + clipMax3(levels[pad+1]) +
				clipMax3(levels[pad+stride+1]) + clipMax3(levels[pad+(stride<<1)]) + clipMax3(levels[pad+2])
			ctx = minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
		}
		if level == 0 {
			w.WriteCDF4Zero(&baseCDFs[ctx])
		} else {
			w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
		}
		if level > NumBaseLevels {
			mag := minInt(int(levels[pad+1]), MaxBaseBRRange) +
				minInt(int(levels[pad+stride]), MaxBaseBRRange) +
				minInt(int(levels[pad+stride+1]), MaxBaseBRRange)
			brCtx := minInt((mag+1)>>1, 6) + int(p.br2DOffset)
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}
	if eob > 1 {
		p := &scanHot[0]
		level := int(absLevels[0])
		if level == 0 {
			w.WriteCDF4Zero(&baseCDFs[0])
		} else {
			w.WriteCDF4(&baseCDFs[0], minInt(level, 3))
		}
		if level > NumBaseLevels {
			pad := p.padded
			mag := int(levels[pad+1]) + int(levels[pad+stride]) + int(levels[pad+stride+1])
			brCtx := minInt((mag+1)>>1, 6)
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	for word, bits64 := range nonZeroBits {
		for nz := bits64; nz != 0; nz &= nz - 1 {
			c := word*64 + bits.TrailingZeros64(nz)
			p := &scanHot[c]
			pos := int(p.pos)
			level := int(absLevels[c])
			sign := int((signBits[word] >> uint(c&63)) & 1)
			if pos == 0 {
				w.WriteBinaryCDFTrusted(&cdfs.DCSign[CoeffPlaneY][0], sign)
			} else {
				w.WriteBit(sign)
			}
			if level >= MaxBaseBRRange {
				writeGolombCounter(&w, level-MaxBaseBRRange)
			}
		}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: uint16(maxScanLine),
		CulLevel:    uint8(culLevel),
	}, w.Tell() - base
}

// CountCoefficientsTXB16x16UV2DTrusted is the exact output-free rate-pricing
// specialization for 16x16 chroma/Class2D trials with zero txb_skip/dc_sign
// contexts. It adapts cdfs identically to WriteCoefficientsTXB and returns the
// same Tell delta that a byte writer would observe.
func CountCoefficientsTXB16x16UV2DTrusted(cdfs *CoeffCDFs, coeffs []int16) (TXBDecodeResult, int) {
	return CountCoefficientsTXB16x16UV2DTrustedArray(cdfs, (*[256]int16)(coeffs))
}

// CountCoefficientsTXB16x16UV2DTrustedArray is
// CountCoefficientsTXB16x16UV2DTrusted for callers that can prove the
// 256-coefficient shape before the hot call.
func CountCoefficientsTXB16x16UV2DTrustedArray(cdfs *CoeffCDFs, coeff256 *[256]int16) (TXBDecodeResult, int) {
	const (
		maxEOB     = 256
		scratchLen = 400
		stride     = uint16(20)
		txCtx      = 2
		txBR       = 2
	)
	scanHot := &coeffScanHot16x16Y2D
	w := entropy.NewBitCounter()
	base := w.Tell()

	eob := 0
	for c := maxEOB - 1; c >= 0; c-- {
		if coeff256[scanHot[c].pos] != 0 {
			eob = c + 1
			break
		}
	}

	w.WriteBinaryCDFTrusted(&cdfs.TXBSkip[txCtx][0], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}, w.Tell() - base
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF(&cdfs.EOBFlag256[CoeffPlaneUV][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteBinaryCDFTrusted(&cdfs.EOBExtra[txCtx][CoeffPlaneUV][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [scratchLen]uint8
	var absLevels [maxEOB]uint16
	var nonZeroBits [4]uint64
	var signBits [4]uint64
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff256[pos]
		if cv == 0 {
			continue
		}
		levels[p.padded] = coeffAbsClamp127(cv)
		if pos > maxScanLine {
			maxScanLine = pos
		}
		level := absInt(int(cv))
		absLevels[c] = uint16(level)
		word := c >> 6
		bit := uint(c & 63)
		nonZeroBits[word] |= 1 << bit
		if cv < 0 {
			signBits[word] |= 1 << bit
		}
		culLevel += level
		if pos == 0 {
			dcValue = int(cv)
		}
	}
	var lowerContexts [maxEOB]int8
	lowerContextMap := coeffNZMapContexts2DFullArch(levels[:], TransformSize16x16, lowerContexts[:])

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneUV]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneUV]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneUV]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF3(&baseEOBCDFs[ctx], minInt(level, 3)-1)
		} else {
			ctx := 0
			if pos != 0 {
				ctx = int(lowerContexts[pos])
				if !lowerContextMap {
					pad := p.padded
					mag := clipMax3(levels[pad+stride]) + clipMax3(levels[pad+1]) +
						clipMax3(levels[pad+stride+1]) + clipMax3(levels[pad+(stride<<1)]) + clipMax3(levels[pad+2])
					ctx = minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
				}
			}
			if level == 0 {
				w.WriteCDF4Zero(&baseCDFs[ctx])
			} else {
				w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
			}
		}
		if level > NumBaseLevels {
			brCtx := 0
			if c == eob-1 {
				brCtx = int(p.brEOBCtx)
			} else if pos != 0 {
				pad := p.padded
				mag := minInt(int(levels[pad+1]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride+1]), MaxBaseBRRange)
				brCtx = minInt((mag+1)>>1, 6) + int(p.br2DOffset)
			} else {
				pad := p.padded
				mag := int(levels[pad+1]) + int(levels[pad+stride]) + int(levels[pad+stride+1])
				brCtx = minInt((mag+1)>>1, 6)
			}
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	for word, bits64 := range nonZeroBits {
		for nz := bits64; nz != 0; nz &= nz - 1 {
			c := word*64 + bits.TrailingZeros64(nz)
			p := &scanHot[c]
			pos := int(p.pos)
			level := int(absLevels[c])
			sign := int((signBits[word] >> uint(c&63)) & 1)
			if pos == 0 {
				w.WriteBinaryCDFTrusted(&cdfs.DCSign[CoeffPlaneUV][0], sign)
			} else {
				w.WriteBit(sign)
			}
			if level >= MaxBaseBRRange {
				writeGolombCounter(&w, level-MaxBaseBRRange)
			}
		}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: uint16(maxScanLine),
		CulLevel:    uint8(culLevel),
	}, w.Tell() - base
}

// WriteCoefficientsTXB16x16Y2DContextTrusted is the validation-free 16x16
// luma/Class2D specialization for real coding with already-derived coefficient
// contexts. If txCDF is non-nil, txSymbol is written after txb_skip and txCDF
// is adapted in place.
func WriteCoefficientsTXB16x16Y2DContextTrusted(w *entropy.Writer, cdfs *CoeffCDFs, coeffs []int16, _ []uint8, txbSkipContext, dcSignContext uint8, txCDF *entropy.CDF, txSymbol int) TXBDecodeResult {
	return WriteCoefficientsTXB16x16Y2DContextTrustedArray(w, cdfs, (*[256]int16)(coeffs), txbSkipContext, dcSignContext, txCDF, txSymbol)
}

// WriteCoefficientsTXB16x16Y2DContextTrustedArray is the pointer-shaped form
// for real 16x16 luma coding with already-derived coefficient contexts.
func WriteCoefficientsTXB16x16Y2DContextTrustedArray(w *entropy.Writer, cdfs *CoeffCDFs, coeffs *[256]int16, txbSkipContext, dcSignContext uint8, txCDF *entropy.CDF, txSymbol int) TXBDecodeResult {
	return writeCoefficientsTXB16x16Plane2DContextTrustedArray(w, cdfs, coeffs, CoeffPlaneY, txbSkipContext, dcSignContext, txCDF, txSymbol)
}

// WriteCoefficientsTXB16x16UV2DContextTrusted is the validation-free 16x16
// chroma/Class2D specialization for real coding with already-derived
// coefficient contexts.
func WriteCoefficientsTXB16x16UV2DContextTrusted(w *entropy.Writer, cdfs *CoeffCDFs, coeffs []int16, _ []uint8, txbSkipContext, dcSignContext uint8) TXBDecodeResult {
	return WriteCoefficientsTXB16x16UV2DContextTrustedArray(w, cdfs, (*[256]int16)(coeffs), txbSkipContext, dcSignContext)
}

// WriteCoefficientsTXB16x16UV2DContextTrustedArray is the pointer-shaped form
// for real 16x16 chroma coding with already-derived coefficient contexts.
func WriteCoefficientsTXB16x16UV2DContextTrustedArray(w *entropy.Writer, cdfs *CoeffCDFs, coeffs *[256]int16, txbSkipContext, dcSignContext uint8) TXBDecodeResult {
	return writeCoefficientsTXB16x16Plane2DContextTrustedArray(w, cdfs, coeffs, CoeffPlaneUV, txbSkipContext, dcSignContext, nil, 0)
}

func writeCoefficientsTXB16x16Plane2DContextTrustedArray(w *entropy.Writer, cdfs *CoeffCDFs, coeff256 *[256]int16, plane CoeffPlaneType, txbSkipContext, dcSignContext uint8, txCDF *entropy.CDF, txSymbol int) TXBDecodeResult {
	const (
		maxEOB     = 256
		scratchLen = 400
		stride     = uint16(20)
		txCtx      = 2
		txBR       = 2
	)
	scanHot := &coeffScanHot16x16Y2D

	eob := 0
	for c := maxEOB - 1; c >= 0; c-- {
		if coeff256[scanHot[c].pos] != 0 {
			eob = c + 1
			break
		}
	}

	w.WriteBinaryCDFTrusted(&cdfs.TXBSkip[txCtx][txbSkipContext], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}
	}
	if txCDF != nil {
		w.WriteCDF(txCDF, txSymbol)
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF(&cdfs.EOBFlag256[plane][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteBinaryCDFTrusted(&cdfs.EOBExtra[txCtx][plane][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [scratchLen]uint8
	var absLevels [maxEOB]uint16
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	var nonZeroBits [4]uint64
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff256[pos]
		if cv == 0 {
			continue
		}
		levels[p.padded] = coeffAbsClamp127(cv)
		if pos > maxScanLine {
			maxScanLine = pos
		}
		level := absInt(int(cv))
		absLevels[c] = uint16(level)
		nonZeroBits[c>>6] |= 1 << uint(c&63)
		culLevel += level
		if pos == 0 {
			dcValue = int(cv)
		}
	}
	var lowerContexts [maxEOB]int8
	lowerContextMap := coeffNZMapContexts2DFullArch(levels[:], TransformSize16x16, lowerContexts[:])

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][plane]
	baseCDFs := &cdfs.CoeffBase[txCtx][plane]
	brCDFs := &cdfs.CoeffBR[txBR][plane]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF3(&baseEOBCDFs[ctx], minInt(level, 3)-1)
		} else {
			ctx := 0
			if pos != 0 {
				ctx = int(lowerContexts[pos])
				if !lowerContextMap {
					pad := p.padded
					mag := clipMax3(levels[pad+stride]) + clipMax3(levels[pad+1]) +
						clipMax3(levels[pad+stride+1]) + clipMax3(levels[pad+(stride<<1)]) + clipMax3(levels[pad+2])
					ctx = minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
				}
			}
			w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
		}
		if level > NumBaseLevels {
			brCtx := 0
			if c == eob-1 {
				brCtx = int(p.brEOBCtx)
			} else if pos != 0 {
				pad := p.padded
				mag := minInt(int(levels[pad+1]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride+1]), MaxBaseBRRange)
				brCtx = minInt((mag+1)>>1, 6) + int(p.br2DOffset)
			} else {
				pad := p.padded
				mag := int(levels[pad+1]) + int(levels[pad+stride]) + int(levels[pad+stride+1])
				brCtx = minInt((mag+1)>>1, 6)
			}
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	for word, bits64 := range nonZeroBits {
		for nz := bits64; nz != 0; nz &= nz - 1 {
			c := word*64 + bits.TrailingZeros64(nz)
			p := &scanHot[c]
			pos := int(p.pos)
			level := int(absLevels[c])
			cv := coeff256[pos]
			sign := int(uint16(cv) >> 15)
			if pos == 0 {
				w.WriteBinaryCDFTrusted(&cdfs.DCSign[plane][dcSignContext], sign)
			} else {
				w.WriteBit(sign)
			}
			if level >= MaxBaseBRRange {
				writeGolomb(w, level-MaxBaseBRRange)
			}
		}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: uint16(maxScanLine),
		CulLevel:    uint8(culLevel),
	}
}

// CountCoefficientsTXB32x32Y2DTrusted is the exact output-free rate-pricing
// specialization for 32x32 luma/Class2D trials with zero txb_skip/dc_sign
// contexts. It adapts cdfs identically to WriteCoefficientsTXB and returns the
// same Tell delta that a byte writer would observe.
func CountCoefficientsTXB32x32Y2DTrusted(cdfs *CoeffCDFs, coeffs []int16) (TXBDecodeResult, int) {
	return countCoefficientsTXB32x32Plane2DTrustedArray(cdfs, (*[1024]int16)(coeffs), CoeffPlaneY, nil, 0)
}

// CountCoefficientsTXB32x32Y2DTrustedArray is
// CountCoefficientsTXB32x32Y2DTrusted for callers that can prove the
// 1024-coefficient shape before the hot call.
func CountCoefficientsTXB32x32Y2DTrustedArray(cdfs *CoeffCDFs, coeff1024 *[1024]int16) (TXBDecodeResult, int) {
	return countCoefficientsTXB32x32Plane2DTrustedArray(cdfs, coeff1024, CoeffPlaneY, nil, 0)
}

// CountCoefficientsTXB32x32Y2DTrustedWithTXType is the inter trial-pricing
// variant: it writes the already-derived transform-type symbol after txb_skip,
// then restores txCDF so trial pricing does not adapt the real transform CDFs.
func CountCoefficientsTXB32x32Y2DTrustedWithTXType(cdfs *CoeffCDFs, coeffs []int16, txCDF *entropy.CDF, txSymbol int) (TXBDecodeResult, int) {
	return countCoefficientsTXB32x32Plane2DTrustedArray(cdfs, (*[1024]int16)(coeffs), CoeffPlaneY, txCDF, txSymbol)
}

// CountCoefficientsTXB32x32Y2DTrustedWithTXTypeArray is the pointer-shaped
// form for inter transform-type trial pricing.
func CountCoefficientsTXB32x32Y2DTrustedWithTXTypeArray(cdfs *CoeffCDFs, coeff1024 *[1024]int16, txCDF *entropy.CDF, txSymbol int) (TXBDecodeResult, int) {
	return countCoefficientsTXB32x32Plane2DTrustedArray(cdfs, coeff1024, CoeffPlaneY, txCDF, txSymbol)
}

// CountCoefficientsTXB32x32UV2DTrusted is the exact output-free rate-pricing
// specialization for 32x32 chroma/Class2D trials with zero txb_skip/dc_sign
// contexts. It adapts cdfs identically to WriteCoefficientsTXB and returns the
// same Tell delta that a byte writer would observe.
func CountCoefficientsTXB32x32UV2DTrusted(cdfs *CoeffCDFs, coeffs []int16) (TXBDecodeResult, int) {
	return countCoefficientsTXB32x32Plane2DTrustedArray(cdfs, (*[1024]int16)(coeffs), CoeffPlaneUV, nil, 0)
}

// CountCoefficientsTXB32x32UV2DTrustedArray is
// CountCoefficientsTXB32x32UV2DTrusted for callers that can prove the
// 1024-coefficient shape before the hot call.
func CountCoefficientsTXB32x32UV2DTrustedArray(cdfs *CoeffCDFs, coeff1024 *[1024]int16) (TXBDecodeResult, int) {
	return countCoefficientsTXB32x32Plane2DTrustedArray(cdfs, coeff1024, CoeffPlaneUV, nil, 0)
}

func countCoefficientsTXB32x32Plane2DTrustedArray(cdfs *CoeffCDFs, coeff1024 *[1024]int16, plane CoeffPlaneType, txCDF *entropy.CDF, txSymbol int) (TXBDecodeResult, int) {
	const (
		maxEOB     = 1024
		scratchLen = 1296
		stride     = uint16(36)
		txCtx      = 3
		txBR       = 3
	)
	scanHot := &coeffScanHot32x32Y2D
	w := entropy.NewBitCounter()
	base := w.Tell()

	eob := 0
	for c := maxEOB - 1; c >= 0; c-- {
		if coeff1024[scanHot[c].pos] != 0 {
			eob = c + 1
			break
		}
	}

	w.WriteBinaryCDFTrusted(&cdfs.TXBSkip[txCtx][0], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}, w.Tell() - base
	}
	if txCDF != nil {
		saved := *txCDF
		w.WriteCDF(txCDF, txSymbol)
		*txCDF = saved
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF(&cdfs.EOBFlag1024[plane][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteBinaryCDFTrusted(&cdfs.EOBExtra[txCtx][plane][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [scratchLen]uint8
	var nonZeroBits [16]uint64
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff1024[pos]
		if cv == 0 {
			continue
		}
		levels[p.padded] = coeffAbsClamp127(cv)
		if pos > maxScanLine {
			maxScanLine = pos
		}
		level := absInt(int(cv))
		nonZeroBits[c>>6] |= 1 << uint(c&63)
		culLevel += level
		if pos == 0 {
			dcValue = int(cv)
		}
	}
	var lowerContexts [maxEOB]int8
	lowerContextMap := coeffNZMapContexts2DFullArch(levels[:], TransformSize32x32, lowerContexts[:])

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][plane]
	baseCDFs := &cdfs.CoeffBase[txCtx][plane]
	brCDFs := &cdfs.CoeffBR[txBR][plane]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := minInt(int(levels[p.padded]), MaxBaseBRRange)
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF3(&baseEOBCDFs[ctx], minInt(level, 3)-1)
		} else {
			ctx := 0
			if pos != 0 {
				ctx = int(lowerContexts[pos])
				if !lowerContextMap {
					pad := p.padded
					mag := clipMax3(levels[pad+stride]) + clipMax3(levels[pad+1]) +
						clipMax3(levels[pad+stride+1]) + clipMax3(levels[pad+(stride<<1)]) + clipMax3(levels[pad+2])
					ctx = minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
				}
			}
			if level == 0 {
				w.WriteCDF4Zero(&baseCDFs[ctx])
			} else {
				w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
			}
		}
		if level > NumBaseLevels {
			brCtx := 0
			if c == eob-1 {
				brCtx = int(p.brEOBCtx)
			} else if pos != 0 {
				pad := p.padded
				mag := minInt(int(levels[pad+1]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride+1]), MaxBaseBRRange)
				brCtx = minInt((mag+1)>>1, 6) + int(p.br2DOffset)
			} else {
				pad := p.padded
				mag := int(levels[pad+1]) + int(levels[pad+stride]) + int(levels[pad+stride+1])
				brCtx = minInt((mag+1)>>1, 6)
			}
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	for word, bits64 := range nonZeroBits {
		for nz := bits64; nz != 0; nz &= nz - 1 {
			c := word*64 + bits.TrailingZeros64(nz)
			p := &scanHot[c]
			pos := int(p.pos)
			cv := coeff1024[pos]
			sign := int(uint16(cv) >> 15)
			if pos == 0 {
				w.WriteBinaryCDFTrusted(&cdfs.DCSign[plane][0], sign)
			} else {
				w.WriteBit(sign)
			}
			if levels[p.padded] >= MaxBaseBRRange {
				level := absInt(int(cv))
				writeGolombCounter(&w, level-MaxBaseBRRange)
			}
		}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: uint16(maxScanLine),
		CulLevel:    uint8(culLevel),
	}, w.Tell() - base
}

// WriteCoefficientsTXB32x32Y2DContextTrusted is the validation-free 32x32
// luma/Class2D specialization for real coding with already-derived coefficient
// contexts. If txCDF is non-nil, txSymbol is written after txb_skip and txCDF
// is adapted in place.
func WriteCoefficientsTXB32x32Y2DContextTrusted(w *entropy.Writer, cdfs *CoeffCDFs, coeffs []int16, _ []uint8, txbSkipContext, dcSignContext uint8, txCDF *entropy.CDF, txSymbol int) TXBDecodeResult {
	return WriteCoefficientsTXB32x32Y2DContextTrustedArray(w, cdfs, (*[1024]int16)(coeffs), txbSkipContext, dcSignContext, txCDF, txSymbol)
}

// WriteCoefficientsTXB32x32Y2DContextTrustedArray is the pointer-shaped form
// for real 32x32 luma coding with already-derived coefficient contexts.
func WriteCoefficientsTXB32x32Y2DContextTrustedArray(w *entropy.Writer, cdfs *CoeffCDFs, coeffs *[1024]int16, txbSkipContext, dcSignContext uint8, txCDF *entropy.CDF, txSymbol int) TXBDecodeResult {
	return writeCoefficientsTXB32x32Plane2DContextTrustedArray(w, cdfs, coeffs, CoeffPlaneY, txbSkipContext, dcSignContext, txCDF, txSymbol)
}

// WriteCoefficientsTXB32x32Y2DContextTrustedZeroedLevels is the encoder-owned
// scratch form of WriteCoefficientsTXB32x32Y2DContextTrustedArray. levelsScratch
// must be zeroed on entry and is restored to zero before return.
func WriteCoefficientsTXB32x32Y2DContextTrustedZeroedLevels(w *entropy.Writer, cdfs *CoeffCDFs, coeffs *[1024]int16, levelsScratch []uint8, txbSkipContext, dcSignContext uint8, txCDF *entropy.CDF, txSymbol int) TXBDecodeResult {
	const scratchLen = 1296
	if len(levelsScratch) < scratchLen {
		return WriteCoefficientsTXB32x32Y2DContextTrustedArray(w, cdfs, coeffs, txbSkipContext, dcSignContext, txCDF, txSymbol)
	}
	return writeCoefficientsTXB32x32Plane2DContextTrustedArrayLevels(w, cdfs, coeffs, (*[scratchLen]uint8)(levelsScratch[:scratchLen]), true, CoeffPlaneY, txbSkipContext, dcSignContext, txCDF, txSymbol)
}

// WriteCoefficientsTXB32x32UV2DContextTrusted is the validation-free 32x32
// chroma/Class2D specialization for real coding with already-derived
// coefficient contexts.
func WriteCoefficientsTXB32x32UV2DContextTrusted(w *entropy.Writer, cdfs *CoeffCDFs, coeffs []int16, _ []uint8, txbSkipContext, dcSignContext uint8) TXBDecodeResult {
	return WriteCoefficientsTXB32x32UV2DContextTrustedArray(w, cdfs, (*[1024]int16)(coeffs), txbSkipContext, dcSignContext)
}

// WriteCoefficientsTXB32x32UV2DContextTrustedArray is the pointer-shaped form
// for real 32x32 chroma coding with already-derived coefficient contexts.
func WriteCoefficientsTXB32x32UV2DContextTrustedArray(w *entropy.Writer, cdfs *CoeffCDFs, coeffs *[1024]int16, txbSkipContext, dcSignContext uint8) TXBDecodeResult {
	return writeCoefficientsTXB32x32Plane2DContextTrustedArray(w, cdfs, coeffs, CoeffPlaneUV, txbSkipContext, dcSignContext, nil, 0)
}

func writeCoefficientsTXB32x32Plane2DContextTrustedArray(w *entropy.Writer, cdfs *CoeffCDFs, coeff1024 *[1024]int16, plane CoeffPlaneType, txbSkipContext, dcSignContext uint8, txCDF *entropy.CDF, txSymbol int) TXBDecodeResult {
	const scratchLen = 1296
	var levels [scratchLen]uint8
	return writeCoefficientsTXB32x32Plane2DContextTrustedArrayLevels(w, cdfs, coeff1024, &levels, false, plane, txbSkipContext, dcSignContext, txCDF, txSymbol)
}

func writeCoefficientsTXB32x32Plane2DContextTrustedArrayLevels(w *entropy.Writer, cdfs *CoeffCDFs, coeff1024 *[1024]int16, levels *[1296]uint8, clearLevels bool, plane CoeffPlaneType, txbSkipContext, dcSignContext uint8, txCDF *entropy.CDF, txSymbol int) TXBDecodeResult {
	const (
		maxEOB = 1024
		stride = uint16(36)
		txCtx  = 3
		txBR   = 3
	)
	scanHot := &coeffScanHot32x32Y2D

	eob := 0
	for c := maxEOB - 1; c >= 0; c-- {
		if coeff1024[scanHot[c].pos] != 0 {
			eob = c + 1
			break
		}
	}

	w.WriteBinaryCDFTrusted(&cdfs.TXBSkip[txCtx][txbSkipContext], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}
	}
	if txCDF != nil {
		w.WriteCDF(txCDF, txSymbol)
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF(&cdfs.EOBFlag1024[plane][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteBinaryCDFTrusted(&cdfs.EOBExtra[txCtx][plane][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	var nonZeroBits [16]uint64
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff1024[pos]
		if cv == 0 {
			continue
		}
		levels[p.padded] = coeffAbsClamp127(cv)
		if pos > maxScanLine {
			maxScanLine = pos
		}
		level := absInt(int(cv))
		nonZeroBits[c>>6] |= 1 << uint(c&63)
		culLevel += level
		if pos == 0 {
			dcValue = int(cv)
		}
	}
	var lowerContexts [maxEOB]int8
	lowerContextMap := coeffNZMapContexts2DFullArch(levels[:], TransformSize32x32, lowerContexts[:])

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][plane]
	baseCDFs := &cdfs.CoeffBase[txCtx][plane]
	brCDFs := &cdfs.CoeffBR[txBR][plane]

	c := eob - 1
	p := &scanHot[c]
	level := minInt(int(levels[p.padded]), MaxBaseBRRange)
	w.WriteCDF3(&baseEOBCDFs[int(p.lowerEOBCtx)], minInt(level, 3)-1)
	if level > NumBaseLevels {
		brCDF := &brCDFs[int(p.brEOBCtx)]
		baseRange := level - 1 - NumBaseLevels
		for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
			k := minInt(baseRange-idx, BRCDFSize-1)
			w.WriteCDF4(brCDF, k)
			if k < BRCDFSize-1 {
				break
			}
		}
	}
	for c = eob - 2; c > 0; c-- {
		p := &scanHot[c]
		level := minInt(int(levels[p.padded]), MaxBaseBRRange)
		pad := p.padded
		ctx := int(lowerContexts[p.pos])
		if !lowerContextMap {
			mag := clipMax3(levels[pad+stride]) + clipMax3(levels[pad+1]) +
				clipMax3(levels[pad+stride+1]) + clipMax3(levels[pad+(stride<<1)]) + clipMax3(levels[pad+2])
			ctx = minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
		}
		if level == 0 {
			w.WriteCDF4Zero(&baseCDFs[ctx])
		} else {
			w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
		}
		if level > NumBaseLevels {
			mag := minInt(int(levels[pad+1]), MaxBaseBRRange) +
				minInt(int(levels[pad+stride]), MaxBaseBRRange) +
				minInt(int(levels[pad+stride+1]), MaxBaseBRRange)
			brCtx := minInt((mag+1)>>1, 6) + int(p.br2DOffset)
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}
	if eob > 1 {
		p := &scanHot[0]
		level := minInt(int(levels[p.padded]), MaxBaseBRRange)
		if level == 0 {
			w.WriteCDF4Zero(&baseCDFs[0])
		} else {
			w.WriteCDF4(&baseCDFs[0], minInt(level, 3))
		}
		if level > NumBaseLevels {
			pad := p.padded
			mag := int(levels[pad+1]) + int(levels[pad+stride]) + int(levels[pad+stride+1])
			brCtx := minInt((mag+1)>>1, 6)
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	for word, bits64 := range nonZeroBits {
		for nz := bits64; nz != 0; nz &= nz - 1 {
			c := word*64 + bits.TrailingZeros64(nz)
			p := &scanHot[c]
			pos := int(p.pos)
			cv := coeff1024[pos]
			sign := int(uint16(cv) >> 15)
			if pos == 0 {
				w.WriteBinaryCDFTrusted(&cdfs.DCSign[plane][dcSignContext], sign)
			} else {
				w.WriteBit(sign)
			}
			level := levels[p.padded]
			if clearLevels {
				levels[p.padded] = 0
			}
			if level >= MaxBaseBRRange {
				level := absInt(int(cv))
				writeGolomb(w, level-MaxBaseBRRange)
			}
		}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: uint16(maxScanLine),
		CulLevel:    uint8(culLevel),
	}
}

// WriteCoefficientsTXB4x4Y2DContextTrusted is the validation-free 4x4
// luma/Class2D specialization for real coding with already-derived coefficient
// contexts. The afterSkip hook, when non-nil, is called after a nonzero
// txb_skip symbol and before the EOB token, matching TXBEncodeRequest.AfterSkip.
func WriteCoefficientsTXB4x4Y2DContextTrusted(w *entropy.Writer, cdfs *CoeffCDFs, coeffs []int16, _ []uint8, txbSkipContext, dcSignContext uint8, afterSkip func() error) (TXBDecodeResult, error) {
	return WriteCoefficientsTXB4x4Y2DContextTrustedArray(w, cdfs, (*[16]int16)(coeffs), txbSkipContext, dcSignContext, afterSkip)
}

// WriteCoefficientsTXB4x4Y2DContextTrustedTXType is the validation-free 4x4
// luma/Class2D specialization for real coding when the inter tx_type CDF and
// symbol are already known. txCDF is adapted in place when non-nil.
func WriteCoefficientsTXB4x4Y2DContextTrustedTXType(w *entropy.Writer, cdfs *CoeffCDFs, coeffs []int16, _ []uint8, txbSkipContext, dcSignContext uint8, txCDF *entropy.CDF, txSymbol int) TXBDecodeResult {
	return WriteCoefficientsTXB4x4Y2DContextTrustedTXTypeArray(w, cdfs, (*[16]int16)(coeffs), txbSkipContext, dcSignContext, txCDF, txSymbol)
}

// WriteCoefficientsTXB4x4Y2DContextTrustedTXTypeArray is the pointer-shaped form
// for real 4x4 luma coding with a pre-resolved inter tx_type symbol.
func WriteCoefficientsTXB4x4Y2DContextTrustedTXTypeArray(w *entropy.Writer, cdfs *CoeffCDFs, coeffs *[16]int16, txbSkipContext, dcSignContext uint8, txCDF *entropy.CDF, txSymbol int) TXBDecodeResult {
	result, _ := writeCoefficientsTXB4x4Y2DContextTrustedArray(w, cdfs, coeffs, txbSkipContext, dcSignContext, nil, txCDF, txSymbol)
	return result
}

// WriteCoefficientsTXB4x4Y2DContextTrustedArray is the pointer-shaped form for
// real 4x4 luma coding with already-derived coefficient contexts.
func WriteCoefficientsTXB4x4Y2DContextTrustedArray(w *entropy.Writer, cdfs *CoeffCDFs, coeffs *[16]int16, txbSkipContext, dcSignContext uint8, afterSkip func() error) (TXBDecodeResult, error) {
	return writeCoefficientsTXB4x4Y2DContextTrustedArray(w, cdfs, coeffs, txbSkipContext, dcSignContext, afterSkip, nil, 0)
}

func writeCoefficientsTXB4x4Y2DContextTrustedArray(w *entropy.Writer, cdfs *CoeffCDFs, coeffs *[16]int16, txbSkipContext, dcSignContext uint8, afterSkip func() error, txCDF *entropy.CDF, txSymbol int) (TXBDecodeResult, error) {
	const (
		maxEOB = 16
		stride = uint8(8)
		txCtx  = 0
		txBR   = 0
	)
	scanHot := &coeffScanHot4x4Y2D

	eob := 0
	for c := maxEOB - 1; c >= 0; c-- {
		if coeffs[scanHot[c].pos] != 0 {
			eob = c + 1
			break
		}
	}

	w.WriteBinaryCDFTrusted(&cdfs.TXBSkip[txCtx][txbSkipContext], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}, nil
	}
	if afterSkip != nil {
		if err := afterSkip(); err != nil {
			return TXBDecodeResult{}, err
		}
	} else if txCDF != nil {
		w.WriteCDF(txCDF, txSymbol)
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF5(&cdfs.EOBFlag16[CoeffPlaneY][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteBinaryCDFTrusted(&cdfs.EOBExtra[txCtx][CoeffPlaneY][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [64]uint8
	var absLevels [maxEOB]uint16
	var signBits uint16
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeffs[pos]
		if cv == 0 {
			continue
		}
		levels[p.padded] = coeffAbsClamp127(cv)
		if pos > maxScanLine {
			maxScanLine = pos
		}
		level := absInt(int(cv))
		absLevels[c] = uint16(level)
		if cv < 0 {
			signBits |= uint16(1) << uint(c)
		}
		culLevel += level
		if pos == 0 {
			dcValue = int(cv)
		}
	}

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneY]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneY]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneY]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF3(&baseEOBCDFs[ctx], minInt(level, 3)-1)
		} else {
			ctx := 0
			if pos != 0 {
				pad := p.padded
				mag := clipMax3(levels[pad+stride]) + clipMax3(levels[pad+1]) +
					clipMax3(levels[pad+stride+1]) + clipMax3(levels[pad+(stride<<1)]) + clipMax3(levels[pad+2])
				ctx = minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
			}
			w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
		}
		if level > NumBaseLevels {
			brCtx := 0
			if c == eob-1 {
				brCtx = int(p.brEOBCtx)
			} else if pos != 0 {
				pad := p.padded
				mag := minInt(int(levels[pad+1]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride+1]), MaxBaseBRRange)
				brCtx = minInt((mag+1)>>1, 6) + int(p.br2DOffset)
			} else {
				pad := p.padded
				mag := int(levels[pad+1]) + int(levels[pad+stride]) + int(levels[pad+stride+1])
				brCtx = minInt((mag+1)>>1, 6)
			}
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if level == 0 {
			continue
		}
		sign := int((signBits >> uint(c)) & 1)
		if pos == 0 {
			w.WriteBinaryCDFTrusted(&cdfs.DCSign[CoeffPlaneY][dcSignContext], sign)
		} else {
			w.WriteBit(sign)
		}
		if level >= MaxBaseBRRange {
			writeGolomb(w, level-MaxBaseBRRange)
		}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: uint16(maxScanLine),
		CulLevel:    uint8(culLevel),
	}, nil
}

// CountCoefficientsTXB4x4Y2DTrusted is the exact output-free rate-pricing
// specialization for 4x4 luma/Class2D trials with zero txb_skip/dc_sign
// contexts. It adapts cdfs identically to WriteCoefficientsTXB and returns the
// same Tell delta that a byte writer would observe.
func CountCoefficientsTXB4x4Y2DTrusted(cdfs *CoeffCDFs, coeffs []int16) (TXBDecodeResult, int) {
	return CountCoefficientsTXB4x4Y2DTrustedArray(cdfs, (*[16]int16)(coeffs))
}

// CountCoefficientsTXB4x4Y2DTrustedArray is CountCoefficientsTXB4x4Y2DTrusted
// for callers that can prove the 16-coefficient shape before the hot call.
func CountCoefficientsTXB4x4Y2DTrustedArray(cdfs *CoeffCDFs, coeff16 *[16]int16) (TXBDecodeResult, int) {
	const (
		maxEOB = 16
		stride = uint8(8)
		txCtx  = 0
		txBR   = 0
	)
	scanHot := &coeffScanHot4x4Y2D
	w := entropy.NewBitCounter()
	base := w.Tell()

	eob := 0
	for c := maxEOB - 1; c >= 0; c-- {
		if coeff16[scanHot[c].pos] != 0 {
			eob = c + 1
			break
		}
	}

	w.WriteBinaryCDFTrusted(&cdfs.TXBSkip[txCtx][0], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}, w.Tell() - base
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF5(&cdfs.EOBFlag16[CoeffPlaneY][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteBinaryCDFTrusted(&cdfs.EOBExtra[txCtx][CoeffPlaneY][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [64]uint8
	var absLevels [maxEOB]uint16
	var signBits uint16
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff16[pos]
		if cv == 0 {
			continue
		}
		levels[p.padded] = coeffAbsClamp127(cv)
		if pos > maxScanLine {
			maxScanLine = pos
		}
		level := absInt(int(cv))
		absLevels[c] = uint16(level)
		if cv < 0 {
			signBits |= uint16(1) << uint(c)
		}
		culLevel += level
		if pos == 0 {
			dcValue = int(cv)
		}
	}

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneY]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneY]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneY]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF3(&baseEOBCDFs[ctx], minInt(level, 3)-1)
		} else {
			ctx := 0
			if pos != 0 {
				pad := p.padded
				mag := clipMax3(levels[pad+stride]) + clipMax3(levels[pad+1]) +
					clipMax3(levels[pad+stride+1]) + clipMax3(levels[pad+(stride<<1)]) + clipMax3(levels[pad+2])
				ctx = minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
			}
			w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
		}
		if level > NumBaseLevels {
			brCtx := 0
			if c == eob-1 {
				brCtx = int(p.brEOBCtx)
			} else if pos != 0 {
				pad := p.padded
				mag := minInt(int(levels[pad+1]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride+1]), MaxBaseBRRange)
				brCtx = minInt((mag+1)>>1, 6) + int(p.br2DOffset)
			} else {
				pad := p.padded
				mag := int(levels[pad+1]) + int(levels[pad+stride]) + int(levels[pad+stride+1])
				brCtx = minInt((mag+1)>>1, 6)
			}
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if level == 0 {
			continue
		}
		sign := int((signBits >> uint(c)) & 1)
		if pos == 0 {
			w.WriteBinaryCDFTrusted(&cdfs.DCSign[CoeffPlaneY][0], sign)
		} else {
			w.WriteBit(sign)
		}
		if level >= MaxBaseBRRange {
			writeGolombCounter(&w, level-MaxBaseBRRange)
		}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: uint16(maxScanLine),
		CulLevel:    uint8(culLevel),
	}, w.Tell() - base
}

// WriteCoefficientsTXB4x4UV2DContextTrusted is the validation-free 4x4
// chroma/Class2D specialization for real coding with already-derived
// coefficient contexts.
func WriteCoefficientsTXB4x4UV2DContextTrusted(w *entropy.Writer, cdfs *CoeffCDFs, coeffs []int16, _ []uint8, txbSkipContext, dcSignContext uint8) TXBDecodeResult {
	return WriteCoefficientsTXB4x4UV2DContextTrustedArray(w, cdfs, (*[16]int16)(coeffs), txbSkipContext, dcSignContext)
}

// WriteCoefficientsTXB4x4UV2DContextTrustedArray is the pointer-shaped form for
// real 4x4 chroma coding with already-derived coefficient contexts.
func WriteCoefficientsTXB4x4UV2DContextTrustedArray(w *entropy.Writer, cdfs *CoeffCDFs, coeffs *[16]int16, txbSkipContext, dcSignContext uint8) TXBDecodeResult {
	const (
		maxEOB = 16
		stride = uint8(8)
		txCtx  = 0
		txBR   = 0
	)
	scanHot := &coeffScanHot4x4Y2D

	eob := 0
	for c := maxEOB - 1; c >= 0; c-- {
		if coeffs[scanHot[c].pos] != 0 {
			eob = c + 1
			break
		}
	}

	w.WriteBinaryCDFTrusted(&cdfs.TXBSkip[txCtx][txbSkipContext], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF5(&cdfs.EOBFlag16[CoeffPlaneUV][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteBinaryCDFTrusted(&cdfs.EOBExtra[txCtx][CoeffPlaneUV][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [64]uint8
	var absLevels [maxEOB]uint16
	var signBits uint16
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeffs[pos]
		if cv == 0 {
			continue
		}
		levels[p.padded] = coeffAbsClamp127(cv)
		if pos > maxScanLine {
			maxScanLine = pos
		}
		level := absInt(int(cv))
		absLevels[c] = uint16(level)
		if cv < 0 {
			signBits |= uint16(1) << uint(c)
		}
		culLevel += level
		if pos == 0 {
			dcValue = int(cv)
		}
	}

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneUV]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneUV]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneUV]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF3(&baseEOBCDFs[ctx], minInt(level, 3)-1)
		} else {
			ctx := 0
			if pos != 0 {
				pad := p.padded
				mag := clipMax3(levels[pad+stride]) + clipMax3(levels[pad+1]) +
					clipMax3(levels[pad+stride+1]) + clipMax3(levels[pad+(stride<<1)]) + clipMax3(levels[pad+2])
				ctx = minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
			}
			w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
		}
		if level > NumBaseLevels {
			brCtx := 0
			if c == eob-1 {
				brCtx = int(p.brEOBCtx)
			} else if pos != 0 {
				pad := p.padded
				mag := minInt(int(levels[pad+1]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride+1]), MaxBaseBRRange)
				brCtx = minInt((mag+1)>>1, 6) + int(p.br2DOffset)
			} else {
				pad := p.padded
				mag := int(levels[pad+1]) + int(levels[pad+stride]) + int(levels[pad+stride+1])
				brCtx = minInt((mag+1)>>1, 6)
			}
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if level == 0 {
			continue
		}
		sign := int((signBits >> uint(c)) & 1)
		if pos == 0 {
			w.WriteBinaryCDFTrusted(&cdfs.DCSign[CoeffPlaneUV][dcSignContext], sign)
		} else {
			w.WriteBit(sign)
		}
		if level >= MaxBaseBRRange {
			writeGolomb(w, level-MaxBaseBRRange)
		}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: uint16(maxScanLine),
		CulLevel:    uint8(culLevel),
	}
}

// CountCoefficientsTXB4x4UV2DTrusted is the exact output-free rate-pricing
// specialization for 4x4 chroma/Class2D trials with zero txb_skip/dc_sign
// contexts. It adapts cdfs identically to WriteCoefficientsTXB and returns the
// same Tell delta that a byte writer would observe.
func CountCoefficientsTXB4x4UV2DTrusted(cdfs *CoeffCDFs, coeffs []int16) (TXBDecodeResult, int) {
	return CountCoefficientsTXB4x4UV2DTrustedArray(cdfs, (*[16]int16)(coeffs))
}

// CountCoefficientsTXB4x4UV2DTrustedArray is
// CountCoefficientsTXB4x4UV2DTrusted for callers that can prove the
// 16-coefficient shape before the hot call.
func CountCoefficientsTXB4x4UV2DTrustedArray(cdfs *CoeffCDFs, coeff16 *[16]int16) (TXBDecodeResult, int) {
	const (
		maxEOB = 16
		stride = uint8(8)
		txCtx  = 0
		txBR   = 0
	)
	scanHot := &coeffScanHot4x4Y2D
	w := entropy.NewBitCounter()
	base := w.Tell()

	eob := 0
	for c := maxEOB - 1; c >= 0; c-- {
		if coeff16[scanHot[c].pos] != 0 {
			eob = c + 1
			break
		}
	}

	w.WriteBinaryCDFTrusted(&cdfs.TXBSkip[txCtx][0], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}, w.Tell() - base
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF5(&cdfs.EOBFlag16[CoeffPlaneUV][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteBinaryCDFTrusted(&cdfs.EOBExtra[txCtx][CoeffPlaneUV][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [64]uint8
	var absLevels [maxEOB]uint16
	var signBits uint16
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff16[pos]
		if cv == 0 {
			continue
		}
		levels[p.padded] = coeffAbsClamp127(cv)
		if pos > maxScanLine {
			maxScanLine = pos
		}
		level := absInt(int(cv))
		absLevels[c] = uint16(level)
		if cv < 0 {
			signBits |= uint16(1) << uint(c)
		}
		culLevel += level
		if pos == 0 {
			dcValue = int(cv)
		}
	}

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneUV]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneUV]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneUV]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF3(&baseEOBCDFs[ctx], minInt(level, 3)-1)
		} else {
			ctx := 0
			if pos != 0 {
				pad := p.padded
				mag := clipMax3(levels[pad+stride]) + clipMax3(levels[pad+1]) +
					clipMax3(levels[pad+stride+1]) + clipMax3(levels[pad+(stride<<1)]) + clipMax3(levels[pad+2])
				ctx = minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
			}
			w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
		}
		if level > NumBaseLevels {
			brCtx := 0
			if c == eob-1 {
				brCtx = int(p.brEOBCtx)
			} else if pos != 0 {
				pad := p.padded
				mag := minInt(int(levels[pad+1]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride+1]), MaxBaseBRRange)
				brCtx = minInt((mag+1)>>1, 6) + int(p.br2DOffset)
			} else {
				pad := p.padded
				mag := int(levels[pad+1]) + int(levels[pad+stride]) + int(levels[pad+stride+1])
				brCtx = minInt((mag+1)>>1, 6)
			}
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if level == 0 {
			continue
		}
		sign := int((signBits >> uint(c)) & 1)
		if pos == 0 {
			w.WriteBinaryCDFTrusted(&cdfs.DCSign[CoeffPlaneUV][0], sign)
		} else {
			w.WriteBit(sign)
		}
		if level >= MaxBaseBRRange {
			writeGolombCounter(&w, level-MaxBaseBRRange)
		}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: uint16(maxScanLine),
		CulLevel:    uint8(culLevel),
	}, w.Tell() - base
}

// WriteCoefficientsTXB8x8Y2DContextTrusted is the validation-free 8x8
// luma/Class2D specialization for real coding with already-derived coefficient
// contexts. Unlike WriteCoefficientsTXB8x8Y2DTrusted, txCDF is adapted in place.
func WriteCoefficientsTXB8x8Y2DContextTrusted(w *entropy.Writer, cdfs *CoeffCDFs, coeffs []int16, _ []uint8, txbSkipContext, dcSignContext uint8, txCDF *entropy.CDF, txSymbol int) TXBDecodeResult {
	return WriteCoefficientsTXB8x8Y2DContextTrustedArray(w, cdfs, (*[64]int16)(coeffs), txbSkipContext, dcSignContext, txCDF, txSymbol)
}

// WriteCoefficientsTXB8x8Y2DContextTrustedArray is the pointer-shaped form for
// real 8x8 luma coding with already-derived coefficient contexts.
func WriteCoefficientsTXB8x8Y2DContextTrustedArray(w *entropy.Writer, cdfs *CoeffCDFs, coeffs *[64]int16, txbSkipContext, dcSignContext uint8, txCDF *entropy.CDF, txSymbol int) TXBDecodeResult {
	return writeCoefficientsTXB8x8Y2DTrustedArray(w, cdfs, coeffs, txbSkipContext, dcSignContext, txCDF, txSymbol, false)
}

func writeCoefficientsTXB8x8Y2DTrustedArray(w *entropy.Writer, cdfs *CoeffCDFs, coeff64 *[64]int16, txbSkipContext, dcSignContext uint8, txCDF *entropy.CDF, txSymbol int, restoreTXCDF bool) TXBDecodeResult {
	const (
		maxEOB = 64
		stride = uint8(12)
		txCtx  = 1
		txBR   = 1
	)
	scanHot := &coeffScanHot8x8Y2D

	eob := 0
	for c := maxEOB - 1; c >= 0; c-- {
		if coeff64[scanHot[c].pos] != 0 {
			eob = c + 1
			break
		}
	}

	w.WriteBinaryCDFTrusted(&cdfs.TXBSkip[txCtx][txbSkipContext], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}
	}
	if txCDF != nil {
		var saved entropy.CDF
		if restoreTXCDF {
			saved = *txCDF
		}
		w.WriteCDF(txCDF, txSymbol)
		if restoreTXCDF {
			*txCDF = saved
		}
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF7(&cdfs.EOBFlag64[CoeffPlaneY][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteBinaryCDFTrusted(&cdfs.EOBExtra[txCtx][CoeffPlaneY][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [256]uint8
	var absLevels [64]uint16
	var signBits uint64
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff64[pos]
		if cv == 0 {
			continue
		}
		levels[p.padded] = coeffAbsClamp127(cv)
		if pos > maxScanLine {
			maxScanLine = pos
		}
		level := absInt(int(cv))
		absLevels[c] = uint16(level)
		if cv < 0 {
			signBits |= uint64(1) << uint(c)
		}
		culLevel += level
		if pos == 0 {
			dcValue = int(cv)
		}
	}

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneY]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneY]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneY]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF3(&baseEOBCDFs[ctx], minInt(level, 3)-1)
		} else {
			ctx := 0
			if pos != 0 {
				pad := p.padded
				mag := clipMax3(levels[pad+stride]) + clipMax3(levels[pad+1]) +
					clipMax3(levels[pad+stride+1]) + clipMax3(levels[pad+(stride<<1)]) + clipMax3(levels[pad+2])
				ctx = minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
			}
			w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
		}
		if level > NumBaseLevels {
			brCtx := 0
			if c == eob-1 {
				brCtx = int(p.brEOBCtx)
			} else if pos != 0 {
				pad := p.padded
				mag := minInt(int(levels[pad+1]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride+1]), MaxBaseBRRange)
				brCtx = minInt((mag+1)>>1, 6) + int(p.br2DOffset)
			} else {
				pad := p.padded
				mag := int(levels[pad+1]) + int(levels[pad+stride]) + int(levels[pad+stride+1])
				brCtx = minInt((mag+1)>>1, 6)
			}
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if level == 0 {
			continue
		}
		sign := int((signBits >> uint(c)) & 1)
		if pos == 0 {
			w.WriteBinaryCDFTrusted(&cdfs.DCSign[CoeffPlaneY][dcSignContext], sign)
		} else {
			w.WriteBit(sign)
		}
		if level >= MaxBaseBRRange {
			writeGolomb(w, level-MaxBaseBRRange)
		}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: uint16(maxScanLine),
		CulLevel:    uint8(culLevel),
	}
}

// WriteCoefficientsTXB8x8UV2DContextTrusted is the validation-free 8x8
// chroma/Class2D specialization for real coding with already-derived
// coefficient contexts.
func WriteCoefficientsTXB8x8UV2DContextTrusted(w *entropy.Writer, cdfs *CoeffCDFs, coeffs []int16, _ []uint8, txbSkipContext, dcSignContext uint8) TXBDecodeResult {
	return WriteCoefficientsTXB8x8UV2DContextTrustedArray(w, cdfs, (*[64]int16)(coeffs), txbSkipContext, dcSignContext)
}

// WriteCoefficientsTXB8x8UV2DContextTrustedArray is the pointer-shaped form for
// real 8x8 chroma coding with already-derived coefficient contexts.
func WriteCoefficientsTXB8x8UV2DContextTrustedArray(w *entropy.Writer, cdfs *CoeffCDFs, coeffs *[64]int16, txbSkipContext, dcSignContext uint8) TXBDecodeResult {
	return writeCoefficientsTXB8x8UV2DContextTrustedArray(w, cdfs, coeffs, txbSkipContext, dcSignContext)
}

func writeCoefficientsTXB8x8UV2DContextTrustedArray(w *entropy.Writer, cdfs *CoeffCDFs, coeffs *[64]int16, txbSkipContext, dcSignContext uint8) TXBDecodeResult {
	const (
		maxEOB = 64
		stride = uint8(12)
		txCtx  = 1
		txBR   = 1
	)
	scanHot := &coeffScanHot8x8Y2D

	eob := 0
	for c := maxEOB - 1; c >= 0; c-- {
		if coeffs[scanHot[c].pos] != 0 {
			eob = c + 1
			break
		}
	}

	w.WriteBinaryCDFTrusted(&cdfs.TXBSkip[txCtx][txbSkipContext], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF(&cdfs.EOBFlag64[CoeffPlaneUV][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteBinaryCDFTrusted(&cdfs.EOBExtra[txCtx][CoeffPlaneUV][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [256]uint8
	var absLevels [64]uint16
	var signBits uint64
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeffs[pos]
		if cv == 0 {
			continue
		}
		levels[p.padded] = coeffAbsClamp127(cv)
		if pos > maxScanLine {
			maxScanLine = pos
		}
		level := absInt(int(cv))
		absLevels[c] = uint16(level)
		if cv < 0 {
			signBits |= uint64(1) << uint(c)
		}
		culLevel += level
		if pos == 0 {
			dcValue = int(cv)
		}
	}

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneUV]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneUV]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneUV]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF3(&baseEOBCDFs[ctx], minInt(level, 3)-1)
		} else {
			ctx := 0
			if pos != 0 {
				pad := p.padded
				mag := clipMax3(levels[pad+stride]) + clipMax3(levels[pad+1]) +
					clipMax3(levels[pad+stride+1]) + clipMax3(levels[pad+(stride<<1)]) + clipMax3(levels[pad+2])
				ctx = minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
			}
			w.WriteCDF4(&baseCDFs[ctx], minInt(level, 3))
		}
		if level > NumBaseLevels {
			brCtx := 0
			if c == eob-1 {
				brCtx = int(p.brEOBCtx)
			} else if pos != 0 {
				pad := p.padded
				mag := minInt(int(levels[pad+1]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride]), MaxBaseBRRange) +
					minInt(int(levels[pad+stride+1]), MaxBaseBRRange)
				brCtx = minInt((mag+1)>>1, 6) + int(p.br2DOffset)
			} else {
				pad := p.padded
				mag := int(levels[pad+1]) + int(levels[pad+stride]) + int(levels[pad+stride+1])
				brCtx = minInt((mag+1)>>1, 6)
			}
			brCDF := &brCDFs[brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF4(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		level := int(absLevels[c])
		if level == 0 {
			continue
		}
		sign := int((signBits >> uint(c)) & 1)
		if pos == 0 {
			w.WriteBinaryCDFTrusted(&cdfs.DCSign[CoeffPlaneUV][dcSignContext], sign)
		} else {
			w.WriteBit(sign)
		}
		if level >= MaxBaseBRRange {
			writeGolomb(w, level-MaxBaseBRRange)
		}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: uint16(maxScanLine),
		CulLevel:    uint8(culLevel),
	}
}

// WriteCoefficientsTXBWithContext derives the txb_skip and dc_sign contexts from
// the caller-owned CoeffEntropyContext carrier, codes the transform block with
// WriteCoefficientsTXB, and marks the carrier with the block's cumulative level
// exactly as the decoder's ReadCoefficientsTXBWithContext does, so encoder and
// decoder carrier state evolve in lockstep. It returns the same TXBDecodeResult
// the decoder would produce for the block.
func WriteCoefficientsTXBWithContext(w *entropy.Writer, cdfs *CoeffCDFs, ctx *CoeffEntropyContext, ctxReq CoeffContextRequest, class transform.Class, coeffs []int16, scan []int16, levels []uint8) (TXBDecodeResult, error) {
	return WriteCoefficientsTXBWithContextHook(w, cdfs, ctx, ctxReq, class, coeffs, scan, levels, nil)
}

// WriteCoefficientsTXBWithContextHook is WriteCoefficientsTXBWithContext with
// the TXBEncodeRequest.AfterSkip hook exposed, for luma transform blocks that
// code a tx_type symbol between txb_skip and the eob token.
func WriteCoefficientsTXBWithContextHook(w *entropy.Writer, cdfs *CoeffCDFs, ctx *CoeffEntropyContext, ctxReq CoeffContextRequest, class transform.Class, coeffs []int16, scan []int16, levels []uint8, afterSkip func() error) (TXBDecodeResult, error) {
	if ctx == nil {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	plane, err := CoeffPlaneTypeForPlane(int(ctxReq.Plane))
	if err != nil {
		return TXBDecodeResult{}, err
	}
	txbCtx, err := ctx.TXBContext(ctxReq)
	if err != nil {
		return TXBDecodeResult{}, err
	}
	result, err := WriteCoefficientsTXB(w, cdfs, TXBEncodeRequest{
		Size:           ctxReq.Size,
		Plane:          plane,
		Class:          class,
		TXBSkipContext: txbCtx.TXBSkipContext,
		DCSignContext:  txbCtx.DCSignContext,
		AfterSkip:      afterSkip,
	}, coeffs, scan, levels)
	if err != nil {
		return TXBDecodeResult{}, err
	}
	if err := ctx.MarkTXB(ctxReq, result); err != nil {
		return TXBDecodeResult{}, err
	}
	return result, nil
}

func boolToSym(b bool) int {
	if b {
		return 1
	}
	return 0
}
