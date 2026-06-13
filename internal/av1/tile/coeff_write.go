package tile

import (
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
	w.WriteCDF(&cdfs.TXBSkip[txCtx][req.TXBSkipContext], boolToSym(eob == 0))
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
	w.WriteCDF(eobCDF, token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteCDF(&cdfs.EOBExtra[txCtx][req.Plane][token-3], firstBit)
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
			w.WriteCDF(&cdfs.CoeffBaseEOB[txCtx][req.Plane][ctx], minInt(level, 3)-1)
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
			w.WriteCDF(&cdfs.DCSign[req.Plane][req.DCSignContext], sign)
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

	w.WriteCDF(&cdfs.TXBSkip[txCtx][0], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}, w.Tell() - base
	}
	if txCDF != nil {
		saved := *txCDF
		w.WriteCDF(txCDF, txSymbol)
		*txCDF = saved
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF(&cdfs.EOBFlag64[CoeffPlaneY][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteCDF(&cdfs.EOBExtra[txCtx][CoeffPlaneY][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [256]uint8
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff64[pos]
		levels[p.padded] = coeffAbsClamp127(cv)
		if cv != 0 {
			if pos > maxScanLine {
				maxScanLine = pos
			}
			level := absInt(int(cv))
			culLevel += level
			if pos == 0 {
				dcValue = int(cv)
			}
		}
	}

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneY]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneY]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneY]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := absInt(int(coeff64[pos]))
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF(&baseEOBCDFs[ctx], minInt(level, 3)-1)
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
		cv := coeff64[pos]
		if cv == 0 {
			continue
		}
		sign := int(uint16(cv) >> 15)
		if pos == 0 {
			w.WriteCDF(&cdfs.DCSign[CoeffPlaneY][0], sign)
		} else {
			w.WriteBit(sign)
		}
		if levels[p.padded] >= MaxBaseBRRange {
			level := absInt(int(cv))
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

	w.WriteCDF(&cdfs.TXBSkip[txCtx][0], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}, w.Tell() - base
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF(&cdfs.EOBFlag64[CoeffPlaneUV][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteCDF(&cdfs.EOBExtra[txCtx][CoeffPlaneUV][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [256]uint8
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff64[pos]
		levels[p.padded] = coeffAbsClamp127(cv)
		if cv != 0 {
			if pos > maxScanLine {
				maxScanLine = pos
			}
			level := absInt(int(cv))
			culLevel += level
			if pos == 0 {
				dcValue = int(cv)
			}
		}
	}

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneUV]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneUV]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneUV]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := absInt(int(coeff64[pos]))
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF(&baseEOBCDFs[ctx], minInt(level, 3)-1)
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
		cv := coeff64[pos]
		if cv == 0 {
			continue
		}
		sign := int(uint16(cv) >> 15)
		if pos == 0 {
			w.WriteCDF(&cdfs.DCSign[CoeffPlaneUV][0], sign)
		} else {
			w.WriteBit(sign)
		}
		if levels[p.padded] >= MaxBaseBRRange {
			level := absInt(int(cv))
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
	return CountCoefficientsTXB16x16Y2DTrustedArray(cdfs, (*[256]int16)(coeffs))
}

// CountCoefficientsTXB16x16Y2DTrustedArray is
// CountCoefficientsTXB16x16Y2DTrusted for callers that can prove the
// 256-coefficient shape before the hot call.
func CountCoefficientsTXB16x16Y2DTrustedArray(cdfs *CoeffCDFs, coeff256 *[256]int16) (TXBDecodeResult, int) {
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

	w.WriteCDF(&cdfs.TXBSkip[txCtx][0], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}, w.Tell() - base
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF(&cdfs.EOBFlag256[CoeffPlaneY][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteCDF(&cdfs.EOBExtra[txCtx][CoeffPlaneY][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [scratchLen]uint8
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff256[pos]
		levels[p.padded] = coeffAbsClamp127(cv)
		if cv != 0 {
			if pos > maxScanLine {
				maxScanLine = pos
			}
			level := absInt(int(cv))
			culLevel += level
			if pos == 0 {
				dcValue = int(cv)
			}
		}
	}

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneY]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneY]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneY]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := absInt(int(coeff256[pos]))
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF(&baseEOBCDFs[ctx], minInt(level, 3)-1)
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
		cv := coeff256[pos]
		if cv == 0 {
			continue
		}
		sign := int(uint16(cv) >> 15)
		if pos == 0 {
			w.WriteCDF(&cdfs.DCSign[CoeffPlaneY][0], sign)
		} else {
			w.WriteBit(sign)
		}
		if levels[p.padded] >= MaxBaseBRRange {
			level := absInt(int(cv))
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

	w.WriteCDF(&cdfs.TXBSkip[txCtx][0], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}, w.Tell() - base
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF(&cdfs.EOBFlag16[CoeffPlaneY][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteCDF(&cdfs.EOBExtra[txCtx][CoeffPlaneY][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [64]uint8
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff16[pos]
		levels[p.padded] = coeffAbsClamp127(cv)
		if cv != 0 {
			if pos > maxScanLine {
				maxScanLine = pos
			}
			level := absInt(int(cv))
			culLevel += level
			if pos == 0 {
				dcValue = int(cv)
			}
		}
	}

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneY]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneY]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneY]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := absInt(int(coeff16[pos]))
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF(&baseEOBCDFs[ctx], minInt(level, 3)-1)
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
		cv := coeff16[pos]
		if cv == 0 {
			continue
		}
		sign := int(uint16(cv) >> 15)
		if pos == 0 {
			w.WriteCDF(&cdfs.DCSign[CoeffPlaneY][0], sign)
		} else {
			w.WriteBit(sign)
		}
		if levels[p.padded] >= MaxBaseBRRange {
			level := absInt(int(cv))
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

	w.WriteCDF(&cdfs.TXBSkip[txCtx][0], boolToSym(eob == 0))
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}, w.Tell() - base
	}

	token, extra, _ := EOBPositionToken(eob)
	w.WriteCDF(&cdfs.EOBFlag16[CoeffPlaneUV][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteCDF(&cdfs.EOBExtra[txCtx][CoeffPlaneUV][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [64]uint8
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff16[pos]
		levels[p.padded] = coeffAbsClamp127(cv)
		if cv != 0 {
			if pos > maxScanLine {
				maxScanLine = pos
			}
			level := absInt(int(cv))
			culLevel += level
			if pos == 0 {
				dcValue = int(cv)
			}
		}
	}

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneUV]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneUV]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneUV]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := absInt(int(coeff16[pos]))
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF(&baseEOBCDFs[ctx], minInt(level, 3)-1)
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
		cv := coeff16[pos]
		if cv == 0 {
			continue
		}
		sign := int(uint16(cv) >> 15)
		if pos == 0 {
			w.WriteCDF(&cdfs.DCSign[CoeffPlaneUV][0], sign)
		} else {
			w.WriteBit(sign)
		}
		if levels[p.padded] >= MaxBaseBRRange {
			level := absInt(int(cv))
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

	w.WriteCDF(&cdfs.TXBSkip[txCtx][txbSkipContext], boolToSym(eob == 0))
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
	w.WriteCDF(&cdfs.EOBFlag64[CoeffPlaneY][0], token-1)
	if offsetBits := int(eobOffsetBits[token]); offsetBits > 0 {
		firstBit := (extra >> (offsetBits - 1)) & 1
		w.WriteCDF(&cdfs.EOBExtra[txCtx][CoeffPlaneY][token-3], firstBit)
		if offsetBits > 1 {
			w.WriteLiteral(uint32(extra&((1<<(offsetBits-1))-1)), offsetBits-1)
		}
	}

	var levels [256]uint8
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := range eob {
		p := &scanHot[c]
		pos := int(p.pos)
		cv := coeff64[pos]
		levels[p.padded] = coeffAbsClamp127(cv)
		if cv != 0 {
			if pos > maxScanLine {
				maxScanLine = pos
			}
			level := absInt(int(cv))
			culLevel += level
			if pos == 0 {
				dcValue = int(cv)
			}
		}
	}

	baseEOBCDFs := &cdfs.CoeffBaseEOB[txCtx][CoeffPlaneY]
	baseCDFs := &cdfs.CoeffBase[txCtx][CoeffPlaneY]
	brCDFs := &cdfs.CoeffBR[txBR][CoeffPlaneY]
	for c := eob - 1; c >= 0; c-- {
		p := &scanHot[c]
		pos := int(p.pos)
		level := absInt(int(coeff64[pos]))
		if c == eob-1 {
			ctx := int(p.lowerEOBCtx)
			w.WriteCDF(&baseEOBCDFs[ctx], minInt(level, 3)-1)
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
		cv := coeff64[pos]
		if cv == 0 {
			continue
		}
		sign := int(uint16(cv) >> 15)
		if pos == 0 {
			w.WriteCDF(&cdfs.DCSign[CoeffPlaneY][dcSignContext], sign)
		} else {
			w.WriteBit(sign)
		}
		if levels[p.padded] >= MaxBaseBRRange {
			level := absInt(int(cv))
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
