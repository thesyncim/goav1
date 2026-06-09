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
		w.WriteLiteral(0, 1)
	}
	for i := length - 1; i >= 0; i-- {
		w.WriteLiteral(uint32((x>>uint(i))&1), 1)
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
func WriteCoefficientsTXB(w *entropy.Writer, cdfs *CoeffCDFs, req TXBEncodeRequest, coeffs []int16, scan []int16, levels []uint8) error {
	if w == nil || cdfs == nil || !req.Plane.Valid() || !req.Class.Valid() {
		return ErrInvalidDecodeState
	}
	geo, ok := coeffGeoPtr(req.Size)
	if !ok {
		return ErrInvalidDecodeState
	}
	maxEOB := int(geo.maxEOB)
	scratchLen := int(geo.scratchLen)
	if len(coeffs) < maxEOB || len(scan) < maxEOB || len(levels) < scratchLen {
		return ErrInvalidDecodeState
	}
	txSize, err := req.Size.TransformSize()
	if err != nil {
		return ErrInvalidDecodeState
	}
	scanSize, err := transform.ScanSize(txSize)
	if err != nil {
		return ErrInvalidDecodeState
	}
	posTable := coeffPosTable[req.Size]
	if len(posTable) < maxEOB {
		return ErrInvalidDecodeState
	}

	// eob = scan index of the last non-zero coefficient plus one.
	eob := 0
	for c := range maxEOB {
		if coeffs[int(scan[c])] != 0 {
			eob = c + 1
		}
	}

	txCtx := int(geo.txCtx)
	txBR := int(geo.txBRCtx)

	// 1) txb_skip (all_zero).
	w.WriteCDF(&cdfs.TXBSkip[txCtx][req.TXBSkipContext], boolToSym(eob == 0))
	if eob == 0 {
		return nil
	}
	if req.AfterSkip != nil {
		// Luma tx_type position (av1_write_coeffs_txb: after txb_skip, before
		// the eob token).
		if err := req.AfterSkip(); err != nil {
			return err
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
		return err
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
	for c := range eob {
		pos := int(scan[c])
		lv := min(absInt(int(coeffs[pos])), 127)
		levels[int(posTable[pos].padded)] = uint8(lv)
	}

	// 3) lower-magnitude level pass, reversed scan order.
	for c := eob - 1; c >= 0; c-- {
		pos := int(scan[c])
		level := absInt(int(coeffs[pos]))
		if c == eob-1 {
			ctx, err := transform.LowerLevelsCtxEOB(scanSize, c)
			if err != nil {
				return ErrInvalidDecodeState
			}
			w.WriteCDF(&cdfs.CoeffBaseEOB[txCtx][req.Plane][ctx], minInt(level, 3)-1)
		} else {
			ctx, err := CoeffLowerLevelsContext(levels, req.Size, req.Class, pos)
			if err != nil {
				return err
			}
			w.WriteCDF(&cdfs.CoeffBase[txCtx][req.Plane][ctx], minInt(level, 3))
		}
		if level > NumBaseLevels {
			var brCtx int
			if c == eob-1 {
				brCtx, err = CoeffBRContextEOB(req.Size, req.Class, pos)
			} else {
				brCtx, err = CoeffBRContext(levels, req.Size, req.Class, pos)
			}
			if err != nil {
				return err
			}
			brCDF := &cdfs.CoeffBR[txBR][req.Plane][brCtx]
			baseRange := level - 1 - NumBaseLevels
			for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
				k := minInt(baseRange-idx, BRCDFSize-1)
				w.WriteCDF(brCDF, k)
				if k < BRCDFSize-1 {
					break
				}
			}
		}
	}

	// 4) sign + golomb pass, forward scan order, non-zero coefficients only.
	for c := range eob {
		pos := int(scan[c])
		v := int(coeffs[pos])
		if v == 0 {
			continue
		}
		level := absInt(v)
		sign := 0
		if v < 0 {
			sign = 1
		}
		if c == 0 {
			w.WriteCDF(&cdfs.DCSign[req.Plane][req.DCSignContext], sign)
		} else {
			w.WriteLiteral(uint32(sign), 1)
		}
		if level >= MaxBaseBRRange {
			writeGolomb(w, level-MaxBaseBRRange)
		}
	}
	return nil
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
	if err := WriteCoefficientsTXB(w, cdfs, TXBEncodeRequest{
		Size:           ctxReq.Size,
		Plane:          plane,
		Class:          class,
		TXBSkipContext: txbCtx.TXBSkipContext,
		DCSignContext:  txbCtx.DCSignContext,
		AfterSkip:      afterSkip,
	}, coeffs, scan, levels); err != nil {
		return TXBDecodeResult{}, err
	}
	result := txbResultFromCoeffs(ctxReq.Size, coeffs, scan)
	if err := ctx.MarkTXB(ctxReq, result); err != nil {
		return TXBDecodeResult{}, err
	}
	return result, nil
}

// txbResultFromCoeffs computes the TXBDecodeResult the decoder derives while
// reading a transform block: eob, max scan line, and the cumulative-level /
// DC-sign context byte (the final culLevel computation in ReadCoefficientsTXB).
func txbResultFromCoeffs(size TransformSize, coeffs []int16, scan []int16) TXBDecodeResult {
	geo, ok := coeffGeoPtr(size)
	if !ok {
		return TXBDecodeResult{}
	}
	maxEOB := int(geo.maxEOB)
	eob := 0
	maxScanLine := 0
	culLevel := 0
	for c := range maxEOB {
		pos := int(scan[c])
		if coeffs[pos] != 0 {
			eob = c + 1
			if pos > maxScanLine {
				maxScanLine = pos
			}
			culLevel += absInt(int(coeffs[pos]))
		}
	}
	if eob == 0 {
		return TXBDecodeResult{AllZero: true}
	}
	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dc := int(coeffs[int(scan[0])]); dc < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dc > 0 {
		culLevel += 2 << CoeffContextBits
	}
	return TXBDecodeResult{
		EOB:         uint16(eob),
		MaxScanLine: uint16(maxScanLine),
		CulLevel:    uint8(culLevel),
	}
}

func boolToSym(b bool) int {
	if b {
		return 1
	}
	return 0
}
