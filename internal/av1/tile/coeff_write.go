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

func boolToSym(b bool) int {
	if b {
		return 1
	}
	return 0
}
