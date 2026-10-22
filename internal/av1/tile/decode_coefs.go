package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// coefLevelsLen is dav1d's t->scratch.levels size.
const coefLevelsLen = entropy.CoefLevelsLen

// coefLoCtxOffsets is dav1d_lo_ctx_offsets (src/tables.c), indexed by
// [w == h, w > h, w < h][min(y, 4)][min(x, 4)].
var coefLoCtxOffsets = [3][5][5]uint8{
	{ // w == h
		{0, 1, 6, 6, 21},
		{1, 6, 6, 21, 21},
		{6, 6, 21, 21, 21},
		{6, 21, 21, 21, 21},
		{21, 21, 21, 21, 21},
	}, { // w > h
		{0, 16, 6, 6, 21},
		{16, 16, 6, 21, 21},
		{16, 16, 21, 21, 21},
		{16, 16, 21, 21, 21},
		{16, 16, 21, 21, 21},
	}, { // w < h
		{0, 11, 11, 11, 11},
		{11, 11, 11, 11, 11},
		{6, 6, 21, 21, 21},
		{6, 21, 21, 21, 21},
		{21, 21, 21, 21, 21},
	},
}

// coefDequant carries the per-TXB dequantization inputs of dav1d's
// decode_coefs residual stage: dq_tbl[0..1], dq_shift and cf_max.
type coefDequant struct {
	dc, ac uint32
	shift  uint8
	cfMax  uint32
}

// coefDecodeResult is what dav1d's decode_coefs hands back to its caller: the
// 0-based eob, plus the level context and the number of coefficient rows the
// inverse transform has to read.
type coefDecodeResult struct {
	eob        int
	culLevel   uint8 // libaom-layout coefficient context byte
	activeRows int
}

// readCoefGolomb is dav1d's read_golomb (src/recon_tmpl.c).
func readCoefGolomb(c *entropy.Cursor) uint32 {
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

// coefLoCtx1D is dav1d's get_lo_ctx for TX_CLASS_H/V, with levels already
// transposed so the class direction runs along +1.
func coefLoCtx1D(levels *[coefLevelsLen]uint8, at int, y uint) (ctx uint, hiMag uint) {
	mag := uint(levels[at+1]) + uint(levels[at+16]) + uint(levels[at+2])
	hiMag = mag
	mag += uint(levels[at+3]) + uint(levels[at+4])
	offset := uint(26)
	if y > 1 {
		offset += 10
	} else {
		offset += y * 5
	}
	return offset + min((mag+64)>>7, 4), hiMag
}

// coefHiCtx is dav1d's hi_tok context tail: (mag > 12 ? 6 : (mag + 1) >> 1).
func coefHiCtx(mag uint) uint {
	mag &= 63
	if mag > 12 {
		return 6
	}
	return (mag + 1) >> 1
}

// decodeCoefTokens ports dav1d decode_coefs (src/recon_tmpl.c) from the eob
// read through the residual, sign and dequantization pass, for a non-skipped
// TXB without quantizer matrices. cf must be zero over the TXB's scan area on
// entry; it receives the dequantized coefficients in raster scan order
// (rc = x * scanHeight + y). During the token pass cf doubles as dav1d's
// token/next-rc chain: cf[rc] = tok << 11 | rc of the previous nonzero.
func decodeCoefTokens(c *entropy.Cursor, cdfs *CoeffCDFs, size TransformSize, geo *coeffGeometry, plane CoeffPlaneType, txtp transform.Type, class transform.Class, dcSignCtx uint8, cf *[1024]int32, levels *[coefLevelsLen]uint8, dq coefDequant) coefDecodeResult {
	txCtx := int(geo.txCtx)
	scanW := uint(geo.scanWidth)
	scanH := uint(geo.scanHeight)

	// find end-of-block (eob)
	is1D := uint8(0)
	if class != transform.Class2D {
		is1D = 1
	}
	eobSymbols := int(eobMultiSizeTable[size]) + 5
	eob := c.ReadCDFSymbolsUnchecked(eobFlagCDFKnown(cdfs, size, plane, is1D), eobSymbols)
	if eob > 1 {
		eobBin := eob - 2
		eobHiBit := c.ReadBinaryCDFUnchecked(&cdfs.EOBExtra[txCtx][plane][eobBin])
		eob = ((eobHiBit | 2) << uint(eobBin)) | int(c.ReadBitsTrusted(uint8(eobBin)))
	}
	// dav1d keeps cf zero between TXBs because its inverse transforms clear
	// what they read. goav1's consumers do not, so clear the scan area here;
	// a DCT_DCT DC-only TXB is reconstructed from cf[0] alone.
	if eob != 0 || txtp != transform.TypeDCTDCT {
		clear(cf[:scanW*scanH])
	}

	// base tokens
	eobCDF := &cdfs.CoeffBaseEOB[txCtx][plane]
	hiCDF := &cdfs.CoeffBR[geo.txBRCtx][plane]
	var rc, dcTok uint
	if eob != 0 {
		loCDF := &cdfs.CoeffBase[txCtx][plane]
		sz := uint(eobMultiSizeTable[size])
		ctx := 1
		if uint(eob) > 2<<sz {
			ctx++
		}
		if uint(eob) > 4<<sz {
			ctx++
		}
		eobTok := uint(c.ReadCDF3Unchecked(&eobCDF[ctx]))
		switch class {
		case transform.Class2D:
			rc, dcTok = decodeCoefs2D(c, loCDF, hiCDF, size, geo, eob, eobTok, cf, levels)
		case transform.ClassHoriz:
			rc, dcTok = decodeCoefs1D(c, loCDF, hiCDF, false, scanW, scanH, eob, eobTok, cf, levels)
		default:
			rc, dcTok = decodeCoefs1D(c, loCDF, hiCDF, true, scanW, scanH, eob, eobTok, cf, levels)
		}
	} else { // dc-only
		tokBR := uint(c.ReadCDF3Unchecked(&eobCDF[0]))
		dcTok = 1 + tokBR
		if tokBR == 2 {
			dcTok = 3 + uint(c.ReadCDF4HighTokenUnchecked(&hiCDF[0]))
		}
		rc = 0
	}

	// residual and sign
	rowMask := scanH - 1
	activeRows := uint(0)
	var culLevel uint
	dcSignLevel := uint8(0)
	if dcTok != 0 {
		dcSign := uint32(c.ReadBinaryCDFUnchecked(&cdfs.DCSign[plane][dcSignCtx]))
		dcDQ := dq.dc
		if dcSign == 0 {
			dcSignLevel = 2 << CoeffContextBits
		} else {
			dcSignLevel = 1 << CoeffContextBits
		}
		if dcTok == 15 {
			dcTok = uint(readCoefGolomb(c)) + 15
			dcTok &= 0xfffff
			dcDQ = ((dcDQ * uint32(dcTok)) & 0xffffff) >> dq.shift
			dcDQ = min(dcDQ, dq.cfMax+dcSign)
		} else {
			dcDQ = (dcDQ * uint32(dcTok)) >> dq.shift
		}
		culLevel = dcTok
		if dcSign != 0 {
			cf[0] = -int32(dcDQ)
		} else {
			cf[0] = int32(dcDQ)
		}
		activeRows = 1
	}
	if rc != 0 {
		acDQ := dq.ac
		for {
			bit, refill := c.ReadBitNoRefillTrusted()
			if refill {
				c.RefillTrusted()
			}
			sign := uint32(bit)
			rcTok := uint(uint32(cf[rc&1023]))
			var tok uint
			var v uint32
			if rcTok >= 15<<11 {
				tok = uint(readCoefGolomb(c)) + 15
				tok &= 0xfffff
				v = ((acDQ * uint32(tok)) & 0xffffff) >> dq.shift
				v = min(v, dq.cfMax+sign)
			} else {
				tok = rcTok >> 11
				v = (acDQ * uint32(tok)) >> dq.shift
			}
			culLevel += tok
			if sign != 0 {
				cf[rc&1023] = -int32(v)
			} else {
				cf[rc&1023] = int32(v)
			}
			activeRows = max(activeRows, (rc&rowMask)+1)
			rc = rcTok & 0x3ff
			if rc == 0 {
				break
			}
		}
	}

	return coefDecodeResult{
		eob:        eob,
		culLevel:   uint8(min(culLevel, CoeffContextMask)) + dcSignLevel,
		activeRows: int(activeRows),
	}
}

// decodeCoefs2D is dav1d's DECODE_COEFS_CLASS(TX_CLASS_2D) together with the
// 2D lo/dc context setup from decode_coefs. It returns the chain head rc and
// the DC token.
func decodeCoefs2D(c *entropy.Cursor, loCDF *[CoeffBaseContexts]entropy.CDF, hiCDF *[CoeffBRContexts]entropy.CDF, size TransformSize, geo *coeffGeometry, eob int, eobTok uint, cf *[1024]int32, levels *[coefLevelsLen]uint8) (uint, uint) {
	offsets := &coefLoCtxOffsets[0]
	if geo.txWide {
		offsets = &coefLoCtxOffsets[1]
	} else if geo.txTall {
		offsets = &coefLoCtxOffsets[2]
	}
	scan := coeffScanTable[size][transform.Class2D]
	scan = scan[:eob+1]
	stride := uint(geo.scanHeight)
	shift := uint(log2Coef[geo.scanHeight])
	mask := stride - 1
	clear(levels[:stride*(uint(geo.scanWidth)+2)])

	// eob
	tok := eobTok + 1
	levelTok := tok * 0x41
	rc := uint(scan[eob])
	x, y := rc>>shift, rc&mask
	if eobTok == 2 {
		ctx := uint(7)
		if x|y > 1 {
			ctx = 14
		}
		tok = 3 + uint(c.ReadCDF4HighTokenUnchecked(&hiCDF[ctx]))
		levelTok = tok + 3<<6
	}
	cf[rc] = int32(tok << 11)
	levels[rc] = uint8(levelTok)
	rc = c.DecodeCoefsAC2D(scan[:eob], shift, stride, offsets, levels, cf, loCDF, hiCDF, rc) // ac
	// dc
	dcTok := uint(c.ReadCDF4Unchecked(&loCDF[0]))
	if dcTok == 3 {
		mag := uint(levels[1]) + uint(levels[stride]) + uint(levels[stride+1])
		dcTok = 3 + uint(c.ReadCDF4HighTokenUnchecked(&hiCDF[coefHiCtx(mag)]))
	}
	return rc, dcTok
}

// decodeCoefs1D is dav1d's DECODE_COEFS_CLASS(TX_CLASS_H) (vert == false) and
// DECODE_COEFS_CLASS(TX_CLASS_V) (vert == true). Both scan i directly; the
// levels array is transposed to stride 16 so the class direction runs along
// +1, and TX_CLASS_V maps i to rc = x << shift2 | y.
func decodeCoefs1D(c *entropy.Cursor, loCDF *[CoeffBaseContexts]entropy.CDF, hiCDF *[CoeffBRContexts]entropy.CDF, vert bool, scanW, scanH uint, eob int, eobTok uint, cf *[1024]int32, levels *[coefLevelsLen]uint8) (uint, uint) {
	const stride = 16
	var shift, shift2, mask uint
	if vert {
		shift, shift2, mask = uint(log2Coef[scanW]), uint(log2Coef[scanH]), scanW-1
		clear(levels[:stride*(scanW+2)])
	} else {
		shift, shift2, mask = uint(log2Coef[scanH]), 0, scanH-1
		clear(levels[:stride*(scanH+2)])
	}

	// eob
	tok := eobTok + 1
	levelTok := tok * 0x41
	x, y := uint(eob)&mask, uint(eob)>>shift
	rc := uint(eob)
	if vert {
		rc = x<<shift2 | y
	}
	if eobTok == 2 {
		ctx := uint(7)
		if y != 0 {
			ctx = 14
		}
		tok = 3 + uint(c.ReadCDF4HighTokenUnchecked(&hiCDF[ctx]))
		levelTok = tok + 3<<6
	}
	cf[rc] = int32(tok << 11)
	levels[x*stride+y] = uint8(levelTok)
	for i := uint(eob - 1); i > 0; i-- { // ac
		x, y = i&mask, i>>shift
		rcI := i
		if vert {
			rcI = x<<shift2 | y
		}
		at := int(x*stride + y)
		ctx, hiMag := coefLoCtx1D(levels, at, y)
		tok = uint(c.ReadCDF4Unchecked(&loCDF[ctx]))
		if tok == 3 {
			hiCtx := uint(7)
			if y > 0 {
				hiCtx = 14
			}
			hiCtx += coefHiCtx(hiMag)
			tok = 3 + uint(c.ReadCDF4HighTokenUnchecked(&hiCDF[hiCtx]))
			levels[at] = uint8(tok + 3<<6)
			cf[rcI] = int32(tok<<11 | rc)
			rc = rcI
		} else {
			t := uint32(tok) * 0x17ff41
			levels[at] = uint8(t)
			t = (t >> 9) & (uint32(rc) + ^uint32(0x7ff))
			if t != 0 {
				rc = rcI
			}
			cf[rcI] = int32(t)
		}
	}
	// dc
	ctx, hiMag := coefLoCtx1D(levels, 0, 0)
	dcTok := uint(c.ReadCDF4Unchecked(&loCDF[ctx]))
	if dcTok == 3 {
		dcTok = 3 + uint(c.ReadCDF4HighTokenUnchecked(&hiCDF[coefHiCtx(hiMag)]))
	}
	return rc, dcTok
}

// log2Coef maps a coefficient scan dimension (4, 8, 16 or 32) to its log2.
var log2Coef = [33]uint8{4: 2, 8: 3, 16: 4, 32: 5}

// decodeCoefsDequantized runs decodeCoefTokens for a nonzero TXB whose plane
// has prepared dequant factors and publishes the result on req.dequant the way
// the tracked quantized reader does. It reports false, consuming nothing, when
// the target cannot take this TXB.
func (s *DecodeState) decodeCoefsDequantized(cdfs *CoeffCDFs, req *TXBDecodeRequest, txtp transform.Type, class transform.Class, scratch *LumaCoeffTreeScratch, geo *coeffGeometry, result *TXBDecodeResult) bool {
	target, txScale, dqMax := prepareTrackedCoeffDequant(req.dequant, req.dequantPlane, geo)
	if target == nil || req.DCSignContext >= 3 {
		return false
	}
	if len(target.Dst) < 1024 {
		return false
	}
	factors := &target.Factors[req.dequantPlane]
	c := s.Reader.Cursor()
	r := decodeCoefTokens(&c, cdfs, req.Size, geo, req.Plane, txtp, class, req.DCSignContext, (*[1024]int32)(target.Dst), &scratch.CoefLevels, coefDequant{
		dc:    uint32(factors[0]),
		ac:    uint32(factors[1]),
		shift: txScale,
		cfMax: uint32(dqMax),
	})
	c.CommitStateTo(&s.Reader)
	target.ActiveRows = r.activeRows
	target.TxScale = txScale
	target.decoded = true
	*result = TXBDecodeResult{
		EOB:      uint16(r.eob + 1),
		CulLevel: r.culLevel,
	}
	return true
}
