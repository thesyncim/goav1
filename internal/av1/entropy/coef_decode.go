package entropy

import (
	"encoding/binary"
	"math/bits"
)

// CoefCDFs is dav1d's CdfCoefContext (src/cdf.h): the coefficient CDFs of a
// tile, indexed by t_dim->ctx and the plane type (0 luma, 1 chroma).
type CoefCDFs struct {
	TXBSkip      [5][13]CDF
	EOBExtra     [5][2][9]CDF
	DCSign       [2][3]CDF
	CoeffBR      [5][2][21]CDF
	CoeffBase    [5][2][42]CDF
	CoeffBaseEOB [5][2][4]CDF
	EOBFlag16    [2][2]CDF
	EOBFlag32    [2][2]CDF
	EOBFlag64    [2][2]CDF
	EOBFlag128   [2][2]CDF
	EOBFlag256   [2][2]CDF
	EOBFlag512   [2][2]CDF
	EOBFlag1024  [2][2]CDF
}

// CoefQMatrices is one level and plane class of dav1d's dav1d_qm_tbl, per
// transform size, in coefficient (cf) order.
type CoefQMatrices [CoefSizes][]uint16

// CoefDequant is one plane's dav1d dq_tbl[0..1] together with cf_max and the
// plane's quantizer matrices (dav1d's f->qm[][plane]), nil when the plane
// dequantizes flat.
type CoefDequant struct {
	DC, AC, Max uint32
	QM          *CoefQMatrices
}

// CoefSizes is the number of transform sizes (dav1d's N_RECT_TX_SIZES).
const CoefSizes = 19

// CoefSize is the part of dav1d's TxfmInfo (src/tables.h) decode_coefs reads
// for one transform size, with the size's 2D scan (dav1d_scans).
type CoefSize struct {
	Scan []int16
	// Size indexes CoefQMatrices.
	Size uint8
	// W4, H4 are t_dim->w and t_dim->h in 4-sample units; Min and Max are
	// t_dim->min and t_dim->max (TX_4X4 = 0 .. TX_64X64 = 4).
	W4, H4, Min, Max uint8
	// Ctx is t_dim->ctx; Slw and Slh are imin(t_dim->lw, TX_32X32) and
	// imin(t_dim->lh, TX_32X32).
	Ctx, Slw, Slh uint8
	// LoCtx selects dav1d_lo_ctx_offsets: 0 square, 1 wider, 2 taller.
	LoCtx uint8
}

// CoefState is what dav1d's decode_coefs reads from the tile state, the frame
// header and Av1Block: the tile's coefficient and transform-type CDFs and a/l
// coefficient contexts, the levels scratch, and the bound block's flags.
type CoefState struct {
	CDF     *CoefCDFs
	TxIntra *[3][4][13]CDF // ts->cdf.m.txtp_intra1 ([1]) and txtp_intra2 ([2])
	TxInter *[4][4]CDF     // txtp_inter1 ([1]), txtp_inter2 ([2]), txtp_inter3 ([3])
	A, L    *[3][32]uint8  // t->a->lcoef/ccoef and t->l, per plane
	Levels  *[CoefLevelsLen]uint8

	// BW4, BH4 are the plane block size in 4x4 units, luma ([0]) and chroma.
	BW4, BH4 [2]uint8
	// YMode is y_mode_nofilt; UVTxtp is dav1d_txtp_from_uvmode[uv_mode].
	YMode, UVTxtp uint8
	Intra         bool
	Lossless      bool
	QZero         bool // !segmentation.qidx[seg_id]
	Reduced       bool // reduced_txtp_set
}

// Transform types as dav1d numbers them (enum TxfmType).
const (
	coefDCTDCT    = 0
	coefIDTX      = 9
	coefVDCT      = 10
	coefHDCT      = 11
	coefVADST     = 12
	coefHADST     = 13
	coefVFlipADST = 14
	coefHFlipADST = 15
	coefWHTWHT    = 16

	coefClass2D = 0
	coefClassH  = 1
	coefClassV  = 2
)

// coefTxClass is dav1d_tx_type_class.
var coefTxClass = [32]uint8{
	coefVDCT: coefClassV, coefHDCT: coefClassH,
	coefVADST: coefClassV, coefHADST: coefClassH,
	coefVFlipADST: coefClassV, coefHFlipADST: coefClassH,
}

// coefTxtpIntra2, coefTxtpIntra1, coefTxtpInter2 and coefTxtpInter1 are
// dav1d_tx_types_per_set at offsets 0, 5, 12 and 24.
var (
	coefTxtpIntra2 = [16]uint8{9, 0, 3, 1, 2}
	coefTxtpIntra1 = [16]uint8{9, 0, 10, 11, 3, 1, 2}
	coefTxtpInter2 = [16]uint8{9, 10, 11, 0, 1, 2, 4, 5, 3, 6, 7, 8}
	coefTxtpInter1 = [16]uint8{9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 4, 5, 3, 6, 7, 8}
)

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

// coefSkipCtxs is the libaom-layout get_skip_ctx table for a luma TXB smaller
// than its block, by [min(above magnitude, 4)][min(left magnitude, 4)].
var coefSkipCtxs = [5][5]uint8{
	{1, 2, 2, 2, 3},
	{2, 4, 4, 4, 5},
	{2, 4, 4, 4, 5},
	{2, 4, 4, 4, 5},
	{3, 5, 5, 5, 6},
}

// The a/l coefficient context byte (libaom's cul_level layout): the level
// magnitude in the low coefCtxBits bits, then a negative-DC flag and a
// positive-DC flag.
const (
	coefCtxBits = 3
	coefCtxMask = 1<<coefCtxBits - 1
)

// coefBool splits the range for an adaptive bool with bound bnd: bit 1 keeps
// [0, bnd), bit 0 the rest. Both outcomes are computed and selected.
func coefBool(dif uint64, rng, bnd uint32) (uint, uint64, uint32) {
	win := uint64(bnd) << (ecWindow - 16)
	dt, rt := dif-win, rng-bnd
	bit, r := uint(1), bnd
	if dif >= win {
		dif, r, bit = dt, rt, 0
	}
	return bit, dif, r
}

// coefB2U is b as 0 or 1, a CSET rather than a branch.
func coefB2U(b bool) uint {
	var v uint
	if b {
		v = 1
	}
	return v
}

// coefNorm is dav1d's ctx_norm without the refill.
func coefNorm(dif uint64, rng uint32, cnt int32) (uint64, uint32, int32) {
	sh := normShift16(rng)
	return ((dif + 1) << uint(sh)) - 1, rng << uint(sh), cnt - sh
}

// coefRefill is dav1d's ctx_refill with the range state in registers and the
// byte position in c: refillState's whole-window fast path, small enough to
// inline into the hot entropy loops.
func (c *Cursor) coefRefill(dif uint64, cnt int32) (uint64, int32) {
	pos := int(c.pos)
	shift := int32(ecWindow) - 9 - (cnt + 15)
	if uint32(shift) <= ecWindow-9 && len(c.src)-pos >= 8 {
		u := binary.BigEndian.Uint64(c.src[pos:])
		dif ^= (u >> uint(56-shift)) &^ (1<<uint(shift&7) - 1)
		n := shift>>3 + 1
		c.pos = uint32(pos + int(n))
		return dif, cnt + 8*n
	}
	return c.coefRefillSlow(dif, cnt)
}

//go:noinline
func (c *Cursor) coefRefillSlow(dif uint64, cnt int32) (uint64, int32) {
	pos, dif, cnt, tellOffs := refillStateSlow(c.src, int(c.pos), dif, cnt, int32(c.tellOffs))
	c.pos = uint32(pos)
	c.tellOffs = int16(tellOffs)
	return dif, cnt
}

func (c *Cursor) coefTell(cnt int32) int {
	return readerTell(int(c.pos), cnt, int32(c.tellOffs))
}

// CoefUVInterTxtp is dav1d's get_uv_inter_txtp (src/env.h): the transform
// type of an inter chroma TXB with t_dim->max and t_dim->min uvMax, uvMin from
// the co-located luma type.
func CoefUVInterTxtp(uvMax, uvMin, ytxtp uint8) uint8 {
	if uvMax == 3 {
		if ytxtp == coefIDTX {
			return coefIDTX
		}
		return coefDCTDCT
	}
	if uvMin == 2 {
		switch ytxtp {
		case coefHFlipADST, coefVFlipADST, coefHADST, coefVADST:
			return coefDCTDCT
		}
	}
	return ytxtp
}

// coefSymbol is the symbol search and CDF adaptation of dav1d's
// msac_decode_symbol_adapt8/16 for an n-symbol CDF (5 <= n <= 16): it returns
// the symbol and its [lower, upper) interval of rng.
func coefSymbol(v *[MaxSymbols + 1]uint16, n int, rng, coded uint32, update bool) (int, uint32, uint32) {
	last := (n - 1) & (MaxSymbols - 1)
	if last < 8 {
		return coefSymbol8(v, last, rng, coded, update)
	}
	if symbol16SIMD {
		return decodeSymbol16SIMD(v, last, rng, coded, update)
	}
	rngHi := rng >> 8
	upper, lower := rng, uint32(0)
	sym := 0
	minP := uint32(ecMinProb * last)
	for i := 0; i < last; i++ {
		b := ((rngHi * (uint32(v[i]) >> ecProbShift)) >> (7 - ecProbShift)) + minP
		minP -= ecMinProb
		if coded < b {
			upper, sym = b, i+1
		} else {
			lower = max(lower, b)
		}
	}
	if update {
		count := v[last+1]
		rate := uint(5 + ((count >> 4) & 3))
		for i := 0; i < last; i++ {
			v[i] = uint16(adaptCDFEntry(uint32(v[i]), rate, i < sym))
		}
		if count < MaxCDFCount {
			v[last+1] = count + 1
		}
	}
	return sym, lower, upper
}

// coefSym8Lanes is, per last = n_symbols - 1 (up to 7), the EC_MIN_PROB *
// (last - i) offsets and the lane mask of the seven bounds coefSymbol8
// evaluates: lanes at and past last are masked to a zero bound and keep their
// CDF entry.
var coefSym8Lanes = func() (t [8][2][7]uint32) {
	for last := range 8 {
		for i := 0; i < last && i < 7; i++ {
			t[last][0][i] = uint32(ecMinProb * (last - i))
			t[last][1][i] = ^uint32(0)
		}
	}
	return t
}()

// coefSymbol8 is dav1d's arm64 msac_decode_symbol_adapt8 in scalar form for
// last = n_symbols - 1 <= 7: every bound is computed (masked lanes are zero,
// which the coded value never falls below), the symbol resolves through
// conditional selects, and the CDF adapts under the same mask.
func coefSymbol8(v *[MaxSymbols + 1]uint16, last int, rng, coded uint32, update bool) (int, uint32, uint32) {
	last &= 7
	t := &coefSym8Lanes[last]
	mp, mk := &t[0], &t[1]
	rngHi := rng >> 8
	c0, c1, c2, c3 := uint32(v[0]), uint32(v[1]), uint32(v[2]), uint32(v[3])
	c4, c5, c6 := uint32(v[4]), uint32(v[5]), uint32(v[6])
	l0 := cdfBound(rngHi, c0, mp[0]) & mk[0]
	l1 := cdfBound(rngHi, c1, mp[1]) & mk[1]
	l2 := cdfBound(rngHi, c2, mp[2]) & mk[2]
	l3 := cdfBound(rngHi, c3, mp[3]) & mk[3]
	l4 := cdfBound(rngHi, c4, mp[4]) & mk[4]
	l5 := cdfBound(rngHi, c5, mp[5]) & mk[5]
	l6 := cdfBound(rngHi, c6, mp[6]) & mk[6]
	upper, lower, sym := rng, l0, 0
	if coded < l0 {
		upper, lower, sym = l0, l1, 1
	}
	if coded < l1 {
		upper, lower, sym = l1, l2, 2
	}
	if coded < l2 {
		upper, lower, sym = l2, l3, 3
	}
	if coded < l3 {
		upper, lower, sym = l3, l4, 4
	}
	if coded < l4 {
		upper, lower, sym = l4, l5, 5
	}
	if coded < l5 {
		upper, lower, sym = l5, l6, 6
	}
	if coded < l6 {
		upper, lower, sym = l6, 0, 7
	}
	if update {
		count := v[last+1]
		rate := uint(5 + ((count >> 4) & 3))
		v[0] = uint16(c0 ^ ((adaptCDFEntry(c0, rate, sym > 0) ^ c0) & mk[0]))
		v[1] = uint16(c1 ^ ((adaptCDFEntry(c1, rate, sym > 1) ^ c1) & mk[1]))
		v[2] = uint16(c2 ^ ((adaptCDFEntry(c2, rate, sym > 2) ^ c2) & mk[2]))
		v[3] = uint16(c3 ^ ((adaptCDFEntry(c3, rate, sym > 3) ^ c3) & mk[3]))
		v[4] = uint16(c4 ^ ((adaptCDFEntry(c4, rate, sym > 4) ^ c4) & mk[4]))
		v[5] = uint16(c5 ^ ((adaptCDFEntry(c5, rate, sym > 5) ^ c5) & mk[5]))
		v[6] = uint16(c6 ^ ((adaptCDFEntry(c6, rate, sym > 6) ^ c6) & mk[6]))
		if count < MaxCDFCount {
			v[last+1] = count + 1
		}
	}
	return sym, lower, upper
}

// coefHiTok is dav1d_msac_decode_hi_tok (src/msac.c) with the range state in
// registers: up to four reads of one four-symbol CDF, summed, stopping at the
// first symbol below 3. It returns the token sum (0..12).
func (c *Cursor) coefHiTok(v *[MaxSymbols + 1]uint16, update bool, dif uint64, rng uint32, cnt int32) (uint, uint64, uint32, int32) {
	c0, c1, c2 := uint32(v[0]), uint32(v[1]), uint32(v[2])
	count := v[4]
	tok := uint(0)
	for range 4 {
		rngHi := rng >> 8
		coded := uint32(dif >> (ecWindow - 16))
		sym, lower, upper := selectSymbol4(rng, coded, cdfBound(rngHi, c0, 3*ecMinProb), cdfBound(rngHi, c1, 2*ecMinProb), cdfBound(rngHi, c2, ecMinProb))
		if traceEntropyReads {
			traceCDFRead(uint16(c0), 4, dif, rng, c.coefTell(cnt))
		}
		dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			rate := uint(5 + ((count >> 4) & 3))
			c0 = adaptCDFEntry(c0, rate, sym > 0)
			c1 = adaptCDFEntry(c1, rate, sym > 1)
			c2 = adaptCDFEntry(c2, rate, sym > 2)
			if count < MaxCDFCount {
				count++
			}
		}
		tok += sym
		if sym < 3 {
			break
		}
	}
	if update {
		v[0], v[1], v[2], v[4] = uint16(c0), uint16(c1), uint16(c2), count
	}
	return tok, dif, rng, cnt
}

// coefGolomb is dav1d's read_golomb with the range state in registers.
func (c *Cursor) coefGolomb(dif uint64, rng uint32, cnt int32) (uint, uint64, uint32, int32) {
	c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
	v := uint(c.ReadCoefGolomb())
	return v, c.dif, uint32(c.rng), int32(c.cnt)
}

// coefCtxWords reads the n context bytes at row[x:] (n = 1, 2, 4, 8 or 16) as
// two words with loads of the transform side's width, the widths the a/l
// context stores use, so the loads forward from them.
func coefCtxWords(row *[32]uint8, x, n uint) (lo, hi uint64) {
	switch n {
	case 1:
		return uint64(row[x]), 0
	case 2:
		return uint64(binary.LittleEndian.Uint16(row[x:])), 0
	case 4:
		return uint64(binary.LittleEndian.Uint32(row[x:])), 0
	case 8:
		return binary.LittleEndian.Uint64(row[x:]), 0
	}
	return binary.LittleEndian.Uint64(row[x:]), binary.LittleEndian.Uint64(row[x+8:])
}

// coefCtxOr folds two context words into the OR of their bytes.
func coefCtxOr(lo, hi uint64) uint8 {
	v := lo | hi
	v |= v >> 32
	v |= v >> 16
	v |= v >> 8
	return uint8(v)
}

// coefCtxSign is the DC-sign sum (positive minus negative entries) of two
// context words.
func coefCtxSign(lo, hi uint64) int {
	const ones = 0x0101010101010101
	neg := (((lo >> coefCtxBits) & ones) + ((hi >> coefCtxBits) & ones)) * ones >> 56
	pos := (((lo >> (coefCtxBits + 1)) & ones) + ((hi >> (coefCtxBits + 1)) & ones)) * ones >> 56
	return int(pos) - int(neg)
}

// coefCtxSet is dav1d's dav1d_memset_likely_pow2 of the TXB's context byte
// into row[x:x+n] for a TXB side n (1, 2, 4, 8 or 16) inside the frame: one
// or two stores of the side's width.
func coefCtxSet(row *[32]uint8, x, n uint, value uint8) {
	v := uint64(value) * 0x0101010101010101
	switch n {
	case 1:
		row[x] = value
	case 2:
		binary.LittleEndian.PutUint16(row[x:], uint16(v))
	case 4:
		binary.LittleEndian.PutUint32(row[x:], uint32(v))
	case 8:
		binary.LittleEndian.PutUint64(row[x:], v)
	default:
		binary.LittleEndian.PutUint64(row[x:], v)
		binary.LittleEndian.PutUint64(row[x+8:], v)
	}
}

// coefCtxSetClipped is coefCtxSet for a TXB crossing the frame edge: the
// visible units take value, the rest zero.
func coefCtxSetClipped(row *[32]uint8, x, n, visible uint, value uint8) {
	run := row[x : x+n]
	for k := range run {
		if uint(k) < visible {
			run[k] = value
		} else {
			run[k] = 0
		}
	}
}

// clearCoefLevels zeroes levels[:n] in 64-byte blocks; CoefLevelsLen leaves
// room for the rounding.
func clearCoefLevels(levels *[CoefLevelsLen]uint8, n uint) {
	for i := uint(0); i < n && i <= CoefLevelsLen-64; i += 64 {
		*(*[64]uint8)(levels[i : i+64]) = [64]uint8{}
	}
}

// DecodeCoefs is dav1d's decode_coefs (src/recon_tmpl.c) for the plane TXB of
// transform size sz at a/l offsets x4, y4, followed by the a/l context write
// of its caller (read_coef_tree / recon_b_*) over the frame-visible visW x
// visH units. The range decoder state stays in registers from the skip flag
// to the last sign. txtp is the co-located luma transform type for an inter
// chroma TXB (dav1d's *txtp in), else ignored.
//
// A coded TXB is dequantized with dq into cf in raster scan order
// (rc = x * scan height + y); cf must be zero over the scan area on entry.
// It returns dav1d's eob (-1 for an all-zero TXB), the transform type and the
// extent of the nonzero coefficients, rows | cols << 8. The caller has proved
// x4 + sz.W4 and y4 + sz.H4 are at most 32.
func (r *Reader) DecodeCoefs(s *CoefState, plane uint, sz *CoefSize, cf *[1024]int32, dq *CoefDequant, x4, y4, visW, visH, txtp uint) (eob int, txtpOut uint, ext uint) {
	c := (*Cursor)(r)
	update := c.allowCDFUpdate
	dif, rng, cnt := c.dif, uint32(c.rng), int32(c.cnt)
	cdfs := s.CDF
	chroma := uint(0)
	if plane != 0 {
		chroma = 1
	}
	if plane >= 3 {
		plane = 0
	}
	a, l := &s.A[plane], &s.L[plane]
	w4, h4 := uint(sz.W4), uint(sz.H4)
	tctx := uint(sz.Ctx)
	if tctx >= 5 {
		tctx = 0
	}

	// does this block have any non-zero coefficients (get_skip_ctx)
	sctx := uint(0)
	bw4, bh4 := uint(s.BW4[chroma]), uint(s.BH4[chroma])
	aLo, aHi := coefCtxWords(a, x4, w4)
	lLo, lHi := coefCtxWords(l, y4, h4)
	if chroma == 0 {
		if bw4 != w4 || bh4 != h4 {
			la := min(coefCtxOr(aLo, aHi)&coefCtxMask, 4)
			ll := min(coefCtxOr(lLo, lHi)&coefCtxMask, 4)
			sctx = uint(coefSkipCtxs[la][ll])
		}
	} else {
		sctx = 7
		if bw4*bh4 > w4*h4 {
			sctx = 10
		}
		sctx += coefB2U(aLo|aHi != 0) + coefB2U(lLo|lHi != 0)
	}
	skip := &cdfs.TXBSkip[tctx][sctx].values
	bnd := ((rng>>8)*(uint32(skip[0])>>ecProbShift))>>(7-ecProbShift) + ecMinProb
	if traceEntropyReads {
		traceCDFRead(skip[0], 2, dif, rng, c.coefTell(cnt))
	}
	allSkip, dif, rng := coefBool(dif, rng, bnd)
	dif, rng, cnt = coefNorm(dif, rng, cnt)
	if cnt < 0 {
		dif, cnt = c.coefRefill(dif, cnt)
	}
	if update {
		updateCDF2(skip, int(allSkip))
	}
	if allSkip != 0 {
		c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
		if visW >= w4 {
			coefCtxSet(a, x4, w4, 0)
		} else {
			coefCtxSetClipped(a, x4, w4, visW, 0)
		}
		if visH >= h4 {
			coefCtxSet(l, y4, h4, 0)
		} else {
			coefCtxSetClipped(l, y4, h4, visH, 0)
		}
		return -1, coefDCTDCT, 0
	}

	// transform type (chroma: derived, luma: explicitly coded)
	intra := uint(0)
	if s.Intra {
		intra = 1
	}
	switch {
	case s.Lossless:
		txtp = coefWHTWHT
	case uint(sz.Max)+intra >= 4:
		txtp = coefDCTDCT
	case chroma != 0:
		if s.Intra {
			txtp = uint(s.UVTxtp)
		} else {
			txtp = uint(CoefUVInterTxtp(sz.Max, sz.Min, uint8(txtp)))
		}
	case s.QZero:
		txtp = coefDCTDCT
	default:
		sq := uint(sz.Min) & 3
		var v *[MaxSymbols + 1]uint16
		var n int
		var set *[16]uint8
		ym := s.YMode
		if ym >= 13 {
			ym = 0
		}
		switch {
		case s.Intra && (s.Reduced || sq == 2):
			v, n, set = &s.TxIntra[2][sq][ym].values, 5, &coefTxtpIntra2
		case s.Intra:
			v, n, set = &s.TxIntra[1][sq][ym].values, 7, &coefTxtpIntra1
		case s.Reduced || sz.Max == 3:
			v = &s.TxInter[3][sq].values
		case sq == 2:
			v, n, set = &s.TxInter[2][sq].values, 12, &coefTxtpInter2
		default:
			v, n, set = &s.TxInter[1][sq].values, 16, &coefTxtpInter1
		}
		if traceEntropyReads {
			traceCDFRead(v[0], max(n, 2), dif, rng, c.coefTell(cnt))
		}
		var lower, upper uint32
		if n == 0 {
			// txtp_inter3: idx ? DCT_DCT : IDTX
			bnd := ((rng>>8)*(uint32(v[0])>>ecProbShift))>>(7-ecProbShift) + ecMinProb
			var idx uint
			idx, dif, upper = coefBool(dif, rng, bnd)
			if update {
				updateCDF2(v, int(idx))
			}
			txtp = (idx - 1) & coefIDTX
		} else {
			var idx int
			idx, lower, upper = coefSymbol(v, n, rng, uint32(dif>>(ecWindow-16)), update)
			dif -= uint64(lower) << (ecWindow - 16)
			upper -= lower
			txtp = uint(set[idx&15])
		}
		dif, rng, cnt = coefNorm(dif, upper, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
	}
	txtp &= 31

	// find end-of-block (eob)
	class := uint(coefTxClass[txtp])
	is1D := uint(0)
	if class != coefClass2D {
		is1D = 1
	}
	slw, slh := uint(sz.Slw)&3, uint(sz.Slh)&3
	tx2dszctx := slw + slh
	var eobBin *[MaxSymbols + 1]uint16
	switch tx2dszctx {
	case 0:
		eobBin = &cdfs.EOBFlag16[chroma][is1D].values
	case 1:
		eobBin = &cdfs.EOBFlag32[chroma][is1D].values
	case 2:
		eobBin = &cdfs.EOBFlag64[chroma][is1D].values
	case 3:
		eobBin = &cdfs.EOBFlag128[chroma][is1D].values
	case 4:
		eobBin = &cdfs.EOBFlag256[chroma][is1D].values
	case 5:
		eobBin = &cdfs.EOBFlag512[chroma][is1D].values
	default:
		eobBin = &cdfs.EOBFlag1024[chroma][is1D].values
	}
	if traceEntropyReads {
		traceCDFRead(eobBin[0], int(5+tx2dszctx), dif, rng, c.coefTell(cnt))
	}
	{
		sym, lower, upper := coefSymbol(eobBin, int(5+tx2dszctx), rng, uint32(dif>>(ecWindow-16)), update)
		dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		eob = sym
	}
	if eob > 1 {
		eobBinIdx := uint(eob - 2)
		hb := &cdfs.EOBExtra[tctx][chroma][eobBinIdx].values
		bnd := ((rng>>8)*(uint32(hb[0])>>ecProbShift))>>(7-ecProbShift) + ecMinProb
		if traceEntropyReads {
			traceCDFRead(hb[0], 2, dif, rng, c.coefTell(cnt))
		}
		var hi uint
		hi, dif, rng = coefBool(dif, rng, bnd)
		dif, rng, cnt = coefNorm(dif, rng, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			updateCDF2(hb, int(hi))
		}
		// dav1d_msac_decode_bools(eob_bin)
		bits := uint(0)
		for range eobBinIdx {
			split := ((rng >> 8) << 7) + ecMinProb
			if traceEntropyReads {
				traceBoolRead(CDFProbTop/2, dif, rng, c.coefTell(cnt))
			}
			var bit uint
			bit, dif, rng = coefBool(dif, rng, split)
			dif, rng, cnt = coefNorm(dif, rng, cnt)
			if cnt < 0 {
				dif, cnt = c.coefRefill(dif, cnt)
			}
			bits = bits<<1 | bit
		}
		eob = int(((hi|2)<<eobBinIdx)|bits) & 1023
	}

	// base tokens
	eobCDF := &cdfs.CoeffBaseEOB[tctx][chroma]
	hiCDF := &cdfs.CoeffBR[min(tctx, 3)][chroma]
	levels := s.Levels
	var rc, dcTok uint
	{
		// eob_base_tok: ctx 0 for a dc-only TXB, else
		// 1 + (eob > 2 << tx2dszctx) + (eob > 4 << tx2dszctx)
		ctx := coefB2U(eob != 0) + coefB2U(uint(eob) > 2<<tx2dszctx) + coefB2U(uint(eob) > 4<<tx2dszctx)
		v := &eobCDF[ctx&3].values
		c0, c1 := uint32(v[0]), uint32(v[1])
		rngHi := rng >> 8
		coded := uint32(dif >> (ecWindow - 16))
		b0 := cdfBound(rngHi, c0, 2*ecMinProb)
		b1 := cdfBound(rngHi, c1, ecMinProb)
		upper, lower, tok := rng, b0, uint(0)
		if coded < b0 {
			upper, lower, tok = b0, b1, 1
		}
		if coded < b1 {
			upper, lower, tok = b1, 0, 2
		}
		if traceEntropyReads {
			traceCDFRead(v[0], 3, dif, rng, c.coefTell(cnt))
		}
		dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			count := v[3]
			rate := uint(4 + ((count >> 4) & 3))
			v[0] = uint16(adaptCDFEntry(c0, rate, tok > 0))
			v[1] = uint16(adaptCDFEntry(c1, rate, tok > 1))
			if count < MaxCDFCount {
				v[3] = count + 1
			}
		}

		if eob == 0 { // dc-only
			dcTok = 1 + tok
			if tok == 2 {
				var t uint
				t, dif, rng, cnt = c.coefHiTok(&hiCDF[0].values, update, dif, rng, cnt)
				dcTok = 3 + t
			}
		} else {
			lo := &cdfs.CoeffBase[tctx][chroma]
			eobTok := tok
			switch class {
			case coefClass2D:
				loCtx := sz.LoCtx
				if loCtx >= 3 {
					loCtx = 0
				}
				offsets := &coefLoCtxOffsets[loCtx]
				scan := sz.Scan[:eob+1]
				stride := uint(4<<slh) & 63
				shift := slh + 2
				mask := stride - 1
				clearCols := min((4<<slw)+2, uint(eob)+3)
				clearCoefLevels(levels, stride*clearCols)

				// eob
				tok := eobTok + 1
				levelTok := tok * 0x41
				rc = uint(scan[eob]) & 1023
				if eobTok == 2 {
					x, y := rc>>shift, rc&mask
					ctx := uint(7)
					if x|y > 1 {
						ctx = 14
					}
					var t uint
					t, dif, rng, cnt = c.coefHiTok(&hiCDF[ctx].values, update, dif, rng, cnt)
					tok = 3 + t
					levelTok = tok + 3<<6
				}
				cf[rc] = int32(tok << 11)
				levels[rc] = uint8(levelTok)
				rc, dif, rng, cnt = c.coefAC2D(scan[:eob], stride, offsets, levels, cf, lo, hiCDF, rc, dif, rng, cnt) // ac
				// dc
				v := &lo[0].values
				c0, c1, c2 := uint32(v[0]), uint32(v[1]), uint32(v[2])
				rngHi := rng >> 8
				coded := uint32(dif >> (ecWindow - 16))
				tok, lower, upper := selectSymbol4(rng, coded, cdfBound(rngHi, c0, 3*ecMinProb), cdfBound(rngHi, c1, 2*ecMinProb), cdfBound(rngHi, c2, ecMinProb))
				if traceEntropyReads {
					traceCDFRead(uint16(c0), 4, dif, rng, c.coefTell(cnt))
				}
				dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
				if cnt < 0 {
					dif, cnt = c.coefRefill(dif, cnt)
				}
				if update {
					count := v[4]
					rate := uint(5 + ((count >> 4) & 3))
					v[0] = uint16(adaptCDFEntry(c0, rate, tok > 0))
					v[1] = uint16(adaptCDFEntry(c1, rate, tok > 1))
					v[2] = uint16(adaptCDFEntry(c2, rate, tok > 2))
					if count < MaxCDFCount {
						count++
					}
					v[4] = count
				}
				dcTok = tok
				if tok == 3 {
					mag := uint(levels[1]) + uint(levels[stride]) + uint(levels[stride+1])
					mag &= 63
					hiCtx := uint(6)
					if mag <= 12 {
						hiCtx = (mag + 1) >> 1
					}
					var t uint
					t, dif, rng, cnt = c.coefHiTok(&hiCDF[hiCtx].values, update, dif, rng, cnt)
					dcTok = 3 + t
				}
			default:
				// TX_CLASS_H and TX_CLASS_V scan i directly; levels is
				// transposed to stride 16 so the class direction runs
				// along +1, and TX_CLASS_V maps i to rc = x << shift2 | y.
				const stride uint = 16
				vert := class == coefClassV
				var shift, shift2, mask uint
				if vert {
					shift, shift2, mask = slw+2, slh+2, (4<<slw)-1
					clearCols := min((4<<slw)+2, uint(eob)+3)
				clearCoefLevels(levels, stride*clearCols)
				} else {
					shift, shift2, mask = slh+2, 0, (4<<slh)-1
					clearCoefLevels(levels, stride*((4<<slh)+2))
				}
				mask &= 15

				rc, dcTok, dif, rng, cnt = c.coefAC1D(vert, shift, shift2, mask, uint(eob), eobTok, levels, cf, lo, hiCDF, dif, rng, cnt)
			}
		}
	}

	// residual and sign
	var qm []uint16
	if dq.QM != nil && txtp < coefIDTX {
		qm = dq.QM[sz.Size%CoefSizes]
	}
	dqShift := uint(max(sz.Ctx, 2)-2) & 3
	cfMax := dq.Max
	rowMask := uint(4<<slh) - 1
	colShift := (slh + 2) & 63
	var culLevel, rows, cols uint
	dcSignLevel := uint8(0)
	if dcTok != 0 {
		// get_dc_sign_ctx
		ds := coefCtxSign(aLo, aHi) + coefCtxSign(lLo, lHi)
		dctx := coefB2U(ds != 0) + coefB2U(ds > 0)
		v := &cdfs.DCSign[chroma][dctx].values
		bnd := ((rng>>8)*(uint32(v[0])>>ecProbShift))>>(7-ecProbShift) + ecMinProb
		if traceEntropyReads {
			traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
		}
		var dcSign uint
		dcSign, dif, rng = coefBool(dif, rng, bnd)
		dif, rng, cnt = coefNorm(dif, rng, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			updateCDF2(v, int(dcSign))
		}
		dcSignLevel = uint8(2-dcSign) << coefCtxBits
		dcDQ := dq.DC
		if qm != nil {
			dcDQ = (dcDQ*uint32(qm[0]) + 16) >> 5
		}
		if dcTok == 15 {
			var g uint
			g, dif, rng, cnt = c.coefGolomb(dif, rng, cnt)
			dcTok = (g + 15) & 0xfffff
			dcDQ = (dcDQ * uint32(dcTok)) & 0xffffff
			dcDQ = min(dcDQ>>dqShift, cfMax+uint32(dcSign))
		} else {
			dcDQ = (dcDQ * uint32(dcTok)) >> dqShift
			if qm != nil {
				dcDQ = min(dcDQ, cfMax+uint32(dcSign))
			}
		}
		culLevel = dcTok
		m := -int32(dcSign)
		cf[0] = (int32(dcDQ) ^ m) - m
		rows, cols = 1, 1
	}
	if rc != 0 {
		acDQ := dq.AC
		for {
			// dav1d_msac_decode_bool_equi
			split := ((rng >> 8) << 7) + ecMinProb
			if traceEntropyReads {
				traceBoolRead(CDFProbTop/2, dif, rng, c.coefTell(cnt))
			}
			var sign uint
			sign, dif, rng = coefBool(dif, rng, split)
			sh := uint(normShift16(rng)) & 3
			cnt -= int32(sh)
			dif = ((dif + 1) << sh) - 1
			rng <<= sh
			if cnt < 0 {
				dif, cnt = c.coefRefill(dif, cnt)
			}

			rc &= 1023
			rcTok := uint(uint32(cf[rc]))
			f := acDQ
			if qm != nil {
				f = (f*uint32(qm[rc]) + 16) >> 5
			}
			var tok uint
			var v uint32
			if rcTok >= 15<<11 {
				var g uint
				g, dif, rng, cnt = c.coefGolomb(dif, rng, cnt)
				tok = (g + 15) & 0xfffff
				v = min(((f*uint32(tok))&0xffffff)>>dqShift, cfMax+uint32(sign))
			} else {
				tok = rcTok >> 11
				v = (f * uint32(tok)) >> dqShift
				if qm != nil {
					v = min(v, cfMax+uint32(sign))
				}
			}
			culLevel += tok
			m := -int32(sign)
			cf[rc] = (int32(v) ^ m) - m
			rows = max(rows, (rc&rowMask)+1)
			cols = max(cols, (rc>>colShift)+1)
			rc = rcTok & 0x3ff
			if rc == 0 {
				break
			}
		}
	}
	c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)

	// context
	ctxv := uint8(min(culLevel, coefCtxMask)) + dcSignLevel
	if visW >= w4 {
		coefCtxSet(a, x4, w4, ctxv)
	} else {
		coefCtxSetClipped(a, x4, w4, visW, ctxv)
	}
	if visH >= h4 {
		coefCtxSet(l, y4, h4, ctxv)
	} else {
		coefCtxSetClipped(l, y4, h4, visH, ctxv)
	}
	return eob, txtp, rows | cols<<8
}

// coefAC2D is the AC loop of dav1d's DECODE_COEFS_CLASS(TX_CLASS_2D): scan[len
// - 1] down to scan[1], with the range state passed in registers. levels holds
// dav1d's packed level bytes (bits 0-5 token, 6-7 min(token, 3)) at rc with
// stride = scan height; cf receives the token chain, cf[rc] = tok << 11 | rc
// of the previous nonzero. It returns the chain head rc.
func (c *Cursor) coefAC2D(scan []int16, stride uint, offsets *[5][5]uint8, levels *[CoefLevelsLen]uint8, cf *[1024]int32, lo *[42]CDF, hi *[21]CDF, rc uint, dif uint64, rng uint32, cnt int32) (uint, uint64, uint32, int32) {
	_ = levels[0]
	_ = cf[0]
	_ = offsets[0]
	_ = lo[0]
	stride &= 63
	shift := uint(bits.TrailingZeros(stride)) & 63
	mask := stride - 1
	update := c.allowCDFUpdate
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
		ctx := uint(offsets[y][x])&31 + min((mag+64)>>7, 4)

		// dav1d_msac_decode_symbol_adapt4(lo_cdf[ctx], 3)
		values := &lo[ctx].values
		rngHi := rng >> 8
		coded := uint32(dif >> (ecWindow - 16))
		upper := rng
		c0 := values[0]
		c1 := values[1]
		c2 := values[2]
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
			traceCDFRead(c0, 4, dif, rng, c.coefTell(cnt))
		}
		dif -= uint64(lower) << (ecWindow - 16)
		rng = upper - lower
		sh := normShift16(rng)
		cnt -= sh
		dif = ((dif + 1) << uint(sh)) - 1
		rng <<= uint(sh)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			count := values[4]
			rate := uint(5 + ((count >> 4) & 3))
			// update_cdf with both adaptations of each entry computed and
			// the symbol selecting between them (dav1d's asm does the same
			// with a lane mask).
			c0u, c1u, c2u := uint(c0), uint(c1), uint(c2)
			n0 := c0u - c0u>>rate
			n1 := c1u - c1u>>rate
			n2 := c2u - c2u>>rate
			u0 := c0u + (CDFProbTop-c0u)>>rate
			u1 := c1u + (CDFProbTop-c1u)>>rate
			u2 := c2u + (CDFProbTop-c2u)>>rate
			if tok > 0 {
				n0 = u0
			}
			if tok > 1 {
				n1 = u1
			}
			if tok > 2 {
				n2 = u2
			}
			values[0] = uint16(n0)
			values[1] = uint16(n1)
			values[2] = uint16(n2)
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
			var t uint
			t, dif, rng, cnt = c.coefHiTok(&hi[hiCtx].values, update, dif, rng, cnt)
			tok = 3 + t
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
	return rc, dif, rng, cnt
}

// coefAC1D is dav1d's DECODE_COEFS_CLASS(TX_CLASS_H) (vert false) and
// DECODE_COEFS_CLASS(TX_CLASS_V) (vert true) from the eob token through DC,
// with the range state passed in registers. Both scan i directly; levels is
// transposed to stride 16 so the class direction runs along +1, and
// TX_CLASS_V maps i to rc = x << shift2 | y. It returns the token chain head
// rc and the DC token.
func (c *Cursor) coefAC1D(vert bool, shift, shift2, mask, eob, eobTok uint, levels *[CoefLevelsLen]uint8, cf *[1024]int32, lo *[42]CDF, hiCDF *[21]CDF, dif uint64, rng uint32, cnt int32) (rc, dcTok uint, _ uint64, _ uint32, _ int32) {
	const stride uint = 16
	update := c.allowCDFUpdate
	shift &= 15
	shift2 &= 15
	// eob
	tok := eobTok + 1
	levelTok := tok * 0x41
	x, y := eob&mask, (eob>>shift)&15
	rc = eob
	if vert {
		rc = x<<shift2 | y
	}
	rc &= 1023
	if eobTok == 2 {
		ctx := uint(7)
		if y != 0 {
			ctx = 14
		}
		var t uint
		t, dif, rng, cnt = c.coefHiTok(&hiCDF[ctx].values, update, dif, rng, cnt)
		tok = 3 + t
		levelTok = tok + 3<<6
	}
	cf[rc] = int32(tok << 11)
	levels[x*stride+y] = uint8(levelTok)
	// ac, then dc at i == 0
	for i := eob - 1; ; i-- {
		x, y = i&mask, (i>>shift)&15
		rcI := i
		if vert {
			rcI = x<<shift2 | y
		}
		rcI &= 1023
		at := x*stride + y
		mag := uint(levels[at+1]) + uint(levels[at+16]) + uint(levels[at+2])
		hiMag := mag
		mag += uint(levels[at+3]) + uint(levels[at+4])
		ctx := uint(26)
		if y > 1 {
			ctx += 10
		} else {
			ctx += y * 5
		}
		ctx += min((mag+64)>>7, 4)

		v := &lo[ctx].values
		c0, c1, c2 := uint32(v[0]), uint32(v[1]), uint32(v[2])
		rngHi := rng >> 8
		coded := uint32(dif >> (ecWindow - 16))
		tok, lower, upper := selectSymbol4(rng, coded, cdfBound(rngHi, c0, 3*ecMinProb), cdfBound(rngHi, c1, 2*ecMinProb), cdfBound(rngHi, c2, ecMinProb))
		if traceEntropyReads {
			traceCDFRead(uint16(c0), 4, dif, rng, c.coefTell(cnt))
		}
		dif, rng, cnt = coefNorm(dif-uint64(lower)<<(ecWindow-16), upper-lower, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			count := v[4]
			rate := uint(5 + ((count >> 4) & 3))
			v[0] = uint16(adaptCDFEntry(c0, rate, tok > 0))
			v[1] = uint16(adaptCDFEntry(c1, rate, tok > 1))
			v[2] = uint16(adaptCDFEntry(c2, rate, tok > 2))
			if count < MaxCDFCount {
				count++
			}
			v[4] = count
		}
		if i == 0 {
			dcTok = tok
			if tok == 3 {
				hiMag &= 63
				hiCtx := uint(6)
				if hiMag <= 12 {
					hiCtx = (hiMag + 1) >> 1
				}
				var t uint
				t, dif, rng, cnt = c.coefHiTok(&hiCDF[hiCtx].values, update, dif, rng, cnt)
				dcTok = 3 + t
			}
			break
		}
		if tok == 3 {
			hiCtx := uint(7)
			if y > 0 {
				hiCtx = 14
			}
			hiMag &= 63
			if hiMag > 12 {
				hiCtx += 6
			} else {
				hiCtx += (hiMag + 1) >> 1
			}
			var t uint
			t, dif, rng, cnt = c.coefHiTok(&hiCDF[hiCtx].values, update, dif, rng, cnt)
			tok = 3 + t
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
	return rc, dcTok, dif, rng, cnt
}
