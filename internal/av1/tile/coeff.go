package tile

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

const (
	CoeffQContexts        = 4
	CoeffTxSizeContexts   = 5
	CoeffPlaneTypes       = 2
	TXBSkipContexts       = 13
	EOBCoefContexts       = 9
	EOBBaseContexts       = 4
	CoeffBaseContexts     = 42
	CoeffBRContexts       = 21
	NumBaseLevels         = 2
	BRCDFSize             = 4
	CoeffBaseRange        = 4 * (BRCDFSize - 1)
	MaxBaseBRRange        = CoeffBaseRange + NumBaseLevels + 1
	CoeffContextBits      = 3
	CoeffContextMask      = (1 << CoeffContextBits) - 1
	maxEOBFlagContexts    = 2
	maxEOBFlagCumulative  = 10
	maxCoeffCumulativeLen = 10
	txPadHorizontal       = 4
)

type CoeffPlaneType uint8

const (
	CoeffPlaneY CoeffPlaneType = iota
	CoeffPlaneUV
)

type CoeffCDFs struct {
	TXBSkip      [CoeffTxSizeContexts][TXBSkipContexts]entropy.CDF
	EOBExtra     [CoeffTxSizeContexts][CoeffPlaneTypes][EOBCoefContexts]entropy.CDF
	DCSign       [CoeffPlaneTypes][3]entropy.CDF
	CoeffBR      [CoeffTxSizeContexts][CoeffPlaneTypes][CoeffBRContexts]entropy.CDF
	CoeffBase    [CoeffTxSizeContexts][CoeffPlaneTypes][CoeffBaseContexts]entropy.CDF
	CoeffBaseEOB [CoeffTxSizeContexts][CoeffPlaneTypes][EOBBaseContexts]entropy.CDF
	EOBFlag16    [CoeffPlaneTypes][maxEOBFlagContexts]entropy.CDF
	EOBFlag32    [CoeffPlaneTypes][maxEOBFlagContexts]entropy.CDF
	EOBFlag64    [CoeffPlaneTypes][maxEOBFlagContexts]entropy.CDF
	EOBFlag128   [CoeffPlaneTypes][maxEOBFlagContexts]entropy.CDF
	EOBFlag256   [CoeffPlaneTypes][maxEOBFlagContexts]entropy.CDF
	EOBFlag512   [CoeffPlaneTypes][maxEOBFlagContexts]entropy.CDF
	EOBFlag1024  [CoeffPlaneTypes][maxEOBFlagContexts]entropy.CDF
}

type TXBSkipRequest struct {
	Size    TransformSize
	Context int
}

type EOBRequest struct {
	Size            TransformSize
	Plane           CoeffPlaneType
	EOBMultiContext int
}

type EOBResult struct {
	Token      int
	Extra      int
	Position   int
	OffsetBits int
}

type CoeffTokenRequest struct {
	Size    TransformSize
	Plane   CoeffPlaneType
	Context int
}

type TXBDecodeRequest struct {
	Size            TransformSize
	Plane           CoeffPlaneType
	Class           transform.Class
	TXBSkipContext  int
	TXBSkipKnown    bool
	TXBSkip         bool
	DCSignContext   int
	EOBMultiContext int
}

type TXBDecodeResult struct {
	EOB         int
	MaxScanLine int
	CulLevel    int
	AllZero     bool
}

var eobGroupStart = [12]int{0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513}
var eobOffsetBits = [12]int{0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
var eobToPosSmall = [33]int{
	0, 1, 2,
	3, 3,
	4, 4, 4, 4,
	5, 5, 5, 5, 5, 5, 5, 5,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
}
var eobToPosLarge = [17]int{
	6,
	7,
	8, 8,
	9, 9, 9, 9,
	10, 10, 10, 10, 10, 10, 10, 10,
	11,
}
var eobMultiSizeTable = [transformSizeCount]int{
	TransformSize4x4:   0,
	TransformSize8x8:   2,
	TransformSize16x16: 4,
	TransformSize32x32: 6,
	TransformSize64x64: 6,
	TransformSize4x8:   1,
	TransformSize8x4:   1,
	TransformSize8x16:  3,
	TransformSize16x8:  3,
	TransformSize16x32: 5,
	TransformSize32x16: 5,
	TransformSize32x64: 6,
	TransformSize64x32: 6,
	TransformSize4x16:  2,
	TransformSize16x4:  2,
	TransformSize8x32:  4,
	TransformSize32x8:  4,
	TransformSize16x64: 5,
	TransformSize64x16: 5,
}

func CoeffQContext(baseQIndex uint8) int {
	if baseQIndex <= 20 {
		return 0
	}
	if baseQIndex <= 60 {
		return 1
	}
	if baseQIndex <= 120 {
		return 2
	}
	return 3
}

func (plane CoeffPlaneType) Valid() bool {
	return plane < CoeffPlaneTypes
}

func CoeffPlaneTypeForPlane(plane int) (CoeffPlaneType, error) {
	switch plane {
	case 0:
		return CoeffPlaneY, nil
	case 1, 2:
		return CoeffPlaneUV, nil
	default:
		return 0, ErrInvalidDecodeState
	}
}

// CoeffTransformSizeContext ports libaom's get_txsize_entropy_ctx().
func CoeffTransformSizeContext(size TransformSize) (int, error) {
	dims, ok := size.Dimensions()
	if !ok {
		return 0, ErrInvalidDecodeState
	}
	return int(dims.Min+dims.Max+1) >> 1, nil
}

// EOBMultiSize ports libaom's txsize_log2_minus4[] table.
func EOBMultiSize(size TransformSize) (int, error) {
	if !size.Valid() {
		return 0, ErrInvalidDecodeState
	}
	return eobMultiSizeTable[size], nil
}

func RecEOBPosition(token int, extra int) (int, error) {
	if token < 0 || token >= len(eobGroupStart) || extra < 0 {
		return 0, ErrInvalidDecodeState
	}
	eob := eobGroupStart[token]
	if eob > 2 {
		eob += extra
	}
	return eob, nil
}

func EOBPositionToken(eob int) (token int, extra int, err error) {
	if eob < 0 || eob > 1024 {
		return 0, 0, ErrInvalidDecodeState
	}
	if eob < len(eobToPosSmall) {
		token = eobToPosSmall[eob]
	} else {
		index := (eob - 1) >> 5
		if index > 16 {
			index = 16
		}
		token = eobToPosLarge[index]
	}
	extra = eob - eobGroupStart[token]
	if extra < 0 || extra >= 1<<eobOffsetBits[token] {
		return 0, 0, ErrInvalidDecodeState
	}
	return token, extra, nil
}

func CoeffLevelsScratchLen(size TransformSize) (int, error) {
	txSize, err := size.TransformSize()
	if err != nil {
		return 0, ErrInvalidDecodeState
	}
	scanSize, err := transform.ScanSize(txSize)
	if err != nil {
		return 0, ErrInvalidDecodeState
	}
	return (scanSize.Width + txPadHorizontal) * (scanSize.Height + txPadHorizontal), nil
}

// CoeffInitLevels ports libaom's av1_txb_init_levels_c into caller-owned
// scratch. coeffs are in AV1 raster order: coeff_idx = col * height + row.
func CoeffInitLevels(coeffs []int16, size TransformSize, levels []uint8) error {
	txSize, err := size.TransformSize()
	if err != nil {
		return ErrInvalidDecodeState
	}
	scanSize, err := transform.ScanSize(txSize)
	if err != nil {
		return ErrInvalidDecodeState
	}
	maxEOB := scanSize.Width * scanSize.Height
	scratchLen, err := CoeffLevelsScratchLen(size)
	if err != nil {
		return err
	}
	if len(coeffs) < maxEOB || len(levels) < scratchLen {
		return ErrInvalidDecodeState
	}
	for i := 0; i < scratchLen; i++ {
		levels[i] = 0
	}
	stride := scanSize.Height + txPadHorizontal
	for col := 0; col < scanSize.Width; col++ {
		src := col * scanSize.Height
		dst := col * stride
		for row := 0; row < scanSize.Height; row++ {
			levels[dst+row] = coeffAbsClamp127(coeffs[src+row])
		}
	}
	return nil
}

// CoeffNZMapContexts ports libaom's av1_get_nz_map_contexts_c. It writes
// contexts only for scan positions before eob and preserves all other entries.
func CoeffNZMapContexts(levels []uint8, size TransformSize, class transform.Class, scan []int16, eob int, contexts []int8) error {
	if !class.Valid() || eob < 0 {
		return ErrInvalidDecodeState
	}
	txSize, err := size.TransformSize()
	if err != nil {
		return ErrInvalidDecodeState
	}
	scanSize, err := transform.ScanSize(txSize)
	if err != nil {
		return ErrInvalidDecodeState
	}
	maxEOB := scanSize.Width * scanSize.Height
	scratchLen, err := CoeffLevelsScratchLen(size)
	if err != nil {
		return err
	}
	if eob > maxEOB || len(scan) < eob || len(contexts) < maxEOB || len(levels) < scratchLen {
		return ErrInvalidDecodeState
	}
	for i := 0; i < eob; i++ {
		pos := int(scan[i])
		if pos < 0 || pos >= maxEOB {
			return ErrInvalidDecodeState
		}
		var ctx int
		var err error
		if i == eob-1 {
			ctx, err = transform.LowerLevelsCtxEOB(scanSize, i)
			if err != nil {
				return ErrInvalidDecodeState
			}
		} else {
			ctx, err = CoeffLowerLevelsContext(levels, size, class, pos)
			if err != nil {
				return err
			}
		}
		contexts[pos] = int8(ctx)
	}
	return nil
}

func (c *CoeffCDFs) InitDefault(baseQIndex uint8) error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	q := CoeffQContext(baseQIndex)
	var next CoeffCDFs
	for tx := 0; tx < CoeffTxSizeContexts; tx++ {
		for ctx := 0; ctx < TXBSkipContexts; ctx++ {
			if err := next.TXBSkip[tx][ctx].Init(defaultCoeffTXBSkip[q][tx][ctx][:]); err != nil {
				return err
			}
		}
		for plane := 0; plane < CoeffPlaneTypes; plane++ {
			for ctx := 0; ctx < EOBCoefContexts; ctx++ {
				if err := next.EOBExtra[tx][plane][ctx].Init(defaultCoeffEOBExtra[q][tx][plane][ctx][:]); err != nil {
					return err
				}
			}
			for ctx := 0; ctx < CoeffBRContexts; ctx++ {
				if err := next.CoeffBR[tx][plane][ctx].Init(defaultCoeffBR[q][tx][plane][ctx][:]); err != nil {
					return err
				}
			}
			for ctx := 0; ctx < CoeffBaseContexts; ctx++ {
				if err := next.CoeffBase[tx][plane][ctx].Init(defaultCoeffBase[q][tx][plane][ctx][:]); err != nil {
					return err
				}
			}
			for ctx := 0; ctx < EOBBaseContexts; ctx++ {
				if err := next.CoeffBaseEOB[tx][plane][ctx].Init(defaultCoeffBaseEOB[q][tx][plane][ctx][:]); err != nil {
					return err
				}
			}
		}
	}
	for plane := 0; plane < CoeffPlaneTypes; plane++ {
		for ctx := 0; ctx < 3; ctx++ {
			if err := next.DCSign[plane][ctx].Init(defaultCoeffDCSign[q][plane][ctx][:]); err != nil {
				return err
			}
		}
		for ctx := 0; ctx < maxEOBFlagContexts; ctx++ {
			if err := next.EOBFlag16[plane][ctx].Init(defaultCoeffEOBFlag16[q][plane][ctx][:]); err != nil {
				return err
			}
			if err := next.EOBFlag32[plane][ctx].Init(defaultCoeffEOBFlag32[q][plane][ctx][:]); err != nil {
				return err
			}
			if err := next.EOBFlag64[plane][ctx].Init(defaultCoeffEOBFlag64[q][plane][ctx][:]); err != nil {
				return err
			}
			if err := next.EOBFlag128[plane][ctx].Init(defaultCoeffEOBFlag128[q][plane][ctx][:]); err != nil {
				return err
			}
			if err := next.EOBFlag256[plane][ctx].Init(defaultCoeffEOBFlag256[q][plane][ctx][:]); err != nil {
				return err
			}
			if err := next.EOBFlag512[plane][ctx].Init(defaultCoeffEOBFlag512[q][plane][ctx][:]); err != nil {
				return err
			}
			if err := next.EOBFlag1024[plane][ctx].Init(defaultCoeffEOBFlag1024[q][plane][ctx][:]); err != nil {
				return err
			}
		}
	}
	*c = next
	return nil
}

func (c *CoeffCDFs) TXBSkipCDF(size TransformSize, context int) (*entropy.CDF, error) {
	tx, err := CoeffTransformSizeContext(size)
	if err != nil {
		return nil, entropy.ErrInvalidCDF
	}
	if c == nil || context < 0 || context >= TXBSkipContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.TXBSkip[tx][context])
}

func (c *CoeffCDFs) EOBExtraCDF(size TransformSize, plane CoeffPlaneType, context int) (*entropy.CDF, error) {
	tx, err := CoeffTransformSizeContext(size)
	if err != nil {
		return nil, entropy.ErrInvalidCDF
	}
	if c == nil || !plane.Valid() || context < 0 || context >= EOBCoefContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.EOBExtra[tx][plane][context])
}

func (c *CoeffCDFs) DCSignCDF(plane CoeffPlaneType, context int) (*entropy.CDF, error) {
	if c == nil || !plane.Valid() || context < 0 || context >= 3 {
		return nil, entropy.ErrInvalidCDF
	}
	return binaryInterRefCDF(&c.DCSign[plane][context])
}

func (c *CoeffCDFs) CoeffBRCDF(size TransformSize, plane CoeffPlaneType, context int) (*entropy.CDF, error) {
	tx, err := CoeffTransformSizeContext(size)
	if err != nil {
		return nil, entropy.ErrInvalidCDF
	}
	if tx > int(TransformSize32x32) {
		tx = int(TransformSize32x32)
	}
	if c == nil || !plane.Valid() || context < 0 || context >= CoeffBRContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return coeffCDF(&c.CoeffBR[tx][plane][context], BRCDFSize)
}

func (c *CoeffCDFs) CoeffBaseCDF(size TransformSize, plane CoeffPlaneType, context int) (*entropy.CDF, error) {
	tx, err := CoeffTransformSizeContext(size)
	if err != nil {
		return nil, entropy.ErrInvalidCDF
	}
	if c == nil || !plane.Valid() || context < 0 || context >= CoeffBaseContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return coeffCDF(&c.CoeffBase[tx][plane][context], NumBaseLevels+2)
}

func (c *CoeffCDFs) CoeffBaseEOBCDF(size TransformSize, plane CoeffPlaneType, context int) (*entropy.CDF, error) {
	tx, err := CoeffTransformSizeContext(size)
	if err != nil {
		return nil, entropy.ErrInvalidCDF
	}
	if c == nil || !plane.Valid() || context < 0 || context >= EOBBaseContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return coeffCDF(&c.CoeffBaseEOB[tx][plane][context], NumBaseLevels+1)
}

func (c *CoeffCDFs) EOBFlagCDF(size TransformSize, plane CoeffPlaneType, context int) (*entropy.CDF, error) {
	eobSize, err := EOBMultiSize(size)
	if err != nil {
		return nil, entropy.ErrInvalidCDF
	}
	if c == nil || !plane.Valid() || context < 0 || context >= maxEOBFlagContexts {
		return nil, entropy.ErrInvalidCDF
	}
	switch eobSize {
	case 0:
		return coeffCDF(&c.EOBFlag16[plane][context], 5)
	case 1:
		return coeffCDF(&c.EOBFlag32[plane][context], 6)
	case 2:
		return coeffCDF(&c.EOBFlag64[plane][context], 7)
	case 3:
		return coeffCDF(&c.EOBFlag128[plane][context], 8)
	case 4:
		return coeffCDF(&c.EOBFlag256[plane][context], 9)
	case 5:
		return coeffCDF(&c.EOBFlag512[plane][context], 10)
	default:
		return coeffCDF(&c.EOBFlag1024[plane][context], 11)
	}
}

func (s *DecodeState) ReadTXBSkip(cdfs *CoeffCDFs, req TXBSkipRequest) (bool, error) {
	if s == nil {
		return false, ErrInvalidDecodeState
	}
	cdf, err := cdfs.TXBSkipCDF(req.Size, req.Context)
	if err != nil {
		return false, err
	}
	return readBoolCDF(s, cdf)
}

func (s *DecodeState) ReadEOB(cdfs *CoeffCDFs, req EOBRequest) (EOBResult, error) {
	if s == nil {
		return EOBResult{}, ErrInvalidDecodeState
	}
	if !req.Plane.Valid() || req.EOBMultiContext < 0 || req.EOBMultiContext >= maxEOBFlagContexts {
		return EOBResult{}, ErrInvalidDecodeState
	}
	cdf, err := cdfs.EOBFlagCDF(req.Size, req.Plane, req.EOBMultiContext)
	if err != nil {
		return EOBResult{}, err
	}
	symbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return EOBResult{}, err
	}
	token := symbol + 1
	if token >= len(eobOffsetBits) {
		return EOBResult{}, ErrInvalidDecodeState
	}
	offsetBits := eobOffsetBits[token]
	extra := 0
	if offsetBits > 0 {
		extraCDF, err := cdfs.EOBExtraCDF(req.Size, req.Plane, token-3)
		if err != nil {
			return EOBResult{}, err
		}
		bit, err := readBoolCDF(s, extraCDF)
		if err != nil {
			return EOBResult{}, err
		}
		if bit {
			extra += 1 << (offsetBits - 1)
		}
		for i := 1; i < offsetBits; i++ {
			raw, err := s.Reader.ReadBit()
			if err != nil {
				return EOBResult{}, err
			}
			if raw != 0 {
				extra += 1 << (offsetBits - 1 - i)
			}
		}
	}
	pos, err := RecEOBPosition(token, extra)
	if err != nil {
		return EOBResult{}, err
	}
	return EOBResult{Token: token, Extra: extra, Position: pos, OffsetBits: offsetBits}, nil
}

func (s *DecodeState) ReadCoeffBaseEOB(cdfs *CoeffCDFs, req CoeffTokenRequest) (int, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	cdf, err := cdfs.CoeffBaseEOBCDF(req.Size, req.Plane, req.Context)
	if err != nil {
		return 0, err
	}
	symbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return 0, err
	}
	return symbol + 1, nil
}

func (s *DecodeState) ReadCoeffBase(cdfs *CoeffCDFs, req CoeffTokenRequest) (int, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	cdf, err := cdfs.CoeffBaseCDF(req.Size, req.Plane, req.Context)
	if err != nil {
		return 0, err
	}
	return s.Reader.ReadCDF(cdf)
}

func (s *DecodeState) ReadCoeffBR(cdfs *CoeffCDFs, req CoeffTokenRequest) (int, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	cdf, err := cdfs.CoeffBRCDF(req.Size, req.Plane, req.Context)
	if err != nil {
		return 0, err
	}
	return s.Reader.ReadCDF(cdf)
}

func (s *DecodeState) ReadDCSign(cdfs *CoeffCDFs, plane CoeffPlaneType, context int) (bool, error) {
	if s == nil {
		return false, ErrInvalidDecodeState
	}
	cdf, err := cdfs.DCSignCDF(plane, context)
	if err != nil {
		return false, err
	}
	return readBoolCDF(s, cdf)
}

// ReadCoefficientsTXB decodes AV1 coefficient syntax into coeffs in AV1 raster
// order: coeff_idx = col * adjusted_height + row. Dequantization and qmatrix
// scaling are intentionally separate from this entropy-syntax layer.
func (s *DecodeState) ReadCoefficientsTXB(cdfs *CoeffCDFs, req TXBDecodeRequest, coeffs []int16, scan []int16, levelsScratch []uint8) (TXBDecodeResult, error) {
	if s == nil {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	if !req.Plane.Valid() || !req.Class.Valid() {
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
	maxEOB := scanSize.Width * scanSize.Height
	if len(coeffs) < maxEOB || len(scan) < maxEOB {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	scratchLen, err := CoeffLevelsScratchLen(req.Size)
	if err != nil {
		return TXBDecodeResult{}, err
	}
	if len(levelsScratch) < scratchLen {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}

	for i := 0; i < maxEOB; i++ {
		coeffs[i] = 0
	}
	for i := 0; i < scratchLen; i++ {
		levelsScratch[i] = 0
	}

	allZero := req.TXBSkip
	if !req.TXBSkipKnown {
		var err error
		allZero, err = s.ReadTXBSkip(cdfs, TXBSkipRequest{Size: req.Size, Context: req.TXBSkipContext})
		if err != nil {
			return TXBDecodeResult{}, fmt.Errorf("read txb skip size=%v ctx=%d: %w", req.Size, req.TXBSkipContext, err)
		}
	}
	if allZero {
		return TXBDecodeResult{AllZero: true}, nil
	}

	eob, err := s.ReadEOB(cdfs, EOBRequest{
		Size:            req.Size,
		Plane:           req.Plane,
		EOBMultiContext: req.EOBMultiContext,
	})
	if err != nil {
		return TXBDecodeResult{}, fmt.Errorf("read eob size=%v plane=%v eobCtx=%d: %w", req.Size, req.Plane, req.EOBMultiContext, err)
	}
	if eob.Position <= 0 || eob.Position > maxEOB {
		return TXBDecodeResult{}, fmt.Errorf("eob position=%d max=%d token=%d extra=%d: %w", eob.Position, maxEOB, eob.Token, eob.Extra, ErrInvalidDecodeState)
	}

	culLevel := 0
	dcValue := 0
	if err := s.readLastCoeffLevel(cdfs, req, eob.Position, scan, levelsScratch, scanSize); err != nil {
		return TXBDecodeResult{}, fmt.Errorf("read last coeff eob=%d: %w", eob.Position, err)
	}
	for c := eob.Position - 2; c >= 0; c-- {
		pos := int(scan[c])
		ctx, err := CoeffLowerLevelsContext(levelsScratch, req.Size, req.Class, pos)
		if err != nil {
			return TXBDecodeResult{}, fmt.Errorf("lower levels ctx c=%d pos=%d: %w", c, pos, err)
		}
		level, err := s.ReadCoeffBase(cdfs, CoeffTokenRequest{Size: req.Size, Plane: req.Plane, Context: ctx})
		if err != nil {
			return TXBDecodeResult{}, fmt.Errorf("read coeff base c=%d pos=%d ctx=%d: %w", c, pos, ctx, err)
		}
		if level > NumBaseLevels {
			brCtx, err := CoeffBRContext(levelsScratch, req.Size, req.Class, pos)
			if err != nil {
				return TXBDecodeResult{}, fmt.Errorf("br ctx c=%d pos=%d level=%d: %w", c, pos, level, err)
			}
			extra, err := s.readBaseRange(cdfs, req.Size, req.Plane, brCtx)
			if err != nil {
				return TXBDecodeResult{}, fmt.Errorf("read base range c=%d pos=%d level=%d brCtx=%d: %w", c, pos, level, brCtx, err)
			}
			level += extra
		}
		if err := setCoeffLevel(levelsScratch, req.Size, pos, level); err != nil {
			return TXBDecodeResult{}, fmt.Errorf("set coeff level c=%d pos=%d level=%d: %w", c, pos, level, err)
		}
	}

	coeffTraceBlock(int(req.Plane), 0, 0, int(req.Size), int(req.Class), eob.Position)
	maxScanLine := 0
	for c := 0; c < eob.Position; c++ {
		pos := int(scan[c])
		level, err := coeffLevel(levelsScratch, req.Size, pos)
		if err != nil {
			return TXBDecodeResult{}, fmt.Errorf("read stored coeff c=%d pos=%d: %w", c, pos, err)
		}
		if level == 0 {
			continue
		}
		if pos > maxScanLine {
			maxScanLine = pos
		}
		negative := false
		if c == 0 {
			negative, err = s.ReadDCSign(cdfs, req.Plane, req.DCSignContext)
		} else {
			var bit uint8
			bit, err = s.Reader.ReadBit()
			negative = bit != 0
		}
		if err != nil {
			return TXBDecodeResult{}, fmt.Errorf("read coeff sign c=%d pos=%d level=%d: %w", c, pos, level, err)
		}
		baseLevel := level
		golombExtra := 0
		if level >= MaxBaseBRRange {
			tail, err := s.readCoeffGolomb()
			if err != nil {
				return TXBDecodeResult{}, fmt.Errorf("read coeff golomb c=%d pos=%d level=%d: %w", c, pos, level, err)
			}
			golombExtra = tail
			level += tail
		}
		signBit := 0
		if negative {
			signBit = 1
		}
		coeffTraceCoeff(c, pos, baseLevel, golombExtra, level, signBit)
		culLevel += level
		if level > int(^uint16(0)>>1) {
			return TXBDecodeResult{}, fmt.Errorf("coeff level overflow c=%d pos=%d level=%d: %w", c, pos, level, ErrInvalidDecodeState)
		}
		signed := int16(level)
		if negative {
			signed = -signed
		}
		if c == 0 {
			dcValue = int(signed)
		}
		coeffs[pos] = signed
	}

	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}

	coeffTraceCulLevel(culLevel)
	return TXBDecodeResult{
		EOB:         eob.Position,
		MaxScanLine: maxScanLine,
		CulLevel:    culLevel,
	}, nil
}

func (s *DecodeState) readLastCoeffLevel(cdfs *CoeffCDFs, req TXBDecodeRequest, eob int, scan []int16, levels []uint8, scanSize transform.Size) error {
	c := eob - 1
	pos := int(scan[c])
	ctx, err := transform.LowerLevelsCtxEOB(scanSize, c)
	if err != nil {
		return ErrInvalidDecodeState
	}
	level, err := s.ReadCoeffBaseEOB(cdfs, CoeffTokenRequest{Size: req.Size, Plane: req.Plane, Context: ctx})
	if err != nil {
		return err
	}
	if level > NumBaseLevels {
		brCtx, err := CoeffBRContextEOB(req.Size, req.Class, pos)
		if err != nil {
			return err
		}
		extra, err := s.readBaseRange(cdfs, req.Size, req.Plane, brCtx)
		if err != nil {
			return err
		}
		level += extra
	}
	return setCoeffLevel(levels, req.Size, pos, level)
}

func (s *DecodeState) readBaseRange(cdfs *CoeffCDFs, size TransformSize, plane CoeffPlaneType, context int) (int, error) {
	level := 0
	for idx := 0; idx < CoeffBaseRange; idx += BRCDFSize - 1 {
		k, err := s.ReadCoeffBR(cdfs, CoeffTokenRequest{Size: size, Plane: plane, Context: context})
		if err != nil {
			return 0, err
		}
		level += k
		if k < BRCDFSize-1 {
			break
		}
	}
	return level, nil
}

func (s *DecodeState) readCoeffGolomb() (int, error) {
	x := 1
	length := 0
	for {
		bit, err := s.Reader.ReadBit()
		if err != nil {
			return 0, err
		}
		length++
		if length > 20 {
			return 0, ErrInvalidDecodeState
		}
		if bit != 0 {
			break
		}
	}
	for i := 0; i < length-1; i++ {
		bit, err := s.Reader.ReadBit()
		if err != nil {
			return 0, err
		}
		x <<= 1
		x += int(bit)
	}
	return x - 1, nil
}

func coeffCDF(cdf *entropy.CDF, symbols int) (*entropy.CDF, error) {
	if cdf == nil || cdf.Symbols() != symbols {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

func CoeffBRContextEOB(size TransformSize, class transform.Class, coeffIndex int) (int, error) {
	if !class.Valid() || coeffIndex < 0 {
		return 0, ErrInvalidDecodeState
	}
	row, col, _, err := coeffPosition(size, coeffIndex)
	if err != nil {
		return 0, err
	}
	if coeffIndex == 0 {
		return 0, nil
	}
	switch class {
	case transform.Class2D:
		if row < 2 && col < 2 {
			return 7, nil
		}
	case transform.ClassHoriz:
		if col == 0 {
			return 7, nil
		}
	case transform.ClassVert:
		if row == 0 {
			return 7, nil
		}
	}
	return 14, nil
}

func CoeffBRContext(levels []uint8, size TransformSize, class transform.Class, coeffIndex int) (int, error) {
	if !class.Valid() || coeffIndex < 0 {
		return 0, ErrInvalidDecodeState
	}
	padded, stride, row, col, err := coeffPaddedPosition(size, coeffIndex)
	if err != nil {
		return 0, err
	}
	if padded+2*stride+2 >= len(levels) || padded+4 >= len(levels) {
		return 0, ErrInvalidDecodeState
	}
	if class == transform.Class2D && coeffIndex != 0 {
		mag := minInt(int(levels[padded+1]), MaxBaseBRRange) +
			minInt(int(levels[padded+stride]), MaxBaseBRRange) +
			minInt(int(levels[padded+stride+1]), MaxBaseBRRange)
		mag = minInt((mag+1)>>1, 6)
		if row < 2 && col < 2 {
			return mag + 7, nil
		}
		return mag + 14, nil
	}

	mag := int(levels[padded+1]) + int(levels[padded+stride])
	switch class {
	case transform.Class2D:
		mag += int(levels[padded+stride+1])
	case transform.ClassHoriz:
		mag += int(levels[padded+(stride<<1)])
	case transform.ClassVert:
		mag += int(levels[padded+2])
	}
	mag = minInt((mag+1)>>1, 6)
	if coeffIndex == 0 {
		return mag, nil
	}
	switch class {
	case transform.Class2D:
		if row < 2 && col < 2 {
			return mag + 7, nil
		}
	case transform.ClassHoriz:
		if col == 0 {
			return mag + 7, nil
		}
	case transform.ClassVert:
		if row == 0 {
			return mag + 7, nil
		}
	}
	return mag + 14, nil
}

func CoeffLowerLevelsContext(levels []uint8, size TransformSize, class transform.Class, coeffIndex int) (int, error) {
	if !class.Valid() || coeffIndex < 0 {
		return 0, ErrInvalidDecodeState
	}
	padded, stride, row, col, err := coeffPaddedPosition(size, coeffIndex)
	if err != nil {
		return 0, err
	}
	if padded+4 >= len(levels) || padded+4*stride >= len(levels) || padded+stride+1 >= len(levels) {
		return 0, ErrInvalidDecodeState
	}
	if class == transform.Class2D && coeffIndex == 0 {
		return 0, nil
	}

	mag := clipMax3(levels[padded+stride]) + clipMax3(levels[padded+1])
	switch class {
	case transform.Class2D:
		mag += clipMax3(levels[padded+stride+1])
		mag += clipMax3(levels[padded+(stride<<1)])
		mag += clipMax3(levels[padded+2])
	case transform.ClassVert:
		mag += clipMax3(levels[padded+2])
		mag += clipMax3(levels[padded+3])
		mag += clipMax3(levels[padded+4])
	case transform.ClassHoriz:
		mag += clipMax3(levels[padded+(stride<<1)])
		mag += clipMax3(levels[padded+3*stride])
		mag += clipMax3(levels[padded+4*stride])
	}
	ctx := minInt((mag+1)>>1, 4)

	switch class {
	case transform.Class2D:
		// libaom's get_lower_levels_ctx_2d adds av1_nz_map_ctx_offset[tx_size]
		// indexed by the *unadjusted* tx_size, so rectangular sizes whose scan
		// is square after av1_get_adjusted_tx_size (TX_32X64, TX_64X32) still
		// receive the row<2/+11 or col<2/+16 bias. We must therefore compare
		// the original transform width/height, not the (possibly square)
		// adjusted scan dimensions.
		dims, ok := size.Dimensions()
		if !ok {
			return 0, ErrInvalidDecodeState
		}
		txWidth := int(dims.W4) << 2
		txHeight := int(dims.H4) << 2
		if txWidth < txHeight && row < 2 {
			return ctx + 11, nil
		}
		if txWidth > txHeight && col < 2 {
			return ctx + 16, nil
		}
		if row+col < 2 {
			return ctx + 1, nil
		}
		if row+col < 4 {
			return ctx + 6, nil
		}
		return ctx + 21, nil
	case transform.ClassHoriz:
		return ctx + coeff1DContextOffset(col), nil
	case transform.ClassVert:
		return ctx + coeff1DContextOffset(row), nil
	default:
		return 0, ErrInvalidDecodeState
	}
}

func coeff1DContextOffset(position int) int {
	if position == 0 {
		return 26
	}
	if position == 1 {
		return 31
	}
	return 36
}

func coeffLevel(levels []uint8, size TransformSize, coeffIndex int) (int, error) {
	padded, _, _, _, err := coeffPaddedPosition(size, coeffIndex)
	if err != nil {
		return 0, err
	}
	if padded >= len(levels) {
		return 0, ErrInvalidDecodeState
	}
	return int(levels[padded]), nil
}

func setCoeffLevel(levels []uint8, size TransformSize, coeffIndex int, level int) error {
	if level < 0 || level > 255 {
		return ErrInvalidDecodeState
	}
	padded, _, _, _, err := coeffPaddedPosition(size, coeffIndex)
	if err != nil {
		return err
	}
	if padded >= len(levels) {
		return ErrInvalidDecodeState
	}
	levels[padded] = uint8(level)
	return nil
}

func coeffPaddedPosition(size TransformSize, coeffIndex int) (padded int, stride int, row int, col int, err error) {
	row, col, stride, err = coeffPosition(size, coeffIndex)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	return col*stride + row, stride, row, col, nil
}

func coeffPosition(size TransformSize, coeffIndex int) (row int, col int, stride int, err error) {
	txSize, err := size.TransformSize()
	if err != nil {
		return 0, 0, 0, ErrInvalidDecodeState
	}
	scanSize, err := transform.ScanSize(txSize)
	if err != nil {
		return 0, 0, 0, ErrInvalidDecodeState
	}
	maxEOB := scanSize.Width * scanSize.Height
	if coeffIndex < 0 || coeffIndex >= maxEOB {
		return 0, 0, 0, ErrInvalidDecodeState
	}
	col = coeffIndex / scanSize.Height
	row = coeffIndex - col*scanSize.Height
	return row, col, scanSize.Height + txPadHorizontal, nil
}

func clipMax3(v uint8) int {
	if v > 3 {
		return 3
	}
	return int(v)
}

func coeffAbsClamp127(v int16) uint8 {
	n := int(v)
	if n < 0 {
		n = -n
	}
	if n > 127 {
		return 127
	}
	return uint8(n)
}
