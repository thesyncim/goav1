package tile

import "github.com/thesyncim/goav1/internal/av1/entropy"

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
	maxEOBFlagContexts    = 2
	maxEOBFlagCumulative  = 10
	maxCoeffCumulativeLen = 10
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

var eobGroupStart = [12]int{0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513}
var eobOffsetBits = [12]int{0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
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

func coeffCDF(cdf *entropy.CDF, symbols int) (*entropy.CDF, error) {
	if cdf == nil || cdf.Symbols() != symbols {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}
