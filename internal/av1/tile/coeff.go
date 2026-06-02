package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

const (
	CoeffQContexts      = 4
	CoeffTxSizeContexts = 5
	CoeffPlaneTypes     = 2
	TXBSkipContexts     = 13
	EOBCoefContexts     = 9
	EOBBaseContexts     = 4
	CoeffBaseContexts   = 42
	CoeffBRContexts     = 21
	NumBaseLevels       = 2
	BRCDFSize           = 4
	CoeffBaseRange      = 4 * (BRCDFSize - 1)
	MaxBaseBRRange      = CoeffBaseRange + NumBaseLevels + 1
	CoeffContextBits    = 3
	CoeffContextMask    = (1 << CoeffContextBits) - 1
	maxEOBFlagContexts  = 2
	txPadHorizontal     = 4
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
	// SkipAllZeroCoeffClear lets fused internal callers avoid full reusable
	// coefficient-scratch clears when they consume coefficients before reuse.
	// The default remains false so public callers still observe a zero-filled
	// coefficient slice for skipped TXBs.
	SkipAllZeroCoeffClear bool

	skipNonZeroCoeffClear bool
	coeffDirtyPos         *[maxCoeffScanLen]int16
	coeffDirtyLen         *int
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

// coeffGeometry caches the scan/level-buffer geometry derived from a
// TransformSize. Every field is a pure function of the transform size, so the
// whole table is computed once at package init and shared read-only across
// decode goroutines. It removes the per-coefficient TransformSize ->
// transform.Size -> ScanSize -> adjustedScanSize chain that dominated the
// coefficient hot path.
type coeffGeometry struct {
	valid      bool
	scanWidth  int
	scanHeight int
	stride     int // scanHeight + txPadHorizontal
	maxEOB     int // scanWidth * scanHeight
	scratchLen int // (scanWidth+pad) * (scanHeight+pad)
	txCtx      uint8
	txBRCtx    uint8
	// txWide/txTall record the unadjusted transform-dimension comparison used
	// by the Class2D lower-levels context bias (libaom indexes
	// av1_nz_map_ctx_offset by the original tx_size). They are loop-invariant
	// per transform size, so precomputing them here lets the per-coefficient
	// context derivation skip a Dimensions() call.
	txWide bool // original txWidth > txHeight
	txTall bool // original txWidth < txHeight
}

var coeffGeometryTable [transformSizeCount]coeffGeometry

// coeffPos is the precomputed scan position for one coefficient index: the
// padded scratch offset (col*stride+row) plus the unpadded row/col. Computing
// these once per transform size at init turns the per-coefficient
// index->position math (a data-dependent division) into a table lookup on the
// hot coefficient-read path. row/col stay below the transform dimension and
// padded stays below scratchLen, all well within int16.
type coeffPos struct {
	padded        int16
	row           int16
	col           int16
	lower2DOffset int8
	br2DOffset    int8
}

// coeffPosTable[size][coeffIndex] holds the precomputed position for every
// valid coefficient index of size, indexed in [0, maxEOB).
var coeffPosTable [transformSizeCount][]coeffPos
var coeffScanTable [transformSizeCount][3][]int16

func init() {
	for size := range transformSizeCount {
		txSize, err := size.TransformSize()
		if err != nil {
			continue
		}
		scanSize, err := transform.ScanSize(txSize)
		if err != nil {
			continue
		}
		stride := scanSize.Height + txPadHorizontal
		maxEOB := scanSize.Width * scanSize.Height
		var txWide, txTall bool
		var txCtx uint8
		if dims, ok := size.Dimensions(); ok {
			txWidth := int(dims.W4) << 2
			txHeight := int(dims.H4) << 2
			txWide = txWidth > txHeight
			txTall = txWidth < txHeight
			txCtx = (dims.Min + dims.Max + 1) >> 1
		}
		txBRCtx := txCtx
		if txBRCtx > uint8(TransformSize32x32) {
			txBRCtx = uint8(TransformSize32x32)
		}
		coeffGeometryTable[size] = coeffGeometry{
			valid:      true,
			scanWidth:  scanSize.Width,
			scanHeight: scanSize.Height,
			stride:     stride,
			maxEOB:     maxEOB,
			scratchLen: (scanSize.Width + txPadHorizontal) * stride,
			txCtx:      txCtx,
			txBRCtx:    txBRCtx,
			txWide:     txWide,
			txTall:     txTall,
		}
		positions := make([]coeffPos, maxEOB)
		for idx := range maxEOB {
			col := idx / scanSize.Height
			row := idx - col*scanSize.Height
			positions[idx] = coeffPos{
				padded: int16(col*stride + row),
				row:    int16(row),
				col:    int16(col),
			}
		}
		for idx := range positions {
			p := &positions[idx]
			row := int(p.row)
			col := int(p.col)
			switch {
			case idx == 0:
				p.lower2DOffset = 0
			case txTall && row < 2:
				p.lower2DOffset = 11
			case txWide && col < 2:
				p.lower2DOffset = 16
			case row+col < 2:
				p.lower2DOffset = 1
			case row+col < 4:
				p.lower2DOffset = 6
			default:
				p.lower2DOffset = 21
			}
			if idx != 0 {
				if row < 2 && col < 2 {
					p.br2DOffset = 7
				} else {
					p.br2DOffset = 14
				}
			}
		}
		coeffPosTable[size] = positions
		for class := transform.Class2D; class <= transform.ClassVert; class++ {
			scan := make([]int16, maxEOB)
			inverse := make([]int16, maxEOB)
			if err := transform.FillDefaultScan(scan, inverse, txSize, class); err == nil {
				coeffScanTable[size][class] = scan
			}
		}
	}
}

// coeffGeo returns the precomputed scan geometry for size. The bool reports
// whether size names a valid transform.
func coeffGeo(size TransformSize) (coeffGeometry, bool) {
	if size >= transformSizeCount {
		return coeffGeometry{}, false
	}
	g := coeffGeometryTable[size]
	return g, g.valid
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
		index := min((eob-1)>>5, 16)
		token = eobToPosLarge[index]
	}
	extra = eob - eobGroupStart[token]
	if extra < 0 || extra >= 1<<eobOffsetBits[token] {
		return 0, 0, ErrInvalidDecodeState
	}
	return token, extra, nil
}

func CoeffLevelsScratchLen(size TransformSize) (int, error) {
	g, ok := coeffGeo(size)
	if !ok {
		return 0, ErrInvalidDecodeState
	}
	return g.scratchLen, nil
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
	for i := range scratchLen {
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
	for i := range eob {
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
	for tx := range CoeffTxSizeContexts {
		for ctx := range TXBSkipContexts {
			if err := next.TXBSkip[tx][ctx].Init(defaultCoeffTXBSkip[q][tx][ctx][:]); err != nil {
				return err
			}
		}
		for plane := range CoeffPlaneTypes {
			for ctx := range EOBCoefContexts {
				if err := next.EOBExtra[tx][plane][ctx].Init(defaultCoeffEOBExtra[q][tx][plane][ctx][:]); err != nil {
					return err
				}
			}
			for ctx := range CoeffBRContexts {
				if err := next.CoeffBR[tx][plane][ctx].Init(defaultCoeffBR[q][tx][plane][ctx][:]); err != nil {
					return err
				}
			}
			for ctx := range CoeffBaseContexts {
				if err := next.CoeffBase[tx][plane][ctx].Init(defaultCoeffBase[q][tx][plane][ctx][:]); err != nil {
					return err
				}
			}
			for ctx := range EOBBaseContexts {
				if err := next.CoeffBaseEOB[tx][plane][ctx].Init(defaultCoeffBaseEOB[q][tx][plane][ctx][:]); err != nil {
					return err
				}
			}
		}
	}
	for plane := range CoeffPlaneTypes {
		for ctx := range 3 {
			if err := next.DCSign[plane][ctx].Init(defaultCoeffDCSign[q][plane][ctx][:]); err != nil {
				return err
			}
		}
		for ctx := range maxEOBFlagContexts {
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
	return coeffBinaryCDF(&c.TXBSkip[tx][context])
}

func (c *CoeffCDFs) EOBExtraCDF(size TransformSize, plane CoeffPlaneType, context int) (*entropy.CDF, error) {
	tx, err := CoeffTransformSizeContext(size)
	if err != nil {
		return nil, entropy.ErrInvalidCDF
	}
	if c == nil || !plane.Valid() || context < 0 || context >= EOBCoefContexts {
		return nil, entropy.ErrInvalidCDF
	}
	return coeffBinaryCDF(&c.EOBExtra[tx][plane][context])
}

func (c *CoeffCDFs) DCSignCDF(plane CoeffPlaneType, context int) (*entropy.CDF, error) {
	if c == nil || !plane.Valid() || context < 0 || context >= 3 {
		return nil, entropy.ErrInvalidCDF
	}
	return coeffBinaryCDF(&c.DCSign[plane][context])
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

// readBoolCDFTrusted reads one boolean symbol from a coefficient CDF that has
// already been validated by its accessor (coeffCDF / coeffBinaryCDF). It skips
// the redundant per-symbol monotonicity scan that ReadCDF would repeat.
// Coefficient CDFs are valid by construction: they are seeded from valid libaom
// defaults and only ever mutated by updateCDFWindow, which preserves the
// inverse-CDF shape.
func readBoolCDFTrusted(s *DecodeState, cdf *entropy.CDF) (bool, error) {
	if s == nil {
		return false, ErrInvalidDecodeState
	}
	return s.Reader.ReadBinaryCDFUnchecked(cdf) != 0, nil
}

func (s *DecodeState) ReadTXBSkip(cdfs *CoeffCDFs, req TXBSkipRequest) (bool, error) {
	if s == nil {
		return false, ErrInvalidDecodeState
	}
	cdf, err := cdfs.TXBSkipCDF(req.Size, req.Context)
	if err != nil {
		return false, err
	}
	return readBoolCDFTrusted(s, cdf)
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
	symbol, err := s.Reader.ReadCDFTrusted(cdf)
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
		bit, err := readBoolCDFTrusted(s, extraCDF)
		if err != nil {
			return EOBResult{}, err
		}
		if bit {
			extra += 1 << (offsetBits - 1)
		}
		if offsetBits > 1 {
			tail, err := s.Reader.ReadBits(uint8(offsetBits - 1))
			if err != nil {
				return EOBResult{}, err
			}
			extra += int(tail)
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
	symbol, err := s.Reader.ReadCDFTrusted(cdf)
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
	return s.Reader.ReadCDFTrusted(cdf)
}

func (s *DecodeState) ReadCoeffBR(cdfs *CoeffCDFs, req CoeffTokenRequest) (int, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	cdf, err := cdfs.CoeffBRCDF(req.Size, req.Plane, req.Context)
	if err != nil {
		return 0, err
	}
	return s.Reader.ReadCDFTrusted(cdf)
}

func (s *DecodeState) ReadDCSign(cdfs *CoeffCDFs, plane CoeffPlaneType, context int) (bool, error) {
	if s == nil {
		return false, ErrInvalidDecodeState
	}
	cdf, err := cdfs.DCSignCDF(plane, context)
	if err != nil {
		return false, err
	}
	return readBoolCDFTrusted(s, cdf)
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
	geo, ok := coeffGeo(req.Size)
	if !ok {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	maxEOB := geo.maxEOB
	if len(coeffs) < maxEOB || len(scan) < maxEOB {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	coeffs = coeffs[:maxEOB]
	scan = scan[:maxEOB]
	scratchLen := geo.scratchLen
	if len(levelsScratch) < scratchLen {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	levelsScratch = levelsScratch[:scratchLen]

	allZero := req.TXBSkip
	if !req.TXBSkipKnown {
		var err error
		allZero, err = s.ReadTXBSkip(cdfs, TXBSkipRequest{Size: req.Size, Context: req.TXBSkipContext})
		if err != nil {
			return TXBDecodeResult{}, err
		}
	}
	if allZero {
		if !req.SkipAllZeroCoeffClear {
			clear(coeffs[:maxEOB])
		}
		return TXBDecodeResult{AllZero: true}, nil
	}
	if !req.skipNonZeroCoeffClear {
		clear(coeffs[:maxEOB])
	}

	eob, err := s.ReadEOB(cdfs, EOBRequest{
		Size:            req.Size,
		Plane:           req.Plane,
		EOBMultiContext: req.EOBMultiContext,
	})
	if err != nil {
		return TXBDecodeResult{}, err
	}
	if eob.Position <= 0 || eob.Position > maxEOB {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}

	// Hoist the per-transform-size geometry and position slice out of the
	// per-coefficient loops. The context derivation and level get/set below all
	// index coeffPosTable[req.Size]/coeffGeometryTable[req.Size] by position, so
	// fetching them once removes the repeated outer-table indexing (and its
	// bounds checks) from every coefficient. coeffGeo already validated req.Size
	// above; restating the bound here is dead on valid input but lets the prover
	// drop the table-index checks.
	if req.Size >= transformSizeCount {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	geoPtr := &coeffGeometryTable[req.Size]
	posSlice := coeffPosTable[req.Size]
	if len(posSlice) < maxEOB {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	posSlice = posSlice[:maxEOB]
	// Hoist the CoeffBase/CoeffBR CDF-array selection out of the per-coefficient
	// loop. CoeffBaseCDF/CoeffBRCDF re-derive the transform-size context (a
	// Dimensions() lookup) and re-validate the plane on every symbol read; here
	// it is loop-invariant. txBR matches CoeffBRCDF's clamp to TransformSize32x32.
	// req.Plane was validated above; the bounds restated here are dead on valid
	// input but keep these one-time array indexes check-free.
	txCtx := int(geo.txCtx)
	txBR := int(geo.txBRCtx)
	if txCtx < 0 || txCtx >= CoeffTxSizeContexts || txBR < 0 || txBR >= CoeffTxSizeContexts ||
		!req.Plane.Valid() || cdfs == nil || req.DCSignContext < 0 || req.DCSignContext >= 3 {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	baseArr := &cdfs.CoeffBase[txCtx][req.Plane]
	baseEOBArr := &cdfs.CoeffBaseEOB[txCtx][req.Plane]
	brArr := &cdfs.CoeffBR[txBR][req.Plane]
	dcSignCDF := &cdfs.DCSign[req.Plane][req.DCSignContext]
	reader := s.Reader.Cursor()

	lastC := eob.Position - 1
	lastPos := int(scan[lastC])
	if lastPos < 0 || lastPos >= maxEOB {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	lastCtx := coeffLowerLevelsCtxEOBFast(maxEOB, lastC)
	lastLevel := reader.ReadCDF3Unchecked(&baseEOBArr[lastCtx]) + 1
	if lastLevel > NumBaseLevels {
		brCtx := coeffBRContextEOBFast(posSlice, req.Class, lastPos)
		extra := readBaseRangeFromArrCursor(&reader, brArr, brCtx)
		lastLevel += extra
	}
	if eob.Position == 1 {
		if coeffTraceEnabled {
			coeffTraceBlock(int(req.Plane), 0, 0, int(req.Size), int(req.Class), eob.Position)
		}
		negative := reader.ReadBinaryCDFUnchecked(dcSignCDF) != 0
		baseLevel := lastLevel
		golombExtra := 0
		if lastLevel >= MaxBaseBRRange {
			tail, err := readCoeffGolombCursor(&reader)
			if err != nil {
				reader.CommitStateTo(&s.Reader)
				return TXBDecodeResult{}, err
			}
			golombExtra = tail
			lastLevel += tail
		}
		signBit := 0
		if negative {
			signBit = 1
		}
		if coeffTraceEnabled {
			coeffTraceCoeff(0, lastPos, baseLevel, golombExtra, lastLevel, signBit)
		}
		if lastLevel > int(^uint16(0)>>1) {
			reader.CommitStateTo(&s.Reader)
			return TXBDecodeResult{}, ErrInvalidDecodeState
		}
		signed := int16(lastLevel)
		if negative {
			signed = -signed
		}
		coeffs[lastPos] = signed
		req.recordCoeffDirty(lastPos)

		culLevel := lastLevel
		if culLevel > CoeffContextMask {
			culLevel = CoeffContextMask
		}
		if signed < 0 {
			culLevel |= 1 << CoeffContextBits
		} else if signed > 0 {
			culLevel += 2 << CoeffContextBits
		}
		if coeffTraceEnabled {
			coeffTraceCulLevel(culLevel)
		}
		reader.CommitStateTo(&s.Reader)
		return TXBDecodeResult{
			EOB:         1,
			MaxScanLine: lastPos,
			CulLevel:    culLevel,
		}, nil
	}

	clear(levelsScratch[:scratchLen])
	lastPadded := int(posSlice[lastPos].padded)
	if lastPadded < 0 || lastPadded >= len(levelsScratch) {
		reader.CommitStateTo(&s.Reader)
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	levelsScratch[lastPadded] = uint8(lastLevel)
	// While the levels map is being filled, coeffs[pos] carries a temporary
	// next pointer through the nonzero coefficients in scan order. The final
	// sign/Golomb pass captures the pointer before overwriting coeffs[pos] with
	// the signed output coefficient, so the public output shape stays unchanged
	// and no extra per-TXB scratch is required.
	headC := lastC
	coeffs[lastPos] = 0

	for c := eob.Position - 2; c >= 0; c-- {
		pos := int(scan[c])
		if pos < 0 || pos >= maxEOB {
			reader.CommitStateTo(&s.Reader)
			return TXBDecodeResult{}, ErrInvalidDecodeState
		}
		// pos is now proven in range, so this posSlice read and the matching
		// levelsScratch store below drop their bounds checks. padded is fetched
		// once and reused for the store (coeffLowerLevelsCtxFast/coeffBRContextFast
		// re-derive it internally, but hoisting it here removes the second
		// posSlice index from the hot store path).
		p := posSlice[pos]
		padded := int(p.padded)
		if padded < 0 || padded >= len(levelsScratch) {
			reader.CommitStateTo(&s.Reader)
			return TXBDecodeResult{}, ErrInvalidDecodeState
		}
		level := 0
		if req.Class == transform.Class2D {
			ctx := 0
			if pos != 0 {
				s1 := padded + geoPtr.stride
				s1p1 := s1 + 1
				s2 := s1 + geoPtr.stride
				p1 := padded + 1
				p2 := padded + 2
				_ = levelsScratch[s2]
				_ = levelsScratch[p2]
				mag := clipMax3(levelsScratch[s1]) + clipMax3(levelsScratch[p1]) +
					clipMax3(levelsScratch[s1p1]) + clipMax3(levelsScratch[s2]) + clipMax3(levelsScratch[p2])
				ctx = minInt((mag+1)>>1, 4) + int(p.lower2DOffset)
			}
			level = reader.ReadCDF4Unchecked(&baseArr[ctx])
			if level > NumBaseLevels {
				s1 := padded + geoPtr.stride
				s1p1 := s1 + 1
				p1 := padded + 1
				_ = levelsScratch[s1p1]
				mag := int(levelsScratch[p1]) + int(levelsScratch[s1]) + int(levelsScratch[s1p1])
				brCtx := minInt((mag+1)>>1, 6) + int(p.br2DOffset)
				extra := readBaseRangeFromArrCursor(&reader, brArr, brCtx)
				level += extra
			}
		} else {
			ctx := coeffLowerLevelsCtxFast(levelsScratch, geoPtr, posSlice, req.Class, pos)
			level = reader.ReadCDF4Unchecked(&baseArr[ctx])
			if level > NumBaseLevels {
				brCtx := coeffBRContextFast(levelsScratch, geoPtr, posSlice, req.Class, pos)
				extra := readBaseRangeFromArrCursor(&reader, brArr, brCtx)
				level += extra
			}
		}
		levelsScratch[padded] = uint8(level)
		if level != 0 {
			coeffs[pos] = int16(headC + 1)
			headC = c
		}
	}

	if coeffTraceEnabled {
		coeffTraceBlock(int(req.Plane), 0, 0, int(req.Size), int(req.Class), eob.Position)
	}
	culLevel := 0
	dcValue := 0
	maxScanLine := 0
	for c := headC; c >= 0; {
		pos := int(scan[c])
		if pos < 0 || pos >= maxEOB {
			reader.CommitStateTo(&s.Reader)
			return TXBDecodeResult{}, ErrInvalidDecodeState
		}
		nextC := int(coeffs[pos]) - 1
		padded := int(posSlice[pos].padded)
		if padded < 0 || padded >= len(levelsScratch) {
			reader.CommitStateTo(&s.Reader)
			return TXBDecodeResult{}, ErrInvalidDecodeState
		}
		level := int(levelsScratch[padded])
		if level == 0 {
			reader.CommitStateTo(&s.Reader)
			return TXBDecodeResult{}, ErrInvalidDecodeState
		}
		if pos > maxScanLine {
			maxScanLine = pos
		}
		negative := false
		if c == 0 {
			negative = reader.ReadBinaryCDFUnchecked(dcSignCDF) != 0
		} else {
			bit := reader.ReadBitTrusted()
			negative = bit != 0
		}
		baseLevel := level
		golombExtra := 0
		if level >= MaxBaseBRRange {
			tail, err := readCoeffGolombCursor(&reader)
			if err != nil {
				reader.CommitStateTo(&s.Reader)
				return TXBDecodeResult{}, err
			}
			golombExtra = tail
			level += tail
		}
		signBit := 0
		if negative {
			signBit = 1
		}
		if coeffTraceEnabled {
			coeffTraceCoeff(c, pos, baseLevel, golombExtra, level, signBit)
		}
		culLevel += level
		if level > int(^uint16(0)>>1) {
			reader.CommitStateTo(&s.Reader)
			return TXBDecodeResult{}, ErrInvalidDecodeState
		}
		signed := int16(level)
		if negative {
			signed = -signed
		}
		if c == 0 {
			dcValue = int(signed)
		}
		coeffs[pos] = signed
		req.recordCoeffDirty(pos)
		c = nextC
	}

	if culLevel > CoeffContextMask {
		culLevel = CoeffContextMask
	}
	if dcValue < 0 {
		culLevel |= 1 << CoeffContextBits
	} else if dcValue > 0 {
		culLevel += 2 << CoeffContextBits
	}

	if coeffTraceEnabled {
		coeffTraceCulLevel(culLevel)
	}
	reader.CommitStateTo(&s.Reader)
	return TXBDecodeResult{
		EOB:         eob.Position,
		MaxScanLine: maxScanLine,
		CulLevel:    culLevel,
	}, nil
}

func (req *TXBDecodeRequest) recordCoeffDirty(pos int) {
	if req == nil || req.coeffDirtyPos == nil || req.coeffDirtyLen == nil {
		return
	}
	n := *req.coeffDirtyLen
	if n >= len(req.coeffDirtyPos) || uint(pos) >= uint(len(req.coeffDirtyPos)) {
		return
	}
	req.coeffDirtyPos[n] = int16(pos)
	*req.coeffDirtyLen = n + 1
}

func readBaseRangeFromArrCursor(reader *entropy.Cursor, arr *[CoeffBRContexts]entropy.CDF, context int) int {
	return reader.ReadCDF4HighTokenUnchecked(&arr[context])
}

func readCoeffGolombCursor(reader *entropy.Cursor) (int, error) {
	x := 1
	length := 0
	for {
		bit := reader.ReadBitTrusted()
		length++
		if length > 20 {
			return 0, ErrInvalidDecodeState
		}
		if bit != 0 {
			break
		}
	}
	if length > 1 {
		suffix := reader.ReadBitsTrusted(uint8(length - 1))
		x = (1 << (length - 1)) | int(suffix)
	}
	return x - 1, nil
}

// coeffCDF returns a coefficient CDF for a trusted read. It verifies only the
// cheap structural invariant (correct symbol count); the per-symbol
// monotonicity scan is intentionally omitted because coefficient CDFs are valid
// by construction (default-seeded, adapted only by updateCDFWindow) and the
// trusted reader does not require it. This drops the redundant O(symbols)
// validation that previously ran on every coefficient symbol.
func coeffCDF(cdf *entropy.CDF, symbols int) (*entropy.CDF, error) {
	if cdf == nil || cdf.Symbols() != symbols {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, nil
}

// coeffBinaryCDF returns a 2-symbol coefficient CDF for a trusted boolean read.
// Like coeffCDF it performs only the structural symbol-count check and leaves
// the monotonicity validation to the (trusted) reader's structural guard.
func coeffBinaryCDF(cdf *entropy.CDF) (*entropy.CDF, error) {
	if cdf == nil || cdf.Symbols() != 2 {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, nil
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
	if size >= transformSizeCount {
		return 0, 0, 0, 0, ErrInvalidDecodeState
	}
	g := &coeffGeometryTable[size]
	if !g.valid || coeffIndex < 0 || coeffIndex >= g.maxEOB {
		return 0, 0, 0, 0, ErrInvalidDecodeState
	}
	p := coeffPosTable[size][coeffIndex]
	return int(p.padded), g.stride, int(p.row), int(p.col), nil
}

func coeffPosition(size TransformSize, coeffIndex int) (row int, col int, stride int, err error) {
	if size >= transformSizeCount {
		return 0, 0, 0, ErrInvalidDecodeState
	}
	g := &coeffGeometryTable[size]
	if !g.valid || coeffIndex < 0 || coeffIndex >= g.maxEOB {
		return 0, 0, 0, ErrInvalidDecodeState
	}
	p := coeffPosTable[size][coeffIndex]
	return int(p.row), int(p.col), g.stride, nil
}

// coeffLowerLevelsCtxFast is the hot-loop variant of CoeffLowerLevelsContext.
// The caller passes the transform's precomputed geometry and position slice
// (hoisted once per TXB), so this avoids the per-coefficient
// coeffGeometryTable/coeffPosTable indexing and the Class2D Dimensions() call.
// The TXB loop has already proven pos and class valid and levels large enough
// for this geometry.
func coeffLowerLevelsCtxFast(levels []uint8, geo *coeffGeometry, posSlice []coeffPos, class transform.Class, pos int) int {
	p := posSlice[pos]
	padded := int(p.padded)
	stride := geo.stride
	// Build neighbour offsets as additive induction chains. The final dummy
	// reads prove the maxima to the bounds-check eliminator; the TXB loop has
	// already validated the table-derived geometry.
	s1 := padded + stride // padded + stride
	s1p1 := s1 + 1        // padded + stride + 1
	s2 := s1 + stride     // padded + 2*stride
	s3 := s2 + stride     // padded + 3*stride
	s4 := s3 + stride     // padded + 4*stride
	p1 := padded + 1
	p2 := padded + 2
	p3 := padded + 3
	p4 := padded + 4
	_ = levels[s4]
	_ = levels[p4]
	if class == transform.Class2D && pos == 0 {
		return 0
	}
	row := int(p.row)
	col := int(p.col)

	mag := clipMax3(levels[s1]) + clipMax3(levels[p1])
	switch class {
	case transform.Class2D:
		mag += clipMax3(levels[s1p1])
		mag += clipMax3(levels[s2])
		mag += clipMax3(levels[p2])
	case transform.ClassVert:
		mag += clipMax3(levels[p2])
		mag += clipMax3(levels[p3])
		mag += clipMax3(levels[p4])
	case transform.ClassHoriz:
		mag += clipMax3(levels[s2])
		mag += clipMax3(levels[s3])
		mag += clipMax3(levels[s4])
	}
	ctx := minInt((mag+1)>>1, 4)

	switch class {
	case transform.Class2D:
		if geo.txTall && row < 2 {
			return ctx + 11
		}
		if geo.txWide && col < 2 {
			return ctx + 16
		}
		if row+col < 2 {
			return ctx + 1
		}
		if row+col < 4 {
			return ctx + 6
		}
		return ctx + 21
	case transform.ClassHoriz:
		return ctx + coeff1DContextOffset(col)
	}
	return ctx + coeff1DContextOffset(row)
}

func coeffLowerLevelsCtx2DFast(levels []uint8, posSlice []coeffPos, pos int, stride int) int {
	p := posSlice[pos]
	if pos == 0 {
		return 0
	}
	padded := int(p.padded)
	s1 := padded + stride
	s1p1 := s1 + 1
	s2 := s1 + stride
	p1 := padded + 1
	p2 := padded + 2
	_ = levels[s2]
	_ = levels[p2]

	mag := clipMax3(levels[s1]) + clipMax3(levels[p1]) +
		clipMax3(levels[s1p1]) + clipMax3(levels[s2]) + clipMax3(levels[p2])
	ctx := minInt((mag+1)>>1, 4)
	return ctx + int(p.lower2DOffset)
}

// coeffBRContextFast is the hot-loop variant of CoeffBRContext using the
// hoisted geometry and position slice.
func coeffBRContextFast(levels []uint8, geo *coeffGeometry, posSlice []coeffPos, class transform.Class, pos int) int {
	p := posSlice[pos]
	padded := int(p.padded)
	stride := geo.stride
	// Same chained-induction BCE as coeffLowerLevelsCtxFast.
	s1 := padded + stride // padded + stride
	s1p1 := s1 + 1        // padded + stride + 1
	s2 := s1 + stride     // padded + 2*stride
	s2p2 := s2 + 2        // padded + 2*stride + 2
	p1 := padded + 1
	p2 := padded + 2
	p4 := padded + 4
	_ = levels[s2p2]
	_ = levels[p4]
	row := int(p.row)
	col := int(p.col)
	if class == transform.Class2D && pos != 0 {
		mag := minInt(int(levels[p1]), MaxBaseBRRange) +
			minInt(int(levels[s1]), MaxBaseBRRange) +
			minInt(int(levels[s1p1]), MaxBaseBRRange)
		mag = minInt((mag+1)>>1, 6)
		if row < 2 && col < 2 {
			return mag + 7
		}
		return mag + 14
	}

	mag := int(levels[p1]) + int(levels[s1])
	switch class {
	case transform.Class2D:
		mag += int(levels[s1p1])
	case transform.ClassHoriz:
		mag += int(levels[s2])
	case transform.ClassVert:
		mag += int(levels[p2])
	}
	mag = minInt((mag+1)>>1, 6)
	if pos == 0 {
		return mag
	}
	switch class {
	case transform.Class2D:
		if row < 2 && col < 2 {
			return mag + 7
		}
	case transform.ClassHoriz:
		if col == 0 {
			return mag + 7
		}
	case transform.ClassVert:
		if row == 0 {
			return mag + 7
		}
	}
	return mag + 14
}

func coeffBRContext2DFast(levels []uint8, posSlice []coeffPos, pos int, stride int) int {
	p := posSlice[pos]
	padded := int(p.padded)
	s1 := padded + stride
	s1p1 := s1 + 1
	p1 := padded + 1
	_ = levels[s1p1]
	mag := int(levels[p1]) + int(levels[s1]) + int(levels[s1p1])
	return minInt((mag+1)>>1, 6) + int(p.br2DOffset)
}

func coeffBRContextEOBFast(posSlice []coeffPos, class transform.Class, pos int) int {
	if pos == 0 {
		return 0
	}
	p := posSlice[pos]
	switch class {
	case transform.Class2D:
		if p.row < 2 && p.col < 2 {
			return 7
		}
	case transform.ClassHoriz:
		if p.col == 0 {
			return 7
		}
	case transform.ClassVert:
		if p.row == 0 {
			return 7
		}
	}
	return 14
}

func coeffLowerLevelsCtxEOBFast(maxEOB int, scanIndex int) int {
	if scanIndex == 0 {
		return 0
	}
	if scanIndex <= maxEOB/8 {
		return 1
	}
	if scanIndex <= maxEOB/4 {
		return 2
	}
	return 3
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
