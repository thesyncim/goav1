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

// CoeffCDFs is dav1d's CdfCoefContext; entropy.(*Reader).DecodeCoefs reads
// it as an entropy.CoefCDFs.
type CoeffCDFs entropy.CoefCDFs

var _ [0]struct{} = [CoeffTxSizeContexts - len(CoeffCDFs{}.TXBSkip)]struct{}{}
var _ [TXBSkipContexts]struct{} = [len(CoeffCDFs{}.TXBSkip[0])]struct{}{}
var _ [EOBCoefContexts]struct{} = [len(CoeffCDFs{}.EOBExtra[0][0])]struct{}{}
var _ [CoeffBRContexts]struct{} = [len(CoeffCDFs{}.CoeffBR[0][0])]struct{}{}
var _ [CoeffBaseContexts]struct{} = [len(CoeffCDFs{}.CoeffBase[0][0])]struct{}{}
var _ [EOBBaseContexts]struct{} = [len(CoeffCDFs{}.CoeffBaseEOB[0][0])]struct{}{}
var _ [maxEOBFlagContexts]struct{} = [len(CoeffCDFs{}.EOBFlag16[0])]struct{}{}

// Equal reports whether c and o hold identical CDFs; it is the test c == *o
// would make, written as one loop per CDF row. TinyGo expands == on an
// aggregate into one comparison per element, so comparing a CoeffCDFs inline
// costs hundreds of kilobytes of wasm in the caller, and that makes wasm-opt
// run for many minutes and tens of gigabytes. TestCoeffCDFsEqualMatchesOperator
// pins that every field is compared.
func (c *CoeffCDFs) Equal(o *CoeffCDFs) bool {
	for i := range c.TXBSkip {
		if !cdfsEqual(c.TXBSkip[i][:], o.TXBSkip[i][:]) {
			return false
		}
	}
	for i := range c.EOBExtra {
		for j := range c.EOBExtra[i] {
			if !cdfsEqual(c.EOBExtra[i][j][:], o.EOBExtra[i][j][:]) {
				return false
			}
		}
	}
	for i := range c.DCSign {
		if !cdfsEqual(c.DCSign[i][:], o.DCSign[i][:]) {
			return false
		}
	}
	for i := range c.CoeffBR {
		for j := range c.CoeffBR[i] {
			if !cdfsEqual(c.CoeffBR[i][j][:], o.CoeffBR[i][j][:]) {
				return false
			}
		}
	}
	for i := range c.CoeffBase {
		for j := range c.CoeffBase[i] {
			if !cdfsEqual(c.CoeffBase[i][j][:], o.CoeffBase[i][j][:]) {
				return false
			}
		}
	}
	for i := range c.CoeffBaseEOB {
		for j := range c.CoeffBaseEOB[i] {
			if !cdfsEqual(c.CoeffBaseEOB[i][j][:], o.CoeffBaseEOB[i][j][:]) {
				return false
			}
		}
	}
	for i := range c.EOBFlag16 {
		if !cdfsEqual(c.EOBFlag16[i][:], o.EOBFlag16[i][:]) {
			return false
		}
	}
	for i := range c.EOBFlag32 {
		if !cdfsEqual(c.EOBFlag32[i][:], o.EOBFlag32[i][:]) {
			return false
		}
	}
	for i := range c.EOBFlag64 {
		if !cdfsEqual(c.EOBFlag64[i][:], o.EOBFlag64[i][:]) {
			return false
		}
	}
	for i := range c.EOBFlag128 {
		if !cdfsEqual(c.EOBFlag128[i][:], o.EOBFlag128[i][:]) {
			return false
		}
	}
	for i := range c.EOBFlag256 {
		if !cdfsEqual(c.EOBFlag256[i][:], o.EOBFlag256[i][:]) {
			return false
		}
	}
	for i := range c.EOBFlag512 {
		if !cdfsEqual(c.EOBFlag512[i][:], o.EOBFlag512[i][:]) {
			return false
		}
	}
	for i := range c.EOBFlag1024 {
		if !cdfsEqual(c.EOBFlag1024[i][:], o.EOBFlag1024[i][:]) {
			return false
		}
	}
	return true
}

func cdfsEqual(a, b []entropy.CDF) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// CopyFrom is *c = *o as one copy per field: TinyGo lowers the assignment of
// a struct this large through a heap temporary.
func (c *CoeffCDFs) CopyFrom(o *CoeffCDFs) {
	copy(c.TXBSkip[:], o.TXBSkip[:])
	copy(c.EOBExtra[:], o.EOBExtra[:])
	copy(c.DCSign[:], o.DCSign[:])
	copy(c.CoeffBR[:], o.CoeffBR[:])
	copy(c.CoeffBase[:], o.CoeffBase[:])
	copy(c.CoeffBaseEOB[:], o.CoeffBaseEOB[:])
	copy(c.EOBFlag16[:], o.EOBFlag16[:])
	copy(c.EOBFlag32[:], o.EOBFlag32[:])
	copy(c.EOBFlag64[:], o.EOBFlag64[:])
	copy(c.EOBFlag128[:], o.EOBFlag128[:])
	copy(c.EOBFlag256[:], o.EOBFlag256[:])
	copy(c.EOBFlag512[:], o.EOBFlag512[:])
	copy(c.EOBFlag1024[:], o.EOBFlag1024[:])
}

type TXBDecodeResult struct {
	EOB         uint16
	MaxScanLine uint16
	CulLevel    uint8
	AllZero     bool
}

var eobGroupStart = [12]uint16{0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513}
var eobOffsetBits = [12]uint8{0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
var eobToPosSmall = [33]uint8{
	0, 1, 2,
	3, 3,
	4, 4, 4, 4,
	5, 5, 5, 5, 5, 5, 5, 5,
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
}
var eobToPosLarge = [17]uint8{
	6,
	7,
	8, 8,
	9, 9, 9, 9,
	10, 10, 10, 10, 10, 10, 10, 10,
	11,
}
var eobMultiSizeTable = [transformSizeCount]uint8{
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
	scanWidth  uint8
	scanHeight uint8
	stride     uint8  // scanHeight + txPadHorizontal
	maxEOB     uint16 // scanWidth * scanHeight
	scratchLen uint16 // (scanWidth+pad) * (scanHeight+pad)
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
var coeffLower2DOffsetTable [transformSizeCount][]uint8
var coeffLowerHorizOffsetTable [transformSizeCount][]uint8
var coeffLowerVertOffsetTable [transformSizeCount][]uint8

// coeffPos is the precomputed scan position for one coefficient index: the
// padded scratch offset (col*stride+row) plus the unpadded row/col. Computing
// these once per transform size at init turns the per-coefficient
// index->position math (a data-dependent division) into a table lookup on the
// hot coefficient-read path. row/col stay below 32 and padded stays below
// scratchLen, all well within uint8/uint16.
type coeffPos struct {
	padded        uint16
	row           uint8
	col           uint8
	lower2DOffset int8
	br2DOffset    int8
}

type coeffPosHot struct {
	padded        uint16
	lower2DOffset int8
	br2DOffset    int8
}

type coeffScanHot = entropy.CoeffScanContext

// coeffPosTable[size][coeffIndex] holds the precomputed position for every
// valid coefficient index of size, indexed in [0, maxEOB).
var coeffPosTable [transformSizeCount][]coeffPos
var coeffPosHotTable [transformSizeCount][]coeffPosHot
var coeffScanTable [transformSizeCount][3][]int16
var coeffScanHotTable [transformSizeCount][3][]coeffScanHot
var coeffScanHot4x4Y2D [16]coeffScanHot
var coeffScanHot8x8Y2D [64]coeffScanHot
var coeffScanHot16x16Y2D [256]coeffScanHot
var coeffScanHot32x32Y2D [1024]coeffScanHot

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
		scanWidth := int(scanSize.Width)
		scanHeight := int(scanSize.Height)
		stride := scanHeight + txPadHorizontal
		maxEOB := scanWidth * scanHeight
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
			scanWidth:  uint8(scanSize.Width),
			scanHeight: uint8(scanSize.Height),
			stride:     uint8(stride),
			maxEOB:     uint16(maxEOB),
			scratchLen: uint16((scanWidth + txPadHorizontal) * stride),
			txCtx:      txCtx,
			txBRCtx:    txBRCtx,
			txWide:     txWide,
			txTall:     txTall,
		}
		positions := make([]coeffPos, maxEOB)
		for idx := range maxEOB {
			col := idx / scanHeight
			row := idx - col*scanHeight
			positions[idx] = coeffPos{
				padded: uint16(col*stride + row),
				row:    uint8(row),
				col:    uint8(col),
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
		lower2DOffsets := make([]uint8, maxEOB)
		lowerHorizOffsets := make([]uint8, maxEOB)
		lowerVertOffsets := make([]uint8, maxEOB)
		for idx, p := range positions {
			lower2DOffsets[idx] = uint8(p.lower2DOffset)
			lowerHorizOffsets[idx] = uint8(coeff1DContextOffset(int(p.col)))
			lowerVertOffsets[idx] = uint8(coeff1DContextOffset(int(p.row)))
		}
		coeffLower2DOffsetTable[size] = lower2DOffsets
		coeffLowerHorizOffsetTable[size] = lowerHorizOffsets
		coeffLowerVertOffsetTable[size] = lowerVertOffsets
		hotPositions := make([]coeffPosHot, maxEOB)
		for idx, p := range positions {
			hotPositions[idx] = coeffPosHot{
				padded:        p.padded,
				lower2DOffset: p.lower2DOffset,
				br2DOffset:    p.br2DOffset,
			}
		}
		coeffPosHotTable[size] = hotPositions
		for class := transform.Class2D; class <= transform.ClassVert; class++ {
			if scan, err := transform.DefaultScan(txSize, class); err == nil {
				coeffScanTable[size][class] = scan
				if class == transform.Class2D {
					scanHot := make([]coeffScanHot, maxEOB)
					for c, rawPos := range scan {
						pos := int(rawPos)
						p := hotPositions[pos]
						scanHot[c] = coeffScanHot{
							Pos:           uint16(pos),
							Padded:        p.padded,
							Lower2DOffset: p.lower2DOffset,
							BR2DOffset:    p.br2DOffset,
							LowerEOBCtx:   coeffLowerLevelsCtxEOBFast(maxEOB, c),
							BREOBCtx:      coeffBRContextEOBFast(positions[pos], class, pos),
						}
					}
					coeffScanHotTable[size][class] = scanHot
					if size == TransformSize4x4 {
						copy(coeffScanHot4x4Y2D[:], scanHot)
					}
					if size == TransformSize8x8 {
						copy(coeffScanHot8x8Y2D[:], scanHot)
					}
					if size == TransformSize16x16 {
						copy(coeffScanHot16x16Y2D[:], scanHot)
					}
					if size == TransformSize32x32 {
						copy(coeffScanHot32x32Y2D[:], scanHot)
					}
				}
			}
		}
	}
}

// coeffGeo returns the precomputed scan geometry for size. The bool reports
// whether size names a valid transform.
func coeffGeo(size TransformSize) (coeffGeometry, bool) {
	g, ok := coeffGeoPtr(size)
	if !ok {
		return coeffGeometry{}, false
	}
	return *g, true
}

func coeffGeoPtr(size TransformSize) (*coeffGeometry, bool) {
	if size >= transformSizeCount {
		return nil, false
	}
	g := &coeffGeometryTable[size]
	return g, g.valid
}

func CoeffQContext(baseQIndex uint8) uint8 {
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
func CoeffTransformSizeContext(size TransformSize) (uint8, error) {
	dims, ok := size.Dimensions()
	if !ok {
		return 0, ErrInvalidDecodeState
	}
	return (dims.Min + dims.Max + 1) >> 1, nil
}

func EOBPositionToken(eob int) (token int, extra int, err error) {
	if eob < 0 || eob > 1024 {
		return 0, 0, ErrInvalidDecodeState
	}
	if eob < len(eobToPosSmall) {
		token = int(eobToPosSmall[eob])
	} else {
		index := min((eob-1)>>5, 16)
		token = int(eobToPosLarge[index])
	}
	extra = eob - int(eobGroupStart[token])
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
	return int(g.scratchLen), nil
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
	scanWidth := int(scanSize.Width)
	scanHeight := int(scanSize.Height)
	maxEOB := scanWidth * scanHeight
	scratchLen, err := CoeffLevelsScratchLen(size)
	if err != nil {
		return err
	}
	if len(coeffs) < maxEOB || len(levels) < scratchLen {
		return ErrInvalidDecodeState
	}
	if coeffInitLevelsArch(coeffs, scanWidth, scanHeight, levels, scratchLen) {
		return nil
	}
	coeffInitLevelsPureGo(coeffs, scanWidth, scanHeight, levels, scratchLen)
	return nil
}

func coeffInitLevelsPureGo(coeffs []int16, scanWidth int, scanHeight int, levels []uint8, scratchLen int) {
	for i := range scratchLen {
		levels[i] = 0
	}
	stride := scanHeight + txPadHorizontal
	for col := 0; col < scanWidth; col++ {
		src := col * scanHeight
		dst := col * stride
		for row := 0; row < scanHeight; row++ {
			levels[dst+row] = coeffAbsClamp127(coeffs[src+row])
		}
	}
}

var defaultCoeffCDFs = makeDefaultCoeffCDFs()

func makeDefaultCoeffCDFs() [CoeffQContexts]CoeffCDFs {
	var defaults [CoeffQContexts]CoeffCDFs
	for q := range defaults {
		if err := initCoeffCDFsForQ(&defaults[q], q); err != nil {
			panic(err)
		}
	}
	return defaults
}

func initCoeffCDFsForQ(c *CoeffCDFs, q int) error {
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

// DefaultCoeffCDFs returns the read-only default coefficient CDF set
// InitDefault copies for baseQIndex. Callers must not modify it.
func DefaultCoeffCDFs(baseQIndex uint8) *CoeffCDFs {
	return &defaultCoeffCDFs[CoeffQContext(baseQIndex)]
}

func (c *CoeffCDFs) InitDefault(baseQIndex uint8) error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	*c = defaultCoeffCDFs[CoeffQContext(baseQIndex)]
	return nil
}

func eobFlagCDFKnown(cdfs *CoeffCDFs, size TransformSize, plane CoeffPlaneType, context uint8) *entropy.CDF {
	switch eobMultiSizeTable[size] {
	case 0:
		return &cdfs.EOBFlag16[plane][context]
	case 1:
		return &cdfs.EOBFlag32[plane][context]
	case 2:
		return &cdfs.EOBFlag64[plane][context]
	case 3:
		return &cdfs.EOBFlag128[plane][context]
	case 4:
		return &cdfs.EOBFlag256[plane][context]
	case 5:
		return &cdfs.EOBFlag512[plane][context]
	default:
		return &cdfs.EOBFlag1024[plane][context]
	}
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

func coeffPaddedPosition(size TransformSize, coeffIndex int) (padded int, stride int, row int, col int, err error) {
	if size >= transformSizeCount {
		return 0, 0, 0, 0, ErrInvalidDecodeState
	}
	g := &coeffGeometryTable[size]
	if !g.valid || coeffIndex < 0 || coeffIndex >= int(g.maxEOB) {
		return 0, 0, 0, 0, ErrInvalidDecodeState
	}
	p := coeffPosTable[size][coeffIndex]
	return int(p.padded), int(g.stride), int(p.row), int(p.col), nil
}

func coeffPosition(size TransformSize, coeffIndex int) (row int, col int, stride int, err error) {
	if size >= transformSizeCount {
		return 0, 0, 0, ErrInvalidDecodeState
	}
	g := &coeffGeometryTable[size]
	if !g.valid || coeffIndex < 0 || coeffIndex >= int(g.maxEOB) {
		return 0, 0, 0, ErrInvalidDecodeState
	}
	p := coeffPosTable[size][coeffIndex]
	return int(p.row), int(p.col), int(g.stride), nil
}

// coeffLowerLevelsCtxFast is the hot-loop variant of CoeffLowerLevelsContext.
// The caller passes the transform's precomputed geometry and position slice
// (hoisted once per TXB), so this avoids the per-coefficient
// coeffGeometryTable/coeffPosTable indexing and the Class2D Dimensions() call.
// The TXB loop has already proven pos and class valid and levels large enough
// for this geometry.
func coeffLowerLevelsCtxFast(levels []uint8, geo *coeffGeometry, posSlice []coeffPos, class transform.Class, pos int) uint8 {
	p := posSlice[pos]
	padded := int(p.padded)
	stride := int(geo.stride)
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
			return uint8(ctx + 11)
		}
		if geo.txWide && col < 2 {
			return uint8(ctx + 16)
		}
		if row+col < 2 {
			return uint8(ctx + 1)
		}
		if row+col < 4 {
			return uint8(ctx + 6)
		}
		return uint8(ctx + 21)
	case transform.ClassHoriz:
		return uint8(ctx + coeff1DContextOffset(col))
	}
	return uint8(ctx + coeff1DContextOffset(row))
}

// coeffBRContextFast is the hot-loop variant of CoeffBRContext using the
// hoisted geometry and position slice.
func coeffBRContextFast(levels []uint8, geo *coeffGeometry, posSlice []coeffPos, class transform.Class, pos int) uint8 {
	p := posSlice[pos]
	padded := int(p.padded)
	stride := int(geo.stride)
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
		if row|col < 2 {
			return uint8(mag + 7)
		}
		return uint8(mag + 14)
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
		return uint8(mag)
	}
	switch class {
	case transform.Class2D:
		if row|col < 2 {
			return uint8(mag + 7)
		}
	case transform.ClassHoriz:
		if col == 0 {
			return uint8(mag + 7)
		}
	case transform.ClassVert:
		if row == 0 {
			return uint8(mag + 7)
		}
	}
	return uint8(mag + 14)
}

func coeffBRContextEOBFast(p coeffPos, class transform.Class, pos int) uint8 {
	if pos == 0 {
		return 0
	}
	switch class {
	case transform.Class2D:
		if p.row|p.col < 2 {
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

func coeffLowerLevelsCtxEOBFast(maxEOB int, scanIndex int) uint8 {
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
	x := int(v)
	// Branchless min(x, 3): for x < 3, the signed shift is -1 and keeps x-3.
	return 3 + ((x - 3) & ((x - 3) >> 31))
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

// CoeffPaddedIndex is libaom/SVT get_padded_idx in this package's coefficient
// layout: the index of coefficient coeffIndex in the padded level map that
// CoeffInitLevels fills. size and coeffIndex must be valid.
func CoeffPaddedIndex(size TransformSize, coeffIndex int) int {
	return int(coeffPosTable[size][coeffIndex].padded)
}

// CoeffLowerLevelsContextTrusted is CoeffLowerLevelsContext for a valid size,
// class and coefficient index and a levels map CoeffInitLevels filled.
func CoeffLowerLevelsContextTrusted(levels []uint8, size TransformSize, class transform.Class, coeffIndex int) int {
	return int(coeffLowerLevelsCtxFast(levels, &coeffGeometryTable[size], coeffPosTable[size], class, coeffIndex))
}

// CoeffBRContextTrusted is CoeffBRContext under the same contract as
// CoeffLowerLevelsContextTrusted.
func CoeffBRContextTrusted(levels []uint8, size TransformSize, class transform.Class, coeffIndex int) int {
	return int(coeffBRContextFast(levels, &coeffGeometryTable[size], coeffPosTable[size], class, coeffIndex))
}
