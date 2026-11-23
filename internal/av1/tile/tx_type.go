package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

const (
	ExtTXSetTypes  = 6
	ExtTXSetsIntra = 3
	ExtTXSetsInter = 4
	ExtTXSizes     = 4
)

type ExtTXSetType uint8

const (
	ExtTXSetDCTOnly ExtTXSetType = iota
	ExtTXSetDCTIDTX
	ExtTXSetDTT4IDTX
	ExtTXSetDTT4IDTX1DDCT
	ExtTXSetDTT9IDTX1DDCT
	ExtTXSetAll16
)

type TransformTypeCDFs struct {
	Intra [ExtTXSetsIntra][ExtTXSizes][intraModeCount]entropy.CDF
	Inter [ExtTXSetsInter][ExtTXSizes]entropy.CDF
}

type IntraTransformTypeRequest struct {
	Size          TransformSize
	Mode          IntraMode
	ReducedTXSet  bool
	SkipTransform bool
	Lossless      bool
	QIndexKnown   bool
	QIndex        uint8
}

type InterTransformTypeRequest struct {
	Size          TransformSize
	ReducedTXSet  bool
	SkipTransform bool
	Lossless      bool
	QIndexKnown   bool
	QIndex        uint8
}

var transformSizeSquare = [transformSizeCount]TransformSize{
	TransformSize4x4:   TransformSize4x4,
	TransformSize8x8:   TransformSize8x8,
	TransformSize16x16: TransformSize16x16,
	TransformSize32x32: TransformSize32x32,
	TransformSize64x64: TransformSize64x64,
	TransformSize4x8:   TransformSize4x4,
	TransformSize8x4:   TransformSize4x4,
	TransformSize8x16:  TransformSize8x8,
	TransformSize16x8:  TransformSize8x8,
	TransformSize16x32: TransformSize16x16,
	TransformSize32x16: TransformSize16x16,
	TransformSize32x64: TransformSize32x32,
	TransformSize64x32: TransformSize32x32,
	TransformSize4x16:  TransformSize4x4,
	TransformSize16x4:  TransformSize4x4,
	TransformSize8x32:  TransformSize8x8,
	TransformSize32x8:  TransformSize8x8,
	TransformSize16x64: TransformSize16x16,
	TransformSize64x16: TransformSize16x16,
}

var transformSizeSquareUp = [transformSizeCount]TransformSize{
	TransformSize4x4:   TransformSize4x4,
	TransformSize8x8:   TransformSize8x8,
	TransformSize16x16: TransformSize16x16,
	TransformSize32x32: TransformSize32x32,
	TransformSize64x64: TransformSize64x64,
	TransformSize4x8:   TransformSize8x8,
	TransformSize8x4:   TransformSize8x8,
	TransformSize8x16:  TransformSize16x16,
	TransformSize16x8:  TransformSize16x16,
	TransformSize16x32: TransformSize32x32,
	TransformSize32x16: TransformSize32x32,
	TransformSize32x64: TransformSize64x64,
	TransformSize64x32: TransformSize64x64,
	TransformSize4x16:  TransformSize16x16,
	TransformSize16x4:  TransformSize16x16,
	TransformSize8x32:  TransformSize32x32,
	TransformSize32x8:  TransformSize32x32,
	TransformSize16x64: TransformSize64x64,
	TransformSize64x16: TransformSize64x64,
}

var extTXSetLookup = [2][2]ExtTXSetType{
	{ExtTXSetDTT4IDTX1DDCT, ExtTXSetDTT4IDTX},
	{ExtTXSetAll16, ExtTXSetDTT9IDTX1DDCT},
}

var extTXSetIndex = [2][ExtTXSetTypes]int8{
	{0, -1, 2, 1, -1, -1},
	{0, 3, -1, -1, 2, 1},
}

var extTXTypeCount = [ExtTXSetTypes]uint8{1, 2, 5, 7, 12, 16}

var extTXTypeUsedFlag = [ExtTXSetTypes]uint16{
	ExtTXSetDCTOnly:       0x0001,
	ExtTXSetDCTIDTX:       0x0201,
	ExtTXSetDTT4IDTX:      0x020f,
	ExtTXSetDTT4IDTX1DDCT: 0x0e0f,
	ExtTXSetDTT9IDTX1DDCT: 0x0fff,
	ExtTXSetAll16:         0xffff,
}

var extTXTypeInd = [ExtTXSetTypes][transform.TypeCount]int8{
	ExtTXSetDCTOnly:       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	ExtTXSetDCTIDTX:       {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	ExtTXSetDTT4IDTX:      {1, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	ExtTXSetDTT4IDTX1DDCT: {1, 5, 6, 4, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0},
	ExtTXSetDTT9IDTX1DDCT: {3, 4, 5, 8, 6, 7, 9, 10, 11, 0, 1, 2, 0, 0, 0, 0},
	ExtTXSetAll16:         {7, 8, 9, 12, 10, 11, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6},
}

var extTXTypeInv = [ExtTXSetTypes][16]transform.Type{
	ExtTXSetDCTOnly:       {transform.TypeDCTDCT},
	ExtTXSetDCTIDTX:       {transform.TypeIDTX, transform.TypeDCTDCT},
	ExtTXSetDTT4IDTX:      {transform.TypeIDTX, transform.TypeDCTDCT, transform.TypeADSTADST, transform.TypeADSTDCT, transform.TypeDCTADST},
	ExtTXSetDTT4IDTX1DDCT: {transform.TypeIDTX, transform.TypeDCTDCT, transform.TypeVDCT, transform.TypeHDCT, transform.TypeADSTADST, transform.TypeADSTDCT, transform.TypeDCTADST},
	ExtTXSetDTT9IDTX1DDCT: {transform.TypeIDTX, transform.TypeVDCT, transform.TypeHDCT, transform.TypeDCTDCT, transform.TypeADSTDCT, transform.TypeDCTADST, transform.TypeFlipADSTDCT, transform.TypeDCTFlipADST, transform.TypeADSTADST, transform.TypeFlipADSTFlipADST, transform.TypeADSTFlipADST, transform.TypeFlipADSTADST},
	ExtTXSetAll16:         {transform.TypeIDTX, transform.TypeVDCT, transform.TypeHDCT, transform.TypeVADST, transform.TypeHADST, transform.TypeVFlipADST, transform.TypeHFlipADST, transform.TypeDCTDCT, transform.TypeADSTDCT, transform.TypeDCTADST, transform.TypeFlipADSTDCT, transform.TypeDCTFlipADST, transform.TypeADSTADST, transform.TypeFlipADSTFlipADST, transform.TypeADSTFlipADST, transform.TypeFlipADSTADST},
}

func TransformSizeSquare(size TransformSize) (TransformSize, error) {
	if !size.Valid() {
		return 0, ErrInvalidDecodeState
	}
	return transformSizeSquare[size], nil
}

func TransformSizeSquareUp(size TransformSize) (TransformSize, error) {
	if !size.Valid() {
		return 0, ErrInvalidDecodeState
	}
	return transformSizeSquareUp[size], nil
}

func ExtTXSetTypeFor(size TransformSize, inter bool, reduced bool) (ExtTXSetType, error) {
	up, err := TransformSizeSquareUp(size)
	if err != nil {
		return 0, err
	}
	if up > TransformSize32x32 {
		return ExtTXSetDCTOnly, nil
	}
	if up == TransformSize32x32 {
		if inter {
			return ExtTXSetDCTIDTX, nil
		}
		return ExtTXSetDCTOnly, nil
	}
	if reduced {
		if inter {
			return ExtTXSetDCTIDTX, nil
		}
		return ExtTXSetDTT4IDTX, nil
	}
	square, err := TransformSizeSquare(size)
	if err != nil {
		return 0, err
	}
	column := 0
	if square == TransformSize16x16 {
		column = 1
	}
	row := 0
	if inter {
		row = 1
	}
	return extTXSetLookup[row][column], nil
}

func ExtTXSetIndex(size TransformSize, inter bool, reduced bool) (int, error) {
	set, err := ExtTXSetTypeFor(size, inter, reduced)
	if err != nil {
		return 0, err
	}
	row := 0
	if inter {
		row = 1
	}
	index := extTXSetIndex[row][set]
	if index < 0 {
		return 0, ErrInvalidDecodeState
	}
	return int(index), nil
}

func ExtTXTypeCount(set ExtTXSetType) (int, error) {
	if set >= ExtTXSetTypes {
		return 0, ErrInvalidDecodeState
	}
	return int(extTXTypeCount[set]), nil
}

func ExtTXTypeAllowed(set ExtTXSetType, typ transform.Type) (bool, error) {
	if set >= ExtTXSetTypes || !typ.Valid() {
		return false, ErrInvalidDecodeState
	}
	return extTXTypeUsedFlag[set]&(1<<typ) != 0, nil
}

func ExtTXSymbolForType(set ExtTXSetType, typ transform.Type) (int, error) {
	allowed, err := ExtTXTypeAllowed(set, typ)
	if err != nil || !allowed {
		return 0, ErrInvalidDecodeState
	}
	return int(extTXTypeInd[set][typ]), nil
}

// IntraChromaTransformType derives the chroma intra tx_type from AV1's UV mode
// mapping and clamps it to DCT_DCT when the current tx set disallows the type.
func IntraChromaTransformType(mode ChromaIntraMode, size TransformSize, reduced bool) (transform.Type, error) {
	lumaMode, err := mode.LumaMode()
	if err != nil {
		return 0, err
	}
	typ, err := intraModeTransformType(lumaMode)
	if err != nil {
		return 0, err
	}
	set, err := ExtTXSetTypeFor(size, false, reduced)
	if err != nil {
		return 0, err
	}
	allowed, err := ExtTXTypeAllowed(set, typ)
	if err != nil {
		return 0, err
	}
	if !allowed {
		return transform.TypeDCTDCT, nil
	}
	return typ, nil
}

var defaultTransformTypeCDFs = makeDefaultTransformTypeCDFs()

func makeDefaultTransformTypeCDFs() TransformTypeCDFs {
	var c TransformTypeCDFs
	if err := c.initDefault(); err != nil {
		panic(err)
	}
	return c
}

func (c *TransformTypeCDFs) InitDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	*c = defaultTransformTypeCDFs
	return nil
}

func (c *TransformTypeCDFs) initDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	var next TransformTypeCDFs
	for tx := range ExtTXSizes {
		for mode := range int(intraModeCount) {
			cdf7 := defaultIntraExtTX7Uniform
			if tx < len(defaultIntraExtTX7Mode) {
				cdf7 = defaultIntraExtTX7Mode[tx][mode]
			}
			if err := next.Intra[1][tx][mode].Init(cdf7[:]); err != nil {
				return err
			}
			cdf5 := defaultIntraExtTX5Uniform
			if tx == 2 {
				cdf5 = defaultIntraExtTX5Size2[mode]
			}
			if err := next.Intra[2][tx][mode].Init(cdf5[:]); err != nil {
				return err
			}
		}
	}
	for tx := range ExtTXSizes {
		if err := next.Inter[1][tx].Init(defaultInterExtTX16[tx][:]); err != nil {
			return err
		}
		if err := next.Inter[2][tx].Init(defaultInterExtTX12[tx][:]); err != nil {
			return err
		}
		if err := next.Inter[3][tx].Init(defaultInterExtTX2[tx][:]); err != nil {
			return err
		}
	}
	*c = next
	return nil
}

func (c *TransformTypeCDFs) IntraCDF(index int, square TransformSize, mode IntraMode, symbols int) (*entropy.CDF, error) {
	if c == nil || index <= 0 || index >= ExtTXSetsIntra || square >= ExtTXSizes || !mode.Valid() {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.Intra[index][square][mode]
	if cdf.Symbols() != symbols {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, nil
}

func (c *TransformTypeCDFs) InterCDF(index int, square TransformSize, symbols int) (*entropy.CDF, error) {
	if c == nil || index <= 0 || index >= ExtTXSetsInter || square >= ExtTXSizes {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.Inter[index][square]
	if cdf.Symbols() != symbols {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, nil
}

func intraModeTransformType(mode IntraMode) (transform.Type, error) {
	switch mode {
	case IntraModeDC, IntraModeD45:
		return transform.TypeDCTDCT, nil
	case IntraModeVertical, IntraModeD113, IntraModeD67, IntraModeSmoothVertical:
		return transform.TypeADSTDCT, nil
	case IntraModeHorizontal, IntraModeD157, IntraModeD203, IntraModeSmoothHorizontal:
		return transform.TypeDCTADST, nil
	case IntraModeD135, IntraModeSmooth, IntraModePaeth:
		return transform.TypeADSTADST, nil
	default:
		return 0, ErrInvalidDecodeState
	}
}
