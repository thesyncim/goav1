package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

const (
	ExtTXSetTypes  = 6
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
	Inter [ExtTXSetsInter][ExtTXSizes]entropy.CDF
}

type InterTransformTypeRequest struct {
	Size          TransformSize
	ReducedTXSet  bool
	SkipTransform bool
	Lossless      bool
}

// InterCoeffTransformSelector reads inter tx_type syntax for each TXB. The
// caller owns and reuses this struct so composed tile decode remains allocation
// free.
type InterCoeffTransformSelector struct {
	State *DecodeState
	CDFs  *TransformTypeCDFs

	ReducedTXSet  bool
	SkipTransform bool
	Lossless      bool
	Color         parser.ColorConfig
	Map           InterTransformTypeMap
}

type InterTransformTypeMap struct {
	Type  [MaxBlockModeSlots][MaxBlockModeSlots]transform.Type
	Valid [MaxBlockModeSlots][MaxBlockModeSlots]uint8
}

func (s *InterCoeffTransformSelector) Reset(state *DecodeState, cdfs *TransformTypeCDFs, reducedTXSet bool, skipTransform bool, lossless bool) {
	s.ResetForColor(state, cdfs, reducedTXSet, skipTransform, lossless, parser.ColorConfig{})
}

func (s *InterCoeffTransformSelector) ResetForColor(state *DecodeState, cdfs *TransformTypeCDFs, reducedTXSet bool, skipTransform bool, lossless bool, color parser.ColorConfig) {
	*s = InterCoeffTransformSelector{
		State:         state,
		CDFs:          cdfs,
		ReducedTXSet:  reducedTXSet,
		SkipTransform: skipTransform,
		Lossless:      lossless,
		Color:         color,
	}
}

func (s *InterCoeffTransformSelector) SelectCoeffTransform(req CoeffTransformRequest) (transform.Type, error) {
	if s == nil || s.State == nil {
		return 0, ErrInvalidDecodeState
	}
	if req.Plane != 0 {
		return s.Map.ChromaType(req.Block, s.Color)
	}
	typ, err := s.State.ReadInterTransformType(s.CDFs, InterTransformTypeRequest{
		Size:          req.Block.Size,
		ReducedTXSet:  s.ReducedTXSet,
		SkipTransform: s.SkipTransform,
		Lossless:      s.Lossless,
	})
	if err != nil {
		return 0, err
	}
	return typ, nil
}

func (s *InterCoeffTransformSelector) RecordCoeffTransform(req CoeffTransformRequest, typ transform.Type) error {
	if s == nil {
		return ErrInvalidDecodeState
	}
	if req.Plane != 0 {
		return nil
	}
	return s.Map.Set(req.Block, typ)
}

func (m *InterTransformTypeMap) Reset() {
	if m == nil {
		return
	}
	*m = InterTransformTypeMap{}
}

func (m *InterTransformTypeMap) Set(block TransformBlock, typ transform.Type) error {
	if m == nil || !typ.Valid() {
		return ErrInvalidDecodeState
	}
	dims, ok := block.Size.Dimensions()
	if !ok || block.X4 < 0 || block.Y4 < 0 ||
		block.X4+int(dims.W4) > MaxBlockModeSlots ||
		block.Y4+int(dims.H4) > MaxBlockModeSlots {
		return ErrInvalidDecodeState
	}
	for y := 0; y < int(dims.H4); y++ {
		for x := 0; x < int(dims.W4); x++ {
			m.Type[block.Y4+y][block.X4+x] = typ
			m.Valid[block.Y4+y][block.X4+x] = 1
		}
	}
	return nil
}

func (m *InterTransformTypeMap) At(x4 int, y4 int) (transform.Type, error) {
	if m == nil || x4 < 0 || y4 < 0 || x4 >= MaxBlockModeSlots || y4 >= MaxBlockModeSlots ||
		m.Valid[y4][x4] == 0 {
		return 0, ErrInvalidDecodeState
	}
	return m.Type[y4][x4], nil
}

func (m *InterTransformTypeMap) ChromaType(block TransformBlock, color parser.ColorConfig) (transform.Type, error) {
	x4 := block.X4
	y4 := block.Y4
	if color.SubsamplingX {
		x4 <<= 1
	}
	if color.SubsamplingY {
		y4 <<= 1
	}
	return m.At(x4, y4)
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

func ExtTXTypeFromSymbol(set ExtTXSetType, symbol int) (transform.Type, error) {
	count, err := ExtTXTypeCount(set)
	if err != nil || symbol < 0 || symbol >= count {
		return 0, ErrInvalidDecodeState
	}
	return extTXTypeInv[set][symbol], nil
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

func (c *TransformTypeCDFs) InitDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	var next TransformTypeCDFs
	for tx := 0; tx < ExtTXSizes; tx++ {
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

func (c *TransformTypeCDFs) InterCDF(index int, square TransformSize, symbols int) (*entropy.CDF, error) {
	if c == nil || index <= 0 || index >= ExtTXSetsInter || square < 0 || square >= ExtTXSizes {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.Inter[index][square]
	if cdf.Symbols() != symbols {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

func (s *DecodeState) ReadInterTransformType(cdfs *TransformTypeCDFs, req InterTransformTypeRequest) (transform.Type, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	if !req.Size.Valid() {
		return 0, ErrInvalidDecodeState
	}
	if req.SkipTransform || req.Lossless {
		return transform.TypeDCTDCT, nil
	}
	set, err := ExtTXSetTypeFor(req.Size, true, req.ReducedTXSet)
	if err != nil {
		return 0, err
	}
	symbols, err := ExtTXTypeCount(set)
	if err != nil {
		return 0, err
	}
	if symbols <= 1 {
		return transform.TypeDCTDCT, nil
	}
	index, err := ExtTXSetIndex(req.Size, true, req.ReducedTXSet)
	if err != nil {
		return 0, err
	}
	square, err := TransformSizeSquare(req.Size)
	if err != nil {
		return 0, err
	}
	cdf, err := cdfs.InterCDF(index, square, symbols)
	if err != nil {
		return 0, err
	}
	symbol, err := s.ReadSymbol(cdf)
	if err != nil {
		return 0, err
	}
	return ExtTXTypeFromSymbol(set, symbol)
}
