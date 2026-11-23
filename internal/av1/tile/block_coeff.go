package tile

import "github.com/thesyncim/goav1/internal/av1/transform"

type BlockCoeffCDFs struct {
	Transform *TransformCDFs
	Coeff     *CoeffCDFs
}

type BlockCoeffScratch struct {
	Coeff LumaCoeffTreeScratch
	// txtpMap is dav1d's t->scratch.txtp_map.
	txtpMap [MaxBlockModeSlots][MaxBlockModeSlots]transform.Type
}

// BlockCoeffBlock is one transform block as the reconstruction consumes it:
// its plane, position and type, the decode result, and, for a TXB read as
// quantized coefficients, those coefficients in scan order.
type BlockCoeffBlock struct {
	Plane     uint8
	Block     TransformBlock
	Transform transform.Type

	Result TXBDecodeResult
	Coeffs []int16
	Scan   []int16
}

// BlockCoeffResult is what the block loop keeps of a block's coefficient
// decode: its transform sizes.
type BlockCoeffResult struct {
	Tree TransformTreeResult
}

const (
	maxCoeffScanLen    = 32 * 32
	maxCoeffScratchLen = (32 + txPadHorizontal) * (32 + txPadHorizontal)
)

type LumaCoeffTreeScratch struct {
	Coeffs      [maxCoeffScanLen]int16
	Scan        [maxCoeffScanLen]int16
	InverseScan [maxCoeffScanLen]int16
	LevelDirty  [maxCoeffScanLen]int16
	Levels      [maxCoeffScratchLen]uint8
	// CoefLevels is dav1d's t->scratch.levels for decodeCoefTokens. It is
	// cleared per TXB by the reader and is separate from Levels, whose dirty
	// positions the quantized readers track.
	CoefLevels [coefLevelsLen]uint8

	coeffDirtyLen uint16
	levelDirtyLen uint16
}

func (s *LumaCoeffTreeScratch) clearCoeffDirty() {
	switch n := s.coeffDirtyLen; n {
	case 0:
		return
	case 1:
		s.Coeffs[coeffDirtyPackedPos(s.InverseScan[0])] = 0
		s.coeffDirtyLen = 0
		return
	case 2:
		s.Coeffs[coeffDirtyPackedPos(s.InverseScan[0])] = 0
		s.Coeffs[coeffDirtyPackedPos(s.InverseScan[1])] = 0
		s.coeffDirtyLen = 0
		return
	case 3:
		s.Coeffs[coeffDirtyPackedPos(s.InverseScan[0])] = 0
		s.Coeffs[coeffDirtyPackedPos(s.InverseScan[1])] = 0
		s.Coeffs[coeffDirtyPackedPos(s.InverseScan[2])] = 0
		s.coeffDirtyLen = 0
		return
	case 4:
		s.Coeffs[coeffDirtyPackedPos(s.InverseScan[0])] = 0
		s.Coeffs[coeffDirtyPackedPos(s.InverseScan[1])] = 0
		s.Coeffs[coeffDirtyPackedPos(s.InverseScan[2])] = 0
		s.Coeffs[coeffDirtyPackedPos(s.InverseScan[3])] = 0
		s.coeffDirtyLen = 0
		return
	}
	dirty := s.InverseScan[:s.coeffDirtyLen:s.coeffDirtyLen]
	for _, pos := range dirty {
		s.Coeffs[coeffDirtyPackedPos(pos)] = 0
	}
	s.coeffDirtyLen = 0
}

func (s *LumaCoeffTreeScratch) coeffDirtyPositions() []int16 {
	if s == nil || s.coeffDirtyLen == 0 {
		return nil
	}
	return s.InverseScan[:s.coeffDirtyLen:s.coeffDirtyLen]
}

func (s *BlockCoeffScratch) CoeffDirtyPositions() []int16 {
	if s == nil {
		return nil
	}
	return s.Coeff.coeffDirtyPositions()
}

func eobMultiContextForClass(class transform.Class) uint8 {
	if class == transform.Class2D {
		return 0
	}
	return 1
}

func (s *LumaCoeffTreeScratch) coeffBuffers(size TransformSize, class transform.Class) ([]int16, []int16, []uint8, error) {
	if s == nil {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	geo, ok := coeffGeoPtr(size)
	if !ok || !class.Valid() {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	return s.coeffBuffersWithGeo(size, class, geo)
}

func (s *LumaCoeffTreeScratch) coeffBuffersWithGeo(size TransformSize, class transform.Class, geo *coeffGeometry) ([]int16, []int16, []uint8, error) {
	if s == nil || geo == nil || !geo.valid || !class.Valid() {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	scanLen := int(geo.maxEOB)
	if scanLen > len(s.Coeffs) {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	scan := coeffScanTable[size][class]
	if len(scan) < scanLen {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	levelsLen := int(geo.scratchLen)
	if levelsLen > len(s.Levels) {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	return s.Coeffs[:scanLen], scan, s.Levels[:levelsLen], nil
}
