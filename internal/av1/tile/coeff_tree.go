package tile

import (
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

const (
	maxCoeffScanLen    = 32 * 32
	maxCoeffScratchLen = (32 + txPadHorizontal) * (32 + txPadHorizontal)
)

type LumaCoeffTreeRequest struct {
	TreeRequest TransformTreeRequest
	Tree        TransformTreeResult

	Class            transform.Class
	TransformType    transform.Type
	UseTransformType bool
	TransformSelect  CoeffTransformSelector
	EOBMultiContext  int
}

type LumaCoeffTreeScratch struct {
	Coeffs      [maxCoeffScanLen]int16
	Scan        [maxCoeffScanLen]int16
	InverseScan [maxCoeffScanLen]int16
	Levels      [maxCoeffScratchLen]uint8
}

type LumaCoeffBlock struct {
	Block     TransformBlock
	Transform transform.Type
	Result    TXBDecodeResult

	Coeffs []int16
	Scan   []int16
}

type LumaCoeffStats struct {
	TXBs     int
	NonZero  int
	AllZero  int
	EOBTotal int
}

type LumaCoeffVisitor func(LumaCoeffBlock) error

type ChromaCoeffTreeRequest struct {
	TreeRequest TransformTreeRequest
	Tree        TransformTreeResult

	Color parser.ColorConfig
	Plane int

	Class            transform.Class
	TransformType    transform.Type
	UseTransformType bool
	TransformSelect  CoeffTransformSelector
	EOBMultiContext  int
}

type ChromaCoeffBlock struct {
	Plane     int
	Block     TransformBlock
	Transform transform.Type

	Result TXBDecodeResult
	Coeffs []int16
	Scan   []int16
}

type ChromaCoeffVisitor func(ChromaCoeffBlock) error

type CoeffTransformRequest struct {
	Plane int
	Block TransformBlock
}

type CoeffTransformSelector interface {
	SelectCoeffTransform(CoeffTransformRequest) (transform.Type, error)
}

type CoeffTransformSelectorFunc func(CoeffTransformRequest) (transform.Type, error)

func (f CoeffTransformSelectorFunc) SelectCoeffTransform(req CoeffTransformRequest) (transform.Type, error) {
	return f(req)
}

func (s *DecodeState) DecodeLumaCoefficients(cdfs *CoeffCDFs, ctx *CoeffEntropyContext, scratch *LumaCoeffTreeScratch, req LumaCoeffTreeRequest, visit LumaCoeffVisitor) (LumaCoeffStats, error) {
	if s == nil || cdfs == nil || ctx == nil || scratch == nil || visit == nil || !req.Class.Valid() {
		return LumaCoeffStats{}, ErrInvalidDecodeState
	}
	if _, err := validateTransformTreeRequest(req.TreeRequest); err != nil {
		return LumaCoeffStats{}, err
	}
	if !req.Tree.Y.Valid() {
		return LumaCoeffStats{}, ErrInvalidDecodeState
	}
	if req.TreeRequest.SkipTransform {
		if err := ctx.ResetBlock(0, req.TreeRequest.Size, req.TreeRequest.X4, req.TreeRequest.Y4); err != nil {
			return LumaCoeffStats{}, err
		}
		return LumaCoeffStats{}, nil
	}

	var stats LumaCoeffStats
	err := req.Tree.ForEachLumaTXB(req.TreeRequest, func(block TransformBlock) error {
		typ, class, err := resolveCoeffTransform(req.TransformSelect, req.TransformType, req.UseTransformType, req.Class, 0, block)
		if err != nil {
			return err
		}
		coeffs, scan, levels, err := scratch.coeffBuffers(block.Size, class)
		if err != nil {
			return err
		}
		result, err := s.ReadCoefficientsTXBWithContext(cdfs, ctx, CoeffContextRequest{
			Plane:      0,
			PlaneBlock: req.TreeRequest.Size,
			Size:       block.Size,
			X4:         block.X4,
			Y4:         block.Y4,
			VisibleW4:  block.VisibleW4,
			VisibleH4:  block.VisibleH4,
		}, TXBDecodeRequest{
			Class:           class,
			EOBMultiContext: req.EOBMultiContext,
		}, coeffs, scan, levels)
		if err != nil {
			return err
		}

		stats.TXBs++
		stats.EOBTotal += result.EOB
		if result.AllZero {
			stats.AllZero++
		} else {
			stats.NonZero++
		}
		return visit(LumaCoeffBlock{
			Block:     block,
			Transform: typ,
			Result:    result,
			Coeffs:    coeffs,
			Scan:      scan,
		})
	})
	if err != nil {
		return stats, err
	}
	return stats, nil
}

func (s *DecodeState) DecodeChromaCoefficients(cdfs *CoeffCDFs, ctx *CoeffEntropyContext, scratch *LumaCoeffTreeScratch, req ChromaCoeffTreeRequest, visit ChromaCoeffVisitor) (LumaCoeffStats, error) {
	if s == nil || cdfs == nil || ctx == nil || scratch == nil || visit == nil ||
		req.Plane < 1 || req.Plane > 2 || !req.Class.Valid() {
		return LumaCoeffStats{}, ErrInvalidDecodeState
	}
	if _, err := validateTransformTreeRequest(req.TreeRequest); err != nil {
		return LumaCoeffStats{}, err
	}
	if !req.Tree.HasUV || !req.Tree.UV.Valid() {
		return LumaCoeffStats{}, ErrInvalidDecodeState
	}
	if !HasChromaBlock(req.TreeRequest, req.Color) {
		return LumaCoeffStats{}, nil
	}
	planeBlock, err := PlaneBlockSize(req.TreeRequest.Size, req.Color, req.Plane)
	if err != nil {
		return LumaCoeffStats{}, err
	}

	ssX := int(boolToShift(req.Color.SubsamplingX))
	ssY := int(boolToShift(req.Color.SubsamplingY))
	x4 := req.TreeRequest.X4 >> ssX
	y4 := req.TreeRequest.Y4 >> ssY
	visibleW4 := ((req.TreeRequest.X4 + int(req.TreeRequest.VisibleW4) + ssX) >> ssX) - x4
	visibleH4 := ((req.TreeRequest.Y4 + int(req.TreeRequest.VisibleH4) + ssY) >> ssY) - y4
	uvDims, ok := req.Tree.UV.Dimensions()
	if !ok || visibleW4 <= 0 || visibleH4 <= 0 ||
		x4+visibleW4 > MaxBlockModeSlots || y4+visibleH4 > MaxBlockModeSlots {
		return LumaCoeffStats{}, ErrInvalidDecodeState
	}

	if req.TreeRequest.SkipTransform {
		if err := ctx.ResetBlock(req.Plane, planeBlock, x4, y4); err != nil {
			return LumaCoeffStats{}, err
		}
		return LumaCoeffStats{}, nil
	}

	var stats LumaCoeffStats
	for y := 0; y < visibleH4; y += int(uvDims.H4) {
		for x := 0; x < visibleW4; x += int(uvDims.W4) {
			block := TransformBlock{
				X4:        x4 + x,
				Y4:        y4 + y,
				Size:      req.Tree.UV,
				VisibleW4: uint8(minInt(int(uvDims.W4), visibleW4-x)),
				VisibleH4: uint8(minInt(int(uvDims.H4), visibleH4-y)),
			}
			typ, class, err := resolveCoeffTransform(req.TransformSelect, req.TransformType, req.UseTransformType, req.Class, req.Plane, block)
			if err != nil {
				return stats, err
			}
			coeffs, scan, levels, err := scratch.coeffBuffers(block.Size, class)
			if err != nil {
				return stats, err
			}
			result, err := s.ReadCoefficientsTXBWithContext(cdfs, ctx, CoeffContextRequest{
				Plane:      req.Plane,
				PlaneBlock: planeBlock,
				Size:       block.Size,
				X4:         block.X4,
				Y4:         block.Y4,
				VisibleW4:  block.VisibleW4,
				VisibleH4:  block.VisibleH4,
			}, TXBDecodeRequest{
				Class:           class,
				EOBMultiContext: req.EOBMultiContext,
			}, coeffs, scan, levels)
			if err != nil {
				return stats, err
			}
			stats.TXBs++
			stats.EOBTotal += result.EOB
			if result.AllZero {
				stats.AllZero++
			} else {
				stats.NonZero++
			}
			if err := visit(ChromaCoeffBlock{
				Plane:     req.Plane,
				Block:     block,
				Transform: typ,
				Result:    result,
				Coeffs:    coeffs,
				Scan:      scan,
			}); err != nil {
				return stats, err
			}
		}
	}
	return stats, nil
}

func resolveCoeffTransform(selector CoeffTransformSelector, typ transform.Type, useType bool, class transform.Class, plane int, block TransformBlock) (transform.Type, transform.Class, error) {
	if selector != nil {
		selected, err := selector.SelectCoeffTransform(CoeffTransformRequest{Plane: plane, Block: block})
		if err != nil {
			return 0, 0, err
		}
		selectedClass, err := selected.Class()
		if err != nil {
			return 0, 0, ErrInvalidDecodeState
		}
		return selected, selectedClass, nil
	}
	if useType {
		selectedClass, err := typ.Class()
		if err != nil {
			return 0, 0, ErrInvalidDecodeState
		}
		return typ, selectedClass, nil
	}
	if !class.Valid() {
		return 0, 0, ErrInvalidDecodeState
	}
	return transform.TypeDCTDCT, class, nil
}

func (s *LumaCoeffTreeScratch) coeffBuffers(size TransformSize, class transform.Class) ([]int16, []int16, []uint8, error) {
	if s == nil {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	txSize, err := size.TransformSize()
	if err != nil {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	scanSize, err := transform.ScanSize(txSize)
	if err != nil {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	scanLen := scanSize.Width * scanSize.Height
	if scanLen > len(s.Scan) || scanLen > len(s.InverseScan) || scanLen > len(s.Coeffs) {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	scan := s.Scan[:scanLen]
	inverse := s.InverseScan[:scanLen]
	if err := transform.FillDefaultScan(scan, inverse, txSize, class); err != nil {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	levelsLen, err := CoeffLevelsScratchLen(size)
	if err != nil {
		return nil, nil, nil, err
	}
	if levelsLen > len(s.Levels) {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	return s.Coeffs[:scanLen], scan, s.Levels[:levelsLen], nil
}
