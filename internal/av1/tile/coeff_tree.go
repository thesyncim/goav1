package tile

import "github.com/thesyncim/goav1/internal/av1/transform"

const (
	maxCoeffScanLen    = 32 * 32
	maxCoeffScratchLen = (32 + txPadHorizontal) * (32 + txPadHorizontal)
)

type LumaCoeffTreeRequest struct {
	TreeRequest TransformTreeRequest
	Tree        TransformTreeResult

	Class           transform.Class
	EOBMultiContext int
}

type LumaCoeffTreeScratch struct {
	Coeffs      [maxCoeffScanLen]int16
	Scan        [maxCoeffScanLen]int16
	InverseScan [maxCoeffScanLen]int16
	Levels      [maxCoeffScratchLen]uint8
}

type LumaCoeffBlock struct {
	Block  TransformBlock
	Result TXBDecodeResult

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
		coeffs, scan, levels, err := scratch.coeffBuffers(block.Size, req.Class)
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
			Class:           req.Class,
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
			Block:  block,
			Result: result,
			Coeffs: coeffs,
			Scan:   scan,
		})
	})
	if err != nil {
		return stats, err
	}
	return stats, nil
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
