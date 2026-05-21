package tile

import "github.com/thesyncim/goav1/internal/av1/transform"

type BlockCoeffCDFs struct {
	Transform *TransformCDFs
	Coeff     *CoeffCDFs
}

type BlockCoeffScratch struct {
	Coeff LumaCoeffTreeScratch
}

type BlockCoeffRequest struct {
	Transform TransformTreeRequest

	LumaClass       transform.Class
	ChromaClass     [2]transform.Class
	EOBMultiContext [3]int
}

type BlockCoeffBlock struct {
	Plane int
	Block TransformBlock

	Result TXBDecodeResult
	Coeffs []int16
	Scan   []int16
}

type BlockCoeffResult struct {
	Tree TransformTreeResult

	Luma   LumaCoeffStats
	Chroma [2]LumaCoeffStats
}

type BlockCoeffVisitor func(BlockCoeffBlock) error

func (r BlockCoeffResult) TotalStats() LumaCoeffStats {
	out := r.Luma
	for _, stats := range r.Chroma {
		out.TXBs += stats.TXBs
		out.NonZero += stats.NonZero
		out.AllZero += stats.AllZero
		out.EOBTotal += stats.EOBTotal
	}
	return out
}

func (s *DecodeState) DecodeBlockCoefficients(cdfs BlockCoeffCDFs, modeCtx *BlockModeContext, coeffCtx *CoeffEntropyContext, scratch *BlockCoeffScratch, req BlockCoeffRequest, visit BlockCoeffVisitor) (BlockCoeffResult, error) {
	if s == nil || cdfs.Transform == nil || cdfs.Coeff == nil ||
		modeCtx == nil || coeffCtx == nil || scratch == nil || visit == nil ||
		!req.LumaClass.Valid() {
		return BlockCoeffResult{}, ErrInvalidDecodeState
	}
	if !req.Transform.Color.MonoChrome {
		if !req.ChromaClass[0].Valid() || !req.ChromaClass[1].Valid() {
			return BlockCoeffResult{}, ErrInvalidDecodeState
		}
	}

	tree, err := s.DecodeTransformTree(cdfs.Transform, modeCtx, req.Transform)
	if err != nil {
		return BlockCoeffResult{}, err
	}
	result := BlockCoeffResult{Tree: tree}

	result.Luma, err = s.DecodeLumaCoefficients(cdfs.Coeff, coeffCtx, &scratch.Coeff, LumaCoeffTreeRequest{
		TreeRequest:     req.Transform,
		Tree:            tree,
		Class:           req.LumaClass,
		EOBMultiContext: req.EOBMultiContext[0],
	}, func(block LumaCoeffBlock) error {
		return visit(BlockCoeffBlock{
			Plane:  0,
			Block:  block.Block,
			Result: block.Result,
			Coeffs: block.Coeffs,
			Scan:   block.Scan,
		})
	})
	if err != nil {
		return result, err
	}

	if req.Transform.Color.MonoChrome || !tree.HasUV {
		return result, nil
	}
	for plane := 1; plane <= 2; plane++ {
		stats, err := s.DecodeChromaCoefficients(cdfs.Coeff, coeffCtx, &scratch.Coeff, ChromaCoeffTreeRequest{
			TreeRequest:     req.Transform,
			Tree:            tree,
			Color:           req.Transform.Color,
			Plane:           plane,
			Class:           req.ChromaClass[plane-1],
			EOBMultiContext: req.EOBMultiContext[plane],
		}, func(block ChromaCoeffBlock) error {
			return visit(BlockCoeffBlock{
				Plane:  block.Plane,
				Block:  block.Block,
				Result: block.Result,
				Coeffs: block.Coeffs,
				Scan:   block.Scan,
			})
		})
		if err != nil {
			return result, err
		}
		result.Chroma[plane-1] = stats
	}
	return result, nil
}
