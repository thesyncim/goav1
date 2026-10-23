package tile

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/transform"
)

type BlockCoeffCDFs struct {
	Transform *TransformCDFs
	Coeff     *CoeffCDFs
}

type BlockCoeffScratch struct {
	Coeff LumaCoeffTreeScratch
	block BlockCoeffBlock
}

// CoeffDequantProvider supplies per-plane dequant factors for the private
// synchronous reconstruction path. It is called only after a nonzero tracked
// 2D TXB has been decoded.
type CoeffDequantProvider interface {
	PrepareCoeffDequantPlane(plane uint8) bool
}

// CoeffDequantTarget is job-owned scratch shared by the tracked coefficient
// reader and its synchronous reconstruction consumer. PreparedPlanes and
// Factors are cached by the provider for the current quantization key; Ready,
// ActiveRows, and TxScale describe only the most recently decoded TXB.
type CoeffDequantTarget struct {
	Dst []int32
	// Arena, when set, switches the target to dav1d's frame-thread cf layout
	// (ts->frame_thread[p].cf): each dequantized TXB is appended to *Arena and
	// Dst is left pointing at its slot, so a deferred reconstruction pass can
	// consume it later. Only the dav1d block walk (readCoefBlocks) fills it;
	// the quantized readers ignore an arena target.
	Arena *[]int32
	// Coef, when set, is the zero-maintained t->cf buffer of the dav1d block
	// walk; Dst is pointed at it after each TXB.
	Coef           *CoefBuffer
	Factors        [3][2]int32 // [Y/U/V][DC/AC]
	ActiveRows     int
	// ActiveCols bounds the nonzero coefficient columns (0 = unknown).
	ActiveCols int
	Provider       CoeffDequantProvider
	BitDepth       uint8
	Ready          bool
	TxScale        uint8
	PreparedPlanes uint8

	decoded bool // set by the tracked reader; published as Ready after recording succeeds
}

type BlockCoeffRequest struct {
	Transform TransformTreeRequest

	LumaType        transform.Type
	ChromaType      [2]transform.Type
	TransformSelect CoeffTransformSelector
	EOBMultiContext [3]uint8

	SkipAllZeroCoeffClear bool

	// Dequant is used only by the internal synchronous pointer-visitor path.
	// Public/value visitors retain the ordinary quantized-coefficient contract.
	Dequant *CoeffDequantTarget
}

type BlockCoeffBlock struct {
	Plane     uint8
	Block     TransformBlock
	Transform transform.Type

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
type blockCoeffPointerVisitor func(*BlockCoeffBlock) error

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
	if visit == nil {
		return BlockCoeffResult{}, ErrInvalidDecodeState
	}
	return s.decodeBlockCoefficients(cdfs, modeCtx, coeffCtx, scratch, req, func(block *BlockCoeffBlock) error {
		return visit(*block)
	})
}

func (s *DecodeState) decodeBlockCoefficientsPtr(cdfs BlockCoeffCDFs, modeCtx *BlockModeContext, coeffCtx *CoeffEntropyContext, scratch *BlockCoeffScratch, req BlockCoeffRequest, visit blockCoeffPointerVisitor) (BlockCoeffResult, error) {
	return s.decodeBlockCoefficients(cdfs, modeCtx, coeffCtx, scratch, req, visit)
}

func (s *DecodeState) decodeBlockCoefficients(cdfs BlockCoeffCDFs, modeCtx *BlockModeContext, coeffCtx *CoeffEntropyContext, scratch *BlockCoeffScratch, req BlockCoeffRequest, visit blockCoeffPointerVisitor) (BlockCoeffResult, error) {
	if s == nil || cdfs.Transform == nil || cdfs.Coeff == nil ||
		modeCtx == nil || coeffCtx == nil || scratch == nil || visit == nil {
		return BlockCoeffResult{}, ErrInvalidDecodeState
	}
	lumaClass, err := req.LumaType.Class()
	if err != nil {
		return BlockCoeffResult{}, ErrInvalidDecodeState
	}
	var chromaClass [2]transform.Class
	if !req.Transform.Color.MonoChrome {
		for i, typ := range req.ChromaType {
			class, err := typ.Class()
			if err != nil {
				return BlockCoeffResult{}, ErrInvalidDecodeState
			}
			chromaClass[i] = class
		}
	}

	tree, err := s.DecodeTransformTree(cdfs.Transform, modeCtx, req.Transform)
	if err != nil {
		return BlockCoeffResult{}, fmt.Errorf("decode transform tree req=%+v: %w", req.Transform, err)
	}
	result := BlockCoeffResult{Tree: tree}

	lumaReq := LumaCoeffTreeRequest{
		TreeRequest:           req.Transform,
		Tree:                  tree,
		Class:                 lumaClass,
		TransformType:         req.LumaType,
		UseTransformType:      true,
		TransformSelect:       req.TransformSelect,
		EOBMultiContext:       req.EOBMultiContext[0],
		SkipAllZeroCoeffClear: req.SkipAllZeroCoeffClear,
	}
	lumaVisit := func(block LumaCoeffBlock) error {
		scratch.block = BlockCoeffBlock{
			Plane:     0,
			Block:     block.Block,
			Transform: block.Transform,
			Result:    block.Result,
			Coeffs:    block.Coeffs,
			Scan:      block.Scan,
		}
		return visit(&scratch.block)
	}

	hasChroma := !req.Transform.Color.MonoChrome && tree.HasUV
	var chromaReq [2]ChromaCoeffTreeRequest
	var chromaPrep [2]chromaCoeffPlanePrep
	chromaVisit := func(block ChromaCoeffBlock) error {
		scratch.block = BlockCoeffBlock{
			Plane:     uint8(block.Plane),
			Block:     block.Block,
			Transform: block.Transform,
			Result:    block.Result,
			Coeffs:    block.Coeffs,
			Scan:      block.Scan,
		}
		return visit(&scratch.block)
	}
	if hasChroma {
		for plane := 1; plane <= 2; plane++ {
			chromaReq[plane-1] = ChromaCoeffTreeRequest{
				TreeRequest:           req.Transform,
				Tree:                  tree,
				Color:                 req.Transform.Color,
				Plane:                 uint8(plane),
				Class:                 chromaClass[plane-1],
				TransformType:         req.ChromaType[plane-1],
				UseTransformType:      true,
				TransformSelect:       req.TransformSelect,
				EOBMultiContext:       req.EOBMultiContext[plane],
				SkipAllZeroCoeffClear: req.SkipAllZeroCoeffClear,
			}
			// prepareChromaCoefficients validates the plane and performs the
			// once-per-block SkipTransform context reset. The non-skip per-unit
			// decode below uses the returned geometry.
			prep, err := s.prepareChromaCoefficients(coeffCtx, chromaReq[plane-1])
			if err != nil {
				return result, fmt.Errorf("prepare chroma coeffs plane=%d req=%+v tree=%+v: %w", plane, req.Transform, tree, err)
			}
			chromaPrep[plane-1] = prep
		}
	}

	// SkipTransform blocks only reset entropy contexts (handled above for chroma
	// and below for luma); no coefficients are read, so the unit interleaving is
	// irrelevant. Take the simple single-call path.
	if req.Transform.SkipTransform {
		result.Luma, err = s.DecodeLumaCoefficients(cdfs.Coeff, coeffCtx, &scratch.Coeff, lumaReq, lumaVisit)
		if err != nil {
			return result, fmt.Errorf("decode luma coeffs req=%+v tree=%+v: %w", req.Transform, tree, err)
		}
		if !req.Transform.Inter {
			var walk skipIntraTXBWalk
			if ok, err := walk.init(&req.Transform, &tree); err != nil {
				return result, err
			} else if ok {
				for walk.next(&scratch.block) {
					if err := visit(&scratch.block); err != nil {
						return result, err
					}
				}
			}
		}
		return result, nil
	}

	// Split the prediction block into BLOCK_64X64 max-units and interleave
	// luma+U+V coefficient decode per unit, mirroring libaom's residual loop in
	// av1/decoder/decodeframe.c (decode_token_recon_block) and spec 5.11.34
	// residual(). For blocks <= 64px there is a single unit, so the resulting
	// luma-then-chroma call sequence is identical to the unwindowed decode.
	//
	// maxUnit4 is BLOCK_64X64 measured in 4x4 luma units; the loop matches
	// libaom's `for (row; row += mu_blocks_high) for (col; col += mu_blocks_wide)`.
	const maxUnit4 = 16
	visW := int(req.Transform.VisibleW4)
	visH := int(req.Transform.VisibleH4)
	muW := minInt(maxUnit4, visW)
	muH := minInt(maxUnit4, visH)
	reqX4 := int(req.Transform.X4)
	reqY4 := int(req.Transform.Y4)
	for row := 0; row < visH; row += muH {
		for col := 0; col < visW; col += muW {
			window := coeffUnitWindow{
				X4Start: uint8(reqX4 + col),
				Y4Start: uint8(reqY4 + row),
				X4End:   uint8(reqX4 + minInt(col+muW, visW)),
				Y4End:   uint8(reqY4 + minInt(row+muH, visH)),
			}
			lumaStats, err := s.decodeLumaCoefficientsInWindow(cdfs.Coeff, coeffCtx, &scratch.Coeff, lumaReq, window, lumaVisit)
			if err != nil {
				return result, fmt.Errorf("decode luma coeffs window=%+v req=%+v tree=%+v: %w", window, req.Transform, tree, err)
			}
			result.Luma.TXBs += lumaStats.TXBs
			result.Luma.NonZero += lumaStats.NonZero
			result.Luma.AllZero += lumaStats.AllZero
			result.Luma.EOBTotal += lumaStats.EOBTotal

			if !hasChroma {
				continue
			}
			for plane := 1; plane <= 2; plane++ {
				prep := chromaPrep[plane-1]
				if !prep.HasChroma {
					continue
				}
				stats, err := s.decodeChromaCoefficientsInWindow(cdfs.Coeff, coeffCtx, &scratch.Coeff, chromaReq[plane-1], window, prep, chromaVisit)
				if err != nil {
					return result, fmt.Errorf("decode chroma coeffs plane=%d window=%+v req=%+v tree=%+v: %w", plane, window, req.Transform, tree, err)
				}
				result.Chroma[plane-1].TXBs += stats.TXBs
				result.Chroma[plane-1].NonZero += stats.NonZero
				result.Chroma[plane-1].AllZero += stats.AllZero
				result.Chroma[plane-1].EOBTotal += stats.EOBTotal
			}
		}
	}
	return result, nil
}

func decodeBlockCoefficientsWithCoeffControllerPtr[T BlockLoopCoeffController](s *DecodeState, cdfs BlockCoeffCDFs, modeCtx *BlockModeContext, coeffCtx *CoeffEntropyContext, scratch *BlockCoeffScratch, req BlockCoeffRequest, loopVisit *BlockLoopVisit, coeffController T) (BlockCoeffResult, error) {
	resetCoeffDequantTXB(req.Dequant)
	if s == nil || cdfs.Transform == nil || cdfs.Coeff == nil ||
		modeCtx == nil || coeffCtx == nil || scratch == nil || loopVisit == nil {
		return BlockCoeffResult{}, ErrInvalidDecodeState
	}
	lumaClass, err := req.LumaType.Class()
	if err != nil {
		return BlockCoeffResult{}, ErrInvalidDecodeState
	}
	var chromaClass [2]transform.Class
	if !req.Transform.Color.MonoChrome {
		for i, typ := range req.ChromaType {
			class, err := typ.Class()
			if err != nil {
				return BlockCoeffResult{}, ErrInvalidDecodeState
			}
			chromaClass[i] = class
		}
	}

	tree, err := s.DecodeTransformTree(cdfs.Transform, modeCtx, req.Transform)
	if err != nil {
		return BlockCoeffResult{}, fmt.Errorf("decode transform tree req=%+v: %w", req.Transform, err)
	}
	result := BlockCoeffResult{Tree: tree}
	hasChroma := !req.Transform.Color.MonoChrome && tree.HasUV

	// dav1d's decode_b hands a decoded block straight to recon_b_intra /
	// recon_b_inter (src/recon_tmpl.c), which derive the chroma geometry once
	// from b and t and walk the TXBs; no per-plane request is built. Skip
	// blocks only reset the coefficient contexts (dav1d's
	// dav1d_memset_likely_pow2 of a/l ccoef in decode_b).
	if req.Transform.SkipTransform || readCoefBlocksSupported(&req) {
		var chromaPrep [2]chromaCoeffPlanePrep
		if hasChroma {
			prep, err := prepareChromaCoefBlock(coeffCtx, &req.Transform, tree)
			if err != nil {
				return result, fmt.Errorf("prepare chroma coeffs req=%+v tree=%+v: %w", req.Transform, tree, err)
			}
			chromaPrep[0], chromaPrep[1] = prep, prep
		}
		if req.Transform.SkipTransform {
			if err := coeffCtx.ResetBlock(0, req.Transform.Size, int(req.Transform.X4), int(req.Transform.Y4)); err != nil {
				return result, err
			}
			if !req.Transform.Inter {
				// dav1d's recon_b_intra still predicts a skip intra block
				// transform block by transform block.
				var walk skipIntraTXBWalk
				if ok, err := walk.init(&req.Transform, &tree); err != nil {
					return result, err
				} else if ok {
					for walk.next(&scratch.block) {
						if err := coeffController.VisitBlockCoeffPtr(loopVisit, &scratch.block); err != nil {
							return result, err
						}
					}
				}
			}
			return result, nil
		}
		if err := readCoefBlocks(s, cdfs, coeffCtx, scratch, &req, tree, &chromaPrep, hasChroma, loopVisit, coeffController, &result); err != nil {
			return result, fmt.Errorf("read coef blocks req=%+v tree=%+v: %w", req.Transform, tree, err)
		}
		return result, nil
	}

	lumaReq := LumaCoeffTreeRequest{
		TreeRequest:           req.Transform,
		Tree:                  tree,
		Class:                 lumaClass,
		TransformType:         req.LumaType,
		UseTransformType:      true,
		TransformSelect:       req.TransformSelect,
		EOBMultiContext:       req.EOBMultiContext[0],
		SkipAllZeroCoeffClear: req.SkipAllZeroCoeffClear,
		dequant:               req.Dequant,
	}

	var chromaReq [2]ChromaCoeffTreeRequest
	var chromaPrep [2]chromaCoeffPlanePrep
	if hasChroma {
		for plane := 1; plane <= 2; plane++ {
			chromaReq[plane-1] = ChromaCoeffTreeRequest{
				TreeRequest:           req.Transform,
				Tree:                  tree,
				Color:                 req.Transform.Color,
				Plane:                 uint8(plane),
				Class:                 chromaClass[plane-1],
				TransformType:         req.ChromaType[plane-1],
				UseTransformType:      true,
				TransformSelect:       req.TransformSelect,
				EOBMultiContext:       req.EOBMultiContext[plane],
				SkipAllZeroCoeffClear: req.SkipAllZeroCoeffClear,
				dequant:               req.Dequant,
			}
			prep, err := s.prepareChromaCoefficients(coeffCtx, chromaReq[plane-1])
			if err != nil {
				return result, fmt.Errorf("prepare chroma coeffs plane=%d req=%+v tree=%+v: %w", plane, req.Transform, tree, err)
			}
			chromaPrep[plane-1] = prep
		}
	}

	const maxUnit4 = 16
	visW := int(req.Transform.VisibleW4)
	visH := int(req.Transform.VisibleH4)
	muW := minInt(maxUnit4, visW)
	muH := minInt(maxUnit4, visH)
	reqX4 := int(req.Transform.X4)
	reqY4 := int(req.Transform.Y4)
	for row := 0; row < visH; row += muH {
		for col := 0; col < visW; col += muW {
			window := coeffUnitWindow{
				X4Start: uint8(reqX4 + col),
				Y4Start: uint8(reqY4 + row),
				X4End:   uint8(reqX4 + minInt(col+muW, visW)),
				Y4End:   uint8(reqY4 + minInt(row+muH, visH)),
			}
			lumaStats, err := decodeLumaCoefficientsInWindowWithCoeffControllerPtr(s, cdfs.Coeff, coeffCtx, &scratch.Coeff, lumaReq, window, scratch, loopVisit, coeffController)
			if err != nil {
				return result, fmt.Errorf("decode luma coeffs window=%+v req=%+v tree=%+v: %w", window, req.Transform, tree, err)
			}
			result.Luma.TXBs += lumaStats.TXBs
			result.Luma.NonZero += lumaStats.NonZero
			result.Luma.AllZero += lumaStats.AllZero
			result.Luma.EOBTotal += lumaStats.EOBTotal

			if !hasChroma {
				continue
			}
			for plane := 1; plane <= 2; plane++ {
				prep := chromaPrep[plane-1]
				if !prep.HasChroma {
					continue
				}
				stats, err := decodeChromaCoefficientsInWindowWithCoeffControllerPtr(s, cdfs.Coeff, coeffCtx, &scratch.Coeff, chromaReq[plane-1], window, prep, scratch, loopVisit, coeffController)
				if err != nil {
					return result, fmt.Errorf("decode chroma coeffs plane=%d window=%+v req=%+v tree=%+v: %w", plane, window, req.Transform, tree, err)
				}
				result.Chroma[plane-1].TXBs += stats.TXBs
				result.Chroma[plane-1].NonZero += stats.NonZero
				result.Chroma[plane-1].AllZero += stats.AllZero
				result.Chroma[plane-1].EOBTotal += stats.EOBTotal
			}
		}
	}
	return result, nil
}
