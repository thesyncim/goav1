package tile

import (
	"fmt"

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

	SkipAllZeroCoeffClear bool
}

type LumaCoeffTreeScratch struct {
	Coeffs      [maxCoeffScanLen]int16
	Scan        [maxCoeffScanLen]int16
	InverseScan [maxCoeffScanLen]int16
	Levels      [maxCoeffScratchLen]uint8

	coeffDirtyLen int
}

func (s *LumaCoeffTreeScratch) clearCoeffDirty() {
	if s == nil {
		return
	}
	for i := 0; i < s.coeffDirtyLen; i++ {
		pos := int(s.InverseScan[i])
		if uint(pos) < uint(len(s.Coeffs)) {
			s.Coeffs[pos] = 0
		}
	}
	s.coeffDirtyLen = 0
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

	SkipAllZeroCoeffClear bool
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
	return s.decodeLumaCoefficientsInWindow(cdfs, ctx, scratch, req, fullCoeffUnitWindow(req.TreeRequest), visit)
}

// decodeLumaCoefficientsInWindow decodes the luma transform blocks of one 64x64
// max-unit (the window). The per-TXB decode is unchanged; only the set of
// transform blocks visited is restricted to the window. Callers must validate
// req and handle the SkipTransform context reset (done once per block).
func (s *DecodeState) decodeLumaCoefficientsInWindow(cdfs *CoeffCDFs, ctx *CoeffEntropyContext, scratch *LumaCoeffTreeScratch, req LumaCoeffTreeRequest, window coeffUnitWindow, visit LumaCoeffVisitor) (LumaCoeffStats, error) {
	var stats LumaCoeffStats
	if visit == nil {
		return stats, ErrInvalidDecodeState
	}
	if _, err := validateTransformTreeRequest(req.TreeRequest); err != nil {
		return stats, err
	}
	if !req.Tree.Y.Valid() {
		return stats, ErrInvalidDecodeState
	}
	if req.TreeRequest.SkipTransform {
		return stats, nil
	}

	if !req.Tree.Variable {
		dims, ok := req.Tree.Y.Dimensions()
		if !ok {
			return stats, ErrInvalidDecodeState
		}
		yStart := maxInt(window.Y4Start, req.TreeRequest.Y4)
		yEnd := minInt(window.Y4End, req.TreeRequest.Y4+int(req.TreeRequest.VisibleH4))
		xStart := maxInt(window.X4Start, req.TreeRequest.X4)
		xEnd := minInt(window.X4End, req.TreeRequest.X4+int(req.TreeRequest.VisibleW4))
		txbReq := TXBDecodeRequest{
			EOBMultiContext:       req.EOBMultiContext,
			SkipAllZeroCoeffClear: req.SkipAllZeroCoeffClear,
		}
		for y := yStart; y < yEnd; y += int(dims.H4) {
			for x := xStart; x < xEnd; x += int(dims.W4) {
				visibleW := minInt(int(dims.W4), req.TreeRequest.X4+int(req.TreeRequest.VisibleW4)-x)
				visibleH := minInt(int(dims.H4), req.TreeRequest.Y4+int(req.TreeRequest.VisibleH4)-y)
				if visibleW <= 0 || visibleH <= 0 {
					return stats, ErrInvalidDecodeState
				}
				block := TransformBlock{
					X4:        x,
					Y4:        y,
					Size:      req.Tree.Y,
					VisibleW4: uint8(visibleW),
					VisibleH4: uint8(visibleH),
				}
				ctxReq := CoeffContextRequest{
					Plane:      0,
					PlaneBlock: req.TreeRequest.Size,
					Size:       block.Size,
					X4:         block.X4,
					Y4:         block.Y4,
					VisibleW4:  block.VisibleW4,
					VisibleH4:  block.VisibleH4,
				}
				typ, result, coeffs, scan, err := s.decodeCoeffTXBWithDeferredTransform(cdfs, ctx, scratch, ctxReq, txbReq, req.TransformSelect, req.TransformType, req.UseTransformType, req.Class, CoeffTransformRequest{
					Plane: 0,
					Block: block,
				})
				if err != nil {
					return stats, fmt.Errorf("decode luma txb block=%+v ctx=%+v: %w", block, ctxReq, err)
				}

				stats.TXBs++
				stats.EOBTotal += result.EOB
				if result.AllZero {
					stats.AllZero++
				} else {
					stats.NonZero++
				}
				if err := visit(LumaCoeffBlock{
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

	err := req.Tree.forEachLumaTXBInWindow(req.TreeRequest, window, func(block TransformBlock) error {
		ctxReq := CoeffContextRequest{
			Plane:      0,
			PlaneBlock: req.TreeRequest.Size,
			Size:       block.Size,
			X4:         block.X4,
			Y4:         block.Y4,
			VisibleW4:  block.VisibleW4,
			VisibleH4:  block.VisibleH4,
		}
		typ, result, coeffs, scan, err := s.decodeCoeffTXBWithDeferredTransform(cdfs, ctx, scratch, ctxReq, TXBDecodeRequest{
			EOBMultiContext:       req.EOBMultiContext,
			SkipAllZeroCoeffClear: req.SkipAllZeroCoeffClear,
		}, req.TransformSelect, req.TransformType, req.UseTransformType, req.Class, CoeffTransformRequest{
			Plane: 0,
			Block: block,
		})
		if err != nil {
			return fmt.Errorf("decode luma txb block=%+v ctx=%+v: %w", block, ctxReq, err)
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

// chromaCoeffPlanePrep holds the per-plane chroma geometry derived once from a
// ChromaCoeffTreeRequest, shared across all 64x64 unit windows of the block.
type chromaCoeffPlanePrep struct {
	HasChroma  bool
	PlaneBlock BlockSize
	X4         int
	Y4         int
	VisibleW4  int
	VisibleH4  int
	UVDims     TransformDimensions
}

// prepareChromaCoefficients validates a chroma plane request, performs the
// once-per-block SkipTransform context reset, and returns the plane geometry.
// HasChroma is false when the block has no chroma to decode for this plane.
func (s *DecodeState) prepareChromaCoefficients(ctx *CoeffEntropyContext, req ChromaCoeffTreeRequest) (chromaCoeffPlanePrep, error) {
	if s == nil || ctx == nil || req.Plane < 1 || req.Plane > 2 || !req.Class.Valid() {
		return chromaCoeffPlanePrep{}, ErrInvalidDecodeState
	}
	if _, err := validateTransformTreeRequest(req.TreeRequest); err != nil {
		return chromaCoeffPlanePrep{}, err
	}
	if !req.Tree.HasUV || !req.Tree.UV.Valid() {
		return chromaCoeffPlanePrep{}, ErrInvalidDecodeState
	}
	if !HasChromaBlock(req.TreeRequest, req.Color) {
		return chromaCoeffPlanePrep{}, nil
	}
	planeBlock, err := PlaneBlockSize(req.TreeRequest.Size, req.Color, req.Plane)
	if err != nil {
		return chromaCoeffPlanePrep{}, err
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
		return chromaCoeffPlanePrep{}, ErrInvalidDecodeState
	}

	if req.TreeRequest.SkipTransform {
		if err := ctx.ResetBlock(req.Plane, planeBlock, x4, y4); err != nil {
			return chromaCoeffPlanePrep{}, err
		}
		return chromaCoeffPlanePrep{}, nil
	}

	return chromaCoeffPlanePrep{
		HasChroma:  true,
		PlaneBlock: planeBlock,
		X4:         x4,
		Y4:         y4,
		VisibleW4:  visibleW4,
		VisibleH4:  visibleH4,
		UVDims:     uvDims,
	}, nil
}

func (s *DecodeState) DecodeChromaCoefficients(cdfs *CoeffCDFs, ctx *CoeffEntropyContext, scratch *LumaCoeffTreeScratch, req ChromaCoeffTreeRequest, visit ChromaCoeffVisitor) (LumaCoeffStats, error) {
	if cdfs == nil || scratch == nil || visit == nil {
		return LumaCoeffStats{}, ErrInvalidDecodeState
	}
	prep, err := s.prepareChromaCoefficients(ctx, req)
	if err != nil {
		return LumaCoeffStats{}, err
	}
	if !prep.HasChroma {
		return LumaCoeffStats{}, nil
	}
	return s.decodeChromaCoefficientsInWindow(cdfs, ctx, scratch, req, fullCoeffUnitWindow(req.TreeRequest), prep, visit)
}

// decodeChromaCoefficientsInWindow decodes the chroma transform blocks of one
// 64x64 luma max-unit (the window) for a single chroma plane. The per-TXB decode
// matches DecodeChromaCoefficients exactly; only the plane-relative loop bounds
// are clamped to the window's subsampled extent, mirroring libaom's
// blk_row/blk_col range [row>>ss, ROUND_POWER_OF_TWO(min(row+mu,max),ss)).
func (s *DecodeState) decodeChromaCoefficientsInWindow(cdfs *CoeffCDFs, ctx *CoeffEntropyContext, scratch *LumaCoeffTreeScratch, req ChromaCoeffTreeRequest, window coeffUnitWindow, prep chromaCoeffPlanePrep, visit ChromaCoeffVisitor) (LumaCoeffStats, error) {
	planeBlock := prep.PlaneBlock
	x4, y4 := prep.X4, prep.Y4
	visibleW4, visibleH4 := prep.VisibleW4, prep.VisibleH4
	uvDims := prep.UVDims

	ssX := int(boolToShift(req.Color.SubsamplingX))
	ssY := int(boolToShift(req.Color.SubsamplingY))

	// Convert the luma unit window into plane-relative chroma 4x4 offsets. The
	// window bounds are absolute superblock-local luma coordinates; subtract the
	// block origin to get the libaom `row`/`col` luma offsets, then subsample.
	lumaXStart := maxInt(window.X4Start-req.TreeRequest.X4, 0)
	lumaYStart := maxInt(window.Y4Start-req.TreeRequest.Y4, 0)
	lumaXEnd := window.X4End - req.TreeRequest.X4
	lumaYEnd := window.Y4End - req.TreeRequest.Y4

	cxStart := lumaXStart >> ssX
	cyStart := lumaYStart >> ssY
	cxEnd := minInt((lumaXEnd+ssX)>>ssX, visibleW4)
	cyEnd := minInt((lumaYEnd+ssY)>>ssY, visibleH4)

	var stats LumaCoeffStats
	for y := cyStart; y < cyEnd; y += int(uvDims.H4) {
		for x := cxStart; x < cxEnd; x += int(uvDims.W4) {
			block := TransformBlock{
				X4:        x4 + x,
				Y4:        y4 + y,
				Size:      req.Tree.UV,
				VisibleW4: uint8(minInt(int(uvDims.W4), visibleW4-x)),
				VisibleH4: uint8(minInt(int(uvDims.H4), visibleH4-y)),
			}
			ctxReq := CoeffContextRequest{
				Plane:      req.Plane,
				PlaneBlock: planeBlock,
				Size:       block.Size,
				X4:         block.X4,
				Y4:         block.Y4,
				VisibleW4:  block.VisibleW4,
				VisibleH4:  block.VisibleH4,
			}
			typ, result, coeffs, scan, err := s.decodeCoeffTXBWithDeferredTransform(cdfs, ctx, scratch, ctxReq, TXBDecodeRequest{
				EOBMultiContext:       req.EOBMultiContext,
				SkipAllZeroCoeffClear: req.SkipAllZeroCoeffClear,
			}, req.TransformSelect, req.TransformType, req.UseTransformType, req.Class, CoeffTransformRequest{
				Plane: req.Plane,
				Block: block,
			})
			if err != nil {
				return stats, fmt.Errorf("decode chroma txb plane=%d block=%+v ctx=%+v: %w", req.Plane, block, ctxReq, err)
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

type coeffTransformRecorder interface {
	RecordCoeffTransform(CoeffTransformRequest, transform.Type) error
}

func (s *DecodeState) decodeCoeffTXBWithDeferredTransform(cdfs *CoeffCDFs, ctx *CoeffEntropyContext, scratch *LumaCoeffTreeScratch, ctxReq CoeffContextRequest, req TXBDecodeRequest, selector CoeffTransformSelector, typ transform.Type, useType bool, class transform.Class, transformReq CoeffTransformRequest) (transform.Type, TXBDecodeResult, []int16, []int16, error) {
	if cdfs == nil || ctx == nil {
		return 0, TXBDecodeResult{}, nil, nil, ErrInvalidDecodeState
	}
	plane, err := CoeffPlaneTypeForPlane(ctxReq.Plane)
	if err != nil {
		return 0, TXBDecodeResult{}, nil, nil, err
	}
	txDims, blockDims, visibleW, visibleH, err := validateCoeffContextRequestWithSpan(ctxReq)
	if err != nil {
		return 0, TXBDecodeResult{}, nil, nil, err
	}
	txbCtx, err := ctx.txbContextKnown(ctxReq, txDims, blockDims)
	if err != nil {
		return 0, TXBDecodeResult{}, nil, nil, fmt.Errorf("read txb context: %w", err)
	}
	req.Size = ctxReq.Size
	req.Plane = plane
	req.TXBSkipContext = txbCtx.TXBSkipContext
	req.DCSignContext = txbCtx.DCSignContext
	geo, ok := coeffGeo(req.Size)
	if !ok || txbCtx.TXBSkipContext < 0 || txbCtx.TXBSkipContext >= TXBSkipContexts {
		return 0, TXBDecodeResult{}, nil, nil, ErrInvalidDecodeState
	}
	allZero := s.Reader.ReadBinaryCDFUnchecked(&cdfs.TXBSkip[geo.txCtx][txbCtx.TXBSkipContext]) != 0

	selected := transform.TypeDCTDCT
	selectedClass := transform.Class2D
	if !allZero {
		selected, selectedClass, err = resolveCoeffTransform(selector, typ, useType, class, transformReq.Plane, transformReq.Block)
		if err != nil {
			return 0, TXBDecodeResult{}, nil, nil, fmt.Errorf("resolve coeff transform: %w", err)
		}
	}
	if allZero && req.SkipAllZeroCoeffClear {
		result := TXBDecodeResult{AllZero: true}
		if err := ctx.markTXBKnown(ctxReq, txDims, visibleW, visibleH, result); err != nil {
			return 0, TXBDecodeResult{}, nil, nil, fmt.Errorf("mark txb result=%+v: %w", result, err)
		}
		if err := recordCoeffTransform(selector, transformReq, selected); err != nil {
			return 0, TXBDecodeResult{}, nil, nil, fmt.Errorf("record coeff transform=%v: %w", selected, err)
		}
		return selected, result, nil, nil, nil
	}

	coeffs, scan, levels, err := scratch.coeffBuffers(ctxReq.Size, selectedClass)
	if err != nil {
		return 0, TXBDecodeResult{}, nil, nil, fmt.Errorf("prepare coeff buffers size=%v class=%v: %w", ctxReq.Size, selectedClass, err)
	}
	scratch.clearCoeffDirty()
	req.Class = selectedClass
	req.EOBMultiContext = eobMultiContextForClass(selectedClass)
	req.TXBSkipKnown = true
	req.TXBSkip = allZero
	req.skipNonZeroCoeffClear = req.SkipAllZeroCoeffClear
	req.coeffDirtyPos = &scratch.InverseScan
	req.coeffDirtyLen = &scratch.coeffDirtyLen
	result, err := s.ReadCoefficientsTXB(cdfs, req, coeffs, scan, levels)
	if err != nil {
		return 0, TXBDecodeResult{}, nil, nil, fmt.Errorf("read coeff txb req=%+v coeffs=%d scan=%d levels=%d selected=%v: %w", req, len(coeffs), len(scan), len(levels), selected, err)
	}
	if err := ctx.markTXBKnown(ctxReq, txDims, visibleW, visibleH, result); err != nil {
		return 0, TXBDecodeResult{}, nil, nil, fmt.Errorf("mark txb result=%+v: %w", result, err)
	}
	if err := recordCoeffTransform(selector, transformReq, selected); err != nil {
		return 0, TXBDecodeResult{}, nil, nil, fmt.Errorf("record coeff transform=%v: %w", selected, err)
	}
	return selected, result, coeffs, scan, nil
}

func recordCoeffTransform(selector CoeffTransformSelector, req CoeffTransformRequest, selected transform.Type) error {
	switch recorder := selector.(type) {
	case nil, *IntraCoeffTransformSelector, CoeffTransformSelectorFunc:
		return nil
	case *InterCoeffTransformSelector:
		return recorder.RecordCoeffTransform(req, selected)
	default:
		if recorder, ok := selector.(coeffTransformRecorder); ok {
			return recorder.RecordCoeffTransform(req, selected)
		}
		return nil
	}
}

func eobMultiContextForClass(class transform.Class) int {
	if class == transform.Class2D {
		return 0
	}
	return 1
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
	geo, ok := coeffGeo(size)
	if !ok || !class.Valid() {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	scanLen := geo.maxEOB
	if scanLen > len(s.Coeffs) {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	scan := coeffScanTable[size][class]
	if len(scan) < scanLen {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	levelsLen := geo.scratchLen
	if levelsLen > len(s.Levels) {
		return nil, nil, nil, ErrInvalidDecodeState
	}
	return s.Coeffs[:scanLen], scan, s.Levels[:levelsLen], nil
}
