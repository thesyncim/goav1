package tile

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// CoefBlock is the per-block binding dav1d's recon_b_intra/recon_b_inter and
// read_coef_blocks (src/recon_tmpl.c) read from Av1Block, t and f while they
// walk a block's transform blocks. decode_b binds it once per block (bindCoefBlock)
// after the transform sizes are read; the exported fields are the walk geometry
// the reconstruction side iterates, the rest is what DecodeCoefs needs.
type CoefBlock struct {
	s       *DecodeState
	cdfs    *CoeffCDFs
	txCDFs  *TransformTypeCDFs
	ctx     *CoeffEntropyContext
	scratch *BlockCoeffScratch
	// txtpMap is dav1d's t->scratch.txtp_map: the luma transform types an
	// inter block's chroma TXBs derive theirs from.
	txtpMap *[MaxBlockModeSlots][MaxBlockModeSlots]transform.Type

	reduced    bool
	qZero      bool
	lumaMode   IntraMode
	chromaMode ChromaIntraMode
	blockDims  [3]BlockDimensions

	// Tree holds the block's transform sizes (b->tx / b->max_ytx, b->uvtx and
	// the var-tx split masks).
	Tree TransformTreeResult
	// X4/Y4 and W4/H4 are the superblock-local luma origin and the
	// frame-clipped size in 4x4 units (dav1d's t->bx & 31 and w4/h4); CX4/CY4
	// and CW4/CH4 the same for chroma, valid when HasChroma.
	X4, Y4, W4, H4     int
	CX4, CY4, CW4, CH4 int
	SSX, SSY           uint
	HasChroma          bool
	Intra              bool
	Skip               bool
	Lossless           bool
}

// CoefDequant is one plane's dav1d dq_tbl[0..1] together with cf_max.
type CoefDequant struct {
	DC, AC, Max uint32
}

// CoefResult is what decode_coefs hands back: the TXB result (EOB is dav1d's
// eob + 1, zero with AllZero), the transform type, and the extent of the
// nonzero dequantized coefficients the inverse transform has to read.
type CoefResult struct {
	TXB        TXBDecodeResult
	Txtp       transform.Type
	ActiveRows uint8
	ActiveCols uint8
}

// bindCoefBlock is the part of dav1d's decode_b that precedes the
// bd_fn.recon_b_* call: the transform sizes (read_tx_size / read_vartx_tree),
// the skip block's a/l coefficient-context reset, and the per-block binding
// the transform-block walk reads.
func (s *DecodeState) bindCoefBlock(cdfs *BlockLoopCDFs, scratch *BlockLoopScratch, req *BlockLoopRequest, visit *BlockLoopVisit) error {
	block := &visit.Block
	pred := &visit.Prediction
	intra := pred.Valid && pred.Intra
	skip := visit.Prefix.SkipTransform
	qIndex := s.CurrentBaseQIdx
	if req.Segmentation.Enabled {
		if visit.SegmentID >= parser.MaxSegments {
			return ErrInvalidDecodeState
		}
		qIndex = quantize.ClampQIndex(qIndex, req.Segmentation.Data.Segments[visit.SegmentID].DeltaQ)
	} else if visit.SegmentID != 0 {
		return ErrInvalidDecodeState
	}
	lossless := qIndex == 0 && req.QuantDeltasZero

	tr := &scratch.coefTree
	tr.Size = block.Size
	tr.X4, tr.Y4 = block.X4, block.Y4
	tr.VisibleW4, tr.VisibleH4 = block.VisibleW4, block.VisibleH4
	tr.HaveTop, tr.HaveLeft = block.HaveTop, block.HaveLeft
	tr.Color = req.Color
	tr.TransformMode = req.TransformMode
	tr.Inter = !intra
	tr.SkipTransform = skip
	tr.Lossless = lossless
	tree, err := s.DecodeTransformTree(cdfs.Transform, &scratch.Mode, tr)
	if err != nil {
		return fmt.Errorf("decode transform tree req=%+v: %w", *tr, err)
	}
	visit.Coefficients = BlockCoeffResult{Tree: tree}

	b := &scratch.coef
	b.s, b.cdfs, b.txCDFs = s, cdfs.Coeff, cdfs.TransformType
	b.ctx, b.scratch = &scratch.CoeffCtx, &scratch.Coeff
	b.reduced = req.ReducedTxSet
	b.qZero = qIndex == 0
	b.Intra, b.Skip, b.Lossless = intra, skip, lossless
	b.txtpMap = nil
	if intra {
		b.lumaMode = pred.LumaMode
		if pred.FilterIntraValid {
			b.lumaMode = filterIntraTransformMode(pred.FilterIntraMode)
		}
		b.chromaMode = pred.ChromaMode
		if !b.chromaMode.Valid() {
			b.chromaMode = ChromaIntraModeDC
		}
	} else {
		b.txtpMap = &scratch.Coeff.txtpMap
	}
	b.Tree = tree
	b.X4, b.Y4 = int(block.X4), int(block.Y4)
	b.W4, b.H4 = int(block.VisibleW4), int(block.VisibleH4)
	lumaDims, ok := block.Size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	b.blockDims[0] = lumaDims
	b.SSX = uint(boolToShift(req.Color.SubsamplingX))
	b.SSY = uint(boolToShift(req.Color.SubsamplingY))
	b.HasChroma = tree.HasUV && hasChromaForBlock(block.Size, b.X4, b.Y4, req.Color)
	if b.HasChroma {
		planeBlock, err := PlaneBlockSize(block.Size, req.Color, 1)
		if err != nil {
			return err
		}
		dims, ok := planeBlock.Dimensions()
		if !ok || !tree.UV.Valid() {
			return ErrInvalidDecodeState
		}
		b.blockDims[1], b.blockDims[2] = dims, dims
		b.CX4, b.CY4 = b.X4>>b.SSX, b.Y4>>b.SSY
		b.CW4 = ((b.X4 + b.W4 + int(b.SSX)) >> b.SSX) - b.CX4
		b.CH4 = ((b.Y4 + b.H4 + int(b.SSY)) >> b.SSY) - b.CY4
		if b.CW4 <= 0 || b.CH4 <= 0 || b.CX4+b.CW4 > MaxBlockModeSlots || b.CY4+b.CH4 > MaxBlockModeSlots {
			return ErrInvalidDecodeState
		}
		if skip {
			for plane := 1; plane <= 2; plane++ {
				if err := b.ctx.ResetBlock(plane, planeBlock, b.CX4, b.CY4); err != nil {
					return err
				}
			}
		}
	}
	if skip {
		return b.ctx.ResetBlock(0, block.Size, b.X4, b.Y4)
	}
	return nil
}

// filterIntraTransformMode is dav1d's dav1d_filter_mode_to_y_mode: the intra
// mode a filter-intra block's transform type is coded under.
func filterIntraTransformMode(mode FilterIntraMode) IntraMode {
	switch mode {
	case FilterIntraModeVertical:
		return IntraModeVertical
	case FilterIntraModeHorizontal:
		return IntraModeHorizontal
	case FilterIntraModeD157:
		return IntraModeD157
	default:
		return IntraModeDC
	}
}

// DecodeCoefs is dav1d's decode_coefs (src/recon_tmpl.c) for one plane TXB at
// a/l offsets x4, y4 of the bound block, followed by the caller-side updates of
// read_coef_tree and recon_b_*: the a/l coefficient contexts and, for inter
// luma, txtp_map. visW/visH are the frame-clipped extent in 4x4 units. With cf
// and dq set, a coded TXB is dequantized into cf in raster scan order (cf must
// be zero over the scan area when cfZeroed, see decodeCoefTokens); with cf nil
// it is read as quantized coefficients into the binding's scratch
// (QuantizedCoefs), the path lossless and quantizer-matrix blocks take.
func (b *CoefBlock) DecodeCoefs(plane uint8, size TransformSize, x4, y4, visW, visH int, cf *[1024]int32, cfZeroed bool, dq *CoefDequant) (CoefResult, error) {
	s := b.s
	if size >= transformSizeCount || plane > 2 {
		return CoefResult{}, ErrInvalidDecodeState
	}
	txDims := &transformDimensions[size]
	tw4, th4, bd := int(txDims.W4), int(txDims.H4), &b.blockDims[plane]
	txbCtx := b.ctx.txbContextTrusted(plane, x4, y4, tw4, th4, int(bd.W4), int(bd.H4))
	geo := &coeffGeometryTable[size]
	if !geo.valid || txbCtx.TXBSkipContext >= TXBSkipContexts {
		return CoefResult{}, ErrInvalidDecodeState
	}
	coeffPlane := CoeffPlaneY
	if plane != 0 {
		coeffPlane = CoeffPlaneUV
	}

	// does this block have any non-zero coefficients
	allSkip := s.Reader.ReadBinaryCDFUnchecked(&b.cdfs.TXBSkip[geo.txCtx][txbCtx.TXBSkipContext]) != 0
	var r CoefResult
	r.Txtp = transform.TypeDCTDCT
	if !allSkip {
		var err error
		if r.Txtp, err = b.readTxtp(plane, size, x4, y4); err != nil {
			return CoefResult{}, err
		}
	}
	if plane == 0 && b.txtpMap != nil {
		if x4+tw4 > MaxBlockModeSlots || y4+th4 > MaxBlockModeSlots {
			return CoefResult{}, ErrInvalidDecodeState
		}
		fillTxtpMap(b.txtpMap, x4, y4, tw4, th4, r.Txtp)
	}
	if allSkip {
		r.TXB.AllZero = true
		b.ctx.setTXBContextKnown(plane, x4, y4, tw4, th4, visW, visH, 0)
		return r, nil
	}
	class := transform.Class2D
	if r.Txtp >= transform.TypeIDTX {
		var err error
		if class, err = r.Txtp.Class(); err != nil {
			return CoefResult{}, ErrInvalidDecodeState
		}
	}
	if cf == nil || dq == nil {
		var err error
		if r.TXB, err = b.readQuantizedTXB(size, coeffPlane, class, txbCtx, geo); err != nil {
			return CoefResult{}, err
		}
	} else {
		// dav1d's dq_shift: imax(0, t_dim->ctx - 2).
		txScale := uint8(0)
		if geo.txCtx > 2 {
			txScale = geo.txCtx - 2
		}
		var c entropy.Cursor
		c.Load(&s.Reader)
		t := decodeCoefTokens(&c, b.cdfs, size, geo, coeffPlane, r.Txtp, class, txbCtx.DCSignContext, cf, cfZeroed, &b.scratch.Coeff.CoefLevels, coefDequant{
			dc:    dq.DC,
			ac:    dq.AC,
			shift: txScale,
			cfMax: dq.Max,
		})
		c.CommitStateTo(&s.Reader)
		r.TXB = TXBDecodeResult{EOB: uint16(t.eob + 1), CulLevel: t.culLevel}
		r.ActiveRows, r.ActiveCols = uint8(t.activeRows), uint8(t.activeCols)
	}
	b.ctx.setTXBContextKnown(plane, x4, y4, tw4, th4, visW, visH, r.TXB.CulLevel)
	return r, nil
}

// QuantizedCoefs returns the quantized coefficients and scan of the TXB the
// last cf-less DecodeCoefs read.
func (b *CoefBlock) QuantizedCoefs(size TransformSize, txtp transform.Type) (coeffs, scan []int16) {
	coeffs, scan, _, _ = b.scratch.Coeff.coeffBuffersWithGeo(size, mustClass(txtp), &coeffGeometryTable[size])
	return coeffs, scan
}

// fillTxtpMap is decode_coefs' dav1d_memset_pow2[t_dim->lw] of txtp into each
// of the TXB's txtp_map rows: one store per row for widths up to 4, 8-byte
// stores above. The caller has bounded x4+w4 and y4+h4.
func fillTxtpMap(m *[MaxBlockModeSlots][MaxBlockModeSlots]transform.Type, x4, y4, w4, h4 int, t transform.Type) {
	x := ctxSplat(uint8(t))
	switch w4 {
	case 1:
		for y := y4; y < y4+h4; y++ {
			m[y][x4] = t
		}
	case 2:
		for y := y4; y < y4+h4; y++ {
			ctxPut2(m[y][x4:], x)
		}
	case 4:
		for y := y4; y < y4+h4; y++ {
			ctxPut4(m[y][x4:], x)
		}
	default:
		for y := y4; y < y4+h4; y++ {
			row := m[y][x4:]
			for i := 0; i < w4; i += 8 {
				ctxPut8(row[i:], x)
			}
		}
	}
}

func mustClass(t transform.Type) transform.Class {
	c, _ := t.Class()
	return c
}

// readTxtp is the transform-type section of dav1d's decode_coefs for a
// non-skipped TXB: chroma types are derived (intra from the UV mode, inter
// from the co-located luma type in txtp_map), luma types are read from the
// per-size set unless qidx is zero. A lossless TXB is a WHT_WHT, which the
// reconstruction keys on the block's quantizer and reads as DCT_DCT.
func (b *CoefBlock) readTxtp(plane uint8, size TransformSize, x4, y4 int) (transform.Type, error) {
	if b.Lossless {
		return transform.TypeDCTDCT, nil
	}
	if plane != 0 {
		if b.Intra {
			return IntraChromaTransformType(b.chromaMode, size, b.reduced)
		}
		// dav1d's recon_b_inter: txtp_map[(by4 + (y << ss_ver)) * 32 +
		// bx4 + (x << ss_hor)], with x/y the TXB offset in the chroma block.
		lx := b.X4 + (x4-b.CX4)<<b.SSX
		ly := b.Y4 + (y4-b.CY4)<<b.SSY
		if uint(lx) >= MaxBlockModeSlots || uint(ly) >= MaxBlockModeSlots {
			return 0, ErrInvalidDecodeState
		}
		luma := b.txtpMap[ly][lx]
		return uvInterTransformType(size, luma)
	}
	if b.qZero {
		return transform.TypeDCTDCT, nil
	}
	inter := 0
	if !b.Intra {
		inter = 1
	}
	p := &txTypeReadPlans[inter][txTypePlanReduced(b.reduced)][size]
	if !p.ok {
		return 0, ErrInvalidDecodeState
	}
	if p.symbols <= 1 {
		return transform.TypeDCTDCT, nil
	}
	var cdf *entropy.CDF
	var err error
	if b.Intra {
		cdf, err = b.txCDFs.IntraCDF(p.index, p.square, b.lumaMode, p.symbols)
	} else {
		cdf, err = b.txCDFs.InterCDF(p.index, p.square, p.symbols)
	}
	if err != nil {
		return 0, err
	}
	symbol, err := b.s.ReadSymbol(cdf)
	if err != nil {
		return 0, err
	}
	return p.typeFromSymbol(symbol)
}

// readQuantizedTXB decodes a nonzero TXB into the quantized coefficient
// scratch when the plane's dequant factors cannot be prepared; the visitor
// then dequantizes during reconstruction.
//
//go:noinline
func (b *CoefBlock) readQuantizedTXB(size TransformSize, plane CoeffPlaneType, class transform.Class, txbCtx TXBContext, geo *coeffGeometry) (TXBDecodeResult, error) {
	scratch := &b.scratch.Coeff
	coeffs, scan, levels, err := scratch.coeffBuffersWithGeo(size, class, geo)
	if err != nil {
		return TXBDecodeResult{}, err
	}
	scratch.clearCoeffDirty()
	req := TXBDecodeRequest{
		Size:                  size,
		Plane:                 plane,
		Class:                 class,
		TXBSkipContext:        txbCtx.TXBSkipContext,
		DCSignContext:         txbCtx.DCSignContext,
		EOBMultiContext:       eobMultiContextForClass(class),
		TXBSkipKnown:          true,
		knownNonZero:          true,
		SkipAllZeroCoeffClear: true,
		skipNonZeroCoeffClear: true,
		coeffDirtyPos:         &scratch.InverseScan,
		coeffDirtyLen:         &scratch.coeffDirtyLen,
		levelDirtyPos:         &scratch.LevelDirty,
		levelDirtyLen:         &scratch.levelDirtyLen,
		levelDirtyScratch:     &scratch.Levels,
		trustedScan:           true,
	}
	if class == transform.Class2D {
		return b.s.readCoefficientsTXBTracked2DWithGeo(b.cdfs, req, coeffs, scan, levels, geo)
	}
	return b.s.readCoefficientsTXBWithGeo(b.cdfs, req, coeffs, scan, levels, geo)
}
