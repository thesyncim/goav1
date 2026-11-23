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

// CoefDequant is one plane's dav1d dq_tbl[0..1] together with cf_max and the
// plane's quantizer matrices (dav1d's f->qm[][plane]), nil when the plane
// dequantizes flat.
type CoefDequant struct {
	DC, AC, Max uint32
	QM          *CoefQMatrices
}

// CoefQMatrices is one level and plane class of dav1d's dav1d_qm_tbl: the
// quantizer matrix of every transform size, libaom's tables in coefficient
// order, which is the cf order DecodeCoefs indexes with rc directly. The
// 64-sample sizes share the matrix of their 32-sample coefficient scan.
type CoefQMatrices [transformSizeCount][]uint16

var coefQMTables [quantize.QMLevels - 1][2]CoefQMatrices

func init() {
	for level := range coefQMTables {
		for class := range coefQMTables[level] {
			for size := range TransformSize(transformSizeCount) {
				ts, err := size.TransformSize()
				if err != nil {
					panic(err)
				}
				coefQMTables[level][class][size] = quantize.QMatrix(uint8(level), class == 1, ts)
			}
		}
	}
}

// QMatrices returns the quantizer matrices of level for luma (chroma false)
// or chroma, nil for the flat level 15.
func QMatrices(level uint8, chroma bool) *CoefQMatrices {
	if level >= quantize.QMLevels-1 {
		return nil
	}
	class := 0
	if chroma {
		class = 1
	}
	return &coefQMTables[level][class]
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
	qIndex, lossless, ok := blockQIndex(req, s.CurrentBaseQIdx, visit.SegmentID)
	if !ok {
		return ErrInvalidDecodeState
	}

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

// blockQIndex is the block's segment-adjusted qindex (the delta-coded current
// qindex clamped with the segment's delta_q, libaom's xd->qindex[segment_id])
// that gates the luma transform-type read, and whether the block is lossless.
func blockQIndex(req *BlockLoopRequest, current, segmentID uint8) (qIndex uint8, lossless, ok bool) {
	qIndex = current
	if req.Segmentation.Enabled {
		if segmentID >= parser.MaxSegments {
			return 0, false, false
		}
		qIndex = quantize.ClampQIndex(qIndex, req.Segmentation.Data.Segments[segmentID].DeltaQ)
	} else if segmentID != 0 {
		return 0, false, false
	}
	return qIndex, qIndex == 0 && req.QuantDeltasZero, true
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
// luma, txtp_map. visW/visH are the frame-clipped extent in 4x4 units. A coded
// TXB is dequantized with dq into cf in raster scan order (cf must be zero over
// the scan area when cfZeroed, see decodeCoefTokens). One msac cursor carries
// every read from the skip flag to the last sign.
func (b *CoefBlock) DecodeCoefs(plane uint8, size TransformSize, x4, y4, visW, visH int, cf *[1024]int32, cfZeroed bool, dq *CoefDequant) (CoefResult, error) {
	s := b.s
	if size >= transformSizeCount || plane > 2 {
		return CoefResult{}, ErrInvalidDecodeState
	}
	tDim := &transformDimensions[size]
	tw4, th4, bd := int(tDim.W4), int(tDim.H4), &b.blockDims[plane]
	if x4+tw4 > MaxBlockModeSlots || y4+th4 > MaxBlockModeSlots {
		return CoefResult{}, ErrInvalidDecodeState
	}
	geo := &coeffGeometryTable[size]
	if !geo.valid {
		return CoefResult{}, ErrInvalidDecodeState
	}
	coeffPlane := CoeffPlaneY
	if plane != 0 {
		coeffPlane = CoeffPlaneUV
	}
	var c entropy.Cursor
	c.Load(&s.Reader)

	// does this block have any non-zero coefficients
	sctx := b.ctx.coefSkipCtx(plane, x4, y4, tw4, th4, int(bd.W4), int(bd.H4))
	allSkip := c.ReadBinaryCDFUnchecked(&b.cdfs.TXBSkip[geo.txCtx][sctx]) != 0
	var r CoefResult
	r.Txtp = transform.TypeDCTDCT
	if !allSkip {
		var err error
		if r.Txtp, err = b.readTxtp(&c, plane, size, tDim, x4, y4); err != nil {
			return CoefResult{}, err
		}
	}
	if plane == 0 && b.txtpMap != nil {
		fillTxtpMap(b.txtpMap, x4, y4, tw4, th4, r.Txtp)
	}
	if allSkip {
		c.CommitStateTo(&s.Reader)
		r.TXB.AllZero = true
		b.ctx.setTXBContextKnown(plane, x4, y4, tw4, th4, visW, visH, 0)
		return r, nil
	}
	class := transform.Class2D
	if r.Txtp >= transform.TypeIDTX && r.Txtp != transform.TypeWHTWHT {
		var err error
		if class, err = r.Txtp.Class(); err != nil {
			return CoefResult{}, ErrInvalidDecodeState
		}
	}
	eob, rc, dcTok := decodeCoefTokens(&c, b.cdfs, size, geo, coeffPlane, r.Txtp, class, cf, cfZeroed, &b.scratch.Coeff.CoefLevels)
	var dcSignCDF *entropy.CDF
	if dcTok != 0 {
		dcSignCDF = &b.cdfs.DCSign[coeffPlane][b.ctx.coefDCSignCtx(plane, x4, y4, tw4, th4)]
	}
	// dav1d's dq_shift: imax(0, t_dim->ctx - 2).
	txScale := uint8(0)
	if geo.txCtx > 2 {
		txScale = geo.txCtx - 2
	}
	var qm []uint16
	if dq.QM != nil && r.Txtp < transform.TypeIDTX {
		qm = dq.QM[size]
	}
	t := decodeCoefResidual(&c, dcSignCDF, eob, rc, dcTok, cf, coefDequant{
		dc:    dq.DC,
		ac:    dq.AC,
		shift: txScale,
		cfMax: dq.Max,
	}, qm, geo)
	c.CommitStateTo(&s.Reader)
	r.TXB = TXBDecodeResult{EOB: uint16(t.eob + 1), CulLevel: t.culLevel}
	r.ActiveRows, r.ActiveCols = uint8(t.activeRows), uint8(t.activeCols)
	b.ctx.setTXBContextKnown(plane, x4, y4, tw4, th4, visW, visH, r.TXB.CulLevel)
	return r, nil
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

// readTxtp is the transform-type section of dav1d's decode_coefs for a
// non-skipped TXB: lossless TXBs are WHT_WHT, 64-sample (and intra
// 32-sample) sides only allow DCT_DCT, chroma types are derived (intra from
// the UV mode, inter from the co-located luma type in txtp_map), and luma
// types are read from the set CDF dav1d indexes by t_dim->min unless qidx is
// zero.
func (b *CoefBlock) readTxtp(c *entropy.Cursor, plane uint8, size TransformSize, tDim *TransformDimensions, x4, y4 int) (transform.Type, error) {
	intra := uint8(0)
	if b.Intra {
		intra = 1
	}
	switch {
	case b.Lossless:
		return transform.TypeWHTWHT, nil
	case tDim.Max+intra >= 4:
		return transform.TypeDCTDCT, nil
	case plane != 0:
		if b.Intra {
			return txtpFromUVMode[b.chromaMode%chromaIntraModeCount], nil
		}
		// dav1d's recon_b_inter: txtp_map[(by4 + (y << ss_ver)) * 32 +
		// bx4 + (x << ss_hor)], with x/y the TXB offset in the chroma block.
		lx := b.X4 + (x4-b.CX4)<<b.SSX
		ly := b.Y4 + (y4-b.CY4)<<b.SSY
		if uint(lx) >= MaxBlockModeSlots || uint(ly) >= MaxBlockModeSlots {
			return 0, ErrInvalidDecodeState
		}
		return uvInterTxtp(tDim, b.txtpMap[ly][lx]), nil
	case b.qZero:
		return transform.TypeDCTDCT, nil
	}
	sq := tDim.Min & 3
	cdfs := b.txCDFs
	if b.Intra {
		mode := b.lumaMode % intraModeCount
		if b.reduced || sq == 2 {
			return extTXTypeInv[ExtTXSetDTT4IDTX][c.ReadCDF5Unchecked(&cdfs.Intra[2][sq][mode])&15], nil
		}
		return extTXTypeInv[ExtTXSetDTT4IDTX1DDCT][c.ReadCDF7Unchecked(&cdfs.Intra[1][sq][mode])&15], nil
	}
	switch {
	case b.reduced || tDim.Max == 3:
		return extTXTypeInv[ExtTXSetDCTIDTX][c.ReadBinaryCDFUnchecked(&cdfs.Inter[3][sq])&1], nil
	case sq == 2:
		return extTXTypeInv[ExtTXSetDTT9IDTX1DDCT][c.ReadCDFWideUnchecked(&cdfs.Inter[2][sq], 12)&15], nil
	}
	return extTXTypeInv[ExtTXSetAll16][c.ReadCDFWideUnchecked(&cdfs.Inter[1][sq], 16)&15], nil
}

// txtpFromUVMode is dav1d_txtp_from_uvmode (src/tables.c).
var txtpFromUVMode = [chromaIntraModeCount]transform.Type{
	ChromaIntraModeVertical:         transform.TypeADSTDCT,
	ChromaIntraModeHorizontal:       transform.TypeDCTADST,
	ChromaIntraModeD135:             transform.TypeADSTADST,
	ChromaIntraModeD113:             transform.TypeADSTDCT,
	ChromaIntraModeD157:             transform.TypeDCTADST,
	ChromaIntraModeD203:             transform.TypeDCTADST,
	ChromaIntraModeD67:              transform.TypeADSTDCT,
	ChromaIntraModeSmooth:           transform.TypeADSTADST,
	ChromaIntraModeSmoothVertical:   transform.TypeADSTDCT,
	ChromaIntraModeSmoothHorizontal: transform.TypeDCTADST,
	ChromaIntraModePaeth:            transform.TypeADSTADST,
}

// uvInterTxtp is dav1d's get_uv_inter_txtp (src/env.h).
func uvInterTxtp(uvtDim *TransformDimensions, ytxtp transform.Type) transform.Type {
	if uvtDim.Max == 3 {
		if ytxtp == transform.TypeIDTX {
			return transform.TypeIDTX
		}
		return transform.TypeDCTDCT
	}
	if uvtDim.Min == 2 {
		switch ytxtp {
		case transform.TypeHFlipADST, transform.TypeVFlipADST, transform.TypeHADST, transform.TypeVADST:
			return transform.TypeDCTDCT
		}
	}
	return ytxtp
}
