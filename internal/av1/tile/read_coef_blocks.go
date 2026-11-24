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
	s   *DecodeState
	ctx *CoeffEntropyContext
	// txtpMap is dav1d's t->scratch.txtp_map: the luma transform types an
	// inter block's chroma TXBs derive theirs from.
	txtpMap *[MaxBlockModeSlots][MaxBlockModeSlots]transform.Type
	// e is what decode_coefs reads from the tile, frame and block state.
	e entropy.CoefState

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
type CoefDequant = entropy.CoefDequant

// CoefQMatrices is one level and plane class of dav1d's dav1d_qm_tbl: the
// quantizer matrix of every transform size, libaom's tables in coefficient
// order, which is the cf order DecodeCoefs indexes with rc directly. The
// 64-sample sizes share the matrix of their 32-sample coefficient scan.
type CoefQMatrices = entropy.CoefQMatrices

var _ [transformSizeCount - entropy.CoefSizes]struct{}
var _ [entropy.CoefSizes - transformSizeCount]struct{}

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

	tree := &visit.Coefficients.Tree
	if err := s.readTxSizes(cdfs.Transform, &scratch.Mode, block, &req.Color, req.TransformMode, !intra, skip, lossless, tree); err != nil {
		return fmt.Errorf("decode transform sizes block=%+v: %w", *block, err)
	}

	b := &scratch.coef
	// The binding's pointers are the same for every block of a tile. Store
	// them only when they change: a pointer store runs the write barrier
	// while a garbage collection is marking, and this runs once per block.
	setPtr(&b.s, s)
	setPtr(&b.ctx, &scratch.CoeffCtx)
	e := &b.e
	setPtr(&e.CDF, (*entropy.CoefCDFs)(cdfs.Coeff))
	setPtr(&e.TxIntra, &cdfs.TransformType.Intra)
	setPtr(&e.TxInter, &cdfs.TransformType.Inter)
	setPtr(&e.A, &scratch.CoeffCtx.Above)
	setPtr(&e.L, &scratch.CoeffCtx.Left)
	setPtr(&e.Levels, &scratch.Coeff.Coeff.CoefLevels)
	e.Reduced = req.ReducedTxSet
	e.QZero = qIndex == 0
	e.Intra, e.Lossless = intra, lossless
	b.Intra, b.Skip, b.Lossless = intra, skip, lossless
	if intra {
		setPtr(&b.txtpMap, nil)
		lumaMode := pred.LumaMode
		if pred.FilterIntraValid {
			lumaMode = filterIntraTransformMode(pred.FilterIntraMode)
		}
		chromaMode := pred.ChromaMode
		if !chromaMode.Valid() {
			chromaMode = ChromaIntraModeDC
		}
		e.YMode = uint8(lumaMode % intraModeCount)
		e.UVTxtp = uint8(txtpFromUVMode[chromaMode%chromaIntraModeCount])
	} else {
		setPtr(&b.txtpMap, &scratch.Coeff.txtpMap)
	}
	b.Tree = *tree
	b.X4, b.Y4 = int(block.X4), int(block.Y4)
	b.W4, b.H4 = int(block.VisibleW4), int(block.VisibleH4)
	lumaDims, ok := block.Size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	e.BW4[0], e.BH4[0] = lumaDims.W4, lumaDims.H4
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
		e.BW4[1], e.BH4[1] = dims.W4, dims.H4
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

// The transform-block walk (threading's reconTXB, dav1d's read_coef_tree)
// calls entropy.Reader.DecodeCoefs, dav1d's decode_coefs, itself: no wrapper
// frame sits between the walk and the symbol reads. These accessors hand it
// the bound block's operands.

// CoefEntropy returns the tile's symbol reader and the block's decode_coefs
// state.
func (b *CoefBlock) CoefEntropy() (*entropy.Reader, *entropy.CoefState) {
	return &b.s.Reader, &b.e
}

// TxtpMap is dav1d's t->scratch.txtp_map for an inter block (the luma
// transform types its chroma TXBs derive theirs from), nil for intra.
func (b *CoefBlock) TxtpMap() *[MaxBlockModeSlots][MaxBlockModeSlots]transform.Type {
	return b.txtpMap
}

// CoefSizeOf is decode_coefs' t_dim for size; ok is false outside the
// transform sizes.
func CoefSizeOf(size TransformSize) (sz *entropy.CoefSize, ok bool) {
	if size >= transformSizeCount {
		return nil, false
	}
	return &coefSizeTable[size], true
}

// FillTxtpMap is decode_coefs' dav1d_memset_pow2[t_dim->lw] of txtp into each
// of the TXB's txtp_map rows: one store per row for widths up to 4, 8-byte
// stores above. The caller has bounded x4+w4 and y4+h4.
func FillTxtpMap(m *[MaxBlockModeSlots][MaxBlockModeSlots]transform.Type, x4, y4, w4, h4 int, t transform.Type) {
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
