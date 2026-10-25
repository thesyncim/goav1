package tile

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// coefBlock is the per-block binding dav1d's recon_b_intra/recon_b_inter and
// read_coef_blocks (src/recon_tmpl.c) read from Av1Block, t and f while they
// walk a block's transform blocks: everything decode_coefs needs that is
// constant across the block.
type coefBlock struct {
	s       *DecodeState
	cdfs    *CoeffCDFs
	txCDFs  *TransformTypeCDFs
	ctx     *CoeffEntropyContext
	scratch *BlockCoeffScratch
	dequant *CoeffDequantTarget
	inter   *InterCoeffTransformSelector

	intra      bool
	reduced    bool
	qZero      bool
	lumaMode   IntraMode
	chromaMode ChromaIntraMode

	// bx4/by4 and w4/h4 are the block's superblock-local luma origin and its
	// frame-clipped size in 4x4 units (dav1d's t->bx & 31 and w4/h4).
	bx4, by4  int
	w4, h4    int
	// cx4/cy4 are the chroma origin (bx4 >> ss_hor, by4 >> ss_ver) and
	// ssX/ssY the chroma subsampling shifts.
	cx4, cy4  int
	ssX, ssY  uint
	blockDims [3]BlockDimensions
	planeBlk  [3]BlockSize

	stats BlockCoeffResult
}

// readCoefBlocksSupported reports whether the dav1d-shaped block walk can take
// this block: the fused dequantized reconstruction target is bound and the
// transform type is read by one of the tile-owned selectors.
func readCoefBlocksSupported(req *BlockCoeffRequest) bool {
	if req.Dequant == nil || !req.SkipAllZeroCoeffClear || coeffTraceEnabled || req.Transform.Lossless {
		return false
	}
	switch req.TransformSelect.(type) {
	case *IntraCoeffTransformSelector, *InterCoeffTransformSelector:
		return true
	}
	return false
}

// readCoefBlocks is the transform-block walk of dav1d's recon_b_intra and
// recon_b_inter (src/recon_tmpl.c): per 64x64 unit, every luma TXB (a grid of
// b->tx for intra, read_coef_tree over max_ytx and tx_split for inter), then
// each chroma plane's grid of b->uvtx. Each TXB runs decodeCoefsTXB and then
// the reconstruction visitor.
func readCoefBlocks[T BlockLoopCoeffController](s *DecodeState, cdfs BlockCoeffCDFs, coeffCtx *CoeffEntropyContext, scratch *BlockCoeffScratch, req *BlockCoeffRequest, tree TransformTreeResult, chromaPrep *[2]chromaCoeffPlanePrep, hasChroma bool, loopVisit *BlockLoopVisit, coeffController T, result *BlockCoeffResult) error {
	tr := &req.Transform
	// Filled in place: a composite literal of this addressed struct was
	// built in a temporary and block-copied with wide loads over the
	// narrower field stores, which cannot store-forward.
	var b coefBlock
	b.s, b.cdfs, b.ctx = s, cdfs.Coeff, coeffCtx
	b.scratch, b.dequant = scratch, req.Dequant
	b.bx4 = int(tr.X4)
	b.by4 = int(tr.Y4)
	b.w4 = int(tr.VisibleW4)
	b.h4 = int(tr.VisibleH4)
	//
	//
	switch sel := req.TransformSelect.(type) {
	case *IntraCoeffTransformSelector:
		b.intra = true
		b.txCDFs = sel.CDFs
		b.reduced = sel.ReducedTXSet
		b.qZero = sel.QIndexKnown && sel.QIndex == 0
		b.lumaMode = sel.LumaMode
		b.chromaMode = sel.ChromaMode
	case *InterCoeffTransformSelector:
		b.inter = sel
		b.txCDFs = sel.CDFs
		b.reduced = sel.ReducedTXSet
		b.qZero = sel.QIndexKnown && sel.QIndex == 0
	}
	lumaDims, ok := tr.Size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	b.blockDims[0] = lumaDims
	b.planeBlk[0] = tr.Size
	if hasChroma {
		for pl := range 2 {
			prep := &chromaPrep[pl]
			if !prep.HasChroma {
				continue
			}
			dims, ok := prep.PlaneBlock.Dimensions()
			if !ok {
				return ErrInvalidDecodeState
			}
			b.blockDims[pl+1] = dims
			b.planeBlk[pl+1] = prep.PlaneBlock
		}
	}
	tDims, ok := tree.Y.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	txW, txH := int(tDims.W4), int(tDims.H4)
	ssX := int(boolToShift(tr.Color.SubsamplingX))
	ssY := int(boolToShift(tr.Color.SubsamplingY))
	b.ssX, b.ssY = uint(ssX), uint(ssY)
	var uvW, uvH int
	hasChroma = hasChroma && chromaPrep[0].HasChroma
	if hasChroma {
		uvW, uvH = int(chromaPrep[0].UVDims.W4), int(chromaPrep[0].UVDims.H4)
		b.cx4, b.cy4 = int(chromaPrep[0].X4), int(chromaPrep[0].Y4)
	}

	b.stats = *result
	err := readCoefBlockUnits(&b, coeffController, loopVisit, tree, chromaPrep, hasChroma, txW, txH, uvW, uvH, ssX, ssY)
	*result = b.stats
	return err
}

// readCoefBlockUnits is the per-64x64-unit loop of dav1d's recon_b_intra /
// recon_b_inter: the unit's luma TXBs, then each chroma plane's.
func readCoefBlockUnits[T BlockLoopCoeffController](b *coefBlock, coeffController T, loopVisit *BlockLoopVisit, tree TransformTreeResult, chromaPrep *[2]chromaCoeffPlanePrep, hasChroma bool, txW, txH, uvW, uvH, ssX, ssY int) error {
	txWLog2, txHLog2 := uint(bits.TrailingZeros(uint(txW))), uint(bits.TrailingZeros(uint(txH)))
	for initY := 0; initY < b.h4; initY += 16 {
		subH4 := min(b.h4, initY+16)
		for initX := 0; initX < b.w4; initX += 16 {
			subW4 := min(b.w4, initX+16)
			for y := initY; y < subH4; y += txH {
				for x := initX; x < subW4; x += txW {
					var err error
					if tree.Variable {
						err = readCoefTree(b, coeffController, loopVisit, tree.Y, 0, &tree.Split, x>>txWLog2, y>>txHLog2, b.bx4+x, b.by4+y)
					} else {
						err = decodeCoefsTXB(b, coeffController, loopVisit, 0, tree.Y, b.bx4+x, b.by4+y, min(txW, b.w4-x), min(txH, b.h4-y))
					}
					if err != nil {
						return err
					}
				}
			}
			if !hasChroma {
				continue
			}
			for pl := range 2 {
				prep := &chromaPrep[pl]
				if !prep.HasChroma {
					continue
				}
				cw4, ch4 := int(prep.VisibleW4), int(prep.VisibleH4)
				subCH4 := min(ch4, (initY+16)>>ssY)
				subCW4 := min(cw4, (initX+16)>>ssX)
				for y := initY >> ssY; y < subCH4; y += uvH {
					for x := initX >> ssX; x < subCW4; x += uvW {
						if err := decodeCoefsTXB(b, coeffController, loopVisit, uint8(pl+1), tree.UV, int(prep.X4)+x, int(prep.Y4)+y, min(uvW, cw4-x), min(uvH, ch4-y)); err != nil {
							return err
						}
					}
				}
			}
		}
	}
	return nil
}

// readCoefTree is dav1d's read_coef_tree (src/recon_tmpl.c): it follows the
// var-tx split masks down to the coded luma transform blocks.
func readCoefTree[T BlockLoopCoeffController](b *coefBlock, coeffController T, loopVisit *BlockLoopVisit, size TransformSize, depth int, split *[2]uint16, xOff, yOff, x4, y4 int) error {
	dims, ok := size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	txW, txH := int(dims.W4), int(dims.H4)
	right, bottom := b.bx4+b.w4, b.by4+b.h4
	if depth < 2 && yOff < 4 && split[depth]&(1<<(yOff*4+xOff)) != 0 && size > TransformSize4x4 {
		sub := dims.Sub
		subDims, ok := sub.Dimensions()
		if !ok {
			return ErrInvalidDecodeState
		}
		txsw, txsh := int(subDims.W4), int(subDims.H4)
		if err := readCoefTree(b, coeffController, loopVisit, sub, depth+1, split, xOff*2, yOff*2, x4, y4); err != nil {
			return err
		}
		if txW >= txH && x4+txsw < right {
			if err := readCoefTree(b, coeffController, loopVisit, sub, depth+1, split, xOff*2+1, yOff*2, x4+txsw, y4); err != nil {
				return err
			}
		}
		if txH >= txW && y4+txsh < bottom {
			if err := readCoefTree(b, coeffController, loopVisit, sub, depth+1, split, xOff*2, yOff*2+1, x4, y4+txsh); err != nil {
				return err
			}
			if txW >= txH && x4+txsw < right {
				if err := readCoefTree(b, coeffController, loopVisit, sub, depth+1, split, xOff*2+1, yOff*2+1, x4+txsw, y4+txsh); err != nil {
					return err
				}
			}
		}
		return nil
	}
	return decodeCoefsTXB(b, coeffController, loopVisit, 0, size, x4, y4, min(txW, right-x4), min(txH, bottom-y4))
}

// decodeCoefsTXB is one transform block of dav1d's recon_b_*: decode_coefs
// (skip, transform type, tokens), the a/l coefficient-context and txtp_map
// updates, and the reconstruction visit. visW/visH are the frame-clipped
// extent in 4x4 units.
func decodeCoefsTXB[T BlockLoopCoeffController](b *coefBlock, coeffController T, loopVisit *BlockLoopVisit, plane uint8, size TransformSize, x4, y4, visW, visH int) error {
	s := b.s
	resetCoeffDequantTXB(b.dequant)
	txDims, ok := size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	ctxReq := CoeffContextRequest{
		Plane:      plane,
		PlaneBlock: b.planeBlk[plane],
		Size:       size,
		X4:         uint8(x4),
		Y4:         uint8(y4),
		VisibleW4:  uint8(visW),
		VisibleH4:  uint8(visH),
	}
	txbCtx := b.ctx.txbContextTrusted(ctxReq, txDims, b.blockDims[plane])
	geo := &coeffGeometryTable[size]
	if !geo.valid || txbCtx.TXBSkipContext >= TXBSkipContexts {
		return ErrInvalidDecodeState
	}
	coeffPlane := CoeffPlaneY
	if plane != 0 {
		coeffPlane = CoeffPlaneUV
	}

	// does this block have any non-zero coefficients
	allSkip := s.Reader.ReadBinaryCDFUnchecked(&b.cdfs.TXBSkip[geo.txCtx][txbCtx.TXBSkipContext]) != 0
	txtp := transform.TypeDCTDCT
	if !allSkip {
		var err error
		if txtp, err = b.readTxtp(plane, size, x4, y4); err != nil {
			return err
		}
	}
	if plane == 0 && b.inter != nil {
		// dav1d's decode_coefs: dav1d_memset_pow2[t_dim->lw] of txtp into
		// t->scratch.txtp_map for each of the TXB's rows. The selector's Type
		// grid is that map; this walk reads back only what the block's own luma
		// TXBs wrote, so it skips the generation tags the legacy reader uses.
		w4, h4 := int(txDims.W4), int(txDims.H4)
		if x4+w4 > MaxBlockModeSlots || y4+h4 > MaxBlockModeSlots {
			return ErrInvalidDecodeState
		}
		m := &b.inter.Map.Type
		for y := y4; y < y4+h4; y++ {
			row := m[y][x4 : x4+w4]
			for i := range row {
				row[i] = txtp
			}
		}
	}

	var result TXBDecodeResult
	stats := &b.stats.Luma
	if plane != 0 {
		stats = &b.stats.Chroma[plane-1]
	}
	stats.TXBs++
	if allSkip {
		result.AllZero = true
		stats.AllZero++
		b.ctx.setTXBContextKnown(ctxReq, txDims, visW, visH, 0)
	} else {
		class, err := txtp.Class()
		if err != nil {
			return ErrInvalidDecodeState
		}
		target, txScale, dqMax := prepareCoeffDequantFactors(b.dequant, plane, geo)
		var cf *[1024]int32
		cfZeroed := false
		if target != nil {
			if target.Coef != nil {
				cf, cfZeroed = target.Coef.take(), true
			} else if target.Arena != nil {
				cf = target.arenaSlot()
			} else if len(target.Dst) >= 1024 {
				cf = (*[1024]int32)(target.Dst)
			}
		}
		if cf == nil {
			if result, err = b.readQuantizedTXB(ctxReq, size, coeffPlane, class, txbCtx, geo); err != nil {
				return err
			}
		} else {
			factors := &target.Factors[plane]
			c := s.Reader.Cursor()
			r := decodeCoefTokens(&c, b.cdfs, size, geo, coeffPlane, txtp, class, txbCtx.DCSignContext, cf, cfZeroed, &b.scratch.Coeff.CoefLevels, coefDequant{
				dc:    uint32(factors[0]),
				ac:    uint32(factors[1]),
				shift: txScale,
				cfMax: uint32(dqMax),
			})
			c.CommitStateTo(&s.Reader)
			target.ActiveRows = r.activeRows
			target.ActiveCols = r.activeCols
			target.TxScale = txScale
			target.Ready = true
			result = TXBDecodeResult{EOB: uint16(r.eob + 1), CulLevel: r.culLevel}
			if target.Coef != nil {
				target.Coef.used(r.activeRows, r.activeCols, int(geo.scanHeight))
				target.Dst = target.Coef.CF[:geo.maxEOB:geo.maxEOB]
			} else if target.Arena != nil {
				target.commitArenaSlot(int(geo.maxEOB))
			}
		}
		stats.NonZero++
		stats.EOBTotal += uint32(result.EOB)
		b.ctx.setTXBContextKnown(ctxReq, txDims, visW, visH, result.CulLevel)
	}

	blk := &b.scratch.block
	*blk = BlockCoeffBlock{
		Plane:     plane,
		Block:     TransformBlock{X4: uint8(x4), Y4: uint8(y4), Size: size, VisibleW4: uint8(visW), VisibleH4: uint8(visH)},
		Transform: txtp,
		Result:    result,
	}
	if !result.AllZero && !b.dequant.Ready {
		blk.Coeffs, blk.Scan, _, _ = b.scratch.Coeff.coeffBuffersWithGeo(size, mustClass(txtp), geo)
	}
	return coeffController.VisitBlockCoeffPtr(loopVisit, blk)
}

func mustClass(t transform.Type) transform.Class {
	c, _ := t.Class()
	return c
}

// readTxtp is the transform-type section of dav1d's decode_coefs for a
// non-skipped TXB: chroma types are derived (intra from the UV mode, inter
// from the co-located luma type in txtp_map), luma types are read from the
// per-size set unless qidx is zero.
func (b *coefBlock) readTxtp(plane uint8, size TransformSize, x4, y4 int) (transform.Type, error) {
	if plane != 0 {
		if b.intra {
			return IntraChromaTransformType(b.chromaMode, size, b.reduced)
		}
		// dav1d's recon_b_inter: txtp_map[(by4 + (y << ss_ver)) * 32 +
		// bx4 + (x << ss_hor)], with x/y the TXB offset in the chroma block.
		lx := b.bx4 + (x4-b.cx4)<<b.ssX
		ly := b.by4 + (y4-b.cy4)<<b.ssY
		if uint(lx) >= MaxBlockModeSlots || uint(ly) >= MaxBlockModeSlots {
			return 0, ErrInvalidDecodeState
		}
		luma := b.inter.Map.Type[ly][lx]
		return uvInterTransformType(size, luma)
	}
	if b.qZero {
		return transform.TypeDCTDCT, nil
	}
	inter := 0
	if !b.intra {
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
	if b.intra {
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
func (b *coefBlock) readQuantizedTXB(ctxReq CoeffContextRequest, size TransformSize, plane CoeffPlaneType, class transform.Class, txbCtx TXBContext, geo *coeffGeometry) (TXBDecodeResult, error) {
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

// arenaSlot returns the next TXB slot of the frame-thread cf arena, dav1d's
// ts->frame_thread[p].cf. The arena grows only while warming up; slots handed
// out earlier keep their backing array, so published slices stay valid.
func (t *CoeffDequantTarget) arenaSlot() *[1024]int32 {
	a := *t.Arena
	off := len(a)
	if cap(a)-off < 1024 {
		a = growCoefArena(a)
		*t.Arena = a
	}
	return (*[1024]int32)(a[off : off+1024])
}

// commitArenaSlot keeps the scan area of the slot just decoded and points Dst
// at it, as dav1d advances ts->frame_thread[p].cf past each TXB.
func (t *CoeffDequantTarget) commitArenaSlot(area int) {
	a := *t.Arena
	off := len(a)
	*t.Arena = a[:off+area]
	t.Dst = a[off : off+area : off+area]
}

// growCoefArena extends the arena's capacity by at least one TXB slot while
// keeping its length. It runs only while the arena warms up.
//
//go:noinline
func growCoefArena(a []int32) []int32 {
	return append(a[:cap(a)], make([]int32, 1024+cap(a)/2)...)[:len(a)]
}

// CoefBuffer is dav1d's t->cf: the coefficient buffer the synchronous
// reconstruction reads right after each TXB is decoded. dav1d's inverse
// transforms zero the coefficients they consumed so decode_coefs can rely on
// a zero buffer; here the next take clears the region the last TXB could have
// left nonzero, its activeCols columns times activeRows rows.
type CoefBuffer struct {
	CF                    [1024]int32
	rows, cols, colStride uint8
}

func (b *CoefBuffer) take() *[1024]int32 {
	// One contiguous clear through the last active row of the last active
	// column; per-column clears of the active rows measured slower.
	if n := (int(b.cols)-1)*int(b.colStride) + int(b.rows); n > 0 {
		clear(b.CF[:n])
	}
	b.rows, b.cols = 0, 0
	return &b.CF
}

func (b *CoefBuffer) used(rows, cols, stride int) {
	b.rows, b.cols, b.colStride = uint8(rows), uint8(cols), uint8(stride)
}
