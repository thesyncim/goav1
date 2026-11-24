package threading

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// ReconB is the block loop's per-block hand-off, dav1d's
// f->bd_fn.recon_b_intra / recon_b_inter (src/recon_tmpl.c) on the fused
// single-thread path and read_coef_blocks on the deferred one: the block's
// prediction (or its deferred-replay record), then per 64x64 unit every luma
// transform block (a grid of b->tx for intra, read_coef_tree over max_ytx and
// tx_split for inter) followed by each chroma plane's grid of b->uvtx. Each
// transform block runs decode_coefs and then its prediction and inverse
// transform, or records its cbi entry for reconBDeferred. A skip block codes
// no coefficients, so the deferred pass reads none (read_coef_blocks).
func (c *frameWorkTileResidualLoopController) ReconB(visit *tile.BlockLoopVisit, b *tile.CoefBlock) error {
	if err := c.reconBBegin(visit); err != nil {
		return err
	}
	if b.Skip && (!b.Intra || c.deferReconstruction) {
		return nil
	}
	var dq *[3]tile.CoefDequant
	if !b.Skip {
		if dq = c.blockDequant(visit.SegmentID); dq == nil {
			return ErrInvalidBatch
		}
	}
	tree := &b.Tree
	yDims, ok := tree.Y.Dimensions()
	if !ok {
		return ErrInvalidBatch
	}
	txW, txH := int(yDims.W4), int(yDims.H4)
	var uvW, uvH int
	if b.HasChroma {
		uvDims, ok := tree.UV.Dimensions()
		if !ok {
			return ErrInvalidBatch
		}
		uvW, uvH = int(uvDims.W4), int(uvDims.H4)
	}
	variable := tree.Variable && !b.Skip
	txWLog2, txHLog2 := uint(bits.TrailingZeros(uint(txW))), uint(bits.TrailingZeros(uint(txH)))
	for initY := 0; initY < b.H4; initY += 16 {
		subH4 := min(b.H4, initY+16)
		for initX := 0; initX < b.W4; initX += 16 {
			subW4 := min(b.W4, initX+16)
			for y := initY; y < subH4; y += txH {
				for x := initX; x < subW4; x += txW {
					var err error
					if variable {
						err = c.reconCoefTree(visit, b, dq, tree.Y, 0, x>>txWLog2, y>>txHLog2, b.X4+x, b.Y4+y)
					} else {
						err = c.reconTXB(visit, b, dq, 0, tree.Y, b.X4+x, b.Y4+y, min(txW, b.W4-x), min(txH, b.H4-y))
					}
					if err != nil {
						return err
					}
				}
			}
			if !b.HasChroma {
				continue
			}
			subCH4 := min(b.CH4, (initY+16)>>b.SSY)
			subCW4 := min(b.CW4, (initX+16)>>b.SSX)
			for pl := uint8(1); pl <= 2; pl++ {
				for y := initY >> b.SSY; y < subCH4; y += uvH {
					for x := initX >> b.SSX; x < subCW4; x += uvW {
						if err := c.reconTXB(visit, b, dq, pl, tree.UV, b.CX4+x, b.CY4+y, min(uvW, b.CW4-x), min(uvH, b.CH4-y)); err != nil {
							return err
						}
					}
				}
			}
		}
	}
	return nil
}

// reconCoefTree is dav1d's read_coef_tree (src/recon_tmpl.c): it follows the
// var-tx split masks down to the coded luma transform blocks.
func (c *frameWorkTileResidualLoopController) reconCoefTree(visit *tile.BlockLoopVisit, b *tile.CoefBlock, dq *[3]tile.CoefDequant, size tile.TransformSize, depth int, xOff, yOff, x4, y4 int) error {
	dims, ok := size.Dimensions()
	if !ok {
		return ErrInvalidBatch
	}
	txW, txH := int(dims.W4), int(dims.H4)
	right, bottom := b.X4+b.W4, b.Y4+b.H4
	split := &b.Tree.Split
	if depth < 2 && yOff < 4 && split[depth]&(1<<(yOff*4+xOff)) != 0 && size > tile.TransformSize4x4 {
		sub := dims.Sub
		subDims, ok := sub.Dimensions()
		if !ok {
			return ErrInvalidBatch
		}
		txsw, txsh := int(subDims.W4), int(subDims.H4)
		if err := c.reconCoefTree(visit, b, dq, sub, depth+1, xOff*2, yOff*2, x4, y4); err != nil {
			return err
		}
		if txW >= txH && x4+txsw < right {
			if err := c.reconCoefTree(visit, b, dq, sub, depth+1, xOff*2+1, yOff*2, x4+txsw, y4); err != nil {
				return err
			}
		}
		if txH >= txW && y4+txsh < bottom {
			if err := c.reconCoefTree(visit, b, dq, sub, depth+1, xOff*2, yOff*2+1, x4, y4+txsh); err != nil {
				return err
			}
			if txW >= txH && x4+txsw < right {
				if err := c.reconCoefTree(visit, b, dq, sub, depth+1, xOff*2+1, yOff*2+1, x4+txsw, y4+txsh); err != nil {
					return err
				}
			}
		}
		return nil
	}
	return c.reconTXB(visit, b, dq, 0, size, x4, y4, min(txW, right-x4), min(txH, bottom-y4))
}

// reconTXB is one transform block of recon_b_* / read_coef_blocks:
// decode_coefs into t->cf (fused) or the frame-thread cf arena (deferred), then
// the TXB's cbi entry, or its reconstruction. An inter TXB is read_coef_tree's
// leaf: nothing for an all-zero TXB, else the inverse transform add straight
// into the bound plane window. An intra TXB is recon_b_intra's prediction
// followed by the add. visW/visH are the frame-clipped extent in 4x4 units.
func (c *frameWorkTileResidualLoopController) reconTXB(visit *tile.BlockLoopVisit, b *tile.CoefBlock, dq *[3]tile.CoefDequant, plane uint8, size tile.TransformSize, x4, y4, visW, visH int) error {
	s := c.scratch
	var cf *[1024]int32
	cfOff := 0
	var r tile.CoefResult
	r.Txtp = transform.TypeDCTDCT
	coded := false
	if !b.Skip {
		// Both buffers are zero where decode_coefs writes (dav1d's itx clears).
		if c.deferReconstruction {
			cfOff = len(s.cfArena)
			cf = s.cfArenaSlot()
		} else {
			cf = s.coefBuf.take()
		}
		// dav1d's read_coef_tree / read_coef_blocks call decode_coefs
		// directly, the txtp_map lookup and fill around it.
		sz, ok := tile.CoefSizeOf(size)
		if !ok || plane > 2 {
			return tile.ErrInvalidDecodeState
		}
		tw4, th4 := int(sz.W4), int(sz.H4)
		if uint(x4)+uint(tw4) > tile.MaxBlockModeSlots || uint(y4)+uint(th4) > tile.MaxBlockModeSlots {
			return tile.ErrInvalidDecodeState
		}
		// recon_b_inter's chroma txtp: txtp_map[(by4 + (y << ss_ver)) * 32 +
		// bx4 + (x << ss_hor)], with x/y the TXB offset in the chroma block.
		m := b.TxtpMap()
		ytxtp := uint(0)
		if plane != 0 && m != nil {
			lx := uint(b.X4 + (x4-b.CX4)<<b.SSX)
			ly := uint(b.Y4 + (y4-b.CY4)<<b.SSY)
			if lx >= tile.MaxBlockModeSlots || ly >= tile.MaxBlockModeSlots {
				return tile.ErrInvalidDecodeState
			}
			ytxtp = uint(m[ly][lx])
		}
		rd, st := b.CoefEntropy()
		eob, txtp, ext := rd.DecodeCoefs(st, uint(plane), sz, cf, &dq[plane], uint(x4), uint(y4), uint(visW), uint(visH), ytxtp)
		r.Txtp = transform.Type(txtp)
		if plane == 0 && m != nil {
			tile.FillTxtpMap(m, x4, y4, tw4, th4, r.Txtp)
		}
		c.stats.TXBs++
		if eob < 0 {
			r.TXB.AllZero = true
			c.stats.AllZero++
		} else {
			r.TXB.EOB = uint16(eob + 1)
			r.ActiveRows, r.ActiveCols = uint8(ext), uint8(ext>>8)
			coded = true
			c.stats.NonZero++
			c.stats.EOBTotal += int32(r.TXB.EOB)
		}
	}
	rs := c.reconHot
	if !c.deferReconstruction && rs == nil {
		rs = c.fusedReconState()
	}
	if !coded {
		if c.deferReconstruction {
			if s.cfOverflow {
				return ErrInvalidBatch
			}
			s.reconCBI = append(s.reconCBI, frameWorkReconCBI{txtp: r.Txtp})
			return nil
		}
		if !b.Intra {
			rs.stats.Residuals++
			return nil
		}
		var blk tile.BlockCoeffBlock
		fillTXBRecord(&blk, plane, size, x4, y4, visW, visH, r.Txtp)
		blk.Result.AllZero = true
		return rs.reconstructTXBWithDequantized(visit, &blk, nil, 0, 0, false)
	}
	meta, ok := frameWorkBlockCoeffTransformMeta(size)
	if !ok {
		return ErrInvalidBatch
	}
	area := int(meta.scanSize.Width) * int(meta.scanSize.Height)
	if c.deferReconstruction {
		// dav1d's read_coef_blocks: the cbi entry, and the coefficients keep
		// their arena slot.
		if s.cfOverflow {
			return ErrInvalidBatch
		}
		s.cfArena = s.cfArena[:cfOff+area]
		s.reconCBI = append(s.reconCBI, frameWorkReconCBI{eob: r.TXB.EOB, txtp: r.Txtp, activeRows: r.ActiveRows, activeCols: r.ActiveCols})
		return nil
	}
	s.coefBuf.used(r.ActiveRows, r.ActiveCols, meta.scanSize.Height)
	dcOnly := r.Txtp == transform.TypeDCTDCT && r.TXB.EOB == 1
	coefs := cf[:area:area]
	if !b.Intra {
		done, err := rs.addResidualInWindow(&visit.Block, plane, uint8(x4), uint8(y4), meta, r.Txtp, r.TXB.EOB, coefs, int(r.ActiveRows), int(r.ActiveCols), dcOnly)
		if done || err != nil {
			return err
		}
	}
	// Field stores, not a composite literal: the literal is built in a stack
	// temporary of byte stores and copied with word loads, a store-forwarding
	// stall per TXB.
	var blk tile.BlockCoeffBlock
	fillTXBRecord(&blk, plane, size, x4, y4, visW, visH, r.Txtp)
	blk.Result = r.TXB
	return rs.reconstructTXBWithDequantized(visit, &blk, coefs, int(r.ActiveRows), int(r.ActiveCols), dcOnly)
}

// itxfmAdd8 is dav1d's dsp->itx.itxfm_add[tx][txtp] table for the 8-bit fused
// kernels: the inverse transform of the coefficients in cf (column stride
// scanH) added to the full tx-size block at dst. It takes every type but IDTX
// and WHT_WHT, which the caller excludes, and reports false, writing nothing,
// for a shape or extent no fused kernel takes. The kernel choice is the one
// reconstruct.ReconstructPlaneBlockVisibleTrustedAtFromDequantized makes.
func itxfmAdd8(dst []byte, stride int, cf []int32, tx tile.TransformSize, txtp transform.Type, scanH, rows, cols int) bool {
	dct := txtp == transform.TypeDCTDCT
	switch tx {
	case tile.TransformSize4x4:
		return transform.InverseAdd4x4(dst, stride, dst, stride, cf, scanH, txtp, rows)
	case tile.TransformSize8x8:
		if dct {
			return transform.InverseAddDCT8x8(dst, stride, dst, stride, cf, scanH, rows, cols)
		}
		return transform.InverseAdd8x8Hybrid(dst, stride, dst, stride, cf, scanH, txtp, rows, cols)
	case tile.TransformSize16x16:
		if dct {
			return transform.InverseAddDCT16x16(dst, stride, dst, stride, cf, scanH, rows, cols)
		}
		return transform.InverseAdd16x16Hybrid(dst, stride, dst, stride, cf, scanH, txtp, rows, cols)
	case tile.TransformSize32x32:
		if !dct {
			return false
		}
		if rows <= 8 && cols <= 8 {
			return transform.InverseAddDCT32x32Low8(dst, stride, dst, stride, cf, scanH)
		}
		if rows <= 16 && cols <= 16 && transform.InverseAddDCT32x32Low16(dst, stride, dst, stride, cf, scanH, rows) {
			return true
		}
		return transform.InverseAddDCT32x32Full(dst, stride, dst, stride, cf, scanH, rows, cols)
	case tile.TransformSize64x64:
		return dct && transform.InverseAddDCT64x64(dst, stride, dst, stride, cf, scanH, rows, cols)
	case tile.TransformSize4x8:
		return transform.InverseAdd4x8(dst, stride, dst, stride, cf, scanH, txtp, rows, cols)
	case tile.TransformSize8x4:
		return transform.InverseAdd8x4(dst, stride, dst, stride, cf, scanH, txtp, rows, cols)
	case tile.TransformSize4x16:
		return transform.InverseAdd4x16(dst, stride, dst, stride, cf, scanH, txtp, rows, cols)
	case tile.TransformSize16x4:
		return transform.InverseAdd16x4(dst, stride, dst, stride, cf, scanH, txtp, rows, cols)
	case tile.TransformSize8x16:
		return transform.InverseAdd8x16(dst, stride, dst, stride, cf, scanH, txtp, rows, cols)
	case tile.TransformSize16x8:
		return transform.InverseAdd16x8(dst, stride, dst, stride, cf, scanH, txtp, rows, cols)
	case tile.TransformSize16x32:
		return dct && transform.InverseAddDCT16x32(dst, stride, dst, stride, cf, scanH, rows, cols)
	case tile.TransformSize32x16:
		return dct && transform.InverseAddDCT32x16(dst, stride, dst, stride, cf, scanH, rows, cols)
	}
	return false
}

func fillTXBRecord(blk *tile.BlockCoeffBlock, plane uint8, size tile.TransformSize, x4, y4, visW, visH int, txtp transform.Type) {
	blk.Plane = plane
	blk.Block.X4, blk.Block.Y4, blk.Block.Size = uint8(x4), uint8(y4), size
	blk.Block.VisibleW4, blk.Block.VisibleH4 = uint8(visW), uint8(visH)
	blk.Transform = txtp
}

// reconBDeferred is the frame-thread pass-2 decode_b of a buffered block,
// dav1d's recon_b_intra / recon_b_inter reading frame_thread.cbi and .cf: the
// block's prediction, then the same transform-block walk ReconB parsed, each
// TXB taking the next cbi entry and, when coded, the next cf scan area.
func (s *frameWorkReconState) reconBDeferred(visit *tile.BlockLoopVisit, cur *frameWorkReconCursor) error {
	if err := s.predictBlockBegin(visit); err != nil {
		return err
	}
	intra := visit.Prediction.Valid && visit.Prediction.Intra
	skip := visit.Prefix.SkipTransform
	if skip && !intra {
		return nil
	}
	vb := &visit.Block
	tree := &visit.Coefficients.Tree
	yDims, ok := tree.Y.Dimensions()
	if !ok {
		return ErrInvalidBatch
	}
	x4, y4, w4, h4 := int(vb.X4), int(vb.Y4), int(vb.VisibleW4), int(vb.VisibleH4)
	txW, txH := int(yDims.W4), int(yDims.H4)
	color := &s.batch.Sequence.ColorConfig
	hasChroma := tree.HasUV && tile.HasChromaBlockAt(vb.Size, x4, y4, *color)
	var uvW, uvH, cx4, cy4, cw4, ch4 int
	var ssX, ssY uint
	if hasChroma {
		uvDims, ok := tree.UV.Dimensions()
		if !ok {
			return ErrInvalidBatch
		}
		uvW, uvH = int(uvDims.W4), int(uvDims.H4)
		ssX, ssY = frameWorkSubsampleShift(color.SubsamplingX), frameWorkSubsampleShift(color.SubsamplingY)
		cx4, cy4 = x4>>ssX, y4>>ssY
		cw4 = ((x4 + w4 + int(ssX)) >> ssX) - cx4
		ch4 = ((y4 + h4 + int(ssY)) >> ssY) - cy4
	}
	variable := tree.Variable && !skip
	txWLog2, txHLog2 := uint(bits.TrailingZeros(uint(txW))), uint(bits.TrailingZeros(uint(txH)))
	for initY := 0; initY < h4; initY += 16 {
		subH4 := min(h4, initY+16)
		for initX := 0; initX < w4; initX += 16 {
			subW4 := min(w4, initX+16)
			for y := initY; y < subH4; y += txH {
				for x := initX; x < subW4; x += txW {
					var err error
					if variable {
						err = s.reconCoefTreeDeferred(visit, cur, tree.Y, 0, x>>txWLog2, y>>txHLog2, x4+x, y4+y)
					} else {
						err = s.reconTXBDeferred(visit, cur, skip, 0, tree.Y, x4+x, y4+y, min(txW, w4-x), min(txH, h4-y))
					}
					if err != nil {
						return err
					}
				}
			}
			if !hasChroma {
				continue
			}
			subCH4 := min(ch4, (initY+16)>>ssY)
			subCW4 := min(cw4, (initX+16)>>ssX)
			for pl := uint8(1); pl <= 2; pl++ {
				for y := initY >> ssY; y < subCH4; y += uvH {
					for x := initX >> ssX; x < subCW4; x += uvW {
						if err := s.reconTXBDeferred(visit, cur, skip, pl, tree.UV, cx4+x, cy4+y, min(uvW, cw4-x), min(uvH, ch4-y)); err != nil {
							return err
						}
					}
				}
			}
		}
	}
	return nil
}

// reconCoefTreeDeferred is read_coef_tree's pass-2 walk of reconCoefTree.
func (s *frameWorkReconState) reconCoefTreeDeferred(visit *tile.BlockLoopVisit, cur *frameWorkReconCursor, size tile.TransformSize, depth int, xOff, yOff, x4, y4 int) error {
	dims, ok := size.Dimensions()
	if !ok {
		return ErrInvalidBatch
	}
	txW, txH := int(dims.W4), int(dims.H4)
	vb := &visit.Block
	right, bottom := int(vb.X4)+int(vb.VisibleW4), int(vb.Y4)+int(vb.VisibleH4)
	split := &visit.Coefficients.Tree.Split
	if depth < 2 && yOff < 4 && split[depth]&(1<<(yOff*4+xOff)) != 0 && size > tile.TransformSize4x4 {
		sub := dims.Sub
		subDims, ok := sub.Dimensions()
		if !ok {
			return ErrInvalidBatch
		}
		txsw, txsh := int(subDims.W4), int(subDims.H4)
		if err := s.reconCoefTreeDeferred(visit, cur, sub, depth+1, xOff*2, yOff*2, x4, y4); err != nil {
			return err
		}
		if txW >= txH && x4+txsw < right {
			if err := s.reconCoefTreeDeferred(visit, cur, sub, depth+1, xOff*2+1, yOff*2, x4+txsw, y4); err != nil {
				return err
			}
		}
		if txH >= txW && y4+txsh < bottom {
			if err := s.reconCoefTreeDeferred(visit, cur, sub, depth+1, xOff*2, yOff*2+1, x4, y4+txsh); err != nil {
				return err
			}
			if txW >= txH && x4+txsw < right {
				if err := s.reconCoefTreeDeferred(visit, cur, sub, depth+1, xOff*2+1, yOff*2+1, x4+txsw, y4+txsh); err != nil {
					return err
				}
			}
		}
		return nil
	}
	return s.reconTXBDeferred(visit, cur, false, 0, size, x4, y4, min(txW, right-x4), min(txH, bottom-y4))
}

// reconTXBDeferred reconstructs one TXB of a buffered block: a skip block's
// TXBs are all zero and read no cbi entry.
func (s *frameWorkReconState) reconTXBDeferred(visit *tile.BlockLoopVisit, cur *frameWorkReconCursor, skip bool, plane uint8, size tile.TransformSize, x4, y4, visW, visH int) error {
	intra := visit.Prediction.Valid && visit.Prediction.Intra
	var blk tile.BlockCoeffBlock
	if skip {
		fillTXBRecord(&blk, plane, size, x4, y4, visW, visH, transform.TypeDCTDCT)
		blk.Result.AllZero = true
		return s.reconstructTXBWithDequantized(visit, &blk, nil, 0, 0, false)
	}
	if cur.cbiOff < 0 || cur.cbiOff >= len(cur.cbi) {
		return ErrInvalidBatch
	}
	e := cur.cbi[cur.cbiOff]
	cur.cbiOff++
	if e.eob == 0 {
		if !intra {
			s.stats.Residuals++
			return nil
		}
		fillTXBRecord(&blk, plane, size, x4, y4, visW, visH, e.txtp)
		blk.Result.AllZero = true
		return s.reconstructTXBWithDequantized(visit, &blk, nil, 0, 0, false)
	}
	meta, ok := frameWorkBlockCoeffTransformMeta(size)
	if !ok {
		return ErrInvalidBatch
	}
	off := cur.cfOff
	end := off + int(meta.scanSize.Width)*int(meta.scanSize.Height)
	if off < 0 || end > len(cur.cf) || e.activeRows == 0 || e.activeCols == 0 {
		return ErrInvalidBatch
	}
	cur.cfOff = end
	dcOnly := e.txtp == transform.TypeDCTDCT && e.eob == 1
	cf := cur.cf[off:end:end]
	done := false
	var err error
	if !intra {
		done, err = s.addResidualInWindow(&visit.Block, plane, uint8(x4), uint8(y4), meta, e.txtp, e.eob, cf, int(e.activeRows), int(e.activeCols), dcOnly)
	}
	if !done && err == nil {
		fillTXBRecord(&blk, plane, size, x4, y4, visW, visH, e.txtp)
		blk.Result.EOB = e.eob
		err = s.reconstructTXBWithDequantized(visit, &blk, cf, int(e.activeRows), int(e.activeCols), dcOnly)
	}
	// dav1d's inverse transforms clear the coefficients they read, so the
	// next frame's decode_coefs finds the arena zero: the nonzero
	// coefficients lie in the first activeCols columns of activeRows.
	clearCoefPrefix(cf, (int(e.activeCols)-1)*int(meta.scanSize.Height)+int(e.activeRows))
	return err
}

// clearCoefPrefix zeroes the first n coefficients of cf (at most len(cf)),
// a short prefix with inline stores rather than a memclr call.
func clearCoefPrefix(cf []int32, n int) {
	switch {
	case n <= 16 && len(cf) >= 16:
		*(*[16]int32)(cf) = [16]int32{}
	case n <= 64 && len(cf) >= 64:
		*(*[64]int32)(cf) = [64]int32{}
	default:
		clear(cf[:min(len(cf), n)])
	}
}

// blockDequant returns the block's dav1d ts->dq[seg_id] with cf_max and the
// planes' f->qm, resolved once per (qindex, segment) key, or nil when the
// key has no valid quantizer.
func (c *frameWorkTileResidualLoopController) blockDequant(segmentID uint8) *[3]tile.CoefDequant {
	key := frameWorkReconQuantKey(c.state.CurrentBaseQIdx, segmentID)
	if c.dqKey != key {
		c.dqKey = key
		c.dqOK = c.resolveDequant(segmentID)
	}
	if !c.dqOK {
		return nil
	}
	return &c.dq
}

func (c *frameWorkTileResidualLoopController) resolveDequant(segmentID uint8) bool {
	bitDepth := c.batch.Sequence.ColorConfig.BitDepth
	switch bitDepth {
	case 8, 10, 12:
	default:
		return false
	}
	_, dqMax := quantize.DequantCoefficientBoundsTrusted(bitDepth)
	qp := &c.batch.Quantization
	levels := [3]uint8{qp.QMatrixLevelY, qp.QMatrixLevelU, qp.QMatrixLevelV}
	for plane := range c.dq {
		q, lossless, err := c.batch.blockQuantizerForPlane(c.state.CurrentBaseQIdx, segmentID, quantize.Plane(plane))
		if err != nil || q.DC <= 0 || q.AC <= 0 {
			return false
		}
		c.dq[plane] = tile.CoefDequant{DC: uint32(q.DC), AC: uint32(q.AC), Max: uint32(dqMax)}
		if qp.UsingQMatrix && !lossless {
			c.dq[plane].QM = tile.QMatrices(levels[plane], plane != 0)
		}
	}
	return true
}

// coefBuffer is dav1d's t->cf: the coefficient buffer the fused reconstruction
// reads right after each TXB is decoded. dav1d's inverse transforms zero the
// coefficients they consumed so decode_coefs can rely on a zero buffer; here
// the next take clears the region the last TXB could have left nonzero, its
// activeCols columns times activeRows rows.
type coefBuffer struct {
	cf                    [1024]int32
	rows, cols, colStride uint8
}

func (b *coefBuffer) take() *[1024]int32 {
	// One contiguous clear through the last active row of the last active
	// column; per-column clears of the active rows measured slower.
	// A prefix of up to 64 coefficients (most TXBs: DC-only, 4x4, low-eob
	// 8x8) is cleared with inline stores, like dav1d's itx zeroing the
	// coefficients it read; only a longer one calls memclr.
	switch n := (int(b.cols)-1)*int(b.colStride) + int(b.rows); {
	case n <= 0:
	case n <= 16:
		*(*[16]int32)(b.cf[:]) = [16]int32{}
	case n <= 64:
		*(*[64]int32)(b.cf[:]) = [64]int32{}
	default:
		clear(b.cf[:n])
	}
	b.rows, b.cols = 0, 0
	return &b.cf
}

func (b *coefBuffer) used(rows, cols, stride uint8) {
	b.rows, b.cols, b.colStride = rows, cols, stride
}

// zeroCFArena makes the cf arena zero for the next job's decode_coefs. The
// lanes clear every coefficient they reconstruct, so after a tile whose lanes
// all completed (clean) it already is; otherwise the prefix earlier jobs used
// is cleared.
func (s *FrameWorkTileResidualScratch) zeroCFArena(clean bool) {
	if !clean {
		clear(s.cfArena[:min(s.cfDirty, cap(s.cfArena))])
	}
	s.cfDirty = 0
}

// cfArenaSlot returns the next TXB slot of the frame-thread cf arena, dav1d's
// ts->frame_thread[p].cf. reserveCFArena sized the arena for the whole job, as
// dav1d sizes frame_thread.cf per frame: it never grows while the lanes read
// and clear it. A TXB past the reservation gets the spare slot and sets
// cfOverflow, which fails the job.
func (s *FrameWorkTileResidualScratch) cfArenaSlot() *[1024]int32 {
	a := s.cfArena
	off := len(a)
	if cap(a)-off < 1024 {
		s.cfOverflow = true
		clear(s.cfSpare[:])
		return &s.cfSpare
	}
	return (*[1024]int32)(a[off : off+1024])
}
