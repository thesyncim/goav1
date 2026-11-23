package tile

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// MarkInterMotionAndFilters commits an inter block's references, motion and
// interpolation filters, after the caller has collected any sub8x8 chroma
// cells.
func (c *BlockModeContext) MarkInterMotionAndFilters(size BlockSize, x4, y4 int, result InterMotionResult, filters motion.InterpFilters, hasChroma bool) error {
	return c.markInterMotionAndFilters(size, x4, y4, result, filters, hasChroma, true, nil)
}

// markInterMotionAndFilters is decode_b's inter set_ctx. keepMotion false
// skips the neighbor-motion arrays and grids (the frame decoder keeps
// neighbor motion in the dav1d refmvs rows instead). blk, when set, adds the
// block's seg_pred/skip_mode/skip, its compound blend and inter-intra flag to
// the same stores.
func (c *BlockModeContext) markInterMotionAndFilters(size BlockSize, x4, y4 int, result InterMotionResult, filters motion.InterpFilters, hasChroma bool, keepMotion bool, blk *interBlockCommit) error {
	if !filters.X.Valid() || !filters.Y.Valid() {
		if err := c.MarkInterMotion(size, x4, y4, result, hasChroma); err != nil {
			return fmt.Errorf("mark inter motion: %w", err)
		}
		if err := c.MarkInterFilters(size, x4, y4, result.References, filters); err != nil {
			return fmt.Errorf("mark inter filters: %w", err)
		}
		return nil
	}
	if err := validateInterMotion(result); err != nil {
		return fmt.Errorf("mark inter motion: %w", err)
	}
	dims, err := c.validateInterContext(size, x4, y4, result.References)
	if err != nil {
		return fmt.Errorf("mark inter motion: %w", err)
	}
	compType := compInterTypeFor(result.References.Compound)
	interIntra := false
	if blk != nil {
		if blk.blendValid {
			compType = compInterTypeOfBlend(blk.blend)
		}
		interIntra = blk.interIntra
	}
	v := interEdgeValues(size, result.References, compType, filters, interIntra, hasChroma)
	if blk != nil {
		v.setPrefix(&blk.prefix)
	}
	c.setBlockEdges(x4, y4, int(dims.W4), int(dims.H4), &v)
	if !keepMotion {
		return nil
	}
	xEnd := x4 + int(dims.W4)
	yEnd := y4 + int(dims.H4)
	for x := x4; x < xEnd; x++ {
		c.AboveInterMotion[x] = result
		c.AboveMotionValid[x] = 1
	}
	for y := y4; y < yEnd; y++ {
		c.LeftInterMotion[y] = result
		c.LeftMotionValid[y] = 1
	}
	// Fill one row and replicate the same complete motion/filter state in
	// one grid walk, as dav1d's final block-context and ref-MV splats do.
	m0 := c.GridInterMotion[y4][x4:xEnd]
	v0 := c.GridMotionValid[y4][x4:xEnd]
	s0 := c.GridBlockSize[y4][x4:xEnd]
	z0 := c.GridBlockSizeVisited[y4][x4:xEnd]
	i0 := c.GridInterp[y4][x4:xEnd]
	f0 := c.GridInterpValid[y4][x4:xEnd]
	fillGridRow(m0, result)
	fillGridRow(v0, 1)
	fillGridRow(s0, size)
	fillGridRow(z0, 1)
	fillGridRow(i0, filters)
	fillGridRow(f0, 1)
	for y := y4 + 1; y < yEnd; y++ {
		copy(c.GridInterMotion[y][x4:xEnd], m0)
		copy(c.GridMotionValid[y][x4:xEnd], v0)
		copy(c.GridBlockSize[y][x4:xEnd], s0)
		copy(c.GridBlockSizeVisited[y][x4:xEnd], z0)
		copy(c.GridInterp[y][x4:xEnd], i0)
		copy(c.GridInterpValid[y][x4:xEnd], f0)
	}
	return nil
}

// interBlockCommit carries the decoded block state the decoder's inter
// set_ctx stores besides the references, motion and filters.
type interBlockCommit struct {
	prefix     BlockModeResult
	blend      CompoundBlendResult
	blendValid bool
	interIntra bool
}

// filterUnsetPair is the filter pair of a block whose filters are not known.
var filterUnsetPair = motion.InterpFilters{X: filterUnset, Y: filterUnset}

// compInterTypeFor is the comp_type of an inter block before its compound
// blend is read: a compound block averages (COMP_INTER_AVG, also skip mode).
func compInterTypeFor(compound bool) uint8 {
	if compound {
		return compInterAvg
	}
	return compInterNone
}

// compInterTypeOfBlend maps a decoded compound blend to dav1d's comp_type.
func compInterTypeOfBlend(b CompoundBlendResult) uint8 {
	switch {
	case b.CompoundGroupIndex != 0 && b.Type == CompoundTypeWedge:
		return compInterWedge
	case b.CompoundGroupIndex != 0:
		return compInterSeg
	case b.CompoundIndex != 0:
		return compInterAvg
	default:
		return compInterWeightedAvg
	}
}

// interEdgeValues are an inter block's set_ctx values.
func interEdgeValues(size BlockSize, refs InterReferencesResult, compType uint8, filters motion.InterpFilters, interIntra, hasChroma bool) blockEdgeValues {
	return blockEdgeValues{
		compType:   ctxSplat(compType),
		ref0:       ctxSplat(uint8(refs.Ref0)),
		ref1:       ctxSplat(uint8(refs.Ref1)),
		filter0:    ctxSplat(uint8(filters.Y)),
		filter1:    ctxSplat(uint8(filters.X)),
		blockSize:  ctxSplat(uint8(size)),
		interIntra: ctxSplat(boolByte(interIntra)),
		chroma:     hasChroma,
	}
}

// setPrefix adds the block's seg_pred/skip_mode/skip; p nil leaves them.
func (v *blockEdgeValues) setPrefix(p *BlockModeResult) {
	if p == nil {
		return
	}
	v.prefix = true
	v.segPred = ctxSplat(boolByte(p.SegmentPredicted))
	v.skipMode = ctxSplat(boolByte(p.SkipMode))
	v.skip = ctxSplat(boolByte(p.SkipTransform))
}

// setBlockEdges runs setBlockEdge on the above edge over the block's width
// and on the left edge over its height, decode_b's loop over t->a and t->l.
func (c *BlockModeContext) setBlockEdges(x4, y4, w4, h4 int, v *blockEdgeValues) {
	setBlockEdge(&c.A, &c.PalA, x4, w4, v)
	setBlockEdge(&c.L, &c.PalL, y4, h4, v)
}

// blockEdgeValues are the per-block values decode_b's set_ctx replicates
// over the block's span of one edge, each splat to eight bytes (the
// rep_macro pattern of case_set).
type blockEdgeValues struct {
	segPred, skipMode, skip  uint64
	intra, mode, compType    uint64
	ref0, ref1, filter0      uint64
	filter1, blockSize       uint64
	interIntra, palSz, palUV uint64
	uvMode                   uint64
	// prefix writes seg_pred/skip_mode/skip; chroma writes uvmode.
	prefix, chroma bool
}

// setBlockEdge is decode_b's case_set over one edge: n (1, 2, 4, 8, 16 or 32)
// entries from off of every BlockContext field, plus the edge's pal_sz_uv.
func setBlockEdge(e *BlockContext, p *edgePalette, off, n int, v *blockEdgeValues) {
	switch n {
	case 1:
		if v.prefix {
			e.SegPred[off] = uint8(v.segPred)
			e.SkipMode[off] = uint8(v.skipMode)
			e.Skip[off] = uint8(v.skip)
		}
		e.Intra[off] = uint8(v.intra)
		e.Mode[off] = IntraMode(v.mode)
		e.CompType[off] = uint8(v.compType)
		e.Ref[0][off] = ReferenceFrame(v.ref0)
		e.Ref[1][off] = ReferenceFrame(v.ref1)
		e.Filter[0][off] = motion.InterpFilter(v.filter0)
		e.Filter[1][off] = motion.InterpFilter(v.filter1)
		e.BlockSize[off] = BlockSize(v.blockSize)
		e.InterIntra[off] = uint8(v.interIntra)
		e.PalSz[off] = uint8(v.palSz)
		p.SzUV[off] = uint8(v.palUV)
		if v.chroma {
			e.UVMode[off] = ChromaIntraMode(v.uvMode)
		}
	case 2:
		if v.prefix {
			ctxPut2(e.SegPred[off:], v.segPred)
			ctxPut2(e.SkipMode[off:], v.skipMode)
			ctxPut2(e.Skip[off:], v.skip)
		}
		ctxPut2(e.Intra[off:], v.intra)
		ctxPut2(e.Mode[off:], v.mode)
		ctxPut2(e.CompType[off:], v.compType)
		ctxPut2(e.Ref[0][off:], v.ref0)
		ctxPut2(e.Ref[1][off:], v.ref1)
		ctxPut2(e.Filter[0][off:], v.filter0)
		ctxPut2(e.Filter[1][off:], v.filter1)
		ctxPut2(e.BlockSize[off:], v.blockSize)
		ctxPut2(e.InterIntra[off:], v.interIntra)
		ctxPut2(e.PalSz[off:], v.palSz)
		ctxPut2(p.SzUV[off:], v.palUV)
		if v.chroma {
			ctxPut2(e.UVMode[off:], v.uvMode)
		}
	case 4:
		if v.prefix {
			ctxPut4(e.SegPred[off:], v.segPred)
			ctxPut4(e.SkipMode[off:], v.skipMode)
			ctxPut4(e.Skip[off:], v.skip)
		}
		ctxPut4(e.Intra[off:], v.intra)
		ctxPut4(e.Mode[off:], v.mode)
		ctxPut4(e.CompType[off:], v.compType)
		ctxPut4(e.Ref[0][off:], v.ref0)
		ctxPut4(e.Ref[1][off:], v.ref1)
		ctxPut4(e.Filter[0][off:], v.filter0)
		ctxPut4(e.Filter[1][off:], v.filter1)
		ctxPut4(e.BlockSize[off:], v.blockSize)
		ctxPut4(e.InterIntra[off:], v.interIntra)
		ctxPut4(e.PalSz[off:], v.palSz)
		ctxPut4(p.SzUV[off:], v.palUV)
		if v.chroma {
			ctxPut4(e.UVMode[off:], v.uvMode)
		}
	default:
		for i := off; i < off+n; i += 8 {
			if v.prefix {
				ctxPut8(e.SegPred[i:], v.segPred)
				ctxPut8(e.SkipMode[i:], v.skipMode)
				ctxPut8(e.Skip[i:], v.skip)
			}
			ctxPut8(e.Intra[i:], v.intra)
			ctxPut8(e.Mode[i:], v.mode)
			ctxPut8(e.CompType[i:], v.compType)
			ctxPut8(e.Ref[0][i:], v.ref0)
			ctxPut8(e.Ref[1][i:], v.ref1)
			ctxPut8(e.Filter[0][i:], v.filter0)
			ctxPut8(e.Filter[1][i:], v.filter1)
			ctxPut8(e.BlockSize[i:], v.blockSize)
			ctxPut8(e.InterIntra[i:], v.interIntra)
			ctxPut8(e.PalSz[i:], v.palSz)
			ctxPut8(p.SzUV[i:], v.palUV)
			if v.chroma {
				ctxPut8(e.UVMode[i:], v.uvMode)
			}
		}
	}
}

// ctxSplat replicates v into every byte, the rep_macro pattern of case_set.
func ctxSplat(v uint8) uint64 { return uint64(v) * 0x0101010101010101 }

type ctxByte interface{ ~uint8 | ~int8 }

// ctxPut2, ctxPut4 and ctxPut8 store the low 2, 4 or 8 bytes of x to
// d[:2], d[:4] or d[:8], one store each.
func ctxPut2[T ctxByte](d []T, x uint64) {
	d = d[:2]
	d[0], d[1] = T(x), T(x>>8)
}

func ctxPut4[T ctxByte](d []T, x uint64) {
	d = d[:4]
	d[0], d[1], d[2], d[3] = T(x), T(x>>8), T(x>>16), T(x>>24)
}

func ctxPut8[T ctxByte](d []T, x uint64) {
	d = d[:8]
	d[0], d[1], d[2], d[3] = T(x), T(x>>8), T(x>>16), T(x>>24)
	d[4], d[5], d[6], d[7] = T(x>>32), T(x>>40), T(x>>48), T(x>>56)
}
