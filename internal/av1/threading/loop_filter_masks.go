package threading

import (
	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// FrameWorkLoopFilterMasks is caller-owned, reused frame-level storage for the
// dav1d-style per-superblock deblocking edge bitmasks (internal/av1/lfmask). It
// mirrors dav1d's f->lf.mask + f->lf.level: one FilterMask per 128x128
// superblock region (sb128w * sb128h of them) holds the compact per-4x4 edge
// bitmasks, and LevelCache is the frame-wide per-4x4 resolved level grid.
//
// The edge bitmasks are built during the tile-residual block walk in decode
// (Z) order (dav1d src/decode.c dav1d_create_lf_mask_{inter,intra}); the
// per-4x4 level cells are order-independent and are filled separately by the
// post-filter consumer. This handle carries only geometry the build needs; it
// is bound once per frame beside the loop-filter record map.
type FrameWorkLoopFilterMasks struct {
	Masks      []lfmask.FilterMask
	LevelCache [][4]uint8

	Cols   int
	Rows   int
	SB128W int
	SB128H int

	Layout    lfmask.Layout
	HasChroma bool
}

// FrameWorkLoopFilterMaskShape returns the 128x128 region grid dimensions for a
// frame-level MI grid of cols x rows luma 4x4 cells. sb128w/sb128h count the
// 128-pixel (32 MI) superblock regions dav1d indexes f->lf.mask by, regardless
// of the coded superblock size (64px superblocks still share a 128 region).
func FrameWorkLoopFilterMaskShape(cols, rows int) (sb128w, sb128h, masks int) {
	sb128w = (cols + 31) >> 5
	sb128h = (rows + 31) >> 5
	return sb128w, sb128h, sb128w * sb128h
}

// Valid reports whether the handle carries a bound mask grid.
func (h *FrameWorkLoopFilterMasks) Valid() bool {
	return h != nil && h.Masks != nil && h.SB128W > 0
}

// region returns the FilterMask covering the 128x128 superblock region that MI
// cell (bx, by) falls in.
func (h *FrameWorkLoopFilterMasks) region(bx, by int) *lfmask.FilterMask {
	return &h.Masks[(by>>5)*h.SB128W+(bx>>5)]
}

// frameWorkLoopFilterMaskScratch is the per-tile-job reusable state for the
// decode-time mask build: dav1d's decomp_tx scratch plus the carried
// above/left edge-context arrays (t->a->tx_lpf_* / t->l.tx_lpf_*). aY/aUV span
// the frame width (reset once at tile start, carried down superblock rows); lY
// /lUV span one 128-region column (reset at each superblock-row change).
type frameWorkLoopFilterMaskScratch struct {
	builder lfmask.Builder
	aY      []uint8
	aUV     []uint8
	lY      [32]uint8
	lUV     [32]uint8

	sbRow   int
	started bool
}

// reset re-initialises the above-context arrays for a new tile job. It mirrors
// dav1d reset_context filling tx_lpf_y with 2 and tx_lpf_uv with 1 at tile
// start; the left context is (re)initialised lazily per superblock row.
func (s *frameWorkLoopFilterMaskScratch) reset(h *FrameWorkLoopFilterMasks) {
	if cap(s.aY) < h.Cols {
		s.aY = make([]uint8, h.Cols)
	} else {
		s.aY = s.aY[:h.Cols]
	}
	for i := range s.aY {
		s.aY[i] = 2
	}
	if h.HasChroma {
		if cap(s.aUV) < h.Cols {
			s.aUV = make([]uint8, h.Cols)
		} else {
			s.aUV = s.aUV[:h.Cols]
		}
		for i := range s.aUV {
			s.aUV[i] = 1
		}
	}
	s.started = false
}

// resetLeft re-initialises the per-superblock-row left context (dav1d
// reset_context on t->l at each superblock row).
func (s *frameWorkLoopFilterMaskScratch) resetLeft(hasChroma bool) {
	for i := range s.lY {
		s.lY[i] = 2
	}
	if hasChroma {
		for i := range s.lUV {
			s.lUV[i] = 1
		}
	}
}

// build ORs one decoded block's deblocking edges into the frame-level masks in
// decode order, maintaining the carried above/left tx-context. sbLog2 is the
// superblock size in MI log2 (5 for 128px, 4 for 64px). It builds only edge
// geometry (a nil level cache); resolved levels are filled by the consumer.
func (h *FrameWorkLoopFilterMasks) build(s *frameWorkLoopFilterMaskScratch, visit *tile.BlockLoopVisit, sbLog2 int, intra bool) {
	h.buildBlock(s, int(visit.Block.MICol), int(visit.Block.MIRow), visit.Block.Size,
		visit.Coefficients.Tree, visit.Prefix.SkipTransform, intra, sbLog2)
}

// OrderedBlock is one block for BuildFromOrderedBlocks (test-only): the record
// map carries no ordering, so tests supply the true decode Z-order explicitly.
type OrderedBlock struct {
	BX, BY        int
	Size          tile.BlockSize
	Tree          tile.TransformTreeResult
	SkipTransform bool
	Intra         bool
}

// BuildFromOrderedBlocks builds the masks by feeding blocks in the given order
// through the decode-time carried-context path (buildBlock). Supplying the true
// single-tile decode Z-order reproduces dav1d's masks bit-for-bit, giving a
// content-independent reference for BuildFromMap's neighbour-derived build.
// sbLog2 is the coded superblock size in MI log2. Test-only.
func (h *FrameWorkLoopFilterMasks) BuildFromOrderedBlocks(blocks []OrderedBlock, sbLog2 int) {
	clear(h.Masks)
	var s frameWorkLoopFilterMaskScratch
	s.reset(h)
	for i := range blocks {
		b := &blocks[i]
		h.buildBlock(&s, b.BX, b.BY, b.Size, b.Tree, b.SkipTransform, b.Intra, sbLog2)
	}
}

// MaskBits returns the raw FilterMask edge bitmasks for direct comparison in
// tests (content-independent parity of two builds). Test-only.
func (h *FrameWorkLoopFilterMasks) MaskBits() []lfmask.FilterMask { return h.Masks }

// buildBlock is the per-block core shared by the decode-time visit-driven build
// and the record-map-driven BuildFromMap: it ORs one block's deblocking edges
// into the frame-level masks and advances the carried above/left tx-context.
func (h *FrameWorkLoopFilterMasks) buildBlock(s *frameWorkLoopFilterMaskScratch, bx, by int, size tile.BlockSize, tree tile.TransformTreeResult, skipTransform, intra bool, sbLog2 int) {
	sbRow := by >> sbLog2
	if !s.started || sbRow != s.sbRow {
		s.resetLeft(h.HasChroma)
		s.sbRow = sbRow
		s.started = true
	}

	by4 := by & 31
	ay := s.aY[bx:]
	ly := s.lY[by4:]

	var auv, luv []uint8
	if h.HasChroma && tree.HasUV {
		ssHor := h.Layout.SSHor
		ssVer := h.Layout.SSVer
		// Only the sub-block that carries the shared chroma emits chroma edges
		// and updates the chroma neighbour context, matching dav1d's has_chroma
		// gate (src/decode.c create_lf_mask_inter/intra: auv/luv passed only when
		// (bw4 > ss_hor || bx&1) && (bh4 > ss_ver || by&1)). Without this, two
		// vertically/horizontally adjacent sub-8-pixel luma blocks that map to
		// one chroma sample both write the shared chroma edge, and the second
		// reads the first's just-written class as its "previous" side instead of
		// the true neighbour -- OR-ing a spurious wider width class into the mask.
		dims, _ := size.Dimensions()
		if (int(dims.W4) > ssHor || bx&1 != 0) && (int(dims.H4) > ssVer || by&1 != 0) {
			auv = s.aUV[bx>>ssHor:]
			luv = s.lUV[by4>>ssVer:]
		}
	}

	m := h.region(bx, by)
	var lc lfmask.LevelCache // nil cells: geometry-only build
	if intra {
		s.builder.CreateIntra(m, lc, lfmask.Levels{}, bx, by, h.Cols, h.Rows,
			size, tree.Y, tree.UV, h.Layout, ay, ly, auv, luv)
		return
	}
	skip := 0
	if skipTransform {
		skip = 1
	}
	s.builder.CreateInter(m, lc, lfmask.Levels{}, bx, by, h.Cols, h.Rows, skip,
		size, tree.Y, tree.Split, tree.UV, h.Layout, ay, ly, auv, luv)
}

// FrameWorkLoopFilterMaskBuildScratch is reusable per-frame state for building
// the deblocking edge masks from a completed loop-filter record map, for encoder
// reconstruction paths that fill the map during coding rather than during a
// decode block walk. Reuse one across a stream of same-sized frames to keep the
// build allocation-free.
//
// Unlike the decode-time build (which is fed blocks in Z-order and carries the
// above/left tx-context forward), BuildFromMap visits block origins in MI-raster
// order. Raster-by-origin order does NOT reproduce dav1d's Z-order left/above
// context: when a taller block on the right of a superblock/tile-column boundary
// has a smaller origin row than the shorter block on its left, raster order
// visits the right block first and reads a stale left context. BuildFromMap
// therefore derives the left/above edge tx-context for every block directly from
// its neighbour records in the shared map (exactly what the edge-list sweep
// reads), which is order-independent and matches the sweep bit-for-bit across
// superblock AND tile-column boundaries.
type FrameWorkLoopFilterMaskBuildScratch struct {
	builder  lfmask.Builder
	neighbor lfmask.Builder
	// ly/ay/luv/auv hold the per-block neighbour-derived left/above context,
	// sliced from these reusable buffers (max block extent is 32 luma 4x4).
	ly  [32]uint8
	ay  [32]uint8
	luv [32]uint8
	auv [32]uint8
	// nbRight/nbBottom are decomposition scratch for one neighbour block.
	nbRight  [32]uint8
	nbBottom [32]uint8
}

// BuildFromMap rebuilds h's deblocking edge masks by walking m's block origins
// in MI-raster order and deriving each block's left/above neighbour tx-context
// directly from the shared record map. This reproduces the edge-list sweep's
// per-edge width derivation bit-for-bit across superblock and tile-column
// boundaries (see FrameWorkLoopFilterMaskBuildScratch). sbLog2 is unused by the
// neighbour-derived build (kept for signature stability). Masks are cleared
// first; the per-4x4 levels are resolved separately by the apply.
func (h *FrameWorkLoopFilterMasks) BuildFromMap(scratch *FrameWorkLoopFilterMaskBuildScratch, m FrameWorkLoopFilterMap, sbLog2 int) error {
	_ = sbLog2
	if !h.Valid() || scratch == nil {
		return ErrInvalidBatch
	}
	if int(m.Stride) != h.Cols || int(m.Rows) != h.Rows {
		return ErrInvalidBatch
	}
	if len(m.Records) < h.Cols*h.Rows {
		return ErrInvalidBatch
	}
	clear(h.Masks)
	stride := int(m.Stride)
	for by := 0; by < h.Rows; by++ {
		base := by * stride
		for bx := 0; bx < h.Cols; bx++ {
			r := &m.Records[base+bx]
			if !r.Valid {
				continue
			}
			// Visit each block once, at its top-left origin cell.
			if int(r.Block.MICol) != bx || int(r.Block.MIRow) != by {
				continue
			}
			h.buildBlockFromMap(scratch, m, bx, by, r)
		}
	}
	return nil
}

// buildBlockFromMap ORs one block's deblocking edges into the frame-level masks,
// deriving the left/above neighbour tx-context from the shared record map so the
// result is independent of visit order.
func (h *FrameWorkLoopFilterMasks) buildBlockFromMap(scratch *FrameWorkLoopFilterMaskBuildScratch, m FrameWorkLoopFilterMap, bx, by int, r *FrameWorkLoopFilterBlockRecord) {
	dims, _ := r.Block.Size.Dimensions()
	bw4 := imin32(h.Cols-bx, int(dims.W4))
	bh4 := imin32(h.Rows-by, int(dims.H4))
	if bw4 <= 0 || bh4 <= 0 {
		return
	}

	// Luma left context: right-edge tx of the block owning (bx-1, by+y), reset
	// value 2 at the frame left edge.
	ly := scratch.ly[:bh4]
	for i := range ly {
		ly[i] = 2
	}
	if bx > 0 {
		h.fillLeftLumaContext(scratch, m, bx, by, bh4, ly)
	}
	// Luma above context: bottom-edge tx of the block owning (bx+x, by-1).
	ay := scratch.ay[:bw4]
	for i := range ay {
		ay[i] = 2
	}
	if by > 0 {
		h.fillAboveLumaContext(scratch, m, bx, by, bw4, ay)
	}

	var auv, luv []uint8
	if h.HasChroma && r.TransformTree.HasUV {
		ssHor := h.Layout.SSHor
		ssVer := h.Layout.SSVer
		// dav1d has_chroma gate (only the sub-block carrying the shared chroma
		// emits chroma edges): (bw4 > ss_hor || bx&1) && (bh4 > ss_ver || by&1).
		if (int(dims.W4) > ssHor || bx&1 != 0) && (int(dims.H4) > ssVer || by&1 != 0) {
			cbw4 := imin32(((h.Cols+ssHor)>>ssHor)-(bx>>ssHor), (int(dims.W4)+ssHor)>>ssHor)
			cbh4 := imin32(((h.Rows+ssVer)>>ssVer)-(by>>ssVer), (int(dims.H4)+ssVer)>>ssVer)
			if cbw4 > 0 && cbh4 > 0 {
				luv = scratch.luv[:cbh4]
				for i := range luv {
					luv[i] = 1
				}
				auv = scratch.auv[:cbw4]
				for i := range auv {
					auv[i] = 1
				}
				if bx > 0 {
					h.fillLeftChromaContext(m, bx, by, cbh4, ssHor, ssVer, luv)
				}
				if by > 0 {
					h.fillAboveChromaContext(m, bx, by, cbw4, ssHor, ssVer, auv)
				}
			}
		}
	}

	region := h.region(bx, by)
	var lc lfmask.LevelCache // nil cells: geometry-only build
	if r.Intra {
		scratch.builder.CreateIntra(region, lc, lfmask.Levels{}, bx, by, h.Cols, h.Rows,
			r.Block.Size, r.TransformTree.Y, r.TransformTree.UV, h.Layout, ay, ly, auv, luv)
		return
	}
	skip := 0
	if r.SkipTransform {
		skip = 1
	}
	scratch.builder.CreateInter(region, lc, lfmask.Levels{}, bx, by, h.Cols, h.Rows, skip,
		r.Block.Size, r.TransformTree.Y, r.TransformTree.Split, r.TransformTree.UV, h.Layout, ay, ly, auv, luv)
}

// neighborRightLuma writes the right-edge luma tx-context of the block owning
// cell (col, row) into scratch.nbRight[0:neighborBH4] and returns the neighbour
// origin row and clamped bh4 so the caller can index the local row. It returns
// ok=false when the cell is not covered by a valid block.
func (h *FrameWorkLoopFilterMasks) neighborRightLuma(scratch *FrameWorkLoopFilterMaskBuildScratch, m FrameWorkLoopFilterMap, col, row int) (nbRow, nbH4 int, ok bool) {
	stride := int(m.Stride)
	nr := &m.Records[row*stride+col]
	if !nr.Valid {
		return 0, 0, false
	}
	nbCol := int(nr.Block.MICol)
	nbRow = int(nr.Block.MIRow)
	dims, _ := nr.Block.Size.Dimensions()
	nbW4 := imin32(h.Cols-nbCol, int(dims.W4))
	nbH4 = imin32(h.Rows-nbRow, int(dims.H4))
	if nbW4 <= 0 || nbH4 <= 0 {
		return 0, 0, false
	}
	if nr.Intra {
		scratch.neighbor.IntraEdgeContext(nbW4, nbH4, nr.TransformTree.Y, scratch.nbRight[:nbH4], scratch.nbBottom[:nbW4])
	} else {
		scratch.neighbor.InterEdgeContext(nbW4, nbH4, nr.TransformTree.Y, nr.TransformTree.Split, scratch.nbRight[:nbH4], scratch.nbBottom[:nbW4])
	}
	return nbRow, nbH4, true
}

// fillLeftLumaContext fills ly[y] with the right-edge tx-context of the block to
// the left of (bx, by+y). Distinct neighbours are decomposed once.
func (h *FrameWorkLoopFilterMasks) fillLeftLumaContext(scratch *FrameWorkLoopFilterMaskBuildScratch, m FrameWorkLoopFilterMap, bx, by, bh4 int, ly []uint8) {
	stride := int(m.Stride)
	y := 0
	for y < bh4 {
		row := by + y
		nr := &m.Records[row*stride+bx-1]
		if !nr.Valid {
			y++
			continue
		}
		nbRow, nbH4, ok := h.neighborRightLuma(scratch, m, bx-1, row)
		if !ok {
			y++
			continue
		}
		// The neighbour covers local rows [nbRow, nbRow+nbH4); apply its context
		// to every row of the current block that falls in that span.
		for y < bh4 {
			local := (by + y) - nbRow
			if local < 0 || local >= nbH4 {
				break
			}
			ly[y] = scratch.nbRight[local]
			y++
		}
	}
}

// fillAboveLumaContext fills ay[x] with the bottom-edge tx-context of the block
// above (bx+x, by).
func (h *FrameWorkLoopFilterMasks) fillAboveLumaContext(scratch *FrameWorkLoopFilterMaskBuildScratch, m FrameWorkLoopFilterMap, bx, by, bw4 int, ay []uint8) {
	stride := int(m.Stride)
	x := 0
	for x < bw4 {
		col := bx + x
		nr := &m.Records[(by-1)*stride+col]
		if !nr.Valid {
			x++
			continue
		}
		nbCol := int(nr.Block.MICol)
		nbRow := int(nr.Block.MIRow)
		dims, _ := nr.Block.Size.Dimensions()
		nbW4 := imin32(h.Cols-nbCol, int(dims.W4))
		nbH4 := imin32(h.Rows-nbRow, int(dims.H4))
		if nbW4 <= 0 || nbH4 <= 0 {
			x++
			continue
		}
		if nr.Intra {
			scratch.neighbor.IntraEdgeContext(nbW4, nbH4, nr.TransformTree.Y, scratch.nbRight[:nbH4], scratch.nbBottom[:nbW4])
		} else {
			scratch.neighbor.InterEdgeContext(nbW4, nbH4, nr.TransformTree.Y, nr.TransformTree.Split, scratch.nbRight[:nbH4], scratch.nbBottom[:nbW4])
		}
		for x < bw4 {
			local := (bx + x) - nbCol
			if local < 0 || local >= nbW4 {
				break
			}
			ay[x] = scratch.nbBottom[local]
			x++
		}
	}
}

// fillLeftChromaContext / fillAboveChromaContext fill the chroma neighbour
// context. Chroma tx is a single uvtx per block (never variable), so each block
// writes a uniform right/bottom chroma class (dav1d mask_edges_chroma stores
// l[y]=twl4c, a[x]=thl4c). The neighbour value is that block's uvtx class. Each
// chroma cell maps to a 2x2 luma group under 4:2:0; the shared-chroma emitting
// sub-block sits at the bottom/right of the group (dav1d has_chroma gate), so we
// read the bottom-right luma cell of the group to pick the owning block.
func (h *FrameWorkLoopFilterMasks) fillLeftChromaContext(m FrameWorkLoopFilterMap, bx, by, cbh4, ssHor, ssVer int, luv []uint8) {
	_ = ssHor
	stride := int(m.Stride)
	cby := by >> ssVer
	for cy := 0; cy < cbh4; cy++ {
		lumaRow := ((cby+cy)<<ssVer) + ssVer
		if lumaRow >= h.Rows {
			lumaRow = h.Rows - 1
		}
		nr := &m.Records[lumaRow*stride+bx-1]
		if !nr.Valid || !nr.TransformTree.HasUV {
			continue
		}
		luv[cy] = chromaWidthClass(nr.TransformTree.UV, true)
	}
}

func (h *FrameWorkLoopFilterMasks) fillAboveChromaContext(m FrameWorkLoopFilterMap, bx, by, cbw4, ssHor, ssVer int, auv []uint8) {
	_ = ssVer
	stride := int(m.Stride)
	cbx := bx >> ssHor
	for cx := 0; cx < cbw4; cx++ {
		lumaCol := ((cbx+cx)<<ssHor) + ssHor
		if lumaCol >= h.Cols {
			lumaCol = h.Cols - 1
		}
		nr := &m.Records[(by-1)*stride+lumaCol]
		if !nr.Valid || !nr.TransformTree.HasUV {
			continue
		}
		auv[cx] = chromaWidthClass(nr.TransformTree.UV, false)
	}
}

// chromaWidthClass returns dav1d's chroma edge tx-context class for a uv tx: for
// the left (right-edge, vertical) context it is log2W!=0, for the above
// (bottom-edge, horizontal) context it is log2H!=0.
func chromaWidthClass(uvtx tile.TransformSize, vertical bool) uint8 {
	td, _ := uvtx.Dimensions()
	if vertical {
		if td.Log2W != 0 {
			return 1
		}
		return 0
	}
	if td.Log2H != 0 {
		return 1
	}
	return 0
}

func imin32(a, b int) int {
	if a < b {
		return a
	}
	return b
}
