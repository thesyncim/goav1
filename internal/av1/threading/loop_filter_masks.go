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
// decode block walk. It holds the carried above/left tx-context arrays; reuse
// one across a stream of same-sized frames to keep the build allocation-free.
type FrameWorkLoopFilterMaskBuildScratch struct {
	s frameWorkLoopFilterMaskScratch
}

// BuildFromMap rebuilds h's deblocking edge masks by walking m's block origins
// in MI-raster order, carrying the above/left tx-context exactly as the decode
// block walk does. Raster origin order visits every block's above and left
// neighbours before the block, and the left context is reset once per 128-region
// row (each MI row owns an independent lY entry indexed by by&31), so it
// reproduces the decode Z-order masks bit-for-bit for a single tile spanning the
// whole frame (see build). sbLog2 is the coded superblock size in MI log2 (4 for
// the encoder's 64px superblocks). Masks are cleared first; the per-4x4 levels
// are resolved separately by the apply. It is single-tile only: the walk carries
// context across the full frame width and does not reset at tile-column
// boundaries.
func (h *FrameWorkLoopFilterMasks) BuildFromMap(scratch *FrameWorkLoopFilterMaskBuildScratch, m FrameWorkLoopFilterMap, sbLog2 int) error {
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
	scratch.s.reset(h)
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
			h.buildBlock(&scratch.s, bx, by, r.Block.Size, r.TransformTree, r.SkipTransform, r.Intra, sbLog2)
		}
	}
	return nil
}
