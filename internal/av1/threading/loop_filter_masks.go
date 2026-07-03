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
	bx := int(visit.Block.MICol)
	by := int(visit.Block.MIRow)
	tree := visit.Coefficients.Tree

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
		auv = s.aUV[bx>>ssHor:]
		luv = s.lUV[by4>>ssVer:]
	}

	m := h.region(bx, by)
	var lc lfmask.LevelCache // nil cells: geometry-only build
	if intra {
		s.builder.CreateIntra(m, lc, lfmask.Levels{}, bx, by, h.Cols, h.Rows,
			visit.Block.Size, tree.Y, tree.UV, h.Layout, ay, ly, auv, luv)
		return
	}
	skip := 0
	if visit.Prefix.SkipTransform {
		skip = 1
	}
	s.builder.CreateInter(m, lc, lfmask.Levels{}, bx, by, h.Cols, h.Rows, skip,
		visit.Block.Size, tree.Y, tree.Split, tree.UV, h.Layout, ay, ly, auv, luv)
}
