package lfmask

import "github.com/thesyncim/goav1/internal/av1/tile"

// Builder owns the per-block transform-decomposition scratch so the mask-edge
// build carries zero heap allocations across the decode block walk. One Builder
// is reused for a whole frame (or per tile thread).
type Builder struct {
	txa [2][2][32][32]uint8
}

// Layout captures the chroma subsampling of the current sequence in dav1d terms:
// ssHor/ssVer are 1 when the corresponding dimension is subsampled.
type Layout struct {
	SSHor int
	SSVer int
	Mono  bool
}

// I420 returns the 4:2:0 layout.
func I420() Layout { return Layout{SSHor: 1, SSVer: 1} }

// I422 returns the 4:2:2 layout.
func I422() Layout { return Layout{SSHor: 1, SSVer: 0} }

// I444 returns the 4:4:4 layout.
func I444() Layout { return Layout{SSHor: 0, SSVer: 0} }

// LevelCache is dav1d's f->lf.level: a frame-wide grid of 4 resolved loop-filter
// levels per luma 4x4 cell (Y vertical, Y horizontal, U, V). Cells is caller
// owned and reused; Stride is in cells (each cell is one [4]uint8).
type LevelCache struct {
	Cells  [][4]uint8
	Stride int
}

// at returns the mutable level cell at luma 4x4 (bx, by).
func (c LevelCache) at(bx, by int) *[4]uint8 {
	return &c.Cells[by*c.Stride+bx]
}

// writesLevels reports whether this cache is populated. A zero LevelCache (nil
// Cells) is used by the decode-time geometry build, which fills only the edge
// bitmasks and leaves level resolution to the consumer; the level cells are
// written separately (in raster order, which is order-independent per block).
func (c LevelCache) writesLevels() bool { return c.Cells != nil }

// Levels are the four resolved base loop-filter levels for a block: Y vertical,
// Y horizontal, U, V. dav1d reads these from ts->lflvl[seg][plane][ref][mode].
type Levels struct {
	YVert uint8
	YHorz uint8
	U     uint8
	V     uint8
}

// CreateIntra builds the luma (and, when auv != nil, chroma) edge masks for one
// intra block and writes its resolved levels into the level cache. It mirrors
// dav1d_create_lf_mask_intra (src/lf_mask.c:259).
//
//   - bx/by:   frame luma 4x4 coordinates of the block top-left
//   - iw/ih:   frame luma 4x4 extent (for right/bottom edge clamping)
//   - bs:      block size
//   - ytx/uvtx transform sizes
//   - ay/ly:   above/left luma edge-context arrays (indexed bx4/by4)
//   - auv/luv: above/left chroma edge-context arrays (nil when no chroma)
func (b *Builder) CreateIntra(m *FilterMask, lc LevelCache, lv Levels, bx, by, iw, ih int, bs tile.BlockSize, ytx, uvtx tile.TransformSize, layout Layout, ay, ly, auv, luv []uint8) {
	dims, _ := bs.Dimensions()
	bw4 := imin(iw-bx, int(dims.W4))
	bh4 := imin(ih-by, int(dims.H4))
	bx4 := bx & 31
	by4 := by & 31

	if bw4 > 0 && bh4 > 0 {
		if lc.writesLevels() {
			for y := 0; y < bh4; y++ {
				for x := 0; x < bw4; x++ {
					cell := lc.at(bx+x, by+y)
					cell[0] = lv.YVert
					cell[1] = lv.YHorz
				}
			}
		}
		maskEdgesIntra(&m.Y, by4, bx4, bw4, bh4, ytx, ay, ly)
	}

	if auv == nil {
		return
	}
	ssVer := layout.SSVer
	ssHor := layout.SSHor
	cbw4 := imin(((iw+ssHor)>>ssHor)-(bx>>ssHor), (int(dims.W4)+ssHor)>>ssHor)
	cbh4 := imin(((ih+ssVer)>>ssVer)-(by>>ssVer), (int(dims.H4)+ssVer)>>ssVer)
	if cbw4 <= 0 || cbh4 <= 0 {
		return
	}
	cbx4 := bx4 >> ssHor
	cby4 := by4 >> ssVer
	if lc.writesLevels() {
		for y := 0; y < cbh4; y++ {
			for x := 0; x < cbw4; x++ {
				cell := lc.at((bx>>ssHor)+x, (by>>ssVer)+y)
				cell[2] = lv.U
				cell[3] = lv.V
			}
		}
	}
	maskEdgesChroma(&m.UV, cby4, cbx4, cbw4, cbh4, 0, uvtx, auv, luv, ssHor, ssVer)
}

// CreateInter builds the luma (and optional chroma) edge masks for one inter
// block and writes its resolved levels into the level cache. It mirrors
// dav1d_create_lf_mask_inter (src/lf_mask.c:321). txMasks carries the variable
// transform split bits (b->tx_split0/1).
func (b *Builder) CreateInter(m *FilterMask, lc LevelCache, lv Levels, bx, by, iw, ih, skip int, bs tile.BlockSize, maxYtx tile.TransformSize, txMasks [2]uint16, uvtx tile.TransformSize, layout Layout, ay, ly, auv, luv []uint8) {
	dims, _ := bs.Dimensions()
	bw4 := imin(iw-bx, int(dims.W4))
	bh4 := imin(ih-by, int(dims.H4))
	bx4 := bx & 31
	by4 := by & 31

	if bw4 > 0 && bh4 > 0 {
		if lc.writesLevels() {
			for y := 0; y < bh4; y++ {
				for x := 0; x < bw4; x++ {
					cell := lc.at(bx+x, by+y)
					cell[0] = lv.YVert
					cell[1] = lv.YHorz
				}
			}
		}
		maskEdgesInter(&m.Y, by4, bx4, bw4, bh4, skip, maxYtx, txMasks, ay, ly, &b.txa)
	}

	if auv == nil {
		return
	}
	ssVer := layout.SSVer
	ssHor := layout.SSHor
	cbw4 := imin(((iw+ssHor)>>ssHor)-(bx>>ssHor), (int(dims.W4)+ssHor)>>ssHor)
	cbh4 := imin(((ih+ssVer)>>ssVer)-(by>>ssVer), (int(dims.H4)+ssVer)>>ssVer)
	if cbw4 <= 0 || cbh4 <= 0 {
		return
	}
	cbx4 := bx4 >> ssHor
	cby4 := by4 >> ssVer
	if lc.writesLevels() {
		for y := 0; y < cbh4; y++ {
			for x := 0; x < cbw4; x++ {
				cell := lc.at((bx>>ssHor)+x, (by>>ssVer)+y)
				cell[2] = lv.U
				cell[3] = lv.V
			}
		}
	}
	maskEdgesChroma(&m.UV, cby4, cbx4, cbw4, cbh4, skip, uvtx, auv, luv, ssHor, ssVer)
}
