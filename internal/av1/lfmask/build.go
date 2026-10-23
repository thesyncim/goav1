package lfmask

import (
	"encoding/binary"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

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
			lc.fill(bx, by, bw4, bh4, 0, lv.YVert, lv.YHorz)
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
		lc.fill(bx>>ssHor, by>>ssVer, cbw4, cbh4, 2, lv.U, lv.V)
	}
	maskEdgesChroma(&m.UV, cby4, cbx4, cbw4, cbh4, 0, uvtx, auv, luv, ssHor, ssVer)
}

// InterEdgeContext walks only the transform-tree leaves that touch the block's
// right or bottom edge and writes the clamped log2 tx-context dav1d carries out
// of the block: right[y] is the width class for the rightmost luma 4x4 cell in
// row y, and bottom[x] is the height class for the bottommost cell in column x.
// These are exactly the neighbour contexts a block to the right or below reads
// for its left/top edge. bw4/bh4 are the block's luma 4x4 extent (already
// clamped to the frame if partial). right and bottom must have len >= bh4 and
// bw4 respectively.
func (b *Builder) InterEdgeContext(bw4, bh4 int, maxYtx tile.TransformSize, txMasks [2]uint16, right, bottom []uint8) {
	td, _ := maxYtx.Dimensions()
	if txMasks == [2]uint16{} {
		// No tree node can split, so every visible cell carries the root
		// transform's same clamped width/height classes.
		b.IntraEdgeContext(bw4, bh4, maxYtx, right, bottom)
		return
	}
	yOff := 0
	for y := 0; y < bh4; y += int(td.H4) {
		xOff := 0
		for x := 0; x < bw4; x += int(td.W4) {
			decompTxEdgeContext(bw4, bh4, right, bottom, y, x, maxYtx, 0, yOff, xOff, txMasks)
			xOff++
		}
		yOff++
	}
}

// decompTxEdgeContext mirrors decompTx's split decisions and tree-position
// offsets, but visits only subtrees that contain a visible right-edge cell or
// bottom-edge cell. Interior leaves cannot affect either carried context.
func decompTxEdgeContext(bw4, bh4 int, right, bottom []uint8, yBase, xBase int, from tile.TransformSize, depth, yOff, xOff int, txMasks [2]uint16) {
	if xBase >= bw4 || yBase >= bh4 {
		return
	}
	td, _ := from.Dimensions()
	w4, h4 := int(td.W4), int(td.H4)
	rightX, bottomY := bw4-1, bh4-1
	touchesRight := rightX >= xBase && rightX < xBase+w4
	touchesBottom := bottomY >= yBase && bottomY < yBase+h4
	if !touchesRight && !touchesBottom {
		return
	}

	isSplit := false
	if from != tile.TransformSize4x4 && depth <= 1 {
		isSplit = (txMasks[depth]>>(uint(yOff*4+xOff)))&1 != 0
	}
	if isSplit {
		sub := td.Sub
		htw4 := w4 >> 1
		hth4 := h4 >> 1
		decompTxEdgeContext(bw4, bh4, right, bottom, yBase, xBase, sub, depth+1, yOff*2, xOff*2, txMasks)
		if td.W4 >= td.H4 {
			decompTxEdgeContext(bw4, bh4, right, bottom, yBase, xBase+htw4, sub, depth+1, yOff*2, xOff*2+1, txMasks)
		}
		if td.H4 >= td.W4 {
			decompTxEdgeContext(bw4, bh4, right, bottom, yBase+hth4, xBase, sub, depth+1, yOff*2+1, xOff*2, txMasks)
			if td.W4 >= td.H4 {
				decompTxEdgeContext(bw4, bh4, right, bottom, yBase+hth4, xBase+htw4, sub, depth+1, yOff*2+1, xOff*2+1, txMasks)
			}
		}
		return
	}

	if touchesRight {
		lw := int(td.Log2W)
		if lw > 2 {
			lw = 2
		}
		endY := imin(yBase+h4, bh4)
		for y := yBase; y < endY; y++ {
			right[y] = uint8(lw)
		}
	}
	if touchesBottom {
		lh := int(td.Log2H)
		if lh > 2 {
			lh = 2
		}
		endX := imin(xBase+w4, bw4)
		for x := xBase; x < endX; x++ {
			bottom[x] = uint8(lh)
		}
	}
}

// IntraEdgeContext writes the right-column / bottom-row tx-context an intra block
// carries out. dav1d fills a[]/l[] with the block's uniform clamped log2W/log2H
// (mask_edges_intra: a[x]=thl4c, l[y]=twl4c), so every right cell is imin(2,
// log2W(tx)) and every bottom cell is imin(2, log2H(tx)).
func (b *Builder) IntraEdgeContext(bw4, bh4 int, ytx tile.TransformSize, right, bottom []uint8) {
	td, _ := ytx.Dimensions()
	twl4c := imin(2, int(td.Log2W))
	thl4c := imin(2, int(td.Log2H))
	for y := 0; y < bh4; y++ {
		right[y] = uint8(twl4c)
	}
	for x := 0; x < bw4; x++ {
		bottom[x] = uint8(thl4c)
	}
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
			lc.fill(bx, by, bw4, bh4, 0, lv.YVert, lv.YHorz)
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
		lc.fill(bx>>ssHor, by>>ssVer, cbw4, cbh4, 2, lv.U, lv.V)
	}
	maskEdgesChroma(&m.UV, cby4, cbx4, cbw4, cbh4, skip, uvtx, auv, luv, ssHor, ssVer)
}

// fill writes one block's two level bytes (components comp and comp+1) over a
// w x h rectangle of cells at (bx, by), dav1d_create_lf_mask_{intra,inter}'s
// level_cache_ptr loops; each row is sliced once.
func (c LevelCache) fill(bx, by, w, h, comp int, l0, l1 uint8) {
	// The pair is one 16-bit store per cell (what the C compiler makes of
	// dav1d's two adjacent byte stores); comp is 0 (Y) or 2 (U/V), and the
	// other pair of the cell is left untouched.
	v := uint16(l0) | uint16(l1)<<8
	base := by*c.Stride + bx
	if comp == 0 {
		for y := 0; y < h; y++ {
			row := c.Cells[base : base+w]
			for x := range row {
				binary.LittleEndian.PutUint16(row[x][0:2], v)
			}
			base += c.Stride
		}
		return
	}
	for y := 0; y < h; y++ {
		row := c.Cells[base : base+w]
		for x := range row {
			binary.LittleEndian.PutUint16(row[x][2:4], v)
		}
		base += c.Stride
	}
}
