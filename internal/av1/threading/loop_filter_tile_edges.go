package threading

import (
	"sync/atomic"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

// FrameWorkLoopFilterTileEdges ports dav1d's tile-boundary transform-context
// backups used to "fix" the deblocking masks of a multi-tile frame
// (src/decode.c dav1d_decode_tile_sbrow, src/lf_apply_tmpl.c
// dav1d_loopfilter_sbrow_cols):
//
//   - right holds f->lf.tx_lpf_right_edge: after each superblock row of a
//     tile, the tile's carried left context t->l.tx_lpf_{y,uv} (the
//     transform-size class of the tile's right-most column per row) is
//     backed up per tile column.
//   - bottom holds the per-tile-row above context f->a[].tx_lpf_{y,uv}: the
//     carried above context after a tile finishes (its bottom-row class per
//     column), indexed per tile row.
//
// Each tile's block walk starts from reset contexts, so the masks at a tile's
// first column/row carry the block's own class; FixTileBoundariesFromEdges
// then applies dav1d's imin(class, neighbour context) from these backups
// instead of re-deriving the neighbour context from a per-MI record map.
// Tiles write disjoint ranges, so concurrently decoded tiles never race.
type FrameWorkLoopFilterTileEdges struct {
	rightY, rightUV   []uint8 // [tileCol][row], stride rows / crows
	bottomY, bottomUV []uint8 // [tileRow][col], stride cols / ccols

	tileCols, tileRows   int
	colStart, rowStart   [parser.MaxTileCols + 1]int // MI units
	rows, crows          int
	cols, ccols          int
	ssHor, ssVer, sbLog2 int
	hasChroma            bool

	// broken is set when a job's geometry does not match the bound tiling, so
	// the backups are incomplete and the caller must use another tile fix.
	broken atomic.Bool
}

func growBytes(b []uint8, n int) []uint8 {
	if cap(b) < n {
		return make([]uint8, n)
	}
	return b[:n]
}

// BindTileEdges sizes the backup storage for this frame's tiling and attaches
// it to h. A single-tile frame needs no fix and detaches it. sbLog2 is the
// superblock size in MI log2 (4 or 5).
func (h *FrameWorkLoopFilterMasks) BindTileEdges(e *FrameWorkLoopFilterTileEdges, tiles parser.TileInfo, sbLog2 int) {
	h.tileEdges = nil
	if e == nil || !h.Valid() || (tiles.Cols <= 1 && tiles.Rows <= 1) ||
		int(tiles.Cols) > parser.MaxTileCols || int(tiles.Rows) > parser.MaxTileRows {
		return
	}
	e.tileCols = int(tiles.Cols)
	e.tileRows = int(tiles.Rows)
	e.ssHor, e.ssVer = h.Layout.SSHor, h.Layout.SSVer
	e.sbLog2 = sbLog2
	e.hasChroma = h.HasChroma
	e.broken.Store(false)
	e.rows = h.Rows
	e.cols = h.Cols
	e.crows = (h.Rows + e.ssVer) >> e.ssVer
	e.ccols = (h.Cols + e.ssHor) >> e.ssHor
	for i := 0; i <= e.tileCols; i++ {
		e.colStart[i] = min(int(tiles.ColStartSB[i])<<sbLog2, h.Cols)
	}
	for i := 0; i <= e.tileRows; i++ {
		e.rowStart[i] = min(int(tiles.RowStartSB[i])<<sbLog2, h.Rows)
	}
	e.rightY = growBytes(e.rightY, e.tileCols*e.rows)
	e.bottomY = growBytes(e.bottomY, e.tileRows*e.cols)
	if e.hasChroma {
		e.rightUV = growBytes(e.rightUV, e.tileCols*e.crows)
		e.bottomUV = growBytes(e.bottomUV, e.tileRows*e.ccols)
	}
	h.tileEdges = e
}

// TileEdgesBound reports whether this frame carries dav1d tile-edge backups.
func (h *FrameWorkLoopFilterMasks) TileEdgesBound() bool { return h != nil && h.tileEdges != nil }

// flushRightEdge backs up the left context of the superblock row the job just
// finished (dav1d decode.c: memcpy(&f->lf.tx_lpf_right_edge[0][align_h *
// tile_col + t->by], &t->l.tx_lpf_y[t->by & 16], sb_step), likewise for uv).
func (s *frameWorkLoopFilterMaskScratch) flushRightEdge() {
	e := s.edges
	if e == nil || !s.started {
		return
	}
	start := s.sbRow << e.sbLog2
	if start >= e.rows {
		return
	}
	n := min(1<<e.sbLog2, e.rows-start)
	copy(e.rightY[s.tileCol*e.rows+start:][:n], s.lY[start&31:][:n])
	if !e.hasChroma {
		return
	}
	cstart := start >> e.ssVer
	cn := min((n+e.ssVer)>>e.ssVer, e.crows-cstart)
	if cn > 0 {
		copy(e.rightUV[s.tileCol*e.crows+cstart:][:cn], s.lUV[(start&31)>>e.ssVer:][:cn])
	}
}

// finishTileEdges flushes the last superblock row's right edge and backs up the
// tile's bottom-row above context (dav1d's per-tile-row f->a tx_lpf_y/uv).
func (s *frameWorkLoopFilterMaskScratch) finishTileEdges() {
	e := s.edges
	if e == nil {
		return
	}
	s.flushRightEdge()
	c0, c1 := e.colStart[s.tileCol], e.colStart[s.tileCol+1]
	if c1 > c0 {
		copy(e.bottomY[s.tileRow*e.cols+c0:s.tileRow*e.cols+c1], s.aY[c0:c1])
	}
	if !e.hasChroma {
		return
	}
	cc0 := c0 >> e.ssHor
	cc1 := min((c1+e.ssHor)>>e.ssHor, e.ccols)
	if cc1 > cc0 {
		copy(e.bottomUV[s.tileRow*e.ccols+cc0:s.tileRow*e.ccols+cc1], s.aUV[cc0:cc1])
	}
}

// FixTileBoundariesFromEdges applies dav1d's "fix lpf strength at tile col/row
// boundaries" (src/lf_apply_tmpl.c dav1d_loopfilter_sbrow_cols) from the
// tile-edge backups: every unit of the first column (row) of a tile is
// re-filed under imin(its class, the neighbouring tile's right (bottom) edge
// context). After the fix the masks equal a single carried-context build.
func (h *FrameWorkLoopFilterMasks) FixTileBoundariesFromEdges() error {
	e := h.tileEdges
	if !h.Valid() || e == nil || e.rows != h.Rows || e.cols != h.Cols || e.broken.Load() {
		return ErrInvalidBatch
	}
	for c := 1; c < e.tileCols; c++ {
		x := e.colStart[c]
		if x <= 0 || x >= h.Cols {
			continue
		}
		pos := x & 31
		for y0 := 0; y0 < h.Rows; y0 += 32 {
			n := min(32, h.Rows-y0)
			fixLumaLane(&h.region(x, y0).Y[0][pos], e.rightY[(c-1)*e.rows+y0:][:n])
			if !e.hasChroma {
				continue
			}
			cy0 := y0 >> e.ssVer
			cn := min(32>>e.ssVer, e.crows-cy0)
			if cn <= 0 {
				continue
			}
			fixChromaLane(&h.region(x, y0).UV[0][pos>>e.ssHor], e.rightUV[(c-1)*e.crows+cy0:][:cn], 16>>e.ssVer)
		}
	}
	for r := 1; r < e.tileRows; r++ {
		y := e.rowStart[r]
		if y <= 0 || y >= h.Rows {
			continue
		}
		pos := y & 31
		for x0 := 0; x0 < h.Cols; x0 += 32 {
			n := min(32, h.Cols-x0)
			fixLumaLane(&h.region(x0, y).Y[1][pos], e.bottomY[(r-1)*e.cols+x0:][:n])
			if !e.hasChroma {
				continue
			}
			cx0 := x0 >> e.ssHor
			cn := min(32>>e.ssHor, e.ccols-cx0)
			if cn <= 0 {
				continue
			}
			fixChromaLane(&h.region(x0, y).UV[1][pos>>e.ssVer], e.bottomUV[(r-1)*e.ccols+cx0:][:cn], 16>>e.ssHor)
		}
	}
	h.TileBoundariesFixed = true
	return nil
}
