// Package lfmask ports dav1d's per-superblock BITMASK loop-filter edge planner.
//
// dav1d builds deblocking edge masks incrementally during the decode block
// walk (src/lf_mask.c): it expands each block's variable transform tree to a
// per-4x4 tx-size grid (decomp_tx), carries per-column/per-row edge tx-context
// arrays (a[]/l[]) across neighbouring blocks so an edge strength resolves once
// as imin(current, previous-side), and ORs one bit per 4x4 edge into a compact
// per-128x128 bitmask (mask_edges_intra / mask_edges_inter / mask_edges_chroma).
// The deblock pass then becomes a pure bit scan with zero neighbour or tree
// lookups (src/loopfilter_tmpl.c).
//
// This package is a faithful, standalone port of that mask-building core. It is
// intentionally decoupled from the decoder so it can be unit-tested against the
// dav1d reference before being wired into the decode block walk.
//
// Ported from dav1d src/lf_mask.c (pinned third_party/upstream/dav1d).
package lfmask

import "github.com/thesyncim/goav1/internal/av1/tile"

// FilterMask holds the dav1d Av1Filter edge bitmasks for one 128x128 region
// (1 or 4 superblocks), pre-superres scaling. Each bit is one 4x4 edge.
//
// Y indexing mirrors dav1d filter_y[dir][pos][strength][half]:
//   - dir  0 = column (vertical edges), 1 = row (horizontal edges)
//   - pos  4x4 position within the region (0..31)
//   - strength luma class 0/1/2 -> filter widths 4/8/16
//   - half  0 = low 16 bits, 1 = high 16 bits of the 32-bit edge column/row
//
// UV mirrors filter_uv with chroma classes 0/1 -> widths 4/6.
//
// Ported from dav1d src/lf_mask.h Av1Filter.
type FilterMask struct {
	Y  [2][32][3][2]uint16
	UV [2][32][2][2]uint16
}

// Clear resets every edge bit to zero. Callers reuse one FilterMask per region
// across frames; Clear must run before re-decoding into it.
func (m *FilterMask) Clear() {
	*m = FilterMask{}
}

// FilterLUT holds the E/I/H deblocking limits keyed by loop-filter level, plus
// the two sharpness scalars dav1d caches. Ported from dav1d src/lf_mask.h
// Av1FilterLUT and src/lf_mask.c dav1d_calc_eih.
type FilterLUT struct {
	E     [64]uint8
	I     [64]uint8
	Sharp [2]uint64
}

// CalcEIH fills the E/I/H limit LUT from a loop-filter sharpness value, matching
// dav1d_calc_eih (src/lf_mask.c:385). E == block limit, I == interior limit,
// H (per-level) == level>>4 is computed at consume time, not stored here.
func CalcEIH(sharpness int) FilterLUT {
	var lut FilterLUT
	sharp := sharpness
	for level := 0; level < 64; level++ {
		limit := level
		if sharp > 0 {
			limit >>= (sharp + 3) >> 2
			if m := 9 - sharp; limit > m {
				limit = m
			}
		}
		if limit < 1 {
			limit = 1
		}
		lut.I[level] = uint8(limit)
		lut.E[level] = uint8(2*(level+2) + limit)
	}
	lut.Sharp[0] = uint64((sharp + 3) >> 2)
	if sharp != 0 {
		lut.Sharp[1] = uint64(9 - sharp)
	} else {
		lut.Sharp[1] = 0xff
	}
	return lut
}

func imin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// decompTx expands a (possibly variable) transform tree rooted at `from` into a
// per-4x4 tx-context grid inside txa, faithfully following dav1d decomp_tx
// (src/lf_mask.c:39).
//
// txa is indexed txa[edge][field][y][x] where:
//   - edge  0 = column direction (holds clamped log2 width for vertical edges)
//     1 = row direction (holds clamped log2 height for horizontal edges)
//   - field 0 = tx size class, 1 = step (in 4x4 units) to the next tx boundary
//
// yBase/xBase are the 4x4 memory offsets of this subtree inside txa (the dav1d
// pointer arithmetic). yOff/xOff are the tree-position offsets used to index the
// per-depth split masks.
func decompTx(txa *[2][2][32][32]uint8, yBase, xBase int, from tile.TransformSize, depth, yOff, xOff int, txMasks [2]uint16) {
	td, _ := from.Dimensions()
	isSplit := false
	if from != tile.TransformSize4x4 && depth <= 1 {
		isSplit = (txMasks[depth]>>(uint(yOff*4+xOff)))&1 != 0
	}
	if isSplit {
		sub := td.Sub
		htw4 := int(td.W4) >> 1
		hth4 := int(td.H4) >> 1
		decompTx(txa, yBase, xBase, sub, depth+1, yOff*2+0, xOff*2+0, txMasks)
		if td.W4 >= td.H4 {
			decompTx(txa, yBase, xBase+htw4, sub, depth+1, yOff*2+0, xOff*2+1, txMasks)
		}
		if td.H4 >= td.W4 {
			decompTx(txa, yBase+hth4, xBase, sub, depth+1, yOff*2+1, xOff*2+0, txMasks)
			if td.W4 >= td.H4 {
				decompTx(txa, yBase+hth4, xBase+htw4, sub, depth+1, yOff*2+1, xOff*2+1, txMasks)
			}
		}
		return
	}
	lw := int(td.Log2W)
	if lw > 2 {
		lw = 2
	}
	lh := int(td.Log2H)
	if lh > 2 {
		lh = 2
	}
	w := int(td.W4)
	h := int(td.H4)
	for y := 0; y < h; y++ {
		row0 := &txa[0][0][yBase+y]
		row1 := &txa[1][0][yBase+y]
		for x := 0; x < w; x++ {
			row0[xBase+x] = uint8(lw)
			row1[xBase+x] = uint8(lh)
		}
		txa[0][1][yBase+y][xBase] = uint8(w)
	}
	for x := 0; x < w; x++ {
		txa[1][1][yBase][xBase+x] = uint8(h)
	}
}

// maskEdgesInter ORs one 128x128 block's inter luma edges into masks and updates
// the carried edge-context arrays a[] (above, per-column) and l[] (left, per-row).
// Ported from dav1d mask_edges_inter (src/lf_mask.c:79).
func maskEdgesInter(masks *[2][32][3][2]uint16, by4, bx4, w4, h4, skip int, maxTx tile.TransformSize, txMasks [2]uint16, a, l []uint8, txa *[2][2][32][32]uint8) {
	td, _ := maxTx.Dimensions()
	yOff := 0
	for y := 0; y < h4; y += int(td.H4) {
		xOff := 0
		for x := 0; x < w4; x += int(td.W4) {
			decompTx(txa, y, x, maxTx, 0, yOff, xOff, txMasks)
			xOff++
		}
		yOff++
	}

	// left block edge
	mask := uint32(1) << uint(by4)
	for y := 0; y < h4; y++ {
		sidx := 0
		if mask >= 0x10000 {
			sidx = 1
		}
		smask := uint16(mask >> uint(sidx<<4))
		masks[0][bx4][imin(int(txa[0][0][y][0]), int(l[y]))][sidx] |= smask
		mask <<= 1
	}

	// top block edge
	mask = uint32(1) << uint(bx4)
	for x := 0; x < w4; x++ {
		sidx := 0
		if mask >= 0x10000 {
			sidx = 1
		}
		smask := uint16(mask >> uint(sidx<<4))
		masks[1][by4][imin(int(txa[1][0][0][x]), int(a[x]))][sidx] |= smask
		mask <<= 1
	}

	if skip == 0 {
		// inner (tx) left|right edges
		mask = uint32(1) << uint(by4)
		for y := 0; y < h4; y++ {
			sidx := 0
			if mask >= 0x10000 {
				sidx = 1
			}
			smask := uint16(mask >> uint(sidx<<4))
			ltx := int(txa[0][0][y][0])
			step := int(txa[0][1][y][0])
			for x := step; x < w4; x += step {
				rtx := int(txa[0][0][y][x])
				masks[0][bx4+x][imin(rtx, ltx)][sidx] |= smask
				ltx = rtx
				step = int(txa[0][1][y][x])
			}
			mask <<= 1
		}
		// inner (tx) top|bottom edges
		mask = uint32(1) << uint(bx4)
		for x := 0; x < w4; x++ {
			sidx := 0
			if mask >= 0x10000 {
				sidx = 1
			}
			smask := uint16(mask >> uint(sidx<<4))
			ttx := int(txa[1][0][0][x])
			step := int(txa[1][1][0][x])
			for y := step; y < h4; y += step {
				btx := int(txa[1][0][y][x])
				masks[1][by4+y][imin(ttx, btx)][sidx] |= smask
				ttx = btx
				step = int(txa[1][1][y][x])
			}
			mask <<= 1
		}
	}

	for y := 0; y < h4; y++ {
		l[y] = txa[0][0][y][w4-1]
	}
	for x := 0; x < w4; x++ {
		a[x] = txa[1][0][h4-1][x]
	}
}

// maskEdgesIntra ORs one 128x128 block's intra luma edges into masks and updates
// the carried a[]/l[] edge-context arrays. Ported from dav1d mask_edges_intra
// (src/lf_mask.c:147).
func maskEdgesIntra(masks *[2][32][3][2]uint16, by4, bx4, w4, h4 int, tx tile.TransformSize, a, l []uint8) {
	td, _ := tx.Dimensions()
	twl4 := int(td.Log2W)
	thl4 := int(td.Log2H)
	twl4c := imin(2, twl4)
	thl4c := imin(2, thl4)

	// left block edge
	mask := uint32(1) << uint(by4)
	for y := 0; y < h4; y++ {
		sidx := 0
		if mask >= 0x10000 {
			sidx = 1
		}
		smask := uint16(mask >> uint(sidx<<4))
		masks[0][bx4][imin(twl4c, int(l[y]))][sidx] |= smask
		mask <<= 1
	}

	// top block edge
	mask = uint32(1) << uint(bx4)
	for x := 0; x < w4; x++ {
		sidx := 0
		if mask >= 0x10000 {
			sidx = 1
		}
		smask := uint16(mask >> uint(sidx<<4))
		masks[1][by4][imin(thl4c, int(a[x]))][sidx] |= smask
		mask <<= 1
	}

	// inner (tx) left|right edges
	hstep := int(td.W4)
	t := uint64(1) << uint(by4)
	inner := uint32((t << uint(h4)) - t)
	inner1 := uint16(inner & 0xffff)
	inner2 := uint16(inner >> 16)
	for x := hstep; x < w4; x += hstep {
		if inner1 != 0 {
			masks[0][bx4+x][twl4c][0] |= inner1
		}
		if inner2 != 0 {
			masks[0][bx4+x][twl4c][1] |= inner2
		}
	}

	// inner (tx) top|bottom edges
	vstep := int(td.H4)
	t = uint64(1) << uint(bx4)
	inner = uint32((t << uint(w4)) - t)
	inner1 = uint16(inner & 0xffff)
	inner2 = uint16(inner >> 16)
	for y := vstep; y < h4; y += vstep {
		if inner1 != 0 {
			masks[1][by4+y][thl4c][0] |= inner1
		}
		if inner2 != 0 {
			masks[1][by4+y][thl4c][1] |= inner2
		}
	}

	for x := 0; x < w4; x++ {
		a[x] = uint8(thl4c)
	}
	for y := 0; y < h4; y++ {
		l[y] = uint8(twl4c)
	}
}

// maskEdgesChroma ORs one block's chroma edges into masks and updates the
// carried a[]/l[] context arrays. Coordinates and extents are chroma 4x4 units.
// Ported from dav1d mask_edges_chroma (src/lf_mask.c:200).
func maskEdgesChroma(masks *[2][32][2][2]uint16, cby4, cbx4, cw4, ch4, skipInter int, tx tile.TransformSize, a, l []uint8, ssHor, ssVer int) {
	td, _ := tx.Dimensions()
	twl4 := int(td.Log2W)
	thl4 := int(td.Log2H)
	twl4c := 0
	if twl4 != 0 {
		twl4c = 1
	}
	thl4c := 0
	if thl4 != 0 {
		thl4c = 1
	}
	vbits := 4 - ssVer
	hbits := 4 - ssHor
	vmask := 16 >> ssVer
	hmask := 16 >> ssHor
	vmax := uint32(1) << uint(vmask)
	hmax := uint32(1) << uint(hmask)

	// left block edge
	mask := uint32(1) << uint(cby4)
	for y := 0; y < ch4; y++ {
		sidx := 0
		if mask >= vmax {
			sidx = 1
		}
		smask := uint16(mask >> uint(sidx<<uint(vbits)))
		masks[0][cbx4][imin(twl4c, int(l[y]))][sidx] |= smask
		mask <<= 1
	}

	// top block edge
	mask = uint32(1) << uint(cbx4)
	for x := 0; x < cw4; x++ {
		sidx := 0
		if mask >= hmax {
			sidx = 1
		}
		smask := uint16(mask >> uint(sidx<<uint(hbits)))
		masks[1][cby4][imin(thl4c, int(a[x]))][sidx] |= smask
		mask <<= 1
	}

	if skipInter == 0 {
		// inner (tx) left|right edges
		hstep := int(td.W4)
		t := uint64(1) << uint(cby4)
		inner := uint32((t << uint(ch4)) - t)
		inner1 := uint16(inner & ((1 << uint(vmask)) - 1))
		inner2 := uint16(inner >> uint(vmask))
		for x := hstep; x < cw4; x += hstep {
			if inner1 != 0 {
				masks[0][cbx4+x][twl4c][0] |= inner1
			}
			if inner2 != 0 {
				masks[0][cbx4+x][twl4c][1] |= inner2
			}
		}

		// inner (tx) top|bottom edges
		vstep := int(td.H4)
		t = uint64(1) << uint(cbx4)
		inner = uint32((t << uint(cw4)) - t)
		inner1 = uint16(inner & ((1 << uint(hmask)) - 1))
		inner2 = uint16(inner >> uint(hmask))
		for y := vstep; y < ch4; y += vstep {
			if inner1 != 0 {
				masks[1][cby4+y][thl4c][0] |= inner1
			}
			if inner2 != 0 {
				masks[1][cby4+y][thl4c][1] |= inner2
			}
		}
	}

	for x := 0; x < cw4; x++ {
		a[x] = uint8(thl4c)
	}
	for y := 0; y < ch4; y++ {
		l[y] = uint8(twl4c)
	}
}
