package tile

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// Neighbor-motion readers over the dav1d refmvs rows: decode.c
// find_matching_ref / derive_warpmv (warped-motion samples), the
// has-overlappable-neighbours findoddzero test, recon_tmpl.c obmc()'s
// neighbor walk and the sub8x8 chroma derivation. They replace the
// libaom-shaped per-superblock grid readers for the frame decoder.

// refmvsFindOddZero ports dav1d's findoddzero over an intra edge context.
func refmvsFindOddZero(buf []uint8, length int) bool {
	for n := 0; n < length; n++ {
		if buf[n*2] == 0 {
			return true
		}
	}
	return false
}

// hasOverlappableNeighbours is decode_b's "has overlappable neighbours"
// test: an inter (non-intra) block at an odd 4x4 position of the left or top
// edge context.
func (c *BlockModeContext) hasOverlappableNeighbours(block *BlockVisit) bool {
	x4, y4 := int(block.X4), int(block.Y4)
	if block.HaveLeft && refmvsFindOddZero(c.LeftIntra[y4+1:], int(block.VisibleH4)>>1) {
		return true
	}
	return block.HaveTop && refmvsFindOddZero(c.AboveIntra[x4+1:], int(block.VisibleW4)>>1)
}

func refmvsMatchesSingleRef(b *refmvsBlock, ref int8) bool {
	return b.ref[0] == ref && b.ref[1] == -1
}

// findMatchingRef ports dav1d's find_matching_ref. ref is the 0-based
// reference frame; masks[0] holds top-edge hits (bit 32 = top-right) and
// masks[1] left-edge hits (bit 32 = top-left).
func (rt *refmvsTile) findMatchingRef(block *BlockVisit, topHasRight bool, ref ReferenceFrame, masks *[2]uint64) {
	dims := blockDimensions[block.Size]
	bw4, bh4 := int32(dims.W4), int32(dims.H4)
	w4, h4 := int32(block.VisibleW4), int32(block.VisibleH4)
	bx, by := int32(block.MICol), int32(block.MIRow)
	base := (by & 31) + 5
	target := int8(ref) + 1
	count := 0
	haveTop, haveLeft := block.HaveTop, block.HaveLeft
	haveTopLeft := haveTop && haveLeft
	haveTopRight := max(bw4, bh4) < 32 && haveTop && bx+bw4 < rt.tileColEnd && topHasRight

	if haveTop {
		top := rt.row(base - 1)
		r2 := &top[bx]
		if refmvsMatchesSingleRef(r2, target) {
			masks[0] |= 1
			count = 1
		}
		aw4 := int32(blockDimensions[r2.bs].W4)
		if aw4 >= bw4 {
			off := bx & (aw4 - 1)
			if off != 0 {
				haveTopLeft = false
			}
			if aw4-off > bw4 {
				haveTopRight = false
			}
		} else {
			mask := uint32(1) << aw4
			for x := aw4; x < w4; x += aw4 {
				r2 = &top[bx+x]
				if refmvsMatchesSingleRef(r2, target) {
					masks[0] |= uint64(mask)
					count++
					if count >= 8 {
						return
					}
				}
				aw4 = int32(blockDimensions[r2.bs].W4)
				mask <<= aw4
			}
		}
	}
	if haveLeft {
		l := &rt.buf[rt.rows[base]+bx-1]
		if refmvsMatchesSingleRef(l, target) {
			masks[1] |= 1
			count++
			if count >= 8 {
				return
			}
		}
		lh4 := int32(blockDimensions[l.bs].H4)
		if lh4 >= bh4 {
			if by&(lh4-1) != 0 {
				haveTopLeft = false
			}
		} else {
			mask := uint32(1) << lh4
			for y := lh4; y < h4; y += lh4 {
				l = &rt.buf[rt.rows[base+y]+bx-1]
				if refmvsMatchesSingleRef(l, target) {
					masks[1] |= uint64(mask)
					count++
					if count >= 8 {
						return
					}
				}
				lh4 = int32(blockDimensions[l.bs].H4)
				mask <<= lh4
			}
		}
	}
	if haveTopLeft && refmvsMatchesSingleRef(&rt.buf[rt.rows[base-1]+bx-1], target) {
		masks[1] |= 1 << 32
		count++
		if count >= 8 {
			return
		}
	}
	if haveTopRight && refmvsMatchesSingleRef(&rt.buf[rt.rows[base-1]+bx+bw4], target) {
		masks[0] |= 1 << 32
	}
}

func refmvsWarpSample(dx, dy, sx, sy int32, b *refmvsBlock) warpSample {
	d := blockDimensions[b.bs]
	x := 16*(2*dx+sx*int32(d.W4)) - 8
	y := 16*(2*dy+sy*int32(d.H4)) - 8
	mv := b.mv[0]
	return warpSample{X: int16(x), Y: int16(y), RefX: int16(x + mv.col()), RefY: int16(y + mv.row()), Motion: mv.vector()}
}

// deriveWarpMV ports dav1d's derive_warpmv: gather the projectable samples
// selected by masks, drop the ones whose motion differs from mv by more than
// the size threshold, then solve the affine model (dav1d_find_affine_int and
// dav1d_get_shear_params).
func (rt *refmvsTile) deriveWarpMV(block *BlockVisit, masks *[2]uint64, mvIn motion.Vector) (WarpedMotionModel, bool, error) {
	var pts [maxWarpSamples]warpSample
	np := 0
	dims := blockDimensions[block.Size]
	bw4, bh4 := int32(dims.W4), int32(dims.H4)
	bx, by := int32(block.MICol), int32(block.MIRow)
	base := (by & 31) + 5
	top := rt.rows[base-1]

	if uint32(masks[0]) == 1 && masks[1]>>32 == 0 {
		b := &rt.buf[top+bx]
		off := bx & (int32(blockDimensions[b.bs].W4) - 1)
		pts[np] = refmvsWarpSample(-off, 0, 1, -1, b)
		np++
	} else {
		xmask := uint32(masks[0])
		for off := int32(0); np < 8 && xmask != 0; {
			tz := int32(bits.TrailingZeros32(xmask))
			off += tz
			xmask >>= tz
			pts[np] = refmvsWarpSample(off, 0, 1, -1, &rt.buf[top+bx+off])
			np++
			xmask &^= 1
		}
	}
	if np < 8 && masks[1] == 1 {
		l := &rt.buf[rt.rows[base]+bx-1]
		off := by & (int32(blockDimensions[l.bs].H4) - 1)
		pts[np] = refmvsWarpSample(0, -off, -1, 1, &rt.buf[rt.rows[base-off]+bx-1])
		np++
	} else {
		ymask := uint32(masks[1])
		for off := int32(0); np < 8 && ymask != 0; {
			tz := int32(bits.TrailingZeros32(ymask))
			off += tz
			ymask >>= tz
			pts[np] = refmvsWarpSample(0, off, -1, 1, &rt.buf[rt.rows[base+off]+bx-1])
			np++
			ymask &^= 1
		}
	}
	if np < 8 && masks[1]>>32 != 0 { // top/left
		pts[np] = refmvsWarpSample(0, 0, -1, -1, &rt.buf[top+bx-1])
		np++
	}
	if np < 8 && masks[0]>>32 != 0 { // top/right
		pts[np] = refmvsWarpSample(bw4, 0, 1, -1, &rt.buf[top+bx+bw4])
		np++
	}

	// select according to motion vector difference against a threshold
	var mvd [maxWarpSamples]int32
	ret := 0
	thresh := 4 * min(max(max(bw4, bh4), 4), 28)
	mx, my := int32(mvIn.Col), int32(mvIn.Row)
	for i := 0; i < np; i++ {
		dx := int32(pts[i].RefX) - int32(pts[i].X) - mx
		dy := int32(pts[i].RefY) - int32(pts[i].Y) - my
		if dx < 0 {
			dx = -dx
		}
		if dy < 0 {
			dy = -dy
		}
		mvd[i] = dx + dy
		if mvd[i] > thresh {
			mvd[i] = -1
		} else {
			ret++
		}
	}
	if ret == 0 {
		ret = 1
	} else {
		for i, j, k := 0, np-1, 0; k < np-ret; k, i, j = k+1, i+1, j-1 {
			for mvd[i] != -1 {
				i++
			}
			for mvd[j] == -1 {
				j--
			}
			if i > j {
				break
			}
			// replace the discarded samples
			mvd[i] = mvd[j]
			pts[i] = pts[j]
		}
	}
	return findWarpProjection(pts[:ret], *block, mvIn)
}

// subChromaCells ports dav1d's sub8x8 chroma derivation (recon_tmpl.c
// recon_b_inter): for a 4-wide or 4-tall block with subsampled chroma, the
// left (r[0][bx-1]), top (r[-1][bx]) and top-left (r[-1][bx-1]) neighbors
// must all be inter for the chroma block to be predicted per luma sub-block.
// Cell filters follow dav1d: t->l.filter[by4], t->a->filter[bx4] and
// tl_4x4_filter (the first 4x4 of the 8x8 split). The trailing self cell's MV
// and reference are filled by the caller.
func (rt *refmvsTile) subChromaCells(c *BlockModeContext, block *BlockVisit, ssX, ssY bool, filters motion.InterpFilters) (SubChromaInterResult, bool) {
	dims := blockDimensions[block.Size]
	is4x := dims.W4 == 1 && ssX
	is4y := dims.H4 == 1 && ssY
	if !is4x && !is4y {
		return SubChromaInterResult{}, false
	}
	bx, by := int32(block.MICol), int32(block.MIRow)
	base := (by & 31) + 5
	var left, top, topLeft *refmvsBlock
	if is4x {
		left = &rt.buf[rt.rows[base]+bx-1]
		if left.ref[0] <= 0 {
			return SubChromaInterResult{}, false
		}
	}
	if is4y {
		top = &rt.buf[rt.rows[base-1]+bx]
		if top.ref[0] <= 0 {
			return SubChromaInterResult{}, false
		}
	}
	if is4x && is4y {
		topLeft = &rt.buf[rt.rows[base-1]+bx-1]
		if topLeft.ref[0] <= 0 {
			return SubChromaInterResult{}, false
		}
	}
	cellW := uint8(dims.W4) * 4
	cellH := uint8(dims.H4) * 4
	if ssX {
		cellW >>= 1
	}
	if ssY {
		cellH >>= 1
	}
	var result SubChromaInterResult
	add := func(offX, offY uint8, b *refmvsBlock, f motion.InterpFilters) {
		cell := SubChromaInterCell{OffsetX: offX, OffsetY: offY, Width: cellW, Height: cellH, InterpFilters: f}
		if b != nil {
			cell.MV = b.mv[0].vector()
			cell.Reference = ReferenceFrame(b.ref[0] - 1)
		}
		result.Cells[result.Count] = cell
		result.Count++
	}
	var offX, offY uint8
	if is4x {
		offX = cellW
	}
	if is4y {
		offY = cellH
	}
	if topLeft != nil {
		add(0, 0, topLeft, rt.tl4x4Filter)
	}
	if top != nil {
		add(offX, 0, top, c.AboveInterp[block.X4])
	}
	if left != nil {
		add(0, offY, left, c.LeftInterp[block.Y4])
	}
	add(offX, offY, nil, filters)
	return result, true
}

func refmvsNeighborMotion(b *refmvsBlock) InterMotionResult {
	m := InterMotionResult{MV: [2]motion.Vector{b.mv[0].vector(), b.mv[1].vector()}}
	m.References.Ref[0] = ReferenceFrame(b.ref[0] - 1)
	m.References.Ref[1] = ReferenceFrameNone
	if b.ref[1] > 0 {
		m.References.Ref[1] = ReferenceFrame(b.ref[1] - 1)
		m.References.Compound = true
	} else if b.ref[1] == 0 {
		m.InterIntra = true
	}
	return m
}

// obmcNeighbors ports the neighbor walk of dav1d's obmc(): the odd 4x4 cell
// of each above/left neighbor (r[-1][bx + x + 1], r[y + 1][bx - 1]) with
// steps clipped to [2, 16], at most min(log2 size, 4) inter neighbors per
// edge. Filters come from the above/left edge contexts like dav1d's
// t->a->filter / t->l.filter.
func (rt *refmvsTile) obmcNeighbors(c *BlockModeContext, block *BlockVisit) OverlappableNeighborSet {
	var set OverlappableNeighborSet
	dims := blockDimensions[block.Size]
	bx, by := int32(block.MICol), int32(block.MIRow)
	x4, y4 := int32(block.X4), int32(block.Y4)
	w4, h4 := int32(block.VisibleW4), int32(block.VisibleH4)
	base := (by & 31) + 5
	if block.HaveTop {
		top := rt.row(base - 1)
		limit := min(int32(dims.Log2W), 4)
		for i, x := int32(0), int32(0); x < w4 && i < limit; {
			a := &top[bx+x+1]
			step := min(max(int32(blockDimensions[a.bs].W4), 2), 16)
			if a.ref[0] > 0 {
				slot := x4 + x + 1
				set.Above[i] = OverlappableNeighbor{
					RelX4:              int8(x),
					Span4:              uint8(min(step, w4-x)),
					Size:               a.bs,
					Motion:             refmvsNeighborMotion(a),
					MotionValid:        true,
					InterpFilters:      c.AboveInterp[slot],
					InterpFiltersValid: c.AboveInterpValid[slot] != 0,
				}
				i++
				set.AboveCount = uint8(i)
			}
			x += step
		}
	}
	if block.HaveLeft {
		limit := min(int32(dims.Log2H), 4)
		for i, y := int32(0), int32(0); y < h4 && i < limit; {
			l := &rt.buf[rt.rows[base+y+1]+bx-1]
			step := min(max(int32(blockDimensions[l.bs].H4), 2), 16)
			if l.ref[0] > 0 {
				slot := y4 + y + 1
				set.Left[i] = OverlappableNeighbor{
					RelY4:              int8(y),
					Span4:              uint8(min(step, h4-y)),
					Size:               l.bs,
					Motion:             refmvsNeighborMotion(l),
					MotionValid:        true,
					InterpFilters:      c.LeftInterp[slot],
					InterpFiltersValid: c.LeftInterpValid[slot] != 0,
				}
				i++
				set.LeftCount = uint8(i)
			}
			y += step
		}
	}
	return set
}
