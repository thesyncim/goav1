package encoder

import "unsafe"

// svt_hme.go is SVT-AV1's hierarchical ME for one 64x64 superblock at
// preset 12 rtc, the search center of the open-loop ME sweep
// (pframe_depth_removal.go):
//
//   - Pyramid: Source/Lib/Codec/pic_analysis_process.c
//     svt_aom_downsample_filtering_input_picture with HME level 1 on: the
//     "quarter" (half-resolution) picture is a 2x2 0-phase average of the
//     source and the "sixteenth" (quarter-resolution) one a 2x2 average of
//     the quarter (svt_aom_downsample_2d_c). Like SVT the reference pyramid
//     is the previous frame's source pyramid (open loop).
//   - Levels and areas: svt_me_ctrl.go (camera content: levels 0 and 1, no
//     pre-HME; screen content adds pre-HME and level 2). hme_search_method
//     is SUB_SAD_SEARCH (every other row, doubled).
//   - Level 0 (motion_estimation.c hme_level0_b64 / hme_level_0): the 16x16
//     sixteenth-resolution block searched over num_hme_sa 2x2 regions. The
//     total area is set_hme_search_params' hme_l0_sa (camera content: sa_min
//     {8, 8}, sa_max {96, 96}); get_hme_l0_search_area scales it by
//     svt_aom_get_scaled_picture_distance of the reference distance: each
//     region is 16 wide (rounded up to 16) by 4, 8 or 12 tall at distance
//     1, 2 or 4 (the L1T3 leaf, middle layer and base).
//   - Level 1 (hme_level1_b64 / hme_level_1): each region's level-0 vector
//     refined on the quarter picture over hme_l1_sa {8, 3}.
//   - The center is the region with the smallest last-level SAD, scanned
//     (0,0), (1,0), (0,1), (1,1) with strict '<'.
//   - me_early_exit_th: when the SB's zero-motion SAD is below
//     me_early_exit_th >> 2, HME is skipped and the center is (0, 0).
//
// The pyramid planes carry replicated borders like SVT's padded pictures, so
// partial superblocks and edge positions search the padding.

const (
	svtHMEL1W = 8 // hme_l1_sa.width
	svtHMEL1H = 3 // hme_l1_sa.height
)

// Pyramid borders (SVT pads its quarter and sixteenth pictures,
// svt_aom_generate_padding): wide enough for a partial superblock's block
// and the full HME search around it.
const (
	svtHalfBorder = 64
	svtSixBorder  = 32
)

// svtPlane is one padded pyramid plane: w x h samples at pix[org], stride
// w + 2*border, edges replicated into the border.
type svtPlane struct {
	pix          []byte
	w, h, stride int
	border, org  int
	// colSum holds the same-parity column prefix sums of the padded plane
	// (buildColSums) when colSumOK: colSum[(r+2)*stride+c] is the sum of
	// pix rows r, r-2, r-4, ... of column c, rows 0 and 1 zero. A SUB_SAD
	// block's reference total is then two loads per column
	// (svtHMESearchColSum).
	colSum   []uint16
	colSumOK bool
}

func (q *svtPlane) size(w, h, border int) {
	q.w, q.h, q.border = w, h, border
	q.stride = w + 2*border
	q.org = border*q.stride + border
	if n := q.stride * (h + 2*border); len(q.pix) < n {
		q.pix = make([]byte, n)
	}
	q.colSumOK = false
}

// buildColSums fills colSum from the padded plane. The sums wrap in uint16;
// a block's column total (at most 16 rows of 255) is the exact difference
// of two of them.
func (q *svtPlane) buildColSums() {
	rows := q.h + 2*q.border
	// 16 slack samples: the colsum loads of svtHMESearchColSum read eight
	// columns at a time and may pass the last column of the last row.
	n := q.stride*(rows+2) + 16
	if len(q.colSum) < n {
		q.colSum = make([]uint16, n)
	}
	clear(q.colSum[:2*q.stride])
	svtColSumRowsArch(q.colSum, q.pix, q.stride, rows)
	q.colSumOK = true
}

// svtColSumRows is the scalar definition of svtColSumRowsArch.
func svtColSumRows(cs []uint16, pix []byte, stride, rows int) {
	for r := 0; r < rows; r++ {
		prev := cs[r*stride : r*stride+stride]
		next := cs[(r+2)*stride : (r+2)*stride+stride]
		row := pix[r*stride : r*stride+stride]
		for c := range row {
			next[c] = prev[c] + uint16(row[c])
		}
	}
}

// pad replicates the interior's edges into the border.
func (q *svtPlane) pad() {
	b, s := q.border, q.stride
	for y := 0; y < q.h; y++ {
		row := q.pix[(b+y)*s : (b+y+1)*s]
		memsetBytes(row[:b], row[b])
		memsetBytes(row[b+q.w:b+q.w+b], row[b+q.w-1])
	}
	top := q.pix[b*s : (b+1)*s]
	bot := q.pix[(b+q.h-1)*s : (b+q.h)*s]
	for y := 0; y < b; y++ {
		copy(q.pix[y*s:(y+1)*s], top)
		copy(q.pix[(b+q.h+y)*s:(b+q.h+y+1)*s], bot)
	}
}

// memsetBytesScalar sets every byte of dst to v, doubling the filled prefix
// with copy (memmove) past the first 16 bytes.
func memsetBytesScalar(dst []byte, v byte) {
	if len(dst) < 32 {
		for i := range dst {
			dst[i] = v
		}
		return
	}
	for i := range dst[:16] {
		dst[i] = v
	}
	for n := 16; n < len(dst); n *= 2 {
		copy(dst[n:], dst[:n])
	}
}

func (q *svtPlane) copyFrom(o *svtPlane) {
	q.size(o.w, o.h, o.border)
	n := q.stride * (q.h + 2*q.border)
	copy(q.pix[:n], o.pix[:n])
	if o.colSumOK {
		m := q.stride*(q.h+2*q.border+2) + 16
		if len(q.colSum) < m {
			q.colSum = make([]uint16, m)
		}
		copy(q.colSum[:m], o.colSum[:m])
		q.colSumOK = true
	}
}

// svtPyramid holds the current source pyramid, the pyramids of the
// reference slots' sources, and the active reference (prevHalf / prevSix,
// views of the selected slot).
type svtPyramid struct {
	curHalf, prevHalf svtPlane
	curSix, prevSix   svtPlane
	haveCur, havePrev bool
	slotHalf          [hmeRefSlots]svtPlane
	slotSix           [hmeRefSlots]svtPlane
	slotOK            [hmeRefSlots]bool
	// hint is the next build's precomputed pyramid (setHint), consumed by it.
	hint svtPyramidHint
	// halfFromHint marks the current half plane as copied from a hint.
	halfFromHint bool
	// l0W, l0H are the level-0 region size of the frame's reference
	// distance (setDistance).
	l0W, l0H int
}

// svtHMENumSA is num_hme_sa_w / num_hme_sa_h: 2x2 HME level-0 regions.
const svtHMENumSA = 2

// svtScaledPictureDistance is svt_aom_get_scaled_picture_distance.
func svtScaledPictureDistance(dist int) int {
	roundUp := 1
	if dist%8 == 0 {
		roundUp = 0
	}
	return dist*5/8 + roundUp
}

// setDistance is get_hme_l0_search_area for a reference dist pictures away:
// the per-region width (rounded up to 16) and height scaled by the scaled
// distance, capped at the sa_max region.
func (p *svtPyramid) setDistance(dist int, l0Min, l0Max svtSearchArea) {
	f := svtScaledPictureDistance(max(dist, 1))
	w := l0Min.w / svtHMENumSA
	p.l0W = min((w*f+15)&^15, (l0Max.w/svtHMENumSA+15)&^15)
	h := l0Min.h / svtHMENumSA
	p.l0H = min(h*f, l0Max.h/svtHMENumSA)
}

// svtDownsample2D is svt_aom_downsample_2d_c with decim_step 2.
func svtDownsample2D(dst []byte, dstStride int, src []byte, srcStride, w, h int) {
	for y := 0; y < h; y++ {
		r0 := src[(2*y)*srcStride:]
		r1 := src[(2*y+1)*srcStride:]
		d := dst[y*dstStride : y*dstStride+w]
		for x := range d {
			s := uint32(r0[2*x]) + uint32(r0[2*x+1]) + uint32(r1[2*x]) + uint32(r1[2*x+1])
			d[x] = uint8((s + 2) >> 2)
		}
	}
}

func buildPyramid(half, six *svtPlane, src SourceFrame420, hint svtPyramidHint) (fromHint bool) {
	hw, hh := src.Width/2, src.Height/2
	half.size(hw, hh, svtHalfBorder)
	six.size(hw/2, hh/2, svtSixBorder)
	if !hint.matches(src) {
		hint = svtPyramidHint{}
	}
	if hint.half.Width == hw && hint.half.Height == hh {
		copyPlaneRows(half.pix[half.org:], half.stride, hint.half.Y, hint.half.YStride, hw, hh)
		fromHint = true
	} else {
		svtDownsample2DArch(half.pix[half.org:], half.stride, src.Y, src.YStride, hw, hh)
		hint.six = SourceFrame420{}
	}
	if hint.six.Width == hw/2 && hint.six.Height == hh/2 {
		copyPlaneRows(six.pix[six.org:], six.stride, hint.six.Y, hint.six.YStride, hw/2, hh/2)
	} else {
		svtDownsample2DArch(six.pix[six.org:], six.stride, half.pix[half.org:], half.stride, hw/2, hh/2)
	}
	half.pad()
	six.pad()
	if svtHMEColSums {
		half.buildColSums()
		six.buildColSums()
	}
	return fromHint
}

// build makes src the current pyramid.
func (p *svtPyramid) build(src SourceFrame420) {
	hint := p.hint
	p.hint = svtPyramidHint{}
	if src.Width/4 < 16 || src.Height/4 < 16 {
		p.halfFromHint = false
		p.haveCur, p.havePrev = false, false
		p.slotOK = [hmeRefSlots]bool{}
		return
	}
	p.halfFromHint = buildPyramid(&p.curHalf, &p.curSix, src, hint)
	p.haveCur = true
}

// storeRef moves the current pyramid into reference slot s (the slot's old
// planes become the next frame's scratch).
func (p *svtPyramid) storeRef(s int) {
	if !p.haveCur {
		p.slotOK[s] = false
		return
	}
	p.slotHalf[s], p.curHalf = p.curHalf, p.slotHalf[s]
	p.slotSix[s], p.curSix = p.curSix, p.slotSix[s]
	p.slotOK[s] = true
	p.haveCur = false
}

// copyRef keeps a copy of the current pyramid as slot s's reference pyramid.
func (p *svtPyramid) copyRef(s int) {
	if !p.haveCur {
		p.slotOK[s] = false
		return
	}
	p.slotHalf[s].copyFrom(&p.curHalf)
	p.slotSix[s].copyFrom(&p.curSix)
	p.slotOK[s] = true
}

// selectRef makes slot s the active reference pyramid.
func (p *svtPyramid) selectRef(s int) {
	p.prevHalf, p.prevSix = p.slotHalf[s], p.slotSix[s]
	p.havePrev = p.slotOK[s] && p.prevHalf.w == p.curHalf.w && p.prevHalf.h == p.curHalf.h
}

// ready reports whether both pyramids exist for this frame.
func (p *svtPyramid) ready() bool { return p != nil && p.haveCur && p.havePrev }

// svtHMESearch is svt_sad_loop_kernel over a w x h area at origin
// (ox, oy) relative to the block at (bx, by): the SUB_SAD bw x bh block
// (bh/2 rows at doubled stride, SAD doubled), y outer, x inner, strict '<'.
// Like SVT's NEON/AVX2 kernels it scores eight positions per step
// (svtSAD16xhx8 per 16 columns) when the width is a multiple of 8, else
// four and then one at a time (the _small kernels), keeping the first
// minimum (update_best_sad_u32). The caller corrects the area to the padded
// plane (svtSearchAreaCorrect), so every position is scorable.
func svtHMESearch(cur, ref *svtPlane, bx, by, bw, bh, ox, oy, w, h int) (bestSAD uint32, bestX, bestY int) {
	bestSAD = ^uint32(0)
	stride := cur.stride
	rows := bh / 2
	if svtHMEColSums && ref.colSumOK && w%8 == 0 && (bw == 16 || bw == 32) && rows <= 16 && ref.stride == stride {
		return svtHMESearchColSum(cur, ref, bx, by, bw, bh, ox, oy, w, h)
	}
	s2 := 2 * stride
	s := cur.pix[cur.org+by*stride+bx:]
	for y := 0; y < h; y++ {
		base := ref.org + (by+oy+y)*stride + bx + ox
		x := 0
		if bw%16 != 0 {
			// A partial superblock's block narrower than the 16-column
			// kernels (svt_sad_loop_kernel's any-width path).
			for ; x < w; x++ {
				if sad := svtSADRectSub(s, s2, ref.pix[base+x:], bw, rows); sad < bestSAD {
					bestSAD, bestX, bestY = sad, ox+x, oy+y
				}
			}
			continue
		}
		if w%8 == 0 {
			for ; x < w; x += 8 {
				var t [8]uint32
				c := 0
				for ; c+32 <= bw; c += 32 {
					svtSAD32xhx8(&t, s[c:], s2, ref.pix[base+x+c:], s2, rows)
				}
				for ; c < bw; c += 16 {
					svtSAD16xhx8(&t, s[c:], s2, ref.pix[base+x+c:], s2, rows)
				}
				// update_best_sad_u32: the eight positions' minimum first,
				// its first lane only when it improves the best.
				m := min(min(min(t[0], t[1]), min(t[2], t[3])), min(min(t[4], t[5]), min(t[6], t[7])))
				if sad := m << 1; sad < bestSAD {
					k := 0
					for t[k] != m {
						k++
					}
					bestSAD, bestX, bestY = sad, ox+x+k, oy+y
				}
			}
			continue
		}
		for ; x+4 <= w; x += 4 {
			var t0, t1, t2, t3 int
			for c := 0; c < bw; c += 16 {
				a0, a1, a2, a3 := sad16xNx4(s[c:], s2, ref.pix[base+x+c:], s2, rows)
				t0, t1, t2, t3 = t0+a0, t1+a1, t2+a2, t3+a3
			}
			for k, t := range [4]int{t0, t1, t2, t3} {
				if sad := uint32(t) << 1; sad < bestSAD {
					bestSAD, bestX, bestY = sad, ox+x+k, oy+y
				}
			}
		}
		for ; x < w; x++ {
			sad := svtSADRectSub(s, s2, ref.pix[base+x:], bw, rows)
			if sad < bestSAD {
				bestSAD, bestX, bestY = sad, ox+x, oy+y
			}
		}
	}
	return bestSAD, bestX, bestY
}

// svtHMESearchColSum is svtHMESearch's eight-position path on the identity
//
//	sum |a - b| = sum a + sum b - 2 * sum min(a, b):
//
// every row and position costs one MIN plus the byte-pair accumulate instead
// of MAX, MIN, SUB plus the accumulate. The source block's total is shared
// by every position, and each position's reference total comes from the
// reference plane's same-parity column sums (svtPlane.colSum): per search
// row, the block's column totals are bottom minus top prefix rows, and the
// eight positions' window totals are their sliding sums. The SADs are equal
// to svtSAD16xhx8's / svtSAD32xhx8's, so the search (y outer, x inner,
// first strict minimum) picks the same position.
func svtHMESearchColSum(cur, ref *svtPlane, bx, by, bw, bh, ox, oy, w, h int) (bestSAD uint32, bestX, bestY int) {
	bestSAD = ^uint32(0)
	stride := cur.stride
	rows := bh / 2
	s2 := 2 * stride
	s := cur.pix[cur.org+by*stride+bx:]
	srcSum := svtBlockSumSubArch(s, s2, bw, rows)
	// Position (x, y) reads the reference rows r0+y, r0+y+2, ...,
	// r0+y+2*(rows-1) from column c0+x (eight positions read bw+7 columns),
	// and column c of those rows totals colSum[(r0+y+2*rows)*stride+c] -
	// colSum[(r0+y)*stride+c] (eight columns loaded at a time, bw+8).
	r0 := ref.border + by + oy
	c0 := ref.border + bx + ox
	_ = s[(rows-1)*s2+bw-1]
	_ = ref.pix[(r0+h-1+2*(rows-1))*stride+c0+w+bw-2]
	_ = ref.colSum[(r0+h-1+2*rows)*stride+c0+w+bw-1]
	sp := unsafe.Pointer(&s[0])
	for y := 0; y < h; y++ {
		top := (r0+y)*stride + c0
		bot := top + 2*rows*stride
		for x := 0; x < w; x += 8 {
			var t [8]uint32
			svtHMEEightColSum(&t, sp, unsafe.Pointer(&ref.pix[top+x]), unsafe.Pointer(&ref.colSum[top+x]), unsafe.Pointer(&ref.colSum[bot+x]), s2, bw, rows, srcSum)
			m := min(min(min(t[0], t[1]), min(t[2], t[3])), min(min(t[4], t[5]), min(t[6], t[7])))
			if sad := m << 1; sad < bestSAD {
				k := 0
				for t[k] != m {
					k++
				}
				bestSAD, bestX, bestY = sad, ox+x+k, oy+y
			}
		}
	}
	return bestSAD, bestX, bestY
}

// svtBlockSumSub is the sum of the bw x rows block at s (row stride s2).
func svtBlockSumSub(s []byte, s2, bw, rows int) uint32 {
	var t uint32
	for r := 0; r < rows; r++ {
		for _, v := range s[r*s2 : r*s2+bw] {
			t += uint32(v)
		}
	}
	return t
}

// svtHMEEightColSumScalar is the definition of svtHMEEightColSum: t[k] is
// the SAD of the bw-wide (16 or 32), rows-tall block at s (stride s2)
// against ref+k (stride s2), from the source total srcSum, the reference
// column totals bot[c] - top[c] and the sum of the per-sample minima.
func svtHMEEightColSumScalar(t *[8]uint32, s []byte, s2 int, ref []byte, top, bot []uint16, bw, rows int, srcSum uint32) {
	for k := range t {
		var q, mn uint32
		for c := k; c < k+bw; c++ {
			q += uint32(bot[c] - top[c])
		}
		for r := 0; r < rows; r++ {
			for c := 0; c < bw; c++ {
				mn += uint32(min(s[r*s2+c], ref[r*s2+k+c]))
			}
		}
		t[k] = srcSum + q - 2*mn
	}
}

// copySlotFrom copies q's reference slot s into p's (the pipelined leaf's
// reference, as the serial chain holds it).
func (p *svtPyramid) copySlotFrom(q *svtPyramid, s int) {
	p.slotOK[s] = q.slotOK[s]
	if !q.slotOK[s] {
		return
	}
	p.slotHalf[s].copyFrom(&q.slotHalf[s])
	p.slotSix[s].copyFrom(&q.slotSix[s])
}

// svtSAD16xhx8Scalar is the reference definition of svtSAD16xhx8: the SADs
// of the 16-wide, rows-tall source block against ref+0..ref+7.
func svtSAD16xhx8Scalar(src []byte, srcStride int, ref []byte, refStride, rows int) [8]uint32 {
	var out [8]uint32
	for k := range out {
		for r := 0; r < rows; r++ {
			a := src[r*srcStride : r*srcStride+16]
			b := ref[r*refStride+k : r*refStride+k+16]
			for c := range a {
				d := int(a[c]) - int(b[c])
				if d < 0 {
					d = -d
				}
				out[k] += uint32(d)
			}
		}
	}
	return out
}

// svtSADRectSub is the doubled SAD of the bw-wide, rows-tall block at
// doubled stride s2 (svt_nxm_sad_kernel at the SUB_SAD stride).
func svtSADRectSub(src []byte, s2 int, ref []byte, bw, rows int) uint32 {
	if (bw == 16 && rows == 8) || (bw == 32 && rows == 16) {
		return uint32(sadRectDualBlock(src, s2, ref, s2, bw, rows)) << 1
	}
	var sum uint32
	for r := 0; r < rows; r++ {
		a := src[r*s2 : r*s2+bw]
		b := ref[r*s2 : r*s2+bw]
		for i := range a {
			d := int(a[i]) - int(b[i])
			if d < 0 {
				d = -d
			}
			sum += uint32(d)
		}
	}
	return sum << 1
}

// svtPyramidHint carries pyramid planes computed elsewhere for the next
// build: in a spatial-layer stream the lower layers' box-filtered luma is,
// sample for sample, the upper layer's half and sixteenth planes
// (scale_box.go), so the upper layer copies them instead of filtering again.
// parent pins the source they were made from.
type svtPyramidHint struct {
	parent    *byte
	half, six SourceFrame420
}

// matches reports whether the hint was made from src.
func (h svtPyramidHint) matches(src SourceFrame420) bool {
	return h.parent != nil && len(src.Y) > 0 && h.parent == &src.Y[0]
}

// setHint hands the next build its precomputed planes.
func (p *svtPyramid) setHint(h svtPyramidHint) { p.hint = h }

func copyPlaneRows(dst []byte, dstStride int, src []byte, srcStride, w, h int) {
	for y := 0; y < h; y++ {
		copy(dst[y*dstStride:y*dstStride+w], src[y*srcStride:y*srcStride+w])
	}
}
