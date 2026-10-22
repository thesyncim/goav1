package encoder

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
//   - Levels: svt_aom_sig_deriv_multi_processes_rtc enables HME levels 0 and
//     1 and disables level 2 at M12 (non-flat). hme_search_method is
//     SUB_SAD_SEARCH (every other row, doubled).
//   - Level 0 (motion_estimation.c hme_level0_b64 / hme_level_0): the 16x16
//     sixteenth-resolution block searched over num_hme_sa 2x2 regions. The
//     total area is set_hme_search_params' rtc non-sc hme_l0_sa.sa_min {8, 8}
//     (sa_max {96, 96}); get_hme_l0_search_area scales it by
//     svt_aom_get_scaled_picture_distance of the reference distance: each
//     region is 16 wide (rounded up to 16) by 4, 8 or 12 tall at distance
//     1, 2 or 4 (the L1T3 leaf, middle layer and base).
//   - Level 1 (hme_level1_b64 / hme_level_1): each region's level-0 vector
//     refined on the quarter picture over hme_l1_sa {8, 3}.
//   - The center is the region with the smallest level-1 SAD, scanned
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
}

func (q *svtPlane) size(w, h, border int) {
	q.w, q.h, q.border = w, h, border
	q.stride = w + 2*border
	q.org = border*q.stride + border
	if n := q.stride * (h + 2*border); len(q.pix) < n {
		q.pix = make([]byte, n)
	}
}

// pad replicates the interior's edges into the border.
func (q *svtPlane) pad() {
	b, s := q.border, q.stride
	for y := 0; y < q.h; y++ {
		row := q.pix[(b+y)*s : (b+y+1)*s]
		l, r := row[b], row[b+q.w-1]
		for x := 0; x < b; x++ {
			row[x] = l
			row[b+q.w+x] = r
		}
	}
	top := q.pix[b*s : (b+1)*s]
	bot := q.pix[(b+q.h-1)*s : (b+q.h)*s]
	for y := 0; y < b; y++ {
		copy(q.pix[y*s:(y+1)*s], top)
		copy(q.pix[(b+q.h+y)*s:(b+q.h+y+1)*s], bot)
	}
}

func (q *svtPlane) copyFrom(o *svtPlane) {
	q.size(o.w, o.h, o.border)
	n := q.stride * (q.h + 2*q.border)
	copy(q.pix[:n], o.pix[:n])
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
	// l0W, l0H are the level-0 region size of the frame's reference
	// distance (setDistance).
	l0W, l0H int
}

// HME level-0 search area at preset 12 rtc non-sc (set_hme_search_params):
// hme_l0_sa.sa_min {8, 8}, sa_max {96, 96}, over num_hme_sa 2x2 regions.
const (
	svtHMEL0SAMin = 8
	svtHMEL0SAMax = 96
	svtHMENumSA   = 2
)

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
func (p *svtPyramid) setDistance(dist int) {
	f := svtScaledPictureDistance(max(dist, 1))
	w := svtHMEL0SAMin / svtHMENumSA
	p.l0W = min((w*f+15)&^15, (svtHMEL0SAMax/svtHMENumSA+15)&^15)
	h := svtHMEL0SAMin / svtHMENumSA
	p.l0H = min(h*f, svtHMEL0SAMax/svtHMENumSA)
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

func buildPyramid(half, six *svtPlane, src SourceFrame420) {
	hw, hh := src.Width/2, src.Height/2
	half.size(hw, hh, svtHalfBorder)
	six.size(hw/2, hh/2, svtSixBorder)
	svtDownsample2DArch(half.pix[half.org:], half.stride, src.Y, src.YStride, hw, hh)
	svtDownsample2DArch(six.pix[six.org:], six.stride, half.pix[half.org:], half.stride, hw/2, hh/2)
	half.pad()
	six.pad()
}

// build makes src the current pyramid.
func (p *svtPyramid) build(src SourceFrame420) {
	if src.Width/4 < 16 || src.Height/4 < 16 {
		p.haveCur, p.havePrev = false, false
		p.slotOK = [hmeRefSlots]bool{}
		return
	}
	buildPyramid(&p.curHalf, &p.curSix, src)
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
// plane (svtHMECorrectArea), so every position is scorable.
func svtHMESearch(cur, ref *svtPlane, bx, by, bw, bh, ox, oy, w, h int) (bestSAD uint32, bestX, bestY int) {
	bestSAD = ^uint32(0)
	stride := cur.stride
	rows := bh / 2
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
				for c := 0; c < bw; c += 16 {
					svtSAD16xhx8(&t, s[c:], s2, ref.pix[base+x+c:], s2, rows)
				}
				for k, v := range t {
					if sad := v << 1; sad < bestSAD {
						bestSAD, bestX, bestY = sad, ox+x+k, oy+y
					}
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

// svtHMECorrectArea is hme_level_0 / hme_level_1's search-area correction,
// statement for statement: the w x h area at origin (ox, oy) of the block
// at (orgX, orgY) is kept on the padded picture (pad = border - 1) of
// width picW and height picH. The left/top width shrink reads the
// already-corrected origin (so it never fires), the right/bottom shrink
// keeps at least one position, and a width of 8 or more is rounded down to
// a multiple of 8.
func svtHMECorrectArea(orgX, orgY, ox, oy, w, h, picW, picH, pad int) (int, int, int, int) {
	if orgX+ox < -pad {
		ox = -pad - orgX
	}
	if orgX+ox < -pad {
		w -= -pad - (orgX + ox)
	}
	if orgX+ox > picW-1 {
		ox -= (orgX + ox) - (picW - 1)
	}
	if orgX+ox+w > picW {
		w = max(1, w-((orgX+ox+w)-picW))
	}
	if w >= 8 {
		w &^= 7
	}
	if orgY+oy < -pad {
		oy = -pad - orgY
	}
	if orgY+oy < -pad {
		h -= -pad - (orgY + oy)
	}
	if orgY+oy > picH-1 {
		oy -= (orgY + oy) - (picH - 1)
	}
	if orgY+oy+h > picH {
		h = max(1, h-((orgY+oy+h)-picH))
	}
	return ox, oy, w, h
}

// center returns the HME search center (full-pel) of the sbW x sbH
// superblock at (px, py) (b64_width x b64_height, 8-aligned) and its
// level-1 SAD. Level 0 searches the (sbW>>2) x (sbH>>2) block, level 1 the
// (sbW>>1) x (sbH>>1) one.
func (p *svtPyramid) center(px, py, sbW, sbH int) (int, int, uint32) {
	var l1SAD [4]uint32
	var l1X, l1Y [4]int
	six, half := &p.curSix, &p.curHalf
	l0W, l0H := p.l0W, p.l0H
	for r := 0; r < 4; r++ {
		srW, srH := r&1, r>>1
		// hme_level_0: sa_origin = -(sa * num_hme_sa) / 2 + sa * sr.
		saW, saH := (l0W+7)&^7, l0H
		ox := -(saW*svtHMENumSA)>>1 + saW*srW
		oy := -(saH*svtHMENumSA)>>1 + saH*srH
		ox, oy, w, h := svtHMECorrectArea(px>>2, py>>2, ox, oy, saW, saH, six.w, six.h, six.border-1)
		_, x0, y0 := svtHMESearch(six, &p.prevSix, px>>2, py>>2, sbW>>2, sbH>>2, ox, oy, w, h)
		// hme_level_1 around the level-0 vector (x4 to full-pel, >>1 to the
		// quarter picture).
		cx, cy := (x0*4)>>1, (y0*4)>>1
		ox, oy, w, h = svtHMECorrectArea(px>>1, py>>1, cx-(svtHMEL1W>>1), cy-(svtHMEL1H>>1), svtHMEL1W, svtHMEL1H, half.w, half.h, half.border-1)
		sad1, x1, y1 := svtHMESearch(half, &p.prevHalf, px>>1, py>>1, sbW>>1, sbH>>1, ox, oy, w, h)
		l1SAD[r], l1X[r], l1Y[r] = sad1, x1*2, y1*2
	}
	best := 0
	for _, r := range [3]int{1, 2, 3} {
		if l1SAD[r] < l1SAD[best] {
			best = r
		}
	}
	return l1X[best], l1Y[best], l1SAD[best]
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
