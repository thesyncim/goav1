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
//     (sa_max {96, 96}); get_hme_l0_search_area at picture distance 1 makes
//     each region 16 wide (rounded up to 16) by 4 tall.
//   - Level 1 (hme_level1_b64 / hme_level_1): each region's level-0 vector
//     refined on the quarter picture over hme_l1_sa {8, 3}.
//   - The center is the region with the smallest level-1 SAD, scanned
//     (0,0), (1,0), (0,1), (1,1) with strict '<'.
//   - me_early_exit_th: when the SB's zero-motion SAD is below
//     me_early_exit_th >> 2, HME is skipped and the center is (0, 0).
//
// SVT's pictures are padded; goav1's planes are not, so candidate positions
// whose block leaves the plane are not scored.

const (
	svtHMEL0RegionW = 16 // per-region level-0 width at distance 1
	svtHMEL0RegionH = 4  // per-region level-0 height at distance 1
	svtHMEL1W       = 8  // hme_l1_sa.width
	svtHMEL1H       = 3  // hme_l1_sa.height
)

// svtPyramid holds the current and previous source pyramids.
type svtPyramid struct {
	halfW, halfH int
	sixW, sixH   int
	curHalf      []byte
	prevHalf     []byte
	curSix       []byte
	prevSix      []byte
	haveCur      bool
	havePrev     bool
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

// build makes src the current pyramid level; with prime it becomes the
// previous one instead (the keyframe that the next inter frame references).
func (p *svtPyramid) build(src SourceFrame420, prime bool) {
	hw, hh := src.Width/2, src.Height/2
	sw, sh := hw/2, hh/2
	if sw < 16 || sh < 16 {
		p.haveCur, p.havePrev = false, false
		return
	}
	if len(p.curHalf) < hw*hh {
		p.curHalf = make([]byte, hw*hh)
		p.prevHalf = make([]byte, hw*hh)
		p.curSix = make([]byte, sw*sh)
		p.prevSix = make([]byte, sw*sh)
	}
	if p.halfW != hw || p.halfH != hh {
		p.haveCur, p.havePrev = false, false
	}
	p.halfW, p.halfH, p.sixW, p.sixH = hw, hh, sw, sh
	if prime {
		svtDownsample2DArch(p.prevHalf, hw, src.Y, src.YStride, hw, hh)
		svtDownsample2DArch(p.prevSix, sw, p.prevHalf, hw, sw, sh)
		p.havePrev, p.haveCur = true, false
		return
	}
	if p.haveCur {
		p.curHalf, p.prevHalf = p.prevHalf, p.curHalf
		p.curSix, p.prevSix = p.prevSix, p.curSix
		p.havePrev = true
	}
	svtDownsample2DArch(p.curHalf, hw, src.Y, src.YStride, hw, hh)
	svtDownsample2DArch(p.curSix, sw, p.curHalf, hw, sw, sh)
	p.haveCur = true
}

// ready reports whether both pyramids exist for this frame.
func (p *svtPyramid) ready() bool { return p != nil && p.haveCur && p.havePrev }

// svtHMESearch is svt_sad_loop_kernel over a w x h area at origin
// (ox, oy) relative to the block at (bx, by): the SUB_SAD bw x bh block
// (bh/2 rows at doubled stride, SAD doubled), y outer, x inner, strict '<'.
func svtHMESearch(src, ref []byte, stride, planeW, planeH, bx, by, bw, bh, ox, oy, w, h int) (bestSAD uint32, bestX, bestY int) {
	bestSAD = ^uint32(0)
	s := src[by*stride+bx:]
	rows := bh / 2
	s2 := 2 * stride
	for y := 0; y < h; y++ {
		ry := by + oy + y
		if ry < 0 || ry+bh > planeH {
			continue
		}
		for x := 0; x < w; {
			rx := bx + ox + x
			// Four positions per kernel call when all four are scorable.
			if x+4 <= w && rx >= 0 && rx+3+bw <= planeW && ry*stride+rx+(2*rows-2)*stride+bw+3 <= len(ref) {
				var t0, t1, t2, t3 int
				for c := 0; c < bw; c += 16 {
					a0, a1, a2, a3 := sad16xNx4(s[c:], s2, ref[ry*stride+rx+c:], s2, rows)
					t0, t1, t2, t3 = t0+a0, t1+a1, t2+a2, t3+a3
				}
				for k, t := range [4]int{t0, t1, t2, t3} {
					if sad := uint32(t) << 1; sad < bestSAD {
						bestSAD, bestX, bestY = sad, ox+x+k, oy+y
					}
				}
				x += 4
				continue
			}
			if rx >= 0 && rx+bw <= planeW {
				sad := uint32(sadRectDualBlock(s, s2, ref[ry*stride+rx:], s2, bw, rows)) << 1
				if sad < bestSAD {
					bestSAD, bestX, bestY = sad, ox+x, oy+y
				}
			}
			x++
		}
	}
	return bestSAD, bestX, bestY
}

// center returns the HME search center (full-pel) of the SB at (px, py).
func (p *svtPyramid) center(px, py int) (int, int) {
	var l1SAD [4]uint32
	var l1X, l1Y [4]int
	for r := 0; r < 4; r++ {
		srW, srH := r&1, r>>1
		// hme_level_0: sa_origin = -(sa * num_hme_sa) / 2 + sa * sr.
		ox := -(svtHMEL0RegionW*2)>>1 + svtHMEL0RegionW*srW
		oy := -(svtHMEL0RegionH*2)>>1 + svtHMEL0RegionH*srH
		sad0, x0, y0 := svtHMESearch(p.curSix, p.prevSix, p.sixW, p.sixW, p.sixH, px>>2, py>>2, 16, 16, ox, oy, svtHMEL0RegionW, svtHMEL0RegionH)
		if sad0 == ^uint32(0) {
			l1SAD[r] = sad0
			continue
		}
		// hme_level_1 around the level-0 vector (x4 to full-pel, >>1 to the
		// quarter picture).
		cx, cy := (x0*4)>>1, (y0*4)>>1
		sad1, x1, y1 := svtHMESearch(p.curHalf, p.prevHalf, p.halfW, p.halfW, p.halfH, px>>1, py>>1, 32, 32,
			cx-(svtHMEL1W>>1), cy-(svtHMEL1H>>1), svtHMEL1W, svtHMEL1H)
		if sad1 == ^uint32(0) {
			l1SAD[r], l1X[r], l1Y[r] = ^uint32(0)-1, cx*2, cy*2
			continue
		}
		l1SAD[r], l1X[r], l1Y[r] = sad1, x1*2, y1*2
	}
	best := 0
	for _, r := range [3]int{1, 2, 3} {
		if l1SAD[r] < l1SAD[best] {
			best = r
		}
	}
	if l1SAD[best] == ^uint32(0) {
		return 0, 0
	}
	return l1X[best], l1Y[best]
}

// ensure sizes p's planes like q's.
func (p *svtPyramid) ensure(q *svtPyramid) {
	hn, sn := q.halfW*q.halfH, q.sixW*q.sixH
	if len(p.curHalf) < hn || len(p.curSix) < sn {
		p.curHalf, p.prevHalf = make([]byte, hn), make([]byte, hn)
		p.curSix, p.prevSix = make([]byte, sn), make([]byte, sn)
	}
	p.halfW, p.halfH, p.sixW, p.sixH = q.halfW, q.halfH, q.sixW, q.sixH
}

// primeFrom makes q's current pyramid p's previous one (the pipelined
// leaf's reference, as the serial swap chain would hold it).
func (p *svtPyramid) primeFrom(q *svtPyramid) {
	// q's most recent pyramid: its current one, or right after a keyframe
	// prime its previous one.
	half, six := q.curHalf, q.curSix
	switch {
	case q.haveCur:
	case q.havePrev:
		half, six = q.prevHalf, q.prevSix
	default:
		p.haveCur, p.havePrev = false, false
		return
	}
	p.ensure(q)
	copy(p.prevHalf, half[:q.halfW*q.halfH])
	copy(p.prevSix, six[:q.sixW*q.sixH])
	p.havePrev, p.haveCur = true, false
}

// adoptCurFrom makes q's current pyramid p's current one.
func (p *svtPyramid) adoptCurFrom(q *svtPyramid) {
	if !q.haveCur {
		return
	}
	p.ensure(q)
	copy(p.curHalf, q.curHalf[:q.halfW*q.halfH])
	copy(p.curSix, q.curSix[:q.sixW*q.sixH])
	p.haveCur = true
}
