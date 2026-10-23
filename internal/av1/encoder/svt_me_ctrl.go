package encoder

import "math"

// svt_me_ctrl.go holds SVT-AV1's per-picture ME/HME controls at preset 12
// rtc (enc_mode_config.c svt_aom_sig_deriv_me, set_me_search_params,
// set_hme_search_params, svt_aom_sig_deriv_multi_processes_rtc's HME level
// flags) for the two content classes, and the hierarchical search of one
// 64x64 superblock that consumes them (motion_estimation.c hme_b64:
// prehme_b64, hme_level0_b64, hme_level1_b64, hme_level2_b64,
// set_final_search_centre_sb; hme_prune_ref_and_adjust_sr's search-region
// divisor; integer_search_b64's area derivation).
//
// Screen content (sc_class1) differs from camera content in:
//
//   - HME levels 1 and 2 on (camera: level 2 off at M12);
//   - hme_l0_sa {32, 32}-{192, 192} (camera {8, 8}-{96, 96}) and me_sa
//     {24, 16}-{32, 16} below 1080p (camera {8, 1}), both modulated by the
//     sequence qp (svt_aom_get_qp_based_th_scaling_factors, enabled at M12);
//   - pre-HME level 1: an 8-wide vertical and a 3-tall horizontal strip on
//     the sixteenth picture (camera: off);
//   - me_sr_adjustment level 5 (camera 3, inert on the 8x3 area);
//   - me_early_exit_th 64*64 (camera 64*64*12) and
//     prev_me_stage_based_exit_th 64*64*4 (camera 0).

// svtSearchArea is SearchArea {width, height}.
type svtSearchArea struct{ w, h int }

// svtMECtrl is one picture's ME/HME control set.
type svtMECtrl struct {
	screen        bool
	earlyExit     uint32 // me_early_exit_th
	prevStageExit uint32 // prev_me_stage_based_exit_th
	hmeL2         bool   // enable_hme_level2_flag
	prehme        bool   // prehme_ctrl.enable (level 1)
	l0Min, l0Max  svtSearchArea
	meMin, meMax  svtSearchArea
	// me_sr_adjustment_ctrls (level 3 for camera content, 5 for screen).
	srMVLenTH, srStationaryTH, srLowHMESADTH uint32
	srDivisor                                int
}

// prehme_ctrl level 1 search regions: [0] the vertical, [1] the horizontal.
var svtPrehmeSAMin = [2]svtSearchArea{{8, 144}, {144, 3}}
var svtPrehmeSAMax = [2]svtSearchArea{{8, 496}, {496, 3}}

// svtQPScale is svt_aom_get_qp_based_th_scaling_factors (scaling enabled).
func svtQPScale(qp int) (weight, denom uint32) {
	weight, denom = uint32(max(10, qp)), 63
	if qp >= 46 {
		ex := -(math.Max(40, float64(qp)) - 35) / 10
		weight, denom = uint32((1.05-math.Exp(ex))*10000), 10000
	}
	return weight, denom
}

func svtScaleArea(a svtSearchArea, weight, denom uint32, floorW, floorH int) svtSearchArea {
	round := func(v int) int { return int((uint32(v)*weight + denom>>1) / denom) }
	return svtSearchArea{max(floorW, round(a.w)), max(floorH, round(a.h))}
}

// svtMEControls derives the preset-12 rtc ME controls for a picture of the
// given resolution class. qp is the sequence qp (static_config.qp: 35 under
// CBR, the user qp under CQP).
func svtMEControls(screen bool, width, height, qp int) svtMECtrl {
	c := svtMECtrl{screen: screen}
	weight, denom := svtQPScale(qp)
	l0Min, l0Max := svtSearchArea{8, 8}, svtSearchArea{96, 96}
	meMin, meMax := svtSearchArea{8, 1}, svtSearchArea{8, 1}
	c.earlyExit = 64 * 64 * 12
	// me_sr_adjustment level 3 with HME level 2 off: thresholds / 4.
	c.srMVLenTH, c.srStationaryTH, c.srLowHMESADTH, c.srDivisor = 4, 12000/4, 12000/4, 8
	if screen {
		l0Min, l0Max = svtSearchArea{32, 32}, svtSearchArea{192, 192}
		if svtInputResolution(width, height) < svtInputSize1080pRange {
			meMin, meMax = svtSearchArea{24, 16}, svtSearchArea{32, 16}
		} else {
			meMin, meMax = svtSearchArea{16, 9}, svtSearchArea{16, 9}
		}
		c.earlyExit = 64 * 64
		c.prevStageExit = 64 * 64 * 4
		c.hmeL2 = true
		c.prehme = true
		c.srMVLenTH, c.srStationaryTH, c.srLowHMESADTH, c.srDivisor = 20, 24000, 24000, 8
	}
	c.l0Min = svtScaleArea(l0Min, weight, denom, 8, 8)
	c.l0Max = svtScaleArea(l0Max, weight, denom, 96, 96)
	c.meMin = svtScaleArea(meMin, weight, denom, 8, 3)
	c.meMax = svtScaleArea(meMax, weight, denom, 8, 3)
	return c
}

// hmeSearchInfo is SearchInfo: one pre-HME region's result.
type hmeSearchInfo struct {
	sad       uint32
	x, y      int
	performed bool
}

// prehme is prehme_b64 for the single reference: the two strip searches on
// the sixteenth pictures, each skipped (zero vector, zero SAD) when the
// zero-motion SAD is below me_early_exit_th.
func (p *svtPyramid) prehme(c *svtMECtrl, px, py, sbW, sbH int, zz uint32, f int) [2]hmeSearchInfo {
	var out [2]hmeSearchInfo
	six := &p.curSix
	for sr := range 2 {
		if zz < c.earlyExit {
			continue
		}
		w := min(svtPrehmeSAMin[sr].w*f, svtPrehmeSAMax[sr].w)
		h := min(svtPrehmeSAMin[sr].h*f, svtPrehmeSAMax[sr].h)
		ox, oy, w, h := svtSearchAreaCorrect(px>>2, py>>2, -(w >> 1), -(h >> 1), w, h, six.w, six.h, six.border-1, false)
		sad, x, y := svtHMESearch(six, &p.prevSix, px>>2, py>>2, sbW>>2, sbH>>2, ox, oy, w, h)
		out[sr] = hmeSearchInfo{sad: sad, x: x * 4, y: y * 4, performed: true}
	}
	return out
}

// hmeB64 is hme_b64 plus set_final_search_centre_sb for the sbW x sbH
// superblock at (px, py) against the single reference: it returns the full-pel
// search center and its SAD (hme_sad). cur and ref are the padded full-
// resolution source pictures level 2 searches.
func (p *svtPyramid) hmeB64(c *svtMECtrl, cur, ref *svtPlane, px, py, sbW, sbH int, zz uint32, f int) (int, int, uint32) {
	var phme [2]hmeSearchInfo
	if c.prehme {
		phme = p.prehme(c, px, py, sbW, sbH, zz, f)
	}
	srBest := 0
	if phme[1].sad < phme[0].sad {
		srBest = 1
	}
	var l0SAD, l1SAD, l2SAD [4]uint32
	var l0X, l0Y, l1X, l1Y, l2X, l2Y [4]int
	six, half := &p.curSix, &p.curHalf
	early := zz < c.earlyExit>>2
	switch {
	case early:
		// hme_level0_b64: zero centers and SADs.
	case c.prevStageExit != 0 && phme[srBest].performed && phme[srBest].sad < c.prevStageExit>>4:
		for r := range 4 {
			l0SAD[r], l0X[r], l0Y[r] = phme[srBest].sad, phme[srBest].x, phme[srBest].y
		}
	default:
		l0W, l0H := p.l0W, p.l0H
		for r := 0; r < 4; r++ {
			srW, srH := r&1, r>>1
			// hme_level_0: sa_origin = -(sa * num_hme_sa) / 2 + sa * sr.
			saW, saH := (l0W+7)&^7, l0H
			ox := -(saW*svtHMENumSA)>>1 + saW*srW
			oy := -(saH*svtHMENumSA)>>1 + saH*srH
			ox, oy, w, h := svtSearchAreaCorrect(px>>2, py>>2, ox, oy, saW, saH, six.w, six.h, six.border-1, true)
			sad, x0, y0 := svtHMESearch(six, &p.prevSix, px>>2, py>>2, sbW>>2, sbH>>2, ox, oy, w, h)
			l0SAD[r], l0X[r], l0Y[r] = sad, x0*4, y0*4
		}
		if c.prehme {
			// get_worst_quadrant, then the better pre-HME region replaces it.
			worst, maxSAD := 0, uint32(0)
			for _, r := range [4]int{0, 1, 2, 3} {
				if l0SAD[r] > maxSAD {
					maxSAD, worst = l0SAD[r], r
				}
			}
			if phme[srBest].sad < l0SAD[worst] {
				l0SAD[worst], l0X[worst], l0Y[worst] = phme[srBest].sad, phme[srBest].x, phme[srBest].y
			}
		}
	}
	if !early {
		for r := 0; r < 4; r++ {
			if c.prevStageExit != 0 && l0SAD[r] < c.prevStageExit>>5 {
				l1SAD[r], l1X[r], l1Y[r] = l0SAD[r], l0X[r], l0Y[r]
				continue
			}
			// hme_level_1 around the level-0 vector (>>1 to the quarter
			// picture).
			cx, cy := l0X[r]>>1, l0Y[r]>>1
			ox, oy, w, h := svtSearchAreaCorrect(px>>1, py>>1, cx-(svtHMEL1W>>1), cy-(svtHMEL1H>>1), svtHMEL1W, svtHMEL1H, half.w, half.h, half.border-1, true)
			sad, x1, y1 := svtHMESearch(half, &p.prevHalf, px>>1, py>>1, sbW>>1, sbH>>1, ox, oy, w, h)
			l1SAD[r], l1X[r], l1Y[r] = sad, x1*2, y1*2
		}
	}
	sads, xs, ys := &l1SAD, &l1X, &l1Y
	if c.hmeL2 {
		for r := 0; r < 4; r++ {
			if c.prevStageExit != 0 && l1SAD[r] < c.prevStageExit>>2 {
				l2SAD[r], l2X[r], l2Y[r] = l1SAD[r], l1X[r], l1Y[r]
				continue
			}
			// hme_level_2: hme_l2_sa {8, 3} around the level-1 vector on the
			// full-resolution pictures, padded by BLOCK_SIZE_64 - 1.
			ox, oy, w, h := svtSearchAreaCorrect(px, py, l1X[r]-(svtHMEL1W>>1), l1Y[r]-(svtHMEL1H>>1), svtHMEL1W, svtHMEL1H, cur.w, cur.h, 63, true)
			l2SAD[r], l2X[r], l2Y[r] = svtHMESearch(cur, ref, px, py, sbW, sbH, ox, oy, w, h)
		}
		sads, xs, ys = &l2SAD, &l2X, &l2Y
	}
	best := 0
	for _, r := range [3]int{1, 2, 3} {
		if sads[r] < sads[best] {
			best = r
		}
	}
	return xs[best], ys[best], sads[best]
}

// meArea is integer_search_b64's search-area derivation before the 8x8
// variance step and the picture-edge correction: me_sa scaled by the
// picture distance, the hme_prune_ref_and_adjust_sr divisor, and the 1x1
// area below me_early_exit_th / 6.
func (c *svtMECtrl) meArea(f, hmeX, hmeY int, hmeSAD, zz uint32) (int, int) {
	w := min(c.meMin.w*f, c.meMax.w)
	h := min(c.meMin.h*f, c.meMax.h)
	div := 1
	if (uint32(abs(hmeX)) <= c.srMVLenTH && uint32(abs(hmeY)) <= c.srMVLenTH && hmeSAD < c.srStationaryTH) ||
		hmeSAD < c.srLowHMESADTH {
		div = c.srDivisor
	}
	w = (max(1, w/div) + 7) &^ 7
	h = max(3, h/div)
	if zz < c.earlyExit/6 {
		return 1, 1
	}
	return w, h
}

// svtSearchAreaCorrect is the search-area correction shared by hme_level_0,
// hme_level_1, hme_level_2 and prehme_core, statement for statement: the
// w x h area at origin (ox, oy) of the block at (orgX, orgY) is kept on the
// padded picture (pad samples) of width picW and height picH. The left/top
// width shrink reads the already-corrected origin (so it never fires), the
// right/bottom shrink keeps at least one position, and the HME levels (not
// pre-HME) round a width of 8 or more down to a multiple of 8.
func svtSearchAreaCorrect(orgX, orgY, ox, oy, w, h, picW, picH, pad int, round8 bool) (int, int, int, int) {
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
	if round8 && w >= 8 {
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
