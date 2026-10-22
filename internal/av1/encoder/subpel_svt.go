package encoder

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// subpel_svt.go is SVT-AV1's MD sub-pel refinement of the ME motion vector,
// ported from:
//
//   - Source/Lib/Codec/product_coding_loop.c read_refine_me_mvs_light_pd1 and
//     md_subpel_search: every block's full-pel ME vector of each reference is
//     refined before the MDS0 candidates are built, with the MV cost measured
//     from the NEWMV predictor (svt_aom_choose_best_av1_mv_pred, DRL 0 at
//     preset 12's approx_inter_rate). Square, rectangular and 64x64 blocks
//     are all refined.
//   - Source/Lib/Codec/mcomp.c svt_av1_find_best_sub_pixel_tree_pruned,
//     first_level_check_fast and svt_check_better_fast: the error metric is
//     the residual VARIANCE (not SAD) plus MV_COST_OPT, the full-pel centre
//     uses the integer-sample variance (vfp->vf), and every sub-pel probe is
//     scored with vfp->svf, the 2-tap BILINEAR sub-pixel variance of
//     Source/Lib/C_DEFAULT/variance.c SUBPIX_VAR (bilinear_filters_2t in
//     Source/Lib/Codec/filter.h). The pruned tree never calls the upsampled
//     (USE_4_TAPS) predictor: is_scaled is 0, so svt_check_better_fast takes
//     svt_estimated_pref_error.
//   - Source/Lib/Codec/enc_mode_config.c md_subpel_me_controls level 6, the
//     level svt_aom_sig_deriv_enc_dec_rtc selects for PD_PASS_1 above M9 (M8
//     flat): quarter-pel max precision, one iteration per step,
//     skip_diag_refinement 3, bias_fp 110, no early exits. That is the
//     regular-PD1 configuration, the one SVT pairs with
//     interpolation_search_level 4 (IFS_MDS3, pframe_ifs.go), which this
//     encoder keeps.
//
// Light-PD1 itself runs level 8 at 720p/1080p (pred_variance_th 100,
// abs_th_mult 8, round_dev_th -15, skip_diag_refinement 4) with the
// interpolation filter search off (svt_aom_sig_deriv_enc_dec_light_pd1 never
// arms IFS; md_stage_0_light_pd1 forces REGULAR). Measured on the 720p
// fwd/rev p720_inter_q20 clip (2026-10-22): level 8 with IFS kept loses 0.17
// dB BD-PSNR (fixed q 140..220); level 8 with IFS off loses 0.28/0.62 dB at
// 1/2 Mbps CBR, while beating the old libaom tree with IFS off. The light-PD1
// pruning (and bias_fp) keeps vectors on full-pel, where SMOOTH/SHARP cannot
// help, so level 8 only fits the IFS-off pipeline it was tuned for.
//
// With skip_diag_refinement 3, org_error starts at besterr (halved for
// 64-sample sides), so first_level_check_fast probes the diagonal only when a
// cardinal probe improved; subpel_iters_per_step 1 keeps
// two_level_checks_fast from the second level.

// svtSubpelCtrls is MdSubPelSearchCtrls for the SUBPEL_TREE_PRUNED levels
// with subpel_iters_per_step 1 (levels 6..10; the second level check never
// runs).
type svtSubpelCtrls struct {
	rounds         int    // AOMMIN(FULL_PEL - forced_stop, 3 - !allow_hp)
	predVarianceTh uint32 // pred_variance_th
	absThMult      uint64 // abs_th_mult
	roundDevTh     int64  // round_dev_th
	skipDiag       uint8  // skip_diag_refinement
	biasFP         uint64 // bias_fp
}

// svtSubpelLevel6 is md_subpel_me_controls(ctx, 6). Quarter-pel max
// precision with allow_high_precision_mv off gives two rounds (half,
// quarter); round_dev_th MAX_SIGNED_VALUE never exits between them.
var svtSubpelLevel6 = svtSubpelCtrls{
	rounds:     2,
	roundDevTh: 1<<31 - 1,
	skipDiag:   3,
	biasFP:     110,
}

// svtSubpelScratch holds the bilinear passes' buffers (variance.c SUBPIX_VAR
// fdata3/temp2) and an edge-replicated reference window for footprints that
// leave the unpadded reference plane.
type svtSubpelScratch struct {
	fdata [(64 + 1) * 64]uint16
	pred  [64 * 64]byte
	win   [(64+1)*(64+1) + 16]byte // +16: the vector kernel's row over-read
}

// svtVarOffs is svt_aom_eb_av1_var_offs: an all-zero row read at stride 0,
// so vf(ref, zeros) is the variance of the reference samples themselves.
var svtVarOffs [128]byte

// svtBilinearFilters2T is bilinear_filters_2t (Source/Lib/Codec/filter.h),
// indexed by the eighth-pel phase.
var svtBilinearFilters2T = [8][2]uint32{
	{128, 0}, {112, 16}, {96, 32}, {80, 48}, {64, 64}, {48, 80}, {32, 96}, {16, 112},
}

// svtSubpelSearch is one SVT light-PD1 sub-pel refinement. The reference
// plane must match the source dimensions. startMV is the full-pel ME vector,
// refMV the NEWMV predictor the MV cost is measured from. It returns the
// refined vector; when it differs from startMV it is a sub-pel vector.
func (st *lossyEncodeState) svtSubpelSearch(src, refPlane []byte, stride, width, height, px, py, bw, bh int,
	startMV, refMV motion.Vector, ctrls *svtSubpelCtrls) motion.Vector {

	st.ensureSubpelProbeCache()
	scratch := &st.subpelCache.svt
	qp := uint64(st.qIndex) >> 2 // pcs->ppcs->picture_qp
	epb := st.rdMult >> 6        // error_per_bit = AOMMAX(rdmult >> RD_EPB_SHIFT, 1)
	if epb < 1 {
		epb = 1
	}
	// early_exit_th = 1020 - (sq_size >> 2) (svt_init_mv_cost_params).
	sq := bw
	if bh > sq {
		sq = bh
	}
	earlyExitTh := uint64(1020 - (sq >> 2))
	limits := svtSubpelLimits(px, py, bw, bh, width, height, refMV)

	srcOff := py*stride + px
	srcBlock := src[srcOff:]

	// svt_upsampled_setup_center_error: full-pel variance plus MV cost.
	best := startMV
	fx, fy := px+int(startMV.Col>>3), py+int(startMV.Row>>3)
	refBlock, refStride := svtRefWindow(scratch, refPlane, stride, width, height, fx, fy, bw, bh)
	_, distortion := realtimeInterResidualSSEVariance(refBlock, srcBlock, refStride, stride, 0, 0, bw, bh)
	besterr := uint64(distortion) + svtMVCostOpt(best, refMV, epb)

	// abs_th_mult early exit: (qp * besterr) < ((w * h) << 5) * abs_th_mult.
	if qp*besterr < uint64((bw*bh)<<5)*ctrls.absThMult {
		return best
	}
	if ctrls.rounds <= 0 {
		return best
	}
	// pred_variance_th: the full-pel prediction is too flat for interpolation
	// to change it.
	if ctrls.predVarianceTh != 0 {
		_, v := realtimeInterResidualSSEVariance(refBlock, svtVarOffs[:], refStride, 0, 0, 0, bw, bh)
		log2 := uint(bits.TrailingZeros(uint(bw * bh)))
		blockVar := (v + (1 << (log2 - 1))) >> log2
		if blockVar < ctrls.predVarianceTh {
			return best
		}
	}
	var orgErr uint64
	if ctrls.skipDiag >= 4 {
		orgErr = 0
	} else {
		demo := uint64(1)
		if ctrls.skipDiag >= 2 && (bw >= 64 || bh >= 64) {
			demo = 2
		}
		if ctrls.skipDiag != 0 {
			orgErr = besterr / demo
		} else {
			orgErr = 1<<31 - 1
		}
	}

	dist := uint64(distortion)
	check := func(this motion.Vector) uint64 {
		// svt_check_better_fast.
		if !limits.contains(this) {
			return 1<<31 - 1
		}
		cost := svtMVCostOpt(this, refMV, epb)
		// MV_COST_OPT early exit against the current best's distortion.
		if bestcost := dist + cost; bestcost > (besterr*earlyExitTh)/1000 {
			return bestcost
		}
		thismse := st.svtSubpelVariance(scratch, srcBlock, stride, refPlane, stride, width, height, px, py, bw, bh, this)
		cost += uint64(thismse)
		weight := uint64(100)
		if ctrls.biasFP != 0 && best.Col&7 == 0 && best.Row&7 == 0 {
			weight = ctrls.biasFP
		}
		if cost*weight/100 < besterr {
			besterr = cost
			best = this
			dist = uint64(thismse)
		}
		return cost
	}

	center := startMV
	hstep := int16(4) // INIT_SUBPEL_STEP_SIZE: half-pel in eighth-pel units
	for iter := 0; iter < ctrls.rounds; iter++ {
		prevErr := besterr
		// first_level_check_fast: the four cardinal directions, then the
		// best quadrant's diagonal unless org_error already bounds besterr.
		left := check(motion.Vector{Row: center.Row, Col: center.Col - hstep})
		right := check(motion.Vector{Row: center.Row, Col: center.Col + hstep})
		up := check(motion.Vector{Row: center.Row - hstep, Col: center.Col})
		down := check(motion.Vector{Row: center.Row + hstep, Col: center.Col})
		if besterr < orgErr {
			diag := motion.Vector{Row: center.Row + hstep, Col: center.Col + hstep}
			if left <= right {
				diag.Col = center.Col - hstep
			}
			if up <= down {
				diag.Row = center.Row - hstep
			}
			check(diag)
		}
		hstep >>= 1
		center = best
		if ctrls.skipDiag != 0 && iter < 1 { // iter < QUARTER_PEL
			orgErr = min(orgErr, besterr)
		}
		b, p := max(besterr, 1), max(prevErr, 1)
		deviation := (int64(b) - int64(p)) * 100 / int64(p)
		if deviation >= ctrls.roundDevTh {
			return best
		}
	}
	return best
}

// svtMVCostOpt is svt_mv_err_cost's MV_COST_OPT arm (mcomp.c), used when
// skip_diag_refinement >= 3: an L1 rate proxy scaled by error_per_bit,
// rounded by RDDIV_BITS + AV1_PROB_COST_SHIFT - RD_EPB_SHIFT +
// PIXEL_TRANSFORM_ERROR_SCALE = 14 bits.
func svtMVCostOpt(mv, ref motion.Vector, epb int64) uint64 {
	dr := int64(mv.Row) - int64(ref.Row)
	if dr < 0 {
		dr = -dr
	}
	dc := int64(mv.Col) - int64(ref.Col)
	if dc < 0 {
		dc = -dc
	}
	return uint64(((dr+dc)<<8*epb + (1 << 13)) >> 14)
}

type svtSubpelMVLimits struct {
	colMin, colMax, rowMin, rowMax int
}

func (l svtSubpelMVLimits) contains(mv motion.Vector) bool {
	return int(mv.Col) >= l.colMin && int(mv.Col) <= l.colMax &&
		int(mv.Row) >= l.rowMin && int(mv.Row) <= l.rowMax
}

// svtSubpelLimits is md_subpel_search's mv_limits (the block may sit up to
// AOM_INTERP_EXTEND samples beyond any picture edge) narrowed by
// svt_av1_set_mv_search_range and svt_av1_set_subpel_mv_search_range around
// the reference MV.
func svtSubpelLimits(px, py, bw, bh, width, height int, refMV motion.Vector) svtSubpelMVLimits {
	const (
		interpExtend  = 4                   // AOM_INTERP_EXTEND
		maxFullPelVal = (1 << (11 - 1)) - 1 // MAX_FULL_PEL_VAL
		mvUpp         = 1 << 14             // MV_UPP (MV_IN_USE_BITS = 14)
		mvLow         = -(1 << 14)          // MV_LOW
		maxMV         = maxFullPelVal * 8   // GET_MV_SUBPEL(MAX_FULL_PEL_VAL)
	)
	miCols, miRows := (width+7)>>3<<1, (height+7)>>3<<1
	miCol, miRow := px>>2, py>>2
	colMin := -((miCol+bw>>2)*4 + interpExtend)
	rowMin := -((miRow+bh>>2)*4 + interpExtend)
	colMax := (miCols-miCol)*4 + interpExtend
	rowMax := (miRows-miRow)*4 + interpExtend
	// svt_av1_set_mv_search_range.
	rc, rr := int(refMV.Col), int(refMV.Row)
	fcMin := rc>>3 - maxFullPelVal + b2i(rc&7 != 0)
	frMin := rr>>3 - maxFullPelVal + b2i(rr&7 != 0)
	fcMax := rc>>3 + maxFullPelVal
	frMax := rr>>3 + maxFullPelVal
	fcMin = max(fcMin, mvLow>>3+1)
	frMin = max(frMin, mvLow>>3+1)
	fcMax = min(fcMax, mvUpp>>3-1)
	frMax = min(frMax, mvUpp>>3-1)
	colMin, colMax = max(colMin, fcMin), min(colMax, fcMax)
	rowMin, rowMax = max(rowMin, frMin), min(rowMax, frMax)
	// svt_av1_set_subpel_mv_search_range.
	return svtSubpelMVLimits{
		colMin: max(mvLow+1, max(colMin*8, rc-maxMV)),
		colMax: min(mvUpp-1, min(colMax*8, rc+maxMV)),
		rowMin: max(mvLow+1, max(rowMin*8, rr-maxMV)),
		rowMax: min(mvUpp-1, min(rowMax*8, rr+maxMV)),
	}
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// svtRefWindow returns the reference samples of the (bw+1) x (bh+1)
// footprint at full-pel (x, y): the plane itself when the footprint is
// inside, else an edge-replicated copy (the padded reference SVT reads).
func svtRefWindow(scratch *svtSubpelScratch, refPlane []byte, stride, width, height, x, y, bw, bh int) ([]byte, int) {
	if x >= 0 && y >= 0 && x+bw < width && y+bh < height {
		return refPlane[y*stride+x:], stride
	}
	ws := bw + 1
	for r := 0; r <= bh; r++ {
		sy := min(max(y+r, 0), height-1)
		row := refPlane[sy*stride:]
		dst := scratch.win[r*ws : r*ws+ws]
		for c := range dst {
			dst[c] = row[min(max(x+c, 0), width-1)]
		}
	}
	return scratch.win[:], ws
}

// svtSubpelVariance is vfp->svf at the sub-pel vector mv: the 2-tap
// bilinear prediction (variance.c aom_var_filter_block2d_bil_first_pass_c /
// _second_pass_c, FILTER_BITS 7) and its residual variance against src.
func (st *lossyEncodeState) svtSubpelVariance(scratch *svtSubpelScratch, src []byte, srcStride int, refPlane []byte, stride, width, height, px, py, bw, bh int, mv motion.Vector) uint32 {
	x, y := px+int(mv.Col>>3), py+int(mv.Row>>3)
	ref, refStride := svtRefWindow(scratch, refPlane, stride, width, height, x, y, bw, bh)
	svtBilinearPredictArch(scratch, ref, refStride, bw, bh, int(mv.Col&7), int(mv.Row&7))
	_, v := realtimeInterResidualSSEVariance(scratch.pred[:], src, bw, srcStride, 0, 0, bw, bh)
	return v
}

// svtBilinearPredict writes the bw x bh bilinear prediction into
// scratch.pred (stride bw). The first pass always filters bh+1 rows and reads
// one column past the block, as the C reference does.
func svtBilinearPredict(scratch *svtSubpelScratch, ref []byte, refStride, bw, bh, xoff, yoff int) {
	fx0, fx1 := svtBilinearFilters2T[xoff][0], svtBilinearFilters2T[xoff][1]
	fy0, fy1 := svtBilinearFilters2T[yoff][0], svtBilinearFilters2T[yoff][1]
	fdata := scratch.fdata[:(bh+1)*bw]
	for r := 0; r <= bh; r++ {
		a := ref[r*refStride : r*refStride+bw+1]
		b := fdata[r*bw : r*bw+bw]
		for c := range b {
			b[c] = uint16((uint32(a[c])*fx0 + uint32(a[c+1])*fx1 + 64) >> 7)
		}
	}
	pred := scratch.pred[:bh*bw]
	for r := 0; r < bh; r++ {
		a0 := fdata[r*bw : r*bw+bw]
		a1 := fdata[(r+1)*bw : (r+1)*bw+bw]
		d := pred[r*bw : r*bw+bw]
		for c := range d {
			d[c] = uint8((uint32(a0[c])*fy0 + uint32(a1[c])*fy1 + 64) >> 7)
		}
	}
}
