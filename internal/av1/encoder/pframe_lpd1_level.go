package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// pframe_lpd1_level.go ports SVT-AV1's light-PD1 level machinery for the
// realtime inter path (Source/Lib/Codec, v4.1.0):
//
//   - Picture level: svt_aom_sig_deriv_mode_decision_config_rtc
//     (enc_mode_config.c:9846-9870), preset 12 (> ENC_M11):
//     pic_lpd1_lvl = is_islice ? 0 : is_base ? 4 : 7 for camera content and
//     is_base ? 0 : 5 for sc_class1 (> ENC_M9).
//   - Per-SB refinement: svt_aom_sig_deriv_enc_dec_common
//     (enc_mode_config.c:6974-7004), rtc above ENC_M6/M7: +2 (+3 from level
//     0) when the SB's me_8x8_cost_variance is under 3000 (the > ENC_M11
//     threshold), -2 for SBs whose qindex cyclic refresh moved off
//     base_q_idx, clamped to 0..7, then set_lpd1_ctrls (enc_mode_config.c:
//     5518). pd1_lvl_refinement is 2 above ENC_M10.
//   - Detector: enc_dec_process.c lpd1_detector_skip_pd0 (line 2188). P
//     pictures at preset 12 run VERY_LIGHT_PD0 (set_pic_lpd0_lvl_rtc level
//     7), so this is the detector that runs (enc_dec_process.c:3044); this
//     encoder has no PD0 pass at all (skip_pd_pass_0).
//   - Statistics: the ME's me_64x64_distortion / me_8x8_cost_variance
//     (motion_estimation.c compute_distortion) per SB of every inter frame,
//     and the coded picture's sb_intra / sb_skip (coding_loop.c update_b,
//     reset per SB at enc_dec_process.c:3110), kept with the reference
//     (rest_process.c copy_statistics_to_ref_obj_ect) together with
//     skip_coded_area (rc_process.c get_ref_skip_percentage).
//
// The ME statistics are the frame's open-loop ME per SB (hme.go sbME, run
// for every SB before mode decision like SVT's pa_me).
//
// Deviation (documented): the detector's current-picture neighbour arm
// (pcs->sb_intra / pcs->sb_skip of the left SB) is skipped for SBs that open
// a tile column, because tile columns code concurrently here and the left
// SB's flags are not final yet. With one tile it is exactly SVT's detector.

// lpd1PD1Level is SVT's Pd1Level: -1 is REGULAR_PD1, 0..5 LPD1_LVL_0..5.
type lpd1PD1Level int8

const (
	lpd1RegularPD1 lpd1PD1Level = -1
	lpd1Levels                  = 6
	// lpd1PD1LvlRefinement is ctx->pd1_lvl_refinement for rtc above ENC_M10
	// (enc_mode_config.c:7026-7032): the detector never demotes below
	// LPD1_LVL_1.
	lpd1PD1LvlRefinement = 2
	// lpd1MEVarRefineTh is the per-SB refinement threshold above ENC_M11
	// (enc_mode_config.c:6981-6983).
	lpd1MEVarRefineTh = 3000
)

// lpd1Ctrls is the subset of SVT's Lpd1Ctrls (md_process.h) the skip-PD0
// detector consumes. cost_th_dist/rate, nz_coeff_th and max_mv_length feed
// only lpd1_detector_post_pd0, which preset 12 never runs.
type lpd1Ctrls struct {
	pd1Level            lpd1PD1Level
	useDetector         [lpd1Levels]bool
	useRefInfo          [lpd1Levels]bool
	me8x8CostVarianceTh [lpd1Levels]uint32
	skipPD0EdgeDistTh   [lpd1Levels]uint32
	skipPD0MEShift      [lpd1Levels]uint16
}

const lpd1U32Max = ^uint32(0)
const lpd1U16Max = ^uint16(0)

// lpd1CtrlsTable is set_lpd1_ctrls (enc_mode_config.c:5518-5770) verbatim
// for the consumed fields.
var lpd1CtrlsTable = [8]lpd1Ctrls{
	0: {pd1Level: lpd1RegularPD1},
	1: {pd1Level: 0,
		useDetector:         [6]bool{0: true},
		useRefInfo:          [6]bool{0: true},
		me8x8CostVarianceTh: [6]uint32{0: 250000 >> 3},
		skipPD0EdgeDistTh:   [6]uint32{0: 1024},
		skipPD0MEShift:      [6]uint16{0: 1}},
	2: {pd1Level: 0,
		useDetector:         [6]bool{0: true},
		useRefInfo:          [6]bool{0: true},
		me8x8CostVarianceTh: [6]uint32{0: 250000 >> 3},
		skipPD0EdgeDistTh:   [6]uint32{0: 1024},
		skipPD0MEShift:      [6]uint16{0: 1}},
	3: {pd1Level: 0,
		useDetector:         [6]bool{0: true},
		useRefInfo:          [6]bool{0: false},
		me8x8CostVarianceTh: [6]uint32{0: 750000 >> 3},
		skipPD0EdgeDistTh:   [6]uint32{0: 16384},
		skipPD0MEShift:      [6]uint16{0: 3}},
	4: {pd1Level: 1,
		useDetector:         [6]bool{0: true, 1: true},
		useRefInfo:          [6]bool{0: false, 1: true},
		me8x8CostVarianceTh: [6]uint32{0: 500000, 1: 500000 >> 3},
		skipPD0EdgeDistTh:   [6]uint32{0: 16384, 1: 16384},
		skipPD0MEShift:      [6]uint16{0: 3, 1: 3}},
	5: {pd1Level: 3,
		useDetector:         [6]bool{0: true, 1: true, 2: true, 3: true},
		useRefInfo:          [6]bool{0: false, 1: false, 2: false, 3: true},
		me8x8CostVarianceTh: [6]uint32{0: lpd1U32Max, 1: lpd1U32Max, 2: lpd1U32Max, 3: lpd1U32Max},
		skipPD0EdgeDistTh:   [6]uint32{0: 16384 * 7, 1: 16384 * 6, 2: 16384 * 6, 3: 16384 * 6},
		skipPD0MEShift:      [6]uint16{0: 5, 1: 5, 2: 5, 3: 5}},
	6: {pd1Level: 4,
		useDetector:         [6]bool{0: true, 1: true, 2: true, 3: true, 4: true},
		useRefInfo:          [6]bool{0: false, 1: false, 2: false, 3: false, 4: true},
		me8x8CostVarianceTh: [6]uint32{0: lpd1U32Max, 1: lpd1U32Max, 2: lpd1U32Max, 3: lpd1U32Max, 4: lpd1U32Max},
		skipPD0EdgeDistTh:   [6]uint32{0: 16384 * 7, 1: 16384 * 7, 2: 16384 * 7, 3: 16384 * 7, 4: 16384 * 2},
		skipPD0MEShift:      [6]uint16{0: 5, 1: 5, 2: 5, 3: 5, 4: 3}},
	7: {pd1Level: 5,
		useDetector:         [6]bool{true, true, true, true, true, true},
		useRefInfo:          [6]bool{false, false, false, false, true, true},
		me8x8CostVarianceTh: [6]uint32{lpd1U32Max, lpd1U32Max, lpd1U32Max, lpd1U32Max, lpd1U32Max, lpd1U32Max},
		skipPD0EdgeDistTh:   [6]uint32{lpd1U32Max, lpd1U32Max, lpd1U32Max, lpd1U32Max, 16384 * 6, 16384 * 6},
		skipPD0MEShift:      [6]uint16{lpd1U16Max, lpd1U16Max, lpd1U16Max, lpd1U16Max, 5, 5}},
}

// lpd1PicLevel is pic_lpd1_lvl at preset 12 rtc
// (enc_mode_config.c:9846-9863): intra pictures and screen-content base
// pictures run the regular path.
func lpd1PicLevel(isIntra, isBase, screenContent bool) uint8 {
	switch {
	case isIntra:
		return 0
	case screenContent:
		if isBase {
			return 0
		}
		return 5
	case isBase:
		return 4
	}
	return 7
}

// lpd1SBStat is one superblock's light-PD1 statistics.
type lpd1SBStat struct {
	// me64Dist / me8x8Var are me_64x64_distortion (pix_num-normalized) and
	// me_8x8_cost_variance from the frame's open-loop ME (hme.go sbME).
	me64Dist uint32
	me8x8Var uint32
	// intra / skip are update_b's sb_intra (any intra block) and sb_skip
	// (no block with coefficients); skipArea accumulates the area of the
	// coefficient-free blocks (tot_skip_coded_area).
	intra bool
	skip  bool
	// mvp64 is sb_64x64_mvp (coding_loop.c update_b): the SB is one
	// 64x64 inter block coded with a predictor mode (not NEWMV).
	mvp64    bool
	skipArea uint32
	// intraArea accumulates the area of the intra blocks
	// (tot_intra_coded_area).
	intraArea uint32
	// minSq is sb_min_sq_size (coding_loop.c update_b): the smallest square
	// block size coded in the SB, 128 before any block; 0xFF where the
	// picture recorded none (keyframes, whose path does not note blocks).
	minSq uint8
}

// lpd1Frame holds one picture's per-SB light-PD1 statistics. A coded
// reference keeps its object (the encoder swaps pointers on refresh), the
// shape of SVT's EbReferenceObject sb_* arrays.
type lpd1Frame struct {
	cols, rows int
	sb         []lpd1SBStat
	// poc is the picture number (EbReferenceObject ref_poc), for the
	// depth-removal reference feedback's distance check.
	poc int
	// meArmed: the frame's ME statistics exist (pa_me ran).
	meArmed bool
	// baseQ is base_q_idx and meQOffset the per-SB b64_me_qindex -
	// base_q_idx (rc_aq.c generate_b64_me_qindex_map).
	baseQ     uint8
	meQOffset []int16

	// Reference-side facts (valid once published).
	valid         bool
	sliceI        bool
	skipPerc      uint8
	// intraPerc is intra_coded_area: the percentage of the picture coded
	// intra (0 for an I_SLICE, copy_statistics_to_ref_obj_ect).
	intraPerc uint8
	width, height int
}

// begin resets the frame's statistics for a new inter picture and takes
// the ME statistics of every SB from the frame's open-loop ME, as SVT's
// detector and lambda modulation read pa_me's per-SB results.
func (f *lpd1Frame) begin(w, h int, hme *hmeState, baseQ uint8) {
	f.resize(w, h)
	f.baseQ = baseQ
	f.meArmed = hme != nil && hme.fme.valid && hme.fme.cols == f.cols && hme.fme.rows == f.rows
	n := f.cols * f.rows
	for i := range f.sb[:n] {
		s := &f.sb[i]
		s.intra, s.skip, s.skipArea, s.mvp64 = false, true, 0, false
		s.intraArea = 0
		s.minSq = 128
		s.me64Dist, s.me8x8Var = 0, 0
		if f.meArmed {
			me := &hme.fme.sb[i]
			s.me64Dist, s.me8x8Var = me.dist64, me.me8x8CostVariance
		}
	}
	if f.meArmed {
		f.buildMEQIndex()
	}
	f.valid, f.sliceI, f.skipPerc, f.intraPerc = false, false, 0, 0
}

func (f *lpd1Frame) resize(w, h int) {
	f.width, f.height = w, h
	f.cols, f.rows = (w+63)/64, (h+63)/64
	n := f.cols * f.rows
	if len(f.sb) < n {
		f.sb = make([]lpd1SBStat, n)
		f.meQOffset = make([]int16, n)
	}
}

// setKey publishes an intra picture as a reference: every SB intra, coded
// with coefficients, no ME statistics (the detector's ref-info arm scores
// an I_SLICE reference +10 without reading them).
func (f *lpd1Frame) setKey(w, h int) {
	f.resize(w, h)
	for i := range f.sb[:f.cols*f.rows] {
		s := &f.sb[i]
		s.intra, s.skip, s.skipArea, s.mvp64 = true, false, 0, false
		s.minSq = 0xFF
		s.me64Dist, s.me8x8Var = 0, 0
	}
	f.valid, f.sliceI, f.skipPerc, f.intraPerc = true, true, 0, 0
}

// publish finalizes a coded inter picture as a reference: skip_coded_area
// becomes the percentage of the 8-aligned picture area.
func (f *lpd1Frame) publish() {
	n := f.cols * f.rows
	var area, intraArea uint64
	for i := 0; i < n; i++ {
		area += uint64(f.sb[i].skipArea)
		intraArea += uint64(f.sb[i].intraArea)
	}
	aw := uint64((f.width + 7) &^ 7)
	ah := uint64((f.height + 7) &^ 7)
	f.skipPerc = uint8(100 * area / (aw * ah))
	f.intraPerc = uint8(100 * intraArea / (aw * ah))
	f.valid, f.sliceI = true, false
}

// meStats returns SB idx's statistics.
func (f *lpd1Frame) meStats(idx int) *lpd1SBStat {
	return &f.sb[idx]
}

// lpd1SBDecision is the per-SB light-PD1 setting the block coder consumes,
// svt_aom_sig_deriv_enc_dec_light_pd1 (enc_mode_config.c:7208-7464) at rtc
// preset 12 for the features this encoder runs.
type lpd1SBDecision struct {
	idx   int
	level lpd1PD1Level
	// bypassTh is lpd1_bypass_tx_th (7410-7416): rtc uses 100 at
	// LPD1_LVL_0 and 200 above; the regular path has none.
	bypassTh int64
	// chromaDetector is lpd1_tx_ctrls.chroma_detector_level, the ENC_M10+
	// override (7340-7347): 4 up to LPD1_LVL_4, 0 above.
	chromaDetector uint8
	// rdoq is rdoq_level != 0 (7262-7290): rtc gives level 1 up to
	// LPD1_LVL_4, but only when pcs->rdoq_level is set, and preset 12 rtc
	// sets pcs->rdoq_level = is_islice ? 1 : 0 (9591-9596), so inter
	// pictures never run RDOQ.
	rdoq bool
	// coeffRateLvl0 is rate_est_level 0 -> coeff_rate_est_lvl 0
	// (7418-7425): only LPD1_LVL_0 keeps rate_est_level 4.
	coeffRateLvl0 bool
	// nearCount and skipDCTh are set_cand_reduction_ctrls'
	// near_count_ctrls.near_count and cand_elimination_ctrls.skip_dc_th at
	// cand_reduction_level 2 (LPD1_LVL_0), 3 (LVL_1..2), 4 (LVL_3) and 5
	// (LVL_4..5), 7245-7272.
	nearCount int
	skipDCTh  int
	// skipInterTxLevel is lpd1_skip_inter_tx_level, the rtc > ENC_M10
	// branch (7378-7393).
	skipInterTxLevel uint8
	// meQFactor is update_lambda's stats_based_sb_lambda_modulation
	// factor for the SB (rc_process.c, rtc branch).
	meQFactor int64
	// subpelLevel is me_subpel_level (svt_aom_sig_deriv_enc_dec_light_pd1):
	// 0 (no sub-pel), 7, 8 or 10 (svtLPD1SubpelLevel).
	subpelLevel uint8
}

// lpd1SBInfo is the reference information light-PD1's signal derivation
// reads (7224-7244): the SB's ME statistics and the LAST reference's
// sb_skip when a same-size reference exists (~0 / 0 otherwise; P pictures
// have no list-1 reference, so l1_was_skip is 1), and ref_skip_percentage.
type lpd1SBInfo struct {
	me8x8Var, me64Dist uint32
	l0WasSkip          bool
	l0Was64x64MVP      bool
	refSkipPerc        uint8
}

var lpd1NoSBInfo = lpd1SBInfo{me8x8Var: lpd1U32Max, me64Dist: lpd1U32Max}

func lpd1DecisionForLevel(level lpd1PD1Level, info lpd1SBInfo, pictureQP uint32, notLastLayer bool) lpd1SBDecision {
	d := lpd1SBDecision{idx: -1, level: level}
	switch {
	case level == lpd1RegularPD1:
		d.bypassTh = 0
	case level <= 0:
		d.bypassTh = 100
	default:
		d.bypassTh = 200
	}
	if level <= 4 {
		d.chromaDetector = 4
	}
	d.rdoq = false
	d.coeffRateLvl0 = level > 0
	switch {
	case level <= 0:
		d.nearCount = 3 // cand_reduction_level 2
	case level <= 3:
		d.nearCount = 1 // levels 3 and 4
	default:
		d.nearCount = 1 // level 5
		if notLastLayer {
			d.skipDCTh = 5
		} else {
			d.skipDCTh = 15
		}
	}
	switch {
	case level <= 2:
		d.skipInterTxLevel = 0
	case level <= 4:
		d.skipInterTxLevel = 1
	default:
		d.skipInterTxLevel = 1
		if (info.l0WasSkip && info.refSkipPerc > 35 &&
			info.me8x8Var < 800*pictureQP && info.me64Dist < 800*pictureQP) ||
			(info.me8x8Var < 100*pictureQP && info.me64Dist < 100*pictureQP) {
			d.skipInterTxLevel = 2
		}
	}
	return d
}

// lpd1SBInfoFor gathers lpd1SBInfo for SB idx of the current picture.
func (st *lossyEncodeState) lpd1SBInfoFor(f *lpd1Frame, idx int) lpd1SBInfo {
	ref := st.lpd1.ref
	if !f.meArmed || ref == nil || !ref.valid || ref.width != f.width || ref.height != f.height {
		return lpd1NoSBInfo
	}
	cur := f.meStats(idx)
	return lpd1SBInfo{
		me8x8Var:      cur.me8x8Var,
		me64Dist:      cur.me64Dist,
		l0WasSkip:     ref.sb[idx].skip,
		l0Was64x64MVP: ref.sb[idx].mvp64,
		refSkipPerc:   ref.skipPerc,
	}
}

// lpd1PictureSetup is what a tile coder needs for one inter picture.
type lpd1PictureSetup struct {
	cur, ref *lpd1Frame
	picLvl   uint8
	// notLastLayer is !is_highest_layer: the base and T1 layers, and the
	// flat structure's T0/T1 positions (svtFlatTemporalLayer).
	notLastLayer bool
	// tlFactor is rd_frame_type_factor[8-bit][gf_update_type] and gfUpdate
	// the low-delay GF_UPDATE picture (every MAX_GF_INTERVAL frames).
	tlFactor int64
	gfUpdate bool
	// skipIntra is pcs->skip_intra (enc_mode_config.c, rtc branch of
	// svt_aom_sig_deriv_mode_decision_config): a non-reference picture whose
	// LAST reference coded at most 50% of its area intra (an I_SLICE
	// reference counts 0, rc_process.c get_ref_intra_percentage) runs
	// intra_level 0, no intra candidate at all.
	skipIntra bool
	// screen is the picture's sc_class1: intra_level 6 (pframe_intra_sc.go)
	// instead of the DC-only candidate.
	screen bool
	// armed marks a picture setup; coders without one (standalone tile
	// paths, the external-reference SVC path) keep the base-picture level
	// pic_lpd1_lvl 4 without per-SB statistics.
	armed bool
}

// lpd1SB returns the light-PD1 decision of the superblock containing luma
// position (px, py). It is computed once per SB when the first block asks;
// the left and top SBs are complete by then in every walk order (serial,
// split, wavefront with its top-right dependency).
func (st *lossyEncodeState) lpd1SB(px, py int) *lpd1SBDecision {
	f := st.lpd1.cur
	if f == nil {
		if st.lpd1Dec.idx != -2 {
			lvl := 4
			if st.lpd1.armed {
				lvl = min(int(st.lpd1.picLvl), 7)
			}
			st.lpd1Dec = lpd1DecisionForLevel(lpd1CtrlsTable[lvl].pd1Level, lpd1NoSBInfo, uint32(st.qIndex)>>2, true)
			st.lpd1Dec.idx = -2
			st.lpd1Dec.meQFactor = 128
			st.lpd1Dec.subpelLevel = svtLPD1SubpelLevel(st.lpd1Dec.level, lpd1NoSBInfo, uint32(st.qIndex)>>2, true, st.keyVisW*st.keyVisH)
		}
		return &st.lpd1Dec
	}
	sbX, sbY := px/64, py/64
	idx := sbY*f.cols + sbX
	if st.lpd1Dec.idx == idx {
		return &st.lpd1Dec
	}
	level := st.lpd1DeriveLevel(f, idx, sbX, sbY)
	st.lpd1Dec = lpd1DecisionForLevel(level, st.lpd1SBInfoFor(f, idx), uint32(st.qIndex)>>2, st.lpd1.notLastLayer)
	st.lpd1Dec.idx = idx
	st.lpd1Dec.meQFactor = lpd1MEQLambdaFactor(f.meQDiff(idx))
	st.lpd1Dec.subpelLevel = svtLPD1SubpelLevel(level, st.lpd1SBInfoFor(f, idx), uint32(st.qIndex)>>2, st.lpd1.notLastLayer, f.width*f.height)
	if lpd1StatsEnabled {
		noteLPD1SBLevel(level)
	}
	return &st.lpd1Dec
}

// lpd1DeriveLevel is the per-SB refinement plus lpd1_detector_skip_pd0.
func (st *lossyEncodeState) lpd1DeriveLevel(f *lpd1Frame, idx, sbX, sbY int) lpd1PD1Level {
	lvl := int(st.lpd1.picLvl)
	if !f.meArmed {
		return lpd1CtrlsTable[min(lvl, 7)].pd1Level
	}
	cur := f.meStats(idx)
	// P pictures only reach here (keyframes have no inter blocks).
	me8 := int(cur.me8x8Var)
	if lvl == 0 {
		if me8 < lpd1MEVarRefineTh {
			lvl += 3
		}
	} else if me8 < lpd1MEVarRefineTh {
		lvl += 2
	}
	// For SBs that are part of the cyclic refresh, use conservative settings.
	if st.sbQIndex != nil && idx < len(st.sbQIndex) && st.sbQIndex[idx] != st.qIndex {
		lvl -= 2
	}
	lvl = max(0, min(lvl, 7))
	ctrls := lpd1CtrlsTable[lvl]
	st.lpd1DetectSkipPD0(&ctrls, f, cur, idx, sbX, sbY)
	return ctrls.pd1Level
}

// lpd1DetectSkipPD0 ports lpd1_detector_skip_pd0 (enc_dec_process.c:2188).
// The arithmetic stays in uint32 so the shifted sums and the reference
// multiples wrap as SVT's do.
func (st *lossyEncodeState) lpd1DetectSkipPD0(ctrls *lpd1Ctrls, f *lpd1Frame, cur *lpd1SBStat, idx, sbX, sbY int) {
	ref := st.lpd1.ref
	refOK := ref != nil && ref.valid && ref.width == f.width && ref.height == f.height
	pictureQP := uint32(st.qIndex) >> 2
	for lvl := lpd1Levels - 1; lvl > int(lpd1RegularPD1); lvl-- {
		if lvl <= lpd1PD1LvlRefinement-1 {
			break
		}
		if int(ctrls.pd1Level) != lvl || !ctrls.useDetector[lvl] {
			continue
		}
		// Use info from ref. frames (if available). The layer check passes
		// for every reference this encoder's structures name (base and T1
		// references sit at or below the current layer).
		if ctrls.useRefInfo[lvl] && refOK {
			score, refs := 0, 0
			if !ref.sliceI {
				r := &ref.sb[idx]
				if r.intra {
					score += 5
				}
				if !r.skip {
					score += 5
				}
				if cur.me64Dist > r.me64Dist*3 {
					score += 5
				}
				if cur.me8x8Var > r.me8x8Var*3 {
					score += 5
				}
			} else {
				score += 10
			}
			refs++
			if refs != 0 && score >= 10*refs {
				ctrls.pd1Level = lpd1PD1Level(lvl - 1)
				continue
			}
		}
		if sbX == 0 || sbY == 0 {
			if cur.me64Dist > ctrls.skipPD0EdgeDistTh[lvl] {
				ctrls.pd1Level = lpd1PD1Level(lvl - 1)
			}
			// me_8x8_cost_variance_th is shifted by 5 then multiplied by 73
			// minus pic_qp, so it must stay under (~0 >> 2).
			if th := ctrls.me8x8CostVarianceTh[lvl]; th < lpd1U32Max>>2 &&
				cur.me8x8Var > (th>>5)*(73-pictureQP) {
				ctrls.pd1Level = lpd1PD1Level(lvl - 1)
			}
			continue
		}
		left := f.meStats(idx - 1)
		top := f.meStats(idx - f.cols)
		shift := ctrls.skipPD0MEShift[lvl]
		switch {
		case shift != lpd1U16Max && cur.me64Dist > (left.me64Dist+top.me64Dist)<<shift:
			ctrls.pd1Level = lpd1PD1Level(lvl - 1)
		case shift != lpd1U16Max && cur.me8x8Var > (left.me8x8Var+top.me8x8Var)<<shift:
			ctrls.pd1Level = lpd1PD1Level(lvl - 1)
		case ctrls.useRefInfo[lvl] && sbX != st.lpd1TileSBCol0:
			// Use info from neighbouring SBs.
			if left.intra && top.intra {
				ctrls.pd1Level = lpd1PD1Level(lvl - 1)
			} else if !left.skip && !top.skip && (left.intra || top.intra) {
				ctrls.pd1Level = lpd1PD1Level(lvl - 1)
			}
		}
	}
}

// lpd1NoteBlock records a coded block into its SB's sb_intra / sb_skip /
// tot_skip_coded_area (coding_loop.c update_b).
func (st *lossyEncodeState) lpd1NoteBlock(px, py, bw, bh int, intra, hasCoeff bool) {
	f := st.lpd1.cur
	if f == nil {
		return
	}
	s := &f.sb[(py/64)*f.cols+px/64]
	if sq := uint8(max(bw, bh)); sq < s.minSq {
		s.minSq = sq
	}
	if intra {
		s.intra = true
		s.intraArea += uint32(bw * bh)
	}
	if hasCoeff {
		s.skip = false
	} else {
		s.skipArea += uint32(bw * bh)
	}
}

// lpd1FreezeSkipRates freezes skip_fac_bits for skip_coeff_ctx 0 from the
// frame-start CDFs: the light-PD1 path forces update_skip_coeff_ctx = 0
// (enc_mode_config.c:7461, rd_cost.c:1514) and SVT's md_rate_est_ctx is
// frame-level.
func (st *lossyEncodeState) lpd1FreezeSkipRates() {
	cdf := &st.modeCDFs.Skip[0]
	st.lpd1SkipBits = [2]int64{int64(entropy.CostSymbol(cdf, 0)), int64(entropy.CostSymbol(cdf, 1))}
}

// beginLPD1Tile installs the picture's light-PD1 setup on a tile coder.
func (st *lossyEncodeState) beginLPD1Tile(miColStart uint16) {
	st.lpd1 = st.lpd1Pending
	st.lpd1Pending = lpd1PictureSetup{}
	st.lpd1Dec = lpd1SBDecision{idx: -1}
	st.lpd1TileSBCol0 = int(miColStart) * 4 / 64
	st.lpd1FreezeSkipRates()
}

// lpd1NoteIntraBlock records an intra block of an inter picture. Intra
// blocks here always code their residual (skip = 0), so they count as
// coefficient-carrying.
func (st *lossyEncodeState) lpd1NoteIntraBlock(block tile.BlockVisit) {
	if st.lpd1.cur == nil {
		return
	}
	bw, bh, _ := pframeBlockDims(block.Size)
	st.lpd1NoteBlock(int(block.MICol)*4, int(block.MIRow)*4, bw, bh, true, true)
}

// publishLPD1 finalizes the coded picture's statistics and swaps them into
// the reference slot old held; old becomes the next picture's object.
func (e *VideoEncoder) publishLPD1(old *lpd1Frame, coded bool) *lpd1Frame {
	cur := e.lpd1Cur
	if cur == nil || !coded {
		if old != nil {
			old.valid = false
		}
		return old
	}
	cur.publish()
	e.lpd1Cur = old
	return cur
}

// lpd1KeyRefs publishes a keyframe into every reference slot's statistics.
func (e *VideoEncoder) lpd1KeyRefs(w, h int) {
	if e.lpd1Last == nil {
		e.lpd1Last = new(lpd1Frame)
	}
	if e.lpd1T1 == nil {
		e.lpd1T1 = new(lpd1Frame)
	}
	e.lpd1Last.setKey(w, h)
	e.lpd1T1.setKey(w, h)
	e.lpd1Last.poc, e.lpd1T1.poc = e.frameIndex, e.frameIndex
}

// lpd1InvalidateRefs drops the reference statistics when a frame refreshes
// slots outside the LAST/T1 bookkeeping (the external-reference path).
func (e *VideoEncoder) lpd1InvalidateRefs() {
	if e.lpd1Last != nil {
		e.lpd1Last.valid = false
	}
	if e.lpd1T1 != nil {
		e.lpd1T1.valid = false
	}
}

// meQDiff returns b64_me_qindex[idx] - base_q_idx for the picture
// (rc_aq.c generate_b64_me_qindex_map, built when
// stats_based_sb_lambda_modulation is set, as it always is): each SB's
// me_8x8_cost_variance places it between the frame's minimum and maximum,
// mapped linearly onto [-8, 8] around the average (built in begin).
func (f *lpd1Frame) meQDiff(idx int) int {
	if !f.meArmed {
		return 0
	}
	return int(f.meQOffset[idx])
}

func (f *lpd1Frame) buildMEQIndex() {
	n := f.cols * f.rows
	var avg uint64
	minDist, maxDist := ^uint64(0), uint64(0)
	for i := 0; i < n; i++ {
		v := uint64(f.sb[i].me8x8Var)
		avg += v
		minDist = min(minDist, v)
		maxDist = max(maxDist, v)
	}
	avg /= uint64(n)
	// min_offset / max_offset = -8 / 8 at every temporal layer; C int
	// (32-bit) arithmetic.
	const minOffset, maxOffset = int32(-8), int32(8)
	for i := 0; i < n; i++ {
		diff := int32(uint64(f.sb[i].me8x8Var) - avg)
		var offset int32
		if diff <= 0 {
			if minDist != avg {
				offset = minOffset * diff / int32(minDist-avg)
			}
		} else if maxDist != avg {
			offset = maxOffset * diff / int32(maxDist-avg)
		}
		offset = min(offset, 9*4-1)
		offset = max(offset, -9*4+1)
		q := max(1, min(int32(f.baseQ)+offset, 255))
		f.meQOffset[i] = int16(q - int32(f.baseQ))
	}
}

// lpd1MEQLambdaFactor is update_lambda's rtc stats_based_sb_lambda_modulation
// factor (rc_process.c): SBs whose me_q_index sits under base_q_idx get a
// lower lambda.
func lpd1MEQLambdaFactor(qdiff int) int64 {
	if qdiff < 0 {
		if qdiff <= -4 {
			return 100
		}
		return 115
	}
	return 128
}

// lpd1FullLambda is ctx->full_lambda_md[EB_8_BIT_MD]
// (md_process.c av1_lambda_assign_md -> rc_process.c
// svt_aom_compute_rd_mult): svt_aom_compute_rd_mult_based_on_qindex at the
// SB qindex (qp_index; def_inter 3.2 for LF_UPDATE, def_arf 3.25 for the
// low-delay GF_UPDATE every MAX_GF_INTERVAL frames), times
// rd_frame_type_factor[8-bit][gf_update_type] (150 for temporal layer 0 and
// internal layers, 180 for the highest layer; flat IPP is layer 0), times
// the SB's me_q_index modulation. lambda_weight is 0 and
// lambda_scale_factors are 128 outside tune IQ.
func (st *lossyEncodeState) lpd1FullLambda(dec *lpd1SBDecision) int64 {
	q := float64(st.yQuant.DC)
	mult := 3.2
	if st.lpd1.gfUpdate {
		mult = 3.25
	}
	rdmult := int64((mult + 0.0015*q) * q * q)
	tl := st.lpd1.tlFactor
	if tl == 0 {
		tl = 150
	}
	rdmult = (rdmult * tl) >> 7
	f := dec.meQFactor
	if f == 0 {
		f = 128
	}
	return (rdmult * f) >> 7
}

// lpd1MaxGFInterval is rc_process.h MAX_GF_INTERVAL.
const lpd1MaxGFInterval = 32

// lpd1NoteIsInterRate records the block's is_inter flag rate from the
// frame-frozen tables (intra_inter_fac_bits[is_inter_ctx][1], with the real
// neighbour context: shut_fast_rate is off in light-PD1), the term of
// av1_inter_fast_cost_light's fast_luma_rate the MDS0 loop leaves out.
func (st *lossyEncodeState) lpd1NoteIsInterRate(modeCtx *tile.BlockModeContext, block tile.BlockVisit) {
	st.lpd1IsInterBits = 0
	if !st.mds0IntraOK {
		return
	}
	ctx, err := modeCtx.IntraContext(int(block.X4), int(block.Y4), block.HaveTop, block.HaveLeft)
	if err == nil && ctx >= 0 && ctx < len(st.mds0IntraRates.IntraFlag) {
		st.lpd1IsInterBits = int64(st.mds0IntraRates.IntraFlag[ctx][1])
	}
}

// svtLPD1SubpelLevel is me_subpel_level in svt_aom_sig_deriv_enc_dec_light_pd1
// (enc_mode_config.c) at preset 12 rtc (non-flat IPP, above ENC_M11):
//
//   - LPD1_LVL_0: 7 up to 1080p, else 10;
//   - above: up to 480p 7 (not-last layer) or 8, up to 1080p 8, else 10, and
//     0 (no sub-pel) when ((l0_was_skip && l1_was_skip && ref_skip_perc > 50)
//     || (l0_was_64x64_mvp && l1_was_64x64_mvp)) && me_8x8_cost_variance <
//     200 * picture_qp && me_64x64_distortion < 200 * picture_qp. P pictures
//     have no list 1, so l1_was_skip and l1_was_64x64_mvp are 1.
//
// numPixels picks svt_aom_derive_input_resolution's range.
func svtLPD1SubpelLevel(level lpd1PD1Level, info lpd1SBInfo, pictureQP uint32, notLastLayer bool, numPixels int) uint8 {
	const (
		inputSize480pTH  = 0xA1400  // INPUT_SIZE_480p_TH
		inputSize1080pTH = 0x535200 // INPUT_SIZE_1080p_TH
	)
	if level <= 0 {
		if numPixels < inputSize1080pTH {
			return 7
		}
		return 10
	}
	var sub uint8
	switch {
	case numPixels < inputSize480pTH:
		sub = 8
		if notLastLayer {
			sub = 7
		}
	case numPixels < inputSize1080pTH:
		sub = 8
	default:
		sub = 10
	}
	if ((info.l0WasSkip && info.refSkipPerc > 50) || info.l0Was64x64MVP) &&
		info.me8x8Var < 200*pictureQP && info.me64Dist < 200*pictureQP {
		sub = 0
	}
	return sub
}

// lpd1NoteMVP64 records sb_64x64_mvp (coding_loop.c update_b): an inter block
// spanning the whole 64x64 SB coded with a mode other than NEWMV/NEW_NEWMV.
func (st *lossyEncodeState) lpd1NoteMVP64(px, py, bw, bh int, mode tile.InterModeResult) {
	f := st.lpd1.cur
	if f == nil || max(bw, bh) != 64 { // blk_geom->sq_size == sb_size
		return
	}
	if mode.Compound {
		if mode.CompoundMode == tile.CompoundInterModeNewNew {
			return
		}
	} else if mode.Mode == tile.InterModeNewMV {
		return
	}
	f.sb[(py/64)*f.cols+px/64].mvp64 = true
}

// lpd1RefIntraPerc is get_ref_intra_percentage for a P picture: the LAST
// reference's intra_coded_area, 0 for an I_SLICE (or missing) reference.
func lpd1RefIntraPerc(ref *lpd1Frame) uint8 {
	if ref == nil || !ref.valid || ref.sliceI {
		return 0
	}
	return ref.intraPerc
}
