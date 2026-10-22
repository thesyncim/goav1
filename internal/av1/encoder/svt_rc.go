package encoder

import "math"

// svt_rc.go ports SVT-AV1's one-pass CBR rate control (Source/Lib/Codec,
// v4.1.0): one RATE_CONTROL shared by every temporal layer.
//
//   - Per frame target: rc_vbr_cbr.c svt_aom_one_pass_rt_rate_alloc,
//     av1_calc_pframe_target_size_one_pass_cbr /
//     av1_calc_iframe_target_size_one_pass_cbr.
//   - Q: rc_pick_q_and_bounds_no_stats_cbr (calc_active_worst_quality_
//     no_stats_cbr, calc_active_best_quality_no_stats_cbr with the
//     reference-q inheritance and per-layer halving toward active worst,
//     av1_rc_regulate_q -> find_closest_qindex_by_rate over
//     svt_av1_rc_bits_per_mb, adjust_q_cbr_flat / adjust_q_cbr with
//     max_delta_per_layer) and svt_av1_rc_calc_qindex_rate_control's
//     reference-q floor.
//   - Feedback: svt_av1_rc_postencode_update
//     (av1_rc_update_rate_correction_factors with the cyclic-refresh segment
//     weighting, avg_frame_qindex, update_buffer_level) and
//     svt_aom_update_rc_counts.
//   - Configuration: pass2_strategy.c rc_cfg (max_intra_bitrate_pct 300,
//     max_inter_bitrate_pct 0, over/under shoot 25), enc_settings.c buffer
//     defaults (600 / 600 / 1000 ms), set_rc_buffer_sizes, svt_av1_rc_init,
//     av1_rc_update_framerate.
//
// Only the 8-bit, non-screen-content and screen-content rate models are
// ported (the encoder's realtime path is 8-bit here); dynamic resize and
// recode are off at preset 12 rtc.

const (
	svtRCBperMBNormBits  = 9
	svtRCFrameOverhead   = 200
	svtRCMinBPBFactor    = 0.005
	svtRCMaxBPBFactor    = 50
	svtRCMaxMBRate       = 250
	svtRCMaxRate1080P    = 2025000
	svtRCVBRMaxSection   = 2000
	svtRCMaxIntraPct     = 300
	svtRCOverShootPct    = 25
	svtRCUnderShootPct   = 25
	svtRCBoostKFHigh     = 5000
	svtRCBoostKFLow      = 400
	svtRCDefaultKFBoost  = 2300
	svtRCStartingBufMs   = 600
	svtRCOptimalBufMs    = 600
	svtRCMaximumBufMs    = 1000
	svtRCQFactor         = 1.1
	svtRCInterNormal     = 0
	svtRCKFStd           = 5
	svtRCMaxTemporalLvls = 6
)

// svtRCMaxDeltaPerLayer is rc_vbr_cbr.c max_delta_per_layer.
var svtRCMaxDeltaPerLayer = [6][6]int{
	{60}, {60, 5}, {60, 20, 2}, {60, 20, 10, 2}, {60, 20, 10, 5, 2}, {60, 30, 20, 10, 5, 2},
}

// svtRateControl is the subset of SVT's RATE_CONTROL / RateControlCfg the
// one-pass CBR path reads.
type svtRateControl struct {
	enabled bool

	bestQuality, worstQuality int
	targetBitRate             int64
	framerate                 float64
	mbs                       int

	avgFrameBandwidth     int
	prevAvgFrameBandwidth int
	maxFrameBandwidth     int
	startingBufferLevel   int64
	optimalBufferLevel    int64
	maximumBufferSize     int64
	bufferLevel           int64
	bitsOffTarget         int64

	avgFrameQIndex [2]int // [KEY_FRAME, INTER_FRAME]
	lastQ          [2]int
	lastBoosted    int
	rateCorrection [svtRCMaxTemporalLvls + 1]float64

	framesSinceKey     int
	framesToKey        int
	frameUpdated       int
	thisKeyFrameForced bool
	kfBoost            int
	pictureNumber      int

	q1Frame, q2Frame   int
	rc1Frame, rc2Frame int

	curAvgBaseMEDist  uint32
	prevAvgBaseMEDist uint32

	// Cyclic-refresh adaptation (rc_aq.c, adjusted by the correction
	// factor update).
	percentRefreshAdjustment  int
	rateRatioQDeltaAdjustment float64
}

const (
	svtFrameKey   = 0
	svtFrameInter = 1
)

// init is set_rc_buffer_sizes + svt_av1_rc_init + av1_rc_update_framerate
// for a CBR stream (SVT re-runs it on every rate change event,
// seq_param_changed).
func (rc *svtRateControl) init(targetBps, fps int, bestQ, worstQ uint8, width, height int) {
	*rc = svtRateControl{enabled: true}
	rc.bestQuality, rc.worstQuality = int(bestQ), int(worstQ)
	rc.targetBitRate = int64(targetBps)
	rc.framerate = float64(fps)
	if rc.framerate < 0.1 {
		rc.framerate = 30
	}
	rc.mbs = ((width + 15) / 16) * ((height + 15) / 16)
	bw := rc.targetBitRate
	rc.startingBufferLevel = svtRCStartingBufMs * bw / 1000
	rc.optimalBufferLevel = svtRCOptimalBufMs * bw / 1000
	rc.maximumBufferSize = svtRCMaximumBufMs * bw / 1000

	rc.avgFrameQIndex = [2]int{rc.worstQuality, rc.worstQuality}
	rc.lastQ = [2]int{rc.worstQuality, rc.worstQuality}
	rc.bufferLevel = rc.startingBufferLevel
	rc.bitsOffTarget = rc.startingBufferLevel
	rc.framesSinceKey = 8
	for i := range rc.rateCorrection {
		rc.rateCorrection[i] = 0.7
	}
	rc.avgFrameBandwidth = int(float64(rc.targetBitRate) / rc.framerate)
	vbrMax := int(int64(rc.avgFrameBandwidth) * svtRCVBRMaxSection / 100)
	rc.maxFrameBandwidth = max(rc.mbs*svtRCMaxMBRate, svtRCMaxRate1080P, vbrMax)
	rc.percentRefreshAdjustment = 5
	rc.rateRatioQDeltaAdjustment = 0.25
}

// svtConvertQIndexToQ is svt_av1_convert_qindex_to_q at 8-bit.
func svtConvertQIndexToQ(q int) float64 {
	return cyclicQRealTable[max(0, min(q, 255))]
}

// svtComputeQDelta is svt_av1_compute_qdelta.
func svtComputeQDelta(qstart, qtarget float64) int {
	start, target := 255, 255
	for i := 0; i < 255; i++ {
		start = i
		if svtConvertQIndexToQ(i) >= qstart {
			break
		}
	}
	for i := 0; i < 255; i++ {
		target = i
		if svtConvertQIndexToQ(i) >= qtarget {
			break
		}
	}
	return target - start
}

// svtRCBitsPerMB is svt_av1_rc_bits_per_mb at 8-bit.
func svtRCBitsPerMB(frameType int, q int, correction float64, screen bool) int {
	enumerator := 1000000
	switch {
	case screen && frameType == svtFrameKey:
		enumerator = 1000000
	case screen:
		enumerator = 750000
	case frameType == svtFrameKey:
		enumerator = 1400000
	}
	return int(float64(enumerator) * correction / svtConvertQIndexToQ(q))
}

// findQIndexByRate is rc_process.c find_qindex_by_rate.
func (rc *svtRateControl) findQIndexByRate(desired int, frameType int, screen bool) int {
	low, high := rc.bestQuality, rc.worstQuality
	for low < high {
		mid := (low + high) >> 1
		if svtRCBitsPerMB(frameType, mid, 1.0, screen) > desired {
			low = mid + 1
		} else {
			high = mid
		}
	}
	return low
}

// computeQDeltaByRate is svt_av1_compute_qdelta_by_rate.
func (rc *svtRateControl) computeQDeltaByRate(frameType, qindex int, ratio float64, screen bool) int {
	base := svtRCBitsPerMB(frameType, qindex, 1.0, screen)
	target := int(ratio * float64(base))
	return rc.findQIndexByRate(target, frameType, screen) - qindex
}

// svtRCFrame is one picture's RC inputs.
type svtRCFrame struct {
	key           bool
	screen        bool
	flat          bool // use_flat_ipp
	hierLevels    int  // hierarchical_levels
	temporalLayer int
	frameOffset   int // distance to the last IDR (frame_offset)
	// The list-0 reference (one reference at preset 12 rtc).
	refValid  bool
	refBaseQ  int
	refLayer  int
	refIsI    bool
	width     int
	height    int
	gfUpdate  bool // low-delay GF_UPDATE (frame_offset % MAX_GF_INTERVAL == 0)
	thisFrame int  // this_frame_target, filled by target()
}

// beginFrame is rc_init_frame_stats' ME bookkeeping: the average of the
// frame's me_64x64_distortion over its SBs (inter frames only).
func (rc *svtRateControl) beginFrame(f *svtRCFrame, hme *hmeState) {
	if f.key || hme == nil || !hme.fme.valid {
		return
	}
	n := hme.fme.cols * hme.fme.rows
	if n == 0 {
		return
	}
	rc.prevAvgBaseMEDist = rc.curAvgBaseMEDist
	var sum uint64
	for i := 0; i < n; i++ {
		sum += uint64(hme.fme.sb[i].dist64)
	}
	rc.curAvgBaseMEDist = uint32(sum / uint64(n))
}

// target is svt_aom_one_pass_rt_rate_alloc.
func (rc *svtRateControl) target(f *svtRCFrame) {
	if f.key {
		rc.kfBoost = svtRCDefaultKFBoost
		// intra_period_length is -1 (keyint -1): frames_to_key restarts at 0.
		rc.thisKeyFrameForced = rc.pictureNumber != 0 && rc.framesToKey == 0
		rc.framesToKey = 0
		f.thisFrame = rc.iframeTarget()
		return
	}
	f.thisFrame = rc.pframeTarget()
}

func (rc *svtRateControl) pframeTarget() int {
	diff := rc.optimalBufferLevel - rc.bufferLevel
	onePct := 1 + rc.optimalBufferLevel/100
	minTarget := max(rc.avgFrameBandwidth>>4, svtRCFrameOverhead)
	target := rc.avgFrameBandwidth
	if diff > 0 {
		pctLow := int(min(diff/onePct, svtRCUnderShootPct))
		target -= target * pctLow / 200
	} else if diff < 0 {
		pctHigh := int(min(-diff/onePct, svtRCOverShootPct))
		target += target * pctHigh / 200
	}
	return max(minTarget, target)
}

func (rc *svtRateControl) iframeTarget() int {
	var target int
	if rc.pictureNumber == 0 {
		target = int(min(rc.startingBufferLevel/2, math.MaxInt32))
	} else {
		kfBoost := 32
		kfBoost = max(kfBoost, int(2*rc.framerate-16))
		if float64(rc.framesSinceKey) < rc.framerate/2 {
			kfBoost = int(float64(kfBoost*rc.framesSinceKey) / (rc.framerate / 2))
		}
		target = ((16 + kfBoost) * rc.avgFrameBandwidth) >> 4
	}
	// av1_rc_clamp_iframe_target_size
	target = min(target, rc.avgFrameBandwidth*svtRCMaxIntraPct/100)
	return min(target, rc.maxFrameBandwidth)
}

// activeWorst is calc_active_worst_quality_no_stats_cbr.
func (rc *svtRateControl) activeWorst(f *svtRCFrame) int {
	if f.key {
		return rc.worstQuality
	}
	critical := rc.optimalBufferLevel >> 3
	ambient := rc.avgFrameQIndex[svtFrameInter]
	if rc.frameUpdated < 4 {
		ambient = min(rc.avgFrameQIndex[svtFrameInter], rc.avgFrameQIndex[svtFrameKey])
	}
	ambient = min(rc.worstQuality, ambient)
	var aw int
	switch {
	case rc.bufferLevel > rc.optimalBufferLevel:
		aw = min(rc.worstQuality, ambient*5/4)
		maxDown := aw / 3
		if maxDown != 0 {
			step := (rc.maximumBufferSize - rc.optimalBufferLevel) / int64(maxDown)
			adj := 0
			if step != 0 {
				adj = int((rc.bufferLevel - rc.optimalBufferLevel) / step)
			}
			aw -= adj
		}
	case rc.bufferLevel > critical:
		aw = min(rc.worstQuality, ambient)
		if critical != 0 {
			step := rc.optimalBufferLevel - critical
			adj := 0
			if step != 0 {
				adj = int(int64(rc.worstQuality-ambient) * (rc.optimalBufferLevel - rc.bufferLevel) / step)
			}
			aw += adj
		}
	default:
		aw = rc.worstQuality
	}
	return aw
}

// svtGetActiveQuality is rc_vbr_cbr.c get_active_quality.
func svtGetActiveQuality(q, boost, low, high int, lowMotion, highMotion *[256]int) int {
	if boost > high {
		return lowMotion[q]
	}
	if boost < low {
		return highMotion[q]
	}
	gap := high - low
	offset := high - boost
	qdiff := highMotion[q] - lowMotion[q]
	return lowMotion[q] + (offset*qdiff+(gap>>1))/gap
}

// activeBest is calc_active_best_quality_no_stats_cbr.
func (rc *svtRateControl) activeBest(f *svtRCFrame, activeWorst int) int {
	best := rc.bestQuality
	if f.key {
		if f.frameOffset > 0 {
			best = svtGetActiveQuality(rc.avgFrameQIndex[svtFrameKey], rc.kfBoost, svtRCBoostKFLow, svtRCBoostKFHigh,
				&svtRCKFLowMotionMinQ, &svtRCKFHighMotionMinQ)
			adj := 1.0
			if f.width*f.height <= 352*288 {
				adj -= 0.25
			}
			qv := svtConvertQIndexToQ(best)
			best += svtComputeQDelta(qv, qv*adj)
		}
		return best
	}
	refQ := 0
	if f.refValid {
		refQ = f.refBaseQ
	}
	if f.flat {
		arfQ := max(0, refQ-30)
		if arfQ < activeWorst {
			return svtRCRTCMinQ[arfQ]
		}
		return svtRCRTCMinQ[max(0, min(activeWorst, 255))]
	}
	arfQ := max(0, refQ-30)
	best = svtRCRTCMinQ[arfQ]
	q := activeWorst
	for delta := f.temporalLayer - f.refLayer; delta > 0; delta-- {
		best = (best + q + 1) / 2
	}
	return best
}

func (rc *svtRateControl) rateCorrectionFactor(key bool) float64 {
	r := rc.rateCorrection[svtRCInterNormal]
	if key {
		r = rc.rateCorrection[svtRCKFStd]
	}
	return min(max(r, svtRCMinBPBFactor), svtRCMaxBPBFactor)
}

// regulateQ is av1_rc_regulate_q (find_closest_qindex_by_rate, then
// adjust_q_cbr_flat / adjust_q_cbr).
func (rc *svtRateControl) regulateQ(f *svtRCFrame, best, worst int) int {
	ft := svtFrameInter
	if f.key {
		ft = svtFrameKey
	}
	cf := rc.rateCorrectionFactor(f.key)
	targetBPM := int((uint64(f.thisFrame) << svtRCBperMBNormBits) / uint64(rc.mbs))
	low, high := best, worst
	for low < high {
		mid := (low + high) >> 1
		if svtRCBitsPerMB(ft, mid, cf, f.screen) > targetBPM {
			low = mid + 1
		} else {
			high = mid
		}
	}
	cur := low
	curBits := svtRCBitsPerMB(ft, cur, cf, f.screen)
	curDiff := math.MaxInt32
	if curBits <= targetBPM {
		curDiff = targetBPM - curBits
	}
	prevDiff := math.MaxInt32
	if curDiff != math.MaxInt32 && cur != best {
		prevDiff = svtRCBitsPerMB(ft, cur-1, cf, f.screen) - targetBPM
	}
	q := cur
	if curDiff > prevDiff {
		q = cur - 1
	}
	if f.flat {
		return rc.adjustQCBRFlat(f, q)
	}
	return rc.adjustQCBR(f, q)
}

func (rc *svtRateControl) bandwidthChanged() bool {
	d := rc.avgFrameBandwidth - rc.prevAvgFrameBandwidth
	if d < 0 {
		d = -d
	}
	return float64(d) > 0.1*float64(rc.avgFrameBandwidth)
}

// adjustQCBRFlat is rc_vbr_cbr.c adjust_q_cbr_flat.
func (rc *svtRateControl) adjustQCBRFlat(f *svtRCFrame, q int) int {
	overshootBufferLow := rc.rc1Frame == -1 && rc.curAvgBaseMEDist > 1000 &&
		rc.bufferLevel < rc.optimalBufferLevel>>1 && rc.framesSinceKey > 4
	maxDeltaUp := 90
	if overshootBufferLow {
		maxDeltaUp = 120
	}
	var maxDeltaDown int
	if f.screen {
		maxDeltaDown = min(max(rc.q1Frame/16, 1), 8)
	} else {
		maxDeltaDown = min(max(rc.q1Frame/8, 1), 16)
	}
	if !f.key && rc.framesSinceKey > 1 && rc.q1Frame > 0 && rc.q2Frame > 0 && !rc.bandwidthChanged() {
		if rc.rc1Frame*rc.rc2Frame == -1 && rc.q1Frame != rc.q2Frame && !overshootBufferLow {
			qclamp := min(max(q, min(rc.q1Frame, rc.q2Frame)), max(rc.q1Frame, rc.q2Frame))
			if rc.rc1Frame == -1 && q > qclamp && rc.framesSinceKey > 10 {
				q = (q + qclamp) >> 1
			} else {
				q = qclamp
			}
		}
		if rc.prevAvgBaseMEDist > 0 && rc.framesSinceKey > 5 && rc.curAvgBaseMEDist > 0 {
			delta := float64(rc.curAvgBaseMEDist)/float64(rc.prevAvgBaseMEDist) - 1.0
			if delta < 0.0 && rc.bufferLevel > rc.optimalBufferLevel>>2 && q > rc.worstQuality>>1 {
				adj := 1.0 + 0.5*math.Tanh(4.0*delta)
				qv := svtConvertQIndexToQ(q)
				q += svtComputeQDelta(qv, qv*adj)
			} else if rc.q1Frame-q > 0 && delta > 0.1 &&
				rc.bufferLevel < min(rc.maximumBufferSize, rc.optimalBufferLevel<<1) {
				q = (3*q + rc.q1Frame) >> 2
			}
		}
		if rc.q1Frame-q > maxDeltaDown {
			q = rc.q1Frame - maxDeltaDown
		} else if q-rc.q1Frame > maxDeltaUp {
			q = rc.q1Frame + maxDeltaUp
		}
	}
	return max(min(q, rc.worstQuality), rc.bestQuality)
}

// adjustQCBR is rc_vbr_cbr.c adjust_q_cbr (hierarchical low delay).
func (rc *svtRateControl) adjustQCBR(f *svtRCFrame, q int) int {
	hl := min(max(f.hierLevels, 0), 5)
	tl := min(max(f.temporalLayer, 0), 5)
	maxDelta := svtRCMaxDeltaPerLayer[hl][tl]
	var maxDeltaDown int
	if f.screen {
		maxDeltaDown = min(maxDelta, max(1, rc.q1Frame/2))
	} else {
		maxDeltaDown = min(maxDelta, max(1, rc.q1Frame/3))
	}
	if !f.key && rc.framesSinceKey > 1 && !rc.bandwidthChanged() {
		if f.temporalLayer == 0 && rc.prevAvgBaseMEDist > 0 && rc.framesSinceKey > 5 && rc.curAvgBaseMEDist > 0 {
			delta := float64(rc.curAvgBaseMEDist)/float64(rc.prevAvgBaseMEDist) - 1.0
			if delta < 0.0 && rc.bufferLevel > rc.optimalBufferLevel>>2 && q > rc.worstQuality>>1 {
				adj := 1.0 + 0.5*math.Tanh(4.0*delta)
				qv := svtConvertQIndexToQ(q)
				q += svtComputeQDelta(qv, qv*adj)
			}
		}
		if rc.q1Frame-q > maxDeltaDown {
			q = rc.q1Frame - maxDeltaDown
		}
	}
	return max(min(q, rc.worstQuality), rc.bestQuality)
}

// pickQ is rc_pick_q_and_bounds_no_stats_cbr followed by
// svt_av1_rc_calc_qindex_rate_control's clamp and reference-q floor.
func (rc *svtRateControl) pickQ(f *svtRCFrame) uint8 {
	aw := rc.activeWorst(f)
	ab := rc.activeBest(f, aw)
	ab = min(max(ab, rc.bestQuality), rc.worstQuality)
	aw = min(max(aw, ab), rc.worstQuality)
	top := aw
	if f.key && !rc.thisKeyFrameForced && f.frameOffset != 0 {
		qd := rc.computeQDeltaByRate(svtFrameKey, aw, 2.0, f.screen)
		top = max(aw+qd, ab)
	}
	var q int
	if f.key && rc.thisKeyFrameForced {
		q = rc.lastBoosted
	} else {
		q = rc.regulateQ(f, ab, aw)
		if q > top {
			if f.thisFrame >= rc.maxFrameBandwidth {
				top = q
			} else {
				q = top
			}
		}
	}
	q = min(max(q, rc.bestQuality), rc.worstQuality)
	// Limit the qindex based on the reference's (find_min_ref_base_q_idx;
	// gop_constraint_rc is off).
	refQ := -1
	if f.refValid && !f.refIsI && (f.refLayer < f.temporalLayer || f.flat) {
		refQ = f.refBaseQ
	}
	if f.temporalLayer != 0 && !f.flat {
		q = max(q, refQ)
	} else {
		q = max(q, refQ-16)
	}
	return uint8(min(max(q, rc.bestQuality), rc.worstQuality))
}

// svtCRStats carries the cyclic refresh segment state the correction
// factor update weighs (cr->actual_num_seg*_sbs, qindex_delta).
type svtCRStats struct {
	apply        bool
	seg1, seg2   int
	sbCount      int
	delta1       int
	delta2       int
	percentFresh int
}

// estimateBitsAtQ is av1_estimate_bits_at_q.
func (rc *svtRateControl) estimateBitsAtQ(frameType, q int, correction float64, screen bool) int {
	bpm := svtRCBitsPerMB(frameType, max(0, min(q, 255)), correction, screen)
	return max(svtRCFrameOverhead, int((uint64(bpm)*uint64(rc.mbs))>>svtRCBperMBNormBits))
}

// postencode is svt_av1_rc_postencode_update + svt_aom_update_rc_counts.
func (rc *svtRateControl) postencode(f *svtRCFrame, qindex int, frameBits int, cr svtCRStats) {
	ft := svtFrameInter
	if f.key {
		ft = svtFrameKey
	}
	// av1_rc_update_rate_correction_factors
	rcf := rc.rateCorrectionFactor(f.key)
	var projected int
	if cr.apply && cr.sbCount > 0 {
		w1 := float64(cr.seg1) / float64(cr.sbCount)
		w2 := float64(cr.seg2) / float64(cr.sbCount)
		projected = int(math.Round((1.0-w1-w2)*float64(rc.estimateBitsAtQ(ft, qindex, rcf, f.screen)) +
			w1*float64(rc.estimateBitsAtQ(ft, qindex+cr.delta1, rcf, f.screen)) +
			w2*float64(rc.estimateBitsAtQ(ft, qindex+cr.delta2, rcf, f.screen))))
	} else {
		projected = rc.estimateBitsAtQ(ft, qindex, rcf, f.screen)
	}
	correction := 100
	if projected > svtRCFrameOverhead {
		correction = int(100 * int64(frameBits) / int64(projected))
	}
	correction = max(correction, 25)
	rc.q2Frame, rc.q1Frame = rc.q1Frame, qindex
	rc.rc2Frame = rc.rc1Frame
	switch {
	case correction > 110:
		rc.rc1Frame = -1
	case correction < 90:
		rc.rc1Frame = 1
	default:
		rc.rc1Frame = 0
	}
	var limit float64
	if correction > 0 {
		if f.screen {
			limit = 0.25 + 0.5*min(0.5, math.Abs(math.Log10(0.01*float64(correction))))
		} else {
			limit = 0.25 + 0.75*min(0.5, math.Abs(math.Log10(0.01*float64(correction))))
		}
	} else {
		limit = 0.75
	}
	if cr.apply {
		if correction > 125 {
			rc.percentRefreshAdjustment = max(rc.percentRefreshAdjustment-1, -5)
			rc.rateRatioQDeltaAdjustment = max(rc.rateRatioQDeltaAdjustment-0.05, -0.0)
		} else if correction < 50 {
			rc.percentRefreshAdjustment = min(rc.percentRefreshAdjustment+1, 5)
			rc.rateRatioQDeltaAdjustment = min(rc.rateRatioQDeltaAdjustment+0.05, 0.25)
		}
	}
	if correction > 101 {
		correction = int(100 + float64(correction-100)*limit)
		rcf = rcf * float64(correction) / 100
		rcf = min(rcf, svtRCMaxBPBFactor)
	} else if correction < 99 {
		tmp := 100 / float64(correction)
		tmp = 1.0 + (tmp-1.0)*limit
		tmp = 1.0 / tmp
		correction = int(100 * tmp)
		rcf = rcf * float64(correction) / 100
		rcf = max(rcf, svtRCMinBPBFactor)
	}
	rcf = min(max(rcf, svtRCMinBPBFactor), svtRCMaxBPBFactor)
	if f.key {
		rc.rateCorrection[svtRCKFStd] = rcf
	} else {
		rc.rateCorrection[svtRCInterNormal] = rcf
	}

	if f.key {
		rc.avgFrameQIndex[svtFrameKey] = (3*rc.avgFrameQIndex[svtFrameKey] + qindex + 2) >> 2
		rc.lastQ[svtFrameKey] = int(svtConvertQIndexToQ(qindex))
		rc.frameUpdated = 0
	} else {
		rc.frameUpdated++
		// LF_UPDATE frames only (GF/ARF/INTNL_ARF keep the inter average).
		if !f.gfUpdate {
			rc.avgFrameQIndex[svtFrameInter] = (3*rc.avgFrameQIndex[svtFrameInter] + qindex + 2) >> 2
			rc.lastQ[svtFrameInter] = int(svtConvertQIndexToQ(qindex))
		}
	}
	// constrained_gf_group holds (intra_period_length -1 clips
	// baseline_gf_interval to frames_to_key), so GF_UPDATE does not count.
	if qindex < rc.lastBoosted || f.key {
		rc.lastBoosted = qindex
	}
	// update_buffer_level (every frame here is showable).
	rc.bitsOffTarget += int64(rc.avgFrameBandwidth - frameBits)
	rc.bitsOffTarget = min(rc.bitsOffTarget, rc.maximumBufferSize)
	rc.bufferLevel = rc.bitsOffTarget
	rc.prevAvgFrameBandwidth = rc.avgFrameBandwidth
	if f.key {
		rc.framesSinceKey = 0
	}
	// svt_aom_update_rc_counts
	rc.framesSinceKey++
	rc.framesToKey--
	rc.pictureNumber++
}
