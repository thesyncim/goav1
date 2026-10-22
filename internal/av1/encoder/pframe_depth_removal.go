package encoder

// pframe_depth_removal.go ports SVT-AV1's depth-removal signal chain for the
// realtime P-frame partitioner (ENCODER_DEPTH_PLAN.md Phase 1):
//
//   1. One shared full-pel sweep per 64x64 superblock over LAST's recon
//      (open_loop_me_fullpel_search_sblock, Source/Lib/Codec/
//      motion_estimation.c:754) — an 8-wide x 3-tall search area (the preset-12
//      rtc me_sa {8,1} floor-clamped to MAX(3, h), enc_mode_config.c
//      set_me_search_params + motion_estimation.c:1255-1281) centered on the
//      HME seed after SVT's zero-center arbitration (check_00_center,
//      motion_estimation.c:1059). At every position the 64 8x8 SADs are
//      computed and hierarchically summed 8x8 -> 16x16 -> 32x32 -> 64x64 with
//      an INDEPENDENT (best SAD, best MV) argmin per PU — 85 PUs total
//      (svt_ext_sad_calculation_8x8_16x16_c, motion_estimation.c:99 +
//      svt_ext_sad_calculation_32x32_64x64_c, motion_estimation.c:163).
//   2. compute_distortion (motion_estimation.c:2739): dist_64/32/16/8 sums of
//      the per-PU best SADs plus me_8x8_cost_variance.
//   3. The TRUE disallow_below_16x16 gate (set_depth_removal_level_controls,
//      Source/Lib/Codec/enc_mode_config.c:2935-3248): cost arm (RDCOST against
//      av1_lambda_mode_decision8_bit_sad) and deviation arm
//      (dev_16x16_to_8x8 < dev_th) with the me_8x8_cost_variance noise guard
//      and QP scaling, at preset-12 levels 9 (base frames) / 14 (leaf frames).
//
// The sweep is THE full-pel ME of every full superblock of every inter
// frame (SVT runs its open-loop ME once per SB and MD reads the per-PU
// results; pframe_motion.go maps each leaf to its PU as
// svt_aom_get_me_block_offset does). Its SADs are SVT's SUB_SAD_SEARCH
// (svt_aom_sig_deriv_me sets me_search_method = SUB_SAD_SEARCH for every
// non-TF picture): the even rows of each 8x8, doubled
// (svt_ext_eight_sad_calculation_8x8_16x16). goav1 sweeps LAST's recon where
// SVT sweeps the open-loop pyramid.

// realtimeSBMultiSizeME holds one superblock's multi-size full-pel ME results:
// an independent (bestSAD, bestMV) per PU in SVT's ME tier-zero layout
// (me_ctx->p_sb_best_sad / p_sb_best_mv binding, motion_estimation.c:
// 1298-1307) plus the compute_distortion sums. Value arrays only — the struct
// lives inline in realtimeVarPartSB so the per-frame grid stays zero-alloc.
//
// PU indexing (ME_TIER_ZERO_PU_*):
//
//	[0]      64x64
//	[1..4]   32x32 quadrants, raster order over the SB
//	[5..20]  16x16, quadrant-grouped: index 5 + q*4 + w where q is the 32x32
//	         quadrant (raster) and w the 16x16 within it (raster) — the
//	         open_loop_me_get_search_point_results_block walk order
//	[21..84] 8x8: index 21 + n16*4 + c with c the corner within the 16x16
//	         (0 TL, 1 TR, 2 BL, 3 BR)
type realtimeSBMultiSizeME struct {
	valid   bool
	bestSAD [85]uint32
	// bestMV holds FULL-PEL offsets (x, y) relative to the block position,
	// not eighth-pel motion.Vector units.
	bestMV [85][2]int16

	// compute_distortion (motion_estimation.c:2739-2785) sums. The SB-area
	// normalization ((dist * 64*64) / pix_num) is the identity here because
	// the sweep only runs on full 64x64 superblocks inside the frame.
	dist64, dist32, dist16, dist8 uint32
	me8x8CostVariance             uint32
}

const realtimeSweepSADMax = ^uint32(0)

// realtimeSweepSearchArea is the preset-12 rtc full-pel search area: me_sa
// sa_min/sa_max = {8, 1} (set_me_search_params, enc_mode_config.c, rtc
// non-sc >M10 branch), then width rounded to a multiple of 8 and height
// floored to 3 (motion_estimation.c:1255-1259).
const (
	realtimeSweepSearchW = 8
	realtimeSweepSearchH = 3
)

// realtimeSweepEarlyExitTH is me_ctx->me_early_exit_th at preset 12 rtc
// (enc_mode_config.c:914): BLOCK_SIZE_64 * BLOCK_SIZE_64 * 12.
const realtimeSweepEarlyExitTH = 64 * 64 * 12

func (me *realtimeSBMultiSizeME) reset() {
	me.valid = false
	for i := range me.bestSAD {
		me.bestSAD[i] = realtimeSweepSADMax
		me.bestMV[i] = [2]int16{}
	}
}

func (me *realtimeSBMultiSizeME) update(pu int, sad uint32, mvX, mvY int) {
	// Strict '<' keeps the earliest candidate in scan order on ties, matching
	// svt_ext_sad_calculation_8x8_16x16_c's improvement checks.
	if sad < me.bestSAD[pu] {
		me.bestSAD[pu] = sad
		me.bestMV[pu] = [2]int16{int16(mvX), int16(mvY)}
	}
}

// sweepSearchPoint4 ports open_loop_me_get_eight_search_point_results_block
// (motion_estimation.c:407) narrowed to a four-candidate group so the existing
// batched sad8x8x4 kernels do the pixel work: it scores the 64 8x8 blocks of
// the SB at (px, py) against the reference positions (mvX..mvX+3, mvY) and
// hierarchically folds the sums exactly as
// svt_ext_all_sad_calculation_8x8_16x16 + svt_ext_eight_sad_calculation_
// 32x32_64x64 do (full-SAD variant). Requires src/ref strides equal (the
// sad8x8x4 kernel contract); callers fall back to sweepSearchPoint1 otherwise.
func (me *realtimeSBMultiSizeME) sweepSearchPoint4(src, ref []byte, stride int, srcBase, refBase, mvX, mvY int) {
	var sad16 [16][4]uint32
	for n := range 16 {
		q, w := n>>2, n&3
		x16 := (q&1)*32 + (w&1)*16
		y16 := (q>>1)*32 + (w>>1)*16
		var sum0, sum1, sum2, sum3 uint32
		for c := range 4 {
			xo := x16 + (c&1)*8
			yo := y16 + (c>>1)*8
			so := srcBase + yo*stride + xo
			ro := refBase + yo*stride + xo
			// SUB_SAD_SEARCH (sig_deriv_me: me_search_method at every rtc preset):
			// the even rows of the 8x8 block, doubled.
			s0, s1, s2, s3 := sad8x4x4(src[so:], ref[ro:], ref[ro+1:], ref[ro+2:], ref[ro+3:], 2*stride)
			s0, s1, s2, s3 = s0<<1, s1<<1, s2<<1, s3<<1
			pu := 21 + n*4 + c
			me.update(pu, uint32(s0), mvX, mvY)
			me.update(pu, uint32(s1), mvX+1, mvY)
			me.update(pu, uint32(s2), mvX+2, mvY)
			me.update(pu, uint32(s3), mvX+3, mvY)
			sum0 += uint32(s0)
			sum1 += uint32(s1)
			sum2 += uint32(s2)
			sum3 += uint32(s3)
		}
		sad16[n] = [4]uint32{sum0, sum1, sum2, sum3}
		me.update(5+n, sum0, mvX, mvY)
		me.update(5+n, sum1, mvX+1, mvY)
		me.update(5+n, sum2, mvX+2, mvY)
		me.update(5+n, sum3, mvX+3, mvY)
	}
	for i := range 4 {
		var sad64 uint32
		for q := range 4 {
			s32 := sad16[q*4][i] + sad16[q*4+1][i] + sad16[q*4+2][i] + sad16[q*4+3][i]
			me.update(1+q, s32, mvX+i, mvY)
			sad64 += s32
		}
		me.update(0, sad64, mvX+i, mvY)
	}
}

// sweepSearchPoint1 ports open_loop_me_get_search_point_results_block
// (motion_estimation.c:456) for a single candidate position; the
// unequal-stride and odd-tail path.
func (me *realtimeSBMultiSizeME) sweepSearchPoint1(src []byte, srcStride int, ref []byte, refStride int, srcBase, refBase, mvX, mvY int) {
	var sad16 [16]uint32
	for n := range 16 {
		q, w := n>>2, n&3
		x16 := (q&1)*32 + (w&1)*16
		y16 := (q>>1)*32 + (w>>1)*16
		var sum uint32
		for c := range 4 {
			xo := x16 + (c&1)*8
			yo := y16 + (c>>1)*8
			s := uint32(sad8x4DualPureGo(src[srcBase+yo*srcStride+xo:], 2*srcStride, ref[refBase+yo*refStride+xo:], 2*refStride)) << 1
			me.update(21+n*4+c, s, mvX, mvY)
			sum += s
		}
		sad16[n] = sum
		me.update(5+n, sum, mvX, mvY)
	}
	var sad64 uint32
	for q := range 4 {
		s32 := sad16[q*4] + sad16[q*4+1] + sad16[q*4+2] + sad16[q*4+3]
		me.update(1+q, s32, mvX, mvY)
		sad64 += s32
	}
	me.update(0, sad64, mvX, mvY)
}

// subSADRect is the row-subsampled, doubled SAD of the bw x bh source block
// at src[srcOff] against ref[refOff] (both padded planes at one stride): the
// get_zz_sad of a partial superblock.
func subSADRect(src []byte, srcOff int, ref []byte, refOff, stride, bw, bh int) uint32 {
	var sum uint32
	for r := 0; r < bh; r += 2 {
		a := src[srcOff+r*stride : srcOff+r*stride+bw]
		b := ref[refOff+r*stride : refOff+r*stride+bw]
		for c := range a {
			d := int(a[c]) - int(b[c])
			if d < 0 {
				d = -d
			}
			sum += uint32(d)
		}
	}
	return sum << 1
}

// motionEstimateSB is svt_aom_motion_estimation_b64 for one superblock at
// preset 12 rtc: init_zz_sad, the HME center (svt_hme.go), the 8x3 sweep with
// per-PU argmins (open_loop_me_fullpel_search_sblock) and compute_distortion,
// on the padded current (h.cur) and previous (h.prev) source pictures. It
// returns the HME level-1 SAD and me_8x8_distortion.
func (h *hmeState) motionEstimateSB(me *realtimeSBMultiSizeME, width, height, px, py int, isRef bool) (uint32, uint64) {
	me.reset()
	cur, prev := &h.cur, &h.prev
	stride := cur.stride
	bw, bh := min(64, width-px), min(64, height-py)
	full := bw == 64 && bh == 64
	sbOff := cur.org + py*stride + px

	// init_zz_sad / get_zz_sad: the SB's zero-motion SAD, row-subsampled and
	// doubled, normalized for an incomplete SB.
	var zz uint32
	if full {
		zz = uint32(sweepSAD64Sub(cur.pix, stride, sbOff, prev.pix, sbOff))
	} else {
		zz = subSADRect(cur.pix, sbOff, prev.pix, sbOff, stride, bw, bh)
		zz = uint32(uint64(zz) * 64 * 64 / uint64(bw*bh))
	}

	// HME center, skipped below me_early_exit_th >> 2.
	xc, yc := 0, 0
	var l1 uint32
	if zz >= realtimeSweepEarlyExitTH>>2 && h.svt.ready() {
		xc, yc, l1 = h.svt.center(px, py)
	}
	// Keep the reference block within one superblock of the picture (SVT's
	// search-area correction against its padding).
	xc = min(max(xc, -px-63), width-1-px)
	yc = min(max(yc, -py-63), height-1-py)

	nx, ny := realtimeSweepSearchW, realtimeSweepSearchH
	earlyExit := zz < realtimeSweepEarlyExitTH/6
	if earlyExit {
		nx, ny = 1, 1
	}
	if !earlyExit && (xc != 0 || yc != 0) && isRef {
		// check_00_center: the HME center survives only when its subsampled
		// 64x64 SAD beats the zero-motion SAD; ties collapse to zero motion.
		zeroSAD := sweepSAD64Sub(cur.pix, stride, sbOff, prev.pix, sbOff)
		hmeSAD := sweepSAD64Sub(cur.pix, stride, sbOff, prev.pix, sbOff+yc*stride+xc)
		if zeroSAD <= hmeSAD {
			xc, yc = 0, 0
		}
	}
	// x/y_search_area_origin = center - (area >> 1).
	x0, y0 := xc-(nx>>1), yc-(ny>>1)
	me.sweepArea(cur.pix, stride, sbOff, prev.pix, prev.border+px, prev.border+py, x0, y0, nx, ny)

	// compute_distortion: per-size sums of the per-PU best SADs, normalized
	// to the SB area, plus the 8x8 cost variance.
	var dist32, dist16, dist8 uint64
	for i := 1; i < 5; i++ {
		dist32 += uint64(me.bestSAD[i])
	}
	for i := 5; i < 21; i++ {
		dist16 += uint64(me.bestSAD[i])
	}
	for i := 21; i < 85; i++ {
		dist8 += uint64(me.bestSAD[i])
	}
	mean8 := dist8 / 64
	var sumOfSq uint64
	for i := 21; i < 85; i++ {
		diff := int64(me.bestSAD[i]) - int64(mean8)
		sumOfSq += uint64(diff * diff)
	}
	me.me8x8CostVariance = uint32(sumOfSq / 64)
	pix := uint64(bw * bh)
	me.dist64 = uint32(uint64(me.bestSAD[0]) * 4096 / pix)
	me.dist32 = uint32(dist32 * 4096 / pix)
	me.dist16 = uint32(dist16 * 4096 / pix)
	me.dist8 = uint32(dist8 * 4096 / pix)
	me.valid = true
	return l1, dist8 * 4096 / pix
}

// sweepSAD64Sub is the subsampled 64x64 SAD of check_00_center.
func sweepSAD64Sub(src []byte, stride, srcOff int, ref []byte, refOff int) int {
	s2 := stride << 1
	return (sad32x32Dual(src[srcOff:], s2, ref[refOff:], s2) + sad32x32Dual(src[srcOff+32:], s2, ref[refOff+32:], s2)) << 1
}

// sweepArea scores the nx x ny positions (x0+i, y0+j) of the SB whose source
// block starts at src[srcBase] (stride) against ref, where the reference
// block at motion (mx, my) starts at ref[(refOy+my)*stride + refOx + mx]
// (source and reference share the stride). Positions run y outer, x inner
// (open_loop_me_fullpel_search_sblock).
func (me *realtimeSBMultiSizeME) sweepArea(src []byte, stride, srcBase int, ref []byte, refOx, refOy, x0, y0, nx, ny int) {
	for yi := 0; yi < ny; yi++ {
		mvY := y0 + yi
		rowBase := (refOy+mvY)*stride + refOx
		xi := 0
		if nx == 8 && me.sweepSearchPoint8(src, ref, stride, srcBase, rowBase+x0, x0, mvY) {
			xi = 8
		}
		for ; xi+4 <= nx; xi += 4 {
			mvX := x0 + xi
			me.sweepSearchPoint4(src, ref, stride, srcBase, rowBase+mvX, mvX, mvY)
		}
		for ; xi < nx; xi++ {
			mvX := x0 + xi
			me.sweepSearchPoint1(src, stride, ref, stride, srcBase, rowBase+mvX, mvX, mvY)
		}
	}
}

// depthRemovalCtrls carries the per-level parameters of
// set_depth_removal_level_controls (enc_mode_config.c:2979-3130). Phase 1
// consumes the disallow_below_16x16 arms; Phase 2 the 32x32/64x64 cost
// multipliers and dev_32x32_to_16x16_th. costMult4x4 (disallow_4x4) still
// only rides along — goav1's realtime tree never goes below 8x8.
type depthRemovalCtrls struct {
	enabled     bool
	costMult4x4 uint32
	costMult16  uint32
	costMult32  uint32
	costMult64  uint32
	dev16To8Th  int64
	dev32To16Th int64
	qpScale     int64
}

const realtimeMaxSignedValue = int64(^uint64(0) >> 1) // MAX_SIGNED_VALUE

// depthRemovalLevels[level] mirrors the upstream switch verbatim.
var depthRemovalLevels = [16]depthRemovalCtrls{
	0:  {},
	1:  {enabled: true, costMult4x4: 64, dev16To8Th: 0, dev32To16Th: 0, qpScale: 1},
	2:  {enabled: true, costMult4x4: 64, dev16To8Th: 10, dev32To16Th: 0, qpScale: 1},
	3:  {enabled: true, costMult4x4: 64, dev16To8Th: 20, dev32To16Th: 0, qpScale: 1},
	4:  {enabled: true, costMult4x4: 64, dev16To8Th: 30, dev32To16Th: 0, qpScale: 1},
	5:  {enabled: true, costMult4x4: 64, costMult16: 6, costMult32: 6, dev16To8Th: 40, dev32To16Th: 0, qpScale: 1},
	6:  {enabled: true, costMult4x4: 64, costMult16: 6, costMult32: 6, dev16To8Th: 50, dev32To16Th: 25, qpScale: 1},
	7:  {enabled: true, costMult4x4: 64, costMult16: 6, costMult32: 6, dev16To8Th: 50, dev32To16Th: 25, qpScale: 2},
	8:  {enabled: true, costMult4x4: 64, costMult16: 16, costMult32: 8, costMult64: 8, dev16To8Th: 100, dev32To16Th: 50, qpScale: 3},
	9:  {enabled: true, costMult4x4: 64, costMult16: 32, costMult32: 8, costMult64: 8, dev16To8Th: 100, dev32To16Th: 50, qpScale: 3},
	10: {enabled: true, costMult4x4: 64, costMult16: 128, costMult32: 8, costMult64: 8, dev16To8Th: 200, dev32To16Th: 75, qpScale: 3},
	11: {enabled: true, costMult4x4: 64, costMult16: 128, costMult32: 8, costMult64: 8, dev16To8Th: 250, dev32To16Th: 125, qpScale: 3},
	12: {enabled: true, costMult4x4: 64, costMult16: 128, costMult32: 16, costMult64: 8, dev16To8Th: 250, dev32To16Th: 150, qpScale: 4},
	13: {enabled: true, costMult4x4: 64, costMult16: 256, costMult32: 16, costMult64: 8, dev16To8Th: 250, dev32To16Th: 150, qpScale: 4},
	14: {enabled: true, costMult4x4: 64, costMult16: 256, costMult32: 16, costMult64: 16, dev16To8Th: 250, dev32To16Th: 150, qpScale: 4},
	15: {enabled: true, costMult4x4: 64, costMult16: 384, costMult32: 24, costMult64: 24, dev16To8Th: 300, dev32To16Th: 200, qpScale: 4},
}

// depthRemovalLevelForFrame is the preset-12 (>M8) rtc non-sc branch of the
// pic_depth_removal_level ladder (enc_mode_config.c:9287-9295), with SVT's
// input-resolution ranges (svt_aom_derive_input_resolution,
// sequence_control_set.c:120 + definitions.h INPUT_SIZE_*_TH).
func depthRemovalLevelForFrame(width, height int, isBase bool) uint8 {
	const (
		inputSize360pTH = 0x4CE00 // <= INPUT_SIZE_360p_RANGE
		inputSize480pTH = 0xA1400 // <= INPUT_SIZE_480p_RANGE
	)
	numPixels := width * height
	switch {
	case numPixels < inputSize360pTH:
		return 7
	case numPixels < inputSize480pTH:
		if isBase {
			return 9
		}
		return 11
	default:
		if isBase {
			return 9
		}
		return 14
	}
}

// Noise-guard thresholds (enc_mode_config.c:9-10).
const (
	low8x8DistVarTH  = 25000
	high8x8DistVarTH = 50000
)

// depthRemovalRDCost is RDCOST (rd_cost.h:34): rate scaled by the lambda
// multiplier at AV1_PROB_COST_SHIFT=9 precision plus distortion at
// RDDIV_BITS=7.
func depthRemovalRDCost(lambda, rate, dist int64) int64 {
	return ((rate*lambda + (1 << 8)) >> 9) + (dist << 7)
}

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// realtimeDepthRemovalDecision is one SB's depth-removal evaluation across
// the three per-depth flags of set_depth_removal_level_controls
// (enc_mode_config.c:3207-3232): disallow_below_16x16 (cost + dev arms),
// disallow_below_32x32 (cost arm + the dual 32->16/32->8 deviation arm) and
// disallow_below_64x64 (cost arm only, verbatim upstream).
type realtimeDepthRemovalDecision struct {
	below16 bool
	below32 bool
	below64 bool

	cost16Arm bool
	dev16Arm  bool
	cost32Arm bool
	dev32Arm  bool

	dev16     int64
	dev32To16 int64
	dev32To8  int64
}

// realtimeDepthRemovalDecide ports the depth-removal decisions of
// set_depth_removal_level_controls (enc_mode_config.c:3179-3245): the
// me_8x8_cost_variance noise guard, the QP threshold scaling, the absolute
// RD-cost arms and the deviation arms at every depth. SVT's SB-geometry
// guards ((sb_geom->width % 64) == 0 || > 32 for the 64 arm, the mod-32
// equivalent for the 32 arm, dimensions_require_8x8 for the 16 arm) are
// satisfied by construction — the sweep only runs on full 64x64 SBs, so
// width == height == 64 passes all three. The sb_min_sq_size reference
// feedback (:3134-3166) is Phase 4; the delta-qp level modulation
// (:2960-2972) never arms here because goav1 codes frame-uniform q.
func realtimeDepthRemovalDecide(level uint8, qIndex uint8, me *realtimeSBMultiSizeME) realtimeDepthRemovalDecision {
	var dec realtimeDepthRemovalDecision
	if int(level) >= len(depthRemovalLevels) {
		return dec
	}
	ctrls := &depthRemovalLevels[level]
	if !ctrls.enabled || !me.valid {
		return dec
	}
	// pcs->picture_qp is the 0..63 quantizer; quantizer_to_qindex maps it to
	// qindex = qp * 4 (rc tables), so the inverse here is qindex >> 2.
	pictureQP := int64(qIndex) >> 2
	// fast_lambda_md[EB_8_BIT_MD] = av1_lambda_mode_decision8_bit_sad[qindex]
	// (svt_aom_lambda_assign, rc_process.c:451/458).
	fastLambda := int64(av1LambdaModeDecision8BitSAD[qIndex])

	dev16Th := ctrls.dev16To8Th
	dev32To16Th := ctrls.dev32To16Th
	// dev th = f(me_8x8_cost_variance) — the noise guard, verbatim
	// (enc_mode_config.c:3181-3191): the low band quadruples the 16 threshold
	// (32 untouched), the middle band doubles 16 and HALVES 32, the high band
	// zeroes both.
	me8x8CostVariance := int64(me.me8x8CostVariance)
	me8x8CostVariance /= maxInt64(maxInt64(63-(pictureQP+10), 1), 1)
	if me8x8CostVariance < low8x8DistVarTH {
		dev16Th <<= 2
	} else if me8x8CostVariance < high8x8DistVarTH {
		dev16Th <<= 1
		dev32To16Th >>= 1
	} else {
		dev16Th = 0
		dev32To16Th = 0
	}
	// dev th = f(QP) (enc_mode_config.c:3193-3195).
	qpFactor := maxInt64(maxInt64(63-(pictureQP+10), 1)>>4, 1) * ctrls.qpScale
	dev16Th *= qpFactor
	dev32To16Th *= qpFactor
	// dev_32x32_to_8x8_th = f(dev_32x32_to_16x16_th); a bit higher
	// (enc_mode_config.c:3196-3197).
	dev32To8Th := (dev32To16Th * ((1 << 2) + 1)) >> 2

	const costThRate = int64(1) << 13
	const sbSize = int64(64 * 64)
	var costTh16, costTh32, costTh64 int64
	if ctrls.costMult16 != 0 {
		costTh16 = depthRemovalRDCost(fastLambda, costThRate, (sbSize>>3)*int64(ctrls.costMult16))
	}
	if ctrls.costMult32 != 0 {
		costTh32 = depthRemovalRDCost(fastLambda, costThRate, (sbSize>>3)*int64(ctrls.costMult32))
	}
	if ctrls.costMult64 != 0 {
		costTh64 = depthRemovalRDCost(fastLambda, costThRate, (sbSize>>3)*int64(ctrls.costMult64))
	}

	cost64 := depthRemovalRDCost(fastLambda, 0, int64(me.dist64))
	cost32 := depthRemovalRDCost(fastLambda, 0, int64(me.dist32))
	cost16 := depthRemovalRDCost(fastLambda, 0, int64(me.dist16))
	cost8 := depthRemovalRDCost(fastLambda, 0, int64(me.dist8))

	dec.dev32To16 = ((maxInt64(cost32, 1) - maxInt64(cost16, 1)) * 1000) / maxInt64(cost16, 1)
	dec.dev32To8 = ((maxInt64(cost32, 1) - maxInt64(cost8, 1)) * 1000) / maxInt64(cost8, 1)
	dec.dev16 = ((maxInt64(cost16, 1) - maxInt64(cost8, 1)) * 1000) / maxInt64(cost8, 1)

	// enc_mode_config.c:3223-3226 — the 64 arm is cost-only upstream.
	dec.below64 = ctrls.costMult64 != 0 && cost64 < costTh64
	// enc_mode_config.c:3228-3232.
	dec.cost32Arm = ctrls.costMult32 != 0 && cost32 < costTh32
	dec.dev32Arm = dec.dev32To16 < dev32To16Th && dec.dev32To8 < dev32To8Th
	dec.below32 = dec.cost32Arm || dec.dev32Arm
	// enc_mode_config.c:3234-3237.
	dec.cost16Arm = ctrls.costMult16 != 0 && cost16 < costTh16
	dec.dev16Arm = dec.dev16 < dev16Th
	dec.below16 = dec.cost16Arm || dec.dev16Arm
	return dec
}

// av1LambdaModeDecision8BitSAD ports av1_lambda_mode_decision8_bit_sad
// (lambda_rate_tables.h:22), the SAD-domain fast lambda indexed by qindex.
var av1LambdaModeDecision8BitSAD = [256]uint32{
	86, 173, 173, 194, 216, 238, 259, 259, 281, 303, 324, 346, 368, 389, 411, 411,
	433, 454, 476, 498, 519, 541, 563, 563, 584, 606, 628, 649, 671, 693, 693, 714,
	736, 758, 779, 801, 823, 823, 844, 866, 888, 909, 931, 931, 953, 974, 996, 1018,
	1039, 1039, 1061, 1083, 1104, 1126, 1148, 1148, 1169, 1191, 1213, 1234, 1234, 1256, 1278, 1299,
	1321, 1343, 1343, 1364, 1386, 1408, 1429, 1429, 1451, 1473, 1494, 1516, 1516, 1538, 1559, 1581,
	1603, 1603, 1624, 1646, 1668, 1689, 1689, 1711, 1733, 1754, 1754, 1776, 1798, 1819, 1841, 1841,
	1884, 1906, 1949, 1993, 2014, 2058, 2079, 2123, 2144, 2188, 2209, 2253, 2274, 2318, 2339, 2383,
	2404, 2448, 2469, 2513, 2534, 2556, 2599, 2621, 2664, 2707, 2751, 2794, 2837, 2902, 2946, 2989,
	3032, 3076, 3119, 3162, 3206, 3249, 3292, 3336, 3379, 3422, 3487, 3552, 3596, 3661, 3726, 3769,
	3834, 3899, 3942, 4007, 4051, 4116, 4159, 4224, 4311, 4376, 4441, 4506, 4571, 4636, 4701, 4766,
	4831, 4896, 4982, 5047, 5134, 5199, 5264, 5351, 5416, 5481, 5567, 5654, 5740, 5827, 5892, 5979,
	6065, 6152, 6239, 6325, 6412, 6499, 6585, 6694, 6780, 6867, 6975, 7062, 7149, 7257, 7365, 7452,
	7560, 7669, 7777, 7885, 7994, 8102, 8210, 8319, 8427, 8557, 8665, 8795, 8903, 9033, 9163, 9293,
	9423, 9553, 9683, 9835, 9987, 10117, 10290, 10442, 10593, 10767, 10940, 11113, 11308, 11481, 11676, 11893,
	12110, 12326, 12543, 12781, 13041, 13301, 13561, 13865, 14168, 14471, 14818, 15164, 15533, 15944, 16356, 16789,
	17244, 17742, 18262, 18826, 19411, 20039, 20689, 21404, 22140, 22920, 23787, 24675, 25650, 26690, 27773, 28943,
}
