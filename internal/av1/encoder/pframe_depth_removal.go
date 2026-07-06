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
// The sweep is full-SAD (not SVT's SUB_SAD subsampling) per the pinned
// threshold-scale note: goav1 sweeps LAST's recon with full SADs while SVT
// sweeps the open-loop pyramid; the deviation arms are ratio-based and
// scale-free either way. Scalar Go + the existing batched SAD kernels for this
// slice; the dedicated NEON sweep kernel is Phase 4.

import (
	"fmt"
	"os"
	"sync/atomic"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

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

// sweepCheck00Center ports check_00_center (motion_estimation.c:1059): the
// HME search center survives only when its subsampled 64x64 SAD beats the
// zero-MV SAD; ties collapse to zero motion. subsample_sad = 1 (every other
// row, doubled) as upstream.
func sweepCheck00Center(src, ref SourceFrame420, px, py, xc, yc int) (int, int) {
	srcOff := py*src.YStride + px
	srcStride2 := src.YStride << 1
	refStride2 := ref.YStride << 1
	zeroOff := py*ref.YStride + px
	zeroSAD := (sad32x32Dual(src.Y[srcOff:], srcStride2, ref.Y[zeroOff:], refStride2) +
		sad32x32Dual(src.Y[srcOff+32:], srcStride2, ref.Y[zeroOff+32:], refStride2)) << 1
	hmeOff := (py+yc)*ref.YStride + (px + xc)
	hmeSAD := (sad32x32Dual(src.Y[srcOff:], srcStride2, ref.Y[hmeOff:], refStride2) +
		sad32x32Dual(src.Y[srcOff+32:], srcStride2, ref.Y[hmeOff+32:], refStride2)) << 1
	// search_center_cost = MIN(zero_mv_cost, hme_mv_cost); equality selects
	// the zero center.
	if zeroSAD <= hmeSAD {
		return 0, 0
	}
	return xc, yc
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
			s0, s1, s2, s3 := sad8x8x4(src[so:], ref[ro:], ref[ro+1:], ref[ro+2:], ref[ro+3:], stride)
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
			s := uint32(sad8x8Dual(src[srcBase+yo*srcStride+xo:], srcStride, ref[refBase+yo*refStride+xo:], refStride))
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

// realtimeSBMultiSizeMESweep runs the shared per-SB full-pel sweep: the 8x3
// preset-12 search area centered on the HME seed (after check_00_center) with
// per-PU argmins, then the compute_distortion sums. src is the source frame,
// ref is LAST's reconstruction; (px, py) must be a full 64x64 SB inside the
// frame (the realtimeVarPartitionSB caller guarantees it).
func (st *lossyEncodeState) realtimeSBMultiSizeMESweep(me *realtimeSBMultiSizeME, src, ref SourceFrame420, px, py int) {
	me.reset()

	// HME search center for this SB (the seeds are frame-start read-only, so
	// wavefront lanes see identical values).
	xc, yc := 0, 0
	if st.hme != nil {
		xc, yc, _ = st.hme.seedAt(px, py)
	}
	// goav1's recon planes carry no out-of-frame padding (SVT's ME buffers
	// do), so the center and the search window clamp to positions whose whole
	// 64x64 reference block stays inside the frame.
	xc = minInt(maxInt(xc, -px), src.Width-64-px)
	yc = minInt(maxInt(yc, -py), src.Height-64-py)

	// me_early_exit_th (enc_mode_config.c:914; preset-12 rtc = 64*64*12): a
	// tiny zero-MV SB SAD collapses the search area to 1x1 and skips the
	// zero-center arbitration (motion_estimation.c:1261-1266). The static-64
	// bypass grid carries that zz-SAD when it hit (zero MV, full-SAD below
	// the bypass threshold, which is below th/6 = 8192).
	nx := src.Width - 64 + 1
	if nx > realtimeSweepSearchW {
		nx = realtimeSweepSearchW
	}
	ny := src.Height - 64 + 1
	if ny > realtimeSweepSearchH {
		ny = realtimeSweepSearchH
	}
	earlyExit := false
	if st.grid64Cols > 0 {
		idx64 := (py/64)*st.grid64Cols + px/64
		if uint(idx64) < uint(len(st.sad64Grid)) && uint(idx64) < uint(len(st.mv64Grid)) &&
			sadCacheValid(st.sad64Grid[idx64], st.sadCacheEpoch) &&
			st.mv64Grid[idx64] == (motion.Vector{}) &&
			sadCacheValue(st.sad64Grid[idx64]) < realtimeSweepEarlyExitTH/6 {
			nx, ny = 1, 1
			earlyExit = true
		}
	}
	if !earlyExit && (xc != 0 || yc != 0) && st.depthRemovalIsRef {
		// SVT gates check_00_center on me_ctx->is_ref (motion_estimation.c:
		// 1267-1281): only pictures that will be referenced re-arbitrate the
		// HME center against zero motion.
		xc, yc = sweepCheck00Center(src, ref, px, py, xc, yc)
	}

	// x/y_search_area_origin = center - (area >> 1) (motion_estimation.c:
	// 1352-1354), then edge-corrected. SVT shrinks the area at picture edges;
	// with unpadded references we shift the window instead so every full-area
	// SB scores the same 8x3 = 24 positions.
	x0 := minInt(maxInt(xc-(nx>>1), -px), src.Width-64-px-(nx-1))
	y0 := minInt(maxInt(yc-(ny>>1), -py), src.Height-64-py-(ny-1))

	srcBase := py*src.YStride + px
	sameStride := src.YStride == ref.YStride
	// open_loop_me_fullpel_search_sblock (motion_estimation.c:754): y outer,
	// x inner in batches (batch width 4 here for the sad8x8x4 kernels).
	for yi := 0; yi < ny; yi++ {
		mvY := y0 + yi
		rowBase := (py+mvY)*ref.YStride + px
		xi := 0
		if sameStride {
			for ; xi+4 <= nx; xi += 4 {
				mvX := x0 + xi
				me.sweepSearchPoint4(src.Y, ref.Y, src.YStride, srcBase, rowBase+mvX, mvX, mvY)
			}
		}
		for ; xi < nx; xi++ {
			mvX := x0 + xi
			me.sweepSearchPoint1(src.Y, src.YStride, ref.Y, ref.YStride, srcBase, rowBase+mvX, mvX, mvY)
		}
	}

	// compute_distortion (motion_estimation.c:2739-2785): per-size sums of the
	// per-PU best SADs plus the 8x8 cost variance.
	me.dist64 = me.bestSAD[0]
	var dist32, dist16, dist8 uint32
	for i := 1; i < 5; i++ {
		dist32 += me.bestSAD[i]
	}
	for i := 5; i < 21; i++ {
		dist16 += me.bestSAD[i]
	}
	for i := 21; i < 85; i++ {
		dist8 += me.bestSAD[i]
	}
	me.dist32, me.dist16, me.dist8 = dist32, dist16, dist8
	mean8 := uint64(dist8) / 64
	var sumOfSq uint64
	for i := 21; i < 85; i++ {
		diff := int64(me.bestSAD[i]) - int64(mean8)
		sumOfSq += uint64(diff * diff)
	}
	me.me8x8CostVariance = uint32(sumOfSq / 64)
	me.valid = true
}

// depthRemovalCtrls carries the per-level parameters of
// set_depth_removal_level_controls (enc_mode_config.c:2979-3130). Only the
// disallow_below_16x16 arms are consumed in Phase 1; the 32x32/64x64
// multipliers and dev_32x32_to_16x16_th ride along for the Phase 2 arms.
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
	if depthRemovalDisabled {
		return 0
	}
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

// depthRemovalBelow16Decision is one SB's disallow_below_16x16 evaluation.
type depthRemovalBelow16Decision struct {
	disallow bool
	costArm  bool
	devArm   bool
	dev16    int64
}

// realtimeDepthRemovalBelow16 ports the disallow_below_16x16 decision of
// set_depth_removal_level_controls (enc_mode_config.c:3179-3245): the
// me_8x8_cost_variance noise guard, the QP threshold scaling, the absolute
// RD-cost arm and the 16x16-to-8x8 deviation arm. The sb_min_sq_size
// reference feedback (:3134-3166) is Phase 4; the delta-qp level modulation
// (:2960-2972) never arms here because goav1 codes frame-uniform q.
func realtimeDepthRemovalBelow16(level uint8, qIndex uint8, me *realtimeSBMultiSizeME) depthRemovalBelow16Decision {
	var dec depthRemovalBelow16Decision
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
	// dev th = f(me_8x8_cost_variance) — the noise guard, verbatim
	// (enc_mode_config.c:3179-3189).
	me8x8CostVariance := int64(me.me8x8CostVariance)
	me8x8CostVariance /= maxInt64(maxInt64(63-(pictureQP+10), 1), 1)
	if me8x8CostVariance < low8x8DistVarTH {
		dev16Th <<= 2
	} else if me8x8CostVariance < high8x8DistVarTH {
		dev16Th <<= 1
	} else {
		dev16Th = 0
	}
	// dev th = f(QP) (enc_mode_config.c:3191-3195).
	dev16Th *= maxInt64(maxInt64(63-(pictureQP+10), 1)>>4, 1) * ctrls.qpScale

	const costThRate = int64(1) << 13
	const sbSize = int64(64 * 64)
	var disallowBelow16CostTh int64
	if ctrls.costMult16 != 0 {
		disallowBelow16CostTh = depthRemovalRDCost(fastLambda, costThRate, (sbSize>>3)*int64(ctrls.costMult16))
	}

	cost16 := depthRemovalRDCost(fastLambda, 0, int64(me.dist16))
	cost8 := depthRemovalRDCost(fastLambda, 0, int64(me.dist8))
	dec.dev16 = ((maxInt64(cost16, 1) - maxInt64(cost8, 1)) * 1000) / maxInt64(cost8, 1)

	dec.costArm = ctrls.costMult16 != 0 && cost16 < disallowBelow16CostTh
	dec.devArm = dec.dev16 < dev16Th
	// The dimensions_require_8x8 guard is satisfied by construction: the
	// sweep only runs on full 64x64 SBs.
	dec.disallow = dec.costArm || dec.devArm
	return dec
}

// ---------------------------------------------------------------------------
// Instrumentation (Phase 1 first-class): the dev_16 distribution and the
// per-arm 16x16 removal attribution, env-gated (GOAV1_DEPTH_REMOVAL_STATS=1)
// so the production hot path pays one predictable branch. Counters are
// process-global atomics because wavefront lanes build SBs concurrently; the
// per-frame dump in encodePReusing prints and resets after the lanes join.

var depthRemovalStatsEnabled = os.Getenv("GOAV1_DEPTH_REMOVAL_STATS") != ""

// depthRemovalDisabled is a measurement kill-switch: with
// GOAV1_DEPTH_REMOVAL_DISABLE=1 the level ladder returns 0, which skips the
// sweep and the TRUE gate entirely — the decision path is then byte-identical
// to the pre-port encoder (the 18b20370 proxy stays active either way), so
// one binary serves both sides of an interleaved A/B.
var depthRemovalDisabled = os.Getenv("GOAV1_DEPTH_REMOVAL_DISABLE") != ""

// dev16 histogram bucket upper bounds (exclusive); the last bucket is open.
var depthRemovalDev16Buckets = [...]int64{5, 10, 25, 50, 100, 250, 500}

type depthRemovalStatsT struct {
	sbs         atomic.Uint64
	disallowSBs atomic.Uint64
	costArmSBs  atomic.Uint64
	devArmSBs   atomic.Uint64

	dev16HistBase [len(depthRemovalDev16Buckets) + 1]atomic.Uint64
	dev16HistLeaf [len(depthRemovalDev16Buckets) + 1]atomic.Uint64

	// Per-16x16 removal attribution: a "removed" slot is one the variance
	// tree wanted to split below 16x16 (val16 > thresholds[3]) that some arm
	// clamped to PartitionNone.
	wouldSplit16  atomic.Uint64
	removed16Cost atomic.Uint64 // TRUE cost arm only
	removed16Dev  atomic.Uint64 // TRUE dev arm only
	removed16Prox atomic.Uint64 // landed 18b20370 proxy only
	removed16Mult atomic.Uint64 // more than one arm fired
}

var depthRemovalStats depthRemovalStatsT

func depthRemovalDev16Bucket(dev16 int64) int {
	for i, hi := range depthRemovalDev16Buckets {
		if dev16 < hi {
			return i
		}
	}
	return len(depthRemovalDev16Buckets)
}

// noteDepthRemovalSB records one swept SB's gate outcome. wouldSplit and
// proxyFired describe the 16 16x16 slots in [lvl1][lvl2] order.
func noteDepthRemovalSB(dec depthRemovalBelow16Decision, isBase bool, wouldSplit, proxyFired *[4][4]bool) {
	s := &depthRemovalStats
	s.sbs.Add(1)
	if dec.disallow {
		s.disallowSBs.Add(1)
	}
	if dec.costArm {
		s.costArmSBs.Add(1)
	}
	if dec.devArm {
		s.devArmSBs.Add(1)
	}
	b := depthRemovalDev16Bucket(dec.dev16)
	if isBase {
		s.dev16HistBase[b].Add(1)
	} else {
		s.dev16HistLeaf[b].Add(1)
	}
	for i := range wouldSplit {
		for j := range wouldSplit[i] {
			if !wouldSplit[i][j] {
				continue
			}
			s.wouldSplit16.Add(1)
			arms := 0
			if dec.costArm {
				arms++
			}
			if dec.devArm {
				arms++
			}
			if proxyFired[i][j] {
				arms++
			}
			switch {
			case arms > 1:
				s.removed16Mult.Add(1)
			case dec.costArm:
				s.removed16Cost.Add(1)
			case dec.devArm:
				s.removed16Dev.Add(1)
			case proxyFired[i][j]:
				s.removed16Prox.Add(1)
			}
		}
	}
}

func histSnapshotReset(h *[len(depthRemovalDev16Buckets) + 1]atomic.Uint64) [len(depthRemovalDev16Buckets) + 1]uint64 {
	var out [len(depthRemovalDev16Buckets) + 1]uint64
	for i := range h {
		out[i] = h[i].Swap(0)
	}
	return out
}

// dumpDepthRemovalStats prints and resets the per-frame counters. Called from
// the frame-serial tail of encodePReusing, after every lane has joined.
func dumpDepthRemovalStats(frame uint64) {
	s := &depthRemovalStats
	sbs := s.sbs.Swap(0)
	if sbs == 0 {
		return
	}
	base := histSnapshotReset(&s.dev16HistBase)
	leaf := histSnapshotReset(&s.dev16HistLeaf)
	fmt.Fprintf(os.Stderr,
		"depthrm frame=%d sbs=%d disallow=%d cost=%d dev=%d dev16base=%v dev16leaf=%v wouldsplit16=%d removed(cost=%d dev=%d proxy=%d multi=%d)\n",
		frame, sbs, s.disallowSBs.Swap(0), s.costArmSBs.Swap(0), s.devArmSBs.Swap(0),
		base, leaf,
		s.wouldSplit16.Swap(0), s.removed16Cost.Swap(0), s.removed16Dev.Swap(0),
		s.removed16Prox.Swap(0), s.removed16Mult.Swap(0))
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
