package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// pframe_mds0.go is the light-PD1 MDS0 inter candidate decision, ported from
// SVT-AV1:
//
//   - Candidate set: inject_mvp_candidates_ii_light_pd1 and
//     inject_new_candidates_light_pd1 (Source/Lib/Codec/mode_decision.c):
//     NEARESTMV, then up to near_count NEARMV DRL entries, then the ME NEWMV,
//     deduplicated by vector (mv_is_already_injected). At preset 12 the
//     SB's candidate reduction level caps near_count (lpd1SBDecision) and
//     choose_best_av1_mv_pred keeps NEWMV on DRL index 0
//     (approx_inter_rate > 1 path).
//   - Pricing: fast_loop_core_light_pd1
//     (Source/Lib/Codec/product_coding_loop.c): each candidate is fully
//     predicted, distortion is the residual variance shifted by 4 to the
//     squared-metric scale, and the winner minimizes
//     RDCOST(full_lambda, rate, dist).
//   - Rate: av1_inter_fast_cost_light (Source/Lib/Codec/rd_cost.c) at
//     preset 12's approx_inter_rate=2: mode cascade and DRL bits from the
//     frame-frozen rate tables (md_rate_estimation.c
//     svt_aom_estimate_syntax_rate), reference bits skipped, NEWMV vector
//     bits approximated as 1296 + factor*(|dx|+|dy|) with factor 20 for
//     screen content else 50.
//
// There is no GLOBALMV candidate: the light-PD1 injectors add only NEAREST,
// the NEAR DRL entries and the ME NEWMV (inject_zz_backup_candidate only
// fires on an empty list, which NEAREST rules out); identical vectors share
// one prediction so the mode symbols' rate alone separates them.
//
// The layer scoping mirrors SVT-AV1 preset 12's temporal-layer modulation
// (Source/Lib/Codec/enc_mode_config.c: pic_lpd1_lvl = is_base ? 4 : 7):
// reference frames buy the fuller decision; droppable leaves run a
// detector-capped variant (see the gate in encodePBlock).

// The NEAR DRL loop is capped by the SB's near_count_ctrls.near_count
// (lpd1SBDecision.nearCount, set_cand_reduction_ctrls via the per-SB
// light-PD1 level: 3 at LPD1_LVL_0, 1 above), as
// inject_mvp_candidates_ii_light_pd1 does (mode_decision.c:1400-1404).

// mds0MaxCands bounds NEAREST + 3 NEAR + NEW.
const mds0MaxCands = 5

type mds0Cand struct {
	mode tile.InterMode
	drl  int8
	mv   motion.Vector
}

// mds0PickInterMode runs the MDS0 fast loop over the single-reference
// candidates for the block and returns the rate-distortion winner. The third
// result says predY contains that winner's regular-filter predictor. ok=false
// keeps the caller on its legacy choice (stack resolution failed, or no
// candidate could be predicted).
func (st *lossyEncodeState) mds0PickInterMode(src SourceFrame420, refPlane []byte, refStride int,
	stack *tile.ReferenceMVStackResult,
	lumaPX, lumaPY, bw, bh int, meMV motion.Vector) (mds0Cand, bool, bool) {

	// SVT's light-PD1 MDS0 prices every block's candidates; there is no
	// dead-zone bypass ahead of it. When every candidate's residual
	// quantizes to zero the rate term alone decides, which is exactly where
	// a vector-free NEARESTMV/NEARMV beats the ME NEWMV. The ME prediction's
	// distortion is cached for the candidate loop below.
	meDist := int64(-1)
	meSSE := uint32(0)
	predYValid := false
	predYMV := motion.Vector{}
	// The sub-pel search already left the refined vector's prediction in
	// predY (pframe_motion.go).
	var perr error
	if !st.mePredY.valid || st.mePredY.mv != meMV {
		perr = st.predictLuma(st.predY[:bw*bh], refPlane, refStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, meMV, motion.RegularFilters)
	}
	if perr == nil {
		predYValid, predYMV = true, meMV
		sse, meVar := realtimeInterResidualSSEVariance(src.Y, st.predY[:bw*bh], src.YStride, bw, lumaPX, lumaPY, bw, bh)
		meSSE = sse
		st.predStats.set(meMV, motion.RegularFilters, meSSE, meVar)
		meDist = int64(meVar) << 4
	}

	var cands [mds0MaxCands]mds0Cand
	n := 0

	// NEARESTMV: the first inter MV injected, never deduplicated
	// (inject_mvp_candidates_ii_light_pd1).
	nearestRefs, err := stack.Stack.ResolveInterMVReferences(tile.InterModeResult{Mode: tile.InterModeNearestMV}, 0, false, st.forceIntegerMV)
	if err != nil {
		return mds0Cand{}, false, false
	}
	cands[n] = mds0Cand{mode: tile.InterModeNearestMV, mv: nearestRefs.Nearest[0]}
	n++

	// NEARMV DRL 0..2: stack slots 1+drl, capped by near_count and by the
	// slots the stack actually holds (svt_aom_get_max_drl_index shape:
	// NEARMV drl d needs ref_mv_count >= d+2).
	maxNear := int(stack.Stack.Count) - 1
	if nc := st.lpd1SB(lumaPX, lumaPY).nearCount; maxNear > nc {
		maxNear = nc
	}
	for d := 0; d < maxNear; d++ {
		r, rerr := stack.Stack.ResolveInterMVReferences(tile.InterModeResult{Mode: tile.InterModeNearMV}, d, false, st.forceIntegerMV)
		if rerr != nil {
			break
		}
		if mds0AlreadyInjected(cands[:n], r.Near[0]) {
			continue
		}
		cands[n] = mds0Cand{mode: tile.InterModeNearMV, drl: int8(d), mv: r.Near[0]}
		n++
	}

	// NEWMV from ME, deduplicated against the injected MVP candidates
	// (inject_new_candidates_light_pd1); DRL index 0 per preset 12's
	// choose_best_av1_mv_pred approx_inter_rate>1 path.
	if !mds0AlreadyInjected(cands[:n], meMV) {
		cands[n] = mds0Cand{mode: tile.InterModeNewMV, mv: meMV}
		n++
	}

	// NEWMV vector-bit predictor: stack slot 0 (choose_best_av1_mv_pred at
	// approx_inter_rate>1 returns DRL 0).
	newPred := nearestRefs.Nearest[0]
	if r, rerr := stack.Stack.ResolveInterMVReferences(tile.InterModeResult{Mode: tile.InterModeNewMV}, 0, false, st.forceIntegerMV); rerr == nil {
		newPred = r.Residual[0]
	}
	mvFactor := int64(50)
	if st.allowScreenContentTools {
		mvFactor = 20
	}

	// fast_loop_core_light_pd1 prices at full_lambda (lpd1FullLambda).
	lambda := st.lpd1FullLambda(st.lpd1SB(lumaPX, lumaPY))
	var dists [mds0MaxCands]int64
	var sses [mds0MaxCands]uint32
	st.mds0BestRate = -1
	best := -1
	bestCost := int64(0)
	bestRate := int64(0)
	bestMV := motion.Vector{}
	for i := 0; i < n; i++ {
		cand := &cands[i]
		// Distortion: one prediction per candidate (the injectors
		// deduplicate vectors, mv_is_already_injected), the ME vector's
		// cached; the residual variance <<4 matches
		// fast_loop_core_light_pd1's squared-metric scale.
		dist := int64(-1)
		source := uint8(0)
		if cand.mv == meMV {
			dist = meDist
			if dist >= 0 {
				source = 1
				sses[i] = meSSE
			}
		}
		if dist < 0 {
			if perr := st.predictLuma(st.sadScratch[:bw*bh], refPlane, refStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, cand.mv, motion.RegularFilters); perr != nil {
				dists[i] = -1
				continue
			}
			sse, variance := realtimeInterResidualSSEVariance(src.Y, st.sadScratch[:bw*bh], src.YStride, bw, lumaPX, lumaPY, bw, bh)
			dist = int64(variance) << 4
			sses[i] = sse
		}
		dists[i] = dist
		// fast_loop_core_light_pd1: a candidate whose distortion alone,
		// RDCOST(full_lambda, 0, dist), already exceeds the best cost can
		// never be the single candidate sent to MDS3; skip its rate.
		if best >= 0 && dist<<7 > bestCost {
			continue
		}

		rate := int64(st.mds0Rates.SingleInterModeBits(stack.ModeContext, cand.mode))
		if cand.mode == tile.InterModeNewMV || cand.mode == tile.InterModeNearMV {
			drlReq, derr := stack.Stack.DRLRequestForMode(tile.InterModeResult{Mode: cand.mode})
			if derr != nil {
				continue
			}
			rate += int64(st.mds0Rates.DRLIndexBits(drlReq, int(cand.drl)))
		}
		if cand.mode == tile.InterModeNewMV {
			dr := int64(cand.mv.Row) - int64(newPred.Row)
			if dr < 0 {
				dr = -dr
			}
			dc := int64(cand.mv.Col) - int64(newPred.Col)
			if dc < 0 {
				dc = -dc
			}
			rate += 1296 + mvFactor*(dr+dc)
		}
		// RDCOST (SVT Source/Lib/Codec/rd_cost.h): (rate*lambda >> 9) +
		// (dist << 7), the same shape the skip decision already uses.
		cost := ((rate*lambda + 256) >> 9) + (dist << 7)
		if best < 0 || cost < bestCost {
			if !predYValid || predYMV != cand.mv {
				switch source {
				case 0:
					// This candidate was just predicted into sadScratch.
					copy(st.predY[:bw*bh], st.sadScratch[:bw*bh])
					predYValid = true
				case 1:
					// The cached ME pixels were overwritten after a different
					// candidate became best; restore them only on this path.
					predYValid = st.predictLuma(st.predY[:bw*bh], refPlane, refStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, meMV, motion.RegularFilters) == nil
				}
				if predYValid {
					predYMV = cand.mv
				}
			}
			best, bestCost = i, cost
			bestRate = rate
			bestMV = cand.mv
		}
	}
	if best < 0 {
		return mds0Cand{}, false, false
	}
	// The winner's regular-filter prediction statistics (dist is its
	// variance << 4) for the later stages (pred_stats.go).
	st.predStats.set(bestMV, motion.RegularFilters, sses[best], uint32(dists[best]>>4))
	st.mds0BestCost, st.mds0BestVar = bestCost, dists[best]>>4
	st.mds0BestRate, st.mds0BestSSE = bestRate, sses[best]
	return cands[best], true, predYValid && predYMV == bestMV
}

// mds0WinnerSAD re-measures the luma SAD once MDS0 has swapped the motion
// vector away from the ME result: the block's later stages (the SAD skip
// shortcut in decideInterTXBlock among them) must judge the prediction
// actually coded, not the stale ME match. retained says predY holds the
// winner's prediction; otherwise the shortcut is disarmed so the
// coefficient proof decides skip.
func (st *lossyEncodeState) mds0WinnerSAD(src SourceFrame420, lumaPX, lumaPY, bw, bh int, retained bool, fullSAD int) int {
	if !retained {
		return max(fullSAD, bw*bh)
	}
	return sadRectDualBlock(src.Y[lumaPY*src.YStride+lumaPX:], src.YStride, st.predY[:bw*bh], bw, bw, bh)
}

// intraPBlockGeometry is the luma side of an intra block inside an inter
// frame: the square size, its whole-block transform and scan. 8x8 keeps the
// legacy fallback's mode search; 16x16 and 32x32 come from the MDS0 DC arm.
func intraPBlockGeometry(st *lossyEncodeState, size tile.BlockSize) (int, tile.TransformSize, []int16) {
	switch size {
	case tile.BlockSize16x16:
		return 16, tile.TransformSize16x16, st.scan16
	case tile.BlockSize32x32:
		return 32, tile.TransformSize32x32, st.scan32
	}
	return 8, tile.TransformSize8x8, st.scan8
}

// fillIntraDCPred writes the n x n DC_PRED block from the reconstructed
// edges.
func fillIntraDCPred(pred []byte, reconPlane []byte, stride, px, py, n int, haveTop, haveLeft bool) {
	dc := dcPredictN(reconPlane, stride, px, py, n, haveTop, haveLeft)
	for i := range pred[:n*n] {
		pred[i] = dc
	}
}

// mds0IntraDCWins is the DC_PRED arm of SVT-AV1's light-PD1 MDS0
// (mode_decision.c generate_md_stage_0_cand_light_pd1 inject_intra_candidates
// with the DC-only intra level preset 12 sets): after the inter candidates,
// DC is skipped when the best inter residual variance already sits under
// the SB's cand_elimination_ctrls.skip_dc_th (lpd1SBDecision.skipDCTh) per
// sample (product_coding_loop.c fast_loop_core_light_pd1),
// else priced like the inter candidates - residual variance << 4 plus
// svt_aom_intra_fast_cost's rate - against the inter winner, which pays its
// is_inter flag as av1_inter_fast_cost_light does. Square 16x16 and 32x32
// blocks only: 8x8 keeps its own fallback and 64x64 would need a TX64.
func (st *lossyEncodeState) mds0IntraDCWins(src SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, modeCtx *tile.BlockModeContext, lumaPX, lumaPY, bw, bh int) bool {
	if !st.mds0IntraOK || bw != bh || (bw != 16 && bw != 32) || st.color.MonoChrome {
		return false
	}
	if st.mds0BestVar < int64(st.lpd1SB(lumaPX, lumaPY).skipDCTh*bw*bh) {
		return false
	}
	ctx, err := modeCtx.IntraContext(int(block.X4), int(block.Y4), block.HaveTop, block.HaveLeft)
	if err != nil || ctx < 0 || ctx >= len(st.mds0IntraRates.IntraFlag) {
		return false
	}
	// fast_loop_core_light_pd1 scores the DC candidate like any other: the
	// svt_aom_mefn_ptr[bsize].vf variance of source against prediction. The
	// DC predictor is flat and the variance is shift-invariant, so the
	// residual variance is the source block's own, measured against zero
	// through the same pixel-statistics kernels.
	sse, sum := svtPD0PixelStats(src.Y[lumaPY*src.YStride+lumaPX:], src.YStride, svtZeroRow[:], 0, bw)
	n := int64(bw * bh)
	variance := int64(sse) - (int64(sum)*int64(sum))/n
	dist := variance << 4
	cflAllowed := 0
	if bw <= 32 {
		cflAllowed = 1
	}
	rate := int64(st.mds0IntraRates.IntraFlag[ctx][0]) +
		int64(st.mds0IntraRates.LumaDCBits(block.Size)) +
		int64(st.mds0IntraRates.UVModeDC[cflAllowed])
	lambda := st.lpd1FullLambda(st.lpd1SB(lumaPX, lumaPY))
	intraCost := ((rate*lambda + 256) >> 9) + (dist << 7)
	interCost := st.mds0BestCost + ((int64(st.mds0IntraRates.IntraFlag[ctx][1])*lambda + 256) >> 9)
	return intraCost < interCost
}

// mds0PredictorMatchesCodedFilters reports whether predY, retained by MDS0
// with regular filters, is still the predictor the block will code. A full-pel
// vector is independent of interpolation filters; a sub-pel vector requires
// the regular filters to remain selected.
func mds0PredictorMatchesCodedFilters(retained bool, mv motion.Vector, codedFilters motion.InterpFilters) bool {
	if !retained {
		return false
	}
	if mv.Row%8 == 0 && mv.Col%8 == 0 {
		return true
	}
	return codedFilters == motion.RegularFilters
}

// mds0AlreadyInjected is SVT-AV1's mv_is_already_injected over the built
// candidate list (Source/Lib/Codec/mode_decision.c).
func mds0AlreadyInjected(cands []mds0Cand, mv motion.Vector) bool {
	for i := range cands {
		if cands[i].mv == mv {
			return true
		}
	}
	return false
}

// svtZeroRow is a zero reference row; read at stride 0 it is an all-zero
// block of any size up to 64 wide.
var svtZeroRow [64]byte
