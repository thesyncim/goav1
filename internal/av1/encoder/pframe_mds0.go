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
//     candidate reduction level pins near_count to 3
//     (Source/Lib/Codec/enc_mode_config.c set_cand_reduction_ctrls) and
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
// GLOBALMV(0,0) stays in the candidate set as in libaom's realtime pick
// (av1/encoder/nonrd_pickmode.c ref_mode_set keeps GLOBALMV), preserving this
// encoder's zero-motion syntax choice; identical vectors share one prediction
// so the mode symbols' rate alone separates them.
//
// The layer scoping mirrors SVT-AV1 preset 12's temporal-layer modulation
// (Source/Lib/Codec/enc_mode_config.c: pic_lpd1_lvl = is_base ? 4 : 7):
// reference frames buy the fuller decision; droppable leaves run a
// detector-capped variant (see the gate in encodePBlock).

// mds0NearCount is SVT-AV1's near_count at preset 12 (enc_mode_config.c
// set_cand_reduction_ctrls, cand_reduction_level 1 for P frames).
const mds0NearCount = 3

// mds0MaxCands bounds NEAREST + 3 NEAR + GLOBAL + NEW.
const mds0MaxCands = 6

type mds0Cand struct {
	mode tile.InterMode
	drl  int8
	mv   motion.Vector
}

// mds0PickInterMode runs the MDS0 fast loop over the single-reference
// candidates for the block and returns the rate-distortion winner. ok=false
// keeps the caller on its legacy choice (stack resolution failed, or no
// candidate could be predicted).
func (st *lossyEncodeState) mds0PickInterMode(src SourceFrame420, refPlane []byte, refStride int,
	stack *tile.ReferenceMVStackResult,
	lumaPX, lumaPY, bw, bh int, meMV motion.Vector) (mds0Cand, bool) {

	// Skip-certain bypass, ported from libaom's realtime skip-txfm test
	// (av1/encoder/nonrd_pickmode.c set_early_term_based_on_uv_plane): when
	// the ME candidate's residual variance sits under the quantizer dead
	// zone (ac_thr = ac_quant^2 >> 6, applied per 16x16 of samples), every
	// candidate here quantizes to all-zero AC and a mode swap is pure
	// syntax churn, so the block keeps the legacy classification. The ME
	// prediction's distortion is cached for the candidate loop below.
	meDist := int64(-1)
	if perr := predictInto(st.sadScratch[:bw*bh], refPlane, refStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, meMV, false, false, &st.scaledScratch); perr == nil {
		meSSE, meVar := realtimeInterResidualSSEVariance(src.Y, st.sadScratch[:bw*bh], src.YStride, bw, lumaPX, lumaPY, bw, bh)
		acThr := (int64(st.yQuant.AC) * int64(st.yQuant.AC) >> 6) * int64(bw*bh) / 256
		// Upstream narrows the dead-zone test on high-motion HD blocks
		// (same function: 720p+, non-screen tuning, source_sad above low,
		// per-pixel SSE over 1000 shift dc_thr/ac_thr right by 4).
		if src.Width*src.Height >= 1280*720 && !st.allowScreenContentTools &&
			int64(meSSE) > int64(bw*bh)*1000 &&
			st.realtimeContentStateForBlock(lumaPX, lumaPY).sourceSADNonRD > realtimeSourceSADLow {
			acThr >>= 4
		}
		if int64(meVar) < acThr {
			return mds0Cand{}, false
		}
		meDist = int64(meVar) << 4
	}

	var cands [mds0MaxCands]mds0Cand
	n := 0

	// NEARESTMV: the first inter MV injected, never deduplicated
	// (inject_mvp_candidates_ii_light_pd1).
	nearestRefs, err := stack.Stack.ResolveInterMVReferences(tile.InterModeResult{Mode: tile.InterModeNearestMV}, 0, false, st.forceIntegerMV)
	if err != nil {
		return mds0Cand{}, false
	}
	cands[n] = mds0Cand{mode: tile.InterModeNearestMV, mv: nearestRefs.Nearest[0]}
	n++

	// NEARMV DRL 0..2: stack slots 1+drl, capped by near_count and by the
	// slots the stack actually holds (svt_aom_get_max_drl_index shape:
	// NEARMV drl d needs ref_mv_count >= d+2).
	maxNear := int(stack.Stack.Count) - 1
	if maxNear > mds0NearCount {
		maxNear = mds0NearCount
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

	// GLOBALMV zero motion (libaom nonrd_pickmode keeps GLOBALMV in the
	// realtime mode set); duplicates reuse the earlier prediction below.
	cands[n] = mds0Cand{mode: tile.InterModeGlobalMV}
	n++

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

	var dists [mds0MaxCands]int64
	best := -1
	bestCost := int64(0)
	for i := 0; i < n; i++ {
		cand := &cands[i]
		// Distortion: one prediction per unique vector; the residual
		// variance <<4 matches fast_loop_core_light_pd1's squared-metric
		// scale.
		dist := int64(-1)
		if cand.mv == meMV {
			dist = meDist
		}
		for j := 0; dist < 0 && j < i; j++ {
			if cands[j].mv == cand.mv && dists[j] >= 0 {
				dist = dists[j]
				break
			}
		}
		if dist < 0 {
			if perr := predictInto(st.sadScratch[:bw*bh], refPlane, refStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, cand.mv, false, false, &st.scaledScratch); perr != nil {
				dists[i] = -1
				continue
			}
			_, variance := realtimeInterResidualSSEVariance(src.Y, st.sadScratch[:bw*bh], src.YStride, bw, lumaPX, lumaPY, bw, bh)
			dist = int64(variance) << 4
		}
		dists[i] = dist

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
		cost := ((rate*st.rdMult + 256) >> 9) + (dist << 7)
		if best < 0 || cost < bestCost {
			best, bestCost = i, cost
		}
	}
	if best < 0 {
		return mds0Cand{}, false
	}
	return cands[best], true
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
