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
	nearest, ok := stack.Stack.SingleNearestMV(false, st.forceIntegerMV)
	if !ok {
		return mds0Cand{}, false, false
	}
	// An intra-bordered block injects no MVP candidates
	// (use_neighbouring_mode_ctrls): only the ME NEWMV below.
	maxNear := -1
	if !st.intraBordered {
		cands[n] = mds0Cand{mode: tile.InterModeNearestMV, mv: nearest}
		n++
		// NEARMV DRL 0..2: stack slots 1+drl, capped by near_count and by
		// the slots the stack actually holds (svt_aom_get_max_drl_index
		// shape: NEARMV drl d needs ref_mv_count >= d+2).
		maxNear = int(stack.Stack.Count) - 1
		if nc := st.lpd1SB(lumaPX, lumaPY).nearCount; maxNear > nc {
			maxNear = nc
		}
	}
	for d := 0; d < maxNear; d++ {
		near, ok := stack.Stack.SingleNearMV(d, false, st.forceIntegerMV)
		if !ok {
			break
		}
		if mds0AlreadyInjected(cands[:n], near) {
			continue
		}
		cands[n] = mds0Cand{mode: tile.InterModeNearMV, drl: int8(d), mv: near}
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
	newPred := nearest
	if r, ok := stack.Stack.SingleNewMVReference(0, false, st.forceIntegerMV); ok {
		newPred = r
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
	// Candidate predictions alternate between two scratch buffers, so the
	// best one so far is never overwritten and predY keeps the ME vector's
	// pixels throughout, as SVT keeps one prediction buffer per candidate
	// (candidate_buffer_ptr_array): nothing is re-predicted, and the winner
	// is copied once.
	bufs := [2]*[4096]byte{&st.sadScratch, &st.mds0Scratch}
	cur, bestBuf := 0, -1
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
			if perr := st.predictLuma(bufs[cur][:bw*bh], refPlane, refStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, cand.mv, motion.RegularFilters); perr != nil {
				dists[i] = -1
				continue
			}
			sse, variance := realtimeInterResidualSSEVariance(src.Y, bufs[cur][:bw*bh], src.YStride, bw, lumaPX, lumaPY, bw, bh)
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
			if source == 0 {
				// The winner so far lives in bufs[cur]; the next candidates
				// predict into the other buffer.
				bestBuf = cur
				cur ^= 1
			} else {
				// The ME vector: predY still holds its pixels.
				bestBuf = -1
			}
			best, bestCost = i, cost
			bestRate = rate
			bestMV = cand.mv
		}
	}
	if best < 0 {
		return mds0Cand{}, false, false
	}
	if bestBuf >= 0 {
		copy(st.predY[:bw*bh], bufs[bestBuf][:bw*bh])
		predYValid, predYMV = true, bestMV
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
	case tile.BlockSize64x64:
		return 64, tile.TransformSize64x64, st.scan32
	}
	return 8, tile.TransformSize8x8, st.scan8
}

// fillIntraDCPred writes the n x n DC_PRED block from the reconstructed
// edges.
func fillIntraDCPred(pred []byte, reconPlane []byte, stride, px, py, n int, haveTop, haveLeft bool) {
	dc := dcPredictN(reconPlane, stride, px, py, n, haveTop, haveLeft)
	memsetBytes(pred[:n*n], dc)
}

// mds0IntraWins is the intra arm of SVT-AV1's light-PD1 MDS0
// (mode_decision.c generate_md_stage_0_cand_light_pd1 inject_intra_candidates,
// product_coding_loop.c fast_loop_core_light_pd1) after the inter
// candidates. Camera content injects DC_PRED only (intra_level
// MAX_INTRA_LEVEL - 1 at preset 12); screen content injects DC, V, H and
// SMOOTH (pframe_intra_sc.go). Each candidate is skipped when the best
// candidate so far already has a residual variance under skip_dc_th (DC) or
// dc_only_th (the others) per sample, else priced like the inter
// candidates - residual variance << 4 plus svt_aom_intra_fast_cost's rate -
// against the best so far; the inter winner pays its is_inter flag as
// av1_inter_fast_cost_light does. Square 16x16, 32x32 and 64x64 blocks (SVT
// injects intra below 128x128; 64x64 codes TX_64X64): 8x8 keeps its own
// fallback. A win leaves the candidate in st.mds0Intra (and a non-DC
// prediction in intraPredScratch[mds0Intra.pred]).
func (st *lossyEncodeState) mds0IntraWins(src SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, modeCtx *tile.BlockModeContext, lumaPX, lumaPY, bw, bh int) bool {
	if !st.mds0IntraOK || st.lpd1.skipIntra || bw != bh || (bw != 16 && bw != 32 && bw != 64) || st.color.MonoChrome {
		return false
	}
	ctx, err := modeCtx.IntraContext(int(block.X4), int(block.Y4), block.HaveTop, block.HaveLeft)
	if err != nil || ctx < 0 || ctx >= len(st.mds0IntraRates.IntraFlag) {
		return false
	}
	dec := st.lpd1SB(lumaPX, lumaPY)
	n := int64(bw * bh)
	modes := svtSCIntraModes[:1]
	dcOnlyTh, elim := svtDCOnlyTh(dec.level, st.lpd1.notLastLayer)
	if st.lpd1.screen && st.mds0IntraModeRates.ok {
		modes = svtSCIntraModes[:]
		// md_me_dist under dc_only_th: DC only.
		if elim && st.mdMEDist != ^uint64(0) && st.mdMEDist < uint64(dcOnlyTh)*uint64(n) {
			modes = modes[:1]
		}
	}
	cflAllowed := 0
	if bw <= 32 {
		cflAllowed = 1
	}
	lambda := st.lpd1FullLambda(dec)
	bestCost := st.mds0BestCost + ((int64(st.mds0IntraRates.IntraFlag[ctx][1])*lambda + 256) >> 9)
	bestVar := st.mds0BestVar
	won := false
	srcBlock := src.Y[lumaPY*src.YStride+lumaPX:]
	for i, mode := range modes {
		if mode == tile.IntraModeDC {
			if bestVar < int64(dec.skipDCTh)*n {
				continue
			}
		} else if elim && bestVar < int64(dcOnlyTh)*n {
			continue
		}
		var sse uint32
		var sum int32
		var lumaRate, chromaRate int64
		buf := 0
		if st.mds0Intra.valid && st.mds0Intra.pred == 0 {
			buf = 1
		}
		if mode == tile.IntraModeDC {
			// The DC predictor is flat and the variance is shift-invariant,
			// so the residual variance is the source block's own, measured
			// against zero through the same pixel-statistics kernels.
			sse, sum = svtPD0PixelStats(srcBlock, src.YStride, svtZeroRow[:], 0, bw)
			lumaRate = int64(st.mds0IntraRates.IntraFlag[ctx][0]) + int64(st.mds0IntraRates.LumaDCBits(block.Size))
			chromaRate = int64(st.mds0IntraRates.UVModeDC[cflAllowed])
		} else {
			pred := st.intraPredScratch[buf][:bw*bh]
			if err := st.svtPredictIntraBlock(pred, recon.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, block.HaveTop, block.HaveLeft, mode); err != nil {
				continue
			}
			sse, sum = svtPD0PixelStats(srcBlock, src.YStride, pred, bw, bw)
			l, c := st.mds0IntraModeRates.rate(i, block.Size, cflAllowed == 1)
			lumaRate = int64(st.mds0IntraRates.IntraFlag[ctx][0]) + l
			chromaRate = c
		}
		variance := int64(sse) - (int64(sum)*int64(sum))/n
		dist := variance << 4
		if dist<<7 > bestCost {
			continue
		}
		cost := (((lumaRate+chromaRate)*lambda + 256) >> 9) + (dist << 7)
		if cost >= bestCost {
			continue
		}
		bestCost, bestVar, won = cost, variance, true
		// The winner's statistics for the full loop: luma_fast_dist (the
		// variance), full_dist (its vf sse; for DC derived from the source
		// moments once the DC value is known) and fast_luma_rate.
		st.mds0Intra = mds0IntraStats{
			valid:    true,
			mode:     mode,
			pred:     buf,
			variance: uint32(variance),
			srcSSE:   sse,
			srcSum:   sum,
			lumaRate: lumaRate,
		}
	}
	return won
}

// mds0IntraStats is the intra MDS0 winner's fast-loop output
// (fast_loop_core_light_pd1): its mode, its residual variance
// (luma_fast_dist), the vf sum of squares and sum (against zero for DC,
// against the prediction otherwise) and its fast_luma_rate. pred names the
// intraPredScratch buffer holding a non-DC prediction.
type mds0IntraStats struct {
	valid    bool
	mode     tile.IntraMode
	pred     int
	variance uint32
	srcSSE   uint32
	srcSum   int32
	lumaRate int64
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
