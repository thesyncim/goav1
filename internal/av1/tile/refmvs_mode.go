package tile

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// This file ports the inter mode / DRL / motion-vector syntax of dav1d's
// decode_b (src/decode.c) and read_mv_residual / read_mv_component_diff. The
// candidate stack comes straight from refmvsTile.find (dav1d_refmvs_find), the
// DRL index is dav1d's absolute mvstack slot, and MVs are assigned from that
// slot exactly like decode_b's assign_comp_mv and single-reference branches.

// refmvsDRLContexts is dav1d's get_drl_context for stack indices 0, 1 and 2,
// packed two bits each for entropy.DecodeInterMode: with a = weight[i] < 640
// and b = weight[i+1] < 640 the context is b << a.
func refmvsDRLContexts(stack *[8]refmvsCandidate) uint {
	a0, a1 := uint(boolInt32(stack[0].weight < 640)), uint(boolInt32(stack[1].weight < 640))
	a2, a3 := uint(boolInt32(stack[2].weight < 640)), uint(boolInt32(stack[3].weight < 640))
	return a1<<a0 | (a2<<a1)<<2 | (a3<<a2)<<4
}

// readMVResidualRefmvs ports dav1d's read_mv_residual: returns ref + residual
// and whether the result stays inside the legal MV range.
func readMVResidualRefmvs(c *entropy.Cursor, cdfs *MVCDFs, ref refmvsMV, mvPrec int) (refmvsMV, bool) {
	row, col := c.DecodeMVResidual(&cdfs.Joint, (*entropy.MVComponentCDFs)(&cdfs.Components[mvComponentRow]),
		(*entropy.MVComponentCDFs)(&cdfs.Components[mvComponentCol]), ref.row(), ref.col(), mvPrec)
	ok := row > MVLower && row < MVUpper && col > MVLower && col < MVUpper
	return makeRefmvsMV(row, col), ok
}

// decodeMotionModeRefmvs ports decode_b's "motion variation" section: the
// OBMC/warp gate (switchable motion mode, no inter-intra, min side >= 8 px,
// not a warped GLOBALMV, has overlappable neighbours), find_matching_ref,
// the motion_mode / obmc symbol and derive_warpmv. It fills the motion-mode,
// warp and OBMC-neighbor fields of result.
func (s *DecodeState) decodeMotionModeRefmvs(cdfs *MotionModeCDFs, c *BlockModeContext, rt *refmvsTile, req *BlockLoopRequest, block *BlockVisit, refs InterReferencesResult, skipMode bool, result *BlockPredictionModeResult) (MotionMode, error) {
	result.MotionMode = MotionModeTranslation
	result.MotionModeValid = true
	dims := blockDimensions[block.Size]
	ref := refs.Ref0
	if refs.Compound || skipMode || !req.SwitchableMotionMode || result.InterIntra.Enabled ||
		min(dims.W4, dims.H4) < 2 ||
		(!req.ForceIntegerMV && result.InterMode.Mode == InterModeGlobalMV && req.GlobalMotionTypes[ref] > parser.GlobalMotionTranslation) ||
		!c.hasOverlappableNeighbours(block) {
		return MotionModeTranslation, nil
	}
	var masks [2]uint64
	topHasRight := refmvsTopHasRight(req.SBSizeMIB, block)
	rt.findMatchingRef(block, topHasRight, ref, &masks)
	allowWarp := !req.ScaledReferences[ref] && !req.ForceIntegerMV && req.AllowWarpedMotion && masks[0]|masks[1] != 0
	// The symbols are read straight from the tile's range decoder, as
	cur := &s.Reader // dav1d reads through &ts->msac.
	var mode MotionMode
	if allowWarp {
		mode = MotionMode(cur.ReadCDF3Unchecked(&cdfs.MotionMode[block.Size]))
	} else {
		mode = MotionMode(cur.ReadBinaryCDFUnchecked(&cdfs.OBMC[block.Size]))
	}
	result.MotionMode = mode
	if mode == MotionModeWarp {
		model, invalid, err := rt.deriveWarpMV(block, &masks, result.InterMotion.MV[0])
		if err != nil {
			return 0, fmt.Errorf("project warped motion: %w", err)
		}
		result.WarpedMotionInvalid = invalid
		if !invalid {
			s.blockMotionExtras(result).WarpedMotion = model
			result.WarpedMotionValid = true
		}
	}
	if mode != MotionModeTranslation {
		s.blockMotionExtras(result).OverlappableNeighbors = rt.obmcNeighbors(c, block)
		result.OverlappableNeighborsValid = true
	}
	return mode, nil
}

// decodeInterModeRefmvs ports decode_b's inter-mode syntax after the
// reference frames are known: dav1d_refmvs_find, the (comp_)inter_mode
// symbols, the DRL bits and the MV assignment/residuals. It fills the mode,
// DRL and motion fields of result.
func (s *DecodeState) decodeInterModeRefmvs(cdfs *InterModeCDFs, mvCDFs *MVCDFs, modeCtx *BlockModeContext, rt *refmvsTile, req *BlockLoopRequest, block *BlockVisit, refs InterReferencesResult, skipMode bool, segment *parser.SegmentData, globalMVs [2]motion.Vector, result *BlockPredictionModeResult) error {
	var found refmvsResult
	target := makeRefmvsRefPair(int8(refs.Ref0)+1, -1)
	if refs.Compound {
		target = makeRefmvsRefPair(int8(refs.Ref0)+1, int8(refs.Ref1)+1)
	}
	rt.find(&found, target, block.Size, refmvsTopHasRight(req.SBSizeMIB, block), int32(block.MIRow), int32(block.MICol), globalMVs)
	if refmvsOracleEnabled {
		refmvsOracleCompare(modeCtx, rt, &found, req, block, refs, globalMVs)
	}
	stack := &found.stack
	nMVs := found.count
	ctx := found.ctx
	rf := &rt.frame
	mvPrec := 0
	if rf.allowHighPrecisionMV {
		mvPrec = 1
	}
	if rf.forceIntegerMV {
		mvPrec--
	}
	// The symbols are read straight from the tile's range decoder, as
	c := (*entropy.Cursor)(&s.Reader) // dav1d reads through &ts->msac.
	motionResult := &result.InterMotion
	motionResult.References = refs // the rest was cleared with the result
	ok := true

	if refs.Compound {
		var mode CompoundInterMode
		var drlIdx int
		if skipMode {
			mode = CompoundInterModeNearestNearest
		} else {
			var m uint
			m, drlIdx, result.DRLIndexValid = c.DecodeInterMode((*entropy.InterModeCDFs)(cdfs), true, false, uint(ctx), int(nMVs), refmvsDRLContexts(stack))
			mode = CompoundInterMode(m)
		}
		result.DRLIndex = uint8(drlIdx)
		motionResult.Mode = InterModeResult{Compound: true, CompoundMode: mode}
		comps := compoundInterModeComponents[mode]
		cand := &stack[drlIdx&7]
		for idx, im := range [2]InterMode{comps.First, comps.Second} {
			switch im {
			case InterModeNearMV, InterModeNearestMV:
				motionResult.MV[idx] = rf.fixMVPrecision(cand.mv[idx]).vector()
			case InterModeGlobalMV:
				motionResult.MV[idx] = globalMVs[idx]
			case InterModeNewMV:
				mv, valid := readMVResidualRefmvs(c, mvCDFs, cand.mv[idx], mvPrec)
				ok = ok && valid
				motionResult.MV[idx] = mv.vector()
			}
		}
	} else {
		segForced := req.Segmentation.Enabled && (segment.Skip || segment.GlobalMV)
		m, drlIdx, drl := c.DecodeInterMode((*entropy.InterModeCDFs)(cdfs), false, segForced, uint(ctx), int(nMVs), refmvsDRLContexts(stack))
		mode := InterMode(m)
		result.DRLIndex, result.DRLIndexValid = uint8(drlIdx), drl
		switch mode {
		case InterModeGlobalMV:
			motionResult.MV[0] = globalMVs[0]
		case InterModeNewMV:
			ref := stack[drlIdx&7].mv[0]
			if nMVs <= 1 {
				ref = rf.fixMVPrecision(ref)
			}
			mv, valid := readMVResidualRefmvs(c, mvCDFs, ref, mvPrec)
			ok = valid
			motionResult.MV[0] = mv.vector()
		default:
			mv := stack[drlIdx&7].mv[0]
			if drlIdx < 2 {
				mv = rf.fixMVPrecision(mv)
			}
			motionResult.MV[0] = mv.vector()
		}
		motionResult.Mode = InterModeResult{Mode: mode}
	}
	if !ok {
		return ErrInvalidDecodeState
	}
	result.InterMode = motionResult.Mode
	result.InterModeValid = true
	result.InterMotionValid = true
	return nil
}
