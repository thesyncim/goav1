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

// refmvsDRLContext ports dav1d's get_drl_context.
func refmvsDRLContext(stack *[8]refmvsCandidate, idx int32) int {
	if stack[idx].weight >= 640 {
		return int(boolInt32(stack[idx+1].weight < 640))
	}
	if stack[idx+1].weight < 640 {
		return 2
	}
	return 0
}

// readMVComponentDiffRefmvs ports dav1d's read_mv_component_diff. mvPrec is
// hp - force_integer_mv (-1, 0 or 1).
func readMVComponentDiffRefmvs(c *entropy.Cursor, comp *MVComponentCDFs, mvPrec int) int32 {
	sign := c.ReadBinaryCDFUnchecked(&comp.Sign)
	cl := c.ReadCDFSymbolsUnchecked(&comp.Classes, MVClasses)
	var up int
	fp, hp := 3, 1
	if cl == 0 {
		up = c.ReadBinaryCDFUnchecked(&comp.Class0)
		if mvPrec >= 0 {
			fp = c.ReadCDF4Unchecked(&comp.Class0FP[up])
			if mvPrec > 0 {
				hp = c.ReadBinaryCDFUnchecked(&comp.Class0HP)
			}
		}
	} else {
		up = 1 << cl
		for n := range cl {
			up |= c.ReadBinaryCDFUnchecked(&comp.Bits[n]) << n
		}
		if mvPrec >= 0 {
			fp = c.ReadCDF4Unchecked(&comp.FP)
			if mvPrec > 0 {
				hp = c.ReadBinaryCDFUnchecked(&comp.HP)
			}
		}
	}
	diff := int32((up<<3)|(fp<<1)|hp) + 1
	if sign != 0 {
		return -diff
	}
	return diff
}

// readMVResidualRefmvs ports dav1d's read_mv_residual: returns ref + residual
// and whether the result stays inside the legal MV range.
func readMVResidualRefmvs(c *entropy.Cursor, cdfs *MVCDFs, ref refmvsMV, mvPrec int) (refmvsMV, bool) {
	joint := c.ReadCDF4Unchecked(&cdfs.Joint)
	row, col := ref.row(), ref.col()
	if joint&2 != 0 { // MV_JOINT_V
		row += readMVComponentDiffRefmvs(c, &cdfs.Components[mvComponentRow], mvPrec)
	}
	if joint&1 != 0 { // MV_JOINT_H
		col += readMVComponentDiffRefmvs(c, &cdfs.Components[mvComponentCol], mvPrec)
	}
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
	ref := refs.Ref[0]
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

	cur := s.Reader.Cursor()
	var mode MotionMode
	if allowWarp {
		mode = MotionMode(cur.ReadCDF3Unchecked(&cdfs.MotionMode[block.Size]))
	} else {
		mode = MotionMode(cur.ReadBinaryCDFUnchecked(&cdfs.OBMC[block.Size]))
	}
	cur.CommitStateTo(&s.Reader)
	result.MotionMode = mode
	if mode == MotionModeWarp {
		model, invalid, err := rt.deriveWarpMV(block, &masks, result.InterMotion.MV[0])
		if err != nil {
			return 0, fmt.Errorf("project warped motion: %w", err)
		}
		result.WarpedMotionInvalid = invalid
		if !invalid {
			result.WarpedMotion = model
			result.WarpedMotionValid = true
		}
	}
	if mode != MotionModeTranslation {
		result.OverlappableNeighbors = rt.obmcNeighbors(c, block)
		result.OverlappableNeighborsValid = true
	}
	return mode, nil
}

var compoundModeHasNear = [compoundInterModeCount]bool{
	CompoundInterModeNearNear: true,
	CompoundInterModeNearNew:  true,
	CompoundInterModeNewNear:  true,
}

// decodeInterModeRefmvs ports decode_b's inter-mode syntax after the
// reference frames are known: dav1d_refmvs_find, the (comp_)inter_mode
// symbols, the DRL bits and the MV assignment/residuals. It fills the mode,
// DRL and motion fields of result.
func (s *DecodeState) decodeInterModeRefmvs(cdfs *InterModeCDFs, mvCDFs *MVCDFs, modeCtx *BlockModeContext, rt *refmvsTile, req *BlockLoopRequest, block *BlockVisit, refs InterReferencesResult, skipMode bool, segment *parser.SegmentData, globalMVs [2]motion.Vector, result *BlockPredictionModeResult) error {
	var found refmvsResult
	target := [2]int8{int8(refs.Ref[0]) + 1, -1}
	if refs.Compound {
		target[1] = int8(refs.Ref[1]) + 1
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

	c := s.Reader.Cursor()
	motionResult := &result.InterMotion
	*motionResult = InterMotionResult{References: refs}
	drlIdx := int32(0)
	drlRead := false
	ok := true

	if refs.Compound {
		var mode CompoundInterMode
		if skipMode {
			mode = CompoundInterModeNearestNearest
		} else {
			mode = CompoundInterMode(c.ReadCDFSymbolsUnchecked(&cdfs.Compound[ctx], int(compoundInterModeCount)))
			if mode == CompoundInterModeNewNew {
				drlRead = true
				if nMVs > 1 {
					drlIdx += int32(c.ReadBinaryCDFUnchecked(&cdfs.DRL[refmvsDRLContext(stack, 0)]))
					if drlIdx == 1 && nMVs > 2 {
						drlIdx += int32(c.ReadBinaryCDFUnchecked(&cdfs.DRL[refmvsDRLContext(stack, 1)]))
					}
				}
			} else if compoundModeHasNear[mode] {
				drlRead = true
				drlIdx = 1
				if nMVs > 2 {
					drlIdx += int32(c.ReadBinaryCDFUnchecked(&cdfs.DRL[refmvsDRLContext(stack, 1)]))
					if drlIdx == 2 && nMVs > 3 {
						drlIdx += int32(c.ReadBinaryCDFUnchecked(&cdfs.DRL[refmvsDRLContext(stack, 2)]))
					}
				}
			}
		}
		motionResult.Mode = InterModeResult{Compound: true, CompoundMode: mode}
		comps := compoundInterModeComponents[mode]
		for idx, im := range [2]InterMode{comps.First, comps.Second} {
			switch im {
			case InterModeNearMV, InterModeNearestMV:
				motionResult.MV[idx] = rf.fixMVPrecision(stack[drlIdx].mv[idx]).vector()
			case InterModeGlobalMV:
				motionResult.MV[idx] = globalMVs[idx]
			case InterModeNewMV:
				mv, valid := readMVResidualRefmvs(&c, mvCDFs, stack[drlIdx].mv[idx], mvPrec)
				ok = ok && valid
				motionResult.MV[idx] = mv.vector()
				result.MVResidualValid[idx] = true
			}
		}
	} else {
		var mode InterMode
		segForced := req.Segmentation.Enabled && (segment.Skip || segment.GlobalMV)
		if segForced || c.ReadBinaryCDFUnchecked(&cdfs.NewMV[ctx&7]) != 0 {
			if segForced || c.ReadBinaryCDFUnchecked(&cdfs.GlobalMV[(ctx>>3)&1]) == 0 {
				mode = InterModeGlobalMV
				motionResult.MV[0] = globalMVs[0]
			} else {
				if c.ReadBinaryCDFUnchecked(&cdfs.RefMV[(ctx>>4)&15]) != 0 {
					mode = InterModeNearMV
					drlRead = true
					drlIdx = 1
					if nMVs > 2 {
						drlIdx += int32(c.ReadBinaryCDFUnchecked(&cdfs.DRL[refmvsDRLContext(stack, 1)]))
						if drlIdx == 2 && nMVs > 3 {
							drlIdx += int32(c.ReadBinaryCDFUnchecked(&cdfs.DRL[refmvsDRLContext(stack, 2)]))
						}
					}
				} else {
					mode = InterModeNearestMV
				}
				mv := stack[drlIdx].mv[0]
				if drlIdx < 2 {
					mv = rf.fixMVPrecision(mv)
				}
				motionResult.MV[0] = mv.vector()
			}
		} else {
			mode = InterModeNewMV
			drlRead = true
			if nMVs > 1 {
				drlIdx += int32(c.ReadBinaryCDFUnchecked(&cdfs.DRL[refmvsDRLContext(stack, 0)]))
				if drlIdx == 1 && nMVs > 2 {
					drlIdx += int32(c.ReadBinaryCDFUnchecked(&cdfs.DRL[refmvsDRLContext(stack, 1)]))
				}
			}
			ref := stack[0].mv[0]
			if nMVs > 1 {
				ref = stack[drlIdx].mv[0]
			} else {
				ref = rf.fixMVPrecision(ref)
			}
			mv, valid := readMVResidualRefmvs(&c, mvCDFs, ref, mvPrec)
			ok = valid
			motionResult.MV[0] = mv.vector()
			result.MVResidualValid[0] = true
		}
		motionResult.Mode = InterModeResult{Mode: mode}
	}
	c.CommitStateTo(&s.Reader)
	if !ok {
		return ErrInvalidDecodeState
	}
	result.InterMode = motionResult.Mode
	result.InterModeValid = true
	result.ReferenceMVStackValid = true
	result.DRLIndex = uint8(drlIdx)
	result.DRLIndexValid = drlRead
	result.InterMVReferencesValid = !interModeUsesGlobalOnly(motionResult.Mode)
	result.InterMotionValid = true
	return nil
}
