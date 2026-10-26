package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// pframe_motion.go is the inter block's motion stage shared by the fused
// (encodePBlock) and split (decidePBlock) walks: the full-pel ME vector, its
// SVT light-PD1 sub-pel refinement per reference (subpel_svt.go), and the
// LAST / GOLDEN / compound reference choice.

// pblockMotion is the motion stage's result.
type pblockMotion struct {
	mv              motion.Vector
	fullSAD         int
	scaledReference bool
	refs            tile.InterReferencesResult
	refPlanes       SourceFrame420
	compoundMV      [2]motion.Vector
	// stack is the reference-MV stack for refs when stackOK; the sub-pel
	// search builds it first for its MV cost reference (SVT generates the
	// MVP table before read_refine_me_mvs_light_pd1).
	stack   tile.ReferenceMVStackResult
	stackOK bool
}

// pblockStackRequest is the reference-MV stack request for the block.
func (st *lossyEncodeState) pblockStackRequest(block tile.BlockVisit, walkReq tile.BlockWalkRequest, miCols, miRows uint16, refs tile.InterReferencesResult) tile.ReferenceMVStackRequest {
	return tile.ReferenceMVStackRequest{
		MICol:          block.MICol,
		MIRow:          block.MIRow,
		TileMIColStart: walkReq.MIColStart,
		TileMIRowStart: walkReq.MIRowStart,
		TileMIColEnd:   walkReq.MIColEnd,
		TileMIRowEnd:   walkReq.MIRowEnd,
		FrameMIRows:    miRows,
		FrameMICols:    miCols,
		Size:           block.Size,
		References:     refs,
		X4:             block.X4,
		Y4:             block.Y4,
		HaveTop:        block.HaveTop,
		HaveLeft:       block.HaveLeft,
		HaveTopRight:   tile.BlockHasTopRight(16, block),
		ForceIntegerMV: st.forceIntegerMV,
	}
}

// svtNewMVPredictor is the NEWMV predictor svt_aom_choose_best_av1_mv_pred
// returns at preset 12 (DRL 0, approx_inter_rate > 1): the MV cost reference
// of the sub-pel search and of the MDS0 NEWMV rate.
func (st *lossyEncodeState) svtNewMVPredictor(stack *tile.ReferenceMVStackResult) motion.Vector {
	if r, ok := stack.Stack.SingleNewMVReference(0, false, st.forceIntegerMV); ok {
		return r
	}
	return motion.Vector{}
}

// pblockSubpel refines the full-pel vector mv on refPlane with the SVT
// light-PD1 sub-pel search and returns the coded vector with its exact
// regular-filter SAD (the later stages' gates and the SAD skip shortcut read
// SAD, never the search's variance metric). For LAST the refined prediction
// stays in predY (st.mePredY) for MDS0 to reuse.
func (st *lossyEncodeState) pblockSubpel(src SourceFrame420, refPlane []byte, refStride, lumaPX, lumaPY, bw, bh int,
	mv motion.Vector, fullSAD int, refMV motion.Vector, intoPredY bool) (motion.Vector, int) {
	// The SB's me_subpel_level (svtLPD1SubpelLevel); level 0 disables the
	// search, skip_zz_mv skips it from a zero vector (md_subpel_search).
	dec := st.lpd1SB(lumaPX, lumaPY)
	ctrls := svtSubpelCtrlsFor(dec.subpelLevel)
	if ctrls == nil || (ctrls.skipZeroMV && mv == (motion.Vector{})) {
		return mv, fullSAD
	}
	refined, besterr := st.svtSubpelSearch(src.Y, refPlane, src.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, mv, refMV, ctrls, st.lpd1FullLambda(dec))
	// md_me_dist: the smallest post-sub-pel cost over the references searched.
	st.mdMEDist = min(st.mdMEDist, besterr)
	if refined == mv {
		return mv, fullSAD
	}
	dst := st.sadScratch[:bw*bh]
	if intoPredY {
		dst = st.predY[:bw*bh]
	}
	if err := st.predictLuma(dst, refPlane, refStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, refined, motion.RegularFilters); err != nil {
		return mv, fullSAD
	}
	sad := sadRectDualBlock(src.Y[lumaPY*src.YStride+lumaPX:], src.YStride, dst, bw, bw, bh)
	if intoPredY {
		st.mePredY = mePredYKey{valid: true, mv: refined}
	}
	return refined, sad
}

// mePredYKey records that predY holds LAST's regular-filter prediction at mv.
type mePredYKey struct {
	valid bool
	mv    motion.Vector
}

// pblockMotionSearch runs the block's motion stage.
func (st *lossyEncodeState) pblockMotionSearch(src, ref SourceFrame420, golden *SourceFrame420, block tile.BlockVisit,
	modeCtx *tile.BlockModeContext, referenceMode parser.ReferenceMode, walkReq tile.BlockWalkRequest, miCols, miRows uint16,
	lumaPX, lumaPY, bw, bh int) (pblockMotion, error) {

	st.mePredY.valid = false
	st.mdMEDist = ^uint64(0)
	if st.zmvForced(block.Size, lumaPX, lumaPY) {
		// force_zeromv_skip_for_blk: zero-motion LAST only.
		var o pblockMotion
		o.refs = tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone}}
		o.refPlanes = ref
		o.fullSAD = st.leafFullPelSAD(src, ref, lumaPX, lumaPY, bw, bh, motion.Vector{})
		return o, nil
	}
	var out pblockMotion
	scaledReference := ref.Width != src.Width || ref.Height != src.Height
	out.scaledReference = scaledReference
	var mv motion.Vector
	fullSAD := -1
	if scaledReference {
		if err := predictIntoScaled(st.sadScratch[:bw*bh], ref.Y, ref.YStride, ref.Width, ref.Height, src.Width, src.Height, lumaPX, lumaPY, bw, bh, mv, false, false, &st.scaledScratch); err != nil {
			return out, fmt.Errorf("scaled reference luma probe: %w", err)
		}
		fullSAD = sadRectDualBlock(src.Y[lumaPY*src.YStride+lumaPX:], src.YStride, st.sadScratch[:bw*bh], bw, bw, bh)
	} else {
		// The frame ME's vector for the PU covering this block (zero motion
		// when the frame has no ME), scored with its exact SAD.
		if me := st.hme.sbME(lumaPX, lumaPY); me != nil {
			pu := svtMEBlockOffset(lumaPX&63, lumaPY&63, max(bw, bh))
			mv = motion.Vector{Row: me.bestMV[pu][1] * 8, Col: me.bestMV[pu][0] * 8}
		}
		fullSAD = st.leafFullPelSAD(src, ref, lumaPX, lumaPY, bw, bh, mv)
		// read_refine_me_mvs_light_pd1: refine LAST's ME vector with the
		// NEWMV predictor of LAST's MVP table as the MV cost reference.
		if st.allowSubpelRefinement() {
			lastRefs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone}}
			err := st.referenceMVStack(&out.stack, modeCtx, lastRefs, &block, walkReq, miCols, miRows)
			if err != nil {
				return out, fmt.Errorf("build ref mv stack: %w", err)
			}
			out.stackOK = true
			mv, fullSAD = st.pblockSubpel(src, ref.Y, ref.YStride, lumaPX, lumaPY, bw, bh, mv, fullSAD, st.svtNewMVPredictor(&out.stack), true)
		}
	}

	// Reference selection: when LAST left a poor match, probe the GOLDEN
	// anchor; occluded-then-revealed content predicts from the older anchor
	// when the previous frame cannot. Compound covers 8x8 and 16x16 leaves,
	// which get a true golden motion search; larger blocks keep the cheap
	// zero-motion anchor probe.
	refs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone}}
	refPlanes := ref
	compound := false
	var compoundMV [2]motion.Vector
	goldenEligible := (bw == 8 && bh == 8) ||
		(bw == 16 && bh == 16) || (bw == 32 && bh == 32) ||
		(bw == 16 && bh == 8) || (bw == 8 && bh == 16) ||
		(bw == 32 && bh == 16) || (bw == 16 && bh == 32) ||
		(bw == 64 && bh == 16) || (bw == 16 && bh == 64)
	if !scaledReference && golden != nil && golden.Y != nil && goldenEligible && fullSAD > bw*bh*4 {
		lastMV, lastSAD := mv, fullSAD
		var gmv motion.Vector
		gsad := 1 << 30
		compoundCapable := bw == bh && bw <= 16
		goldenRefPruned := false
		var goldenStack tile.ReferenceMVStackResult
		goldenStackOK := false
		if compoundCapable {
			gdx, gdy, s := fullPelDiamondSearch(src.Y, golden.Y, src.YStride, src.Width, src.Height, lumaPX, lumaPY, bw)
			gmv, gsad = motion.Vector{Row: int16(gdy * 8), Col: int16(gdx * 8)}, s
			// dist_based_ref_pruning: drop GOLDEN's subpel + compound MD work once
			// its full-pel distortion is >30% worse than LAST (perform_md_reference_pruning).
			goldenRefPruned = int64(gsad)*100 > int64(lastSAD)*(100+goldenRefPruneMaxDevPct)
			if st.allowSubpelRefinement() && !goldenRefPruned {
				goldenRefs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameGolden, tile.ReferenceFrameNone}}
				err := st.referenceMVStack(&goldenStack, modeCtx, goldenRefs, &block, walkReq, miCols, miRows)
				if err != nil {
					return out, fmt.Errorf("build ref mv stack: %w", err)
				}
				goldenStackOK = true
				gmv, gsad = st.pblockSubpel(src, golden.Y, golden.YStride, lumaPX, lumaPY, bw, bh, gmv, gsad, st.svtNewMVPredictor(&goldenStack), false)
			}
		} else {
			base := lumaPY*src.YStride + lumaPX
			gsad = sadRectBlock(src.Y, golden.Y, base, base, src.YStride, bw, bh, lastSAD)
		}
		if compoundCapable && !goldenRefPruned && referenceMode == parser.ReferenceModeSelect && gsad+32 >= lastSAD && gsad <= lastSAD+bw*bh*4 {
			if err := predictCompoundInto(st.sadScratch[:bw*bh], ref.Y, ref.YStride, golden.Y, golden.YStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, lastMV, gmv, false, false, &st.compBuf0, &st.compBuf1, &st.compScratch); err == nil {
				srcBlock := src.Y[lumaPY*src.YStride+lumaPX:]
				var compoundSAD int
				if bw == 8 {
					compoundSAD = sad8x8Dual(srcBlock, src.YStride, st.sadScratch[:64], 8)
				} else {
					compoundSAD = sad16x16Dual(srcBlock, src.YStride, st.sadScratch[:256], 16)
				}
				compoundBias := 64 + 12*st.sadPerBit
				if compoundSAD+compoundBias < fullSAD {
					compound = true
					compoundMV = [2]motion.Vector{lastMV, gmv}
					fullSAD = compoundSAD
				}
			}
		}
		// The anchor must win clearly: LAST keeps the cheaper ref symbol and
		// denser MV predictors.
		if !compound && gsad+32 < fullSAD {
			refs.Ref[0] = tile.ReferenceFrameGolden
			refPlanes = *golden
			mv, fullSAD = gmv, gsad
			out.stack, out.stackOK = goldenStack, goldenStackOK
			st.mePredY.valid = false
		}
		if compound {
			refs = tile.InterReferencesResult{
				Ref:      [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameGolden},
				Compound: true,
				Unidir:   true,
			}
			out.stackOK = false
			st.mePredY.valid = false
		}
	}
	out.mv, out.fullSAD = mv, fullSAD
	out.refs, out.refPlanes = refs, refPlanes
	out.compoundMV = compoundMV
	return out, nil
}

// pblockStack returns the reference-MV stack for the motion stage's refs,
// reusing the one the sub-pel search built; otherwise it builds it in mo.
func (st *lossyEncodeState) pblockStack(mo *pblockMotion, block *tile.BlockVisit, modeCtx *tile.BlockModeContext, walkReq tile.BlockWalkRequest, miCols, miRows uint16) (*tile.ReferenceMVStackResult, error) {
	if mo.stackOK {
		return &mo.stack, nil
	}
	if err := st.referenceMVStack(&mo.stack, modeCtx, mo.refs, block, walkReq, miCols, miRows); err != nil {
		return &mo.stack, fmt.Errorf("build ref mv stack: %w", err)
	}
	return &mo.stack, nil
}

// leafFullPelSAD is the luma SAD of the bw x bh block at full-pel motion mv,
// through the edge-emulating predictor when the reference block leaves the
// plane.
func (st *lossyEncodeState) leafFullPelSAD(src, ref SourceFrame420, px, py, bw, bh int, mv motion.Vector) int {
	dx, dy := int(mv.Col>>3), int(mv.Row>>3)
	if px+dx >= 0 && py+dy >= 0 && px+dx+bw <= ref.Width && py+dy+bh <= ref.Height {
		return sadRectDualBlock(src.Y[py*src.YStride+px:], src.YStride, ref.Y[(py+dy)*ref.YStride+px+dx:], ref.YStride, bw, bh)
	}
	if err := st.predictLuma(st.sadScratch[:bw*bh], ref.Y, ref.YStride, ref.Width, ref.Height, px, py, bw, bh, mv, motion.RegularFilters); err != nil {
		return 1 << 30
	}
	return sadRectDualBlock(src.Y[py*src.YStride+px:], src.YStride, st.sadScratch[:bw*bh], bw, bw, bh)
}

// svtMEBlockOffset is svt_aom_get_me_block_offset
// (Source/Lib/Codec/mode_decision.c) in realtimeSBMultiSizeME's PU layout: a
// block reads the square PU of side max(width, height) that holds it.
// (x, y) is the block origin within the superblock.
func svtMEBlockOffset(x, y, side int) int {
	switch {
	case side >= 64:
		return 0
	case side >= 32:
		return 1 + (y/32)*2 + x/32
	}
	q := (y/32)*2 + x/32
	w := ((y&31)/16)*2 + (x&31)/16
	if side >= 16 {
		return 5 + q*4 + w
	}
	c := ((y&15)/8)*2 + (x&15)/8
	return 21 + (q*4+w)*4 + c
}
