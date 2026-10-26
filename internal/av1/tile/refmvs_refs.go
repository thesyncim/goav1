package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// This file ports decode_b's reference-frame syntax (the comp flag,
// comp_dir, the uni/bidir compound references and the single-reference tree)
// with dav1d's src/env.h context helpers over the above/left edge contexts.

// envRefCounts gathers the per-reference counts that dav1d's
// av1_get_*_ref_ctx helpers take from the top and left neighbours: every
// non-intra neighbour adds its first reference, and compound neighbours also
// their second.
func envRefCounts(c *BlockModeContext, x4, y4 int, haveTop, haveLeft bool) (cnt [referenceFrameCount]uint8) {
	if haveTop && c.AboveIntra[x4] == 0 {
		if r := c.AboveRef[0][x4]; r >= 0 {
			cnt[r]++
		}
		if c.AboveCompound[x4] != 0 {
			if r := c.AboveRef[1][x4]; r >= 0 {
				cnt[r]++
			}
		}
	}
	if haveLeft && c.LeftIntra[y4] == 0 {
		if r := c.LeftRef[0][y4]; r >= 0 {
			cnt[r]++
		}
		if c.LeftCompound[y4] != 0 {
			if r := c.LeftRef[1][y4]; r >= 0 {
				cnt[r]++
			}
		}
	}
	return cnt
}

// envCompareCounts is the shared tail of the av1_get_*_ref_ctx helpers.
func envCompareCounts(a, b uint8) int {
	if a == b {
		return 1
	}
	if a < b {
		return 0
	}
	return 2
}

// av1_get_ref_ctx (also av1_get_uni_p_ctx): forward vs backward references.
func envRefCtx(n *[referenceFrameCount]uint8) int {
	return envCompareCounts(n[0]+n[1]+n[2]+n[3], n[4]+n[5]+n[6])
}

// av1_get_fwd_ref_ctx (av1_get_ref_3_ctx).
func envFwdRefCtx(n *[referenceFrameCount]uint8) int {
	return envCompareCounts(n[0]+n[1], n[2]+n[3])
}

// av1_get_fwd_ref_1_ctx (av1_get_ref_4_ctx).
func envFwdRef1Ctx(n *[referenceFrameCount]uint8) int { return envCompareCounts(n[0], n[1]) }

// av1_get_fwd_ref_2_ctx (av1_get_ref_5_ctx, av1_get_uni_p2_ctx).
func envFwdRef2Ctx(n *[referenceFrameCount]uint8) int { return envCompareCounts(n[2], n[3]) }

// av1_get_bwd_ref_ctx (av1_get_ref_2_ctx).
func envBwdRefCtx(n *[referenceFrameCount]uint8) int {
	return envCompareCounts(n[4]+n[5], n[6])
}

// av1_get_bwd_ref_1_ctx (av1_get_ref_6_ctx).
func envBwdRef1Ctx(n *[referenceFrameCount]uint8) int { return envCompareCounts(n[4], n[5]) }

// av1_get_uni_p1_ctx.
func envUniP1Ctx(n *[referenceFrameCount]uint8) int {
	return envCompareCounts(n[1], n[2]+n[3])
}

// envUnidir is env.h's has_uni_comp.
func envUnidir(r0, r1 ReferenceFrame) bool {
	return (r0 < ReferenceFrameBWD) == (r1 < ReferenceFrameBWD)
}

// envCompCtx ports get_comp_ctx. Intra neighbours carry ref[0] == -1, which
// dav1d's unsigned ">= 4U" test counts with the backward references.
func envCompCtx(c *BlockModeContext, x4, y4 int, haveTop, haveLeft bool) int {
	if haveTop {
		aRef := c.AboveRef[0][x4]
		if haveLeft {
			lRef := c.LeftRef[0][y4]
			if c.AboveCompound[x4] != 0 {
				if c.LeftCompound[y4] != 0 {
					return 4
				}
				return 2 + boolInt(lRef.Backward() || c.LeftIntra[y4] != 0)
			}
			if c.LeftCompound[y4] != 0 {
				return 2 + boolInt(aRef.Backward() || c.AboveIntra[x4] != 0)
			}
			return boolInt(lRef.Backward() != aRef.Backward())
		}
		if c.AboveCompound[x4] != 0 {
			return 3
		}
		return boolInt(aRef.Backward())
	}
	if haveLeft {
		if c.LeftCompound[y4] != 0 {
			return 3
		}
		return boolInt(c.LeftRef[0][y4].Backward())
	}
	return 1
}

// envCompDirCtx ports get_comp_dir_ctx.
func envCompDirCtx(c *BlockModeContext, x4, y4 int, haveTop, haveLeft bool) int {
	if haveTop && haveLeft {
		aIntra := c.AboveIntra[x4] != 0
		lIntra := c.LeftIntra[y4] != 0
		if aIntra && lIntra {
			return 2
		}
		if aIntra || lIntra {
			if aIntra {
				if c.LeftCompound[y4] == 0 {
					return 2
				}
				return 1 + 2*boolInt(envUnidir(c.LeftRef[0][y4], c.LeftRef[1][y4]))
			}
			if c.AboveCompound[x4] == 0 {
				return 2
			}
			return 1 + 2*boolInt(envUnidir(c.AboveRef[0][x4], c.AboveRef[1][x4]))
		}
		aComp := c.AboveCompound[x4] != 0
		lComp := c.LeftCompound[y4] != 0
		aRef0 := c.AboveRef[0][x4]
		lRef0 := c.LeftRef[0][y4]
		if !aComp && !lComp {
			return 1 + 2*boolInt(aRef0.Backward() == lRef0.Backward())
		}
		if !aComp || !lComp {
			if aComp {
				if !envUnidir(aRef0, c.AboveRef[1][x4]) {
					return 1
				}
			} else if !envUnidir(lRef0, c.LeftRef[1][y4]) {
				return 1
			}
			return 3 + boolInt(aRef0.Backward() == lRef0.Backward())
		}
		aUni := envUnidir(aRef0, c.AboveRef[1][x4])
		lUni := envUnidir(lRef0, c.LeftRef[1][y4])
		if !aUni && !lUni {
			return 0
		}
		if !aUni || !lUni {
			return 2
		}
		return 3 + boolInt((aRef0 == ReferenceFrameBWD) == (lRef0 == ReferenceFrameBWD))
	}
	if haveLeft {
		if c.LeftIntra[y4] != 0 || c.LeftCompound[y4] == 0 {
			return 2
		}
		return 4 * boolInt(envUnidir(c.LeftRef[0][y4], c.LeftRef[1][y4]))
	}
	if haveTop {
		if c.AboveIntra[x4] != 0 || c.AboveCompound[x4] == 0 {
			return 2
		}
		return 4 * boolInt(envUnidir(c.AboveRef[0][x4], c.AboveRef[1][x4]))
	}
	return 2
}

// readInterRefsRefmvs ports decode_b's reference-frame syntax: skip-mode
// references, the segment reference feature, the comp flag, then either the
// compound (uni or bidir) or the single-reference tree, on one cursor.
func (s *DecodeState) readInterRefsRefmvs(cdfs *InterRefCDFs, c *BlockModeContext, req *BlockLoopRequest, block *BlockVisit, skipMode bool, segment *parser.SegmentData) (InterReferencesResult, error) {
	if skipMode {
		refs := req.SkipModeRefs
		if !refs[0].Valid() || !refs[1].Valid() {
			return InterReferencesResult{}, ErrInvalidDecodeState
		}
		return InterReferencesResult{Ref: refs, Compound: true}, nil
	}
	if req.Segmentation.Enabled {
		if segment.RefFrame > 0 {
			ref := ReferenceFrame(segment.RefFrame - 1)
			if !ref.Valid() {
				return InterReferencesResult{}, ErrInvalidDecodeState
			}
			return InterReferencesResult{Ref: [2]ReferenceFrame{ref, ReferenceFrameNone}}, nil
		}
		if segment.RefFrame == 0 {
			return InterReferencesResult{}, ErrInvalidDecodeState
		}
		if segment.Skip || segment.GlobalMV {
			return InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}, nil
		}
	}
	x4, y4 := int(block.X4), int(block.Y4)
	haveTop, haveLeft := block.HaveTop, block.HaveLeft
	dims := blockDimensions[block.Size]
	cur := s.Reader.Cursor()
	compound := false
	if min(dims.W4, dims.H4) > 1 {
		switch req.ReferenceMode {
		case parser.ReferenceModeSingle:
		case parser.ReferenceModeCompound:
			compound = true
		case parser.ReferenceModeSelect:
			compound = cur.ReadBinaryCDFUnchecked(&cdfs.CompInter[envCompCtx(c, x4, y4, haveTop, haveLeft)]) != 0
		default:
			return InterReferencesResult{}, ErrInvalidDecodeState
		}
	}
	n := envRefCounts(c, x4, y4, haveTop, haveLeft)
	var refs InterReferencesResult
	if compound {
		refs.Compound = true
		if readEnvBool(&cur, &cdfs.CompRefType[envCompDirCtx(c, x4, y4, haveTop, haveLeft)]) {
			// bidir - first reference (fw)
			if readEnvBool(&cur, &cdfs.CompFwdRef[0][envFwdRefCtx(&n)]) {
				refs.Ref[0] = ReferenceFrameLast3 + ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.CompFwdRef[2][envFwdRef2Ctx(&n)]))
			} else {
				refs.Ref[0] = ReferenceFrameLast + ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.CompFwdRef[1][envFwdRef1Ctx(&n)]))
			}
			// second reference (bw)
			if readEnvBool(&cur, &cdfs.CompBwdRef[0][envBwdRefCtx(&n)]) {
				refs.Ref[1] = ReferenceFrameAltref
			} else {
				refs.Ref[1] = ReferenceFrameBWD + ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.CompBwdRef[1][envBwdRef1Ctx(&n)]))
			}
		} else {
			// unidir
			refs.Unidir = true
			if readEnvBool(&cur, &cdfs.UniCompRef[0][envRefCtx(&n)]) {
				refs.Ref = [2]ReferenceFrame{ReferenceFrameBWD, ReferenceFrameAltref}
			} else {
				refs.Ref[0] = ReferenceFrameLast
				refs.Ref[1] = ReferenceFrameLast2 + ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.UniCompRef[1][envUniP1Ctx(&n)]))
				if refs.Ref[1] == ReferenceFrameLast3 {
					refs.Ref[1] += ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.UniCompRef[2][envFwdRef2Ctx(&n)]))
				}
			}
		}
	} else {
		refs.Ref[1] = ReferenceFrameNone
		if readEnvBool(&cur, &cdfs.SingleRef[0][envRefCtx(&n)]) {
			if readEnvBool(&cur, &cdfs.SingleRef[1][envBwdRefCtx(&n)]) {
				refs.Ref[0] = ReferenceFrameAltref
			} else {
				refs.Ref[0] = ReferenceFrameBWD + ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.SingleRef[5][envBwdRef1Ctx(&n)]))
			}
		} else if readEnvBool(&cur, &cdfs.SingleRef[2][envFwdRefCtx(&n)]) {
			refs.Ref[0] = ReferenceFrameLast3 + ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.SingleRef[4][envFwdRef2Ctx(&n)]))
		} else {
			refs.Ref[0] = ReferenceFrameLast + ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.SingleRef[3][envFwdRef1Ctx(&n)]))
		}
	}
	cur.CommitStateTo(&s.Reader)
	return refs, nil
}

func readEnvBool(c *entropy.Cursor, cdf *entropy.CDF) bool {
	return c.ReadBinaryCDFUnchecked(cdf) != 0
}

// envFilterCtx ports get_filter_ctx: a neighbour contributes its filter for
// direction dir only when it is an inter block using ref.
func envFilterCtx(c *BlockModeContext, x4, y4 int, haveTop, haveLeft bool, comp bool, dir int, ref ReferenceFrame) int {
	aFilter, lFilter := switchableFilterMissing, switchableFilterMissing
	if haveTop && c.AboveInterpValid[x4] != 0 && (c.AboveRef[0][x4] == ref || c.AboveRef[1][x4] == ref) {
		aFilter = envFilterDir(c.AboveInterp[x4], dir)
	}
	if haveLeft && c.LeftInterpValid[y4] != 0 && (c.LeftRef[0][y4] == ref || c.LeftRef[1][y4] == ref) {
		lFilter = envFilterDir(c.LeftInterp[y4], dir)
	}
	ctx := dir*interFilterDirOffset + boolInt(comp)*interFilterCompOffset
	switch {
	case aFilter == lFilter:
		return ctx + aFilter
	case aFilter == switchableFilterMissing:
		return ctx + lFilter
	case lFilter == switchableFilterMissing:
		return ctx + aFilter
	default:
		return ctx + switchableFilterMissing
	}
}

func envFilterDir(f motion.InterpFilters, dir int) int {
	filter := f.Y
	if dir == 1 {
		filter = f.X
	}
	if !switchableMotionFilter(filter) {
		return switchableFilterMissing
	}
	return int(filter)
}

// readInterpFiltersRefmvs ports decode_b's subpel filter section: no symbol
// for skip mode, warped motion or non-translational global motion
// (has_subpel_filter), the frame filter when it is not switchable, otherwise
// one or (dual_filter) two switchable symbols with the get_filter_ctx
// contexts. It returns the filters and the number of symbols read.
func (s *DecodeState) readInterpFiltersRefmvs(cdfs *InterpFilterCDFs, c *BlockModeContext, req *BlockLoopRequest, block *BlockVisit, refs InterReferencesResult, mode InterModeResult, motionMode MotionMode, globalTypes [2]parser.GlobalMotionType, skipMode bool) (motion.InterpFilters, int, error) {
	if skipMode || motionMode == MotionModeWarp || nonTransGlobalMotion(InterpFilterRequest{Size: block.Size, References: refs, Mode: mode, GlobalTypes: globalTypes}) {
		filter, err := unswitchableMotionFilter(req.InterpolationFilter)
		return motion.InterpFilters{X: filter, Y: filter}, 0, err
	}
	if req.InterpolationFilter != parser.InterpolationSwitchable {
		filter, err := fixedMotionFilter(req.InterpolationFilter)
		return motion.InterpFilters{X: filter, Y: filter}, 0, err
	}
	x4, y4 := int(block.X4), int(block.Y4)
	cur := s.Reader.Cursor()
	y := motion.InterpFilter(cur.ReadCDF3Unchecked(&cdfs.Switchable[envFilterCtx(c, x4, y4, block.HaveTop, block.HaveLeft, refs.Compound, 0, refs.Ref[0])]))
	x, reads := y, 1
	if req.EnableDualFilter {
		x = motion.InterpFilter(cur.ReadCDF3Unchecked(&cdfs.Switchable[envFilterCtx(c, x4, y4, block.HaveTop, block.HaveLeft, refs.Compound, 1, refs.Ref[0])]))
		reads++
	}
	cur.CommitStateTo(&s.Reader)
	return motion.InterpFilters{X: x, Y: y}, reads, nil
}

// envSingleRefContext is the context of single_ref bit (decode_b's ref[0..5]
// CDFs): av1_get_ref_ctx and av1_get_ref_2..6_ctx.
func envSingleRefContext(n *[referenceFrameCount]uint8, bit int) (int, error) {
	switch bit {
	case 0:
		return envRefCtx(n), nil
	case 1:
		return envBwdRefCtx(n), nil
	case 2:
		return envFwdRefCtx(n), nil
	case 3:
		return envFwdRef1Ctx(n), nil
	case 4:
		return envFwdRef2Ctx(n), nil
	case 5:
		return envBwdRef1Ctx(n), nil
	}
	return 0, ErrInvalidDecodeState
}

// envCompFwdRefContext is the context of comp_fwd_ref bit.
func envCompFwdRefContext(n *[referenceFrameCount]uint8, bit int) (int, error) {
	switch bit {
	case 0:
		return envFwdRefCtx(n), nil
	case 1:
		return envFwdRef1Ctx(n), nil
	case 2:
		return envFwdRef2Ctx(n), nil
	}
	return 0, ErrInvalidDecodeState
}

// envCompBwdRefContext is the context of comp_bwd_ref bit.
func envCompBwdRefContext(n *[referenceFrameCount]uint8, bit int) (int, error) {
	switch bit {
	case 0:
		return envBwdRefCtx(n), nil
	case 1:
		return envBwdRef1Ctx(n), nil
	}
	return 0, ErrInvalidDecodeState
}

// envUniCompRefContext is the context of comp_uni_ref bit
// (av1_get_uni_p_ctx, av1_get_uni_p1_ctx, av1_get_uni_p2_ctx).
func envUniCompRefContext(n *[referenceFrameCount]uint8, bit int) (int, error) {
	switch bit {
	case 0:
		return envRefCtx(n), nil
	case 1:
		return envUniP1Ctx(n), nil
	case 2:
		return envFwdRef2Ctx(n), nil
	}
	return 0, ErrInvalidDecodeState
}
