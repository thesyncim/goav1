package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// This file ports decode_b's reference-frame syntax (the comp flag,
// comp_dir, the uni/bidir compound references and the single-reference tree)
// with dav1d's src/env.h context helpers over the above/left edge contexts.

// envRefCountSet holds the per-reference counts that dav1d's
// av1_get_*_ref_ctx helpers take from the top and left neighbours, one byte
// per reference: every non-intra neighbour adds its first reference, and
// compound neighbours also their second (each count is at most 4). Packing
// them in one register keeps the counts out of memory across the reference
// tree reads.
type envRefCountSet uint64

func (n envRefCountSet) at(r int) uint8 { return uint8(n >> (uint(r) * 8 & 63)) }

func envRefCountOf(r ReferenceFrame) envRefCountSet {
	if r < 0 {
		return 0
	}
	return 1 << (uint(r) * 8 & 63)
}

func envRefCounts(c *BlockModeContext, x4, y4 int, haveTop, haveLeft bool) envRefCountSet {
	var cnt envRefCountSet
	if haveTop && c.A.Intra[x4] == 0 {
		cnt += envRefCountOf(c.A.Ref[0][x4])
		if c.A.CompType[x4] != 0 {
			cnt += envRefCountOf(c.A.Ref[1][x4])
		}
	}
	if haveLeft && c.L.Intra[y4] == 0 {
		cnt += envRefCountOf(c.L.Ref[0][y4])
		if c.L.CompType[y4] != 0 {
			cnt += envRefCountOf(c.L.Ref[1][y4])
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
func envRefCtx(n envRefCountSet) int {
	// The byte sums cannot carry: all counts together are at most 4.
	return envCompareCounts(uint8(uint32(n)*0x01010101>>24), uint8(uint32(n>>32)*0x010101>>16))
}

// av1_get_fwd_ref_ctx (av1_get_ref_3_ctx).
func envFwdRefCtx(n envRefCountSet) int {
	return envCompareCounts(n.at(0)+n.at(1), n.at(2)+n.at(3))
}

// av1_get_fwd_ref_1_ctx (av1_get_ref_4_ctx).
func envFwdRef1Ctx(n envRefCountSet) int { return envCompareCounts(n.at(0), n.at(1)) }

// av1_get_fwd_ref_2_ctx (av1_get_ref_5_ctx, av1_get_uni_p2_ctx).
func envFwdRef2Ctx(n envRefCountSet) int { return envCompareCounts(n.at(2), n.at(3)) }

// av1_get_bwd_ref_ctx (av1_get_ref_2_ctx).
func envBwdRefCtx(n envRefCountSet) int {
	return envCompareCounts(n.at(4)+n.at(5), n.at(6))
}

// av1_get_bwd_ref_1_ctx (av1_get_ref_6_ctx).
func envBwdRef1Ctx(n envRefCountSet) int { return envCompareCounts(n.at(4), n.at(5)) }

// av1_get_uni_p1_ctx.
func envUniP1Ctx(n envRefCountSet) int {
	return envCompareCounts(n.at(1), n.at(2)+n.at(3))
}

// envUnidir is env.h's has_uni_comp.
func envUnidir(r0, r1 ReferenceFrame) bool {
	return (r0 < ReferenceFrameBWD) == (r1 < ReferenceFrameBWD)
}

// envCompCtx ports get_comp_ctx. Intra neighbours carry ref[0] == -1, which
// dav1d's unsigned ">= 4U" test counts with the backward references.
func envCompCtx(c *BlockModeContext, x4, y4 int, haveTop, haveLeft bool) int {
	if haveTop {
		aRef := c.A.Ref[0][x4]
		if haveLeft {
			lRef := c.L.Ref[0][y4]
			if c.A.CompType[x4] != 0 {
				if c.L.CompType[y4] != 0 {
					return 4
				}
				return 2 + boolInt(lRef.Backward() || c.L.Intra[y4] != 0)
			}
			if c.L.CompType[y4] != 0 {
				return 2 + boolInt(aRef.Backward() || c.A.Intra[x4] != 0)
			}
			return boolInt(lRef.Backward() != aRef.Backward())
		}
		if c.A.CompType[x4] != 0 {
			return 3
		}
		return boolInt(aRef.Backward())
	}
	if haveLeft {
		if c.L.CompType[y4] != 0 {
			return 3
		}
		return boolInt(c.L.Ref[0][y4].Backward())
	}
	return 1
}

// envCompDirCtx ports get_comp_dir_ctx.
func envCompDirCtx(c *BlockModeContext, x4, y4 int, haveTop, haveLeft bool) int {
	if haveTop && haveLeft {
		aIntra := c.A.Intra[x4] != 0
		lIntra := c.L.Intra[y4] != 0
		if aIntra && lIntra {
			return 2
		}
		if aIntra || lIntra {
			if aIntra {
				if c.L.CompType[y4] == 0 {
					return 2
				}
				return 1 + 2*boolInt(envUnidir(c.L.Ref[0][y4], c.L.Ref[1][y4]))
			}
			if c.A.CompType[x4] == 0 {
				return 2
			}
			return 1 + 2*boolInt(envUnidir(c.A.Ref[0][x4], c.A.Ref[1][x4]))
		}
		aComp := c.A.CompType[x4] != 0
		lComp := c.L.CompType[y4] != 0
		aRef0 := c.A.Ref[0][x4]
		lRef0 := c.L.Ref[0][y4]
		if !aComp && !lComp {
			return 1 + 2*boolInt(aRef0.Backward() == lRef0.Backward())
		}
		if !aComp || !lComp {
			if aComp {
				if !envUnidir(aRef0, c.A.Ref[1][x4]) {
					return 1
				}
			} else if !envUnidir(lRef0, c.L.Ref[1][y4]) {
				return 1
			}
			return 3 + boolInt(aRef0.Backward() == lRef0.Backward())
		}
		aUni := envUnidir(aRef0, c.A.Ref[1][x4])
		lUni := envUnidir(lRef0, c.L.Ref[1][y4])
		if !aUni && !lUni {
			return 0
		}
		if !aUni || !lUni {
			return 2
		}
		return 3 + boolInt((aRef0 == ReferenceFrameBWD) == (lRef0 == ReferenceFrameBWD))
	}
	if haveLeft {
		if c.L.Intra[y4] != 0 || c.L.CompType[y4] == 0 {
			return 2
		}
		return 4 * boolInt(envUnidir(c.L.Ref[0][y4], c.L.Ref[1][y4]))
	}
	if haveTop {
		if c.A.Intra[x4] != 0 || c.A.CompType[x4] == 0 {
			return 2
		}
		return 4 * boolInt(envUnidir(c.A.Ref[0][x4], c.A.Ref[1][x4]))
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
		return InterReferencesResult{Ref0: refs[0], Ref1: refs[1], Compound: true}, nil
	}
	if req.Segmentation.Enabled {
		if segment.RefFrame > 0 {
			ref := ReferenceFrame(segment.RefFrame - 1)
			if !ref.Valid() {
				return InterReferencesResult{}, ErrInvalidDecodeState
			}
			return InterReferencesResult{Ref0: ref, Ref1: ReferenceFrameNone}, nil
		}
		if segment.RefFrame == 0 {
			return InterReferencesResult{}, ErrInvalidDecodeState
		}
		if segment.Skip || segment.GlobalMV {
			return InterReferencesResult{Ref0: ReferenceFrameLast, Ref1: ReferenceFrameNone}, nil
		}
	}
	x4, y4, haveTop, haveLeft := int(block.X4), int(block.Y4), block.HaveTop, block.HaveLeft
	dims := blockDimensions[block.Size]
	// The symbols are read straight from the tile's range decoder, as
	cur := &s.Reader // dav1d reads through &ts->msac.
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
		if readEnvBool(cur, &cdfs.CompRefType[envCompDirCtx(c, x4, y4, haveTop, haveLeft)]) {
			// bidir - first reference (fw)
			if readEnvBool(cur, &cdfs.CompFwdRef[0][envFwdRefCtx(n)]) {
				refs.Ref0 = ReferenceFrameLast3 + ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.CompFwdRef[2][envFwdRef2Ctx(n)]))
			} else {
				refs.Ref0 = ReferenceFrameLast + ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.CompFwdRef[1][envFwdRef1Ctx(n)]))
			}
			// second reference (bw)
			if readEnvBool(cur, &cdfs.CompBwdRef[0][envBwdRefCtx(n)]) {
				refs.Ref1 = ReferenceFrameAltref
			} else {
				refs.Ref1 = ReferenceFrameBWD + ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.CompBwdRef[1][envBwdRef1Ctx(n)]))
			}
		} else {
			// unidir
			refs.Unidir = true
			if readEnvBool(cur, &cdfs.UniCompRef[0][envRefCtx(n)]) {
				refs.Ref0, refs.Ref1 = ReferenceFrameBWD, ReferenceFrameAltref
			} else {
				refs.Ref0 = ReferenceFrameLast
				refs.Ref1 = ReferenceFrameLast2 + ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.UniCompRef[1][envUniP1Ctx(n)]))
				if refs.Ref1 == ReferenceFrameLast3 {
					refs.Ref1 += ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.UniCompRef[2][envFwdRef2Ctx(n)]))
				}
			}
		}
	} else {
		refs.Ref1 = ReferenceFrameNone
		if readEnvBool(cur, &cdfs.SingleRef[0][envRefCtx(n)]) {
			if readEnvBool(cur, &cdfs.SingleRef[1][envBwdRefCtx(n)]) {
				refs.Ref0 = ReferenceFrameAltref
			} else {
				refs.Ref0 = ReferenceFrameBWD + ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.SingleRef[5][envBwdRef1Ctx(n)]))
			}
		} else if readEnvBool(cur, &cdfs.SingleRef[2][envFwdRefCtx(n)]) {
			refs.Ref0 = ReferenceFrameLast3 + ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.SingleRef[4][envFwdRef2Ctx(n)]))
		} else {
			refs.Ref0 = ReferenceFrameLast + ReferenceFrame(cur.ReadBinaryCDFUnchecked(&cdfs.SingleRef[3][envFwdRef1Ctx(n)]))
		}
	}
	return refs, nil
}

func readEnvBool(c *entropy.Reader, cdf *entropy.CDF) bool {
	return c.ReadBinaryCDFUnchecked(cdf) != 0
}

// envFilterCtx ports get_filter_ctx: a neighbour contributes its filter for
// direction dir only when it is an inter block using ref; intra and bilinear
// entries hold filterUnset (switchableFilterMissing).
func envFilterCtx(c *BlockModeContext, x4, y4 int, haveTop, haveLeft bool, comp bool, dir int, ref ReferenceFrame) int {
	aFilter, lFilter := switchableFilterMissing, switchableFilterMissing
	if haveTop && (c.A.Ref[0][x4] == ref || c.A.Ref[1][x4] == ref) {
		aFilter = int(c.A.Filter[dir][x4])
	}
	if haveLeft && (c.L.Ref[0][y4] == ref || c.L.Ref[1][y4] == ref) {
		lFilter = int(c.L.Filter[dir][y4])
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
	// The symbols are read straight from the tile's range decoder, as
	cur := &s.Reader // dav1d reads through &ts->msac.
	y := motion.InterpFilter(cur.ReadCDF3Unchecked(&cdfs.Switchable[envFilterCtx(c, int(block.X4), int(block.Y4), block.HaveTop, block.HaveLeft, refs.Compound, 0, refs.Ref0)]))
	x, reads := y, 1
	if req.EnableDualFilter {
		x = motion.InterpFilter(cur.ReadCDF3Unchecked(&cdfs.Switchable[envFilterCtx(c, int(block.X4), int(block.Y4), block.HaveTop, block.HaveLeft, refs.Compound, 1, refs.Ref0)]))
		reads++
	}
	return motion.InterpFilters{X: x, Y: y}, reads, nil
}

// envSingleRefContext is the context of single_ref bit (decode_b's ref[0..5]
// CDFs): av1_get_ref_ctx and av1_get_ref_2..6_ctx.
func envSingleRefContext(n envRefCountSet, bit int) (int, error) {
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
func envCompFwdRefContext(n envRefCountSet, bit int) (int, error) {
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
func envCompBwdRefContext(n envRefCountSet, bit int) (int, error) {
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
func envUniCompRefContext(n envRefCountSet, bit int) (int, error) {
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
