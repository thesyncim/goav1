package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/prediction"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// pframe_intra_sc.go is the intra candidate set SVT-AV1 injects in inter
// pictures of screen content at preset 12 rtc (v4.1.0):
//
//   - svt_aom_get_intra_mode_levels_rtc: intra_level = is_islice ? 4 : 6 for
//     sc_class1 (camera content runs MAX_INTRA_LEVEL - 1, DC_PRED only) and
//     dist_based_ang_intra_level 1.
//   - set_intra_ctrls level 6: intra_mode_end SMOOTH_PRED,
//     angular_pred_level[6] = 4, no pruning.
//   - inject_intra_candidates (mode_decision.c): angular_pred_level >= 4
//     masks D45_PRED and up, and >= 3 keeps only angle delta 0, so the
//     candidates are DC_PRED, V_PRED, H_PRED and SMOOTH_PRED, each with
//     uv_mode = intra_luma_to_chroma[mode] (ind_uv_avail is off in light PD1)
//     and the luma angle delta for the chroma angle.
//   - generate_md_stage_0_cand_light_pd1: DC only when md_me_dist (the
//     smallest post-sub-pel ME cost) is under dc_only_th per sample.
//   - fast_loop_core_light_pd1: an intra candidate is skipped when the best
//     candidate so far has a luma_fast_dist under dc_only_th (skip_dc_th for
//     DC) per sample, else scored as variance << 4 plus svt_aom_intra_fast_cost
//     (is_inter flag, mb_mode_fac_bits by size group, angle delta, uv mode and
//     uv angle delta) at full lambda.
//
// The chroma transform type is implicit from the uv mode
// (tile.IntraChromaTransformType; this encoder never sets reduced_tx_set).

// svtSCIntraModes are inject_intra_candidates' modes at intra_level 6, in
// injection order.
var svtSCIntraModes = [4]tile.IntraMode{tile.IntraModeDC, tile.IntraModeVertical, tile.IntraModeHorizontal, tile.IntraModeSmooth}

// svtSCChromaModes is intra_luma_to_chroma for svtSCIntraModes.
var svtSCChromaModes = [4]tile.ChromaIntraMode{tile.ChromaIntraModeDC, tile.ChromaIntraModeVertical, tile.ChromaIntraModeHorizontal, tile.ChromaIntraModeSmooth}

// svtSizeGroup is size_group_lookup for the square sizes light-PD1 injects
// intra on.
func svtSizeGroup(size tile.BlockSize) int {
	switch size {
	case tile.BlockSize8x8:
		return 1
	case tile.BlockSize16x16:
		return 2
	}
	return 3
}

// svtIntraModeRates are the frame-frozen svt_aom_intra_fast_cost terms of
// the screen-content candidates (md_rate_estimation.c from the frame-start
// CDFs): mb_mode_fac_bits by size group, the zero angle delta of V/H
// (angle_delta_fac_bits[mode - V_PRED][MAX_ANGLE_DELTA], shared by luma and
// chroma) and intra_uv_mode_fac_bits[cfl_allowed][mode][uv_mode].
type svtIntraModeRates struct {
	ok     bool
	yMode  [4][4]uint32
	angle0 [4]uint32
	uvMode [2][4]uint32
}

func (r *svtIntraModeRates) estimate(c *tile.IntraModeCDFs) {
	r.ok = false
	for g := range r.yMode {
		cdf, err := c.YModeCDF(g)
		if err != nil {
			return
		}
		for i, m := range svtSCIntraModes {
			r.yMode[g][i] = entropy.CostSymbol(cdf, int(m))
		}
	}
	for i, m := range svtSCIntraModes {
		r.angle0[i] = 0
		if m == tile.IntraModeVertical || m == tile.IntraModeHorizontal {
			cdf, err := c.AngleDeltaCDF(m)
			if err != nil {
				return
			}
			r.angle0[i] = entropy.CostSymbol(cdf, tile.AngleDeltaMax)
		}
		for cfl := range r.uvMode {
			cdf, err := c.UVModeCDF(cfl == 1, m)
			if err != nil {
				return
			}
			r.uvMode[cfl][i] = entropy.CostSymbol(cdf, int(svtSCChromaModes[i]))
		}
	}
	r.ok = true
}

// rate is svt_aom_intra_fast_cost's luma (without the is_inter flag) and
// chroma rate of candidate i for a square block.
func (r *svtIntraModeRates) rate(i int, size tile.BlockSize, cflAllowed bool) (luma, chroma int64) {
	luma = int64(r.yMode[svtSizeGroup(size)][i]) + int64(r.angle0[i])
	c := 0
	if cflAllowed {
		c = 1
	}
	chroma = int64(r.uvMode[c][i]) + int64(r.angle0[i])
	return luma, chroma
}

// svtResolveIntraEdges fills above / left with the AV1 intra edge samples
// (spec 7.11.2) of the w x h block at (x, y) on a planeW x planeH 8-bit
// plane: the reconstructed neighbours clamped to the plane, else the left
// (above) neighbour, else 127 (129). V_PRED, H_PRED and SMOOTH_PRED read
// nothing else, and none of them filters its edges.
func svtResolveIntraEdges(plane []byte, stride, planeW, planeH, x, y, w, h int, haveTop, haveLeft bool, above, left []uint16) prediction.IntraEdges {
	switch {
	case haveTop:
		row := plane[(y-1)*stride:]
		for i := range w {
			above[i] = uint16(row[min(x+i, planeW-1)])
		}
	case haveLeft:
		v := uint16(plane[y*stride+x-1])
		for i := range w {
			above[i] = v
		}
	default:
		for i := range w {
			above[i] = 127
		}
	}
	switch {
	case haveLeft:
		for j := range h {
			left[j] = uint16(plane[min(y+j, planeH-1)*stride+x-1])
		}
	case haveTop:
		v := uint16(plane[(y-1)*stride+x])
		for j := range h {
			left[j] = v
		}
	default:
		for j := range h {
			left[j] = 129
		}
	}
	return prediction.IntraEdges{Above: above[:w], Left: left[:h], AboveAvailable: true, LeftAvailable: true, AboveLeftAvailable: true}
}

// svtPredictIntraBlock predicts the w x h block at (x, y) with one of the
// screen-content candidate modes into dst (stride w).
func (st *lossyEncodeState) svtPredictIntraBlock(dst, plane []byte, stride, planeW, planeH, x, y, w, h int, haveTop, haveLeft bool, mode tile.IntraMode) error {
	if mode == tile.IntraModeDC {
		memsetBytes(dst[:w*h], dcPredictRect(plane, stride, x, y, w, h, haveTop, haveLeft))
		return nil
	}
	edges := svtResolveIntraEdges(plane, stride, planeW, planeH, x, y, w, h, haveTop, haveLeft, st.intraEdgeAbove[:], st.intraEdgeLeft[:])
	var pm prediction.IntraMode
	switch mode {
	case tile.IntraModeVertical:
		pm = prediction.IntraModeVertical
	case tile.IntraModeHorizontal:
		pm = prediction.IntraModeHorizontal
	default:
		pm = prediction.IntraModeSmooth
	}
	return prediction.PredictIntraPlaneBlock(frame.Plane{Pix: dst, Stride: w, Width: w, Height: h}, 1, 8, 0, 0, w, h, pm, edges)
}

// svtSCChromaMode is intra_luma_to_chroma for the candidate modes.
func svtSCChromaMode(mode tile.IntraMode) tile.ChromaIntraMode {
	for i, m := range svtSCIntraModes {
		if m == mode {
			return svtSCChromaModes[i]
		}
	}
	return tile.ChromaIntraModeDC
}

// svtIntraUVTxType is the implicit transform type of an intra chroma block.
func svtIntraUVTxType(mode tile.ChromaIntraMode, size tile.TransformSize) transform.Type {
	t, err := tile.IntraChromaTransformType(mode, size, false)
	if err != nil {
		return transform.TypeDCTDCT
	}
	return t
}

// svtDCOnlyTh is cand_elimination_ctrls.dc_only_th for the light-PD1 levels
// (set_cand_reduction_ctrls levels 2..5, non-flat rtc): 30 on referenced
// pictures, else 200 (LPD1_LVL_0..2) or 600 (LPD1_LVL_3..5). REGULAR_PD1
// runs cand_reduction_level 1, which disables candidate elimination.
func svtDCOnlyTh(level lpd1PD1Level, notLastLayer bool) (th int, enabled bool) {
	switch {
	case level == lpd1RegularPD1:
		return 0, false
	case notLastLayer:
		return 30, true
	case level <= 2:
		return 200, true
	}
	return 600, true
}

// chromaPlaneDims is the chroma plane size of src.
func (st *lossyEncodeState) chromaPlaneDims(src SourceFrame420) (int, int) {
	w, h := src.Width, src.Height
	if st.color.SubsamplingX {
		w = (w + 1) >> 1
	}
	if st.color.SubsamplingY {
		h = (h + 1) >> 1
	}
	return w, h
}

// writeUVAngleDelta codes intra_angle_info_uv: the zero angle delta of a
// directional chroma mode, whose luma-mode angle-delta CDF matches the luma
// mode the candidate's uv mode derives from.
func (st *lossyEncodeState) writeUVAngleDelta(block tile.BlockVisit, mode tile.IntraMode, uvMode tile.ChromaIntraMode) error {
	if uvMode == tile.ChromaIntraModeDC {
		return nil
	}
	if err := tile.WriteIntraAngleDelta(st.w, &st.intraCDFs, tile.IntraAngleDeltaRequest{Size: block.Size, Mode: mode}, 0); err != nil {
		return fmt.Errorf("intra uv angle delta: %w", err)
	}
	return nil
}
