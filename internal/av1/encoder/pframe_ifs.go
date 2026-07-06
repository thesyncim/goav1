package encoder

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// pframe_ifs.go is the interpolation filter search for reference (base-layer)
// frames, ported from SVT-AV1's interpolation_filter_search
// (Source/Lib/Codec/enc_inter_prediction.c). Preset 12 keeps
// interpolation_search_level = 4 for is_base pictures (enc_mode_config.c: the
// >M8 arm keeps level 4 and only non-base pictures drop to 0 on high
// ref_skip_percentage), signals frm_hdr->interpolation_filter = SWITCHABLE,
// and disables sequence-level dual filters (sequence_control_set.c
// enable_dual_filter = 0), so the search covers the three equal-pair combos
// of filter_sets:
//
//   - Rate: svt_aom_get_switchable_rate (Source/Lib/Codec/rd_cost.c) over the
//     frame-frozen switchable_interp_fac_bitss tables.
//   - Full-pel bypass: the filter cannot change a full-pel prediction, so the
//     combo is chosen on switchable rate alone (is_fp path).
//   - Distortion: the luma prediction under each filter, its SSE against the
//     source, and model_rd_from_sse's Laplacian rate/distortion model
//     (svt_av1_model_rd_from_var_lapndz), then
//     RDCOST(full_lambda, rs + rate, dist).

// modelRDNorm ports model_rd_norm (Source/Lib/Codec/enc_inter_prediction.c):
// the Laplacian-source rate and distortion curves sampled on the shared
// xsq_iq_q10 grid with linear interpolation.
func modelRDNorm(xsqQ10 int32) (rQ10, dQ10 int32) {
	// Normalized rate: Rn(x) = H(sqrt(r)) + sqrt(r)*[1 + H(r)/(1 - r)],
	// r = exp(-sqrt(2) * x), x = qpstep / sqrt(variance).
	rateTabQ10 := [...]int32{
		65536, 6086, 5574, 5275, 5063, 4899, 4764, 4651, 4553, 4389, 4255, 4142, 4044, 3958, 3881, 3811, 3748, 3635,
		3538, 3453, 3376, 3307, 3244, 3186, 3133, 3037, 2952, 2877, 2809, 2747, 2690, 2638, 2589, 2501, 2423, 2353,
		2290, 2232, 2179, 2130, 2084, 2001, 1928, 1862, 1802, 1748, 1698, 1651, 1608, 1530, 1460, 1398, 1342, 1290,
		1243, 1199, 1159, 1086, 1021, 963, 911, 864, 821, 781, 745, 680, 623, 574, 530, 490, 455, 424,
		395, 345, 304, 269, 239, 213, 190, 171, 154, 126, 104, 87, 73, 61, 52, 44, 38, 28,
		21, 16, 12, 10, 8, 6, 5, 3, 2, 1, 1, 1, 0, 0,
	}
	// Normalized distortion: Dn(x) = 1 - 1/sqrt(2) * x / sinh(x/sqrt(2)).
	distTabQ10 := [...]int32{
		0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9,
		11, 12, 13, 15, 16, 17, 18, 21, 24, 26, 29, 31, 34, 36, 39, 44, 49, 54,
		59, 64, 69, 73, 78, 88, 97, 106, 115, 124, 133, 142, 151, 167, 184, 200, 215, 231,
		245, 260, 274, 301, 327, 351, 375, 397, 418, 439, 458, 495, 528, 559, 587, 613, 637, 659,
		680, 717, 749, 777, 801, 823, 842, 859, 874, 899, 919, 936, 949, 960, 969, 977, 983, 994,
		1001, 1006, 1010, 1013, 1015, 1017, 1018, 1020, 1022, 1022, 1023, 1023, 1023, 1024,
	}
	xsqIQQ10 := [...]int32{
		0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64,
		72, 80, 88, 96, 112, 128, 144, 160, 176, 192, 208, 224, 256,
		288, 320, 352, 384, 416, 448, 480, 544, 608, 672, 736, 800, 864,
		928, 992, 1120, 1248, 1376, 1504, 1632, 1760, 1888, 2016, 2272, 2528, 2784,
		3040, 3296, 3552, 3808, 4064, 4576, 5088, 5600, 6112, 6624, 7136, 7648, 8160,
		9184, 10208, 11232, 12256, 13280, 14304, 15328, 16352, 18400, 20448, 22496, 24544, 26592,
		28640, 30688, 32736, 36832, 40928, 45024, 49120, 53216, 57312, 61408, 65504, 73696, 81888,
		90080, 98272, 106464, 114656, 122848, 131040, 147424, 163808, 180192, 196576, 212960, 229344, 245728,
	}
	tmp := (xsqQ10 >> 2) + 8
	k := int32(bits.Len32(uint32(tmp))) - 1 - 3 // get_msb(tmp) - 3
	xq := (k << 3) + ((tmp >> k) & 0x7)
	const oneQ10 = int32(1) << 10
	aQ10 := ((xsqQ10 - xsqIQQ10[xq]) << 10) >> (2 + k)
	bQ10 := oneQ10 - aQ10
	rQ10 = (rateTabQ10[xq]*bQ10 + rateTabQ10[xq+1]*aQ10) >> 10
	dQ10 = (distTabQ10[xq]*bQ10 + distTabQ10[xq+1]*aQ10) >> 10
	return rQ10, dQ10
}

// modelRDFromVarLapndz ports svt_av1_model_rd_from_var_lapndz
// (Source/Lib/Codec/enc_inter_prediction.c): rate in 512-per-bit units and
// distortion in the sse domain for a Laplacian source of the given variance
// quantized at qstep.
func modelRDFromVarLapndz(variance int64, nLog2 uint, qstep int64) (rate int64, dist int64) {
	if variance == 0 {
		return 0, 0
	}
	const maxXSqQ10 = 245727
	xsq64 := ((uint64(qstep*qstep) << (nLog2 + 10)) + uint64(variance>>1)) / uint64(variance)
	if xsq64 > maxXSqQ10 {
		xsq64 = maxXSqQ10
	}
	rQ10, dQ10 := modelRDNorm(int32(xsq64))
	// ROUND_POWER_OF_TWO(r_q10 << n_log2, 10 - AV1_PROB_COST_SHIFT).
	rate = ((int64(rQ10) << nLog2) + 1) >> 1
	dist = (variance*int64(dQ10) + 512) >> 10
	return rate, dist
}

// modelRDFromSSE ports model_rd_from_sse's simple_model_rd_from_var = 0 arm
// as model_rd_for_sb invokes it (Source/Lib/Codec/enc_inter_prediction.c):
// the quantizer is the AC dequant in QTX scale, dequant_shift is
// bit_depth - 5 = 3 at 8-bit, and the modeled distortion is scaled by 16.
func modelRDFromSSE(nLog2 uint, acDequantQTX int64, sse int64) (rate int64, dist int64) {
	qstep := acDequantQTX >> 3
	rate, dist = modelRDFromVarLapndz(sse, nLog2, qstep)
	return rate, dist << 4
}

// predictIntoFilters is predictInto with a caller-chosen interpolation filter
// pair; the reference plane must match the current frame dimensions.
func predictIntoFilters(dst []byte, refPlane []byte, stride, width, height, px, py, bw, bh int, mv motion.Vector, ssX, ssY bool, filters motion.InterpFilters, scratch *motion.ConvolveScratch) error {
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(px, py, mv, ssX, ssY)
	if err != nil {
		return err
	}
	dstPlane := frame.Plane{Pix: dst, Stride: bw, Width: bw, Height: bh}
	ref := frame.Plane{Pix: refPlane, Stride: stride, Width: width, Height: height}
	return motion.PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSizeScratch(dstPlane, ref, 1, 8, 0, 0, refX, refY, bw, bh, bw, bh, subX, subY, filters, scratch)
}

// interpolationFilterSearch runs the IFS combo loop for one single-reference
// block whose mode and vector are final, returning the winning filter pair
// (X = Y with sequence dual filters disabled, as SVT-AV1 configures them).
func (st *lossyEncodeState) interpolationFilterSearch(src SourceFrame420, refPlane []byte, refStride int,
	modeCtx *tile.BlockModeContext, ifsReq tile.InterpFilterRequest,
	lumaPX, lumaPY, bw, bh int, mv motion.Vector) motion.InterpFilters {

	ctx0, err := modeCtx.SwitchableInterpContext(ifsReq, 0)
	if err != nil {
		return motion.RegularFilters
	}
	ctx1 := -1
	if ifsReq.EnableDualFilter {
		if c, cerr := modeCtx.SwitchableInterpContext(ifsReq, 1); cerr == nil {
			ctx1 = c
		} else {
			return motion.RegularFilters
		}
	}
	// The filter cannot change a full-pel prediction, so full-pel vectors
	// pick the combo on switchable rate alone (interpolation_filter_search's
	// is_fp bypass).
	isFP := mv.Row%8 == 0 && mv.Col%8 == 0
	nLog2 := uint(bits.TrailingZeros(uint(bw * bh)))
	best := motion.InterpEightTapRegular
	bestRD := int64(1) << 62
	for f := motion.InterpEightTapRegular; f <= motion.InterpMultiTapSharp; f++ {
		rs := int64(st.interpRates.SwitchableBits(ctx0, f))
		if ctx1 >= 0 {
			rs += int64(st.interpRates.SwitchableBits(ctx1, f))
		}
		var rd int64
		if isFP {
			rd = (rs*st.rdMult + 256) >> 9
		} else {
			if perr := predictIntoFilters(st.sadScratch[:bw*bh], refPlane, refStride, src.Width, src.Height, lumaPX, lumaPY, bw, bh, mv, false, false, motion.InterpFilters{X: f, Y: f}, st.scaledScratch.Conv()); perr != nil {
				continue
			}
			sse, _ := realtimeInterResidualSSEVariance(src.Y, st.sadScratch[:bw*bh], src.YStride, bw, lumaPX, lumaPY, bw, bh)
			rate, dist := modelRDFromSSE(nLog2, int64(st.yQuant.AC), int64(sse))
			rd = (((rs+rate)*st.rdMult + 256) >> 9) + (dist << 7)
		}
		if rd < bestRD {
			bestRD = rd
			best = f
		}
	}
	return motion.InterpFilters{X: best, Y: best}
}
