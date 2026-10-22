package encoder

// pframe_lpd1_intra.go runs an intra block of an inter frame through SVT-AV1's
// light-PD1 full loop, the same full_loop_core_light_pd1
// (Source/Lib/Codec/product_coding_loop.c) its inter candidates take. The DC
// candidate gets no separate path there; the light-PD1 TX shortcuts apply to
// it as they do to inter candidates:
//
//   - lpd1_tx_shortcut_detector: use_tx_shortcuts_mds3 from the MDS0 winner's
//     luma_fast_dist, which for a DC winner is its residual variance.
//   - get_perform_tx_flag: the lpd1_skip_inter_tx_level arms only take inter
//     candidates, but lpd1_bypass_tx_th prices every candidate on its
//     fast_luma_rate and its prediction SSE (full_dist).
//   - zero_y_coeff_exit: a luma block without coefficients skips the chroma
//     TX unless chroma_complexity_check flags a plane. For an intra
//     candidate that check is the chroma source variance against
//     th = 75 (detector level <= 1) or 150.
//   - chroma_complexity_check_pred, then svt_aom_full_loop_chroma_light_pd1's
//     per-plane N4_SHAPE for planes not detected as complex.
//
// Preset 12 rtc runs no RDOQ in inter pictures (pcs->rdoq_level =
// is_islice), so intra blocks quantize with quantize_b as before.

import (
	"fmt"
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

// intraChromaGeom is the chroma side of an intra block in an inter frame.
type intraChromaGeom struct {
	cw, ch     int
	x, y       int
	tx         tile.TransformSize
	haveTop    bool
	haveLeft   bool
	planeBlock tile.BlockSize
}

// intraFullLoopResult reports the quantized planes of an intra block.
type intraFullLoopResult struct {
	lumaGeo, chromaGeo   txBlockGeometry
	lumaZero, uZero, vZero bool
}

// lpd1IntraFullLoop predicts the chroma DC planes into st.predU/st.predV and
// quantizes the planes of an intra block whose luma prediction is in
// st.predY (stride n): luma into qY, chroma into qU and qV. Planes the full
// loop skips are left all-zero; their reconstruction is the prediction.
func (st *lossyEncodeState) lpd1IntraFullLoop(src SourceFrame420, recon *SourceFrame420, block tile.BlockVisit,
	lumaPX, lumaPY, n int, lumaTX tile.TransformSize, cg *intraChromaGeom, qY, qU, qV []int16) (intraFullLoopResult, error) {
	res := intraFullLoopResult{uZero: true, vZero: true}
	pred := st.predY[:n*n]
	light := st.lpd1Tx.enabled && videoColorIs420(st.color)
	var dec *lpd1SBDecision
	useShortcuts := false
	performTX := true
	if light {
		dec = st.lpd1SB(lumaPX, lumaPY)
		if s := &st.mds0Intra; s.valid {
			useShortcuts, _ = st.lpd1TxDetect(s.variance, n, n)
			// full_dist is the vf sse against the DC prediction:
			// sum (s - dc)^2 = sum s^2 - 2 dc sum s + n dc^2.
			dc := int64(pred[0])
			sse := int64(s.srcSSE) - 2*dc*int64(s.srcSum) + int64(n*n)*dc*dc
			performTX = !st.lpd1BypassTXRate(s.lumaRate, uint32(sse), n, n, dec.bypassTh, st.lpd1FullLambda(dec))
		}
	}
	var err error
	if performTX {
		res.lumaGeo, res.lumaZero, err = st.prepareIntraTXB(src.Y, src.YStride, lumaPX, lumaPY, n, n, st.yQuant, lumaTX, pred, qY)
		if err != nil {
			return res, err
		}
	} else {
		// perform_tx = 0: no luma coefficients (y_has_coeff = 0).
		if res.lumaGeo, err = txBlockGeometryForTransformSize(lumaTX); err != nil {
			return res, err
		}
		clear(qY[:res.lumaGeo.coeffCount])
		res.lumaZero = true
	}
	if cg == nil {
		return res, nil
	}
	if res.chromaGeo, err = txBlockGeometryForTransformSize(cg.tx); err != nil {
		return res, err
	}
	// zero_y_coeff_exit + chroma_complexity_check.
	comp := lpd1ComponentChroma
	complexity := lpd1ComponentLuma
	if light && st.lpd1Tx.zeroYCoeffExit && res.lumaZero {
		comp = lpd1ComponentLuma
		if dec.chromaDetector != 0 {
			comp = lpd1IntraChromaComplexity(src, cg, dec.chromaDetector)
			if dec.chromaDetector <= 3 {
				complexity = comp
			}
		}
	}
	// The DC predictions are the reconstruction of a plane the loop skips.
	predU, predV := st.predU[:cg.cw*cg.ch], st.predV[:cg.cw*cg.ch]
	fillRect(predU, dcPredictRect(recon.U, src.ChromaStride, cg.x, cg.y, cg.cw, cg.ch, cg.haveTop, cg.haveLeft))
	fillRect(predV, dcPredictRect(recon.V, src.ChromaStride, cg.x, cg.y, cg.cw, cg.ch, cg.haveTop, cg.haveLeft))
	if comp == lpd1ComponentLuma {
		clear(qU[:res.chromaGeo.coeffCount])
		clear(qV[:res.chromaGeo.coeffCount])
		return res, nil
	}
	n4U, n4V := false, false
	if light {
		if dec.chromaDetector != 0 && dec.chromaDetector <= 3 && complexity != lpd1ComponentChroma &&
			(useShortcuts || st.lpd1Tx.useUVShortcutsOnYCoeffs) {
			complexity = st.lpd1ChromaComplexityPred(src, lumaPX, lumaPY, n, cg.x, cg.y, cg.cw, cg.ch, complexity)
		}
		n4U, n4V = st.lpd1ChromaN4(block.Size, useShortcuts, complexity, res.lumaZero, !performTX, qY, lumaTX, false)
	}
	if comp == lpd1ComponentChroma || comp == lpd1ComponentChromaCb {
		st.pfN4 = n4U
		_, res.uZero, err = st.prepareIntraTXB(src.U, src.ChromaStride, cg.x, cg.y, cg.cw, cg.ch, st.uQuant, cg.tx, predU, qU)
		st.pfN4 = false
		if err != nil {
			return res, err
		}
	} else {
		clear(qU[:res.chromaGeo.coeffCount])
	}
	if comp == lpd1ComponentChroma || comp == lpd1ComponentChromaCr {
		st.pfN4 = n4V
		_, res.vZero, err = st.prepareIntraTXB(src.V, src.ChromaStride, cg.x, cg.y, cg.cw, cg.ch, st.vQuant, cg.tx, predV, qV)
		st.pfN4 = false
		if err != nil {
			return res, err
		}
	} else {
		clear(qV[:res.chromaGeo.coeffCount])
	}
	return res, nil
}

// lpd1IntraChromaComplexity is chroma_complexity_check's intra arm: each
// chroma plane's source variance (vf against svt_aom_eb_av1_var_offs, a
// flat 128 block, so the plain variance) rounded to a per-pixel value, and
// a plane above th (75 at detector level <= 1, else 150) is complex.
func lpd1IntraChromaComplexity(src SourceFrame420, cg *intraChromaGeom, level uint8) lpd1Component {
	shift := uint(bits.TrailingZeros(uint(cg.cw * cg.ch)))
	off := cg.y*src.ChromaStride + cg.x
	blockVar := func(plane []byte) uint32 {
		sse, sum := pixelStatsRect(plane[off:], src.ChromaStride, svtZeroRow[:], 0, cg.cw, cg.ch)
		v := varianceFromStats(sse, sum, shift)
		return (v + (1 << (shift - 1))) >> shift
	}
	th := uint32(150)
	if level <= 1 {
		th = 75
	}
	cb, cr := blockVar(src.U) > th, blockVar(src.V) > th
	switch {
	case cb && cr:
		return lpd1ComponentChroma
	case cb:
		return lpd1ComponentChromaCb
	case cr:
		return lpd1ComponentChromaCr
	}
	return lpd1ComponentLuma
}

func fillRect(dst []byte, v byte) {
	for i := range dst {
		dst[i] = v
	}
}

// pixelStatsRect dispatches the chroma block sizes of the light-PD1 path to
// the fixed-size pixel-statistics kernels (svt_aom_mefn_ptr[bsize_uv].vf).
func pixelStatsRect(src []byte, srcStride int, ref []byte, refStride, w, h int) (uint32, int32) {
	switch {
	case w == 4 && h == 4:
		return pixelStats4x4(src, srcStride, ref, refStride)
	case w == 8 && h == 8:
		return pixelStats8x8(src, srcStride, ref, refStride)
	case w == 16 && h == 16:
		return pixelStats16x16(src, srcStride, ref, refStride)
	case w == 32 && h == 32:
		return pixelStats32x32(src, srcStride, ref, refStride)
	case w == 16 && h == 8:
		return pixelStats16x8(src, srcStride, ref, refStride)
	case w == 8 && h == 16:
		return pixelStats8x16(src, srcStride, ref, refStride)
	case w == 32 && h == 16:
		return pixelStats32x16(src, srcStride, ref, refStride)
	case w == 16 && h == 32:
		return pixelStats16x32(src, srcStride, ref, refStride)
	}
	return pixelStatsPureGo(src, srcStride, ref, refStride, w, h)
}

// intraChromaGeometry fills cg with the chroma plane block, transform and
// position of an intra block, and returns the block's chroma scan.
func (st *lossyEncodeState) intraChromaGeometry(cg *intraChromaGeom, block tile.BlockVisit, lumaPX, lumaPY int) (*intraChromaGeom, tile.BlockSize, []int16, error) {
	chromaBlock, err := tile.PlaneBlockSize(block.Size, st.color, 1)
	if err != nil {
		return nil, chromaBlock, nil, fmt.Errorf("chroma plane block: %w", err)
	}
	cw, ch, err := planeBlockPixels(chromaBlock)
	if err != nil {
		return nil, chromaBlock, nil, fmt.Errorf("chroma plane block dimensions: %w", err)
	}
	chromaTX, err := tile.MaxTransformSize(block.Size, st.color, 1)
	if err != nil {
		return nil, chromaBlock, nil, fmt.Errorf("chroma transform size: %w", err)
	}
	scan, ok := st.scanForTransformSize(chromaTX)
	if !ok {
		scan, ok = st.scanForThinTransformSize(chromaTX)
	}
	if !ok {
		return nil, chromaBlock, nil, fmt.Errorf("encoder: unsupported chroma transform %d", chromaTX)
	}
	*cg = intraChromaGeom{
		cw: cw, ch: ch,
		x: chromaXForColor(lumaPX, st.color), y: chromaYForColor(lumaPY, st.color),
		tx:      chromaTX,
		haveTop: block.HaveTop, haveLeft: block.HaveLeft,
		planeBlock: chromaBlock,
	}
	return cg, chromaBlock, scan, nil
}
