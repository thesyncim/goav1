package encoder

// pframe_lpd1_tx.go ports SVT-AV1's light-PD1 TX shortcut detectors, the
// mechanism preset 12 uses to stop paying for transform work on easy blocks:
//
//   - Controls: enc_mode_config.c set_lpd1_tx_ctrls(). rtc pictures are
//     forced to level 3 ("rtc forced to use lvl 3 because using the more
//     aggressive levels for only non-base pictures causes concentrated bit
//     allocation for flat rtc", enc_mode_config.c:7325-7338):
//     zero_y_coeff_exit=1, use_mds3_shortcuts_th=30, use_neighbour_info=0.
//     The chroma detector level is the ENC_M10+ override
//     (enc_mode_config.c:7340-7355): 4 while lpd1_level <= LPD1_LVL_4 (base
//     pictures, pic_lpd1_lvl=3 -> LPD1_LVL_1), 0 above it (droppable
//     leaves, pic_lpd1_lvl=7 -> LPD1_LVL_5).
//   - Detector: product_coding_loop.c lpd1_tx_shortcut_detector(). The MDS0
//     winner's luma residual variance (fast_loop_core_light_pd1 stores
//     fn_ptr->vf as luma_fast_dist) is compared against
//     th * bwidth * bheight * qp_index / 100. use_neighbour_info=0 at level
//     3, so the neighbour-multiplier arm is not ported.
//   - Consumption: zero_y_coeff_exit + chroma_complexity_check() (skip the
//     chroma TX when luma coded all-zero and chroma is not complex) and
//     search_dct_dct_only() (skip the 8x8 TX-type trial, DCT_DCT only).
//
// PINNED REVERTS (measured on the four-clip corpus, 2026-07-06):
//   - get_perform_tx_flag's skip_tx_th=30 / lpd1_skip_inter_tx_level=1 arm
//     (drop the leaf-frame luma TX when both neighbours coded skip) measured
//     RED in the single-tile config: realB −0.29 dB rate-adjusted at −6.8%
//     rate (skip cascades through the neighbour feedback frame-wide; SVT
//     guards the arm with its per-SB LPD1 detector plus
//     l0_was_skip/ref_skip_perc/me_8x8_cost_variance picture stats this
//     encoder does not collect). Do not re-enable without those stats.
//   - Pinning shortcut blocks to the largest square leaf (LPD1's tx_depth-0
//     shape) moved realA/realB −0.17/−0.2 dB rate-adjusted; TX-size
//     reduction is the separate flag-gated largest-TX experiment.
//
import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// lpd1TxCtrls mirrors the consumed subset of SVT-AV1's Lpd1TxCtrls
// (md_process.h). The zero value disables every shortcut.
type lpd1TxCtrls struct {
	enabled bool
	// zeroYCoeffExit: skip chroma TX when the luma TXBs code all-zero
	// (subject to the chroma complexity detector).
	zeroYCoeffExit bool
	// useMds3ShortcutsTh: percentage threshold arming use_tx_shortcuts_mds3.
	useMds3ShortcutsTh uint16
	// chromaDetectorLevel selects chroma_complexity_check behaviour
	// (0 = no detector: chroma dropped whenever luma is all-zero).
	chromaDetectorLevel uint8
}

// setLPD1TxCtrls applies SVT-AV1 set_lpd1_tx_ctrls level 3 — the level rtc
// pictures are forced to at every preset that runs LPD1 — with the picture
// modulation derived from isBase (temporal layer 0 / referenced):
// chroma_complexity_check level 4 on base pictures, 0 on droppable leaves.
func (st *lossyEncodeState) setLPD1TxCtrls(enabled bool, isBase bool) {
	if !enabled {
		st.lpd1Tx = lpd1TxCtrls{}
		return
	}
	ctrls := lpd1TxCtrls{
		enabled:            true,
		zeroYCoeffExit:     true,
		useMds3ShortcutsTh: 30,
	}
	if isBase {
		ctrls.chromaDetectorLevel = 4
	} else {
		ctrls.chromaDetectorLevel = 0
	}
	st.lpd1Tx = ctrls
}

// lpd1TxDetect is lpd1_tx_shortcut_detector's use_tx_shortcuts_mds3 arm
// (product_coding_loop.c): compare the winner's luma residual variance
// against th_normalizer = bheight * bwidth * qp_index. use_neighbour_info is
// 0 at level 3, so the neighbour-multiplier refinement is not ported.
func (st *lossyEncodeState) lpd1TxDetect(variance uint32, bw, bh int) bool {
	if !st.lpd1Tx.enabled {
		return false
	}
	thNormalizer := uint64(bw) * uint64(bh) * uint64(st.qIndex)
	return 100*uint64(variance) < uint64(st.lpd1Tx.useMds3ShortcutsTh)*thNormalizer
}

// lpd1Component mirrors SVT's COMPONENT_TYPE subset the chroma detector
// returns: luma-only (chroma can be dropped), one chroma plane, or both.
type lpd1Component uint8

const (
	lpd1ComponentLuma lpd1Component = iota
	lpd1ComponentChromaCb
	lpd1ComponentChromaCr
	lpd1ComponentChroma
)

// lpd1ChromaComplexity is chroma_complexity_check's inter arm
// (product_coding_loop.c): compare the full-pel luma/chroma SADs of the
// winner's motion; a chroma plane whose SAD exceeds the (shifted) luma SAD is
// complex and keeps its TX. Rows are subsampled by shift (chroma_detector
// level >= 2: 4 rows sampled for blocks above 16px chroma height) and the
// luma SAD is computed over the chroma-sized region so the SADs compare
// (y_dist <<= 2 at level >= 2). SVT reads padded reference pictures; this
// encoder's planes are tight, so the displaced windows clamp to the plane
// (the same discipline as realtimeSAD64Clamped).
func (st *lossyEncodeState) lpd1ChromaComplexity(src SourceFrame420, refPlanes SourceFrame420, mv motion.Vector,
	lumaPX, lumaPY, cbw, cbh int) lpd1Component {
	shift := 0
	if cbh > 8 {
		shift = 2
	} else if cbh > 4 {
		shift = 1
	}
	rows := cbh >> shift
	if rows == 0 {
		rows = 1
	}
	step := 1 << shift

	mvX, mvY := int(mv.Col>>3), int(mv.Row>>3)
	refX := max(0, min(lumaPX+mvX, refPlanes.Width-cbw))
	refY := max(0, min(lumaPY+mvY, refPlanes.Height-(rows-1)*step-1))
	yDist := lpd1SampledSAD(src.Y, refPlanes.Y, lumaPY*src.YStride+lumaPX, refY*refPlanes.YStride+refX,
		src.YStride, refPlanes.YStride, cbw, rows, step)

	cw := chromaWidthForColor(refPlanes.Width, st.color)
	ch := chromaHeightForColor(refPlanes.Height, st.color)
	srcCX := chromaXForColor(lumaPX, st.color)
	srcCY := chromaYForColor(lumaPY, st.color)
	refCX := max(0, min((lumaPX+mvX)>>1, cw-cbw))
	refCY := max(0, min((lumaPY+mvY)>>1, ch-(rows-1)*step-1))
	cbDist := lpd1SampledSAD(src.U, refPlanes.U, srcCY*src.ChromaStride+srcCX, refCY*refPlanes.ChromaStride+refCX,
		src.ChromaStride, refPlanes.ChromaStride, cbw, rows, step)
	crDist := lpd1SampledSAD(src.V, refPlanes.V, srcCY*src.ChromaStride+srcCX, refCY*refPlanes.ChromaStride+refCX,
		src.ChromaStride, refPlanes.ChromaStride, cbw, rows, step)

	// Shift y_dist to ensure chroma is much higher than luma (level >= 2).
	yDist <<= 2
	switch {
	case cbDist > yDist && crDist > yDist:
		return lpd1ComponentChroma
	case cbDist > yDist:
		return lpd1ComponentChromaCb
	case crDist > yDist:
		return lpd1ComponentChromaCr
	}
	return lpd1ComponentLuma
}

// lpd1SampledSAD is the row-subsampled SAD the chroma detector uses
// (svt_nxm_sad_kernel called with stride << shift and height >> shift).
func lpd1SampledSAD(src, ref []byte, srcOff, refOff, srcStride, refStride, w, rows, step int) int {
	total := 0
	for r := 0; r < rows; r++ {
		s := src[srcOff+r*step*srcStride:]
		d := ref[refOff+r*step*refStride:]
		for c := 0; c < w; c++ {
			diff := int(s[c]) - int(d[c])
			if diff < 0 {
				diff = -diff
			}
			total += diff
		}
	}
	return total
}

// decideInterTXBlock runs one inter block's transform/quantize preparation
// and skip decision — shared verbatim by the fused (encodePBlock) and split
// (decidePBlock) paths so the wavefront byte-identity oracle holds by
// construction. It quantizes all planes up front, takes the rate-priced skip
// decision, and applies the light-PD1 TX shortcut detectors when armed:
// zero_y_coeff_exit + the chroma complexity detector (drop the chroma TX
// when luma coded all-zero) and search_dct_dct_only (skip the 8x8 TX-type
// trial, DCT_DCT only; the TX-size plan stays on the calculate_tx_size
// heuristic — see the pinned reverts in the file header). With st.lpd1Tx
// disabled the flow is byte-identical to the pre-port encoder.
//
// detectMV is the winner's first motion vector (block_mi.mv[0]) and
// refPlanes its first reference's planes, the chroma detector's inputs.
func (st *lossyEncodeState) decideInterTXBlock(src, refPlanes SourceFrame420, detectMV motion.Vector,
	block tile.BlockVisit,
	lumaPX, lumaPY, bw, bh, cbw, cbh, chromaPX, chromaPY int, hasChroma bool,
	fullSAD int) (skip, splitTX bool, txPlan realtimeInterTXPlan, txType transform.Type, err error) {

	// A near-perfect match skips the proof: with the luma SAD at a quarter
	// sample per pixel, no realtime quantizer step keeps a coefficient
	// (skip is an encoder choice the decoder honors either way).
	skip = fullSAD*4 <= bw*bh
	useRealtimeTXPlan := videoColorIs420(st.color)
	lpd1UseShortcuts := false
	if useRealtimeTXPlan {
		sse, variance := realtimeInterResidualSSEVariance(src.Y, st.predY[:bw*bh], src.YStride, bw, lumaPX, lumaPY, bw, bh)
		if st.lpd1Tx.enabled {
			lpd1UseShortcuts = st.lpd1TxDetect(variance, bw, bh)
		}
		txLevel := realtimeTXSizeLevelBasedOnQstep(src.Width, src.Height, st.effortLevel)
		txPlan, err = realtimeInterTXPlanForBlock(block.Size, st.qIndex, st.yQuant.AC, txLevel, sse, variance)
		if err != nil {
			return false, false, txPlan, transform.TypeDCTDCT, err
		}
	}
	dctRdD := int64(0)
	if !skip {
		st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
		lumaZero := true
		if txPlan.Variable() {
			leafArea := txPlan.leafSize * txPlan.leafSize
			if lerr := txPlan.ForEachLeaf(func(i, dx, dy int) error {
				if !st.prepareInterTXB(src.Y, st.predY[dy*bw+dx:], bw, src.YStride, lumaPX+dx, lumaPY+dy, txPlan.leafSize, txPlan.leafSize, st.yQuant, st.lumaQ2[i*leafArea:(i+1)*leafArea]) {
					lumaZero = false
				}
				return nil
			}); lerr != nil {
				return false, false, txPlan, transform.TypeDCTDCT, fmt.Errorf("prepare realtime luma tx: %w", lerr)
			}
		} else {
			lumaZero = st.prepareInterTXB(src.Y, st.predY[:bw*bh], bw, src.YStride, lumaPX, lumaPY, bw, bh, st.yQuant, st.lumaQ[:bw*bh])
		}
		if hasChroma {
			// zero_y_coeff_exit: an all-zero luma block skips the chroma TX
			// unless the chroma complexity detector flags a plane
			// (full_loop_core_light_pd1's perform_chroma derivation).
			performChroma := true
			comp := lpd1ComponentChroma
			if useRealtimeTXPlan && st.lpd1Tx.enabled && st.lpd1Tx.zeroYCoeffExit && lumaZero {
				comp = lpd1ComponentLuma
				if st.lpd1Tx.chromaDetectorLevel != 0 {
					comp = st.lpd1ChromaComplexity(src, refPlanes, detectMV, lumaPX, lumaPY, cbw, cbh)
				}
				performChroma = comp != lpd1ComponentLuma
			}
			if performChroma {
				uZero, vZero := true, true
				if comp == lpd1ComponentChroma || comp == lpd1ComponentChromaCb {
					uZero = st.prepareInterTXB(src.U, st.predU[:cbw*cbh], cbw, src.ChromaStride, chromaPX, chromaPY, cbw, cbh, st.uQuant, st.uQ[:cbw*cbh])
				} else {
					clear(st.uQ[:cbw*cbh])
				}
				if comp == lpd1ComponentChroma || comp == lpd1ComponentChromaCr {
					vZero = st.prepareInterTXB(src.V, st.predV[:cbw*cbh], cbw, src.ChromaStride, chromaPX, chromaPY, cbw, cbh, st.vQuant, st.vQ[:cbw*cbh])
				} else {
					clear(st.vQ[:cbw*cbh])
				}
				skip = lumaZero && uZero && vZero
			} else {
				skip = lumaZero
			}
		} else {
			skip = lumaZero
		}
		dctRdD = st.rdDcode
		if !skip {
			// Rate-priced skip (RDCOST shapes): code when distortion saved
			// outweighs the coefficient rate at the working quantizer.
			// Dropped planes contribute equal dcode/dskip terms, so their
			// omission leaves the comparison unchanged.
			rdCode := ((st.rdRcode*st.rdMult + 256) >> 9) + (st.rdDcode << 7)
			rdSkip := st.rdDskip << 7
			if rdSkip <= rdCode {
				skip = true
			}
		}
		if !skip && txPlan.Variable() {
			splitTX = true
		}
	}
	if !useRealtimeTXPlan {
		skip = false
		splitTX = false
	}

	txType = transform.TypeDCTDCT
	if !skip && !splitTX && !lpd1UseShortcuts &&
		bw == 8 && bh == 8 && st.qIndex <= 96 && videoColorIs420(st.color) {
		txType = st.chooseInter8x8TXType(src, lumaPX, lumaPY, dctRdD)
	}
	return skip, splitTX, txPlan, txType, nil
}
