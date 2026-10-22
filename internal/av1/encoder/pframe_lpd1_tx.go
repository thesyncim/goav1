package encoder

// pframe_lpd1_tx.go ports SVT-AV1's light-PD1 TX shortcut detectors, the
// mechanism preset 12 uses to stop paying for transform work on easy blocks:
//
//   - Controls: enc_mode_config.c set_lpd1_tx_ctrls(). rtc pictures are
//     forced to level 3 ("rtc forced to use lvl 3 because using the more
//     aggressive levels for only non-base pictures causes concentrated bit
//     allocation for flat rtc", enc_mode_config.c:7325-7338):
//     zero_y_coeff_exit=1, skip_tx_th=30, use_mds3_shortcuts_th=30,
//     use_neighbour_info=0. The chroma detector level, the TX bypass
//     threshold, lpd1_skip_inter_tx_level, rdoq_level and rate_est_level
//     follow the SB's light-PD1 level (pframe_lpd1_level.go
//     lpd1SBDecision).
//   - Detector: product_coding_loop.c lpd1_tx_shortcut_detector(). The MDS0
//     winner's luma residual variance (fast_loop_core_light_pd1 stores
//     fn_ptr->vf as luma_fast_dist) is compared against
//     th * bwidth * bheight * qp_index / 100 for use_tx_shortcuts_mds3 and
//     lpd1_allow_skipping_tx. use_neighbour_info=0 at level 3, so the
//     neighbour-multiplier arm is not ported.
//   - Consumption: get_perform_tx_flag (lpd1PerformTX: the skip-inter-TX
//     arms, then lpd1_bypass_tx_th), zero_y_coeff_exit +
//     chroma_complexity_check() (skip the chroma TX when luma coded all-zero
//     and chroma is not complex) and search_dct_dct_only() (skip the 8x8
//     TX-type trial, DCT_DCT only).
//
// The skip-inter-TX arm measured red on 2026-07-06 without SVT's per-SB
// level and its l0_was_skip/ref_skip_perc/me_8x8_cost_variance inputs; it
// now runs with them, as SVT does.
//
import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// lpd1TxCtrls mirrors the consumed subset of SVT-AV1's Lpd1TxCtrls
// (md_process.h). The zero value disables every shortcut. The chroma
// detector level is per SB (lpd1SBDecision.chromaDetector).
type lpd1TxCtrls struct {
	enabled bool
	// zeroYCoeffExit: skip chroma TX when the luma TXBs code all-zero
	// (subject to the chroma complexity detector).
	zeroYCoeffExit bool
	// useMds3ShortcutsTh: percentage threshold arming use_tx_shortcuts_mds3.
	useMds3ShortcutsTh uint16
	// skipTxTh: percentage threshold arming lpd1_allow_skipping_tx.
	skipTxTh uint16
}

// setLPD1TxCtrls applies SVT-AV1 set_lpd1_tx_ctrls level 3, the level rtc
// pictures are forced to at every preset that runs LPD1
// (enc_mode_config.c:6238-6247): zero_y_coeff_exit, skip_tx_th 30,
// use_mds3_shortcuts_th 30, use_neighbour_info 0. skip_nrst_nrst_luma_tx
// only applies to NEAREST_NEARESTMV, a compound mode the light path here
// never codes.
func (st *lossyEncodeState) setLPD1TxCtrls(enabled bool) {
	if !enabled {
		st.lpd1Tx = lpd1TxCtrls{}
		return
	}
	st.lpd1Tx = lpd1TxCtrls{
		enabled:            true,
		zeroYCoeffExit:     true,
		useMds3ShortcutsTh: 30,
		skipTxTh:           30,
	}
}

// lpd1TxDetect is lpd1_tx_shortcut_detector (product_coding_loop.c): the
// winner's luma residual variance against th_normalizer = bheight * bwidth
// * qp_index, where qp_index is the SB qindex (md_process.c
// svt_aom_mode_decision_configure_sb). It returns use_tx_shortcuts_mds3 and
// lpd1_allow_skipping_tx. use_neighbour_info is 0 at level 3, so the
// neighbour-multiplier refinement is not ported.
func (st *lossyEncodeState) lpd1TxDetect(variance uint32, bw, bh int) (useShortcuts, allowSkipTX bool) {
	if !st.lpd1Tx.enabled {
		return false, false
	}
	thNormalizer := uint64(bw) * uint64(bh) * uint64(st.quantQ)
	dist := 100 * uint64(variance)
	return dist < uint64(st.lpd1Tx.useMds3ShortcutsTh)*thNormalizer,
		dist < uint64(st.lpd1Tx.skipTxTh)*thNormalizer
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
// when luma coded all-zero), get_perform_tx_flag (drop the luma TX) and
// search_dct_dct_only (skip the 8x8 TX-type trial, DCT_DCT only). The
// light path codes tx_depth 0 with SVT's coefficient pricing; RDOQ follows
// the SB's rdoq_level.
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
	st.rdoqBlock = false
	useRealtimeTXPlan := videoColorIs420(st.color)
	lumaBypass := false
	lpd1UseShortcuts := false
	lightPath := false
	var dec *lpd1SBDecision
	if useRealtimeTXPlan {
		dec = st.lpd1SB(lumaPX, lumaPY)
		sse, variance, ok := uint32(0), uint32(0), false
		if st.predYKey.valid {
			sse, variance, ok = st.predStats.lookup(st.predYKey.mv, st.predYKey.filters)
		}
		if !ok {
			sse, variance = realtimeInterResidualSSEVariance(src.Y, st.predY[:bw*bh], src.YStride, bw, lumaPX, lumaPY, bw, bh)
		}
		// The TX detectors and the bypass read the MDS0 winner's
		// regular-filter statistics (luma_fast_dist, full_dist = its vf sse),
		// not the IFS-chosen prediction's.
		mdsSSE, mdsVar := sse, variance
		if st.mds0BestRate >= 0 {
			mdsSSE, mdsVar = st.mds0BestSSE, uint32(st.mds0BestVar)
		}
		allowSkipTX := false
		if st.lpd1Tx.enabled {
			lpd1UseShortcuts, allowSkipTX = st.lpd1TxDetect(mdsVar, bw, bh)
		}
		txLevel := realtimeTXSizeLevelBasedOnQstep(src.Width, src.Height, st.effortLevel)
		// The light-PD1 TX path (tx_depth 0, SVT coefficient pricing, the TX
		// bypass) needs the RDOQ rate tables; RDOQ itself follows the SB's
		// rdoq_level.
		lightPath = st.armRDOQ()
		st.rdoqBlock = lightPath && dec.rdoq
		if leaf, ok := svtDepth0TXLeaf(block.Size); lightPath && ok {
			txPlan, err = realtimeInterTXPlanForLeafSize(block.Size, leaf)
		} else {
			txPlan, err = realtimeInterTXPlanForBlock(block.Size, st.qIndex, st.yQuant.AC, txLevel, sse, variance)
		}
		if err != nil {
			return false, false, txPlan, transform.TypeDCTDCT, err
		}
		// get_perform_tx_flag: perform_tx = 0 skips only the luma transform;
		// chroma still runs (full_loop_core_light_pd1).
		lumaBypass = lightPath && !skip && !st.lpd1PerformTX(dec, allowSkipTX, mdsSSE, bw, bh)
	}
	st.mds0BestRate = -1
	dctRdD := int64(0)
	if !skip {
		st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
		st.setRDOQArm(0)
		lumaZero := true
		if lumaBypass {
			if txPlan.Variable() {
				clear(st.lumaQ2[:bw*bh])
			} else {
				clear(st.lumaQ[:bw*bh])
			}
		} else if txPlan.Variable() {
			leafArea := txPlan.leafSize * txPlan.leafSize
			if lerr := txPlan.ForEachLeaf(func(i, dx, dy int) error {
				if !st.prepareInterTXBDecide(src.Y, st.predY[dy*bw+dx:], bw, src.YStride, lumaPX+dx, lumaPY+dy, txPlan.leafSize, txPlan.leafSize, st.yQuant, st.lumaQ2[i*leafArea:(i+1)*leafArea]) {
					lumaZero = false
				}
				return nil
			}); lerr != nil {
				return false, false, txPlan, transform.TypeDCTDCT, fmt.Errorf("prepare realtime luma tx: %w", lerr)
			}
		} else if bw == 8 && bh == 8 {
			// The 8x8 TX-type trial (chooseInter8x8TXType) prices its
			// alternatives against this block's absolute DCT distortion, so
			// the luma terms stay complete here.
			lumaZero = st.prepareInterTXB(src.Y, st.predY[:bw*bh], bw, src.YStride, lumaPX, lumaPY, bw, bh, st.yQuant, st.lumaQ[:bw*bh])
		} else {
			lumaZero = st.prepareInterTXBDecide(src.Y, st.predY[:bw*bh], bw, src.YStride, lumaPX, lumaPY, bw, bh, st.yQuant, st.lumaQ[:bw*bh])
		}
		if hasChroma {
			// zero_y_coeff_exit: an all-zero luma block skips the chroma TX
			// unless the chroma complexity detector flags a plane
			// (full_loop_core_light_pd1's perform_chroma derivation).
			performChroma := true
			comp := lpd1ComponentChroma
			if useRealtimeTXPlan && st.lpd1Tx.enabled && st.lpd1Tx.zeroYCoeffExit && lumaZero {
				comp = lpd1ComponentLuma
				if dec.chromaDetector != 0 {
					comp = st.lpd1ChromaComplexity(src, refPlanes, detectMV, lumaPX, lumaPY, cbw, cbh)
				}
				performChroma = comp != lpd1ComponentLuma
			}
			st.setRDOQArm(1)
			if performChroma {
				uZero, vZero := true, true
				if comp == lpd1ComponentChroma || comp == lpd1ComponentChromaCb {
					uZero = st.prepareInterTXBDecide(src.U, st.predU[:cbw*cbh], cbw, src.ChromaStride, chromaPX, chromaPY, cbw, cbh, st.uQuant, st.uQ[:cbw*cbh])
				} else {
					clear(st.uQ[:cbw*cbh])
				}
				if comp == lpd1ComponentChroma || comp == lpd1ComponentChromaCr {
					vZero = st.prepareInterTXBDecide(src.V, st.predV[:cbw*cbh], cbw, src.ChromaStride, chromaPX, chromaPY, cbw, cbh, st.vQuant, st.vQ[:cbw*cbh])
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
		st.rdoqArm = 0
		dctRdD = st.rdDcode
		if !skip {
			// Rate-priced skip (RDCOST shapes): code when distortion saved
			// outweighs the coefficient rate at the working quantizer.
			// Dropped planes contribute equal dcode/dskip terms, so their
			// omission leaves the comparison unchanged. The rate is the real
			// coefficient coder's cost of every TXB, txb_skip and EOB
			// symbols included, as SVT-AV1's full loop prices the coded
			// alternative (full_loop.c svt_aom_txb_estimate_coeff_bits);
			// the per-coefficient estimate remains only for layouts the
			// trial counters do not cover.
			rate := (st.rdRcode*st.rdMult + 256) >> 9
			if lightPath {
				// svt_aom_full_cost prices at full_lambda.
				rate = (st.rdRcode*st.lpd1FullLambda(dec) + 256) >> 9
				lumaTX, _ := st.interLumaTXAndScan(bw, bh)
				chromaTX := tile.TransformSize4x4
				if hasChroma {
					chromaTX, _ = tile.MaxTransformSize(block.Size, st.color, 1)
				}
				if r, ok := st.svtInterCoeffCost(txPlan, lumaTX, chromaTX, hasChroma, dec.coeffRateLvl0, st.lpd1FullLambda(dec)); ok {
					rate = r
				}
			} else if r, ok := st.exactInterCoeffCost(txPlan, bw, bh, cbw, cbh, hasChroma, lumaPX, lumaPY); ok {
				rate = r
			}
			rdCode := rate + (st.rdDcode << 7)
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
	st.lpd1NoteBlock(lumaPX, lumaPY, bw, bh, false, !skip)
	return skip, splitTX, txPlan, txType, nil
}

// exactInterCoeffCost prices the block's quantized luma and chroma
// coefficients with the real coefficient coder against the frame-frozen
// trial CDFs, lambda-scaled like the RDCOST rate term. ok=false leaves the
// caller on its per-coefficient estimate (rectangular or 64-wide layouts).
//
// The pricing contexts restart from the frame-start defaults at every
// superblock: adaptation inside a superblock follows its fixed block order,
// so the serial, split and wavefront walks price identically.
func (st *lossyEncodeState) exactInterCoeffCost(txPlan realtimeInterTXPlan, bw, bh, cbw, cbh int, hasChroma bool, lumaPX, lumaPY int) (int64, bool) {
	if !st.skipRateReady {
		if err := st.skipRateBase.InitDefault(st.qIndex); err != nil {
			return 0, false
		}
		st.skipRateCDFs = st.skipRateBase
		st.skipRateDirty = 0
		st.skipRateReady = true
		st.skipRateSB = -1
	}
	if sb := (lumaPY>>6)<<16 | lumaPX>>6; sb != st.skipRateSB {
		st.restoreSkipRateCDFs()
		st.skipRateSB = sb
	}
	if hasChroma && (cbw != cbh || cbw > 32) {
		return 0, false
	}
	var bits int
	if txPlan.Variable() {
		leaf := txPlan.leafSize
		area := leaf * leaf
		if err := txPlan.ForEachLeaf(func(i, dx, dy int) error {
			bits += st.skipRateTXBBits(true, st.lumaQ2[i*area:(i+1)*area], leaf)
			return nil
		}); err != nil {
			return 0, false
		}
	} else {
		if bw != bh || bw > 32 {
			return 0, false
		}
		bits += st.skipRateTXBBits(true, st.lumaQ[:bw*bh], bw)
	}
	if hasChroma {
		bits += st.skipRateTXBBits(false, st.uQ[:cbw*cbh], cbw)
		bits += st.skipRateTXBBits(false, st.vQ[:cbw*cbh], cbw)
	}
	if bits < 0 {
		return 0, false
	}
	return st.txbBitsToRate(bits), true
}

// skipRateTXBBits counts one square TXB against the skip-pricing contexts;
// -1 marks an unsupported size.
func (st *lossyEncodeState) skipRateTXBBits(luma bool, q []int16, n int) int {
	c := &st.skipRateCDFs
	plane := tile.CoeffPlaneUV
	if luma {
		plane = tile.CoeffPlaneY
	}
	txCtx := -1
	switch n {
	case 4:
		txCtx = 0
	case 8:
		txCtx = 1
	case 16:
		txCtx = 2
	case 32:
		txCtx = 3
	}
	if txCtx >= 0 {
		st.skipRateDirty |= 1 << (txCtx*tile.CoeffPlaneTypes + int(plane))
	}
	var bits int
	switch {
	case luma && n == 8:
		_, bits = tile.CountCoefficientsTXB8x8Y2DTrustedArray(c, (*[64]int16)(q), nil, 0)
	case luma && n == 16:
		_, bits = tile.CountCoefficientsTXB16x16Y2DTrustedArray(c, (*[256]int16)(q))
	case luma && n == 32:
		_, bits = tile.CountCoefficientsTXB32x32Y2DTrustedArray(c, (*[1024]int16)(q))
	case !luma && n == 4:
		_, bits = tile.CountCoefficientsTXB4x4UV2DTrustedArray(c, (*[16]int16)(q))
	case !luma && n == 8:
		_, bits = tile.CountCoefficientsTXB8x8UV2DTrustedArray(c, (*[64]int16)(q))
	case !luma && n == 16:
		_, bits = tile.CountCoefficientsTXB16x16UV2DTrustedArray(c, (*[256]int16)(q))
	case !luma && n == 32:
		_, bits = tile.CountCoefficientsTXB32x32UV2DTrustedArray(c, (*[1024]int16)(q))
	default:
		return -1 << 30
	}
	return bits
}

// restoreSkipRateCDFs returns the coefficient families the last superblock's
// pricing touched to their frame-start state. A (tx-size context, plane)
// pair's counters touch only that pair's TXB-skip, EOB, base, BR and DC-sign
// tables, so copying those rows equals restoring the whole set.
func (st *lossyEncodeState) restoreSkipRateCDFs() {
	d := st.skipRateDirty
	if d == 0 {
		return
	}
	c, b := &st.skipRateCDFs, &st.skipRateBase
	planeDirty := [tile.CoeffPlaneTypes]bool{}
	for tx := 0; tx < tile.CoeffTxSizeContexts; tx++ {
		for p := 0; p < tile.CoeffPlaneTypes; p++ {
			if d&(1<<(tx*tile.CoeffPlaneTypes+p)) == 0 {
				continue
			}
			planeDirty[p] = true
			c.TXBSkip[tx] = b.TXBSkip[tx]
			c.EOBExtra[tx][p] = b.EOBExtra[tx][p]
			c.CoeffBR[min(tx, 3)][p] = b.CoeffBR[min(tx, 3)][p]
			c.CoeffBase[tx][p] = b.CoeffBase[tx][p]
			c.CoeffBaseEOB[tx][p] = b.CoeffBaseEOB[tx][p]
		}
	}
	for p, dirty := range planeDirty {
		if !dirty {
			continue
		}
		c.DCSign[p] = b.DCSign[p]
		c.EOBFlag16[p] = b.EOBFlag16[p]
		c.EOBFlag32[p] = b.EOBFlag32[p]
		c.EOBFlag64[p] = b.EOBFlag64[p]
		c.EOBFlag128[p] = b.EOBFlag128[p]
		c.EOBFlag256[p] = b.EOBFlag256[p]
		c.EOBFlag512[p] = b.EOBFlag512[p]
		c.EOBFlag1024[p] = b.EOBFlag1024[p]
	}
	st.skipRateDirty = 0
}

// lpd1InitBitEst is product_coding_loop.c INIT_BIT_EST.
const lpd1InitBitEst = 6000

// lpd1BypassTX is get_perform_tx_flag's lpd1_bypass_tx_th test: the block
// skips the transform when the estimated skip cost of the MDS0 winner (its
// luma rate plus the skip flag, its prediction SSE) is under th
// (lpd1_bypass_tx_th, the SB's light-PD1 level; see pframe_lpd1_level.go)
// percent of a one-unit-per-pixel coded cost, at SVT's full_lambda
// (lpd1FullLambda); skip_coeff_ctx is 0 on the light-PD1
// path (update_skip_coeff_ctx = 0) with frame-frozen skip_fac_bits.
func (st *lossyEncodeState) lpd1BypassTX(sse uint32, bw, bh int, th, lambda int64) bool {
	rate := st.mds0BestRate
	if rate < 0 || th == 0 {
		return false
	}
	// fast_luma_rate carries the is_inter flag (av1_inter_fast_cost_light).
	rate += st.lpd1IsInterBits
	skip0, skip1 := st.lpd1SkipBits[0], st.lpd1SkipBits[1]
	est := rdoqCostRD(lambda, rate+skip1, int64(sse)<<4)
	thCost := rdoqCostRD(lambda, rate+skip0+lpd1InitBitEst, int64(bw*bh)<<4)
	return est*100 < th*thCost
}

// lpd1PerformTX is get_perform_tx_flag (product_coding_loop.c) for the
// single-reference inter candidate the light path codes: with
// lpd1_allow_skipping_tx, lpd1_skip_inter_tx_level 2 drops the luma TX of
// every inter candidate and level 1 drops it when the left and above
// blocks are available and both coded skip; otherwise lpd1_bypass_tx_th
// decides. skip_nrst_nrst_luma_tx targets NEAREST_NEARESTMV only.
func (st *lossyEncodeState) lpd1PerformTX(dec *lpd1SBDecision, allowSkipTX bool, sse uint32, bw, bh int) bool {
	if allowSkipTX {
		if dec.skipInterTxLevel == 2 {
			return false
		}
		if st.lpd1NbrSkipCtx == 2 && dec.skipInterTxLevel == 1 {
			return false
		}
	}
	return !st.lpd1BypassTX(sse, bw, bh, dec.bypassTh, st.lpd1FullLambda(dec))
}
