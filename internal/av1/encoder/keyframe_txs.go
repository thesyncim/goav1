package encoder

import (
	"sync"

	"github.com/thesyncim/goav1/internal/av1/quantize"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// keyframe_txs.go ports SVT-AV1's I-slice transform-size search at the rtc
// presets (v4.1.0). svt_aom_sig_deriv_mode_decision_config_rtc sets txs_level
// is_islice ? 4 : 0 above M8 (the QP banding that lowers it needs a
// configured qp above 58, which CBR never has), and the frame header then
// signals TX_MODE_SELECT. set_txs_controls level 4:
//
//   - intra_class_max_depth_sq/nsq 1: a block tests its largest transform
//     (depth 0) and the four quarter-size transforms (depth 1);
//     get_start_end_tx_depth ends at depth 0 for a block that crosses the
//     8-aligned picture.
//   - prev_depth_coeff_exit_th 2: depth 1 runs only when depth 0 coded at
//     least two coefficients (perform_tx_partitioning, best_coeff_count).
//   - quadrant_th_sf 100: after each depth-1 transform block, the depth is
//     abandoned when its cost so far (tx-size bits included) exceeds the
//     proportional share of the depth-0 cost.
//   - The txt search is off above M11 (txt_level 0), so every transform is
//     DCT_DCT; the depth-1 group offsets do not apply.
//
// md_stage_3 runs the search with spatial SSE (spatial_sse_full_loop_level 3
// is SSSE_MDS3, and an intra depth > 0 always reconstructs), the full
// coefficient rate (rate_est_level 1: coeff_rate_est_lvl 1 with the real
// txb_skip and dc_sign contexts, the luma tx_type rate included) and
// RDCOST(full_lambda, coeff bits + tx-size bits, SSE << 4). Each depth-1
// transform block is predicted from the reconstruction of the ones before
// it. The rate tables are SVT's frame-start ones; a keyframe starts from the
// default CDFs.
//
// Scope: camera-content 8-bit 4:2:0 keyframes, whose leaves code DC_PRED only
// (keySelectIntraMode). Screen-content keyframes keep TX_MODE_LARGEST.

// keyTXSEnabled reports whether a keyframe tile of this configuration runs the
// transform-size search and codes TX_MODE_SELECT.
func keyTXSEnabled(allowScreenContentTools bool, color parser.ColorConfig) bool {
	return !allowScreenContentTools && videoColorIs420(color) && color.BitDepth == 8
}

// keyTXSScratch is the depth search's lane-owned scratch.
type keyTXSScratch struct {
	recon0 [64 * 64]byte
	q0     [32 * 32]int16
	q1     [4 * 32 * 32]int16
	pred   [32 * 32]byte
	above  [tile.MaxBlockModeSlots]uint8
	left   [tile.MaxBlockModeSlots]uint8
}

// keyTXSRateTable holds the frame-start tx-size and luma tx_type rates.
type keyTXSRateTable struct {
	ok bool
	// txSize[category][ctx][depth] is cost_selected_tx_size's rate.
	txSize [4][3][3]int64
	// txType[log2(n)-2] is the DCT_DCT intra tx_type rate of an n x n
	// DC_PRED transform (zero where the set codes no symbol).
	txType [5]int64
}

var (
	keyTXSRatesOnce sync.Once
	keyTXSRates     keyTXSRateTable
)

func keyTXSRateTables() *keyTXSRateTable {
	keyTXSRatesOnce.Do(func() {
		var sz tile.TransformCDFs
		if sz.InitDefault() != nil {
			return
		}
		t := &keyTXSRates
		for cat := range 4 {
			for ctx := range 3 {
				cdf, err := sz.TxSizeCDF(cat, ctx)
				if err != nil {
					return
				}
				for d := range min(cdf.Symbols(), 3) {
					t.txSize[cat][ctx][d] = int64(entropy.CostSymbol(cdf, d))
				}
			}
		}
		var ty tile.TransformTypeCDFs
		if ty.InitDefault() != nil {
			return
		}
		for i, size := range [...]tile.TransformSize{tile.TransformSize4x4, tile.TransformSize8x8, tile.TransformSize16x16, tile.TransformSize32x32, tile.TransformSize64x64} {
			set, err := tile.ExtTXSetTypeFor(size, false, false)
			if err != nil {
				return
			}
			symbols, err := tile.ExtTXTypeCount(set)
			if err != nil {
				return
			}
			if symbols <= 1 {
				continue
			}
			index, err := tile.ExtTXSetIndex(size, false, false)
			if err != nil {
				return
			}
			square, err := tile.TransformSizeSquare(size)
			if err != nil {
				return
			}
			cdf, err := ty.IntraCDF(index, square, tile.IntraModeDC, symbols)
			if err != nil {
				return
			}
			symbol, err := tile.ExtTXSymbolForType(set, transform.TypeDCTDCT)
			if err != nil {
				return
			}
			t.txType[i] = int64(entropy.CostSymbol(cdf, symbol))
		}
		t.ok = true
	})
	return &keyTXSRates
}

// keyTXSCoeffCost is svt_av1_cost_coeffs_txb for one n x n DCT_DCT luma TXB at
// its real txb_skip and dc_sign contexts, the tx_type rate included, over the
// frame-start rate tables.
func (st *lossyEncodeState) keyTXSCoeffCost(qcoeff []int16, size tile.TransformSize, n int, ctx tile.TXBContext, eob int) int64 {
	g := rdoqGeometry(size, 0)
	tc := &st.keyTXSCoeffTab.coeff[g.txCtx][0]
	if eob == 0 {
		return int64(tc.txbSkip[ctx.TXBSkipContext][1])
	}
	cost := int64(tc.txbSkip[ctx.TXBSkipContext][0])
	cost += keyTXSRateTables().txType[keyTXSLog2(n)]
	cost += int64(g.eobCost(eob, &st.keyTXSCoeffTab.eob[g.eobSel][0], tc))
	scan := g.scan
	levels := st.levels
	_ = tile.CoeffInitLevels(qcoeff, g.size, levels)
	for c := eob - 1; c >= 0; c-- {
		pos := int(scan[c])
		v := int32(qcoeff[pos])
		level := rdoqAbs(v)
		if c == eob-1 {
			cost += int64(tc.baseEOB[g.lowerLevelsCtxEOB(c)][min(level, 3)-1])
		} else {
			cost += int64(tc.base[g.lowerLevelsCtx(levels, pos)][min(level, 3)])
		}
		if v == 0 {
			continue
		}
		if c == 0 {
			sign := 0
			if v < 0 {
				sign = 1
			}
			cost += int64(tc.dcSign[ctx.DCSignContext][sign])
		} else {
			cost += int64(rdoqCostLiteral(1))
		}
		if level > rdoqNumBaseLevels {
			brCtx := 0
			if eob != 1 {
				brCtx = g.brCtx(levels, pos)
			}
			cost += int64(rdoqBRCost(level, &tc.lps[brCtx]))
		}
	}
	return cost
}

func keyTXSLog2(n int) int {
	switch n {
	case 4:
		return 0
	case 8:
		return 1
	case 16:
		return 2
	case 32:
		return 3
	}
	return 4
}

// keyTXSSSE is the spatial SSE of an n x n block of src against recon.
func keyTXSSSE(src, recon []byte, stride, px, py, n int) int64 {
	off := py*stride + px
	a, b := src[off:], recon[off:]
	var sse uint32
	switch n {
	case 4:
		sse, _ = sseVariance4x4(a, stride, b, stride)
	case 8:
		sse, _ = sseVariance8x8(a, stride, b, stride)
	case 16:
		sse, _ = sseVariance16x16(a, stride, b, stride)
	case 32:
		sse, _ = sseVariance32x32(a, stride, b, stride)
	default:
		sse, _ = sseVariance64x64(a, stride, b, stride)
	}
	return int64(sse)
}

// keyTXSCost is RDCOST(full_lambda, rate, dist): rate in 1/512 bits, dist the
// spatial SSE (svt_spatial_full_distortion_kernel << 4).
func (st *lossyEncodeState) keyTXSCost(rate, sse int64) int64 {
	return ((rate*st.keyTXSLambda + 256) >> entropy.ProbCostShift) + (sse<<4)<<7
}

// keyLumaTXS codes a DC_PRED keyframe leaf's luma with SVT's transform-size
// search (perform_tx_partitioning): it reconstructs the winner into recon,
// appends its quantized coefficients to rec in coding order, marks the decision
// pass's luma coefficient contexts and returns the chosen transform size.
// pred holds the block's DC prediction.
func (st *lossyEncodeState) keyLumaTXS(src SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, scratch *tile.BlockLoopScratch, rec *pframeSplitRecord, pred []byte, n int, lumaTX tile.TransformSize) (tile.TransformSize, error) {
	return st.keyLuma(src, recon, block, scratch, rec, pred, n, lumaTX, true)
}

// keyLumaDepth0 codes a keyframe leaf's luma at its largest transform, with
// the decision pass's coefficient contexts marked as keyLumaTXS does.
func (st *lossyEncodeState) keyLumaDepth0(src SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, scratch *tile.BlockLoopScratch, rec *pframeSplitRecord, pred []byte, n int, lumaTX tile.TransformSize) error {
	_, err := st.keyLuma(src, recon, block, scratch, rec, pred, n, lumaTX, false)
	return err
}

func (st *lossyEncodeState) keyLuma(src SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, scratch *tile.BlockLoopScratch, rec *pframeSplitRecord, pred []byte, n int, lumaTX tile.TransformSize, search bool) (tile.TransformSize, error) {
	stride := src.YStride
	px, py := int(block.MICol)*4, int(block.MIRow)*4
	coeffCtx := &scratch.CoeffCtx
	q := st.yQuant
	ctxReq := tile.CoeffContextRequest{Plane: 0, PlaneBlock: block.Size, Size: lumaTX, X4: block.X4, Y4: block.Y4}
	blockDims, _ := block.Size.Dimensions()
	txDims, _ := lumaTX.Dimensions()

	// Depth 0: the block's largest transform.
	ctx0 := coeffCtx.TXBContextTrusted(ctxReq, txDims, blockDims)
	st.keyRDOQPlane, st.keyRDOQCtx = 0, ctx0
	geo0, _, err := st.prepareIntraTXB(src.Y, stride, px, py, n, n, q, lumaTX, pred, st.lumaQ[:])
	if err != nil {
		return 0, err
	}
	scan0, _ := st.scanForTransformSize(lumaTX)
	cn0 := geo0.coeffCount
	res0 := interTXBResult(st.lumaQ[:cn0], scan0, cn0)
	if !res0.AllZero {
		res0.CulLevel = tile.CoeffCulLevel(st.lumaQ[:cn0], scan0, int(res0.EOB))
	}
	if err := st.reconInterTXBResult(recon.Y, pred, n, stride, px, py, n, n, q, st.lumaQ[:cn0], geo0, scan0, transform.TypeDCTDCT, res0); err != nil {
		return 0, err
	}
	// get_start_end_tx_depth: no search for a block crossing the picture, and
	// prev_depth_coeff_exit_th 2.
	st.keyLumaEOB = int(res0.EOB)
	if !search || px+n > src.Width || py+n > src.Height || n < 8 || res0.EOB < 2 {
		copy(rec.grabCoeffs(cn0), st.lumaQ[:cn0])
		return lumaTX, coeffCtx.MarkTXB(ctxReq, res0)
	}
	rates := keyTXSRateTables()
	maxDims, _ := lumaTX.Dimensions()
	category := int(maxDims.Max) - 1
	sizeCtx, err := scratch.Mode.SelectedTransformContextWithAvailability(lumaTX, int(block.X4), int(block.Y4), block.HaveTop, block.HaveLeft)
	if err != nil || !rates.ok {
		copy(rec.grabCoeffs(cn0), st.lumaQ[:cn0])
		return lumaTX, coeffCtx.MarkTXB(ctxReq, res0)
	}
	rate0 := st.keyTXSCoeffCost(st.lumaQ[:cn0], lumaTX, n, ctx0, int(res0.EOB)) + rates.txSize[category][sizeCtx][0]
	cost0 := st.keyTXSCost(rate0, keyTXSSSE(src.Y, recon.Y, stride, px, py, n))

	s := st.keyTXS
	for r := range n {
		copy(s.recon0[r*n:(r+1)*n], recon.Y[(py+r)*stride+px:])
	}
	copy(s.q0[:cn0], st.lumaQ[:cn0])
	s.above = coeffCtx.Above[0]
	s.left = coeffCtx.Left[0]

	// Depth 1: four quarter-size transforms in raster order, each predicted
	// from the reconstruction so far.
	sub := n / 2
	subTX := txSizeSquare(sub)
	subDims, _ := subTX.Dimensions()
	subGeo, err := txBlockGeometryForTransformSize(subTX)
	if err != nil {
		return 0, err
	}
	subScan, _ := st.scanForTransformSize(subTX)
	cs := subGeo.coeffCount
	var bits, dist int64
	won := true
	eobSum := 0
	for i := range 4 {
		dx, dy := (i&1)*sub, (i>>1)*sub
		haveTop := block.HaveTop || dy > 0
		haveLeft := block.HaveLeft || dx > 0
		spred := s.pred[:sub*sub]
		dc := dcPredictN(recon.Y, stride, px+dx, py+dy, sub, haveTop, haveLeft)
		if i == 0 {
			st.keyLumaSubDC = dc
		}
		memsetBytes(spred, dc)
		qc := s.q1[i*cs : (i+1)*cs]
		subReq := tile.CoeffContextRequest{Plane: 0, PlaneBlock: block.Size, Size: subTX, X4: block.X4 + uint8(dx/4), Y4: block.Y4 + uint8(dy/4)}
		ctx := coeffCtx.TXBContextTrusted(subReq, subDims, blockDims)
		st.keyRDOQCtx = ctx
		if _, _, err := st.prepareIntraTXB(src.Y, stride, px+dx, py+dy, sub, sub, q, subTX, spred, qc); err != nil {
			return 0, err
		}
		res := interTXBResult(qc, subScan, cs)
		eobSum += int(res.EOB)
		if !res.AllZero {
			res.CulLevel = tile.CoeffCulLevel(qc, subScan, int(res.EOB))
		}
		if err := st.reconInterTXBResult(recon.Y, spred, sub, stride, px+dx, py+dy, sub, sub, q, qc, subGeo, subScan, transform.TypeDCTDCT, res); err != nil {
			return 0, err
		}
		bits += st.keyTXSCoeffCost(qc, subTX, sub, ctx, int(res.EOB))
		dist += keyTXSSSE(src.Y, recon.Y, stride, px+dx, py+dy, sub)
		if err := coeffCtx.MarkTXB(subReq, res); err != nil {
			return 0, err
		}
		if st.keyTXSCost(bits, dist) > cost0 {
			won = false
			break
		}
		// quadrant_th_sf 100.
		if st.keyTXSCost(bits+rates.txSize[category][sizeCtx][1], dist) > (int64(i+1)*cost0)/4 {
			won = false
			break
		}
	}
	if won && st.keyTXSCost(bits+rates.txSize[category][sizeCtx][1], dist) < cost0 {
		copy(rec.grabCoeffs(4*cs), s.q1[:4*cs])
		st.keyLumaEOB = eobSum
		return subTX, nil
	}
	// Depth 0 stays: restore its reconstruction and contexts.
	for r := range n {
		copy(recon.Y[(py+r)*stride+px:(py+r)*stride+px+n], s.recon0[r*n:(r+1)*n])
	}
	coeffCtx.Above[0] = s.above
	coeffCtx.Left[0] = s.left
	copy(rec.grabCoeffs(cn0), s.q0[:cn0])
	return lumaTX, coeffCtx.MarkTXB(ctxReq, res0)
}

// txSizeSquare is the n x n transform size.
func txSizeSquare(n int) tile.TransformSize {
	switch n {
	case 4:
		return tile.TransformSize4x4
	case 8:
		return tile.TransformSize8x8
	case 16:
		return tile.TransformSize16x16
	case 32:
		return tile.TransformSize32x32
	}
	return tile.TransformSize64x64
}

// keyTXSCoeffTable is the coefficient rate table of the default CDFs at
// qIndex (a keyframe's frame-start tables).
func keyTXSCoeffTable(qIndex uint8) *rdoqCosts {
	t := &rdoqDefaultTables[tile.CoeffQContext(qIndex)]
	t.once.Do(func() { t.costs.fill(tile.DefaultCoeffCDFs(qIndex)) })
	return &t.costs
}

// keyTXShortcut is tx_shortcut_detector for a keyframe leaf. SVT-AV1's rtc
// presets run tx_shortcut_level 3 above M9 (M11 flat) on every picture
// (use_mds3_shortcuts_th 10, apply_pf_on_coeffs, chroma_detector_level 1). A
// keyframe leaf has one candidate, so MDS1 is skipped (enable_skipping_mds1
// at nic levels 8 and up) and md_stage_0's variance of the DC_PRED
// prediction decides whether MDS3 takes the shortcuts:
// 100 * luma_fast_dist < 10 * bw * bh * qindex. A shortcut block codes its
// luma transforms, at both tx depths, with the N4 partial-frequency shape
// (only the low quarter of each dimension).
func keyTXShortcut(src []byte, stride int, pred []byte, px, py, n int, qIndex uint8) bool {
	a := src[py*stride+px:]
	var v uint32
	switch n {
	case 8:
		_, v = sseVariance8x8(a, stride, pred, n)
	case 16:
		_, v = sseVariance16x16(a, stride, pred, n)
	case 32:
		_, v = sseVariance32x32(a, stride, pred, n)
	case 64:
		_, v = sseVariance64x64(a, stride, pred, n)
	default:
		return false
	}
	return 100*uint64(v) < 10*uint64(n*n)*uint64(qIndex)
}

// keyZero is the all-zero reference of svt_aom_eb_av1_var_offs.
var keyZero [32 * 32]byte

// keyChromaN4 reports whether a keyframe leaf codes its chroma with the N4
// shape: svt_aom_full_loop_uv takes it when chroma_complexity_check_pred
// (use_var) leaves the block at COMPONENT_LUMA and either the block took the
// tx shortcut or its luma coded fewer than (w_uv >> 4) * (h_uv >> 4)
// coefficients (apply_pf_on_coeffs; cnt_nz_coeff is the luma EOB sum). The
// detector compares the luma and chroma prediction SADs over the chroma
// block's extent, every 2^shift-th row, and the chroma source variances
// against 150. lumaPred is the winning luma prediction (n x n), or nil for a
// split block, whose top-left quarter predicts st.keyLumaSubDC.
func (st *lossyEncodeState) keyChromaN4(src SourceFrame420, g *keyChromaGeom, lumaPred []byte, n, px, py int, dcU, dcV uint8, shortcut bool) bool {
	cw, ch := g.cw, g.ch
	if !shortcut && !(st.keyLumaEOB < (cw>>4)*(ch>>4) || st.keyLumaEOB == 0) {
		return false
	}
	shift := 0
	if ch > 8 {
		shift = 2
	} else if ch > 4 {
		shift = 1
	}
	var yDist, cbDist, crDist int
	for r := 0; r < ch>>shift; r++ {
		yr := (py + r<<shift) * src.YStride
		cr := (g.y + r<<shift) * src.ChromaStride
		for c := range cw {
			p := int(st.keyLumaSubDC)
			if lumaPred != nil {
				p = int(lumaPred[(r<<shift)*n+c])
			}
			yDist += absInt(int(src.Y[yr+px+c]) - p)
			cbDist += absInt(int(src.U[cr+g.x+c]) - int(dcU))
			crDist += absInt(int(src.V[cr+g.x+c]) - int(dcV))
		}
	}
	yDist <<= 1
	if cbDist > yDist || crDist > yDist {
		return false
	}
	const th = 150
	off := g.y*src.ChromaStride + g.x
	return keyBlockVar(src.U[off:], src.ChromaStride, cw, ch) <= th && keyBlockVar(src.V[off:], src.ChromaStride, cw, ch) <= th
}

// keyBlockVar is ROUND_POWER_OF_TWO(variance, log2(w * h)) of a w x h source
// block (the variance kernel against the zero offsets).
func keyBlockVar(a []byte, stride, w, h int) int {
	var v uint32
	switch {
	case w == 4 && h == 4:
		_, v = sseVariance4x4(a, stride, keyZero[:], w)
	case w == 8 && h == 8:
		_, v = sseVariance8x8(a, stride, keyZero[:], w)
	case w == 16 && h == 16:
		_, v = sseVariance16x16(a, stride, keyZero[:], w)
	default:
		_, v = sseVariance32x32(a, stride, keyZero[:], w)
	}
	log2 := 0
	for 1<<log2 < w*h {
		log2++
	}
	return int((v + 1<<(log2-1)) >> log2)
}

// keyRDOQQuantize is svt_aom_quantize_inv_quantize with perform_rdoq for a
// keyframe TXB. SVT-AV1's rtc presets set rdoq_level is_islice ? 1 : 0
// above M8 (M9 flat). Level 1 runs the full optimize_b on every I-slice TXB:
// luma and chroma (skip_uv 0), every tx type, eob_th and eob_fast_th 255,
// no cut-off. It starts from the fp quantizer and prices with the frame-start
// rates at the TXB's real txb_skip / dc_sign contexts, at
// full_lambda * plane_rd_mult[rtc][intra] (17 luma, 13 chroma) >> 2.
func (st *lossyEncodeState) keyRDOQQuantize(tran []int32, qcoeff []int16, size tile.TransformSize, q quantize.Quantizer) {
	plane := st.keyRDOQPlane
	g := rdoqGeometry(size, plane)
	if err := quantize.QuantizeBlockScaledFP(qcoeff[:g.area], g.ch, tran[:g.area], g.ch, g.cw, g.ch, q, uint8(g.shift)); err != nil {
		return
	}
	eob := rdoqEOBFromIScan(qcoeff[:g.area], g.iscan)
	if eob == 0 {
		return
	}
	tab := st.keyTXSCoeffTab
	s := rdoqState{
		g:       g,
		tc:      &tab.coeff[g.txCtx][plane],
		ec:      &tab.eob[g.eobSel][plane],
		rdmult:  (st.keyTXSLambda*rdoqPlaneRDMult[plane] + 2) >> 2,
		scan:    g.scan,
		tcoeff:  tran,
		qcoeff:  qcoeff,
		dqcoeff: st.rdoqDQ[:],
		levels:  st.levels,
		dqDC:    q.DC,
		dqAC:    q.AC,
	}
	s.optimizeB(eob, int(st.keyRDOQCtx.TXBSkipContext), int(st.keyRDOQCtx.DCSignContext))
}
