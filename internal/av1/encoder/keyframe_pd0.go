package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// keyframe_pd0.go is SVT-AV1's intra-slice partition decision at the rtc
// presets above M11 (v4.1.0): PD_PASS_0 at light-PD0 level 4, whose tree PD1
// then codes as is. Trace for a camera-content I slice:
//
//   - pic_lpd0_lvl = 6 (set_pic_lpd0_lvl_rtc: is_islice ? 6 : 7), i.e.
//     pd0_level = LPD0_LVL_4 (set_lpd0_ctrls case 6). The lpd0_detector does
//     not run at rtc non-sc above M9 (use_lpd0_classifier,
//     enc_dec_process.c). Depth removal is off on I slices
//     (set_depth_removal_level_controls) and disallow_8x8 holds the minimum
//     square at 16 (svt_aom_get_disallow_8x8_rtc above M10) unless the SB's
//     dimensions need 8x8 blocks.
//   - svt_aom_sig_deriv_enc_dec_light_pd0: depth_early_exit level 2
//     (split_cost_th 50, early_exit_th 900), parent_cost_bias 1000, PF off
//     (DEFAULT_SHAPE), rate_est_level 0 (coeff_rate_est_lvl 0, lpd0_qp_offset
//     8) and subres level 2 on a complete SB with 8x8 disallowed (an I slice
//     is never the highest layer), else 1; subres needs 4x4 disallowed.
//   - md_encode_block_light_pd0 injects DC_PRED only
//     (inject_intra_candidates_light_pd0) and runs md_stage_3_light_pd0:
//     full_loop_core_light_pd0 checks at the 64x64 block whether subres is
//     safe (check_is_subres_safe: the even- and odd-row prediction SADs within
//     5%), then perform_tx_light_pd0 codes the DCT_DCT luma residual of every
//     2^step-th row at qindex + 8 with the b quantizer and prices it as
//     RDCOST(full_sb_lambda, 5000 + 1600 * resolution factor + 100 * eob +
//     skip_fac_bits[0][0] + partition_fac_bits[0][PARTITION_NONE], the
//     transform-domain SSE plus the dropped 64-extent energy).
//   - The block's reconstruction (the step's rows, each repeated 2^step
//     times) feeds the following blocks' DC prediction through the MD
//     neighbour arrays, updated as svt_aom_pick_partition_lpd0 and
//     test_split_partition_lpd0 do; outside the SB they hold the final
//     reconstruction of the neighbouring SBs.
//   - test_split_partition_lpd0 prices SPLIT at RDCOST(full_sb_lambda,
//     2 * partition_fac_bits[bsl * 4][SPLIT], 0) (use_accurate_part_ctx is off
//     above M8), exits before quadrant 0 when parent * 50 * 1000 <= split *
//     1e6 and before the others at 900, and keeps the parent when 1000 *
//     parent <= 1000 * split.
//
// One deviation: this keyframe coder codes square blocks only, so a frame
// edge node whose tested shape is PART_H or PART_V splits instead.

// keyPD0 is one superblock's light-PD0 pass state.
type keyPD0 struct {
	st         *lossyEncodeState
	sb         *svtPD0SB
	src, recon SourceFrame420
	sbX, sbY   int
	miW, miH   int
	minSq      int
	subresStep int
	// subresSafe is is_subres_safe for the SB: -1 until the 64x64 block
	// checks it.
	subresSafe int
	lambda     int64
	q          quantize.Quantizer
	// top[x] / left[y] are the MD neighbour arrays over the SB (SB-relative),
	// with their frame-edge availability.
	top, left [64]uint8
	haveTopSB bool
	haveLfSB  bool
	// node frontiers: the bottom row and right column of each node's tested
	// block, by depth.
	nodeBottom [4][64]uint8
	nodeRight  [4][64]uint8
	pred       [64 * 64]byte
	recBlock   [64 * 64]byte
	residual   [64 * 64]int16
	tran       [64 * 64]int32
	qcoeff     [64 * 64]int16
	scratch    [64 * 64]int32
}

// keyPD0Rates are the frame-start rate terms the pass reads.
type keyPD0Rates struct {
	ready     bool
	qIndex    uint8
	width     int
	height    int
	parts     svtPD0PartitionRates
	skip      int64
	coeffBase int64
	lambda    int64
	q         quantize.Quantizer
}

func (st *lossyEncodeState) keyPD0RatesFor(width, height int) *keyPD0Rates {
	r := &st.keyPD0Rates
	if r.ready && r.qIndex == st.qIndex && r.width == width && r.height == height {
		return r
	}
	*r = keyPD0Rates{qIndex: st.qIndex, width: width, height: height}
	var parts tile.PartitionCDFs
	if parts.InitDefault() != nil {
		return r
	}
	r.parts.estimate(&parts)
	var modes tile.BlockModeCDFs
	if modes.InitDefault() != nil {
		return r
	}
	skipCDF, err := modes.SkipCDF(0)
	if err != nil {
		return r
	}
	r.skip = int64(entropy.CostSymbol(skipCDF, 0))
	factor := [7]int64{0, 1, 2, 3, 4, 4, 4}
	r.coeffBase = 5000 + factor[svtInputResolution(width, height)]*1600
	r.lambda = int64(svtCDEFFullLambda(st.qIndex, true, false, 150))
	q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, uint8(min(255, int(st.qIndex)+8)), 8, quantize.PlaneY)
	if err != nil {
		return r
	}
	r.q = q
	r.ready = true
	return r
}

// keyPD0ForSB returns the light-PD0 tree of the keyframe SB at (px, py),
// building it on first use.
func (st *lossyEncodeState) keyPD0ForSB(src, recon SourceFrame420, px, py, tileX0 int) *svtPD0SB {
	if st.pd0Cols <= 0 {
		return nil
	}
	idx := (py/64)*st.pd0Cols + px/64
	if idx < 0 || idx >= len(st.pd0Grid) {
		return nil
	}
	sb := &st.pd0Grid[idx]
	if sb.ready {
		return sb
	}
	rates := st.keyPD0RatesFor(src.Width, src.Height)
	if !rates.ready {
		return nil
	}
	if st.keyPD0 == nil {
		st.keyPD0 = new(keyPD0)
	}
	p := st.keyPD0
	*sb = svtPD0SB{ready: true, part64: tile.PartitionSplit}
	for q := range sb.part32 {
		sb.part32[q] = tile.PartitionSplit
	}
	for i := range sb.part16 {
		sb.part16[i] = svtPD0Fallback(px+(i&3)*16, py+(i>>2)*16, 16, src.Width, src.Height)
	}
	p.st, p.sb, p.src, p.recon, p.sbX, p.sbY = st, sb, src, recon, px, py
	p.miW, p.miH = (src.Width/4)*4, (src.Height/4)*4
	sbW, sbH := min(64, src.Width-px), min(64, src.Height-py)
	p.minSq = 16
	if svtDimensionsRequire8x8(sbW, sbH) {
		p.minSq = 8
	}
	// set_subres_controls: level 2 with 8x8 disallowed, else 1; off on an
	// incomplete 64x64.
	p.subresStep = 0
	if sbW == 64 && sbH == 64 {
		p.subresStep = 2
		if p.minSq < 16 {
			p.subresStep = 1
		}
	}
	p.subresSafe = -1
	p.lambda = rates.lambda
	p.q = rates.q
	p.haveTopSB = py > 0
	// The neighbour arrays are per tile: a tile's first SB column has no left.
	p.haveLfSB = px > tileX0
	if p.haveTopSB {
		row := recon.Y[(py-1)*recon.YStride+px:]
		copy(p.top[:sbW], row[:sbW])
	}
	if p.haveLfSB {
		for y := range sbH {
			p.left[y] = recon.Y[(py+y)*recon.YStride+px-1]
		}
	}
	p.pick(rates, 0, 0, 64, 0, 0)
	return sb
}

func (p *keyPD0) record(x, y, n int, part tile.Partition) {
	switch n {
	case 64:
		p.sb.part64 = part
	case 32:
		p.sb.part32[(y/32)*2+x/32] = part
	case 16:
		p.sb.part16[(y/16)*4+x/16] = part
	}
}

// commit is mode_decision_update_neighbor_arrays_light_pd0 for the node at
// depth d: its tested block's bottom row and right column become the
// neighbour arrays.
func (p *keyPD0) commit(d, x, y, n int) {
	copy(p.top[x:x+n], p.nodeBottom[d][:n])
	copy(p.left[y:y+n], p.nodeRight[d][:n])
}

// dc is the DC_PRED value of the n x n block at SB-relative (x, y) from the
// neighbour arrays (n x n blocks only: the rectangular edge shapes split).
func (p *keyPD0) dc(x, y, n int) uint8 {
	haveTop := y > 0 || p.haveTopSB
	haveLeft := x > 0 || p.haveLfSB
	sum, count := 0, 0
	if haveTop {
		for i := range n {
			sum += int(p.top[x+i])
		}
		count += n
	}
	if haveLeft {
		for i := range n {
			sum += int(p.left[y+i])
		}
		count += n
	}
	if count == 0 {
		return 128
	}
	return uint8((sum + count/2) / count)
}

// blockCost is md_encode_block_light_pd0 for the n x n block at (x, y): it
// returns the block cost and leaves the block's frontier at depth d.
func (p *keyPD0) blockCost(rates *keyPD0Rates, x, y, n, d int) int64 {
	src := p.src
	srcOff := (p.sbY+y)*src.YStride + p.sbX + x
	dc := p.dc(x, y, n)
	// check_is_subres_safe at the 64x64 block.
	step := p.subresStep
	if step > 0 && n == 64 && p.subresSafe < 0 {
		even, odd := 0, 0
		for r := 0; r < n; r++ {
			row := src.Y[srcOff+r*src.YStride:]
			s := 0
			for c := range n {
				v := int(row[c]) - int(dc)
				if v < 0 {
					v = -v
				}
				s += v
			}
			if r&1 == 0 {
				even += s
			} else {
				odd += s
			}
		}
		dev := ((max(even, 1) - max(odd, 1)) * 100) / max(odd, 1)
		if dev < 0 {
			dev = -dev
		}
		p.subresSafe = 0
		if dev <= 5 {
			p.subresSafe = 1
		}
	}
	if p.subresSafe != 1 {
		step = 0
	}
	if n < 16 {
		step = min(step, 1)
	}
	th := n >> step
	geo, err := txBlockGeometryForDimensions(n, th)
	if err != nil {
		step, th = 0, n
		geo, _ = txBlockGeometryForDimensions(n, n)
	}
	for r := range th {
		row := src.Y[srcOff+(r<<step)*src.YStride:]
		res := p.residual[r*n:]
		for c := range n {
			res[c] = int16(int(row[c]) - int(dc))
		}
	}
	cn := geo.coeffCount
	var energy uint64
	if n == 64 {
		energy, err = transform.ForwardDCT64ExtentEnergy(p.tran[:cn], geo.coeffHeight, p.residual[:n*th], n, geo.size)
	} else {
		err = forwardTransformBlock8BitResidualTrusted(p.tran[:cn], p.residual[:n*th], p.scratch[:n*th], n, th, transform.TypeDCTDCT)
	}
	if err != nil {
		return 1 << 62
	}
	if quantize.QuantizeBlockScaledB(p.qcoeff[:cn], geo.coeffHeight, p.tran[:cn], geo.coeffHeight, geo.coeffWidth, geo.coeffHeight, p.q, geo.txScale) != nil {
		return 1 << 62
	}
	scan, ok := p.st.scanForTransformSize(geo.tx)
	if !ok {
		scan, ok = p.st.scanForThinTransformSize(geo.tx)
	}
	eob := 0
	if ok {
		for i := cn - 1; i >= 0; i-- {
			if p.qcoeff[scan[i]] != 0 {
				eob = i + 1
				break
			}
		}
	}
	// Transform-domain SSE against the dequantized levels
	// (svt_aom_picture_full_distortion32_bits_single), plus the dropped
	// 64-extent energy, at the TX scale (MAX_TX_SCALE 1) and the subres step.
	var dist int64
	for i := range cn {
		t := int64(p.tran[i])
		dq := int64(0)
		if lv := int64(p.qcoeff[i]); lv != 0 {
			deq := int64(p.q.AC)
			if i == 0 {
				deq = int64(p.q.DC)
			}
			a := lv
			if a < 0 {
				a = -a
			}
			dq = (a * deq) >> geo.txScale
			if lv < 0 {
				dq = -dq
			}
		}
		e := t - dq
		dist += e * e
	}
	dist += int64(energy)
	if shift := (1 - int(geo.txScale)) * 2; shift >= 0 {
		dist >>= shift
	} else {
		dist <<= -shift
	}
	dist <<= step
	rate := rates.coeffBase + 100*int64(eob) + rates.skip + rates.parts.none
	cost := depthRemovalRDCost(p.lambda, rate, dist)

	// The block's reconstruction for the neighbour arrays.
	pred := p.pred[:n*th]
	memsetBytes(pred, dc)
	rec := p.recBlock[:n*th]
	if eob == 0 {
		memsetBytes(rec, dc)
	} else if err := p.st.reconInterTXBResult(rec, pred, n, n, 0, 0, n, th, p.q, p.qcoeff[:cn], geo, scan, transform.TypeDCTDCT, interTXBResult(p.qcoeff[:cn], scan, cn)); err != nil {
		memsetBytes(rec, dc)
	}
	copy(p.nodeBottom[d][:n], rec[(th-1)*n:th*n])
	for r := range n {
		p.nodeRight[d][r] = rec[(r>>step)*n+n-1]
	}
	return cost
}

// pick is svt_aom_pick_partition_lpd0 for the n x n node at (x, y); quad is
// its quadrant index under its parent (mds->index), d its depth.
func (p *keyPD0) pick(rates *keyPD0Rates, x, y, n, quad, d int) (bool, int64) {
	bx, by := p.sbX+x, p.sbY+y
	half := n / 2
	hasRows := by+half < p.miH
	hasCols := bx+half < p.miW
	valid := false
	var cost int64
	if hasRows && hasCols && bx+n <= p.src.Width && by+n <= p.src.Height {
		cost = p.blockCost(rates, x, y, n, d)
		valid = true
		p.record(x, y, n, tile.PartitionNone)
	}
	if n <= p.minSq {
		if valid && quad < 3 {
			p.commit(d, x, y, n)
		}
		return valid, cost
	}
	bsl := 0
	switch n {
	case 64:
		bsl = 3
	case 32:
		bsl = 2
	case 16:
		bsl = 1
	}
	var splitRate int64
	switch {
	case hasRows && hasCols:
		splitRate = rates.parts.split[bsl]
	case !hasRows && hasCols:
		splitRate = rates.parts.horz[bsl]
	case hasRows && !hasCols:
		splitRate = rates.parts.vert[bsl]
	}
	splitCost := depthRemovalRDCost(p.lambda, 2*splitRate, 0)
	lastQuadValid := true
	lastChild := -1
	for i := range 4 {
		cx, cy := x+(i&1)*half, y+(i>>1)*half
		if p.sbY+cy >= p.miH || p.sbX+cx >= p.miW {
			lastQuadValid = false
			continue
		}
		th := int64(900)
		if i == 0 {
			th = 50
		}
		if valid && cost*th*1000 <= splitCost*1000*1000 {
			p.commit(d, x, y, n)
			return true, cost
		}
		ok, c := p.pick(rates, cx, cy, half, i, d+1)
		if !ok {
			if valid {
				p.record(x, y, n, tile.PartitionNone)
				p.commit(d, x, y, n)
			}
			return valid, cost
		}
		splitCost += c
		lastChild = i
	}
	if valid && 1000*cost <= splitCost*1000 {
		p.record(x, y, n, tile.PartitionNone)
		p.commit(d, x, y, n)
		return true, cost
	}
	p.record(x, y, n, tile.PartitionSplit)
	// The fourth quadrant's arrays, unless it split itself.
	if lastQuadValid && lastChild == 3 && p.childPartition(x+half, y+half, half) != tile.PartitionSplit {
		p.commit(d+1, x+half, y+half, half)
	}
	return true, splitCost
}

func (p *keyPD0) childPartition(x, y, n int) tile.Partition {
	switch n {
	case 32:
		return p.sb.part32[(y/32)*2+x/32]
	case 16:
		return p.sb.part16[(y/16)*4+x/16]
	}
	return tile.PartitionNone
}
