package encoder

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// pframe_vlpd0.go is SVT-AV1's inter-picture partition decision at preset 12
// rtc: the VERY_LIGHT_PD0 pass (PD_PASS_0) whose tree PD1 then codes as is.
// Trace for a non-I, non-transition rtc picture above M11 (v4.1.0):
//
//   - pic_lpd0_lvl = 7 (set_pic_lpd0_lvl_rtc, enc_mode_config.c:8689), i.e.
//     pd0_level = VERY_LIGHT_PD0 (set_lpd0_ctrls case 7); the lpd0_detector
//     never runs at rtc non-sc >M9 (use_lpd0_classifier,
//     enc_dec_process.c:2928).
//   - skip_pd_pass_0 needs disallow_below_64x64, which depth-removal level 5
//     never sets (64 cost multiplier 0), so PD0 always runs.
//   - svt_aom_pick_partition_lpd0 (product_coding_loop.c:9949) walks the
//     square depths set_blocks_to_be_tested allows (enc_dec_process.c:1476:
//     max 64, min 16 under disallow_8x8, 32 under disallow_below_32x32).
//     Each block's cost is md_encode_block_light_pd0's VLPD0 arm
//     (product_coding_loop.c:7787): RDCOST(full_sb_lambda,
//     partition_fac_bits[0][PARTITION_NONE], mds0_best_cost), where
//     mds0_best_cost is the smallest full-pel variance between the source and
//     the reference reconstruction at the block's unipred ME vectors
//     (fast_loop_core_light_pd0 VLPD0 arm, product_coding_loop.c:967;
//     inject_new_candidates_light_pd0, mode_decision.c:2326), with the
//     zero-vector backup when ME left none (inject_zz_backup_candidate).
//     Intra is off (intra_level 0 above M10, sig_deriv_enc_dec_light_pd0).
//   - test_split_partition_lpd0 (product_coding_loop.c:9852) prices SPLIT at
//     RDCOST(full_sb_lambda, 2 * partition_fac_bits[bsl*4][SPLIT], 0) (the
//     doubling because use_accurate_part_ctx is off above M8; neighbour
//     contexts are not tracked in PD0), exits before each quadrant once the
//     parent (scaled by parent_cost_bias) is already no worse
//     (depth_early_exit level 0: both thresholds 1000), and keeps the parent
//     when parent_cost_bias * parent <= split * 1000.
//   - parent_cost_bias (svt_aom_sig_deriv_enc_dec_light_pd0,
//     enc_mode_config.c:7113-7131) = init_bias_tab[input_resolution] -
//     base_q_idx, then +150 / +50 / -50 / -150 by the SB's
//     me_8x8_cost_variance band.
//   - PD1 keeps exactly that tree: light-PD1 forces PD0_DEPTH_PRED_PART_ONLY
//     (enc_dec_process.c:3029) and refinement level 10 is the same mode.
//
// The inputs are the source, the reference reconstruction, the frame ME and
// the frame-start partition CDFs, so the per-SB tree is a pure function of
// the SB position: serial, split and wavefront walks see the same tree.
//
// Incomplete superblocks (frame edges) run the same pass with SVT's edge
// shapes: a node whose lower (right) half leaves the frame tests PART_H
// (PART_V) instead of PART_N when it is larger than nsq_geom level 4's 16x16
// minimum (set_blocks_to_test, enc_dec_process.c:1386; the shape check in
// svt_aom_pick_partition_lpd0), and its SPLIT is priced with the gathered
// split_or_horz / split_or_vert rate (svt_aom_partition_rate_cost,
// rd_cost.c:1847). One deviation: this encoder codes no block that crosses
// the frame edge, so a PART_N or PART_H/V block that would is not tested and
// the node splits.

// svtPD0SB is one superblock's VLPD0 tree: the partition of the 64x64 node,
// of its four 32x32 quadrants and of the sixteen 16x16 nodes (raster order
// over the SB). Nodes the chosen tree never reaches keep the edge fallback.
type svtPD0SB struct {
	ready  bool
	part64 tile.Partition
	part32 [4]tile.Partition
	part16 [16]tile.Partition
}

// svtPD0PartitionRates is the frame-frozen partition_fac_bits subset VLPD0
// reads (md_rate_estimation.c svt_aom_estimate_syntax_rate from the
// frame-start CDFs; update_se is off at rtc above M8): PARTITION_NONE of
// context 0 (the 8x8 CDF, read for every block size), and per square size
// 8/16/32/64 (bsl 0..3, context bsl*PARTITION_PLOFFSET) the PARTITION_SPLIT
// rate and the split entry of the partition_horz_alike / _vert_alike
// gathered rates.
type svtPD0PartitionRates struct {
	none  int64
	split [4]int64
	horz  [4]int64
	vert  [4]int64
}

func (r *svtPD0PartitionRates) estimate(cdfs *tile.PartitionCDFs) {
	r.none = int64(entropy.CostSymbol(&cdfs.Partition[tile.BlockLevel8x8][0], int(tile.PartitionNone)))
	for bsl, level := range [4]tile.BlockLevel{tile.BlockLevel8x8, tile.BlockLevel16x16, tile.BlockLevel32x32, tile.BlockLevel64x64} {
		cdf := &cdfs.Partition[level][0]
		r.split[bsl] = int64(entropy.CostSymbol(cdf, int(tile.PartitionSplit)))
		// partition_gather_horz_alike / _vert_alike (cabac_context_model.h:
		// 770-800): the split entry costs the summed mass of the partitions
		// that split the block's missing half.
		r.horz[bsl] = int64(entropy.CostProb15(svtPartitionMass(cdf, tile.PartitionH, tile.PartitionSplit,
			tile.PartitionTTopSplit, tile.PartitionTBottomSplit, tile.PartitionTLeftSplit, tile.PartitionH4)))
		r.vert[bsl] = int64(entropy.CostProb15(svtPartitionMass(cdf, tile.PartitionV, tile.PartitionSplit,
			tile.PartitionTTopSplit, tile.PartitionTLeftSplit, tile.PartitionTRightSplit, tile.PartitionV4)))
	}
}

// svtPartitionMass sums cdf_element_prob over the listed partitions the CDF
// codes.
func svtPartitionMass(cdf *entropy.CDF, parts ...tile.Partition) int {
	vals := cdf.Values()
	n := cdf.Symbols()
	mass := 0
	for _, p := range parts {
		s := int(p)
		if s >= n {
			continue
		}
		prev := entropy.CDFProbTop
		if s > 0 {
			prev = int(vals[s-1])
		}
		mass += prev - int(vals[s])
	}
	return mass
}

// svtInitBiasTab is init_bias_tab of svt_aom_sig_deriv_enc_dec_light_pd0,
// indexed by SVT's input-resolution range.
var svtInitBiasTab = [7]int64{1200, 1150, 1150, 1150, 1100, 1050, 1000}

// svtInputResolution is svt_aom_derive_input_resolution
// (sequence_control_set.c:120, definitions.h INPUT_SIZE_*_TH).
func svtInputResolution(width, height int) int {
	n := width * height
	switch {
	case n < 0x28500:
		return 0
	case n < 0x4CE00:
		return 1
	case n < 0xA1400:
		return 2
	case n < 0x16DA00:
		return 3
	case n < 0x535200:
		return 4
	case n < 0x140A000:
		return 5
	}
	return 6
}

func (st *lossyEncodeState) prepareSVTPD0(gridCols, gridRows int) {
	st.pd0Cols = gridCols
	need := gridCols * gridRows
	if need <= 0 {
		return
	}
	if len(st.pd0Grid) < need {
		st.pd0Grid = make([]svtPD0SB, need)
	}
	for i := range st.pd0Grid[:need] {
		st.pd0Grid[i].ready = false
	}
}

// svtPD0ForSB returns the VLPD0 tree of the superblock at (px, py),
// building it on first use.
func (st *lossyEncodeState) svtPD0ForSB(src, ref SourceFrame420, px, py int) (*svtPD0SB, error) {
	if st.pd0Cols <= 0 || src.Width != ref.Width || src.Height != ref.Height {
		return nil, nil
	}
	idx := (py/64)*st.pd0Cols + px/64
	if idx < 0 || idx >= len(st.pd0Grid) {
		return nil, nil
	}
	sb := &st.pd0Grid[idx]
	if sb.ready {
		return sb, nil
	}
	if err := st.buildSVTPD0(sb, src, ref, px, py); err != nil {
		return nil, err
	}
	return sb, nil
}

// svtPD0Fallback is the partition of a node the VLPD0 tree leaves
// untested: a 16x16 inside the frame stays whole, anything else splits.
func svtPD0Fallback(bx, by, n, width, height int) tile.Partition {
	if n == 16 && bx+16 <= width && by+16 <= height {
		return tile.PartitionNone
	}
	return tile.PartitionSplit
}

func (st *lossyEncodeState) buildSVTPD0(sb *svtPD0SB, src, ref SourceFrame420, px, py int) error {
	*sb = svtPD0SB{ready: true, part64: tile.PartitionSplit}
	for q := range sb.part32 {
		sb.part32[q] = tile.PartitionSplit
	}
	for i := range sb.part16 {
		sb.part16[i] = svtPD0Fallback(px+(i&3)*16, py+(i>>2)*16, 16, src.Width, src.Height)
	}
	// full_sb_lambda_md[EB_8_BIT_MD] at the SB's qindex.
	if err := st.useSBQuant(px, py); err != nil {
		return err
	}
	lambda := st.lpd1FullLambda(st.lpd1SB(px, py))
	me := st.hme.sbME(px, py)
	if me != nil && !me.valid {
		me = nil
	}

	// set_blocks_to_be_tested's min_sq_size: disallow_8x8 holds it at 16
	// unless the SB's dimensions need 8x8 blocks; depth removal raises it
	// (evaluated on complete SBs, where its SB-geometry guards all hold).
	sbW, sbH := min(64, src.Width-px), min(64, src.Height-py)
	minSq := 16
	if svtDimensionsRequire8x8(sbW, sbH) {
		minSq = 8
	}
	if st.depthRemovalLevel != 0 && me != nil && sbW == 64 && sbH == 64 {
		dec := realtimeDepthRemovalDecide(st.svtDepthRemovalLevelForSB(), st.svtDepthRemovalInputs(px, py), me)
		switch {
		case dec.below64:
			sb.part64 = tile.PartitionNone
			return nil
		case dec.below32:
			minSq = 32
		}
	}

	bias := svtInitBiasTab[svtInputResolution(src.Width, src.Height)] - int64(st.qIndex)
	if me != nil {
		switch v := me.me8x8CostVariance; {
		case v > 1000:
			bias += 150
		case v > 500:
			bias += 50
		case v > 250:
			bias -= 50
		default:
			bias -= 150
		}
	} else {
		bias -= 150
	}

	p := svtPD0Pass{st: st, sb: sb, src: src, ref: ref, me: me, sbX: px, sbY: py,
		miW: (src.Width / 4) * 4, miH: (src.Height / 4) * 4, lambda: lambda, bias: bias, minSq: minSq}
	p.pick(0, 0, 64)
	return nil
}

// svtPD0Pass carries one superblock's svt_aom_pick_partition_lpd0 state.
type svtPD0Pass struct {
	st       *lossyEncodeState
	sb       *svtPD0SB
	src, ref SourceFrame420
	me       *realtimeSBMultiSizeME
	sbX, sbY int
	// miW, miH are the frame size in luma samples at MI granularity, the
	// bounds the AV1 partition syntax tests halves against.
	miW, miH int
	lambda   int64
	bias     int64
	minSq    int
}

func (p *svtPD0Pass) record(x, y, n int, part tile.Partition) {
	switch n {
	case 64:
		p.sb.part64 = part
	case 32:
		p.sb.part32[(y/32)*2+x/32] = part
	case 16:
		p.sb.part16[(y/16)*4+x/16] = part
	}
}

// pick is svt_aom_pick_partition_lpd0 for the n x n node at (x, y) within
// the SB: it returns whether the node has a valid partition and its RD cost
// (its tested shape's block cost, or the split cost when SPLIT wins), and
// records the winning partition.
func (p *svtPD0Pass) pick(x, y, n int) (bool, int64) {
	bx, by := p.sbX+x, p.sbY+y
	half := n / 2
	hasRows := by+half < p.miH
	hasCols := bx+half < p.miW
	frameW, frameH := p.src.Width, p.src.Height

	// The tested shape (set_blocks_to_test, then the pick_partition_lpd0
	// shape check): PART_N when both halves are present, PART_H / PART_V
	// above the 16x16 NSQ minimum when one is missing.
	valid := false
	var cost int64
	shape := tile.PartitionNone
	w, h := n, n
	switch {
	case hasRows && hasCols:
	case !hasRows && hasCols && n > 16:
		shape, h = tile.PartitionH, half
	case hasRows && !hasCols && n > 16:
		shape, w = tile.PartitionV, half
	default:
		shape = partitionInvalid
	}
	if shape != partitionInvalid && bx+w <= frameW && by+h <= frameH {
		dist := p.st.svtPD0BlockDist(p.src, p.ref, p.me, p.sbX, p.sbY, x, y, w, h)
		cost = depthRemovalRDCost(p.lambda, p.st.pd0Rates.none, dist)
		valid = true
		p.record(x, y, n, shape)
	}
	if n <= p.minSq {
		return valid, cost
	}

	// test_split_partition_lpd0.
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
		splitRate = p.st.pd0Rates.split[bsl]
	case !hasRows && hasCols:
		splitRate = p.st.pd0Rates.horz[bsl]
	case hasRows && !hasCols:
		splitRate = p.st.pd0Rates.vert[bsl]
	}
	splitCost := depthRemovalRDCost(p.lambda, 2*splitRate, 0)
	for i := range 4 {
		cx, cy := x+(i&1)*half, y+(i>>1)*half
		if p.sbY+cy >= p.miH || p.sbX+cx >= p.miW {
			continue
		}
		// depth_early_exit level 0: split_cost_th = early_exit_th = 0 -> 1000.
		if valid && cost*1000*p.bias <= splitCost*1000*1000 {
			return true, cost
		}
		ok, c := p.pick(cx, cy, half)
		if !ok {
			if valid {
				p.record(x, y, n, shape)
			}
			return valid, cost
		}
		splitCost += c
	}
	if valid && p.bias*cost <= splitCost*1000 {
		p.record(x, y, n, shape)
		return true, cost
	}
	p.record(x, y, n, tile.PartitionSplit)
	return true, splitCost
}

// partitionInvalid marks a node with no testable shape.
const partitionInvalid = tile.Partition(0xFF)

// svtPD0BlockDist is the VLPD0 mds0_best_cost of the w x h block at (x, y)
// within the SB at (sbX, sbY): fast_loop_core_light_pd0's
// svt_aom_mefn_ptr[bsize].vf variance between the source and the reference
// reconstruction at the block's ME vector, full-pel (mv >> 3), the PU being
// svt_aom_get_me_block_offset's (the square of side max(w, h)). goav1's
// frame ME searches one reference, so the PU contributes one candidate; with
// no ME the zero-vector backup is the candidate. Positions that leave the
// frame read edge-replicated samples, the padding SVT's reference buffers
// carry.
func (st *lossyEncodeState) svtPD0BlockDist(src, ref SourceFrame420, me *realtimeSBMultiSizeME, sbX, sbY, x, y, w, h int) int64 {
	dx, dy := 0, 0
	if me != nil {
		pu := svtMEBlockOffset(x, y, max(w, h))
		dx, dy = int(me.bestMV[pu][0]), int(me.bestMV[pu][1])
	}
	bx, by := sbX+x, sbY+y
	rx, ry := bx+dx, by+dy
	srcBlk := src.Y[by*src.YStride+bx:]
	var sse uint32
	var sum int32
	if rx >= 0 && ry >= 0 && rx+w <= ref.Width && ry+h <= ref.Height {
		sse, sum = svtPD0RectStats(srcBlk, src.YStride, ref.Y[ry*ref.YStride+rx:], ref.YStride, w, h)
	} else if p := st.refPads.lookup(ref.Y, ref.Width, ref.Height); p != nil && rx >= -p.border && ry >= -p.border && rx+w <= ref.Width+p.border && ry+h <= ref.Height+p.border {
		// SVT reads its padded reference picture directly; the padded copy
		// replicates the edges exactly as the clamped window below does.
		sse, sum = svtPD0RectStats(srcBlk, src.YStride, p.buf[(ry+p.border)*p.stride+rx+p.border:], p.stride, w, h)
	} else {
		buf := st.pd0EdgeRef[:w*h]
		for r := range h {
			yy := min(max(ry+r, 0), ref.Height-1)
			row := ref.Y[yy*ref.YStride:]
			d := buf[r*w : r*w+w]
			for c := range d {
				d[c] = row[min(max(rx+c, 0), ref.Width-1)]
			}
		}
		sse, sum = svtPD0RectStats(srcBlk, src.YStride, buf, w, w, h)
	}
	// svt_aom_variance{w}x{h}: sse - sum^2 / (w*h), the division a shift.
	log2N := uint(bits.Len(uint(w*h)) - 1)
	return int64(sse) - int64((int64(sum)*int64(sum))>>log2N)
}

// svtPD0RectStats is the (sse, sum) pair of the w x h residual for the
// shapes VLPD0 tests.
func svtPD0RectStats(src []byte, srcStride int, ref []byte, refStride, w, h int) (uint32, int32) {
	switch {
	case w == h:
		return svtPD0PixelStats(src, srcStride, ref, refStride, w)
	case w == 64 && h == 32:
		return pixelStats64x32(src, srcStride, ref, refStride)
	case w == 32 && h == 64:
		return pixelStats32x64(src, srcStride, ref, refStride)
	case w == 32 && h == 16:
		return pixelStats32x16(src, srcStride, ref, refStride)
	case w == 16 && h == 32:
		return pixelStats16x32(src, srcStride, ref, refStride)
	}
	return pixelStatsPureGo(src, srcStride, ref, refStride, w, h)
}

// svtPD0PixelStats is the (sse, sum) pair of the n x n residual through the
// dispatched pixel-statistics kernels.
func svtPD0PixelStats(src []byte, srcStride int, ref []byte, refStride, n int) (uint32, int32) {
	switch n {
	case 64:
		s0, m0 := pixelStats64x32(src, srcStride, ref, refStride)
		s1, m1 := pixelStats64x32(src[32*srcStride:], srcStride, ref[32*refStride:], refStride)
		return s0 + s1, m0 + m1
	case 32:
		return pixelStats32x32(src, srcStride, ref, refStride)
	case 16:
		return pixelStats16x16(src, srcStride, ref, refStride)
	}
	return pixelStats8x8(src, srcStride, ref, refStride)
}

// svtDepthRemovalLevelForSB is the SB's depth-removal level after the
// delta-q modulation (the SB quantizer is already selected by useSBQuant).
func (st *lossyEncodeState) svtDepthRemovalLevelForSB() uint8 {
	return svtDepthRemovalLevelForSB(st.depthRemovalLevel, st.sbQIndex != nil, st.quantQ, svtPictureQP(st.qIndex))
}

// svtDepthRemovalInputs gathers set_depth_removal_level_controls' per-SB
// inputs for the SB at (px, py) once useSBQuant selected its quantizer.
func (st *lossyEncodeState) svtDepthRemovalInputs(px, py int) svtDepthRemovalInputs {
	dec := st.lpd1SB(px, py)
	// svt_aom_compute_fast_lambda: av1_lambda_mode_decision8_bit_sad at the
	// SB qindex through update_lambda's temporal-layer factor and the rtc
	// me_q_index modulation (rc_process.c:393-454).
	rdmult := int64(av1LambdaModeDecision8BitSAD[st.quantQ])
	tl := st.lpd1.tlFactor
	if tl == 0 {
		tl = 150
	}
	rdmult = (rdmult * tl) >> 7
	f := dec.meQFactor
	if f == 0 {
		f = 128
	}
	in := svtDepthRemovalInputs{
		pictureQP:  svtPictureQP(st.qIndex),
		fastLambda: (rdmult * f) >> 7,
		refMinSq:   0xFF,
	}
	cur, ref := st.lpd1.cur, st.lpd1.ref
	if cur != nil && ref != nil && ref.valid && !ref.sliceI && ref.width == cur.width && ref.height == cur.height &&
		cur.poc-ref.poc <= 1 && ref.poc-cur.poc <= 1 {
		in.refMinSq = ref.sb[(py/64)*ref.cols+px/64].minSq
	}
	return in
}
