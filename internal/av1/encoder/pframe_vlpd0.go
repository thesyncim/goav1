package encoder

import (
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
// Incomplete superblocks (frame edges) keep the caller's edge rules; SVT
// injects PART_H/PART_V there (nsq_geom level 4), which this encoder does not
// code.

// svtPD0SB is one superblock's VLPD0 tree: the 64x64 node and its four 32x32
// quadrants (16x16 is the minimum depth).
type svtPD0SB struct {
	ready  bool
	part64 tile.Partition
	part32 [4]tile.Partition
}

// svtPD0PartitionRates is the frame-frozen partition_fac_bits subset VLPD0
// reads (md_rate_estimation.c svt_aom_estimate_syntax_rate from the
// frame-start CDFs; update_se is off at rtc above M8): PARTITION_NONE of
// context 0 (the 8x8 CDF, read for every block size) and PARTITION_SPLIT of
// context bsl*PARTITION_PLOFFSET for 16x16, 32x32 and 64x64.
type svtPD0PartitionRates struct {
	none  int64
	split [3]int64 // 16x16, 32x32, 64x64
}

func (r *svtPD0PartitionRates) estimate(cdfs *tile.PartitionCDFs) {
	r.none = int64(entropy.CostSymbol(&cdfs.Partition[tile.BlockLevel8x8][0], int(tile.PartitionNone)))
	r.split[0] = int64(entropy.CostSymbol(&cdfs.Partition[tile.BlockLevel16x16][0], int(tile.PartitionSplit)))
	r.split[1] = int64(entropy.CostSymbol(&cdfs.Partition[tile.BlockLevel32x32][0], int(tile.PartitionSplit)))
	r.split[2] = int64(entropy.CostSymbol(&cdfs.Partition[tile.BlockLevel64x64][0], int(tile.PartitionSplit)))
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

// svtPD0ForSB returns the VLPD0 tree of the complete superblock at (px, py),
// building it on first use; nil for incomplete superblocks.
func (st *lossyEncodeState) svtPD0ForSB(src, ref SourceFrame420, px, py int) (*svtPD0SB, error) {
	if st.pd0Cols <= 0 || px+64 > src.Width || py+64 > src.Height ||
		src.Width != ref.Width || src.Height != ref.Height {
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

func (st *lossyEncodeState) buildSVTPD0(sb *svtPD0SB, src, ref SourceFrame420, px, py int) error {
	*sb = svtPD0SB{ready: true, part64: tile.PartitionSplit}
	for i := range sb.part32 {
		sb.part32[i] = tile.PartitionNone
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

	// Depth removal (set_depth_removal_level_controls) clamps min_sq_size;
	// disallow_8x8 already holds it at 16 for a complete SB.
	minSq := 16
	if st.depthRemovalLevel != 0 && me != nil {
		dec := realtimeDepthRemovalDecide(st.depthRemovalLevel, st.qIndex, me)
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

	p := svtPD0Pass{st: st, src: src, ref: ref, me: me, sbX: px, sbY: py, lambda: lambda, bias: bias, minSq: minSq}
	_, split64 := p.pick(0, 0, 64)
	if !split64 {
		sb.part64 = tile.PartitionNone
		return nil
	}
	sb.part64 = tile.PartitionSplit
	sb.part32 = p.split32
	return nil
}

// svtPD0Pass carries one superblock's svt_aom_pick_partition_lpd0 state.
type svtPD0Pass struct {
	st       *lossyEncodeState
	src, ref SourceFrame420
	me       *realtimeSBMultiSizeME
	sbX, sbY int
	lambda   int64
	bias     int64
	minSq    int
	split32  [4]tile.Partition
}

// pick is svt_aom_pick_partition_lpd0 for the n x n node at (x, y) within
// the SB: it returns the node's RD cost (its own block cost, or the split
// cost when SPLIT wins) and whether SPLIT won.
func (p *svtPD0Pass) pick(x, y, n int) (int64, bool) {
	dist := p.st.svtPD0BlockDist(p.src, p.ref, p.me, p.sbX, p.sbY, x, y, n)
	cost := depthRemovalRDCost(p.lambda, p.st.pd0Rates.none, dist)
	if n <= p.minSq {
		return cost, false
	}
	// test_split_partition_lpd0.
	var splitRate int64
	switch n {
	case 64:
		splitRate = p.st.pd0Rates.split[2]
	case 32:
		splitRate = p.st.pd0Rates.split[1]
	default:
		splitRate = p.st.pd0Rates.split[0]
	}
	splitCost := depthRemovalRDCost(p.lambda, 2*splitRate, 0)
	half := n / 2
	var childSplit [4]bool
	for i := range 4 {
		// depth_early_exit level 0: split_cost_th = early_exit_th = 0 -> 1000.
		if cost*1000*p.bias <= splitCost*1000*1000 {
			return cost, false
		}
		c, s := p.pick(x+(i&1)*half, y+(i>>1)*half, half)
		childSplit[i] = s
		splitCost += c
	}
	if p.bias*cost <= splitCost*1000 {
		return cost, false
	}
	if n == 64 {
		for i, s := range childSplit {
			if s {
				p.split32[i] = tile.PartitionSplit
			}
		}
	}
	return splitCost, true
}

// svtPD0BlockDist is the VLPD0 mds0_best_cost of the n x n block at (x, y)
// within the SB at (sbX, sbY): fast_loop_core_light_pd0's
// svt_aom_mefn_ptr[bsize].vf variance between the source and the reference
// reconstruction at the block's ME vector, full-pel (mv >> 3). goav1's frame
// ME searches one reference, so the PU contributes one candidate; with no ME
// the zero-vector backup is the candidate. Positions that leave the frame
// read edge-replicated samples, the padding SVT's reference buffers carry.
func (st *lossyEncodeState) svtPD0BlockDist(src, ref SourceFrame420, me *realtimeSBMultiSizeME, sbX, sbY, x, y, n int) int64 {
	dx, dy := 0, 0
	if me != nil {
		pu := svtMEBlockOffset(x, y, n)
		dx, dy = int(me.bestMV[pu][0]), int(me.bestMV[pu][1])
	}
	bx, by := sbX+x, sbY+y
	rx, ry := bx+dx, by+dy
	srcBlk := src.Y[by*src.YStride+bx:]
	var sse uint32
	var sum int32
	if rx >= 0 && ry >= 0 && rx+n <= ref.Width && ry+n <= ref.Height {
		sse, sum = svtPD0PixelStats(srcBlk, src.YStride, ref.Y[ry*ref.YStride+rx:], ref.YStride, n)
	} else {
		buf := st.pd0EdgeRef[:n*n]
		for r := range n {
			yy := min(max(ry+r, 0), ref.Height-1)
			row := ref.Y[yy*ref.YStride:]
			d := buf[r*n : r*n+n]
			for c := range d {
				d[c] = row[min(max(rx+c, 0), ref.Width-1)]
			}
		}
		sse, sum = svtPD0PixelStats(srcBlk, src.YStride, buf, n, n)
	}
	// svt_aom_variance{n}x{n}: sse - sum^2 / (n*n), the division a shift.
	var log2N uint
	switch n {
	case 64:
		log2N = 12
	case 32:
		log2N = 10
	case 16:
		log2N = 8
	default:
		log2N = 6
	}
	return int64(sse) - int64((int64(sum)*int64(sum))>>log2N)
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
