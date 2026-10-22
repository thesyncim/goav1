package encoder

import "github.com/thesyncim/goav1/internal/av1/quantize"

// cyclic_refresh.go ports SVT-AV1's realtime CBR cyclic refresh
// (Source/Lib/Codec/rc_aq.c svt_aom_cyclic_refresh_init and
// cyclic_sb_qp_derivation, with svt_av1_compute_deltaq from rc_process.c):
// each inter frame lowers the qindex of a sliding window of a quarter of the
// 64x64 superblocks, signalled through delta_q, so static content converges
// to a higher quality than the frame quantizer alone reaches. Superblocks in
// the window whose motion-compensated distortion sits below the frame
// average get the larger BOOST2 offset.
//
// Two inputs differ from upstream, both documented substitutions:
//   - SVT's me_8x8_distortion per superblock is replaced by the sum of this
//     encoder's hierarchical motion search region SADs (quarter resolution)
//     covering the superblock; only its comparison against the frame average
//     matters.
//   - percent_refresh_adjustment and rate_ratio_qdelta_adjustment are driven
//     by SVT's rate-correction feedback (av1_rc_update_rate_correction_factors);
//     this encoder's controller has no correction factor, so they stay at
//     their reset values (5 and 0.25).

const (
	crPercentRefresh         = 20 + 5
	crRateRatioAdjustment    = 0.25
	crMaxQDeltaPerc          = 60
	crRateBoostFac           = 15
	crBoostMax               = 10
	crMaxRateTargetRatio     = 4.0
	crFlatHierarchicalLevels = 2
)

// cyclicRefresh is the per-stream state.
type cyclicRefresh struct {
	sbEnd     int
	avgInterQ int
	grid      []uint8
	sbDist    []int64
}

func (cr *cyclicRefresh) reset(worstQ uint8) {
	cr.sbEnd = 0
	cr.avgInterQ = int(worstQ)
}

// noteInterQ is the avg_frame_qindex[INTER_FRAME] update of
// svt_av1_rc_postencode_update.
func (cr *cyclicRefresh) noteInterQ(q uint8) {
	cr.avgInterQ = (3*cr.avgInterQ + int(q) + 2) >> 2
}

// cyclicQRealTable is svt_av1_convert_qindex_to_q at 8-bit (AC step / 4).
var cyclicQRealTable = func() (t [256]float64) {
	for q := range t {
		ac, err := quantize.ACQuant(uint8(q), 0, 8)
		if err != nil {
			panic(err)
		}
		t[q] = float64(ac) / 4.0
	}
	return t
}()

// cyclicComputeDeltaQ is svt_av1_compute_deltaq for a CBR inter frame:
// svt_av1_compute_qdelta_by_rate over the svt_av1_rc_bits_per_mb model,
// floored at -max_qdelta_perc% of q.
func cyclicComputeDeltaQ(q int, ratio float64, bestQ, worstQ int) int {
	base := int(1000000.0 / cyclicQRealTable[q])
	target := int(ratio * float64(base))
	low, high := bestQ, worstQ
	for low < high {
		mid := (low + high) >> 1
		if int(1000000.0/cyclicQRealTable[mid]) > target {
			low = mid + 1
		} else {
			high = mid
		}
	}
	delta := low - q
	return max(delta, -crMaxQDeltaPerc*q/100)
}

// plan fills cr.grid with the qindex of every 64x64 superblock of an inter
// frame, or returns nil when cyclic refresh is off for the frame
// (svt_aom_cyclic_refresh_init's gates). hme supplies the per-region
// distortion; framesSinceKey counts inter frames since the last keyframe.
func (cr *cyclicRefresh) plan(baseQ uint8, bestQ, worstQ uint8, width, height int, hme *hmeState, framesSinceKey int, avgFrameLowMotion int) []uint8 {
	qpThresh := max(16, int(bestQ)+4)
	qpMaxThresh := 118 * 255 >> 7
	if cr.avgInterQ > qpMaxThresh || cr.avgInterQ < qpThresh {
		return nil
	}
	if avgFrameLowMotion > 0 && avgFrameLowMotion < 50 {
		return nil
	}
	cols := (width + 63) / 64
	rows := (height + 63) / 64
	sbCnt := cols * rows
	if sbCnt == 0 {
		return nil
	}
	sbStart := cr.sbEnd
	sbEnd := sbStart + sbCnt*crPercentRefresh/100
	if sbEnd >= sbCnt {
		cr.sbEnd = 0
	} else {
		cr.sbEnd = sbEnd
	}
	rateRatio := 1.15
	if framesSinceKey < 4*(1<<crFlatHierarchicalLevels)*100/crPercentRefresh {
		rateRatio = 1.50
	}
	rateRatio += crRateRatioAdjustment

	if len(cr.grid) < sbCnt {
		cr.grid = make([]uint8, sbCnt)
		cr.sbDist = make([]int64, sbCnt)
	}
	grid := cr.grid[:sbCnt]
	dist := cr.sbDist[:sbCnt]
	haveDist := hme != nil && hme.cols > 0
	var avg int64
	if haveDist {
		clear(dist)
		for ry := 0; ry < hme.rows; ry++ {
			for rx := 0; rx < hme.cols; rx++ {
				sb := min(ry/2, rows-1)*cols + min(rx/2, cols-1)
				dist[sb] += int64(hme.seedSADs[ry*hme.cols+rx])
			}
		}
		for _, d := range dist {
			avg += d
		}
		avg /= int64(sbCnt)
	}
	inWindow := func(i int) bool { return i >= sbStart && i < sbEnd }
	boostFac := crRateBoostFac
	if haveDist && avg > 0 {
		var seg2Dist int64
		seg2 := 0
		for i, d := range dist {
			if inWindow(i) && d < avg {
				seg2Dist += d
				seg2++
			}
		}
		if seg2 > 0 {
			seg2Dist /= int64(seg2)
			dev := (avg - seg2Dist) * 100 / avg
			boostFac += int(crBoostMax * dev * dev / (100 * 100))
		}
	}
	delta1 := cyclicComputeDeltaQ(int(baseQ), rateRatio, int(bestQ), int(worstQ))
	delta2 := cyclicComputeDeltaQ(int(baseQ), min(crMaxRateTargetRatio, 0.1*float64(boostFac)*rateRatio), int(bestQ), int(worstQ))
	for i := range grid {
		offset := 0
		if inWindow(i) {
			offset = delta1
			if haveDist && dist[i] < avg {
				offset = delta2
			}
		}
		grid[i] = uint8(min(max(int(baseQ)+offset, 1), 255))
	}
	return grid
}
