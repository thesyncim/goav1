package encoder

import "github.com/thesyncim/goav1/internal/av1/quantize"

// cyclic_refresh.go ports SVT-AV1's realtime CBR cyclic refresh
// (Source/Lib/Codec/rc_aq.c svt_aom_cyclic_refresh_init and
// cyclic_sb_qp_derivation, with svt_av1_compute_deltaq from rc_process.c):
// inter frames lower the qindex of a sliding window of the 64x64
// superblocks, signalled through delta_q, so static content converges to a
// higher quality than the frame quantizer alone reaches. Superblocks in the
// window whose ME distortion (me_8x8_distortion, hme.go) sits below the
// frame average get the larger BOOST2 offset. The refresh percentage and
// rate ratio adapt through the shared rate control
// (svtRateControl.percentRefreshAdjustment / rateRatioQDeltaAdjustment,
// driven by av1_rc_update_rate_correction_factors).

const (
	crMaxQDeltaPerc      = 60
	crBoostMax           = 10
	crMaxRateTargetRatio = 4.0
)

// cyclicRefresh is the per-stream state (enc_ctx->cr_sb_end).
type cyclicRefresh struct {
	sbEnd  int
	grid   []uint8
	sbDist []int64
}

func (cr *cyclicRefresh) reset() {
	cr.sbEnd = 0
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
func cyclicComputeDeltaQ(rc *svtRateControl, q int, ratio float64, screen bool) int {
	delta := rc.computeQDeltaByRate(svtFrameInter, q, ratio, screen)
	return max(delta, -crMaxQDeltaPerc*q/100)
}

// svtCRFrame is the picture context cyclic refresh reads.
type svtCRFrame struct {
	key               bool
	screen            bool
	flat              bool
	temporalLayer     int
	hierLevels        int
	avgFrameLowMotion int
}

// plan runs svt_aom_cyclic_refresh_init and cyclic_sb_qp_derivation for a
// picture coded at baseQ: it fills cr.grid with every 64x64 superblock's
// qindex, or returns nil when cyclic refresh is off for the picture, and the
// segment statistics the rate-correction update weighs.
func (cr *cyclicRefresh) plan(rc *svtRateControl, f svtCRFrame, baseQ uint8, width, height int, hme *hmeState) ([]uint8, svtCRStats) {
	var stats svtCRStats
	if f.key {
		// Reset adaptive elements for intra only frames.
		rc.percentRefreshAdjustment = 5
		rc.rateRatioQDeltaAdjustment = 0.25
	}
	percent := 20 + rc.percentRefreshAdjustment
	if f.screen {
		percent += 5
	}
	stats.percentFresh = percent
	apply := !f.key && (f.flat || f.temporalLayer == 0)
	qpThresh := max(16, rc.bestQuality+4)
	qpMaxThresh := 118 * 255 >> 7
	if rc.avgFrameQIndex[svtFrameInter] > qpMaxThresh || rc.avgFrameQIndex[svtFrameInter] < qpThresh {
		apply = false
	}
	if f.avgFrameLowMotion != 0 && f.avgFrameLowMotion < 50 {
		apply = false
	}
	if !apply {
		return nil, stats
	}
	cols := (width + 63) / 64
	rows := (height + 63) / 64
	sbCnt := cols * rows
	if sbCnt == 0 {
		return nil, stats
	}
	sbStart := cr.sbEnd
	sbEnd := sbStart + sbCnt*percent/100
	if sbEnd >= sbCnt {
		cr.sbEnd = 0
	} else {
		cr.sbEnd = sbEnd
	}
	var rateRatio float64
	var boostFac int
	if percent > 0 {
		if !f.screen {
			rateRatio = 1.15
			if rc.framesSinceKey < 4*(1<<f.hierLevels)*100/percent {
				rateRatio = 1.50
			}
			rateRatio += rc.rateRatioQDeltaAdjustment
			boostFac = 15
		} else {
			fromSC := min(0.75, float64(rc.framesSinceKey/10)*0.1)
			rateRatio = 2.25 + rc.rateRatioQDeltaAdjustment - fromSC
			if rc.rc1Frame < 0 || rc.rc2Frame < 0 {
				rateRatio -= 0.25
			}
			boostFac = 10
		}
	} else {
		rateRatio = 1.50 + rc.rateRatioQDeltaAdjustment
	}

	if len(cr.grid) < sbCnt {
		cr.grid = make([]uint8, sbCnt)
		cr.sbDist = make([]int64, sbCnt)
	}
	grid := cr.grid[:sbCnt]
	dist := cr.sbDist[:sbCnt]
	// me_8x8_distortion per SB and norm_me_dist, their frame average
	// (initial_rc_process.c), from the frame's open-loop ME (hme.go).
	haveDist := hme != nil && hme.fme.valid && hme.fme.cols == cols && hme.fme.rows == rows
	var avg int64
	if haveDist {
		for i := range dist {
			dist[i] = int64(hme.fme.me8x8Dist[i])
			avg += dist[i]
		}
		avg /= int64(sbCnt)
	}
	inWindow := func(i int) bool { return i >= sbStart && i < sbEnd }
	var seg2Dist int64
	for i, d := range dist {
		if !inWindow(i) {
			continue
		}
		if haveDist && d < avg {
			seg2Dist += d
			stats.seg2++
		} else {
			stats.seg1++
		}
	}
	if !f.screen && stats.seg2 > 0 && avg > 0 {
		seg2Dist /= int64(stats.seg2)
		dev := (avg - seg2Dist) * 100 / avg
		boostFac += int(crBoostMax * dev * dev / (100 * 100))
	}
	delta1 := cyclicComputeDeltaQ(rc, int(baseQ), rateRatio, f.screen)
	delta2 := cyclicComputeDeltaQ(rc, int(baseQ), min(crMaxRateTargetRatio, 0.1*float64(boostFac)*rateRatio), f.screen)
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
	stats.apply = true
	stats.sbCount = sbCnt
	stats.delta1, stats.delta2 = delta1, delta2
	return grid, stats
}
