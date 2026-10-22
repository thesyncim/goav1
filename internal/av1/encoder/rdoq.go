package encoder

import (
	"math/bits"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// rdoq.go ports SVT-AV1's rate-distortion optimized quantization:
// full_loop.c svt_av1_optimize_b with update_coeff_general/update_coeff_eob/
// update_coeff_simple/update_skip, over the coefficient rate tables of
// md_rate_estimation.c svt_aom_estimate_coefficients_rate. The realtime
// light-PD1 path runs it on every DCT_DCT transform block (rdoq_level 1)
// after the fp quantizer, with the skip and DC-sign contexts at zero
// (product_coding_loop.c perform_dct_dct_tx_light_pd1).

const (
	rdoqNumBaseLevels  = 2
	rdoqCoeffBaseRange = 12
	rdoqBRCDFSize      = 4
	rdoqMaxNZ          = 4
)

// rdoqCoeffCost is SVT's LvMapCoeffCost for one transform-size context and
// plane type, in 512-per-bit units.
type rdoqCoeffCost struct {
	txbSkip  [tile.TXBSkipContexts][2]int32
	baseEOB  [tile.EOBBaseContexts][3]int32
	base     [tile.CoeffBaseContexts][8]int32
	eobExtra [tile.EOBCoefContexts][2]int32
	dcSign   [3][2]int32
	lps      [tile.CoeffBRContexts][2*rdoqCoeffBaseRange + 2]int32
}

// rdoqCosts holds the per-qindex tables; eob is SVT's LvMapEobCost indexed by
// txsize_log2_minus4 and plane type.
type rdoqCosts struct {
	coeff [tile.CoeffTxSizeContexts][tile.CoeffPlaneTypes]rdoqCoeffCost
	eob   [7][tile.CoeffPlaneTypes][2][11]int32
}

func rdoqCostLiteral(n int) int32 { return int32(n) << entropy.ProbCostShift }

// fill is svt_aom_estimate_coefficients_rate over cdfs.
func (c *rdoqCosts) fill(cdfs *tile.CoeffCDFs) {
	cost := func(cdf *entropy.CDF, s int) int32 { return int32(entropy.CostSymbol(cdf, s)) }
	eobCDFs := [7]*[tile.CoeffPlaneTypes][2]entropy.CDF{
		(*[tile.CoeffPlaneTypes][2]entropy.CDF)(&cdfs.EOBFlag16),
		(*[tile.CoeffPlaneTypes][2]entropy.CDF)(&cdfs.EOBFlag32),
		(*[tile.CoeffPlaneTypes][2]entropy.CDF)(&cdfs.EOBFlag64),
		(*[tile.CoeffPlaneTypes][2]entropy.CDF)(&cdfs.EOBFlag128),
		(*[tile.CoeffPlaneTypes][2]entropy.CDF)(&cdfs.EOBFlag256),
		(*[tile.CoeffPlaneTypes][2]entropy.CDF)(&cdfs.EOBFlag512),
		(*[tile.CoeffPlaneTypes][2]entropy.CDF)(&cdfs.EOBFlag1024),
	}
	for m, set := range eobCDFs {
		for plane := range tile.CoeffPlaneTypes {
			for ctx := range 2 {
				cdf := &set[plane][ctx]
				for s := range min(cdf.Symbols(), 11) {
					c.eob[m][plane][ctx][s] = cost(cdf, s)
				}
			}
		}
	}
	for tx := range tile.CoeffTxSizeContexts {
		for plane := range tile.CoeffPlaneTypes {
			p := &c.coeff[tx][plane]
			for ctx := range tile.TXBSkipContexts {
				for s := range 2 {
					p.txbSkip[ctx][s] = cost(&cdfs.TXBSkip[tx][ctx], s)
				}
			}
			for ctx := range tile.EOBBaseContexts {
				for s := range 3 {
					p.baseEOB[ctx][s] = cost(&cdfs.CoeffBaseEOB[tx][plane][ctx], s)
				}
			}
			for ctx := range tile.CoeffBaseContexts {
				b := &p.base[ctx]
				for s := range 4 {
					b[s] = cost(&cdfs.CoeffBase[tx][plane][ctx], s)
				}
				b[4] = 0
				b[5] = b[1] + rdoqCostLiteral(1) - b[0]
				b[6] = b[2] - b[1]
				b[7] = b[3] - b[2]
			}
			for ctx := range tile.EOBCoefContexts {
				for s := range 2 {
					p.eobExtra[ctx][s] = cost(&cdfs.EOBExtra[tx][plane][ctx], s)
				}
			}
			for ctx := range 3 {
				for s := range 2 {
					p.dcSign[ctx][s] = cost(&cdfs.DCSign[plane][ctx], s)
				}
			}
			brTx := min(tx, 3)
			for ctx := range tile.CoeffBRContexts {
				var brRate [rdoqBRCDFSize]int32
				for s := range rdoqBRCDFSize {
					brRate[s] = cost(&cdfs.CoeffBR[brTx][plane][ctx], s)
				}
				lps := &p.lps[ctx]
				prev := int32(0)
				i := 0
				for ; i < rdoqCoeffBaseRange; i += rdoqBRCDFSize - 1 {
					j := 0
					for ; j < rdoqBRCDFSize-1; j++ {
						lps[i+j] = prev + brRate[j]
					}
					prev += brRate[j]
				}
				lps[i] = prev
				lps[rdoqCoeffBaseRange+1] = lps[0]
				for i := 1; i <= rdoqCoeffBaseRange; i++ {
					lps[i+rdoqCoeffBaseRange+1] = lps[i] - lps[i-1]
				}
			}
		}
	}
}

// rdoqTXB is the per-transform-block geometry svt_av1_optimize_b reads. The
// coded coefficient area is cw x ch (column-major, stride ch): a 64-point
// extent codes its low 32 (av1_get_adjusted_tx_size; SVT repacks them with
// svt_handle_transform64x64 and friends).
type rdoqTXB struct {
	size   tile.TransformSize
	class  transform.Class
	cw, ch int // coded width and height
	area   int // (height << bwl), the scan area
	txArea int // tx_size_wide * tx_size_high, 64-point extents included
	shift  int // av1_get_tx_scale_tab
	txCtx  int // get_txsize_entropy_ctx
	eobSel int // txsize_log2_minus4
	plane  int // plane type: 0 luma, 1 chroma
	scan   []int16
	iscan  []int16
}

func rdoqCostRD(rdmult int64, rate int64, dist int64) int64 {
	return ((rate*rdmult + 256) >> entropy.ProbCostShift) + dist<<7
}

func rdoqCoeffDist(t, dq int32, shift int) int64 {
	d := int64(t-dq) << shift
	return d * d
}

func rdoqGolombCost(absQC int32) int32 {
	if absQC >= 1+rdoqNumBaseLevels+rdoqCoeffBaseRange {
		r := absQC - rdoqCoeffBaseRange - rdoqNumBaseLevels
		length := bits.Len32(uint32(r))
		return rdoqCostLiteral(2*length - 1)
	}
	return 0
}

func rdoqBRCost(level int32, lps *[2*rdoqCoeffBaseRange + 2]int32) int32 {
	baseRange := min(level-1-rdoqNumBaseLevels, rdoqCoeffBaseRange)
	return lps[baseRange] + rdoqGolombCost(level)
}

var rdoqGolombBitsCost = [32]int32{0, 512, 512 * 3, 512 * 3, 512 * 5, 512 * 5, 512 * 5, 512 * 5,
	512 * 7, 512 * 7, 512 * 7, 512 * 7, 512 * 7, 512 * 7, 512 * 7, 512 * 7,
	512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9,
	512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9, 512 * 9}

var rdoqGolombCostDiff = [32]int32{0, 512, 512 * 2, 0, 512 * 2, 0, 0, 0, 512 * 2, 0, 0, 0, 0, 0, 0, 0,
	512 * 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

func rdoqBRCostWithDiff(level int32, lps *[2*rdoqCoeffBaseRange + 2]int32, diff *int32) int32 {
	baseRange := min(level-1-rdoqNumBaseLevels, rdoqCoeffBaseRange)
	golombBits := int32(0)
	if level <= rdoqCoeffBaseRange+1+rdoqNumBaseLevels {
		*diff += lps[baseRange+rdoqCoeffBaseRange+1]
	}
	if level >= rdoqCoeffBaseRange+1+rdoqNumBaseLevels {
		r := level - rdoqCoeffBaseRange - rdoqNumBaseLevels
		if r < 32 {
			golombBits = rdoqGolombBitsCost[r]
			*diff += rdoqGolombCostDiff[r]
		} else {
			golombBits = rdoqGolombCost(level)
			if r&(r-1) == 0 {
				*diff += 1024
			}
		}
	}
	return lps[baseRange] + golombBits
}

func (g *rdoqTXB) lowerLevelsCtx(levels []uint8, ci int) int {
	return tile.CoeffLowerLevelsContextTrusted(levels, g.size, g.class, ci)
}

func (g *rdoqTXB) lowerLevelsCtxEOB(si int) int {
	switch {
	case si == 0:
		return 0
	case si <= g.area/8:
		return 1
	case si <= g.area/4:
		return 2
	}
	return 3
}

func (g *rdoqTXB) brCtx(levels []uint8, ci int) int {
	return tile.CoeffBRContextTrusted(levels, g.size, g.class, ci)
}

func (g *rdoqTXB) brCtxEOB(ci int) int {
	ctx, _ := tile.CoeffBRContextEOB(g.size, g.class, ci)
	return ctx
}

// coeffCostGeneral is get_coeff_cost_general.
func (g *rdoqTXB) coeffCostGeneral(isLast bool, ci int, absQC int32, sign int, coeffCtx, dcSignCtx int, tc *rdoqCoeffCost, levels []uint8) int32 {
	var cost int32
	if isLast {
		cost += tc.baseEOB[coeffCtx][min(absQC, 3)-1]
	} else {
		cost += tc.base[coeffCtx][min(absQC, 3)]
	}
	if absQC != 0 {
		if ci == 0 {
			cost += tc.dcSign[dcSignCtx][sign]
		} else {
			cost += rdoqCostLiteral(1)
		}
		if absQC > rdoqNumBaseLevels {
			var br int
			if isLast {
				br = g.brCtxEOB(ci)
			} else {
				br = g.brCtx(levels, ci)
			}
			cost += rdoqBRCost(absQC, &tc.lps[br])
		}
	}
	return cost
}

// coeffCostEOB is get_coeff_cost_eob.
func (g *rdoqTXB) coeffCostEOB(ci int, absQC int32, sign int, coeffCtx, dcSignCtx int, tc *rdoqCoeffCost) int32 {
	cost := tc.baseEOB[coeffCtx][min(absQC, 3)-1]
	if absQC != 0 {
		if ci == 0 {
			cost += tc.dcSign[dcSignCtx][sign]
		} else {
			cost += rdoqCostLiteral(1)
		}
		if absQC > rdoqNumBaseLevels {
			cost += rdoqBRCost(absQC, &tc.lps[g.brCtxEOB(ci)])
		}
	}
	return cost
}

// eobCost is get_eob_cost.
func (g *rdoqTXB) eobCost(eob int, ec *[2][11]int32, tc *rdoqCoeffCost) int32 {
	pt, extra, _ := tile.EOBPositionToken(eob)
	multiCtx := 0
	if g.class != transform.Class2D {
		multiCtx = 1
	}
	cost := ec[multiCtx][pt-1]
	if offsetBits := rdoqEOBOffsetBits[pt]; offsetBits > 0 {
		bit := 0
		if extra&(1<<(offsetBits-1)) != 0 {
			bit = 1
		}
		cost += tc.eobExtra[pt-3][bit]
		if offsetBits > 1 {
			cost += rdoqCostLiteral(offsetBits - 1)
		}
	}
	return cost
}

// rdoqEOBOffsetBits is svt_aom_eob_offset_bits (k_eob_offset_bits) by EOB
// position token.
var rdoqEOBOffsetBits = [12]int{0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

func rdoqAbs(v int32) int32 {
	if v < 0 {
		return -v
	}
	return v
}

// rdoqState carries the in-place buffers of one optimize_b call.
type rdoqState struct {
	g       *rdoqTXB
	tc      *rdoqCoeffCost
	ec      *[2][11]int32
	rdmult  int64
	scan    []int16
	tcoeff  []int32
	qcoeff  []int16
	dqcoeff []int32
	levels  []uint8
	dqDC    int32
	dqAC    int32
}

func (s *rdoqState) dqv(ci int) int32 {
	if ci == 0 {
		return s.dqDC
	}
	return s.dqAC
}

func (s *rdoqState) setLevel(ci int, v int32) {
	s.levels[tile.CoeffPaddedIndex(s.g.size, ci)] = uint8(min(v, 127))
}

func (s *rdoqState) qcLowDQ(absQC int32, sign int, dqv int32) (int16, int32) {
	absLow := absQC - 1
	absDQLow := (absLow * dqv) >> s.g.shift
	if sign != 0 {
		return int16(-absLow), -absDQLow
	}
	return int16(absLow), absDQLow
}

// updateCoeffGeneral is update_coeff_general.
func (s *rdoqState) updateCoeffGeneral(accuRate *int64, accuDist *int64, si, eob, dcSignCtx int) {
	g, tc := s.g, s.tc
	ci := int(s.scan[si])
	dqv := s.dqv(ci)
	qc := int32(s.qcoeff[ci])
	isLast := si == eob-1
	var coeffCtx int
	if isLast {
		coeffCtx = g.lowerLevelsCtxEOB(si)
	} else {
		coeffCtx = g.lowerLevelsCtx(s.levels, ci)
	}
	if qc == 0 {
		*accuRate += int64(tc.base[coeffCtx][0])
		return
	}
	sign := 0
	if qc < 0 {
		sign = 1
	}
	absQC := rdoqAbs(qc)
	tqc := s.tcoeff[ci]
	dqc := s.dqcoeff[ci]
	dist := rdoqCoeffDist(tqc, dqc, g.shift)
	dist0 := rdoqCoeffDist(tqc, 0, g.shift)
	rate := g.coeffCostGeneral(isLast, ci, absQC, sign, coeffCtx, dcSignCtx, tc, s.levels)
	rd := rdoqCostRD(s.rdmult, int64(rate), dist)

	var qcLow int16
	var dqcLow, absQCLow int32
	var distLow int64
	var rateLow int32
	if absQC == 1 {
		distLow = dist0
		rateLow = tc.base[coeffCtx][0]
	} else {
		qcLow, dqcLow = s.qcLowDQ(absQC, sign, dqv)
		absQCLow = absQC - 1
		distLow = rdoqCoeffDist(tqc, dqcLow, g.shift)
		rateLow = g.coeffCostGeneral(isLast, ci, absQCLow, sign, coeffCtx, dcSignCtx, tc, s.levels)
	}
	rdLow := rdoqCostRD(s.rdmult, int64(rateLow), distLow)
	if rdLow < rd {
		s.qcoeff[ci] = qcLow
		s.dqcoeff[ci] = dqcLow
		s.setLevel(ci, absQCLow)
		*accuRate += int64(rateLow)
		*accuDist += distLow - dist0
	} else {
		*accuRate += int64(rate)
		*accuDist += dist - dist0
	}
}

// updateCoeffEOB is update_coeff_eob.
func (s *rdoqState) updateCoeffEOB(accuRate *int64, accuDist *int64, eob *int, nzNum *int, nzCI *[rdoqMaxNZ + 1]int, si, dcSignCtx int) {
	g, tc := s.g, s.tc
	ci := int(s.scan[si])
	dqv := s.dqv(ci)
	qc := int32(s.qcoeff[ci])
	coeffCtx := g.lowerLevelsCtx(s.levels, ci)
	if qc == 0 {
		*accuRate += int64(tc.base[coeffCtx][0])
		return
	}
	lowerLevel := false
	absQC := rdoqAbs(qc)
	tqc := s.tcoeff[ci]
	dqc := s.dqcoeff[ci]
	sign := 0
	if qc < 0 {
		sign = 1
	}
	dist0 := rdoqCoeffDist(tqc, 0, g.shift)
	dist := rdoqCoeffDist(tqc, dqc, g.shift) - dist0
	rate := int64(g.coeffCostGeneral(false, ci, absQC, sign, coeffCtx, dcSignCtx, tc, s.levels))
	rd := rdoqCostRD(s.rdmult, *accuRate+rate, *accuDist+dist)

	var qcLow int16
	var dqcLow, absQCLow int32
	var distLow, rdLow int64
	var rateLow int64
	if absQC == 1 {
		rateLow = int64(tc.base[coeffCtx][0])
		rdLow = rdoqCostRD(s.rdmult, *accuRate+rateLow, *accuDist)
	} else {
		qcLow, dqcLow = s.qcLowDQ(absQC, sign, dqv)
		absQCLow = absQC - 1
		distLow = rdoqCoeffDist(tqc, dqcLow, g.shift) - dist0
		rateLow = int64(g.coeffCostGeneral(false, ci, absQCLow, sign, coeffCtx, dcSignCtx, tc, s.levels))
		rdLow = rdoqCostRD(s.rdmult, *accuRate+rateLow, *accuDist+distLow)
	}

	lowerLevelNewEOB := false
	newEOB := si + 1
	coeffCtxNewEOB := g.lowerLevelsCtxEOB(si)
	newEOBCost := int64(g.eobCost(newEOB, s.ec, tc))
	rateCoeffEOB := newEOBCost + int64(g.coeffCostEOB(ci, absQC, sign, coeffCtxNewEOB, dcSignCtx, tc))
	distNewEOB := dist
	rdNewEOB := rdoqCostRD(s.rdmult, rateCoeffEOB, distNewEOB)
	if absQCLow > 0 {
		rateCoeffEOBLow := newEOBCost + int64(g.coeffCostEOB(ci, absQCLow, sign, coeffCtxNewEOB, dcSignCtx, tc))
		rdNewEOBLow := rdoqCostRD(s.rdmult, rateCoeffEOBLow, distLow)
		if rdNewEOBLow < rdNewEOB {
			lowerLevelNewEOB = true
			rdNewEOB = rdNewEOBLow
			rateCoeffEOB = rateCoeffEOBLow
			distNewEOB = distLow
		}
	}
	if rdLow < rd {
		lowerLevel = true
		rd = rdLow
		rate = rateLow
		dist = distLow
	}
	if rdNewEOB < rd {
		for ni := 0; ni < *nzNum; ni++ {
			last := nzCI[ni]
			s.levels[tile.CoeffPaddedIndex(g.size, last)] = 0
			s.qcoeff[last] = 0
			s.dqcoeff[last] = 0
		}
		*eob = newEOB
		*nzNum = 0
		*accuRate = rateCoeffEOB
		*accuDist = distNewEOB
		lowerLevel = lowerLevelNewEOB
	} else {
		*accuRate += rate
		*accuDist += dist
	}
	if lowerLevel {
		s.qcoeff[ci] = qcLow
		s.dqcoeff[ci] = dqcLow
		s.setLevel(ci, absQCLow)
	}
	if s.qcoeff[ci] != 0 {
		nzCI[*nzNum] = ci
		*nzNum++
	}
}

// updateCoeffSimple is update_coeff_simple with get_two_coeff_cost_simple.
func (s *rdoqState) updateCoeffSimple(accuRate *int64, si int) {
	g, tc := s.g, s.tc
	ci := int(s.scan[si])
	dqv := s.dqv(ci)
	qc := int32(s.qcoeff[ci])
	coeffCtx := g.lowerLevelsCtx(s.levels, ci)
	if qc == 0 {
		*accuRate += int64(tc.base[coeffCtx][0])
		return
	}
	absQC := rdoqAbs(qc)
	absTQC := rdoqAbs(s.tcoeff[ci])
	absDQC := rdoqAbs(s.dqcoeff[ci])
	// get_two_coeff_cost_simple
	cost := tc.base[coeffCtx][min(absQC, 3)]
	diff := int32(0)
	if absQC <= 3 {
		diff = tc.base[coeffCtx][absQC+4]
	}
	cost += rdoqCostLiteral(1)
	if absQC > rdoqNumBaseLevels {
		br := g.brCtx(s.levels, ci)
		brDiff := int32(0)
		cost += rdoqBRCostWithDiff(absQC, &tc.lps[br], &brDiff)
		diff += brDiff
	}
	rateLow := cost - diff
	if absDQC < absTQC {
		*accuRate += int64(cost)
		return
	}
	dist := rdoqCoeffDist(absTQC, absDQC, g.shift)
	rd := rdoqCostRD(s.rdmult, int64(cost), dist)
	absQCLow := absQC - 1
	absDQCLow := (absQCLow * dqv) >> g.shift
	distLow := rdoqCoeffDist(absTQC, absDQCLow, g.shift)
	rdLow := rdoqCostRD(s.rdmult, int64(rateLow), distLow)
	if rdLow < rd {
		if qc < 0 {
			s.qcoeff[ci] = int16(-absQCLow)
			s.dqcoeff[ci] = -absDQCLow
		} else {
			s.qcoeff[ci] = int16(absQCLow)
			s.dqcoeff[ci] = absDQCLow
		}
		s.setLevel(ci, absQCLow)
		*accuRate += int64(rateLow)
	} else {
		*accuRate += int64(cost)
	}
}

// optimizeB is svt_av1_optimize_b with sharpness 0 and full RDOQ
// (rdoq_ctrls cut_off_num 0). qcoeff holds the fp-quantized levels of tcoeff
// and eob their end of block; it rewrites qcoeff in place and returns the new
// end of block.
func (s *rdoqState) optimizeB(eob, txbSkipCtx, dcSignCtx int) int {
	g, tc := s.g, s.tc
	nonSkipCost := int64(tc.txbSkip[txbSkipCtx][0])
	skipCost := int64(tc.txbSkip[txbSkipCtx][1])
	for si := range eob {
		ci := int(s.scan[si])
		qc := int32(s.qcoeff[ci])
		if qc == 0 {
			continue
		}
		dq := (rdoqAbs(qc) * s.dqv(ci)) >> g.shift
		if qc < 0 {
			dq = -dq
		}
		s.dqcoeff[ci] = dq
	}
	_ = tile.CoeffInitLevels(s.qcoeff, g.size, s.levels)
	accuRate := int64(g.eobCost(eob, s.ec, tc))
	var accuDist int64
	si := eob - 1
	ci := int(s.scan[si])
	qc := int32(s.qcoeff[ci])
	absQC := rdoqAbs(qc)
	nzNum := 1
	nzCI := [rdoqMaxNZ + 1]int{ci}
	if absQC >= 2 {
		s.updateCoeffGeneral(&accuRate, &accuDist, si, eob, dcSignCtx)
		si--
	} else {
		sign := 0
		if qc < 0 {
			sign = 1
		}
		coeffCtx := g.lowerLevelsCtxEOB(si)
		accuRate += int64(g.coeffCostEOB(ci, absQC, sign, coeffCtx, dcSignCtx, tc))
		tqc := s.tcoeff[ci]
		accuDist += rdoqCoeffDist(tqc, s.dqcoeff[ci], g.shift) - rdoqCoeffDist(tqc, 0, g.shift)
		si--
	}
	for ; si >= 0 && nzNum <= rdoqMaxNZ; si-- {
		s.updateCoeffEOB(&accuRate, &accuDist, &eob, &nzNum, &nzCI, si, dcSignCtx)
	}
	if si == -1 && nzNum <= rdoqMaxNZ {
		// update_skip
		rd := rdoqCostRD(s.rdmult, accuRate+nonSkipCost, accuDist)
		rdNewEOB := rdoqCostRD(s.rdmult, skipCost, 0)
		if rdNewEOB < rd {
			for i := 0; i < nzNum; i++ {
				s.qcoeff[nzCI[i]] = 0
				s.dqcoeff[nzCI[i]] = 0
			}
			return 0
		}
	}
	for ; si >= 1; si-- {
		s.updateCoeffSimple(&accuRate, si)
	}
	if si == 0 {
		var dummy int64
		s.updateCoeffGeneral(&accuRate, &dummy, si, eob, dcSignCtx)
	}
	return eob
}

// rdoqGeometries holds the RDOQ geometry of every DCT_DCT transform size by
// plane type, with its default 2D scan and inverse scan.
var rdoqGeometries = func() (g [2][int(tile.TransformSize64x16) + 1]rdoqTXB) {
	for s := range g[0] {
		size := tile.TransformSize(s)
		txSize, err := size.TransformSize()
		if err != nil {
			panic(err)
		}
		coded, err := transform.ScanSize(txSize)
		if err != nil {
			panic(err)
		}
		txScale, err := quantize.TransformScale(int(txSize.Width), int(txSize.Height))
		if err != nil {
			panic(err)
		}
		txCtx, err := tile.CoeffTransformSizeContext(size)
		if err != nil {
			panic(err)
		}
		cw, ch := int(coded.Width), int(coded.Height)
		area := cw * ch
		scan := make([]int16, area)
		iscan := make([]int16, area)
		if err := transform.FillDefaultScan(scan, iscan, txSize, transform.Class2D); err != nil {
			panic(err)
		}
		for si, ci := range scan {
			iscan[ci] = int16(si)
		}
		for plane := range 2 {
			g[plane][s] = rdoqTXB{
				size:   size,
				class:  transform.Class2D,
				cw:     cw,
				ch:     ch,
				area:   area,
				txArea: int(txSize.Width) * int(txSize.Height),
				shift:  int(txScale),
				txCtx:  int(txCtx),
				eobSel: bits.Len(uint(area)) - 1 - 4,
				plane:  plane,
				scan:   scan,
				iscan:  iscan,
			}
		}
	}
	return g
}()

// rdoqGeometry returns the RDOQ geometry of a DCT_DCT transform block.
func rdoqGeometry(size tile.TransformSize, plane int) *rdoqTXB {
	return &rdoqGeometries[plane][size]
}

// rdoqSquareGeometry returns the RDOQ geometry of an n x n DCT_DCT block.
func rdoqSquareGeometry(n int, plane int) (*rdoqTXB, bool) {
	switch n {
	case 4:
		return rdoqGeometry(tile.TransformSize4x4, plane), true
	case 8:
		return rdoqGeometry(tile.TransformSize8x8, plane), true
	case 16:
		return rdoqGeometry(tile.TransformSize16x16, plane), true
	case 32:
		return rdoqGeometry(tile.TransformSize32x32, plane), true
	case 64:
		return rdoqGeometry(tile.TransformSize64x64, plane), true
	}
	return nil, false
}

// rdoqQuantize quantizes the n x n DCT_DCT coefficients tran into qcoeff with
// SVT's RDOQ pipeline (svt_aom_quantize_inv_quantize with perform_rdoq: the
// fp quantizer, then svt_av1_optimize_b) and reports whether every level is
// zero. The caller has armed st.rdoq.
func (st *lossyEncodeState) rdoqQuantize(tran []int32, qcoeff []int16, n int, plane int, q quantize.Quantizer) bool {
	g, ok := rdoqSquareGeometry(n, plane)
	if !ok {
		return false
	}
	return st.rdoqQuantizeTX(tran, qcoeff, g, q)
}

// rdoqQuantizeTX is rdoqQuantize for the transform geometry g (any DCT_DCT
// size, 64-point extents coding their low 32).
func (st *lossyEncodeState) rdoqQuantizeTX(tran []int32, qcoeff []int16, g *rdoqTXB, q quantize.Quantizer) bool {
	plane := g.plane
	if err := quantize.QuantizeBlockScaledFP(qcoeff[:g.area], g.ch, tran[:g.area], g.ch, g.cw, g.ch, q, uint8(g.shift)); err != nil {
		return false
	}
	eob := rdoqEOBFromIScan(qcoeff[:g.area], g.iscan)
	if eob == 0 {
		return true
	}
	s := rdoqState{
		g:       g,
		tc:      &st.rdoq.coeff[g.txCtx][plane],
		ec:      &st.rdoq.eob[g.eobSel][plane],
		rdmult:  st.rdoqLambda(plane),
		scan:    g.scan,
		tcoeff:  tran,
		qcoeff:  qcoeff,
		dqcoeff: st.rdoqDQ[:],
		levels:  st.levels,
		dqDC:    q.DC,
		dqAC:    q.AC,
	}
	return s.optimizeB(eob, 0, 0) == 0
}

// rdoqPlaneRDMult is SVT's plane_rd_mult[allintra || rtc][is_inter] row for
// rtc inter blocks: luma 17, chroma 13.
var rdoqPlaneRDMult = [2]int64{17, 13}

// rdoqLambda is optimize_b's rdmult: SVT's full lambda (svt_aom_compute_rd_mult
// for a flat-IPP realtime inter frame: the inter rdmult times
// rd_frame_type_factor[8-bit][ARF_UPDATE] = 150/128, following the
// superblock quantizer through st.rdMult) scaled by plane_rd_mult (rweight
// 100, rshift 2 at sharpness 0). Preset 12 rtc runs RDOQ on intra pictures
// only (pcs->rdoq_level = is_islice), so inter blocks never reach it.
func (st *lossyEncodeState) rdoqLambda(plane int) int64 {
	full := (st.rdMult * 150) >> 7
	return (full*rdoqPlaneRDMult[plane] + 2) >> 2
}

// armRDOQ builds the coefficient rate tables for the frame's qindex (the
// trial CDFs are the qindex defaults).
func (st *lossyEncodeState) armRDOQ() bool {
	if !st.armTrial() {
		return false
	}
	if !st.rdoqReady || st.rdoqQIndex != st.qIndex {
		st.rdoq.fill(&st.trialCDFs)
		st.rdoqQIndex = st.qIndex
		st.rdoqReady = true
	}
	return true
}

// setRDOQArm arms RDOQ for the following inter TXBs of plane type plane when
// the block runs it (st.rdoqBlock).
func (st *lossyEncodeState) setRDOQArm(plane int8) {
	st.rdoqArm = 0
	if st.rdoqBlock {
		st.rdoqArm = plane + 1
	}
}

// svtDepth0TXLeaf is the luma transform leaf of SVT's light-PD1 TX path at
// tx_depth 0 (perform_dct_dct_tx_light_pd1: tx_depth_to_tx_size[0][bsize]):
// the block's largest rectangular transform (max_txsize_rect_lookup: TX_64X64
// for a 64x64 block, TX_32X16 for 32x16, TX_64X16 for 64x16), never split.
// The leaf returned covers the block's longer side so the transform tree has
// no split; blocks under 8x8 keep their 8x8 floor.
func svtDepth0TXLeaf(block tile.BlockSize) (int, bool) {
	dims, ok := block.Dimensions()
	if !ok {
		return 0, false
	}
	leaf := 4 << max(int(dims.Log2W), int(dims.Log2H))
	return min(max(leaf, 8), 64), true
}

// prepareInterTXBDecideRDOQ is prepareInterTXBDecide for an RDOQ-armed TXB
// whose residual is in resScratch: the row-skipping DCT at the fp zero bins
// (the quantizer RDOQ starts from), the RDOQ quantizer, and the skip
// comparison's difference terms of the surviving levels.
func (st *lossyEncodeState) prepareInterTXBDecideRDOQ(geo *txBlockGeometry, w int, q quantize.Quantizer, qcoeff []int16) bool {
	dcZero, acZero := quantize.ZeroBinFP(q, geo.txScale)
	thr0, thr := min(dcZero, acZero), acZero
	var mask uint8
	switch w {
	case 8:
		mask = transform.ForwardDCT8x8RowSkip8BitTrusted((*[64]int32)(st.tranScratch[:64]), (*[64]int16)(st.resScratch[:64]), thr0, thr)
	case 16:
		mask = transform.ForwardDCT16x16RowSkip8BitTrusted((*[256]int32)(st.tranScratch[:256]), (*[256]int16)(st.resScratch[:256]), thr0, thr)
	default:
		mask = transform.ForwardDCT32x32RowSkip8BitTrusted((*[1024]int32)(st.tranScratch[:1024]), (*[1024]int16)(st.resScratch[:1024]), thr0, thr)
	}
	cn := geo.coeffCount
	if st.pfN4 {
		svtPFN4Zero(st.tranScratch[:cn], w, w, geo.coeffWidth, geo.coeffHeight)
		mask = svtPFN4Mask(mask, w)
	}
	if mask == 0 {
		clear(qcoeff[:cn])
		return true
	}
	if st.rdoqQuantize(st.tranScratch[:cn], qcoeff[:cn], w, int(st.rdoqArm-1), q) {
		return true
	}
	return st.accumulateInterTXBStatsDiff(geo, q, qcoeff)
}

// rdoqEOBFromIScan returns the end of block of qcoeff: one past the largest
// scan position of a non-zero level (iscan maps a coefficient to its scan
// position).
func rdoqEOBFromIScan(qcoeff []int16, iscan []int16) int {
	return rdoqEOBImpl(qcoeff, iscan)
}

var rdoqEOBImpl = rdoqEOBPureGo

// rdoqEOBPureGo skips runs of four zero levels by one 64-bit test.
func rdoqEOBPureGo(qcoeff []int16, iscan []int16) int {
	eob := 0
	n := len(qcoeff) &^ 3
	iscan = iscan[:len(qcoeff)]
	for i := 0; i < n; i += 4 {
		if *(*uint64)(unsafe.Pointer(&qcoeff[i])) == 0 {
			continue
		}
		for j := i; j < i+4; j++ {
			if qcoeff[j] != 0 {
				eob = max(eob, int(iscan[j])+1)
			}
		}
	}
	for i := n; i < len(qcoeff); i++ {
		if qcoeff[i] != 0 {
			eob = max(eob, int(iscan[i])+1)
		}
	}
	return eob
}

// svtCoeffCost is svt_av1_cost_coeffs_txb without CDF updates for one n x n
// DCT_DCT TXB of plane type plane (txb_skip_ctx 0 and dc_sign_ctx 0, as the
// light-PD1 path passes), in 512-per-bit units. It overwrites st.levels.
func (st *lossyEncodeState) svtCoeffCost(qcoeff []int16, size tile.TransformSize, plane int) int64 {
	g := rdoqGeometry(size, plane)
	return st.svtCoeffCostEOB(qcoeff, g, rdoqEOBFromIScan(qcoeff[:g.area], g.iscan))
}

// svtCoeffCostEOB is svtCoeffCost for a TXB whose end of block is known.
func (st *lossyEncodeState) svtCoeffCostEOB(qcoeff []int16, g *rdoqTXB, eob int) int64 {
	plane := g.plane
	tc := &st.rdoq.coeff[g.txCtx][plane]
	if eob == 0 {
		return int64(tc.txbSkip[0][1])
	}
	scan := g.scan
	levels := st.levels
	_ = tile.CoeffInitLevels(qcoeff, g.size, levels)
	cost := int64(tc.txbSkip[0][0])
	cost += int64(g.eobCost(eob, &st.rdoq.eob[g.eobSel][plane], tc))
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
			cost += int64(tc.dcSign[0][sign])
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

// svtLumaCoeffBits is perform_dct_dct_tx_light_pd1's luma rate
// (product_coding_loop.c:5268-5275): 6000 + 1000*eob for a TXB with eob
// below (txbwidth * txbheight) >> 6 (the full transform area, 64-point
// extents included) at coeff_rate_est_lvl 2 (rate_est_level 4, LPD1_LVL_0)
// and 0 (rate_est_level 0, LPD1_LVL_1 and above); otherwise the table cost
// at level 2 and 6000 + 400*eob at level 0 (lvl0).
func (st *lossyEncodeState) svtLumaCoeffBits(qcoeff []int16, size tile.TransformSize, lvl0 bool) int64 {
	g := rdoqGeometry(size, 0)
	eob := rdoqEOBFromIScan(qcoeff[:g.area], g.iscan)
	if eob < g.txArea>>6 {
		return 6000 + int64(eob)*1000
	}
	if lvl0 {
		return 6000 + int64(eob)*400
	}
	return st.svtCoeffCostEOB(qcoeff, g, eob)
}

// svtInterCoeffCost prices the coded alternative of the skip decision the
// light-PD1 way (svtLumaCoeffBits per luma TXB, svtCoeffCost per chroma TXB),
// scaled by full_lambda (lpd1FullLambda).
func (st *lossyEncodeState) svtInterCoeffCost(txPlan realtimeInterTXPlan, lumaTX, chromaTX tile.TransformSize, hasChroma bool, lumaLvl0 bool, lambda int64) (int64, bool) {
	var cost int64
	if txPlan.Variable() {
		leaf := txPlan.leafSize
		area := leaf * leaf
		if err := txPlan.ForEachLeaf(func(i, dx, dy int) error {
			cost += st.svtLumaCoeffBits(st.lumaQ2[i*area:(i+1)*area], txPlan.leafTX, lumaLvl0)
			return nil
		}); err != nil {
			return 0, false
		}
	} else {
		cost += st.svtLumaCoeffBits(st.lumaQ[:], lumaTX, lumaLvl0)
	}
	if hasChroma {
		cost += st.svtCoeffCost(st.uQ[:], chromaTX, 1)
		cost += st.svtCoeffCost(st.vQ[:], chromaTX, 1)
	}
	return (cost*lambda + 256) >> 9, true
}
