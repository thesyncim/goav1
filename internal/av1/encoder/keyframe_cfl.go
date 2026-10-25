package encoder

import (
	"sync"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/prediction"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// keyframe_cfl.go ports SVT-AV1's chroma-from-luma search for intra slices.
// At the rtc presets (enc_mode_config.c svt_aom_sig_deriv_mode_decision_config_rtc)
// cfl_level is 2 on I slices (is_islice ? 2 : 0 above M6; screen content uses
// is_base ? 2 : 0, which a keyframe also is): CflCtrls enabled with
// itr_th 1 (set_cfl_ctrls). product_coding_loop.c cfl_prediction then, without
// independent chroma results (ind_uv_avail is 0 on this path), prices the
// block's UV_DC_PRED chroma, searches the CfL alphas (md_cfl_rd_pick_alpha)
// and keeps CfL when its cost is lower.

// CfL joint-sign helpers (av1/common/cfl.h).
const (
	cflSigns      = 3
	cflJointSigns = cflSigns*cflSigns - 1
	cflAlphabet   = 16
	cflSignZero   = 0
	cflSignNeg    = 1
	cflSignPos    = 2
)

// keyIntraModeCount is the number of luma intra modes (DC_PRED..PAETH_PRED).
const keyIntraModeCount = int(tile.IntraModePaeth) + 1

func cflSignU(js int) int    { return ((js + 1) * 11) >> 5 }
func cflSignV(js int) int    { return (js + 1) - cflSigns*cflSignU(js) }
func cflContextU(js int) int { return js + 1 - cflSigns }
func cflContextV(js int) int { return cflSignV(js)*cflSigns + cflSignU(js) - cflSigns }
func cflPlaneSignToJoint(plane, a, b int) int {
	if plane == 0 {
		return a*cflSigns + b - 1
	}
	return b*cflSigns + a - 1
}

// keyCFLRateTable is the part of md_rate_estimation.c's MdRateEstimationContext
// the keyframe CfL search reads: intra_uv_mode_fac_bits[CFL_ALLOWED][y][uv]
// for UV_DC_PRED and UV_CFL_PRED, and cfl_alpha_fac_bits (the joint-sign cost
// folded into the U entries). A keyframe starts from the default CDFs, so the
// frame-start table is the same for every keyframe.
type keyCFLRateTable struct {
	ok    bool
	uvDC  [keyIntraModeCount]int64
	uvCFL [keyIntraModeCount]int64
	alpha [cflJointSigns][2][cflAlphabet]int64
}

var (
	keyCFLRatesOnce sync.Once
	keyCFLRates     keyCFLRateTable
)

func keyCFLRateTables() *keyCFLRateTable {
	keyCFLRatesOnce.Do(func() {
		var c tile.IntraModeCDFs
		if c.InitDefault() != nil {
			return
		}
		t := &keyCFLRates
		for m := tile.IntraMode(0); int(m) < keyIntraModeCount; m++ {
			cdf, err := c.UVModeCDF(true, m)
			if err != nil {
				return
			}
			t.uvDC[m] = int64(entropy.CostSymbol(cdf, int(tile.ChromaIntraModeDC)))
			t.uvCFL[m] = int64(entropy.CostSymbol(cdf, int(tile.ChromaIntraModeCFL)))
		}
		signCDF, err := c.CFLSignCDF()
		if err != nil {
			return
		}
		for js := range cflJointSigns {
			for plane, sign := range [2]int{cflSignU(js), cflSignV(js)} {
				if sign == cflSignZero {
					continue
				}
				ctx := cflContextU(js)
				if plane == 1 {
					ctx = cflContextV(js)
				}
				cdf, err := c.CFLAlphaCDF(ctx)
				if err != nil {
					return
				}
				for a := range cflAlphabet {
					t.alpha[js][plane][a] = int64(entropy.CostSymbol(cdf, a))
				}
			}
			s := int64(entropy.CostSymbol(signCDF, js))
			for a := range cflAlphabet {
				t.alpha[js][0][a] += s
			}
		}
		t.ok = true
	})
	return &keyCFLRates
}

// rateTerm is RDCOST's rate term for a rate in 1/512 bits under the keyframe
// coder's multiplier.
func (st *lossyEncodeState) rateTerm(rate512 int64) int64 {
	return (rate512*st.rdMult + 256) >> 9
}

// keyCFLPlaneCost is av1_cost_calc_cfl for one chroma plane: the CfL
// prediction at alphaQ3 over the plane's DC prediction dc, then the block's
// coefficient rate and distortion.
func (st *lossyEncodeState) keyCFLPlaneCost(src SourceFrame420, plane int, g *keyChromaGeom, dc uint8, alphaQ3 int) int64 {
	data, q, qc := src.U, st.uQuant, st.uQ[:]
	if plane == 2 {
		data, q, qc = src.V, st.vQuant, st.vQ[:]
	}
	n := g.cw * g.ch
	pred := st.cflPred[:n]
	if !st.keyCFLPredict(pred, g, dc, alphaQ3) {
		return 1 << 59
	}
	st.rdDcode = 0
	st.prepareInterTXB(data, pred, g.cw, src.ChromaStride, g.x, g.y, g.cw, g.ch, q, qc[:n])
	return st.trialTXBBits(tile.CoeffPlaneUV, qc[:n], g.cw) + st.rdDcode<<7
}

// keyCFLPredict writes the CfL prediction (cfl_predict_lbd over the DC
// prediction) into pred.
func (st *lossyEncodeState) keyCFLPredict(pred []byte, g *keyChromaGeom, dc uint8, alphaQ3 int) bool {
	memsetBytes(pred, dc)
	if alphaQ3 == 0 {
		return true
	}
	p := frame.Plane{Pix: pred, Stride: g.cw, Width: g.cw, Height: g.ch}
	return prediction.PredictCFLPlaneBlock(p, 1, 8, 0, 0, g.cw, g.ch, st.cflAC[:], alphaQ3) == nil
}

// keyCFLDecide is cfl_prediction for a keyframe leaf whose luma is
// reconstructed: the UV_DC_PRED reference cost, the AC of the reconstructed
// luma (compute_cfl_ac_components) and md_cfl_rd_pick_alpha. It reports
// whether CfL wins and its alphas; dcU / dcV are the planes' DC predictions.
func (st *lossyEncodeState) keyCFLDecide(src SourceFrame420, recon *SourceFrame420, g *keyChromaGeom, lumaMode tile.IntraMode, lumaPX, lumaPY, n int, dcU, dcV uint8) (bool, tile.CFLAlphaResult) {
	rates := keyCFLRateTables()
	if !rates.ok || g.cw != n>>1 || g.ch != n>>1 {
		return false, tile.CFLAlphaResult{}
	}
	dcs := [3]uint8{0, dcU, dcV}
	// Reference: UV_DC_PRED for both planes plus its mode rate. A zero alpha
	// predicts the plane's DC as well, so these plane costs are also the
	// alpha-zero entries of the search below.
	var zeroCost [2]int64
	nonCFL := st.rateTerm(rates.uvDC[lumaMode])
	for plane := range 2 {
		zeroCost[plane] = st.keyCFLPlaneCost(src, plane+1, g, dcs[plane+1], 0)
		nonCFL += zeroCost[plane]
	}
	// compute_cfl_ac_components: 4:2:0 luma subsampling of the reconstructed
	// block and its average removal.
	lumaOff := lumaPY*src.YStride + lumaPX
	if prediction.SubsampleLuma8ToQ3(st.cflQ3[:], recon.Y[lumaOff:], src.YStride, n, n, true, true) != nil ||
		prediction.SubtractCFLAverage(st.cflQ3[:], st.cflAC[:], g.cw, g.ch) != nil {
		return false, tile.CFLAlphaResult{}
	}

	const maxCost = int64(1) << 62
	modeRD := st.rateTerm(rates.uvCFL[lumaMode])
	var bestRDUV [cflJointSigns][2]int64
	var bestC [cflJointSigns][2]int
	for js := range bestRDUV {
		bestRDUV[js] = [2]int64{maxCost, maxCost}
	}
	for plane := range 2 {
		// Alpha zero in this plane, priced under both joint signs that leave
		// the plane's own sign zero.
		dist := zeroCost[plane]
		jsNeg := cflPlaneSignToJoint(plane, cflSignZero, cflSignNeg)
		jsPos := cflPlaneSignToJoint(plane, cflSignZero, cflSignPos)
		bestRDUV[jsNeg][plane] = dist + st.rateTerm(rates.alpha[jsNeg][plane][0])
		bestRDUV[jsPos][plane] = dist + st.rateTerm(rates.alpha[jsPos][plane][0])
	}
	const itrTH = 1 // set_cfl_ctrls level 2
	bestRD := maxCost
	bestJS, found := 0, false
	for plane := range 2 {
		for pnSign := cflSignNeg; pnSign < cflSigns; pnSign++ {
			progress := 0
			for c := range cflAlphabet {
				flag := 0
				if c > itrTH && progress < c {
					break
				}
				alphaQ3 := c + 1
				if pnSign == cflSignNeg {
					alphaQ3 = -alphaQ3
				}
				coeffDist := st.keyCFLPlaneCost(src, plane+1, g, dcs[plane+1], alphaQ3)
				for i := range cflSigns {
					js := cflPlaneSignToJoint(plane, pnSign, i)
					thisRD := coeffDist + st.rateTerm(rates.alpha[js][plane][c])
					if thisRD >= bestRDUV[js][plane] {
						continue
					}
					bestRDUV[js][plane] = thisRD
					bestC[js][plane] = c
					flag = itrTH
					if bestRDUV[js][1-plane] == maxCost {
						continue
					}
					thisRD += modeRD + bestRDUV[js][1-plane]
					if thisRD >= bestRD {
						continue
					}
					bestRD, bestJS, found = thisRD, js, true
				}
				progress += flag
			}
		}
	}
	if !found || bestRD >= nonCFL {
		return false, tile.CFLAlphaResult{}
	}
	u, v := bestC[bestJS][0], bestC[bestJS][1]
	return true, tile.CFLAlphaResult{Index: uint8(u<<4 + v), JointSign: int8(bestJS)}
}
