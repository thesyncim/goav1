package transform

import (
	"math"
	"testing"
)

// adst16F follows inverseADST16Core in adst.go stage for stage, with the
// (c-4096)*x + x forms written as the plain rotations they equal.
func adst16F(c []float64, stride int, g *gainRecorder) {
	var in [16]float64
	for i := range in {
		in[i] = c[i*stride]
	}
	rot := func(a, ca, b, cb float64) float64 { return g.rot(rs(a*ca+b*cb, 12)) }
	var t, ta [16]float64
	t[0] = rot(in[15], 4091, in[0], 201)
	t[1] = rot(in[15], 201, in[0], -4091)
	t[2] = rot(in[13], 3973, in[2], 995)
	t[3] = rot(in[13], 995, in[2], -3973)
	t[4] = rot(in[11], 3703, in[4], 1751)
	t[5] = rot(in[11], 1751, in[4], -3703)
	t[6] = rot(in[9], 3290, in[6], 2440)
	t[7] = rot(in[9], 2440, in[6], -3290)
	t[8] = rot(in[7], 2751, in[8], 3035)
	t[9] = rot(in[7], 3035, in[8], -2751)
	t[10] = rot(in[5], 2106, in[10], 3513)
	t[11] = rot(in[5], 3513, in[10], -2106)
	t[12] = rot(in[3], 1380, in[12], 3857)
	t[13] = rot(in[3], 3857, in[12], -1380)
	t[14] = rot(in[1], 601, in[14], 4052)
	t[15] = rot(in[1], 4052, in[14], -601)
	for i := 0; i < 8; i++ {
		ta[i] = t[i] + t[i+8]
		ta[i+8] = t[i] - t[i+8]
	}
	t[8] = rot(ta[8], 4017, ta[9], 799)
	t[9] = rot(ta[8], 799, ta[9], -4017)
	t[10] = rot(ta[10], 2276, ta[11], 3406)
	t[11] = rot(ta[10], 3406, ta[11], -2276)
	t[12] = rot(ta[13], 4017, ta[12], -799)
	t[13] = rot(ta[13], 799, ta[12], 4017)
	t[14] = rot(ta[15], 2276, ta[14], -3406)
	t[15] = rot(ta[15], 3406, ta[14], 2276)
	for i := 0; i < 4; i++ {
		t[i] = ta[i] + ta[i+4]
		t[i+4] = ta[i] - ta[i+4]
	}
	for i := 8; i < 12; i++ {
		ta[i] = t[i] + t[i+4]
		ta[i+4] = t[i] - t[i+4]
	}
	ta[4] = rot(t[4], 3784, t[5], 1567)
	ta[5] = rot(t[4], 1567, t[5], -3784)
	ta[6] = rot(t[7], 3784, t[6], -1567)
	ta[7] = rot(t[7], 1567, t[6], 3784)
	t[12] = rot(ta[12], 3784, ta[13], 1567)
	t[13] = rot(ta[12], 1567, ta[13], -3784)
	t[14] = rot(ta[15], 3784, ta[14], -1567)
	t[15] = rot(ta[15], 1567, ta[14], 3784)
	var out [16]float64
	out[0] = t[0] + t[2]
	out[15] = -(t[1] + t[3])
	ta[2] = t[0] - t[2]
	ta[3] = t[1] - t[3]
	out[3] = -(ta[4] + ta[6])
	out[12] = ta[5] + ta[7]
	t[6] = ta[4] - ta[6]
	t[7] = ta[5] - ta[7]
	out[1] = -(ta[8] + ta[10])
	out[14] = ta[9] + ta[11]
	t[10] = ta[8] - ta[10]
	t[11] = ta[9] - ta[11]
	out[2] = t[12] + t[14]
	out[13] = -(t[13] + t[15])
	ta[14] = t[12] - t[14]
	ta[15] = t[13] - t[15]
	out[7] = -rot(ta[2], 2896, ta[3], 2896)
	out[8] = rot(ta[2], 2896, ta[3], -2896)
	out[4] = rot(t[6], 2896, t[7], 2896)
	out[11] = -rot(t[6], 2896, t[7], -2896)
	out[6] = rot(t[10], 2896, t[11], 2896)
	out[9] = -rot(t[10], 2896, t[11], -2896)
	out[5] = -rot(ta[14], 2896, ta[15], 2896)
	out[10] = rot(ta[14], 2896, ta[15], -2896)
	for i := range out {
		c[i*stride] = out[i]
	}
}

// adst8F follows inverseADST8To in adst.go stage for stage.
func adst8F(c []float64, stride int, g *gainRecorder) {
	var in [8]float64
	for i := range in {
		in[i] = c[i*stride]
	}
	rot := func(a, ca, b, cb float64) float64 { return g.rot(rs(a*ca+b*cb, 12)) }
	var t, ta [8]float64
	ta[0] = rot(in[7], 4076, in[0], 401)
	ta[1] = rot(in[7], 401, in[0], -4076)
	ta[2] = rot(in[5], 3612, in[2], 1931)
	ta[3] = rot(in[5], 1931, in[2], -3612)
	ta[4] = rot(in[3], 2598, in[4], 3166)
	ta[5] = rot(in[3], 3166, in[4], -2598)
	ta[6] = rot(in[1], 1189, in[6], 3920)
	ta[7] = rot(in[1], 3920, in[6], -1189)
	for i := 0; i < 4; i++ {
		t[i] = ta[i] + ta[i+4]
		t[i+4] = ta[i] - ta[i+4]
	}
	ta[4] = rot(t[4], 3784, t[5], 1567)
	ta[5] = rot(t[4], 1567, t[5], -3784)
	ta[6] = rot(t[7], 3784, t[6], -1567)
	ta[7] = rot(t[7], 1567, t[6], 3784)
	var out [8]float64
	out[0] = t[0] + t[2]
	out[7] = -(t[1] + t[3])
	t[2] = t[0] - t[2]
	t[3] = t[1] - t[3]
	out[1] = -(ta[4] + ta[6])
	out[6] = ta[5] + ta[7]
	t[6] = ta[4] - ta[6]
	t[7] = ta[5] - ta[7]
	out[3] = -rot(t[2], 2896, t[3], 2896)
	out[4] = rot(t[2], 2896, t[3], -2896)
	out[2] = rot(t[6], 2896, t[7], 2896)
	out[5] = -rot(t[6], 2896, t[7], -2896)
	for i := range out {
		c[i*stride] = out[i]
	}
}

// TestITXHybridGainBounds checks that the fused hybrid kernels' guard
// (itx16ColumnL1Bound, 32767-64) keeps every rotation output of the ADST8
// and ADST16 networks inside int16: an output is bounded by the largest
// single-input gain times the per-lane L1 norm of the inputs, plus the
// rounding slack.
func TestITXHybridGainBounds(t *testing.T) {
	for _, tc := range []struct {
		n int
		f func([]float64, int, *gainRecorder)
	}{{8, adst8F}, {16, adst16F}} {
		g := gainsFor(tc.n, tc.f)
		slack := roundingSlack(tc.n, tc.f, len(g))
		maxCoef, maxSlack := 0.0, 0.0
		for i, row := range g {
			for _, v := range row {
				maxCoef = math.Max(maxCoef, v)
			}
			maxSlack = math.Max(maxSlack, slack[i])
		}
		t.Logf("ADST%d: %d rotation outputs; max |coef| %.4f -> per-column L1 bound %d; max rounding slack %.2f",
			tc.n, len(g), maxCoef, int(32767/maxCoef), maxSlack)
		const bound = 32767 - 64
		if maxCoef*bound+maxSlack > 32767 {
			t.Fatalf("guard %d does not certify ADST%d: %.4f * %d + %.2f exceeds 32767", bound, tc.n, maxCoef, bound, maxSlack)
		}
	}
}

// TestITXHybridFloatModelTracksScalar checks the ADST float models against
// the integer networks on small inputs, where rounding is the only
// difference.
func TestITXHybridFloatModelTracksScalar(t *testing.T) {
	for _, tc := range []struct {
		n int
		f func([]float64, int, *gainRecorder)
	}{{8, adst8F}, {16, adst16F}} {
		for trial := 0; trial < 200; trial++ {
			ci := make([]int32, tc.n)
			cf := make([]float64, tc.n)
			for i := range ci {
				v := int32((trial*7919+i*104729)%201) - 100
				ci[i] = v
				cf[i] = float64(v)
			}
			if tc.n == 8 {
				inverseADST8(ci, 1, -32768, 32767)
			} else {
				inverseADST16(ci, 1, -32768, 32767)
			}
			tc.f(cf, 1, &gainRecorder{perturb: -1})
			for i := range ci {
				if d := math.Abs(float64(ci[i]) - cf[i]); d > 64 {
					t.Fatalf("ADST%d trial %d out %d: int %d float %.2f", tc.n, trial, i, ci[i], cf[i])
				}
			}
		}
	}
}
