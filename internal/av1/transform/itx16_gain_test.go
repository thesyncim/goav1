package transform

import (
	"math"
	"testing"
)

// Float models of the inverse DCT butterfly networks in dct.go: the same
// structure with exact (unrounded, unclamped) real arithmetic, recording every
// rotation output. Fed with unit vectors they give the coefficient g[i][j] of
// input j in rotation output i, which bounds the true integer value of that
// output by sum_j |g[i][j]| |x_j| plus rounding slack (clamps only shrink
// magnitudes). This is the exactness contract of the int16 column kernels: a
// kernel equals the scalar reference whenever every rotation output fits
// int16, since the butterfly sums are then saturating adds of exact int16
// operands, equal to the scalar's clip to the same int16 range.

type gainRecorder struct {
	outs    [][]float64 // rotation outputs, one coefficient vector each
	n       int
	cur     int
	perturb int // rotation output index to perturb by +1 (-1 = none)
	calls   int
}

func (g *gainRecorder) rot(v float64) float64 {
	if g.calls == g.perturb {
		v += 1
	}
	g.calls++
	if g.outs != nil {
		g.outs[g.cur] = append(g.outs[g.cur], v)
	}
	return v
}

func rs(x float64, n int) float64 { return x / float64(int64(1)<<uint(n)) }

func dct4F(c []float64, stride int, g *gainRecorder) {
	in0, in1, in2, in3 := c[0], c[stride], c[2*stride], c[3*stride]
	t0 := g.rot(rs((in0+in2)*181, 8))
	t1 := g.rot(rs((in0-in2)*181, 8))
	t2 := g.rot(rs(in1*1567-in3*3784, 12))
	t3 := g.rot(rs(in1*3784+in3*1567, 12))
	c[0], c[stride], c[2*stride], c[3*stride] = t0+t3, t1+t2, t1-t2, t0-t3
}

func dct8F(c []float64, stride int, g *gainRecorder) {
	dct4F(c, stride<<1, g)
	in1, in3, in5, in7 := c[stride], c[3*stride], c[5*stride], c[7*stride]
	t4a := g.rot(rs(in1*799-in7*4017, 12))
	t5a := g.rot(rs(in5*1703-in3*1138, 11))
	t6a := g.rot(rs(in5*1138+in3*1703, 11))
	t7a := g.rot(rs(in1*4017+in7*799, 12))
	t4 := t4a + t5a
	t5a = t4a - t5a
	t7 := t7a + t6a
	t6a = t7a - t6a
	t5 := g.rot(rs((t6a-t5a)*181, 8))
	t6 := g.rot(rs((t6a+t5a)*181, 8))
	t0, t1, t2, t3 := c[0], c[2*stride], c[4*stride], c[6*stride]
	c[0], c[stride], c[2*stride], c[3*stride] = t0+t7, t1+t6, t2+t5, t3+t4
	c[4*stride], c[5*stride], c[6*stride], c[7*stride] = t3-t4, t2-t5, t1-t6, t0-t7
}

func dct16F(c []float64, stride int, g *gainRecorder) {
	dct8F(c, stride<<1, g)
	in1, in3, in5, in7 := c[stride], c[3*stride], c[5*stride], c[7*stride]
	in9, in11, in13, in15 := c[9*stride], c[11*stride], c[13*stride], c[15*stride]
	t8a := g.rot(rs(in1*401-in15*4076, 12))
	t9a := g.rot(rs(in9*1583-in7*1299, 11))
	t10a := g.rot(rs(in5*1931-in11*3612, 12))
	t11a := g.rot(rs(in13*3920-in3*1189, 12))
	t12a := g.rot(rs(in13*1189+in3*3920, 12))
	t13a := g.rot(rs(in5*3612+in11*1931, 12))
	t14a := g.rot(rs(in9*1299+in7*1583, 11))
	t15a := g.rot(rs(in1*4076+in15*401, 12))
	t8, t9 := t8a+t9a, t8a-t9a
	t10, t11 := t11a-t10a, t11a+t10a
	t12, t13 := t12a+t13a, t12a-t13a
	t14, t15 := t15a-t14a, t15a+t14a
	t9a = g.rot(rs(t14*1567-t9*3784, 12))
	t14a = g.rot(rs(t14*3784+t9*1567, 12))
	t10a = g.rot(rs(-(t13*3784 + t10*1567), 12))
	t13a = g.rot(rs(t13*1567-t10*3784, 12))
	t8a = t8 + t11
	t9 = t9a + t10a
	t10 = t9a - t10a
	t11a = t8 - t11
	t12a = t15 - t12
	t13 = t14a - t13a
	t14 = t14a + t13a
	t15a = t15 + t12
	t10a = g.rot(rs((t13-t10)*181, 8))
	t13a = g.rot(rs((t13+t10)*181, 8))
	t11 = g.rot(rs((t12a-t11a)*181, 8))
	t12 = g.rot(rs((t12a+t11a)*181, 8))
	e := [8]float64{c[0], c[2*stride], c[4*stride], c[6*stride], c[8*stride], c[10*stride], c[12*stride], c[14*stride]}
	o := [8]float64{t15a, t14, t13a, t12, t11, t10a, t9, t8a}
	for i := 0; i < 8; i++ {
		c[i*stride] = e[i] + o[i]
		c[(15-i)*stride] = e[i] - o[i]
	}
}

// dct32F follows inverseDCT32 in dct.go stage for stage.
func dct32F(c []float64, stride int, g *gainRecorder) {
	dct16F(c, stride<<1, g)
	var in [32]float64
	for i := range in {
		in[i] = c[i*stride]
	}
	rot := func(a float64, ca float64, b float64, cb float64, n int) float64 { return g.rot(rs(a*ca+b*cb, n)) }
	t16a := rot(in[1], 201, in[31], -4091, 12)
	t17a := rot(in[17], 3035, in[15], -2751, 12)
	t18a := rot(in[9], 1751, in[23], -3703, 12)
	t19a := rot(in[25], 3857, in[7], -1380, 12)
	t20a := rot(in[5], 995, in[27], -3973, 12)
	t21a := rot(in[21], 3513, in[11], -2106, 12)
	t22a := rot(in[13], 2440, in[19], -3290, 12)
	t23a := rot(in[29], 4052, in[3], -601, 12)
	t24a := rot(in[29], 601, in[3], 4052, 12)
	t25a := rot(in[13], 3290, in[19], 2440, 12)
	t26a := rot(in[21], 2106, in[11], 3513, 12)
	t27a := rot(in[5], 3973, in[27], 995, 12)
	t28a := rot(in[25], 1380, in[7], 3857, 12)
	t29a := rot(in[9], 3703, in[23], 1751, 12)
	t30a := rot(in[17], 2751, in[15], 3035, 12)
	t31a := rot(in[1], 4091, in[31], 201, 12)
	t16, t17 := t16a+t17a, t16a-t17a
	t18, t19 := t19a-t18a, t19a+t18a
	t20, t21 := t20a+t21a, t20a-t21a
	t22, t23 := t23a-t22a, t23a+t22a
	t24, t25 := t24a+t25a, t24a-t25a
	t26, t27 := t27a-t26a, t27a+t26a
	t28, t29 := t28a+t29a, t28a-t29a
	t30, t31 := t31a-t30a, t31a+t30a
	t17a = rot(t30, 799, t17, -4017, 12)
	t30a = rot(t30, 4017, t17, 799, 12)
	t18a = rot(t29, -4017, t18, -799, 12)
	t29a = rot(t29, 799, t18, -4017, 12)
	t21a = rot(t26, 3406, t21, -2276, 12)
	t26a = rot(t26, 2276, t21, 3406, 12)
	t22a = rot(t25, -2276, t22, -3406, 12)
	t25a = rot(t25, 3406, t22, -2276, 12)
	t16a, t19a = t16+t19, t16-t19
	t17, t18 = t17a+t18a, t17a-t18a
	t20a, t23a = t23-t20, t23+t20
	t21, t22 = t22a-t21a, t22a+t21a
	t24a, t27a = t24+t27, t24-t27
	t25, t26 = t25a+t26a, t25a-t26a
	t28a, t31a = t31-t28, t31+t28
	t29, t30 = t30a-t29a, t30a+t29a
	t18a = rot(t29, 1567, t18, -3784, 12)
	t29a = rot(t29, 3784, t18, 1567, 12)
	t19 = rot(t28a, 1567, t19a, -3784, 12)
	t28 = rot(t28a, 3784, t19a, 1567, 12)
	t20 = rot(t27a, -3784, t20a, -1567, 12)
	t27 = rot(t27a, 1567, t20a, -3784, 12)
	t21a = rot(t26, -3784, t21, -1567, 12)
	t26a = rot(t26, 1567, t21, -3784, 12)
	t16, t23 = t16a+t23a, t16a-t23a
	t17a, t22a = t17+t22, t17-t22
	t18, t21 = t18a+t21a, t18a-t21a
	t19a, t20a = t19+t20, t19-t20
	t24, t31 = t31a-t24a, t31a+t24a
	t25a, t30a = t30-t25, t30+t25
	t26, t29 = t29a-t26a, t29a+t26a
	t27a, t28a = t28-t27, t28+t27
	t20 = g.rot(rs((t27a-t20a)*181, 8))
	t27 = g.rot(rs((t27a+t20a)*181, 8))
	t21a = g.rot(rs((t26-t21)*181, 8))
	t26a = g.rot(rs((t26+t21)*181, 8))
	t22 = g.rot(rs((t25a-t22a)*181, 8))
	t25 = g.rot(rs((t25a+t22a)*181, 8))
	t23a = g.rot(rs((t24-t23)*181, 8))
	t24a = g.rot(rs((t24+t23)*181, 8))
	o := [16]float64{t31, t30a, t29, t28a, t27, t26a, t25, t24a, t23a, t22, t21a, t20, t19a, t18, t17a, t16}
	var e [16]float64
	for i := range e {
		e[i] = c[2*i*stride]
	}
	for i := 0; i < 16; i++ {
		c[i*stride] = e[i] + o[i]
		c[(31-i)*stride] = e[i] - o[i]
	}
}

// rotOutputs runs the float network on c with rotation output p perturbed by
// +1 (p < 0: none) and returns every rotation output in call order.
func rotOutputs(f func([]float64, int, *gainRecorder), c []float64, p int) []float64 {
	rec := &gainRecorder{outs: make([][]float64, 1), perturb: p}
	f(c, 1, rec)
	return rec.outs[0]
}

// gainsFor returns, for the N-point inverse DCT, |g[i][j]|: the coefficient
// magnitude of input j in rotation output i.
func gainsFor(n int, f func([]float64, int, *gainRecorder)) [][]float64 {
	var outs [][]float64
	for j := 0; j < n; j++ {
		c := make([]float64, n)
		c[j] = 1
		vals := rotOutputs(f, c, -1)
		if outs == nil {
			outs = make([][]float64, len(vals))
		}
		for i, v := range vals {
			outs[i] = append(outs[i], math.Abs(v))
		}
	}
	return outs
}

// roundingSlack returns, per rotation output i, the largest total effect of
// the +-0.5 rounding errors of all earlier rotation outputs on it:
// 0.5 * sum_k |h[i][k]| where h[i][k] is the gain from a unit perturbation of
// output k to output i.
func roundingSlack(n int, f func([]float64, int, *gainRecorder), numRot int) []float64 {
	slack := make([]float64, numRot)
	for k := 0; k < numRot; k++ {
		c := make([]float64, n)
		vals := rotOutputs(f, c, k)
		for i, v := range vals {
			if i != k {
				slack[i] += 0.5 * math.Abs(v)
			}
		}
	}
	return slack
}

func TestITX16GainBounds(t *testing.T) {
	for _, tc := range []struct {
		n int
		f func([]float64, int, *gainRecorder)
	}{{8, dct8F}, {16, dct16F}, {32, dct32F}} {
		g := gainsFor(tc.n, tc.f)
		slack := roundingSlack(tc.n, tc.f, len(g))
		maxRowL1, maxCoef, maxSlack := 0.0, 0.0, 0.0
		worst := -1
		for i, row := range g {
			l1 := 0.0
			for _, v := range row {
				l1 += v
				maxCoef = math.Max(maxCoef, v)
			}
			if l1 > maxRowL1 {
				maxRowL1, worst = l1, i
			}
			maxSlack = math.Max(maxSlack, slack[i])
		}
		t.Logf("DCT%d: %d rotation outputs; max row L1 gain %.4f (output %d) -> uniform input bound %d; max |coef| %.4f -> per-column L1 bound %d; max rounding slack %.2f",
			tc.n, len(g), maxRowL1, worst, int(32767/maxRowL1), maxCoef, int(32767/maxCoef), maxSlack)
	}
}

// TestITX16FloatModelTracksScalar checks the float networks against the integer
// kernels on small inputs (where rounding is the only difference).
func TestITX16FloatModelTracksScalar(t *testing.T) {
	for _, tc := range []struct {
		n int
		f func([]float64, int, *gainRecorder)
	}{{16, dct16F}, {32, dct32F}} {
		for trial := 0; trial < 200; trial++ {
			ci := make([]int32, tc.n)
			cf := make([]float64, tc.n)
			for i := range ci {
				v := int32((trial*7919+i*104729)%201) - 100
				ci[i] = v
				cf[i] = float64(v)
			}
			inverseDCT1D(ci, 1, tc.n, -32768, 32767)
			tc.f(cf, 1, &gainRecorder{perturb: -1})
			for i := range ci {
				if d := math.Abs(float64(ci[i]) - cf[i]); d > 64 {
					t.Fatalf("DCT%d trial %d out %d: int %d float %.2f", tc.n, trial, i, ci[i], cf[i])
				}
			}
		}
	}
}
