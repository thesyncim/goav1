package transform

import (
	"math"
	"testing"
)

// dct64F follows inverseDCT64 in dct.go stage for stage (the structure the
// generated itx64Odd* networks fold their zero operands from): the DCT32 of
// the even inputs and the ten-stage odd half.
func dct64F(c []float64, stride int, g *gainRecorder) {
	dct32F(c, stride<<1, g)
	var e [32]float64
	for i := range e {
		e[i] = c[2*i*stride]
	}
	var x, y [64]float64
	for i := range x {
		x[i] = c[i*stride]
	}
	cp := func(i int) float64 { return float64(itx64Cos[i]) }
	rot := func(a, ca, b, cb float64) float64 { return g.rot(rs(a*ca+b*cb, 12)) }
	// Stage 2: pairs (32+j, 63-j) from inputs (a, 64-a) in bit-reversed order.
	a := [16]int{1, 33, 17, 49, 9, 41, 25, 57, 5, 37, 21, 53, 13, 45, 29, 61}
	for j := 0; j < 16; j++ {
		lo, hi := 32+j, 63-j
		ia, ib := a[j], 64-a[j]
		y[lo] = rot(x[ia], cp(ib), x[ib], -cp(ia))
		y[hi] = rot(x[ia], cp(ia), x[ib], cp(ib))
	}
	x = y
	// Stage 3.
	for b := 32; b < 64; b += 4 {
		y[b] = x[b] + x[b+1]
		y[b+1] = x[b] - x[b+1]
		y[b+2] = x[b+3] - x[b+2]
		y[b+3] = x[b+2] + x[b+3]
	}
	x = y
	// Stage 4.
	for m, pq := range [4][2]int{{4, 60}, {36, 28}, {20, 44}, {52, 12}} {
		p, q := cp(pq[0]), cp(pq[1])
		lo, hi := 33+4*m, 62-4*m
		y[lo] = rot(x[lo], -p, x[hi], q)
		y[hi] = rot(x[lo], q, x[hi], p)
		y[lo+1] = rot(x[lo+1], -q, x[hi-1], -p)
		y[hi-1] = rot(x[lo+1], -p, x[hi-1], q)
	}
	x = y
	// Stage 5.
	for b := 32; b < 64; b += 8 {
		y[b] = x[b] + x[b+3]
		y[b+1] = x[b+1] + x[b+2]
		y[b+2] = x[b+1] - x[b+2]
		y[b+3] = x[b] - x[b+3]
		y[b+4] = x[b+7] - x[b+4]
		y[b+5] = x[b+6] - x[b+5]
		y[b+6] = x[b+5] + x[b+6]
		y[b+7] = x[b+4] + x[b+7]
	}
	x = y
	// Stage 6.
	for _, pql := range [2][4]int{{8, 56, 34, 61}, {40, 24, 42, 53}} {
		p, q := cp(pql[0]), cp(pql[1])
		lo, hi := pql[2], pql[3]
		for i := 0; i < 2; i++ {
			y[lo+i] = rot(x[lo+i], -p, x[hi-i], q)
			y[hi-i] = rot(x[lo+i], q, x[hi-i], p)
		}
		for i := 2; i < 4; i++ {
			y[lo+i] = rot(x[lo+i], -q, x[hi-i], -p)
			y[hi-i] = rot(x[lo+i], -p, x[hi-i], q)
		}
	}
	x = y
	// Stage 7.
	for b := 32; b < 64; b += 16 {
		for i := 0; i < 4; i++ {
			y[b+i] = x[b+i] + x[b+7-i]
			y[b+7-i] = x[b+i] - x[b+7-i]
			y[b+8+i] = x[b+15-i] - x[b+8+i]
			y[b+15-i] = x[b+8+i] + x[b+15-i]
		}
	}
	x = y
	// Stage 8.
	for i := 0; i < 4; i++ {
		y[36+i] = rot(x[36+i], -cp(16), x[59-i], cp(48))
		y[59-i] = rot(x[36+i], cp(48), x[59-i], cp(16))
		y[40+i] = rot(x[40+i], -cp(48), x[55-i], -cp(16))
		y[55-i] = rot(x[40+i], -cp(16), x[55-i], cp(48))
	}
	x = y
	// Stage 9.
	for i := 0; i < 8; i++ {
		y[32+i] = x[32+i] + x[47-i]
		y[47-i] = x[32+i] - x[47-i]
		y[48+i] = x[63-i] - x[48+i]
		y[63-i] = x[48+i] + x[63-i]
	}
	x = y
	// Stage 10.
	for i := 0; i < 8; i++ {
		y[40+i] = rot(x[40+i], -cp(32), x[55-i], cp(32))
		y[55-i] = rot(x[40+i], cp(32), x[55-i], cp(32))
	}
	x = y
	for i := 0; i < 32; i++ {
		c[i*stride] = e[i] + x[63-i]
		c[(63-i)*stride] = e[i] - x[63-i]
	}
}

// TestITX64GainBounds logs the DCT64 bounds like TestITX16GainBounds and
// checks that the fused 64x64 kernel's guard (itx16ColumnL1Bound, 32767-64)
// keeps every rotation output inside int16: an output is bounded by the
// largest single-input gain times the per-lane L1 norm of the inputs, plus
// the rounding slack.
func TestITX64GainBounds(t *testing.T) {
	g := gainsFor(64, dct64F)
	slack := roundingSlack(64, dct64F, len(g))
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
	t.Logf("DCT64: %d rotation outputs; max row L1 gain %.4f (output %d) -> uniform input bound %d; max |coef| %.4f -> per-column L1 bound %d; max rounding slack %.2f",
		len(g), maxRowL1, worst, int(32767/maxRowL1), maxCoef, int(32767/maxCoef), maxSlack)
	const bound = 32767 - 64
	if maxCoef*bound+maxSlack > 32767 {
		t.Fatalf("guard %d does not certify DCT64: %.4f * %d + %.2f exceeds 32767", bound, maxCoef, bound, maxSlack)
	}
}

// TestITX64FloatModelTracksScalar checks dct64F against the integer DCT64 on
// small inputs, where rounding is the only difference.
func TestITX64FloatModelTracksScalar(t *testing.T) {
	for trial := 0; trial < 200; trial++ {
		ci := make([]int32, 64)
		cf := make([]float64, 64)
		for i := range ci {
			v := int32((trial*7919+i*104729)%201) - 100
			ci[i] = v
			cf[i] = float64(v)
		}
		inverseDCT1D(ci, 1, 64, -32768, 32767)
		dct64F(cf, 1, &gainRecorder{perturb: -1})
		for i := range ci {
			if d := math.Abs(float64(ci[i]) - cf[i]); d > 64 {
				t.Fatalf("trial %d out %d: int %d float %.2f", trial, i, ci[i], cf[i])
			}
		}
	}
}
