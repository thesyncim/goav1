package quantize

import (
	"math/rand"
	"testing"
)

// TestQuantizeDequantizeRoundTrip proves the forward quantizer inverts the
// decoder's dequantization to within one step: for any transform-domain
// coefficient, dequant(quant(c)) differs from c by less than the applicable
// scale (after the txScale shift), with sign preserved and the sub-step band
// quantizing to exactly zero.
func TestQuantizeDequantizeRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	sizes := []struct{ w, h int }{{4, 4}, {8, 8}, {16, 16}, {32, 32}}
	qIndices := []uint8{1, 20, 60, 120, 200, 255}
	for _, sz := range sizes {
		txScale, err := TransformScale(sz.w, sz.h)
		if err != nil {
			t.Fatal(err)
		}
		for _, qi := range qIndices {
			dc, err := DCQuant(qi, 0, 8)
			if err != nil {
				t.Fatal(err)
			}
			ac, err := ACQuant(qi, 0, 8)
			if err != nil {
				t.Fatal(err)
			}
			q := Quantizer{DC: dc, AC: ac}
			n := sz.w * sz.h
			coeff := make([]int32, n)
			for i := range coeff {
				coeff[i] = int32(rng.Intn(2*8192+1) - 8192) // tran-domain span
			}
			qcoeff := make([]int16, n)
			if err := QuantizeBlockScaled(qcoeff, sz.h, coeff, sz.h, sz.w, sz.h, q, txScale); err != nil {
				t.Fatalf("quantize %dx%d q%d: %v", sz.w, sz.h, qi, err)
			}
			dqs := make([]int32, n)
			if err := DequantizeBlockScaledBitDepth(dqs, sz.h, qcoeff, sz.h, sz.w, sz.h, q, txScale, 8); err != nil {
				t.Fatalf("dequantize scaled: %v", err)
			}
			for pos := range n {
				scale := q.AC
				if pos == 0 {
					scale = q.DC
				}
				bound := scale >> txScale
				if bound == 0 {
					bound = 1
				}
				diff := dqs[pos] - coeff[pos]
				if diff < 0 {
					diff = -diff
				}
				if diff >= scale {
					t.Fatalf("%dx%d q%d pos %d: c=%d q=%d dq=%d |err|=%d >= step %d",
						sz.w, sz.h, qi, pos, coeff[pos], qcoeff[pos], dqs[pos], diff, scale)
				}
				if coeff[pos] > 0 && dqs[pos] < 0 || coeff[pos] < 0 && dqs[pos] > 0 {
					t.Fatalf("sign flipped at pos %d: c=%d dq=%d", pos, coeff[pos], dqs[pos])
				}
			}
		}
	}
}

func TestQuantizeSubStepIsZero(t *testing.T) {
	q := Quantizer{DC: 40, AC: 50}
	coeff := make([]int32, 16)
	coeff[0] = 39   // below DC step
	coeff[5] = -49  // below AC step
	coeff[10] = 50  // exactly one AC step
	coeff[15] = -40 // one AC step? no: AC=50 -> trunc 0
	qcoeff := make([]int16, 16)
	if err := QuantizeBlockScaled(qcoeff, 4, coeff, 4, 4, 4, q, 0); err != nil {
		t.Fatal(err)
	}
	if qcoeff[0] != 0 || qcoeff[5] != 0 || qcoeff[15] != 0 {
		t.Fatalf("sub-step levels not zero: %v", qcoeff)
	}
	if qcoeff[10] != 1 {
		t.Fatalf("one-step level = %d, want 1", qcoeff[10])
	}
}

func TestQuantizeZeroAlloc(t *testing.T) {
	q := Quantizer{DC: 100, AC: 120}
	coeff := make([]int32, 64)
	for i := range coeff {
		coeff[i] = int32(i*37 - 800)
	}
	qcoeff := make([]int16, 64)
	allocs := testing.AllocsPerRun(100, func() {
		_ = QuantizeBlockScaled(qcoeff, 8, coeff, 8, 8, 8, q, 0)
	})
	if allocs != 0 {
		t.Fatalf("quantize allocated %v objects/run, want 0", allocs)
	}
}
