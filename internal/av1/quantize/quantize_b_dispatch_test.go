package quantize

import (
	"math/rand"
	"testing"
)

// TestQuantizeBBlockImplMatchesScalar proves the vector zbin-quantize kernel
// bit-exact with the scalar rule across block sizes, tx scales, quantizer
// steps, and the full coefficient range the forward transforms emit.
func TestQuantizeBBlockImplMatchesScalar(t *testing.T) {
	if quantizeBBlockImpl == nil {
		t.Skip("no vector kernel on this architecture")
	}
	rng := rand.New(rand.NewSource(61))
	for _, n := range []int{4, 8, 16, 32} {
		for _, ts := range []uint8{0, 1} {
			for trial := 0; trial < 200; trial++ {
				q := Quantizer{
					DC: int32(4 + rng.Intn(8000)),
					AC: int32(4 + rng.Intn(8000)),
				}
				zbinFactor := int32(80)
				if q.DC < 148 {
					zbinFactor = 84
				}
				mk := func(step int32) quantBParams {
					quant, shift := invertQuant(step)
					return quantBParams{
						zbin:  roundPowerOfTwo(roundPowerOfTwo(zbinFactor*step, 7), ts),
						round: roundPowerOfTwo((48*step)>>7, ts),
						quant: quant,
						shift: shift,
					}
				}
				dc, ac := mk(q.DC), mk(q.AC)
				coeff := make([]int32, n*n)
				for i := range coeff {
					coeff[i] = int32(rng.Intn(1<<22)) - 1<<21
				}
				want := make([]int16, n*n)
				got := make([]int16, n*n)
				for i := range coeff {
					p := &ac
					if i == 0 {
						p = &dc
					}
					want[i] = quantizeScalarB(coeff[i], p, ts)
				}
				if !quantizeBBlockImpl(got, coeff, n, q, ts) {
					t.Fatalf("n=%d ts=%d: kernel refused", n, ts)
				}
				for i := range want {
					if want[i] != got[i] {
						t.Fatalf("n=%d ts=%d trial=%d: q[%d] impl %d want %d (coeff %d)", n, ts, trial, i, got[i], want[i], coeff[i])
					}
				}
			}
		}
	}
}
