package quantize

import (
	"math/rand"
	"testing"
)

// TestQuantizeFPBlockImplMatchesScalar proves the vector fp-quantize kernel
// bit-exact with the scalar rule across block sizes, tx scales, quantizer
// steps, and the full coefficient range the forward transforms emit.
func TestQuantizeFPBlockImplMatchesScalar(t *testing.T) {
	if quantizeFPBlockImpl == nil {
		t.Skip("no vector kernel on this architecture")
	}
	rng := rand.New(rand.NewSource(59))
	for _, n := range []int{4, 8, 16, 32} {
		for _, ts := range []uint8{0, 1} {
			for trial := 0; trial < 200; trial++ {
				q := Quantizer{
					DC: int32(4 + rng.Intn(8000)),
					AC: int32(4 + rng.Intn(8000)),
				}
				coeff := make([]int32, n*n)
				for i := range coeff {
					coeff[i] = int32(rng.Intn(1<<22)) - 1<<21
				}
				want := make([]int16, n*n)
				got := make([]int16, n*n)
				quantDC := int64(1<<16) / int64(q.DC)
				roundDC := roundPowerOfTwo((64*q.DC)>>7, ts)
				quantAC := int64(1<<16) / int64(q.AC)
				roundAC := roundPowerOfTwo((64*q.AC)>>7, ts)
				for i := range coeff {
					if i == 0 {
						want[i] = quantizeScalarFP(coeff[i], q.DC, quantDC, roundDC, ts)
					} else {
						want[i] = quantizeScalarFP(coeff[i], q.AC, quantAC, roundAC, ts)
					}
				}
				if !quantizeFPBlockImpl(got, coeff, n, q, ts) {
					t.Fatalf("n=%d ts=%d: kernel refused", n, ts)
				}
				for i := range want {
					if want[i] != got[i] {
						t.Fatalf("n=%d ts=%d trial=%d: q[%d] impl %d want %d (coeff %d dc=%v)", n, ts, trial, i, got[i], want[i], coeff[i], i == 0)
					}
				}
			}
		}
	}
}
