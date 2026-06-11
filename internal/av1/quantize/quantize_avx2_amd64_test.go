//go:build amd64 && !purego

package quantize

import (
	"math/rand"
	"testing"
)

// TestQuantizeBBlockAVX2MatchesScalar proves the AVX2 zbin kernel bit-exact
// with the scalar rule across the quantizer table range and both tx scales,
// calling the kernel directly so Rosetta hosts verify it despite CPUID
// hiding AVX2.
func TestQuantizeBBlockAVX2MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(73))
	for _, n := range []int{4, 8, 16, 32} {
		for _, ts := range []uint8{0, 1} {
			for trial := 0; trial < 300; trial++ {
				q := Quantizer{DC: int32(rng.Intn(6900) + 4), AC: int32(rng.Intn(6900) + 4)}
				coeff := make([]int32, n*n)
				for i := range coeff {
					coeff[i] = int32(rng.Intn(1<<22)) - 1<<21
				}
				got := make([]int16, n*n)
				if !quantizeBBlockAVX2(got, coeff, n, q, ts) {
					continue
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
				want := make([]int16, n*n)
				for i := range coeff {
					p := &ac
					if i == 0 {
						p = &dc
					}
					want[i] = quantizeScalarB(coeff[i], p, ts)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("n=%d ts=%d q=%+v trial %d: qcoeff[%d] avx2 %d want %d (coeff %d)",
							n, ts, q, trial, i, got[i], want[i], coeff[i])
					}
				}
			}
		}
	}
}

// TestQuantizeFPBlockAVX2MatchesScalar proves the AVX2 fp kernel bit-exact
// with the scalar rule.
func TestQuantizeFPBlockAVX2MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(79))
	for _, n := range []int{4, 8, 16, 32} {
		for _, ts := range []uint8{0, 1} {
			for trial := 0; trial < 300; trial++ {
				q := Quantizer{DC: int32(rng.Intn(6900) + 4), AC: int32(rng.Intn(6900) + 4)}
				coeff := make([]int32, n*n)
				for i := range coeff {
					coeff[i] = int32(rng.Intn(1<<22)) - 1<<21
				}
				got := make([]int16, n*n)
				if !quantizeFPBlockAVX2(got, coeff, n, q, ts) {
					continue
				}
				want := make([]int16, n*n)
				for i := range coeff {
					step, round := q.AC, roundPowerOfTwo((64*int32(q.AC))>>7, ts)
					if i == 0 {
						step, round = q.DC, roundPowerOfTwo((64*int32(q.DC))>>7, ts)
					}
					want[i] = quantizeScalarFP(coeff[i], step, int64(1<<16)/int64(step), round, ts)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("n=%d ts=%d q=%+v trial %d: qcoeff[%d] avx2 %d want %d (coeff %d)",
							n, ts, q, trial, i, got[i], want[i], coeff[i])
					}
				}
			}
		}
	}
}
