//go:build amd64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// TestForwardDCT4x4AVX2MatchesPureGo proves the 4x4 kernel bit-exact with
// the portable reference, called directly like the 8x8 test.
func TestForwardDCT4x4AVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(71))
	const resStride, coeffStride = 13, 9
	residual := make([]int16, resStride*8)
	for trial := range 3000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255
		}
		want := make([]int32, coeffStride*8)
		got := make([]int32, coeffStride*8)
		forwardDCT4x4PureGo(want, coeffStride, residual, resStride)
		forwardDCT4x4AVX2(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] avx2 %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}
