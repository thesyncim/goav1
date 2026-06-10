package transform

import (
	"math/rand"
	"testing"
)

// TestForwardDCT8x8ImplMatchesPureGo proves the dispatched 8x8 forward DCT
// kernel bit-exact with the portable reference across random 8-bit residual
// ranges and strides.
func TestForwardDCT8x8ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(53))
	const resStride, coeffStride = 23, 17
	residual := make([]int16, resStride*16)
	for trial := range 3000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255 // 8-bit residual range
		}
		want := make([]int32, coeffStride*16)
		got := make([]int32, coeffStride*16)
		forwardDCT8x8PureGo(want, coeffStride, residual, resStride)
		forwardDCT8x8Impl(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] impl %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func BenchmarkForwardDCT8x8(b *testing.B) {
	var residual [64]int16
	for i := range residual {
		residual[i] = int16(i*7%400) - 200
	}
	var coeff [64]int32
	b.ReportAllocs()
	for b.Loop() {
		forwardDCT8x8Impl(coeff[:], 8, residual[:], 8)
	}
}
