//go:build amd64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// TestForwardDCT8x8AVX2MatchesPureGo proves the AVX2 kernel bit-exact with
// the portable reference across random 8-bit residual ranges and strides,
// calling the kernel directly so hosts whose CPUID hides AVX2 (Rosetta)
// still exercise it.
func TestForwardDCT8x8AVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(67))
	const resStride, coeffStride = 23, 17
	residual := make([]int16, resStride*16)
	for trial := range 3000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255
		}
		want := make([]int32, coeffStride*16)
		got := make([]int32, coeffStride*16)
		forwardDCT8x8PureGo(want, coeffStride, residual, resStride)
		forwardDCT8x8AVX2(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] avx2 %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

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

// TestForwardDCT16x16AVX2MatchesPureGo proves the 16x16 AVX2 kernel bit-exact
// with the portable reference across random 8-bit residual ranges and
// strides, called directly like the 8x8 test so Rosetta hosts (CPUID hides
// AVX2) still exercise the VEX path.
func TestForwardDCT16x16AVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(83))
	const resStride, coeffStride = 37, 19
	residual := make([]int16, resStride*32)
	for trial := range 2000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255
		}
		want := make([]int32, coeffStride*32)
		got := make([]int32, coeffStride*32)
		forwardDCT16x16PureGo(want, coeffStride, residual, resStride)
		forwardDCT16x16AVX2(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] avx2 %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

