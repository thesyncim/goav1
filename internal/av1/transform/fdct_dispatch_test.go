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

// TestForwardDCT4x4ImplMatchesPureGo proves the dispatched 4x4 forward DCT
// kernel bit-exact with the portable reference across random 8-bit residual
// ranges and strides.
func TestForwardDCT4x4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(61))
	const resStride, coeffStride = 13, 9
	residual := make([]int16, resStride*8)
	for trial := range 3000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255 // 8-bit residual range
		}
		want := make([]int32, coeffStride*8)
		got := make([]int32, coeffStride*8)
		forwardDCT4x4PureGo(want, coeffStride, residual, resStride)
		forwardDCT4x4Impl(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] impl %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func BenchmarkForwardDCT4x4(b *testing.B) {
	var residual [16]int16
	for i := range residual {
		residual[i] = int16(i*29%400) - 200
	}
	var coeff [16]int32
	b.ReportAllocs()
	for b.Loop() {
		forwardDCT4x4Impl(coeff[:], 4, residual[:], 4)
	}
}

// TestForwardDCT16x16ImplMatchesPureGo proves the dispatched 16x16 forward
// DCT kernel bit-exact with the portable reference across random 8-bit
// residual ranges and strides.
func TestForwardDCT16x16ImplMatchesPureGo(t *testing.T) {
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
		forwardDCT16x16Impl(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] impl %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func BenchmarkForwardDCT16x16(b *testing.B) {
	var residual [256]int16
	for i := range residual {
		residual[i] = int16(i*11%400) - 200
	}
	var coeff [256]int32
	b.ReportAllocs()
	for b.Loop() {
		forwardDCT16x16Impl(coeff[:], 16, residual[:], 16)
	}
}
