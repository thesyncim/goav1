//go:build goexperiment.simd && (amd64 || arm64) && !purego

package quantize

import (
	"math"
	"math/rand"
	"testing"
)

// fpSIMDCoeffs builds a block of coefficients that covers the boundaries the
// fp rule branches on: zero, the skip threshold (|c| << (1+ts) against the
// dequant step), the 32767 clamp, the int32 extremes, and log-uniform random
// magnitudes of both signs.
func fpSIMDCoeffs(rng *rand.Rand, count int, dequant int32, ts uint8) []int32 {
	coeff := make([]int32, count)
	for i := range coeff {
		mag := rng.Int63n(int64(1) << uint(rng.Intn(32)))
		v := int32(mag)
		if rng.Intn(2) == 0 {
			v = -v
		}
		coeff[i] = v
	}
	thresh := dequant >> (1 + ts)
	extremes := []int32{
		0, 1, -1, thresh - 1, thresh, -thresh, -thresh + 1, thresh + 1,
		32766, 32767, 32768, -32767, -32768, -32769,
		65535, 65536, 1 << 20, -(1 << 20), 1<<30 - 1, -(1 << 30),
		math.MaxInt32, math.MinInt32, math.MinInt32 + 1, math.MaxInt32 - 1,
		// Magnitudes whose rounding add wraps int32: the keep test must drop them.
		math.MaxInt32 - 1<<14, math.MaxInt32 - 1<<13 - 7, -(math.MaxInt32 - 1<<14),
	}
	// Scatter the boundary values through the block so every vector lane
	// position sees them, including the DC slot.
	for range count / 2 {
		coeff[rng.Intn(count)] = extremes[rng.Intn(len(extremes))]
	}
	return coeff
}

// TestQuantizeFPBlockSIMDMatchesScalar proves the SIMD fp kernel bit-exact
// with quantizeScalarFP over every square block size, every tx scale, quantizer
// steps across the table range, and coefficients at the extremes of int32.
func TestQuantizeFPBlockSIMDMatchesScalar(t *testing.T) {
	if !quantizeSIMDSupported() {
		t.Skip("SIMD quantize kernels not supported on this CPU")
	}
	rng := rand.New(rand.NewSource(0x51dfa))
	for _, n := range []int{4, 8, 16, 32} {
		for ts := uint8(0); ts <= 2; ts++ {
			for trial := 0; trial < 300; trial++ {
				q := Quantizer{
					DC: int32(4 + rng.Intn(29244)),
					AC: int32(4 + rng.Intn(29244)),
				}
				if trial == 0 {
					q = Quantizer{DC: 4, AC: 4}
				}
				coeff := fpSIMDCoeffs(rng, n*n, q.AC, ts)
				got := make([]int16, n*n)
				if !quantizeFPBlockSIMD(got, coeff, n, q, ts) {
					t.Fatalf("n=%d ts=%d q=%+v trial %d: kernel refused in-range input", n, ts, q, trial)
				}
				quantDC := int64(1<<16) / int64(q.DC)
				roundDC := roundPowerOfTwo((64*q.DC)>>7, ts)
				quantAC := int64(1<<16) / int64(q.AC)
				roundAC := roundPowerOfTwo((64*q.AC)>>7, ts)
				for i := range coeff {
					want := quantizeScalarFP(coeff[i], q.AC, quantAC, roundAC, ts)
					if i == 0 {
						want = quantizeScalarFP(coeff[i], q.DC, quantDC, roundDC, ts)
					}
					if got[i] != want {
						t.Fatalf("n=%d ts=%d q=%+v trial %d: qcoeff[%d] simd %d want %d (coeff %d)",
							n, ts, q, trial, i, got[i], want, coeff[i])
					}
				}
			}
		}
	}
}

// TestQuantizeFPBlockSIMDRejectsOutOfRangeSteps keeps the guard that protects
// the int32 lane math: steps below 4 (quant above 1<<14) are refused so the
// scalar rule handles them.
func TestQuantizeFPBlockSIMDRejectsOutOfRangeSteps(t *testing.T) {
	if !quantizeSIMDSupported() {
		t.Skip("SIMD quantize kernels not supported on this CPU")
	}
	coeff := make([]int32, 16)
	got := make([]int16, 16)
	if quantizeFPBlockSIMD(got, coeff, 4, Quantizer{DC: 3, AC: 3}, 0) {
		t.Fatal("accepted AC/DC step 3 (quant above 1<<14)")
	}
	if quantizeFPBlockSIMD(got, coeff, 3, Quantizer{DC: 8, AC: 8}, 0) {
		t.Fatal("accepted a block whose count is not a multiple of 16")
	}
}
