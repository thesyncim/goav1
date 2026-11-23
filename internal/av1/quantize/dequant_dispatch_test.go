// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package quantize

import (
	"math/rand"
	"testing"
)

// dequantColumnImpl is the dispatch slot bound to the architecture-best column
// kernel (NEON on arm64, pure-Go elsewhere). These tests assert that whatever
// is bound is bit-identical to the pure-Go reference over the full parameter
// space the decoder exercises: every supported scale, transform scale, and
// bit-depth clamp.

func dqBoundsForBitDepth(t *testing.T, bd uint8) (int32, int32) {
	t.Helper()
	lo, hi, ok := dqCoeffBounds(bd)
	if !ok {
		t.Fatalf("dqCoeffBounds(%d) not ok", bd)
	}
	return lo, hi
}

func TestDequantColumnDispatchMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5eed))

	// Coefficient lengths chosen to cover the 8-wide vector body plus every
	// possible tail remainder (n%8) and the empty/short cases.
	lengths := []int{0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 1024}
	scales := []int32{4, 9, 38, 44, 74, 88, 1828, 7312, 29247}
	txScales := []uint8{0, 1, 2}
	bitDepths := []uint8{8, 10, 12}

	for _, n := range lengths {
		// Build a coefficient buffer that includes the int16 extremes and a
		// spread of random values so masking, shifting, sign reapplication and
		// clamping are all exercised.
		coeff := make([]int16, n)
		for i := range coeff {
			switch i % 5 {
			case 0:
				coeff[i] = 32767
			case 1:
				coeff[i] = -32768
			case 2:
				coeff[i] = 0
			default:
				coeff[i] = int16(rng.Intn(65536) - 32768)
			}
		}
		for _, scale := range scales {
			for _, txScale := range txScales {
				for _, bd := range bitDepths {
					dqMin, dqMax := dqBoundsForBitDepth(t, bd)

					want := make([]int32, n)
					dequantColumnPureGo(want, coeff, scale, txScale, dqMin, dqMax)

					got := make([]int32, n)
					dequantColumnImpl(got, coeff, scale, txScale, dqMin, dqMax)

					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("n=%d scale=%d txScale=%d bd=%d coeff[%d]=%d: got %d want %d",
								n, scale, txScale, bd, i, coeff[i], got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestDequantColumnDispatchZeroAlloc(t *testing.T) {
	coeff := make([]int16, 1024)
	for i := range coeff {
		coeff[i] = int16(i*131 - 32768)
	}
	dst := make([]int32, 1024)
	allocs := testing.AllocsPerRun(200, func() {
		dequantColumnImpl(dst, coeff, 88, 1, -32768, 32767)
	})
	if allocs != 0 {
		t.Fatalf("dequantColumnImpl allocs = %v, want 0", allocs)
	}
}

func BenchmarkDequantColumn(b *testing.B) {
	coeff := make([]int16, 1024)
	for i := range coeff {
		coeff[i] = int16(i*131 - 32768)
	}
	dst := make([]int32, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dequantColumnImpl(dst, coeff, 88, 1, -32768, 32767)
	}
}

func BenchmarkDequantColumnPureGo(b *testing.B) {
	coeff := make([]int16, 1024)
	for i := range coeff {
		coeff[i] = int16(i*131 - 32768)
	}
	dst := make([]int32, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dequantColumnPureGo(dst, coeff, 88, 1, -32768, 32767)
	}
}
