// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package quantize

import (
	"math/rand"
	"testing"
)

// These tests invoke the AVX2 column kernel directly rather than through the
// dispatch slot. The dispatch slot only binds AVX2 when CPUID reports
// OS-enabled AVX2; under Apple Rosetta 2 the guest CPUID hides AVX so the slot
// falls back to pure-Go even though Rosetta can execute the AVX2 instructions.
// Calling dequantColumnAVX2 directly exercises the real assembly path on every
// amd64 host (native and Rosetta) and asserts bit-identity with the reference.

// TestDequantColumnAVX2MatchesPureGo sweeps the full decoder parameter space:
// every supported AC/DC scale, every transform scale, every bit-depth clamp,
// across all vector-body and tail remainders, with int16 extremes seeded so
// masking, the arithmetic shift, sign reapplication, and clamping are all hit.
func TestDequantColumnAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xa1c2))

	lengths := []int{0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 1024}
	scales := []int32{4, 9, 38, 44, 74, 88, 1828, 7312, 29247}
	txScales := []uint8{0, 1, 2}
	bitDepths := []uint8{8, 10, 12}

	for _, n := range lengths {
		coeff := make([]int16, n)
		for i := range coeff {
			switch i % 7 {
			case 0:
				coeff[i] = 32767
			case 1:
				coeff[i] = -32768 // INT16_MIN: abs must stay safe.
			case 2:
				coeff[i] = 0
			case 3:
				coeff[i] = -1
			case 4:
				coeff[i] = 1
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
					dequantColumnAVX2(got, coeff, scale, txScale, dqMin, dqMax)

					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("n=%d scale=%d txScale=%d bd=%d coeff[%d]=%d: AVX2 got %d want %d",
								n, scale, txScale, bd, i, coeff[i], got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestDequantColumnAVX2ZeroAlloc(t *testing.T) {
	coeff := make([]int16, 1024)
	for i := range coeff {
		coeff[i] = int16(i*131 - 32768)
	}
	dst := make([]int32, 1024)
	allocs := testing.AllocsPerRun(200, func() {
		dequantColumnAVX2(dst, coeff, 88, 1, -32768, 32767)
	})
	if allocs != 0 {
		t.Fatalf("dequantColumnAVX2 allocs = %v, want 0", allocs)
	}
}

func BenchmarkDequantColumnAVX2(b *testing.B) {
	coeff := make([]int16, 1024)
	for i := range coeff {
		coeff[i] = int16(i*131 - 32768)
	}
	dst := make([]int32, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dequantColumnAVX2(dst, coeff, 88, 1, -32768, 32767)
	}
}
