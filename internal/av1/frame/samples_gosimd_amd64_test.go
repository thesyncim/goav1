// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package frame

import (
	"simd/archsimd"
	"testing"
)

// TestLoadSampleRows8DispatchBindsSIMD asserts the dispatcher selects the Go
// SIMD kernel for vector-sized rectangles exactly when the CPU advertises AVX2,
// and keeps the scalar reference for shapes the kernel does not own.
func TestLoadSampleRows8DispatchBindsSIMD(t *testing.T) {
	if got, want := sampleRows8UseSIMD(16, 1), archsimd.X86.AVX2(); got != want {
		t.Fatalf("sampleRows8UseSIMD(16, 1)=%v want AVX2=%v", got, want)
	}
	if sampleRows8UseSIMD(15, 4) || sampleRows8UseSIMD(16, 0) {
		t.Fatal("SIMD selected for a shape without a vector body")
	}
}

// widenLCG is a tiny deterministic generator so the differential test is
// reproducible without pulling in math/rand.
type widenLCG struct{ s uint64 }

func (g *widenLCG) next() uint64 {
	g.s = g.s*6364136223846793005 + 1442695040888963407
	return g.s >> 11
}

// TestLoadSampleRows8SIMDMatchesPureGo runs the AVX2 widen and the pure-Go
// reference over identical inputs and asserts element-for-element equality. It
// calls the SIMD kernel directly (not through the dispatcher) so it exercises
// the vector path even on hosts that do not advertise AVX2 in CPUID — notably
// Rosetta 2, which executes AVX2 but reports it absent. It sweeps widths that
// cover the 16-wide body and the scalar row tail, with a dst stride wider than
// the visible width to catch any past-width write.
func TestLoadSampleRows8SIMDMatchesPureGo(t *testing.T) {
	widths := []int{1, 8, 9, 13, 15, 16, 17, 23, 24, 31, 32, 33, 48, 63, 64, 65, 127}
	heights := []int{1, 2, 3, 5, 8}

	for _, width := range widths {
		for _, height := range heights {
			g := widenLCG{s: uint64(width*131 + height*17 + 7)}
			srcStride := width + 5
			dstStride := width + 11

			src := make([]byte, srcStride*height)
			for i := range src {
				src[i] = byte(g.next())
			}

			// Seed both destinations with a sentinel so any stray write past
			// the visible [0,width) region shows up as a mismatch.
			gotDst := make([]uint16, dstStride*height)
			wantDst := make([]uint16, dstStride*height)
			for i := range gotDst {
				gotDst[i] = 0xBEEF
				wantDst[i] = 0xBEEF
			}

			loadSampleRows8PureGo(wantDst, dstStride, src, srcStride, width, height)
			loadSampleRows8SIMD(gotDst, dstStride, src, srcStride, width, height)

			for i := range gotDst {
				if gotDst[i] != wantDst[i] {
					t.Fatalf("w=%d h=%d elem %d: simd=%#04x ref=%#04x", width, height, i, gotDst[i], wantDst[i])
				}
			}
		}
	}
}

func TestLoadSampleRows8SIMDIsZeroAlloc(t *testing.T) {
	const w, h = 64, 64
	src := make([]byte, w*h)
	for i := range src {
		src[i] = byte(i)
	}
	dst := make([]uint16, w*h)
	allocs := testing.AllocsPerRun(1000, func() {
		loadSampleRows8SIMD(dst, w, src, w, w, h)
	})
	if allocs != 0 {
		t.Fatalf("loadSampleRows8SIMD allocated: %f", allocs)
	}
}
