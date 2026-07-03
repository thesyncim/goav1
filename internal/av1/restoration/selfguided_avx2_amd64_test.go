// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package restoration

import (
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// TestSGRAVX2ReportsDetection records whether the dispatcher bound the AVX2
// kernels. Under Rosetta 2 on Apple silicon CPUID does not advertise AVX2, so
// auto-dispatch stays pure-Go even though the VEX instructions themselves
// execute; the differential tests below call the AVX2 funcs directly so they
// exercise the SIMD regardless.
func TestSGRAVX2ReportsDetection(t *testing.T) {
	t.Logf("cpu.Detected.AVX2 = %v (SSE41=%v SSE42=%v AVX512=%v)",
		cpu.Detected.AVX2, cpu.Detected.SSE41, cpu.Detected.SSE42, cpu.Detected.AVX512)
}

// TestBoxsumAVX2MatchesPureGo compares the AVX2 box-filter sums against the
// scalar reference across radii, sizes (including widths whose interior band is
// smaller than the 8-lane vector so the wrapper must fall back), and both the
// plain and squared variants. It calls boxsumAVX2 directly, independent of
// cpu.Detected.
func TestBoxsumAVX2MatchesPureGo(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xB0B)
	sizes := []struct{ w, h int }{{4, 4}, {5, 3}, {8, 8}, {17, 19}, {64, 64}, {70, 70}, {1, 1}, {3, 9}, {64, 1}, {12, 12}}
	for _, r := range []int{0, 1, 2} {
		for _, sz := range sizes {
			width := sz.w + 2*SGRProjBorderHorz
			height := sz.h + 2*SGRProjBorderVert
			stride := sgrBufferStride(sz.w)
			srcStride := width
			src := make([]int32, srcStride*height)
			for i := range src {
				src[i] = int32(rnd.pseudoUniform(4096))
			}
			for _, squared := range []bool{false, true} {
				want := make([]int32, stride*height)
				got := make([]int32, stride*height)
				boxsum(src, 0, width, height, srcStride, r, squared, want, stride)
				boxsumAVX2(src, 0, width, height, srcStride, r, squared, got, stride)
				if !slices.Equal(want, got) {
					for i := range want {
						if want[i] != got[i] {
							t.Fatalf("r=%d sz=%dx%d squared=%v idx=%d got=%d want=%d", r, sz.w, sz.h, squared, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

// TestBoxsumAVX2IsZeroAlloc protects the hot-path contract that the AVX2 box
// sums do not allocate per call.
func TestBoxsumAVX2IsZeroAlloc(t *testing.T) {
	const w, h = 64, 64
	width := w + 2*SGRProjBorderHorz
	height := h + 2*SGRProjBorderVert
	stride := sgrBufferStride(w)
	src := make([]int32, width*height)
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x4243)
	for i := range src {
		src[i] = int32(rnd.pseudoUniform(4096))
	}
	dst := make([]int32, stride*height)
	if allocs := testing.AllocsPerRun(200, func() {
		boxsumAVX2(src, 0, width, height, width, 2, true, dst, stride)
	}); allocs != 0 {
		t.Fatalf("boxsumAVX2 allocated %f times per call", allocs)
	}
}
