// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package restoration

import (
	"slices"
	"testing"
)

// TestBoxsumSeparableMatchesBoxsum asserts the libaom-style separable running
// sum produces byte-identical A/B buffers to the brute-force reference across
// radii, sizes (including the exact decoder extents width+6/height+6 and the
// degenerate fallback sizes), and both the plain and squared variants.
func TestBoxsumSeparableMatchesBoxsum(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x5EFA)
	// Decoder always drives widthExt = unit + 6, heightExt = unit + 6, unit in
	// [1,64]; include those plus tiny sizes that trip the brute-force fallback.
	sizes := []struct{ w, h int }{
		{7, 7}, {8, 8}, {10, 12}, {13, 9}, {23, 31}, {70, 70}, {70, 1 + 6}, {1 + 6, 70},
		{7, 8}, {9, 7}, {5, 5}, {4, 4}, {3, 3}, {6, 6}, {2, 9}, {9, 2},
	}
	for _, r := range []int{1, 2} {
		for _, sz := range sizes {
			width := sz.w
			height := sz.h
			stride := sgrBufferStride(width)
			srcStride := width + 5 // exercise srcStride != width and a nonzero origin
			srcOrigin := 2*srcStride + 3
			src := make([]int32, srcOrigin+srcStride*height+width+8)
			for i := range src {
				src[i] = int32(rnd.pseudoUniform(4096))
			}
			for _, squared := range []bool{false, true} {
				want := make([]int32, stride*height)
				got := make([]int32, stride*height)
				boxsum(src, srcOrigin, width, height, srcStride, r, squared, want, stride)
				boxsumSeparable(src, srcOrigin, width, height, srcStride, r, squared, got, stride)
				if !slices.Equal(want, got) {
					for i := range want {
						if want[i] != got[i] {
							t.Fatalf("r=%d sz=%dx%d squared=%v idx=%d got=%d want=%d",
								r, sz.w, sz.h, squared, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

// TestSGRApplySeparableMatchesReference drives the full
// ApplySelfguidedRestoration with the separable box sum bound and compares the
// destination against the pure-Go brute-force reference across bit depths, all
// 16 parameter sets, and a mix of sizes.
func TestSGRApplySeparableMatchesReference(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x0B0C)
	sizes := []struct{ w, h int }{{1, 1}, {4, 4}, {5, 5}, {8, 8}, {13, 11}, {31, 64}, {64, 16}, {64, 64}, {2, 3}, {7, 7}}
	for _, bitDepth := range []uint8{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		for _, sz := range sizes {
			for eps := range SGRProjParams {
				stride := sz.w + 2*SGRProjBorderHorz + 5
				origin := SGRProjBorderVert*stride + SGRProjBorderHorz
				src := make([]uint16, stride*(sz.h+2*SGRProjBorderVert))
				for i := range src {
					src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
				}
				scratchLen, err := SelfguidedScratchLen(sz.w, sz.h)
				if err != nil {
					t.Fatal(err)
				}
				xqd := [2]int8{int8(rnd.pseudoUniform(96) - 48), int8(rnd.pseudoUniform(96) - 48)}

				gotDst := make([]uint16, sz.w*sz.h)
				gotScratch := make([]int32, scratchLen)
				withBoxsum(boxsumSeparable, func() {
					if err := ApplySelfguidedRestoration(src, stride, origin, gotDst, sz.w, sz.w, sz.h, eps, xqd, bitDepth, gotScratch); err != nil {
						t.Fatalf("separable bd=%d sz=%dx%d eps=%d: %v", bitDepth, sz.w, sz.h, eps, err)
					}
				})

				wantDst := make([]uint16, sz.w*sz.h)
				wantScratch := make([]int32, scratchLen)
				withPureGoSGR(func() {
					if err := ApplySelfguidedRestoration(src, stride, origin, wantDst, sz.w, sz.w, sz.h, eps, xqd, bitDepth, wantScratch); err != nil {
						t.Fatalf("purego bd=%d sz=%dx%d eps=%d: %v", bitDepth, sz.w, sz.h, eps, err)
					}
				})

				if !slices.Equal(gotDst, wantDst) {
					for i := range wantDst {
						if gotDst[i] != wantDst[i] {
							t.Fatalf("bd=%d sz=%dx%d eps=%d dst[%d]=%d want %d", bitDepth, sz.w, sz.h, eps, i, gotDst[i], wantDst[i])
						}
					}
				}
			}
		}
	}
}

// TestBoxsumSeparableIsZeroAlloc protects the hot-path contract that the
// separable box sum allocates nothing per call.
func TestBoxsumSeparableIsZeroAlloc(t *testing.T) {
	src, so, w, h, ss, dst, ds := benchBoxsumInput()
	allocs := testing.AllocsPerRun(50, func() {
		boxsumSeparable(src, so, w, h, ss, 2, true, dst, ds)
		boxsumSeparable(src, so, w, h, ss, 2, false, dst, ds)
		boxsumSeparable(src, so, w, h, ss, 1, true, dst, ds)
		boxsumSeparable(src, so, w, h, ss, 1, false, dst, ds)
	})
	if allocs != 0 {
		t.Fatalf("boxsumSeparable allocated %f times per call", allocs)
	}
}

// withBoxsum temporarily binds the box-sum dispatch slot to fn's kernel while
// keeping the blend kernels as-is, runs body, then restores the binding.
func withBoxsum(kernel func([]int32, int, int, int, int, int, bool, []int32, int), body func()) {
	prev := boxsumImpl
	boxsumImpl = kernel
	defer func() { boxsumImpl = prev }()
	body()
}

// benchBoxsumInput builds a representative extended-block source for a 64x64 SGR
// unit (widthExt=heightExt=70) matching the decoder's calculateIntermediate call.
func benchBoxsumInput() (src []int32, srcOrigin, width, height, srcStride int, dst []int32, dstStride int) {
	width = 64 + 2*SGRProjBorderHorz
	height = 64 + 2*SGRProjBorderVert
	srcStride = width
	src = make([]int32, srcStride*height)
	rnd := newRestorationRandom(restorationDeterministicSeed)
	for i := range src {
		src[i] = int32(rnd.pseudoUniform(4096))
	}
	dstStride = sgrBufferStride(64)
	dst = make([]int32, dstStride*height)
	return src, 0, width, height, srcStride, dst, dstStride
}

func BenchmarkBoxsumScalar(b *testing.B) {
	src, so, w, h, ss, dst, ds := benchBoxsumInput()
	b.ReportAllocs()
	for b.Loop() {
		boxsum(src, so, w, h, ss, 2, true, dst, ds)
		boxsum(src, so, w, h, ss, 2, false, dst, ds)
	}
}

func BenchmarkBoxsumSeparable(b *testing.B) {
	src, so, w, h, ss, dst, ds := benchBoxsumInput()
	b.ReportAllocs()
	for b.Loop() {
		boxsumSeparable(src, so, w, h, ss, 2, true, dst, ds)
		boxsumSeparable(src, so, w, h, ss, 2, false, dst, ds)
	}
}

func BenchmarkBoxsumSeparableR1(b *testing.B) {
	src, so, w, h, ss, dst, ds := benchBoxsumInput()
	b.ReportAllocs()
	for b.Loop() {
		boxsumSeparable(src, so, w, h, ss, 1, true, dst, ds)
		boxsumSeparable(src, so, w, h, ss, 1, false, dst, ds)
	}
}
