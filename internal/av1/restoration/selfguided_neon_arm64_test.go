// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

import (
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// TestSGRNEONReportsDetection records whether the dispatcher bound the NEON
// kernels (always true on arm64 hosts Go runs on).
func TestSGRNEONReportsDetection(t *testing.T) {
	t.Logf("cpu.Detected.NEON = %v", cpu.Detected.NEON)
}

// TestBoxsumNEONMatchesPureGo compares the NEON box-filter sums against the
// scalar reference across radii, sizes (including widths smaller than the
// 4-lane interior band so the wrapper must fall back), and both the plain and
// squared variants.
func TestBoxsumNEONMatchesPureGo(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xB0B)
	sizes := []struct{ w, h int }{{4, 4}, {5, 3}, {8, 8}, {17, 19}, {64, 64}, {70, 70}, {1, 1}, {3, 9}, {64, 1}}
	for _, r := range []int{1, 2} {
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
				boxsumNEON(src, 0, width, height, srcStride, r, squared, got, stride)
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

// TestSGRBlendNEONMatchesPureGo drives the full ApplySelfguidedRestoration with
// the NEON kernels bound (the production dispatch) and again with the pure-Go
// reference forced, asserting the destinations are byte-identical across bit
// depths, all 16 parameter sets, and a mix of 4-aligned and ragged widths.
func TestSGRBlendNEONMatchesPureGo(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xCAFE)
	sizes := []struct{ w, h int }{{4, 4}, {5, 5}, {8, 8}, {13, 11}, {31, 64}, {64, 16}, {64, 64}, {1, 1}, {2, 3}, {7, 7}}
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
				scratch := make([]int32, scratchLen)
				if err := ApplySelfguidedRestoration(src, stride, origin, gotDst, sz.w, sz.w, sz.h, eps, xqd, bitDepth, scratch); err != nil {
					t.Fatalf("neon bd=%d sz=%dx%d eps=%d: %v", bitDepth, sz.w, sz.h, eps, err)
				}

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

// withPureGoSGR temporarily forces the SGR dispatch slots to the pure-Go
// reference, runs fn, then restores the previous bindings.
func withPureGoSGR(fn func()) {
	pb, ps, pf := boxsumImpl, selfguidedImpl, selfguidedFastImpl
	boxsumImpl, selfguidedImpl, selfguidedFastImpl = boxsum, selfguided, selfguidedFast
	defer func() { boxsumImpl, selfguidedImpl, selfguidedFastImpl = pb, ps, pf }()
	fn()
}

func BenchmarkApplySelfguidedRestorationPureGo(b *testing.B) {
	scratchLen, _ := SelfguidedScratchLen(64, 64)
	scratch := make([]int32, scratchLen)
	stride := 64 + 2*SGRProjBorderHorz
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	src := make([]uint16, stride*(64+2*SGRProjBorderVert))
	dst := make([]uint16, 64*64)
	rnd := newRestorationRandom(restorationDeterministicSeed)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(1 << 12))
	}
	b.SetBytes(int64(64 * 64 * 2))
	b.ReportAllocs()
	pb, ps, pf := boxsumImpl, selfguidedImpl, selfguidedFastImpl
	boxsumImpl, selfguidedImpl, selfguidedFastImpl = boxsum, selfguided, selfguidedFast
	defer func() { boxsumImpl, selfguidedImpl, selfguidedFastImpl = pb, ps, pf }()
	for b.Loop() {
		_ = ApplySelfguidedRestoration(src, stride, origin, dst, 64, 64, 64, 15, [2]int8{8, 11}, 12, scratch)
	}
}
