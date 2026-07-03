// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package restoration

import (
	"slices"
	"testing"
)

// withPureGoSGRAVX2 temporarily forces the SGR dispatch slots to the pure-Go
// reference, runs fn, then restores the previous bindings.
func withPureGoSGRAVX2(fn func()) {
	pb, ps, pf := boxsumImpl, selfguidedImpl, selfguidedFastImpl
	boxsumImpl, selfguidedImpl, selfguidedFastImpl = boxsum, selfguided, selfguidedFast
	defer func() { boxsumImpl, selfguidedImpl, selfguidedFastImpl = pb, ps, pf }()
	fn()
}

// withAVX2SGR temporarily forces the SGR dispatch slots to the AVX2 kernels
// (independent of cpu.Detected, so the SIMD is exercised even when
// auto-dispatch falls back under Rosetta), runs fn, then restores.
func withAVX2SGR(fn func()) {
	pb, ps, pf := boxsumImpl, selfguidedImpl, selfguidedFastImpl
	boxsumImpl, selfguidedImpl, selfguidedFastImpl = boxsumAVX2, selfguidedAVX2, selfguidedFastAVX2
	defer func() { boxsumImpl, selfguidedImpl, selfguidedFastImpl = pb, ps, pf }()
	fn()
}

// TestSGRBlendAVX2MatchesPureGo drives the full ApplySelfguidedRestoration with
// the AVX2 kernels forced (box sums + blend) and again with the pure-Go
// reference, asserting the destinations are byte-identical across bit depths,
// all 16 parameter sets, and a mix of 8-aligned and ragged widths.
func TestSGRBlendAVX2MatchesPureGo(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xCAFE)
	sizes := []struct{ w, h int }{{4, 4}, {5, 5}, {8, 8}, {13, 11}, {31, 64}, {64, 16}, {64, 64}, {1, 1}, {2, 3}, {7, 7}, {16, 9}, {40, 40}}
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
				withAVX2SGR(func() {
					if err := ApplySelfguidedRestoration(src, stride, origin, gotDst, sz.w, sz.w, sz.h, eps, xqd, bitDepth, scratch); err != nil {
						t.Fatalf("avx2 bd=%d sz=%dx%d eps=%d: %v", bitDepth, sz.w, sz.h, eps, err)
					}
				})

				wantDst := make([]uint16, sz.w*sz.h)
				wantScratch := make([]int32, scratchLen)
				withPureGoSGRAVX2(func() {
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

// TestSGRBlendAVX2IsZeroAlloc protects the hot-path contract that the AVX2
// blend kernels do not allocate per call (beyond the scratch the caller owns).
func TestSGRBlendAVX2IsZeroAlloc(t *testing.T) {
	const w, h = 64, 64
	const bitDepth uint8 = 12
	max := uint16((1 << bitDepth) - 1)
	stride := w + 2*SGRProjBorderHorz
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	src := make([]uint16, stride*(h+2*SGRProjBorderVert))
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x9977)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
	}
	dst := make([]uint16, w*h)
	scratchLen, _ := SelfguidedScratchLen(w, h)
	scratch := make([]int32, scratchLen)
	withAVX2SGR(func() {
		if allocs := testing.AllocsPerRun(50, func() {
			_ = ApplySelfguidedRestoration(src, stride, origin, dst, w, w, h, 15, [2]int8{8, 11}, bitDepth, scratch)
		}); allocs != 0 {
			t.Fatalf("AVX2 SGR allocated %f times per call", allocs)
		}
	})
}

func BenchmarkApplySelfguidedRestorationAVX2(b *testing.B) {
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
	withAVX2SGR(func() {
		for b.Loop() {
			_ = ApplySelfguidedRestoration(src, stride, origin, dst, 64, 64, 64, 15, [2]int8{8, 11}, 12, scratch)
		}
	})
}
