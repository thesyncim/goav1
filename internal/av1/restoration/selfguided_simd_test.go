// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package restoration

import (
	"reflect"
	"runtime"
	"slices"
	"testing"
)

// withSIMDSGR temporarily forces the SGR dispatch slots to the Go-native SIMD
// kernels (box sums + blend), independent of the detection gate, so the SIMD path
// runs regardless of the host. It restores the previous bindings afterwards.
func withSIMDSGR(fn func()) {
	pb, ps, pf := boxsumImpl, selfguidedImpl, selfguidedFastImpl
	boxsumImpl, selfguidedImpl, selfguidedFastImpl = sgrSIMDBoxsumKernel, selfguidedSIMD, selfguidedFastSIMD
	defer func() { boxsumImpl, selfguidedImpl, selfguidedFastImpl = pb, ps, pf }()
	fn()
}

// TestSGRSIMDIsBound guards the wiring: when the host detects the SIMD features the
// self-guided dispatch slots resolve to the Go-native SIMD kernels.
func TestSGRSIMDIsBound(t *testing.T) {
	if !sgrSIMDDetected() {
		t.Skip("Go SIMD detection failed on this host")
	}
	nameOf := func(v any) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	for _, c := range []struct {
		name string
		got  any
		want any
	}{
		{"boxsumImpl", boxsumImpl, sgrSIMDBoxsumKernel},
		{"selfguidedImpl", selfguidedImpl, selfguidedSIMD},
		{"selfguidedFastImpl", selfguidedFastImpl, selfguidedFastSIMD},
	} {
		if got, want := nameOf(c.got), nameOf(c.want); got != want {
			t.Fatalf("%s = %s, want %s", c.name, got, want)
		}
	}
}

// TestSGRBlendRowSIMDMatchesScalar is the direct differential for the nine-tap
// blend: sgrBlendRowSIMD must equal the scalar stencil formula for every weight
// set the driver uses, over random and extreme int32 A/B/dgd values, with
// lane-aligned and ragged column counts.
func TestSGRBlendRowSIMDMatchesScalar(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x51d1)
	weightSets := [][9]int32{
		{3, 4, 3, 4, 4, 4, 3, 4, 3},
		{5, 6, 5, 0, 0, 0, 5, 6, 5},
		{0, 0, 0, 5, 6, 5, 0, 0, 0},
	}
	shifts := []int{SGRProjSgrBits + 5 - SGRProjRstBits, SGRProjSgrBits + 4 - SGRProjRstBits, SGRProjSgrBits + 5 - SGRProjRstBits + 1}
	for iter := 0; iter < 200; iter++ {
		cols := 1 + rnd.pseudoUniform(40)
		n := cols + 4
		fill := func() []int32 {
			s := make([]int32, n+2)
			for i := range s {
				switch rnd.pseudoUniform(8) {
				case 0:
					s[i] = 0x7fffffff >> rnd.pseudoUniform(4)
				case 1:
					s[i] = -0x7fffffff >> rnd.pseudoUniform(4)
				default:
					s[i] = int32(rnd.pseudoUniform(1<<20)) - (1 << 19)
				}
			}
			return s
		}
		aPrev, aCur, aNext := fill(), fill(), fill()
		bPrev, bCur, bNext := fill(), fill(), fill()
		dgd := fill()
		for _, w := range weightSets {
			for _, shift := range shifts {
				lanes := cols &^ (sgrBlendLanes - 1)
				if lanes == 0 {
					continue
				}
				got := make([]int32, lanes)
				sgrBlendRowSIMD(got, dgd, aPrev, aCur, aNext, bPrev, bCur, bNext, &w, shift, lanes)
				for c := 0; c < lanes; c++ {
					a := aPrev[c]*w[0] + aPrev[c+1]*w[1] + aPrev[c+2]*w[2] +
						aCur[c]*w[3] + aCur[c+1]*w[4] + aCur[c+2]*w[5] +
						aNext[c]*w[6] + aNext[c+1]*w[7] + aNext[c+2]*w[8]
					b := bPrev[c]*w[0] + bPrev[c+1]*w[1] + bPrev[c+2]*w[2] +
						bCur[c]*w[3] + bCur[c+1]*w[4] + bCur[c+2]*w[5] +
						bNext[c]*w[6] + bNext[c+1]*w[7] + bNext[c+2]*w[8]
					want := roundPowerOfTwo(a*dgd[c]+b, shift)
					if got[c] != want {
						t.Fatalf("iter=%d w=%v shift=%d col=%d got=%d want=%d", iter, w, shift, c, got[c], want)
					}
				}
			}
		}
	}
}

// TestSGRSIMDMatchesPureGo drives the full ApplySelfguidedRestoration with the SIMD
// kernels forced (box sums + blend) and again with the pure-Go reference, asserting
// the destinations are byte-identical across bit depths, all parameter sets, and a
// mix of lane-aligned and ragged widths.
func TestSGRSIMDMatchesPureGo(t *testing.T) {
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
				withSIMDSGR(func() {
					if err := ApplySelfguidedRestoration(src, stride, origin, gotDst, sz.w, sz.w, sz.h, eps, xqd, bitDepth, scratch); err != nil {
						t.Fatalf("simd bd=%d sz=%dx%d eps=%d: %v", bitDepth, sz.w, sz.h, eps, err)
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

// TestSGRSIMDIsZeroAlloc protects the hot-path contract that the SIMD self-guided
// drivers do not allocate per call beyond the scratch the caller owns.
func TestSGRSIMDIsZeroAlloc(t *testing.T) {
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
	withSIMDSGR(func() {
		if allocs := testing.AllocsPerRun(50, func() {
			_ = ApplySelfguidedRestoration(src, stride, origin, dst, w, w, h, 15, [2]int8{8, 11}, bitDepth, scratch)
		}); allocs != 0 {
			t.Fatalf("SIMD SGR allocated %f times per call", allocs)
		}
	})
}

func BenchmarkApplySelfguidedRestorationSIMD(b *testing.B) {
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
	withSIMDSGR(func() {
		for b.Loop() {
			_ = ApplySelfguidedRestoration(src, stride, origin, dst, 64, 64, 64, 15, [2]int8{8, 11}, 12, scratch)
		}
	})
}
