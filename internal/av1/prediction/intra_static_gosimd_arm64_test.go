// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import (
	"math/rand"
	"reflect"
	"runtime"
	"testing"
)

func simdFuncName(v any) string {
	return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
}

// TestIntraStaticSIMDIsBound guards the wiring: under the goexperiment.simd
// build the PAETH/SMOOTH dispatch slots must resolve to the Go-native SIMD
// kernels, while the remaining intra kernels keep their NEON asm (still compiled
// here). If either regresses, the dispatch inits are fighting over the slots.
func TestIntraStaticSIMDIsBound(t *testing.T) {
	cases := []struct {
		name     string
		got      any
		wantFunc any
	}{
		{"paeth", predictPaethImpl, predictPaethSIMD},
		{"smooth", predictSmoothImpl, predictSmoothSIMD},
		{"smooth_v", predictSmoothVerticalImpl, predictSmoothVerticalSIMD},
		{"smooth_h", predictSmoothHorizontalImpl, predictSmoothHorizontalSIMD},
		{"sumSamples", sumSamplesImpl, sumSamplesNEON},
		{"applyCFL", applyCFLImpl, applyCFLNEON},
		{"subsampleLuma8", subsampleLuma8Impl, subsampleLuma8NEON},
		{"dirRowInterp8", dirRowInterp8Impl, dirRowInterp8NEON},
		{"dirAboveRun8", dirAboveRun8Impl, dirAboveRun8NEON},
		{"dirLeftCol8", dirLeftCol8Impl, dirLeftCol8NEON},
		{"filterIntra8", predictFilterIntra8Impl, predictFilterIntraBlockDirect8NEON},
		{"filterIntra16", predictFilterIntra16Impl, predictFilterIntraBlockDirect16NEON},
	}
	for _, c := range cases {
		if got, want := simdFuncName(c.got), simdFuncName(c.wantFunc); got != want {
			t.Errorf("%s dispatch = %s, want %s", c.name, got, want)
		}
	}
}

// randomEdges fills above/left edge sample slices in [0,max] from rng.
func randomEdges(rng *rand.Rand, w, h int, max uint16) (above, left []uint16, aboveLeft uint16) {
	above = make([]uint16, w)
	left = make([]uint16, h)
	for i := range above {
		above[i] = uint16(rng.Intn(int(max) + 1))
	}
	for i := range left {
		left[i] = uint16(rng.Intn(int(max) + 1))
	}
	aboveLeft = uint16(rng.Intn(int(max) + 1))
	return
}

// TestPaethSIMDMatchesPureGo is the randomized byte-exactness differential for
// the Go-native SIMD PAETH kernel: it must equal predictPaethPureGo sample for
// sample over random edges, every AV1 block size and 8/10/12-bit depth. Widths
// of 4 exercise the scalar fallback; 8/16/32/64 exercise the vector path.
func TestPaethSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x9151))
	for _, depth := range []struct {
		bps int
		max uint16
	}{{1, 0xff}, {2, 0x3ff}, {2, 0xfff}} {
		for _, w := range dispatchBlockDims {
			for _, h := range dispatchBlockDims {
				for iter := 0; iter < 8; iter++ {
					above, left, aboveLeft := randomEdges(rng, w, h, depth.max)
					base := makeDispatchBlock(w, h, depth.bps)
					got := cloneBlock(base)
					want := cloneBlock(base)
					predictPaethSIMD(got, depth.bps, above, left, aboveLeft)
					predictPaethPureGo(want, depth.bps, above, left, aboveLeft)
					diffBlocks(t, "paeth-simd", int(depth.max), w*depth.bps, h, got, want)
				}
			}
		}
	}
}

// TestSmoothSIMDMatchesPureGo is the randomized byte-exactness differential for
// the full SMOOTH and the two 1D SMOOTH kernels.
func TestSmoothSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x7331))
	for _, depth := range []struct {
		bps int
		max uint16
	}{{1, 0xff}, {2, 0x3ff}, {2, 0xfff}} {
		for _, w := range dispatchBlockDims {
			for _, h := range dispatchBlockDims {
				weightsW, _ := smoothWeightsForSize(w)
				weightsH, _ := smoothWeightsForSize(h)
				for iter := 0; iter < 8; iter++ {
					above, left, _ := randomEdges(rng, w, h, depth.max)
					belowPred := left[h-1]
					rightPred := above[w-1]
					base := makeDispatchBlock(w, h, depth.bps)

					got := cloneBlock(base)
					want := cloneBlock(base)
					predictSmoothSIMD(got, depth.bps, weightsW, weightsH, above, left, belowPred, rightPred)
					predictSmoothPureGo(want, depth.bps, weightsW, weightsH, above, left, belowPred, rightPred)
					diffBlocks(t, "smooth-simd", int(depth.max), w*depth.bps, h, got, want)

					got = cloneBlock(base)
					want = cloneBlock(base)
					predictSmoothVerticalSIMD(got, depth.bps, weightsH, above, belowPred)
					predictSmoothVerticalPureGo(want, depth.bps, weightsH, above, belowPred)
					diffBlocks(t, "smooth_v-simd", int(depth.max), w*depth.bps, h, got, want)

					got = cloneBlock(base)
					want = cloneBlock(base)
					predictSmoothHorizontalSIMD(got, depth.bps, weightsW, left, rightPred)
					predictSmoothHorizontalPureGo(want, depth.bps, weightsW, left, rightPred)
					diffBlocks(t, "smooth_h-simd", int(depth.max), w*depth.bps, h, got, want)
				}
			}
		}
	}
}

// TestIntraStaticSIMDZeroAlloc protects the hot-path zero-allocation contract
// for the four Go-native SIMD kernels directly (the dispatch-level check already
// covers the resolved slots).
func TestIntraStaticSIMDZeroAlloc(t *testing.T) {
	const w, h = 32, 32
	above, _ := samplesFor(w, 0xff, 7)
	left, _ := samplesFor(h, 0xff, 99)
	weightsW, _ := smoothWeightsForSize(w)
	weightsH, _ := smoothWeightsForSize(h)
	block := makeDispatchBlock(w, h, 1)

	checks := []struct {
		name string
		fn   func()
	}{
		{"paeth", func() { predictPaethSIMD(block, 1, above, left, 123) }},
		{"smooth", func() { predictSmoothSIMD(block, 1, weightsW, weightsH, above, left, left[h-1], above[w-1]) }},
		{"smooth_v", func() { predictSmoothVerticalSIMD(block, 1, weightsH, above, left[h-1]) }},
		{"smooth_h", func() { predictSmoothHorizontalSIMD(block, 1, weightsW, left, above[w-1]) }},
	}
	for _, c := range checks {
		if allocs := testing.AllocsPerRun(200, c.fn); allocs != 0 {
			t.Errorf("%s SIMD allocated %f times per call", c.name, allocs)
		}
	}
}

// Benchmarks: SIMD vs NEON asm vs pure-Go for each kernel across block sizes.

func benchPaethVariants(b *testing.B, fn predictPaethFunc) {
	for _, s := range benchSizes() {
		w, h := s[0], s[1]
		above, left, _, _ := benchInputs(w, h)
		block := makeDispatchBlock(w, h, 1)
		b.Run(benchName(w, h, ""), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fn(block, 1, above, left, 123)
			}
		})
	}
}

func BenchmarkPaethSIMD(b *testing.B) { benchPaethVariants(b, predictPaethSIMD) }
func BenchmarkPaethNEON(b *testing.B) { benchPaethVariants(b, predictPaethNEON) }
func BenchmarkPaethPure(b *testing.B) { benchPaethVariants(b, predictPaethPureGo) }

func benchSmoothVariants(b *testing.B, fn predictSmoothFunc) {
	for _, s := range benchSizes() {
		w, h := s[0], s[1]
		above, left, weightsW, weightsH := benchInputs(w, h)
		block := makeDispatchBlock(w, h, 1)
		b.Run(benchName(w, h, ""), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fn(block, 1, weightsW, weightsH, above, left, left[h-1], above[w-1])
			}
		})
	}
}

func BenchmarkSmoothSIMD(b *testing.B) { benchSmoothVariants(b, predictSmoothSIMD) }
func BenchmarkSmoothNEON(b *testing.B) { benchSmoothVariants(b, predictSmoothNEON) }
func BenchmarkSmoothPure(b *testing.B) { benchSmoothVariants(b, predictSmoothPureGo) }

func benchSmoothVVariants(b *testing.B, fn predictSmoothVerticalFunc) {
	for _, s := range benchSizes() {
		w, h := s[0], s[1]
		above, left, _, weightsH := benchInputs(w, h)
		block := makeDispatchBlock(w, h, 1)
		b.Run(benchName(w, h, ""), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fn(block, 1, weightsH, above, left[h-1])
			}
		})
	}
}

func BenchmarkSmoothVSIMD(b *testing.B) { benchSmoothVVariants(b, predictSmoothVerticalSIMD) }
func BenchmarkSmoothVNEON(b *testing.B) { benchSmoothVVariants(b, predictSmoothVerticalNEON) }
func BenchmarkSmoothVPure(b *testing.B) { benchSmoothVVariants(b, predictSmoothVerticalPureGo) }

func benchSmoothHVariants(b *testing.B, fn predictSmoothHorizontalFunc) {
	for _, s := range benchSizes() {
		w, h := s[0], s[1]
		above, left, weightsW, _ := benchInputs(w, h)
		block := makeDispatchBlock(w, h, 1)
		b.Run(benchName(w, h, ""), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fn(block, 1, weightsW, left, above[w-1])
			}
		})
	}
}

func BenchmarkSmoothHSIMD(b *testing.B) { benchSmoothHVariants(b, predictSmoothHorizontalSIMD) }
func BenchmarkSmoothHNEON(b *testing.B) { benchSmoothHVariants(b, predictSmoothHorizontalNEON) }
func BenchmarkSmoothHPure(b *testing.B) { benchSmoothHVariants(b, predictSmoothHorizontalPureGo) }
