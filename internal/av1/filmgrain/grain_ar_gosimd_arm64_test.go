// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package filmgrain

import (
	"math/rand"
	"slices"
	"testing"
)

// TestARLag3MatchesScalar checks the vector lag-3 auto-regression against
// the scalar reference for luma and every chroma layout and bit depth.
func TestARLag3MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x3a3))
	for _, bd := range []uint8{8, 10, 12} {
		grainMin, grainMax := -(1 << (bd - 1)), (1<<(bd-1))-1
		for iter := 0; iter < 8; iter++ {
			shift := 6 + iter%4
			round := 1 << (shift - 1)
			lp := LumaGrainParams{BitDepth: bd}
			for i := range lp.ARCoeffs {
				lp.ARCoeffs[i] = int8(rng.Intn(256) - 128)
			}
			luma := randomGrainTemplate(rng, LumaGrainSamples, bd)
			want, got := slices.Clone(luma), slices.Clone(luma)
			generateLumaGrainARLag3(want, &lp, round, shift, grainMin, grainMax)
			lumaGrainARLag3(got, &lp, round, shift, grainMin, grainMax)
			if !slices.Equal(got, want) {
				t.Fatalf("luma bd=%d iter=%d", bd, iter)
			}
			for _, ss := range [][2]bool{{false, false}, {true, false}, {true, true}} {
				cp := ChromaGrainParams{BitDepth: bd, SubsamplingX: ss[0], SubsamplingY: ss[1], NumYPoints: uint8(iter % 2)}
				for i := range cp.ARCoeffs {
					cp.ARCoeffs[i] = int8(rng.Intn(256) - 128)
				}
				w, h := chromaGrainDimensions(ss[0], ss[1])
				uv := randomGrainTemplate(rng, ChromaGrainSamples, bd)
				want, got := slices.Clone(uv), slices.Clone(uv)
				generateChromaGrainARLag3(want, luma, &cp, w, h, round, shift, grainMin, grainMax)
				chromaGrainARLag3(got, luma, &cp, w, h, round, shift, grainMin, grainMax)
				if !slices.Equal(got, want) {
					t.Fatalf("chroma bd=%d iter=%d ss=%v", bd, iter, ss)
				}
			}
		}
	}
}
