// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE for the BSD-2-Clause grant.

package filmgrain

import (
	"math/rand"
	"slices"
	"testing"
)

// TestApplyRowsTableMatchRows checks the 10/12-bit table rows (dav1d's 16bpc
// block kernels) against the LUT-interpolating rows on every overlap,
// clipping, chroma layout and scaling mode, including partial edge blocks.
func TestApplyRowsTableMatchRows(t *testing.T) {
	rng := rand.New(rand.NewSource(0x16b))
	scaling := randomScalingLUT(rng)
	for _, bd := range []uint8{10, 12} {
		grainY := randomGrainTemplate(rng, LumaGrainSamples, bd)
		grainUV := randomGrainTemplate(rng, ChromaGrainSamples, bd)
		var table [ScalingTableSize]uint8
		if err := BuildScalingTable(&table, scaling, bd); err != nil {
			t.Fatal(err)
		}
		for _, width := range []int{7, 16, 17, 32, 33, 65, 96} {
			for mode := 0; mode < 16; mode++ {
				height := 32
				if mode&1 != 0 {
					height = 1 + mode
				}
				p := LumaRowParams{Seed: uint16(mode*977 + int(bd)), Width: width, Height: height, Stride: width + 5, Row: mode % 3,
					BitDepth: bd, ScalingShift: uint8(8 + mode%4), Overlap: mode&2 != 0, ClipToRestrictedRange: mode&4 != 0}
				src := randomSamplePlane(rng, height*p.Stride+11, bd)
				want, got := slices.Clone(src), slices.Clone(src)
				if err := ApplyLumaRow(want, src, grainY, scaling, p); err != nil {
					t.Fatal(err)
				}
				if err := ApplyLumaRowTable(got, src, grainY, scaling, &table, p); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(got, want) {
					t.Fatalf("luma bd=%d width=%d mode=%d", bd, width, mode)
				}
				for _, ss := range [][2]bool{{false, false}, {true, false}, {true, true}} {
					sx, sy := chromaSubsamplingShift(ss[0]), chromaSubsamplingShift(ss[1])
					h := min(height, 32>>sy)
					cp := ChromaRowParams{Seed: p.Seed, Width: width, Height: h, Stride: width + 3, LumaStride: (width << sx) + 7,
						Row: p.Row, BitDepth: bd, ScalingShift: p.ScalingShift, Overlap: p.Overlap, ClipToRestrictedRange: p.ClipToRestrictedRange,
						SubsamplingX: ss[0], SubsamplingY: ss[1], ChromaScalingFromLuma: mode&1 != 0, MatrixIdentity: mode&8 != 0,
						ChromaMult: uint8(mode * 17), ChromaLumaMult: uint8(255 - mode*17), ChromaOffset: uint16(mode * 34)}
					uv := randomSamplePlane(rng, h*cp.Stride+11, bd)
					lum := randomSamplePlane(rng, (h<<sy)*cp.LumaStride, bd)
					want, got := slices.Clone(uv), slices.Clone(uv)
					if err := ApplyChromaRow(want, uv, lum, grainUV, scaling, cp); err != nil {
						t.Fatal(err)
					}
					if err := ApplyChromaRowTable(got, uv, lum, grainUV, scaling, &table, cp); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(got, want) {
						t.Fatalf("chroma bd=%d width=%d mode=%d ss=%v", bd, width, mode, ss)
					}
				}
			}
		}
	}
}
