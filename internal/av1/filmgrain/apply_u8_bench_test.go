// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE for the BSD-2-Clause grant.

package filmgrain

import (
	"math/rand"
	"testing"
)

// benchApplyU8 places grain on one 1280x720 4:2:0 8-bit frame through the
// display (U8) row kernels, the decoder's direct path.
func benchApplyU8(b *testing.B, chroma bool, csfl bool) {
	const w, h = 1280, 720
	rng := rand.New(rand.NewSource(1))
	grainY := randomGrainTemplate(rng, LumaGrainSamples, 8)
	grainUV := randomGrainTemplate(rng, ChromaGrainSamples, 8)
	scaling := randomScalingLUT(rng)
	luma := make([]byte, w*h)
	for i := range luma {
		luma[i] = byte(rng.Intn(256))
	}
	dst := make([]byte, w*h)
	uv := make([]byte, w/2*h/2)
	for i := range uv {
		uv[i] = byte(rng.Intn(256))
	}
	b.ResetTimer()
	for b.Loop() {
		if !chroma {
			for row := 0; row*32 < h; row++ {
				p := LumaRowParams{Seed: 7, Width: w, Height: min(32, h-row*32), Stride: w, Row: row, BitDepth: 8, ScalingShift: 10, Overlap: true}
				if err := ApplyLumaRowU8(dst[row*32*w:], luma[row*32*w:], grainY, scaling, p); err != nil {
					b.Fatal(err)
				}
			}
			continue
		}
		for row := 0; row*16 < h/2; row++ {
			p := ChromaRowParams{Seed: 7, Width: w / 2, Height: min(16, h/2-row*16), Stride: w / 2, LumaStride: w, Row: row, BitDepth: 8,
				ScalingShift: 10, SubsamplingX: true, SubsamplingY: true, Overlap: true, ChromaScalingFromLuma: csfl,
				ChromaMult: 100, ChromaLumaMult: 160, ChromaOffset: 300}
			if err := ApplyChromaRowU8(dst[row*16*w/2:], uv[row*16*w/2:], luma[row*32*w:], grainUV, scaling, p); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkApplyU8Luma720p(b *testing.B)       { benchApplyU8(b, false, false) }
func BenchmarkApplyU8Chroma720p(b *testing.B)     { benchApplyU8(b, true, false) }
func BenchmarkApplyU8ChromaCSFL720p(b *testing.B) { benchApplyU8(b, true, true) }
