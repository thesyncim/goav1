// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package filmgrain_test

import (
	"math/rand"
	"testing"

	fg "github.com/thesyncim/goav1/internal/av1/filmgrain"
)

// benchApplyFrame applies film grain to every 32-line luma band (and the
// matching 4:2:0 chroma bands) of a full frame with overlap enabled, so the
// top-seam, left-seam and corner overlap regions are all exercised across a
// realistic block count. It only uses exported API so it can be built against
// both the fused and the pre-fusion trees for an A/B comparison.
func benchApplyFrame(b *testing.B, bd uint8) {
	const width, height = 512, 512
	const stride = width + 16
	rng := rand.New(rand.NewSource(0x5EED))

	grainLen := fg.LumaGrainSamples
	lumaGrain := make([]int16, grainLen)
	cbGrain := make([]int16, grainLen)
	crGrain := make([]int16, grainLen)
	grainMin := -(1 << (bd - 1))
	grainMax := (1 << (bd - 1)) - 1
	for i := range lumaGrain {
		lumaGrain[i] = int16(grainMin + rng.Intn(grainMax-grainMin+1))
		cbGrain[i] = int16(grainMin + rng.Intn(grainMax-grainMin+1))
		crGrain[i] = int16(grainMin + rng.Intn(grainMax-grainMin+1))
	}
	scaling := make([]uint8, fg.ScalingLUTSize)
	for i := range scaling {
		scaling[i] = uint8(rng.Intn(256))
	}

	luma := make([]uint16, height*stride)
	span := 1 << bd
	for i := range luma {
		luma[i] = uint16(rng.Intn(span))
	}
	cW, cH := width/2, height/2
	cStride := cW + 16
	cb := make([]uint16, cH*cStride)
	cr := make([]uint16, cH*cStride)
	for i := range cb {
		cb[i] = uint16(rng.Intn(span))
		cr[i] = uint16(rng.Intn(span))
	}

	work := make([]uint16, len(luma))
	cbWork := make([]uint16, len(cb))
	crWork := make([]uint16, len(cr))

	lumaRows := (height + 31) / 32
	chromaRows := (cH + 15) / 16

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		copy(work, luma)
		copy(cbWork, cb)
		copy(crWork, cr)
		for row := 0; row < lumaRows; row++ {
			start := row * 32 * stride
			h := 32
			if rem := height - row*32; rem < h {
				h = rem
			}
			_ = fg.ApplyLumaRow(work[start:], work[start:], lumaGrain, scaling, fg.LumaRowParams{
				Seed: 0x2AAA, Width: width, Height: h, Stride: stride, Row: row,
				BitDepth: bd, ScalingShift: 8, Overlap: true,
			})
		}
		for row := 0; row < chromaRows; row++ {
			cstart := row * 16 * cStride
			lstart := row * 32 * stride
			h := 16
			if rem := cH - row*16; rem < h {
				h = rem
			}
			p := fg.ChromaRowParams{
				Seed: 0x2AAA, Width: cW, Height: h, Stride: cStride, LumaStride: stride, Row: row,
				BitDepth: bd, ScalingShift: 8, SubsamplingX: true, SubsamplingY: true, Overlap: true,
				ChromaMult: 140, ChromaLumaMult: 118, ChromaOffset: 260,
			}
			_ = fg.ApplyChromaRow(cbWork[cstart:], cbWork[cstart:], work[lstart:], cbGrain, scaling, p)
			_ = fg.ApplyChromaRow(crWork[cstart:], crWork[cstart:], work[lstart:], crGrain, scaling, p)
		}
	}
}

func BenchmarkApplyFilmGrainFrame8(b *testing.B)  { benchApplyFrame(b, 8) }
func BenchmarkApplyFilmGrainFrame10(b *testing.B) { benchApplyFrame(b, 10) }
