// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package filmgrain

import (
	"math/rand"
	"testing"
)

// This file cross-checks the fused overlap apply path (blendGrainRow +
// gather + applyGrainSegment, apply.go) against an independent per-sample
// reference that walks every grain-block region with the primitive
// blendLumaOverlap / blendChromaOverlap / applyLumaSample / applyChromaSample
// helpers exactly as the pre-fusion loops did. It covers all overlap positions
// (interior, left seam, top seam, corner) across every bit depth and chroma
// subsampling, which the two committed 4:2:0 film-grain conformance clips do not
// span on their own.

// applyLumaRowRef is a per-sample reference mirroring ApplyLumaRow's outer
// offset/RNG bookkeeping and the original four-region blend.
func applyLumaRowRef(t *testing.T, dst []uint16, src []uint16, grain []int16, scaling []uint8, params LumaRowParams) {
	rows := 1
	if params.Overlap && params.Row > 0 {
		rows = 2
	}
	var randoms [2]Random
	for i := 0; i < rows; i++ {
		rng, err := NewStripeRandom(params.Seed, (params.Row-i)*LumaBlockSize)
		if err != nil {
			t.Fatal(err)
		}
		randoms[i] = rng
	}
	var offsets [2][2]uint8
	for blockX := 0; blockX < params.Width; blockX += LumaBlockSize {
		blockWidth := LumaBlockSize
		if remaining := params.Width - blockX; remaining < blockWidth {
			blockWidth = remaining
		}
		if params.Overlap && blockX > 0 {
			for i := 0; i < rows; i++ {
				offsets[1][i] = offsets[0][i]
			}
		}
		for i := 0; i < rows; i++ {
			offset, _ := randoms[i].Number(8)
			offsets[0][i] = uint8(offset)
		}
		yStart := 0
		if params.Overlap && params.Row > 0 {
			yStart = min(params.Height, LumaOverlapSamples)
		}
		xStart := 0
		if params.Overlap && blockX > 0 {
			xStart = min(blockWidth, LumaOverlapSamples)
		}
		for y := 0; y < params.Height; y++ {
			for x := 0; x < blockWidth; x++ {
				var g int16
				switch {
				case y >= yStart && x >= xStart:
					g = lumaGrainSample(grain, offsets[0][0], 0, 0, x, y)
				case y >= yStart && x < xStart:
					current := lumaGrainSample(grain, offsets[0][0], 0, 0, x, y)
					left := lumaGrainSample(grain, offsets[1][0], 1, 0, x, y)
					g = blendLumaOverlap(left, current, x, params.BitDepth)
				case y < yStart && x >= xStart:
					current := lumaGrainSample(grain, offsets[0][0], 0, 0, x, y)
					top := lumaGrainSample(grain, offsets[0][1], 0, 1, x, y)
					g = blendLumaOverlap(top, current, y, params.BitDepth)
				default:
					top := lumaGrainSample(grain, offsets[0][1], 0, 1, x, y)
					topLeft := lumaGrainSample(grain, offsets[1][1], 1, 1, x, y)
					top = blendLumaOverlap(topLeft, top, x, params.BitDepth)
					current := lumaGrainSample(grain, offsets[0][0], 0, 0, x, y)
					left := lumaGrainSample(grain, offsets[1][0], 1, 0, x, y)
					current = blendLumaOverlap(left, current, x, params.BitDepth)
					g = blendLumaOverlap(top, current, y, params.BitDepth)
				}
				applyLumaRowSample(dst, src, scaling, params, blockX+x, y, g)
			}
		}
	}
}

// applyChromaRowRef is the chroma counterpart of applyLumaRowRef.
func applyChromaRowRef(t *testing.T, dst []uint16, src []uint16, luma []uint16, grain []int16, scaling []uint8, params ChromaRowParams) {
	rows := 1
	if params.Overlap && params.Row > 0 {
		rows = 2
	}
	var randoms [2]Random
	for i := 0; i < rows; i++ {
		rng, err := NewStripeRandom(params.Seed, (params.Row-i)*LumaBlockSize)
		if err != nil {
			t.Fatal(err)
		}
		randoms[i] = rng
	}
	shiftX := chromaSubsamplingShift(params.SubsamplingX)
	shiftY := chromaSubsamplingShift(params.SubsamplingY)
	chromaBlockWidth := LumaBlockSize >> shiftX
	var offsets [2][2]uint8
	for blockX := 0; blockX < params.Width; blockX += chromaBlockWidth {
		blockWidth := chromaBlockWidth
		if remaining := params.Width - blockX; remaining < blockWidth {
			blockWidth = remaining
		}
		if params.Overlap && blockX > 0 {
			for i := 0; i < rows; i++ {
				offsets[1][i] = offsets[0][i]
			}
		}
		for i := 0; i < rows; i++ {
			offset, _ := randoms[i].Number(8)
			offsets[0][i] = uint8(offset)
		}
		yStart := 0
		if params.Overlap && params.Row > 0 {
			yStart = min(params.Height, LumaOverlapSamples>>shiftY)
		}
		xStart := 0
		if params.Overlap && blockX > 0 {
			xStart = min(blockWidth, LumaOverlapSamples>>shiftX)
		}
		for y := 0; y < params.Height; y++ {
			for x := 0; x < blockWidth; x++ {
				var g int16
				switch {
				case y >= yStart && x >= xStart:
					g = chromaGrainSample(grain, offsets[0][0], shiftX, shiftY, 0, 0, x, y)
				case y >= yStart && x < xStart:
					current := chromaGrainSample(grain, offsets[0][0], shiftX, shiftY, 0, 0, x, y)
					left := chromaGrainSample(grain, offsets[1][0], shiftX, shiftY, 1, 0, x, y)
					g = blendChromaOverlap(left, current, x, shiftX, params.BitDepth)
				case y < yStart && x >= xStart:
					current := chromaGrainSample(grain, offsets[0][0], shiftX, shiftY, 0, 0, x, y)
					top := chromaGrainSample(grain, offsets[0][1], shiftX, shiftY, 0, 1, x, y)
					g = blendChromaOverlap(top, current, y, shiftY, params.BitDepth)
				default:
					top := chromaGrainSample(grain, offsets[0][1], shiftX, shiftY, 0, 1, x, y)
					topLeft := chromaGrainSample(grain, offsets[1][1], shiftX, shiftY, 1, 1, x, y)
					top = blendChromaOverlap(topLeft, top, x, shiftX, params.BitDepth)
					current := chromaGrainSample(grain, offsets[0][0], shiftX, shiftY, 0, 0, x, y)
					left := chromaGrainSample(grain, offsets[1][0], shiftX, shiftY, 1, 0, x, y)
					current = blendChromaOverlap(left, current, x, shiftX, params.BitDepth)
					g = blendChromaOverlap(top, current, y, shiftY, params.BitDepth)
				}
				applyChromaRowSample(dst, src, luma, scaling, params, shiftX, blockX+x, y, g)
			}
		}
	}
}

func randomGrainTemplate(rng *rand.Rand, n int, bd uint8) []int16 {
	grainMin := -(1 << (bd - 1))
	grainMax := (1 << (bd - 1)) - 1
	span := grainMax - grainMin + 1
	out := make([]int16, n)
	for i := range out {
		out[i] = int16(grainMin + rng.Intn(span))
	}
	return out
}

func randomSamplePlane(rng *rand.Rand, n int, bd uint8) []uint16 {
	span := 1 << bd
	out := make([]uint16, n)
	for i := range out {
		out[i] = uint16(rng.Intn(span))
	}
	return out
}

func randomScalingLUT(rng *rand.Rand) []uint8 {
	lut := make([]uint8, ScalingLUTSize)
	for i := range lut {
		lut[i] = uint8(rng.Intn(256))
	}
	return lut
}

func TestApplyLumaRowFusedMatchesPerSampleReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0x0A11))
	for _, bd := range []uint8{8, 10, 12} {
		for _, shift := range []uint8{8, 11} {
			for _, restricted := range []bool{false, true} {
				// Multiple rows so Row>0 exercises the top seam, and Width>32
				// so blockX>0 exercises the left seam and the corner.
				for _, row := range []int{0, 1, 2} {
					const width, height, stride = 70, 32, 96
					src := randomSamplePlane(rng, height*stride, bd)
					grain := randomGrainTemplate(rng, LumaGrainSamples, bd)
					scaling := randomScalingLUT(rng)
					params := LumaRowParams{
						Seed:                  uint16(0x1234 + row),
						Width:                 width,
						Height:                height,
						Stride:                stride,
						Row:                   row,
						BitDepth:              bd,
						ScalingShift:          shift,
						Overlap:               true,
						ClipToRestrictedRange: restricted,
					}
					got := make([]uint16, len(src))
					want := make([]uint16, len(src))
					copy(got, src)
					copy(want, src)
					if err := ApplyLumaRow(got, got, grain, scaling, params); err != nil {
						t.Fatal(err)
					}
					applyLumaRowRef(t, want, want, grain, scaling, params)
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("bd=%d shift=%d restricted=%v row=%d i=%d: fused=%d ref=%d",
								bd, shift, restricted, row, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestApplyChromaRowFusedMatchesPerSampleReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0x0C22))
	type sub struct {
		x, y bool
	}
	for _, bd := range []uint8{8, 10, 12} {
		for _, ss := range []sub{{true, true}, {true, false}, {false, false}} {
			for _, csfl := range []bool{false, true} {
				for _, restricted := range []bool{false, true} {
					for _, row := range []int{0, 1, 2} {
						shiftX := 0
						if ss.x {
							shiftX = 1
						}
						shiftY := 0
						if ss.y {
							shiftY = 1
						}
						const chromaWidth = 40
						blockH := LumaBlockSize >> shiftY
						chromaStride := chromaWidth + 8
						lumaStride := ((chromaWidth - 1) << shiftX) + 2 + 8
						lumaRows := ((blockH - 1) << shiftY) + 1 + 2
						src := randomSamplePlane(rng, blockH*chromaStride, bd)
						luma := randomSamplePlane(rng, lumaRows*lumaStride, bd)
						grain := randomGrainTemplate(rng, ChromaGrainSamples, bd)
						scaling := randomScalingLUT(rng)
						params := ChromaRowParams{
							Seed:                  uint16(0x5678 + row),
							Width:                 chromaWidth,
							Height:                blockH,
							Stride:                chromaStride,
							LumaStride:            lumaStride,
							Row:                   row,
							BitDepth:              bd,
							ScalingShift:          9,
							SubsamplingX:          ss.x,
							SubsamplingY:          ss.y,
							Overlap:               true,
							ClipToRestrictedRange: restricted,
							ChromaScalingFromLuma: csfl,
							ChromaMult:            140,
							ChromaLumaMult:        118,
							ChromaOffset:          260,
						}
						got := make([]uint16, len(src))
						want := make([]uint16, len(src))
						copy(got, src)
						copy(want, src)
						if err := ApplyChromaRow(got, got, luma, grain, scaling, params); err != nil {
							t.Fatalf("bd=%d ss=%v csfl=%v row=%d: %v", bd, ss, csfl, row, err)
						}
						applyChromaRowRef(t, want, want, luma, grain, scaling, params)
						for i := range want {
							if got[i] != want[i] {
								t.Fatalf("bd=%d ss=%v csfl=%v restricted=%v row=%d i=%d: fused=%d ref=%d",
									bd, ss, csfl, restricted, row, i, got[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}
