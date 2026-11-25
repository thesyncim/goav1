package filmgrain

import (
	"bytes"
	"errors"
	"fmt"
	"math/rand"
	"slices"
	"testing"
	"unsafe"
)

// TestApplyChromaRowOddLumaWidth checks the last chroma column of a
// horizontally subsampled plane over an odd-width luma plane. dav1d's
// apply_grain_row pads luma[w] = luma[w-1] before fguv averages the luma
// pairs, so the reference here is the same row applied over a luma copy padded
// that way. The run under test gets an unpadded luma plane, cut off at the
// last visible sample of its last row and holding an extreme value past the
// edge on every other row, so a read past the edge either changes the output
// or panics. Every entry point (uint16 scalar, 10/12-bit table, 8-bit) and so
// every dispatched kernel is covered, at widths whose last chroma block is
// full (64), partial (51), a lone overlap-seam column (17) and one column (1).
func TestApplyChromaRowOddLumaWidth(t *testing.T) {
	rng := rand.New(rand.NewSource(0x0DD))
	type sub struct{ x, y bool }
	// detected records, per entry point, bit depth, layout and scaling mode,
	// whether some width and row made the pre-fix read past the edge change
	// the output, so every combination is shown to exercise the fix.
	detected := map[string]bool{}
	for _, bd := range []uint8{8, 10, 12} {
		for _, ss := range []sub{{true, true}, {true, false}} {
			for _, chromaWidth := range []int{64, 51, 17, 1} {
				for _, csfl := range []bool{false, true} {
					for _, row := range []int{0, 1} {
						lumaWidth := 2*chromaWidth - 1
						shiftY := 0
						if ss.y {
							shiftY = 1
						}
						// An odd luma height leaves a short last chroma band.
						height := (LumaBlockSize >> shiftY) - row*3
						lumaRows := ((height - 1) << shiftY) + 1
						chromaStride := chromaWidth + 5
						lumaStride := lumaWidth + 7
						src := randomSamplePlane(rng, height*chromaStride, bd)
						lumaLen := (lumaRows-1)*lumaStride + lumaWidth
						luma := randomSamplePlane(rng, lumaLen, bd)
						padded := make([]uint16, lumaLen+1)
						copy(padded, luma)
						for y := range lumaRows {
							edge := y*lumaStride + lumaWidth
							padded[edge] = padded[edge-1]
							if edge < lumaLen {
								luma[edge] = padded[edge-1] ^ uint16(1<<bd-1)
							}
						}
						grain := randomGrainTemplate(rng, ChromaGrainSamples, bd)
						scaling := randomScalingLUT(rng)
						params := ChromaRowParams{
							Seed:                  uint16(0x0DD0 + row),
							Width:                 chromaWidth,
							Height:                height,
							Stride:                chromaStride,
							LumaStride:            lumaStride,
							Row:                   row,
							BitDepth:              bd,
							ScalingShift:          9,
							SubsamplingX:          ss.x,
							SubsamplingY:          ss.y,
							Overlap:               true,
							ChromaScalingFromLuma: csfl,
							ChromaMult:            140,
							ChromaLumaMult:        118,
							ChromaOffset:          260,
						}
						odd := params
						odd.OddLumaWidth = true
						check := func(path string, got, want, unfixed []uint16) {
							t.Helper()
							key := fmt.Sprintf("%s bd=%d ss=%v csfl=%v", path, bd, ss, csfl)
							if _, ok := detected[key]; !ok {
								detected[key] = false
							}
							for i := range want {
								if got[i] != want[i] {
									t.Fatalf("%s bd=%d ss=%v w=%d csfl=%v row=%d i=%d: got %d, padded reference %d",
										path, bd, ss, chromaWidth, csfl, row, i, got[i], want[i])
								}
							}
							for i := range want {
								if unfixed[i] != want[i] {
									detected[key] = true
									break
								}
							}
						}
						run := func(apply func(dst, luma []uint16, p ChromaRowParams) error) (got, want, unfixed []uint16) {
							t.Helper()
							got = append([]uint16(nil), src...)
							want = append([]uint16(nil), src...)
							unfixed = append([]uint16(nil), src...)
							if err := apply(got, luma, odd); err != nil {
								t.Fatalf("odd-width apply: %v", err)
							}
							if err := apply(want, padded, params); err != nil {
								t.Fatalf("padded apply: %v", err)
							}
							// The pre-fix behaviour: the unpadded plane read as if even.
							unpadded := append(append([]uint16(nil), luma...), padded[lumaLen]^uint16(1<<bd-1))
							if err := apply(unfixed, unpadded, params); err != nil {
								t.Fatalf("unfixed apply: %v", err)
							}
							return got, want, unfixed
						}

						got, want, unfixed := run(func(dst, l []uint16, p ChromaRowParams) error {
							return ApplyChromaRow(dst, dst, l, grain, scaling, p)
						})
						check("ApplyChromaRow", got, want, unfixed)
						if bd != 8 {
							var table [ScalingTableSize]uint8
							if err := BuildScalingTable(&table, scaling, bd); err != nil {
								t.Fatal(err)
							}
							got, want, unfixed = run(func(dst, l []uint16, p ChromaRowParams) error {
								return ApplyChromaRowTable(dst, dst, l, grain, scaling, &table, p)
							})
							check("ApplyChromaRowTable", got, want, unfixed)
							continue
						}
						got, want, unfixed = run(func(dst, l []uint16, p ChromaRowParams) error {
							d8, l8 := u16ToU8(dst), u16ToU8(l)
							if err := ApplyChromaRowU8(d8, d8, l8, grain, scaling, p); err != nil {
								return err
							}
							for i, v := range d8 {
								dst[i] = uint16(v)
							}
							return nil
						})
						check("ApplyChromaRowU8", got, want, unfixed)
					}
				}
			}
		}
	}
	for key, ok := range detected {
		if !ok {
			t.Errorf("%s: no case reads past the odd luma edge; it does not exercise the fix", key)
		}
	}
}

func u16ToU8(s []uint16) []uint8 {
	out := make([]uint8, len(s))
	for i, v := range s {
		out[i] = uint8(v)
	}
	return out
}

// TestApplyChromaRowOddLumaWidthRejects444 keeps OddLumaWidth to subsampled
// planes: without SubsamplingX each chroma column has its own luma sample.
func TestApplyChromaRowOddLumaWidthRejects444(t *testing.T) {
	params := ChromaRowParams{
		Width: 8, Height: 1, Stride: 8, LumaStride: 8,
		BitDepth: 8, ScalingShift: 9, OddLumaWidth: true,
	}
	buf := make([]uint8, 8)
	if err := ApplyChromaRowU8(buf, buf, buf, make([]int16, ChromaGrainSamples), make([]uint8, ScalingLUTSize), params); err != ErrInvalidParams {
		t.Fatalf("ApplyChromaRowU8 = %v, want ErrInvalidParams", err)
	}
}

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

// fuzzSegmentInputs builds a deterministic but adversarial set of inputs for
// the apply kernel: full-range source samples, the full int16 grain range,
// and 0..255 scale values, so the differential test exercises the rounding,
// the arithmetic shift sign, and both clamp bounds.
func fuzzSegmentInputs(rng *rand.Rand, n int, bitDepth uint8) (dst, src, scale []uint16, grain []int16) {
	maxSample := (1 << bitDepth) - 1
	dst = make([]uint16, n)
	src = make([]uint16, n)
	scale = make([]uint16, n)
	grain = make([]int16, n)
	grainLimit := 1 << (bitDepth - 1)
	for i := 0; i < n; i++ {
		src[i] = uint16(rng.Intn(maxSample + 1))
		scale[i] = uint16(rng.Intn(256))
		grain[i] = int16(rng.Intn(2*grainLimit) - grainLimit)
	}
	// Sprinkle in saturating extremes.
	if n > 0 {
		src[0], scale[0], grain[0] = uint16(maxSample), 255, int16(grainLimit-1)
		src[n-1], scale[n-1], grain[n-1] = 0, 255, int16(-grainLimit)
	}
	return dst, src, scale, grain
}

func BenchmarkApplyGrainSegmentPureGo(b *testing.B) {
	const n = 64
	dst, src, scale, grain := fuzzSegmentInputs(rand.New(rand.NewSource(7)), n, 8)
	b.ReportAllocs()
	for b.Loop() {
		applyGrainSegmentPureGo(dst, src, scale, grain, 8, 0, 255)
	}
}

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

func TestApplyLumaRowNoOverlap(t *testing.T) {
	const (
		width  = 4
		height = 2
		stride = 8
	)
	dst, src := testLumaRowBuffers(width, height, stride, 100)
	grain := make([]int16, LumaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestLumaRowGrain(t, grain, 0xd9, 0, 0, 0, 0, 64)

	params := LumaRowParams{
		Seed:         0,
		Width:        width,
		Height:       height,
		Stride:       stride,
		BitDepth:     8,
		ScalingShift: 8,
	}
	if err := ApplyLumaRow(dst, src, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[0]; got != 116 {
		t.Fatalf("first sample=%d want 116", got)
	}
	if got := dst[1]; got != 100 {
		t.Fatalf("neighbor sample=%d want 100", got)
	}
	if got := dst[stride]; got != 100 {
		t.Fatalf("next-row sample=%d want 100", got)
	}
}

func TestApplyLumaRowLeftOverlap(t *testing.T) {
	const (
		width  = 34
		height = 1
		stride = 40
	)
	dst, src := testLumaRowBuffers(width, height, stride, 100)
	grain := make([]int16, LumaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestLumaRowGrain(t, grain, 0xec, 0, 0, 0, 0, 20)
	setTestLumaRowGrain(t, grain, 0xd9, 1, 0, 0, 0, 100)

	params := LumaRowParams{
		Seed:         0,
		Width:        width,
		Height:       height,
		Stride:       stride,
		BitDepth:     8,
		ScalingShift: 8,
		Overlap:      true,
	}
	if err := ApplyLumaRow(dst, src, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[32]; got != 124 {
		t.Fatalf("left-overlap sample=%d want 124", got)
	}
	if got := dst[33]; got != 100 {
		t.Fatalf("second overlap sample=%d want 100", got)
	}
}

func TestApplyLumaRowTopOverlap(t *testing.T) {
	const (
		width  = 1
		height = 2
		stride = 8
	)
	dst, src := testLumaRowBuffers(width, height, stride, 100)
	grain := make([]int16, LumaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestLumaRowGrain(t, grain, 0x6b, 0, 0, 0, 0, 20)
	setTestLumaRowGrain(t, grain, 0xd9, 0, 1, 0, 0, 100)

	params := LumaRowParams{
		Seed:         0,
		Width:        width,
		Height:       height,
		Stride:       stride,
		Row:          1,
		BitDepth:     8,
		ScalingShift: 8,
		Overlap:      true,
	}
	if err := ApplyLumaRow(dst, src, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[0]; got != 124 {
		t.Fatalf("top-overlap sample=%d want 124", got)
	}
	if got := dst[stride]; got != 100 {
		t.Fatalf("second row sample=%d want 100", got)
	}
}

func TestApplyLumaRowCornerOverlap(t *testing.T) {
	const (
		width  = 34
		height = 2
		stride = 40
	)
	dst, src := testLumaRowBuffers(width, height, stride, 100)
	grain := make([]int16, LumaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestLumaRowGrain(t, grain, 0xb5, 0, 0, 0, 0, 20)
	setTestLumaRowGrain(t, grain, 0x6b, 1, 0, 0, 0, 100)
	setTestLumaRowGrain(t, grain, 0xec, 0, 1, 0, 0, 20)
	setTestLumaRowGrain(t, grain, 0xd9, 1, 1, 0, 0, 60)

	params := LumaRowParams{
		Seed:         0,
		Width:        width,
		Height:       height,
		Stride:       stride,
		Row:          1,
		BitDepth:     8,
		ScalingShift: 8,
		Overlap:      true,
	}
	if err := ApplyLumaRow(dst, src, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[32]; got != 126 {
		t.Fatalf("corner-overlap sample=%d want 126", got)
	}
}

func TestApplyChromaRowNoOverlap(t *testing.T) {
	const (
		width      = 4
		height     = 2
		stride     = 8
		lumaStride = 8
	)
	dst, src := testChromaRowBuffers(width, height, stride, 100)
	luma := testChromaRowLuma(width, height, lumaStride, false, false, 80)
	grain := make([]int16, ChromaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestChromaRowGrain(t, grain, 0xd9, false, false, 0, 0, 0, 0, 64)

	params := ChromaRowParams{
		Seed:                  0,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		LumaStride:            lumaStride,
		BitDepth:              8,
		ScalingShift:          8,
		ChromaScalingFromLuma: true,
	}
	if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[0]; got != 116 {
		t.Fatalf("first sample=%d want 116", got)
	}
	if got := dst[1]; got != 100 {
		t.Fatalf("neighbor sample=%d want 100", got)
	}
}

func TestApplyChromaRowUsesChromaScalingFormula(t *testing.T) {
	const (
		width      = 1
		height     = 1
		stride     = 4
		lumaStride = 4
	)
	dst, src := testChromaRowBuffers(width, height, stride, 100)
	luma := testChromaRowLuma(width, height, lumaStride, false, false, 80)
	grain := make([]int16, ChromaGrainSamples)
	var lut [ScalingLUTSize]uint8
	lut[100] = 128
	setTestChromaRowGrain(t, grain, 0xd9, false, false, 0, 0, 0, 0, 32)

	// libaom add_noise_to_block applies the +128 bias on the mults and the
	// +256 bias on the offset (cb_mult-128, cb_luma_mult-128, cb_offset-256).
	// With raw cb_luma_mult=208 (=>+80), cb_mult=128 (=>0), cb_offset=256 (=>0)
	// and average_luma=80 the scaling index resolves to (80*80)>>6 = 100.
	params := ChromaRowParams{
		Seed:           0,
		Width:          width,
		Height:         height,
		Stride:         stride,
		LumaStride:     lumaStride,
		BitDepth:       8,
		ScalingShift:   8,
		ChromaMult:     128,
		ChromaLumaMult: 208,
		ChromaOffset:   256,
	}
	if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[0]; got != 116 {
		t.Fatalf("chroma-scaled sample=%d want 116", got)
	}
}

func TestApplyChromaRowLeftOverlap420(t *testing.T) {
	const (
		width      = 17
		height     = 1
		stride     = 20
		lumaStride = 40
	)
	dst, src := testChromaRowBuffers(width, height, stride, 100)
	luma := testChromaRowLuma(width, height, lumaStride, true, true, 80)
	grain := make([]int16, ChromaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestChromaRowGrain(t, grain, 0xec, true, true, 0, 0, 0, 0, 20)
	setTestChromaRowGrain(t, grain, 0xd9, true, true, 1, 0, 0, 0, 100)

	params := ChromaRowParams{
		Seed:                  0,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		LumaStride:            lumaStride,
		BitDepth:              8,
		ScalingShift:          8,
		SubsamplingX:          true,
		SubsamplingY:          true,
		Overlap:               true,
		ChromaScalingFromLuma: true,
	}
	if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[16]; got != 122 {
		t.Fatalf("left-overlap sample=%d want 122", got)
	}
}

func TestApplyChromaRowTopOverlap420(t *testing.T) {
	const (
		width      = 1
		height     = 1
		stride     = 8
		lumaStride = 8
	)
	dst, src := testChromaRowBuffers(width, height, stride, 100)
	luma := testChromaRowLuma(width, height, lumaStride, true, true, 80)
	grain := make([]int16, ChromaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestChromaRowGrain(t, grain, 0x6b, true, true, 0, 0, 0, 0, 20)
	setTestChromaRowGrain(t, grain, 0xd9, true, true, 0, 1, 0, 0, 100)

	params := ChromaRowParams{
		Seed:                  0,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		LumaStride:            lumaStride,
		Row:                   1,
		BitDepth:              8,
		ScalingShift:          8,
		SubsamplingX:          true,
		SubsamplingY:          true,
		Overlap:               true,
		ChromaScalingFromLuma: true,
	}
	if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[0]; got != 122 {
		t.Fatalf("top-overlap sample=%d want 122", got)
	}
}

func TestApplyChromaRowCornerOverlap420(t *testing.T) {
	const (
		width      = 17
		height     = 1
		stride     = 20
		lumaStride = 40
	)
	dst, src := testChromaRowBuffers(width, height, stride, 100)
	luma := testChromaRowLuma(width, height, lumaStride, true, true, 80)
	grain := make([]int16, ChromaGrainSamples)
	lut := testLumaRowScaling(64)
	setTestChromaRowGrain(t, grain, 0xb5, true, true, 0, 0, 0, 0, 20)
	setTestChromaRowGrain(t, grain, 0x6b, true, true, 1, 0, 0, 0, 100)
	setTestChromaRowGrain(t, grain, 0xec, true, true, 0, 1, 0, 0, 20)
	setTestChromaRowGrain(t, grain, 0xd9, true, true, 1, 1, 0, 0, 60)

	params := ChromaRowParams{
		Seed:                  0,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		LumaStride:            lumaStride,
		Row:                   1,
		BitDepth:              8,
		ScalingShift:          8,
		SubsamplingX:          true,
		SubsamplingY:          true,
		Overlap:               true,
		ChromaScalingFromLuma: true,
	}
	if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[16]; got != 125 {
		t.Fatalf("corner-overlap sample=%d want 125", got)
	}
}

func TestApplyChromaRowClipsRestrictedRange(t *testing.T) {
	const (
		width      = 1
		height     = 1
		stride     = 4
		lumaStride = 4
	)
	dst, src := testChromaRowBuffers(width, height, stride, 250)
	luma := testChromaRowLuma(width, height, lumaStride, false, false, 80)
	grain := make([]int16, ChromaGrainSamples)
	lut := testLumaRowScaling(255)
	setTestChromaRowGrain(t, grain, 0xd9, false, false, 0, 0, 0, 0, 128)

	params := ChromaRowParams{
		Seed:                  0,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		LumaStride:            lumaStride,
		BitDepth:              8,
		ScalingShift:          8,
		ClipToRestrictedRange: true,
		MatrixIdentity:        true,
		ChromaScalingFromLuma: true,
	}
	if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
		t.Fatal(err)
	}
	if got := dst[0]; got != ChromaIdentityMax {
		t.Fatalf("restricted sample=%d want %d", got, ChromaIdentityMax)
	}
}

func TestApplyLumaRowRejectsInvalidInputs(t *testing.T) {
	dst, src := testLumaRowBuffers(4, 2, 8, 100)
	grain := make([]int16, LumaGrainSamples)
	lut := testLumaRowScaling(64)
	params := LumaRowParams{Width: 4, Height: 2, Stride: 8, BitDepth: 8, ScalingShift: 8}
	tests := []struct {
		name   string
		dst    []uint16
		src    []uint16
		grain  []int16
		lut    []uint8
		params LumaRowParams
	}{
		{name: "short-dst", dst: dst[:8], src: src, grain: grain, lut: lut[:], params: params},
		{name: "short-src", dst: dst, src: src[:8], grain: grain, lut: lut[:], params: params},
		{name: "short-grain", dst: dst, src: src, grain: grain[:LumaGrainSamples-1], lut: lut[:], params: params},
		{name: "short-lut", dst: dst, src: src, grain: grain, lut: lut[:ScalingLUTSize-1], params: params},
		{name: "width", dst: dst, src: src, grain: grain, lut: lut[:], params: withLumaRowWidth(params, 0)},
		{name: "height", dst: dst, src: src, grain: grain, lut: lut[:], params: withLumaRowHeight(params, LumaBlockSize+1)},
		{name: "stride", dst: dst, src: src, grain: grain, lut: lut[:], params: withLumaRowStride(params, 3)},
		{name: "row", dst: dst, src: src, grain: grain, lut: lut[:], params: withLumaRowIndex(params, -1)},
		{name: "bit-depth", dst: dst, src: src, grain: grain, lut: lut[:], params: withLumaRowBitDepth(params, 9)},
		{name: "scaling-shift", dst: dst, src: src, grain: grain, lut: lut[:], params: withLumaRowScalingShift(params, 7)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ApplyLumaRow(tt.dst, tt.src, tt.grain, tt.lut, tt.params); !errors.Is(err, ErrInvalidParams) {
				t.Fatalf("ApplyLumaRow err=%v want %v", err, ErrInvalidParams)
			}
		})
	}
}

func TestApplyChromaRowRejectsInvalidInputs(t *testing.T) {
	dst, src := testChromaRowBuffers(4, 2, 8, 100)
	luma := testChromaRowLuma(4, 2, 8, false, false, 80)
	grain := make([]int16, ChromaGrainSamples)
	lut := testLumaRowScaling(64)
	params := ChromaRowParams{Width: 4, Height: 2, Stride: 8, LumaStride: 8, BitDepth: 8, ScalingShift: 8}
	tests := []struct {
		name   string
		dst    []uint16
		src    []uint16
		luma   []uint16
		grain  []int16
		lut    []uint8
		params ChromaRowParams
	}{
		{name: "short-dst", dst: dst[:8], src: src, luma: luma, grain: grain, lut: lut[:], params: params},
		{name: "short-src", dst: dst, src: src[:8], luma: luma, grain: grain, lut: lut[:], params: params},
		{name: "short-luma", dst: dst, src: src, luma: luma[:8], grain: grain, lut: lut[:], params: params},
		{name: "short-grain", dst: dst, src: src, luma: luma, grain: grain[:ChromaGrainSamples-1], lut: lut[:], params: params},
		{name: "short-lut", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:ScalingLUTSize-1], params: params},
		{name: "width", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:], params: withChromaRowWidth(params, 0)},
		{name: "height", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:], params: withChromaRowHeight(params, LumaBlockSize+1)},
		{name: "stride", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:], params: withChromaRowStride(params, 3)},
		{name: "luma-stride", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:], params: withChromaRowLumaStride(params, 3)},
		{name: "row", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:], params: withChromaRowIndex(params, -1)},
		{name: "bit-depth", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:], params: withChromaRowBitDepth(params, 9)},
		{name: "scaling-shift", dst: dst, src: src, luma: luma, grain: grain, lut: lut[:], params: withChromaRowScalingShift(params, 7)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ApplyChromaRow(tt.dst, tt.src, tt.luma, tt.grain, tt.lut, tt.params); !errors.Is(err, ErrInvalidParams) {
				t.Fatalf("ApplyChromaRow err=%v want %v", err, ErrInvalidParams)
			}
		})
	}
}

// TestApplyLumaRowRejectsOverflowStride confirms that an attacker-controlled
// Stride combined with a non-trivial Height cannot wrap the slice-size compare
// via integer overflow.
func TestApplyLumaRowRejectsOverflowStride(t *testing.T) {
	dst, src := testLumaRowBuffers(4, 1, 8, 100)
	grain := make([]int16, LumaGrainSamples)
	lut := testLumaRowScaling(64)
	const maxInt = int(^uint(0) >> 1)
	params := LumaRowParams{
		Width:        4,
		Height:       8,
		Stride:       maxInt/4 + 1,
		BitDepth:     8,
		ScalingShift: 8,
	}
	if err := ApplyLumaRow(dst, src, grain, lut[:], params); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("ApplyLumaRow err=%v want %v", err, ErrInvalidParams)
	}
}

// TestApplyChromaRowRejectsOverflowStride mirrors the luma overflow guard for
// the chroma path so an attacker-controlled Stride/LumaStride pair cannot wrap
// the slice-size compare.
func TestApplyChromaRowRejectsOverflowStride(t *testing.T) {
	dst, src := testChromaRowBuffers(4, 1, 8, 100)
	luma := testChromaRowLuma(4, 1, 8, false, false, 80)
	grain := make([]int16, ChromaGrainSamples)
	lut := testLumaRowScaling(64)
	const maxInt = int(^uint(0) >> 1)
	for _, name := range []string{"stride", "luma-stride"} {
		params := ChromaRowParams{
			Width:        4,
			Height:       4,
			Stride:       8,
			LumaStride:   8,
			BitDepth:     8,
			ScalingShift: 8,
		}
		switch name {
		case "stride":
			params.Stride = maxInt/4 + 1
		case "luma-stride":
			params.LumaStride = maxInt/4 + 1
		}
		t.Run(name, func(t *testing.T) {
			if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); !errors.Is(err, ErrInvalidParams) {
				t.Fatalf("ApplyChromaRow err=%v want %v", err, ErrInvalidParams)
			}
		})
	}
}

func TestApplyLumaRowAllocs(t *testing.T) {
	const (
		width  = 64
		height = 32
		stride = 64
	)
	dst, src := testLumaRowBuffers(width, height, stride, 100)
	grain := testPatternLumaRowGrain()
	lut := testLumaRowScaling(64)
	params := LumaRowParams{
		Seed:                  0x1234,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		Row:                   1,
		BitDepth:              8,
		ScalingShift:          8,
		Overlap:               true,
		ClipToRestrictedRange: true,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ApplyLumaRow(dst, src, grain, lut[:], params); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyLumaRow allocated: %f", allocs)
	}
}

func TestApplyChromaRowAllocs(t *testing.T) {
	const (
		width      = 32
		height     = 16
		stride     = 32
		lumaStride = 64
	)
	dst, src := testChromaRowBuffers(width, height, stride, 100)
	luma := testChromaRowLuma(width, height, lumaStride, true, true, 80)
	grain := testPatternChromaRowGrain()
	lut := testLumaRowScaling(64)
	params := ChromaRowParams{
		Seed:                  0x1234,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		LumaStride:            lumaStride,
		Row:                   1,
		BitDepth:              8,
		ScalingShift:          8,
		SubsamplingX:          true,
		SubsamplingY:          true,
		Overlap:               true,
		ClipToRestrictedRange: true,
		ChromaScalingFromLuma: true,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyChromaRow allocated: %f", allocs)
	}
}

func BenchmarkApplyLumaRow(b *testing.B) {
	const (
		width  = 64
		height = 32
		stride = 64
	)
	dst, src := testLumaRowBuffers(width, height, stride, 100)
	grain := testPatternLumaRowGrain()
	lut := testLumaRowScaling(64)
	params := LumaRowParams{
		Seed:         0x1234,
		Width:        width,
		Height:       height,
		Stride:       stride,
		Row:          1,
		BitDepth:     8,
		ScalingShift: 8,
		Overlap:      true,
	}
	b.ReportAllocs()
	for b.Loop() {
		if err := ApplyLumaRow(dst, src, grain, lut[:], params); err != nil {
			b.Fatal(err)
		}
	}
	if dst[0] == 0 {
		b.Fatal("unexpected zero sample")
	}
}

func BenchmarkApplyChromaRow(b *testing.B) {
	const (
		width      = 32
		height     = 16
		stride     = 32
		lumaStride = 64
	)
	dst, src := testChromaRowBuffers(width, height, stride, 100)
	luma := testChromaRowLuma(width, height, lumaStride, true, true, 80)
	grain := testPatternChromaRowGrain()
	lut := testLumaRowScaling(64)
	params := ChromaRowParams{
		Seed:                  0x1234,
		Width:                 width,
		Height:                height,
		Stride:                stride,
		LumaStride:            lumaStride,
		Row:                   1,
		BitDepth:              8,
		ScalingShift:          8,
		SubsamplingX:          true,
		SubsamplingY:          true,
		Overlap:               true,
		ChromaScalingFromLuma: true,
	}
	b.ReportAllocs()
	for b.Loop() {
		if err := ApplyChromaRow(dst, src, luma, grain, lut[:], params); err != nil {
			b.Fatal(err)
		}
	}
	if dst[0] == 0 {
		b.Fatal("unexpected zero sample")
	}
}

func testLumaRowBuffers(width int, height int, stride int, value uint16) ([]uint16, []uint16) {
	dst := make([]uint16, stride*height)
	src := make([]uint16, stride*height)
	for y := range height {
		for x := range width {
			src[y*stride+x] = value
		}
	}
	return dst, src
}

func testChromaRowBuffers(width int, height int, stride int, value uint16) ([]uint16, []uint16) {
	dst := make([]uint16, stride*height)
	src := make([]uint16, stride*height)
	for y := range height {
		for x := range width {
			src[y*stride+x] = value
		}
	}
	return dst, src
}

func testChromaRowLuma(width int, height int, stride int, subsamplingX bool, subsamplingY bool, value uint16) []uint16 {
	shiftX := chromaSubsamplingShift(subsamplingX)
	shiftY := chromaSubsamplingShift(subsamplingY)
	rows := ((height - 1) << shiftY) + 1
	samples := make([]uint16, stride*rows)
	for y := range rows {
		for x := 0; x < ((width-1)<<shiftX)+1+shiftX; x++ {
			samples[y*stride+x] = value
		}
	}
	return samples
}

func testLumaRowScaling(value uint8) [ScalingLUTSize]uint8 {
	var lut [ScalingLUTSize]uint8
	for i := range lut {
		lut[i] = value
	}
	return lut
}

func testPatternLumaRowGrain() []int16 {
	grain := make([]int16, LumaGrainSamples)
	for y := range LumaGrainHeight {
		for x := range LumaGrainWidth {
			grain[y*LumaGrainWidth+x] = int16((x+y)%64 - 32)
		}
	}
	return grain
}

func testPatternChromaRowGrain() []int16 {
	grain := make([]int16, ChromaGrainSamples)
	for y := range ChromaGrainHeight {
		for x := range ChromaGrainWidth {
			grain[y*ChromaGrainWidth+x] = int16((x+y)%64 - 32)
		}
	}
	return grain
}

func setTestLumaRowGrain(t *testing.T, grain []int16, offset uint8, blockCol int, blockRow int, x int, y int, value int16) {
	t.Helper()
	col := lumaGrainOffset(int(offset>>4)) + x + LumaBlockSize*blockCol
	row := lumaGrainOffset(int(offset&0x0f)) + y + LumaBlockSize*blockRow
	if col < 0 || col >= LumaGrainWidth || row < 0 || row >= LumaGrainHeight {
		t.Fatalf("grain coordinate out of range: col=%d row=%d", col, row)
	}
	grain[row*LumaGrainWidth+col] = value
}

func setTestChromaRowGrain(t *testing.T, grain []int16, offset uint8, subsamplingX bool, subsamplingY bool, blockCol int, blockRow int, x int, y int, value int16) {
	t.Helper()
	shiftX := chromaSubsamplingShift(subsamplingX)
	shiftY := chromaSubsamplingShift(subsamplingY)
	col := chromaGrainOffset(int(offset>>4), shiftX) + x + (LumaBlockSize>>shiftX)*blockCol
	row := chromaGrainOffset(int(offset&0x0f), shiftY) + y + (LumaBlockSize>>shiftY)*blockRow
	width, height := chromaGrainDimensions(subsamplingX, subsamplingY)
	if col < 0 || col >= width || row < 0 || row >= height {
		t.Fatalf("grain coordinate out of range: col=%d row=%d", col, row)
	}
	grain[row*ChromaGrainWidth+col] = value
}

func withLumaRowWidth(params LumaRowParams, width int) LumaRowParams {
	params.Width = width
	return params
}

func withLumaRowHeight(params LumaRowParams, height int) LumaRowParams {
	params.Height = height
	return params
}

func withLumaRowStride(params LumaRowParams, stride int) LumaRowParams {
	params.Stride = stride
	return params
}

func withLumaRowIndex(params LumaRowParams, row int) LumaRowParams {
	params.Row = row
	return params
}

func withLumaRowBitDepth(params LumaRowParams, bitDepth uint8) LumaRowParams {
	params.BitDepth = bitDepth
	return params
}

func withLumaRowScalingShift(params LumaRowParams, shift uint8) LumaRowParams {
	params.ScalingShift = shift
	return params
}

func withChromaRowWidth(params ChromaRowParams, width int) ChromaRowParams {
	params.Width = width
	return params
}

func withChromaRowHeight(params ChromaRowParams, height int) ChromaRowParams {
	params.Height = height
	return params
}

func withChromaRowStride(params ChromaRowParams, stride int) ChromaRowParams {
	params.Stride = stride
	return params
}

func withChromaRowLumaStride(params ChromaRowParams, stride int) ChromaRowParams {
	params.LumaStride = stride
	return params
}

func withChromaRowIndex(params ChromaRowParams, row int) ChromaRowParams {
	params.Row = row
	return params
}

func withChromaRowBitDepth(params ChromaRowParams, bitDepth uint8) ChromaRowParams {
	params.BitDepth = bitDepth
	return params
}

func withChromaRowScalingShift(params ChromaRowParams, shift uint8) ChromaRowParams {
	params.ScalingShift = shift
	return params
}

// FuzzApplyLumaRow exercises ApplyLumaRow with random widths, heights,
// strides, row indices, bit depths, scaling shifts, and overlap flags. The
// function must either return ErrInvalidParams without panicking or complete
// successfully without writing outside the configured row region.
func FuzzApplyLumaRow(f *testing.F) {
	f.Add(uint8(0), uint8(4), uint8(2), uint8(0), uint8(8), uint8(8), int16(0))
	f.Add(uint8(0xff), uint8(34), uint8(LumaBlockSize), uint8(3), uint8(10), uint8(11), int16(64))
	f.Add(uint8(0x55), uint8(LumaBlockSize+2), uint8(LumaBlockSize), uint8(7), uint8(8), uint8(9), int16(-32))
	f.Add(uint8(0xaa), uint8(1), uint8(1), uint8(0), uint8(12), uint8(10), int16(127))

	f.Fuzz(func(t *testing.T, flags uint8, rawWidth uint8, rawHeight uint8, rawRow uint8, rawBitDepth uint8, rawShift uint8, grainValue int16) {
		width := int(rawWidth)
		height := int(rawHeight)
		stride := width + int(flags&7)
		if stride <= 0 {
			stride = 1
		}
		if width > 0 && stride < width {
			stride = width
		}

		bitDepth := uint8(8)
		switch rawBitDepth % 3 {
		case 1:
			bitDepth = 10
		case 2:
			bitDepth = 12
		}
		shift := uint8(8) + rawShift%4
		row := int(rawRow)
		if flags&0x80 != 0 {
			row = -1
		}

		dstLen := stride * height
		if dstLen < 0 {
			return
		}
		const maxSamples = 1 << 16
		if dstLen > maxSamples {
			return
		}
		dst := make([]uint16, dstLen)
		src := make([]uint16, dstLen)
		baseSrc := uint16(100)
		if bitDepth == 10 {
			baseSrc = 400
		} else if bitDepth == 12 {
			baseSrc = 1024
		}
		for y := range height {
			for x := range width {
				src[y*stride+x] = baseSrc
			}
		}
		grain := make([]int16, LumaGrainSamples)
		for i := range grain {
			grain[i] = grainValue
		}
		lut := testLumaRowScaling(64)

		params := LumaRowParams{
			Seed:                  uint16(flags) << 8,
			Width:                 width,
			Height:                height,
			Stride:                stride,
			Row:                   row,
			BitDepth:              bitDepth,
			ScalingShift:          shift,
			Overlap:               flags&0x10 != 0,
			ClipToRestrictedRange: flags&0x20 != 0,
		}
		err := ApplyLumaRow(dst, src, grain, lut[:], params)
		if err != nil {
			return
		}
		for y := range height {
			for x := width; x < stride; x++ {
				if dst[y*stride+x] != 0 {
					t.Fatalf("dst(%d,%d)=%d expected outside-region zero", x, y, dst[y*stride+x])
				}
			}
		}
		maxSample := uint16((1 << bitDepth) - 1)
		for y := range height {
			for x := range width {
				if dst[y*stride+x] > maxSample {
					t.Fatalf("dst(%d,%d)=%d exceeds bit-depth max %d", x, y, dst[y*stride+x], maxSample)
				}
			}
		}
	})
}

// FuzzApplyChromaRow exercises ApplyChromaRow with random subsampling, scaling
// modes, and row geometries. As with the luma fuzz, the function must either
// reject the input cleanly or complete without panicking or scribbling past
// the destination row band.
func FuzzApplyChromaRow(f *testing.F) {
	f.Add(uint8(0), uint8(4), uint8(2), uint8(0), uint8(8), uint8(8), int16(0))
	f.Add(uint8(0x3f), uint8(17), uint8(LumaBlockSize), uint8(1), uint8(8), uint8(11), int16(-64))
	f.Add(uint8(0x7f), uint8(16), uint8(LumaBlockSize/2), uint8(2), uint8(10), uint8(8), int16(96))
	f.Add(uint8(0xff), uint8(2), uint8(2), uint8(0), uint8(12), uint8(9), int16(-128))

	f.Fuzz(func(t *testing.T, flags uint8, rawWidth uint8, rawHeight uint8, rawRow uint8, rawBitDepth uint8, rawShift uint8, grainValue int16) {
		subsamplingX := flags&1 != 0
		subsamplingY := flags&2 != 0
		shiftX := chromaSubsamplingShift(subsamplingX)
		shiftY := chromaSubsamplingShift(subsamplingY)

		width := int(rawWidth)
		height := int(rawHeight)
		stride := width + int((flags>>2)&7)
		if width > 0 && stride < width {
			stride = width
		}
		if stride <= 0 {
			stride = 1
		}
		lumaStride := max(((width-1)<<shiftX)+1+shiftX+int((flags>>5)&3), 1)

		bitDepth := uint8(8)
		switch rawBitDepth % 3 {
		case 1:
			bitDepth = 10
		case 2:
			bitDepth = 12
		}
		shift := uint8(8) + rawShift%4
		row := int(rawRow)

		dstLen := stride * height
		lumaRows := max(((height-1)<<shiftY)+1, 0)
		lumaLen := lumaStride * lumaRows
		if dstLen < 0 || lumaLen < 0 {
			return
		}
		const maxSamples = 1 << 16
		if dstLen > maxSamples || lumaLen > maxSamples {
			return
		}

		dst := make([]uint16, dstLen)
		src := make([]uint16, dstLen)
		luma := make([]uint16, lumaLen)
		baseSrc := uint16(100)
		baseLuma := uint16(80)
		if bitDepth == 10 {
			baseSrc = 400
			baseLuma = 320
		} else if bitDepth == 12 {
			baseSrc = 1024
			baseLuma = 800
		}
		for y := range height {
			for x := range width {
				src[y*stride+x] = baseSrc
			}
		}
		for y := range lumaRows {
			for x := range lumaStride {
				luma[y*lumaStride+x] = baseLuma
			}
		}
		grain := make([]int16, ChromaGrainSamples)
		for i := range grain {
			grain[i] = grainValue
		}
		lut := testLumaRowScaling(64)

		params := ChromaRowParams{
			Seed:                  uint16(flags) << 4,
			Width:                 width,
			Height:                height,
			Stride:                stride,
			LumaStride:            lumaStride,
			Row:                   row,
			BitDepth:              bitDepth,
			ScalingShift:          shift,
			SubsamplingX:          subsamplingX,
			SubsamplingY:          subsamplingY,
			Overlap:               flags&0x04 != 0,
			ClipToRestrictedRange: flags&0x08 != 0,
			MatrixIdentity:        flags&0x10 != 0,
			ChromaScalingFromLuma: flags&0x80 != 0,
			ChromaMult:            uint8(grainValue & 0x7f),
			ChromaLumaMult:        uint8((grainValue >> 7) & 0x7f),
			ChromaOffset:          uint16(rawBitDepth) << 1,
		}
		err := ApplyChromaRow(dst, src, luma, grain, lut[:], params)
		if err != nil {
			return
		}
		for y := range height {
			for x := width; x < stride; x++ {
				if dst[y*stride+x] != 0 {
					t.Fatalf("dst(%d,%d)=%d expected outside-region zero", x, y, dst[y*stride+x])
				}
			}
		}
		maxSample := uint16((1 << bitDepth) - 1)
		for y := range height {
			for x := range width {
				if dst[y*stride+x] > maxSample {
					t.Fatalf("dst(%d,%d)=%d exceeds bit-depth max %d", x, y, dst[y*stride+x], maxSample)
				}
			}
		}
	})
}

// The reference walks each pixel and overlap region independently of the new
// byte block/segment path; full buffers include stride padding and end canaries.
func TestApplyRowsU8MatchPerSampleReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0x892f))
	grainY := randomGrainTemplate(rng, LumaGrainSamples, 8)
	grainUV := randomGrainTemplate(rng, ChromaGrainSamples, 8)
	scaling := randomScalingLUT(rng)
	for i := range scaling {
		if i&3 == 0 {
			scaling[i] = 255
		}
	}
	for _, width := range []int{1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65} {
		for mode := 0; mode < 16; mode++ {
			height := 32
			if mode&1 != 0 {
				height = 1 + mode
			}
			p := LumaRowParams{Seed: uint16(mode * 3511), Width: width, Height: height, Stride: width + 7, Row: mode % 3,
				BitDepth: 8, ScalingShift: uint8(8 + mode%4), Overlap: mode&2 != 0, ClipToRestrictedRange: mode&4 != 0}
			src16 := randomSamplePlane(rng, height*p.Stride+13, 8)
			src := grainTestBytes(src16)
			before := bytes.Clone(src)
			for _, alias := range []bool{false, true} {
				dst := bytes.Repeat([]byte{0xa5}, len(src))
				want := make([]uint16, len(src))
				for i := range want {
					want[i] = 0xa5
				}
				if alias {
					copy(dst, src)
					copy(want, src16)
				}
				input := src
				if alias {
					input = dst
				}
				applyLumaRowRef(t, want, src16, grainY, scaling, p)
				if err := ApplyLumaRowU8(dst, input, grainY, scaling, p); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(dst, grainTestBytes(want)) {
					t.Fatalf("luma width=%d mode=%d alias=%v", width, mode, alias)
				}
			}
			if !bytes.Equal(src, before) {
				t.Fatal("source modified")
			}

			for _, ss := range [][2]bool{{false, false}, {true, false}, {true, true}} {
				sx, sy := chromaSubsamplingShift(ss[0]), chromaSubsamplingShift(ss[1])
				h := min(height, 32>>sy)
				cp := ChromaRowParams{Seed: p.Seed, Width: width, Height: h, Stride: width + 7, LumaStride: (width << sx) + 9,
					Row: p.Row, BitDepth: 8, ScalingShift: p.ScalingShift, Overlap: p.Overlap, ClipToRestrictedRange: p.ClipToRestrictedRange,
					SubsamplingX: ss[0], SubsamplingY: ss[1], ChromaScalingFromLuma: mode&1 != 0, MatrixIdentity: mode&8 != 0,
					ChromaMult: uint8(mode * 17), ChromaLumaMult: uint8(255 - mode*17), ChromaOffset: uint16(mode * 34)}
				uv16 := randomSamplePlane(rng, h*cp.Stride+13, 8)
				uv := grainTestBytes(uv16)
				lum16 := randomSamplePlane(rng, (h<<sy)*cp.LumaStride, 8)
				lum := grainTestBytes(lum16)
				for _, alias := range []bool{false, true} {
					dst := bytes.Repeat([]byte{0xa5}, len(uv))
					want := make([]uint16, len(uv))
					for i := range want {
						want[i] = 0xa5
					}
					if alias {
						copy(dst, uv)
						copy(want, uv16)
					}
					input := uv
					if alias {
						input = dst
					}
					applyChromaRowRef(t, want, uv16, lum16, grainUV, scaling, cp)
					if err := ApplyChromaRowU8(dst, input, lum, grainUV, scaling, cp); err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(dst, grainTestBytes(want)) {
						t.Fatalf("chroma width=%d mode=%d ss=%v alias=%v", width, mode, ss, alias)
					}
				}
				if !bytes.Equal(uv, grainTestBytes(uv16)) || !bytes.Equal(lum, grainTestBytes(lum16)) {
					t.Fatal("chroma source modified")
				}
			}
		}
	}
}

func grainTestBytes(samples []uint16) []byte {
	out := make([]byte, len(samples))
	for i, v := range samples {
		out[i] = byte(v)
	}
	return out
}

func TestApplyRowsU8RejectInvalidBeforeMutation(t *testing.T) {
	grain := make([]int16, LumaGrainSamples)
	scaling := make([]uint8, ScalingLUTSize)
	p := LumaRowParams{Width: 8, Height: 2, Stride: 8, BitDepth: 8, ScalingShift: 8}
	for _, change := range []func(*LumaRowParams){
		func(p *LumaRowParams) { p.BitDepth = 10 }, func(p *LumaRowParams) { p.ScalingShift = 7 },
		func(p *LumaRowParams) { p.Stride = int(^uint(0) >> 1) }, func(p *LumaRowParams) { p.Height = 33 },
	} {
		bad := p
		change(&bad)
		dst := bytes.Repeat([]byte{0xab}, 16)
		if err := ApplyLumaRowU8(dst, dst, grain, scaling, bad); err != ErrInvalidParams {
			t.Fatalf("params=%+v err=%v", bad, err)
		}
		if !bytes.Equal(dst, bytes.Repeat([]byte{0xab}, 16)) {
			t.Fatal("invalid request mutated destination")
		}
	}
	if err := ApplyLumaRowU8(make([]byte, 15), make([]byte, 16), grain, scaling, p); err != ErrInvalidParams {
		t.Fatal(err)
	}
	cp := ChromaRowParams{Width: 8, Height: 2, Stride: 8, LumaStride: 16, BitDepth: 8, ScalingShift: 8, SubsamplingX: true}
	for _, change := range []func(*ChromaRowParams){
		func(p *ChromaRowParams) { p.BitDepth = 12 }, func(p *ChromaRowParams) { p.ScalingShift = 12 },
		func(p *ChromaRowParams) { p.LumaStride = int(^uint(0) >> 1) }, func(p *ChromaRowParams) { p.Height = 33 },
	} {
		bad := cp
		change(&bad)
		dst := bytes.Repeat([]byte{0xab}, 16)
		if err := ApplyChromaRowU8(dst, dst, make([]byte, 32), grain, scaling, bad); err != ErrInvalidParams {
			t.Fatalf("params=%+v err=%v", bad, err)
		}
		if !bytes.Equal(dst, bytes.Repeat([]byte{0xab}, 16)) {
			t.Fatal("invalid chroma request mutated destination")
		}
	}
}

// TestRound2ViaHalvingAdd checks the identity the arm64 U8 kernels use for
// dav1d's srshl by -scaling_shift, which archsimd lacks:
// round2(p, s) == ((p >> (s-1)) + 1) >> 1 (sshl, then srhadd with zero) for
// every scaling*grain product an 8-bit frame can produce.
func TestRound2ViaHalvingAdd(t *testing.T) {
	for s := 8; s <= 11; s++ {
		for p := -255 * 128; p <= 255*127; p++ {
			if got, want := ((p>>(s-1))+1)>>1, roundPowerOfTwo(p, s); got != want {
				t.Fatalf("p=%d s=%d: got %d want %d", p, s, got, want)
			}
		}
	}
}

// TestChromaIndexSaturatingSum checks that dav1d's int16 sqadd of
// avg*uv_luma_mult + src*uv_mult (used by the arm64 U8 chroma kernels) gives
// the same clipped scaling index as the exact sum.
func TestChromaIndexSaturatingSum(t *testing.T) {
	for _, lm := range []int{-128, -127, -1, 0, 1, 126, 127} {
		for _, m := range []int{-128, -127, -1, 0, 1, 126, 127} {
			for _, off := range []int{-256, -255, -1, 0, 1, 254, 255} {
				for avg := 0; avg < 256; avg++ {
					for src := 0; src < 256; src++ {
						exact := avg*lm + src*m
						sat := min(max(exact, -32768), 32767)
						if got, want := clipInt((sat>>6)+off, 0, 255), clipInt((exact>>6)+off, 0, 255); got != want {
							t.Fatalf("lm=%d m=%d off=%d avg=%d src=%d: got %d want %d", lm, m, off, avg, src, got, want)
						}
					}
				}
			}
		}
	}
}

func TestApplyLumaSample(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 0, Scaling: 64}}); err != nil {
		t.Fatal(err)
	}
	got := applyLumaSample(128, 32, lut[:], 8, 8, false)
	if got != 136 {
		t.Fatalf("sample=%d want 136", got)
	}
	got = applyLumaSample(512, 32, lut[:], 10, 8, false)
	if got != 520 {
		t.Fatalf("hbd sample=%d want 520", got)
	}
}

func TestApplyLumaSampleClips(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 0, Scaling: 255}}); err != nil {
		t.Fatal(err)
	}
	got := applyLumaSample(10, -128, lut[:], 8, 8, true)
	if got != LumaLegalMin {
		t.Fatalf("restricted low sample=%d want %d", got, LumaLegalMin)
	}
	got = applyLumaSample(250, 128, lut[:], 8, 8, true)
	if got != LumaLegalMax {
		t.Fatalf("restricted high sample=%d want %d", got, LumaLegalMax)
	}
}

func TestBlendLumaOverlap(t *testing.T) {
	got := blendLumaOverlap(100, 20, 0, 8)
	if got != 95 {
		t.Fatalf("first overlap=%d want 95", got)
	}
	got = blendLumaOverlap(100, 20, 1, 8)
	if got != 70 {
		t.Fatalf("second overlap=%d want 70", got)
	}
}

func TestBlendLumaOverlapClips(t *testing.T) {
	got := blendLumaOverlap(300, 300, 0, 8)
	if got != 127 {
		t.Fatalf("high overlap=%d want 127", got)
	}
	got = blendLumaOverlap(-300, -300, 0, 8)
	if got != -128 {
		t.Fatalf("low overlap=%d want -128", got)
	}
}

// fuzzScaleLUT returns a 256-entry scaling LUT. Beyond fully random tables it
// injects structured shapes (flat, ramps, saturated) so the interpolation and
// the uint8 truncation of start+round are exercised at their extremes.
func fuzzScaleLUT(rng *rand.Rand, shape int) []uint8 {
	lut := make([]uint8, ScalingLUTSize)
	switch shape {
	case 0:
		for i := range lut {
			lut[i] = uint8(rng.Intn(256))
		}
	case 1: // monotone ramp (typical film-grain shape)
		for i := range lut {
			lut[i] = uint8(i)
		}
	case 2: // descending ramp: forces negative diffs (arithmetic shift sign)
		for i := range lut {
			lut[i] = uint8(255 - i)
		}
	case 3: // flat
		v := uint8(rng.Intn(256))
		for i := range lut {
			lut[i] = v
		}
	default: // adjacent extremes: max |diff| interpolation
		for i := range lut {
			if i&1 == 0 {
				lut[i] = 0
			} else {
				lut[i] = 255
			}
		}
	}
	return lut
}

// fuzzScaleSrc returns a sample row for the given bit depth, saturated at both
// ends and salted with a few out-of-range samples to exercise the clamp.
func fuzzScaleSrc(rng *rand.Rand, n int, bitDepth uint8) []uint16 {
	maxSample := (1 << bitDepth) - 1
	src := make([]uint16, n)
	for i := range src {
		src[i] = uint16(rng.Intn(maxSample + 1))
	}
	if n > 0 {
		src[0] = 0
		src[n-1] = uint16(maxSample)
	}
	// Out-of-range samples (defense-in-depth clamp path).
	for _, i := range []int{n / 3, n / 2, (2 * n) / 3} {
		if i >= 0 && i < n {
			src[i] = uint16(1024 + rng.Intn(64512))
		}
	}
	return src
}

func TestBuildScaleRow10MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5CA1E10))
	lengths := []int{1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100}
	for shape := 0; shape < 5; shape++ {
		for rep := 0; rep < 4; rep++ {
			lut := fuzzScaleLUT(rng, shape)
			for _, n := range lengths {
				src := fuzzScaleSrc(rng, n, 10)
				ref := make([]uint16, n)
				got := make([]uint16, n)
				for i := 0; i < n; i++ {
					ref[i] = uint16(scaleLUT10(lut, int(src[i])))
				}
				buildScaleRow10(got, src, lut)
				for i := 0; i < n; i++ {
					if got[i] != ref[i] {
						t.Fatalf("10-bit shape=%d n=%d i=%d src=%d: row=%d scalar=%d",
							shape, n, i, src[i], got[i], ref[i])
					}
				}
			}
		}
	}
}

// TestBuildScaleRow10ExhaustiveIndex checks every legal 10-bit sample index
// against the scalar, so the x==255 boundary (idx in [1020,1023]) and every
// interpolation weight are covered without relying on random draws.
func TestBuildScaleRow10ExhaustiveIndex(t *testing.T) {
	rng := rand.New(rand.NewSource(0xEECE))
	src := make([]uint16, 1024)
	for i := range src {
		src[i] = uint16(i)
	}
	for shape := 0; shape < 5; shape++ {
		lut := fuzzScaleLUT(rng, shape)
		ref := make([]uint16, len(src))
		got := make([]uint16, len(src))
		for i := range src {
			ref[i] = uint16(scaleLUT10(lut, int(src[i])))
		}
		buildScaleRow10(got, src, lut)
		for i := range src {
			if got[i] != ref[i] {
				t.Fatalf("10-bit exhaustive shape=%d idx=%d: row=%d scalar=%d",
					shape, i, got[i], ref[i])
			}
		}
	}
}

func TestBuildScaleRow10ZeroAlloc(t *testing.T) {
	const n = 64
	rng := rand.New(rand.NewSource(1))
	lut := fuzzScaleLUT(rng, 0)
	src := fuzzScaleSrc(rng, n, 10)
	scale := make([]uint16, n)
	if a := testing.AllocsPerRun(1000, func() {
		buildScaleRow10(scale, src, lut)
	}); a != 0 {
		t.Fatalf("buildScaleRow10 allocated: %f", a)
	}
}

func BenchmarkBuildScaleRow10(b *testing.B) {
	const n = 32
	rng := rand.New(rand.NewSource(7))
	lut := fuzzScaleLUT(rng, 1)
	src := fuzzScaleSrc(rng, n, 10)
	scale := make([]uint16, n)
	b.ReportAllocs()
	for b.Loop() {
		buildScaleRow10(scale, src, lut)
	}
}

func TestChromaGrainParamsHotSize(t *testing.T) {
	if size := unsafe.Sizeof(ChromaGrainParams{}); size > 36 {
		t.Fatalf("ChromaGrainParams size=%d max=36", size)
	}
}

func TestGenerateChromaGrainWhiteNoise420(t *testing.T) {
	grain := make([]int16, ChromaGrainSamples)
	params := ChromaGrainParams{
		Seed:            0x1234,
		Plane:           ChromaPlaneCb,
		BitDepth:        8,
		GrainScaleShift: 0,
		SubsamplingX:    true,
		SubsamplingY:    true,
	}
	if err := GenerateChromaGrain(grain, nil, params); err != nil {
		t.Fatal(err)
	}
	wantCb := [...]int16{12, -37, 4, -18, -13, 6, -5, -43, -8, 11, -29, 48, 67, 57, 20, 13}
	for i, want := range wantCb {
		if grain[i] != want {
			t.Fatalf("cb grain[%d]=%d want %d", i, grain[i], want)
		}
	}
	if got := grain[3*ChromaGrainWidth+3]; got != 14 {
		t.Fatalf("cb grain[3][3]=%d want 14", got)
	}
	if grain[44] != 0 || grain[38*ChromaGrainWidth] != 0 {
		t.Fatalf("subsampled inactive samples=%d %d want zero", grain[44], grain[38*ChromaGrainWidth])
	}

	params.Plane = ChromaPlaneCr
	if err := GenerateChromaGrain(grain, nil, params); err != nil {
		t.Fatal(err)
	}
	wantCr := [...]int16{5, -38, 7, 23, 0, -33, 34, 15, -5, -25, -11, 48, 29, -11, 8, 3}
	for i, want := range wantCr {
		if grain[i] != want {
			t.Fatalf("cr grain[%d]=%d want %d", i, grain[i], want)
		}
	}
}

func TestGenerateChromaGrainAppliesLumaAR(t *testing.T) {
	grain := make([]int16, ChromaGrainSamples)
	luma := make([]int16, LumaGrainSamples)
	for i := range luma {
		luma[i] = 32
	}
	params := ChromaGrainParams{
		Seed:            0x1234,
		Plane:           ChromaPlaneCb,
		BitDepth:        8,
		NumYPoints:      1,
		GrainScaleShift: 0,
		SubsamplingX:    true,
		SubsamplingY:    true,
		ARCoeffShift:    6,
	}
	params.ARCoeffs[0] = 64
	if err := GenerateChromaGrain(grain, luma, params); err != nil {
		t.Fatal(err)
	}
	if got := grain[3*ChromaGrainWidth+3]; got != 46 {
		t.Fatalf("luma AR grain[3][3]=%d want 46", got)
	}
}

func TestGenerateChromaGrainAppliesChromaAR(t *testing.T) {
	baseline := make([]int16, ChromaGrainSamples)
	params := ChromaGrainParams{
		Seed:            0x1234,
		Plane:           ChromaPlaneCb,
		BitDepth:        8,
		GrainScaleShift: 0,
		SubsamplingX:    true,
		SubsamplingY:    true,
	}
	if err := GenerateChromaGrain(baseline, nil, params); err != nil {
		t.Fatal(err)
	}

	grain := make([]int16, ChromaGrainSamples)
	params.ARCoeffLag = 1
	params.ARCoeffShift = 6
	params.ARCoeffs[3] = 64
	if err := GenerateChromaGrain(grain, nil, params); err != nil {
		t.Fatal(err)
	}
	want := baseline[3*ChromaGrainWidth+3] + baseline[3*ChromaGrainWidth+2]
	if got := grain[3*ChromaGrainWidth+3]; got != want {
		t.Fatalf("chroma AR grain[3][3]=%d want %d", got, want)
	}
}

func TestGenerateChromaGrainARLagSpecializationsMatchReference(t *testing.T) {
	for _, tt := range []struct {
		name         string
		lag          uint8
		subsamplingX bool
		subsamplingY bool
	}{
		{name: "lag1-420", lag: 1, subsamplingX: true, subsamplingY: true},
		{name: "lag2-420", lag: 2, subsamplingX: true, subsamplingY: true},
		{name: "lag3-420", lag: 3, subsamplingX: true, subsamplingY: true},
		{name: "lag1-444", lag: 1},
		{name: "lag2-444", lag: 2},
		{name: "lag3-444", lag: 3},
	} {
		t.Run(tt.name, func(t *testing.T) {
			params := ChromaGrainParams{
				Seed:            0x4567,
				Plane:           ChromaPlaneCr,
				BitDepth:        10,
				NumYPoints:      1,
				GrainScaleShift: 1,
				SubsamplingX:    tt.subsamplingX,
				SubsamplingY:    tt.subsamplingY,
				ARCoeffLag:      tt.lag,
				ARCoeffShift:    7,
			}
			for i := range params.ARCoeffs {
				params.ARCoeffs[i] = int8((i%9)-4) * 3
			}
			luma := make([]int16, LumaGrainSamples)
			for i := range luma {
				luma[i] = int16((i % 31) - 15)
			}
			want := referenceChromaGrain(t, luma, params)
			got := make([]int16, ChromaGrainSamples)
			if err := GenerateChromaGrain(got, luma, params); err != nil {
				t.Fatal(err)
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("grain[%d]=%d want %d", i, got[i], want[i])
				}
			}
		})
	}
}

func TestGenerateChromaGrainRejectsInvalidInputs(t *testing.T) {
	grain := make([]int16, ChromaGrainSamples)
	luma := make([]int16, LumaGrainSamples)
	tests := []struct {
		name   string
		dst    []int16
		luma   []int16
		params ChromaGrainParams
	}{
		{name: "short", dst: grain[:ChromaGrainSamples-1], params: ChromaGrainParams{Plane: ChromaPlaneCb, BitDepth: 8}},
		{name: "plane", dst: grain, params: ChromaGrainParams{Plane: 0, BitDepth: 8}},
		{name: "bit-depth", dst: grain, params: ChromaGrainParams{Plane: ChromaPlaneCb, BitDepth: 9}},
		{name: "points", dst: grain, params: ChromaGrainParams{Plane: ChromaPlaneCb, BitDepth: 8, NumYPoints: MaxLumaScalingPoints + 1}},
		{name: "grain-shift", dst: grain, params: ChromaGrainParams{Plane: ChromaPlaneCb, BitDepth: 8, GrainScaleShift: 4}},
		{name: "lag", dst: grain, params: ChromaGrainParams{Plane: ChromaPlaneCb, BitDepth: 8, ARCoeffLag: 4}},
		{name: "coeff-shift", dst: grain, params: ChromaGrainParams{Plane: ChromaPlaneCb, BitDepth: 8, ARCoeffLag: 1, ARCoeffShift: 5}},
		{name: "short-luma", dst: grain, luma: luma[:LumaGrainSamples-1], params: ChromaGrainParams{Plane: ChromaPlaneCb, BitDepth: 8, NumYPoints: 1, ARCoeffShift: 6}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := GenerateChromaGrain(tt.dst, tt.luma, tt.params); !errors.Is(err, ErrInvalidParams) {
				t.Fatalf("GenerateChromaGrain err=%v want %v", err, ErrInvalidParams)
			}
		})
	}
}

func referenceChromaGrain(t *testing.T, luma []int16, params ChromaGrainParams) []int16 {
	t.Helper()
	grain := make([]int16, ChromaGrainSamples)
	rng, err := NewPlaneRandom(params.Seed, int(params.Plane))
	if err != nil {
		t.Fatal(err)
	}
	width, height := chromaGrainDimensions(params.SubsamplingX, params.SubsamplingY)
	gaussianShift := int(12 - params.BitDepth + params.GrainScaleShift)
	for y := range height {
		for x := range width {
			index, err := rng.Number(GaussianBits)
			if err != nil {
				t.Fatal(err)
			}
			grain[y*ChromaGrainWidth+x] = int16(roundPowerOfTwo(int(Gaussian(index)), gaussianShift))
		}
	}
	lag := int(params.ARCoeffLag)
	coeffShift := int(params.ARCoeffShift)
	roundingOffset := 1 << (coeffShift - 1)
	grainMin := -(1 << (params.BitDepth - 1))
	grainMax := (1 << (params.BitDepth - 1)) - 1
	for y := 3; y < height; y++ {
		for x := 3; x < width-3; x++ {
			sum := 0
			pos := 0
			for deltaRow := -lag; deltaRow <= 0; deltaRow++ {
				for deltaCol := -lag; deltaCol <= lag; deltaCol++ {
					if deltaRow == 0 && deltaCol == 0 {
						if params.NumYPoints != 0 {
							sum += chromaLumaAverage(luma, x, y, params.SubsamplingX, params.SubsamplingY) * int(params.ARCoeffs[pos])
						}
						break
					}
					sample := int(grain[(y+deltaRow)*ChromaGrainWidth+x+deltaCol])
					sum += int(params.ARCoeffs[pos]) * sample
					pos++
				}
			}
			v := int(grain[y*ChromaGrainWidth+x]) + ((sum + roundingOffset) >> coeffShift)
			grain[y*ChromaGrainWidth+x] = int16(clipInt(v, grainMin, grainMax))
		}
	}
	return grain
}

func TestGenerateChromaGrainAllocs(t *testing.T) {
	grain := make([]int16, ChromaGrainSamples)
	luma := make([]int16, LumaGrainSamples)
	params := ChromaGrainParams{
		Seed:            0x1234,
		Plane:           ChromaPlaneCb,
		BitDepth:        8,
		NumYPoints:      1,
		GrainScaleShift: 0,
		SubsamplingX:    true,
		SubsamplingY:    true,
		ARCoeffLag:      1,
		ARCoeffShift:    6,
	}
	params.ARCoeffs[3] = 64
	params.ARCoeffs[4] = 32
	allocs := testing.AllocsPerRun(1000, func() {
		if err := GenerateChromaGrain(grain, luma, params); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("GenerateChromaGrain allocated: %f", allocs)
	}
}

func BenchmarkGenerateChromaGrain(b *testing.B) {
	grain := make([]int16, ChromaGrainSamples)
	luma := make([]int16, LumaGrainSamples)
	params := ChromaGrainParams{
		Seed:            0x1234,
		Plane:           ChromaPlaneCb,
		BitDepth:        8,
		NumYPoints:      1,
		GrainScaleShift: 0,
		SubsamplingX:    true,
		SubsamplingY:    true,
		ARCoeffLag:      1,
		ARCoeffShift:    6,
	}
	params.ARCoeffs[3] = 64
	params.ARCoeffs[4] = 32
	b.ReportAllocs()
	for b.Loop() {
		if err := GenerateChromaGrain(grain, luma, params); err != nil {
			b.Fatal(err)
		}
	}
}

func TestGaussianSequence(t *testing.T) {
	if len(gaussianSequence) != GaussianSequenceLen {
		t.Fatalf("len=%d want %d", len(gaussianSequence), GaussianSequenceLen)
	}
	var sum int64
	var abs int64
	var weighted int64
	for i, v := range gaussianSequence {
		value := int64(v)
		sum += value
		if value < 0 {
			abs -= value
		} else {
			abs += value
		}
		weighted += int64(i+1) * value
	}
	if sum != 1120 || abs != 837192 || weighted != -8011268 {
		t.Fatalf("gaussian checksum sum=%d abs=%d weighted=%d", sum, abs, weighted)
	}
	wantFirst := [...]int16{56, 568, -180, 172, 124, -84, 172, -64}
	for i, want := range wantFirst {
		if got := Gaussian(uint16(i)); got != want {
			t.Fatalf("Gaussian(%d)=%d want %d", i, got, want)
		}
	}
	wantMid := [...]int16{320, 1032, 216, 320, -8, -64, 156, -1016}
	for i, want := range wantMid {
		index := 1020 + i
		if got := Gaussian(uint16(index)); got != want {
			t.Fatalf("Gaussian(%d)=%d want %d", index, got, want)
		}
	}
	wantLast := [...]int16{-80, 32, -16, 280, 288, 944, 428, -484}
	for i, want := range wantLast {
		index := GaussianSequenceLen - len(wantLast) + i
		if got := Gaussian(uint16(index)); got != want {
			t.Fatalf("Gaussian(%d)=%d want %d", index, got, want)
		}
	}
	if Gaussian(GaussianSequenceLen) != Gaussian(0) {
		t.Fatalf("masked index sample=%d first=%d", Gaussian(GaussianSequenceLen), Gaussian(0))
	}
}

func TestGaussianSequenceRange(t *testing.T) {
	minValue := int16(2047)
	maxValue := int16(-2048)
	for i, sample := range gaussianSequence {
		if sample < -2048 || sample > 2047 || sample%4 != 0 {
			t.Fatalf("gaussian[%d]=%d outside normative range", i, sample)
		}
		if sample < minValue {
			minValue = sample
		}
		if sample > maxValue {
			maxValue = sample
		}
	}
	if minValue != -1752 || maxValue != 1688 {
		t.Fatalf("range=%d..%d", minValue, maxValue)
	}
}

func TestGaussianAllocs(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		var sum int16
		for i := range uint16(256) {
			sum += Gaussian(i)
		}
		if sum == 0 {
			t.Fatal("unexpected zero gaussian sum")
		}
	})
	if allocs != 0 {
		t.Fatalf("Gaussian allocated: %f", allocs)
	}
}

func BenchmarkGaussian(b *testing.B) {
	var sum int16
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sum += Gaussian(uint16(i))
	}
	if sum == 0 {
		b.Fatal("unexpected zero gaussian sum")
	}
}

func TestGenerateLumaGrainZeroPoints(t *testing.T) {
	grain := make([]int16, LumaGrainSamples)
	for i := range grain {
		grain[i] = 99
	}
	if err := GenerateLumaGrain(grain, LumaGrainParams{BitDepth: 8}); err != nil {
		t.Fatal(err)
	}
	for i, sample := range grain {
		if sample != 0 {
			t.Fatalf("grain[%d]=%d want 0", i, sample)
		}
	}
}

func TestGenerateLumaGrainWhiteNoise(t *testing.T) {
	grain := make([]int16, LumaGrainSamples)
	params := LumaGrainParams{
		Seed:            0x1234,
		BitDepth:        8,
		NumYPoints:      1,
		ARCoeffShift:    6,
		GrainScaleShift: 0,
	}
	if err := GenerateLumaGrain(grain, params); err != nil {
		t.Fatal(err)
	}
	wantFirst := [...]int16{-29, -1, -31, -20, 49, 8, -56, 54, 18, -7, -35, -47, -51, -3, 60, 26}
	for i, want := range wantFirst {
		if grain[i] != want {
			t.Fatalf("grain[%d]=%d want %d", i, grain[i], want)
		}
	}
	if got := grain[3*LumaGrainWidth+3]; got != -14 {
		t.Fatalf("grain[3][3]=%d want -14", got)
	}
}

func TestGenerateLumaGrainAppliesAR(t *testing.T) {
	grain := make([]int16, LumaGrainSamples)
	params := LumaGrainParams{
		Seed:            0x1234,
		BitDepth:        8,
		NumYPoints:      1,
		ARCoeffLag:      1,
		ARCoeffShift:    6,
		GrainScaleShift: 0,
	}
	params.ARCoeffs[3] = 64
	if err := GenerateLumaGrain(grain, params); err != nil {
		t.Fatal(err)
	}
	wantRow := [...]int16{-24, 42, -26, -40, 24, 42, 13, -66, -67, -70, -74}
	row := grain[3*LumaGrainWidth:]
	for i, want := range wantRow {
		if row[i] != want {
			t.Fatalf("row[3][%d]=%d want %d", i, row[i], want)
		}
	}
}

func TestGenerateLumaGrainARLagSpecializationsMatchReference(t *testing.T) {
	for _, lag := range []uint8{1, 2, 3} {
		t.Run("lag"+string(rune('0'+lag)), func(t *testing.T) {
			params := LumaGrainParams{
				Seed:            0x4567,
				BitDepth:        10,
				NumYPoints:      1,
				GrainScaleShift: 1,
				ARCoeffLag:      lag,
				ARCoeffShift:    7,
			}
			for i := range params.ARCoeffs {
				params.ARCoeffs[i] = int8((i%9)-4) * 3
			}
			want := referenceLumaGrain(t, params)
			got := make([]int16, LumaGrainSamples)
			if err := GenerateLumaGrain(got, params); err != nil {
				t.Fatal(err)
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("grain[%d]=%d want %d", i, got[i], want[i])
				}
			}
		})
	}
}

func TestGenerateLumaGrainRejectsInvalidInputs(t *testing.T) {
	grain := make([]int16, LumaGrainSamples)
	tests := []struct {
		name   string
		dst    []int16
		params LumaGrainParams
	}{
		{name: "short", dst: grain[:LumaGrainSamples-1], params: LumaGrainParams{BitDepth: 8}},
		{name: "bit-depth", dst: grain, params: LumaGrainParams{BitDepth: 9}},
		{name: "points", dst: grain, params: LumaGrainParams{BitDepth: 8, NumYPoints: MaxLumaScalingPoints + 1}},
		{name: "grain-shift", dst: grain, params: LumaGrainParams{BitDepth: 8, GrainScaleShift: 4}},
		{name: "lag", dst: grain, params: LumaGrainParams{BitDepth: 8, ARCoeffLag: 4}},
		{name: "coeff-shift", dst: grain, params: LumaGrainParams{BitDepth: 8, NumYPoints: 1, ARCoeffLag: 1, ARCoeffShift: 5}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := GenerateLumaGrain(tt.dst, tt.params); !errors.Is(err, ErrInvalidParams) {
				t.Fatalf("GenerateLumaGrain err=%v want %v", err, ErrInvalidParams)
			}
		})
	}
}

func referenceLumaGrain(t *testing.T, params LumaGrainParams) []int16 {
	t.Helper()
	grain := make([]int16, LumaGrainSamples)
	rng := NewRandom(params.Seed)
	gaussianShift := int(12 - params.BitDepth + params.GrainScaleShift)
	for y := range LumaGrainHeight {
		for x := range LumaGrainWidth {
			index, err := rng.Number(GaussianBits)
			if err != nil {
				t.Fatal(err)
			}
			grain[y*LumaGrainWidth+x] = int16(roundPowerOfTwo(int(Gaussian(index)), gaussianShift))
		}
	}

	lag := int(params.ARCoeffLag)
	coeffShift := int(params.ARCoeffShift)
	roundingOffset := 1 << (coeffShift - 1)
	grainMin := -(1 << (params.BitDepth - 1))
	grainMax := (1 << (params.BitDepth - 1)) - 1
	for y := 3; y < LumaGrainHeight; y++ {
		for x := 3; x < LumaGrainWidth-3; x++ {
			sum := 0
			pos := 0
			for deltaRow := -lag; deltaRow <= 0; deltaRow++ {
				for deltaCol := -lag; deltaCol <= lag; deltaCol++ {
					if deltaRow == 0 && deltaCol == 0 {
						break
					}
					sample := int(grain[(y+deltaRow)*LumaGrainWidth+x+deltaCol])
					sum += int(params.ARCoeffs[pos]) * sample
					pos++
				}
			}
			v := int(grain[y*LumaGrainWidth+x]) + ((sum + roundingOffset) >> coeffShift)
			grain[y*LumaGrainWidth+x] = int16(clipInt(v, grainMin, grainMax))
		}
	}
	return grain
}

func TestGenerateLumaGrainAllocs(t *testing.T) {
	grain := make([]int16, LumaGrainSamples)
	params := LumaGrainParams{
		Seed:            0x1234,
		BitDepth:        8,
		NumYPoints:      1,
		ARCoeffLag:      1,
		ARCoeffShift:    6,
		GrainScaleShift: 0,
	}
	params.ARCoeffs[3] = 64
	allocs := testing.AllocsPerRun(1000, func() {
		if err := GenerateLumaGrain(grain, params); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("GenerateLumaGrain allocated: %f", allocs)
	}
}

func BenchmarkGenerateLumaGrain(b *testing.B) {
	grain := make([]int16, LumaGrainSamples)
	params := LumaGrainParams{
		Seed:            0x1234,
		BitDepth:        8,
		NumYPoints:      1,
		ARCoeffLag:      1,
		ARCoeffShift:    6,
		GrainScaleShift: 0,
	}
	params.ARCoeffs[3] = 64
	b.ReportAllocs()
	for b.Loop() {
		if err := GenerateLumaGrain(grain, params); err != nil {
			b.Fatal(err)
		}
	}
}

func TestRandomNumber(t *testing.T) {
	rng := NewRandom(0x1234)
	wantValues := []uint16{0x89, 0x44, 0x22, 0x91, 0xc8, 0xe4, 0xf2, 0xf9}
	wantRegisters := []uint16{0x891a, 0x448d, 0x2246, 0x9123, 0xc891, 0xe448, 0xf224, 0xf912}
	for i := range wantValues {
		got, err := rng.Number(8)
		if err != nil {
			t.Fatal(err)
		}
		if got != wantValues[i] || rng.register != wantRegisters[i] {
			t.Fatalf("step %d value=%#x register=%#x want %#x %#x", i, got, rng.register, wantValues[i], wantRegisters[i])
		}
	}
}

func TestRandomNumberElevenBits(t *testing.T) {
	rng := NewRandom(0x1234)
	for i, want := range []uint16{1096, 548, 274, 1161, 1604} {
		got, err := rng.Number(11)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("step %d value=%d want %d", i, got, want)
		}
	}
}

func TestRandomSeeds(t *testing.T) {
	cb, err := NewPlaneRandom(0x1234, 1)
	if err != nil {
		t.Fatal(err)
	}
	if cb.register != 0xa710 {
		t.Fatalf("cb register=%#x", cb.register)
	}
	cr, err := NewPlaneRandom(0x1234, 2)
	if err != nil {
		t.Fatal(err)
	}
	if cr.register != 0x5bec {
		t.Fatalf("cr register=%#x", cr.register)
	}
	stripe, err := NewStripeRandom(0x1234, 32)
	if err != nil {
		t.Fatal(err)
	}
	if stripe.register != 0xc522 {
		t.Fatalf("stripe register=%#x", stripe.register)
	}
}

func TestRandomRejectsInvalidInputs(t *testing.T) {
	rng := NewRandom(0)
	if _, err := rng.Number(0); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("Number(0) err=%v want %v", err, ErrInvalidParams)
	}
	if _, err := rng.Number(17); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("Number(17) err=%v want %v", err, ErrInvalidParams)
	}
	if _, err := NewPlaneRandom(0, 0); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("NewPlaneRandom err=%v want %v", err, ErrInvalidParams)
	}
	if _, err := NewStripeRandom(0, -1); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("NewStripeRandom err=%v want %v", err, ErrInvalidParams)
	}
}

func TestRandomAllocs(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		rng := NewRandom(0x1234)
		for range 64 {
			if _, err := rng.Number(11); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := NewPlaneRandom(rng.register, 1); err != nil {
			t.Fatal(err)
		}
		if _, err := NewStripeRandom(rng.register, 64); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("filmgrain random allocated: %f", allocs)
	}
}

func BenchmarkRandomNumber(b *testing.B) {
	rng := NewRandom(0x1234)
	b.ReportAllocs()
	for b.Loop() {
		_, _ = rng.Number(11)
	}
}

func TestLumaGrainSample(t *testing.T) {
	grain := testIndexedLumaGrain()
	tests := []struct {
		name     string
		offset   uint8
		blockCol int
		blockRow int
		x        int
		y        int
		wantCol  int
		wantRow  int
	}{
		{name: "current", offset: 0x00, x: 0, y: 0, wantCol: 9, wantRow: 9},
		{name: "left", offset: 0x00, blockCol: 1, x: 0, y: 0, wantCol: 41, wantRow: 9},
		{name: "top", offset: 0x00, blockRow: 1, x: 0, y: 0, wantCol: 9, wantRow: 41},
		{name: "top-left", offset: 0x00, blockCol: 1, blockRow: 1, x: 0, y: 0, wantCol: 41, wantRow: 41},
		{name: "max-current", offset: 0xff, x: 31, y: 31, wantCol: 70, wantRow: 70},
		{name: "max-overlap", offset: 0xff, blockCol: 1, blockRow: 1, x: 1, y: 1, wantCol: 72, wantRow: 72},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := lumaGrainSample(grain, tt.offset, tt.blockCol, tt.blockRow, tt.x, tt.y)
			want := int16(tt.wantRow*LumaGrainWidth + tt.wantCol)
			if got != want {
				t.Fatalf("sample=%d want %d", got, want)
			}
		})
	}
}

func TestChromaGrainSample(t *testing.T) {
	grain := testIndexedChromaGrain()
	tests := []struct {
		name         string
		offset       uint8
		subsamplingX bool
		subsamplingY bool
		blockCol     int
		blockRow     int
		x            int
		y            int
		wantCol      int
		wantRow      int
	}{
		{name: "420-current", offset: 0x00, subsamplingX: true, subsamplingY: true, x: 0, y: 0, wantCol: 6, wantRow: 6},
		{name: "420-left", offset: 0x00, subsamplingX: true, subsamplingY: true, blockCol: 1, x: 0, y: 0, wantCol: 22, wantRow: 6},
		{name: "420-top", offset: 0x00, subsamplingX: true, subsamplingY: true, blockRow: 1, x: 0, y: 0, wantCol: 6, wantRow: 22},
		{name: "420-top-left", offset: 0x00, subsamplingX: true, subsamplingY: true, blockCol: 1, blockRow: 1, x: 0, y: 0, wantCol: 22, wantRow: 22},
		{name: "420-max-current", offset: 0xff, subsamplingX: true, subsamplingY: true, x: 15, y: 15, wantCol: 36, wantRow: 36},
		{name: "420-max-overlap", offset: 0xff, subsamplingX: true, subsamplingY: true, blockCol: 1, blockRow: 1, x: 0, y: 0, wantCol: 37, wantRow: 37},
		{name: "422-current", offset: 0x00, subsamplingX: true, x: 0, y: 0, wantCol: 6, wantRow: 9},
		{name: "422-top", offset: 0x00, subsamplingX: true, blockRow: 1, x: 0, y: 0, wantCol: 6, wantRow: 41},
		{name: "422-max-current", offset: 0xff, subsamplingX: true, x: 15, y: 31, wantCol: 36, wantRow: 70},
		{name: "422-max-overlap", offset: 0xff, subsamplingX: true, blockCol: 1, blockRow: 1, x: 0, y: 1, wantCol: 37, wantRow: 72},
		{name: "444-current", offset: 0x00, x: 0, y: 0, wantCol: 9, wantRow: 9},
		{name: "444-overlap", offset: 0xff, blockCol: 1, blockRow: 1, x: 1, y: 1, wantCol: 72, wantRow: 72},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := chromaGrainSample(grain, tt.offset, chromaSubsamplingShift(tt.subsamplingX), chromaSubsamplingShift(tt.subsamplingY), tt.blockCol, tt.blockRow, tt.x, tt.y)
			want := int16(tt.wantRow*ChromaGrainWidth + tt.wantCol)
			if got != want {
				t.Fatalf("sample=%d want %d", got, want)
			}
		})
	}
}

func testIndexedLumaGrain() []int16 {
	grain := make([]int16, LumaGrainSamples)
	for i := range grain {
		grain[i] = int16(i)
	}
	return grain
}

func testIndexedChromaGrain() []int16 {
	grain := make([]int16, ChromaGrainSamples)
	for i := range grain {
		grain[i] = int16(i)
	}
	return grain
}

func TestBuildScalingLUT(t *testing.T) {
	tests := []struct {
		name    string
		points  []ScalingPoint
		samples map[int]uint8
	}{
		{
			name:    "empty",
			samples: map[int]uint8{0: 0, 1: 0, 255: 0},
		},
		{
			name:    "single point",
			points:  []ScalingPoint{{Value: 10, Scaling: 20}},
			samples: map[int]uint8{0: 20, 9: 20, 10: 20, 255: 20},
		},
		{
			name:    "rising segment",
			points:  []ScalingPoint{{Value: 10, Scaling: 20}, {Value: 20, Scaling: 40}},
			samples: map[int]uint8{0: 20, 9: 20, 10: 20, 11: 22, 14: 28, 15: 30, 19: 38, 20: 40, 255: 40},
		},
		{
			name:    "falling segment",
			points:  []ScalingPoint{{Value: 0, Scaling: 100}, {Value: 10, Scaling: 0}},
			samples: map[int]uint8{0: 100, 1: 90, 2: 80, 3: 70, 4: 60, 5: 50, 6: 40, 7: 30, 8: 20, 9: 10, 10: 0, 255: 0},
		},
		{
			name:    "uneven segments",
			points:  []ScalingPoint{{Value: 3, Scaling: 7}, {Value: 8, Scaling: 11}, {Value: 20, Scaling: 2}},
			samples: map[int]uint8{0: 7, 2: 7, 3: 7, 4: 8, 5: 9, 6: 9, 7: 10, 8: 11, 9: 10, 19: 3, 20: 2, 255: 2},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var lut [ScalingLUTSize]uint8
			if err := BuildScalingLUT(lut[:], tt.points); err != nil {
				t.Fatal(err)
			}
			for x, want := range tt.samples {
				if got := lut[x]; got != want {
					t.Fatalf("lut[%d]=%d want %d", x, got, want)
				}
			}
		})
	}
}

func TestNormativeDimensions(t *testing.T) {
	if GaussianSequenceLen != 2048 || GaussianBits != 11 {
		t.Fatalf("gaussian bits=%d len=%d", GaussianBits, GaussianSequenceLen)
	}
	if LumaGrainWidth != 82 || LumaGrainHeight != 73 || LumaGrainSamples != 5986 {
		t.Fatalf("luma grain dimensions=%dx%d samples=%d", LumaGrainWidth, LumaGrainHeight, LumaGrainSamples)
	}
	if LumaBlockSize != 32 || NoiseStripeHeight != 34 || LumaOverlapSamples != 2 {
		t.Fatalf("luma block=%d stripe=%d overlap=%d", LumaBlockSize, NoiseStripeHeight, LumaOverlapSamples)
	}
}

func TestScaleLUT(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 0, Scaling: 0}, {Value: 255, Scaling: 255}}); err != nil {
		t.Fatal(err)
	}
	tests := map[int]uint8{
		0:    0,
		1:    0,
		2:    1,
		3:    1,
		4:    1,
		510:  128,
		511:  128,
		512:  128,
		1022: 255,
		1023: 255,
	}
	for index, want := range tests {
		got := scaleLUT(lut[:], index, 10)
		if got != want {
			t.Fatalf("scaleLUT(%d, 10)=%d want %d", index, got, want)
		}
	}
	got := scaleLUT(lut[:], 4095, 12)
	if got != 255 {
		t.Fatalf("scaleLUT(4095, 12)=%d want 255", got)
	}
}

func TestScalingRejectsInvalidInputs(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:ScalingLUTSize-1], nil); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("short lut err=%v want %v", err, ErrInvalidParams)
	}
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 10}, {Value: 10}}); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("duplicate point err=%v want %v", err, ErrInvalidParams)
	}
}

// TestScaleLUTClampsOutOfRangeIndex exercises the defense-in-depth clamp in
// the internal scaleLUT helper. Internal apply callers feed raw uint16 sample
// values that, on a malformed bitstream, can exceed (1<<bitDepth)-1 and would
// otherwise drive lut[x] out of bounds.
func TestScaleLUTClampsOutOfRangeIndex(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 0, Scaling: 0}, {Value: 255, Scaling: 255}}); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name     string
		index    int
		bitDepth uint8
		want     uint8
	}{
		{"8bit above max", 1 << 12, 8, lut[ScalingLUTSize-1]},
		{"10bit above max", 1 << 14, 10, lut[ScalingLUTSize-1]},
		{"12bit above max", 1 << 16, 12, lut[ScalingLUTSize-1]},
		{"negative index", -1, 10, lut[0]},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := scaleLUT(lut[:], tc.index, tc.bitDepth)
			if got != tc.want {
				t.Fatalf("scaleLUT(%d, %d)=%d want %d", tc.index, tc.bitDepth, got, tc.want)
			}
		})
	}
}

// TestApplyLumaSampleClampsOutOfRangeOrig confirms applyLumaSample does not
// panic when the original sample is outside the legal bitdepth range: it clips
// defensively so concurrent apply paths cannot OOB.
func TestApplyLumaSampleClampsOutOfRangeOrig(t *testing.T) {
	var lut [ScalingLUTSize]uint8
	if err := BuildScalingLUT(lut[:], []ScalingPoint{{Value: 0, Scaling: 0}, {Value: 255, Scaling: 255}}); err != nil {
		t.Fatal(err)
	}
	// applyLumaSample must not panic for any uint16 input under any
	// supported bitdepth even when the sample exceeds the bitdepth range.
	for _, bd := range []uint8{8, 10, 12} {
		for _, orig := range []uint16{0, 1, 1<<bd - 1, 1 << bd, (1 << 16) - 1} {
			_ = applyLumaSample(orig, 0, lut[:], bd, 8, false)
			_ = applyLumaSample(orig, 0, lut[:], bd, 11, true)
		}
	}
}

func TestScalingAllocs(t *testing.T) {
	points := [...]ScalingPoint{{Value: 3, Scaling: 7}, {Value: 8, Scaling: 11}, {Value: 20, Scaling: 2}}
	var lut [ScalingLUTSize]uint8
	allocs := testing.AllocsPerRun(1000, func() {
		if err := BuildScalingLUT(lut[:], points[:]); err != nil {
			t.Fatal(err)
		}
		for _, index := range [...]int{0, 17, 255, 510, 1023, 2047, 4095} {
			_ = scaleLUT(lut[:], index, 12)
		}
	})
	if allocs != 0 {
		t.Fatalf("filmgrain scaling allocated: %f", allocs)
	}
}

func BenchmarkBuildScalingLUT(b *testing.B) {
	points := [...]ScalingPoint{{Value: 3, Scaling: 7}, {Value: 8, Scaling: 11}, {Value: 20, Scaling: 2}}
	var lut [ScalingLUTSize]uint8
	b.ReportAllocs()
	for b.Loop() {
		_ = BuildScalingLUT(lut[:], points[:])
	}
}
