// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package filmgrain

import (
	"fmt"
	"math/rand"
	"testing"
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
