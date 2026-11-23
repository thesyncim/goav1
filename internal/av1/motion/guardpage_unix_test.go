// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

package motion

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/guardpage"
)

// guardPlane returns a w x h plane of bps-byte samples whose Pix is exactly
// (h-1)*stride + w*bps bytes, ending at a PROT_NONE page (or, with before,
// starting right after one), filled with samples below 1<<bitDepth.
func guardPlane(t *testing.T, rng *rand.Rand, w, h, stride, bps, bitDepth int, before bool) frame.Plane {
	n := (h-1)*stride + w*bps
	var pix []byte
	if before {
		pix = guardpage.Before[byte](t, n)
	} else {
		pix = guardpage.After[byte](t, n)
	}
	for i := 0; i < n; i += bps {
		v := rng.Intn(1 << bitDepth)
		pix[i] = byte(v)
		if bps == 2 {
			pix[i+1] = byte(v >> 8)
		}
	}
	return frame.Plane{Pix: pix, Stride: stride, Width: w, Height: h}
}

func heapPlane(p frame.Plane) frame.Plane {
	p.Pix = slices.Clone(p.Pix)
	return p
}

var guardSizes = [][2]int{{2, 2}, {4, 4}, {4, 8}, {4, 16}, {8, 4}, {8, 8}, {8, 32}, {16, 4}, {16, 16}, {32, 8}, {32, 32}, {64, 16}, {64, 64}, {128, 128}, {16, 64}, {24, 6}}

// guardRefPos picks a reference origin near an edge, inside, or outside the
// plane so the in-place window and the emu-edge path are both reached.
func guardRefPos(rng *rand.Rand, iter, rw, rh, w, h int) (int, int) {
	switch iter % 5 {
	case 0:
		return rw - w - 4 + rng.Intn(8), rh - h - 4 + rng.Intn(8)
	case 1:
		return rng.Intn(8) - 4, rng.Intn(8) - 4
	case 2:
		return rw - w, rh - h
	case 3:
		return 4, 4
	default:
		return rng.Intn(rw+2*w) - w - 3, rng.Intn(rh+2*h) - h - 3
	}
}

func TestMotionKernelsStayInsideGuardedPlanes(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	filters := []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpBilinear, InterpMultiTapSharp}
	for _, bps := range []int{1, 2} {
		bd := 8
		if bps == 2 {
			bd = 10
		}
		for _, sz := range guardSizes {
			w, h := sz[0], sz[1]
			// The reference is barely larger than the block, so the
			// in-place window sits against the guard page.
			rw, rh := w+3, h+3
			for _, before := range []bool{false, true} {
				for iter := range 40 {
					ref := guardPlane(t, rng, rw, rh, rw*bps, bps, bd, before)
					dst := guardPlane(t, rng, w, h, w*bps, bps, bd, before)
					refX, refY := guardRefPos(rng, iter, rw, rh, w, h)
					f := InterpFilters{X: filters[rng.Intn(4)], Y: filters[rng.Intn(4)]}
					subX, subY := rng.Intn(16), rng.Intn(16)
					switch iter % 4 {
					case 0:
						subX = 0
					case 1:
						subY = 0
					case 2:
						subX, subY = 0, 0
					}
					mv := Vector{Row: int16((refY*8 + subY/2)), Col: int16((refX*8 + subX/2))}
					name := fmt.Sprintf("bps=%d %dx%d before=%v ref=%d,%d sub=%d,%d %v", bps, w, h, before, refX, refY, subX, subY, f)
					heapDst, heapRef := heapPlane(dst), heapPlane(ref)
					guardpage.Check(t, name+" put", func() {
						var err, herr error
						if bps == 1 {
							err = PredictTranslation8(&dst, &ref, 0, 0, w, h, w, h, mv, false, false, f, new(ConvolveScratch))
							herr = PredictTranslation8(&heapDst, &heapRef, 0, 0, w, h, w, h, mv, false, false, f, new(ConvolveScratch))
						} else {
							err = PredictTranslation16(&dst, &ref, uint8(bd), 0, 0, w, h, w, h, mv, false, false, f, new(ConvolveScratch))
							herr = PredictTranslation16(&heapDst, &heapRef, uint8(bd), 0, 0, w, h, w, h, mv, false, false, f, new(ConvolveScratch))
						}
						if (err == nil) != (herr == nil) || !slices.Equal(dst.Pix, heapDst.Pix) {
							t.Fatalf("%s: guarded put differs from heap put (err %v / %v)", name, err, herr)
						}
					})
					guardpage.Check(t, name+" prep", func() {
						var a, b CompoundConvBuf
						var err, herr error
						if bps == 1 {
							err = PrepTranslation8(&a, &ref, 0, 0, w, h, w, h, mv, false, false, f, new(CompoundConvolveScratch))
							herr = PrepTranslation8(&b, &heapRef, 0, 0, w, h, w, h, mv, false, false, f, new(CompoundConvolveScratch))
						} else {
							err = PrepTranslation16(&a, &ref, uint8(bd), 0, 0, w, h, w, h, mv, false, false, f, new(CompoundConvolveScratch))
							herr = PrepTranslation16(&b, &heapRef, uint8(bd), 0, 0, w, h, w, h, mv, false, false, f, new(CompoundConvolveScratch))
						}
						if (err == nil) != (herr == nil) || a.Data != b.Data {
							t.Fatalf("%s: guarded prep differs from heap prep (err %v / %v)", name, err, herr)
						}
						guardBlends(t, name, rng, &a, &b, w, h, bps, bd, before)
					})
				}
			}
		}
	}
}

// guardBlends runs the compound blends and mask builder into a guarded
// destination and mask that are exactly the block's size.
func guardBlends(t *testing.T, name string, rng *rand.Rand, a, b *CompoundConvBuf, w, h, bps, bd int, before bool) {
	dst := guardPlane(t, rng, w, h, w*bps, bps, bd, before)
	heap := heapPlane(dst)
	if err := BlendCompoundAvg(dst, a, b, bps, uint8(bd), 0, 0, w, h, 9, 7); err != nil {
		t.Fatalf("%s: BlendCompoundAvg: %v", name, err)
	}
	if err := BlendCompoundAvg(heap, a, b, bps, uint8(bd), 0, 0, w, h, 9, 7); err != nil || !slices.Equal(dst.Pix, heap.Pix) {
		t.Fatalf("%s: BlendCompoundAvg guarded differs (%v)", name, err)
	}
	if bps == 1 && BlendAvg8(&dst, a, b, 0, 0, w, h) {
		BlendAvg8(&heap, a, b, 0, 0, w, h)
		if !slices.Equal(dst.Pix, heap.Pix) {
			t.Fatalf("%s: BlendAvg8 guarded differs", name)
		}
	}
	var mask []byte
	if before {
		mask = guardpage.Before[byte](t, w*h)
	} else {
		mask = guardpage.After[byte](t, w*h)
	}
	if err := BuildDiffWtdMaskD16(mask, w, a, b, uint8(bd), w, h, rng.Intn(2) == 0); err != nil {
		t.Fatalf("%s: BuildDiffWtdMaskD16: %v", name, err)
	}
	if err := BlendCompoundMaskD16(dst, a, b, bps, uint8(bd), 0, 0, w, h, mask, w, false, false); err != nil {
		t.Fatalf("%s: BlendCompoundMaskD16: %v", name, err)
	}
	if err := BlendCompoundMaskD16(heap, a, b, bps, uint8(bd), 0, 0, w, h, slices.Clone(mask), w, false, false); err != nil || !slices.Equal(dst.Pix, heap.Pix) {
		t.Fatalf("%s: BlendCompoundMaskD16 guarded differs (%v)", name, err)
	}
	if w >= 4 && h >= 4 && w%2 == 0 && h%2 == 0 {
		w2, h2 := w/2, h/2
		sub := guardPlane(t, rng, w2, h2, w2*bps, bps, bd, before)
		sa, sb := *a, *b
		sa.Width, sa.Height, sb.Width, sb.Height = uint8(w2), uint8(h2), uint8(w2), uint8(h2)
		if err := BlendCompoundMaskD16(sub, &sa, &sb, bps, uint8(bd), 0, 0, w2, h2, mask, w, true, true); err != nil {
			t.Fatalf("%s: BlendCompoundMaskD16 subsampled: %v", name, err)
		}
	}
}

func warpReduce(v int) int16 {
	if v < 0 {
		return int16(-((-v + 32) >> 6) * 64)
	}
	return int16(((v + 32) >> 6) * 64)
}

func TestWarpAndScaledStayInsideGuardedPlanes(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	sizes := [][2]int{{8, 8}, {16, 8}, {8, 16}, {16, 16}, {32, 32}, {64, 64}, {4, 4}, {4, 8}, {8, 4}}
	var ran [4]int
	for _, bps := range []int{1, 2} {
		bd := 8
		if bps == 2 {
			bd = 10
		}
		for _, sz := range sizes {
			w, h := sz[0], sz[1]
			for _, before := range []bool{false, true} {
				for iter := range 24 {
					rw, rh := w+8, h+8
					ref := guardPlane(t, rng, rw, rh, rw*bps, bps, bd, before)
					heapRef := heapPlane(ref)
					px, py := guardRefPos(rng, iter, rw, rh, w, h)
					matrix := [6]int32{
						int32(px)<<16 + int32(rng.Intn(1<<16)),
						int32(py)<<16 + int32(rng.Intn(1<<16)),
						1<<16 + int32(rng.Intn(8192)) - 4096,
						int32(rng.Intn(8192)) - 4096,
						int32(rng.Intn(8192)) - 4096,
						1<<16 + int32(rng.Intn(8192)) - 4096,
					}
					alpha, beta := warpReduce(rng.Intn(1024)-512), warpReduce(rng.Intn(1024)-512)
					gamma, delta := warpReduce(rng.Intn(1024)-512), warpReduce(rng.Intn(1024)-512)
					name := fmt.Sprintf("bps=%d %dx%d before=%v iter=%d", bps, w, h, before, iter)
					guardpage.Check(t, name+" warp", func() {
						dst := guardPlane(t, rng, w, h, w*bps, bps, bd, before)
						heap := heapPlane(dst)
						err := PredictWarpedPlaneBlockBitDepth(dst, ref, bps, uint8(bd), 0, 0, w, h, matrix, alpha, beta, gamma, delta, false, false)
						herr := PredictWarpedPlaneBlockBitDepth(heap, heapRef, bps, uint8(bd), 0, 0, w, h, matrix, alpha, beta, gamma, delta, false, false)
						if (err == nil) != (herr == nil) || !slices.Equal(dst.Pix, heap.Pix) {
							t.Fatalf("%s: guarded warp differs (%v / %v)", name, err, herr)
						}
						if err == nil {
							ran[0]++
						}
						var a, b CompoundConvBuf
						err = PredictWarpedCompoundToConvBuf(&a, ref, bps, uint8(bd), 0, 0, w, h, matrix, alpha, beta, gamma, delta, false, false)
						herr = PredictWarpedCompoundToConvBuf(&b, heapRef, bps, uint8(bd), 0, 0, w, h, matrix, alpha, beta, gamma, delta, false, false)
						if (err == nil) != (herr == nil) || a.Data != b.Data {
							t.Fatalf("%s: guarded warp compound differs (%v / %v)", name, err, herr)
						}
						if err == nil {
							ran[1]++
						}
					})
					guardpage.Check(t, name+" scaled", func() {
						f := []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpBilinear, InterpMultiTapSharp}[iter%4]
						xt, err := SubpelKernelTableFor(f, w)
						if err != nil {
							t.Fatal(err)
						}
						yt, _ := SubpelKernelTableFor(f, h)
						steps := []int64{ScaleSubpelScale, ScaleSubpelScale * 3 / 2, ScaleSubpelScale * 2, ScaleSubpelScale * 5 / 4}
						xStep, yStep := steps[rng.Intn(4)], steps[rng.Intn(4)]
						startX := int64(px)<<ScaleSubpelBits + rng.Int63n(1<<ScaleSubpelBits)
						startY := int64(py)<<ScaleSubpelBits + rng.Int63n(1<<ScaleSubpelBits)
						var a, b CompoundConvBuf
						err = PredictScaledCompoundRefToConvBufWithScratch(&a, ref, bps, uint8(bd), w, h, startX, xStep, startY, yStep, xt, yt, new(ScaledConvolveScratch))
						herr := PredictScaledCompoundRefToConvBufWithScratch(&b, heapRef, bps, uint8(bd), w, h, startX, xStep, startY, yStep, xt, yt, new(ScaledConvolveScratch))
						if (err == nil) != (herr == nil) || a.Data != b.Data {
							t.Fatalf("%s: guarded scaled compound differs (%v / %v)", name, err, herr)
						}
						if err == nil {
							ran[2]++
						}
						dst := guardPlane(t, rng, w, h, w*bps, bps, bd, before)
						heap := heapPlane(dst)
						if bps == 1 {
							err = ConvolveScale2D8ClampedWithScratch(dst, ref, 0, 0, w, h, startX, xStep, startY, yStep, xt, yt, new(ScaledConvolveScratch))
							herr = ConvolveScale2D8ClampedWithScratch(heap, heapRef, 0, 0, w, h, startX, xStep, startY, yStep, xt, yt, new(ScaledConvolveScratch))
						} else {
							err = ConvolveScale2DHighBDClampedWithScratch(dst, ref, uint8(bd), 0, 0, w, h, startX, xStep, startY, yStep, xt, yt, new(ScaledConvolveScratch))
							herr = ConvolveScale2DHighBDClampedWithScratch(heap, heapRef, uint8(bd), 0, 0, w, h, startX, xStep, startY, yStep, xt, yt, new(ScaledConvolveScratch))
						}
						if (err == nil) != (herr == nil) || !slices.Equal(dst.Pix, heap.Pix) {
							t.Fatalf("%s: guarded scaled put differs (%v / %v)", name, err, herr)
						}
						if err == nil {
							ran[3]++
						}
					})
				}
			}
		}
	}
	if slices.Contains(ran[:], 0) {
		t.Fatalf("a kernel never ran (warp, warp compound, scaled compound, scaled put): %v", ran)
	}
}
