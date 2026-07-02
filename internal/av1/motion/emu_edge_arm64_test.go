// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"math/rand"
	"testing"
)

// emuEdgeTestGeometries yields block/ref placements whose 8-tap halo
// overhangs every combination of plane edges (single sides, corners, and
// windows fully outside the plane), which is exactly the population the
// emu_edge path serves.
func emuEdgeTestGeometries(rng *rand.Rand, refW int, refH int, w int, h int) [][2]int {
	return [][2]int{
		{-w - 2, rng.Intn(refH)},             // fully left
		{refW + 2, rng.Intn(refH)},           // fully right
		{rng.Intn(refW), -h - 2},             // fully above
		{rng.Intn(refW), refH + 2},           // fully below
		{-2, -3},                             // top-left corner
		{refW - w + 2, -3},                   // top-right corner
		{-2, refH - h + 3},                   // bottom-left corner
		{refW - w + 2, refH - h + 3},         // bottom-right corner
		{-1, rng.Intn(refH - 1)},             // left edge
		{refW - w + 1, rng.Intn(refH - 1)},   // right edge
		{rng.Intn(refW - 1), -1},             // top edge
		{rng.Intn(refW - 1), refH - h + 1},   // bottom edge
		{-w - refW, -h - refH},               // far outside corner
		{rng.Intn(refW - 1), rng.Intn(refH)}, // interior-ish (halo may clip)
	}
}

// TestConvolve2D8ClampedEmuEdgeMatchesPureGo pins the emu_edge fast path
// (dav1d src/recon_tmpl.c mc(): emu_edge + plain 8tap) byte-exact against the
// per-tap clamped pure-Go reference for the single-prediction 2D kernels,
// through both the I8MM and NEON clamped entries.
func TestConvolve2D8ClampedEmuEdgeMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xe19e))
	widths := []int{4, 8, 16, 32, 64, 128}
	heights := []int{4, 8, 16, 32}
	phasePairs := [][2][filterTaps]int16{
		{subpelFilters8[3], subpelFilters8[5]},
		{subpelFilters8Smooth[6], subpelFilters8Smooth[11]},
		{subpelFilters8Sharp[9], subpelFilters8Sharp[13]},
		{bilinearFilters[7], bilinearFilters[2]},
	}
	const refW, refH = 96, 72
	ref, _ := testPlane(refW, refH, 1, refW+5)
	for i := range ref.Pix {
		ref.Pix[i] = byte(rng.Intn(256))
	}
	for _, w := range widths {
		for _, h := range heights {
			for _, kernels := range phasePairs {
				for _, at := range emuEdgeTestGeometries(rng, refW, refH, w, h) {
					refX, refY := at[0], at[1]
					gotI8MM, _ := testPlane(w, h, 1, w)
					gotNEON, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					var scratchA, scratchB ConvolveScratch
					convolve2D8ClampedI8MMWithScratch(gotI8MM, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1], &scratchA)
					convolve2D8ClampedNEONWithScratch(gotNEON, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1], &scratchB)
					convolve2D8ClampedPureGo(want, ref, 0, 0, refX, refY, w, h, kernels[0], kernels[1])
					for i := range want.Pix {
						if gotI8MM.Pix[i] != want.Pix[i] {
							t.Fatalf("i8mm %dx%d ref=(%d,%d) sample=%d got=%d want=%d", w, h, refX, refY, i, gotI8MM.Pix[i], want.Pix[i])
						}
						if gotNEON.Pix[i] != want.Pix[i] {
							t.Fatalf("neon %dx%d ref=(%d,%d) sample=%d got=%d want=%d", w, h, refX, refY, i, gotNEON.Pix[i], want.Pix[i])
						}
					}
				}
			}
		}
	}
}

// TestCompoundConvBuf2DEmuEdgeMatchesPureGo pins the emu_edge fast path for
// the compound conv-buf 2D kernels byte-exact against the clamped pure-Go
// reference, through both the I8MM and NEON entries.
func TestCompoundConvBuf2DEmuEdgeMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xc03b))
	widths := []int{4, 8, 16, 32, 64, 128}
	heights := []int{4, 8, 16, 32}
	phasePairs := [][2][filterTaps]int16{
		{subpelFilters8[3], subpelFilters8[5]},
		{subpelFilters8Smooth[6], subpelFilters8Smooth[11]},
		{subpelFilters8Sharp[9], subpelFilters8Sharp[13]},
		{bilinearFilters[7], bilinearFilters[2]},
	}
	const offsetBits = 19
	const refW, refH = 96, 72
	ref, _ := testPlane(refW, refH, 1, refW+3)
	for i := range ref.Pix {
		ref.Pix[i] = byte(rng.Intn(256))
	}
	for _, w := range widths {
		for _, h := range heights {
			for _, kernels := range phasePairs {
				for _, at := range emuEdgeTestGeometries(rng, refW, refH, w, h) {
					refX, refY := at[0], at[1]
					gotI8MM := make([]uint16, w*h)
					gotNEON := make([]uint16, w*h)
					want := make([]uint16, w*h)
					var scratchA, scratchB CompoundConvolveScratch
					predictInterCompoundRef8ToConvBuf2DI8MM(gotI8MM, ref, refX, refY, w, h, kernels[0], kernels[1], offsetBits, &scratchA)
					predictInterCompoundRef8ToConvBuf2DNEON(gotNEON, ref, refX, refY, w, h, kernels[0], kernels[1], offsetBits, &scratchB)
					predictInterCompoundRef8ToConvBuf2DPureGo(want, ref, refX, refY, w, h, kernels[0], kernels[1], offsetBits, nil)
					for i := range want {
						if gotI8MM[i] != want[i] {
							t.Fatalf("i8mm %dx%d ref=(%d,%d) sample=%d got=%d want=%d", w, h, refX, refY, i, gotI8MM[i], want[i])
						}
						if gotNEON[i] != want[i] {
							t.Fatalf("neon %dx%d ref=(%d,%d) sample=%d got=%d want=%d", w, h, refX, refY, i, gotNEON[i], want[i])
						}
					}
				}
			}
		}
	}
}

// TestConvolve1D8ClampedEmuEdgeMatchesPureGo pins the emu_edge fast path for
// the 1D clamped scratch wrappers byte-exact against the per-tap clamped
// pure-Go reference.
func TestConvolve1D8ClampedEmuEdgeMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x1d3e))
	widths := []int{4, 8, 16, 32, 64, 128}
	heights := []int{4, 8, 16, 32}
	kernels := [][filterTaps]int16{
		subpelFilters8[3],
		subpelFilters8Smooth[6],
		subpelFilters8Sharp[9],
		bilinearFilters[7],
	}
	const refW, refH = 96, 72
	ref, _ := testPlane(refW, refH, 1, refW+5)
	for i := range ref.Pix {
		ref.Pix[i] = byte(rng.Intn(256))
	}
	for _, w := range widths {
		for _, h := range heights {
			for _, kernel := range kernels {
				for _, at := range emuEdgeTestGeometries(rng, refW, refH, w, h) {
					refX, refY := at[0], at[1]
					gotX, _ := testPlane(w, h, 1, w)
					gotY, _ := testPlane(w, h, 1, w)
					wantX, _ := testPlane(w, h, 1, w)
					wantY, _ := testPlane(w, h, 1, w)
					var scratchX, scratchY ConvolveScratch
					convolveX8ClampedNEONWithScratch(gotX, ref, 0, 0, refX, refY, w, h, kernel, &scratchX)
					convolveY8ClampedNEONWithScratch(gotY, ref, 0, 0, refX, refY, w, h, kernel, &scratchY)
					convolveX8ClampedPureGo(wantX, ref, 0, 0, refX, refY, w, h, kernel)
					convolveY8ClampedPureGo(wantY, ref, 0, 0, refX, refY, w, h, kernel)
					for i := range wantX.Pix {
						if gotX.Pix[i] != wantX.Pix[i] {
							t.Fatalf("x %dx%d ref=(%d,%d) sample=%d got=%d want=%d", w, h, refX, refY, i, gotX.Pix[i], wantX.Pix[i])
						}
						if gotY.Pix[i] != wantY.Pix[i] {
							t.Fatalf("y %dx%d ref=(%d,%d) sample=%d got=%d want=%d", w, h, refX, refY, i, gotY.Pix[i], wantY.Pix[i])
						}
					}
				}
			}
		}
	}
}

// TestConvolve2D8ClampedEmuEdgeZeroAlloc proves the emu_edge fast path stays
// allocation-free (the window plane aliases the caller-owned scratch).
func TestConvolve2D8ClampedEmuEdgeZeroAlloc(t *testing.T) {
	const refW, refH = 64, 64
	ref, _ := testPlane(refW, refH, 1, refW)
	fillMotionTestPlane(ref)
	dst, _ := testPlane(32, 32, 1, 32)
	xKernel := subpelFilters8[3]
	yKernel := subpelFilters8[5]
	var scratch ConvolveScratch
	var compound CompoundConvolveScratch
	out := make([]uint16, 32*32)
	allocs := testing.AllocsPerRun(50, func() {
		convolve2D8ClampedI8MMWithScratch(dst, ref, 0, 0, -5, -9, 32, 32, xKernel, yKernel, &scratch)
		convolve2D8ClampedNEONWithScratch(dst, ref, 0, 0, refW-3, refH-2, 32, 32, xKernel, yKernel, &scratch)
		predictInterCompoundRef8ToConvBuf2DI8MM(out, ref, -5, refH-2, 32, 32, xKernel, yKernel, 19, &compound)
		predictInterCompoundRef8ToConvBuf2DNEON(out, ref, refW-3, -9, 32, 32, xKernel, yKernel, 19, &compound)
	})
	if allocs != 0 {
		t.Fatalf("emu_edge path allocates: %v allocs/run", allocs)
	}
}
