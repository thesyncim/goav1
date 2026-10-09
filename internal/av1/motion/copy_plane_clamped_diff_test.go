// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package motion

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// copyPlaneBlockClampedNaive is the pre-optimization per-pixel-clamp reference
// (emu_edge), kept here as the differential oracle for the region-split fast
// path in copyPlaneBlockClamped.
func copyPlaneBlockClampedNaive(dst frame.Plane, ref frame.Plane, bps, dstX, dstY, refX, refY, width, height int) {
	for y := range height {
		sy := clampInt(refY+y, 0, ref.Height-1)
		for x := range width {
			sx := clampInt(refX+x, 0, ref.Width-1)
			switch bps {
			case 1:
				dst.Pix[(dstY+y)*dst.Stride+dstX+x] = ref.Pix[sy*ref.Stride+sx]
			case 2:
				do := (dstY+y)*dst.Stride + (dstX+x)*2
				ro := sy*ref.Stride + sx*2
				dst.Pix[do] = ref.Pix[ro]
				dst.Pix[do+1] = ref.Pix[ro+1]
			}
		}
	}
}

func makeRefPlane(w, h, bps int) frame.Plane {
	stride := w * bps
	pix := make([]byte, stride*h)
	for i := range pix {
		// Deterministic non-uniform content so any region mismatch shows up.
		pix[i] = byte((i*37 + 11) & 0xff)
	}
	return frame.Plane{Pix: pix, Stride: stride, Width: w, Height: h}
}

func TestCopyPlaneBlockClampedMatchesNaive(t *testing.T) {
	refDims := [][2]int{{1, 1}, {2, 3}, {8, 8}, {16, 9}, {13, 7}}
	blocks := [][2]int{{1, 1}, {4, 4}, {8, 8}, {5, 3}, {16, 16}}
	// Offsets span fully-left, straddle-left, in-bounds, straddle-right,
	// fully-right, and the exact-edge cases in both axes.
	offs := []int{-20, -9, -3, -1, 0, 1, 3, 7, 12, 20}
	for _, bps := range []int{1, 2} {
		for _, rd := range refDims {
			ref := makeRefPlane(rd[0], rd[1], bps)
			for _, blk := range blocks {
				w, h := blk[0], blk[1]
				dstStride := (w + 4) * bps
				got := frame.Plane{Pix: make([]byte, dstStride*(h+2)), Stride: dstStride, Width: w + 4, Height: h + 2}
				want := frame.Plane{Pix: make([]byte, dstStride*(h+2)), Stride: dstStride, Width: w + 4, Height: h + 2}
				for _, rx := range offs {
					for _, ry := range offs {
						for i := range got.Pix {
							got.Pix[i] = 0xAA
							want.Pix[i] = 0xAA
						}
						copyPlaneBlockClampedNaive(want, ref, bps, 1, 1, rx, ry, w, h)
						if err := copyPlaneBlockClamped(got, ref, bps, 1, 1, rx, ry, w, h); err != nil {
							t.Fatalf("bps=%d ref=%dx%d blk=%dx%d rx=%d ry=%d: err %v", bps, rd[0], rd[1], w, h, rx, ry, err)
						}
						for i := range want.Pix {
							if got.Pix[i] != want.Pix[i] {
								t.Fatalf("bps=%d ref=%dx%d blk=%dx%d rx=%d ry=%d: byte %d got %d want %d",
									bps, rd[0], rd[1], w, h, rx, ry, i, got.Pix[i], want.Pix[i])
							}
						}
					}
				}
			}
		}
	}
}
