// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package motion

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func TestConvolve8HalfExtremesAndExactWindows(t *testing.T) {
	for _, tbl := range avx2FilterTables() {
		for phase := 0; phase < 16; phase++ {
			k := tbl[phase]
			for _, shape := range [][2]int{{4, 4}, {8, 16}, {16, 8}, {32, 32}, {128, 128}} {
				w, h := shape[0], shape[1]
				for pattern := 0; pattern < 3; pattern++ {
					stride := w + 9 // odd stride, with seven horizontal tap samples
					ref := frame.Plane{Pix: make([]byte, (h+6)*stride+w+7), Stride: stride, Width: w + 7, Height: h + 7}
					for y := 0; y < ref.Height; y++ {
						for x := 0; x < ref.Width; x++ {
							if pattern == 1 || pattern == 2 && (x+y)&1 != 0 {
								ref.Pix[y*stride+x] = 255
							}
						}
					}
					mkDst := func() frame.Plane {
						return frame.Plane{Pix: make([]byte, (h-1)*(w+1)+w), Stride: w + 1, Width: w, Height: h}
					}
					var scratch ConvolveScratch
					for axis := 0; axis < 3; axis++ {
						got, want := mkDst(), mkDst()
						switch axis {
						case 0:
							convolveX8GoSIMD(got, ref, 0, 0, 3, 3, w, h, k)
							convolveX8PureGo(want, ref, 0, 0, 3, 3, w, h, k)
						case 1:
							convolveY8GoSIMD(got, ref, 0, 0, 3, 3, w, h, k)
							convolveY8PureGo(want, ref, 0, 0, 3, 3, w, h, k)
						default:
							convolve2D8GoSIMDWithScratch(got, ref, 0, 0, 3, 3, w, h, k, k, &scratch)
							convolve2D8PureGo(want, ref, 0, 0, 3, 3, w, h, k, k)
						}
						for y := 0; y < h; y++ {
							for x := 0; x < w; x++ {
								idx := y*(w+1) + x
								if got.Pix[idx] != want.Pix[idx] {
									t.Fatalf("axis=%d phase=%d shape=%dx%d pattern=%d pixel=(%d,%d): got=%d want=%d", axis, phase, w, h, pattern, x, y, got.Pix[idx], want.Pix[idx])
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestConvolve8HalfRejectsUnsafeTaps(t *testing.T) {
	for _, k := range [][filterTaps]int16{
		{1, 0, 0, 0, 0, 0, 0, 0},
		{128, 128, 128, 128, 128, 128, 128, 128},
		{-128, -128, -128, -128, -128, -128, -128, -128},
	} {
		if _, ok := u8HalfTaps(&k, 8194); ok {
			t.Fatalf("unsafe taps accepted: %v", k)
		}
	}
}
