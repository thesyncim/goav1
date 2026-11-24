// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package motion

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// TestConvolveScale2D8Step2MatchesScalar checks the 2:1 phase-split kernel
// against the scalar scaled convolve for every filter, block shape and
// sub-pel start phase.
func TestConvolveScale2D8Step2MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const refW, refH = 200, 180
	ref := frame.Plane{Width: refW, Height: refH, Stride: refW + 16, Pix: make([]byte, (refW+16)*refH+64)}
	for i := range ref.Pix {
		ref.Pix[i] = byte(rng.Intn(256))
	}
	sizes := []int{1, 2, 3, 4, 7, 8, 12, 15, 16, 24, 31, 32, 48, 63, 64, 96, 127, 128}
	var im, imRef scaledIM
	checked := 0
	for _, fx := range []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpMultiTapSharp, InterpBilinear} {
		for _, fy := range []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpMultiTapSharp, InterpBilinear} {
			for _, w := range sizes {
				for _, h := range sizes {
					xTable, err := SubpelKernelTableFor(fx, w)
					if err != nil {
						t.Fatal(err)
					}
					yTable, err := SubpelKernelTableFor(fy, h)
					if err != nil {
						t.Fatal(err)
					}
					for trial := 0; trial < 3; trial++ {
						startX := int64(3<<ScaleSubpelBits) + rng.Int63n(int64(refW-w/2-12)<<ScaleSubpelBits)
						startY := int64(3<<ScaleSubpelBits) + rng.Int63n(int64(refH-h/2-12)<<ScaleSubpelBits)
						imH, ok := scaledIMHeight(h, startY, scaledStep2Half)
						if !ok || !scaledRefRegionFits(ref, w, imH, startX, scaledStep2Half, startY) {
							continue
						}
						got := frame.Plane{Width: w, Height: h, Stride: w + 5, Pix: make([]byte, (w+5)*h)}
						want := frame.Plane{Width: w, Height: h, Stride: w + 5, Pix: make([]byte, (w+5)*h)}
						if !convolveScale2D8Step2GoSIMD(got, ref, 0, 0, w, h, startX, startY, xTable, yTable, imH, &im) {
							continue
						}
						convolveScale2D8(want, ref, 0, 0, w, h, startX, scaledStep2Half, startY, scaledStep2Half, xTable, yTable, imH, &imRef)
						if !bytes.Equal(got.Pix, want.Pix) {
							t.Fatalf("filters %d/%d %dx%d start %d,%d: mismatch", fx, fy, w, h, startX, startY)
						}
						checked++
					}
				}
			}
		}
	}
	if checked < 1000 {
		t.Fatalf("checked only %d cases", checked)
	}
	t.Logf("checked %d cases", checked)
}

// BenchmarkConvolveScale2D8Step2 measures the 2:1 phase-split scaled kernel.
func BenchmarkConvolveScale2D8Step2(b *testing.B) {
	rng := rand.New(rand.NewSource(2))
	const refW, refH = 200, 180
	ref := frame.Plane{Width: refW, Height: refH, Stride: refW + 16, Pix: make([]byte, (refW+16)*refH+64)}
	for i := range ref.Pix {
		ref.Pix[i] = byte(rng.Intn(256))
	}
	var im scaledIM
	for _, w := range []int{8, 16, 32} {
		xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, w)
		yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, w)
		startX := int64(20<<ScaleSubpelBits) + 37
		startY := int64(20<<ScaleSubpelBits) + 91
		imH, ok := scaledIMHeight(w, startY, scaledStep2Half)
		if !ok {
			b.Fatal("scaledIMHeight declined")
		}
		dst := frame.Plane{Width: w, Height: w, Stride: w, Pix: make([]byte, w*w)}
		if !convolveScale2D8Step2GoSIMD(dst, ref, 0, 0, w, w, startX, startY, xTable, yTable, imH, &im) {
			b.Fatal("convolveScale2D8Step2GoSIMD declined")
		}
		b.Run(fmt.Sprintf("%dx%d", w, w), func(b *testing.B) {
			for b.Loop() {
				convolveScale2D8Step2GoSIMD(dst, ref, 0, 0, w, w, startX, startY, xTable, yTable, imH, &im)
			}
		})
	}
}
