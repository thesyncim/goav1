// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package motion

import (
	"fmt"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// BenchmarkMotion8ABMatrix uses the exported prediction and blend entry points
// so the same source file can run against both the assembly and Go SIMD trees.
// All setup and scratch allocation stay outside the measured loops.
func BenchmarkMotion8ABMatrix(b *testing.B) {
	shapes := [...][2]int{{4, 4}, {8, 16}, {32, 32}, {128, 128}}
	filters := [...]struct {
		name string
		kind InterpFilter
	}{{"regular", InterpEightTapRegular}, {"sharp", InterpMultiTapSharp}, {"bilinear", InterpBilinear}}
	for _, shape := range shapes {
		w, h := shape[0], shape[1]
		name := fmt.Sprintf("%dx%d", w, h)
		refSide := w
		if h > refSide {
			refSide = h
		}
		refSide += 16
		ref := frame.Plane{Pix: make([]byte, refSide*refSide), Stride: refSide, Width: refSide, Height: refSide}
		for i := range ref.Pix {
			ref.Pix[i] = byte((i*37 + i/refSide*11) & 255)
		}
		dst := frame.Plane{Pix: make([]byte, (h-1)*(w+1)+w), Stride: w + 1, Width: w, Height: h}
		var singleScratch ConvolveScratch
		var compoundScratch CompoundConvolveScratch
		var conv0, conv1 CompoundConvBuf
		conv0.Width, conv0.Height = uint8(w), uint8(h)
		conv1.Width, conv1.Height = uint8(w), uint8(h)
		for i := 0; i < w*h; i++ {
			conv0.Data[i] = uint16(6144 + (i*17)&4095)
			conv1.Data[i] = uint16(6144 + (i*29)&4095)
		}
		for _, filter := range filters {
			pair := InterpFilters{X: filter.kind, Y: filter.kind}
			for _, axis := range [...]struct {
				name string
				sx   int
				sy   int
			}{{"X", 3, 0}, {"Y", 0, 5}, {"2D", 3, 5}} {
				b.Run("Single/"+axis.name+"/"+filter.name+"/"+name, func(b *testing.B) {
					b.ReportAllocs()
					for b.Loop() {
						if err := PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSizeScratch(dst, ref, 1, 8, 0, 0, 8, 8, w, h, w, h, axis.sx, axis.sy, pair, &singleScratch); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
			for _, axis := range [...]struct {
				name string
				sx   int
				sy   int
			}{{"copy", 0, 0}, {"X", 3, 0}, {"Y", 0, 5}, {"2D", 3, 5}} {
				b.Run("Compound/"+axis.name+"/"+filter.name+"/"+name, func(b *testing.B) {
					b.ReportAllocs()
					for b.Loop() {
						if err := PredictInterCompoundRefToConvBufWithScratch(&conv0, ref, 1, 8, 8, 8, w, h, axis.sx, axis.sy, pair, &compoundScratch); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
		b.Run("Blend/"+name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if err := BlendCompoundAvg(dst, &conv0, &conv1, 1, 8, 0, 0, w, h, 9, 7); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
	warpRef := frame.Plane{Pix: make([]byte, 96*96), Stride: 96, Width: 96, Height: 96}
	warpDst := frame.Plane{Pix: make([]byte, 96*96), Stride: 96, Width: 96, Height: 96}
	for y := 0; y < 96; y++ {
		for x := 0; x < 96; x++ {
			warpRef.Pix[y*96+x] = byte((x*17 + y*29 + x*y) & 255)
		}
	}
	warpMatrix := [6]int32{42805, -7571, 65230, -57, 0, 65509}
	for _, tc := range [...]struct {
		name  string
		gamma int16
	}{{"gamma0", 0}, {"gamma96", 96}} {
		b.Run("Warp/"+tc.name+"/64x64", func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if err := PredictWarpedPlaneBlockBitDepth(warpDst, warpRef, 1, 8, 16, 16, 64, 64, warpMatrix, -320, -64, tc.gamma, -64, false, false); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
	edgeRef := frame.Plane{Pix: make([]byte, 32*32), Stride: 32, Width: 32, Height: 32}
	for k := range edgeRef.Pix {
		edgeRef.Pix[k] = byte((k*37 + k/32*11) & 255)
	}
	var edgeBuf CompoundConvBuf
	edgeBuf.Width, edgeBuf.Height = 32, 32
	var edgeScratch CompoundConvolveScratch
	regular := InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}
	b.Run("Compound/2D/edge/32x32", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if err := PredictInterCompoundRefToConvBufWithScratch(&edgeBuf, edgeRef, 1, 8, 0, 0, 32, 32, 3, 5, regular, &edgeScratch); err != nil {
				b.Fatal(err)
			}
		}
	})
}
