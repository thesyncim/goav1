package motion

import (
	"fmt"
	"testing"
)

// BenchmarkHBDAB measures the public prediction and blending entries used by
// the decoder. The same file can be supplied to the assembly worktree through
// go test's overlay flag for an identical A/B workload.
func BenchmarkHBDAB(b *testing.B) {
	shapes := [][2]int{{4, 4}, {4, 8}, {8, 8}, {8, 16}, {16, 16}, {32, 32}, {64, 64}, {128, 128}}
	for _, bd := range []uint8{10, 12} {
		for _, shape := range shapes {
			w, h := shape[0], shape[1]
			side := w
			if h > side {
				side = h
			}
			dst, ref := benchPlanes(side, int(bd))
			name := fmt.Sprintf("%d/%dx%d", bd, w, h)
			for _, op := range []struct {
				name string
				subX int
				subY int
			}{
				{"X", 6, 0}, {"Y", 0, 10}, {"2D", 6, 10},
			} {
				var scratch ConvolveScratch
				b.Run("Single/"+op.name+"/"+name, func(b *testing.B) {
					b.ReportAllocs()
					for b.Loop() {
						if err := PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSizeScratch(dst, ref, 2, bd, 0, 0, filterTaps, filterTaps, w, h, w, h, op.subX, op.subY, RegularFilters, &scratch); err != nil {
							b.Fatal(err)
						}
					}
				})
				var buf CompoundConvBuf
				var compoundScratch CompoundConvolveScratch
				b.Run("Compound/"+op.name+"/"+name, func(b *testing.B) {
					b.ReportAllocs()
					for b.Loop() {
						if err := PredictInterCompoundRefToConvBufWithScratch(&buf, ref, 2, bd, filterTaps, filterTaps, w, h, op.subX, op.subY, RegularFilters, &compoundScratch); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
			var buf0, buf1 CompoundConvBuf
			var scratch CompoundConvolveScratch
			if err := PredictInterCompoundRefToConvBufWithScratch(&buf0, ref, 2, bd, filterTaps, filterTaps, w, h, 0, 0, RegularFilters, &scratch); err != nil {
				b.Fatal(err)
			}
			buf1 = buf0
			b.Run("Compound/Copy/"+name, func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if err := PredictInterCompoundRefToConvBufWithScratch(&buf0, ref, 2, bd, filterTaps, filterTaps, w, h, 0, 0, RegularFilters, &scratch); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("Blend/Avg/"+name, func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if err := BlendCompoundAvg(dst, &buf0, &buf1, 2, bd, 0, 0, w, h, 8, 8); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkHBDABSharp exercises the full eight-tap filters, whose endpoints
// are nonzero at these phases. Regular filters above exercise four and six
// nonzero taps according to block width.
func BenchmarkHBDABSharp(b *testing.B) {
	sharp := InterpFilters{X: InterpMultiTapSharp, Y: InterpMultiTapSharp}
	for _, bd := range []uint8{10, 12} {
		for _, side := range []int{8, 16, 32, 64} {
			dst, ref := benchPlanes(side, int(bd))
			name := fmt.Sprintf("%d/%dx%d", bd, side, side)
			for _, op := range []struct {
				name       string
				subX, subY int
			}{{"X", 6, 0}, {"Y", 0, 10}, {"2D", 6, 10}} {
				var scratch ConvolveScratch
				b.Run("Single/"+op.name+"/"+name, func(b *testing.B) {
					b.ReportAllocs()
					for b.Loop() {
						if err := PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSizeScratch(dst, ref, 2, bd, 0, 0, filterTaps, filterTaps, side, side, side, side, op.subX, op.subY, sharp, &scratch); err != nil {
							b.Fatal(err)
						}
					}
				})
				var buf CompoundConvBuf
				var compoundScratch CompoundConvolveScratch
				b.Run("Compound/"+op.name+"/"+name, func(b *testing.B) {
					b.ReportAllocs()
					for b.Loop() {
						if err := PredictInterCompoundRefToConvBufWithScratch(&buf, ref, 2, bd, filterTaps, filterTaps, side, side, op.subX, op.subY, sharp, &compoundScratch); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}
