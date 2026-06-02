package motion

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// Benchmarks targeting the individual convolve routines directly so per-routine
// speedups can be measured. The reference plane is padded so the non-clamped
// (fast) path is taken; clamped variants are exercised by placing the block at
// the frame origin with a negative effective offset is not possible here, so a
// dedicated clamped bench drives loadSample8Clamped directly.

func benchPlanes(side, bd int) (dst, ref frame.Plane) {
	bps := 1
	if bd > 8 {
		bps = 2
	}
	pad := filterTaps
	refSide := side + 2*pad
	r, _ := testPlane(refSide, refSide, bps, refSide*bps)
	d, _ := testPlane(side, side, bps, side*bps)
	if bps == 1 {
		fillMotionTestPlane(r)
	} else {
		fillHighBDMotionTestPlane(r, uint16((1<<bd)-1))
	}
	return d, r
}

func runConvolveBench(b *testing.B, w, h int, fn func()) {
	b.ReportAllocs()
	for b.Loop() {
		fn()
	}
}

// 2D 8-tap convolve, 32x32 block (common large luma).
func BenchmarkConvolve2D8_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	runConvolveBench(b, 32, 32, func() {
		convolve2D8Impl(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk, yk)
	})
}

// 2D 8-tap convolve, 8x8 block.
func BenchmarkConvolve2D8_8(b *testing.B) {
	dst, ref := benchPlanes(8, 8)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	runConvolveBench(b, 8, 8, func() {
		convolve2D8Impl(dst, ref, 0, 0, filterTaps, filterTaps, 8, 8, xk, yk)
	})
}

// 2D 4-tap convolve (small block), 4x4.
func BenchmarkConvolve2D8_4tap_4(b *testing.B) {
	dst, ref := benchPlanes(4, 8)
	xk := subpelFilters4[3]
	yk := subpelFilters4[5]
	runConvolveBench(b, 4, 4, func() {
		convolve2D8Impl(dst, ref, 0, 0, filterTaps, filterTaps, 4, 4, xk, yk)
	})
}

// 2D mixed: 4 wide x 16 tall (4-tap X, 8-tap Y) - the common per-axis small case.
func BenchmarkConvolve2D8_mixed_4x16(b *testing.B) {
	dst, ref := benchPlanes(16, 8)
	xk := subpelFilters4[3]
	yk := subpelFilters8[5]
	runConvolveBench(b, 4, 16, func() {
		convolve2D8Impl(dst, ref, 0, 0, filterTaps, filterTaps, 4, 16, xk, yk)
	})
}

// 1D X convolve, 32x32.
func BenchmarkConvolveX8_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters8[3]
	runConvolveBench(b, 32, 32, func() {
		convolveX8Impl(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk)
	})
}

// 1D Y convolve, 32x32.
func BenchmarkConvolveY8_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	yk := subpelFilters8[5]
	runConvolveBench(b, 32, 32, func() {
		convolveY8Impl(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, yk)
	})
}

func BenchmarkCompoundConvBuf2D8_32(b *testing.B) {
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	var scratch CompoundConvolveScratch
	runConvolveBench(b, 32, 32, func() {
		if err := PredictInterCompoundRefToConvBufWithScratch(&buf, ref, 1, 8, filterTaps, filterTaps, 32, 32, 3, 5, RegularFilters, &scratch); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkCompoundConvBuf2D8_8(b *testing.B) {
	_, ref := benchPlanes(8, 8)
	var buf CompoundConvBuf
	var scratch CompoundConvolveScratch
	runConvolveBench(b, 8, 8, func() {
		if err := PredictInterCompoundRefToConvBufWithScratch(&buf, ref, 1, 8, filterTaps, filterTaps, 8, 8, 3, 5, RegularFilters, &scratch); err != nil {
			b.Fatal(err)
		}
	})
}

// Clamped 2D, 16x16 (forces edge-clamped loads).
func BenchmarkConvolve2D8Clamped_16(b *testing.B) {
	dst, _ := testPlane(16, 16, 1, 16)
	ref, _ := testPlane(16, 16, 1, 16)
	fillMotionTestPlane(ref)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	runConvolveBench(b, 16, 16, func() {
		convolve2D8ClampedImpl(dst, ref, 0, 0, 0, 0, 16, 16, xk, yk)
	})
}

// Clamped 2D, 16x16 block straddling the right frame edge on a larger plane:
// most columns are resident (fast interior) and only the rightmost few taps
// fall off the frame. This is the common edge-block shape the optimized
// clamped path targets, unlike the fully-cornered Clamped_16 bench above.
func BenchmarkConvolve2D8ClampedEdge_16(b *testing.B) {
	const plane = 64
	ref, _ := testPlane(plane, plane, 1, plane)
	fillMotionTestPlane(ref)
	dst, _ := testPlane(16, 16, 1, 16)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	// refX places the right edge of the tap window just past the frame.
	refX := plane - 13
	refY := 20
	runConvolveBench(b, 16, 16, func() {
		convolve2D8ClampedImpl(dst, ref, 0, 0, refX, refY, 16, 16, xk, yk)
	})
}

// Clamped 2D width-4 edge block (the small inter block that the NEON path
// always routes to pure-Go).
func BenchmarkConvolve2D8ClampedEdge_4x4(b *testing.B) {
	const plane = 64
	ref, _ := testPlane(plane, plane, 1, plane)
	fillMotionTestPlane(ref)
	dst, _ := testPlane(4, 4, 1, 4)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	runConvolveBench(b, 4, 4, func() {
		convolve2D8ClampedImpl(dst, ref, 0, 0, 0, 0, 4, 4, xk, yk)
	})
}

// High-BD 2D, 32x32 at bd=10.
func BenchmarkConvolve2DHighBD_32(b *testing.B) {
	dst, ref := benchPlanes(32, 10)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	runConvolveBench(b, 32, 32, func() {
		convolve2DHighBDImpl(dst, ref, 10, (1<<10)-1, 0, 0, filterTaps, filterTaps, 32, 32, xk, yk)
	})
}

// High-BD X 1D, 32x32 at bd=10.
func BenchmarkConvolveXHighBD_32(b *testing.B) {
	dst, ref := benchPlanes(32, 10)
	xk := subpelFilters8[3]
	runConvolveBench(b, 32, 32, func() {
		convolveXHighBDImpl(dst, ref, 10, (1<<10)-1, 0, 0, filterTaps, filterTaps, 32, 32, xk)
	})
}

// High-BD Y 1D, 32x32 at bd=10.
func BenchmarkConvolveYHighBD_32(b *testing.B) {
	dst, ref := benchPlanes(32, 10)
	yk := subpelFilters8[5]
	runConvolveBench(b, 32, 32, func() {
		convolveYHighBDImpl(dst, ref, 10, (1<<10)-1, 0, 0, filterTaps, filterTaps, 32, 32, yk)
	})
}
