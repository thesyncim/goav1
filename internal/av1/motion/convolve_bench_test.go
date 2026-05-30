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

// Clamped 2D, 16x16 (forces edge-clamped loads).
func BenchmarkConvolve2D8Clamped_16(b *testing.B) {
	dst, _ := testPlane(16, 16, 1, 16)
	ref, _ := testPlane(16, 16, 1, 16)
	fillMotionTestPlane(ref)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	runConvolveBench(b, 16, 16, func() {
		convolve2D8Clamped(dst, ref, 0, 0, 0, 0, 16, 16, xk, yk)
	})
}

// High-BD 2D, 32x32 at bd=10.
func BenchmarkConvolve2DHighBD_32(b *testing.B) {
	dst, ref := benchPlanes(32, 10)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	runConvolveBench(b, 32, 32, func() {
		convolve2DHighBD(dst, ref, 10, (1<<10)-1, 0, 0, filterTaps, filterTaps, 32, 32, xk, yk)
	})
}

// High-BD X 1D, 32x32 at bd=10.
func BenchmarkConvolveXHighBD_32(b *testing.B) {
	dst, ref := benchPlanes(32, 10)
	xk := subpelFilters8[3]
	runConvolveBench(b, 32, 32, func() {
		convolveXHighBD(dst, ref, 10, (1<<10)-1, 0, 0, filterTaps, filterTaps, 32, 32, xk)
	})
}
