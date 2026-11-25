package motion

import (
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"testing"
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

// 2D 8-tap convolve, 32x32 block using caller-owned scratch.
func BenchmarkConvolve2D8WithScratch_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	var scratch ConvolveScratch
	runConvolveBench(b, 32, 32, func() {
		convolve2D8WithScratchImpl(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk, yk, &scratch)
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

func BenchmarkConvolve2D8WithScratch_4tap_4(b *testing.B) {
	dst, ref := benchPlanes(4, 8)
	xk := subpelFilters4[3]
	yk := subpelFilters4[5]
	var scratch ConvolveScratch
	runConvolveBench(b, 4, 4, func() {
		convolve2D8WithScratchImpl(dst, ref, 0, 0, filterTaps, filterTaps, 4, 4, xk, yk, &scratch)
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

func BenchmarkConvolve2D8WithScratch_mixed_4x16(b *testing.B) {
	dst, ref := benchPlanes(16, 8)
	xk := subpelFilters4[3]
	yk := subpelFilters8[5]
	var scratch ConvolveScratch
	runConvolveBench(b, 4, 16, func() {
		convolve2D8WithScratchImpl(dst, ref, 0, 0, filterTaps, filterTaps, 4, 16, xk, yk, &scratch)
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

func BenchmarkConvolveX8_4tap_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters4[3]
	runConvolveBench(b, 32, 32, func() {
		convolveX8Impl(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk)
	})
}

func BenchmarkConvolveY8_4tap_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	yk := subpelFilters4[5]
	runConvolveBench(b, 32, 32, func() {
		convolveY8Impl(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, yk)
	})
}

func BenchmarkConvolveX8ClampedEdge_16(b *testing.B) {
	const plane = 64
	ref, _ := testPlane(plane, plane, 1, plane)
	fillMotionTestPlane(ref)
	dst, _ := testPlane(16, 16, 1, 16)
	xk := subpelFilters8[3]
	refX := plane - 13
	refY := 20
	runConvolveBench(b, 16, 16, func() {
		convolveX8ClampedImpl(dst, ref, 0, 0, refX, refY, 16, 16, xk)
	})
}

func BenchmarkConvolveY8ClampedEdge_16(b *testing.B) {
	const plane = 64
	ref, _ := testPlane(plane, plane, 1, plane)
	fillMotionTestPlane(ref)
	dst, _ := testPlane(16, 16, 1, 16)
	yk := subpelFilters8[5]
	refX := 20
	refY := plane - 13
	runConvolveBench(b, 16, 16, func() {
		convolveY8ClampedImpl(dst, ref, 0, 0, refX, refY, 16, 16, yk)
	})
}

func BenchmarkCompoundConvBuf2D8_32(b *testing.B) {
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	var scratch CompoundConvolveScratch
	runConvolveBench(b, 32, 32, func() {
		if err := PredictInterCompoundRefToConvBuf(&buf, ref, 1, 8, filterTaps, filterTaps, 32, 32, 32, 32, 3, 5, RegularFilters, &scratch); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkCompoundConvBuf2D8_8(b *testing.B) {
	_, ref := benchPlanes(8, 8)
	var buf CompoundConvBuf
	var scratch CompoundConvolveScratch
	runConvolveBench(b, 8, 8, func() {
		if err := PredictInterCompoundRefToConvBuf(&buf, ref, 1, 8, filterTaps, filterTaps, 8, 8, 8, 8, 3, 5, RegularFilters, &scratch); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkCompoundConvBuf2D8_4x16(b *testing.B) {
	_, ref := benchPlanes(16, 8)
	var buf CompoundConvBuf
	var scratch CompoundConvolveScratch
	runConvolveBench(b, 4, 16, func() {
		if err := PredictInterCompoundRefToConvBuf(&buf, ref, 1, 8, filterTaps, filterTaps, 4, 16, 4, 16, 3, 5, RegularFilters, &scratch); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkScaledCompoundConvBufIdentity8_32(b *testing.B) {
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	var scratch ScaledConvolveScratch
	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	startX := int64(filterTaps)*ScaleSubpelScale + int64(3<<ScaleExtraBits)
	startY := int64(filterTaps)*ScaleSubpelScale + int64(5<<ScaleExtraBits)
	runConvolveBench(b, 32, 32, func() {
		if err := PredictScaledCompoundRefToConvBufWithScratch(&buf, ref, 1, 8, 32, 32, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkScaledCompoundConvBufIdentity8GenericDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	var scratch ScaledConvolveScratch
	out, ok := compoundConvBufView(&buf, 32, 32)
	if !ok {
		b.Fatal("invalid convbuf")
	}
	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	startX := int64(filterTaps)*ScaleSubpelScale + int64(3<<ScaleExtraBits)
	startY := int64(filterTaps)*ScaleSubpelScale + int64(5<<ScaleExtraBits)
	imH, ok := scaledIMHeight(32, startY, ScaleSubpelScale)
	if !ok {
		b.Fatal("invalid intermediate height")
	}
	runConvolveBench(b, 32, 32, func() {
		predictScaledCompoundRefToConvBufGeneric(out, ref, 1, 8, 32, 32, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, imH, &scratch)
	})
}

func BenchmarkScaledCompoundConvBufYIdentity8_32(b *testing.B) {
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	var scratch ScaledConvolveScratch
	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	startX := int64(filterTaps)*ScaleSubpelScale + ScaleSubpelScale/4
	startY := int64(filterTaps) * ScaleSubpelScale
	xStep := int64(ScaleSubpelScale * 3 / 4)
	runConvolveBench(b, 32, 32, func() {
		if err := PredictScaledCompoundRefToConvBufWithScratch(&buf, ref, 1, 8, 32, 32, startX, xStep, startY, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkScaledCompoundConvBufYIdentity8GenericDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	var scratch ScaledConvolveScratch
	out, ok := compoundConvBufView(&buf, 32, 32)
	if !ok {
		b.Fatal("invalid convbuf")
	}
	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	startX := int64(filterTaps)*ScaleSubpelScale + ScaleSubpelScale/4
	startY := int64(filterTaps) * ScaleSubpelScale
	xStep := int64(ScaleSubpelScale * 3 / 4)
	imH, ok := scaledIMHeight(32, startY, ScaleSubpelScale)
	if !ok {
		b.Fatal("invalid intermediate height")
	}
	runConvolveBench(b, 32, 32, func() {
		predictScaledCompoundRefToConvBufGeneric(out, ref, 1, 8, 32, 32, startX, xStep, startY, ScaleSubpelScale, xTable, yTable, imH, &scratch)
	})
}

func BenchmarkScaledCompoundConvBufIdentityHighBD_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	var buf CompoundConvBuf
	var scratch ScaledConvolveScratch
	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	startX := int64(filterTaps)*ScaleSubpelScale + int64(3<<ScaleExtraBits)
	startY := int64(filterTaps)*ScaleSubpelScale + int64(5<<ScaleExtraBits)
	runConvolveBench(b, 32, 32, func() {
		if err := PredictScaledCompoundRefToConvBufWithScratch(&buf, ref, 2, 10, 32, 32, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkScaledCompoundConvBufIdentityHighBDGenericDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	var buf CompoundConvBuf
	var scratch ScaledConvolveScratch
	out, ok := compoundConvBufView(&buf, 32, 32)
	if !ok {
		b.Fatal("invalid convbuf")
	}
	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	startX := int64(filterTaps)*ScaleSubpelScale + int64(3<<ScaleExtraBits)
	startY := int64(filterTaps)*ScaleSubpelScale + int64(5<<ScaleExtraBits)
	imH, ok := scaledIMHeight(32, startY, ScaleSubpelScale)
	if !ok {
		b.Fatal("invalid intermediate height")
	}
	runConvolveBench(b, 32, 32, func() {
		predictScaledCompoundRefToConvBufGeneric(out, ref, 2, 10, 32, 32, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, imH, &scratch)
	})
}

func BenchmarkScaledCompoundConvBufYIdentityHighBD_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	var buf CompoundConvBuf
	var scratch ScaledConvolveScratch
	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	startX := int64(filterTaps)*ScaleSubpelScale + ScaleSubpelScale/4
	startY := int64(filterTaps) * ScaleSubpelScale
	xStep := int64(ScaleSubpelScale * 3 / 4)
	runConvolveBench(b, 32, 32, func() {
		if err := PredictScaledCompoundRefToConvBufWithScratch(&buf, ref, 2, 10, 32, 32, startX, xStep, startY, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkScaledCompoundConvBufYIdentityHighBDGenericDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	var buf CompoundConvBuf
	var scratch ScaledConvolveScratch
	out, ok := compoundConvBufView(&buf, 32, 32)
	if !ok {
		b.Fatal("invalid convbuf")
	}
	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapRegular, 32)
	if err != nil {
		b.Fatal(err)
	}
	startX := int64(filterTaps)*ScaleSubpelScale + ScaleSubpelScale/4
	startY := int64(filterTaps) * ScaleSubpelScale
	xStep := int64(ScaleSubpelScale * 3 / 4)
	imH, ok := scaledIMHeight(32, startY, ScaleSubpelScale)
	if !ok {
		b.Fatal("invalid intermediate height")
	}
	runConvolveBench(b, 32, 32, func() {
		predictScaledCompoundRefToConvBufGeneric(out, ref, 2, 10, 32, 32, startX, xStep, startY, ScaleSubpelScale, xTable, yTable, imH, &scratch)
	})
}

func BenchmarkCompoundConvBufX8_32(b *testing.B) {
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	runConvolveBench(b, 32, 32, func() {
		if err := PredictInterCompoundRefToConvBuf(&buf, ref, 1, 8, filterTaps, filterTaps, 32, 32, 32, 32, 3, 0, RegularFilters, nil); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkCompoundConvBufX8_4x16(b *testing.B) {
	_, ref := benchPlanes(16, 8)
	var buf CompoundConvBuf
	runConvolveBench(b, 4, 16, func() {
		if err := PredictInterCompoundRefToConvBuf(&buf, ref, 1, 8, filterTaps, filterTaps, 4, 16, 4, 16, 3, 0, RegularFilters, nil); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkCompoundConvBufY8_32(b *testing.B) {
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	runConvolveBench(b, 32, 32, func() {
		if err := PredictInterCompoundRefToConvBuf(&buf, ref, 1, 8, filterTaps, filterTaps, 32, 32, 32, 32, 0, 5, RegularFilters, nil); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkCompoundConvBufY8_4x16(b *testing.B) {
	_, ref := benchPlanes(16, 8)
	var buf CompoundConvBuf
	runConvolveBench(b, 4, 16, func() {
		if err := PredictInterCompoundRefToConvBuf(&buf, ref, 1, 8, filterTaps, filterTaps, 4, 16, 4, 16, 0, 5, RegularFilters, nil); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkCompoundConvBufCopy8_32(b *testing.B) {
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	runConvolveBench(b, 32, 32, func() {
		if err := PredictInterCompoundRefToConvBuf(&buf, ref, 1, 8, filterTaps, filterTaps, 32, 32, 32, 32, 0, 0, RegularFilters, nil); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkCompoundConvBufCopy8_4x16(b *testing.B) {
	_, ref := benchPlanes(16, 8)
	var buf CompoundConvBuf
	runConvolveBench(b, 4, 16, func() {
		if err := PredictInterCompoundRefToConvBuf(&buf, ref, 1, 8, filterTaps, filterTaps, 4, 16, 4, 16, 0, 0, RegularFilters, nil); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkCompoundConvBuf2DHighBD_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	var buf CompoundConvBuf
	var scratch CompoundConvolveScratch
	runConvolveBench(b, 32, 32, func() {
		if err := PredictInterCompoundRefToConvBuf(&buf, ref, 2, 10, filterTaps, filterTaps, 32, 32, 32, 32, 3, 5, RegularFilters, &scratch); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkCompoundConvBuf2DHighBDClamped_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	var buf CompoundConvBuf
	var scratch CompoundConvolveScratch
	runConvolveBench(b, 32, 32, func() {
		if err := PredictInterCompoundRefToConvBuf(&buf, ref, 2, 10, 0, 0, 32, 32, 32, 32, 3, 5, RegularFilters, &scratch); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkCompoundConvBufXHighBD_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	var buf CompoundConvBuf
	runConvolveBench(b, 32, 32, func() {
		if err := PredictInterCompoundRefToConvBuf(&buf, ref, 2, 10, filterTaps, filterTaps, 32, 32, 32, 32, 3, 0, RegularFilters, nil); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkCompoundConvBufYHighBD_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	var buf CompoundConvBuf
	runConvolveBench(b, 32, 32, func() {
		if err := PredictInterCompoundRefToConvBuf(&buf, ref, 2, 10, filterTaps, filterTaps, 32, 32, 32, 32, 0, 5, RegularFilters, nil); err != nil {
			b.Fatal(err)
		}
	})
}

func BenchmarkCompoundConvBufCopyHighBD_32(b *testing.B) {
	_, ref := benchPlanes(32, 10)
	var buf CompoundConvBuf
	runConvolveBench(b, 32, 32, func() {
		if err := PredictInterCompoundRefToConvBuf(&buf, ref, 2, 10, filterTaps, filterTaps, 32, 32, 32, 32, 0, 0, RegularFilters, nil); err != nil {
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

func BenchmarkConvolve2D8ClampedEdge_16WithScratch(b *testing.B) {
	const plane = 64
	ref, _ := testPlane(plane, plane, 1, plane)
	fillMotionTestPlane(ref)
	dst, _ := testPlane(16, 16, 1, 16)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	refX := plane - 13
	refY := 20
	var scratch ConvolveScratch
	runConvolveBench(b, 16, 16, func() {
		convolve2D8ClampedWithScratchImpl(dst, ref, 0, 0, refX, refY, 16, 16, xk, yk, &scratch)
	})
}

// Clamped 2D width-4 edge block (the small inter block that the GoSIMD path
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
						if err := PredictInterCompoundRefToConvBuf(&buf, ref, 2, bd, filterTaps, filterTaps, w, h, w, h, op.subX, op.subY, RegularFilters, &compoundScratch); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
			var buf0, buf1 CompoundConvBuf
			var scratch CompoundConvolveScratch
			if err := PredictInterCompoundRefToConvBuf(&buf0, ref, 2, bd, filterTaps, filterTaps, w, h, w, h, 0, 0, RegularFilters, &scratch); err != nil {
				b.Fatal(err)
			}
			buf1 = buf0
			b.Run("Compound/Copy/"+name, func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if err := PredictInterCompoundRefToConvBuf(&buf0, ref, 2, bd, filterTaps, filterTaps, w, h, w, h, 0, 0, RegularFilters, &scratch); err != nil {
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
						if err := PredictInterCompoundRefToConvBuf(&buf, ref, 2, bd, filterTaps, filterTaps, side, side, side, side, op.subX, op.subY, sharp, &compoundScratch); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

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
						if err := PredictInterCompoundRefToConvBuf(&conv0, ref, 1, 8, 8, 8, w, h, w, h, axis.sx, axis.sy, pair, &compoundScratch); err != nil {
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
			if err := PredictInterCompoundRefToConvBuf(&edgeBuf, edgeRef, 1, 8, 0, 0, 32, 32, 32, 32, 3, 5, regular, &edgeScratch); err != nil {
				b.Fatal(err)
			}
		}
	})
}
