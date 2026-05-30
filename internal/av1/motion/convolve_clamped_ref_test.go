package motion

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// This file pins the per-tap-clamping pure-Go convolve reference (the spec
// oracle, identical to the original convolve*ClampedPureGo before the
// edge/interior split optimization) and asserts the production clamped
// functions stay bit-identical to it across the width/tap/edge matrix.

func refConvolveX8Clamped(dst frame.Plane, ref frame.Plane, dstX, dstY, refX, refY, width, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	for y := 0; y < height; y++ {
		sy := refY + y
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		for x := 0; x < width; x++ {
			sx := refX + x - fo
			sum := k0*int(loadSample8Clamped(ref, sx, sy)) +
				k1*int(loadSample8Clamped(ref, sx+1, sy)) +
				k2*int(loadSample8Clamped(ref, sx+2, sy)) +
				k3*int(loadSample8Clamped(ref, sx+3, sy)) +
				k4*int(loadSample8Clamped(ref, sx+4, sy)) +
				k5*int(loadSample8Clamped(ref, sx+5, sy)) +
				k6*int(loadSample8Clamped(ref, sx+6, sy)) +
				k7*int(loadSample8Clamped(ref, sx+7, sy))
			res := roundPowerOfTwo(sum, round0Bits)
			dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
		}
	}
}

func refConvolveY8Clamped(dst frame.Plane, ref frame.Plane, dstX, dstY, refX, refY, width, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	for y := 0; y < height; y++ {
		sy := refY + y - fo
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		for x := 0; x < width; x++ {
			sx := refX + x
			sum := k0*int(loadSample8Clamped(ref, sx, sy)) +
				k1*int(loadSample8Clamped(ref, sx, sy+1)) +
				k2*int(loadSample8Clamped(ref, sx, sy+2)) +
				k3*int(loadSample8Clamped(ref, sx, sy+3)) +
				k4*int(loadSample8Clamped(ref, sx, sy+4)) +
				k5*int(loadSample8Clamped(ref, sx, sy+5)) +
				k6*int(loadSample8Clamped(ref, sx, sy+6)) +
				k7*int(loadSample8Clamped(ref, sx, sy+7))
			dstRow[x] = byte(clipPixel(roundPowerOfTwo(sum, filterBits)))
		}
	}
}

func refConvolve2D8Clamped(dst frame.Plane, ref frame.Plane, dstX, dstY, refX, refY, width, height int, xKernel, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int16
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	const xBias = 1 << (8 + filterBits - 1)
	xk0, xk1, xk2, xk3, xk4, xk5, xk6, xk7 := int(xKernel[0]), int(xKernel[1]), int(xKernel[2]), int(xKernel[3]), int(xKernel[4]), int(xKernel[5]), int(xKernel[6]), int(xKernel[7])
	for y := 0; y < imH; y++ {
		sy := refY - foY + y
		imRow := im[y*imStride:]
		for x := 0; x < width; x++ {
			sx := refX + x - foX
			sum := xBias +
				xk0*int(loadSample8Clamped(ref, sx, sy)) +
				xk1*int(loadSample8Clamped(ref, sx+1, sy)) +
				xk2*int(loadSample8Clamped(ref, sx+2, sy)) +
				xk3*int(loadSample8Clamped(ref, sx+3, sy)) +
				xk4*int(loadSample8Clamped(ref, sx+4, sy)) +
				xk5*int(loadSample8Clamped(ref, sx+5, sy)) +
				xk6*int(loadSample8Clamped(ref, sx+6, sy)) +
				xk7*int(loadSample8Clamped(ref, sx+7, sy))
			imRow[x] = int16(roundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := 8 + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	yBias := 1 << offsetBits
	yk0, yk1, yk2, yk3, yk4, yk5, yk6, yk7 := int(yKernel[0]), int(yKernel[1]), int(yKernel[2]), int(yKernel[3]), int(yKernel[4]), int(yKernel[5]), int(yKernel[6]), int(yKernel[7])
	for y := 0; y < height; y++ {
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		col := im[y*imStride : (y+7)*imStride+width]
		for x := 0; x < width; x++ {
			sum := yBias + yk0*int(col[x]) + yk1*int(col[x+imStride]) + yk2*int(col[x+2*imStride]) +
				yk3*int(col[x+3*imStride]) + yk4*int(col[x+4*imStride]) + yk5*int(col[x+5*imStride]) +
				yk6*int(col[x+6*imStride]) + yk7*int(col[x+7*imStride])
			res := roundPowerOfTwo(sum, round1Bits) - roundOffset
			dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, bits)))
		}
	}
}

// clampedRefCase describes one block placement against the reference plane.
type clampedRefCase struct {
	name               string
	refX, refY         int
	width, height      int
	fourTapX, fourTapY bool
}

func TestConvolveClampedMatchesPerTapReference(t *testing.T) {
	const planeW, planeH = 48, 48
	ref, _ := testPlane(planeW, planeH, 1, planeW)
	fillMotionTestPlane(ref)

	// Subpel index 3 (8-tap regular) and 5 (8-tap, different phase); 4-tap from
	// subpelFilters4. The clamped functions branch on whether the kernel has
	// zero end taps, so cover both shapes.
	xk8 := subpelFilters8[3]
	yk8 := subpelFilters8[5]
	xk4 := subpelFilters4[3]
	yk4 := subpelFilters4[5]

	pick := func(four bool, eight, fourK [filterTaps]int16) [filterTaps]int16 {
		if four {
			return fourK
		}
		return eight
	}

	cases := []clampedRefCase{
		// Interior (fully resident halo, tap window inside the plane).
		{"interior_16x16_8tap", 12, 12, 16, 16, false, false},
		{"interior_8x8_8tap", 16, 16, 8, 8, false, false},
		{"interior_4x4_8tap", 20, 20, 4, 4, false, false},
		{"interior_4x16_4tapX_8tapY", 20, 12, 4, 16, true, false},
		{"interior_16x16_4tap", 12, 12, 16, 16, true, true},
		// Top-left corner (negative tap coordinates -> clamp active).
		{"corner_16x16_8tap", 0, 0, 16, 16, false, false},
		{"corner_8x8_8tap", 0, 0, 8, 8, false, false},
		{"corner_4x4_8tap", 0, 0, 4, 4, false, false},
		{"corner_4x16_4tap", 0, 0, 4, 16, true, true},
		// Left edge only / top edge only.
		{"left_8x8_8tap", 0, 20, 8, 8, false, false},
		{"top_8x8_8tap", 20, 0, 8, 8, false, false},
		// Bottom-right corner (tap coordinates beyond Width/Height).
		{"br_16x16_8tap", planeW - 16, planeH - 16, 16, 16, false, false},
		{"br_8x8_8tap", planeW - 8, planeH - 8, 8, 8, false, false},
		{"br_4x4_4tap", planeW - 4, planeH - 4, 4, 4, true, true},
		// Block straddling the right edge so part of every row needs clamping.
		{"right_straddle_8x8_8tap", planeW - 4, 16, 8, 8, false, false},
		{"bottom_straddle_8x8_8tap", 16, planeH - 4, 8, 8, false, false},
	}

	for _, c := range cases {
		xk := pick(c.fourTapX, xk8, xk4)
		yk := pick(c.fourTapY, yk8, yk4)

		t.Run("2D/"+c.name, func(t *testing.T) {
			got, _ := testPlane(c.width, c.height, 1, c.width)
			want, _ := testPlane(c.width, c.height, 1, c.width)
			convolve2D8ClampedPureGo(got, ref, 0, 0, c.refX, c.refY, c.width, c.height, xk, yk)
			refConvolve2D8Clamped(want, ref, 0, 0, c.refX, c.refY, c.width, c.height, xk, yk)
			assertClampedEqual(t, got, want, c.width, c.height)
		})
		t.Run("X/"+c.name, func(t *testing.T) {
			got, _ := testPlane(c.width, c.height, 1, c.width)
			want, _ := testPlane(c.width, c.height, 1, c.width)
			convolveX8ClampedPureGo(got, ref, 0, 0, c.refX, c.refY, c.width, c.height, xk)
			refConvolveX8Clamped(want, ref, 0, 0, c.refX, c.refY, c.width, c.height, xk)
			assertClampedEqual(t, got, want, c.width, c.height)
		})
		t.Run("Y/"+c.name, func(t *testing.T) {
			got, _ := testPlane(c.width, c.height, 1, c.width)
			want, _ := testPlane(c.width, c.height, 1, c.width)
			convolveY8ClampedPureGo(got, ref, 0, 0, c.refX, c.refY, c.width, c.height, yk)
			refConvolveY8Clamped(want, ref, 0, 0, c.refX, c.refY, c.width, c.height, yk)
			assertClampedEqual(t, got, want, c.width, c.height)
		})
	}
}

func assertClampedEqual(t *testing.T, got, want frame.Plane, width, height int) {
	t.Helper()
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			g := got.Pix[y*got.Stride+x]
			w := want.Pix[y*want.Stride+x]
			if g != w {
				t.Fatalf("mismatch at (%d,%d): got %d want %d", x, y, g, w)
			}
		}
	}
}
