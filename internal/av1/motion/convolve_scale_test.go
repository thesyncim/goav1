// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package motion

import (
	"fmt"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// libaomConvolveScale2DRef is a faithful port of libaom's
// av1_convolve_2d_scale_c (av1/common/convolve.c). It takes the same
// post-pos>>10 source pointer convention as libaom: the caller pre-offsets
// the reference plane by (refX0, refY0) integer-pel, and passes only the Q10
// sub-pel fractional component plus per-output Q10 step.
func libaomConvolveScale2DRef(dst frame.Plane, src frame.Plane, refX0, refY0 int,
	dstX, dstY, width, height int, subpelXQN, xStepQN, subpelYQN, yStepQN int,
	xTable, yTable SubpelKernelTable, conv_params_round_0, conv_params_round_1 int) {
	const imStride = maxBlockSize
	var im [scaledIMMaxHeight * maxBlockSize]int16
	imH := (((height-1)*yStepQN + subpelYQN) >> ScaleSubpelBits) + filterTaps
	foHoriz := filterTaps/2 - 1
	foVert := filterTaps/2 - 1
	bd := 8

	// Horizontal pass: src_horiz = src - fo_vert * src_stride
	startRow := refY0 - foVert
	for y := range imH {
		srcRow := startRow + y
		xQN := subpelXQN
		for x := range width {
			xInt := (xQN >> ScaleSubpelBits)
			xFilterIdx := (xQN & ScaleSubpelMask) >> ScaleExtraBits
			kernel := xTable[xFilterIdx]
			sum := 1 << (bd + filterBits - 1)
			for k := range filterTaps {
				col := refX0 + xInt + k - foHoriz
				sum += int(kernel[k]) * int(src.Pix[srcRow*src.Stride+col])
			}
			im[y*imStride+x] = int16(libaomRoundPowerOfTwo(sum, conv_params_round_0))
			xQN += xStepQN
		}
	}

	// Vertical pass: src_vert = im_block + fo_vert * im_stride
	offsetBits := bd + 2*filterBits - conv_params_round_0
	roundOffset := (1 << (offsetBits - conv_params_round_1)) + (1 << (offsetBits - conv_params_round_1 - 1))
	bits := 2*filterBits - conv_params_round_0 - conv_params_round_1
	for x := range width {
		yQN := subpelYQN
		for y := range height {
			yInt := (yQN >> ScaleSubpelBits)
			yFilterIdx := (yQN & ScaleSubpelMask) >> ScaleExtraBits
			kernel := yTable[yFilterIdx]
			sum := 1 << offsetBits
			// src_y = &src_vert[(yQN>>10) * im_stride]; tap k reads
			// src_y[(k - fo_vert) * im_stride] = im_block[(fo_vert + yInt + k - fo_vert) * im_stride] = im[(yInt + k) * im_stride]
			for k := range filterTaps {
				sum += int(kernel[k]) * int(im[(yInt+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(sum, conv_params_round_1) - roundOffset
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(res, bits)))
			yQN += yStepQN
		}
	}
}

func libaomConvolveScale2DHighBDRef(dst frame.Plane, src frame.Plane, bd int, refX0, refY0 int,
	dstX, dstY, width, height int, subpelXQN, xStepQN, subpelYQN, yStepQN int,
	xTable, yTable SubpelKernelTable, conv_params_round_0, conv_params_round_1 int) {
	const imStride = maxBlockSize
	var im [scaledIMMaxHeight * maxBlockSize]int32
	imH := (((height-1)*yStepQN + subpelYQN) >> ScaleSubpelBits) + filterTaps
	foHoriz := filterTaps/2 - 1
	foVert := filterTaps/2 - 1
	max := uint16((1 << bd) - 1)

	startRow := refY0 - foVert
	for y := range imH {
		srcRow := startRow + y
		xQN := subpelXQN
		for x := range width {
			xInt := (xQN >> ScaleSubpelBits)
			xFilterIdx := (xQN & ScaleSubpelMask) >> ScaleExtraBits
			kernel := xTable[xFilterIdx]
			sum := 1 << (bd + filterBits - 1)
			for k := range filterTaps {
				col := refX0 + xInt + k - foHoriz
				sum += int(kernel[k]) * int(getSample(src, 2, col, srcRow))
			}
			im[y*imStride+x] = int32(libaomRoundPowerOfTwo(sum, conv_params_round_0))
			xQN += xStepQN
		}
	}

	offsetBits := bd + 2*filterBits - conv_params_round_0
	roundOffset := (1 << (offsetBits - conv_params_round_1)) + (1 << (offsetBits - conv_params_round_1 - 1))
	bits := 2*filterBits - conv_params_round_0 - conv_params_round_1
	for x := range width {
		yQN := subpelYQN
		for y := range height {
			yInt := (yQN >> ScaleSubpelBits)
			yFilterIdx := (yQN & ScaleSubpelMask) >> ScaleExtraBits
			kernel := yTable[yFilterIdx]
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(kernel[k]) * int(im[(yInt+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(sum, conv_params_round_1) - roundOffset
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, bits), max))
			yQN += yStepQN
		}
	}
}

// scaledTestSourcePlane fills a plane with deterministic pseudo-random bytes
// for libaom-shaped scaled convolver fuzzing.
func scaledTestSourcePlane(width, height, bytesPerSample, seed int) frame.Plane {
	plane, _ := testPlane(width, height, bytesPerSample, width*bytesPerSample)
	state := uint32(0xc0ffee | uint32(seed))
	for i := range plane.Pix {
		state = state*1664525 + 1013904223
		plane.Pix[i] = byte(state >> 24)
	}
	return plane
}

// TestConvolveScale2D8MatchesLibaomIdentity verifies the scaled convolver
// matches the same-size 2D 8-tap convolver under identity scaling. With
// xStep = yStep = ScaleSubpelScale and zero subpel, the result must equal the
// existing convolve2D8 output bit-exact.
func TestConvolveScale2D8MatchesLibaomIdentity(t *testing.T) {
	const w, h = 16, 8
	src := scaledTestSourcePlane(w+filterTaps+8, h+filterTaps+8, 1, 0xabcd)
	got, _ := testPlane(w, h, 1, w)
	want, _ := testPlane(w, h, 1, w)

	// Pre-offset position so the centre tap is at the block origin (8, 8) in
	// the input plane, leaving foX=foY=3 taps of headroom on each edge.
	const refX0, refY0 = 8, 8
	startX := int64(refX0) * ScaleSubpelScale
	startY := int64(refY0) * ScaleSubpelScale

	for _, filter := range []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpMultiTapSharp, InterpBilinear} {
		xTable, err := SubpelKernelTableFor(filter, w)
		if err != nil {
			t.Fatalf("xTable: %v", err)
		}
		yTable, err := SubpelKernelTableFor(filter, h)
		if err != nil {
			t.Fatalf("yTable: %v", err)
		}
		clearPlane(got)
		clearPlane(want)
		if err := ConvolveScale2D8(got, src, 0, 0, w, h, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable); err != nil {
			t.Fatalf("filter %d: ConvolveScale2D8: %v", filter, err)
		}
		// Same-size reference convolver uses subpel=0 → integer-pel only.
		xKern, _ := interpKernel(filter, w, 0)
		yKern, _ := interpKernel(filter, h, 0)
		libaomConvolve2DRef(want, src, refX0, refY0, 0, 0, w, h, xKern, yKern)
		comparePlanes(t, "identity_"+filterName(filter), got, want, w, h, 1)
	}
}

// TestConvolveScale2D8MatchesLibaomReference confirms the new scaled
// convolver matches libaom's av1_convolve_2d_scale_c bit-exact under a range
// of typical scaled-position / step configurations.
func TestConvolveScale2D8MatchesLibaomReference(t *testing.T) {
	const w, h = 16, 16
	const srcPad = 32 // generous padding for tap headroom across scaled steps
	src := scaledTestSourcePlane(w+srcPad, h+srcPad, 1, 0x1234)

	type tc struct {
		name                                         string
		refX0, refY0, subpelX, xStep, subpelY, yStep int
	}
	cases := []tc{
		{"identity", 8, 8, 0, ScaleSubpelScale, 0, ScaleSubpelScale},
		{"halfStep_zeroSubpel", 8, 8, 0, ScaleSubpelScale / 2, 0, ScaleSubpelScale / 2}, // 2x upscale
		{"halfStep_halfSubpel", 8, 8, ScaleSubpelScale / 2, ScaleSubpelScale / 2, ScaleSubpelScale / 2, ScaleSubpelScale / 2},
		{"doubleStep_zeroSubpel", 8, 8, 0, 2 * ScaleSubpelScale, 0, 2 * ScaleSubpelScale}, // 2x downscale
		{"3over4Step_q4subpel", 8, 8, 256, (ScaleSubpelScale * 3) / 4, 192, (ScaleSubpelScale * 3) / 4},
		{"asymmetric", 8, 8, 64, ScaleSubpelScale - 13, 320, ScaleSubpelScale + 47},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			for _, filterX := range []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpMultiTapSharp, InterpBilinear} {
				for _, filterY := range []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpMultiTapSharp, InterpBilinear} {
					xTable, err := SubpelKernelTableFor(filterX, w)
					if err != nil {
						t.Fatalf("xTable: %v", err)
					}
					yTable, err := SubpelKernelTableFor(filterY, h)
					if err != nil {
						t.Fatalf("yTable: %v", err)
					}
					got, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					startX := int64(c.refX0)*ScaleSubpelScale + int64(c.subpelX)
					startY := int64(c.refY0)*ScaleSubpelScale + int64(c.subpelY)
					if err := ConvolveScale2D8(got, src, 0, 0, w, h,
						startX, int64(c.xStep), startY, int64(c.yStep),
						xTable, yTable); err != nil {
						t.Fatalf("%s X=%d Y=%d: %v", c.name, filterX, filterY, err)
					}
					libaomConvolveScale2DRef(want, src, c.refX0, c.refY0, 0, 0, w, h,
						c.subpelX, c.xStep, c.subpelY, c.yStep, xTable, yTable,
						round0Bits, round1Bits)
					comparePlanes(t, c.name+"/"+filterName(filterX)+"/"+filterName(filterY), got, want, w, h, 1)
				}
			}
		})
	}
}

func TestConvolveScale2D8ClampedMatchesLibaomReferenceAtEdge(t *testing.T) {
	const w, h = 8, 8
	src := scaledTestSourcePlane(w, h, 1, 0xfeed)
	// Position the block so the leftmost taps fall outside the source — only
	// the *Clamped* variant should handle this.
	const refX0, refY0 = 0, 0
	startX := int64(refX0) * ScaleSubpelScale
	startY := int64(refY0) * ScaleSubpelScale

	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, w)
	if err != nil {
		t.Fatalf("xTable: %v", err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapRegular, h)
	if err != nil {
		t.Fatalf("yTable: %v", err)
	}
	got, _ := testPlane(w, h, 1, w)

	// Bordered reference for the libaom-shape reference impl: replicate src
	// edges into a padded buffer.
	const pad = 12
	padded, _ := testPlane(w+2*pad, h+2*pad, 1, w+2*pad)
	for y := range h + 2*pad {
		sy := y - pad
		if sy < 0 {
			sy = 0
		} else if sy >= h {
			sy = h - 1
		}
		for x := range w + 2*pad {
			sx := x - pad
			if sx < 0 {
				sx = 0
			} else if sx >= w {
				sx = w - 1
			}
			padded.Pix[y*padded.Stride+x] = src.Pix[sy*src.Stride+sx]
		}
	}

	want, _ := testPlane(w, h, 1, w)
	libaomConvolveScale2DRef(want, padded, pad+refX0, pad+refY0, 0, 0, w, h,
		0, ScaleSubpelScale, 0, ScaleSubpelScale, xTable, yTable, round0Bits, round1Bits)
	if err := ConvolveScale2D8Clamped(got, src, 0, 0, w, h, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable); err != nil {
		t.Fatalf("ConvolveScale2D8Clamped: %v", err)
	}
	comparePlanes(t, "clamped_edge", got, want, w, h, 1)
}

func TestConvolveScale2DHighBDMatchesLibaomReference(t *testing.T) {
	const w, h = 16, 16
	const srcPad = 32
	const bd = 10
	src := highBDPlane(w+srcPad, h+srcPad, 0x9999, bd)

	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, w)
	if err != nil {
		t.Fatalf("xTable: %v", err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapSmooth, h)
	if err != nil {
		t.Fatalf("yTable: %v", err)
	}

	for _, step := range []int{ScaleSubpelScale, ScaleSubpelScale / 2, 2 * ScaleSubpelScale, ScaleSubpelScale - 17} {
		got, _ := testPlane(w, h, 2, w*2)
		want, _ := testPlane(w, h, 2, w*2)
		const refX0, refY0 = 8, 8
		const subpelX, subpelY = 31, 47
		startX := int64(refX0)*ScaleSubpelScale + int64(subpelX)
		startY := int64(refY0)*ScaleSubpelScale + int64(subpelY)
		if err := ConvolveScale2DHighBD(got, src, bd, 0, 0, w, h,
			startX, int64(step), startY, int64(step), xTable, yTable); err != nil {
			t.Fatalf("step=%d: %v", step, err)
		}
		libaomConvolveScale2DHighBDRef(want, src, bd, refX0, refY0, 0, 0, w, h,
			subpelX, step, subpelY, step, xTable, yTable, round0Bits, round1Bits)
		comparePlanes(t, "hb_step", got, want, w, h, 2)
	}
}

func TestConvolveScale2DHighBDClampedMatchesSuperResReference(t *testing.T) {
	const srcW, srcH = 160, 128
	const curW, curH = 107, 128
	const bd = 10
	src := highBDPlane(srcW, srcH, 0x5eed, bd)
	sf, err := NewScaleFactors(srcW, srcH, curW, curH)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}

	const pad = 24
	padded, _ := testPlane(srcW+2*pad, srcH+2*pad, 2, (srcW+2*pad)*2)
	for y := range srcH + 2*pad {
		sy := y - pad
		if sy < 0 {
			sy = 0
		} else if sy >= srcH {
			sy = srcH - 1
		}
		for x := range srcW + 2*pad {
			sx := x - pad
			if sx < 0 {
				sx = 0
			} else if sx >= srcW {
				sx = srcW - 1
			}
			setSample(padded, 2, x, y, getSample(src, 2, sx, sy))
		}
	}

	for _, tc := range []struct {
		name           string
		x, y           int
		blockW, blockH int
		mv             Vector
		filterX        InterpFilter
		filterY        InterpFilter
	}{
		{name: "top_left", x: 0, y: 0, blockW: 16, blockH: 16, filterX: InterpEightTapRegular, filterY: InterpEightTapSmooth},
		{name: "middle", x: 48, y: 0, blockW: 16, blockH: 16, filterX: InterpEightTapRegular, filterY: InterpEightTapSmooth},
		{name: "right_edge", x: curW - 16, y: 0, blockW: 16, blockH: 16, filterX: InterpEightTapRegular, filterY: InterpEightTapSmooth},
		{name: "bottom_right", x: curW - 16, y: curH - 16, blockW: 16, blockH: 16, filterX: InterpEightTapRegular, filterY: InterpEightTapSmooth},
		{name: "moving_obmc_area", x: 96, y: 16, blockW: 8, blockH: 16, mv: Vector{Col: 28, Row: 24}, filterX: InterpEightTapRegular, filterY: InterpEightTapRegular},
		{name: "moving_direct_area", x: 96, y: 0, blockW: 11, blockH: 16, mv: Vector{Col: 28, Row: 24}, filterX: InterpEightTapRegular, filterY: InterpEightTapRegular},
	} {
		xTable, err := SubpelKernelTableFor(tc.filterX, tc.blockW)
		if err != nil {
			t.Fatalf("%s xTable: %v", tc.name, err)
		}
		yTable, err := SubpelKernelTableFor(tc.filterY, tc.blockH)
		if err != nil {
			t.Fatalf("%s yTable: %v", tc.name, err)
		}
		startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(tc.x, tc.y, tc.mv, false, false)
		if err != nil {
			t.Fatalf("%s ScaledBlockOrigin: %v", tc.name, err)
		}
		got, _ := testPlane(tc.blockW, tc.blockH, 2, tc.blockW*2)
		if err := ConvolveScale2DHighBDClamped(got, src, bd, 0, 0, tc.blockW, tc.blockH, startX, xStep, startY, yStep, xTable, yTable); err != nil {
			t.Fatalf("%s ConvolveScale2DHighBDClamped: %v", tc.name, err)
		}

		intX := int(startX >> ScaleSubpelBits)
		intY := int(startY >> ScaleSubpelBits)
		subpelX := int(startX & ScaleSubpelMask)
		subpelY := int(startY & ScaleSubpelMask)
		want, _ := testPlane(tc.blockW, tc.blockH, 2, tc.blockW*2)
		libaomConvolveScale2DHighBDRef(want, padded, bd, pad+intX, pad+intY, 0, 0, tc.blockW, tc.blockH,
			subpelX, int(xStep), subpelY, int(yStep), xTable, yTable, round0Bits, round1Bits)
		comparePlanes(t, fmt.Sprintf("superres10_%s", tc.name), got, want, tc.blockW, tc.blockH, 2)
	}
}

// TestConvolveScale2D8ClampedMatchesLibaomSVCL2T1FrameTopLeft pins the
// scaled 8-tap convolver against libaom for the exact SVC L2T1 spatial=1
// frame-top-left block: a 1280x720 enhancement layer references the 640x360
// base, so the start position is libaom's biased (-224, -224) Q10 and the
// step is exactly 0.5 sample per output pel. The leftmost / topmost taps
// fall outside the source plane and must be edge-replicated, exercising the
// Clamped variant against a libaom-shape padded reference. Locks the bias
// (3d13790) + clamping behaviour against silent SVC L2T1 spatial=1 drift.
func TestConvolveScale2D8ClampedMatchesLibaomSVCL2T1FrameTopLeft(t *testing.T) {
	const blockW, blockH = 16, 16
	const srcW, srcH = 64, 64
	src := scaledTestSourcePlane(srcW, srcH, 1, 0x5717)

	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(0, 0, Vector{}, false, false)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}
	if startX >= 0 || startY >= 0 {
		t.Fatalf("expected biased negative startX/startY, got startX=%d startY=%d", startX, startY)
	}

	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, blockW)
	if err != nil {
		t.Fatalf("xTable: %v", err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapRegular, blockH)
	if err != nil {
		t.Fatalf("yTable: %v", err)
	}

	got, _ := testPlane(blockW, blockH, 1, blockW)
	if err := ConvolveScale2D8Clamped(got, src, 0, 0, blockW, blockH, startX, xStep, startY, yStep, xTable, yTable); err != nil {
		t.Fatalf("ConvolveScale2D8Clamped: %v", err)
	}

	const pad = 16
	padded, _ := testPlane(srcW+2*pad, srcH+2*pad, 1, srcW+2*pad)
	for y := range srcH + 2*pad {
		sy := y - pad
		if sy < 0 {
			sy = 0
		} else if sy >= srcH {
			sy = srcH - 1
		}
		for x := range srcW + 2*pad {
			sx := x - pad
			if sx < 0 {
				sx = 0
			} else if sx >= srcW {
				sx = srcW - 1
			}
			padded.Pix[y*padded.Stride+x] = src.Pix[sy*src.Stride+sx]
		}
	}
	intX := int(startX >> ScaleSubpelBits)
	intY := int(startY >> ScaleSubpelBits)
	subpelX := int(startX & ScaleSubpelMask)
	subpelY := int(startY & ScaleSubpelMask)

	want, _ := testPlane(blockW, blockH, 1, blockW)
	libaomConvolveScale2DRef(want, padded, pad+intX, pad+intY, 0, 0, blockW, blockH,
		subpelX, int(xStep), subpelY, int(yStep), xTable, yTable, round0Bits, round1Bits)

	comparePlanes(t, "svc_l2t1_top_left", got, want, blockW, blockH, 1)
}

// TestConvolveScale2D8ClampedMatchesLibaomSVCL2T1RightEdge pins the analogous
// right-edge case: the last column block of an SVC L2T1 spatial=1 frame
// references the rightmost columns of the base plane, with several taps
// overhanging the source plane and replicated.
func TestConvolveScale2D8ClampedMatchesLibaomSVCL2T1RightEdge(t *testing.T) {
	const blockW, blockH = 16, 16
	const srcW, srcH = 64, 64
	src := scaledTestSourcePlane(srcW, srcH, 1, 0x9b0a)

	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	dstX := 2*srcW - blockW
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(dstX, 0, Vector{}, false, false)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}

	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, blockW)
	if err != nil {
		t.Fatalf("xTable: %v", err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapSmooth, blockH)
	if err != nil {
		t.Fatalf("yTable: %v", err)
	}

	got, _ := testPlane(blockW, blockH, 1, blockW)
	if err := ConvolveScale2D8Clamped(got, src, 0, 0, blockW, blockH, startX, xStep, startY, yStep, xTable, yTable); err != nil {
		t.Fatalf("ConvolveScale2D8Clamped: %v", err)
	}

	const pad = 24
	padded, _ := testPlane(srcW+2*pad, srcH+2*pad, 1, srcW+2*pad)
	for y := range srcH + 2*pad {
		sy := y - pad
		if sy < 0 {
			sy = 0
		} else if sy >= srcH {
			sy = srcH - 1
		}
		for x := range srcW + 2*pad {
			sx := x - pad
			if sx < 0 {
				sx = 0
			} else if sx >= srcW {
				sx = srcW - 1
			}
			padded.Pix[y*padded.Stride+x] = src.Pix[sy*src.Stride+sx]
		}
	}
	intX := int(startX >> ScaleSubpelBits)
	intY := int(startY >> ScaleSubpelBits)
	subpelX := int(startX & ScaleSubpelMask)
	subpelY := int(startY & ScaleSubpelMask)

	want, _ := testPlane(blockW, blockH, 1, blockW)
	libaomConvolveScale2DRef(want, padded, pad+intX, pad+intY, 0, 0, blockW, blockH,
		subpelX, int(xStep), subpelY, int(yStep), xTable, yTable, round0Bits, round1Bits)

	comparePlanes(t, "svc_l2t1_right_edge", got, want, blockW, blockH, 1)
}

// TestConvolveScale2D8ClampedMatchesLibaomSVCL2T1FuzzGrid sweeps the L2T1
// scaled convolver across a coarse grid of destination positions, motion
// vectors, and filter combinations, comparing every pixel against the
// libaom-faithful reference implementation. Any drift in the dispatched
// scale-factor / sub-pel split / kernel-table lookup would manifest as a
// non-matching sample here.
func TestConvolveScale2D8ClampedMatchesLibaomSVCL2T1FuzzGrid(t *testing.T) {
	const srcW, srcH = 96, 96
	src := scaledTestSourcePlane(srcW, srcH, 1, 0xa5a5)
	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	const blockW, blockH = 16, 16

	const pad = 32
	padded, _ := testPlane(srcW+2*pad, srcH+2*pad, 1, srcW+2*pad)
	for y := range srcH + 2*pad {
		sy := y - pad
		if sy < 0 {
			sy = 0
		} else if sy >= srcH {
			sy = srcH - 1
		}
		for x := range srcW + 2*pad {
			sx := x - pad
			if sx < 0 {
				sx = 0
			} else if sx >= srcW {
				sx = srcW - 1
			}
			padded.Pix[y*padded.Stride+x] = src.Pix[sy*src.Stride+sx]
		}
	}

	dstPositions := []struct{ x, y int }{
		{0, 0},
		{16, 0},
		{0, 16},
		{32, 32},
		{2*srcW - blockW, 0},
		{0, 2*srcH - blockH},
		{2*srcW - blockW, 2*srcH - blockH},
	}
	mvs := []Vector{
		{Row: 0, Col: 0},
		{Row: 4, Col: 4},
		{Row: -3, Col: 5},
		{Row: -8, Col: 0},
	}
	filters := []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpMultiTapSharp, InterpBilinear}

	for _, dst := range dstPositions {
		for _, mv := range mvs {
			for _, fX := range filters {
				for _, fY := range filters {
					fX, fY := fX, fY
					name := fmt.Sprintf("dst%dx%d_mv(%d,%d)_%s/%s", dst.x, dst.y, mv.Row, mv.Col, filterName(fX), filterName(fY))
					t.Run(name, func(t *testing.T) {
						startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(dst.x, dst.y, mv, false, false)
						if err != nil {
							t.Fatalf("ScaledBlockOrigin: %v", err)
						}
						xTable, err := SubpelKernelTableFor(fX, blockW)
						if err != nil {
							t.Fatalf("xTable: %v", err)
						}
						yTable, err := SubpelKernelTableFor(fY, blockH)
						if err != nil {
							t.Fatalf("yTable: %v", err)
						}
						got, _ := testPlane(blockW, blockH, 1, blockW)
						if err := ConvolveScale2D8Clamped(got, src, 0, 0, blockW, blockH, startX, xStep, startY, yStep, xTable, yTable); err != nil {
							t.Fatalf("ConvolveScale2D8Clamped: %v", err)
						}
						intX := int(startX >> ScaleSubpelBits)
						intY := int(startY >> ScaleSubpelBits)
						subpelX := int(startX & ScaleSubpelMask)
						subpelY := int(startY & ScaleSubpelMask)
						want, _ := testPlane(blockW, blockH, 1, blockW)
						libaomConvolveScale2DRef(want, padded, pad+intX, pad+intY, 0, 0, blockW, blockH,
							subpelX, int(xStep), subpelY, int(yStep), xTable, yTable, round0Bits, round1Bits)
						comparePlanes(t, name, got, want, blockW, blockH, 1)
					})
				}
			}
		}
	}
}

func TestConvolveScale2D8RejectsInvalidInputs(t *testing.T) {
	src := scaledTestSourcePlane(32, 32, 1, 0)
	dst, _ := testPlane(16, 16, 1, 16)
	xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 16)
	yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 16)
	if err := ConvolveScale2D8(dst, src, 0, 0, 0, 16, 0, ScaleSubpelScale, 0, ScaleSubpelScale, xTable, yTable); err == nil {
		t.Fatal("expected ErrInvalidMotion for width=0")
	}
	if err := ConvolveScale2D8(dst, src, 0, 0, 16, 16, 0, 0, 0, ScaleSubpelScale, xTable, yTable); err == nil {
		t.Fatal("expected ErrInvalidMotion for xStep=0")
	}
	if err := ConvolveScale2D8(dst, src, 0, 0, 16, 16, 0, ScaleSubpelScale, 0, 0, xTable, yTable); err == nil {
		t.Fatal("expected ErrInvalidMotion for yStep=0")
	}
	if err := ConvolveScale2D8(dst, src, 0, 0, 16, 16, -1, ScaleSubpelScale, 0, ScaleSubpelScale, xTable, yTable); err == nil {
		t.Fatal("expected ErrInvalidMotion for negative startX (taps left of plane)")
	}
}

func TestSubpelKernelTableForInvalidFilter(t *testing.T) {
	if _, err := SubpelKernelTableFor(interpFilterCount, 16); err == nil {
		t.Fatal("expected ErrInvalidMotion for invalid filter")
	}
}

func TestConvolveScale2D8ZeroAllocation(t *testing.T) {
	src := scaledTestSourcePlane(32, 32, 1, 0x55)
	dst, _ := testPlane(16, 16, 1, 16)
	xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 16)
	yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 16)
	startX := int64(8) * ScaleSubpelScale
	startY := int64(8) * ScaleSubpelScale
	var scratch ScaledConvolveScratch

	allocs := testing.AllocsPerRun(50, func() {
		if err := ConvolveScale2D8(dst, src, 0, 0, 16, 16, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable); err != nil {
			t.Fatalf("ConvolveScale2D8: %v", err)
		}
		if err := ConvolveScale2D8WithScratch(dst, src, 0, 0, 16, 16, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
			t.Fatalf("ConvolveScale2D8WithScratch: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("expected 0 allocs, got %v", allocs)
	}
}

func BenchmarkConvolveScale2DHighBDClampedInterior32(b *testing.B) {
	const bd = 10
	src := highBDPlane(96, 96, 0xace, bd)
	dst, _ := testPlane(32, 32, 2, 64)
	xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	startX := int64(16) * ScaleSubpelScale
	startY := int64(16) * ScaleSubpelScale
	var scratch ScaledConvolveScratch

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ConvolveScale2DHighBDClampedWithScratch(dst, src, bd, 0, 0, 32, 32, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConvolveScale2DHighBDClampedEdge32(b *testing.B) {
	const bd = 10
	src := highBDPlane(40, 40, 0xace, bd)
	dst, _ := testPlane(32, 32, 2, 64)
	xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	var scratch ScaledConvolveScratch

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ConvolveScale2DHighBDClampedWithScratch(dst, src, bd, 0, 0, 32, 32, 0, ScaleSubpelScale, 0, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConvolveScale2D8ClampedInterior32(b *testing.B) {
	src := scaledTestSourcePlane(96, 96, 1, 0x55)
	dst, _ := testPlane(32, 32, 1, 32)
	xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	startX := int64(16) * ScaleSubpelScale
	startY := int64(16) * ScaleSubpelScale

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ConvolveScale2D8Clamped(dst, src, 0, 0, 32, 32, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConvolveScale2D8ClampedInterior32WithScratch(b *testing.B) {
	src := scaledTestSourcePlane(96, 96, 1, 0x55)
	dst, _ := testPlane(32, 32, 1, 32)
	xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	startX := int64(16) * ScaleSubpelScale
	startY := int64(16) * ScaleSubpelScale
	var scratch ScaledConvolveScratch

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ConvolveScale2D8ClampedWithScratch(dst, src, 0, 0, 32, 32, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func filterName(f InterpFilter) string {
	switch f {
	case InterpEightTapRegular:
		return "EightTapRegular"
	case InterpEightTapSmooth:
		return "EightTapSmooth"
	case InterpMultiTapSharp:
		return "MultiTapSharp"
	case InterpBilinear:
		return "Bilinear"
	default:
		return "unknown"
	}
}

func comparePlanes(t *testing.T, name string, got, want frame.Plane, w, h, bytesPerSample int) {
	t.Helper()
	for y := range h {
		for x := range w {
			g := getSample(got, bytesPerSample, x, y)
			wv := getSample(want, bytesPerSample, x, y)
			if g != wv {
				t.Fatalf("%s sample(%d,%d)=%d want %d", name, x, y, g, wv)
			}
		}
	}
}

func clearPlane(p frame.Plane) {
	for i := range p.Pix {
		p.Pix[i] = 0
	}
}

func highBDPlane(width, height, seed, bd int) frame.Plane {
	plane, _ := testPlane(width, height, 2, width*2)
	state := uint32(0x55aa55aa | uint32(seed))
	max := uint16((1 << bd) - 1)
	for y := range height {
		for x := range width {
			state = state*1664525 + 1013904223
			v := uint16(state>>16) & max
			setSample(plane, 2, x, y, v)
		}
	}
	return plane
}
