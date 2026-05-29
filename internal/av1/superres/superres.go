// Ported from libaom: av1/common/resize.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package superres

import "github.com/thesyncim/goav1/internal/av1/frame"

const (
	Num          = 8
	DenomMin     = 9
	DenomMax     = 16
	FilterBits   = 6
	FilterShifts = 1 << FilterBits
	FilterTaps   = 8
	FilterOffset = 3
	ScaleBits    = 14
	ScaleMask    = (1 << ScaleBits) - 1
	ExtraBits    = ScaleBits - FilterBits

	filterRoundBits = 7
)

// UpscalePlane horizontally upscales src into dst using AV1's normative
// superres filter. The caller supplies already-decimated luma or chroma planes.
//
// The whole src plane (src.Width columns) is treated as both the coded
// (downscaled_plane_width) span used to derive the convolution step/phase and
// the physical sampling extent. For libaom's MI-aligned super-res source span
// (where downscaled_plane_width < the physical aligned width), use
// UpscalePlaneCoded.
func UpscalePlane(src frame.SamplePlane, dst frame.SamplePlane, bitDepth uint8) error {
	return UpscalePlaneCoded(src, dst, src.Width, bitDepth)
}

// UpscalePlaneCoded ports av1_upscale_normative_rows for a single-tile-column
// frame plane (av1/common/resize.c). It derives the convolution step (x_step_qn)
// and initial subpel phase (x0_qn) from codedWidth — libaom's
// downscaled_plane_width = ROUND_POWER_OF_TWO(cm->width, ss_x) — while sampling
// from the physical src.Width columns, which libaom sets to the MI-aligned tile
// span src_width = (mi_col_end - mi_col_start) << (MI_SIZE_LOG2 - ss_x).
//
// libaom's av1_superres_upscale (resize.c ~line 1317) upscales from a copy of
// the frame allocated at aligned_width = ALIGN_POWER_OF_TWO(cm->width, 3), whose
// columns in [codedWidth, src.Width) hold the genuine MI-aligned reconstructed /
// border-extended pixels carried by the coded frame buffer — NOT a clamped copy
// of the last coded column. The per-tile-column right pad in
// upscale_normative_rect then replicates src column src.Width-1 beyond the span,
// which the srcLast clamp below reproduces.
func UpscalePlaneCoded(src frame.SamplePlane, dst frame.SamplePlane, codedWidth int, bitDepth uint8) error {
	if err := validatePlanesCoded(src, dst, codedWidth, bitDepth); err != nil {
		return err
	}
	// x_step_qn / x0_qn are derived from the coded (downscaled) plane width, not
	// the physical aligned source extent, matching av1_get_upscale_convolve_step
	// and get_upscale_convolve_x0.
	stepX := ((codedWidth << ScaleBits) + (dst.Width / 2)) / dst.Width
	err := (dst.Width * stepX) - (codedWidth << ScaleBits)
	initialSubpelX := ((-((dst.Width - codedWidth) << (ScaleBits - 1))) + dst.Width/2) / dst.Width
	initialSubpelX += (1 << (ExtraBits - 1)) - err/2
	initialSubpelX &= ScaleMask

	maxValue := (1 << bitDepth) - 1
	srcWidth := src.Width
	dstWidth := dst.Width
	srcLast := srcWidth - 1
	for y := 0; y < src.Height; y++ {
		// Reslice each row to its exact extent so the compiler can drop
		// per-tap bounds checks on the source read and the destination write.
		srcRow := src.Pix[y*src.Stride : y*src.Stride+srcWidth : y*src.Stride+srcWidth]
		dstRow := dst.Pix[y*dst.Stride : y*dst.Stride+dstWidth : y*dst.Stride+dstWidth]
		for x := 0; x < dstWidth; x++ {
			srcX := -(1 << ScaleBits) + initialSubpelX + x*stepX
			srcXPx := srcX >> ScaleBits
			srcXSubpel := (srcX & ScaleMask) >> ExtraBits
			filter := &upscaleFilter[srcXSubpel]
			base := srcXPx - FilterOffset
			var sum int
			if base >= 0 && base+FilterTaps-1 <= srcLast {
				// Central case: the whole 8-tap window is in bounds, so no
				// per-tap clamping is needed. Reslice to a fixed-size window
				// to eliminate bounds checks on the unrolled taps.
				w := srcRow[base : base+FilterTaps : base+FilterTaps]
				sum = int(w[0])*int(filter[0]) +
					int(w[1])*int(filter[1]) +
					int(w[2])*int(filter[2]) +
					int(w[3])*int(filter[3]) +
					int(w[4])*int(filter[4]) +
					int(w[5])*int(filter[5]) +
					int(w[6])*int(filter[6]) +
					int(w[7])*int(filter[7])
			} else {
				for k := 0; k < FilterTaps; k++ {
					sampleX := clipInt(base+k, 0, srcLast)
					sum += int(srcRow[sampleX]) * int(filter[k])
				}
			}
			dstRow[x] = uint16(clipInt(roundPowerOfTwo(sum, filterRoundBits), 0, maxValue))
		}
	}
	return nil
}

func validatePlanesCoded(src frame.SamplePlane, dst frame.SamplePlane, codedWidth int, bitDepth uint8) error {
	if bitDepth < 8 || bitDepth > 12 ||
		!samplePlaneFits(src) || !samplePlaneFits(dst) ||
		dst.Height != src.Height ||
		codedWidth <= 0 ||
		codedWidth > src.Width ||
		dst.Width <= codedWidth ||
		src.Width > int(^uint(0)>>1)>>ScaleBits {
		return frame.ErrInvalidPlane
	}
	return nil
}

func samplePlaneFits(plane frame.SamplePlane) bool {
	if plane.Width <= 0 || plane.Height <= 0 || plane.Stride < plane.Width {
		return false
	}
	need := (plane.Height-1)*plane.Stride + plane.Width
	return need >= 0 && len(plane.Pix) >= need
}

func roundPowerOfTwo(value int, bits int) int {
	if bits <= 0 {
		return value
	}
	return (value + (1 << (bits - 1))) >> bits
}

func clipInt(v int, lo int, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

var upscaleFilter = [FilterShifts][FilterTaps]int16{
	{0, 0, 0, 128, 0, 0, 0, 0}, {0, 0, -1, 128, 2, -1, 0, 0},
	{0, 1, -3, 127, 4, -2, 1, 0}, {0, 1, -4, 127, 6, -3, 1, 0},
	{0, 2, -6, 126, 8, -3, 1, 0}, {0, 2, -7, 125, 11, -4, 1, 0},
	{-1, 2, -8, 125, 13, -5, 2, 0}, {-1, 3, -9, 124, 15, -6, 2, 0},
	{-1, 3, -10, 123, 18, -6, 2, -1}, {-1, 3, -11, 122, 20, -7, 3, -1},
	{-1, 4, -12, 121, 22, -8, 3, -1}, {-1, 4, -13, 120, 25, -9, 3, -1},
	{-1, 4, -14, 118, 28, -9, 3, -1}, {-1, 4, -15, 117, 30, -10, 4, -1},
	{-1, 5, -16, 116, 32, -11, 4, -1}, {-1, 5, -16, 114, 35, -12, 4, -1},
	{-1, 5, -17, 112, 38, -12, 4, -1}, {-1, 5, -18, 111, 40, -13, 5, -1},
	{-1, 5, -18, 109, 43, -14, 5, -1}, {-1, 6, -19, 107, 45, -14, 5, -1},
	{-1, 6, -19, 105, 48, -15, 5, -1}, {-1, 6, -19, 103, 51, -16, 5, -1},
	{-1, 6, -20, 101, 53, -16, 6, -1}, {-1, 6, -20, 99, 56, -17, 6, -1},
	{-1, 6, -20, 97, 58, -17, 6, -1}, {-1, 6, -20, 95, 61, -18, 6, -1},
	{-2, 7, -20, 93, 64, -18, 6, -2}, {-2, 7, -20, 91, 66, -19, 6, -1},
	{-2, 7, -20, 88, 69, -19, 6, -1}, {-2, 7, -20, 86, 71, -19, 6, -1},
	{-2, 7, -20, 84, 74, -20, 7, -2}, {-2, 7, -20, 81, 76, -20, 7, -1},
	{-2, 7, -20, 79, 79, -20, 7, -2}, {-1, 7, -20, 76, 81, -20, 7, -2},
	{-2, 7, -20, 74, 84, -20, 7, -2}, {-1, 6, -19, 71, 86, -20, 7, -2},
	{-1, 6, -19, 69, 88, -20, 7, -2}, {-1, 6, -19, 66, 91, -20, 7, -2},
	{-2, 6, -18, 64, 93, -20, 7, -2}, {-1, 6, -18, 61, 95, -20, 6, -1},
	{-1, 6, -17, 58, 97, -20, 6, -1}, {-1, 6, -17, 56, 99, -20, 6, -1},
	{-1, 6, -16, 53, 101, -20, 6, -1}, {-1, 5, -16, 51, 103, -19, 6, -1},
	{-1, 5, -15, 48, 105, -19, 6, -1}, {-1, 5, -14, 45, 107, -19, 6, -1},
	{-1, 5, -14, 43, 109, -18, 5, -1}, {-1, 5, -13, 40, 111, -18, 5, -1},
	{-1, 4, -12, 38, 112, -17, 5, -1}, {-1, 4, -12, 35, 114, -16, 5, -1},
	{-1, 4, -11, 32, 116, -16, 5, -1}, {-1, 4, -10, 30, 117, -15, 4, -1},
	{-1, 3, -9, 28, 118, -14, 4, -1}, {-1, 3, -9, 25, 120, -13, 4, -1},
	{-1, 3, -8, 22, 121, -12, 4, -1}, {-1, 3, -7, 20, 122, -11, 3, -1},
	{-1, 2, -6, 18, 123, -10, 3, -1}, {0, 2, -6, 15, 124, -9, 3, -1},
	{0, 2, -5, 13, 125, -8, 2, -1}, {0, 1, -4, 11, 125, -7, 2, 0},
	{0, 1, -3, 8, 126, -6, 2, 0}, {0, 1, -3, 6, 127, -4, 1, 0},
	{0, 1, -2, 4, 127, -3, 1, 0}, {0, 0, -1, 2, 128, -1, 0, 0},
}
