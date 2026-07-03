// Ported from libaom: av1/common/convolve.c (av1_dist_wtd_convolve_* and the
// highbd variants), aom_dsp/blend_a64_mask.c (aom_*_blend_a64_d16_mask_c) and
// av1/common/reconinter.c (diffwtd_mask_d16). These implement the AV1 compound
// inter-prediction path, which keeps each reference predictor at the un-rounded
// 16-bit CONV_BUF precision (round_1 = COMPOUND_ROUND1_BITS) and only rounds to
// the final pixel after blending. Blending two already-rounded 8-bit predictors
// (the single-prediction path) loses up to a few LSBs of precision, so compound
// blocks must go through this dedicated path.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

const (
	// libaom convolve.h: ROUND0_BITS / COMPOUND_ROUND1_BITS for the compound
	// (do_average) convolve. round_0 stays 3 and round_1 stays 7 for all bit
	// depths <= 10 (the highbd intbufrange adjustment only fires at bd == 12).
	compoundRound0Bits = round0Bits
	compoundRound1Bits = 7
	// aom_dsp/blend.h
	compoundBlendA64RoundBits = 6
	compoundBlendA64MaxAlpha  = 64
	// CONV_BUF intermediate scratch sizing.
	compoundMaxConvSamples = maxBlockSize * maxBlockSize
	compoundIMMaxSamples   = (maxBlockSize + filterTaps - 1) * maxBlockSize
)

type compoundIM [compoundIMMaxSamples]int32
type compoundIM16 [compoundIMMaxSamples]int16

// CompoundConvolveScratch carries reusable temporary storage for compound
// two-dimensional interpolation. The 2D path overwrites every sample it reads,
// so framework callers keep this scratch live across blocks to avoid clearing a
// large stack array on every compound reference.
type CompoundConvolveScratch struct {
	im  compoundIM
	im8 compoundIM16
	// edge is the dav1d-style emulated-edge window (src/mc_tmpl.c emu_edge_c):
	// clamped compound references materialize their halo into it once and then
	// run the plain SIMD kernels over it.
	edge [emuEdgeStride * emuEdgeRows]byte
}

// compoundRound0 ports get_conv_params_no_round() round_0 selection for the
// compound (is_compound) convolve. round_0 = ROUND0_BITS (3) for bd <= 10 and
// is raised to 5 at bd == 12 by the intbufrange adjustment (intbufrange =
// bd + FILTER_BITS - round_0 + 2 > 16). Unlike the single-prediction path,
// COMPOUND_ROUND1_BITS is left unchanged for compound (libaom only lowers
// round_1 when !is_compound), so callers keep compoundRound1Bits.
func compoundRound0(bitDepth uint8) int {
	round0 := compoundRound0Bits
	intbufrange := int(bitDepth) + filterBits - round0 + 2
	if intbufrange > 16 {
		round0 += intbufrange - 16
	}
	return round0
}

func roundPowerOfTwo3(value int) int {
	return (value + 4) >> 3
}

func roundPowerOfTwo7(value int) int {
	return (value + 64) >> 7
}

// CompoundConvBuf holds one reference's un-rounded 16-bit CONV_BUF predictor
// for compound inter prediction.
type CompoundConvBuf struct {
	Data   [compoundMaxConvSamples]uint16
	Width  uint8
	Height uint8
}

func compoundConvBufView(buf *CompoundConvBuf, width int, height int) ([]uint16, bool) {
	if width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return nil, false
	}
	buf.Width = uint8(width)
	buf.Height = uint8(height)
	return buf.Data[:width*height], true
}

// PredictInterCompoundRefToConvBuf fills buf with the un-rounded 16-bit CONV_BUF
// predictor (av1_dist_wtd_convolve_* with do_average == 0) for one reference.
// The block origin (refX, refY) and subpel phases match the single-prediction
// path; the result is later combined by a compound blend.
func PredictInterCompoundRefToConvBuf(buf *CompoundConvBuf, ref frame.Plane, bytesPerSample int, bitDepth uint8, refX int, refY int, width int, height int, subX int, subY int, filters InterpFilters) error {
	return PredictInterCompoundRefToConvBufWithScratch(buf, ref, bytesPerSample, bitDepth, refX, refY, width, height, subX, subY, filters, nil)
}

// PredictInterCompoundRefToConvBufWithScratch is PredictInterCompoundRefToConvBuf
// using optional caller-owned scratch for the large 2D intermediate block.
func PredictInterCompoundRefToConvBufWithScratch(buf *CompoundConvBuf, ref frame.Plane, bytesPerSample int, bitDepth uint8, refX int, refY int, width int, height int, subX int, subY int, filters InterpFilters, scratch *CompoundConvolveScratch) error {
	if buf == nil || !filters.X.Valid() || !filters.Y.Valid() {
		return ErrInvalidMotion
	}
	if subX < 0 || subX > subpelQ4Mask || subY < 0 || subY > subpelQ4Mask {
		return ErrInvalidMotion
	}
	if !bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
		return ErrInvalidMotion
	}
	out, ok := compoundConvBufView(buf, width, height)
	if !ok {
		return ErrInvalidMotion
	}
	bd := int(bitDepth)
	xKernel, err := interpKernel(filters.X, width, subX)
	if err != nil {
		return err
	}
	yKernel, err := interpKernel(filters.Y, height, subY)
	if err != nil {
		return err
	}
	round0 := compoundRound0(bitDepth)
	offsetBits := bd + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	if bytesPerSample == 1 {
		predictInterCompoundRef8ToConvBuf(out, ref, refX, refY, width, height, subX, subY, xKernel, yKernel, round0, offsetBits, roundOffset, scratch)
		return nil
	}
	predictInterCompoundRefHighBDToConvBuf(out, ref, bd, refX, refY, width, height, subX, subY, xKernel, yKernel, round0, offsetBits, roundOffset, scratch)
	return nil
}

func predictInterCompoundRefHighBDToConvBuf(out []uint16, ref frame.Plane, bitDepth int, refX int, refY int, width int, height int, subX int, subY int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, roundOffset int, scratch *CompoundConvolveScratch) {
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	switch {
	case subX != 0 && subY != 0:
		if scratch != nil {
			if planeRegionFits(ref, 2, refX-foX, refY-foY, width+filterTaps-1, height+filterTaps-1) {
				predictInterCompoundRefHighBDToConvBuf2DResidentImpl(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, &scratch.im)
			} else {
				predictInterCompoundRefHighBDToConvBuf2DClampedImpl(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, &scratch.im)
			}
		} else {
			var im compoundIM
			if planeRegionFits(ref, 2, refX-foX, refY-foY, width+filterTaps-1, height+filterTaps-1) {
				predictInterCompoundRefHighBDToConvBuf2DResidentImpl(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, &im)
			} else {
				predictInterCompoundRefHighBDToConvBuf2DClampedImpl(out, ref, refX, refY, width, height, xKernel, yKernel, round0, offsetBits, bitDepth, &im)
			}
		}
	case subX != 0:
		if planeRegionFits(ref, 2, refX-foX, refY, width+filterTaps-1, height) {
			predictInterCompoundRefHighBDToConvBufXResidentImpl(out, ref, refX, refY, width, height, xKernel, round0, roundOffset)
		} else {
			predictInterCompoundRefHighBDToConvBufXClamped(out, ref, refX, refY, width, height, xKernel, round0, roundOffset)
		}
	case subY != 0:
		if planeRegionFits(ref, 2, refX, refY-foY, width, height+filterTaps-1) {
			predictInterCompoundRefHighBDToConvBufYResidentImpl(out, ref, refX, refY, width, height, yKernel, round0, roundOffset)
		} else {
			predictInterCompoundRefHighBDToConvBufYClamped(out, ref, refX, refY, width, height, yKernel, round0, roundOffset)
		}
	default:
		if planeRegionFits(ref, 2, refX, refY, width, height) {
			predictInterCompoundRefHighBDToConvBufCopyResidentImpl(out, ref, refX, refY, width, height, round0, roundOffset)
		} else {
			predictInterCompoundRefHighBDToConvBufCopyClamped(out, ref, refX, refY, width, height, round0, roundOffset)
		}
	}
}

func predictInterCompoundRefHighBDToConvBuf2DResident(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, bitDepth int, im *compoundIM) {
	const imStride = maxBlockSize
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	xBias := 1 << (bitDepth + filterBits - 1)
	round0Bias := 1 << (round0 - 1)
	stride := ref.Stride

	xk0, xk1, xk2, xk3, xk4, xk5, xk6, xk7 := int(xKernel[0]), int(xKernel[1]), int(xKernel[2]), int(xKernel[3]), int(xKernel[4]), int(xKernel[5]), int(xKernel[6]), int(xKernel[7])
	xFourTap := xk0 == 0 && xk1 == 0 && xk6 == 0 && xk7 == 0
	for y := range imH {
		base := (refY-foY+y)*stride + (refX-foX)*2
		srcRow := ref.Pix[base : base+(width+filterTaps-1)*2]
		imRow := im[y*imStride:]
		if xFourTap {
			for x := range width {
				o := x * 2
				s2 := int(uint16(srcRow[o+4]) | uint16(srcRow[o+5])<<8)
				s3 := int(uint16(srcRow[o+6]) | uint16(srcRow[o+7])<<8)
				s4 := int(uint16(srcRow[o+8]) | uint16(srcRow[o+9])<<8)
				s5 := int(uint16(srcRow[o+10]) | uint16(srcRow[o+11])<<8)
				sum := xBias + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5
				imRow[x] = int32((sum + round0Bias) >> round0)
			}
			continue
		}
		for x := range width {
			o := x * 2
			s0 := int(uint16(srcRow[o]) | uint16(srcRow[o+1])<<8)
			s1 := int(uint16(srcRow[o+2]) | uint16(srcRow[o+3])<<8)
			s2 := int(uint16(srcRow[o+4]) | uint16(srcRow[o+5])<<8)
			s3 := int(uint16(srcRow[o+6]) | uint16(srcRow[o+7])<<8)
			s4 := int(uint16(srcRow[o+8]) | uint16(srcRow[o+9])<<8)
			s5 := int(uint16(srcRow[o+10]) | uint16(srcRow[o+11])<<8)
			s6 := int(uint16(srcRow[o+12]) | uint16(srcRow[o+13])<<8)
			s7 := int(uint16(srcRow[o+14]) | uint16(srcRow[o+15])<<8)
			sum := xBias + xk0*s0 + xk1*s1 + xk2*s2 + xk3*s3 + xk4*s4 + xk5*s5 + xk6*s6 + xk7*s7
			imRow[x] = int32((sum + round0Bias) >> round0)
		}
	}

	yBias := 1 << offsetBits
	yk0, yk1, yk2, yk3, yk4, yk5, yk6, yk7 := int(yKernel[0]), int(yKernel[1]), int(yKernel[2]), int(yKernel[3]), int(yKernel[4]), int(yKernel[5]), int(yKernel[6]), int(yKernel[7])
	yFourTap := yk0 == 0 && yk1 == 0 && yk6 == 0 && yk7 == 0
	for y := range height {
		outRow := out[y*width:]
		if yFourTap {
			col := im[(y+2)*imStride : (y+6)*imStride+width]
			for x := range width {
				sum := yBias + yk2*int(col[x]) + yk3*int(col[x+imStride]) +
					yk4*int(col[x+2*imStride]) + yk5*int(col[x+3*imStride])
				outRow[x] = uint16(roundPowerOfTwo7(sum))
			}
			continue
		}
		col := im[y*imStride : (y+7)*imStride+width]
		for x := range width {
			sum := yBias + yk0*int(col[x]) + yk1*int(col[x+imStride]) + yk2*int(col[x+2*imStride]) +
				yk3*int(col[x+3*imStride]) + yk4*int(col[x+4*imStride]) + yk5*int(col[x+5*imStride]) +
				yk6*int(col[x+6*imStride]) + yk7*int(col[x+7*imStride])
			outRow[x] = uint16(roundPowerOfTwo7(sum))
		}
	}
}

func predictInterCompoundRefHighBDToConvBuf2DClamped(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, bitDepth int, im *compoundIM) {
	const imStride = maxBlockSize
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	for y := range imH {
		for x := range width {
			sum := 1 << (bitDepth + filterBits - 1)
			for k := range filterTaps {
				sum += int(xKernel[k]) * int(loadHighBDSampleClamped(ref, refX+x-foX+k, refY-foY+y))
			}
			im[y*imStride+x] = int32(roundPowerOfTwo(sum, round0))
		}
	}
	for y := range height {
		for x := range width {
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			out[y*width+x] = uint16(roundPowerOfTwo(sum, compoundRound1Bits))
		}
	}
}

func predictInterCompoundRefHighBDToConvBufXResident(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	bits := filterBits - compoundRound1Bits
	scale := 1 << bits
	round0Bias := 1 << (round0 - 1)
	stride := ref.Stride
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	firstTap := refX - fo
	for y := range height {
		base := (refY+y)*stride + firstTap*2
		srcRow := ref.Pix[base : base+(width+filterTaps-1)*2]
		outRow := out[y*width:]
		if fourTap {
			for x := range width {
				o := x * 2
				s2 := int(uint16(srcRow[o+4]) | uint16(srcRow[o+5])<<8)
				s3 := int(uint16(srcRow[o+6]) | uint16(srcRow[o+7])<<8)
				s4 := int(uint16(srcRow[o+8]) | uint16(srcRow[o+9])<<8)
				s5 := int(uint16(srcRow[o+10]) | uint16(srcRow[o+11])<<8)
				res := ((k2*s2 + k3*s3 + k4*s4 + k5*s5 + round0Bias) >> round0) * scale
				outRow[x] = uint16(res + roundOffset)
			}
			continue
		}
		for x := range width {
			o := x * 2
			s0 := int(uint16(srcRow[o]) | uint16(srcRow[o+1])<<8)
			s1 := int(uint16(srcRow[o+2]) | uint16(srcRow[o+3])<<8)
			s2 := int(uint16(srcRow[o+4]) | uint16(srcRow[o+5])<<8)
			s3 := int(uint16(srcRow[o+6]) | uint16(srcRow[o+7])<<8)
			s4 := int(uint16(srcRow[o+8]) | uint16(srcRow[o+9])<<8)
			s5 := int(uint16(srcRow[o+10]) | uint16(srcRow[o+11])<<8)
			s6 := int(uint16(srcRow[o+12]) | uint16(srcRow[o+13])<<8)
			s7 := int(uint16(srcRow[o+14]) | uint16(srcRow[o+15])<<8)
			sum := k0*s0 + k1*s1 + k2*s2 + k3*s3 + k4*s4 + k5*s5 + k6*s6 + k7*s7
			outRow[x] = uint16(((sum+round0Bias)>>round0)*scale + roundOffset)
		}
	}
}

func predictInterCompoundRefHighBDToConvBufXClamped(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	bits := filterBits - compoundRound1Bits
	scale := 1 << bits
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	for y := range height {
		outRow := out[y*width:]
		if fourTap {
			for x := range width {
				sx := refX + x - fo
				sum := k2*int(loadHighBDSampleClamped(ref, sx+2, refY+y)) +
					k3*int(loadHighBDSampleClamped(ref, sx+3, refY+y)) +
					k4*int(loadHighBDSampleClamped(ref, sx+4, refY+y)) +
					k5*int(loadHighBDSampleClamped(ref, sx+5, refY+y))
				outRow[x] = uint16(roundPowerOfTwo(sum, round0)*scale + roundOffset)
			}
			continue
		}
		for x := range width {
			sx := refX + x - fo
			sum := k0*int(loadHighBDSampleClamped(ref, sx, refY+y)) +
				k1*int(loadHighBDSampleClamped(ref, sx+1, refY+y)) +
				k2*int(loadHighBDSampleClamped(ref, sx+2, refY+y)) +
				k3*int(loadHighBDSampleClamped(ref, sx+3, refY+y)) +
				k4*int(loadHighBDSampleClamped(ref, sx+4, refY+y)) +
				k5*int(loadHighBDSampleClamped(ref, sx+5, refY+y)) +
				k6*int(loadHighBDSampleClamped(ref, sx+6, refY+y)) +
				k7*int(loadHighBDSampleClamped(ref, sx+7, refY+y))
			outRow[x] = uint16(roundPowerOfTwo(sum, round0)*scale + roundOffset)
		}
	}
}

func predictInterCompoundRefHighBDToConvBufYResident(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	bits := filterBits - round0
	scale := 1 << bits
	stride := ref.Stride
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	for y := range height {
		sy := refY + y - fo
		base := sy*stride + refX*2
		col := ref.Pix[base : base+(filterTaps-1)*stride+width*2]
		outRow := out[y*width:]
		if fourTap {
			for x := range width {
				o := x * 2
				s2 := int(uint16(col[o+2*stride]) | uint16(col[o+2*stride+1])<<8)
				s3 := int(uint16(col[o+3*stride]) | uint16(col[o+3*stride+1])<<8)
				s4 := int(uint16(col[o+4*stride]) | uint16(col[o+4*stride+1])<<8)
				s5 := int(uint16(col[o+5*stride]) | uint16(col[o+5*stride+1])<<8)
				outRow[x] = uint16(roundPowerOfTwo7((k2*s2+k3*s3+k4*s4+k5*s5)*scale) + roundOffset)
			}
			continue
		}
		for x := range width {
			o := x * 2
			s0 := int(uint16(col[o]) | uint16(col[o+1])<<8)
			s1 := int(uint16(col[o+stride]) | uint16(col[o+stride+1])<<8)
			s2 := int(uint16(col[o+2*stride]) | uint16(col[o+2*stride+1])<<8)
			s3 := int(uint16(col[o+3*stride]) | uint16(col[o+3*stride+1])<<8)
			s4 := int(uint16(col[o+4*stride]) | uint16(col[o+4*stride+1])<<8)
			s5 := int(uint16(col[o+5*stride]) | uint16(col[o+5*stride+1])<<8)
			s6 := int(uint16(col[o+6*stride]) | uint16(col[o+6*stride+1])<<8)
			s7 := int(uint16(col[o+7*stride]) | uint16(col[o+7*stride+1])<<8)
			sum := k0*s0 + k1*s1 + k2*s2 + k3*s3 + k4*s4 + k5*s5 + k6*s6 + k7*s7
			outRow[x] = uint16(roundPowerOfTwo7(sum*scale) + roundOffset)
		}
	}
}

func predictInterCompoundRefHighBDToConvBufYClamped(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	bits := filterBits - round0
	scale := 1 << bits
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	for y := range height {
		sy := refY + y - fo
		outRow := out[y*width:]
		if fourTap {
			for x := range width {
				sx := refX + x
				sum := k2*int(loadHighBDSampleClamped(ref, sx, sy+2)) +
					k3*int(loadHighBDSampleClamped(ref, sx, sy+3)) +
					k4*int(loadHighBDSampleClamped(ref, sx, sy+4)) +
					k5*int(loadHighBDSampleClamped(ref, sx, sy+5))
				outRow[x] = uint16(roundPowerOfTwo7(sum*scale) + roundOffset)
			}
			continue
		}
		for x := range width {
			sx := refX + x
			sum := k0*int(loadHighBDSampleClamped(ref, sx, sy)) +
				k1*int(loadHighBDSampleClamped(ref, sx, sy+1)) +
				k2*int(loadHighBDSampleClamped(ref, sx, sy+2)) +
				k3*int(loadHighBDSampleClamped(ref, sx, sy+3)) +
				k4*int(loadHighBDSampleClamped(ref, sx, sy+4)) +
				k5*int(loadHighBDSampleClamped(ref, sx, sy+5)) +
				k6*int(loadHighBDSampleClamped(ref, sx, sy+6)) +
				k7*int(loadHighBDSampleClamped(ref, sx, sy+7))
			outRow[x] = uint16(roundPowerOfTwo7(sum*scale) + roundOffset)
		}
	}
}

var predictInterCompoundRefHighBDToConvBufCopyResidentImpl = predictInterCompoundRefHighBDToConvBufCopyResidentPureGo
var predictInterCompoundRefHighBDToConvBuf2DResidentImpl = predictInterCompoundRefHighBDToConvBuf2DResident

// predictInterCompoundRefHighBDToConvBuf2DClampedImpl is the dispatch slot for
// the non-resident (edge-overhanging) HBD compound 2D convolve. It defaults to
// the pure-Go per-tap-clamping reference; arm64 rebinds it to an emu_edge NEON
// path (see compound_neon_arm64.go).
var predictInterCompoundRefHighBDToConvBuf2DClampedImpl = predictInterCompoundRefHighBDToConvBuf2DClamped
var predictInterCompoundRefHighBDToConvBufXResidentImpl = predictInterCompoundRefHighBDToConvBufXResident
var predictInterCompoundRefHighBDToConvBufYResidentImpl = predictInterCompoundRefHighBDToConvBufYResident

func predictInterCompoundRefHighBDToConvBufCopyResidentPureGo(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, round0 int, roundOffset int) {
	bits := 2*filterBits - compoundRound1Bits - round0
	scale := 1 << bits
	stride := ref.Stride
	for y := range height {
		base := (refY+y)*stride + refX*2
		srcRow := ref.Pix[base : base+width*2]
		outRow := out[y*width:]
		for x := range width {
			o := x * 2
			s := int(uint16(srcRow[o]) | uint16(srcRow[o+1])<<8)
			outRow[x] = uint16(s*scale + roundOffset)
		}
	}
}

func predictInterCompoundRefHighBDToConvBufCopyClamped(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, round0 int, roundOffset int) {
	bits := 2*filterBits - compoundRound1Bits - round0
	scale := 1 << bits
	for y := range height {
		outRow := out[y*width:]
		for x := range width {
			outRow[x] = uint16(int(loadHighBDSampleClamped(ref, refX+x, refY+y))*scale + roundOffset)
		}
	}
}

func predictInterCompoundRef8ToConvBuf(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, subX int, subY int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, round0 int, offsetBits int, roundOffset int, scratch *CompoundConvolveScratch) {
	switch {
	case subX != 0 && subY != 0:
		predictInterCompoundRef8ToConvBuf2DImpl(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, scratch)
	case subX != 0:
		predictInterCompoundRef8ToConvBufXImpl(out, ref, refX, refY, width, height, xKernel, roundOffset)
	case subY != 0:
		predictInterCompoundRef8ToConvBufYImpl(out, ref, refX, refY, width, height, yKernel, round0, roundOffset)
	default:
		predictInterCompoundRef8ToConvBufCopyImpl(out, ref, refX, refY, width, height, round0, roundOffset)
	}
}

var predictInterCompoundRef8ToConvBuf2DImpl = predictInterCompoundRef8ToConvBuf2DPureGo

func predictInterCompoundRef8ToConvBuf2DPureGo(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, offsetBits int, scratch *CompoundConvolveScratch) {
	if scratch != nil {
		predictInterCompoundRef8ToConvBuf2DWithIM(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, &scratch.im)
		return
	}
	var im compoundIM
	predictInterCompoundRef8ToConvBuf2DWithIM(out, ref, refX, refY, width, height, xKernel, yKernel, offsetBits, &im)
}

func predictInterCompoundRef8ToConvBuf2DWithIM(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16, offsetBits int, im *compoundIM) {
	const imStride = maxBlockSize
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	xBias := 1 << (8 + filterBits - 1)

	xk0, xk1, xk2, xk3, xk4, xk5, xk6, xk7 := int(xKernel[0]), int(xKernel[1]), int(xKernel[2]), int(xKernel[3]), int(xKernel[4]), int(xKernel[5]), int(xKernel[6]), int(xKernel[7])
	xFourTap := xk0 == 0 && xk1 == 0 && xk6 == 0 && xk7 == 0
	firstTap := refX - foX
	xLo, xHi := clampedXInterior(firstTap, filterTaps, ref.Width, width)
	for y := range imH {
		cy := clampInt(refY-foY+y, 0, ref.Height-1)
		rowBase := cy * ref.Stride
		imRow := im[y*imStride:]
		if xFourTap {
			for x := 0; x < xLo; x++ {
				sx := firstTap + x
				sum := xBias +
					xk2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
					xk3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
					xk4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
					xk5*int(loadSample8ClampedRow(ref, sx+5, rowBase))
				imRow[x] = int32(roundPowerOfTwo3(sum))
			}
			if xHi > xLo {
				srcRow := ref.Pix[rowBase+firstTap+xLo:]
				for x := xLo; x < xHi; x++ {
					i := x - xLo
					s := srcRow[i : i+filterTaps]
					sum := xBias + xk2*int(s[2]) + xk3*int(s[3]) + xk4*int(s[4]) + xk5*int(s[5])
					imRow[x] = int32(roundPowerOfTwo3(sum))
				}
			}
			for x := xHi; x < width; x++ {
				sx := firstTap + x
				sum := xBias +
					xk2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
					xk3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
					xk4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
					xk5*int(loadSample8ClampedRow(ref, sx+5, rowBase))
				imRow[x] = int32(roundPowerOfTwo3(sum))
			}
			continue
		}
		for x := 0; x < xLo; x++ {
			sx := firstTap + x
			sum := xBias +
				xk0*int(loadSample8ClampedRow(ref, sx, rowBase)) +
				xk1*int(loadSample8ClampedRow(ref, sx+1, rowBase)) +
				xk2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
				xk3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
				xk4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
				xk5*int(loadSample8ClampedRow(ref, sx+5, rowBase)) +
				xk6*int(loadSample8ClampedRow(ref, sx+6, rowBase)) +
				xk7*int(loadSample8ClampedRow(ref, sx+7, rowBase))
			imRow[x] = int32(roundPowerOfTwo3(sum))
		}
		if xHi > xLo {
			srcRow := ref.Pix[rowBase+firstTap+xLo:]
			for x := xLo; x < xHi; x++ {
				i := x - xLo
				s := srcRow[i : i+filterTaps]
				sum := xBias + xk0*int(s[0]) + xk1*int(s[1]) + xk2*int(s[2]) + xk3*int(s[3]) +
					xk4*int(s[4]) + xk5*int(s[5]) + xk6*int(s[6]) + xk7*int(s[7])
				imRow[x] = int32(roundPowerOfTwo3(sum))
			}
		}
		for x := xHi; x < width; x++ {
			sx := firstTap + x
			sum := xBias +
				xk0*int(loadSample8ClampedRow(ref, sx, rowBase)) +
				xk1*int(loadSample8ClampedRow(ref, sx+1, rowBase)) +
				xk2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
				xk3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
				xk4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
				xk5*int(loadSample8ClampedRow(ref, sx+5, rowBase)) +
				xk6*int(loadSample8ClampedRow(ref, sx+6, rowBase)) +
				xk7*int(loadSample8ClampedRow(ref, sx+7, rowBase))
			imRow[x] = int32(roundPowerOfTwo3(sum))
		}
	}

	yBias := 1 << offsetBits
	yk0, yk1, yk2, yk3, yk4, yk5, yk6, yk7 := int(yKernel[0]), int(yKernel[1]), int(yKernel[2]), int(yKernel[3]), int(yKernel[4]), int(yKernel[5]), int(yKernel[6]), int(yKernel[7])
	yFourTap := yk0 == 0 && yk1 == 0 && yk6 == 0 && yk7 == 0
	for y := range height {
		outRow := out[y*width:]
		if yFourTap {
			col := im[(y+2)*imStride : (y+6)*imStride+width]
			for x := range width {
				sum := yBias + yk2*int(col[x]) + yk3*int(col[x+imStride]) +
					yk4*int(col[x+2*imStride]) + yk5*int(col[x+3*imStride])
				outRow[x] = uint16(roundPowerOfTwo7(sum))
			}
			continue
		}
		col := im[y*imStride : (y+7)*imStride+width]
		for x := range width {
			sum := yBias + yk0*int(col[x]) + yk1*int(col[x+imStride]) + yk2*int(col[x+2*imStride]) +
				yk3*int(col[x+3*imStride]) + yk4*int(col[x+4*imStride]) + yk5*int(col[x+5*imStride]) +
				yk6*int(col[x+6*imStride]) + yk7*int(col[x+7*imStride])
			outRow[x] = uint16(roundPowerOfTwo7(sum))
		}
	}
}

var predictInterCompoundRef8ToConvBufXImpl = predictInterCompoundRef8ToConvBufXPureGo

func predictInterCompoundRef8ToConvBufXPureGo(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, roundOffset int) {
	fo := filterTaps/2 - 1
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	firstTap := refX - fo
	xLo, xHi := clampedXInterior(firstTap, filterTaps, ref.Width, width)
	for y := range height {
		cy := clampInt(refY+y, 0, ref.Height-1)
		rowBase := cy * ref.Stride
		outRow := out[y*width:]
		if fourTap {
			for x := 0; x < xLo; x++ {
				sx := firstTap + x
				sum := k2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
					k3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
					k4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
					k5*int(loadSample8ClampedRow(ref, sx+5, rowBase))
				outRow[x] = uint16(roundPowerOfTwo3(sum) + roundOffset)
			}
			if xHi > xLo {
				srcRow := ref.Pix[rowBase+firstTap+xLo:]
				for x := xLo; x < xHi; x++ {
					i := x - xLo
					s := srcRow[i : i+filterTaps]
					sum := k2*int(s[2]) + k3*int(s[3]) + k4*int(s[4]) + k5*int(s[5])
					outRow[x] = uint16(roundPowerOfTwo3(sum) + roundOffset)
				}
			}
			for x := xHi; x < width; x++ {
				sx := firstTap + x
				sum := k2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
					k3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
					k4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
					k5*int(loadSample8ClampedRow(ref, sx+5, rowBase))
				outRow[x] = uint16(roundPowerOfTwo3(sum) + roundOffset)
			}
			continue
		}
		for x := 0; x < xLo; x++ {
			sx := firstTap + x
			sum := k0*int(loadSample8ClampedRow(ref, sx, rowBase)) +
				k1*int(loadSample8ClampedRow(ref, sx+1, rowBase)) +
				k2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
				k3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
				k4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
				k5*int(loadSample8ClampedRow(ref, sx+5, rowBase)) +
				k6*int(loadSample8ClampedRow(ref, sx+6, rowBase)) +
				k7*int(loadSample8ClampedRow(ref, sx+7, rowBase))
			outRow[x] = uint16(roundPowerOfTwo3(sum) + roundOffset)
		}
		if xHi > xLo {
			srcRow := ref.Pix[rowBase+firstTap+xLo:]
			for x := xLo; x < xHi; x++ {
				i := x - xLo
				s := srcRow[i : i+filterTaps]
				sum := k0*int(s[0]) + k1*int(s[1]) + k2*int(s[2]) + k3*int(s[3]) +
					k4*int(s[4]) + k5*int(s[5]) + k6*int(s[6]) + k7*int(s[7])
				outRow[x] = uint16(roundPowerOfTwo3(sum) + roundOffset)
			}
		}
		for x := xHi; x < width; x++ {
			sx := firstTap + x
			sum := k0*int(loadSample8ClampedRow(ref, sx, rowBase)) +
				k1*int(loadSample8ClampedRow(ref, sx+1, rowBase)) +
				k2*int(loadSample8ClampedRow(ref, sx+2, rowBase)) +
				k3*int(loadSample8ClampedRow(ref, sx+3, rowBase)) +
				k4*int(loadSample8ClampedRow(ref, sx+4, rowBase)) +
				k5*int(loadSample8ClampedRow(ref, sx+5, rowBase)) +
				k6*int(loadSample8ClampedRow(ref, sx+6, rowBase)) +
				k7*int(loadSample8ClampedRow(ref, sx+7, rowBase))
			outRow[x] = uint16(roundPowerOfTwo3(sum) + roundOffset)
		}
	}
}

var predictInterCompoundRef8ToConvBufYImpl = predictInterCompoundRef8ToConvBufYPureGo

func predictInterCompoundRef8ToConvBufYPureGo(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, kernel [filterTaps]int16, round0 int, roundOffset int) {
	fo := filterTaps/2 - 1
	bits := filterBits - round0
	scale := 1 << bits
	stride := ref.Stride
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	fourTap := k0 == 0 && k1 == 0 && k6 == 0 && k7 == 0
	xLo, xHi := clampedXInterior(refX, 1, ref.Width, width)
	for y := range height {
		sy := refY + y - fo
		outRow := out[y*width:]
		yResident := sy >= 0 && sy+filterTaps-1 <= ref.Height-1
		if yResident && xHi > xLo {
			if fourTap {
				col := ref.Pix[(sy+2)*stride+refX+xLo:]
				for x := xLo; x < xHi; x++ {
					i := x - xLo
					sum := k2*int(col[i]) + k3*int(col[i+stride]) +
						k4*int(col[i+2*stride]) + k5*int(col[i+3*stride])
					res := sum * scale
					outRow[x] = uint16(roundPowerOfTwo(res, compoundRound1Bits) + roundOffset)
				}
			} else {
				col := ref.Pix[sy*stride+refX+xLo:]
				for x := xLo; x < xHi; x++ {
					i := x - xLo
					sum := k0*int(col[i]) + k1*int(col[i+stride]) + k2*int(col[i+2*stride]) +
						k3*int(col[i+3*stride]) + k4*int(col[i+4*stride]) + k5*int(col[i+5*stride]) +
						k6*int(col[i+6*stride]) + k7*int(col[i+7*stride])
					res := sum * scale
					outRow[x] = uint16(roundPowerOfTwo(res, compoundRound1Bits) + roundOffset)
				}
			}
			for x := 0; x < xLo; x++ {
				outRow[x] = compoundY8ClampedConvBufSample(ref, refX+x, sy, scale, roundOffset, fourTap, k0, k1, k2, k3, k4, k5, k6, k7)
			}
			for x := xHi; x < width; x++ {
				outRow[x] = compoundY8ClampedConvBufSample(ref, refX+x, sy, scale, roundOffset, fourTap, k0, k1, k2, k3, k4, k5, k6, k7)
			}
			continue
		}
		for x := range width {
			outRow[x] = compoundY8ClampedConvBufSample(ref, refX+x, sy, scale, roundOffset, fourTap, k0, k1, k2, k3, k4, k5, k6, k7)
		}
	}
}

func compoundY8ClampedConvBufSample(ref frame.Plane, sx int, sy int, scale int, roundOffset int, fourTap bool, k0 int, k1 int, k2 int, k3 int, k4 int, k5 int, k6 int, k7 int) uint16 {
	if fourTap {
		sum := k2*int(loadSample8Clamped(ref, sx, sy+2)) +
			k3*int(loadSample8Clamped(ref, sx, sy+3)) +
			k4*int(loadSample8Clamped(ref, sx, sy+4)) +
			k5*int(loadSample8Clamped(ref, sx, sy+5))
		res := sum * scale
		return uint16(roundPowerOfTwo(res, compoundRound1Bits) + roundOffset)
	}
	sum := k0*int(loadSample8Clamped(ref, sx, sy)) +
		k1*int(loadSample8Clamped(ref, sx, sy+1)) +
		k2*int(loadSample8Clamped(ref, sx, sy+2)) +
		k3*int(loadSample8Clamped(ref, sx, sy+3)) +
		k4*int(loadSample8Clamped(ref, sx, sy+4)) +
		k5*int(loadSample8Clamped(ref, sx, sy+5)) +
		k6*int(loadSample8Clamped(ref, sx, sy+6)) +
		k7*int(loadSample8Clamped(ref, sx, sy+7))
	res := sum * scale
	return uint16(roundPowerOfTwo(res, compoundRound1Bits) + roundOffset)
}

var predictInterCompoundRef8ToConvBufCopyImpl = predictInterCompoundRef8ToConvBufCopyPureGo

func predictInterCompoundRef8ToConvBufCopyPureGo(out []uint16, ref frame.Plane, refX int, refY int, width int, height int, round0 int, roundOffset int) {
	bits := 2*filterBits - compoundRound1Bits - round0
	scale := 1 << bits
	xLo, xHi := clampedXInterior(refX, 1, ref.Width, width)
	for y := range height {
		cy := clampInt(refY+y, 0, ref.Height-1)
		rowBase := cy * ref.Stride
		outRow := out[y*width:]
		if xHi > xLo {
			srcRow := ref.Pix[rowBase+refX+xLo:]
			for x := xLo; x < xHi; x++ {
				outRow[x] = uint16(int(srcRow[x-xLo])*scale + roundOffset)
			}
		}
		for x := 0; x < xLo; x++ {
			sx := clampInt(refX+x, 0, ref.Width-1)
			outRow[x] = uint16(int(ref.Pix[rowBase+sx])*scale + roundOffset)
		}
		for x := xHi; x < width; x++ {
			sx := clampInt(refX+x, 0, ref.Width-1)
			outRow[x] = uint16(int(ref.Pix[rowBase+sx])*scale + roundOffset)
		}
	}
}

// PredictScaledCompoundRefToConvBuf fills buf with the un-rounded 16-bit
// CONV_BUF predictor for one scaled reference. It mirrors
// av1_dist_wtd_convolve_2d_scale_c / av1_highbd_dist_wtd_convolve_2d_scale_c
// with do_average == 0, so compound blends can keep the scaled predictor at
// the same precision as the same-size translational path.
func PredictScaledCompoundRefToConvBuf(buf *CompoundConvBuf, ref frame.Plane, bytesPerSample int, bitDepth uint8, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable) error {
	return PredictScaledCompoundRefToConvBufWithScratch(buf, ref, bytesPerSample, bitDepth, width, height,
		startX, xStep, startY, yStep, xTable, yTable, nil)
}

// PredictScaledCompoundRefToConvBufWithScratch is
// PredictScaledCompoundRefToConvBuf using optional caller-owned scratch for the
// large scaled intermediate block.
func PredictScaledCompoundRefToConvBufWithScratch(buf *CompoundConvBuf, ref frame.Plane, bytesPerSample int, bitDepth uint8, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable, scratch *ScaledConvolveScratch) error {
	if buf == nil || xStep <= 0 || yStep <= 0 ||
		!bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
		return ErrInvalidMotion
	}
	out, ok := compoundConvBufView(buf, width, height)
	if !ok {
		return ErrInvalidMotion
	}
	imH, ok := scaledIMHeight(height, startY, yStep)
	if !ok {
		return ErrInvalidMotion
	}
	if !planeRegionFits(ref, bytesPerSample, 0, 0, ref.Width, ref.Height) {
		return ErrInvalidMotion
	}
	if xStep == ScaleSubpelScale && yStep == ScaleSubpelScale &&
		predictScaledCompoundRefToConvBufIdentityStep(out, ref, bytesPerSample, bitDepth, width, height, startX, startY, xTable, yTable, scratch) {
		return nil
	}
	if bytesPerSample == 1 && yStep == ScaleSubpelScale {
		yFilterIdx := int(scaledSubpel(startY) >> ScaleExtraBits)
		if scaledKernelIsIdentity(yTable[yFilterIdx]) {
			predictScaledCompoundRef8ToConvBufYIdentity(out, ref, width, height, startX, xStep, startY, xTable)
			return nil
		}
	}
	if bytesPerSample == 2 && yStep == ScaleSubpelScale {
		yFilterIdx := int(scaledSubpel(startY) >> ScaleExtraBits)
		if scaledKernelIsIdentity(yTable[yFilterIdx]) {
			predictScaledCompoundRefHighBDToConvBufYIdentity(out, ref, int(bitDepth), width, height, startX, xStep, startY, xTable)
			return nil
		}
	}
	predictScaledCompoundRefToConvBufGeneric(out, ref, bytesPerSample, bitDepth, width, height,
		startX, xStep, startY, yStep, xTable, yTable, imH, scratch)
	return nil
}

func predictScaledCompoundRefToConvBufIdentityStep(out []uint16, ref frame.Plane, bytesPerSample int, bitDepth uint8, width int, height int,
	startX int64, startY int64, xTable SubpelKernelTable, yTable SubpelKernelTable, scratch *ScaledConvolveScratch) bool {
	refX, refY, subX, subY, ok := scaledIdentityStepOrigin(startX, startY)
	if !ok {
		return false
	}
	xKernel := xTable[subX]
	yKernel := yTable[subY]
	round0 := compoundRound0(bitDepth)
	offsetBits := int(bitDepth) + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	if bytesPerSample == 1 {
		predictInterCompoundRef8ToConvBuf(out, ref, refX, refY, width, height, subX, subY, xKernel, yKernel, round0, offsetBits, roundOffset, nil)
		return true
	}
	if subX != 0 && subY != 0 && scratch == nil {
		return false
	}
	var compoundScratch *CompoundConvolveScratch
	if scratch != nil {
		compoundScratch = &scratch.compound
	}
	predictInterCompoundRefHighBDToConvBuf(out, ref, int(bitDepth), refX, refY, width, height, subX, subY, xKernel, yKernel, round0, offsetBits, roundOffset, compoundScratch)
	return true
}

func predictScaledCompoundRef8ToConvBufYIdentity(out []uint16, ref frame.Plane, width int, height int,
	startX int64, xStep int64, startY int64, xTable SubpelKernelTable) {
	const roundOffset = (1 << (19 - compoundRound1Bits)) + (1 << (19 - compoundRound1Bits - 1))
	foX := filterTaps/2 - 1
	baseY := int(scaledIntFloor(startY))
	for y := range height {
		rowBase := clampInt(baseY+y, 0, ref.Height-1) * ref.Stride
		xPos := startX
		outRow := out[y*width:]
		for x := range width {
			xInt := int(scaledIntFloor(xPos)) - foX
			xFilterIdx := int(scaledSubpel(xPos) >> ScaleExtraBits)
			kernel := xTable[xFilterIdx]
			k0, k1, k2, k3 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3])
			k4, k5, k6, k7 := int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
			sum := 0
			if xInt >= 0 && xInt+filterTaps <= ref.Width {
				src := ref.Pix[rowBase+xInt : rowBase+xInt+filterTaps : rowBase+xInt+filterTaps]
				_ = src[7]
				sum = k0*int(src[0]) + k1*int(src[1]) + k2*int(src[2]) + k3*int(src[3]) +
					k4*int(src[4]) + k5*int(src[5]) + k6*int(src[6]) + k7*int(src[7])
			} else {
				sum = k0*int(loadSample8ClampedRow(ref, xInt, rowBase)) +
					k1*int(loadSample8ClampedRow(ref, xInt+1, rowBase)) +
					k2*int(loadSample8ClampedRow(ref, xInt+2, rowBase)) +
					k3*int(loadSample8ClampedRow(ref, xInt+3, rowBase)) +
					k4*int(loadSample8ClampedRow(ref, xInt+4, rowBase)) +
					k5*int(loadSample8ClampedRow(ref, xInt+5, rowBase)) +
					k6*int(loadSample8ClampedRow(ref, xInt+6, rowBase)) +
					k7*int(loadSample8ClampedRow(ref, xInt+7, rowBase))
			}
			outRow[x] = uint16(roundPowerOfTwo3(sum) + roundOffset)
			xPos += xStep
		}
	}
}

func predictScaledCompoundRefHighBDToConvBufYIdentity(out []uint16, ref frame.Plane, bitDepth int, width int, height int,
	startX int64, xStep int64, startY int64, xTable SubpelKernelTable) {
	round0 := compoundRound0(uint8(bitDepth))
	offsetBits := bitDepth + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	foX := filterTaps/2 - 1
	baseY := int(scaledIntFloor(startY))
	round0Bias := 1 << (round0 - 1)
	for y := range height {
		rowBase := clampInt(baseY+y, 0, ref.Height-1) * ref.Stride
		xPos := startX
		outRow := out[y*width:]
		for x := range width {
			xInt := int(scaledIntFloor(xPos)) - foX
			xFilterIdx := int(scaledSubpel(xPos) >> ScaleExtraBits)
			kernel := xTable[xFilterIdx]
			k0, k1, k2, k3 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3])
			k4, k5, k6, k7 := int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
			sum := 0
			if xInt >= 0 && xInt+filterTaps <= ref.Width {
				src := ref.Pix[rowBase+xInt*2 : rowBase+(xInt+filterTaps)*2 : rowBase+(xInt+filterTaps)*2]
				sum = k0*int(uint16(src[0])|uint16(src[1])<<8) +
					k1*int(uint16(src[2])|uint16(src[3])<<8) +
					k2*int(uint16(src[4])|uint16(src[5])<<8) +
					k3*int(uint16(src[6])|uint16(src[7])<<8) +
					k4*int(uint16(src[8])|uint16(src[9])<<8) +
					k5*int(uint16(src[10])|uint16(src[11])<<8) +
					k6*int(uint16(src[12])|uint16(src[13])<<8) +
					k7*int(uint16(src[14])|uint16(src[15])<<8)
			} else {
				sum = k0*int(loadHighBDSampleClamped(ref, xInt, baseY+y)) +
					k1*int(loadHighBDSampleClamped(ref, xInt+1, baseY+y)) +
					k2*int(loadHighBDSampleClamped(ref, xInt+2, baseY+y)) +
					k3*int(loadHighBDSampleClamped(ref, xInt+3, baseY+y)) +
					k4*int(loadHighBDSampleClamped(ref, xInt+4, baseY+y)) +
					k5*int(loadHighBDSampleClamped(ref, xInt+5, baseY+y)) +
					k6*int(loadHighBDSampleClamped(ref, xInt+6, baseY+y)) +
					k7*int(loadHighBDSampleClamped(ref, xInt+7, baseY+y))
			}
			outRow[x] = uint16(((sum + round0Bias) >> round0) + roundOffset)
			xPos += xStep
		}
	}
}

func predictScaledCompoundRefToConvBufGeneric(out []uint16, ref frame.Plane, bytesPerSample int, bitDepth uint8, width int, height int,
	startX int64, xStep int64, startY int64, yStep int64,
	xTable SubpelKernelTable, yTable SubpelKernelTable, imH int, scratch *ScaledConvolveScratch) {
	const imStride = maxBlockSize
	im, pooled := scaledHighBDIMForScratch(scratch)
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	startRow := int(scaledIntFloor(startY)) - foY
	round0 := compoundRound0(bitDepth)
	for y := range imH {
		srcRow := startRow + y
		xPos := startX
		for x := range width {
			xInt := int(scaledIntFloor(xPos)) - foX
			xFilterIdx := int(scaledSubpel(xPos) >> ScaleExtraBits)
			kernel := xTable[xFilterIdx]
			sum := 1 << (int(bitDepth) + filterBits - 1)
			for k := range filterTaps {
				var sample int
				if bytesPerSample == 1 {
					sample = int(loadSample8Clamped(ref, xInt+k, srcRow))
				} else {
					sample = int(loadHighBDSampleClamped(ref, xInt+k, srcRow))
				}
				sum += int(kernel[k]) * sample
			}
			im[y*imStride+x] = int32(roundPowerOfTwo(sum, round0))
			xPos += xStep
		}
	}

	offsetBits := int(bitDepth) + 2*filterBits - round0
	baseY := int(scaledIntFloor(startY))
	for x := range width {
		yPos := startY
		for y := range height {
			yInt := int(scaledIntFloor(yPos)) - baseY
			yFilterIdx := int(scaledSubpel(yPos) >> ScaleExtraBits)
			kernel := yTable[yFilterIdx]
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(kernel[k]) * int(im[(yInt+k)*imStride+x])
			}
			out[y*width+x] = uint16(roundPowerOfTwo(sum, compoundRound1Bits))
			yPos += yStep
		}
	}
	putScaledHighBDIM(im, pooled)
}

// PredictWarpedCompoundToConvBuf fills buf with the un-rounded 16-bit CONV_BUF
// warp predictor for one compound reference (av1_warp_affine_c with
// is_compound && !do_average). It mirrors the warp filter in warp.go but emits
// the intermediate value at COMPOUND_ROUND1_BITS precision instead of the final
// clipped pixel, so the compound blend can round once after combining the two
// references. matX/matY are the block's true plane coordinates (libaom
// p_col/p_row); the result is written into buf at (0,0).
func PredictWarpedCompoundToConvBuf(buf *CompoundConvBuf, ref frame.Plane, bytesPerSample int, bitDepth uint8, matX int, matY int, width int, height int, matrix [6]int32, alpha int16, beta int16, gamma int16, delta int16, subsamplingX bool, subsamplingY bool) error {
	if buf == nil || width <= 0 || height <= 0 || width > maxBlockSize || height > maxBlockSize {
		return ErrInvalidMotion
	}
	if !planeRegionFits(ref, bytesPerSample, 0, 0, ref.Width, ref.Height) ||
		!bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
		return ErrInvalidMotion
	}
	out, ok := compoundConvBufView(buf, width, height)
	if !ok {
		return ErrInvalidMotion
	}
	ssX := 0
	if subsamplingX {
		ssX = 1
	}
	ssY := 0
	if subsamplingY {
		ssY = 1
	}
	bd := int(bitDepth)
	a, be, g, d := int(alpha), int(beta), int(gamma), int(delta)
	// av1_highbd_warp_affine_c with is_compound: reduce_bits_horiz = round_0
	// (bd-dependent, 5 at bd == 12) and reduce_bits_vert = round_1
	// (COMPOUND_ROUND1_BITS, unchanged for compound).
	reduceBitsHoriz := compoundRound0(bitDepth)
	reduceBitsVert := compoundRound1Bits
	offsetBitsHoriz := bd + filterBits - 1
	offsetBitsVert := bd + 2*filterBits - reduceBitsHoriz
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	for i := matY; i < matY+height; i += 8 {
		for j := matX; j < matX+width; j += 8 {
			var baseSY int
			if bytesPerSample == 1 {
				baseSY = warpHorizontal8(&tmp, ref, i, j, matrix, a, be, g, d, ssX, ssY, reduceBitsHoriz, offsetBitsHoriz)
			} else {
				baseSY = warpHorizontalHighBD(&tmp, ref, i, j, matrix, a, be, g, d, ssX, ssY, reduceBitsHoriz, offsetBitsHoriz)
			}
			for k := -4; k < minWarpInt(4, matY+height-i-4); k++ {
				sy := baseSY + d*(k+4)
				for l := -4; l < minWarpInt(4, matX+width-j-4); l++ {
					offs := roundPowerOfTwo(sy, warpedDiffPrecBits) + warpedPixelPrecShifts
					if offs < 0 || offs >= len(warpedFilter) {
						sy += g
						continue
					}
					coeffs := warpedFilter[offs]
					sum := 1 << offsetBitsVert
					for m := range filterTaps {
						sum += int(coeffs[m]) * int(tmp[(k+m+4)*warpedIntermediateColumns+(l+4)])
					}
					sum = roundPowerOfTwo(sum, reduceBitsVert)
					out[(i-matY+k+4)*width+(j-matX+l+4)] = uint16(sum)
					sy += g
				}
			}
		}
	}
	return nil
}

// BlendCompoundAvg blends two CONV_BUF predictors with distance weights
// (fwdOffset/bckOffset; both 8 for a plain average) and writes the final pixels
// (av1_dist_wtd_convolve_* do_average branch).
func BlendCompoundAvg(dst frame.Plane, buf0 *CompoundConvBuf, buf1 *CompoundConvBuf, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, fwdOffset int, bckOffset int) error {
	if buf0 == nil || buf1 == nil ||
		int(buf0.Width) != width || int(buf0.Height) != height ||
		int(buf1.Width) != width || int(buf1.Height) != height {
		return ErrInvalidMotion
	}
	if !planeRegionFits(dst, bytesPerSample, dstX, dstY, width, height) ||
		!bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
		return ErrInvalidMotion
	}
	bd := int(bitDepth)
	round0 := compoundRound0(bitDepth)
	offsetBits := bd + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	roundBits := 2*filterBits - round0 - compoundRound1Bits
	src0 := buf0.Data[:width*height]
	src1 := buf1.Data[:width*height]
	if bytesPerSample == 1 {
		blendCompoundAvg8Impl(dst, src0, src1, dstX, dstY, width, height, fwdOffset, bckOffset, roundOffset, roundBits)
		return nil
	}
	max, _ := highBDMax(bitDepth)
	blendCompoundAvgHighBD(dst, src0, src1, max, dstX, dstY, width, height, fwdOffset, bckOffset, roundOffset, roundBits)
	return nil
}

// BlendCompoundMaskD16 blends two CONV_BUF predictors through a per-pixel A64
// soft mask (wedge / diff-weighted) at the d16 precision and writes the final
// pixels (aom_*_blend_a64_d16_mask_c).
func BlendCompoundMaskD16(dst frame.Plane, buf0 *CompoundConvBuf, buf1 *CompoundConvBuf, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mask []byte, maskStride int, subX bool, subY bool) error {
	if buf0 == nil || buf1 == nil ||
		int(buf0.Width) != width || int(buf0.Height) != height ||
		int(buf1.Width) != width || int(buf1.Height) != height {
		return ErrInvalidMotion
	}
	if !planeRegionFits(dst, bytesPerSample, dstX, dstY, width, height) ||
		!bitDepthMatchesSampleWidth(bytesPerSample, bitDepth) {
		return ErrInvalidMotion
	}
	bd := int(bitDepth)
	round0 := compoundRound0(bitDepth)
	offsetBits := bd + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	roundBits := 2*filterBits - round0 - compoundRound1Bits
	src0 := buf0.Data[:width*height]
	src1 := buf1.Data[:width*height]
	max := uint16(0)
	if bytesPerSample != 1 {
		max, _ = highBDMax(bitDepth)
	}
	for y := range height {
		for x := range width {
			m, ok := compoundMaskSample(mask, maskStride, y, x, subX, subY)
			if !ok {
				return ErrInvalidMotion
			}
			i := y*width + x
			res := (m*int(src0[i]) + (compoundBlendA64MaxAlpha-m)*int(src1[i])) >> compoundBlendA64RoundBits
			res -= roundOffset
			res = roundPowerOfTwo(res, roundBits)
			if bytesPerSample == 1 {
				dst.Pix[(dstY+y)*dst.Stride+dstX+x] = clipPixel(res)
			} else {
				storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(res, max))
			}
		}
	}
	return nil
}

// BuildDiffWtdMaskD16 builds the difference-weighted compound mask from two
// CONV_BUF predictors (av1_build_compound_diffwtd_mask_d16_c).
func BuildDiffWtdMaskD16(mask []byte, maskStride int, buf0 *CompoundConvBuf, buf1 *CompoundConvBuf, bitDepth uint8, width int, height int, invert bool) error {
	if buf0 == nil || buf1 == nil ||
		int(buf0.Width) != width || int(buf0.Height) != height ||
		int(buf1.Width) != width || int(buf1.Height) != height {
		return ErrInvalidMotion
	}
	if maskStride < width || len(mask) < (height-1)*maskStride+width {
		return ErrInvalidMotion
	}
	bd := int(bitDepth)
	round := 2*filterBits - compoundRound0(bitDepth) - compoundRound1Bits + (bd - 8)
	const maskBase = 38
	const diffFactor = 16
	src0 := buf0.Data[:width*height]
	src1 := buf1.Data[:width*height]
	for y := range height {
		for x := range width {
			i := y*width + x
			diff := int(src0[i]) - int(src1[i])
			if diff < 0 {
				diff = -diff
			}
			diff = roundPowerOfTwo(diff, round)
			m := max(maskBase+diff/diffFactor, 0)
			if m > compoundBlendA64MaxAlpha {
				m = compoundBlendA64MaxAlpha
			}
			if invert {
				m = compoundBlendA64MaxAlpha - m
			}
			mask[y*maskStride+x] = byte(m)
		}
	}
	return nil
}

var blendCompoundAvg8Impl = blendCompoundAvg8PureGo

func blendCompoundAvg8PureGo(dst frame.Plane, src0 []uint16, src1 []uint16, dstX int, dstY int, width int, height int, fwdOffset int, bckOffset int, roundOffset int, roundBits int) {
	for y := range height {
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		srcOff := y * width
		for x := range width {
			i := srcOff + x
			tmp := int(src0[i])*fwdOffset + int(src1[i])*bckOffset
			tmp >>= 4 // DIST_PRECISION_BITS
			tmp -= roundOffset
			dstRow[x] = clipPixel(roundPowerOfTwo(tmp, roundBits))
		}
	}
}

func blendCompoundAvgHighBD(dst frame.Plane, src0 []uint16, src1 []uint16, max uint16, dstX int, dstY int, width int, height int, fwdOffset int, bckOffset int, roundOffset int, roundBits int) {
	for y := range height {
		for x := range width {
			i := y*width + x
			tmp := int(src0[i])*fwdOffset + int(src1[i])*bckOffset
			tmp >>= 4 // DIST_PRECISION_BITS
			tmp -= roundOffset
			storeHighBDSample(dst, dstX+x, dstY+y, clipPixelHighBD(roundPowerOfTwo(tmp, roundBits), max))
		}
	}
}

// compoundMaskSample reads a (possibly subsampled) A64 mask sample, mirroring
// aom_*_blend_a64_d16_mask_c subw/subh handling.
func compoundMaskSample(mask []byte, stride int, row int, col int, subX bool, subY bool) (int, bool) {
	switch {
	case !subX && !subY:
		idx := row*stride + col
		if idx < 0 || idx >= len(mask) {
			return 0, false
		}
		return int(mask[idx]), true
	case subX && subY:
		base := (2 * row) * stride
		base2 := (2*row + 1) * stride
		if base2+2*col+1 >= len(mask) {
			return 0, false
		}
		sum := int(mask[base+2*col]) + int(mask[base2+2*col]) +
			int(mask[base+2*col+1]) + int(mask[base2+2*col+1])
		return roundPowerOfTwoCompound(sum, 2), true
	case subX:
		idx := row*stride + 2*col
		if idx+1 >= len(mask) {
			return 0, false
		}
		return (int(mask[idx]) + int(mask[idx+1]) + 1) >> 1, true
	default:
		base := (2 * row) * stride
		base2 := (2*row + 1) * stride
		if base2+col >= len(mask) {
			return 0, false
		}
		return (int(mask[base+col]) + int(mask[base2+col]) + 1) >> 1, true
	}
}

func roundPowerOfTwoCompound(value int, bits int) int {
	return (value + (1 << (bits - 1))) >> bits
}
