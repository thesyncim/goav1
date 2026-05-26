// Ported from libaom:
//   av1/common/scale.c
//   av1/common/scale.h
//   av1/common/reconinter.c (av1_make_inter_predictor scaled subpel-params setup)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package motion

// Constants matching AV1 spec §7.11.3.3 (motion_vector_scaling) and §7.11.3.4
// (block inter prediction). Mirrors libaom's REF_SCALE_SHIFT / SCALE_SUBPEL_BITS
// / SCALE_EXTRA_BITS.
const (
	// RefScaleShift is the Q14 fixed-point shift used for per-axis scale
	// factors derived from reference vs current frame dimensions.
	RefScaleShift = 14

	// ScaleSubpelBits is the Q10 sub-pel precision of the scaled
	// inter-prediction sample stepper (1 / 1024 sample units).
	ScaleSubpelBits = 10

	// ScaleSubpelScale is 2^ScaleSubpelBits and represents one full sample at
	// scaled-stepper precision.
	ScaleSubpelScale = 1 << ScaleSubpelBits

	// ScaleSubpelMask masks the integer part out of a Q10 stepper position.
	ScaleSubpelMask = ScaleSubpelScale - 1

	// ScaleExtraBits is the gap between the Q10 stepper position and the Q4
	// sub-pel filter index (ScaleSubpelBits - subpelQ4Bits = 6).
	ScaleExtraBits = ScaleSubpelBits - subpelQ4Bits

	// ScaleExtraOff rounds a Q10 stepper position to the nearest Q4 filter
	// index when reducing precision (mirrors libaom's SCALE_EXTRA_OFF).
	ScaleExtraOff = (1 << ScaleExtraBits) >> 1
)

// AV1 spec §6.8.6 (uncompressed_header semantics for frame_size_with_refs)
// caps scaled-reference prediction so the per-axis ratio stays within
// libaom's documented range:
//   - 2 * FrameSize >= RefFrameSize    (downscaling factor at most 2x)
//   - 16 * RefFrameSize >= FrameSize   (upscaling factor at most 16x)
const (
	// MaxUpscaleNumerator bounds RefSize * 16 >= CurSize.
	MaxUpscaleNumerator = 16
	// MaxDownscaleNumerator bounds CurSize * 2 >= RefSize.
	MaxDownscaleNumerator = 2
)

// ScaleFactors holds the per-axis Q14 reference scale factor and Q10 sample
// stepper for AV1 scaled inter prediction. Construction validates that the
// reference frame size is within AV1's allowed scale range relative to the
// current frame.
//
// XScaleFP / YScaleFP are the Q14 ratios (refSize << 14) / curSize, matching
// AV1 spec §7.11.3.3 XScale / YScale. XStepQN / YStepQN are the Q10 per-output-
// sample step, matching libaom's SubpelParams.xs / ys.
type ScaleFactors struct {
	// XScaleFP is RefScaleShift-precision (refWidth << 14) rounded by
	// half-curWidth, divided by curWidth.
	XScaleFP int32
	YScaleFP int32

	// XStepQN is ScaleSubpelBits-precision per-output-sample step:
	// xStep = ((refWidth << ScaleSubpelBits) + curWidth/2) / curWidth.
	XStepQN int32
	YStepQN int32

	// Identity reports whether the reference and current dimensions match,
	// in which case the same-size translational interpolator can be used.
	Identity bool
}

// NewScaleFactors computes the Q14 reference scale factors and Q10 sample
// stepper between a reference plane of refWidth x refHeight and a current
// plane of curWidth x curHeight. Mirrors libaom's av1_setup_scale_factors_for_
// frame() pre-checks: dimensions must be positive, and AV1 disallows
// downscaling beyond 1:2 or upscaling beyond 16:1 on either axis.
//
// Returns ErrInvalidMotion if any of the AV1 size constraints fail or if any
// intermediate would overflow int32 (a degenerate case for tiles much larger
// than spec-allowed maxima).
func NewScaleFactors(refWidth int, refHeight int, curWidth int, curHeight int) (ScaleFactors, error) {
	if refWidth <= 0 || refHeight <= 0 || curWidth <= 0 || curHeight <= 0 {
		return ScaleFactors{}, ErrInvalidMotion
	}
	if !validScaleRatio(refWidth, curWidth) || !validScaleRatio(refHeight, curHeight) {
		return ScaleFactors{}, ErrInvalidMotion
	}
	xScaleFP, ok := fixedPointScale(refWidth, curWidth, RefScaleShift)
	if !ok {
		return ScaleFactors{}, ErrInvalidMotion
	}
	yScaleFP, ok := fixedPointScale(refHeight, curHeight, RefScaleShift)
	if !ok {
		return ScaleFactors{}, ErrInvalidMotion
	}
	xStep, ok := fixedPointScale(refWidth, curWidth, ScaleSubpelBits)
	if !ok {
		return ScaleFactors{}, ErrInvalidMotion
	}
	yStep, ok := fixedPointScale(refHeight, curHeight, ScaleSubpelBits)
	if !ok {
		return ScaleFactors{}, ErrInvalidMotion
	}
	return ScaleFactors{
		XScaleFP: xScaleFP,
		YScaleFP: yScaleFP,
		XStepQN:  xStep,
		YStepQN:  yStep,
		Identity: refWidth == curWidth && refHeight == curHeight,
	}, nil
}

// ScaledBlockOrigin maps a destination block whose top-left lies at
// (dstX, dstY) in current-frame plane samples and references an integer-pel
// motion vector mv to the scaled reference-plane origin and Q10 stepper for
// AV1's scaled subpel interpolator.
//
// mv is the AV1-Q3 motion vector stored on the block (eighth-pel units).
// subsamplingX / subsamplingY toggle the chroma-subsampled axis convention
// mirroring ReferenceOriginSubsampled: luma multiplies the MV by 2 to reach
// Q4 prediction units; subsampled chroma axes keep the MV as-is.
//
// Returned (startX, startY) is the Q10 sub-pel position of the first output
// sample in the *reference plane*, and (xStep, yStep) the per-output-sample
// Q10 stride. Callers split startX / startY into:
//
//	intX     = startX >> ScaleSubpelBits
//	fracX_Q4 = (startX >> ScaleExtraBits) & subpelQ4Mask
//
// to feed an 8-tap scaled convolver (signatures listed in the package-level
// TODO below).
func (sf ScaleFactors) ScaledBlockOrigin(dstX int, dstY int, mv Vector, subsamplingX bool, subsamplingY bool) (startX int64, startY int64, xStep int64, yStep int64, err error) {
	if sf.XScaleFP <= 0 || sf.YScaleFP <= 0 || sf.XStepQN <= 0 || sf.YStepQN <= 0 {
		return 0, 0, 0, 0, ErrInvalidMotion
	}
	scaleX := int64(2)
	if subsamplingX {
		scaleX = 1
	}
	scaleY := int64(2)
	if subsamplingY {
		scaleY = 1
	}
	// Q4 source position (dst*16 + mv_Q4). This matches the same-size
	// referenceOriginQ4 convention; for scaled prediction we lift the result
	// to Q10 (the convolver stepper precision) before applying the Q14 scale.
	origX := int64(dstX)*subpelQ4Scale + int64(mv.Col)*scaleX
	origY := int64(dstY)*subpelQ4Scale + int64(mv.Row)*scaleY
	// Lift to Q10 source units: origQ10 = origQ4 << ScaleExtraBits. Then
	// scaledQ10 = (origQ10 * scaleFP + (1<<(RefScaleShift-1))) >> RefScaleShift
	// — half-up rounding mirrors libaom's av1_make_inter_predictor() scaled
	// position derivation (av1/common/reconinter.c).
	startX = scaleQ10Position(origX<<ScaleExtraBits, int64(sf.XScaleFP))
	startY = scaleQ10Position(origY<<ScaleExtraBits, int64(sf.YScaleFP))
	xStep = int64(sf.XStepQN)
	yStep = int64(sf.YStepQN)
	return startX, startY, xStep, yStep, nil
}

func scaleQ10Position(origQ10 int64, scaleFP int64) int64 {
	const half = int64(1) << (RefScaleShift - 1)
	prod := origQ10 * scaleFP
	if prod >= 0 {
		return (prod + half) >> RefScaleShift
	}
	// Symmetric round-half-away-from-zero for negatives, matching libaom's
	// ROUND_POWER_OF_TWO_SIGNED_64(value, n).
	return -((-prod + half) >> RefScaleShift)
}

// SplitScaledPosition decomposes a Q10 scaled-subpel position into its
// integer-pel and Q4 sub-pel-filter components, matching libaom's
// av1_make_inter_predictor() subpel split.
func SplitScaledPosition(posQN int64) (intPart int64, subpelQ4 int) {
	intPart = posQN >> ScaleSubpelBits
	subpelQ4 = int((posQN >> ScaleExtraBits) & subpelQ4Mask)
	return intPart, subpelQ4
}

// subpelQ4Scale is the integer multiplier that converts an integer-pel position
// to Q4 sub-pel-filter units (1 << subpelQ4Bits). Kept private to this file to
// avoid leaking the AV1-internal precision shift outside of motion's scaling
// math.
const subpelQ4Scale = int64(1) << subpelQ4Bits

func fixedPointScale(otherSize int, thisSize int, shift int) (int32, bool) {
	if otherSize <= 0 || thisSize <= 0 || shift < 0 || shift > 30 {
		return 0, false
	}
	// libaom: get_fixed_point_scale_factor:
	//   ((other << shift) + this/2) / this
	num := int64(otherSize)<<shift + int64(thisSize)/2
	if num < 0 {
		return 0, false
	}
	val := num / int64(thisSize)
	if val < 0 || val > int64(int32(^uint32(0)>>1)) {
		return 0, false
	}
	return int32(val), true
}

// validScaleRatio enforces AV1's per-axis reference-vs-current scale bounds
// from frame_size_with_refs(): a 1:16 upscale and a 2:1 downscale are the
// extremes. Both directions must hold.
func validScaleRatio(refSize int, curSize int) bool {
	if refSize <= 0 || curSize <= 0 {
		return false
	}
	// Upscale bound: ref*16 must reach cur.
	if int64(refSize)*MaxUpscaleNumerator < int64(curSize) {
		return false
	}
	// Downscale bound: cur*2 must reach ref.
	if int64(curSize)*MaxDownscaleNumerator < int64(refSize) {
		return false
	}
	return true
}

// TODO(scaled-subpel-kernel): Wire up an 8-tap scaled convolver and route
// scaled-reference prediction blocks through it. The same-size translational
// fast path (predictInterPlaneBlock8 / predictInterPlaneBlockHighBD) cannot
// step the source position by a non-unit stride, so a new family is required
// — matching libaom's av1_convolve_2d_scale_c / av1_highbd_convolve_2d_scale_c:
//
//   func convolve2DScale8(
//       dst frame.Plane, ref frame.Plane,
//       dstX, dstY int, width, height int,
//       startXQ10 int64, xStepQ10 int64,
//       startYQ10 int64, yStepQ10 int64,
//       xKernels [16][filterTaps]int16,
//       yKernels [16][filterTaps]int16,
//   )
//
//   func convolve2DScaleHighBD(... bitDepth uint8, max uint16, ...)
//
// (Plus *Clamped variants that extend the source plane edges so out-of-bounds
// taps replicate the boundary, as predictInterPlaneBlock8 does today.)
//
// Once those kernels are in, the predict.go validator can flip from "reject
// scaled refs" to "route scaled refs through the new path" behind a build/env
// guard so the existing 7-vector fast suite keeps the rejection until the
// scaled kernels are conformance-tested against libaom.
