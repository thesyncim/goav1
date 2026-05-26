// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"os"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

// frameWorkScaledRefAllowedEnv toggles routing scaled-reference inter blocks
// through the new scaled 8-tap convolver. When unset, the threading
// prediction path keeps rejecting size-mismatched references, leaving the
// 7-vector fast suite (which has no SVC inputs) on its battle-tested
// same-size path. Set GOAV1_SCALED_PRED=1 to opt in.
const frameWorkScaledRefAllowedEnv = "GOAV1_SCALED_PRED"

// frameWorkScaledRefEnabled returns true when the build was compiled with the
// goav1_scaled_pred tag or when the runtime environment variable opts in.
// Both routes exist so the conformance-driven scaled-pred path can be enabled
// without recompilation while still letting the build tag pin it on for
// developer workflows that rely on -tags.
func frameWorkScaledRefEnabled() bool {
	if frameWorkScaledRefBuildEnabled {
		return true
	}
	return os.Getenv(frameWorkScaledRefAllowedEnv) != ""
}

// frameWorkSameOrScaledReferencePlane validates a reference plane against
// geom for either same-size sampling or, when the scaled-prediction path is
// enabled, the AV1 scale-factor range. It mirrors libaom's
// av1_setup_scale_factors_for_frame checks via motion.NewScaleFactors.
//
// Returns (true, nil) if the reference is same-size; (false, nil) if a
// non-identity scaling is acceptable for the scaled convolver; or an error if
// the reference dimensions are unusable.
func frameWorkSameOrScaledReferencePlane(geom frameWorkPredictionPlaneGeometry, ref frame.Plane) (bool, error) {
	if ref.Width == geom.Output.Width && ref.Height == geom.Output.Height {
		return true, nil
	}
	if !frameWorkScaledRefEnabled() {
		return false, ErrInvalidBatch
	}
	if _, err := motion.NewScaleFactors(ref.Width, ref.Height, geom.Output.Width, geom.Output.Height); err != nil {
		return false, ErrInvalidBatch
	}
	return false, nil
}

// frameWorkPredictScaledReferencePlane writes one plane block from a smaller
// or larger reference into dst at (dstX, dstY). It is used only when the
// scaled-prediction path is enabled and the reference dimensions differ from
// the output dimensions (verified by frameWorkSameOrScaledReferencePlane).
// dst is expected to be the output plane, so its Width / Height supply the
// curWidth / curHeight used for the Q14 reference scale factors. Callers
// staging into a block-sized scratch buffer (inter-intra, masked compound)
// must use frameWorkPredictScaledReferencePlaneToBuffer so the scale factors
// stay anchored to the output frame (libaom's xd->block_ref_scale_factors).
func frameWorkPredictScaledReferencePlane(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8,
	dstX int, dstY int, blockX int, blockY int, width int, height int, mv motion.Vector,
	subsamplingX bool, subsamplingY bool, filters motion.InterpFilters) error {
	return frameWorkPredictScaledReferencePlaneWithDims(dst, ref, dst.Width, dst.Height, bytesPerSample, bitDepth,
		dstX, dstY, blockX, blockY, width, height, mv, subsamplingX, subsamplingY, filters)
}

// frameWorkPredictScaledReferencePlaneToBuffer is the inter-intra / masked
// compound staging form of the scaled-reference dispatch. The dst plane is a
// block-sized scratch buffer, so its Width / Height are not the output frame
// dimensions; the helper lifts the actual output plane size out of geom
// (geom.Output.Width / Height) before computing libaom's Q14 scale factors.
//
// libaom mirror: av1_make_inter_predictor() in av1/common/reconinter.c picks
// up scale factors from xd->block_ref_scale_factors, which are set per-frame
// by av1_setup_scale_factors_for_frame() from the output frame dimensions —
// independent of whatever buffer the inter prediction is staged into.
func frameWorkPredictScaledReferencePlaneToBuffer(dst frame.Plane, ref frame.Plane,
	geom frameWorkPredictionPlaneGeometry, bitDepth uint8,
	dstX int, dstY int, blockX int, blockY int, mv motion.Vector, filters motion.InterpFilters) error {
	return frameWorkPredictScaledReferencePlaneWithDims(dst, ref, geom.Output.Width, geom.Output.Height,
		geom.BytesPerSample, bitDepth, dstX, dstY, blockX, blockY, geom.Width, geom.Height, mv,
		geom.SubsamplingX, geom.SubsamplingY, filters)
}

// frameWorkPredictScaledReferencePlaneWithDims is the explicit-output-dims
// form of the scaled-reference dispatch. curWidth / curHeight must be the
// output frame plane dimensions (not the destination buffer dimensions),
// matching libaom's xd->block_ref_scale_factors derivation.
func frameWorkPredictScaledReferencePlaneWithDims(dst frame.Plane, ref frame.Plane, curWidth int, curHeight int,
	bytesPerSample int, bitDepth uint8,
	dstX int, dstY int, blockX int, blockY int, width int, height int, mv motion.Vector,
	subsamplingX bool, subsamplingY bool, filters motion.InterpFilters) error {
	sf, err := motion.NewScaleFactors(ref.Width, ref.Height, curWidth, curHeight)
	if err != nil {
		return ErrInvalidBatch
	}
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(blockX, blockY, mv, subsamplingX, subsamplingY)
	if err != nil {
		return ErrInvalidBatch
	}
	xTable, err := motion.SubpelKernelTableFor(filters.X, width)
	if err != nil {
		return ErrInvalidBatch
	}
	yTable, err := motion.SubpelKernelTableFor(filters.Y, height)
	if err != nil {
		return ErrInvalidBatch
	}
	switch bytesPerSample {
	case 1:
		if err := motion.ConvolveScale2D8Clamped(dst, ref, dstX, dstY, width, height, startX, xStep, startY, yStep, xTable, yTable); err != nil {
			return ErrInvalidBatch
		}
	case 2:
		if err := motion.ConvolveScale2DHighBDClamped(dst, ref, bitDepth, dstX, dstY, width, height, startX, xStep, startY, yStep, xTable, yTable); err != nil {
			return ErrInvalidBatch
		}
	default:
		return ErrInvalidBatch
	}
	return nil
}
