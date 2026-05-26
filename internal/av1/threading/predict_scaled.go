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
func frameWorkPredictScaledReferencePlane(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8,
	dstX int, dstY int, blockX int, blockY int, width int, height int, mv motion.Vector,
	subsamplingX bool, subsamplingY bool, filters motion.InterpFilters) error {
	sf, err := motion.NewScaleFactors(ref.Width, ref.Height, dst.Width, dst.Height)
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
