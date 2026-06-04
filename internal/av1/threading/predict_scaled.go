// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"os"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

// frameWorkScaledRefDisabledEnv force-disables routing scaled-reference inter
// blocks through the scaled 8-tap convolver. Scaled-reference prediction is now
// on by default (libaom: av1_make_inter_predictor() routes through
// av1_convolve_2d_scale_c whenever av1_is_scaled(sf)), so a size-mismatched
// reference - e.g. an SVC enhancement layer (spatial>0) referencing a smaller
// base-layer frame - is handled instead of crashing with ErrInvalidBatch. Set
// GOAV1_SCALED_PRED=0 to fall back to the legacy same-size-only reject (used
// only for debugging / bisecting the scaled path). The same-size fast path is
// unaffected because frameWorkSameOrScaledReferencePlane short-circuits on
// equal dimensions before consulting this gate at all.
const frameWorkScaledRefDisabledEnv = "GOAV1_SCALED_PRED"

// frameWorkScaledRefEnabled reports whether the scaled-reference convolver is
// active. It is unconditionally on unless explicitly disabled with
// GOAV1_SCALED_PRED=0; the goav1_scaled_pred build tag remains a no-op kept for
// source compatibility.
func frameWorkScaledRefEnabled() bool {
	if frameWorkScaledRefBuildEnabled {
		return true
	}
	return os.Getenv(frameWorkScaledRefDisabledEnv) != "0"
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
	// Compare against the current frame's CODED (cropped) plane dimensions, not
	// geom.Output.Width/Height: the predictor's output plane is extended to the
	// MI-aligned write extent (frameWorkExtendPlaneToClip), which can exceed the
	// coded dimensions for the bottom/right partial superblock. libaom keys the
	// reference same-size / scale-factor decision off the coded frame
	// dimensions (av1_setup_scale_factors_for_frame), so a partial-superblock
	// block must not be mistaken for a scaled-reference block.
	curWidth, curHeight, ok := geom.codedDimensions()
	if !ok {
		return false, ErrInvalidBatch
	}
	if ref.Width == curWidth && ref.Height == curHeight {
		return true, nil
	}
	if !frameWorkScaledRefEnabled() {
		return false, ErrInvalidBatch
	}
	if _, err := motion.NewScaleFactors(ref.Width, ref.Height, curWidth, curHeight); err != nil {
		return false, ErrInvalidBatch
	}
	return false, nil
}

// frameWorkPredictScaledReferencePlane writes one plane block from a smaller
// or larger reference into geom.Output at (dstX, dstY). It is used only when the
// scaled-prediction path is enabled and the reference dimensions differ from
// the current coded frame dimensions (verified by
// frameWorkSameOrScaledReferencePlane). The Q14 reference scale factors are
// anchored to the current frame's CODED (cropped) plane dimensions
// (geom.CodedWidth / geom.CodedHeight), mirroring libaom's
// av1_setup_scale_factors_for_frame(): the scale factors are derived from the
// coded frame size, never from the MI/SB-aligned write extent that
// geom.Output.Width / Height span. Using the extended Output height here would
// shrink y_scale_fp / y_step (e.g. 720 -> 768 on the SVC L2T1 enhancement
// layer) and walk the vertical reference taps off by a growing per-row offset.
// Callers staging into a block-sized scratch buffer (inter-intra, masked
// compound) must use frameWorkPredictScaledReferencePlaneToBuffer.
func frameWorkPredictScaledReferencePlane(geom frameWorkPredictionPlaneGeometry, ref frame.Plane, bytesPerSample int, bitDepth uint8,
	dstX int, dstY int, blockX int, blockY int, width int, height int, mv motion.Vector,
	subsamplingX bool, subsamplingY bool, filters motion.InterpFilters) error {
	return frameWorkPredictScaledReferencePlaneWithFilterSize(geom, ref, bytesPerSample, bitDepth,
		dstX, dstY, blockX, blockY, width, height, width, height, mv, subsamplingX, subsamplingY, filters)
}

func frameWorkPredictScaledReferencePlaneWithFilterSize(geom frameWorkPredictionPlaneGeometry, ref frame.Plane, bytesPerSample int, bitDepth uint8,
	dstX int, dstY int, blockX int, blockY int, width int, height int, filterWidth int, filterHeight int, mv motion.Vector,
	subsamplingX bool, subsamplingY bool, filters motion.InterpFilters) error {
	return frameWorkPredictScaledReferencePlaneWithFilterSizeScratch(geom, ref, bytesPerSample, bitDepth,
		dstX, dstY, blockX, blockY, width, height, filterWidth, filterHeight, mv, subsamplingX, subsamplingY, filters, nil)
}

func frameWorkPredictScaledReferencePlaneWithFilterSizeScratch(geom frameWorkPredictionPlaneGeometry, ref frame.Plane, bytesPerSample int, bitDepth uint8,
	dstX int, dstY int, blockX int, blockY int, width int, height int, filterWidth int, filterHeight int, mv motion.Vector,
	subsamplingX bool, subsamplingY bool, filters motion.InterpFilters, scratch *motion.ScaledConvolveScratch) error {
	curWidth, curHeight, ok := frameWorkScaledReferenceCurrentDims(geom)
	if !ok {
		return ErrInvalidBatch
	}
	return frameWorkPredictScaledReferencePlaneWithDimsAndFilterSizeScratch(geom.Output, ref, curWidth, curHeight, bytesPerSample, bitDepth,
		dstX, dstY, blockX, blockY, width, height, filterWidth, filterHeight, mv, subsamplingX, subsamplingY, filters, scratch)
}

// frameWorkScaledReferenceCurrentDims returns the current frame's coded
// (cropped) plane dimensions used to derive the Q14 reference scale factors,
// falling back to the Output plane dimensions only when the coded extent was
// not recorded. It mirrors frameWorkSameOrScaledReferencePlane's curWidth /
// curHeight selection so the same-size / scaled decision and the scale-factor
// computation stay consistent.
func frameWorkScaledReferenceCurrentDims(geom frameWorkPredictionPlaneGeometry) (int, int, bool) {
	return geom.codedDimensions()
}

// frameWorkPredictScaledReferencePlaneToBuffer is the inter-intra / masked
// compound staging form of the scaled-reference dispatch. The dst plane is a
// block-sized scratch buffer, so its Width / Height are not the current frame
// dimensions; the helper lifts the current frame's coded plane size out of geom
// (geom.CodedWidth / CodedHeight) before computing libaom's Q14 scale factors.
//
// libaom mirror: av1_make_inter_predictor() in av1/common/reconinter.c picks
// up scale factors from xd->block_ref_scale_factors, which are set per-frame
// by av1_setup_scale_factors_for_frame() from the output frame dimensions —
// independent of whatever buffer the inter prediction is staged into.
func frameWorkPredictScaledReferencePlaneToBuffer(dst frame.Plane, ref frame.Plane,
	geom frameWorkPredictionPlaneGeometry, bitDepth uint8,
	dstX int, dstY int, blockX int, blockY int, mv motion.Vector, filters motion.InterpFilters) error {
	return frameWorkPredictScaledReferencePlaneToBufferWithFilterSize(dst, ref, geom, bitDepth,
		dstX, dstY, blockX, blockY, geom.width(), geom.height(), mv, filters)
}

func frameWorkPredictScaledReferencePlaneToBufferWithFilterSize(dst frame.Plane, ref frame.Plane,
	geom frameWorkPredictionPlaneGeometry, bitDepth uint8,
	dstX int, dstY int, blockX int, blockY int, filterWidth int, filterHeight int, mv motion.Vector, filters motion.InterpFilters) error {
	return frameWorkPredictScaledReferencePlaneToBufferWithFilterSizeScratch(dst, ref, geom, bitDepth,
		dstX, dstY, blockX, blockY, filterWidth, filterHeight, mv, filters, nil)
}

func frameWorkPredictScaledReferencePlaneToBufferWithFilterSizeScratch(dst frame.Plane, ref frame.Plane,
	geom frameWorkPredictionPlaneGeometry, bitDepth uint8,
	dstX int, dstY int, blockX int, blockY int, filterWidth int, filterHeight int, mv motion.Vector, filters motion.InterpFilters,
	scratch *motion.ScaledConvolveScratch) error {
	curWidth, curHeight, ok := frameWorkScaledReferenceCurrentDims(geom)
	if !ok {
		return ErrInvalidBatch
	}
	return frameWorkPredictScaledReferencePlaneWithDimsAndFilterSizeScratch(dst, ref, curWidth, curHeight,
		geom.bytesPerSample(), bitDepth, dstX, dstY, blockX, blockY, geom.width(), geom.height(),
		filterWidth, filterHeight, mv, geom.SubsamplingX, geom.SubsamplingY, filters, scratch)
}

func frameWorkPredictScaledReferencePlaneToConvBuf(buf *motion.CompoundConvBuf, ref frame.Plane,
	geom frameWorkPredictionPlaneGeometry, bitDepth uint8, mv motion.Vector, filters motion.InterpFilters) error {
	return frameWorkPredictScaledReferencePlaneToConvBufScratch(buf, ref, geom, bitDepth, mv, filters, nil)
}

func frameWorkPredictScaledReferencePlaneToConvBufScratch(buf *motion.CompoundConvBuf, ref frame.Plane,
	geom frameWorkPredictionPlaneGeometry, bitDepth uint8, mv motion.Vector, filters motion.InterpFilters,
	scratch *motion.ScaledConvolveScratch) error {
	curWidth, curHeight, ok := frameWorkScaledReferenceCurrentDims(geom)
	if !ok {
		return ErrInvalidBatch
	}
	sf, err := motion.NewScaleFactors(ref.Width, ref.Height, curWidth, curHeight)
	if err != nil {
		return ErrInvalidBatch
	}
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(geom.X, geom.Y, mv, geom.SubsamplingX, geom.SubsamplingY)
	if err != nil {
		return ErrInvalidBatch
	}
	xTable, err := motion.SubpelKernelTableFor(filters.X, geom.width())
	if err != nil {
		return ErrInvalidBatch
	}
	yTable, err := motion.SubpelKernelTableFor(filters.Y, geom.height())
	if err != nil {
		return ErrInvalidBatch
	}
	if err := motion.PredictScaledCompoundRefToConvBufWithScratch(buf, ref, geom.bytesPerSample(), bitDepth,
		geom.width(), geom.height(), startX, xStep, startY, yStep, xTable, yTable, scratch); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

// frameWorkPredictScaledReferencePlaneWithDims is the explicit-output-dims
// form of the scaled-reference dispatch. curWidth / curHeight must be the
// output frame plane dimensions (not the destination buffer dimensions),
// matching libaom's xd->block_ref_scale_factors derivation.
func frameWorkPredictScaledReferencePlaneWithDims(dst frame.Plane, ref frame.Plane, curWidth int, curHeight int,
	bytesPerSample int, bitDepth uint8,
	dstX int, dstY int, blockX int, blockY int, width int, height int, mv motion.Vector,
	subsamplingX bool, subsamplingY bool, filters motion.InterpFilters) error {
	return frameWorkPredictScaledReferencePlaneWithDimsAndFilterSize(dst, ref, curWidth, curHeight,
		bytesPerSample, bitDepth, dstX, dstY, blockX, blockY, width, height, width, height, mv,
		subsamplingX, subsamplingY, filters)
}

func frameWorkPredictScaledReferencePlaneWithDimsAndFilterSize(dst frame.Plane, ref frame.Plane, curWidth int, curHeight int,
	bytesPerSample int, bitDepth uint8,
	dstX int, dstY int, blockX int, blockY int, width int, height int, filterWidth int, filterHeight int, mv motion.Vector,
	subsamplingX bool, subsamplingY bool, filters motion.InterpFilters) error {
	return frameWorkPredictScaledReferencePlaneWithDimsAndFilterSizeScratch(dst, ref, curWidth, curHeight, bytesPerSample, bitDepth,
		dstX, dstY, blockX, blockY, width, height, filterWidth, filterHeight, mv, subsamplingX, subsamplingY, filters, nil)
}

func frameWorkPredictScaledReferencePlaneWithDimsAndFilterSizeScratch(dst frame.Plane, ref frame.Plane, curWidth int, curHeight int,
	bytesPerSample int, bitDepth uint8,
	dstX int, dstY int, blockX int, blockY int, width int, height int, filterWidth int, filterHeight int, mv motion.Vector,
	subsamplingX bool, subsamplingY bool, filters motion.InterpFilters, scratch *motion.ScaledConvolveScratch) error {
	sf, err := motion.NewScaleFactors(ref.Width, ref.Height, curWidth, curHeight)
	if err != nil {
		return ErrInvalidBatch
	}
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(blockX, blockY, mv, subsamplingX, subsamplingY)
	if err != nil {
		return ErrInvalidBatch
	}
	xTable, err := motion.SubpelKernelTableFor(filters.X, filterWidth)
	if err != nil {
		return ErrInvalidBatch
	}
	yTable, err := motion.SubpelKernelTableFor(filters.Y, filterHeight)
	if err != nil {
		return ErrInvalidBatch
	}
	switch bytesPerSample {
	case 1:
		if err := motion.ConvolveScale2D8ClampedWithScratch(dst, ref, dstX, dstY, width, height, startX, xStep, startY, yStep, xTable, yTable, scratch); err != nil {
			return ErrInvalidBatch
		}
	case 2:
		if err := motion.ConvolveScale2DHighBDClampedWithScratch(dst, ref, bitDepth, dstX, dstY, width, height, startX, xStep, startY, yStep, xTable, yTable, scratch); err != nil {
			return ErrInvalidBatch
		}
	default:
		return ErrInvalidBatch
	}
	return nil
}
