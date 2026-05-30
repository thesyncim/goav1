// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package prediction

// The intra static predictors (PAETH and the three SMOOTH variants) run for
// every intra block and dominate intra-frame decode time. Each per-block fill
// loop has a function-pointer dispatch slot bound exactly once at package init
// (see intra_dispatch_*.go). On targets with a tuned SIMD variant the slot
// points at it; everywhere else it points at the pure-Go reference. The
// *PureGo functions in intra_static.go are the canonical bit-exact reference:
// every SIMD variant MUST match them sample for sample, including the rounding
// (divideRound) and the PAETH tie-breaking order.
//
// The NEON wrappers handle only the common 8-bit (bytesPerSample==1) path with
// width that is a multiple of 8; every other shape and bit depth falls back to
// the pure-Go reference inside the wrapper.

type predictPaethFunc func(block planeBlock, bytesPerSample int, above []uint16, left []uint16, aboveLeft uint16)

type predictSmoothFunc func(block planeBlock, bytesPerSample int, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16)

type predictSmoothVerticalFunc func(block planeBlock, bytesPerSample int, weights []uint16, above []uint16, belowPred uint16)

type predictSmoothHorizontalFunc func(block planeBlock, bytesPerSample int, weights []uint16, left []uint16, rightPred uint16)

// sumSamplesFunc sums a slice of uint16 edge samples. The DC predictor uses it
// for the above/left reductions that feed the block-average division. SIMD
// variants must return the exact 64-bit integer sum the pure-Go reference does.
type sumSamplesFunc func(samples []uint16) int

// applyCFLFunc applies the chroma-from-luma residual to a plane block that
// already holds the chroma DC prediction. acQ3 uses CFLBufLine stride. SIMD
// variants must reproduce roundPowerOfTwoSigned(alpha*ac,6) and the [0,max]
// clamp sample-for-sample (see PredictCFLPlaneBlockVisible).
type applyCFLFunc func(block planeBlock, bytesPerSample int, visibleWidth int, visibleHeight int, acQ3 []int16, alphaQ3 int, max uint16)

// subsampleLuma8Func subsamples an 8-bit luma block to Q3 (see
// SubsampleLuma8ToQ3). outW/outH are the post-subsample dimensions. SIMD
// variants must match the pure-Go reductions and shifts exactly.
type subsampleLuma8Func func(outputQ3 []uint16, input []uint8, inputStride int, width int, height int, outW int, outH int, subX bool, subY bool)

// dirRowInterp8Func fills a single 8-bit destination row for the directional
// Z1/Z3 interpolation. above is the edge pointer at the libaom origin; base is
// the starting reference index, advancing by 1 per column (the non-upsampled
// baseInc); shift is the constant per-row fractional phase in [0,31]. Columns
// whose base reaches maxBase clamp to above[maxBase]. SIMD variants must match
// roundPowerOfTwo(p0*(32-shift)+p1*shift,5) bit-for-bit.
type dirRowInterp8Func func(dst []byte, above []uint16, base int, shift int, maxBase int, width int)

var (
	predictPaethImpl            predictPaethFunc            = predictPaethPureGo
	predictSmoothImpl           predictSmoothFunc           = predictSmoothPureGo
	predictSmoothVerticalImpl   predictSmoothVerticalFunc   = predictSmoothVerticalPureGo
	predictSmoothHorizontalImpl predictSmoothHorizontalFunc = predictSmoothHorizontalPureGo

	sumSamplesImpl     sumSamplesFunc     = sumSamplesPureGo
	applyCFLImpl       applyCFLFunc       = applyCFLPureGo
	subsampleLuma8Impl subsampleLuma8Func = subsampleLuma8PureGo
	dirRowInterp8Impl  dirRowInterp8Func  = dirRowInterp8PureGo
)
