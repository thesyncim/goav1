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

// subsampleLuma16Func is the high-bit-depth counterpart. It also preserves the
// scalar range check against max and returns ErrInvalidPrediction on overflow.
type subsampleLuma16Func func(outputQ3 []uint16, input []uint16, inputStride int, width int, height int, outW int, outH int, subX bool, subY bool, max uint16) error

// subtractCFLAverageFunc subtracts the rounded block average from Q3 luma
// samples. numPelLog2 is log2(width*height), computed by the validated public
// wrapper.
type subtractCFLAverageFunc func(srcQ3 []uint16, dstQ3 []int16, width int, height int, numPelLog2 int)

// dirRowInterp8Func fills a single 8-bit destination row for the directional
// Z1/Z3 interpolation. above is the edge pointer at the libaom origin; base is
// the starting reference index, advancing by 1 per column (the non-upsampled
// baseInc); shift is the constant per-row fractional phase in [0,31]. Columns
// whose base reaches maxBase clamp to above[maxBase]. SIMD variants must match
// roundPowerOfTwo(p0*(32-shift)+p1*shift,5) bit-for-bit.
type dirRowInterp8Func func(dst []byte, above []uint16, base int, shift int, maxBase int, width int)

// dirAboveRun8Func fills count contiguous 8-bit destination bytes for the
// zone-2 "above" branch: ref is positioned at the first source index, shift is
// the constant per-row fractional phase in [0,31], and the source index
// advances by 1 per output. There is no clamp (the reference span is
// pre-validated). SIMD variants must match
// roundPowerOfTwo(ref[i]*(32-shift)+ref[i+1]*shift,5) bit-for-bit.
type dirAboveRun8Func func(dst []byte, ref []uint16, shift int, count int)

// dirLeftCol8Func fills count 8-bit destination samples down a single zone-3
// column. ref is positioned at the first source index (advancing by 1 per row),
// shift is the constant per-column fractional phase, dst is the column's top
// sample and stride is the per-row byte step. No clamp (callers only pass the
// in-range row run). SIMD variants must match
// roundPowerOfTwo(ref[i]*(32-shift)+ref[i+1]*shift,5) bit-for-bit.
type dirLeftCol8Func func(dst []byte, stride int, ref []uint16, shift int, count int)

// predictFilterIntra8Func fills an 8-bit AV1 filter-intra block (see
// predictFilterIntraBlockDirect8). The predictor is the only intra predictor
// that lacked a SIMD slot; SIMD variants must reproduce the 7-tap filter,
// roundPowerOfTwo(sum, 4) rounding and [0,max] clamp sample-for-sample.
type predictFilterIntra8Func func(block planeBlock, width int, height int, mode FilterIntraMode, edges IntraEdges, max int)

// predictFilterIntra16Func is the 16-bit (high-bit-depth) counterpart. SIMD
// variants must reproduce the 7-tap filter, roundPowerOfTwo(sum, 4) rounding and
// [0,max] clamp sample-for-sample.
type predictFilterIntra16Func func(block planeBlock, width int, height int, mode FilterIntraMode, edges IntraEdges, max int)

var (
	predictPaethImpl            predictPaethFunc            = predictPaethPureGo
	predictSmoothImpl           predictSmoothFunc           = predictSmoothPureGo
	predictSmoothVerticalImpl   predictSmoothVerticalFunc   = predictSmoothVerticalPureGo
	predictSmoothHorizontalImpl predictSmoothHorizontalFunc = predictSmoothHorizontalPureGo

	sumSamplesImpl         sumSamplesFunc         = sumSamplesPureGo
	applyCFLImpl           applyCFLFunc           = applyCFLPureGo
	subsampleLuma8Impl     subsampleLuma8Func     = subsampleLuma8PureGo
	subsampleLuma16Impl    subsampleLuma16Func    = subsampleLuma16PureGo
	subtractCFLAverageImpl subtractCFLAverageFunc = subtractCFLAveragePureGo
	dirRowInterp8Impl      dirRowInterp8Func      = dirRowInterp8PureGo
	dirAboveRun8Impl       dirAboveRun8Func       = dirAboveRun8PureGo
	dirLeftCol8Impl        dirLeftCol8Func        = dirLeftCol8PureGo

	predictFilterIntra8Impl  predictFilterIntra8Func  = predictFilterIntraBlockDirect8
	predictFilterIntra16Impl predictFilterIntra16Func = predictFilterIntraBlockDirect16
)
