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

var (
	predictPaethImpl            predictPaethFunc            = predictPaethPureGo
	predictSmoothImpl           predictSmoothFunc           = predictSmoothPureGo
	predictSmoothVerticalImpl   predictSmoothVerticalFunc   = predictSmoothVerticalPureGo
	predictSmoothHorizontalImpl predictSmoothHorizontalFunc = predictSmoothHorizontalPureGo
)
