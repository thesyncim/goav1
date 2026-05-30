// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

// The convolve kernels below are the hottest part of inter prediction (~14-24%
// of single-worker decode). Each has a function-pointer dispatch slot that is
// bound exactly once at package init (see convolve_dispatch_*.go). On targets
// with a tuned SIMD variant the slot points at it; everywhere else it points at
// the pure-Go reference. The pure-Go *PureGo functions in vector.go are the
// canonical bit-exact reference: every SIMD variant MUST match them sample for
// sample, including the staged rounding and clipping.
//
// The slots cover only the non-edge-clamped 8-bit paths (convolveX8,
// convolveY8, convolve2D8). The clamped and high-bit-depth variants remain
// pure-Go.

type convolve1DFunc func(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, kernel [filterTaps]int16)

type convolve2DFunc func(dst frame.Plane, ref frame.Plane, dstX int, dstY int, refX int, refY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16)

var (
	convolveX8Impl  convolve1DFunc = convolveX8PureGo
	convolveY8Impl  convolve1DFunc = convolveY8PureGo
	convolve2D8Impl convolve2DFunc = convolve2D8PureGo
)
