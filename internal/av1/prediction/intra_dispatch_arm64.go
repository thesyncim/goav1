// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package prediction

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best intra static predictors on arm64. When NEON
// is available (mandatory on every arm64 chip Go runs on) the PAETH and SMOOTH
// fill loops route through the hand-written NEON asm; otherwise they keep the
// pure-Go reference. The assignment happens once, before any decoder goroutine
// starts, so the steady-state cost is a single indirect call.
//
// The NEON wrappers fall back to pure-Go for non-8-bit samples and widths that
// are not a multiple of 8, so the asm only handles the common shapes that
// dominate intra decode time.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		predictPaethImpl = predictPaethNEON
		predictSmoothImpl = predictSmoothNEON
		predictSmoothVerticalImpl = predictSmoothVerticalNEON
		predictSmoothHorizontalImpl = predictSmoothHorizontalNEON
		sumSamplesImpl = sumSamplesNEON
		applyCFLImpl = applyCFLNEON
		subsampleLuma8Impl = subsampleLuma8NEON
		dirRowInterp8Impl = dirRowInterp8NEON
		dirAboveRun8Impl = dirAboveRun8NEON
		dirLeftCol8Impl = dirLeftCol8NEON
		predictFilterIntra8Impl = predictFilterIntraBlockDirect8NEON
		predictFilterIntra16Impl = predictFilterIntraBlockDirect16NEON
		return
	}
	predictPaethImpl = predictPaethPureGo
	predictSmoothImpl = predictSmoothPureGo
	predictSmoothVerticalImpl = predictSmoothVerticalPureGo
	predictSmoothHorizontalImpl = predictSmoothHorizontalPureGo
	sumSamplesImpl = sumSamplesPureGo
	applyCFLImpl = applyCFLPureGo
	subsampleLuma8Impl = subsampleLuma8PureGo
	dirRowInterp8Impl = dirRowInterp8PureGo
	dirAboveRun8Impl = dirAboveRun8PureGo
	dirLeftCol8Impl = dirLeftCol8PureGo
}
