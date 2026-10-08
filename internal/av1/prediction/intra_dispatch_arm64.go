// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && !goexperiment.simd

package prediction

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best intra kernels on arm64. When NEON is
// available (mandatory on every arm64 chip Go runs on) the DC sum, CfL apply and
// the directional interpolation route through the hand-written NEON asm;
// otherwise they keep the pure-Go reference. The PAETH and SMOOTH predictors have
// no asm in this build and always use the pure-Go reference. The assignment
// happens once, before any decoder goroutine starts, so the steady-state cost is
// a single indirect call.
//
// The NEON wrappers fall back to pure-Go for non-8-bit samples and widths that
// are not a multiple of 8, so the asm only handles the common shapes that
// dominate intra decode time.
//
// Under the goexperiment.simd build the Go-native SIMD kernels bind instead
// (intra_static_gosimd_dispatch_arm64.go); this file is excluded there so the
// two inits never fight over the same slots.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		sumSamplesImpl = sumSamplesNEON
		applyCFLImpl = applyCFLNEON
		subsampleLuma8Impl = subsampleLuma8NEON
		dirRowInterp8Impl = dirRowInterp8NEON
		dirAboveRun8Impl = dirAboveRun8NEON
		dirLeftCol8Impl = dirLeftCol8NEON
		return
	}
	sumSamplesImpl = sumSamplesPureGo
	applyCFLImpl = applyCFLPureGo
	subsampleLuma8Impl = subsampleLuma8PureGo
	dirRowInterp8Impl = dirRowInterp8PureGo
	dirAboveRun8Impl = dirAboveRun8PureGo
	dirLeftCol8Impl = dirLeftCol8PureGo
}
