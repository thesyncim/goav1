// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the wide deblocking kernels under the goexperiment.simd build (the
// NEON binding in filter_wide_dispatch_arm64.go is excluded there via
// !goexperiment.simd). The eight-sample kernel routes through the Go-native SIMD
// implementation (filter8_gosimd_arm64.go); the six- and fourteen-sample kernels
// keep the hand-written NEON asm, which is still compiled in this build
// (filter_wide_neon_arm64.go is tagged arm64 && !purego), so they do not
// regress. Every binding is byte-identical to the pure-Go reference.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		// filter4_dispatch_arm64.go (the NEON narrow binding) is excluded in this
		// build via !goexperiment.simd, so bind the narrow kernels here too — both
		// the 8-bit and 16-bit Go-native SIMD forms.
		filter4EdgeImpl = filter4EdgeSIMD
		filter4Edge16Impl = filter4Edge16SIMD
		filter6EdgeImpl = filter6EdgeNEON
		filter8EdgeImpl = filter8EdgeSIMD
		filter14EdgeImpl = filter14EdgeNEON
		filter6Edge16Impl = filter6Edge16NEON
		filter8Edge16Impl = filter8Edge16SIMD
		filter14Edge16Impl = filter14Edge16NEON
		return
	}
	filter4EdgeImpl = filter4EdgePureGo
	filter4Edge16Impl = filter4Edge16PureGo
	filter6EdgeImpl = filter6EdgePureGo
	filter8EdgeImpl = filter8EdgePureGo
	filter14EdgeImpl = filter14EdgePureGo
	filter6Edge16Impl = filter6Edge16PureGo
	filter8Edge16Impl = filter8Edge16PureGo
	filter14Edge16Impl = filter14Edge16PureGo
}
