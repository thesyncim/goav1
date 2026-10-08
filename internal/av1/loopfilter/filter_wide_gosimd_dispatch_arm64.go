// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init keeps measured SIMD winners live and routes losing 8-bit kernels to
// NEON. HBD six- and fourteen-tap SIMD kernels remain selected; HBD eight-tap
// stays on NEON after its paired comparison.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		filter4EdgeImpl = filter4EdgeSIMD
		filter4Edge16Impl = filter4Edge16SIMD
		filter6EdgeImpl = filter6EdgeNEON
		filter8EdgeImpl = filter8EdgeNEON
		filter14EdgeImpl = filter14EdgeNEON
		filter6Edge16Impl = filter6Edge16SIMD
		filter8Edge16Impl = filter8Edge16NEON
		filter14Edge16Impl = filter14Edge16SIMD
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
