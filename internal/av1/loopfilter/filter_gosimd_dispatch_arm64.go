// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the Go-native SIMD deblocking kernels on arm64. NEON is mandatory
// on every arm64 chip Go runs on; the cpu check keeps the same gate as the
// pre-SIMD dispatch. Without it the slots keep the pure-Go reference. The
// assignment happens once, before any decoder goroutine starts, so the
// steady-state cost is a single indirect call.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		filter4EdgeImpl = filter4EdgeSIMD
		filter4Edge16Impl = filter4Edge16SIMD
		filter6EdgeImpl = filter6EdgeSIMD
		filter8EdgeImpl = filter8EdgeSIMD
		filter14EdgeImpl = filter14EdgeSIMD
		filter6Edge16Impl = filter6Edge16SIMD
		filter8Edge16Impl = filter8Edge16SIMD
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
