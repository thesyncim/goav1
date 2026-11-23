// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the Go-native SIMD self-guided kernels on arm64 (NEON is mandatory on
// every arm64 chip Go runs on). The box sums use boxsumSeparable, which needs no
// SIMD to beat the brute-force window, and the per-pixel blend runs through
// sgrBlendRowSIMD. The data-dependent LUT gather inside calculateIntermediate stays
// scalar in both paths. The assignment happens once, before any decoder goroutine
// starts, so the steady-state cost is a single indirect call.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	boxsumImpl = boxsumSeparable
	if cpu.Detected.NEON {
		selfguidedImpl = selfguidedSIMD
		selfguidedFastImpl = selfguidedFastSIMD
		return
	}
	selfguidedImpl = selfguided
	selfguidedFastImpl = selfguidedFast
}
