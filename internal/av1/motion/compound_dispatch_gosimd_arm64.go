// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the compound predictors under the goexperiment.simd build. It mirrors
// compound_dispatch_arm64.go except for three 8-bit paths routed through official
// Go-native SIMD implementations: horizontal CONV_BUF, two-dimensional CONV_BUF,
// and average/distance blend. Unsupported widths, filter shapes, and blend
// rounding parameters continue through the NEON/I8MM asm tier.
func init() {
	compoundNEONBind()
	if cpu.Detected.NEON {
		predictInterCompoundRef8ToConvBufXImpl = compoundX8GoSIMD
		predictInterCompoundRef8ToConvBuf2DImpl = compound2D8GoSIMD
		blendCompoundAvg8Impl = blendCompoundAvg8GoSIMD
	}
}
