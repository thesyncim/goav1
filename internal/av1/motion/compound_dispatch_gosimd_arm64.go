// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the compound predictors under the goexperiment.simd build. It mirrors
// compound_dispatch_arm64.go (the !goexperiment.simd sibling) -- every predictor
// keeps its proven NEON/I8MM asm tier -- except two passes routed through Go-native
// SIMD kernels: the 8-bit horizontal CONV_BUF pass (USMMLA + int16 round-narrow
// tail) and the 8-bit compound average/distance blend (16-wide UMULL/UMLAL +
// SQSHRUN fuse). Both fall back to the asm tier for the shapes they do not cover
// (width-4, edge-overhang, odd taps for X; roundBits!=4 and width<8 for the blend),
// so every case stays accelerated and byte-exact.
func init() {
	compoundNEONBind()
	if cpu.Detected.NEON {
		predictInterCompoundRef8ToConvBufXImpl = compoundX8GoSIMD
		predictInterCompoundRef8ToConvBuf2DImpl = compound2D8GoSIMD
		blendCompoundAvg8Impl = blendCompoundAvg8GoSIMD
	}
}
