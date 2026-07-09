// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the compound predictors under the goexperiment.simd build. It mirrors
// compound_dispatch_arm64.go (the !goexperiment.simd sibling) -- every predictor
// keeps its proven NEON/I8MM asm tier -- except the 8-bit horizontal CONV_BUF pass,
// which routes through the Go-native SIMD kernel (USMMLA + int16 round-narrow tail).
// compoundX8GoSIMD itself falls back to the asm tier for width-4, edge-overhanging,
// odd-tap and nonzero-end-tap shapes, so every case stays accelerated and byte-exact.
func init() {
	compoundNEONBind()
	if cpu.Detected.NEON {
		predictInterCompoundRef8ToConvBufXImpl = compoundX8GoSIMD
	}
}
