// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && !goexperiment.simd

package motion

// init binds the compound predictors to their NEON/I8MM asm implementations for
// the production (non-simd) build. The goexperiment.simd sibling
// (compound_dispatch_gosimd_arm64.go) binds the same set but overrides the 8-bit
// horizontal pass with the Go-native SIMD kernel.
func init() {
	compoundNEONBind()
}
