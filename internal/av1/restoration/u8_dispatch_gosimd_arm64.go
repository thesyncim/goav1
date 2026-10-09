// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

// init binds the Go-native SIMD 8-bit-pixel self-guided projection on arm64. NEON
// is mandatory on every arm64 chip Go runs on, so the binding is unconditional.
// The 8-bit Wiener passes are bound in wiener_dispatch_arm64.go.
func init() {
	sgrWeightedRowU8Impl = sgrWeightedRowU8SIMD
}
