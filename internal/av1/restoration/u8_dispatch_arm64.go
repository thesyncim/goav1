// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && !goexperiment.simd

package restoration

// init binds the pure-Go 8-bit-pixel self-guided projection on default
// (non-goexperiment.simd) arm64 builds. The Go-native SIMD binding lives in
// u8_dispatch_gosimd_arm64.go, and the 8-bit Wiener passes are bound in
// wiener_dispatch_*.go.
func init() {
	sgrWeightedRowU8Impl = sgrWeightedRowU8
}
