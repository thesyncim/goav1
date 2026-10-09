// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego && !goexperiment.simd

package restoration

// init binds the pure-Go 8-bit-pixel self-guided projection on default
// (non-goexperiment.simd) amd64 builds. The Go-native AVX2 binding lives in
// u8_dispatch_gosimd_amd64.go.
func init() {
	sgrWeightedRowU8Impl = sgrWeightedRowU8
}
