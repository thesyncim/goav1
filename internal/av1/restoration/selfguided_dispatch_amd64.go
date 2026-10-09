// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego && !goexperiment.simd

package restoration

// init binds the pure-Go self-guided kernels on default (non-goexperiment.simd)
// amd64 builds. The Go-native AVX2 bindings live in
// selfguided_dispatch_gosimd_amd64.go.
func init() {
	boxsumImpl = boxsum
	selfguidedImpl = selfguided
	selfguidedFastImpl = selfguidedFast
}
