// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego && !goexperiment.simd

package restoration

// init binds the pure-Go self-guided kernels on default (non-goexperiment.simd)
// arm64 builds. The box sums use boxsumSeparable (libaom's separable boxsum1/
// boxsum2 running sum), which is byte-identical to boxsum and O(1)-amortized per
// output. The Go-native SIMD bindings live in selfguided_dispatch_gosimd_arm64.go.
func init() {
	boxsumImpl = boxsumSeparable
	selfguidedImpl = selfguided
	selfguidedFastImpl = selfguidedFast
}
