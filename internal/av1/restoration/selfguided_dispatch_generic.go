// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (!amd64 && !arm64) || (arm64 && purego) || (amd64 && purego)

package restoration

// init binds the pure-Go self-guided kernels on architectures the dispatcher
// does not special-case, and on the arm64/amd64 purego builds where the SIMD
// asm is excluded. This file only keeps the dispatch wiring symmetric across
// builds.
func init() {
	// boxsumSeparable (libaom's separable boxsum1/boxsum2 running sum) replaces
	// the O((2r+1)^2) brute-force boxsum: it is byte-identical and O(1)-amortized
	// per output, a clear win on the scalar path.
	boxsumImpl = boxsumSeparable
	selfguidedImpl = selfguided
	selfguidedFastImpl = selfguidedFast
}
