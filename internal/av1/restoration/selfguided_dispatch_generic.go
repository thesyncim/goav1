// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !arm64 || (arm64 && purego)

package restoration

// init binds the pure-Go self-guided kernels on architectures the dispatcher
// does not special-case, and on the arm64 purego build where the NEON asm is
// excluded. This file only keeps the dispatch wiring symmetric across builds.
func init() {
	boxsumImpl = boxsum
	selfguidedImpl = selfguided
	selfguidedFastImpl = selfguidedFast
}
