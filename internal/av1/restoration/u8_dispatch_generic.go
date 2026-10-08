// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (!amd64 && !arm64) || (arm64 && purego) || (amd64 && purego)

package restoration

// init binds the pure-Go 8-bit-pixel self-guided projection on architectures the
// dispatcher does not special-case, and on the arm64/amd64 purego builds where
// the SIMD asm is excluded. This file only keeps the dispatch wiring symmetric
// across builds.
func init() {
	sgrWeightedRowU8Impl = sgrWeightedRowU8
}
