// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (!amd64 && !arm64) || (arm64 && purego)

package loopfilter

// init binds the pure-Go narrow deblocking kernel on architectures the
// dispatcher does not special-case, and on the arm64 purego build where the
// NEON asm is excluded. This file only keeps the dispatch wiring symmetric
// across builds.
func init() {
	filter4EdgeImpl = filter4EdgePureGo
}
