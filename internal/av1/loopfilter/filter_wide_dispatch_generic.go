// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (!amd64 && !arm64) || (arm64 && purego)

package loopfilter

// init binds the pure-Go six/eight/fourteen-sample deblocking kernels on
// architectures the dispatcher does not special-case, and on the arm64 purego
// build where the NEON asm is excluded. This file keeps the dispatch wiring
// symmetric across builds.
func init() {
	filter6EdgeImpl = filter6EdgePureGo
	filter8EdgeImpl = filter8EdgePureGo
	filter14EdgeImpl = filter14EdgePureGo
}
