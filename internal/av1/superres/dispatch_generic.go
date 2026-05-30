// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (!amd64 && !arm64) || (arm64 && purego)

package superres

// init binds the pure-Go super-res row kernel on architectures the dispatcher
// does not special-case, and on arm64 purego builds where the NEON assembly is
// excluded. This file's only purpose is to keep the dispatch wiring symmetric.
func init() {
	upscaleRowImpl = upscaleRowPureGo
}
