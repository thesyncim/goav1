// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (!amd64 && !arm64) || (arm64 && purego)

package quantize

// init binds the pure-Go dequant column variant on architectures that the
// quantize dispatcher does not yet special-case, and on arm64 when the purego
// build tag excludes the NEON assembly. This file keeps the dispatch wiring
// symmetric across architectures.
func init() {
	dequantColumnImpl = dequantColumnPureGo
}
