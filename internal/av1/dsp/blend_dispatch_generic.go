// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (!arm64 && !amd64) || ((arm64 || amd64) && purego)

package dsp

// init binds the pure-Go BlendA64Mask inner loop on every target without a
// tuned variant: architectures with no hand-written kernel, plus arm64/amd64
// built with the purego tag (which excludes all assembly).
func init() {
	blendA64MaskImpl = blendA64MaskPureGo
}
