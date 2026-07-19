// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !arm64 || purego

package filmgrain

// buildScaleRow10 is the 10-bit scaling-LUT gather entry point on architectures
// without a dedicated kernel (and on the arm64 purego build where the NEON asm
// is excluded). It calls the pure-Go reference directly; keeping it a concrete
// call preserves the zero-alloc property of the apply path (the caller's scale
// scratch buffer stays on the stack).
func buildScaleRow10(scale []uint16, src []uint16, lut []uint8) {
	buildScaleRow10PureGo(scale, src, lut)
}
