// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best convolve variants on arm64. When NEON is
// available (mandatory on every arm64 chip Go runs on) the 8-bit
// non-edge-clamped kernels route through the hand-written NEON asm; otherwise
// they keep the pure-Go reference. The assignment happens once, before any
// decoder goroutine starts, so the steady-state cost is a single indirect call.
//
// The NEON wrappers themselves fall back to pure-Go for width-4 and 4-tap
// blocks, so the only blocks the asm handles are the common width>=8 8-tap
// shapes that dominate decode time.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		convolveX8Impl = convolveX8NEON
		convolveY8Impl = convolveY8NEON
		// convolve2D8 keeps the pure-Go reference: the 2D NEON kernel is not
		// implemented yet. The 1D X/Y NEON variants already accelerate the
		// separable single-axis passes.
		convolve2D8Impl = convolve2D8PureGo
		return
	}
	convolveX8Impl = convolveX8PureGo
	convolveY8Impl = convolveY8PureGo
	convolve2D8Impl = convolve2D8PureGo
}
