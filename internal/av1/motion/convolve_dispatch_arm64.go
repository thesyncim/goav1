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
// The NEON wrappers handle width-4 (dedicated 4-lane kernels) and every
// width>=8 8-tap shape. The I8MM X wrapper handles width>=8 8-lane filters,
// including 4-tap filters via zeroed end taps, and falls back to NEON for
// width-4 or odd-width blocks.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		convolveX8Impl = convolveX8NEON
		convolveY8Impl = convolveY8NEON
		if cpu.Detected.I8MM {
			convolveX8Impl = convolveX8I8MM
		}
		// The 2D NEON kernel handles width-4 and every width>=8 8-tap shape; its
		// Go wrapper falls back to pure-Go only for non-multiple-of-8 widths != 4.
		convolve2D8Impl = convolve2D8NEON
		convolve2D8WithScratchImpl = convolve2D8NEONWithScratch

		// Edge-clamped 8-bit: the wrappers route to the fast NEON path when the
		// tap window is fully resident and clamp via pure-Go otherwise.
		convolveX8ClampedImpl = convolveX8ClampedNEON
		convolveY8ClampedImpl = convolveY8ClampedNEON
		convolve2D8ClampedImpl = convolve2D8ClampedNEON
		convolve2D8ClampedWithScratchImpl = convolve2D8ClampedNEONWithScratch

		// High-bit-depth (10/12-bit). The 1D Y, 1D X and 2D non-clamped kernels
		// have NEON; the clamped HBD variants reuse the same in-bounds fast-path
		// trick.
		convolveXHighBDImpl = convolveXHighBDNEON
		convolveYHighBDImpl = convolveYHighBDNEON
		convolve2DHighBDImpl = convolve2DHighBDNEON
		convolveXHighBDClampedImpl = convolveXHighBDClampedNEON
		convolveYHighBDClampedImpl = convolveYHighBDClampedNEON
		convolve2DHighBDClampedImpl = convolve2DHighBDClampedNEON
		return
	}
	convolveX8Impl = convolveX8PureGo
	convolveY8Impl = convolveY8PureGo
	convolve2D8Impl = convolve2D8PureGo
	convolve2D8WithScratchImpl = convolve2D8WithScratchDefault
}
