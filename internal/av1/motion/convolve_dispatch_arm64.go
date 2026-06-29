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
// width>=8 8-tap shape. The I8MM X/Y wrappers handle the resident source-shaped
// matrix/dot-product tiers and fall back to NEON for unsupported widths, heights
// or tap counts.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		convolveX8Impl = convolveX8NEON
		convolveY8Impl = convolveY8NEON
		if cpu.Detected.I8MM {
			convolveX8Impl = convolveX8I8MM
			convolveY8Impl = convolveY8I8MM
		}
		// The 2D kernels handle width-4 and every width>=8 8-tap shape; their Go
		// wrappers fall back to the next proven tier for unsupported widths.
		// Keep the no-scratch slot on NEON: indirect dispatch into the I8MM
		// wrapper loses enough stack-scratch optimization on Apple M4 that the
		// measured row no longer wins. The caller-scratch hot path does win.
		convolve2D8Impl = convolve2D8NEON
		convolve2D8WithScratchImpl = convolve2D8NEONWithScratch
		if cpu.Detected.I8MM {
			convolve2D8WithScratchImpl = convolve2D8I8MMWithScratch
		}

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
