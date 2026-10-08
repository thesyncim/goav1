// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the convolve variants under the goexperiment.simd build. It mirrors
// convolve_dispatch_arm64.go except that selected 8-bit paths use official
// Go-native SIMD implementations. Unsupported shapes continue through the asm
// tiers, so all paths retain the same scalar-reference output.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		convolveX8Impl = convolveX8NEON
		convolveY8Impl = convolveY8NEON
		if cpu.Detected.I8MM {
			// Route supported wide horizontal blocks through the official SIMD
			// implementation; width-4 and unsupported filters use the I8MM asm.
			convolveX8Impl = convolveX8GoSIMDDispatch
			convolveY8Impl = convolveY8I8MM
		}

		// Both-axes-fractional 2D: the caller-scratch path is the decode hotspot
		// (predictInterPlaneBlock8). Unsupported widths and tap packings fall back
		// to the asm tier inside the wrapper.
		// The no-scratch slot keeps the asm tier: it is a cold path (scaled
		// references) whose per-call stack intermediate makes the GoSIMD's extra
		// horizontal work a net loss on small blocks.
		convolve2D8Impl = convolve2D8NEON
		convolve2D8WithScratchImpl = convolve2D8GoSIMDScratch

		// Edge-clamped 8-bit: the wrappers route to the fast path when the tap
		// window is fully resident and clamp via pure-Go otherwise. The
		// WithScratch slots additionally run dav1d's emu_edge model for
		// non-resident halos.
		convolveX8ClampedImpl = convolveX8ClampedNEON
		convolveY8ClampedImpl = convolveY8ClampedNEON
		convolveX8ClampedWithScratchImpl = convolveX8ClampedNEONWithScratch
		convolveY8ClampedWithScratchImpl = convolveY8ClampedNEONWithScratch
		convolve2D8ClampedImpl = convolve2D8ClampedNEON
		convolve2D8ClampedWithScratchImpl = convolve2D8ClampedNEONWithScratch
		if cpu.Detected.I8MM {
			convolve2D8ClampedWithScratchImpl = convolve2D8ClampedI8MMWithScratch
		}

		// High-bit-depth (10/12-bit): keep the NEON asm tier.
		convolveXHighBDImpl = convolveXHighBDNEON
		convolveYHighBDImpl = convolveYHighBDNEON
		convolve2DHighBDImpl = convolve2DHighBDNEON
		convolveXHighBDClampedImpl = convolveXHighBDClampedNEON
		convolveYHighBDClampedImpl = convolveYHighBDClampedNEON
		convolve2DHighBDClampedImpl = convolve2DHighBDClampedNEON
		convolve2DHighBDWithScratchImpl = convolve2DHighBDNEONWithScratch
		convolve2DHighBDClampedWithScratchImpl = convolve2DHighBDClampedNEONWithScratch
		return
	}
	convolveX8Impl = convolveX8PureGo
	convolveY8Impl = convolveY8PureGo
	convolve2D8Impl = convolve2D8PureGo
	convolve2D8WithScratchImpl = convolve2D8WithScratchDefault
}
