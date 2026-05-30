// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package motion

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best convolve variants on amd64. When AVX2 is
// available the 8-bit and high-bit-depth kernels route through the hand-written
// AVX2 asm; otherwise they keep the pure-Go reference. The assignment happens
// once, before any decoder goroutine starts, so the steady-state cost is a
// single indirect call.
//
// The AVX2 wrappers fall back to pure-Go for width<8 (8-bit) / width<4 (HBD)
// and 4-tap blocks, so the asm handles only the common width-multiple 8-tap
// shapes that dominate decode time. AVX2 detection is CPUID-based (see the cpu
// package); on hosts that do not advertise AVX2 (e.g. amd64 under Rosetta 2)
// the slots keep the pure-Go reference.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.AVX2 {
		convolveX8Impl = convolveX8AVX2
		convolveY8Impl = convolveY8AVX2
		convolve2D8Impl = convolve2D8AVX2

		convolveX8ClampedImpl = convolveX8ClampedAVX2
		convolveY8ClampedImpl = convolveY8ClampedAVX2
		convolve2D8ClampedImpl = convolve2D8ClampedAVX2

		convolveXHighBDImpl = convolveXHighBDAVX2
		convolveYHighBDImpl = convolveYHighBDAVX2
		convolve2DHighBDImpl = convolve2DHighBDAVX2
		convolveXHighBDClampedImpl = convolveXHighBDClampedAVX2
		convolveYHighBDClampedImpl = convolveYHighBDClampedAVX2
		convolve2DHighBDClampedImpl = convolve2DHighBDClampedAVX2
		return
	}
	convolveX8Impl = convolveX8PureGo
	convolveY8Impl = convolveY8PureGo
	convolve2D8Impl = convolve2D8PureGo
}
