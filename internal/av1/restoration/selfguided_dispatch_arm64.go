// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best self-guided kernels on arm64. When NEON is
// available (mandatory on every arm64 chip Go runs on) the per-pixel blend
// routes through the hand-written NEON asm; the data-dependent LUT gather inside
// calculateIntermediate stays scalar in both paths. The assignment happens once,
// before any decoder goroutine starts, so the steady-state cost is a single
// indirect call.
//
// The box sums use boxsumSeparable (libaom's separable boxsum1/boxsum2 running
// sum) unconditionally: its O(1)-amortized work beats the NEON brute-force
// window re-summation by ~4x for r=2 and ~1.7x for r=1 on this hardware, so the
// NEON box-sum asm is no longer the fastest path.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	boxsumImpl = boxsumSeparable
	if cpu.Detected.NEON {
		selfguidedImpl = selfguidedNEON
		selfguidedFastImpl = selfguidedFastNEON
		return
	}
	selfguidedImpl = selfguided
	selfguidedFastImpl = selfguidedFast
}
