// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best self-guided kernels on arm64. When NEON is
// available (mandatory on every arm64 chip Go runs on) the box sums and the
// per-pixel blend route through the hand-written NEON asm; the data-dependent
// LUT gather inside calculateIntermediate stays scalar in both paths. The
// assignment happens once, before any decoder goroutine starts, so the
// steady-state cost is a single indirect call.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		boxsumImpl = boxsumNEON
		selfguidedImpl = selfguidedNEON
		selfguidedFastImpl = selfguidedFastNEON
		return
	}
	boxsumImpl = boxsum
	selfguidedImpl = selfguided
	selfguidedFastImpl = selfguidedFast
}
