// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package restoration

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best self-guided kernels on amd64. When AVX2 is
// available the box sums and the per-pixel blend route through the hand-written
// AVX2 asm; the data-dependent LUT gather inside calculateIntermediate stays
// scalar in both paths. The assignment happens once, before any decoder
// goroutine starts, so the steady-state cost is a single indirect call.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	boxsumImpl = boxsum
	selfguidedImpl = selfguided
	selfguidedFastImpl = selfguidedFast
	if cpu.Detected.AVX2 {
		boxsumImpl = boxsumAVX2
		selfguidedImpl = selfguidedAVX2
		selfguidedFastImpl = selfguidedFastAVX2
	}
}
