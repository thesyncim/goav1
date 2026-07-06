// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package restoration

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best 8-bit-pixel restoration kernels on amd64,
// exactly like the u16 dispatchers in wiener_dispatch_amd64.go /
// selfguided_dispatch_amd64.go. When AVX2 is available the kernels route
// through the hand-written AVX2 asm; otherwise they keep the pure-Go
// references. The assignment happens once, before any decoder goroutine
// starts, so the steady-state cost is a single indirect call.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.AVX2 {
		wienerHorizontalU8Impl = wienerHorizontalU8AVX2
		wienerVerticalU8Impl = wienerVerticalU8AVX2
		sgrWeightedRowU8Impl = sgrWeightedRowU8AVX2
		return
	}
	wienerHorizontalU8Impl = wienerHorizontalU8
	wienerVerticalU8Impl = wienerVerticalU8
	sgrWeightedRowU8Impl = sgrWeightedRowU8
}
