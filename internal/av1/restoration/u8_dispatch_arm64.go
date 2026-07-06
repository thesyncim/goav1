// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best 8-bit-pixel restoration kernels on arm64,
// exactly like the u16 dispatchers in wiener_dispatch_arm64.go /
// selfguided_dispatch_arm64.go. When NEON is available (mandatory on every
// arm64 chip Go runs on) the kernels route through the hand-written NEON asm;
// otherwise they keep the pure-Go references. The assignment happens once,
// before any decoder goroutine starts, so the steady-state cost is a single
// indirect call.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		wienerHorizontalU8Impl = wienerHorizontalU8NEON
		wienerVerticalU8Impl = wienerVerticalU8NEON
		sgrWeightedRowU8Impl = sgrWeightedRowU8NEON
		return
	}
	wienerHorizontalU8Impl = wienerHorizontalU8
	wienerVerticalU8Impl = wienerVerticalU8
	sgrWeightedRowU8Impl = sgrWeightedRowU8
}
