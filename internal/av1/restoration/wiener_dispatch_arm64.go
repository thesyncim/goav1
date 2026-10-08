// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best Wiener passes on arm64. When NEON is
// available (mandatory on every arm64 chip Go runs on) the kernels route
// through the hand-written NEON asm; otherwise they keep the pure-Go reference.
// The assignment happens once, before any decoder goroutine starts, so the
// steady-state cost is a single indirect call.
//
// The NEON wrappers fall back to pure-Go for widths below the 8-lane vector
// (and for non-multiple-of-8 tails) so the asm only handles full 8-wide column
// groups. All bit-depths share one code path because the samples (<= 4095) fit
// in positive int16 lanes and the int32 accumulator never overflows.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		wienerHorizontalImpl = wienerHorizontalNEON
		wienerHorizontalTrustedImpl = wienerHorizontalNEONTrusted
		wienerVerticalImpl = wienerVerticalNEON
		return
	}
	wienerHorizontalImpl = wienerHorizontal
	wienerHorizontalTrustedImpl = wienerHorizontalTrusted
	wienerVerticalImpl = wienerVertical
}
