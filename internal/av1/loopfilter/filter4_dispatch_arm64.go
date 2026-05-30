// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package loopfilter

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best 8-bit narrow deblocking kernel on arm64.
// When NEON is available (mandatory on every arm64 chip Go runs on) the kernel
// routes through the hand-written NEON asm; otherwise it keeps the pure-Go
// reference. The assignment happens once, before any decoder goroutine starts,
// so the steady-state cost is a single indirect call.
//
// The NEON wrapper itself processes eight edge positions per vector and falls
// back to pure-Go for any tail shorter than eight positions, so the asm only
// ever sees full 8-wide groups. Both edge orientations are handled: horizontal
// edges load contiguous tap rows, vertical edges gather four strided columns.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		filter4EdgeImpl = filter4EdgeNEON
		return
	}
	filter4EdgeImpl = filter4EdgePureGo
}
