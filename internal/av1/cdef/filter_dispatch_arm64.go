// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package cdef

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best CDEF block filter on arm64. When NEON is
// available (mandatory on every arm64 chip Go runs on) the kernel routes
// through the hand-written NEON asm; otherwise it keeps the pure-Go reference.
// The assignment happens once, before any decoder goroutine starts, so the
// steady-state cost is a single indirect call.
//
// The NEON wrapper itself falls back to pure-Go for non-8-wide blocks (the
// 8x4 / 4x8 / 4x4 chroma shapes), so the asm only handles the common 8-wide
// rows that dominate decode time. All bit-depths share one code path because
// the CDEF buffer is 16-bit regardless of source depth.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		filterBlockImpl = filterBlockNEON
		return
	}
	filterBlockImpl = filterBlockPureGo
}
