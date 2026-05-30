// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package cdef

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best CDEF block filter on amd64. When the CPU
// advertises AVX2 the kernel routes through the hand-written AVX2 asm;
// otherwise it keeps the pure-Go reference. The assignment happens once,
// before any decoder goroutine starts, so the steady-state cost is a single
// indirect call.
//
// The AVX2 wrapper itself falls back to pure-Go for non-8-wide blocks (the
// 8x4 / 4x8 / 4x4 chroma shapes), so the asm only handles the common 8-wide
// rows that dominate decode time. All bit-depths share one code path because
// the CDEF buffer is 16-bit regardless of source depth.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.AVX2 {
		filterBlockImpl = filterBlockAVX2
		return
	}
	filterBlockImpl = filterBlockPureGo
}
