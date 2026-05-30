// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64

package motion

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the convolve dispatch slots on amd64. No tuned amd64 variant ships
// yet, so every slot keeps the pure-Go reference. The assignment happens once,
// before any decoder goroutine starts, so the steady-state cost is a single
// indirect call.
//
// TODO: add SSE4.1/AVX2 variants here, gated on the matching cpu.Detected flag.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	convolveX8Impl = convolveX8PureGo
	convolveY8Impl = convolveY8PureGo
	convolve2D8Impl = convolve2D8PureGo
}
