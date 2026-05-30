// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64

package loopfilter

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the narrow deblocking kernel dispatch slot on amd64. No tuned
// amd64 variant ships yet, so the slot keeps the pure-Go reference. The
// assignment happens once, before any decoder goroutine starts.
//
// TODO: add SSE4.1/AVX2 variants here, gated on the matching cpu.Detected flag.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	filter4EdgeImpl = filter4EdgePureGo
}
