// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package loopfilter

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the narrow deblocking kernel dispatch slot on amd64. When the CPU
// advertises AVX2 (and the cpu package reports it), the slot routes through the
// hand-written AVX2 asm, which processes sixteen 8-bit horizontal edge
// positions per 256-bit vector and falls back to the pure-Go reference for
// vertical edges, high bit depth, and short tails. Otherwise the slot keeps the
// pure-Go reference. The assignment happens once, before any decoder goroutine
// starts, so the steady-state cost is a single indirect call.
//
// Note: cpu.Detected.AVX2 is only set once the cpu package gains CPUID-based
// detection; until then this binding stays on pure-Go on every amd64 host
// (including Apple/Rosetta, which does not expose AVX2 at all).
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	// The 10/12-bit (two-byte) narrow kernel keeps the pure-Go reference on
	// amd64; only the 8-bit path has an AVX2 variant today.
	filter4Edge16Impl = filter4Edge16PureGo
	// amd64 keeps the pure-Go filter4 reference: the prototyped AVX2 variant
	// diverged from the reference under direct execution (not byte-exact yet).
	filter4EdgeImpl = filter4EdgePureGo
}
