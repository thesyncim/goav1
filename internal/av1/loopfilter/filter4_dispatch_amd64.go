// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package loopfilter

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the narrow deblocking kernel dispatch slot on amd64. When the CPU
// advertises AVX2 (and the cpu package reports it), the 8-bit slot routes
// through the hand-written AVX2 asm, which processes sixteen horizontal edge
// positions per 256-bit vector and falls back to the pure-Go reference for
// vertical edges, high bit depth, and short tails. Otherwise the slot keeps the
// pure-Go reference. The assignment happens once, before any decoder goroutine
// starts, so the steady-state cost is a single indirect call.
//
// cpu.Detected.AVX2 is false on hosts whose CPUID does not advertise AVX2 —
// notably Apple/Rosetta, which translates AVX2 instructions but reports them
// absent — so production on such hosts stays on pure-Go. The AVX2 kernel's
// byte-exactness is nonetheless verified by TestFilter4EdgeAVX2MatchesPureGo,
// which calls it directly (under Rosetta) regardless of the CPUID report.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	// The 10/12-bit (two-byte) narrow kernel keeps the pure-Go reference on
	// amd64; only the 8-bit path has an AVX2 variant today.
	filter4Edge16Impl = filter4Edge16PureGo
	if cpu.Detected.AVX2 {
		filter4EdgeImpl = filter4EdgeAVX2
		return
	}
	filter4EdgeImpl = filter4EdgePureGo
}
