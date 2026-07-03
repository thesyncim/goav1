// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package loopfilter

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best 8-bit wide deblocking kernels on amd64. When
// the CPU advertises AVX2 (and the cpu package reports it) the six/eight/
// fourteen-sample kernels route through the hand-written AVX2 asm, which
// processes sixteen horizontal edge positions per 256-bit vector and falls back
// to the pure-Go reference for vertical edges, high bit depth, and short tails.
// Otherwise they keep the pure-Go reference. The assignment happens once, before
// any decoder goroutine starts, so the steady-state cost is a single indirect
// call.
//
// cpu.Detected.AVX2 is false on hosts whose CPUID does not advertise AVX2 —
// notably Apple/Rosetta, which translates AVX2 instructions but reports them
// absent — so production on such hosts stays on pure-Go. The AVX2 kernels'
// byte-exactness is nonetheless verified by TestWideFilterAVX2MatchesPureGo,
// which calls them directly (under Rosetta) regardless of the CPUID report.
//
// The 10/12-bit (two-byte) wide kernels keep the pure-Go reference on amd64;
// only the 8-bit wide path has an AVX2 variant today.
func init() {
	_ = cpu.Detected
	if cpu.Detected.AVX2 {
		filter6EdgeImpl = filter6EdgeAVX2
		filter8EdgeImpl = filter8EdgeAVX2
		filter14EdgeImpl = filter14EdgeAVX2
		return
	}
	filter6EdgeImpl = filter6EdgePureGo
	filter8EdgeImpl = filter8EdgePureGo
	filter14EdgeImpl = filter14EdgePureGo
}
