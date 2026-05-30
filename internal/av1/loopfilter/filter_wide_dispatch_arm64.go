// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package loopfilter

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best 8-bit wide deblocking kernels on arm64.
// When NEON is available (mandatory on every arm64 chip Go runs on) the
// six/eight/fourteen-sample kernels route through the hand-written NEON asm;
// otherwise they keep the pure-Go reference. The NEON wrappers handle only
// 8-bit horizontal edges in full groups of eight positions and fall back to
// pure-Go for vertical edges, high bit depth, and short tails.
func init() {
	_ = cpu.Detected
	if cpu.Detected.NEON {
		filter6EdgeImpl = filter6EdgeNEON
		filter8EdgeImpl = filter8EdgeNEON
		// filter14 keeps the pure-Go reference: its flat2 decision plus twelve
		// weighted-average outputs exceed the register budget of a single
		// contiguous NEON pass, so it is left on the audited scalar path.
		filter14EdgeImpl = filter14EdgePureGo
		return
	}
	filter6EdgeImpl = filter6EdgePureGo
	filter8EdgeImpl = filter8EdgePureGo
	filter14EdgeImpl = filter14EdgePureGo
}
