// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best 8-bit-pixel self-guided projection on arm64,
// exactly like the u16 dispatcher in selfguided_dispatch_arm64.go. The 8-bit
// Wiener passes are bound in wiener_dispatch_*.go. When NEON is available
// (mandatory on every arm64 chip Go runs on) the projection routes through the
// hand-written NEON asm; otherwise it keeps the pure-Go reference.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		sgrWeightedRowU8Impl = sgrWeightedRowU8NEON
		return
	}
	sgrWeightedRowU8Impl = sgrWeightedRowU8
}
