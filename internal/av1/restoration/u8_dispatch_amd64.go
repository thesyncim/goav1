// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package restoration

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best 8-bit-pixel self-guided projection on amd64,
// exactly like the u16 dispatcher in selfguided_dispatch_amd64.go. The 8-bit
// Wiener passes are bound in wiener_dispatch_*.go. When AVX2 is available the
// projection routes through the hand-written AVX2 asm; otherwise it keeps the
// pure-Go reference.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.AVX2 {
		sgrWeightedRowU8Impl = sgrWeightedRowU8AVX2
		return
	}
	sgrWeightedRowU8Impl = sgrWeightedRowU8
}
